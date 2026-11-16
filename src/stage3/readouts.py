"""Shared statistics, target assignments, C blocks and immutable selected scorers."""
import hashlib
import inspect
import numpy as np
from scipy import sparse
from src.common.contracts import fingerprint
from src.stage2.schema import check, plain, frozen
from .artifacts import Store, write
from .data import CS, ROLES
from .engine import statistics, transform, fit_heads, metrics, choose


def assignment(seed):
    check(seed=='true' or seed in ('seed0','seed1','seed2'), 'Unknown target assignment')
    return seed


def donor_map(plan,aid,split,draw=None):
    assignment(aid)
    check(split in ('FIT','DEV-S','SCREEN') and (draw is None or (split=='SCREEN' and 1<=draw<=1999)), 'Closed assignment partition/draw')
    ids=sorted(i for i,t in plan['targets'].items() if t['split']==split)
    if aid=='true':
        check(draw is None,'No true-label null draws'); return dict(zip(ids,ids))
    seed=int(aid[-1])
    if draw is None: prefix=f'G9Q6-G0-label-v1|{seed}|{split}|'
    else: prefix=f'G9Q6-G0-null-v1|{draw}|{seed}|'
    donors=sorted(ids,key=lambda i:(hashlib.sha256((prefix+i).encode()).digest(),i))
    return dict(zip(ids,donors))


def targets(plan,rows,aid,draw=None):
    check(rows and len({r['split'] for r in rows})==1,'Single-partition rows required')
    split=rows[0]['split']; mapping=donor_map(plan,aid,split,draw)
    yy=[]; masks=[]
    for r in rows:
        check(r['image_id'] in mapping,'Wrong target population')
        values=plan['targets'][mapping[r['image_id']]]['values']
        yy.append([0 if v is None else v for v in values]); masks.append([v is not None for v in values])
    return np.asarray(yy,dtype='<f8'),np.asarray(masks,dtype=bool)


def weights(rows,group):
    return np.full(len(rows),0.25 if group['source']=='e03' and group['language'] is None else 1.,dtype='<f8')


def contexts(rows): return [r['language'] or 'image' for r in rows]


def expected_rows(plan,gid,split,store):
    g=plan['groups'][gid]
    if g['source']=='vlm':
        from .data import _plan
        from src.stage2.plans import planned_rows
        m=_plan(store,g['model_plan'],'stage2_model_plan')
        rows=[r for u in m['work_units'] if u['split']==split for r in planned_rows(m,u['chunk_id'],g['condition'])]
        return [r for r in rows if r['language']==g['language']]
    dp=store.json(plan['data_plan'],'plan.json','stage2_data_plan')
    condition=g['condition'] if g['source']=='vlm' else 'caption_only'
    rows=[r for r in dp['rows'][condition] if r['split']==split and
          (g['source']=='e03' and g['language'] is None or r['language']==g['language'])]
    if g['source']=='e03': rows=[dict(r,row_index=i,image_loss_weight=.25 if g['language'] is None else 1.) for i,r in enumerate(rows)]
    return rows


def publish_target_assignments(plan,root,producer):
    check(plan.root==__import__('pathlib').Path(root).resolve(),'Root mismatch')
    maps={aid:{s:donor_map(plan,aid,s) for s in ('FIT','DEV-S','SCREEN')} for aid in ('true','seed0','seed1','seed2')}
    return write(plan,'cache/shared/stage3-assignments','assignments',{'maps':maps,'assignment_identity':fingerprint(maps)}, {},producer)


def _body(plan,ref,kind,store=None,fit=True):
    store=store or Store(plan.root,plan.protocol)
    store.dataset=plan.get('dataset_fingerprint')
    b=store.json(ref,'body.json','stage3_'+kind)
    if fit: check(b['fit_identity']==plan['fit_identity'],'Incompatible fitted artifact')
    return b,store


def stats_identity(plan,gid,rows):
    g=plain(plan['groups'][gid])
    return fingerprint({'FIT':g['releases']['FIT'],'site':g.get('site'),'language':g['language'],
                        'dimension':g['dimension'],'rows':plain(rows),'weights':weights(rows,g).tolist(),
                        'runtime':plain(plan['runtime']),'statistics':inspect.getsource(__import__('src.stage3.engine',fromlist=['statistics']).statistics)})


def load_statistics(plan,ref,gid,rows,store=None):
    b,s=_body(plan,ref,'statistics',store,fit=False)
    check(b['input_identity']==stats_identity(plan,gid,rows),'Wrong source statistics')
    d=plan['groups'][gid]['dimension']
    stats={k:s.array(ref,k+'.npy','<f8',(d,)) for k in ('mean','variance','scale')}
    check(np.all(stats['variance']>=0) and np.array_equal(stats['scale'],np.where(stats['variance']<=1e-12,1,np.sqrt(stats['variance']))),'Invalid source scales')
    stats['center']=b['center']; stats['reference']=s.ref(ref)
    return stats


def load_or_publish_statistics(plan,group_id,session,producer,supplied_refs=()):
    check(session.plan is plan and session.role=='source_fit','Statistics require FIT session')
    rows,x=session.read(group_id)
    matches=[]
    for ref in supplied_refs:
        b,_=_body(plan,ref,'statistics',session.store,fit=False)
        if b['input_identity']==stats_identity(plan,group_id,rows): matches.append(ref)
    check(len(matches)<=1,'Conflicting/duplicate statistics references')
    if matches: return load_statistics(plan,matches[0],group_id,rows,session.store)
    stats=statistics(x,weights(rows,plan['groups'][group_id]))
    ref=write(plan,'cache/'+plan['fit_identity'][:16]+'/preprocessing','statistics',
              {'group_id':group_id,'input_identity':stats_identity(plan,group_id,rows),'center':stats['center']},
              {k:stats[k] for k in ('mean','variance','scale')},producer)
    return load_statistics(plan,ref,group_id,rows)


def fit_candidate_block(plan,group_id,assignment_id,C,session,statistics,producer):
    """session is {source_fit: NumericSession, source_select: NumericSession}."""
    assignment(assignment_id); check(C in CS,'Unknown C')
    fit,dev=session['source_fit'],session['source_select']
    check(fit.plan is dev.plan is plan and fit.role=='source_fit' and dev.role=='source_select','Fitting roles mismatch')
    rows,x=fit.read(group_id); dr,dx=dev.read(group_id)
    stats=load_statistics(plan,statistics['reference'],group_id,rows,fit.store)
    g=plan['groups'][group_id]
    check(stats['center']==(not sparse.issparse(x)),'Scaler centering policy mismatch')
    y,known=targets(plan,rows,assignment_id); dy,dk=targets(plan,dr,assignment_id)
    coef,intercept,valid,records=fit_heads(_transformed(fit,group_id,stats),y,known,weights(rows,g),C)
    logits=np.asarray(_transformed(dev,group_id,stats)@coef.T)+intercept
    report=metrics(logits,dy,dk,valid,contexts(dr),[list(plan['categories']).index(c) for c in plan['primary']])
    check(report['selection_score'] is not None,'Undefined primary DEV-S cohort')
    body={'group_id':group_id,'assignment_id':assignment_id,'C':C,'statistics':plain(stats['reference']),
          'fit_rows':fingerprint(plain(rows)),'dev_rows':plain(dr),'target_identity':fingerprint({'FIT':donor_map(plan,assignment_id,'FIT'),'DEV-S':donor_map(plan,assignment_id,'DEV-S')}),
          'records':records,'metrics':report}
    return write(plan,'checkpoints/'+plan['fit_identity'][:16]+'/'+group_id[:16],'candidate',body,
                 {'coefficients':coef,'intercepts':intercept,'estimable':valid,'dev_logits':logits},producer)


def _transformed(session,gid,stats):
    _,x=session.read(gid)  # Role/ownership guards still run on cache hits.
    key=(gid,stats['reference']['sha256'])
    old=getattr(session,'_working',None)
    if old is not None and old[0]==key: return old[1]
    a=transform(x,stats)
    if sparse.issparse(a):
        for v in (a.data,a.indices,a.indptr):v.flags.writeable=False
    else:a.flags.writeable=False
    session._working=(key,a)
    return a


def candidate(plan,ref,store=None):
    b,s=_body(plan,ref,'candidate',store)
    check(b['group_id'] in plan['groups'] and b['C'] in CS,'Bad candidate identity')
    assignment(b['assignment_id']); d=plan['groups'][b['group_id']]['dimension']
    fr=expected_rows(plan,b['group_id'],'FIT',s)
    check(b['fit_rows']==fingerprint(fr) and b['dev_rows']==expected_rows(plan,b['group_id'],'DEV-S',s),'Candidate source rows mismatch')
    stats=load_statistics(plan,b['statistics'],b['group_id'],fr,s)
    coef=s.array(ref,'coefficients.npy','<f8',(25,d)); intercept=s.array(ref,'intercepts.npy','<f8',(25,))
    valid=s.array(ref,'estimable.npy','|b1',(25,)); logits=s.array(ref,'dev_logits.npy','<f8',(len(b['dev_rows']),25))
    check(len(b['records'])==25 and all(bool(valid[k])==(r['status']=='fitted') for k,r in enumerate(b['records'])),'Fit/status mismatch')
    for k,r in enumerate(b['records']):
        if valid[k]: check(r['attempts'] and r['attempts'][-1]['converged'],'Nonconverged candidate')
        else: check(not np.any(coef[k]) and intercept[k]==0,'Unestimable padding mismatch')
    check(b['target_identity']==fingerprint({'FIT':donor_map(plan,b['assignment_id'],'FIT'),'DEV-S':donor_map(plan,b['assignment_id'],'DEV-S')}),'Target assignment mismatch')
    yy,kk=targets(plan,b['dev_rows'],b['assignment_id'])
    expected=metrics(logits,yy,kk,valid,contexts(b['dev_rows']),[list(plan['categories']).index(c) for c in plan['primary']])
    check(expected==b['metrics'],'Changed selection metrics')
    return b,coef,intercept,valid


def select_system(plan,group_id,assignment_id,candidate_refs,producer):
    check(len(candidate_refs)==4,'Four complete candidates required')
    store=Store(plan.root,plan.protocol); scores={}; refs={}; scaler=None
    for ref in candidate_refs:
        b,*_=candidate(plan,ref,store)
        check(b['group_id']==group_id and b['assignment_id']==assignment_id and b['C'] not in scores,'Candidate selection mismatch')
        check(scaler is None or scaler==b['statistics'],'Candidate source scalers differ')
        scaler=b['statistics']; scores[b['C']]=b['metrics']['selection_score']; refs[b['C']]=store.ref(ref)
    C=choose(scores)
    return write(plan,'checkpoints/'+plan['fit_identity'][:16]+'/selections','selection',
                 {'group_id':group_id,'assignment_id':assignment_id,'scores':{str(c):v for c,v in scores.items()},
                  'selected_C':C,'selected':refs[C],'candidates':[refs[c] for c in CS]}, {},producer)


class FittedSystem:
    def __init__(self,plan,ref,role,expected):
        check(role in ROLES,'Stage 3 scoring role required')
        self.plan,self.role=plan,role
        b,self.store=_body(plan,ref,'selection')
        for k,v in expected.items(): check(b.get(k)==v,'Unexpected selected-system identity')
        check(b['selected_C']==choose({float(k):v for k,v in b['scores'].items()}),'Selection tie/C mismatch')
        cb,self.coef,self.intercept,self.valid=candidate(plan,b['selected'],self.store)
        check(cb['C']==b['selected_C'] and all(cb[k]==b[k] for k in ('group_id','assignment_id')),'Selected candidate mismatch')
        check((role!='g0_score' or b['assignment_id']!='true'),'Ordinary SCREEN scores are closed')
        self.body,self.candidate=b,cb
        # Scoring can load small statistics without any source feature matrix.
        sb,_=_body(plan,cb['statistics'],'statistics',self.store,fit=False)
        check(sb['input_identity']==stats_identity(plan,cb['group_id'],expected_rows(plan,cb['group_id'],'FIT',self.store)), 'Selected statistics source mismatch')
        d=self.coef.shape[1]
        self.stats={k:self.store.array(cb['statistics'],k+'.npy','<f8',(d,)) for k in ('mean','variance','scale')}
        self.stats['center']=sb['center']
        self.stats['reference']=cb['statistics']

    def score(self,session):
        check(session.plan is self.plan and session.role==self.role,'Score session role/plan mismatch')
        rows,x=session.read(self.body['group_id'])
        z=np.asarray(_transformed(session,self.body['group_id'],self.stats)@self.coef.T)+self.intercept
        return rows,z,self.valid


def load_fitted_system(selection_manifest,root,plan,role,expected):
    check(__import__('pathlib').Path(root).resolve()==plan.root,'Root mismatch')
    check({'group_id','assignment_id'}<=set(expected),'Explicit selected identity required')
    return FittedSystem(plan,selection_manifest,role,expected)
