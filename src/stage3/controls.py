"""Shared control scoring, vectorized masked AUROC and complete-family reduction."""
import numpy as np
from scipy.stats import rankdata
from src.common.contracts import fingerprint
from src.stage2.schema import check, plain
from .artifacts import Store, write
from .readouts import load_fitted_system, targets, contexts, metrics, _body, donor_map


def auc_batch(scores,y,known):
    """B x N x K labels/masks; exact average-tie ranks after exclusions."""
    check(y.shape==known.shape and y.shape[1:]==scores.shape,'AUC batch shape mismatch')
    result=np.empty((len(y),scores.shape[1]),dtype='f8')
    for k in range(scores.shape[1]):
        z=scores[:,k]; ranks=rankdata(z,method='average')
        pos=y[:,:,k]*known[:,:,k]; npos=pos.sum(axis=1); neg=known[:,:,k].sum(axis=1)-npos
        check(np.all(npos>0)&np.all(neg>0),'Undefined primary null AUROC')
        if known[:,:,k].all(): rank_sum=pos@ranks
        else:
            correction=(z[:,None]<z[None,:]).astype('f8')+0.5*(z[:,None]==z[None,:])
            adjusted=ranks[None,:]-(~known[:,:,k])@correction
            rank_sum=np.sum(pos*adjusted,axis=1)
        result[:,k]=(rank_sum-npos*(npos+1)/2)/(npos*neg)
    return result


def score_g0(plan,selected_control_refs,session,producer):
    check(session.plan is plan and session.role=='g0_score','G0 SCREEN session required')
    refs=[]
    for ref in selected_control_refs:
        b,_=_body(plan,ref,'selection')
        check(b['assignment_id'] in ('seed0','seed1','seed2'),'Control selection required')
        system=load_fitted_system(ref,plan.root,plan,'g0_score',{k:b[k] for k in ('group_id','assignment_id')})
        rows,z,valid=system.score(session)
        y,k=targets(plan,rows,b['assignment_id'])
        report=metrics(z,y,k,valid,contexts(rows),[list(plan['categories']).index(c) for c in plan['primary']])
        check(report['selection_score'] is not None,'Undefined primary control SCREEN result')
        refs.append(write(plan,'predictions/'+plan['fit_identity'][:16]+'/g0','control_scores',
                          {'selection':plain(ref),'group_id':b['group_id'],'assignment_id':b['assignment_id'],
                           'rows':plain(rows),'metrics':report}, {'logits':z,'estimable':valid},producer))
    return refs


def _null_group(plan,gid,refs,producer):
    check(len(refs)==3,'All three seed scores required before null computation')
    store=Store(plan.root,plan.protocol); seen=set(); maxima=np.zeros(1999); observed=0.
    primary=[list(plan['categories']).index(c) for c in plan['primary']]
    for ref in refs:
        b,_=_body(plan,ref,'control_scores',store)
        aid=b['assignment_id']
        check(b['group_id']==gid and aid in ('seed0','seed1','seed2') and aid not in seen,'Missing/duplicate control seed')
        seen.add(aid); rows=b['rows']; z=store.array(ref,'logits.npy','<f8',(len(rows),25))
        check(all(r['split']=='SCREEN' for r in rows),'G0 score partition mismatch')
        valid=store.array(ref,'estimable.npy','|b1',(25,)); check(valid[primary].all(),'Unestimable primary control')
        yy,kk=targets(plan,rows,aid)
        check(metrics(z,yy,kk,valid,contexts(rows),primary)==b['metrics'],'Control metric mismatch')
        obs=[]
        c=np.asarray(contexts(rows))
        for context in sorted(set(c)):
            ix=c==context
            auc=auc_batch(z[ix][:,primary],yy[None,ix][:,:,primary],kk[None,ix][:,:,primary])[0]
            obs.extend(auc.tolist()+[float(auc.mean())])
        observed=max(observed,float(np.max(np.abs(np.asarray(obs)-.5))))
        for start in range(1,2000,100):
            labels=[targets(plan,rows,aid,draw) for draw in range(start,min(start+100,2000))]
            ys=np.stack([v[0] for v in labels]); ks=np.stack([v[1] for v in labels]); values=[]
            for context in sorted(set(c)):
                ix=c==context
                a=auc_batch(z[ix][:,primary],ys[:,ix][:,:,primary],ks[:,ix][:,:,primary])
                values.extend([np.max(np.abs(a-.5),axis=1),np.abs(a.mean(axis=1)-.5)])
            maxima[start-1:start-1+len(labels)]=np.maximum(maxima[start-1:start-1+len(labels)],np.max(values,axis=0))
        print(__import__('json').dumps({'event':'null_seed_complete','group_id':gid,'assignment':aid}),flush=True)
    check(seen=={'seed0','seed1','seed2'},'All three seed scores required')
    return write(plan,'results/'+plan['fit_identity'][:16]+'/g0','null_group',
                 {'group_id':gid,'scores':[store.ref(r) for r in refs],'draw_ids':list(range(1,2000)),
                  'observed':observed,'family_identity':fingerprint({'groups':plain(plan['groups']),'primary':plain(plan['primary']),'draws':1999})},
                 {'maxima':maxima},producer)


def evaluate_g0(plan,control_score_refs,assignment_refs,resume_refs,producer):
    """Accept complete group score triples; partial total coverage yields no gate."""
    store=Store(plan.root,plan.protocol)
    check(len(assignment_refs)==1,'Exactly one shared assignment artifact required')
    ab,_=_body(plan,assignment_refs[0],'assignments',store)
    maps={a:{s:donor_map(plan,a,s) for s in ('FIT','DEV-S','SCREEN')} for a in ('true','seed0','seed1','seed2')}
    check(ab['maps']==maps and ab['assignment_identity']==fingerprint(maps),'Shared assignments mismatch')
    groups={}; completed={}
    for ref in resume_refs:
        b,_=_body(plan,ref,'null_group',store)
        gid=b['group_id']; check(gid in plan['groups'] and gid not in completed,'Duplicate/foreign null result')
        check(b['draw_ids']==list(range(1,2000)) and b['family_identity']==fingerprint({'groups':plain(plan['groups']),'primary':plain(plan['primary']),'draws':1999}),'Null family mismatch')
        store.array(ref,'maxima.npy','<f8',(1999,)); completed[gid]=ref
    for ref in control_score_refs:
        b,_=_body(plan,ref,'control_scores',store); groups.setdefault(b['group_id'],[]).append(ref)
    for gid,refs in groups.items():
        check(gid in plan['groups'],'Foreign control group')
        if gid in completed:
            old,_=_body(plan,completed[gid],'null_group',store)
            check(sorted(old['scores'],key=lambda r:r['sha256'])==sorted([store.ref(r) for r in refs],key=lambda r:r['sha256']),'Conflicting resumed score refs')
        else: completed[gid]=_null_group(plan,gid,refs,producer)
    if set(completed)!=set(plan['groups']): return {'complete':False,'groups':plain(completed),'gate':None}
    maximum=np.zeros(1999); observed=0.
    for gid,ref in completed.items():
        b,_=_body(plan,ref,'null_group',store)
        observed=max(observed,b['observed']); maximum=np.maximum(maximum,store.array(ref,'maxima.npy','<f8',(1999,)))
    p=(1+int(np.sum(maximum>=observed)))/2000
    gate=write(plan,'results/'+plan['fit_identity'][:16]+'/g0','g0_gate',
               {'groups':plain(completed),'assignments':plain(assignment_refs[0]),'observed':observed,
                'p':p,'flagged':p<=.01,'structural_checks':'passed'}, {'maxima':maximum},producer)
    return {'complete':True,'groups':plain(completed),'gate':plain(gate)}
