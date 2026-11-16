"""Accepted-suite metadata compiler and role-scoped numeric sessions."""
import hashlib
from collections.abc import Mapping
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from src.common.contracts import fingerprint, TargetRecord, LANGUAGES
from src.stage2.schema import check, frozen, plain, specification
from src.stage2.plans import text_recipes, planned_rows
from .artifacts import Store, write

CS = (0.01, 0.1, 1.0, 10.0)
ROLES = {'source_fit':'FIT', 'source_select':'DEV-S', 'g0_score':'SCREEN'}


def runtime():
    import platform
    from importlib.metadata import version
    from threadpoolctl import threadpool_info
    result = {'python':platform.python_version(), **{n:version(n) for n in ('numpy','scipy','scikit-learn')}}
    check(all(result[k] == v for k,v in {'numpy':'2.1.3','scipy':'1.18.1','scikit-learn':'1.6.1'}.items()),
          'Use the pinned CPU requirements')
    pools = threadpool_info()
    check(all(p['num_threads'] == 1 for p in pools), 'Set BLAS/OpenMP threads to one before imports')
    result['threadpools'] = [{k:p.get(k) for k in ('internal_api','prefix','version','num_threads','threading_layer','architecture')}
                             for p in pools]
    return result


def numerical_identity():
    import inspect
    from .readouts import weights, donor_map, targets
    from src.common.storage import file_sha256
    return {'engine_sha256': file_sha256(Path(__file__).with_name('engine.py')),
            'target_weight_semantics':fingerprint([inspect.getsource(f) for f in (weights,donor_map,targets)]),
            'recipe':'sum-weighted-logloss-L2-unpenalized-intercept.v1',
            'C':list(CS), 'tol':1e-8, 'iterations':[1000,5000], 'variance_threshold':1e-12}


class FitPlan(Mapping):
    def __init__(self, body, root, protocol, reference, store=None):
        self.body, self.root, self.protocol = frozen(body), Path(root).resolve(), protocol
        self.reference = frozen(reference)
        self.store = store or Store(root, protocol)
    def __getitem__(self,k): return self.body[k]
    def __iter__(self): return iter(self.body)
    def __len__(self): return len(self.body)


def _plan(store, ref, kind):
    b = store.json(ref,'plan.json',kind)
    raw = {k:v for k,v in b.items() if k not in ('plan_fingerprint','job_id')}
    check(fingerprint(raw) == b['plan_fingerprint'] and b['kind'] == kind
          and b['specification'] == specification(), 'Stage 2 plan identity mismatch')
    return b


def suite_inventory(suite, store):
    """No upstream image/model calls, including on metadata cache misses."""
    suite_ref = store.ref(suite)
    s = store.json(suite_ref,'report.json','stage2_suite_index')
    check(s['schema']=='stage2.suite.v1' and s['status']=='passed' and not s['blockers'], 'Suite not accepted')
    check(set(s['model_plans']) == set(s['resource_reports']) == {'qwen','gemma'}, 'Both models required')
    data = _plan(store,s['data_plan'],'stage2_data_plan')
    store.dataset=data['dataset_fingerprint']
    check(store.marker(suite_ref)[2]['dataset_fingerprint']==store.dataset,'Suite/dataset mismatch')
    check(data['protocol_fingerprint'] == fingerprint(store.protocol) and not data['confirm_processing'], 'Data identity/role mismatch')
    check(data['recipes'] == text_recipes(), 'Unexpected E03 recipes')
    dp = data['references']['dataset']
    dataset = store.json(dp,'dataset.json','stage1_dataset')
    check(dataset['dataset_fingerprint'] == data['dataset_fingerprint'] and
          data['source_identity'] == {k:dataset[k] for k in ('records_sha256','targets_sha256','split_manifest_sha256','category_order')}, 'Dataset metadata mismatch')
    check(dataset['category_order'] == store.protocol['category_order'], 'Category order mismatch')
    store.complete(data['references']['data_validation'],'stage1_data_validation')
    target_bytes = store.payload(dp,'targets.jsonl')
    check(hashlib.sha256(target_bytes).hexdigest() == dataset['targets_sha256'], 'Target hash mismatch')
    targets = {}
    for line in target_bytes.splitlines():
        r = __import__('json').loads(line)
        t = TargetRecord(r['image_id'],r['split'],tuple(r['values']))
        check(t.image_id not in targets, 'Duplicate target image')
        targets[t.image_id] = r
    partitions = store.json(dp,'splits.json')
    for split in store.protocol['split_sizes']:
        ids = sorted(r['image_id'] for r in targets.values() if r['split']==split)
        check(ids == sorted(partitions[split]) and len(ids)==store.protocol['split_sizes'][split], 'Split/target coverage mismatch')
    check(hashlib.sha256(store.payload(dp,'splits.json')).hexdigest() == dataset['split_manifest_sha256'], 'Split hash mismatch')
    # Records authenticate original row keys/caption provenance, without touching images.
    records_bytes = store.payload(dp,'records.jsonl')
    check(hashlib.sha256(records_bytes).hexdigest()==dataset['records_sha256'], 'Record hash mismatch')
    from src.common.contracts import ModelInput
    from src.stage2.plans import compile_inventory
    from types import SimpleNamespace
    records = [ModelInput(**__import__('json').loads(l)) for l in records_bytes.splitlines()]
    expected = compile_inventory(SimpleNamespace(records=records))
    check(all(data[k]==v for k,v in expected.items()), 'Data plan rows disagree with records')
    for r in records:
        check(r.image_id in targets and r.split==targets[r.image_id]['split'], 'Record target join mismatch')
    models, releases, groups = {}, {}, {}
    for model, ref in s['model_plans'].items():
        print(__import__('json').dumps({'event':'verify_model_metadata','model':model}),flush=True)
        m = _plan(store,ref,'stage2_model_plan')
        check(m['model']==model and m['references']['data_plan']==s['data_plan'], 'Model/shared plan mismatch')
        check(all(m[k]==data[k] for k in ('dataset_fingerprint','source_identity','rows','work_units','target_reference')), 'Model data mismatch')
        check([len(m['required_sites'][c]) for c in ('image_caption','caption_only','image_only')]==[8,8,2], 'Unexpected physical site inventory')
        receipt = store.json(m['references']['readiness'],'readiness.json','stage2_model_readiness')
        check(receipt['readiness_fingerprint']==fingerprint({k:v for k,v in receipt.items() if k!='readiness_fingerprint'}), 'Readiness fingerprint mismatch')
        check(all(receipt[k]==m[k] for k in ('model','model_config','execution_identity','dimensions','adapter')), 'Model/receipt mismatch')
        store.complete(s['resource_reports'][model],'stage2_resource_report')
        models[model]=m
    check(len(s['feature_releases']) == 24, 'Expected 24 feature releases')
    for ref in s['feature_releases']:
        b = store.json(ref,'release.json','stage2_feature_release')
        candidates = [name for name in models if b['plan']==s['model_plans'][name]]
        check(len(candidates)==1, 'Foreign feature release')
        model, = candidates
        m = models[model]
        key=(model,b['condition'],b['split'])
        check(key not in releases and b['condition'] in m['required_sites'] and b['split'] in ('FIT','DEV-S','DEV-C','SCREEN'), 'Duplicate/invalid release')
        units=[u for u in m['work_units'] if u['split']==b['split']]
        check(b['sites']==m['required_sites'][b['condition']] and b['plan_fingerprint']==m['plan_fingerprint']
              and len(b['chunks'])==len(units) and b['row_count']==sum(len(u['rows'][b['condition']]) for u in units), 'Release metadata coverage mismatch')
        releases[key]=ref
    for model,m in models.items():
        for condition in ('image_caption','caption_only','image_only'):
            for split in ('FIT','DEV-S','DEV-C','SCREEN'):
                check((model,condition,split) in releases, 'Missing release')
            for site in m['required_sites'][condition]:
                for language in ((None,) if condition=='image_only' else LANGUAGES):
                    name='|'.join([model,condition,site,language or 'image'])
                    gid=fingerprint(name)
                    groups[gid]={'group_id':gid,'name':name,'source':'vlm','model':model,'condition':condition,
                                 'site':site,'language':language,'dimension':m['dimensions'][site],
                                 'model_plan':s['model_plans'][model],
                                 'releases':{split:releases[(model,condition,split)] for split in ('FIT','DEV-S','DEV-C','SCREEN')}}
    check(len(s['text_families'])==13, 'Expected 13 E03 families')
    for rid,ref in s['text_families'].items():
        b=store.json(ref,'family.json','stage2_text_family')
        pre=store.json(b['preprocessor'],'fit.json','stage2_text_preprocessor')
        check(b['recipe'] in data['recipes'] and b['recipe']['recipe_fingerprint']==rid and b['plan']==s['data_plan']
              and b['family_complete'] and set(b['splits'])=={'FIT','DEV-S','DEV-C','SCREEN'}, 'E03 family mismatch')
        check(pre['plan']==s['data_plan'] and pre['recipe']==b['recipe'] and pre['preprocessing_fingerprint']==b['preprocessing_fingerprint'], 'Preprocessor/family mismatch')
        check(pre['preprocessing_fingerprint']==fingerprint({k:pre[k] for k in ('recipe','versions','fit_rows_sha256','dimension','preprocessor_files')}), 'Preprocessor identity mismatch')
        for name in pre['preprocessor_files']: store.payload(b['preprocessor'],name)
        gid=fingerprint(['e03',rid])
        groups[gid]={'group_id':gid,'name':'e03|'+rid,'source':'e03','family':ref,'recipe':b['recipe'],
                     'language':b['recipe']['language'],'dimension':pre['dimension'],'releases':b['splits']}
    check(len(groups)==145, 'Expected exactly 145 systems')
    print(__import__('json').dumps({'event':'suite_metadata_verified','groups':len(groups),'images_opened':0}),flush=True)
    return {'suite':suite_ref,'data_plan':s['data_plan'],'model_plans':s['model_plans'],
            'dataset_fingerprint':data['dataset_fingerprint'],'groups':groups,
            'targets':{k:v for k,v in targets.items() if v['split']!='CONFIRM'},
            'categories':dataset['category_order'],'primary':store.protocol['primary_categories']}


def compile_fit_plan(suite_manifest, root, protocol, producer):
    from src.common.storage import publish, new_run_id
    from src.common.contracts import canonical_bytes
    store=Store(root,protocol)
    body=suite_inventory(suite_manifest,store)
    actual=runtime()
    check(producer['software_versions']==actual, 'Record actual pinned numerical runtime')
    body.update(schema='stage3.plan.v1',runtime=actual,numerical=numerical_identity())
    body['fit_identity']=fingerprint({k:body[k] for k in ('suite','groups','categories','targets','numerical','runtime')})
    classes={}
    for gid,g in body['groups'].items():
        cls=g['model'] if g['source']=='vlm' else g['recipe']['family']
        classes.setdefault(cls,[]).append(g)
    body['profiling_groups']=[sorted(gs,key=lambda g:(-g['dimension'],g['group_id']))[0]['group_id'] for _,gs in sorted(classes.items())]
    run=new_run_id()
    meta=dict(kind='stage3_plan',status='passed',run_id=run,code_commit=producer['code_commit'],
              software_versions=actual,storage_scope=producer['storage_scope'],protocol_fingerprint=fingerprint(protocol),
              dataset_fingerprint=body['dataset_fingerprint'],model_fingerprint=None,seeds={'g0':[0,1,2]},upstream_artifacts=[body['suite']])
    folder=publish(Path(root),'cache/shared/stage3-plans/'+run,meta,{'body.json':canonical_bytes(body)+b'\n'})
    return FitPlan(body,root,protocol,store.ref(folder/'_COMPLETE.json'),store)


def load_fit_plan(manifest,root,protocol):
    store=Store(root,protocol)
    b=store.json(manifest,'body.json','stage3_plan')
    check(b['runtime']==runtime() and b['numerical']==numerical_identity(), 'Numerical environment/implementation changed')
    original=suite_inventory(b['suite'],store)
    check(all(b[k]==v for k,v in original.items()), 'Plan no longer matches accepted suite')
    check(b['fit_identity']==fingerprint({k:b[k] for k in ('suite','groups','categories','targets','numerical','runtime')}), 'Fit identity mismatch')
    return FitPlan(b,root,protocol,store.ref(manifest),store)


class NumericSession:
    def __init__(self,plan,groups,role):
        check(role in ROLES, 'Unauthorized Stage 3 role')
        self.plan,self.groups,self.role=plan,frozenset(groups),role
        check(self.groups and self.groups<=set(plan['groups']), 'Unknown/empty group set')
        self.store=Store(plan.root,plan.protocol)
        self.store.dataset=plan.get('dataset_fingerprint')
        self._current=None

    def read(self,gid,role=None):
        check(gid in self.groups and (role is None or role==self.role), 'Session ownership/role violation')
        if self._current is not None and self._current[0]==gid:
            return self._current[1:]
        split=ROLES[self.role]
        g=plain(self.plan['groups'][gid]); ref=g['releases'][split]
        if g['source']=='vlm':
            b=self.store.json(ref,'release.json','stage2_feature_release')
            m=_plan(self.store,g['model_plan'],'stage2_model_plan')
            expected=[u for u in m['work_units'] if u['split']==split]
            check(len(b['chunks'])==len(expected), 'Chunk coverage mismatch')
            rows,parts=[],[]
            for cr,u in zip(b['chunks'],expected):
                cb=self.store.json(cr,'chunk.json','stage2_feature_chunk')
                rr=self.store.rows(cr)
                check(cb['plan']==g['model_plan'] and cb['plan_fingerprint']==m['plan_fingerprint']
                      and cb['execution_identity']==m['execution_identity'] and cb['model_config']==m['model_config']
                      and cb['source_identity']==m['source_identity'] and cb['chunk_id']==u['chunk_id']
                      and cb['split']==split and cb['condition']==g['condition']
                      and rr==planned_rows(m,u['chunk_id'],g['condition']), 'Chunk/row provenance mismatch')
                spec=cb['sites'][g['site']]
                check(spec['dimension']==g['dimension'] and spec['spec']==m['execution_identity']['site_specs'][g['site']], 'Site mismatch')
                a=self.store.array(cr,spec['file'],'<f4',(len(rr),g['dimension']))
                check([hashlib.sha256(v.tobytes()).hexdigest() for v in a]==cb['vector_digests'][g['site']], 'Vector content mismatch')
                ix=[i for i,r in enumerate(rr) if r['language']==g['language']]
                rows.extend(rr[i] for i in ix); parts.append(a[ix])
            x=np.concatenate(parts)
        else:
            b=self.store.json(ref,'text_release.json','stage2_text_release')
            fam=self.store.json(g['family'],'family.json','stage2_text_family')
            check(b['split']==split and b['preprocessor']==fam['preprocessor'] and b['preprocessing_fingerprint']==fam['preprocessing_fingerprint'], 'Text identity mismatch')
            rows=self.store.rows(ref)
            dp=self.store.json(self.plan['data_plan'],'plan.json','stage2_data_plan')
            expected=[dict(r,row_index=i,image_loss_weight=0.25 if g['language'] is None else 1.) for i,r in
                      enumerate(r for r in dp['rows']['caption_only'] if r['split']==split and (g['language'] is None or r['language']==g['language']))]
            check(rows==expected and b['shape']==[len(rows),g['dimension']], 'Text row identity mismatch')
            if g['recipe']['family']=='length':
                x=self.store.array(ref,'matrix.npy','<f4',b['shape'])
            else:
                a=self.store.array(ref,'data.npy','<f4'); i=self.store.array(ref,'indices.npy','<i4'); p=self.store.array(ref,'indptr.npy','<i4')
                check(a.ndim==i.ndim==p.ndim==1 and len(p)==len(rows)+1 and p[0]==0 and p[-1]==len(a)==len(i)
                      and np.all(np.diff(p.astype('i8'))>=0) and np.all((i>=0)&(i<g['dimension'])) and np.all(a>0), 'Invalid CSR')
                for j in range(len(rows)): check(np.all(np.diff(i[p[j]:p[j+1]].astype('i8'))>0), 'CSR columns not unique/sorted')
                x=csr_matrix((a,i,p),shape=b['shape'])
                norms=np.asarray(x.astype('f8').multiply(x).sum(axis=1)).ravel()
                check(np.all((norms==0)|np.isclose(norms,1,atol=4e-7,rtol=4e-7)), 'TF-IDF norm mismatch')
        check(len(rows)>0 and all(r['split']==split and r['image_id'] in self.plan['targets'] for r in rows), 'Closed/unjoined rows')
        for r in rows: check(self.plan['targets'][r['image_id']]['split']==split, 'Cross-split target join')
        if hasattr(x,'data') and hasattr(x,'indices'):
            for a in (x.data,x.indices,x.indptr): a.flags.writeable=False
        else: x.flags.writeable=False
        rows=tuple(frozen(r) for r in rows)
        self._current=(gid,rows,x)
        return rows,x


def open_numeric_session(plan, root, groups, role):
    check(Path(root).resolve()==plan.root, 'Root mismatch')
    return NumericSession(plan,groups,role)
