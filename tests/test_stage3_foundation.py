"""CPU fixture checks, not production Stage 3 signal or a completed G0 gate."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from scipy import sparse
from sklearn.metrics import roc_auc_score
from src.common.contracts import fingerprint, canonical_bytes, ModelInput, ContractError
from src.common.storage import publish
from src.stage2.schema import plain, jsonl, specification
from src.stage2.plans import compile_inventory, text_recipes, planned_rows
from src.stage3.artifacts import Store, array_bytes
from src.stage3.data import suite_inventory, FitPlan, runtime, numerical_identity, open_numeric_session, compile_fit_plan, load_fit_plan
from src.stage3.engine import statistics, transform, objective_gradient, fit_heads, choose, metrics
from src.stage3.readouts import (donor_map,targets,load_or_publish_statistics,fit_candidate_block,
                                select_system,load_fitted_system,publish_target_assignments)
from src.stage3.controls import auc_batch, evaluate_g0

REPO=Path(__file__).resolve().parents[1]
PROTOCOL=json.loads((REPO/'configs/protocol.json').read_text(encoding='utf-8'))


class Fixture:
    def __init__(self,root):
        self.root=root; self.protocol=copy.deepcopy(PROTOCOL)
        self.protocol['split_sizes']={s:4 for s in ('FIT','DEV-S','DEV-C','SCREEN','CONFIRM')}
        self.store=Store(root,self.protocol)
        self.producer={'code_commit':'0'*40,'storage_scope':'local-development','software_versions':runtime()}
        self.counter=0

    def put(self,kind,name,body,files=None):
        self.counter+=1
        meta=dict(kind=kind,status='passed',run_id='fixture'+str(self.counter),code_commit='0'*40,
                  protocol_fingerprint=fingerprint(self.protocol),dataset_fingerprint='e'*64,model_fingerprint=None,
                  seeds={},software_versions={'fixture':'synthetic'},upstream_artifacts=[],storage_scope='local-development')
        folder=publish(self.root,'fixture/'+str(self.counter),meta,{name:canonical_bytes(body)+b'\n',**(files or {})})
        return self.store.ref(folder/'_COMPLETE.json')

    def build(self):
        target=[]; records=[]; partitions={}
        for si,split in enumerate(self.protocol['split_sizes']):
            partitions[split]=[]
            for j in range(4):
                image=f'{si*4+j+1:016x}'; partitions[split].append(image)
                values=[j%2]*25
                if split=='SCREEN' and j==0: values[16]=None
                target.append({'image_id':image,'split':split,'values':values})
                for lang in ('en','hi','bn','te'):
                    records.append(ModelInput(image,'never-open.jpg','a'*64,lang,image+lang,'native caption',split))
        rb=jsonl([vars(r) for r in records]); tb=jsonl(target); sb=canonical_bytes(partitions)+b'\n'
        source={'category_order':self.protocol['category_order'],'records_sha256':hashlib.sha256(rb).hexdigest(),
                'targets_sha256':hashlib.sha256(tb).hexdigest(),'split_manifest_sha256':hashlib.sha256(sb).hexdigest()}
        dataset=self.put('stage1_dataset','dataset.json',dict(source,dataset_fingerprint='e'*64),{'records.jsonl':rb,'targets.jsonl':tb,'splits.json':sb})
        validation=self.put('stage1_data_validation','report.json',{'fixture':True})
        dp=dict(compile_inventory(SimpleNamespace(records=records)),kind='stage2_data_plan',schema='stage2.plan.v1',
                specification=specification(),protocol_fingerprint=fingerprint(self.protocol),dataset_fingerprint='e'*64,
                source_identity=source,target_reference={'artifact':dataset,'file':'targets.jsonl','primary_categories':self.protocol['primary_categories']},
                references={'dataset':dataset,'data_validation':validation},recipes=text_recipes(),confirm_processing=False)
        dp['plan_fingerprint']=fingerprint(dp); data=self.put('stage2_data_plan','plan.json',dp)
        models={}; resources={}; releases=[]
        for model in ('qwen','gemma'):
            sites={f'decoder:{n}:{p}':{'depth':n,'pooling':p} for n in (1,2,3,4) for p in ('last_caption','caption_mean')}
            sites.update({'encoder:None:visual_mean':{'stage':'encoder'},'projector:None:visual_mean':{'stage':'projector'}})
            for site,spec in sites.items(): spec.update(module_path='fixture.'+site,pooling=site.split(':')[-1],output='block output')
            shared={'model':model,'model_config':{'config_fingerprint':fingerprint(model)},'execution_identity':{'site_specs':sites,'execution_fingerprint':fingerprint(model)},
                    'dimensions':{s:2 for s in sites},'adapter':{'fixture':True}}
            receipt=dict(shared); receipt['readiness_fingerprint']=fingerprint(receipt)
            rr=self.put('stage2_model_readiness','readiness.json',receipt)
            mp=dict(dp,**shared); mp.pop('plan_fingerprint'); mp.update(kind='stage2_model_plan',references={'data_plan':data,'readiness':rr},
                 required_sites={c:sorted(s for s in sites if s.startswith('decoder:')==(c!='image_only')) for c in ('image_caption','caption_only','image_only')})
            mp['plan_fingerprint']=fingerprint(mp); mr=self.put('stage2_model_plan','plan.json',mp); models[model]=mr
            resources[model]=self.put('stage2_resource_report','report.json',{'fixture':True})
            for split in ('FIT','DEV-S','DEV-C','SCREEN'):
                u=next(u for u in mp['work_units'] if u['split']==split)
                for condition in mp['required_sites']:
                    rows=planned_rows(mp,u['chunk_id'],condition); cat={}; files={'rows.jsonl':jsonl(rows)}; digests={}
                    for index,site in enumerate(mp['required_sites'][condition]):
                        a=np.array([[int(r['image_id'],16)%2,1.] for r in rows],dtype='<f4')
                        name=f'site_{index:03d}.npy'; files[name]=array_bytes(a)
                        cat[site]={'file':name,'dimension':2,'spec':sites[site]}
                        digests[site]=[hashlib.sha256(v.tobytes()).hexdigest() for v in a]
                    cr=self.put('stage2_feature_chunk','chunk.json',dict(plan=mr,plan_fingerprint=mp['plan_fingerprint'],
                      execution_identity=shared['execution_identity'],model_config=shared['model_config'],source_identity=source,
                      chunk_id=u['chunk_id'],split=split,condition=condition,sites=cat,vector_digests=digests),files)
                    releases.append(self.put('stage2_feature_release','release.json',dict(plan=mr,plan_fingerprint=mp['plan_fingerprint'],
                       condition=condition,split=split,sites=mp['required_sites'][condition],chunks=[cr],row_count=len(rows))))
        families={}
        for recipe in text_recipes():
            fr=[dict(r,row_index=i,image_loss_weight=.25 if recipe['language'] is None else 1.) for i,r in enumerate(
                r for r in dp['rows']['caption_only'] if r['split']=='FIT' and (recipe['language'] is None or r['language']==recipe['language']))]
            spec=canonical_bytes({'fixture':True})
            pre=dict(plan=data,recipe=recipe,versions={'fixture':True},fit_rows_sha256=fingerprint(fr),dimension=2,
                     preprocessor_files={'preprocessor.json':hashlib.sha256(spec).hexdigest()})
            pre['preprocessing_fingerprint']=fingerprint({k:pre[k] for k in ('recipe','versions','fit_rows_sha256','dimension','preprocessor_files')})
            pr=self.put('stage2_text_preprocessor','fit.json',pre,{'preprocessor.json':spec}); splits={}
            for split in ('FIT','DEV-S','DEV-C','SCREEN'):
                rows=[dict(r,row_index=i,image_loss_weight=.25 if recipe['language'] is None else 1.) for i,r in enumerate(
                    r for r in dp['rows']['caption_only'] if r['split']==split and (recipe['language'] is None or r['language']==recipe['language']))]
                files={'rows.jsonl':jsonl(rows)}
                if recipe['family']=='length': files['matrix.npy']=array_bytes(np.ones((len(rows),2),dtype='<f4'))
                else:
                    x=sparse.csr_matrix(np.full((len(rows),2),1/np.sqrt(2),dtype='<f4'))
                    files.update({n+'.npy':array_bytes(v) for n,v in [('data',x.data),('indices',x.indices),('indptr',x.indptr)]})
                splits[split]=self.put('stage2_text_release','text_release.json',{'split':split,'preprocessor':pr,
                  'preprocessing_fingerprint':pre['preprocessing_fingerprint'],'shape':[len(rows),2]},files)
            rid=recipe['recipe_fingerprint']
            families[rid]=self.put('stage2_text_family','family.json',dict(recipe=recipe,plan=data,preprocessor=pr,
               preprocessing_fingerprint=pre['preprocessing_fingerprint'],family_complete=True,splits=splits))
        suite=self.put('stage2_suite_index','report.json',dict(schema='stage2.suite.v1',status='passed',blockers=[],data_plan=data,
                       model_plans=models,resource_reports=resources,feature_releases=releases,text_families=families))
        self.suite=suite
        return compile_fit_plan(suite,self.root,self.protocol,self.producer)


class MathTests(unittest.TestCase):
    def test_independent_process_readback_and_corruption(self):
        import subprocess,sys
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            def bundle(kind,body,name):
                meta=dict(kind=kind,status='passed',run_id=name,code_commit='0'*40,protocol_fingerprint=fingerprint(PROTOCOL),
                          dataset_fingerprint='e'*64,model_fingerprint=None,seeds={},software_versions={'fixture':'synthetic'},upstream_artifacts=[])
                p=publish(root,name,meta,{'body.json':canonical_bytes(body)})
                return Store(root,PROTOCOL).ref(p/'_COMPLETE.json')
            gate=bundle('stage3_g0_gate',{'fixture':True},'gate')
            pending=bundle('stage3_registry_readback',{'candidates':[],'selections':[],'g0_gate':gate},'pending')
            command=[sys.executable,'-m','src.stage3.operations','readback','--phase1-root',str(root),'--manifest',str(root/pending['path'])]
            result=subprocess.run(command,cwd=REPO,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            (root/gate['path']).with_name('body.json').write_bytes(b'bad')
            self.assertNotEqual(subprocess.run(command,cwd=REPO,text=True,capture_output=True).returncode,0)
    def test_convergence_retry_is_fixed_and_failure_blocks(self):
        from sklearn.exceptions import ConvergenceWarning
        import warnings
        fake=SimpleNamespace(classes_=np.array([0,1]),coef_=np.zeros((1,2)),intercept_=np.zeros(1),n_iter_=np.array([1000]))
        def fail(*a,**kw):
            warnings.warn('fixture exhaustion',ConvergenceWarning)
            return fake
        fake.fit=fail
        with patch('src.stage3.engine.LogisticRegression',return_value=fake) as maker:
            with self.assertRaises(ContractError):
                fit_heads(np.ones((4,2)),np.tile([0.,1,0,1],(25,1)).T,np.ones((4,25),dtype=bool),np.ones(4),.1)
        self.assertEqual([c.kwargs['max_iter'] for c in maker.call_args_list],[1000,5000])

    def test_stage2_reference_feature_reader_equivalence(self):
        from test_stage2_foundation import ArtifactFixture
        from src.stage2.features import publish_feature_chunk,publish_release_index,load_feature_release
        from src.stage2.schema import reference
        f=ArtifactFixture(methodName='runTest');f.setUp()
        try:
            chunk,files=f.payload()
            cr=publish_feature_chunk(f.root,f.plan,chunk,'caption_only','equivalence',files,f.producer())
            rr=publish_release_index(f.root,f.plan,'caption_only','FIT',[cr],'equivalence-release')
            old=load_feature_release(rr,f.root,PROTOCOL,'source_fit',{'plan_fingerprint':f.plan['plan_fingerprint'],'condition':'caption_only','split':'FIT'})
            rows,a=old.read_matrix('decoder:1:last_caption','bn')
            group={'source':'vlm','model':'qwen','condition':'caption_only','site':'decoder:1:last_caption','language':'bn','dimension':2,
                   'model_plan':plain(f.plan['manifest_reference']),'releases':{'FIT':reference(rr,f.root)}}
            body={'groups':{'fixture':group},'targets':{r.image_id:{'split':r.split} for r in f.inputs}}
            plan=FitPlan(body,f.root,PROTOCOL,{'path':'fixture/_COMPLETE.json','sha256':'0'*64})
            with patch('src.stage3.data._plan',return_value=plain(f.plan)):
                newrows,b=open_numeric_session(plan,f.root,['fixture'],'source_fit').read('fixture')
            self.assertEqual(plain(rows),plain(newrows));np.testing.assert_array_equal(a,b)
        finally:f.tearDown()

    def test_objective_gradient_intercept(self):
        x=np.array([[1.,2.],[-1,0],[0,1]]); y=np.array([1.,0,1]); a=np.array([.25,.5,.25]); w=np.array([.1,-.2]); b=.3
        loss,g,gb=objective_gradient(x,y,a,w,b,.1); h=1e-6
        for j in range(2):
            ww=w.copy();ww[j]+=h; vv=w.copy();vv[j]-=h
            self.assertAlmostEqual(g[j],(objective_gradient(x,y,a,ww,b,.1)[0]-objective_gradient(x,y,a,vv,b,.1)[0])/(2*h),places=7)
        self.assertAlmostEqual(gb,(objective_gradient(x,y,a,w,b+h,.1)[0]-objective_gradient(x,y,a,w,b-h,.1)[0])/(2*h),places=7)

    def test_sparse_dense_scaling_and_fractional_replication(self):
        x=np.array([[0.,2,1],[1,2,0],[2,2,3],[3,2,0]])
        d=statistics(x,np.ones(4)); s=statistics(sparse.csr_matrix(x),np.ones(4))
        np.testing.assert_allclose(d['mean'],s['mean']);np.testing.assert_allclose(d['variance'],s['variance'])
        self.assertEqual(d['scale'][1],1.)
        w=np.array([.2,.5,-.3]); b=.8
        np.testing.assert_allclose(transform(x,d)@w+b,transform(sparse.csr_matrix(x),s)@w+b-d['mean']@(w/d['scale']))
        y=np.tile([0.,1,0,1],(25,1)).T;known=np.ones_like(y,dtype=bool)
        c,i,_,_=fit_heads(transform(x,d),y,known,np.ones(4),.1)
        xx=np.repeat(transform(x,d),4,axis=0); yy=np.repeat(y,4,axis=0)
        c2,i2,_,_=fit_heads(xx,yy,np.ones_like(yy,dtype=bool),np.full(16,.25),.1)
        np.testing.assert_allclose(c,c2,atol=1e-12);np.testing.assert_allclose(i,i2,atol=1e-12)

    def test_masks_unestimable_and_ties(self):
        x=np.zeros((4,2),dtype='f8'); y=np.zeros((4,25)); k=np.ones_like(y,dtype=bool)
        c,i,v,r=fit_heads(x,y,k,np.ones(4),.01)
        self.assertFalse(v.any());self.assertFalse(c.any())
        self.assertEqual(choose({.01:.5,.1:.5+1e-13,1.:.4,10.:.3}),.01)
        with self.assertRaises(ContractError): choose({.01:None})

    def test_vectorized_auc_with_ties_and_moving_masks(self):
        z=np.array([[0.,1.],[0,2],[2,0],[3,3],[4,2],[1,4]])
        rng=np.random.default_rng(12)
        y=np.tile(np.array([[0,1,0,1,0,1],[1,0,1,0,1,0]]).T,(5,1,1))
        k=np.ones_like(y,dtype=bool)
        for b in range(5): k[b,b,0]=False
        got=auc_batch(z,y,k)
        for b in range(5):
            for j in range(2): self.assertAlmostEqual(got[b,j],roc_auc_score(y[b,k[b,:,j],j],z[k[b,:,j],j]))


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.f=Fixture(Path(self.temp.name));self.plan=self.f.build()
    def tearDown(self): self.temp.cleanup()
    def test_inventory_reload_and_no_image_access(self):
        self.assertEqual(len(self.plan['groups']),145)
        with patch('src.common.data.load_dataset',side_effect=AssertionError('images forbidden')):
            reloaded=load_fit_plan(self.plan.reference,self.plan.root,self.plan.protocol)
        self.assertEqual(reloaded['fit_identity'],self.plan['fit_identity'])
        self.assertEqual(len(self.plan['profiling_groups']),6)
    def test_numeric_sessions_role_cache_and_corruption(self):
        gids=[g for g,b in self.plan['groups'].items() if b['source']=='vlm' and b['language']=='hi'][:2]
        s=open_numeric_session(self.plan,self.plan.root,gids,'source_fit');rows,x=s.read(gids[0]); reads=dict(s.store.reads)
        self.assertEqual(len(rows),4);self.assertEqual(s.read(gids[0])[1].shape,x.shape);self.assertEqual(reads,dict(s.store.reads))
        with self.assertRaises(ContractError): s.read(gids[0],'g0_score')
        with self.assertRaises(ValueError): x[0,0]=123
        for gid in gids: s.read(gid)
        self.assertTrue(all(n==1 for n in s.store.reads.values()))
        p=next(self.plan.root/r for r in s.store.reads if r.endswith('site_000.npy'))
        p.write_bytes(b'corrupt')
        with self.assertRaises(ContractError): open_numeric_session(self.plan,self.plan.root,gids,'source_fit').read(gids[0])
    def test_all_text_families_read_sparse_dense(self):
        for gid,g in self.plan['groups'].items():
            if g['source']!='e03':continue
            rows,x=open_numeric_session(self.plan,self.plan.root,[gid],'source_fit').read(gid)
            self.assertEqual(len(rows),16 if g['language'] is None else 4)
            self.assertEqual(sparse.issparse(x),g['recipe']['family']!='length')
    def test_true_control_fit_select_reload_and_reuse(self):
        gid=next(g for g,b in self.plan['groups'].items() if b['source']=='vlm' and b['language']=='en')
        sessions={r:open_numeric_session(self.plan,self.plan.root,[gid],r) for r in ('source_fit','source_select')}
        stats=load_or_publish_statistics(self.plan,gid,sessions['source_fit'],self.f.producer,[])
        with patch('src.stage3.readouts.statistics',side_effect=AssertionError('must reuse')):
            reused=load_or_publish_statistics(self.plan,gid,sessions['source_fit'],self.f.producer,[stats['reference']])
        self.assertEqual(stats['reference'],reused['reference'])
        refs=[fit_candidate_block(self.plan,gid,'true',.01,sessions,stats,self.f.producer)]
        with patch('src.stage3.readouts.transform',side_effect=AssertionError('reuse source working arrays')):
            refs.extend(fit_candidate_block(self.plan,gid,'true',c,sessions,stats,self.f.producer) for c in (.1,1.,10.))
        selected=select_system(self.plan,gid,'true',refs,self.f.producer)
        reader=load_fitted_system(selected,self.plan.root,self.plan,'source_select',{'group_id':gid,'assignment_id':'true'})
        rows,z,v=reader.score(sessions['source_select']);self.assertEqual(z.shape,(4,25));self.assertTrue(v.all())
        with self.assertRaises(ContractError): load_fitted_system(selected,self.plan.root,self.plan,'g0_score',{'group_id':gid,'assignment_id':'true'})
        with self.assertRaises(ContractError): select_system(self.plan,gid,'true',refs[:-1],self.f.producer)
        control=fit_candidate_block(self.plan,gid,'seed0',.01,sessions,stats,self.f.producer)
        self.assertNotEqual(control,refs[0])
        from src.stage3.recovery import publish_run_index,load_run_index,read_completed
        index=publish_run_index(self.plan,{'candidates':refs,'selections':[selected],'statistics':[stats['reference']]},self.f.producer)
        entries=load_run_index(self.plan,index)
        self.assertEqual(len(entries['candidates']),4)
        read_completed(self.plan,refs[0],'candidate',{'group_id':gid,'assignment_id':'true','C':.01})
        with self.assertRaises(ContractError):read_completed(self.plan,refs[0],'candidate',{'assignment_id':'seed0'})
    def test_assignments_masks_grouping_and_partial_gate(self):
        gid=next(g for g,b in self.plan['groups'].items() if b['source']=='e03' and b['language'] is None)
        rows,_=open_numeric_session(self.plan,self.plan.root,[gid],'g0_score').read(gid)
        for seed in ('seed0','seed1','seed2'):
            y,k=targets(self.plan,rows,seed)
            for image in set(r['image_id'] for r in rows):
                ix=[i for i,r in enumerate(rows) if r['image_id']==image]
                self.assertTrue(np.all(y[ix]==y[ix[0]]));self.assertTrue(np.all(k[ix]==k[ix[0]]))
            self.assertEqual(int((~k[:,16]).sum()),4)
        with self.assertRaises(ContractError): donor_map(self.plan,'seed0','CONFIRM')
        ref=publish_target_assignments(self.plan,self.plan.root,self.f.producer)
        result=evaluate_g0(self.plan,[],[ref],[],self.f.producer)
        self.assertFalse(result['complete']);self.assertIsNone(result['gate'])
        from src.stage3.validation import validate_stage3
        with self.assertRaises(ContractError):validate_stage3(self.plan,[],[],None,self.f.producer)
    def test_null_group_resume_and_missing_seed_guard(self):
        from src.stage3.controls import score_g0
        gid=next(g for g,b in self.plan['groups'].items() if b['source']=='vlm' and b['language']=='en')
        sessions={r:open_numeric_session(self.plan,self.plan.root,[gid],r) for r in ('source_fit','source_select')}
        stats=load_or_publish_statistics(self.plan,gid,sessions['source_fit'],self.f.producer,[])
        selections=[]
        for aid in ('seed0','seed1','seed2'):
            refs=[fit_candidate_block(self.plan,gid,aid,c,sessions,stats,self.f.producer) for c in (.01,.1,1.,10.)]
            selections.append(select_system(self.plan,gid,aid,refs,self.f.producer))
        scores=score_g0(self.plan,selections,open_numeric_session(self.plan,self.plan.root,[gid],'g0_score'),self.f.producer)
        assignments=publish_target_assignments(self.plan,self.plan.root,self.f.producer)
        with self.assertRaises(ContractError):evaluate_g0(self.plan,scores[:2],[assignments],[],self.f.producer)
        result=evaluate_g0(self.plan,scores,[assignments],[],self.f.producer)
        self.assertFalse(result['complete']);self.assertEqual(set(result['groups']),{gid})
        with patch('src.stage3.controls._null_group',side_effect=AssertionError('must reuse')):
            resumed=evaluate_g0(self.plan,scores,[assignments],list(result['groups'].values()),self.f.producer)
        self.assertEqual(result,resumed)
        with self.assertRaises(ContractError):evaluate_g0(self.plan,[],[assignments],list(result['groups'].values())*2,self.f.producer)
    def test_dependency_mutation_blocks(self):
        p=self.plan.root/self.f.suite['path'];p.write_bytes(b'bad')
        with self.assertRaises(ContractError):load_fit_plan(self.plan.reference,self.plan.root,self.plan.protocol)


if __name__=='__main__':unittest.main()
