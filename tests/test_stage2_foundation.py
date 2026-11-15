"""Synthetic interface tests, never represented as actual VLM/E03 execution."""
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.common.contracts import (ContractError, ModelInput, LANGUAGES, ExtractionRequest,
                                  FeatureSite, fingerprint, canonical_bytes)
from src.common.storage import publish, file_sha256
from src.models.hf import HFAdapter
from src.stage2.schema import (REPO, SPLITS, CONDITIONS, frozen, plain, json_bytes,
                               jsonl, reference, metadata, publish_json)
from src.stage2.plans import compile_inventory, text_recipes, planned_rows, load_job_plan
from src.stage2.evidence import extract_with_evidence, compare_public_path, validate_token_evidence
from src.stage2.features import (publish_feature_chunk, publish_release_index,
                                load_feature_release, verify_feature_chunk)
from src.stage2.text import (text_rows, publish_text_preprocessor, publish_text_release,
                            publish_text_family, load_text_family, preprocessing_versions)
from src.stage2.readiness import discover_evidence, check_sources, assert_runtime
from src.stage2.operations import resource_projection, storage_probe

PROTOCOL = json.loads((REPO/'configs/protocol.json').read_text())


def producer(run='fixture'):
    return {'run_id': run, 'code_commit': '0'*40, 'software_versions': {'fixture': 'synthetic'},
            'storage_scope': 'local-development'}


def source_row(i=1, language='en', split='FIT'):
    return ModelInput(f'{i:016x}', f'fixture/{i}.jpg', 'a'*64, language,
                      f'{i}:{language}', 'বাংলা e\u0301  a\tb!', split)


def token(row, condition):
    caption = [] if condition == 'image_only' else [2, 3]
    images = [] if condition == 'caption_only' else [0]
    return {'row_key': list(row.row_key), 'condition': condition, 'token_count': 5,
            'caption_positions': caption, 'last_caption': caption[-1] if caption else None,
            'image_positions': images, 'role_and_other_positions': [i for i in range(5) if i not in caption+images],
            'processor_attention_mask': [1]*5, 'input_ids_sha256': 'b'*64,
            'native_serialization_sha256': 'c'*64, 'caption_alignment': 'fixture only'}


class BridgeFixture(HFAdapter):
    def __init__(self):
        self.config = {'max_input_tokens': 1024}
        self.calls = []
    def describe(self):
        return {'supported_sites': ['decoder:1:last_caption', 'decoder:1:caption_mean', 'encoder:None:visual_mean']}
    def _forward(self, row, sites, condition='image_caption', attention=False, purpose='stage1_smoke'):
        self.calls.append((row.row_key, condition, attention, purpose))
        item = token(row, condition)
        item.pop('native_serialization_sha256')
        item['native_serialization'] = 'Caption:\n'+row.caption_text
        item['decoded_processor_sequence'] = 'not used for reconstruction'
        return {s.key: [1.25, -2.5] for s in sites}, item


class InventoryTests(unittest.TestCase):
    def test_full_frozen_partitions_counts_order_and_confirm_exclusion(self):
        from src.stage2.schema import VerifiedPlan
        a=VerifiedPlan({'nested':{'value':[1]}})
        self.assertNotEqual(a,VerifiedPlan({'nested':{'value':[2]}}))
        with self.assertRaises((TypeError,AttributeError)): a._body={}
        with self.assertRaises(TypeError): a['nested']['value'][0]=2
        audit = json.loads((REPO/'data/manifests/g9q6_split_planning_audit.json').read_text())
        records = [replace(source_row(language=l, split=s), image_id=i) for s, ids in audit['partitions'].items() for i in ids for l in LANGUAGES]
        inventory = compile_inventory(SimpleNamespace(records=records[::-1]))
        self.assertEqual(inventory['counts'], {'image_caption':11520, 'caption_only':11520, 'image_only':2880})
        self.assertEqual([sum(u['split']==s for u in inventory['work_units']) for s in SPLITS], [68,6,6,12])
        for c, rows in inventory['rows'].items():
            self.assertNotIn('CONFIRM', {r['split'] for r in rows})
            self.assertEqual(len({r['input_view_fingerprint'] for r in rows}), len(rows))
        self.assertTrue(all(r['language'] is None and r['caption_id'] is None for r in inventory['rows']['image_only']))
        self.assertEqual(len(text_recipes()),13)

    def test_roles_and_same_forward_bridge(self):
        adapter = BridgeFixture()
        sites = [FeatureSite('decoder',1,p) for p in ('last_caption','caption_mean')]
        request = ExtractionRequest([source_row()],sites,purpose='preliminary_features')
        result = extract_with_evidence(adapter,request)
        self.assertEqual(len(adapter.calls),1)
        self.assertFalse(adapter.calls[0][2])
        self.assertEqual(result.features.vectors[sites[0].key], [[1.25,-2.5]])
        self.assertEqual(result.evidence[0]['caption_positions'],[2,3])
        self.assertNotIn('native_serialization',result.evidence[0])
        compare_public_path(adapter,request)
        self.assertEqual(len(adapter.calls),3)
        with self.assertRaises(ContractError):
            extract_with_evidence(adapter,replace(request,inputs=[replace(source_row(),split='CONFIRM')]))
        self.assertEqual(len(adapter.calls),3)
        adapter._forward = lambda *a,**k: ({s.key:[9.,9.] for s in sites}, token(source_row(),'caption_only'))

    def test_bad_masks_and_overlimit_are_rejected(self):
        item = token(source_row(),'image_caption')
        validate_token_evidence(item,'image_caption')
        for changed in ({'caption_positions':[0,2,3]}, {'last_caption':4}, {'processor_attention_mask':[1,0]},
                        {'token_count':1025}, {'image_positions':[]}, {'role_and_other_positions':[1]}):
            with self.assertRaises(ContractError): validate_token_evidence(dict(item,**changed),'image_caption')


class ArtifactFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.probe = reference(storage_probe(self.root, PROTOCOL, 'e'*64, producer('probe')), self.root)
        self.inputs = [source_row(i,l,s) for i,s in ((1,'FIT'),(2,'DEV-S'),(3,'DEV-C'),(4,'SCREEN')) for l in LANGUAGES]
        inv = compile_inventory(SimpleNamespace(records=self.inputs))
        self.data = dict(inv,kind='stage2_data_plan',schema='stage2.plan.v1',plan_fingerprint='d'*64,
                         dataset_fingerprint='e'*64,protocol_fingerprint=fingerprint(PROTOCOL),recipes=text_recipes(),
                         source_identity={'records_sha256':'1'*64,'targets_sha256':'2'*64,'split_manifest_sha256':'3'*64},
                         target_reference={}, specification={})
        dm=publish_json(self.root,'plans/data',metadata('stage2_data_plan',PROTOCOL,'e'*64,producer()),'plan.json',self.data)
        self.data['manifest_reference']=reference(dm,self.root)
        self.plan=copy.deepcopy(self.data)
        sites={'decoder:1:last_caption':{'module_path':'blocks.0','pooling':'last_caption','output':'block output'},
               'decoder:1:caption_mean':{'module_path':'blocks.0','pooling':'caption_mean','output':'block output'},
               'encoder:None:visual_mean':{'module_path':'vision','pooling':'visual_mean','output':'mean visual'}}
        self.plan.update(kind='stage2_model_plan',model='qwen',model_config={'config_fingerprint':'f'*64,'max_input_tokens':1024,'precision':'bfloat16'},bridge_sha256='b'*64,
            execution_identity={'execution_fingerprint':'a'*64,'site_specs':sites}, dimensions={k:2 for k in sites},
            required_sites={c:sorted(k for k in sites if (k.startswith('decoder:') if c!='image_only' else not k.startswith('decoder:'))) for c in CONDITIONS},
            audit_sites=['encoder:None:visual_mean'], job_id='fixture',references={'readiness':{'path':'fixture/_COMPLETE.json','sha256':'0'*64}})
        mm=publish_json(self.root,'plans/model',metadata('stage2_model_plan',PROTOCOL,'e'*64,producer(), 'f'*64),'plan.json',self.plan)
        self.plan['manifest_reference']=reference(mm,self.root)
        self.plan=frozen(self.plan)
        # Isolate the artifact layer from upstream dataset acceptance. Tests below
        # still use real publish/verify/checksums/arrays and complete row inventories.
        self.patchers=[patch('src.stage2.features.load_job_plan',return_value=self.plan),
                       patch('src.stage2.text.load_job_plan',side_effect=lambda *a,**k:frozen(self.data)),
                       patch('src.stage2.features._index_producer',side_effect=lambda p,r,pr,run:producer(run))]
        for p in self.patchers: p.start()
    def tearDown(self):
        for p in self.patchers: p.stop()
        self.temp.cleanup()
    def payload(self,condition='caption_only',split='FIT',delta=0):
        unit=next(u for u in self.plan['work_units'] if u['split']==split)
        rows=planned_rows(self.plan,unit['chunk_id'],condition)
        folder=self.root/('payload'+str(len(list(self.root.glob('payload*')))))
        folder.mkdir()
        (folder/'rows.jsonl').write_bytes(jsonl(rows))
        evidence=[]
        for r in rows:
            inp=next(i for i in self.inputs if list(i.row_key)==r['source_row_key'])
            item=token(inp,condition)
            item.update(visibility_reference=plain(self.plan['references']['readiness']),
                        visibility_scope='inherited Stage 1; no bulk attention measurement')
            evidence.append(item)
        (folder/'token_evidence.jsonl').write_bytes(jsonl(evidence))
        vectors={k:np.full((len(rows),2),1+delta,dtype='<f4') for k in self.plan['required_sites'][condition]}
        for i,(k,a) in enumerate(vectors.items()): np.save(folder/f'site_{i:03d}.npy',a,allow_pickle=False)
        audits={'repeated_vectors':{str(i):{k:a[i].tolist() for k,a in vectors.items()} for i in sorted({0,len(rows)-1})},'visual_references':[]}
        if split=='FIT': audits['public_path_vectors']={str(i):{k:a[i].tolist() for k,a in vectors.items()} for i in range(len(rows))}
        (folder/'audits.json').write_bytes(json_bytes(audits))
        return unit['chunk_id'], {p.name:p for p in folder.iterdir()}

    def producer(self, run='fixture'):
        return dict(producer(run), storage_probe=self.probe)


class FeatureArtifactTests(ArtifactFixture):
    def test_immutable_release_role_restart_and_corruption(self):
        chunk,files=self.payload()
        marker=publish_feature_chunk(self.root,self.plan,chunk,'caption_only','attempt1',files,self.producer())
        with self.assertRaises(FileExistsError): publish_feature_chunk(self.root,self.plan,chunk,'caption_only','attempt1',files,self.producer())
        index=publish_release_index(self.root,self.plan,'caption_only','FIT',[marker],'release1')
        reader=load_feature_release(index,self.root,PROTOCOL,'source_fit',{'plan_fingerprint':self.plan['plan_fingerprint'],'condition':'caption_only','split':'FIT'})
        rows,a=reader.read_matrix('decoder:1:last_caption','bn')
        self.assertEqual(a.shape,(1,2));self.assertEqual(rows[0]['language'],'bn')
        with self.assertRaises(ContractError): load_feature_release(index,self.root,PROTOCOL,'exploratory_score',{'split':'FIT'})
        (marker.parent/'site_000.npy').write_bytes(b'broken')
        with self.assertRaises(ContractError): reader.read_matrix('decoder:1:last_caption')

    def test_duplicate_numerical_conflict_and_missing_chunk(self):
        chunk,files=self.payload()
        a=publish_feature_chunk(self.root,self.plan,chunk,'caption_only','attempt1',files,self.producer())
        b=publish_feature_chunk(self.root,self.plan,chunk,'caption_only','attempt2',files,self.producer())
        publish_release_index(self.root,self.plan,'caption_only','FIT',[a,b],'identical')
        chunk,changed=self.payload(delta=1)
        c=publish_feature_chunk(self.root,self.plan,chunk,'caption_only','attempt3',changed,self.producer())
        with self.assertRaisesRegex(ContractError,'conflict'): publish_release_index(self.root,self.plan,'caption_only','FIT',[a,c],'conflict')
        with self.assertRaises(ContractError): publish_release_index(self.root,self.plan,'caption_only','FIT',[],'missing')

    def test_wrong_rows_dtype_nonfinite_and_missing_audits_rejected(self):
        for mode in ('rows','dtype','nan','repeat'):
            chunk,files=self.payload()
            if mode=='rows': files['rows.jsonl'].write_bytes(jsonl(read_json_rows(files['rows.jsonl'])[::-1]))
            elif mode=='dtype': np.save(files['site_000.npy'],np.ones((4,2),dtype='<f8'))
            elif mode=='nan': np.save(files['site_000.npy'],np.full((4,2),np.nan,dtype='<f4'))
            else: files['audits.json'].write_bytes(json_bytes({'repeated_vectors':{}}))
            with self.assertRaises(ContractError): publish_feature_chunk(self.root,self.plan,chunk,'caption_only','bad_'+mode,files,self.producer())

    def test_visual_reference_context_mismatch_blocks(self):
        chunk,files=self.payload('image_only')
        ref=publish_feature_chunk(self.root,self.plan,chunk,'image_only','image',files,self.producer())
        chunk,combined=self.payload('image_caption')
        audits=json.loads(combined['audits.json'].read_text())
        audits['visual_references']=[{'artifact':reference(ref,self.root),'row_index':0} for _ in range(4)]
        combined['audits.json'].write_bytes(json_bytes(audits))
        np.save(combined['rows.jsonl'].parent/'visual_000.npy',np.ones((4,2),dtype='<f4'))
        combined['visual_000.npy']=combined['rows.jsonl'].parent/'visual_000.npy'
        publish_feature_chunk(self.root,self.plan,chunk,'image_caption','good',combined,self.producer())
        np.save(combined['visual_000.npy'],np.zeros((4,2),dtype='<f4'))
        with self.assertRaisesRegex(ContractError,'invariance'): publish_feature_chunk(self.root,self.plan,chunk,'image_caption','bad',combined,self.producer())

    def test_missing_workers_produce_blocked_suite_diagnostic(self):
        from src.stage2.operations import validate_suite
        with patch('src.stage2.plans.load_job_plan',return_value=frozen(self.data)):
            marker=validate_suite(self.root,PROTOCOL,{'data_plan':self.data['manifest_reference']['path']},producer('suite'))
        body=json.loads((marker.parent/'report.json').read_text())
        self.assertEqual(body['status'],'blocked')
        self.assertEqual(len(body['blockers']),3)
        self.assertFalse(body['scientific_results'])


def read_json_rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]


class TextArtifactTests(ArtifactFixture):
    def preprocessor(self,family='length'):
        recipe=next(r for r in self.data['recipes'] if r['family']==family and r['language']=='en')
        folder=self.root/'textpayload';folder.mkdir(exist_ok=True)
        spec={'fit_rows_sha256':fingerprint(text_rows(self.data,recipe,'FIT')),'fit_split':'FIT'}
        if family=='length': spec.update(columns=recipe['columns'],fitted_parameters=False)
        else:
            terms=[{'n':1,'tokens':['a']},{'n':2,'tokens':['a','b']}]
            spec.update(terms=terms,term_order_sha256=fingerprint(terms),document_frequency=[2,2],corpus_term_count=[3,2],n_fit_documents=1)
            # Sparse fixture needs two FIT images to meet frozen min_df=2.
        (folder/'preprocessor.json').write_bytes(json_bytes(spec))
        versions=preprocessing_versions('tests/test_stage2_foundation.py')
        meta=dict(self.producer('pre'),preprocessing_versions=versions,analyzer_file='tests/test_stage2_foundation.py')
        return recipe,folder,meta

    def test_length_family_partial_split_roles_reload_and_runtime_mismatch(self):
        recipe,folder,meta=self.preprocessor()
        pre=publish_text_preprocessor(self.root,self.data,recipe['recipe_fingerprint'],{'preprocessor.json':folder/'preprocessor.json'},meta)
        releases={}
        for split in ('FIT','DEV-S'):
            rows=text_rows(self.data,recipe,split)
            (folder/'rows.jsonl').write_bytes(jsonl(rows));np.save(folder/'matrix.npy',np.ones((len(rows),2),dtype='<f4'))
            releases[split]=publish_text_release(self.root,pre,split,{k:folder/k for k in ('rows.jsonl','matrix.npy')},dict(meta,run_id=split,zero_rows=0))
        fam=publish_text_family(self.root,pre,releases,'family',meta)
        reader=load_text_family(fam,self.root,PROTOCOL,'source_fit',{'plan_fingerprint':self.data['plan_fingerprint'],'preprocessing_fingerprint':json.loads((pre.parent/'fit.json').read_text())['preprocessing_fingerprint']})
        rows,a=reader.read_matrix('FIT')
        self.assertEqual(a.shape,(1,2))
        with self.assertRaises(ContractError): reader.read_matrix('DEV-S')
        with self.assertRaises(ContractError): reader.read_matrix('CONFIRM')
        with self.assertRaises(ContractError): publish_text_release(self.root,pre,'FIT',{k:folder/k for k in ('rows.jsonl','matrix.npy')},dict(meta,preprocessing_versions={}))
        self.assertFalse(json.loads((fam.parent/'family.json').read_text())['family_complete'])

    def test_unknown_recipe_and_changed_fit_identity_block(self):
        recipe,folder,meta=self.preprocessor()
        spec=json.loads((folder/'preprocessor.json').read_text());spec['fit_split']='SCREEN'
        (folder/'preprocessor.json').write_bytes(json_bytes(spec))
        with self.assertRaises(ContractError): publish_text_preprocessor(self.root,self.data,recipe['recipe_fingerprint'],{'preprocessor.json':folder/'preprocessor.json'},meta)

    def test_sparse_portable_csr_reload_zero_rows_and_invalid_columns(self):
        # Two synthetic FIT documents permit the fixed min_df without changing it.
        self.inputs.extend(source_row(5,l,'FIT') for l in LANGUAGES)
        self.data.update(compile_inventory(SimpleNamespace(records=self.inputs)))
        recipe,folder,meta=self.preprocessor('word_tfidf')
        spec=json.loads((folder/'preprocessor.json').read_text())
        spec['n_fit_documents']=2
        (folder/'preprocessor.json').write_bytes(json_bytes(spec))
        np.save(folder/'idf.npy',np.ones(2,dtype='<f8'))
        pre=publish_text_preprocessor(self.root,self.data,recipe['recipe_fingerprint'],{k:folder/k for k in ('preprocessor.json','idf.npy')},meta)
        (folder/'rows.jsonl').write_bytes(jsonl(text_rows(self.data,recipe,'FIT')))
        np.save(folder/'data.npy',np.array([1],dtype='<f4'))
        np.save(folder/'indices.npy',np.array([1],dtype='<i4'))
        np.save(folder/'indptr.npy',np.array([0,1,1],dtype='<i4'))
        files={k:folder/k for k in ('rows.jsonl','data.npy','indices.npy','indptr.npy')}
        release=publish_text_release(self.root,pre,'FIT',files,dict(meta,run_id='sparse',zero_rows=1))
        family=publish_text_family(self.root,pre,{'FIT':release},'sparse_family',meta)
        reader=load_text_family(family,self.root,PROTOCOL,'source_fit',{'plan_fingerprint':self.data['plan_fingerprint'],'preprocessing_fingerprint':json.loads((pre.parent/'fit.json').read_text())['preprocessing_fingerprint']})
        rows,values=reader.read_matrix('FIT')
        self.assertEqual(values.shape,(2,2)); self.assertEqual(values.nnz,1)
        self.assertEqual(values[1].nnz,0)
        np.save(folder/'indices.npy',np.array([2],dtype='<i4'))
        with self.assertRaises(ContractError): publish_text_release(self.root,pre,'FIT',files,dict(meta,run_id='badcolumns',zero_rows=1))
        np.save(folder/'indices.npy',np.array([1],dtype='<i8'))
        with self.assertRaises(ContractError): publish_text_release(self.root,pre,'FIT',files,dict(meta,run_id='baddtype',zero_rows=1))


class TypedReadinessTests(unittest.TestCase):
    def test_historical_evidence_receipt_roundtrip_and_wrong_kind_gate(self):
        from src.common.contracts import resolve_model, INSPECTION_CHECKS
        from src.common.cache_keys import execution_fingerprint
        from src.stage2.readiness import publish_readiness, verify_readiness, MODEL_FILES
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            config=json.loads((REPO/'configs/models.json').read_text())['models']['qwen']
            config.update(chat_template_sha256='a'*64,image_token_ids=[99],image_processor_class='FixtureProcessor')
            config=resolve_model({'models':{'qwen':config}},'qwen')
            dfp='e'*64
            def put(kind,run,name,body,extra=None):
                m=metadata(kind,PROTOCOL,dfp,producer(run),config['config_fingerprint'] if kind not in ('stage1_dataset','stage1_data_validation') else None)
                m.update(code_commit='d73133ce51247f10dd6bc7af35ee9fe9b59ed517')
                if extra:m.update(extra)
                return publish_json(root,'evidence/'+run,m,name,body)
            source_identity={'records_sha256':'1'*64,'targets_sha256':'2'*64,'split_manifest_sha256':'3'*64,'category_order':PROTOCOL['category_order']}
            data=put('stage1_dataset','data','dataset.json',source_identity)
            validation=put('stage1_data_validation','data_validation','report.json',{})
            m=metadata('stage1_runtime_configuration',PROTOCOL,dfp,producer('config'),config['config_fingerprint'])
            cp=publish_json(root,'evidence/config',m,'configuration_report.json',
                {'status':'passed','configuration_resolved':True,'weights_loaded':False,'blockers':[],
                 'model_fingerprint':config['config_fingerprint']}, {'models.json':json_bytes({'models':{'qwen':config}})})
            sites={f'decoder:{d}:{p}':{'module_path':f'blocks.{d-1}','pooling':p,'output':'block output'}
                   for d in (1,2,3,4) for p in ('last_caption','caption_mean')}
            sites['encoder:None:visual_mean']={'module_path':'vision','pooling':'visual_mean','output':'visual mean'}
            backend={'device_type':'cuda','device_name':'Fixture L4','batch_size':1,'eval':True,'padding':False,'truncation':False}
            src={Path(p).name:file_sha256(REPO/p) for p in MODEL_FILES[:2]}
            execution={'execution_fingerprint':execution_fingerprint(config,backend,src),'backend':backend,'source_sha256':src,'site_specs':sites}
            info={'decoder_blocks':4,'decoder_depths':[1,2,3,4],'midpoint':2,
                  'supported_sites':sorted(sites),'repeat_atol':0,'repeat_rtol':0,'model_fingerprint':config['config_fingerprint']}
            inspection={'checks':{k:True for k in INSPECTION_CHECKS},'records':[{'fixture':True}]}
            conditions={c:sorted(k for k in sites if c=='image_caption' or
                        (k.startswith('decoder:') if c=='caption_only' else not k.startswith('decoder:'))) for c in CONDITIONS}
            report={'status':'passed','gpu_checks_passed':True,'code_dirty':False,'blockers':[],
                    'resolved_config':config,'inspection':inspection,'adapter':info,'conditions_validated':conditions,
                    'repeat_extraction':True,'visual_context_invariance':True,'dimensions':{k:2 for k in sites},'execution_identity':execution}
            mp=put('stage1_model_validation','model','model_validation.json',report,
                   {'source_sha256':file_sha256(REPO/'src/stage1/validate_model.py')})
            smoke={'kind':'stage1_smoke','status':'passed','blockers':[],'inspection':inspection,'adapter':info,
                   'checks':{k:True for k in ('data_contract','adapter_inspection','finite_features','repeat_extraction','toy_optimizer','permutation_plumbing')},
                   'toy_optimizer':[{'fixture':True}],'permutation_plumbing':[{}, {}, {}]}
            sp=put('stage1_smoke','smoke','report.json',smoke,
                   {'script_sha256':file_sha256(REPO/'src/stage1/run_validity_smoke.py')})
            audit=json.loads((REPO/'data/manifests/g9q6_split_planning_audit.json').read_text())
            records=[replace(source_row(language=l,split=s),image_id=i) for s,ids in audit['partitions'].items() for i in ids for l in LANGUAGES]
            bundle=SimpleNamespace(dataset_fingerprint=dfp,records=records)
            refs=[reference(data,root),reference(validation,root)]
            with patch('src.stage2.readiness.verify_data',return_value=(bundle,refs)), patch('src.stage2.plans.verify_data',return_value=(bundle,refs)):
                receipt=publish_readiness(root,PROTOCOL,'qwen',data,validation,cp,mp,sp,producer('receipt'))
                resolved=verify_readiness(receipt,root,PROTOCOL,config)
                self.assertEqual(resolved['original_commits']['model_validation'],'d73133ce51247f10dd6bc7af35ee9fe9b59ed517')
                from src.stage2.plans import publish_data_plan,publish_model_plan,resolve_worker_settings
                shared=publish_data_plan(root,PROTOCOL,data,validation,producer('shared'))
                model_plan=publish_model_plan(root,PROTOCOL,shared,receipt,producer('model_plan'))
                settings=resolve_worker_settings(model_plan,root,PROTOCOL,'qwen')
                self.assertEqual(settings['plan']['counts']['image_only'],2880)
                self.assertEqual(settings['dataset_manifest'],str(data))
                self.assertEqual(settings['model_config'],str(cp.parent/'models.json'))
                with self.assertRaises(ContractError): resolve_worker_settings(model_plan,root,PROTOCOL,'gemma')
                cpu=resolve_worker_settings(shared,root,PROTOCOL)
                self.assertNotIn('model_config',cpu)
                with self.assertRaises(TypeError): resolved['model']='gemma'
                with self.assertRaises(ContractError): publish_readiness(root,PROTOCOL,'qwen',data,validation,cp,sp,sp,producer('wrongkind'))
                with patch('src.stage2.readiness.git_source_hash',return_value='b'*64):
                    with self.assertRaises(ContractError): verify_readiness(receipt,root,PROTOCOL,config)


class ReadinessTests(unittest.TestCase):
    def test_notebook_entry_cells_compile_and_have_no_saved_outputs(self):
        import ast
        for name in ('prepare_stage2','validate_stage2'):
            notebook=json.loads((REPO/'scripts/stage2'/(name+'.ipynb')).read_text(encoding='utf-8'))
            for cell in notebook['cells']:
                if cell['cell_type']=='code':
                    self.assertEqual(cell['outputs'],[])
                    self.assertIsNone(cell['execution_count'])
                    ast.parse(''.join(cell['source']))

    def test_resource_projection_uses_observed_dimensions_and_exact_remaining_units(self):
        plan={'counts':{'image_caption':8,'caption_only':8,'image_only':2},
              'dimensions':{'decoder':3,'visual':2},
              'required_sites':{'image_caption':['decoder'],'caption_only':['decoder'],'image_only':['visual']}}
        measurements={'token_ranges':{},'condition_seconds_per_input':{c:2.0 for c in CONDITIONS},
            'forward_seconds':36.,'audit_seconds':3.,'write_seconds':1.,'verification_seconds':1.,
            'gpu_peak_allocated_bytes':100,'gpu_peak_reserved_bytes':200,'raw_bytes':208,
            'sidecar_bytes':104,'free_space_bytes':None,'remaining_inputs':{'image_caption':4,'caption_only':4,'image_only':1}}
        result=resource_projection(plan,measurements)
        self.assertEqual(result['expected_raw_matrix_bytes'],208)
        self.assertEqual(result['projected_bytes_with_overhead_and_25_percent_reserve'],390)
        self.assertEqual(result['estimated_remaining_forward_seconds'],18.)
        self.assertFalse(result['free_space_available'])

    def test_discovery_rejects_wrong_kind_and_conflicting_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);config={'config_fingerprint':'f'*64}
            for run,kind,execution in [('one','stage1_smoke','a'),('two','stage1_model_validation','a')]:
                meta=metadata(kind,PROTOCOL,'e'*64,producer(run),'f'*64)
                publish_json(root,'smoke/'+config['config_fingerprint']+'/'+run,meta,
                    'report.json' if kind=='stage1_smoke' else 'model_validation.json',
                    {'status':'passed','execution_identity':execution})
            ref,all_refs=discover_evidence(root,PROTOCOL,config,'stage1_model_validation','e'*64)
            self.assertEqual(len(all_refs),1);self.assertIn('/two/',ref['path'])
            publish_json(root,'smoke/'+config['config_fingerprint']+'/three',metadata('stage1_model_validation',PROTOCOL,'e'*64,producer('three'),'f'*64),
                'model_validation.json',{'status':'passed','execution_identity':'b'})
            with self.assertRaisesRegex(ContractError,'Conflicting'): discover_evidence(root,PROTOCOL,config,'stage1_model_validation','e'*64)

    def test_actual_backend_rechecked_after_adapter_load(self):
        plan={'model_config':{'x':1},'execution_identity':{'backend':'L4'}}
        a=SimpleNamespace(config={'x':1},describe_execution=lambda:{'backend':'L4'})
        assert_runtime(a,plan)
        a.describe_execution=lambda:{'backend':'different'}
        with self.assertRaises(ContractError): assert_runtime(a,plan)

    def test_plan_payload_self_hash_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            marker=publish_json(Path(temp),'plans/bad',metadata('stage2_data_plan',PROTOCOL,'e'*64,producer()),'plan.json',{'plan_fingerprint':'a'*64})
            with self.assertRaisesRegex(ContractError,'fingerprint'): load_job_plan(marker,Path(temp),PROTOCOL)


if __name__=='__main__': unittest.main()
