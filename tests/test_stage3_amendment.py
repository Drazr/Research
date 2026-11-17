"""Computational amendment and immutable recovery; synthetic inputs only."""
import copy
import ast
import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from test_stage3_foundation import Fixture, REPO
from src.common.contracts import ContractError, canonical_bytes
from src.common.storage import publish, new_run_id
from src.stage2.schema import plain
from src.stage3.data import FitPlan, fit_fingerprint, load_fit_plan, open_numeric_session
from src.stage3.amendment import BASE, OLD_ENGINE, _source_proof, amend_solver_budget, validate_compatibility
from src.stage3.readouts import (load_or_publish_statistics, fit_candidate_block, select_system,
                                publish_target_assignments, candidate)
from src.stage3.recovery import publish_run_index, load_run_index, read_completed
from src.stage3.run_g0 import run_g0


class BudgetTests(unittest.TestCase):
    @classmethod
    def diagnostic(cls,report):
        _,_,meta=cls.f.store.marker(cls.old.reference)
        run=new_run_id()
        meta=dict(meta,kind='stage3_solver_diagnostic',run_id=run,software_versions=cls.f.producer['software_versions'])
        folder=publish(cls.f.root,'diagnostics/'+run,meta,{'report.json':canonical_bytes(report)})
        return cls.f.store.ref(folder/'_COMPLETE.json')

    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.f=Fixture(Path(cls.temp.name)); current=cls.f.build()
        body=plain(current.body)
        body['numerical'].update(engine_sha256=OLD_ENGINE,iterations=[1000,5000])
        body['fit_identity']=fit_fingerprint(body)
        ref=cls.f.put('stage3_plan','body.json',body)
        cls.old=FitPlan(body,cls.f.root,cls.f.protocol,ref)
        cls.gid=next(g for g,v in body['groups'].items() if v['source']=='e03' and v['language'] is None)
        report={'diagnostic_only':True,'accepted_fit':False,'plan':ref,'solver':'lbfgs','tol':1e-8,
                'C':10.,'assignment':'seed0','head':4,'group_id':cls.gid,
                'attempts':[{'max_iter':20000,'n_iter':5511,'convergence_check_passed':True}]}
        cls.diag=cls.diagnostic(report)
        cls.new=amend_solver_budget(ref,cls.diag,cls.f.root,cls.f.protocol,cls.f.producer)

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_01_reload_changes_identity_but_not_inputs(self):
        new=load_fit_plan(self.new.reference,self.f.root,self.f.protocol)
        self.assertNotEqual(self.old['fit_identity'],new['fit_identity'])
        self.assertEqual(plain(new.legacy),{self.old['fit_identity']:plain(self.old.reference)})
        self.assertEqual(new['groups'],self.old['groups'])
        self.assertEqual(new['targets'],self.old['targets'])
        with self.assertRaisesRegex(ContractError,'Numerical environment'):
            load_fit_plan(self.old.reference,self.f.root,self.f.protocol)

    def test_02_changed_recipe_runtime_inputs_or_certificate_block(self):
        for field in ('C','tol','recipe','iterations'):
            body=plain(self.new.body); body['numerical'][field]='changed'
            with self.assertRaises(ContractError): validate_compatibility(body,self.new.store)
        for field in ('runtime','targets','primary'):
            body=plain(self.new.body); body[field]={}
            with self.assertRaises(ContractError): validate_compatibility(body,self.new.store)
        body=plain(self.new.body); body['compatibility']['legacy_fit_identity']='0'*64
        with self.assertRaises(ContractError): validate_compatibility(body,self.new.store)
        body=plain(self.new.body); body['compatibility']['rule']='unrecognized'
        with self.assertRaises(ContractError): validate_compatibility(body,self.new.store)
        report=self.f.store.json(self.diag,'report.json');report['accepted_fit']=True
        bad=self.diagnostic(report)
        with self.assertRaises(ContractError):
            amend_solver_budget(self.old.reference,bad,self.f.root,self.f.protocol,self.f.producer)

    def test_03_source_equivalence_and_normally_converging_outputs(self):
        _source_proof()
        old={};exec(subprocess.check_output(['git','show',BASE+':src/stage3/engine.py'],cwd=REPO),old)
        from src.stage3.engine import fit_heads
        rng=np.random.default_rng(9);x=rng.normal(size=(16,3))
        y=np.tile(np.arange(16)%2,(25,1)).T.astype('f8');known=np.ones(y.shape,dtype=bool)
        before=old['fit_heads'](x,y,known,np.ones(16),10.)
        after=fit_heads(x,y,known,np.ones(16),10.)
        for a,b in zip(before[:3],after[:3]): np.testing.assert_array_equal(a,b)
        for a,b in zip(before[3],after[3]):
            for aa,bb in zip(a['attempts'],b['attempts']):
                self.assertEqual({k:v for k,v in aa.items() if k!='seconds'},
                                 {k:v for k,v in bb.items() if k!='seconds'})

    def test_04_old_complete_g0_recovers_without_refits_or_feature_reads(self):
        gid=self.gid;old=self.old
        assignments=publish_target_assignments(old,old.root,self.f.producer)
        session=open_numeric_session(old,old.root,[gid],'source_fit')
        stats=load_or_publish_statistics(old,gid,session,self.f.producer)['reference']
        result=run_g0(old,[gid],['seed0','seed1','seed2'],assignments,self.f.producer,
                      supplied={'statistics':[stats]})
        self.assertEqual(result['counts']['null_groups'],1)
        with patch('src.stage3.run_g0.fit_candidate_block',side_effect=AssertionError('refit')), \
             patch('src.stage3.run_g0.open_numeric_session',side_effect=AssertionError('feature read')):
            resumed=run_g0(self.new,[gid],['seed0','seed1','seed2'],assignments,self.f.producer,
                           resume_indexes=result['indexes'])
        self.assertEqual(resumed['counts']['candidates'],12)
        self.assertEqual(resumed['counts']['null_groups'],1)
        self.assertEqual(result['entries']['candidates'],resumed['entries']['candidates'])

    def test_05_legacy_ordinary_selection_and_partial_block_reuse(self):
        gid=next(g for g,v in self.old['groups'].items() if v['source']=='vlm')
        sessions={r:open_numeric_session(self.old,self.old.root,[gid],r) for r in ('source_fit','source_select')}
        stats=load_or_publish_statistics(self.old,gid,sessions['source_fit'],self.f.producer)
        refs=[fit_candidate_block(self.old,gid,'true',C,sessions,stats,self.f.producer) for C in (.01,.1,1.,10.)]
        selected=select_system(self.old,gid,'true',refs,self.f.producer)
        idx=publish_run_index(self.old,{'statistics':[stats['reference']],'candidates':refs,'selections':[selected]},self.f.producer)
        from src.stage3.fit_readouts import Recovery
        with patch('src.stage3.readouts.fit_heads',side_effect=AssertionError('refit')):
            recovery=Recovery(self.new,[gid],[idx])
        self.assertEqual(len(recovery.candidates),4)
        self.assertEqual(recovery.selections[gid],plain(selected))
        # An unapproved same-recipe plan reference must not be accepted.
        body=plain(self.old.body);other=self.f.put('stage3_plan','body.json',body)
        foreign=FitPlan(body,self.old.root,self.old.protocol,other)
        foreign_idx=publish_run_index(foreign,{},self.f.producer)
        with self.assertRaisesRegex(ContractError,'Incompatible fitted artifact'):
            load_run_index(self.new,foreign_idx)

    def test_06_partial_legacy_control_only_fits_missing_Cs(self):
        gid=next(g for g,v in self.old['groups'].items() if v['source']=='vlm' and v['language']=='hi')
        sessions={r:open_numeric_session(self.old,self.old.root,[gid],r) for r in ('source_fit','source_select')}
        stats=load_or_publish_statistics(self.old,gid,sessions['source_fit'],self.f.producer)
        ref=fit_candidate_block(self.old,gid,'seed0',.01,sessions,stats,self.f.producer)
        idx=publish_run_index(self.old,{'statistics':[stats['reference']],'candidates':[ref]},self.f.producer)
        assignments=publish_target_assignments(self.old,self.old.root,self.f.producer)
        with patch('src.stage3.run_g0.fit_candidate_block',wraps=fit_candidate_block) as fitter:
            result=run_g0(self.new,[gid],['seed0'],assignments,self.f.producer,resume_indexes=[idx],emit=lambda e:None)
        self.assertEqual([c.args[3] for c in fitter.call_args_list],[.1,1.,10.])
        self.assertEqual(result['counts']['candidates'],4)
        self.assertIn(plain(ref),result['entries']['candidates'])

    def test_07_third_attempt_only_after_original_budgets_fail(self):
        import warnings
        from types import SimpleNamespace
        from sklearn.exceptions import ConvergenceWarning
        from src.stage3.engine import fit_heads
        def construct(**kw):
            limit=kw['max_iter']
            model=SimpleNamespace(classes_=np.array([0,1]),coef_=np.zeros((1,2)),
                                  intercept_=np.zeros(1),n_iter_=np.array([limit if limit<20000 else 5511]))
            def fit(*a,**k):
                if limit<20000:warnings.warn('exhaustion',ConvergenceWarning)
                return model
            model.fit=fit;return model
        known=np.zeros((4,25),dtype=bool);known[:,0]=True
        with patch('src.stage3.engine.LogisticRegression',side_effect=construct) as maker:
            result=fit_heads(np.ones((4,2)),np.tile([0.,1,0,1],(25,1)).T,known,np.ones(4),10.)
        self.assertEqual([c.kwargs['max_iter'] for c in maker.call_args_list],[1000,5000,20000])
        self.assertTrue(result[2][0])
        self.assertEqual([a['converged'] for a in result[3][0]['attempts']],[False,False,True])

    def test_08_additional_engine_change_rejected(self):
        with patch('pathlib.Path.read_bytes',return_value=b'changed engine'):
            with self.assertRaisesRegex(ContractError,'exact budget-only'):_source_proof()

    def test_09_initializer_receipt_is_idempotent_and_creates_shared_parents(self):
        nb=json.loads((REPO/'scripts/stage3/amend_solver_budget.ipynb').read_text())
        tree=ast.parse(''.join(nb['cells'][3]['source']))
        child=next(n.value.value for n in tree.body if isinstance(n,ast.Assign)
                   and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='child')
        config=self.f.root/'configs';config.mkdir(exist_ok=True)
        handoff=config/'old.json';receipt=config/'new.json'
        handoff.write_text(json.dumps({'plan':self.old.reference['path'],
                                      'assignments':{'fixture':'retained'},'ownership':'configs/owners.json'}))
        args=['initializer',str(self.f.root),str(self.f.root/self.old.reference['path']),
              str(self.f.root/self.diag['path']),str(handoff),str(receipt)]
        import os
        previous=Path.cwd()
        read_text=Path.read_text
        def fixture_protocol(path,*a,**kw):
            if path.as_posix()=='configs/protocol.json':return json.dumps(self.f.protocol)
            return read_text(path,*a,**kw)
        try:
            os.chdir(REPO)
            with patch('sys.argv',args),patch('src.stage3.interfaces.current_producer',return_value=self.f.producer),patch('pathlib.Path.read_text',fixture_protocol),contextlib.redirect_stdout(io.StringIO()):
                exec(child,{})
                first=receipt.read_bytes()
                exec(child,{})
            self.assertEqual(first,receipt.read_bytes())
        finally:os.chdir(previous)
        job=self.new['fit_identity'][:16]
        self.assertTrue((self.f.root/'checkpoints'/job/'selections').is_dir())
        self.assertTrue(all((self.f.root/'checkpoints'/job/g[:16]).is_dir() for g in self.new['groups']))


if __name__=='__main__': unittest.main()
