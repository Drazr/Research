"""Audited budget-only amendment; immutable successful legacy units stay valid."""
import hashlib
import subprocess
from pathlib import Path
from src.common.contracts import canonical_bytes,fingerprint
from src.common.storage import publish,new_run_id
from src.stage2.schema import check,plain
from .artifacts import Store
from .data import FitPlan,fit_fingerprint,numerical_identity,runtime,suite_inventory

BASE='91c5131c32e7a675c5ef2fbaf1353da2c693eff3'
OLD_ENGINE='2647b8ab38f75fac72b70d6ea9baf0f97b58deee2fda9d1ba155eb50453f7b42'
RULE='converged-lbfgs-budget-extension.v1'


def _source_proof():
    repo=Path(__file__).resolve().parents[2]
    old=subprocess.check_output(['git','show',BASE+':src/stage3/engine.py'],cwd=repo)
    check(hashlib.sha256(old).hexdigest()==OLD_ENGINE,'Unrecognized legacy numerical implementation')
    old=old.replace(b'\r\n',b'\n')
    new=Path(__file__).with_name('engine.py').read_bytes().replace(b'\r\n',b'\n')
    needle=b'for limit in (1000,5000):'
    check(old.count(needle)==1 and old.replace(needle,b'for limit in (1000,5000,20000):')==new,
          'Compatibility requires the exact budget-only source change')


def validate_compatibility(body,store):
    _source_proof()
    c=body['compatibility']
    check(set(c)=={'rule','legacy_plan','legacy_fit_identity','diagnostic'} and c['rule']==RULE,
          'Unknown compatibility certificate')
    old=store.json(c['legacy_plan'],'body.json','stage3_plan')
    check('compatibility' not in old and old['fit_identity']==c['legacy_fit_identity']==fit_fingerprint(old),
          'Invalid legacy plan identity')
    expected=dict(numerical_identity(),engine_sha256=OLD_ENGINE,iterations=[1000,5000])
    check(old['numerical']==expected and body['numerical']==numerical_identity(),
          'Amendment may extend the retry budget only')
    old_core={k:v for k,v in old.items() if k not in ('numerical','fit_identity')}
    new_core={k:plain(v) for k,v in body.items() if k not in ('numerical','fit_identity','compatibility')}
    check(old_core==new_core,'Input, target, runtime or selection policy changed')
    diagnostic=store.json(c['diagnostic'],'report.json','stage3_solver_diagnostic')
    _,_,marker=store.marker(c['diagnostic'],'stage3_solver_diagnostic')
    check(marker['software_versions']==old['runtime'],'Diagnostic used a different numerical runtime')
    check(diagnostic['diagnostic_only'] is True and diagnostic['accepted_fit'] is False
          and diagnostic['plan']==c['legacy_plan'] and diagnostic['solver']=='lbfgs'
          and diagnostic['tol']==1e-8 and diagnostic['C']==10 and diagnostic['assignment']=='seed0'
          and diagnostic['head']==4 and diagnostic['group_id'] in old['groups'],
          'Wrong diagnostic evidence for the computational amendment')
    attempts=diagnostic['attempts']
    check(len(attempts)==1 and attempts[0]['max_iter']==20000
          and attempts[0]['convergence_check_passed'] is True
          and 5000<attempts[0]['n_iter']<20000,'Extended diagnostic did not establish the budget case')
    return {old['fit_identity']:c['legacy_plan']}


def amend_solver_budget(legacy_manifest,diagnostic_manifest,root,protocol,producer):
    """Publish a new plan. No arrays/checkpoints/old markers are rewritten."""
    store=Store(root,protocol)
    old_ref=store.ref(legacy_manifest)
    old=store.json(old_ref,'body.json','stage3_plan')
    check(old['runtime']==runtime()==producer['software_versions'],'Use the same pinned runtime')
    original=suite_inventory(old['suite'],store)
    check(all(old[k]==v for k,v in original.items()),'Legacy suite/input identity changed')
    body=dict(old,numerical=numerical_identity(),compatibility={
        'rule':RULE,'legacy_plan':old_ref,'legacy_fit_identity':old['fit_identity'],
        'diagnostic':store.ref(diagnostic_manifest)})
    body['fit_identity']=fit_fingerprint(body)
    validate_compatibility(body,store)
    run=new_run_id()
    meta=dict(kind='stage3_plan',status='passed',run_id=run,code_commit=producer['code_commit'],
              software_versions=producer['software_versions'],storage_scope=producer['storage_scope'],
              protocol_fingerprint=fingerprint(protocol),dataset_fingerprint=body['dataset_fingerprint'],
              model_fingerprint=None,seeds={'g0':[0,1,2]},
              upstream_artifacts=[old_ref,body['compatibility']['diagnostic'],body['suite']])
    folder=publish(Path(root),'cache/shared/stage3-plans/'+run,meta,{'body.json':canonical_bytes(body)+b'\n'})
    return FitPlan(body,root,protocol,store.ref(folder/'_COMPLETE.json'),store)
