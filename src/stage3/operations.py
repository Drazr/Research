"""Coordinator CLI only; CPU worker orchestration is intentionally absent."""
import argparse
import json
from pathlib import Path
from src.common.contracts import ContractError
from src.common.storage import phase1_root
from src.stage2.operations import current_producer as original_producer
from .data import compile_fit_plan, load_fit_plan, runtime
from .artifacts import Store
from .readouts import publish_target_assignments


def current_producer(scope='local-development'):
    p=original_producer(scope=scope)
    p['software_versions']=runtime()
    return p


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('task',choices=['prepare','validate','readback'])
    parser.add_argument('--phase1-root',required=True)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--inventory')
    parser.add_argument('--storage-scope',choices=['local-development','persistent-drive'],default='persistent-drive')
    a=parser.parse_args(); root=phase1_root(a.phase1_root)
    repo=Path(__file__).resolve().parents[2]
    protocol=json.loads((repo/'configs/protocol.json').read_text(encoding='utf-8'))
    if a.task=='readback':
        store=Store(root,protocol)
        b=store.json(a.manifest,'body.json','stage3_registry_readback')
        for ref in b['candidates']+b['selections']+[b['g0_gate']]: store.complete(ref)
        print(json.dumps({'event':'independent_readback','status':'passed'}),flush=True)
        return
    producer=current_producer(a.storage_scope)
    print(json.dumps({'event':'stage3_task_start','task':a.task,'code_commit':producer['code_commit']}),flush=True)
    if a.task=='prepare':
        plan=compile_fit_plan(a.manifest,root,protocol,producer)
        assignments=publish_target_assignments(plan,root,producer)
        result={'plan':str(root/plan.reference['path']),'assignments':dict(assignments),
                'counts':{'groups':145,'candidate_blocks':2320,'selections':580},
                'profiling_groups':list(plan['profiling_groups']),'stage3_complete':False}
    else:
        from .validation import validate_stage3
        if not a.inventory: raise ContractError('Explicit worker inventory JSON required')
        plan=load_fit_plan(a.manifest,root,protocol)
        inv=json.loads(Path(a.inventory).read_text(encoding='utf-8'))
        ref=validate_stage3(plan,inv['candidates'],inv['selections'],inv['g0_gate'],producer)
        result={'manifest':str(root/ref['path']),'status':'passed','stage3_complete':True}
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    try: main()
    except (ContractError,OSError,KeyError,ValueError) as exc:
        print(json.dumps({'status':'blocked','blockers':[str(exc)]}),flush=True)
        raise SystemExit(2)
