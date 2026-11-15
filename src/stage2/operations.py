"""Coordinator preparation/validation and independent storage readback processes."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from src.common.contracts import fingerprint, ContractError
from src.common.storage import new_run_id, phase1_root, verify
from .schema import (REPO, SPLITS, CONDITIONS, ROLES, artifact, check, metadata,
                     plain, publish_json, read_json, read_rows, reference)


def current_producer(run_id=None, scope='local-development'):
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip()
    check(not dirty, 'Use a clean frozen checkout for artifact production')
    import importlib.metadata
    versions = {'python': platform.python_version()}
    for name in ('numpy', 'scipy', 'scikit-learn'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {'run_id': run_id or new_run_id(), 'code_commit': commit,
            'storage_scope': scope, 'software_versions': versions}


def storage_probe(root, protocol, dataset, producer):
    """Publish uniquely, verify in a fresh process, then publish a readback receipt."""
    meta = metadata('stage2_storage_probe_payload', protocol, dataset, producer)
    marker = publish_json(root, 'cache/shared/stage2-plans/' + meta['run_id'], meta,
                          'probe.json', {'nonce': new_run_id(), 'remote_sync_guarantee': False})
    child = subprocess.run([sys.executable, '-m', 'src.stage2.operations', 'probe-read',
                    '--phase1-root', str(root), '--manifest', str(marker)], cwd=REPO, check=True,
                   capture_output=True, text=True)
    observed = json.loads(child.stdout)
    check(observed['pid'] != os.getpid() and observed['payload'] == reference(marker, root), 'Independent readback evidence mismatch')
    receipt = dict(producer, run_id=new_run_id())
    meta = metadata('stage2_storage_probe', protocol, dataset, receipt)
    return publish_json(root, 'cache/shared/stage2-plans/' + meta['run_id'], meta, 'probe.json',
                        {'payload': reference(marker, root), 'independent_process_readback': True,
                         'reader_pid': observed['pid'], 'writer_pid': os.getpid(),
                         'remote_sync_guarantee': False})


def verify_storage_probe(probe_reference, root, protocol, dataset):
    from .schema import dereference
    p, m = dereference(probe_reference, root, protocol, 'stage2_storage_probe')
    b = read_json(p.parent/'probe.json')
    check(m['dataset_fingerprint'] == dataset and b['independent_process_readback'] is True
          and type(b['reader_pid']) is int and type(b['writer_pid']) is int
          and b['reader_pid'] != b['writer_pid'], 'Independent storage readback receipt required')
    dereference(b['payload'], root, protocol, 'stage2_storage_probe_payload')


def resource_projection(plan, measurements):
    """Measured first complete chunk only; never estimates hidden dimensions."""
    required = ('token_ranges', 'condition_seconds_per_input', 'forward_seconds', 'audit_seconds',
                'write_seconds', 'verification_seconds', 'gpu_peak_allocated_bytes',
                'gpu_peak_reserved_bytes', 'raw_bytes', 'sidecar_bytes', 'free_space_bytes',
                'remaining_inputs')
    check(all(k in measurements for k in required), 'Incomplete measured first-chunk resource report')
    check(set(measurements['condition_seconds_per_input']) == set(CONDITIONS) and
          set(measurements['remaining_inputs']) == set(CONDITIONS), 'Resource condition coverage mismatch')
    for key in required:
        if key in ('token_ranges', 'condition_seconds_per_input', 'remaining_inputs', 'free_space_bytes'):
            continue
        check(isinstance(measurements[key], (int, float)) and measurements[key] >= 0, 'Invalid resource measurement: ' + key)
    dims = plan['dimensions']
    raw = 4 * sum(plan['counts'][c] * sum(dims[k] for k in plan['required_sites'][c]) for c in CONDITIONS)
    ratio = measurements['sidecar_bytes'] / max(1, measurements['raw_bytes'])
    check(all(isinstance(t, (int, float)) and t >= 0 for t in measurements['condition_seconds_per_input'].values()), 'Invalid throughput measurements')
    check(all(type(n) is int and 0 <= n <= plan['counts'][c] for c, n in measurements['remaining_inputs'].items()), 'Invalid remaining work counts')
    return dict(measurements, expected_raw_matrix_bytes=raw,
                projected_bytes_with_overhead_and_25_percent_reserve=int(raw*(1+ratio)*1.25),
                estimated_remaining_forward_seconds=sum(measurements['remaining_inputs'][c]*measurements['condition_seconds_per_input'][c] for c in CONDITIONS),
                estimate_scope='Measured first complete FIT chunk; Drive free space may be misleading',
                free_space_available=measurements['free_space_bytes'] is not None)


def publish_resource_report(root, plan, first_chunk_manifests, measurements, producer):
    from .features import verify_feature_chunk
    from .schema import dereference
    protocol = read_json(REPO/'configs/protocol.json')
    first = plan['work_units'][0]['chunk_id']
    check(set(first_chunk_manifests) == set(CONDITIONS), 'All first FIT condition chunks required for projection')
    refs, ranges, raw_bytes, sidecar_bytes = {}, {}, 0, 0
    for condition in CONDITIONS:
        marker = first_chunk_manifests[condition]
        b, rows, arrays, _ = verify_feature_chunk(marker, root, protocol, plan)
        check(b['condition'] == condition and b['chunk_id'] == first, 'Resource report must use first complete FIT chunk')
        p, m = artifact(marker, root, protocol, 'stage2_feature_chunk')
        refs[condition] = reference(p, root)
        tokens = read_rows(p.parent/'token_evidence.jsonl')
        ranges[condition] = {}
        for language in (('image',) if condition == 'image_only' else ('en','hi','bn','te')):
            counts = [t['token_count'] for r,t in zip(rows,tokens) if language == 'image' or r['language'] == language]
            ranges[condition][language] = [min(counts), max(counts)]
        raw_bytes += sum(a.nbytes for a in arrays.values())
        sidecar_bytes += sum((p.parent/name).stat().st_size for name in m['files']) - sum(a.nbytes for a in arrays.values())
    measured = dict(measurements, token_ranges=ranges, raw_bytes=raw_bytes, sidecar_bytes=sidecar_bytes)
    body = {'schema':'stage2.resources.v1', 'plan':plain(plan['manifest_reference']),
            'plan_fingerprint':plan['plan_fingerprint'], 'first_chunks':refs,
            'measurements':measured, 'projection':resource_projection(plan, measured)}
    meta = metadata('stage2_resource_report',protocol,plan['dataset_fingerprint'],producer,plan['model_config']['config_fingerprint'])
    meta['upstream_artifacts'] = list(refs.values())
    return publish_json(root, 'results/stage2-validation/'+meta['run_id'],meta,'resources.json',body)


def verify_resource_report(ref, root, protocol, plan):
    from .schema import dereference
    p, m = dereference(ref,root,protocol,'stage2_resource_report')
    b = read_json(p.parent/'resources.json')
    check(b['schema']=='stage2.resources.v1' and b['plan']==plain(plan['manifest_reference'])
          and b['plan_fingerprint']==plan['plan_fingerprint'] and
          m['model_fingerprint']==plan['model_config']['config_fingerprint'], 'Resource report identity mismatch')
    check(b['projection']==resource_projection(plan,b['measurements']), 'Resource projection mismatch')
    check(set(b['first_chunks'])==set(CONDITIONS), 'Incomplete resource first chunk evidence')
    for ref in b['first_chunks'].values():
        dereference(ref,root,protocol,'stage2_feature_chunk')


def validate_suite(root, protocol, inventory, producer):
    """Explicit references only. Missing workers yield a blocked diagnostic."""
    from .plans import load_job_plan
    from .features import load_feature_release
    from .text import load_text_family
    data = load_job_plan(inventory['data_plan'], root, protocol)
    check(data['kind'] == 'stage2_data_plan', 'Suite requires shared data plan')
    blockers, models, families, resources = [], {}, {}, {}
    accepted = []
    for model in ('qwen', 'gemma'):
        supplied = inventory.get('model_plans', {}).get(model)
        if not supplied:
            blockers.append(model + ': model plan missing'); continue
        plan = load_job_plan(supplied, root, protocol)
        check(plan['model'] == model and plan['dataset_fingerprint'] == data['dataset_fingerprint'], 'Suite model/data mismatch')
        from .schema import dereference
        dp, _ = dereference(plan['references']['data_plan'], root, protocol, 'stage2_data_plan')
        check(reference(dp, root) == plain(data['manifest_reference']), 'Suite uses different shared data plans')
        resource = inventory.get('resource_reports',{}).get(model)
        if resource:
            verify_resource_report(reference(resource,root),root,protocol,plan)
            resources[model] = reference(resource,root)
        else:
            blockers.append(model+': measured first-chunk resource report missing')
        seen = set()
        for marker in inventory.get('feature_releases', {}).get(model, []):
            p, _ = artifact(marker, root, protocol, 'stage2_feature_release')
            b = read_json(p.parent/'release.json')
            role = next((r for r, s in ROLES.items() if s == b['split']), None)
            release = load_feature_release(p, root, protocol, role,
                {'plan_fingerprint': plan['plan_fingerprint'], 'condition': b['condition'], 'split': b['split']})
            key = (release.condition, release.split)
            check(key not in seen, 'Duplicate suite release binding')
            seen.add(key); accepted.append(reference(p, root))
        missing = sorted(set((c, s) for c in CONDITIONS for s in SPLITS) - seen)
        if missing:
            blockers.append(model + ': missing releases ' + str(missing))
        models[model] = reference(supplied, root)
    for marker in inventory.get('text_families', []):
        p, _ = artifact(marker, root, protocol, 'stage2_text_family')
        b = read_json(p.parent/'family.json')
        rid = b['recipe']['recipe_fingerprint']
        check(rid not in families, 'Duplicate suite text family')
        expected = {'plan_fingerprint': data['plan_fingerprint'], 'preprocessing_fingerprint': b['preprocessing_fingerprint']}
        for role, split in ROLES.items():
            if split not in b['splits']:
                blockers.append(rid + ': missing ' + split); continue
            reader = load_text_family(p, root, protocol, role, expected)
            reader.read_matrix(split)
        families[rid] = reference(p, root)
    missing = {r['recipe_fingerprint'] for r in data['recipes']} - set(families)
    if missing:
        blockers.append(str(len(missing)) + ' E03 families missing')
    body = {'schema': 'stage2.suite.v1', 'status': 'blocked' if blockers else 'passed',
            'data_plan': plain(data['manifest_reference']), 'model_plans': models,
            'feature_releases': accepted, 'resource_reports': resources,
            'text_families': families, 'blockers': blockers,
            'scientific_results': False, 'full_G0_complete': False}
    meta = metadata('stage2_validation' if blockers else 'stage2_suite_index', protocol, data['dataset_fingerprint'], producer)
    meta['status'] = body['status']
    meta['upstream_artifacts'] = ([plain(data['manifest_reference'])] + list(models.values())
                                + accepted + list(resources.values()) + list(families.values()))
    return publish_json(root, 'results/stage2-validation/' + meta['run_id'], meta, 'report.json', body)


def prepare(root, protocol, dataset, validation, bindings, scope, shared_plan=None):
    from .plans import publish_data_plan, publish_model_plan
    from .readiness import publish_readiness
    producer = current_producer(scope=scope)
    if shared_plan:
        from .plans import load_job_plan
        data=load_job_plan(shared_plan,root,protocol)
        check(data['kind']=='stage2_data_plan','Reuse a shared data plan')
        data_plan=Path(root).resolve()/data['manifest_reference']['path']
        dataset=Path(root).resolve()/data['references']['dataset']['path']
        validation=Path(root).resolve()/data['references']['data_validation']['path']
    else:
        data_plan = publish_data_plan(root, protocol, dataset, validation, producer)
        data=read_json(data_plan.parent/'plan.json')
        validation=Path(root).resolve()/data['references']['data_validation']['path']
    results = {'data_plan': str(data_plan), 'models': {}, 'stage2_complete': False}
    # Independent receipt attempts cannot block the already published shared plan.
    for model, binding in bindings.items():
        check(model in ('qwen', 'gemma'), 'Unknown MODEL binding')
        try:
            receipt = publish_readiness(root, protocol, model, dataset, validation,
                binding['configuration_manifest'], binding.get('model_validation'), binding.get('smoke'),
                current_producer(scope=scope))
            model_plan = publish_model_plan(root, protocol, data_plan, receipt, current_producer(scope=scope))
            results['models'][model] = {'status': 'passed', 'plan_manifest': str(model_plan), 'readiness': str(receipt)}
        except (ValueError, OSError, KeyError) as exc:
            meta = metadata('stage2_readiness_diagnostic', protocol,
                read_json(data_plan)['dataset_fingerprint'], current_producer(scope=scope))
            meta['status'] = 'blocked'
            diagnostic = publish_json(root, 'results/stage2-validation/' + meta['run_id'], meta,
                'report.json', {'status': 'blocked', 'model': model, 'blockers': [str(exc)],
                                'historical_passes_revoked': False})
            results['models'][model] = {'status': 'blocked', 'diagnostic': str(diagnostic)}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'validate', 'preflight', 'probe-read'])
    parser.add_argument('--phase1-root', required=True)
    parser.add_argument('--protocol', type=Path, default=REPO/'configs/protocol.json')
    parser.add_argument('--dataset-manifest')
    parser.add_argument('--data-validation')
    parser.add_argument('--bindings', type=Path)
    parser.add_argument('--inventory', type=Path)
    parser.add_argument('--manifest')
    parser.add_argument('--model', choices=['qwen', 'gemma'])
    parser.add_argument('--storage-scope', choices=['local-development', 'persistent-drive'], default='persistent-drive')
    args = parser.parse_args()
    root, protocol = phase1_root(args.phase1_root), read_json(args.protocol)
    if args.command == 'probe-read':
        p, _ = artifact(args.manifest, root, protocol, 'stage2_storage_probe_payload')
        print(json.dumps({'pid': os.getpid(), 'payload': reference(p, root)}))
        return 0
    if args.command == 'prepare':
        check(args.manifest or args.dataset_manifest, 'Dataset manifest or existing shared plan required')
        result = prepare(root, protocol, args.dataset_manifest, args.data_validation,
                         read_json(args.bindings) if args.bindings else {}, args.storage_scope,args.manifest)
    elif args.command == 'preflight':
        from .plans import resolve_worker_settings
        result = plain(resolve_worker_settings(args.manifest, root, protocol, args.model))
        # Show resolved paths/identities, not the very large row inventory.
        result['plan'] = {k: result['plan'][k] for k in ('kind', 'plan_fingerprint', 'counts')}
    else:
        check(args.inventory, 'Explicit suite inventory JSON required')
        marker = validate_suite(root, protocol, read_json(args.inventory), current_producer(scope=args.storage_scope))
        result = {'manifest': str(marker), 'status': read_json(marker)['status']}
    print(json.dumps(result, indent=2))
    return 2 if result.get('status') == 'blocked' else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ContractError, OSError, KeyError) as exc:
        print(json.dumps({'status': 'blocked', 'blockers': [str(exc)]}))
        sys.exit(2)
