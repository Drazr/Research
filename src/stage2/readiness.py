"""Typed Stage 1 evidence reuse, preserving original commits and fail-closed gates."""
import subprocess
from pathlib import Path

from src.common.contracts import (INSPECTION_CHECKS, decoder_depths, fingerprint,
                                  resolve_model)
from src.common.cache_keys import execution_fingerprint, INFERENCE_FIELDS
from src.common.storage import file_sha256, within
from .schema import (REPO, artifact, check, dereference, frozen, metadata, plain,
                     publish_json, read_json, reference, specification, contract)

MODEL_FILES = ('src/models/hf.py', 'src/models/spans.py', 'src/models/registry.py',
               'src/common/contracts.py', 'src/common/cache_keys.py')


def git_source_hash(commit, path):
    import hashlib
    try:
        data = subprocess.check_output(['git', 'show', commit + ':' + path], cwd=REPO,
                                       stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as exc:
        raise ValueError('Original source unavailable; fetch original commit before evidence reuse') from exc
    return hashlib.sha256(data).hexdigest()


def check_sources(meta, script):
    # Smoke historically records only its entry-file hash; recover other files
    # from its original immutable commit, never invent a current code_commit.
    expected = {p: file_sha256(REPO / p) for p in MODEL_FILES + (script,)}
    for path, digest in expected.items():
        recorded = meta.get('source_files_sha256', {}).get(path)
        original = git_source_hash(meta['code_commit'], path)
        check(recorded in (None, original), 'Recorded source contradicts original commit: ' + path)
        check(original == digest, 'Relevant source changed; affected validation required: ' + path)
    entry = meta.get('script_sha256', meta.get('source_sha256'))
    check(entry == expected[script], 'Entry-point evidence source mismatch')
    return expected


def verify_data(dataset_manifest, validation_manifest, root, protocol):
    from src.common.data import load_dataset
    path, data = artifact(dataset_manifest, root, protocol, 'stage1_dataset')
    bundle = load_dataset(path, root, protocol)
    vp, validation = artifact(validation_manifest, root, protocol, 'stage1_data_validation')
    report = read_json(vp.parent / 'report.json')
    check(validation['dataset_fingerprint'] == bundle.dataset_fingerprint,
          'Data-validation dataset mismatch')
    check(report.get('status') == 'passed' and report.get('deterministic_reload') is True,
          'Passed deterministic data validation required')
    check(report.get('dataset_fingerprint') == data['dataset_fingerprint'] and
          report.get('protocol_fingerprint') == fingerprint(protocol), 'Data report identity mismatch')
    check(all(v is True for v in report.get('checks', {}).values()) and bool(report.get('checks')),
          'Incomplete data-validation checks')
    refs = [reference(path, root), reference(vp, root)]
    return bundle, refs


def discover_data_validation(dataset_manifest, root, protocol):
    dp, dm = artifact(dataset_manifest, root, protocol, 'stage1_dataset')
    folder = within(Path(root).resolve(), 'shared/'+fingerprint(protocol)+'/validation')
    candidates=[]
    for marker in sorted(folder.glob('*/_COMPLETE.json')):
        try:
            p,m=artifact(marker,root,protocol,'stage1_data_validation')
            report=read_json(p.parent/'report.json')
            if m['dataset_fingerprint']==dm['dataset_fingerprint'] and report.get('status')=='passed' and report.get('deterministic_reload') is True:
                candidates.append(reference(p,root))
        except (ValueError,OSError):
            continue
    check(candidates,'No passed data-validation artifact found in declared shared namespace')
    candidates.sort(key=lambda r:r['sha256'])
    return within(Path(root).resolve(),candidates[0]['path']), candidates


def _model_evidence(root, protocol, model_config, model, refs):
    cp, cm = dereference(refs['configuration'], root, protocol, 'stage1_runtime_configuration')
    generated = resolve_model(read_json(cp.parent / 'models.json'), model)
    check(generated == plain(model_config), 'Generated model configuration mismatch')
    cr = read_json(cp.parent / 'configuration_report.json')
    check(cr.get('status') == 'passed' and cr.get('configuration_resolved') is True and
          not cr.get('blockers') and cr.get('weights_loaded') is False and
          cr.get('model_fingerprint') == generated['config_fingerprint'], 'Invalid configuration report')
    mp, mm = dereference(refs['model_validation'], root, protocol, 'stage1_model_validation')
    sp, sm = dereference(refs['smoke'], root, protocol, 'stage1_smoke')
    for meta in (cm, mm, sm):
        check(meta['dataset_fingerprint'] == refs['dataset_fingerprint'] and
              meta['model_fingerprint'] == generated['config_fingerprint'], 'Model evidence identity mismatch')
    mr, sr = read_json(mp.parent / 'model_validation.json'), read_json(sp.parent / 'report.json')
    check(mr.get('status') == 'passed' and mr.get('gpu_checks_passed') is True and
          mr.get('code_dirty') is False and not mr.get('blockers'), 'Actual clean model validation required')
    check(mr.get('resolved_config') == generated, 'Validation used a different resolved configuration')
    for report in (mr, sr):
        check(report.get('status') == 'passed' and not report.get('blockers'), 'Passed model payload required')
        check(all(report.get('inspection', {}).get('checks', {}).get(k) is True
                  for k in INSPECTION_CHECKS), 'Missing actual inspection checks')
        check(report.get('inspection', {}).get('records'), 'Actual FIT token/visibility records required')
    for k in ('data_contract', 'adapter_inspection', 'finite_features', 'repeat_extraction',
              'toy_optimizer', 'permutation_plumbing'):
        check(sr.get('checks', {}).get(k) is True, 'Final smoke check missing: ' + k)
    check(sr.get('kind') == 'stage1_smoke' and sr.get('toy_optimizer') and
          len(sr.get('permutation_plumbing', [])) == 3, 'Final smoke is not optimizer/permutation evidence')
    info = mr['adapter']
    depths = list(decoder_depths(info['decoder_blocks']))
    check(info['decoder_depths'] == depths and info['midpoint'] == (info['decoder_blocks'] + 1)//2,
          'Invalid fixed depth inventory')
    decoder = {f'decoder:{d}:{p}' for d in depths for p in ('last_caption', 'caption_mean')}
    visual = {k for k in info['supported_sites'] if k.startswith(('encoder:', 'projector:'))}
    check(bool(visual) and set(info['supported_sites']) == decoder | visual, 'Incomplete scheduled sites')
    expected = {'image_caption': sorted(decoder | visual), 'caption_only': sorted(decoder),
                'image_only': sorted(visual)}
    check({k: sorted(v) for k, v in mr.get('conditions_validated', {}).items()} == expected,
          'Actual three-condition validation required')
    check(mr.get('repeat_extraction') is True and mr.get('visual_context_invariance') is True,
          'Actual repeat/visual validation required')
    check(set(mr['dimensions']) == decoder | visual and all(type(d) is int and d > 0
          for d in mr['dimensions'].values()), 'Invalid actual dimensions')
    for key in ('repeat_atol', 'repeat_rtol'):
        check(info[key] == generated[key] == 0, 'Unversioned repeat tolerance change')
    execution = mr['execution_identity']
    check(execution['execution_fingerprint'] == execution_fingerprint(generated,
          execution['backend'], execution['source_sha256']), 'Execution fingerprint mismatch')
    check(execution['source_sha256'] == {Path(p).name: file_sha256(REPO / p)
          for p in MODEL_FILES[:2]}, 'Inference source changed')
    check(set(execution['site_specs']) == decoder | visual, 'Physical site catalog mismatch')
    backend = execution['backend']
    check(backend.get('device_type') == 'cuda' and 'L4' in backend.get('device_name', '') and
          backend.get('batch_size') == 1 and backend.get('eval') is True and
          backend.get('truncation') is False and backend.get('padding') is False,
          'Actual selected L4 backend required')
    source = check_sources(mm, 'src/stage1/validate_model.py')
    source.update(check_sources(sm, 'src/stage1/run_validity_smoke.py'))
    check(sr['adapter']['model_fingerprint'] == generated['config_fingerprint'] and
          sr['adapter']['decoder_blocks'] == info['decoder_blocks'], 'Smoke adapter mismatch')
    return {'model': model, 'model_config': generated, 'execution_identity': execution,
            'dimensions': mr['dimensions'], 'adapter': info, 'source_compatibility': source,
            'original_commits': {k: v['code_commit'] for k, v in
                (('configuration', cm), ('model_validation', mm), ('smoke', sm))},
            'conditions_validated': expected}


def discover_evidence(root, protocol, model_config, kind, dataset_fingerprint):
    """Inspect only the explicit model namespace. No newest-file fallback."""
    check(kind in ('stage1_model_validation', 'stage1_smoke'), 'Unsupported discovery kind')
    candidates = []
    folder = within(Path(root).resolve(), 'smoke/' + model_config['config_fingerprint'])
    for marker in sorted(folder.glob('*/_COMPLETE.json')):
        try:
            p, m = artifact(marker, root, protocol, kind)
            if m['dataset_fingerprint'] != dataset_fingerprint:
                continue
            if m.get('model_fingerprint') != model_config['config_fingerprint']:
                continue
            payload = read_json(p.parent / ('model_validation.json' if kind == 'stage1_model_validation' else 'report.json'))
            if payload.get('status') != 'passed' or payload.get('blockers'):
                continue
            if kind == 'stage1_model_validation' and payload.get('resolved_config') not in (None, plain(model_config)):
                continue
            candidates.append((reference(p, root), payload))
        except (ValueError, OSError):
            continue
    check(candidates, 'No compatible typed passed evidence found in declared model namespace')
    # A same-alias report with a different execution/site catalog cannot be guessed.
    identities = {fingerprint({k: p.get(k) for k in
                  ('resolved_config', 'execution_identity', 'dimensions', 'adapter')}) for _, p in candidates}
    check(len(identities) == 1, 'Conflicting evidence identities; supply an explicit binding')
    refs = sorted((r for r, _ in candidates), key=lambda r: r['sha256'])
    return refs[0], refs


def publish_readiness(root, protocol, model, dataset_manifest, data_validation,
                      configuration_manifest, model_validation, smoke, producer):
    bundle, data_refs = verify_data(dataset_manifest, data_validation, root, protocol)
    cp, _ = artifact(configuration_manifest, root, protocol, 'stage1_runtime_configuration')
    config = resolve_model(read_json(cp.parent / 'models.json'), model)
    refs = {'dataset': data_refs[0], 'data_validation': data_refs[1],
            'dataset_fingerprint': bundle.dataset_fingerprint,
            'configuration': reference(cp, root)}
    candidates = {}
    for key, supplied, kind in (('model_validation', model_validation, 'stage1_model_validation'),
                               ('smoke', smoke, 'stage1_smoke')):
        if supplied:
            refs[key] = reference(artifact(supplied, root, protocol, kind)[0], root)
        else:
            refs[key], candidates[key] = discover_evidence(root, protocol, config, kind, bundle.dataset_fingerprint)
    body = _model_evidence(root, protocol, config, model, refs)
    body.update(schema='stage2.readiness.v1', specification=specification(),
                references=refs, discovery_candidates=candidates)
    body['readiness_fingerprint'] = fingerprint(body)
    meta = metadata('stage2_model_readiness', protocol, bundle.dataset_fingerprint, producer, config['config_fingerprint'])
    meta['upstream_artifacts'] = [v for v in refs.values() if isinstance(v, dict)]
    return publish_json(root, 'cache/shared/stage2-plans/' + meta['run_id'], meta, 'readiness.json', body)


@contract
def verify_readiness(receipt_manifest, root, protocol, model_config):
    path, meta = artifact(receipt_manifest, root, protocol, 'stage2_model_readiness')
    body = read_json(path.parent / 'readiness.json')
    digest = body.pop('readiness_fingerprint', None)
    check(digest == fingerprint(body), 'Readiness fingerprint mismatch')
    check(body['schema'] == 'stage2.readiness.v1' and body['specification'] == specification(), 'Readiness schema/specification mismatch')
    refs = body['references']
    dp, _ = dereference(refs['dataset'], root, protocol, 'stage1_dataset')
    vp, _ = dereference(refs['data_validation'], root, protocol, 'stage1_data_validation')
    bundle, _ = verify_data(dp, vp, root, protocol)
    check(meta['dataset_fingerprint'] == refs['dataset_fingerprint'] == bundle.dataset_fingerprint,
          'Readiness dataset mismatch')
    actual = _model_evidence(root, protocol, model_config, body['model'], refs)
    check(all(body[k] == v for k, v in actual.items()), 'Readiness payload contradicts upstream evidence')
    check(meta['model_fingerprint'] == model_config['config_fingerprint'], 'Receipt model mismatch')
    body['readiness_fingerprint'] = digest
    return frozen(body)


def assert_runtime(adapter, plan):
    """Call after load, before any production request; never trust an alias."""
    check(adapter.config == plain(plan['model_config']), 'Loaded config differs from plan')
    check(adapter.describe_execution() == plain(plan['execution_identity']), 'Actual backend/execution differs from accepted plan')
