"""Strict feature serialization, restart verification and role-scoped releases."""
import hashlib
from pathlib import Path
import numpy as np

from src.common.contracts import fingerprint
from src.common.storage import file_sha256
from .schema import (FEATURE_SCHEMA, SPLITS, CONDITIONS, ROLES, artifact, authorize, check,
                     component, dereference, frozen, metadata as run_metadata, plain, publish_json,
                     read_json, read_rows, reference, jsonl, contract, VerifiedPlan)
from .plans import load_job_plan, planned_rows
from .evidence import validate_token_evidence


def matrix(path, n, d):
    try:
        a = np.load(path, allow_pickle=False)
    except (ValueError, OSError) as exc:
        raise ValueError('Unreadable non-pickle dense matrix') from exc
    check(isinstance(a, np.ndarray) and a.dtype.str == '<f4' and a.shape == (n, d)
          and a.flags.c_contiguous and np.isfinite(a).all(), 'Invalid dense shape/dtype/order/finite values')
    return a


def vector_digest(vector):
    return hashlib.sha256(np.asarray(vector, dtype='<f4').tobytes(order='C')).hexdigest()


def _plan(plan, root):
    protocol = plain(plan['protocol']) if 'protocol' in plan else None
    # Resolved plans always carry their protocol through the canonical frozen file.
    if protocol is None:
        from .schema import REPO
        protocol = read_json(REPO / 'configs/protocol.json')
    if isinstance(plan, VerifiedPlan):
        # Full source/image validation occurs at plan load. Recheck the immutable
        # plan bytes here without rereading all 3,600 images on every bulk flush.
        dereference(plain(plan['manifest_reference']),root,protocol,'stage2_model_plan')
        check(plan['kind'] == 'stage2_model_plan', 'Model job plan required')
        return plan,protocol
    verified = load_job_plan(plan['manifest_reference']['path'], root, protocol)
    check(plain(verified) == plain(plan), 'Caller plan differs from verified immutable artifact')
    check(plan['kind'] == 'stage2_model_plan', 'Model job plan required')
    return verified, protocol


def _check_chunk(folder, body, plan, root, protocol):
    from .operations import verify_storage_probe
    verify_storage_probe(body['storage_probe'], root, protocol, plan['dataset_fingerprint'])
    if body['chunk_id'] != plan['work_units'][0]['chunk_id']:
        from .operations import verify_resource_report
        verify_resource_report(body['resource_report'],root,protocol,plan)
    check(body['schema'] == FEATURE_SCHEMA and body['plan_fingerprint'] == plan['plan_fingerprint']
          and body['execution_fingerprint'] == plan['execution_identity']['execution_fingerprint']
          and body['source_identity'] == plain(plan['source_identity'])
          and body['model_fingerprint'] == plan['model_config']['config_fingerprint'], 'Feature chunk identity mismatch')
    check(body['model_config'] == plain(plan['model_config']) and
          body['execution_identity'] == plain(plan['execution_identity']) and
          body['bridge_sha256'] == plan['bridge_sha256'] and
          body['policies'] == {'batch_size':1, 'precision':plan['model_config']['precision'],
                               'stored_dtype':'<f4', 'normalization':'none'}, 'Chunk execution/policy provenance mismatch')
    expected = planned_rows(plan, body['chunk_id'], body['condition'])
    rows = read_rows(folder / 'rows.jsonl')
    check(rows == expected and body['rows_sha256'] == file_sha256(folder / 'rows.jsonl'), 'Feature ordered row coverage mismatch')
    condition = body['condition']
    sites = list(plan['required_sites'][condition])
    check(body['sites'] == {k: {'file': f'site_{i:03d}.npy', 'dimension': plan['dimensions'][k],
                              'spec': plain(plan['execution_identity']['site_specs'][k])}
          for i, k in enumerate(sites)}, 'Feature physical site catalog mismatch')
    arrays = {k: matrix(folder / body['sites'][k]['file'], len(rows), plan['dimensions'][k]) for k in sites}
    digests = {k: [vector_digest(v) for v in a] for k, a in arrays.items()}
    check(body['vector_digests'] == digests, 'Canonical float32 vector digest mismatch')
    evidence = read_rows(folder / 'token_evidence.jsonl')
    check(len(evidence) == len(rows), 'Token evidence coverage mismatch')
    for row, item in zip(rows, evidence):
        validate_token_evidence(item, condition, row['source_row_key'], plan['model_config']['max_input_tokens'])
        check(item.get('visibility_reference') == plain(plan['references']['readiness']) and
              item.get('visibility_scope') == 'inherited Stage 1; no bulk attention measurement',
              'Actual spans and inherited visibility evidence must be distinguished')
    audits = read_json(folder / 'audits.json')
    boundaries = sorted({0, len(rows)-1})
    check(sorted(int(k) for k in audits.get('repeated_vectors', {})) == boundaries, 'First/last row repeat evidence required')
    for index in boundaries:
        values = audits['repeated_vectors'][str(index)]
        check(set(values) == set(sites), 'Repeat site coverage mismatch')
        for key in sites:
            check(vector_digest(values[key]) == digests[key][index] and
                  np.asarray(values[key]).shape == (plan['dimensions'][key],), 'Boundary repeat mismatch')
    first_unit = plan['work_units'][0]
    if body['chunk_id'] == first_unit['chunk_id']:
        first_image = rows[0]['image_id']
        indexes = [i for i, r in enumerate(rows) if r['image_id'] == first_image]
        initial = audits.get('public_path_vectors', {})
        check(set(initial) == {str(i) for i in indexes}, 'First FIT image public-path equality required for all core views')
        for i in indexes:
            check(set(initial[str(i)]) == set(sites), 'Public equality site inventory mismatch')
            for key in sites:
                check(vector_digest(initial[str(i)][key]) == digests[key][i], 'Bridge/public equality mismatch')
    visual_refs = audits.get('visual_references', [])
    if condition == 'image_caption':
        check(len(visual_refs) == len(rows), 'Every combined view requires explicit image-only reference')
        visual = {k: matrix(folder / f'visual_{i:03d}.npy', len(rows), plan['dimensions'][k])
                  for i, k in enumerate(plan['audit_sites'])}
        ref_cache = {}
        for i, (row, ref) in enumerate(zip(rows, visual_refs)):
            p, _ = dereference(ref['artifact'], root, protocol, 'stage2_feature_chunk')
            if str(p) not in ref_cache:
                rb = read_json(p.parent / 'chunk.json')
                check(rb['condition'] == 'image_only', 'Visual reference must be explicit image_only')
                ref_cache[str(p)] = _check_chunk(p.parent, rb, plan, root, protocol)
            rr, aa, _ = ref_cache[str(p)]
            j = ref['row_index']
            check(type(j) is int and 0 <= j < len(rr) and rr[j]['image_id'] == row['image_id']
                  and rr[j]['split'] == row['split'], 'Visual reference image/split mismatch')
            for key in plan['audit_sites']:
                check(vector_digest(visual[key][i]) == vector_digest(aa[key][j]), 'Visual context invariance failed')
        expected_visual = {k: [vector_digest(v) for v in a] for k, a in visual.items()}
        check(body['visual_digests'] == expected_visual, 'Visual content digest mismatch')
    else:
        check(not visual_refs and not body['visual_digests'], 'Unexpected visual audit aliases')
    return rows, arrays, digests


@contract
def publish_feature_chunk(root, plan, chunk_id, condition, attempt_id, payload_paths, metadata):
    plan, protocol = _plan(plan, root)
    component(attempt_id)
    rows = planned_rows(plan, chunk_id, condition)
    files = {k: Path(v) for k, v in payload_paths.items()}
    sites = list(plan['required_sites'][condition])
    expected_files = {'rows.jsonl', 'token_evidence.jsonl', 'audits.json'} | {f'site_{i:03d}.npy' for i in range(len(sites))}
    if condition == 'image_caption':
        expected_files |= {f'visual_{i:03d}.npy' for i in range(len(plan['audit_sites']))}
    check(set(files) == expected_files, 'Feature payload file inventory mismatch')
    check(read_rows(files['rows.jsonl']) == rows, 'Producer rows differ from plan')
    # Validate from a staging directory with canonical names before completion.
    import tempfile, shutil
    with tempfile.TemporaryDirectory() as temp:
        folder = Path(temp)
        for k, v in files.items():
            shutil.copyfile(v, folder / k)
        body = {'schema': FEATURE_SCHEMA, 'chunk_id': chunk_id, 'condition': condition,
                'storage_probe': metadata['storage_probe'],
                'resource_report': metadata.get('resource_report'),
                'split': rows[0]['split'], 'plan_fingerprint': plan['plan_fingerprint'],
                'plan': plain(plan['manifest_reference']),
                'execution_fingerprint': plan['execution_identity']['execution_fingerprint'],
                'source_identity': plain(plan['source_identity']),
                'model_fingerprint': plan['model_config']['config_fingerprint'],
                'model_config': plain(plan['model_config']),
                'execution_identity': plain(plan['execution_identity']),
                'bridge_sha256': plan['bridge_sha256'],
                'policies': {'batch_size':1,'precision':plan['model_config']['precision'],
                             'stored_dtype':'<f4','normalization':'none'},
                'rows_sha256': file_sha256(folder / 'rows.jsonl'),
                'sites': {k: {'file': f'site_{i:03d}.npy', 'dimension': plan['dimensions'][k],
                              'spec': plain(plan['execution_identity']['site_specs'][k])} for i, k in enumerate(sites)},
                'vector_digests': {k: [vector_digest(v) for v in matrix(folder/f'site_{i:03d}.npy', len(rows), plan['dimensions'][k])]
                                   for i, k in enumerate(sites)},
                'visual_digests': {k: [vector_digest(v) for v in matrix(folder/f'visual_{i:03d}.npy', len(rows), plan['dimensions'][k])]
                                   for i, k in enumerate(plan['audit_sites'])} if condition == 'image_caption' else {}}
        _check_chunk(folder, body, plan, root, protocol)
        meta = run_metadata('stage2_feature_chunk', protocol, plan['dataset_fingerprint'], metadata, plan['model_config']['config_fingerprint'])
        meta.update(plan_fingerprint=plan['plan_fingerprint'], execution_fingerprint=body['execution_fingerprint'],
                    upstream_artifacts=[plain(plan['manifest_reference'])])
        output = publish_json(root, f"features/{plan['job_id']}/{attempt_id}/{body['split']}/{condition}/{chunk_id}",
                              meta, 'chunk.json', body, {k: folder/k for k in files})
    verify_feature_chunk(output, root, protocol, plan)
    return output


@contract
def verify_feature_chunk(manifest, root, protocol, plan):
    p, m = artifact(manifest, root, protocol, 'stage2_feature_chunk')
    check(m['dataset_fingerprint'] == plan['dataset_fingerprint'] and
          m['model_fingerprint'] == plan['model_config']['config_fingerprint'] and
          m['plan_fingerprint'] == plan['plan_fingerprint'] and
          m['execution_fingerprint'] == plan['execution_identity']['execution_fingerprint'], 'Feature outer identity mismatch')
    body = read_json(p.parent / 'chunk.json')
    dereference(body['plan'], root, protocol, 'stage2_model_plan')
    check(body['plan'] == plain(plan['manifest_reference']), 'Feature plan reference mismatch')
    rows, arrays, digests = _check_chunk(p.parent, body, plan, root, protocol)
    check(body['split'] == rows[0]['split'], 'Chunk split mismatch')
    return body, rows, arrays, digests


@contract
def publish_release_index(root, plan, condition, split, chunk_manifests, run_id):
    plan, protocol = _plan(plan, root)
    check(condition in CONDITIONS and split in SPLITS, 'Invalid release condition/split')
    component(run_id)
    expected = [u['chunk_id'] for u in plan['work_units'] if u['split'] == split]
    candidates = {}
    for manifest in chunk_manifests:
        body, rows, _, digests = verify_feature_chunk(manifest, root, protocol, plan)
        check(body['condition'] == condition and body['split'] == split, 'Release contains wrong condition/split')
        ref = reference(manifest, root)
        candidates.setdefault(body['chunk_id'], []).append((ref, digests))
    check(set(candidates) == set(expected), 'Incomplete/extra release chunk coverage')
    chosen, retained = [], []
    for chunk in expected:
        values = candidates[chunk]
        check(len({fingerprint(d) for _, d in values}) == 1, 'Numerical reproducibility conflict for the same feature keys')
        ordered = sorted((r for r, _ in values), key=lambda r: r['sha256'])
        chosen.append(ordered[0]); retained.extend(ordered)
    body = {'schema': FEATURE_SCHEMA, 'plan': plain(plan['manifest_reference']),
            'plan_fingerprint': plan['plan_fingerprint'], 'condition': condition, 'split': split,
            'chunks': chosen, 'all_compatible_attempts': retained,
            'row_count': sum(len(u['rows'][condition]) for u in plan['work_units'] if u['split'] == split),
            'sites': list(plan['required_sites'][condition])}
    producer = _index_producer(plan, root, protocol, run_id)
    meta = run_metadata('stage2_feature_release', protocol, plan['dataset_fingerprint'], producer, plan['model_config']['config_fingerprint'])
    meta['upstream_artifacts'] = chosen
    return publish_json(root, f"features/{plan['job_id']}/indexes/{run_id}", meta, 'release.json', body)


def _index_producer(plan, root, protocol, run_id):
    from .operations import current_producer
    _, meta = artifact(plan['manifest_reference']['path'],root,protocol,'stage2_model_plan')
    return current_producer(run_id, meta['storage_scope'])


class FeatureRelease:
    def __init__(self, manifest, root, protocol, role, expected):
        self.manifest, self.root, self.protocol, self.role, self.expected = manifest, root, protocol, role, expected
        self._reference = reference(manifest, root)
        p, m = artifact(manifest, root, protocol, 'stage2_feature_release')
        body = read_json(p.parent / 'release.json')
        check(body['schema'] == FEATURE_SCHEMA, 'Feature release schema mismatch')
        authorize(role, body['split'])  # before arrays, including cache hits
        check({'plan_fingerprint','condition','split'} <= set(expected), 'Explicit plan/condition/split release identities required')
        for key, value in expected.items():
            check(body.get(key) == plain(value), 'Unexpected release identity: ' + key)
        pp, _ = dereference(body['plan'], root, protocol, 'stage2_model_plan')
        plan = load_job_plan(pp, root, protocol)
        check(body['plan_fingerprint'] == plan['plan_fingerprint'] and m['dataset_fingerprint'] == plan['dataset_fingerprint']
              and m['model_fingerprint'] == plan['model_config']['config_fingerprint'], 'Release common identity mismatch')
        expected_units = [u for u in plan['work_units'] if u['split'] == body['split']]
        chunks = []
        for ref in body['chunks']:
            cp, _ = dereference(ref, root, protocol, 'stage2_feature_chunk')
            cb, rr, _, cd = verify_feature_chunk(cp, root, protocol, plan)
            chunks.append((cb, rr, None, cd))
        check([b['chunk_id'] for b, _, _, _ in chunks] == [u['chunk_id'] for u in expected_units]
              and all(b['condition'] == body['condition'] and b['split'] == body['split'] for b, _, _, _ in chunks),
              'Release exact chunk coverage mismatch')
        check(body['row_count'] == sum(len(r) for _, r, _, _ in chunks) and
              body['sites'] == list(plan['required_sites'][body['condition']]), 'Release counts/sites mismatch')
        # Independently recheck every retained duplicate, not only the chosen attempt.
        selected = {b['chunk_id']: d for b, _, _, d in chunks}
        for ref in body['all_compatible_attempts']:
            cp, _ = dereference(ref, root, protocol, 'stage2_feature_chunk')
            cb, _, _, cd = verify_feature_chunk(cp, root, protocol, plan)
            check(cb['condition'] == body['condition'] and cb['split'] == body['split']
                  and selected.get(cb['chunk_id']) == cd, 'Retained candidate numerical conflict')
        self.rows = tuple(frozen(r) for _, rr, _, _ in chunks for r in rr)
        self.site_catalog = frozen({k: {'dimension': plan['dimensions'][k], 'spec': plain(plan['execution_identity']['site_specs'][k])}
                                    for k in body['sites']})
        self.condition, self.split = body['condition'], body['split']
        self._chunks = tuple(body['chunks'])
        self.plan = plan

    def read_matrix(self, site_key, language=None):
        authorize(self.role, self.split)
        dereference(self._reference,self.root,self.protocol,'stage2_feature_release')
        check(site_key in self.site_catalog, 'Unknown release site')
        check(language is None or language in ('en', 'hi', 'bn', 'te'), 'Unknown language')
        check(self.condition != 'image_only' or language is None, 'image_only has no language observations')
        # Reverify on every read, including if files changed after reader construction.
        rows, values = [], []
        for ref in self._chunks:
            cp, _ = dereference(ref, self.root, self.protocol, 'stage2_feature_chunk')
            _, rr, aa, _ = verify_feature_chunk(cp, self.root, self.protocol, self.plan)
            for i, row in enumerate(rr):
                if language is None or row['language'] == language:
                    rows.append(frozen(row)); values.append(aa[site_key][i])
        return tuple(rows), np.asarray(values, dtype='<f4')


@contract
def load_feature_release(index_manifest, root, protocol, role, expected):
    return FeatureRelease(index_manifest, root, protocol, role, expected)
