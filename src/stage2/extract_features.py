"""Plan-bound Stage 2 VLM extraction. No fitting, generation or scoring."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time

import numpy as np

from src.common.contracts import ContractError, ExtractionRequest, FeatureSite, LANGUAGES
from src.common.cache_keys import input_view_fingerprint
from src.common.data import load_dataset
from src.common.storage import new_run_id, phase1_root
from src.models.registry import create_adapter
from src.stage2.interfaces import (
    assert_runtime, compare_public_path, current_producer, extract_with_evidence,
    planned_rows, publish_feature_chunk, publish_release_index, publish_resource_report,
    resolve_worker_settings, storage_probe, vector_digest, verify_feature_chunk,
)
from src.stage2.schema import (
    REPO, CONDITIONS, SPLITS, artifact, check, dereference, json_bytes, jsonl,
    metadata, plain, publish_json, read_json, reference,
)

VISIBILITY_SCOPE = 'inherited Stage 1; no bulk attention measurement'


def safe_error(exc):
    message = str(exc)
    token = os.environ.get('HF_TOKEN')
    if token:
        message = message.replace(token, '[REDACTED]')
    return re.sub(r'hf_[A-Za-z0-9]+', '[REDACTED]', message)


def sites_for(plan, condition):
    keys = list(plan['required_sites'][condition])
    if condition == 'image_caption':
        keys += list(plan['audit_sites'])
    sites = []
    for key in keys:
        stage, depth, pooling = key.split(':')
        sites.append(FeatureSite(stage, None if depth == 'None' else int(depth), pooling))
    return sites


def bind_inputs(plan, records):
    """Validate all joins and permissions before any adapter/cache use."""
    selected = {}
    for row in records:
        if row.split == 'CONFIRM':
            continue
        check(row.row_key not in selected, 'Duplicate source row')
        selected[row.row_key] = row
    used = set()
    for unit in plan['work_units']:
        check(unit['split'] in SPLITS and 0 < len(unit['image_ids']) <= 32,
              'Invalid preliminary image chunk')
        for condition in CONDITIONS:
            for row in planned_rows(plan, unit['chunk_id'], condition):
                key = tuple(row['source_row_key'])
                check(key in selected, 'Missing selected source row')
                source = selected[key]
                check(source.split == row['split'] == unit['split'] and source.image_id == row['image_id'],
                      'Source partition/image mismatch')
                check(input_view_fingerprint(source, condition) == row['input_view_fingerprint'],
                      'Selected caption/image input differs from plan')
                check(condition != 'image_only' or source.language == 'en', 'Image reference needs English pointer')
                if condition != 'image_only':
                    check(source.language == row['language'] and source.caption_id == row['caption_id'] and
                          hashlib.sha256(source.caption_text.encode('utf-8')).hexdigest() == row['caption_text_sha256'],
                          'Selected caption identity mismatch')
                    used.add(key)
                ExtractionRequest([source], sites_for(plan, condition), condition,
                                  purpose='preliminary_features').validate()
    check(used == set(selected), 'Plan/source eligible population differs')
    return selected


def load_resume(indexes, root, protocol, plan):
    """Explicit immutable progress indexes only; no directory discovery."""
    candidates, issues, resource, measurements = {}, [], None, {}
    for index in indexes:
        p, meta = artifact(index, root, protocol, 'stage2_extraction_progress')
        body = read_json(p.parent / 'progress.json')
        check(body['schema'] == 'stage2.extraction-progress.v1' and
              body['plan'] == plain(plan['manifest_reference']) and
              body['plan_fingerprint'] == meta['plan_fingerprint'] == plan['plan_fingerprint'] and
              body['execution_fingerprint'] == meta['execution_fingerprint'] == plan['execution_identity']['execution_fingerprint'] and
              meta['dataset_fingerprint'] == plan['dataset_fingerprint'] and
              meta['model_fingerprint'] == plan['model_config']['config_fingerprint'], 'Resume plan/schema mismatch')
        dereference(body['plan'], root, protocol, 'stage2_model_plan')
        if body.get('resource_report'):
            try:
                dereference(body['resource_report'], root, protocol, 'stage2_resource_report')
                resource = body['resource_report']
            except (ValueError, OSError) as exc:
                issues.append({'artifact': body['resource_report'], 'error': safe_error(exc),
                               'action': 'rebuild resource report from verified first-chunk measurements'})
        for ref in body['chunks']:
            try:
                cp, _ = dereference(ref, root, protocol, 'stage2_feature_chunk')
                chunk, _, _, digests = verify_feature_chunk(cp, root, protocol, plan)
            except (ValueError, OSError) as exc:
                issues.append({'artifact': ref, 'error': safe_error(exc), 'action': 'reextract affected work'})
                continue
            key = (chunk['chunk_id'], chunk['condition'])
            entries = candidates.setdefault(key, [])
            check(all(digests == old[1] and chunk['visual_digests'] == old[2] for old in entries),
                  'Numerical reproducibility conflict in resume candidates')
            if not any(old[0] == ref for old in entries):
                entries.append((ref, digests, chunk['visual_digests']))
            measured = body.get('first_measurements', {}).get(chunk['condition'])
            if chunk['chunk_id'] == plan['work_units'][0]['chunk_id'] and measured and measured['artifact'] == ref:
                old = measurements.get(chunk['condition'])
                if old is None or ref['sha256'] < old['artifact']['sha256']:
                    measurements[chunk['condition']] = measured
    accepted = {k: [v[0] for v in sorted(vs, key=lambda v: v[0]['sha256'])] for k, vs in candidates.items()}
    if resource:
        rp, _ = dereference(resource, root, protocol, 'stage2_resource_report')
        first_refs = read_json(rp.parent/'resources.json')['first_chunks']
        first = plan['work_units'][0]['chunk_id']
        if any(first_refs[c] not in accepted.get((first, c), []) for c in CONDITIONS):
            issues.append({'artifact': resource, 'error': 'First-chunk reference no longer accepted',
                           'action': 'rebuild measured resource report'})
            resource = None
    return accepted, issues, resource, measurements


class Extractor:
    """One model, at most one 32-image CPU chunk, one input per forward."""
    def __init__(self, root, protocol, plan, inputs, producer, resume_indexes=(), emit=None):
        self.root, self.protocol, self.plan = Path(root), protocol, plan
        self.inputs, self.producer = inputs, producer
        self.emit = emit or (lambda value: print(json.dumps(value), flush=True))
        self.context = {'operation': 'resume verification'}
        self.accepted, self.issues, self.resource, self.first_measurements = load_resume(resume_indexes, root, protocol, plan)
        self.releases = []
        self.probe = None
        self.adapter = None
        self.latest_progress = None
        self.seconds = {c: 0.0 for c in CONDITIONS}
        self.calls = {c: 0 for c in CONDITIONS}
        self.audit_seconds = self.write_seconds = self.verification_seconds = 0.0
        self.audit_by_condition = {c: 0.0 for c in CONDITIONS}

    def snapshot(self):
        producer = dict(self.producer, run_id=new_run_id())
        body = {'schema': 'stage2.extraction-progress.v1', 'plan': plain(self.plan['manifest_reference']),
                'plan_fingerprint': self.plan['plan_fingerprint'],
                'execution_fingerprint': self.plan['execution_identity']['execution_fingerprint'],
                'chunks': [r for refs in self.accepted.values() for r in refs],
                'releases': self.releases, 'resource_report': self.resource,
                'first_measurements': self.first_measurements,
                'issues': self.issues, 'stage2_complete': False}
        meta = metadata('stage2_extraction_progress', self.protocol, self.plan['dataset_fingerprint'],
                        producer, self.plan['model_config']['config_fingerprint'])
        meta.update(plan_fingerprint=body['plan_fingerprint'], execution_fingerprint=body['execution_fingerprint'],
                    upstream_artifacts=[body['plan']] + body['chunks'] + self.releases +
                    ([self.resource] if self.resource else []))
        self.latest_progress = publish_json(self.root, 'results/stage2-validation/' + producer['run_id'],
                                            meta, 'progress.json', body)
        self.emit({'event': 'resume_index', 'manifest': str(self.latest_progress),
                   'sha256': reference(self.latest_progress, self.root)['sha256']})

    def forward(self, row, condition, initial=False, audit=False):
        self.context = {'operation': 'forward', 'image_id': row.image_id, 'source_row_key': list(row.row_key),
                        'condition': condition, 'sites': [s.key for s in sites_for(self.plan, condition)]}
        request = ExtractionRequest([row], sites_for(self.plan, condition), condition,
                                    purpose='preliminary_features')
        start = time.perf_counter()
        result = (compare_public_path if initial else extract_with_evidence)(self.adapter, request)
        elapsed = time.perf_counter() - start
        if audit:
            self.audit_seconds += elapsed
            self.audit_by_condition[condition] += elapsed
        else:
            self.seconds[condition] += elapsed
            self.calls[condition] += 1
        values = {k: np.asarray(v[0], dtype='<f4') for k, v in result.features.vectors.items()}
        for key, value in values.items():
            check(value.shape == (self.plan['dimensions'][key],) and np.isfinite(value).all(),
                  'Nonfinite/wrong-dimension vector at ' + key)
        evidence = dict(result.evidence[0], visibility_reference=plain(self.plan['references']['readiness']),
                        visibility_scope=VISIBILITY_SCOPE)
        return values, evidence

    def chunk(self, unit):
        chunk_id = unit['chunk_id']
        pending = [c for c in CONDITIONS if (chunk_id, c) not in self.accepted]
        if not pending:
            return
        first = chunk_id == self.plan['work_units'][0]['chunk_id']
        tables = {c: planned_rows(self.plan, chunk_id, c) for c in CONDITIONS}
        buffers = {c: {'values': [], 'evidence': [], 'public': {}, 'visual_checks': []} for c in pending}
        image_values = {}
        image_marker = None
        if 'image_only' not in pending:
            image_marker = self.root / self.accepted[(chunk_id, 'image_only')][0]['path']
            _, rows, arrays, _ = verify_feature_chunk(image_marker, self.root, self.protocol, self.plan)
            image_values = {r['image_id']: {k: a[i] for k, a in arrays.items()} for i, r in enumerate(rows)}
        by_image = {c: {} for c in CONDITIONS}
        for c, rows in tables.items():
            for row in rows:
                by_image[c].setdefault(row['image_id'], []).append(row)
        for image in unit['image_ids']:
            for condition in ('image_only', 'image_caption', 'caption_only'):
                if condition not in pending:
                    continue
                for row in by_image[condition][image]:
                    source = self.inputs[tuple(row['source_row_key'])]
                    initial = first and image == unit['image_ids'][0]
                    values, evidence = self.forward(source, condition, initial)
                    buf = buffers[condition]
                    buf['values'].append(values); buf['evidence'].append(evidence)
                    if initial:
                        buf['public'][str(row['row_index'])] = {
                            k: values[k].tolist() for k in self.plan['required_sites'][condition]}
                    if condition == 'image_only':
                        image_values[image] = values
                    if condition == 'image_caption':
                        for key in self.plan['audit_sites']:
                            a, b = values[key], image_values[image][key]
                            same = vector_digest(a) == vector_digest(b)
                            detail = {'image_id': image, 'language': row['language'], 'site': key,
                                      'equal': same, 'max_absolute_difference': float(np.max(np.abs(a-b))),
                                      'combined_digest': vector_digest(a), 'image_only_digest': vector_digest(b)}
                            buf['visual_checks'].append(detail)
                            self.context['visual_comparison'] = detail
                            check(same, 'Visual context invariance failed at ' + key)
        # The image-only artifact must complete before combined references can publish.
        for condition in ('image_only', 'image_caption', 'caption_only'):
            if condition not in pending:
                continue
            rows, buf = tables[condition], buffers[condition]
            repeated = {}
            for i in sorted({0, len(rows)-1}):
                source = self.inputs[tuple(rows[i]['source_row_key'])]
                values, _ = self.forward(source, condition, audit=True)
                for key, value in values.items():
                    self.context['site'] = key
                    original = buf['values'][i][key]
                    same = vector_digest(value) == vector_digest(original)
                    if not same:
                        self.context['repeat_comparison'] = {
                            'row_index': i, 'expected_digest': vector_digest(original),
                            'observed_digest': vector_digest(value),
                            'max_absolute_difference': float(np.max(np.abs(original-value))),
                            'expected_vector': original.tolist(), 'observed_vector': value.tolist()}
                    check(same, 'Boundary repeat mismatch at ' + key)
                repeated[str(i)] = {k: values[k].tolist() for k in self.plan['required_sites'][condition]}
            refs = []
            if condition == 'image_caption':
                image_ref = reference(image_marker, self.root)
                positions = {r['image_id']: i for i, r in enumerate(tables['image_only'])}
                refs = [{'artifact': image_ref, 'row_index': positions[r['image_id']]} for r in rows]
            audits = {'repeated_vectors': repeated, 'public_path_vectors': buf['public'],
                      'visual_references': refs, 'visual_comparisons': buf['visual_checks']}
            self.context = {'operation': 'publication', 'chunk_id': chunk_id, 'condition': condition}
            start = time.perf_counter()
            with tempfile.TemporaryDirectory(prefix='stage2-extract-') as temp:
                folder = Path(temp)
                (folder/'rows.jsonl').write_bytes(jsonl(rows))
                (folder/'token_evidence.jsonl').write_bytes(jsonl(buf['evidence']))
                (folder/'audits.json').write_bytes(json_bytes(audits))
                for prefix, keys in (('site', self.plan['required_sites'][condition]),
                                     ('visual', self.plan['audit_sites'] if condition == 'image_caption' else ())):
                    for i, key in enumerate(keys):
                        a = np.ascontiguousarray([v[key] for v in buf['values']], dtype='<f4')
                        np.save(folder/f'{prefix}_{i:03d}.npy', a, allow_pickle=False)
                meta = dict(self.producer, storage_probe=self.probe, resource_report=self.resource)
                marker = publish_feature_chunk(self.root, self.plan, chunk_id, condition, new_run_id(),
                                                {p.name: p for p in folder.iterdir()}, meta)
            write_seconds = time.perf_counter() - start
            self.write_seconds += write_seconds
            start = time.perf_counter()
            verify_feature_chunk(marker, self.root, self.protocol, self.plan)
            verification_seconds = time.perf_counter() - start
            self.verification_seconds += verification_seconds
            ref = reference(marker, self.root)
            self.accepted[(chunk_id, condition)] = [ref]
            if first:
                torch = getattr(self.adapter, 'torch', None)
                cuda = getattr(torch, 'cuda', None)
                self.first_measurements[condition] = {
                    'artifact': ref, 'input_count': self.calls[condition],
                    'forward_seconds': self.seconds[condition],
                    'audit_seconds': self.audit_by_condition[condition],
                    'write_seconds': write_seconds, 'verification_seconds': verification_seconds,
                    'gpu_peak_allocated_bytes': cuda.max_memory_allocated() if cuda and cuda.is_available() else 0,
                    'gpu_peak_reserved_bytes': cuda.max_memory_reserved() if cuda and cuda.is_available() else 0,
                    'timing_scope': 'production path including first-image public equality; boundary repeats separate'}
            if condition == 'image_only':
                image_marker = marker
            self.snapshot()

    def resources(self):
        if self.resource:
            return
        first = self.plan['work_units'][0]['chunk_id']
        markers = {c: self.root / self.accepted[(first, c)][0]['path'] for c in CONDITIONS}
        check(set(self.first_measurements) == set(CONDITIONS), 'Missing immutable first-chunk measurements')
        measured = self.first_measurements
        check(all(measured[c]['input_count'] == len(self.plan['work_units'][0]['rows'][c])
                  and measured[c]['artifact'] in self.accepted[(first, c)] for c in CONDITIONS),
              'First-chunk measurement coverage mismatch')
        try:
            free = shutil.disk_usage(self.root).free
        except OSError:
            free = None
        remaining = {c: sum(len(u['rows'][c]) for u in self.plan['work_units']
                            if (u['chunk_id'], c) not in self.accepted) for c in CONDITIONS}
        measurements = {'condition_seconds_per_input': {c: measured[c]['forward_seconds']/measured[c]['input_count'] for c in CONDITIONS},
                        'timing_scope': 'First-chunk production includes nine public-path comparison forwards; boundary repeats measured separately',
                        'core_inputs': {c: measured[c]['input_count'] for c in CONDITIONS},
                        'public_comparison_inputs': {'image_only': 1, 'image_caption': 4, 'caption_only': 4},
                        'boundary_repeat_inputs': {c: len({0, measured[c]['input_count']-1}) for c in CONDITIONS},
                        'remaining_inputs': remaining,
                        **{key: sum(measured[c][key] for c in CONDITIONS) for key in
                           ('forward_seconds', 'audit_seconds', 'write_seconds', 'verification_seconds')},
                        **{key: max(measured[c][key] for c in CONDITIONS) for key in
                           ('gpu_peak_allocated_bytes', 'gpu_peak_reserved_bytes')},
                        'free_space_bytes': free}
        marker = publish_resource_report(self.root, self.plan, markers, measurements,
                                        dict(self.producer, run_id=new_run_id()))
        self.resource = reference(marker, self.root)
        self.emit({'event': 'resource_report', 'artifact': self.resource})
        self.snapshot()

    def run(self, adapter_factory):
        try:
            self.snapshot()  # record rejected candidates before any model work
            self.context = {'operation': 'storage readback probe'}
            self.probe = reference(storage_probe(self.root, self.protocol, self.plan['dataset_fingerprint'],
                                                  dict(self.producer, run_id=new_run_id())), self.root)
            needs_work = any((u['chunk_id'], c) not in self.accepted for u in self.plan['work_units'] for c in CONDITIONS)
            if needs_work:
                self.context = {'operation': 'model load/runtime identity'}
                self.adapter = adapter_factory()
                assert_runtime(self.adapter, self.plan)
                torch = getattr(self.adapter, 'torch', None)
                if torch and torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
            for unit in self.plan['work_units']:
                self.chunk(unit)
                if unit['chunk_id'] == self.plan['work_units'][0]['chunk_id']:
                    self.resources()
                # Release each complete partition immediately for downstream roles.
                last = [u for u in self.plan['work_units'] if u['split'] == unit['split']][-1]
                if unit['chunk_id'] == last['chunk_id']:
                    for c in CONDITIONS:
                        markers = [self.root/r['path'] for u in self.plan['work_units'] if u['split'] == unit['split']
                                   for r in self.accepted[(u['chunk_id'], c)]]
                        marker = publish_release_index(self.root, self.plan, c, unit['split'], markers, new_run_id())
                        self.releases.append(reference(marker, self.root))
                        self.emit({'event': 'feature_release', 'condition': c, 'split': unit['split'],
                                   'artifact': self.releases[-1]})
                    self.snapshot()
            return {'status': 'passed', 'model': self.plan['model'], 'resume_index': str(self.latest_progress),
                    'resource_report': self.resource, 'feature_releases': self.releases,
                    'stage2_complete': False, 'scientific_results': False}
        except Exception as exc:
            producer = dict(self.producer, run_id=new_run_id())
            meta = metadata('stage2_extraction_diagnostic', self.protocol, self.plan['dataset_fingerprint'],
                            producer, self.plan['model_config']['config_fingerprint'])
            meta['status'] = 'blocked' if isinstance(exc, (ContractError, OSError)) else 'failed'
            meta['upstream_artifacts'] = [plain(self.plan['manifest_reference'])]
            body = {'status': meta['status'], 'context': self.context, 'error_type': type(exc).__name__,
                    'plan': plain(self.plan['manifest_reference']),
                    'error': safe_error(exc), 'resume_index': str(self.latest_progress) if self.latest_progress else None,
                    'completed_chunks_preserved': True, 'stage2_complete': False}
            marker = publish_json(self.root, 'results/stage2-validation/' + producer['run_id'], meta, 'report.json', body)
            self.emit(dict(body, diagnostic=str(marker)))
            raise


def preflight(root, manifest, model, protocol, resume_indexes=()):
    settings = resolve_worker_settings(manifest, root, protocol, model)
    plan = settings['plan']
    check(plan['kind'] == 'stage2_model_plan', 'Extraction requires a model plan')
    bundle = load_dataset(Path(settings['dataset_manifest']), root, protocol)
    inputs = bind_inputs(plan, bundle.records)
    load_resume(resume_indexes, root, protocol, plan)
    return settings, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=['qwen', 'gemma'])
    parser.add_argument('--phase1-root', required=True)
    parser.add_argument('--plan-manifest', required=True)
    parser.add_argument('--resume-index', action='append', default=[])
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--storage-scope', choices=['persistent-drive', 'local-development'], default='persistent-drive')
    args = parser.parse_args()
    root = phase1_root(args.phase1_root)
    check(root.is_dir(), 'PHASE1_ROOT must exist and be mounted before launch')
    protocol = read_json(REPO/'configs/protocol.json')
    producer = current_producer(scope=args.storage_scope)
    settings, inputs = preflight(root, args.plan_manifest, args.model, protocol, args.resume_index)
    plan = settings['plan']
    if args.preflight:
        print(json.dumps({'status': 'passed', 'model': args.model,
              'resolved': {k: plain(v) for k, v in settings.items() if k != 'plan'},
              'plan_fingerprint': plan['plan_fingerprint'], 'counts': plain(plan['counts']),
              'required_sites': plain(plan['required_sites']), 'code_commit': producer['code_commit'],
              'weights_loaded': False}))
        return 0
    def factory():
        adapter = create_adapter(args.model, plain(plan['model_config']), root)
        from importlib.metadata import version
        for name in ('torch', 'torchvision', 'transformers', 'tokenizers', 'safetensors', 'huggingface-hub', 'Pillow'):
            producer['software_versions'][name] = version(name)
        return adapter
    worker = Extractor(root, protocol, plan, inputs, producer, args.resume_index)
    result = worker.run(factory)
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(json.dumps({'status': 'blocked' if isinstance(exc, (ContractError, OSError)) else 'failed',
                          'error_type': type(exc).__name__, 'error': safe_error(exc)}))
        sys.exit(2 if isinstance(exc, (ContractError, OSError)) else 1)
