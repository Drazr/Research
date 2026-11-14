"""FIT-only S1-C adapter diagnostics. Blocked configuration is never a pass."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.common.contracts import (ContractError, ExtractionRequest, FeatureSite, INSPECTION_CHECKS,
                                  canonical_bytes, fingerprint, resolve_model)
from src.common.storage import new_run_id, phase1_root, publish


def execute(args):
    root = phase1_root(args.phase1_root)
    registry = json.loads(Path(args.models_config).read_text(encoding='utf-8'))
    protocol = json.loads(Path(args.protocol).read_text(encoding='utf-8'))
    raw = registry['models'][args.model]
    run_id = args.run_id or new_run_id()
    if not re.fullmatch('[A-Za-z0-9_-]+', run_id):
        raise ContractError('Unsafe run ID')
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    report = {'status': 'blocked', 'model': args.model, 'blockers': [],
              'scientific_results': False, 'gpu_checks_passed': False,
              'storage_scope': args.storage_scope, 'code_dirty': dirty,
              'unsupported_or_unvalidated_sites': ['post_caption', 'encoder', 'projector',
                  'actual_checkpoint_token_spans', 'actual_checkpoint_attention', 'actual_checkpoint_residual_hooks'],
              'unresolved_fields': [k for k, v in raw.items() if (v is None and k not in ('quantization', 'config_fingerprint')) or (k == 'software_versions' and not v)]}
    identity = fingerprint(raw)
    metadata = dict(kind='stage1_model_validation', status='blocked', run_id=run_id,
                    code_commit=commit, protocol_fingerprint=fingerprint(protocol),
                    dataset_fingerprint=None, model_fingerprint=None, seeds={'smoke': 0},
                    software_versions={'python': platform.python_version()}, upstream_artifacts=[])
    try:
        config = resolve_model(registry, args.model)
        if dirty is not False:
            raise ContractError('Run real validation from a clean Git checkout with a known commit')
        identity = metadata['model_fingerprint'] = config['config_fingerprint']
        report['resolved_config'] = config
        if not args.dataset_manifest:
            raise ContractError('Completed validated dataset artifact required: --dataset-manifest')
        from src.common.data import load_dataset
        bundle = load_dataset(Path(args.dataset_manifest), root, protocol)
        bundle.validate(protocol)
        metadata['dataset_fingerprint'] = bundle.dataset_fingerprint
        metadata['upstream_artifacts'] = [str(Path(args.dataset_manifest).resolve())]
        fit_ids = sorted({r.image_id for r in bundle.records if r.split == 'FIT'},
                         key=lambda x: hashlib.sha256(('stage1-model-smoke-v1|' + x).encode()).digest())[:args.max_images]
        inputs = sorted([r for r in bundle.records if r.image_id in fit_ids], key=lambda r: r.row_key)
        if not inputs:
            raise ContractError('No FIT inputs available')
        for row in inputs:
            row.require_fit()
        from src.models.registry import create_adapter
        adapter = create_adapter(args.model, config, root)
        info = dict(adapter.describe())
        report['adapter'] = info
        report['unsupported_or_unvalidated_sites'] = info['unsupported_sites']
        inspection = adapter.inspect(inputs)
        report['inspection'] = inspection
        if any(inspection.get('checks', {}).get(k) is not True for k in INSPECTION_CHECKS):
            raise ContractError('Required inspection check failed')
        if [tuple(r['row_key']) for r in inspection['records']] != [r.row_key for r in inputs]:
            raise ContractError('Inspection row identity mismatch')
        sites = []
        for key in info['supported_sites']:
            stage, depth, pooling = key.split(':')
            sites.append(FeatureSite(stage, None if depth == 'None' else int(depth), pooling))
        if not sites:
            raise ContractError('No supported extraction sites')
        first = adapter.extract(inputs, sites)
        second = adapter.extract(inputs, sites)
        first.validate(inputs, sites)
        second.validate(inputs, sites)
        widths = {len(first.vectors[s.key][0]) for s in sites if s.stage == 'decoder'}
        if len(widths) != 1:
            raise ContractError('Inconsistent decoder residual-stream dimensions')
        atol, rtol = info['repeat_atol'], info['repeat_rtol']
        if not all(isinstance(v, (int, float)) and 0 <= v < 1 for v in (atol, rtol)):
            raise ContractError('Explicit repeat tolerances in [0,1) required')
        for key in first.vectors:
            for a, b in zip(first.vectors[key], second.vectors[key]):
                if len(a) != len(b) or any(abs(x-y) > atol + rtol*abs(x) for x, y in zip(a, b)):
                    raise ContractError('Repeat extraction mismatch: ' + key)
        decoder_sites = [s for s in sites if s.stage == 'decoder']
        visual_sites = [s for s in sites if s.stage in ('encoder', 'projector')]
        expected_decoder = {FeatureSite('decoder', d, p).key for d in info['decoder_depths']
                            for p in ('last_caption', 'caption_mean')}
        if not visual_sites or {s.key for s in decoder_sites} != expected_decoder:
            raise ContractError('Scheduled E01/E02/E04 minimum site inventory unavailable')
        text_request = ExtractionRequest(inputs, decoder_sites, condition='caption_only')
        text_first = adapter.extract_request(text_request)
        text_second = adapter.extract_request(text_request)
        for batch in (text_first, text_second):
            batch.validate_request(text_request)
        image_inputs = list({r.image_id: r for r in inputs}.values())
        image_request = ExtractionRequest(image_inputs, visual_sites, condition='image_only')
        image_first = adapter.extract_request(image_request)
        image_second = adapter.extract_request(image_request)
        for batch in (image_first, image_second):
            batch.validate_request(image_request)
        for a_batch, b_batch in ((text_first, text_second), (image_first, image_second)):
            for key in a_batch.vectors:
                for a, b in zip(a_batch.vectors[key], b_batch.vectors[key]):
                    if len(a) != len(b) or any(abs(x-y) > atol + rtol*abs(x) for x, y in zip(a, b)):
                        raise ContractError('Condition repeat extraction mismatch: ' + key)
        # Reuse is permitted only after actual visual invariance across contexts.
        image_index = {r.image_id: i for i, r in enumerate(image_inputs)}
        for i, row in enumerate(inputs):
            for site in visual_sites:
                a = first.vectors[site.key][i]
                b = image_first.vectors[site.key][image_index[row.image_id]]
                if len(a) != len(b) or any(abs(x-y) > atol + rtol*abs(x) for x, y in zip(a, b)):
                    raise ContractError('Visual site depends on caption context; audit before reuse: ' + site.key)
        report['execution_identity'] = adapter.describe_execution()
        report['conditions_validated'] = {
            'image_caption': [s.key for s in sites],
            'caption_only': [s.key for s in decoder_sites],
            'image_only': [s.key for s in visual_sites]}
        report['image_only_rows'] = len(image_inputs)
        report['visual_context_invariance'] = True
        report.update(status='passed', repeat_extraction=True,
                      dimensions={k: len(v[0]) for k, v in first.vectors.items()},
                      gpu_checks_passed=str(config['device']).startswith('cuda'))
        metadata['software_versions'].update(config['software_versions'])
    except (ContractError, ImportError, FileNotFoundError) as exc:
        report['blockers'].append(str(exc))
    except Exception as exc:
        report.update(status='failed')
        report['blockers'].append(type(exc).__name__ + ': ' + str(exc))
    metadata['status'] = report['status']
    runtime_torch = sys.modules.get('torch')
    if runtime_torch is not None and runtime_torch.cuda.is_available():
        report['observed_gpu_memory'] = {
            'device':runtime_torch.cuda.get_device_name(0),
            'peak_allocated_bytes':runtime_torch.cuda.max_memory_allocated(0),
            'peak_reserved_bytes':runtime_torch.cuda.max_memory_reserved(0),
            'scope':'this validation process, including weight loading; no fit guarantee for later workloads'}
    metadata['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metadata['source_files_sha256'] = {p.relative_to(REPO).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ('src/common','src/models') for p in sorted((REPO/folder).glob('*.py'))}
    location = publish(root, f'smoke/{identity}/{run_id}', metadata,
                       {'model_validation.json': canonical_bytes(report) + b'\n'})
    print(json.dumps({'status': report['status'], 'artifact': str(location), 'blockers': report['blockers']}))
    return {'passed': 0, 'blocked': 2, 'failed': 1}[report['status']]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['qwen', 'gemma'], required=True)
    p.add_argument('--models-config', default=str(REPO / 'configs/models.json'))
    p.add_argument('--protocol', default=str(REPO / 'configs/protocol.json'))
    p.add_argument('--phase1-root')
    p.add_argument('--dataset-manifest')
    p.add_argument('--storage-scope', choices=['local-development', 'persistent-drive'], required=True)
    p.add_argument('--run-id')
    p.add_argument('--max-images', type=int, default=2)
    args = p.parse_args()
    if args.max_images < 1:
        p.error('max-images must be positive')
    return execute(args)


if __name__ == '__main__':
    sys.exit(main())
