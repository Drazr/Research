"""Coordinator's real-data smoke runner. Blocked dependencies never count as a pass."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.common.contracts import (ContractError, FeatureBatch, FeatureSite, INSPECTION_CHECKS,
                                  canonical_bytes, decoder_depths, fingerprint, resolve_model)
from src.common.storage import new_run_id, phase1_root, publish

def execute(args):
    protocol = json.loads(Path(args.protocol).read_text(encoding='utf-8'))
    registry = json.loads(Path(args.models_config).read_text(encoding='utf-8'))
    root = phase1_root(args.phase1_root)
    run_id = args.run_id or new_run_id()
    if not re.fullmatch('[A-Za-z0-9_-]+', run_id):
        raise ContractError('run-id must be a single portable path component')
    report = {'kind': 'stage1_smoke', 'status': 'blocked', 'checks': {}, 'blockers': [],
              'full_G0_shuffled_label_controls': 'pending-stage3', 'scientific_results': False,
              'storage_scope': args.storage_scope, 'software_versions': {'python': platform.python_version()}}
    metadata = dict(kind='stage1_smoke', status='blocked', run_id=run_id,
                    code_commit=args.code_commit, protocol_fingerprint=fingerprint(protocol),
                    dataset_fingerprint=None, model_fingerprint=None, seeds={'smoke': 0},
                    software_versions=report['software_versions'], upstream_artifacts=[])
    raw_config = registry.get('models', {}).get(args.model, {})
    namespace_model = fingerprint(raw_config)
    try:
        config = resolve_model(registry, args.model)
        metadata['model_fingerprint'] = namespace_model = config['config_fingerprint']
        if not args.code_commit or not re.fullmatch('[0-9a-f]{40}', args.code_commit):
            raise ContractError('A real code commit must be supplied through --code-commit')
        try:
            actual_commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True,stderr=subprocess.DEVNULL).strip()
            dirty = bool(subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True,stderr=subprocess.DEVNULL).strip())
        except (OSError,subprocess.CalledProcessError) as exc:
            raise ContractError('Run real smoke checks from a clean Git checkout') from exc
        if dirty or actual_commit != args.code_commit:
            raise ContractError('Dirty checkout or --code-commit differs from the actual checkout')
        if not args.dataset_manifest:
            raise ContractError('Provide the completed dataset artifact using --dataset-manifest')
        data_module = importlib.import_module('src.common.data')
        model_module = importlib.import_module('src.models.registry')
        bundle = data_module.load_dataset(Path(args.dataset_manifest), root, protocol)
        bundle.validate(protocol)
        metadata['dataset_fingerprint'] = bundle.dataset_fingerprint
        metadata['upstream_artifacts'] = [str(Path(args.dataset_manifest).resolve())]
        fit_ids = sorted({r.image_id for r in bundle.records if r.split == 'FIT'},
                         key=lambda x: hashlib.sha256(('stage1-smoke-v1|' + x).encode()).digest())
        chosen = fit_ids[:args.max_images]
        if len(chosen) < 4:
            raise ContractError('At least four independent FIT images required for toy checks')
        rank = {iid: n for n, iid in enumerate(chosen)}
        inputs = sorted((r for r in bundle.records if r.image_id in rank),
                        key=lambda r: (rank[r.image_id], r.language))
        for r in inputs:
            r.require_fit()
        adapter = model_module.create_adapter(args.model, config, root)
        info = dict(adapter.describe())
        if info.get('model_fingerprint') != namespace_model or not info.get('hook_convention'):
            raise ContractError('Adapter identity/hook convention missing or mismatched')
        sites = [FeatureSite('decoder', d, 'last_caption') for d in decoder_depths(info['decoder_blocks'])]
        if not {s.key for s in sites}.issubset(set(info.get('supported_sites', []))):
            raise ContractError('Default decoder sites are unsupported; report them explicitly')
        inspection = dict(adapter.inspect(inputs))
        if any(inspection.get('checks', {}).get(k) is not True for k in INSPECTION_CHECKS):
            raise ContractError('Model inspection failed or lacks required evidence checks')
        if len(inspection.get('records', [])) != len(inputs):
            raise ContractError('Per-input inspection evidence required')
        if [tuple(r.get('row_key', ())) for r in inspection['records']] != [r.row_key for r in inputs]:
            raise ContractError('Inspection evidence row identity/order mismatch')
        rows = {s.key: [] for s in sites}
        for start in range(0, len(inputs), args.batch_size):
            batch_inputs = inputs[start:start + args.batch_size]
            result = adapter.extract(batch_inputs, sites)
            result.validate(batch_inputs, sites)
            for s in sites:
                rows[s.key].extend(result.vectors[s.key])
        FeatureBatch([r.row_key for r in inputs], rows).validate(inputs, sites)
        repeated = adapter.extract(inputs[:1], sites)
        repeated.validate(inputs[:1], sites)
        atol, rtol = info.get('repeat_atol'), info.get('repeat_rtol')
        if not isinstance(atol, (float, int)) or not isinstance(rtol, (float, int)) or not 0 <= atol < 1 or not 0 <= rtol < 1:
            raise ContractError('Explicit repeat tolerances in [0,1) required')
        for s in sites:
            if len(rows[s.key][0]) != len(repeated.vectors[s.key][0]):
                raise ContractError('Repeated extraction changed feature dimension')
            if any(abs(float(a)-float(b)) > atol + rtol * abs(float(a))
                   for a, b in zip(rows[s.key][0], repeated.vectors[s.key][0])):
                raise ContractError('Repeated extraction exceeds registered tolerances')
        # Optional ML packages are imported only after real upstream inputs validate.
        import numpy as np
        import sklearn
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        train_ids = set(chosen[:max(2, len(chosen)*3//4)])
        train = np.array([r.image_id in train_ids for r in inputs])
        mid = FeatureSite('decoder', (info['decoder_blocks'] + 1)//2, 'last_caption').key
        x = np.asarray(rows[mid], dtype=float)
        diagnostics = []
        for k, category in enumerate(bundle.category_order):
            known = np.array([bundle.targets[r.image_id].values[k] is not None for r in inputs])
            y = np.array([bundle.targets[r.image_id].values[k] or 0 for r in inputs])
            tr, va = train & known, ~train & known
            if len(set(y[tr])) < 2 or not va.any():
                continue
            scaler = StandardScaler().fit(x[tr])
            head = LogisticRegression(C=1.0, max_iter=300, class_weight=None).fit(scaler.transform(x[tr]), y[tr])
            probabilities = head.predict_proba(scaler.transform(x[va]))
            if not np.isfinite(probabilities).all() or (head.n_iter_ >= 300).any():
                raise ContractError('Toy optimizer failed convergence/finite-score check')
            diagnostics.append({'category': category, 'train_rows': int(tr.sum()), 'validation_rows': int(va.sum())})
            break
        if not diagnostics:
            raise ContractError('Toy optimizer unestimable on this FIT smoke selection; not a pass')
        # Verify vector/mask permutation wiring at image level, shared across languages.
        permutation_checks = []
        for seed in (0, 1, 2):
            ids = sorted(train_ids)
            perm = np.random.default_rng(seed).permutation(len(ids))
            mapped = {iid: bundle.targets[ids[int(j)]].values for iid, j in zip(ids, perm)}
            vectors = [mapped[r.image_id] for r in inputs if r.image_id in train_ids]
            if len(vectors) != 4*len(ids):
                raise ContractError('Permutation view grouping failed')
            permutation_checks.append({'seed': seed, 'image_vectors': len(ids), 'mask_carried_with_values': True})
        report['software_versions'].update(numpy=np.__version__, sklearn=sklearn.__version__)
        report.update(status='passed', checks={'data_contract': True, 'adapter_inspection': True,
                      'finite_features': True, 'repeat_extraction': True, 'toy_optimizer': True,
                      'permutation_plumbing': True}, inspection=inspection, adapter=info,
                      toy_optimizer=diagnostics, permutation_plumbing=permutation_checks,
                      note='Pooled-view FIT-only optimizer plumbing; no scientific transfer/calibration result.')
    except (ContractError, ModuleNotFoundError, FileNotFoundError) as exc:
        report['blockers'].append(str(exc))
    except Exception as exc:
        report['status'] = 'failed'
        report['blockers'].append(type(exc).__name__ + ': ' + str(exc))
    metadata['status'] = report['status']
    runtime_torch = sys.modules.get('torch')
    if runtime_torch is not None and runtime_torch.cuda.is_available():
        report['observed_gpu_memory'] = {
            'device':runtime_torch.cuda.get_device_name(0),
            'peak_allocated_bytes':runtime_torch.cuda.max_memory_allocated(0),
            'peak_reserved_bytes':runtime_torch.cuda.max_memory_reserved(0),
            'scope':'this smoke process, including weight loading; no fit guarantee for later workloads'}
    metadata['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    location = publish(root, f'smoke/{namespace_model}/{run_id}', metadata,
                       {'report.json': canonical_bytes(report) + b'\n'})
    print(json.dumps({'status': report['status'], 'artifact': str(location), 'blockers': report['blockers']}))
    return 0 if report['status'] == 'passed' else 2 if report['status'] == 'blocked' else 1

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, choices=['qwen', 'gemma'])
    p.add_argument('--phase1-root')
    p.add_argument('--dataset-manifest')
    p.add_argument('--models-config', default=str(REPO/'configs/models.json'))
    p.add_argument('--protocol', default=str(REPO/'configs/protocol.json'))
    p.add_argument('--code-commit')
    p.add_argument('--run-id')
    p.add_argument('--max-images', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--storage-scope', choices=['local-development', 'persistent-drive'], required=True)
    a = p.parse_args()
    if a.max_images < 4 or a.batch_size < 1:
        p.error('max-images >=4 and batch-size >=1 are required')
    return execute(a)

if __name__ == '__main__':
    sys.exit(main())
