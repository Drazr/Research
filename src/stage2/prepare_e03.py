"""CPU-only E03 worker using the frozen Stage 2 public artifact interfaces."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from src.common.contracts import ContractError, fingerprint
from src.common.data import load_dataset
from src.common.storage import new_run_id, phase1_root
from src.stage2.interfaces import (
    current_producer, load_text_family, preprocessing_versions, publish_text_family,
    publish_text_preprocessor, publish_text_release, resolve_worker_settings,
    storage_probe, text_rows,
)
from src.stage2.schema import (REPO, ROLES, SPLITS, artifact, check, dereference,
    plain, publish_json, read_json, reference)
from .e03_text import (caption_texts, equal_matrices, fit_identities, fit_preprocessor,
    read_preprocessor, transform, write_matrix, write_preprocessor)

ANALYZER = 'src/stage2/e03_text.py'
CHECKPOINT_SCHEMA = 'stage2.e03-checkpoint.v1'


def emit(event, **values):
    print(json.dumps(dict(event=event, **plain(values)), ensure_ascii=False), flush=True)


def resolve(root, manifest):
    protocol = read_json(REPO/'configs/protocol.json')
    settings = resolve_worker_settings(manifest, root, protocol)
    plan = settings['plan']
    check(plan['kind'] == 'stage2_data_plan', 'E03 requires a shared data plan, without MODEL')
    versions = preprocessing_versions(ANALYZER)
    check(versions['numpy'] == '2.1.3' and versions['scikit-learn'] == '1.6.1',
          'Install requirements-stage2-cpu.txt before E03 production')
    return protocol, settings, plan, versions


def preflight(root, manifest):
    _, settings, plan, versions = resolve(root, manifest)
    current_producer()  # Clean checkout and actual HEAD are required even at preflight.
    emit('preflight', phase1_root=str(root), plan=plain(plan['manifest_reference']),
         dataset_manifest=settings['dataset_manifest'], versions=versions,
         recipes=plain(plan['recipes']), counts=plain(plan['counts']), model_independent=True)


def load_fit(ref, root, protocol, plan, recipe, versions):
    marker, _ = dereference(ref, root, protocol, 'stage2_text_preprocessor')
    body = read_json(marker.parent/'fit.json')
    check(body['plan'] == plain(plan['manifest_reference']) and body['recipe'] == recipe
          and body['versions'] == versions and body['plan_fingerprint'] == plan['plan_fingerprint'],
          'Resume preprocessor plan/recipe/runtime mismatch')
    spec, idf = read_preprocessor(marker.parent, recipe, text_rows(plan, recipe, 'FIT'))
    return marker, body, spec, idf


def verify_family(root, manifest, split):
    """Fresh-process full reconstruction and bitwise comparison, without fitting."""
    protocol = read_json(REPO/'configs/protocol.json')
    marker, _ = artifact(manifest, root, protocol, 'stage2_text_family')
    body = read_json(marker.parent/'family.json')
    pp, _ = dereference(body['plan'], root, protocol, 'stage2_data_plan')
    protocol, settings, plan, versions = resolve(root, pp)
    recipe = body['recipe']
    _, _, spec, idf = load_fit(body['preprocessor'], root, protocol, plan, recipe, versions)
    role = next(r for r, s in ROLES.items() if s == split)
    reader = load_text_family(marker, root, protocol, role,
        {k: body[k] for k in ('plan_fingerprint', 'preprocessing_fingerprint')})
    rows, values = reader.read_matrix(split)
    bundle = load_dataset(Path(settings['dataset_manifest']), root, protocol)
    texts = caption_texts(bundle.records, rows, split)
    check(equal_matrices(transform(recipe, spec, idf, texts), values),
          'Independent reload differs from reconstructed E03 transform')
    emit('reload_passed', family=reference(marker, root), split=split,
         rows=len(rows), shape=list(values.shape))


def read_checkpoint(marker, root, protocol, plan, versions):
    p, _ = artifact(marker, root, protocol, 'stage2_e03_checkpoint')
    body = read_json(p.parent/'inventory.json')
    check(body['schema'] == CHECKPOINT_SCHEMA and body['plan'] == plain(plan['manifest_reference'])
          and body['versions'] == versions, 'Resume checkpoint identity/runtime mismatch')
    entries = body['families']
    recipes = {r['recipe_fingerprint']: plain(r) for r in plan['recipes']}
    check(set(entries) <= set(recipes), 'Resume has unknown/extra E03 recipe')
    for rid, entry in entries.items():
        recipe = recipes[rid]
        _, fit, _, _ = load_fit(entry['preprocessor'], root, protocol, plan, recipe, versions)
        check(entry['fit_identities'] == fit_identities(text_rows(plan, recipe, 'FIT')),
              'Resume FIT identity hashes mismatch')
        check(set(entry['splits']) <= set(SPLITS), 'CONFIRM/unknown resume split')
        if entry['splits']:
            fp, _ = dereference(entry['family'], root, protocol, 'stage2_text_family')
            family = read_json(fp.parent/'family.json')
            check(family['preprocessor'] == entry['preprocessor'] and family['splits'] == entry['splits'],
                  'Resume family/split reference mismatch')
            for split in entry['splits']:
                role = next(r for r, s in ROLES.items() if s == split)
                reader = load_text_family(fp, root, protocol, role,
                    {'plan_fingerprint': plan['plan_fingerprint'],
                     'preprocessing_fingerprint': fit['preprocessing_fingerprint']})
                reader.read_matrix(split)
    return entries


def checkpoint(root, protocol, plan, versions, entries, producer):
    body = {'schema': CHECKPOINT_SCHEMA, 'plan': plain(plan['manifest_reference']),
            'versions': versions, 'families': entries,
            'complete': len(entries) == 13 and all(set(e['splits']) == set(SPLITS) for e in entries.values())}
    from src.stage2.schema import metadata
    meta = metadata('stage2_e03_checkpoint', protocol, plan['dataset_fingerprint'],
                    dict(producer, run_id=new_run_id()))
    meta['upstream_artifacts'] = [plain(plan['manifest_reference'])] + [e.get('family', e['preprocessor']) for e in entries.values()]
    marker = publish_json(root, 'cache/shared/stage2-e03-checkpoints/'+meta['run_id'],
                          meta, 'inventory.json', body)
    emit('checkpoint', resume_index=reference(marker, root), complete=body['complete'],
         families={rid: e.get('family') for rid, e in entries.items()})
    return marker


def run(root, manifest, resume_index=None, splits=SPLITS, scope='persistent-drive'):
    protocol, settings, plan, versions = resolve(root, manifest)
    check(tuple(splits) == SPLITS[:len(splits)] and splits, 'Request a nonempty ordered eligible split prefix')
    producer = current_producer(scope=scope)
    # Record the remaining CPU profile/dependencies as producer provenance too;
    # the frozen preprocessing identity has the coordinator's exact six fields.
    producer['software_versions'].update({name: version(name) for name in ('Pillow', 'joblib', 'threadpoolctl')})
    entries = read_checkpoint(resume_index, root, protocol, plan, versions) if resume_index else {}
    emit('settings', code_commit=producer['code_commit'], phase1_root=str(root),
         plan=plain(plan['manifest_reference']), versions=versions, splits=list(splits),
         resume_index=str(resume_index) if resume_index else None, storage_scope=scope)
    bundle = load_dataset(Path(settings['dataset_manifest']), root, protocol)
    probe = storage_probe(root, protocol, plan['dataset_fingerprint'], dict(producer, run_id=new_run_id()))
    producer.update(analyzer_file=ANALYZER, preprocessing_versions=versions, storage_probe=reference(probe, root))
    recipes = [plain(r) for r in plan['recipes']]
    last = None
    # Fit once per family, publish independently, and checkpoint immediately.
    for recipe in recipes:
        rid = recipe['recipe_fingerprint']
        if rid in entries:
            continue
        rows = text_rows(plan, recipe, 'FIT')
        texts = caption_texts(bundle.records, rows, 'FIT')
        spec, idf = fit_preprocessor(recipe, rows, texts)
        with tempfile.TemporaryDirectory() as temporary:
            paths = write_preprocessor(temporary, spec, idf)
            loaded_spec, loaded_idf = read_preprocessor(temporary, recipe, rows)
            check(equal_matrices(transform(recipe, spec, idf, texts),
                                 transform(recipe, loaded_spec, loaded_idf, texts)),
                  'Fitted serialization changed transform')
            fitted = publish_text_preprocessor(root, plan, rid, paths, dict(producer, run_id=new_run_id()))
        entries[rid] = {'preprocessor': reference(fitted, root), 'fit_identities': fit_identities(rows), 'splits': {}}
        last = checkpoint(root, protocol, plan, versions, entries, producer)
    # Complete every family's FIT, then DEV-S, before touching DEV-C/SCREEN.
    for split in splits:
        for recipe in recipes:
            rid = recipe['recipe_fingerprint']
            entry = entries[rid]
            if split in entry['splits']:
                continue
            fitted, _, spec, idf = load_fit(entry['preprocessor'], root, protocol, plan, recipe, versions)
            rows = text_rows(plan, recipe, split)
            values = transform(recipe, spec, idf, caption_texts(bundle.records, rows, split))
            with tempfile.TemporaryDirectory() as temporary:
                paths, zero = write_matrix(temporary, rows, values)
                released = publish_text_release(root, fitted, split, paths,
                    dict(producer, run_id=new_run_id(), zero_rows=zero))
            candidate = dict(entry['splits'], **{split: reference(released, root)})
            family = publish_text_family(root, fitted,
                {s: root/ref['path'] for s, ref in candidate.items()}, new_run_id(), producer)
            # No success/checkpoint until another Python process reloads and
            # reconstructs all values with the serialized FIT preprocessor.
            subprocess.run([sys.executable, '-m', 'src.stage2.prepare_e03', 'verify-release',
                '--phase1-root', str(root), '--manifest', str(family), '--split', split], cwd=REPO, check=True)
            entry.update(splits=candidate, family=reference(family, root))
            emit('release_passed', recipe=rid, family=reference(family, root), split=split,
                 shape=list(values.shape), zero_rows=zero)
            last = checkpoint(root, protocol, plan, versions, entries, producer)
    if last is None:
        last = checkpoint(root, protocol, plan, versions, entries, producer)
    emit('finished', resume_index=reference(last, root),
         complete=all(set(e['splits']) == set(SPLITS) for e in entries.values()),
         scope='E03 preprocessing only; no Stage 2 suite acceptance')
    return last


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('preflight', 'run', 'verify-release'))
    parser.add_argument('--phase1-root', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--resume-index')
    parser.add_argument('--through-split', choices=SPLITS, default='SCREEN')
    parser.add_argument('--split', choices=SPLITS)
    parser.add_argument('--storage-scope', choices=('local-development', 'persistent-drive'), default='persistent-drive')
    args = parser.parse_args()
    try:
        root = phase1_root(args.phase1_root)
        if args.command == 'preflight':
            preflight(root, args.manifest)
        elif args.command == 'verify-release':
            check(args.split in SPLITS, 'An explicit eligible split is required')
            verify_family(root, args.manifest, args.split)
        else:
            run(root, args.manifest, args.resume_index, SPLITS[:SPLITS.index(args.through_split)+1], args.storage_scope)
    except (ContractError, OSError, ValueError, subprocess.CalledProcessError) as exc:
        emit('blocked', error=str(exc), scope='No incomplete release is accepted')
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
