"""Immutable shared data and independent per-model work inventories."""
from pathlib import Path
from src.common.contracts import LANGUAGES, ModelInput, fingerprint
from src.common.cache_keys import input_view_fingerprint, feature_cache_key
from .schema import (SPLITS, CONDITIONS, PLAN_SCHEMA, artifact, check, dereference,
                     frozen, metadata, plain, publish_json, read_json, reference, specification, contract, VerifiedPlan)
from .evidence import bridge_hash
from .readiness import verify_data, verify_readiness


def text_recipes():
    common = {'schema': 'stage2.text-recipe.v1', 'draw': 0,
              'normalization': 'none', 'fit_split': 'FIT', 'storage_dtype': '<f4'}
    recipes = []
    for family in ('word_tfidf', 'char_tfidf', 'length', 'pooled_char_tfidf'):
        for language in (LANGUAGES if family != 'pooled_char_tfidf' else (None,)):
            recipe = dict(common, family=family, language=language)
            if family == 'length':
                recipe.update(columns=['log1p_unicode_codepoints', 'log1p_str_split_tokens'],
                              arithmetic='float64 then float32', fitted_parameters=False)
            else:
                word = family == 'word_tfidf'
                recipe.update(analyzer='str.split case-sensitive token tuples' if word else 'literal Unicode codepoint substrings',
                    ngrams=[1, 2] if word else [3, 4, 5], min_df=2,
                    max_features=20000 if word else 50000,
                    cap_order='descending FIT raw term count; Unicode lexical ties',
                    column_order='Unicode lexicographic tuples/strings',
                    tf='1+ln(count)', idf='ln((1+N_FIT)/(1+df))+1', norm='row L2',
                    arithmetic='float64 then float32', idf_dtype='<f8', format='csr',
                    indices_dtype='<i4', indptr_dtype='<i4', zero_rows='preserve',
                    empty_vocabulary='blocked', explicit_zeros=False)
            recipe['recipe_fingerprint'] = fingerprint(recipe)
            recipes.append(recipe)
    return recipes


def row_identity(row, condition):
    image_only = condition == 'image_only'
    return {'image_id': row.image_id, 'split': row.split, 'condition': condition,
            'language': None if image_only else row.language,
            'caption_id': None if image_only else row.caption_id,
            'caption_draw_key': None if image_only else 0,
            'caption_text_sha256': None if image_only else __import__('hashlib').sha256(row.caption_text.encode('utf-8')).hexdigest(),
            'source_row_key': list(row.row_key),
            'input_view_fingerprint': input_view_fingerprint(row, condition)}


def compile_inventory(bundle):
    """Pure compiler for an already accepted DatasetBundle, no input IO/tokenization."""
    units, rows = [], {c: [] for c in CONDITIONS}
    for split in SPLITS:
        selected = sorted((r for r in bundle.records if r.split == split),
                          key=lambda r: (r.image_id, LANGUAGES.index(r.language)))
        ids = sorted({r.image_id for r in selected})
        by_image = {i: [] for i in ids}
        for row in selected:
            by_image[row.image_id].append(row)
        for offset in range(0, len(ids), 32):
            images = ids[offset:offset+32]
            chunk_id = split + '_' + str(offset//32).zfill(4)
            unit = {'chunk_id': chunk_id, 'split': split, 'image_ids': images, 'rows': {}}
            for condition in CONDITIONS:
                inputs = [r for i in images for r in by_image[i]
                          if condition != 'image_only' or r.language == 'en']
                table = [row_identity(r, condition) for r in inputs]
                unit['rows'][condition] = table
                rows[condition].extend(table)
            units.append(unit)
    return {'work_units': units, 'rows': rows,
            'counts': {c: len(r) for c, r in rows.items()}, 'chunk_size_images': 32}


def publish_data_plan(root, protocol, dataset_manifest, data_validation, producer):
    from .readiness import discover_data_validation
    candidates=[]
    if not data_validation:
        data_validation,candidates=discover_data_validation(dataset_manifest,root,protocol)
    bundle, refs = verify_data(dataset_manifest, data_validation, root, protocol)
    inventory = compile_inventory(bundle)
    check(inventory['counts'] == {'image_caption': 11520, 'caption_only': 11520, 'image_only': 2880}
          and len(inventory['work_units']) == 92, 'Frozen population inventory mismatch')
    path, meta = artifact(dataset_manifest, root, protocol, 'stage1_dataset')
    dataset = read_json(path.parent / 'dataset.json')
    body = dict(inventory, schema=PLAN_SCHEMA, kind='stage2_data_plan', specification=specification(),
                dataset_fingerprint=bundle.dataset_fingerprint, protocol_fingerprint=fingerprint(protocol),
                references={'dataset': refs[0], 'data_validation': refs[1]}, recipes=text_recipes(),
                data_validation_candidates=candidates,
                source_identity={k: dataset[k] for k in ('records_sha256', 'targets_sha256', 'split_manifest_sha256', 'category_order')},
                target_reference={'artifact': refs[0], 'file': 'targets.jsonl', 'primary_categories': protocol['primary_categories']},
                confirm_processing=False)
    body['plan_fingerprint'] = fingerprint(body)
    meta = metadata('stage2_data_plan', protocol, bundle.dataset_fingerprint, producer)
    meta['upstream_artifacts'] = refs
    return publish_json(root, 'cache/shared/stage2-plans/' + meta['run_id'], meta, 'plan.json', body)


def publish_model_plan(root, protocol, data_plan_manifest, receipt_manifest, producer):
    data = load_job_plan(data_plan_manifest, root, protocol)
    check(data['kind'] == 'stage2_data_plan', 'Shared data plan required')
    rp, _ = artifact(receipt_manifest, root, protocol, 'stage2_model_readiness')
    config = read_json(rp.parent / 'readiness.json')['model_config']
    receipt = verify_readiness(rp, root, protocol, config)
    check(receipt['references']['dataset_fingerprint'] == data['dataset_fingerprint'], 'Plan/receipt dataset mismatch')
    body = {k: plain(data[k]) for k in ('schema', 'specification', 'dataset_fingerprint', 'protocol_fingerprint',
                                      'source_identity', 'work_units', 'rows', 'counts', 'chunk_size_images', 'target_reference')}
    sites = receipt['execution_identity']['site_specs']
    body.update(kind='stage2_model_plan', model=receipt['model'], model_config=plain(config),
                execution_identity=plain(receipt['execution_identity']), dimensions=plain(receipt['dimensions']),
                adapter=plain(receipt['adapter']), bridge_sha256=bridge_hash(),
                references={'data_plan': reference(data_plan_manifest, root), 'readiness': reference(rp, root)},
                required_sites={c: sorted(k for k in sites if
                    (k.startswith('decoder:') if c != 'image_only' else not k.startswith('decoder:'))) for c in CONDITIONS},
                audit_sites=sorted(k for k in sites if not k.startswith('decoder:')),
                unsupported=['post_caption', 'decoder_visual_token_pooling'],
                required_releases=[{'condition': c, 'split': s} for s in SPLITS for c in CONDITIONS])
    body['plan_fingerprint'] = fingerprint(body)
    body['job_id'] = body['model'] + '_' + body['execution_identity']['execution_fingerprint'][:12] + '_' + body['plan_fingerprint'][:12]
    meta = metadata('stage2_model_plan', protocol, body['dataset_fingerprint'], producer, config['config_fingerprint'])
    meta['upstream_artifacts'] = list(body['references'].values())
    return publish_json(root, 'cache/shared/stage2-plans/' + meta['run_id'], meta, 'plan.json', body)


@contract
def load_job_plan(plan_manifest, root, protocol):
    path, meta = artifact(plan_manifest, root, protocol)
    check(meta['kind'] in ('stage2_data_plan', 'stage2_model_plan'), 'Wrong plan artifact kind')
    body = read_json(path.parent / 'plan.json')
    digest = body.pop('plan_fingerprint', None)
    job = body.pop('job_id', None)
    check(digest == fingerprint(body), 'Plan fingerprint mismatch')
    check(body['schema'] == PLAN_SCHEMA and body['kind'] == meta['kind'] and
          body['specification'] == specification() and body['protocol_fingerprint'] == fingerprint(protocol)
          and body['dataset_fingerprint'] == meta['dataset_fingerprint'], 'Plan schema/common identity mismatch')
    if meta['kind'] == 'stage2_data_plan':
        dp, _ = dereference(body['references']['dataset'], root, protocol, 'stage1_dataset')
        vp, _ = dereference(body['references']['data_validation'], root, protocol, 'stage1_data_validation')
        bundle, _ = verify_data(dp, vp, root, protocol)
        expected = compile_inventory(bundle)
        check(all(body[k] == v for k, v in expected.items()) and body['recipes'] == text_recipes(), 'Plan inventory/recipe mismatch')
        dataset = read_json(dp.parent / 'dataset.json')
        check(body['source_identity'] == {k: dataset[k] for k in
              ('records_sha256', 'targets_sha256', 'split_manifest_sha256', 'category_order')}, 'Frozen source/target/split identity mismatch')
        check(body['target_reference'] == {'artifact': body['references']['dataset'], 'file': 'targets.jsonl',
              'primary_categories': protocol['primary_categories']} and body['confirm_processing'] is False,
              'Target/mask reference mismatch')
    else:
        dp, _ = dereference(body['references']['data_plan'], root, protocol, 'stage2_data_plan')
        data = load_job_plan(dp, root, protocol)
        rp, _ = dereference(body['references']['readiness'], root, protocol, 'stage2_model_readiness')
        receipt = verify_readiness(rp, root, protocol, body['model_config'])
        for key in ('model', 'model_config', 'execution_identity', 'dimensions', 'adapter'):
            check(body[key] == plain(receipt[key]), 'Model plan/readiness mismatch: ' + key)
        for key in ('work_units', 'rows', 'counts', 'source_identity', 'target_reference', 'chunk_size_images'):
            check(body[key] == plain(data[key]), 'Model plan/shared inventory mismatch: ' + key)
        sites = body['execution_identity']['site_specs']
        required = {c: sorted(k for k in sites if (k.startswith('decoder:') if c != 'image_only'
                                                  else not k.startswith('decoder:'))) for c in CONDITIONS}
        check(body['required_sites'] == required and body['audit_sites'] == sorted(k for k in sites if not k.startswith('decoder:')),
              'Model plan site mismatch')
        check(body['bridge_sha256'] == bridge_hash(), 'Evidence bridge implementation changed')
        check(body['required_releases'] == [{'condition': c, 'split': s} for s in SPLITS for c in CONDITIONS]
              and body['unsupported'] == ['post_caption', 'decoder_visual_token_pooling'], 'Model release/unsupported inventory mismatch')
        check(job == body['model'] + '_' + body['execution_identity']['execution_fingerprint'][:12] + '_' + digest[:12], 'Job prefix collision/identity mismatch')
        body['job_id'] = job
    body['plan_fingerprint'] = digest
    body['manifest_reference'] = reference(path, root)
    return VerifiedPlan(body)


def resolve_worker_settings(plan_manifest, root, protocol, model=None):
    plan = load_job_plan(plan_manifest, root, protocol)
    if plan['kind'] == 'stage2_model_plan':
        check(model == plan['model'], 'MODEL differs from bound model plan')
        rp, _ = dereference(plan['references']['readiness'], root, protocol, 'stage2_model_readiness')
        receipt = read_json(rp.parent / 'readiness.json')
        refs = receipt['references']
        return frozen({'plan': plan, 'dataset_manifest': str(Path(root).resolve()/refs['dataset']['path']),
                       'model_config': str((Path(root).resolve()/refs['configuration']['path']).parent/'models.json'),
                       'readiness_manifest': str(rp), 'model': model})
    check(model is None, 'E03 shared data plan has no MODEL')
    return frozen({'plan': plan, 'dataset_manifest': str(Path(root).resolve()/plan['references']['dataset']['path'])})


def planned_rows(plan, chunk_id, condition):
    check(condition in CONDITIONS, 'Unknown condition')
    units = [u for u in plan['work_units'] if u['chunk_id'] == chunk_id]
    check(len(units) == 1, 'Unknown/duplicate chunk')
    result = []
    for index, row in enumerate(units[0]['rows'][condition]):
        row = plain(row)
        row['row_index'] = index
        row['feature_keys'] = {k: feature_cache_key(plan['execution_identity']['execution_fingerprint'],
            row['input_view_fingerprint'], plain(plan['execution_identity']['site_specs'][k]))
            for k in plan['required_sites'][condition]}
        result.append(row)
    return result
