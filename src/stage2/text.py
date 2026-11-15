"""E03 artifact contracts only. Analyzers and fitting belong to the E03 worker."""
from pathlib import Path
import numpy as np
from src.common.contracts import LANGUAGES, fingerprint
from src.common.storage import file_sha256
from .schema import (TEXT_SCHEMA, SPLITS, ROLES, artifact, authorize, check, component,
                     dereference, frozen, metadata as run_metadata, plain, publish_json, read_json,
                     read_rows, reference, contract)
from .plans import load_job_plan
from .features import matrix

VERSIONS = ('python', 'unicode', 'numpy', 'scipy', 'scikit-learn', 'analyzer_sha256')


def preprocessing_versions(analyzer_file):
    """Record the actual CPU runtime and the worker's analyzer implementation."""
    import platform, unicodedata
    from importlib.metadata import version
    from .schema import REPO
    from src.common.storage import within
    path = within(REPO, analyzer_file)
    return {'python':platform.python_version(), 'unicode':unicodedata.unidata_version,
            'numpy':version('numpy'), 'scipy':version('scipy'), 'scikit-learn':version('scikit-learn'),
            'analyzer_sha256':file_sha256(path)}


def _producer_runtime(producer):
    check(producer['preprocessing_versions']==preprocessing_versions(producer['analyzer_file']),
          'Text producer runtime/analyzer differs from supplied version identity')


def text_rows(plan, recipe, split):
    check(split in SPLITS, 'CONFIRM/unknown text split is closed')
    language = recipe['language']
    rows = [plain(r) for r in plan['rows']['caption_only']
            if r['split'] == split and (language is None or r['language'] == language)]
    return [dict(r, row_index=i, image_loss_weight=0.25 if language is None else 1.0)
            for i, r in enumerate(rows)]


def _recipe(plan, recipe_id):
    recipes = [plain(r) for r in plan['recipes'] if r['recipe_fingerprint'] == recipe_id]
    check(len(recipes) == 1, 'Unknown text recipe')
    return recipes[0]


def _runtime(versions):
    from src.common.contracts import require_sha256
    check(set(versions) == set(VERSIONS) and all(isinstance(v, str) and v for v in versions.values()), 'Freeze actual text preprocessing versions')
    require_sha256(versions['analyzer_sha256'])
    # The selected profile is fixed, SciPy and Python/Unicode are recorded exactly.
    check(versions['numpy'] == '2.1.3' and versions['scikit-learn'] == '1.6.1', 'E03 selected CPU dependency profile mismatch')


def _check_preprocessor(folder, body, plan):
    check(body['schema'] == TEXT_SCHEMA and body['plan_fingerprint'] == plan['plan_fingerprint'], 'Text preprocessor plan/schema mismatch')
    recipe = _recipe(plan, body['recipe']['recipe_fingerprint'])
    check(set(body['preprocessor_files']) == ({'preprocessor.json'} if recipe['family'] == 'length'
          else {'preprocessor.json', 'idf.npy'}), 'Preprocessor exact file inventory mismatch')
    check(body['recipe'] == recipe, 'Text recipe mismatch')
    _runtime(body['versions'])
    check(body['fit_rows_sha256'] == fingerprint(text_rows(plan, recipe, 'FIT')), 'Preprocessor FIT record/text identity mismatch')
    spec = read_json(folder / 'preprocessor.json')
    check(spec['fit_rows_sha256'] == body['fit_rows_sha256'] and spec['fit_split'] == 'FIT', 'Preprocessing fit must use exact FIT documents')
    if recipe['family'] == 'length':
        check(spec == {'fit_rows_sha256': body['fit_rows_sha256'], 'fit_split': 'FIT',
                       'columns': recipe['columns'], 'fitted_parameters': False}, 'Length column/arithmetic identity mismatch')
        check(body['dimension'] == 2, 'Length has exactly two columns')
    else:
        terms = spec['terms']
        check(isinstance(terms, list) and 0 < len(terms) <= recipe['max_features'], 'Empty/over-cap vocabulary')
        word = recipe['family'] == 'word_tfidf'
        if word:
            check(all(isinstance(t, dict) and set(t) == {'n', 'tokens'} and type(t['n']) is int
                      and t['n'] in (1, 2) and isinstance(t['tokens'], list) and len(t['tokens']) == t['n']
                      and all(isinstance(s, str) and s for s in t['tokens']) for t in terms), 'Word ngram tuple encoding required')
            order = [tuple(t['tokens']) for t in terms]
        else:
            check(all(isinstance(t, str) and len(t) in (3, 4, 5) for t in terms), 'Literal character ngram encoding required')
            order = terms
        check(order == sorted(set(order)), 'Vocabulary columns must be unique lexicographic terms')
        check(spec['term_order_sha256'] == fingerprint(terms) and body['dimension'] == len(terms), 'Vocabulary column identity mismatch')
        df, count = spec['document_frequency'], spec['corpus_term_count']
        n = len(text_rows(plan, recipe, 'FIT'))
        check(spec['n_fit_documents'] == n and len(df) == len(count) == len(terms)
              and all(type(d) is int and 2 <= d <= n for d in df)
              and all(type(c) is int and c >= d for c, d in zip(count, df)), 'Invalid FIT document/term counts')
        idf = np.load(folder/'idf.npy', allow_pickle=False)
        check(idf.dtype.str == '<f8' and idf.shape == (len(terms),) and idf.flags.c_contiguous
              and np.isfinite(idf).all() and np.array_equal(idf, np.log((1+n)/(1+np.asarray(df, dtype=np.float64)))+1), 'FIT-only IDF formula/dtype mismatch')
    check(body['preprocessing_fingerprint'] == fingerprint({k: body[k] for k in
          ('recipe', 'versions', 'fit_rows_sha256', 'dimension', 'preprocessor_files')}), 'Preprocessor identity mismatch')
    for name, digest in body['preprocessor_files'].items():
        check(file_sha256(folder/name) == digest, 'Preprocessor content hash mismatch')
    return recipe


@contract
def publish_text_preprocessor(root, plan, recipe_id, payload_paths, metadata):
    _producer_runtime(metadata)
    from .schema import REPO
    protocol = read_json(REPO/'configs/protocol.json')
    plan = load_job_plan(plan['manifest_reference']['path'], root, protocol)
    check(plan['kind'] == 'stage2_data_plan', 'Text preprocessing uses model-independent data plan')
    recipe = _recipe(plan, recipe_id)
    files = {k: Path(v) for k, v in payload_paths.items()}
    check(set(files) == ({'preprocessor.json'} if recipe['family'] == 'length' else {'preprocessor.json', 'idf.npy'}), 'Preprocessor file inventory mismatch')
    body = {'schema': TEXT_SCHEMA, 'plan': plain(plan['manifest_reference']), 'plan_fingerprint': plan['plan_fingerprint'],
            'storage_probe': metadata['storage_probe'],
            'recipe': recipe, 'versions': metadata['preprocessing_versions'],
            'fit_rows_sha256': fingerprint(text_rows(plan, recipe, 'FIT')),
            'dimension': 2 if recipe['family'] == 'length' else len(read_json(files['preprocessor.json'])['terms']),
            'preprocessor_files': {k: file_sha256(p) for k, p in files.items()}}
    body['preprocessing_fingerprint'] = fingerprint({k: body[k] for k in
        ('recipe', 'versions', 'fit_rows_sha256', 'dimension', 'preprocessor_files')})
    import tempfile, shutil
    with tempfile.TemporaryDirectory() as temp:
        folder = Path(temp)
        for k, v in files.items():
            shutil.copyfile(v, folder/k)
        _check_preprocessor(folder, body, plan)
        from .operations import verify_storage_probe
        verify_storage_probe(body['storage_probe'], root, protocol, plan['dataset_fingerprint'])
        meta = run_metadata('stage2_text_preprocessor', protocol, plan['dataset_fingerprint'], metadata)
        meta['upstream_artifacts'] = [plain(plan['manifest_reference'])]
        return publish_json(root, f"cache/shared/stage2-text/{recipe_id[:16]}/{meta['run_id']}", meta,
                            'fit.json', body, {k: folder/k for k in files})


def _load_preprocessor(manifest, root, protocol):
    p, m = artifact(manifest, root, protocol, 'stage2_text_preprocessor')
    b = read_json(p.parent/'fit.json')
    pp, _ = dereference(b['plan'], root, protocol, 'stage2_data_plan')
    plan = load_job_plan(pp, root, protocol)
    from .operations import verify_storage_probe
    verify_storage_probe(b['storage_probe'], root, protocol, plan['dataset_fingerprint'])
    check(m['dataset_fingerprint'] == plan['dataset_fingerprint'] and m['model_fingerprint'] is None, 'Text common identity mismatch')
    recipe = _check_preprocessor(p.parent, b, plan)
    return p, b, plan, recipe


def _text_matrix(folder, body, plan, recipe, preprocessor):
    rows = read_rows(folder/'rows.jsonl')
    expected = text_rows(plan, recipe, body['split'])
    check(rows == expected and body['rows_sha256'] == file_sha256(folder/'rows.jsonl'), 'Text row/split/order mismatch')
    n, d = len(rows), preprocessor['dimension']
    if recipe['family'] == 'length':
        values = matrix(folder/'matrix.npy', n, d)
    else:
        # Portable CSR is four ordinary non-pickle arrays, never a Python object.
        from scipy.sparse import csr_matrix
        arrays = {k: np.load(folder/(k+'.npy'), allow_pickle=False) for k in ('data', 'indices', 'indptr')}
        data, indices, indptr = (arrays[k] for k in ('data', 'indices', 'indptr'))
        check(data.dtype.str == '<f4' and indices.dtype.str == indptr.dtype.str == '<i4'
              and all(a.ndim == 1 and a.flags.c_contiguous for a in arrays.values()), 'CSR exact dtype/layout mismatch')
        check(len(indptr) == n+1 and indptr[0] == 0 and indptr[-1] == len(data) == len(indices)
              and (np.diff(indptr.astype(np.int64)) >= 0).all() and len(data) <= np.iinfo(np.int32).max,
              'CSR row/nnz bounds mismatch')
        check((indices >= 0).all() and (indices < d).all() and np.isfinite(data).all()
              and (data > 0).all(), 'CSR values/columns invalid or explicit zero')
        for i in range(n):
            check((np.diff(indices[indptr[i]:indptr[i+1]].astype(np.int64)) > 0).all(), 'CSR columns must be strictly sorted unique')
        values = csr_matrix((data, indices, indptr), shape=(n, d))
        norms = np.sqrt(np.asarray(values.astype(np.float64).multiply(values).sum(axis=1)).ravel())
        check(np.all((norms == 0) | np.isclose(norms, 1, atol=2e-7, rtol=2e-7)), 'TF-IDF nonzero rows must be L2 normalized')
    zero = int(np.sum(np.asarray(values.getnnz(axis=1)) == 0)) if hasattr(values, 'getnnz') else int(np.sum(~np.any(values, axis=1)))
    check(body['zero_rows'] == zero and body['shape'] == [n, d], 'Text zero-row/shape report mismatch')
    return rows, values


@contract
def publish_text_release(root, preprocessor_manifest, split, payload_paths, metadata):
    _producer_runtime(metadata)
    from .schema import REPO
    protocol = read_json(REPO/'configs/protocol.json')
    pp, pre, plan, recipe = _load_preprocessor(preprocessor_manifest, root, protocol)
    check(metadata['preprocessing_versions'] == pre['versions'], 'Transform runtime/preprocessor identity mismatch')
    files = {k: Path(v) for k, v in payload_paths.items()}
    names = {'rows.jsonl', 'matrix.npy'} if recipe['family'] == 'length' else {'rows.jsonl', 'data.npy', 'indices.npy', 'indptr.npy'}
    check(set(files) == names and split in SPLITS, 'Text split/payload inventory mismatch')
    body = {'schema': TEXT_SCHEMA, 'preprocessor': reference(pp, root), 'plan': plain(plan['manifest_reference']),
            'plan_fingerprint': plan['plan_fingerprint'], 'preprocessing_fingerprint': pre['preprocessing_fingerprint'],
            'split': split, 'rows_sha256': file_sha256(files['rows.jsonl']),
            'shape': [len(text_rows(plan, recipe, split)), pre['dimension']], 'zero_rows': metadata['zero_rows']}
    import tempfile, shutil
    with tempfile.TemporaryDirectory() as temp:
        folder = Path(temp)
        for k, v in files.items():
            shutil.copyfile(v, folder/k)
        _text_matrix(folder, body, plan, recipe, pre)
        meta = run_metadata('stage2_text_release', protocol, plan['dataset_fingerprint'], metadata)
        meta['upstream_artifacts'] = [reference(pp, root)]
        return publish_json(root, f"cache/shared/stage2-text/{recipe['recipe_fingerprint'][:16]}/{meta['run_id']}",
                            meta, 'text_release.json', body, {k: folder/k for k in files})


def _load_split(manifest, root, protocol):
    p, m = artifact(manifest, root, protocol, 'stage2_text_release')
    body = read_json(p.parent/'text_release.json')
    pp, _ = dereference(body['preprocessor'], root, protocol, 'stage2_text_preprocessor')
    _, pre, plan, recipe = _load_preprocessor(pp, root, protocol)
    check(body['schema'] == TEXT_SCHEMA and body['plan'] == plain(plan['manifest_reference']) and
          body['plan_fingerprint'] == plan['plan_fingerprint'] and
          body['preprocessing_fingerprint'] == pre['preprocessing_fingerprint'] and
          m['dataset_fingerprint'] == plan['dataset_fingerprint'] and m['model_fingerprint'] is None, 'Text release identity mismatch')
    rows, values = _text_matrix(p.parent, body, plan, recipe, pre)
    return body, rows, values


@contract
def publish_text_family(root, recipe, split_payloads, run_id, metadata):
    """recipe is the completed FIT-preprocessor manifest; splits are completed refs."""
    from .schema import REPO
    protocol = read_json(REPO/'configs/protocol.json')
    pp, pre, plan, spec = _load_preprocessor(recipe, root, protocol)
    check(split_payloads and set(split_payloads) <= set(SPLITS), 'Explicit eligible text splits required')
    refs = {}
    for split in SPLITS:
        if split not in split_payloads:
            continue
        b, _, _ = _load_split(split_payloads[split], root, protocol)
        check(b['split'] == split and b['preprocessor'] == reference(pp, root), 'Text family split/preprocessor mismatch')
        refs[split] = reference(split_payloads[split], root)
    check('FIT' in refs, 'Family handoff requires complete FIT release')
    body = {'schema': TEXT_SCHEMA, 'preprocessor': reference(pp, root), 'recipe': spec,
            'plan': plain(plan['manifest_reference']), 'plan_fingerprint': plan['plan_fingerprint'],
            'preprocessing_fingerprint': pre['preprocessing_fingerprint'], 'splits': refs,
            'family_complete': set(refs) == set(SPLITS)}
    meta = run_metadata('stage2_text_family', protocol, plan['dataset_fingerprint'], dict(metadata, run_id=run_id))
    meta['upstream_artifacts'] = [reference(pp, root)] + list(refs.values())
    return publish_json(root, f"cache/shared/stage2-text/{spec['recipe_fingerprint'][:16]}/{component(run_id)}", meta, 'family.json', body)


class TextFamily:
    def __init__(self, manifest, root, protocol, role, expected):
        p, m = artifact(manifest, root, protocol, 'stage2_text_family')
        body = read_json(p.parent/'family.json')
        check(body['schema'] == TEXT_SCHEMA and {'plan_fingerprint','preprocessing_fingerprint'} <= set(expected), 'Text family schema/expected plan/preprocessing identity required')
        check(role in ROLES and ROLES[role] in body['splits'], 'Role has no completed text split')
        for key, value in expected.items():
            check(body.get(key) == plain(value), 'Unexpected text family identity: ' + key)
        pp, _ = dereference(body['preprocessor'], root, protocol, 'stage2_text_preprocessor')
        _, pre, plan, spec = _load_preprocessor(pp, root, protocol)
        check(body['plan'] == plain(plan['manifest_reference']) and body['plan_fingerprint'] == plan['plan_fingerprint']
              and body['recipe'] == spec and body['preprocessing_fingerprint'] == pre['preprocessing_fingerprint']
              and m['dataset_fingerprint'] == plan['dataset_fingerprint'] and m['model_fingerprint'] is None
              and body['family_complete'] == (set(body['splits']) == set(SPLITS)), 'Text family common identity mismatch')
        check('FIT' in body['splits'] and set(body['splits']) <= set(SPLITS), 'Invalid text split catalog')
        # Check all child hashes without loading unauthorized split values.
        for split, ref in body['splits'].items():
            cp, _ = dereference(ref, root, protocol, 'stage2_text_release')
            cb = read_json(cp.parent/'text_release.json')
            check(cb['split'] == split and cb['preprocessor'] == body['preprocessor'], 'Text child identity mismatch')
        self.root, self.protocol, self.role = root, protocol, role
        self._reference=reference(manifest,root)
        self.body, self.recipe = frozen(body), frozen(spec)

    def read_matrix(self, split, language=None):
        authorize(self.role, split)
        dereference(self._reference,self.root,self.protocol,'stage2_text_family')
        check(split in self.body['splits'], 'No completed release for this split')
        check(language is None or language in LANGUAGES, 'Unknown text language')
        check(self.recipe['language'] is None or language in (None, self.recipe['language']), 'Language differs from family')
        p, _ = dereference(self.body['splits'][split], self.root, self.protocol, 'stage2_text_release')
        _, rows, values = _load_split(p, self.root, self.protocol)
        selected = [i for i, row in enumerate(rows) if language is None or row['language'] == language]
        return tuple(frozen(rows[i]) for i in selected), values[selected]


@contract
def load_text_family(family_manifest, root, protocol, role, expected):
    return TextFamily(family_manifest, root, protocol, role, expected)
