"""Exact E03 analyzers, FIT statistics and portable numeric serialization.

All numerical preprocessing lives here so its source digest identifies both
fitting and transformation. No sklearn vectorizer defaults are involved.
"""
from collections import Counter
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from src.common.contracts import ContractError, fingerprint
from src.stage2.interfaces import text_recipes
from src.stage2.schema import SPLITS, check, json_bytes, jsonl, plain, read_json


def validate_recipe(recipe):
    recipe = plain(recipe)
    check(recipe in text_recipes(), 'E03 requires an unchanged plan recipe')
    return recipe


def analyze(text, family):
    """Yield tuple word terms or literal code-point substrings, unchanged."""
    check(isinstance(text, str), 'Caption must be original Unicode text')
    if family == 'word_tfidf':
        tokens = text.split()
        for n in (1, 2):
            for i in range(len(tokens) - n + 1):
                yield tuple(tokens[i:i+n])
    else:
        check(family in ('char_tfidf', 'pooled_char_tfidf'), 'Unknown E03 analyzer')
        for n in (3, 4, 5):
            for i in range(len(text) - n + 1):
                yield text[i:i+n]


def caption_texts(records, rows, split):
    """Guard every row before reading any text; metadata-only CONFIRM filtering."""
    check(split in SPLITS, 'CONFIRM/unknown E03 split is closed')
    check(all(r['split'] == split and r['condition'] == 'caption_only'
              and r['caption_draw_key'] == 0 for r in rows), 'E03 row split/view mismatch')
    keys = [tuple(r['source_row_key']) for r in rows]
    check(len(keys) == len(set(keys)), 'Duplicate E03 source rows')
    eligible = {}
    for record in records:
        if record.split not in SPLITS:
            continue
        check(record.row_key not in eligible, 'Duplicate eligible dataset record')
        eligible[record.row_key] = record
    result = []
    for row, key in zip(rows, keys):
        check(key in eligible, 'Missing E03 source record')
        record = eligible[key]
        check(record.split == split and record.image_id == row['image_id']
              and record.language == row['language'] and record.caption_id == row['caption_id'],
              'E03 source identity/group mismatch')
        text = record.caption_text
        check(sha256(text.encode('utf-8')).hexdigest() == row['caption_text_sha256'],
              'Original caption text hash mismatch')
        result.append(text)
    return result


def fit_identities(rows):
    return {'fit_record_ids_sha256': fingerprint([r['source_row_key'] for r in rows]),
            'fit_texts_sha256': fingerprint([r['caption_text_sha256'] for r in rows])}


def choose_terms(document_frequency, corpus_count, min_df, cap):
    eligible = [t for t, df in document_frequency.items() if df >= min_df]
    # Cap by corpus count first; assign columns only after the cap.
    retained = sorted(eligible, key=lambda t: (-corpus_count[t], t))[:cap]
    return sorted(retained)


def fit_preprocessor(recipe, rows, texts):
    recipe = validate_recipe(recipe)
    check(rows and len(rows) == len(texts) and all(r['split'] == 'FIT' for r in rows),
          'Fit requires exact nonempty FIT captions')
    check(all(recipe['language'] is None or r['language'] == recipe['language'] for r in rows),
          'FIT population differs from recipe language')
    check(len({tuple(r['source_row_key']) for r in rows}) == len(rows)
          and all(sha256(t.encode('utf-8')).hexdigest() == r['caption_text_sha256']
                  for r, t in zip(rows, texts)), 'FIT record/text identity mismatch')
    spec = {'fit_split': 'FIT', 'fit_rows_sha256': fingerprint(plain(rows))}
    if recipe['family'] == 'length':
        return dict(spec, columns=recipe['columns'], fitted_parameters=False), None
    df, counts = Counter(), Counter()
    for text in texts:
        count = Counter(analyze(text, recipe['family']))
        counts.update(count)
        df.update(count.keys())
    terms = choose_terms(df, counts, recipe['min_df'], recipe['max_features'])
    check(terms, 'Empty fitted E03 vocabulary; required family blocked')
    encoded = ([{'n': len(t), 'tokens': list(t)} for t in terms]
               if recipe['family'] == 'word_tfidf' else terms)
    frequencies = [df[t] for t in terms]
    spec.update(terms=encoded, term_order_sha256=fingerprint(encoded),
                document_frequency=frequencies, corpus_term_count=[counts[t] for t in terms],
                n_fit_documents=len(texts), **fit_identities(rows))
    idf = np.log((1 + len(texts)) / (1 + np.asarray(frequencies, dtype=np.float64))) + 1
    return spec, np.ascontiguousarray(idf, dtype='<f8')


def transform(recipe, spec, idf, texts):
    recipe = validate_recipe(recipe)
    if recipe['family'] == 'length':
        values = np.asarray([[np.log1p(np.float64(len(t))), np.log1p(np.float64(len(t.split())))]
                             for t in texts], dtype='<f4').reshape(len(texts), 2)
        check(np.isfinite(values).all(), 'Nonfinite E03 length values')
        return values
    terms = ([tuple(t['tokens']) for t in spec['terms']]
             if recipe['family'] == 'word_tfidf' else spec['terms'])
    check(idf.dtype.str == '<f8' and idf.shape == (len(terms),), 'IDF shape/dtype mismatch')
    vocabulary = {t: i for i, t in enumerate(terms)}
    data, indices, indptr = [], [], [0]
    for text in texts:
        counts = Counter(analyze(text, recipe['family']))
        retained = sorted((vocabulary[t], c) for t, c in counts.items() if t in vocabulary)
        cols = [i for i, _ in retained]
        values = (1 + np.log(np.asarray([c for _, c in retained], dtype=np.float64))) * idf[cols]
        if len(values):
            values /= np.sqrt(np.sum(values * values, dtype=np.float64))
        data.extend(values)
        indices.extend(cols)
        indptr.append(len(data))
    bound = np.iinfo(np.int32).max
    check(max(len(texts), len(terms), len(data)) <= bound, 'CSR int32 bound exceeded')
    values = csr_matrix((np.asarray(data, dtype='<f4'), np.asarray(indices, dtype='<i4'),
                         np.asarray(indptr, dtype='<i4')), shape=(len(texts), len(terms)))
    check(np.isfinite(values.data).all() and (values.data > 0).all(),
          'CSR nonfinite/explicit zero values')
    return values


def write_preprocessor(folder, spec, idf):
    folder = Path(folder)
    (folder/'preprocessor.json').write_bytes(json_bytes(spec))
    paths = {'preprocessor.json': folder/'preprocessor.json'}
    if idf is not None:
        np.save(folder/'idf.npy', idf, allow_pickle=False)
        paths['idf.npy'] = folder/'idf.npy'
    return paths


def read_preprocessor(folder, recipe, fit_rows):
    """Reconstruct the transform without fitting or trusting Python objects."""
    recipe = validate_recipe(recipe)
    folder = Path(folder)
    spec = read_json(folder/'preprocessor.json')
    check(spec['fit_split'] == 'FIT' and spec['fit_rows_sha256'] == fingerprint(plain(fit_rows)),
          'Reload FIT identity mismatch')
    if recipe['family'] == 'length':
        check(spec == {'fit_split': 'FIT', 'fit_rows_sha256': fingerprint(plain(fit_rows)),
                       'columns': recipe['columns'], 'fitted_parameters': False}, 'Length recipe mismatch')
        return spec, None
    encoded = spec['terms']
    if recipe['family'] == 'word_tfidf':
        check(all(set(t) == {'n', 'tokens'} and t['n'] in (1, 2)
                  and len(t['tokens']) == t['n'] and all(isinstance(s, str) and s for s in t['tokens'])
                  for t in encoded), 'Invalid word tuple encoding')
        terms = [tuple(t['tokens']) for t in encoded]
    else:
        check(all(isinstance(t, str) and len(t) in (3, 4, 5) for t in encoded), 'Invalid character term')
        terms = encoded
    check(0 < len(terms) <= recipe['max_features'] and terms == sorted(set(terms))
          and spec['term_order_sha256'] == fingerprint(encoded), 'Reload term order mismatch')
    check(all(spec.get(k) == v for k, v in fit_identities(fit_rows).items()), 'Reload FIT hashes mismatch')
    df, counts = spec['document_frequency'], spec['corpus_term_count']
    n = len(fit_rows)
    check(spec['n_fit_documents'] == n and len(df) == len(counts) == len(terms)
          and all(type(d) is int and 2 <= d <= n for d in df)
          and all(type(c) is int and c >= d for c, d in zip(counts, df)), 'Reload FIT counts mismatch')
    idf = np.load(folder/'idf.npy', allow_pickle=False)
    expected = np.log((1+n)/(1+np.asarray(df, dtype=np.float64)))+1
    check(idf.dtype.str == '<f8' and idf.shape == (len(terms),) and np.array_equal(idf, expected),
          'Reload IDF formula/dtype mismatch')
    return spec, idf


def write_matrix(folder, rows, values):
    folder = Path(folder)
    (folder/'rows.jsonl').write_bytes(jsonl(rows))
    paths = {'rows.jsonl': folder/'rows.jsonl'}
    arrays = ({'data': values.data, 'indices': values.indices, 'indptr': values.indptr}
              if isinstance(values, csr_matrix) else {'matrix': values})
    for name, array in arrays.items():
        path = folder/(name+'.npy')
        np.save(path, array, allow_pickle=False)
        paths[path.name] = path
    zero = int(np.sum(np.diff(values.indptr) == 0)) if isinstance(values, csr_matrix) else int(np.sum(~values.any(axis=1)))
    return paths, zero


def equal_matrices(a, b):
    if isinstance(a, csr_matrix) and isinstance(b, csr_matrix):
        return a.shape == b.shape and all(np.array_equal(x, y) and x.dtype == y.dtype for x, y in
               ((a.data, b.data), (a.indices, b.indices), (a.indptr, b.indptr)))
    return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
