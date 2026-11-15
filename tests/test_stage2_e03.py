"""E03 numerical and artifact regressions. Synthetic captions, no real-data claim."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix

from src.common.contracts import ContractError, LANGUAGES, ModelInput, fingerprint
from src.stage2.e03_text import (analyze, caption_texts, choose_terms, equal_matrices,
    fit_preprocessor, read_preprocessor, transform, write_matrix, write_preprocessor)
from src.stage2.interfaces import (load_text_family, preprocessing_versions,
    publish_text_family, publish_text_preprocessor, publish_text_release, storage_probe,
    text_rows, text_recipes)
from src.stage2.plans import compile_inventory
from src.stage2.schema import (REPO, SPLITS, json_bytes, metadata, plain, publish_json,
    read_json, reference, frozen)
from src.stage2 import prepare_e03 as worker

PROTOCOL = read_json(REPO/'configs/protocol.json')


def source(i, split='FIT', language='en', text='a a b'):
    return ModelInput(f'{i:016x}', f'fixture/{i}.jpg', 'a'*64, language,
                      f'{i}:{language}', text, split)


def recipe(family='word_tfidf', language='en'):
    return next(r for r in text_recipes() if r['family'] == family and r['language'] == language)


def rows_for(records, rec, split='FIT'):
    plan = compile_inventory(SimpleNamespace(records=records))
    return text_rows(plan, rec, split)


def producer():
    return {'run_id': 'fixture', 'code_commit': '0'*40,
            'software_versions': {'fixture': 'synthetic acceptance only'},
            'storage_scope': 'local-development'}


class AnalyzerTests(unittest.TestCase):
    def test_unicode_word_and_literal_character_semantics(self):
        text = 'A!\t\tहिन्दी  বাংলা\u00a0తెలుగు\n e\u0301 é 😀'
        tokens = text.split()
        self.assertEqual(list(analyze(text, 'word_tfidf')),
                         [(t,) for t in tokens] + list(zip(tokens, tokens[1:])))
        expected = [text[i:i+n] for n in (3,4,5) for i in range(len(text)-n+1)]
        self.assertEqual(list(analyze(text, 'char_tfidf')), expected)
        self.assertIn('A!\t', expected)
        self.assertNotEqual(list(analyze('e\u0301 e\u0301', 'char_tfidf')),
                            list(analyze('é é', 'char_tfidf')))
        self.assertEqual(list(analyze('a  b', 'char_tfidf')),
                         ['a  ', '  b', 'a  b'])
        self.assertNotEqual(list(analyze('A A', 'word_tfidf')), list(analyze('a a', 'word_tfidf')))

    def test_min_df_document_count_cap_ties_then_column_order(self):
        self.assertEqual(choose_terms({'z':2,'a':2,'b':2,'x':1},
                                     {'z':5,'a':4,'b':4,'x':100}, 2, 2), ['a','z'])
        records = [source(1, text='a a singleton'), source(2, text='a b')]
        rec = recipe()
        rows = rows_for(records, rec)
        spec, _ = fit_preprocessor(rec, rows, [r.caption_text for r in records])
        self.assertEqual(spec['terms'], [{'n':1,'tokens':['a']}])
        self.assertEqual(spec['document_frequency'], [2])
        self.assertEqual(spec['corpus_term_count'], [3])
        # Exercise the actual 20,000 boundary without a modified production recipe.
        counts = {(f't{i:05}',): 2 for i in range(20002)}
        self.assertEqual(choose_terms(counts, counts, 2, 20000), sorted(counts)[:20000])

    def test_known_tfidf_sublinear_smoothed_and_retained_norm(self):
        rec = recipe()
        texts = ['a a b', 'a b b', 'a c']
        records = [source(i+1, text=t) for i,t in enumerate(texts)]
        rows = rows_for(records, rec)
        spec, idf = fit_preprocessor(rec, rows, texts)
        self.assertEqual(spec['terms'], [{'n':1,'tokens':['a']}, {'n':2,'tokens':['a','b']},
                                         {'n':1,'tokens':['b']}])
        self.assertEqual(spec['document_frequency'], [3,2,2])
        expected_idf = np.array([1, np.log(4/3)+1, np.log(4/3)+1], dtype='<f8')
        np.testing.assert_array_equal(idf, expected_idf)
        values = transform(rec, spec, idf, texts + ['OUT_OF_VOCABULARY', ''])
        raw = np.array([1+np.log(2), expected_idf[1], expected_idf[2]])
        np.testing.assert_array_equal(values.toarray()[0], (raw/np.linalg.norm(raw)).astype('<f4'))
        np.testing.assert_array_equal(values.toarray()[2], np.array([1,0,0], dtype='<f4'))
        self.assertEqual(values.shape, (5,3))
        self.assertEqual(values.getnnz(axis=1).tolist(), [3,3,1,0,0])
        self.assertEqual(values.data.dtype.str, '<f4')
        self.assertEqual(values.indices.dtype.str, '<i4')
        self.assertEqual(values.indptr.dtype.str, '<i4')
        self.assertTrue(values.has_sorted_indices)

    def test_no_dev_fit_language_leakage_or_confirm_text_access(self):
        rec = recipe()
        records = [source(1, text='repeat repeat'), source(2, text='repeat repeat'),
                   source(3, 'DEV-S', text='devonly devonly')]
        rows = rows_for(records, rec)
        spec, idf = fit_preprocessor(rec, rows, caption_texts(records, rows, 'FIT'))
        original = json.dumps(spec, sort_keys=True), idf.copy()
        dev = rows_for(records, rec, 'DEV-S')
        self.assertEqual(transform(rec,spec,idf,caption_texts(records,dev,'DEV-S')).nnz, 0)
        self.assertEqual(json.dumps(spec,sort_keys=True), original[0])
        np.testing.assert_array_equal(idf, original[1])
        with self.assertRaises(ContractError): fit_preprocessor(rec,dev,['devonly devonly'])
        with self.assertRaises(ContractError): fit_preprocessor(recipe(language='hi'),rows,['repeat repeat']*2)
        class Closed:
            split = 'CONFIRM'
            @property
            def caption_text(self): raise AssertionError('CONFIRM was read')
        self.assertEqual(caption_texts(records+[Closed()], rows, 'FIT'), ['repeat repeat']*2)
        bad = [dict(rows[0], split='CONFIRM')]
        with self.assertRaises(ContractError): caption_texts([Closed()],bad,'CONFIRM')
        with self.assertRaises(ContractError): caption_texts([Closed()],bad,'FIT')
        with self.assertRaises(ContractError): caption_texts(records,[dict(rows[0],caption_text_sha256='0'*64)],'FIT')

    def test_lengths_codepoints_whitespace_and_empty_rows(self):
        rec = recipe('length')
        texts = ['e\u0301\t😀  বাংলা', '', '  \t\n']
        records = [source(1), source(2)]
        spec, idf = fit_preprocessor(rec, rows_for(records,rec), [r.caption_text for r in records])
        values = transform(rec,spec,idf,texts)
        expected = np.log1p(np.array([[len(t),len(t.split())] for t in texts], dtype=np.float64)).astype('<f4')
        np.testing.assert_array_equal(values,expected)
        self.assertIsNone(idf)
        self.assertFalse(spec['fitted_parameters'])

    def test_empty_vocab_blocks_and_word_tuple_encoding_is_unambiguous(self):
        rec = recipe()
        records = [source(1,text='only'),source(2,text='different')]
        with self.assertRaises(ContractError):
            fit_preprocessor(rec,rows_for(records,rec),[r.caption_text for r in records])
        texts = ['a|b c a b|c']*2
        records = [source(i+1,text=t) for i,t in enumerate(texts)]
        spec,_ = fit_preprocessor(rec,rows_for(records,rec),texts)
        terms = [tuple(t['tokens']) for t in spec['terms']]
        self.assertIn(('a|b','c'),terms)
        self.assertIn(('a','b|c'),terms)
        self.assertNotEqual(terms.index(('a|b','c')),terms.index(('a','b|c')))

    def test_real_fresh_process_numeric_sparse_and_dense_reload(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for family in ('word_tfidf','char_tfidf','length'):
                folder=root/family
                folder.mkdir()
                rec=recipe(family)
                texts=['a a বাংলা e\u0301','a a বাংলা e\u0301','UNSEEN']
                records=[source(i+1,text=t) for i,t in enumerate(texts)]
                rows=rows_for(records,rec)
                spec,idf=fit_preprocessor(rec,rows,texts)
                write_preprocessor(folder,spec,idf)
                write_matrix(folder,rows,transform(rec,spec,idf,texts))
                (folder/'inputs.json').write_bytes(json_bytes(dict(recipe=rec,rows=rows,texts=texts)))
            code = '''
import json,sys
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from src.stage2.e03_text import read_preprocessor,transform,equal_matrices
for p in Path(sys.argv[1]).iterdir():
    args=json.loads((p/'inputs.json').read_text(encoding='utf-8'))
    spec,idf=read_preprocessor(p,args['recipe'],args['rows'])
    expected=transform(args['recipe'],spec,idf,args['texts'])
    if (p/'matrix.npy').exists(): observed=np.load(p/'matrix.npy',allow_pickle=False)
    else:
        a=[np.load(p/(k+'.npy'),allow_pickle=False) for k in ('data','indices','indptr')]
        observed=csr_matrix(tuple(a),shape=expected.shape)
    assert equal_matrices(expected,observed)
print('fresh-process exact reload passed')
'''
            result=subprocess.run([sys.executable,'-c',code,str(root)],cwd=REPO,capture_output=True,text=True,check=True)
            self.assertIn('passed',result.stdout)


class PublicationTests(unittest.TestCase):
    """Real shared writers/readers; only upstream data acceptance is synthetic."""
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name).resolve()
        self.records=[source(i,s,l, 'a a বাংলা e\u0301' if s=='FIT' else 'UNSEEN')
                      for i,s in ((1,'FIT'),(2,'FIT'),(3,'DEV-S'),(4,'DEV-C'),(5,'SCREEN')) for l in LANGUAGES]
        inv=compile_inventory(SimpleNamespace(records=self.records))
        self.plan=dict(inv,kind='stage2_data_plan',schema='stage2.plan.v1',plan_fingerprint='d'*64,
                       dataset_fingerprint='e'*64,recipes=text_recipes())
        marker=publish_json(self.root,'plans/data',metadata('stage2_data_plan',PROTOCOL,'e'*64,producer()),'plan.json',self.plan)
        self.plan['manifest_reference']=reference(marker,self.root)
        self.settings={'plan':frozen(self.plan),'dataset_manifest':str(marker)}
        self.patches=[patch('src.stage2.text.load_job_plan',return_value=frozen(self.plan)),
                      patch.object(worker,'resolve_worker_settings',return_value=self.settings),
                      patch.object(worker,'load_dataset',return_value=SimpleNamespace(records=self.records)),
                      patch.object(worker,'current_producer',side_effect=lambda **k:producer())]
        for p in self.patches:p.start()
        self.original_run=subprocess.run
        def run_child(command,**kw):
            if 'src.stage2.prepare_e03' in command:
                worker.verify_family(self.root,command[command.index('--manifest')+1],command[-1])
                return SimpleNamespace(returncode=0)
            return self.original_run(command,**kw)
        self.child=patch.object(worker.subprocess,'run',side_effect=run_child)
        self.child.start()
    def tearDown(self):
        self.child.stop()
        for p in self.patches:p.stop()
        self.temp.cleanup()

    def test_all_13_early_handoff_resume_without_refit_roles_and_corruption(self):
        with patch.object(worker,'emit'):
            early=worker.run(self.root,'ignored',splits=('FIT','DEV-S'),scope='local-development')
        body=read_json(early.parent/'inventory.json')
        self.assertFalse(body['complete'])
        self.assertEqual(len(body['families']),13)
        before=copy.deepcopy(body['families'])
        self.assertTrue(all(set(e['splits'])=={'FIT','DEV-S'} for e in before.values()))
        with (patch.object(worker,'fit_preprocessor',side_effect=AssertionError('Resume refit')),
              patch.object(worker,'emit')):
            final=worker.run(self.root,'ignored',early,scope='local-development')
        complete=read_json(final.parent/'inventory.json')
        self.assertTrue(complete['complete'])
        for rid,entry in complete['families'].items():
            self.assertEqual(entry['preprocessor'],before[rid]['preprocessor'])
            self.assertEqual(entry['splits']['FIT'],before[rid]['splits']['FIT'])
            family=self.root/entry['family']['path']
            fb=read_json(family.parent/'family.json')
            reader=load_text_family(family,self.root,PROTOCOL,'source_fit',
                {k:fb[k] for k in ('plan_fingerprint','preprocessing_fingerprint')})
            rows,values=reader.read_matrix('FIT')
            pooled=fb['recipe']['language'] is None
            self.assertEqual(len(rows),8 if pooled else 2)
            self.assertEqual({r['image_loss_weight'] for r in rows},{0.25 if pooled else 1.0})
            if pooled:
                self.assertEqual([r['language'] for r in rows[:4]],list(LANGUAGES))
                pre=read_json((self.root/entry['preprocessor']['path']).parent/'preprocessor.json')
                self.assertEqual(pre['n_fit_documents'],8)
            with self.assertRaises(ContractError):reader.read_matrix('SCREEN')
            with self.assertRaises(ContractError):reader.read_matrix('CONFIRM')
        entry=next(iter(complete['families'].values()))
        data=(self.root/entry['splits']['FIT']['path']).parent/'data.npy'
        data.write_bytes(b'corrupt')
        with self.assertRaises(ContractError):
            worker.read_checkpoint(final,self.root,PROTOCOL,frozen(self.plan),preprocessing_versions(worker.ANALYZER))


if __name__=='__main__':unittest.main()
