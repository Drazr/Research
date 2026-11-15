"""Condition/split and layered-key regressions; no actual checkpoint evidence."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from src.common.contracts import (ContractError, ExtractionRequest, FeatureBatch,
                                  FeatureSite, ModelInput)
from src.common.cache_keys import (input_view_fingerprint, execution_fingerprint,
                                  feature_cache_key, readout_cache_key, analysis_cache_key)
from src.models.hf import HFAdapter

def row(split='FIT', language='en'):
    return ModelInput('a'*16, 'missing.png', 'b'*64, language, 'caption-' + language,
                      'Original ' + language, split)

D = FeatureSite('decoder', 1, 'last_caption')
V = FeatureSite('encoder', None, 'visual_mean')

class ExtractionTests(unittest.TestCase):
    def test_split_roles_are_explicit_and_confirm_always_closed(self):
        for split in ('FIT', 'DEV-S', 'DEV-C', 'SCREEN', 'CONFIRM'):
            for condition, site in (('image_caption', D), ('caption_only', D), ('image_only', V)):
                request = ExtractionRequest([row(split)], [site], condition)
                if split == 'FIT':
                    request.validate()
                else:
                    with self.assertRaises(ContractError): request.validate()
                request = replace(request, purpose='preliminary_features')
                if split == 'CONFIRM':
                    with self.assertRaises(ContractError): request.validate()
                else:
                    request.validate()
        with self.assertRaises(ContractError):
            ExtractionRequest([row()], [D], purpose='confirmation').validate()

    def test_bad_conditions_sites_duplicates_and_splits_rejected_before_forward(self):
        adapter = HFAdapter.__new__(HFAdapter)
        adapter._forward = lambda *a, **k: self.fail('forward must not run')
        bad = [ExtractionRequest([row('CONFIRM')], [D], purpose='preliminary_features'),
               ExtractionRequest([row()], [D], 'image_only'),
               ExtractionRequest([row()], [V], 'caption_only'),
               ExtractionRequest([row(), row(language='hi')], [V], 'image_only'),
               ExtractionRequest([row(), row('SCREEN', 'hi')], [D], purpose='preliminary_features'),
               ExtractionRequest([row()], [D, D]),
               ExtractionRequest([row()], [D], 'other')]
        for request in bad:
            with self.assertRaises(ContractError): adapter.extract_request(request)

    def test_prepare_guards_before_processor_and_image_io(self):
        adapter = HFAdapter.__new__(HFAdapter)
        for condition in ('image_caption', 'caption_only', 'image_only'):
            with self.assertRaises(ContractError):
                adapter._prepare(row('CONFIRM'), condition, 'preliminary_features')

    def test_public_request_routes_condition_and_preserves_view_identity(self):
        adapter = HFAdapter.__new__(HFAdapter)
        adapter.describe = lambda: {'supported_sites': [D.key, V.key]}
        calls = []
        def forward(r, sites, condition, **kwargs):
            calls.append((r.split, condition, kwargs['purpose']))
            return {s.key: [1., 2.] for s in sites}, {}
        adapter._forward = forward
        for condition, site in (('image_caption', D), ('caption_only', D), ('image_only', V)):
            request = ExtractionRequest([row('SCREEN')], [site], condition, 'preliminary_features')
            batch = adapter.extract_request(request)
            batch.validate_request(request)
            self.assertEqual(calls[-1], ('SCREEN', condition, 'preliminary_features'))
            batch.input_condition = 'wrong'
            with self.assertRaises(ContractError): batch.validate_request(request)
        adapter.extract([row()], [D])
        self.assertEqual(calls[-1], ('FIT', 'image_caption', 'stage1_smoke'))

    def test_image_only_serialization_contains_no_caption_or_language_metadata(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest('Pillow unavailable')
        import hashlib
        import tempfile
        class Value:
            def __init__(self, values): self.values = values
            def tolist(self): return self.values
        class Batch(dict):
            def to(self, device): return self
        class Tokenizer:
            all_special_ids = [99]
            def decode(self, *a, **k): return '<image>'
            def __call__(self, *a, **k):
                return {'input_ids': [99], 'offset_mapping': [(0, 7)]}
        class Processor:
            tokenizer = Tokenizer()
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                return '<image>'
            def __call__(self, **kwargs):
                return Batch(input_ids=[Value([99])], attention_mask=[Value([1])])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new('RGB', (2, 2)).save(root / 'image.png')
            r = replace(row('SCREEN'), image_path='image.png',
                        image_sha256=hashlib.sha256((root/'image.png').read_bytes()).hexdigest())
            adapter = HFAdapter.__new__(HFAdapter)
            adapter.root, adapter.processor = root, Processor()
            adapter.config = {'processor_kwargs': {}, 'max_input_tokens': 10,
                              'image_token_ids': [99], 'device': 'cpu', 'adapter': 'qwen'}
            _, evidence = adapter._prepare(r, 'image_only', 'preliminary_features')
            self.assertEqual(adapter.processor.messages, [{'role': 'user', 'content': [{'type': 'image'}]}])
            self.assertEqual(evidence['caption_positions'], [])
            self.assertIsNone(evidence['last_caption'])

class CacheIdentityTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((Path(__file__).resolve().parents[1]/'configs/models.json').read_text())['models']['qwen']
        self.backend = {'device_name': 'fixture L4', 'batch_size': 1}
        self.sources = {'hf.py': 'a'*64}

    def execution(self, config=None, backend=None):
        return execution_fingerprint(config or self.config, backend or self.backend, self.sources)

    def test_tolerances_and_readout_version_do_not_invalidate_features(self):
        changed = dict(self.config, repeat_atol=.001, config_fingerprint='c'*64)
        changed['software_versions'] = dict(changed['software_versions'], **{'scikit-learn': 'different'})
        self.assertEqual(self.execution(), self.execution(changed))

    def test_weight_processor_backend_and_source_changes_invalidate(self):
        for key, value in (('revision', 'c'*40), ('processor_kwargs', {'max_pixels': 100}),
                           ('precision', 'float32'), ('chat_template_sha256', 'd'*64)):
            self.assertNotEqual(self.execution(), self.execution(dict(self.config, **{key: value})))
        self.assertNotEqual(self.execution(), self.execution(backend={'device_name': 'different GPU', 'batch_size': 1}))
        old = self.execution()
        self.sources['hf.py'] = 'b'*64
        self.assertNotEqual(old, self.execution())
        with self.assertRaises(ContractError): self.execution(dict(self.config, new_unclassified_policy=True))

    def test_image_only_is_once_per_image_and_conditions_never_collide(self):
        self.assertEqual(input_view_fingerprint(row(), 'image_only'), input_view_fingerprint(row(language='hi'), 'image_only'))
        ids = {input_view_fingerprint(row(), c) for c in ('image_only', 'caption_only', 'image_caption')}
        self.assertEqual(len(ids), 3)
        self.assertNotEqual(input_view_fingerprint(row(), 'caption_only'), input_view_fingerprint(row(language='hi'), 'caption_only'))

    def test_downstream_choices_have_separate_keys(self):
        feature = feature_cache_key(self.execution(), input_view_fingerprint(row(), 'image_caption'),
                                    {'module_path': 'block.0', 'pooling': 'last_caption', 'output': 'pre-final-norm'})
        a = readout_cache_key([feature], {'targets': 'fixed', 'fit_ids': ['image1']}, {'C': 1})
        b = readout_cache_key([feature], {'targets': 'fixed', 'fit_ids': ['image1']}, {'C': 10})
        self.assertNotEqual(a, b)
        self.assertNotEqual(analysis_cache_key([a], {'threshold': .5, 'bootstrap': 0}),
                            analysis_cache_key([a], {'threshold': .6, 'bootstrap': 1}))
        self.assertEqual(feature, feature_cache_key(self.execution(), input_view_fingerprint(row(), 'image_caption'),
                                    {'module_path': 'block.0', 'pooling': 'last_caption', 'output': 'pre-final-norm'}))

if __name__ == '__main__':
    unittest.main()
