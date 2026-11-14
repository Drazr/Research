"""CPU contract regressions. These fixtures are not checkpoint/GPU validation."""
import unittest
from pathlib import Path
from src.common.contracts import ContractError, ModelInput, decoder_depths
from src.models.spans import caption_positions
from src.models.registry import create_adapter
from src.models.hf import HFAdapter


class ModelContractTests(unittest.TestCase):
    def test_depths(self):
        for blocks in range(1, 100):
            ds = decoder_depths(blocks)
            self.assertEqual(ds[-1], blocks)
            self.assertIn((blocks + 1)//2, ds)
            self.assertEqual(len(ds), len(set(ds)))
        self.assertEqual(decoder_depths(32), (8, 16, 24, 32))
        for invalid in (0, -1, True, 2.5):
            with self.assertRaises(ContractError):
                decoder_depths(invalid)

    def test_original_script_and_punctuation(self):
        for caption in ('Original caption.', 'यह चित्र है।', 'এটি ছবি।', 'ఇది చిత్రం.'):
            text = '<img>Caption:\n' + caption + '<end>'
            offsets = [(i, i+1) for i in range(len(text))]
            selected = caption_positions(text, caption, offsets, [False]*len(text), [0])
            self.assertEqual(''.join(text[i] for i in selected), caption)
            self.assertEqual(selected[-1], text.index(caption)+len(caption)-1)

    def test_ambiguous_missing_crossing_and_special_spans_rejected(self):
        cases = [('xx', 'x', [(0, 1), (1, 2)], [False, False], []),
                 ('abc', 'absent', [(0, 3)], [False], []),
                 ('xa', 'a', [(0, 2)], [False], []),
                 ('a', 'a', [(0, 1)], [True], []),
                 ('a', 'a', [(0, 1)], [False], [0]),
                 ('abc', 'abc', [(0, 1)], [False], [])]
        for args in cases:
            with self.assertRaises(ContractError):
                caption_positions(*args)

    def test_non_fit_rejected_before_io_or_processor(self):
        adapter = HFAdapter.__new__(HFAdapter)
        for split in ('DEV-S', 'DEV-C', 'SCREEN', 'CONFIRM'):
            row = ModelInput('0'*16, 'missing.png', '0'*64, 'en', 'c0', 'Caption.', split)
            with self.assertRaisesRegex(ContractError, 'FIT only'):
                adapter._prepare(row)

    def test_unresolved_registry_never_imports_or_loads_weights(self):
        for key in ('qwen', 'gemma'):
            with self.assertRaises(ContractError):
                create_adapter(key, {'adapter': key}, Path('.'))
        with self.assertRaises(ContractError):
            create_adapter('qwen', {'adapter': 'gemma'}, Path('.'))

    def test_measured_attention_not_assumed_from_order(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest('numpy unavailable')
        from src.models.spans import validate_attention
        attention = np.array([[[1, 0, 0], [.5, .5, 0], [.25, .25, .5]]])
        result = validate_attention(attention, [1, 2], [0])
        self.assertEqual(result['last_caption_to_image_mass'], .25)
        self.assertEqual(result['image_to_caption_mass'], 0)
        attention[0, 2] = [0, 0, 1]
        self.assertEqual(validate_attention(attention, [1, 2], [0])['last_caption_to_image_mass'], 0)
        with self.assertRaises(ContractError):
            validate_attention(np.zeros((1, 3, 3)), [1, 2], [0])

    def test_block_outputs_pooling_and_hook_cleanup(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest('numpy unavailable')
        from contextlib import nullcontext
        from types import SimpleNamespace
        from src.common.contracts import FeatureSite
        class Tensor(np.ndarray):
            def detach(self): return self
            def clone(self): return self.copy()
            def float(self): return self
            def cpu(self): return self
        class Block:
            def __init__(self): self.hooks = []
            def register_forward_hook(self, hook):
                self.hooks.append(hook)
                return SimpleNamespace(remove=lambda: self.hooks.remove(hook))
        blocks = [Block(), Block()]
        class Model:
            training = False
            def parameters(self): return []
            def __call__(self, **kwargs):
                for i, block in enumerate(blocks):
                    h = (np.array([[[1., 2.], [3., 4.], [5., 6.]]])*(i+1)).view(Tensor)
                    for hook in block.hooks: hook(block, (), (h,))
                return SimpleNamespace(last_hidden_state=np.zeros((1, 3, 2)))
        adapter = HFAdapter.__new__(HFAdapter)
        adapter.model = Model()
        adapter.blocks = blocks
        adapter.depths = (1, 2)
        adapter.visual_paths = {}
        adapter.config = {'config_fingerprint': 'a'*64, 'repeat_atol': 0., 'repeat_rtol': 0.}
        adapter.torch = SimpleNamespace(inference_mode=nullcontext)
        adapter._prepare = lambda *args: ({}, {'caption_positions': [0, 1], 'token_count': 3})
        sites = [FeatureSite('decoder', d, p) for d in (1, 2) for p in ('last_caption', 'caption_mean')]
        vectors, _ = adapter._forward(None, sites)
        self.assertEqual(vectors['decoder:2:last_caption'], [6., 8.])
        self.assertEqual(vectors['decoder:1:caption_mean'], [2., 3.])
        self.assertTrue(all(not b.hooks for b in blocks))
        with self.assertRaises(ContractError):
            adapter._forward(None, [sites[0], FeatureSite('decoder', 99, 'last_caption')])
        self.assertTrue(all(not b.hooks for b in blocks))


if __name__ == '__main__':
    unittest.main()
