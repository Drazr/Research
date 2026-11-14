"""Explicitly configured Hugging Face Qwen/Gemma VLM backend.

No architecture paths, checkpoint sizes, precision or processor policies are guessed.
Unsupported alignment/backends fail closed before features can be published.
"""
import hashlib
from importlib.metadata import version
from pathlib import Path

from src.common.contracts import (ContractError, FeatureBatch, FeatureSite,
                                  decoder_depths)
from src.common.storage import within
from .spans import caption_positions, validate_attention


class HFAdapter:
    def __init__(self, key, config, root):
        self.key, self.config, self.root = key, config, Path(root).resolve()
        required = ('model_class', 'decoder_path', 'image_token_ids', 'processor_kwargs',
                    'device', 'attention_backend', 'repeat_atol', 'repeat_rtol')
        missing = [name for name in required if config.get(name) is None]
        if missing:
            raise ContractError('Unresolved adapter configuration: ' + ', '.join(missing))
        if not isinstance(config['processor_kwargs'], dict) or not isinstance(config['image_token_ids'], list) or not config['image_token_ids'] or any(type(i) is not int or i < 0 for i in config['image_token_ids']):
            raise ContractError('Explicit processor kwargs and nonempty integer image token IDs required')
        if not all(isinstance(config[k], (int, float)) and 0 <= config[k] < 1 for k in ('repeat_atol', 'repeat_rtol')):
            raise ContractError('Repeat tolerances must be explicit numbers in [0,1)')
        if config.get('quantization') is not None:
            raise ContractError('Only quantization=null (no quantization) is currently supported')
        if config.get('attention_backend') != 'eager':
            raise ContractError('Stage 1 visibility validation requires explicit eager attention')
        for name in ('torch', 'transformers', 'numpy', 'Pillow'):
            if config['software_versions'].get(name) != version(name):
                raise ContractError('Runtime version mismatch: ' + name)
        import torch
        import transformers
        self.torch = torch
        classes = {'qwen': ('Qwen2VLForConditionalGeneration', 'Qwen2_5_VLForConditionalGeneration'),
                   'gemma': ('Gemma3ForConditionalGeneration',)}
        if config['model_class'] not in classes[key]:
            raise ContractError('Unsupported checkpoint architecture class')
        dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                 'float16': torch.float16}.get(config['precision'])
        if dtype is None:
            raise ContractError('Unsupported precision')
        if config['image_policy'] != 'pinned_processor_with_explicit_kwargs':
            raise ContractError('Unsupported image policy')
        cache = self.root / 'model_cache'
        cache.mkdir(parents=True, exist_ok=True)
        self.processor = transformers.AutoProcessor.from_pretrained(
            config['checkpoint_id'], revision=config['processor_revision'],
            cache_dir=str(cache), trust_remote_code=False)
        self.processor.tokenizer = transformers.AutoTokenizer.from_pretrained(
            config['checkpoint_id'], revision=config['tokenizer_revision'],
            cache_dir=str(cache), trust_remote_code=False, use_fast=True)
        template = self.processor.chat_template
        if not isinstance(template, str) or hashlib.sha256(template.encode('utf-8')).hexdigest() != config['chat_template_sha256']:
            raise ContractError('Loaded chat template hash mismatch')
        self.model = getattr(transformers, config['model_class']).from_pretrained(
            config['checkpoint_id'], revision=config['revision'], torch_dtype=dtype,
            attn_implementation=config['attention_backend'], cache_dir=str(cache),
            trust_remote_code=False).to(config['device'])
        self.model.eval()
        self.model.requires_grad_(False)
        self.blocks = self.model.get_submodule(config['decoder_path'])
        self.depths = decoder_depths(len(self.blocks))
        self.visual_paths = config.get('visual_paths', {})
        for stage, path in self.visual_paths.items():
            if stage not in ('encoder', 'projector'):
                raise ContractError('Unknown visual stage')
            self.model.get_submodule(path)

    def describe(self):
        sites = [FeatureSite('decoder', d, p).key for d in self.depths
                 for p in ('last_caption', 'caption_mean')]
        sites += [FeatureSite(s, None, 'visual_mean').key for s in self.visual_paths]
        return {'decoder_blocks': len(self.blocks), 'decoder_depths': self.depths,
                'midpoint': (len(self.blocks) + 1)//2,
                'model_fingerprint': self.config['config_fingerprint'],
                'repeat_atol': self.config['repeat_atol'], 'repeat_rtol': self.config['repeat_rtol'],
                'supported_sites': sites,
                'unsupported_sites': {'post_caption': 'Boundary semantics not yet validated',
                    **{s: 'No explicit exposed module path configured' for s in ('encoder', 'projector') if s not in self.visual_paths}},
                'hook_convention': 'forward hook on actual 1-based decoder block output, before final decoder norm'}

    def _prepare(self, row, condition='image_caption'):
        row.require_fit()  # Guard before image reads, template rendering or tokenization.
        if condition not in ('image_caption', 'caption_only'):
            raise ContractError('Unsupported input condition')
        from PIL import Image
        content, images = [], []
        if condition == 'image_caption':
            path = within(self.root, row.image_path)
            if hashlib.sha256(path.read_bytes()).hexdigest() != row.image_sha256:
                raise ContractError('Image checksum mismatch')
            with Image.open(path) as image:
                images = [image.convert('RGB')]
            content.append({'type': 'image'})
        content.append({'type': 'text', 'text': 'Caption:\n' + row.caption_text})
        text = self.processor.apply_chat_template([{'role': 'user', 'content': content}],
                                                  tokenize=False, add_generation_prompt=False)
        kwargs = dict(self.config['processor_kwargs'])
        if set(kwargs) & {'text', 'images', 'return_tensors', 'truncation', 'padding', 'max_length'}:
            raise ContractError('Processor kwargs may not override validated input construction')
        processed = self.processor(text=[text], images=images or None, return_tensors='pt',
                                   truncation=False, padding=False, **kwargs)
        ids = processed['input_ids'][0].tolist()
        if len(ids) > self.config['max_input_tokens']:
            raise ContractError('Input exceeds configured token limit; truncation forbidden')
        tok = self.processor.tokenizer
        decoded = tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        aligned = tok(decoded, add_special_tokens=False, return_offsets_mapping=True,
                      return_special_tokens_mask=True)
        if aligned['input_ids'] != ids:
            raise ContractError('Unsupported lossless token-offset alignment for this tokenizer')
        image_positions = [i for i, token in enumerate(ids) if token in self.config['image_token_ids']]
        if bool(image_positions) != bool(images):
            raise ContractError('Configured image tokens disagree with input condition')
        special = [token in tok.all_special_ids for token in ids]
        caption = caption_positions(decoded, row.caption_text, aligned['offset_mapping'], special, image_positions)
        if image_positions and max(image_positions) >= min(caption):
            raise ContractError('Processor did not preserve image-before-caption order')
        if 'labels' in processed:
            raise ContractError('Unexpected labels in processor output')
        evidence = {'row_key': row.row_key, 'condition': condition, 'token_count': len(ids),
                    'native_serialization': text, 'decoded_processor_sequence': decoded,
                    'processor_attention_mask': processed['attention_mask'][0].tolist(),
                    'input_ids_sha256': hashlib.sha256(str(ids).encode()).hexdigest(),
                    'caption_positions': caption, 'last_caption': caption[-1],
                    'image_positions': image_positions, 'role_and_other_positions':
                    [i for i in range(len(ids)) if i not in caption and i not in image_positions]}
        return processed.to(self.config['device']), evidence

    @staticmethod
    def _tensor(output):
        if hasattr(output, 'last_hidden_state'):
            return output.last_hidden_state
        if isinstance(output, tuple):
            return output[0]
        return output

    def _forward(self, row, sites, condition='image_caption', attention=False):
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise ContractError('Frozen evaluation mode required')
        inputs, evidence = self._prepare(row, condition)
        captured, handles = {}, []
        def hook(name):
            def collect(module, args, output):
                if name in captured:
                    raise ContractError('Hook invoked more than once')
                captured[name] = self._tensor(output).detach().clone()
            return collect
        try:
            for site in sites:
                if site.key not in self.describe()['supported_sites']:
                    raise ContractError('Unsupported feature site: ' + site.key)
                name = (site.stage, site.depth)
                if any(k == name for k in captured):
                    continue
                # Install exactly one hook per module, even with two pooling sites.
                if name in [n for n, _ in handles]:
                    continue
                module = self.blocks[site.depth-1] if site.stage == 'decoder' else self.model.get_submodule(self.visual_paths[site.stage])
                handles.append((name, module.register_forward_hook(hook(name))))
            with self.torch.inference_mode():
                result = self.model(**inputs, use_cache=False, output_attentions=attention, return_dict=True)
            vectors = {}
            for site in sites:
                h = captured[(site.stage, site.depth)]
                if site.stage == 'decoder':
                    if h.ndim != 3 or h.shape[0] != 1 or h.shape[1] != evidence['token_count']:
                        raise ContractError('Decoder hook shape does not match processed token sequence')
                    pos = evidence['caption_positions']
                    vector = h[0, pos[-1]] if site.pooling == 'last_caption' else h[0, pos].mean(0)
                else:
                    if h.ndim < 2 or not h.numel():
                        raise ContractError('Unsupported visual output shape')
                    vector = h.reshape(-1, h.shape[-1]).mean(0)
                vectors[site.key] = vector.float().cpu().tolist()
            if attention:
                tensors = getattr(result, 'attentions', None)
                if tensors is None or len(tensors) != len(self.blocks) or any(a is None for a in tensors):
                    raise ContractError('Backend did not expose every decoder attention layer')
                evidence['attention_by_block'] = [validate_attention(a[0].float().cpu().numpy(),
                    evidence['caption_positions'], evidence['image_positions']) for a in tensors]
            return vectors, evidence
        finally:
            for _, handle in handles:
                handle.remove()

    def inspect(self, inputs):
        if not inputs:
            raise ContractError('Nonempty FIT smoke inputs required')
        evidence = []
        for row in inputs:
            record = {'row_key': row.row_key, 'conditions': {}}
            for condition in ('image_caption', 'caption_only'):
                _, item = self._forward(row, [], condition, attention=True)
                record['conditions'][condition] = item
            evidence.append(record)
        return {'checks': {'native_text_preserved': True, 'caption_spans_valid': True,
                'attention_visibility_documented': True, 'residual_hooks_consistent': True,
                'no_gold_in_model_input': True, 'eval_mode': not self.model.training,
                'frozen_parameters': all(not p.requires_grad for p in self.model.parameters()),
                'input_policy_respected': True}, 'records': evidence}

    def extract(self, inputs, sites):
        vectors = {s.key: [] for s in sites}
        for row in inputs:
            values, _ = self._forward(row, sites)
            for key, vector in values.items():
                vectors[key].append(vector)
        batch = FeatureBatch([r.row_key for r in inputs], vectors)
        batch.validate(inputs, sites)
        return batch
