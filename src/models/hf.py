"""Explicitly configured Hugging Face Qwen/Gemma VLM backend.

No architecture paths, checkpoint sizes, precision or processor policies are guessed.
Unsupported alignment/backends fail closed before features can be published.
"""
import hashlib
from importlib.metadata import version
from pathlib import Path

from src.common.contracts import (ContractError, ExtractionRequest, FeatureBatch, FeatureSite,
                                  decoder_depths, require_extraction_split)
from src.common.storage import within
from .spans import caption_positions, map_image_expansion, validate_attention

def validate_device(torch, config):
    """Fail before processor/weight downloads when the requested GPU is absent."""
    if str(config['device']).startswith('cuda'):
        if not torch.cuda.is_available():
            raise ContractError('Configured CUDA device is unavailable; no CPU fallback')
        if config['precision'] == 'bfloat16' and not torch.cuda.is_bf16_supported():
            raise ContractError('Configured CUDA device does not support bfloat16')


class HFAdapter:
    def __init__(self, key, config, root):
        self.key, self.config, self.root = key, config, Path(root).resolve()
        required = ('model_class', 'decoder_path', 'image_token_ids', 'processor_kwargs',
                    'device', 'attention_backend', 'repeat_atol', 'repeat_rtol','image_processor_use_fast')
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
        if type(config['image_processor_use_fast']) is not bool:
            raise ContractError('Image processor fast/slow policy must be explicit')
        for name, expected in config['software_versions'].items():
            if expected != version(name):
                raise ContractError('Runtime version mismatch: ' + name)
        import torch
        import transformers
        self.torch = torch
        validate_device(torch,config)
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
            cache_dir=str(cache), trust_remote_code=False,use_fast=config['image_processor_use_fast'])
        self.processor.tokenizer = transformers.AutoTokenizer.from_pretrained(
            config['checkpoint_id'], revision=config['tokenizer_revision'],
            cache_dir=str(cache), trust_remote_code=False, use_fast=True)
        if config.get('image_processor_class') and type(self.processor.image_processor).__name__ != config['image_processor_class']:
            raise ContractError('Loaded image processor class differs from resolved configuration')
        template = self.processor.chat_template
        if not isinstance(template, str) or hashlib.sha256(template.encode('utf-8')).hexdigest() != config['chat_template_sha256']:
            raise ContractError('Loaded chat template hash mismatch')
        self.model = getattr(transformers, config['model_class']).from_pretrained(
            config['checkpoint_id'], revision=config['revision'], torch_dtype=dtype,
            attn_implementation=config['attention_backend'], cache_dir=str(cache),
            trust_remote_code=False, use_safetensors=True).to(config['device'])
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

    def _prepare(self, row, condition='image_caption', purpose='stage1_smoke'):
        require_extraction_split(row, purpose)  # Before IO, rendering or tokenization.
        if condition not in ('image_caption', 'caption_only', 'image_only'):
            raise ContractError('Unsupported input condition')
        from PIL import Image
        content, images = [], []
        if condition != 'caption_only':
            path = within(self.root, row.image_path)
            if hashlib.sha256(path.read_bytes()).hexdigest() != row.image_sha256:
                raise ContractError('Image checksum mismatch')
            with Image.open(path) as image:
                images = [image.convert('RGB')]
            content.append({'type': 'image'})
        if condition != 'image_only':
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
        # Qwen decoding can normalize native text. Align its original serialization
        # using fast-tokenizer offsets, then verify the complete processed ID stream.
        alignment_text = text if self.config['adapter'] == 'qwen' else decoded
        aligned = tok(alignment_text, add_special_tokens=False, return_offsets_mapping=True,
                      return_special_tokens_mask=True)
        if self.config['adapter'] == 'qwen':
            mapping = map_image_expansion(aligned['input_ids'], ids, self.config['image_token_ids'])
        elif aligned['input_ids'] != ids:
            raise ContractError('Unsupported lossless token-offset alignment for this tokenizer')
        else:
            mapping = {i: i for i in range(len(ids))}
        image_positions = [i for i, token in enumerate(ids) if token in self.config['image_token_ids']]
        if bool(image_positions) != bool(images):
            raise ContractError('Configured image tokens disagree with input condition')
        special = [token in tok.all_special_ids for token in aligned['input_ids']]
        source_image_positions = [i for i, token in enumerate(aligned['input_ids'])
                                  if token in self.config['image_token_ids']]
        caption = ([] if condition == 'image_only' else
                   [mapping[i] for i in caption_positions(alignment_text, row.caption_text,
                        aligned['offset_mapping'], special, source_image_positions)])
        if image_positions and caption and max(image_positions) >= min(caption):
            raise ContractError('Processor did not preserve image-before-caption order')
        if 'labels' in processed:
            raise ContractError('Unexpected labels in processor output')
        evidence = {'row_key': row.row_key, 'condition': condition, 'token_count': len(ids),
                    'native_serialization': text, 'decoded_processor_sequence': decoded,
                    'caption_alignment': 'original serialization offsets; exact text IDs with image expansion' if self.config['adapter'] == 'qwen' else 'lossless decoded offsets',
                    'processor_attention_mask': processed['attention_mask'][0].tolist(),
                    'input_ids_sha256': hashlib.sha256(str(ids).encode()).hexdigest(),
                    'caption_positions': caption, 'last_caption': caption[-1] if caption else None,
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

    def _forward(self, row, sites, condition='image_caption', attention=False, purpose='stage1_smoke'):
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise ContractError('Frozen evaluation mode required')
        inputs, evidence = self._prepare(row, condition, purpose)
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
        return self.extract_request(ExtractionRequest(inputs, sites))

    def extract_request(self, request):
        request.validate()
        if any(s.key not in self.describe()['supported_sites'] for s in request.sites):
            raise ContractError('Unsupported requested extraction site')
        vectors = {s.key: [] for s in request.sites}
        for row in request.inputs:
            values, _ = self._forward(row, request.sites, request.condition, purpose=request.purpose)
            for key, vector in values.items():
                vectors[key].append(vector)
        batch = FeatureBatch([r.row_key for r in request.inputs], vectors,
                             request.view_ids, request.condition)
        batch.validate_request(request)
        return batch

    def describe_execution(self):
        """Actual numerical backend plus per-site physical extraction identity."""
        from src.common.cache_keys import execution_fingerprint
        torch = self.torch
        cuda = str(self.config['device']).startswith('cuda')
        backend = {'device_type': 'cuda' if cuda else 'cpu', 'batch_size': 1,
                   'padding': False, 'truncation': False, 'eval': True,
                   'cuda_version': torch.version.cuda,
                   'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                   'float32_matmul_precision': torch.get_float32_matmul_precision(),
                   'cudnn_version': torch.backends.cudnn.version(),
                   'cudnn_benchmark': torch.backends.cudnn.benchmark,
                   'cudnn_deterministic': torch.backends.cudnn.deterministic,
                   'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                   'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                   'image_processor_class': type(self.processor.image_processor).__name__,
                   'tokenizer_class': type(self.processor.tokenizer).__name__}
        if cuda:
            backend.update(device_name=torch.cuda.get_device_name(self.config['device']),
                           capability=list(torch.cuda.get_device_capability(self.config['device'])))
        source = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (Path(__file__), Path(__file__).with_name('spans.py'))}
        specs = {}
        for key in self.describe()['supported_sites']:
            stage, depth, pooling = key.split(':')
            path = (self.config['decoder_path'] + '.' + str(int(depth)-1)
                    if stage == 'decoder' else self.visual_paths[stage])
            specs[key] = {'module_path': path, 'pooling': pooling,
                          'output': 'forward module output; before final decoder norm' if stage == 'decoder'
                                    else 'exposed visual module output; mean over all non-channel axes'}
        return {'execution_fingerprint': execution_fingerprint(self.config, backend, source),
                'backend': backend, 'source_sha256': source, 'site_specs': specs}
