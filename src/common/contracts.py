"""Stage 1 wire/API contracts. Standard library only; Python >=3.9."""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple

SCHEMA_VERSION = 'stage1.v1'
LANGUAGES = ('en', 'hi', 'bn', 'te')
SPLITS = ('FIT', 'DEV-S', 'DEV-C', 'SCREEN', 'CONFIRM')
INSPECTION_CHECKS = (
    'native_text_preserved', 'caption_spans_valid', 'attention_visibility_documented',
    'residual_hooks_consistent', 'no_gold_in_model_input', 'eval_mode',
    'frozen_parameters', 'input_policy_respected',
)

class ContractError(ValueError):
    pass

def canonical_bytes(value: Any) -> bytes:
    # No Unicode normalization: the original caption characters stay intact.
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')

def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()

def select_caption(candidates: Sequence['ModelInput'], draw_key: int = 0) -> 'ModelInput':
    if not candidates or type(draw_key) is not int or draw_key < 0:
        raise ContractError('Nonempty candidates and nonnegative draw key required')
    groups = {(r.image_id, r.language, r.split) for r in candidates}
    if len(groups) != 1 or len({r.caption_id for r in candidates}) != len(candidates):
        raise ContractError('Caption selection needs one image/language/split and unique source IDs')
    return min(candidates, key=lambda r: (fingerprint(['caption-draw-v1', r.image_id,
                    r.language, draw_key, r.caption_id]), r.caption_id))

def require_sha256(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ContractError('Expected lowercase SHA256')

def relative_path(value: str) -> str:
    if not isinstance(value, str) or not value or '\\' in value or ':' in value:
        raise ContractError('Expected a portable relative POSIX path')
    p = PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts or str(p) != value or value == '.':
        raise ContractError('Unsafe or noncanonical relative path')
    return value

@dataclass(frozen=True)
class ModelInput:
    image_id: str
    image_path: str                 # Relative to PHASE1_ROOT
    image_sha256: str
    language: str
    caption_id: str                 # Stable original-release ID, not a translation ID
    caption_text: str
    split: str

    def __post_init__(self):
        if not isinstance(self.image_id, str) or not re.fullmatch('[0-9a-f]{16}', self.image_id):
            raise ContractError('Image IDs must be canonical 16-character hexadecimal strings')
        relative_path(self.image_path)
        require_sha256(self.image_sha256)
        if self.language not in LANGUAGES or self.split not in SPLITS:
            raise ContractError('Invalid language or split')
        if not isinstance(self.caption_id, str) or not self.caption_id or not isinstance(self.caption_text, str) or not self.caption_text.strip():
            raise ContractError('Original caption ID/text required')

    @property
    def row_key(self) -> Tuple[str, str, str]:
        return (self.image_id, self.language, self.caption_id)

    def require_fit(self) -> None:
        if self.split != 'FIT':
            raise ContractError('Stage 1 model smoke inputs must be FIT only')

@dataclass(frozen=True)
class TargetRecord:
    image_id: str
    split: str
    values: Tuple[Optional[int], ...]  # Exactly 25; None is unknown, not negative

    def __post_init__(self):
        if not isinstance(self.image_id, str) or not re.fullmatch('[0-9a-f]{16}', self.image_id) or self.split not in SPLITS:
            raise ContractError('Invalid target image/split')
        if not isinstance(self.values, tuple) or len(self.values) != 25 or any(v is not None and (type(v) is not int or v not in (0, 1)) for v in self.values):
            raise ContractError('Targets require 25 binary/None values')

    @property
    def known_mask(self) -> Tuple[bool, ...]:
        return tuple(v is not None for v in self.values)

@dataclass
class DatasetBundle:
    records: Sequence[ModelInput]    # One frozen draw-0 caption per image/language
    targets: Mapping[str, TargetRecord]
    category_order: Sequence[str]
    dataset_fingerprint: str
    protocol_fingerprint: str
    validation: Mapping[str, Any]  # Passed data report, with matching fingerprints

    def validate(self, protocol: Mapping[str, Any]) -> None:
        require_sha256(self.dataset_fingerprint)
        if self.protocol_fingerprint != fingerprint(protocol):
            raise ContractError('Protocol fingerprint mismatch')
        if list(self.category_order) != protocol['category_order'] or len(set(self.category_order)) != 25:
            raise ContractError('Category order mismatch')
        if self.validation.get('status') != 'passed' or self.validation.get('dataset_fingerprint') != self.dataset_fingerprint or self.validation.get('protocol_fingerprint') != self.protocol_fingerprint:
            raise ContractError('A matching passed data-validation report is required')
        seen = set()
        groups = {}
        for r in self.records:
            if not isinstance(r, ModelInput):
                raise ContractError('Dataset records must be ModelInput objects')
            key = (r.image_id, r.language)
            if key in seen:
                raise ContractError('Duplicate draw-0 image/language record')
            seen.add(key)
            t = self.targets.get(r.image_id)
            if not isinstance(t, TargetRecord) or t.image_id != r.image_id or t.split != r.split:
                raise ContractError('Target join or split mismatch')
            group = groups.setdefault(r.image_id, [])
            if group and (group[0].image_path, group[0].image_sha256) != (r.image_path, r.image_sha256):
                raise ContractError('Image identity differs across language views')
            group.append(r)
        if not groups or set(groups) != set(self.targets):
            raise ContractError('Empty dataset or unmatched targets')
        if any({r.language for r in rows} != set(LANGUAGES) for rows in groups.values()):
            raise ContractError('Common population must cover all four languages')

@dataclass(frozen=True)
class FeatureSite:
    stage: str                      # decoder, encoder, projector
    depth: Optional[int]            # 1-based decoder block; None for encoder/projector
    pooling: str                    # last_caption, caption_mean, post_caption, visual_mean

    @property
    def key(self) -> str:
        return f'{self.stage}:{self.depth}:{self.pooling}'

@dataclass
class FeatureBatch:
    row_keys: Sequence[Tuple[str, str, str]]
    vectors: Mapping[str, Sequence[Sequence[float]]]
    view_ids: Optional[Sequence[str]] = None
    input_condition: Optional[str] = None

    def validate_request(self, request: 'ExtractionRequest') -> None:
        request.validate()
        self.validate(request.inputs, request.sites)
        if self.input_condition != request.condition or list(self.view_ids or []) != list(request.view_ids):
            raise ContractError('Feature condition/view identity mismatch')

    def validate(self, inputs: Sequence[ModelInput], sites: Sequence[FeatureSite]) -> None:
        if list(self.row_keys) != [r.row_key for r in inputs]:
            raise ContractError('Feature row identity/order mismatch')
        if set(self.vectors) != {s.key for s in sites}:
            raise ContractError('Missing or unexpected feature sites')
        for rows in self.vectors.values():
            if len(rows) != len(inputs) or len(rows) == 0 or not len(rows[0]):
                raise ContractError('Invalid feature dimensions')
            width = len(rows[0])
            if any(len(row) != width or any(not math.isfinite(float(v)) for v in row) for row in rows):
                raise ContractError('Ragged or nonfinite features')


def require_extraction_split(row: ModelInput, purpose: str) -> None:
    # Called before image IO, rendering, tokenization and cache consumption.
    if purpose == 'stage1_smoke':
        row.require_fit()
    elif purpose == 'preliminary_features':
        if row.split not in ('FIT', 'DEV-S', 'DEV-C', 'SCREEN'):
            raise ContractError('CONFIRM is closed to preliminary model processing')
    else:
        raise ContractError('Unknown extraction purpose')


@dataclass(frozen=True)
class ExtractionRequest:
    inputs: Sequence[ModelInput]
    sites: Sequence[FeatureSite]
    condition: str = 'image_caption'
    purpose: str = 'stage1_smoke'
    schema_version: str = 'extraction.v2'

    def validate(self) -> None:
        if self.schema_version != 'extraction.v2' or self.condition not in ('image_caption', 'caption_only', 'image_only'):
            raise ContractError('Unknown extraction schema/condition')
        if not self.inputs or not self.sites or len({s.key for s in self.sites}) != len(self.sites):
            raise ContractError('Nonempty inputs and unique sites required')
        images = {}
        for row in self.inputs:
            require_extraction_split(row, self.purpose)
            identity = (row.image_sha256, row.split)
            if row.image_id in images and images[row.image_id] != identity:
                raise ContractError('Inconsistent image identity/split')
            images[row.image_id] = identity
        if len(set(self.view_ids)) != len(self.inputs):
            raise ContractError('Duplicate extraction views; image-only is once per image')
        for s in self.sites:
            decoder = s.stage == 'decoder' and type(s.depth) is int and s.depth > 0 and s.pooling in ('last_caption', 'caption_mean')
            visual = s.stage in ('encoder', 'projector') and s.depth is None and s.pooling == 'visual_mean'
            if not (decoder or visual):
                raise ContractError('Unsupported feature-site semantics')
            if (self.condition == 'caption_only' and visual) or (self.condition == 'image_only' and decoder):
                raise ContractError('Feature site incompatible with input condition')

    @property
    def view_ids(self) -> Tuple[str, ...]:
        from src.common.cache_keys import input_view_fingerprint
        return tuple(input_view_fingerprint(r, self.condition) for r in self.inputs)

def decoder_depths(blocks: int) -> Tuple[int, ...]:
    if type(blocks) is not int or blocks < 1:
        raise ContractError('Positive decoder block count required')
    return tuple(sorted({(blocks + 3)//4, (blocks + 1)//2, (3*blocks + 3)//4, blocks}))

def resolve_model(registry: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    try:
        config = dict(registry['models'][key])
    except KeyError as exc:
        raise ContractError('Unknown model selector') from exc
    for name in ('checkpoint_id', 'adapter', 'precision', 'image_policy', 'max_input_tokens'):
        if not config.get(name):
            raise ContractError('Unresolved model configuration: ' + name)
    if type(config['max_input_tokens']) is not int or config['max_input_tokens'] < 1:
        raise ContractError('max_input_tokens must be a positive integer')
    for name in ('revision', 'processor_revision', 'tokenizer_revision'):
        if not re.fullmatch('[0-9a-f]{40}', str(config.get(name))):
            raise ContractError('Immutable revision required: ' + name)
    require_sha256(config.get('chat_template_sha256'))
    if not config.get('software_versions'):
        raise ContractError('Pin model software versions')
    declared = config.pop('config_fingerprint', None)
    computed = fingerprint(config)
    if declared is not None and declared != computed:
        raise ContractError('Model configuration fingerprint mismatch')
    config['config_fingerprint'] = computed
    return config

class ModelAdapter(Protocol):
    """Model worker supplies src.models.registry.create_adapter(key, config, root)."""
    def describe(self) -> Mapping[str, Any]:
        """decoder_blocks, model_fingerprint, repeat_atol/rtol, supported_sites, hook_convention."""
        ...

    def inspect(self, inputs: Sequence[ModelInput]) -> Mapping[str, Any]:
        """INSPECTION_CHECKS bools plus per-input token spans/visibility evidence."""
        ...

    def extract(self, inputs: Sequence[ModelInput], sites: Sequence[FeatureSite]) -> FeatureBatch:
        """Frozen eval-mode forward pass; return CPU-convertible row vectors, no generation."""
        ...

    def extract_request(self, request: ExtractionRequest) -> FeatureBatch:
        """Explicit condition/purpose; validate split before IO or cache reuse."""
        ...
