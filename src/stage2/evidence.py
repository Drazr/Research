"""Single compatibility boundary to the existing adapter forward. No new math."""
import hashlib
from dataclasses import dataclass

from src.common.contracts import FeatureBatch, ExtractionRequest
from src.common.storage import file_sha256
from .schema import check, REPO, contract


@dataclass(frozen=True)
class EvidenceBatch:
    features: FeatureBatch
    evidence: tuple


def validate_token_evidence(item, condition, row_key=None, limit=1024):
    check(item.get('condition') == condition, 'Token evidence condition mismatch')
    if row_key is not None:
        check(tuple(item.get('row_key', ())) == tuple(row_key), 'Token evidence source row mismatch')
    n = item.get('token_count')
    check(type(n) is int and 0 < n <= limit, 'Invalid/over-limit processed token count')
    masks = [item.get(k) for k in ('caption_positions', 'image_positions', 'role_and_other_positions')]
    check(all(isinstance(v, list) and all(type(i) is int and 0 <= i < n for i in v)
              and v == sorted(set(v)) for v in masks), 'Invalid ordered token positions')
    flat = sum(masks, [])
    check(sorted(flat) == list(range(n)), 'Token masks must exactly partition the processed sequence')
    caption, image, _ = masks
    check(bool(caption) == (condition != 'image_only'), 'Caption span disagrees with condition')
    check(bool(image) == (condition != 'caption_only'), 'Image span disagrees with condition')
    check(item.get('last_caption') == (caption[-1] if caption else None), 'Wrong last caption position')
    check(not image or not caption or max(image) < min(caption), 'Image-before-caption ordering failed')
    attention = item.get('processor_attention_mask')
    check(isinstance(attention, list) and attention == [1] * n, 'Batch-one unpadded attention mask required')
    from src.common.contracts import require_sha256
    require_sha256(item.get('input_ids_sha256'))
    require_sha256(item.get('native_serialization_sha256'))
    check(bool(item.get('caption_alignment')), 'Alignment method required')


@contract
def extract_with_evidence(adapter, request):
    check(isinstance(request, ExtractionRequest), 'extraction.v2 request required')
    request.validate()  # before any adapter IO
    check(all(s.key in adapter.describe()['supported_sites'] for s in request.sites),
          'Unsupported requested extraction site')
    # Only the current HF implementation (or an explicit fixture subclass) is supported.
    from src.models.hf import HFAdapter
    check(isinstance(adapter, HFAdapter), 'Evidence bridge supports the validated HFAdapter only')
    vectors = {s.key: [] for s in request.sites}
    records = []
    for row in request.inputs:
        values, item = adapter._forward(row, request.sites, request.condition,
                                        attention=False, purpose=request.purpose)
        check(set(values) == set(vectors), 'Forward returned wrong site inventory')
        item = dict(item)
        text = item.pop('native_serialization')
        item.pop('decoded_processor_sequence', None)
        check('attention_by_block' not in item, 'Bulk extraction must not request attention dumps')
        item['native_serialization_sha256'] = hashlib.sha256(text.encode('utf-8')).hexdigest()
        validate_token_evidence(item, request.condition, row.row_key, adapter.config['max_input_tokens'])
        records.append(item)
        for key in vectors:
            vectors[key].append(values[key])
    batch = FeatureBatch([r.row_key for r in request.inputs], vectors, request.view_ids, request.condition)
    batch.validate_request(request)
    return EvidenceBatch(batch, tuple(records))


def bridge_hash():
    return file_sha256(REPO / 'src/stage2/evidence.py')


def compare_public_path(adapter, request):
    """FIT-only first-image equality gate; caller invokes all nine production views."""
    check(all(r.split == 'FIT' for r in request.inputs), 'Initial equality audit must use FIT')
    result = extract_with_evidence(adapter, request)
    public = adapter.extract_request(request)
    public.validate_request(request)
    check(public.vectors == result.features.vectors, 'Bridge/public vectors differ; stop and revalidate')
    return result
