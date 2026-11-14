"""Conservative alignment against the final, image-expanded processor sequence."""
from src.common.contracts import ContractError


def caption_positions(text, caption, offsets, special_mask, image_positions):
    if text.count(caption) != 1:
        raise ContractError('Caption must occur exactly once in decoded processor input')
    start = text.index(caption)
    end = start + len(caption)
    if len(offsets) != len(special_mask):
        raise ContractError('Offset/mask length mismatch')
    selected = []
    covered = set()
    for i, (a, b) in enumerate(offsets):
        if b > start and a < end:
            if special_mask[i] or i in image_positions or a < start or b > end:
                raise ContractError('Caption boundary crosses a special/image/prefix token')
            selected.append(i)
            covered.update(range(a, b))
    required = {i for i in range(start, end) if not text[i].isspace()}
    if not selected or not required.issubset(covered):
        raise ContractError('Caption truncated or offset mapping incomplete')
    if selected != list(range(selected[0], selected[-1] + 1)):
        raise ContractError('Noncontiguous caption content')
    return selected


def validate_attention(matrix, caption, image):
    """Summarize measured heads x queries x keys; never infer visibility from order."""
    import numpy as np
    a = np.asarray(matrix, dtype=float)
    if a.ndim != 3 or a.shape[-1] != a.shape[-2] or not np.isfinite(a).all() or (a < 0).any():
        raise ContractError('Invalid attention tensor')
    if not np.allclose(a.sum(-1), 1, atol=0.02, rtol=0.02):
        raise ContractError('Attention rows are not normalized')
    last = caption[-1]
    return {'last_caption_to_image_mass': float(a[:, last, image].sum(-1).mean()) if image else None,
            'image_to_caption_mass': float(a[:, image, :][:, :, caption].sum(-1).mean()) if image else None,
            'caption_tokens': len(caption), 'image_tokens': len(image),
            'evidence': 'observed attention probabilities; zero mass does not establish causal non-use'}
