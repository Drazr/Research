"""One selector and the coordinator's immutable configuration resolver."""
from src.common.contracts import ContractError, resolve_model


def create_adapter(key, config, root):
    if key not in ('qwen', 'gemma') or config.get('adapter') != key:
        raise ContractError('Selector/adapter mismatch')
    resolved = resolve_model({'models': {key: config}}, key)
    from .hf import HFAdapter
    return HFAdapter(key, resolved, root)
