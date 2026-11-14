"""Layered content identities; provenance is retained separately from reuse keys.

These functions do not authorize a read: validate the ExtractionRequest and
completion hashes before consuming any cached values, including existing ones.
"""
from src.common.contracts import ContractError, fingerprint, require_sha256

INFERENCE_FIELDS = (
    'checkpoint_id', 'revision', 'processor_revision', 'tokenizer_revision',
    'adapter', 'model_class', 'precision', 'quantization', 'image_policy',
    'processor_kwargs', 'image_processor_use_fast', 'chat_template_sha256',
    'max_input_tokens', 'attention_backend', 'image_token_ids',
)
PROVENANCE_FIELDS = {'config_fingerprint', 'repeat_atol', 'repeat_rtol',
                     'decoder_path', 'visual_paths', 'device', 'software_versions',
                     'image_processor_class'}


def input_view_fingerprint(row, condition):
    identity = {'schema': 'input-view.v2', 'condition': condition, 'image_id': row.image_id}
    if condition not in ('image_caption', 'caption_only', 'image_only'):
        raise ContractError('Unknown input condition')
    if condition != 'caption_only':
        identity['image_sha256'] = row.image_sha256
    if condition != 'image_only':
        identity.update(language=row.language, caption_id=row.caption_id, caption_text=row.caption_text)
    # Split belongs in a frozen assignment manifest, not in numeric feature math.
    return fingerprint(identity)


def execution_fingerprint(config, backend_identity, source_hashes):
    missing = set(INFERENCE_FIELDS) - set(config)
    unknown = set(config) - set(INFERENCE_FIELDS) - PROVENANCE_FIELDS
    if missing or unknown:
        raise ContractError('Unclassified inference configuration fields: ' + str(sorted(missing | unknown)))
    if any(config[k] is None for k in INFERENCE_FIELDS if k != 'quantization'):
        raise ContractError('Unresolved inference configuration')
    if not backend_identity or not source_hashes or not config.get('software_versions'):
        raise ContractError('Actual backend, inference source hashes and versions required')
    for value in source_hashes.values():
        require_sha256(value)
    return fingerprint({'schema': 'execution.v2',
        'inference': {k: config[k] for k in INFERENCE_FIELDS},
        'image_processor_class': config.get('image_processor_class'),
        'versions': {k: v for k, v in config['software_versions'].items() if k != 'scikit-learn'},
        'backend': backend_identity, 'inference_source_sha256': source_hashes})


def feature_cache_key(execution_id, view_id, site_spec):
    require_sha256(execution_id)
    require_sha256(view_id)
    if not isinstance(site_spec, dict) or any(not site_spec.get(k) for k in ('module_path', 'pooling', 'output')):
        raise ContractError('Physical hook path, pooling and output convention required')
    return fingerprint({'schema': 'feature.v2', 'execution': execution_id,
                        'input_view': view_id, 'site': site_spec, 'stored_dtype': 'float32'})


def readout_cache_key(feature_refs, training_target_identity, readout_spec):
    if not feature_refs or not training_target_identity or not readout_spec:
        raise ContractError('Ordered feature refs, target/split/mask identity and fitting spec required')
    return fingerprint({'schema': 'readout.v2', 'features': feature_refs,
                        'training_targets': training_target_identity, 'fit': readout_spec})


def analysis_cache_key(prediction_refs, analysis_spec):
    if not prediction_refs or not analysis_spec:
        raise ContractError('Prediction refs and analysis specification required')
    return fingerprint({'schema': 'analysis.v2', 'predictions': prediction_refs, 'analysis': analysis_spec})
