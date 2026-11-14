"""Resolve pinned processor metadata on the actual L4; never load model weights."""
import argparse
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import platform
import re
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.common.contracts import ContractError, canonical_bytes, fingerprint, resolve_model
from src.common.storage import new_run_id, phase1_root, publish, verify, within

PACKAGES = ('torch','torchvision','transformers','numpy','Pillow','scikit-learn',
            'huggingface-hub','tokenizers','safetensors')

def check_versions(config):
    actual = {}
    for package in PACKAGES:
        installed = version(package)
        expected = config['software_versions'].get(package)
        if installed.split('+')[0] != expected:
            raise ContractError(f'Runtime mismatch: {package}; expected {expected}, found {installed}')
        actual[package] = installed
    return actual

def l4_hardware(torch):
    if not torch.cuda.is_available():
        raise ContractError('A CUDA L4 runtime is required; no CPU/precision fallback')
    name = torch.cuda.get_device_name(0)
    memory = torch.cuda.get_device_properties(0).total_memory
    if 'L4' not in name or memory < 22 * 1024**3:
        raise ContractError('Expected a full L4 GPU with approximately 24 GB nominal memory')
    if not torch.cuda.is_bf16_supported():
        raise ContractError('Selected runtime must support bfloat16')
    return {'device_name':name,'total_memory_bytes':memory,'cuda_runtime':torch.version.cuda,
            'bfloat16_supported':True,'python':platform.python_version()}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,choices=['qwen','gemma'])
    p.add_argument('--phase1-root')
    p.add_argument('--dataset-manifest',type=Path,required=True)
    p.add_argument('--code-commit',required=True)
    p.add_argument('--models-config',type=Path,default=REPO/'configs/models.json')
    p.add_argument('--protocol',type=Path,default=REPO/'configs/protocol.json')
    p.add_argument('--storage-scope',required=True,choices=['local-development','persistent-drive'])
    args=p.parse_args(); root=phase1_root(args.phase1_root)
    registry=json.loads(args.models_config.read_text(encoding='utf-8'))
    protocol=json.loads(args.protocol.read_text(encoding='utf-8'))
    config=dict(registry['models'][args.model]);run_id=new_run_id()
    metadata=dict(kind='stage1_runtime_configuration',status='blocked',run_id=run_id,
        code_commit=args.code_commit,protocol_fingerprint=fingerprint(protocol),dataset_fingerprint=None,
        model_fingerprint=None,seeds={},software_versions={'python':platform.python_version()},
        upstream_artifacts=[],storage_scope=args.storage_scope)
    report={'status':'blocked','blockers':[],'weights_loaded':False,'model_inference_ran':False,
            'storage_scope':args.storage_scope,'note':'Configuration readiness only; not VLM validation.'}
    files={}
    try:
        if not re.fullmatch('[0-9a-f]{40}',args.code_commit):
            raise ContractError('Provide the actual code commit SHA')
        manifest=args.dataset_manifest.resolve()
        if manifest.name != '_COMPLETE.json':
            raise ContractError('Expected the completed dataset manifest')
        try: manifest.relative_to(root)
        except ValueError as exc: raise ContractError('Dataset must be under PHASE1_ROOT') from exc
        data=verify(manifest.parent,expected={'kind':'stage1_dataset','protocol_fingerprint':fingerprint(protocol)})
        metadata['dataset_fingerprint']=data['dataset_fingerprint']
        metadata['upstream_artifacts']=[str(manifest)]
        actual=check_versions(config)
        metadata['software_versions'].update(actual)
        import torch
        report['hardware']=l4_hardware(torch)
        from transformers import AutoConfig,AutoProcessor,AutoTokenizer
        # No token is printed, serialized or stored in repository configuration.
        # Hugging Face resolves the user's existing login/HF_TOKEN if needed.
        cache=str(within(root,'model_cache'))
        processor=AutoProcessor.from_pretrained(config['checkpoint_id'],
            revision=config['processor_revision'],cache_dir=cache,trust_remote_code=False,
            use_fast=config['image_processor_use_fast'])
        tokenizer=AutoTokenizer.from_pretrained(config['checkpoint_id'],
            revision=config['tokenizer_revision'],cache_dir=cache,trust_remote_code=False,use_fast=True)
        if not tokenizer.is_fast:
            raise ContractError('Fast tokenizer required for token-span validation')
        template=processor.chat_template
        if not isinstance(template,str):
            raise ContractError('Expected the pinned processor chat template as a string')
        import hashlib
        template_hash=hashlib.sha256(template.encode('utf-8')).hexdigest()
        if config.get('chat_template_sha256') not in (None,template_hash):
            raise ContractError('Pinned processor template checksum mismatch')
        model_config=AutoConfig.from_pretrained(config['checkpoint_id'],revision=config['revision'],
                                               cache_dir=cache,trust_remote_code=False)
        expected_type={'qwen':'qwen2_5_vl','gemma':'gemma3'}[args.model]
        if model_config.model_type != expected_type:
            raise ContractError('Selected checkpoint architecture mismatch')
        field={'qwen':'image_token_id','gemma':'image_token_index'}[args.model]
        token_id=getattr(model_config,field)
        if type(token_id) is not int or token_id not in tokenizer.all_special_ids:
            raise ContractError('Model image token is not a tokenizer special token')
        if config.get('image_token_ids') not in (None,[token_id]):
            raise ContractError('Pinned image-token metadata mismatch')
        config.update(chat_template_sha256=template_hash,image_token_ids=[token_id],
                      software_versions=actual,config_fingerprint=None,
                      image_processor_class=type(processor.image_processor).__name__)
        resolved=resolve_model({'models':{args.model:config}},args.model)
        registry['models'][args.model]=resolved
        metadata['model_fingerprint']=resolved['config_fingerprint']
        report.update(status='passed',configuration_resolved=True,model_fingerprint=resolved['config_fingerprint'],
                      checkpoint_id=resolved['checkpoint_id'],revision=resolved['revision'])
        files['models.json']=canonical_bytes(registry)+b'\n'
    except (ContractError,ImportError,PackageNotFoundError,FileNotFoundError) as exc:
        report['blockers'].append(str(exc))
    except Exception as exc:
        report['blockers'].append(type(exc).__name__+': '+str(exc))
        report['action']='Check pinned runtime and authorized Hugging Face access; Gemma is gated.'
    metadata['status']=report['status']
    files['configuration_report.json']=canonical_bytes(report)+b'\n'
    # This is a generated configuration cache, not a fitted-model checkpoint.
    # Unique runs and manifest fingerprints protect identity without overlong
    # nested hash paths on Windows development roots.
    output=publish(root,f'cache/runtime-configurations/{args.model}/{run_id}',metadata,files)
    print(json.dumps({'status':report['status'],'artifact':str(output),
                      'models_config':str(output/'models.json') if 'models.json' in files else None,
                      'blockers':report['blockers']}))
    return 0 if report['status']=='passed' else 2

if __name__=='__main__':sys.exit(main())
