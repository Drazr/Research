"""Append-only, checksum-verified artifact completion. No cloud atomicity assumption."""
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from pathlib import Path
import uuid
from .contracts import SCHEMA_VERSION, ContractError, canonical_bytes, relative_path, require_sha256

MARKER = '_COMPLETE.json'

def phase1_root(value=None) -> Path:
    value = value or os.environ.get('PHASE1_ROOT')
    if not value:
        raise ContractError('Set PHASE1_ROOT or --phase1-root; no implicit local/Drive fallback')
    return Path(value).expanduser().resolve()

def within(root: Path, relative: str) -> Path:
    relative_path(relative)
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ContractError('Artifact path escapes its root') from exc
    return resolved

def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-') + uuid.uuid4().hex

def publish(root: Path, namespace: str, manifest: dict, files: dict) -> Path:
    """Reserve a unique run dir; partial/failed sync never counts as completion."""
    if manifest.get('status') not in ('passed', 'blocked', 'failed', 'unavailable'):
        raise ContractError('Explicit terminal diagnostic status required')
    for key in ('kind', 'run_id', 'code_commit', 'protocol_fingerprint', 'dataset_fingerprint',
                'model_fingerprint', 'seeds', 'software_versions', 'upstream_artifacts'):
        if key not in manifest:
            raise ContractError('Run manifest missing: ' + key)
    require_sha256(manifest['protocol_fingerprint'])
    if manifest['status'] == 'passed':
        if not isinstance(manifest['code_commit'], str) or not re.fullmatch('[0-9a-f]{40}', manifest['code_commit']):
            raise ContractError('Passed artifacts require a code commit')
        require_sha256(manifest['dataset_fingerprint'])
        if manifest['kind'] == 'stage1_smoke':
            require_sha256(manifest['model_fingerprint'])
    if not files or MARKER in files:
        raise ContractError('Nonempty payload required; completion marker is reserved')
    for name, data in files.items():
        relative_path(name)
        if not isinstance(data, bytes):
            raise ContractError('Artifact payloads must be bytes')
    target = within(root, namespace)
    target.mkdir(parents=True, exist_ok=False)
    checksums = {}
    for name, data in files.items():
        p = within(target, name)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('xb') as f:
            f.write(data)
        checksums[name] = hashlib.sha256(data).hexdigest()
        if hashlib.sha256(p.read_bytes()).hexdigest() != checksums[name]:
            raise ContractError('Written artifact failed verification')
    complete = dict(manifest, schema_version=SCHEMA_VERSION, files=checksums)
    with (target / MARKER).open('xb') as f:
        f.write(canonical_bytes(complete) + b'\n')
    verify(target, required_status=None)
    return target

def verify(path: Path, required_status='passed', expected=None) -> dict:
    try:
        manifest = json.loads((path / MARKER).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ContractError('Artifact incomplete or completion marker unreadable') from exc
    if not isinstance(manifest, dict) or manifest.get('schema_version') != SCHEMA_VERSION or not isinstance(manifest.get('files'), dict) or not manifest['files']:
        raise ContractError('Unrecognized/empty completion manifest')
    if required_status is not None and manifest.get('status') != required_status:
        raise ContractError('Artifact does not have required passed status')
    for key, value in (expected or {}).items():
        if manifest.get(key) != value:
            raise ContractError('Incompatible artifact: ' + key)
    for name, digest in manifest['files'].items():
        require_sha256(digest)
        p = within(path, name)
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise ContractError('Missing, partially synced or corrupt artifact: ' + name)
    return manifest
