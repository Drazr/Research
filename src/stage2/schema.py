"""Shared wire definitions. No scientific or adapter policy is changed here."""
import json
import re
from pathlib import Path
from types import MappingProxyType
from collections.abc import Mapping
from functools import wraps
from dataclasses import dataclass

from src.common.contracts import ContractError, canonical_bytes, fingerprint
from src.common.storage import MARKER, file_sha256, publish, verify, within

REPO = Path(__file__).resolve().parents[2]
SPLITS = ('FIT', 'DEV-S', 'DEV-C', 'SCREEN')
CONDITIONS = ('image_caption', 'caption_only', 'image_only')
ROLES = {'source_fit': 'FIT', 'source_select': 'DEV-S',
         'calibration': 'DEV-C', 'exploratory_score': 'SCREEN'}
PLAN_SCHEMA = 'stage2.plan.v1'
FEATURE_SCHEMA = 'stage2.features.v1'
TEXT_SCHEMA = 'stage2.text.v1'


def check(ok, message):
    if not ok:
        raise ContractError(message)


def contract(function):
    """Malformed wire shapes fail as contract errors, never accidental coercions."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise ContractError(function.__name__ + ': malformed or missing contract field: ' + str(exc)) from exc
    return wrapped


def plain(value):
    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def frozen(value):
    if isinstance(value, dict):
        return MappingProxyType({k: frozen(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(frozen(v) for v in value)
    return value


@dataclass(frozen=True)
class VerifiedPlan(Mapping):
    """A recursively immutable plan that has passed the full upstream verifier."""
    _body: Mapping
    def __init__(self, body):
        object.__setattr__(self, '_body', frozen(body))
    def __getitem__(self, key):
        return self._body[key]
    def __iter__(self):
        return iter(self._body)
    def __len__(self):
        return len(self._body)


def json_bytes(value):
    return canonical_bytes(plain(value)) + b'\n'


def jsonl(rows):
    return b''.join(json_bytes(r) for r in rows)


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ContractError('Unreadable JSON: ' + str(path)) from exc


def read_rows(path):
    try:
        return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines()]
    except (OSError, ValueError) as exc:
        raise ContractError('Unreadable row table: ' + str(path)) from exc


def component(value):
    check(isinstance(value, str) and re.fullmatch('[A-Za-z0-9_-]+', value),
          'Expected a single portable namespace component')
    return value


def marker_path(path, root):
    root = Path(root).resolve()
    path = Path(path)
    path = path.resolve() if path.is_absolute() else within(root, path.as_posix())
    check(path.name == MARKER, 'Expected explicit _COMPLETE.json')
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ContractError('Artifact outside PHASE1_ROOT') from exc
    return path


def reference(path, root):
    path = marker_path(path, root)
    verify(path.parent)
    return {'path': path.relative_to(Path(root).resolve()).as_posix(),
            'sha256': file_sha256(path)}


def artifact(path, root, protocol, kind=None):
    path = marker_path(path, root)
    expected = {'protocol_fingerprint': fingerprint(protocol)}
    if kind:
        expected['kind'] = kind
    return path, verify(path.parent, expected=expected)


def dereference(ref, root, protocol, kind=None):
    path, meta = artifact(ref['path'], root, protocol, kind)
    check(file_sha256(path) == ref['sha256'], 'Upstream completion manifest changed')
    return path, meta


def specification():
    return {'version': 'stage2-plan.v1', 'review_revision': 2,
            'sha256': file_sha256(REPO / 'docs/stage2_contracts.md')}


def source_hashes():
    return {p.relative_to(REPO).as_posix(): file_sha256(p)
            for p in sorted((REPO / 'src/stage2').glob('*.py'))}


def metadata(kind, protocol, dataset, producer, model=None):
    for key in ('run_id', 'code_commit', 'software_versions', 'storage_scope'):
        check(key in producer, 'Producer metadata missing: ' + key)
    component(producer['run_id'])
    check(producer['storage_scope'] in ('local-development', 'persistent-drive'), 'Unknown storage scope')
    check(bool(producer['software_versions']), 'Actual software versions required')
    return dict(kind=kind, status='passed', run_id=producer['run_id'],
                code_commit=producer['code_commit'], protocol_fingerprint=fingerprint(protocol),
                dataset_fingerprint=dataset, model_fingerprint=model,
                seeds={'caption_draw': 0, 'split': protocol['split_seed']},
                software_versions=producer['software_versions'],
                storage_scope=producer['storage_scope'], upstream_artifacts=[],
                source_files_sha256=source_hashes())


def publish_json(root, namespace, meta, name, payload, files=None):
    files = dict(files or {})
    check(name not in files, 'Reserved metadata filename')
    files[name] = json_bytes(payload)
    folder = publish(Path(root).resolve(), namespace, meta, files)
    return folder / MARKER


def authorize(role, split):
    check(role in ROLES and ROLES[role] == split, 'Reader role does not authorize this split')
