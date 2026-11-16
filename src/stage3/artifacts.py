"""Numeric-only, checksum-bound I/O. No Stage 2 recursive verifier imports."""
import io
import json
import hashlib
from collections import OrderedDict, Counter
from pathlib import Path
import numpy as np
from src.common.contracts import ContractError, fingerprint, require_sha256
from src.common.storage import within, publish, new_run_id
from src.stage2.schema import check, plain, frozen


class Store:
    """Private bounded immutable byte snapshots; a fresh Store rechecks disk."""
    def __init__(self, root, protocol, budget=512*1024**2):
        self.root = Path(root).resolve()
        self.protocol = protocol
        self.budget, self.size = budget, 0
        self.cache, self.reads = OrderedDict(), Counter()
        self.dataset = None

    def bytes(self, path, digest):
        require_sha256(digest)
        key = (str(path), digest)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        b = path.read_bytes()
        self.reads[str(path.relative_to(self.root))] += 1
        check(hashlib.sha256(b).hexdigest() == digest, 'Changed/corrupt payload: '+str(path))
        if len(b) <= self.budget:
            while self.cache and self.size+len(b) > self.budget:
                _, old = self.cache.popitem(last=False)
                self.size -= len(old)
            self.cache[key], self.size = b, self.size+len(b)
        return b

    def ref(self, value):
        if isinstance(value, dict) or hasattr(value, 'keys'):
            ref = plain(value)
            check(set(ref) == {'path','sha256'}, 'Explicit manifest reference required')
            require_sha256(ref['sha256'])
            return ref
        p = Path(value)
        p = p.resolve() if p.is_absolute() else within(self.root, p.as_posix())
        rel = p.relative_to(self.root).as_posix()
        check(p.name == '_COMPLETE.json', 'Expected completion manifest')
        return {'path': rel, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}

    def marker(self, ref, kind=None):
        ref = self.ref(ref)
        p = within(self.root, ref['path'])
        check(p.name == '_COMPLETE.json', 'Expected completion manifest')
        m = json.loads(self.bytes(p, ref['sha256']))
        check(m['schema_version'] == 'stage1.v1' and m['status'] == 'passed'
              and m['protocol_fingerprint'] == fingerprint(self.protocol), 'Unaccepted artifact')
        check(kind is None or m['kind'] == kind, 'Wrong artifact kind')
        check(self.dataset is None or m['dataset_fingerprint']==self.dataset,'Cross-dataset artifact')
        check(isinstance(m['files'], dict) and m['files'], 'Empty artifact')
        for name, digest in m['files'].items():
            within(p.parent, name)
            require_sha256(digest)
        return ref, p.parent, m

    def payload(self, ref, name, kind=None):
        _, folder, m = self.marker(ref, kind)
        check(name in m['files'], 'Unlisted payload')
        return self.bytes(within(folder, name), m['files'][name])

    def json(self, ref, name, kind=None):
        return json.loads(self.payload(ref, name, kind))

    def rows(self, ref, name='rows.jsonl'):
        return [json.loads(b) for b in self.payload(ref, name).splitlines()]

    def array(self, ref, name, dtype=None, shape=None):
        a = np.load(io.BytesIO(self.payload(ref, name)), allow_pickle=False)
        check(isinstance(a, np.ndarray) and a.dtype.kind != 'O' and a.flags.c_contiguous,
              'Invalid numeric layout')
        check(dtype is None or a.dtype.str == dtype, 'Wrong numeric dtype')
        check(shape is None or a.shape == tuple(shape), 'Wrong numeric shape')
        check(np.isfinite(a).all(), 'Nonfinite numeric payload')
        a.flags.writeable = False
        return a

    def complete(self, ref, kind=None):
        _, _, m = self.marker(ref, kind)
        for name in m['files']:
            self.payload(ref, name)
        return m


def array_bytes(a):
    stream = io.BytesIO()
    np.save(stream, np.ascontiguousarray(a), allow_pickle=False)
    return stream.getvalue()


def write(plan, namespace, kind, body, arrays, producer):
    """One completed bundle; publication performs independent disk readback."""
    check(producer['software_versions'] == plain(plan['runtime']), 'Producer runtime differs from plan')
    run = new_run_id()
    body = dict(plain(body), schema='stage3.'+kind+'.v1', plan=plain(plan.reference),
                fit_identity=plan['fit_identity'])
    files = {'body.json': __import__('src.common.contracts', fromlist=['canonical_bytes']).canonical_bytes(body)+b'\n'}
    files.update({k+'.npy': array_bytes(v) for k,v in arrays.items()})
    meta = dict(kind='stage3_'+kind, status='passed', run_id=run,
                code_commit=producer['code_commit'], storage_scope=producer['storage_scope'],
                software_versions=producer['software_versions'],
                protocol_fingerprint=fingerprint(plan.protocol), dataset_fingerprint=plan['dataset_fingerprint'],
                model_fingerprint=None, seeds={'g0':[0,1,2]}, upstream_artifacts=[plain(plan.reference)])
    folder = publish(plan.root, namespace+'/'+run, meta, files)
    store = Store(plan.root, plan.protocol)
    ref = store.ref(folder/'_COMPLETE.json')
    store.complete(ref, 'stage3_'+kind)
    return frozen(ref)
