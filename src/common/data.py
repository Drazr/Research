"""Original-release preparation; no translations or mention proxies."""
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen
import zipfile

RELEASE = 'ad44c380c86cb1240666361caabaf5684507d0ef'
BLOB = '6b99def06a621c570edf074c26741868a4df7353'
SOURCES = {
    'captions.zip': (f'https://raw.githubusercontent.com/google/crossmodal-3600/{RELEASE}/web-data/captions.zip', '7964f969d39319a96769499b199042015754f8d5607b6aad2d1c2c2e55f5e562'),
    'images.tgz': ('https://open-images-dataset.s3.amazonaws.com/crossmodal-3600/images.tgz', 'ba57c58d7f7a35d95315157b609ca280d55ba10a107026f67d2e98371e221188'),
    'image_attributions.csv': (f'https://raw.githubusercontent.com/google/crossmodal-3600/{RELEASE}/web-data/image_attributions.csv', '288de413d25e9c70dadfd42ee59256aa1082673e615d0eda02c3875fe2b36ecd'),
    'xm3600_annotation.csv': (f'https://api.github.com/repos/uriberger/cross_lingual_diff_in_descs/git/blobs/{BLOB}', 'acb83d20b8930e52865c673d715a55484c5a528290dd2e8296dfd7cd725e2a09'),
}

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def canonical_id(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-fA-F]{1,16}', value):
        raise ValueError('Image ID must be a lossless hexadecimal string')
    return value.lower().zfill(16)

def fetch_sources(root):
    import uuid
    folder = root / 'cache/shared/source-downloads'
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, (url, digest) in SOURCES.items():
        p = folder / name
        if not p.exists():
            temp = folder / (name + '.' + uuid.uuid4().hex + '.partial')
            try:
                with urlopen(Request(url, headers={'User-Agent':'Research-stage1/1.0'}), timeout=60) as response:
                    if name == 'xm3600_annotation.csv':
                        o = json.load(response)
                        if o['encoding'] != 'base64': raise ValueError('Expected GitHub blob base64')
                        raw = base64.b64decode(o['content'])
                        if hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() != BLOB:
                            raise ValueError('Annotation Git blob mismatch')
                        temp.write_bytes(raw)
                    else:
                        with temp.open('xb') as out:
                            for b in iter(lambda: response.read(1024*1024), b''): out.write(b)
                if sha256(temp) != digest: raise ValueError('Source checksum mismatch: '+name)
                # Hard-link publication is exclusive: another producer cannot be overwritten.
                import os
                try: os.link(temp, p)
                except FileExistsError: pass
            finally:
                if temp.exists(): temp.unlink()
        if sha256(p) != digest: raise ValueError('Source checksum mismatch: '+name)
        paths[name] = p
    return paths

def read_original_captions(path, languages):
    by_id = {}
    with zipfile.ZipFile(path) as z:
        raw = z.read('captions.jsonl')
    for line in raw.splitlines():
        row = json.loads(line)
        iid = canonical_id(row['image/key'])
        if iid in by_id: raise ValueError('Duplicate caption image: '+iid)
        by_id[iid] = {}
        for lang in languages:
            texts = row.get(lang, {}).get('caption', [])
            if not isinstance(texts, list) or any(not isinstance(t,str) or not t.strip() for t in texts):
                raise ValueError('Malformed original caption list')
            by_id[iid][lang] = texts
    return by_id

def read_targets(path, protocol):
    rows = list(csv.reader(io.StringIO(Path(path).read_text(encoding='utf-8-sig'))))
    if rows[0][1:] != protocol['category_order']: raise ValueError('Annotation category order mismatch')
    targets = {}
    allowed = {(x['image_id'], x['category']): x['value'] for x in protocol['invalid_cells']}
    for row in rows[1:]:
        if len(row) != 26: raise ValueError('Malformed annotation row')
        iid = canonical_id(row[0])
        if iid in targets: raise ValueError('Duplicate annotation ID')
        values = []
        for cat, value in zip(protocol['category_order'],row[1:]):
            if value == '': values.append(0)
            elif value == '1': values.append(1)
            elif allowed.get((iid,cat)) == value: values.append(None)
            else: raise ValueError('Unexpected nonbinary annotation cell')
        targets[iid] = values
    return targets

# The source helpers above are followed by this contract-dependent implementation.
from dataclasses import asdict
import platform
import tarfile
from .contracts import (SCHEMA_VERSION, LANGUAGES, ContractError, ModelInput,
                        TargetRecord, DatasetBundle, canonical_bytes, fingerprint, select_caption)
from .storage import publish, verify, within, new_run_id

REPO = Path(__file__).resolve().parents[2]

def json_bytes(value):
    return canonical_bytes(value) + b'\n'

def jsonl(rows):
    return b''.join(json_bytes(row) for row in rows)

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]

def run_metadata(kind, protocol, code_commit, dataset=None, scope='local-development'):
    if not re.fullmatch('[0-9a-f]{40}', code_commit or ''):
        raise ContractError('Supply the real 40-character --code-commit')
    return dict(kind=kind, status='passed', run_id=new_run_id(), code_commit=code_commit,
                protocol_fingerprint=fingerprint(protocol), dataset_fingerprint=dataset,
                model_fingerprint=None, seeds={'caption_draw':0,'split':protocol['split_seed']},
                software_versions={'python':platform.python_version()}, upstream_artifacts=[],
                storage_scope=scope)

def splits_and_support(targets, protocol):
    audit = json.loads((REPO/'data/manifests/g9q6_split_planning_audit.json').read_text())
    ordered = sorted(targets, key=lambda i:hashlib.sha256((protocol['split_seed']+'|'+i).encode()).digest())
    if len(ordered) != sum(protocol['split_sizes'].values()):
        raise ContractError('Incomplete population: request coordinator replacement protocol before publishing')
    partitions, counts, offset = {}, {}, 0
    for split, n in protocol['split_sizes'].items():
        ids = ordered[offset:offset+n]; offset += n
        if ids != audit['partitions'][split]: raise ContractError('Frozen partition mismatch')
        partitions[split] = ids
        counts[split] = {}
        for k, cat in enumerate(protocol['category_order']):
            vals = [targets[i][k] for i in ids]
            counts[split][cat] = dict(positive=vals.count(1),negative=vals.count(0),unknown=vals.count(None))
        if counts[split] != audit['counts'][split]: raise ContractError('Frozen support mismatch')
    primary = [c for c in protocol['category_order'] if all(
        min(counts[s][c]['positive'], counts[s][c]['negative']) >= n
        for s,n in protocol['minimum_positive_and_negative'].items())]
    if primary != protocol['primary_categories']: raise ContractError('Primary cohort mismatch')
    return partitions, counts

def prepare(root, protocol, code_commit, scope='local-development'):
    sources = fetch_sources(root)
    targets = read_targets(sources['xm3600_annotation.csv'], protocol)
    captions = read_original_captions(sources['captions.zip'], LANGUAGES)
    partitions, counts = splits_and_support(targets, protocol)
    split_by_id = {i:s for s,ids in partitions.items() for i in ids}
    identity_sources = {n:{'url':u,'sha256':h} for n,(u,h) in SOURCES.items()}
    identity_sources['caption_revision'] = RELEASE
    identity_sources['annotation_git_blob'] = BLOB
    source_fp = fingerprint(identity_sources)
    namespace = 'datasets/xm3600/'+source_fp
    source_folder = within(root,namespace)
    if not source_folder.exists():
        payload = {n:p.read_bytes() for n,p in sources.items() if n != 'images.tgz'}
        seen = set()
        with tarfile.open(sources['images.tgz']) as archive:
            for member in archive:
                if not member.isfile() or not re.fullmatch('[0-9a-f]{16}\.jpg',member.name):
                    raise ContractError('Unexpected image archive member')
                iid = member.name[:-4]
                if iid in seen: raise ContractError('Duplicate image archive member')
                seen.add(iid)
                payload['images/'+member.name] = archive.extractfile(member).read()
        if seen != set(targets): raise ContractError('Image/annotation join mismatch')
        meta = run_metadata('xm3600_original_sources',protocol,code_commit,source_fp,scope)
        publish(root,namespace,meta,payload)
    verify(source_folder, expected={'protocol_fingerprint':fingerprint(protocol)})
    if set(captions) != set(targets) or any(not captions[i][l] for i in targets for l in LANGUAGES):
        raise ContractError('Incomplete four-language intersection; coordinator protocol required')
    records, views, duplicates = [], {l:[] for l in LANGUAGES}, []
    for iid in sorted(targets):
        image_path = namespace+'/images/'+iid+'.jpg'
        digest = sha256(within(root,image_path))
        for lang in LANGUAGES:
            candidates = []
            texts = captions[iid][lang]
            if len(set(texts)) != len(texts): duplicates.append({'image_id':iid,'language':lang})
            for index,text in enumerate(texts):
                cid = f'xm3600:{RELEASE}:{iid}:{lang}:{index}'
                r = ModelInput(iid,image_path,digest,lang,cid,text,split_by_id[iid])
                candidates.append(r)
                views[lang].append(dict(asdict(r),source_index=index,source_field=f'{lang}.caption[{index}]'))
            records.append(asdict(select_caption(candidates,0)))
    record_bytes = jsonl(records)
    target_bytes = jsonl(dict(image_id=i,split=split_by_id[i],values=targets[i]) for i in sorted(targets))
    split_bytes = json_bytes(partitions)
    identity = dict(sources=identity_sources,category_order=protocol['category_order'],
                    split_manifest_sha256=hashlib.sha256(split_bytes).hexdigest(),
                    caption_recipe='select_caption stage1.v1; draw_key=0; caption_id=xm3600:release:image_id:language:zero_based_source_index',
                    records_sha256=hashlib.sha256(record_bytes).hexdigest(),
                    targets_sha256=hashlib.sha256(target_bytes).hexdigest())
    dfp = fingerprint(identity)
    dataset = dict(identity,schema_version=SCHEMA_VERSION,dataset_fingerprint=dfp,
                   protocol_fingerprint=fingerprint(protocol),source_artifact=namespace)
    report = dict(status='passed',dataset_fingerprint=dfp,protocol_fingerprint=fingerprint(protocol),
                  checks={'source_hashes':True,'full_population':True,'frozen_splits_supports':True,
                          'original_caption_fields':True,'draw_0':True,'unknown_mask':True,
                          'image_checksums_and_decode':True},
                  counts={'images':len(targets),'records':len(records),'captions':{l:len(v) for l,v in views.items()}},
                  exclusions=[],duplicate_caption_groups=duplicates,support=counts,storage_scope=scope,
                  invalid_cells=protocol['invalid_cells'],
                  semantic_verification=False,confirm_model_processing=False)
    files = {'records.jsonl':record_bytes,'targets.jsonl':target_bytes,'dataset.json':json_bytes(dataset),
             'splits.json':split_bytes,'validation.json':json_bytes(report)}
    files.update({'captions/'+l+'.jsonl':jsonl(v) for l,v in views.items()})
    folder = within(root,'shared/'+fingerprint(protocol)+'/manifests/'+dfp)
    if folder.exists():
        load_dataset(folder/'_COMPLETE.json',root,protocol)
        return folder/'_COMPLETE.json'
    meta = run_metadata('stage1_dataset',protocol,code_commit,dfp,scope)
    meta['upstream_artifacts'] = [namespace]
    # Validate candidate metadata and every image before publishing a passed artifact.
    _validate_payload(dataset,report,records,read_targets(sources['xm3600_annotation.csv'],protocol),
                      views,partitions,root,protocol,record_bytes,target_bytes)
    publish(root,folder.relative_to(root).as_posix(),meta,files)
    load_dataset(folder/'_COMPLETE.json',root,protocol)
    return folder/'_COMPLETE.json'

def _validate_payload(dataset,report,records,targets,views,partitions,root,protocol,record_bytes,target_bytes):
    identity = {k:dataset[k] for k in ('sources','category_order','split_manifest_sha256','caption_recipe','records_sha256','targets_sha256')}
    if fingerprint(identity) != dataset['dataset_fingerprint']: raise ContractError('Dataset identity mismatch')
    if hashlib.sha256(record_bytes).hexdigest() != dataset['records_sha256'] or hashlib.sha256(target_bytes).hexdigest() != dataset['targets_sha256']:
        raise ContractError('Selected record/target hash mismatch')
    expected_splits, counts = splits_and_support(targets,protocol)
    if partitions != expected_splits or hashlib.sha256(json_bytes(partitions)).hexdigest() != dataset['split_manifest_sha256']:
        raise ContractError('Split manifest mismatch')
    source_dir = within(root,dataset['source_artifact'])
    verify(source_dir,expected={'protocol_fingerprint':fingerprint(protocol)})
    for name, (_, digest) in SOURCES.items():
        if name != 'images.tgz' and sha256(source_dir/name) != digest:
            raise ContractError('Pinned source checksum mismatch: '+name)
    if dataset['sources'] != dict({n:{'url':u,'sha256':h} for n,(u,h) in SOURCES.items()},caption_revision=RELEASE,annotation_git_blob=BLOB):
        raise ContractError('Unrecognized source provenance')
    if read_targets(source_dir/'xm3600_annotation.csv',protocol) != targets: raise ContractError('Targets differ from pinned annotation')
    originals = read_original_captions(source_dir/'captions.zip',LANGUAGES)
    mapping = {i:s for s,ids in partitions.items() for i in ids}
    expected_records = []
    from PIL import Image
    for iid in sorted(targets):
        path = dataset['source_artifact']+'/images/'+iid+'.jpg'
        image = within(root,path)
        digest = sha256(image)
        with Image.open(image) as im: im.load()
        for lang in LANGUAGES:
            candidates = [ModelInput(iid,path,digest,lang,f'xm3600:{RELEASE}:{iid}:{lang}:{j}',text,mapping[iid])
                          for j,text in enumerate(originals[iid][lang])]
            expected_records.append(asdict(select_caption(candidates,0)))
    if records != expected_records: raise ContractError('Selected text, caption ID, image checksum or split differs from source')
    selected_by_key = {(r['image_id'],r['language']):r for r in expected_records}
    for lang in LANGUAGES:
        expected_view = []
        for iid in sorted(targets):
            r = selected_by_key[(iid,lang)]
            for j,text in enumerate(originals[iid][lang]):
                expected_view.append(dict(r,caption_id=f'xm3600:{RELEASE}:{iid}:{lang}:{j}',caption_text=text,
                                          source_index=j,source_field=f'{lang}.caption[{j}]'))
        if views[lang] != expected_view: raise ContractError('Original caption view mismatch')
    if report.get('support') != counts or report.get('exclusions') != []: raise ContractError('Validation support/exclusion mismatch')

def load_dataset(manifest_path, phase1_root, protocol):
    root = Path(phase1_root).resolve()
    path = Path(manifest_path).resolve()
    if path.name != '_COMPLETE.json': raise ContractError('Expected dataset _COMPLETE.json')
    try: path.relative_to(root)
    except ValueError as exc: raise ContractError('Dataset artifact must be under PHASE1_ROOT') from exc
    folder = path.parent
    manifest = verify(folder,expected={'protocol_fingerprint':fingerprint(protocol),'kind':'stage1_dataset'})
    dataset = json.loads((folder/'dataset.json').read_text(encoding='utf-8'))
    report = json.loads((folder/'validation.json').read_text(encoding='utf-8'))
    records = read_jsonl(folder/'records.jsonl')
    rows = read_jsonl(folder/'targets.jsonl')
    target_objects = {}
    for row in rows:
        t = TargetRecord(row['image_id'],row['split'],tuple(row['values']))
        if t.image_id in target_objects: raise ContractError('Duplicate target')
        target_objects[t.image_id] = t
    bundle = DatasetBundle([ModelInput(**r) for r in records],target_objects,dataset['category_order'],
                           dataset['dataset_fingerprint'],dataset['protocol_fingerprint'],report)
    bundle.validate(protocol)
    if manifest['dataset_fingerprint'] != bundle.dataset_fingerprint: raise ContractError('Completion identity mismatch')
    _validate_payload(dataset,report,records,{i:list(t.values) for i,t in target_objects.items()},
                      {l:read_jsonl(folder/'captions'/ (l+'.jsonl')) for l in LANGUAGES},
                      json.loads((folder/'splits.json').read_text()),root,protocol,
                      (folder/'records.jsonl').read_bytes(),(folder/'targets.jsonl').read_bytes())
    return bundle
