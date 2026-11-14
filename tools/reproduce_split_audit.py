import argparse
import base64
import os
import csv
import hashlib
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen

parser = argparse.ArgumentParser(description='Reproduce the existing pinned annotation/split audit; no model runs.')
parser.add_argument('--phase1-root', default=os.environ.get('PHASE1_ROOT'))
parser.add_argument('--annotation-csv', type=Path)
args = parser.parse_args()
if not args.phase1_root:
    parser.error('Set PHASE1_ROOT or pass --phase1-root; persistent outputs belong in the phase1 folder.')
ROOT = Path(args.phase1_root).expanduser().resolve()
blob_sha = '6b99def06a621c570edf074c26741868a4df7353'
url = 'https://api.github.com/repos/uriberger/cross_lingual_diff_in_descs/git/blobs/' + blob_sha
folder = ROOT / 'cache' / 'shared' / 'annotation' / blob_sha
folder.mkdir(parents=True, exist_ok=True)
csv_path = folder / 'xm3600_annotation.csv'
if args.annotation_csv:
    raw = args.annotation_csv.read_bytes()
elif csv_path.exists():
    raw = csv_path.read_bytes()
else:
    with urlopen(Request(url, headers={'User-Agent': 'XM3600-preliminary-design/1.0'}), timeout=25) as response:
        result = json.load(response)
    assert result['encoding'] == 'base64'
    raw = base64.b64decode(result['content'])
assert hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == blob_sha
csv_path.write_bytes(raw)
rows = list(csv.reader(io.StringIO(raw.decode('utf-8-sig'))))
categories = rows[0][1:]
assert len(categories) == 25 and len(rows) == 3601
by_id = {row[0].lower().zfill(16): row[1:] for row in rows[1:]}
assert len(by_id) == 3600
seed = 'G9Q6-first-set-20261114-v1'
ordered = sorted(by_id, key=lambda iid: hashlib.sha256((seed + '|' + iid).encode()).digest())
sizes = {'FIT':2160, 'DEV-S':180, 'DEV-C':180, 'SCREEN':360, 'CONFIRM':720}
partitions = {}
pos = 0
for name, size in sizes.items():
    partitions[name] = ordered[pos:pos+size]
    pos += size
assert pos == 3600
counts = {}
invalid = []
for name, ids in partitions.items():
    counts[name] = {}
    for k, category in enumerate(categories):
        values = [by_id[i][k] for i in ids]
        counts[name][category] = {'positive':values.count('1'), 'negative':values.count(''), 'unknown':sum(v not in {'1',''} for v in values)}
        for iid in ids:
            if by_id[iid][k] not in {'1',''}:
                invalid.append({'partition':name,'image_id':iid,'category':category,'value':by_id[iid][k]})
minimum = {'FIT':20, 'DEV-S':10, 'DEV-C':10, 'SCREEN':20}
eligible = [cat for cat in categories if all(counts[part][cat]['positive'] >= n and counts[part][cat]['negative'] >= n for part,n in minimum.items())]
audit = {
    'date':'2026-11-14',
    'kind':'Automatic design-time partition/support audit only; no model results or human validation.',
    'csv_blob_sha':blob_sha,
    'source':url,
    'seed':seed,
    'assignment':'Sort canonical hexadecimal image IDs by SHA256(seed + | + image_id), then allocate the fixed contiguous block sizes.',
    'sizes':sizes,
    'partitions':partitions,
    'categories':categories,
    'counts':counts,
    'minimum_positive_and_negative':minimum,
    'primary_categories':eligible,
    'primary_count':len(eligible),
    'invalid_cells':invalid,
    'confirmatory_features_or_predictions_accessed':False,
}
output = ROOT / 'shared' / 'design-audit' / 'manifests'
output.mkdir(parents=True, exist_ok=True)
(output / 'g9q6_split_planning_audit.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k:audit[k] for k in ['kind','sizes','primary_categories','primary_count','invalid_cells']}, indent=2))
