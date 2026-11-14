"""Recheck completed data, including original-source joins and deterministic reload."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from src.common.data import load_dataset, run_metadata, json_bytes
from src.common.contracts import fingerprint
from src.common.storage import phase1_root, publish

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase1-root')
    p.add_argument('--dataset-manifest',type=Path,required=True)
    p.add_argument('--protocol',type=Path,default=REPO/'configs/protocol.json')
    p.add_argument('--code-commit',required=True)
    p.add_argument('--storage-scope',choices=['local-development','persistent-drive'],default='local-development')
    args = p.parse_args()
    root = phase1_root(args.phase1_root)
    protocol = json.loads(args.protocol.read_text())
    meta = run_metadata('stage1_data_validation',protocol,args.code_commit,scope=args.storage_scope)
    meta['upstream_artifacts'] = [str(args.dataset_manifest.resolve())]
    try:
        bundle = load_dataset(args.dataset_manifest,root,protocol)
        again = load_dataset(args.dataset_manifest,root,protocol)
        if [asdict(r) for r in bundle.records] != [asdict(r) for r in again.records] or bundle.targets != again.targets:
            raise ValueError('Deterministic reload failed')
        meta['dataset_fingerprint'] = bundle.dataset_fingerprint
        report = dict(bundle.validation,deterministic_reload=True,code_commit=args.code_commit)
    except (ValueError,OSError,ImportError) as exc:
        meta['status'] = 'failed'
        report = {'status':'failed','error':str(exc)}
    folder = publish(root,'shared/'+fingerprint(protocol)+'/validation/'+meta['run_id'],meta,{'report.json':json_bytes(report)})
    print(json.dumps({'status':meta['status'],'report':str(folder/'report.json')}))
    return 0 if meta['status']=='passed' else 1

if __name__ == '__main__': sys.exit(main())
