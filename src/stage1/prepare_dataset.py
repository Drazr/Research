"""Prepare pinned original XM3600 data once, without model execution."""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from src.common.data import prepare
from src.common.storage import phase1_root

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase1-root')
    p.add_argument('--protocol',type=Path,default=REPO/'configs/protocol.json')
    p.add_argument('--code-commit',required=True)
    p.add_argument('--storage-scope',choices=['local-development','persistent-drive'],default='local-development')
    args = p.parse_args()
    try:
        manifest = prepare(phase1_root(args.phase1_root),json.loads(args.protocol.read_text()),args.code_commit,args.storage_scope)
        print(json.dumps({'status':'passed','dataset_manifest':str(manifest),'storage_scope':args.storage_scope}))
    except (ValueError,OSError,ImportError) as exc:
        print(json.dumps({'status':'failed','error':str(exc)}),file=sys.stderr)
        return 1
    return 0

if __name__ == '__main__': sys.exit(main())
