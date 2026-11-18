"""CPU coordinator entry points for Stage 4 preparation and acceptance."""

import argparse
import json
from pathlib import Path
import subprocess

from src.common.contracts import ContractError
from src.common.storage import phase1_root
from src.stage2.schema import REPO, check
from src.stage3.operations import current_producer
from .analysis import analysis_units
from .plan import load_stage4_plan, prepare_stage4
from .validation import independent_readback, validate_stage4


def emit(event):
    print(json.dumps(event), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("prepare", "validate", "readback"))
    parser.add_argument("--phase1-root", required=True)
    parser.add_argument("--registry")
    parser.add_argument("--manifest")
    parser.add_argument("--inventory")
    parser.add_argument("--readback-ref")
    parser.add_argument("--storage-scope", default="persistent-drive",
                        choices=("persistent-drive", "local-development"))
    args = parser.parse_args()
    root = phase1_root(args.phase1_root)
    protocol = json.loads((REPO / "configs/protocol.json").read_text(encoding="utf-8"))
    if args.task == "readback":
        check(args.readback_ref is not None, "An explicit readback reference is required")
        ref = independent_readback(root, protocol, json.loads(args.readback_ref), progress=emit)
        emit({"event": "stage4_independent_readback", "status": "passed", "reference": ref})
        return
    producer = current_producer(args.storage_scope)
    emit({"event": "stage4_task_start", "task": args.task, "code_commit": producer["code_commit"]})
    if args.task == "prepare":
        check(args.registry is not None, "An accepted Stage 3 registry is required")
        plan = prepare_stage4(root, protocol, args.registry, producer)
        emit({"manifest": str(root / plan.reference["path"]), "reference": dict(plan.reference),
              "prediction_blocks": len(plan["blocks"]), "analysis_units": len(analysis_units(plan)),
              "stage4_complete": False})
    else:
        check(args.manifest and args.inventory, "Explicit plan and worker inventory required")
        plan = load_stage4_plan(root, protocol, args.manifest)
        inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
        check(set(inventory) == {"predictions", "analyses"}, "Wrong Stage 4 inventory fields")
        ref = validate_stage4(plan, inventory["predictions"], inventory["analyses"], producer, progress=emit)
        emit({"manifest": str(root / ref["path"]), "reference": ref, "status": "passed", "stage4_complete": True})


if __name__ == "__main__":
    try:
        main()
    except (ContractError, OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        emit({"status": "blocked", "blockers": [str(exc)], "stage4_complete": False})
        raise SystemExit(2)
