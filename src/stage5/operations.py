"""Stage 5 CPU coordinator for synthesis and final report publication."""

import argparse
import json
from pathlib import Path

# Stage 3 controls native library initialization order in the pinned runtime.
from src.stage3.operations import current_producer
from src.common.contracts import ContractError
from src.common.storage import phase1_root
from src.stage2.schema import REPO
from .coordinator import publish_final_report, publish_synthesis
from .interfaces import open_release


def emit(event):
    print(json.dumps(event), flush=True)


def read_json(path):
    return Path(path).read_text(encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-root", required=True)
    parser.add_argument("--release", required=True, help="JSON object with path and sha256")
    parser.add_argument("--storage-scope", choices=("persistent-drive", "local-development"),
                        default="persistent-drive")
    tasks = parser.add_subparsers(dest="task", required=True)
    synthesis = tasks.add_parser("synthesize")
    synthesis.add_argument("--reports", required=True, help="JSON file keyed by qwen, gemma and e03")
    final = tasks.add_parser("finalize")
    final.add_argument("--synthesis", required=True, help="JSON object with path and sha256")
    final.add_argument("--review", required=True, help="Reviewed decision register JSON file")
    args = parser.parse_args()
    producer = current_producer(args.storage_scope)
    emit({"event": "stage5_task_start", "task": args.task, "code_commit": producer["code_commit"]})
    run = open_release(phase1_root(args.phase1_root), json.loads(read_json(REPO / "configs/protocol.json")),
                       json.loads(args.release))
    if args.task == "synthesize":
        ref = publish_synthesis(run, json.loads(read_json(args.reports)), producer, progress=emit)
    else:
        ref = publish_final_report(run, json.loads(args.synthesis), json.loads(read_json(args.review)), producer)
    emit({"status": "passed", "reference": ref, "manifest": str(run.plan.root / ref["path"]),
          "stage5_complete": args.task == "finalize" and args.storage_scope == "persistent-drive"})


if __name__ == "__main__":
    try:
        main()
    except (ContractError, OSError, ValueError, TypeError, KeyError) as exc:
        emit({"status": "blocked", "blockers": [str(exc)], "stage5_complete": False})
        raise SystemExit(2) from exc
