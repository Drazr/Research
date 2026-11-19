"""CPU coordinator CLI entry points for Phase 2 Stage 1 preparation and verification."""

import argparse
import json
from pathlib import Path
import subprocess

from src.common.contracts import ContractError
from src.common.storage import phase1_root
from src.phase2.interfaces import (
    PHASE1_STAGE4_RELEASE,
    REPO,
    check,
    current_producer,
    plain,
)
from .plan import load_phase2_plan, prepare_phase2, verify_stage1_bundle


def emit(event):
    print(json.dumps(event), flush=True)


def _parse_ref_arg(value):
    if value is None:
        return None
    text = value.strip()
    if text.startswith("{"):
        parsed = json.loads(text)
        check(isinstance(parsed, dict), "JSON reference argument must decode to an object")
        return parsed
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("prepare-phase2", "readback", "verify-stage1-bundle"))
    parser.add_argument("--phase1-root", required=True)
    parser.add_argument("--protocol", type=Path, default=REPO / "configs/protocol.json")
    parser.add_argument("--stage4-release", help="Explicit Stage 4 release {'path', 'sha256'} JSON or path")
    parser.add_argument("--manifest", help="Explicit phase2_plan {'path', 'sha256'} JSON or path")
    parser.add_argument("--readback-ref", help="Alias for --manifest in readback task")
    parser.add_argument("--plan", help="Explicit phase2_plan {'path', 'sha256'} JSON or path")
    parser.add_argument("--e07-audit", help="Explicit phase2_e07_audit {'path', 'sha256'} JSON or path")
    parser.add_argument("--e17-auth", help="Explicit phase2_e17_auth {'path', 'sha256'} JSON or path")
    parser.add_argument(
        "--storage-scope",
        default="persistent-drive",
        choices=("persistent-drive", "local-development"),
    )
    args = parser.parse_args(argv)

    root = phase1_root(args.phase1_root)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))

    if args.task == "readback":
        target = args.manifest or args.readback_ref
        check(target is not None, "An explicit phase2_plan --manifest or --readback-ref is required")
        plan = load_phase2_plan(root, protocol, _parse_ref_arg(target))
        emit({
            "event": "phase2_plan_readback",
            "status": "passed",
            "reference": plain(plan.reference),
            "manifest": str(root / plan.reference["path"]),
            "plan_identity": plan["plan_identity"],
            "lineage_identity": plan["lineage_identity"],
            "confirm_excluded": plan["confirm_excluded"],
            "e07_audit": plain(plan["e07_audit"]),
            "e17_auth": plain(plan["e17_auth"]),
        })
        return 0

    if args.task == "verify-stage1-bundle":
        plan_arg = args.plan or args.manifest
        check(plan_arg is not None, "Explicit --plan is required for verify-stage1-bundle")
        check(args.e07_audit is not None, "Explicit --e07-audit is required for verify-stage1-bundle")
        check(args.e17_auth is not None, "Explicit --e17-auth is required for verify-stage1-bundle")
        summary = verify_stage1_bundle(
            root,
            protocol,
            _parse_ref_arg(plan_arg),
            _parse_ref_arg(args.e07_audit),
            _parse_ref_arg(args.e17_auth),
        )
        emit(dict(plain(summary), event="phase2_stage1_bundle_verified"))
        return 0

    # prepare-phase2
    producer = current_producer(args.storage_scope)
    emit({
        "event": "phase2_task_start",
        "task": args.task,
        "code_commit": producer["code_commit"],
        "storage_scope": producer["storage_scope"],
    })
    stage4_ref = _parse_ref_arg(args.stage4_release) if args.stage4_release else plain(PHASE1_STAGE4_RELEASE)
    plan = prepare_phase2(
        root,
        protocol,
        stage4_ref,
        producer,
        e07_audit_ref=_parse_ref_arg(args.e07_audit),
        e17_auth_ref=_parse_ref_arg(args.e17_auth),
        progress=emit,
    )
    emit({
        "event": "phase2_plan_published",
        "status": "passed",
        "manifest": str(root / plan.reference["path"]),
        "reference": plain(plan.reference),
        "plan_identity": plan["plan_identity"],
        "lineage_identity": plan["lineage_identity"],
        "storage_probe": plain(plan["storage_probe"]),
        "e07_audit": plain(plan["e07_audit"]),
        "e17_auth": plain(plan["e17_auth"]),
        "stage1_complete": bool(plan["e07_audit"] is not None and plan["e17_auth"] is not None),
    })
    return 0


def _format_blocker(exc):
    if isinstance(exc, subprocess.CalledProcessError):
        parts = [str(exc)]
        if exc.stdout:
            parts.append(f"child_stdout={exc.stdout.strip()}")
        if exc.stderr:
            parts.append(f"child_stderr={exc.stderr.strip()}")
        return " | ".join(parts)
    return str(exc)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, OSError, ValueError, KeyError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        emit({"status": "blocked", "blockers": [_format_blocker(exc)], "stage1_complete": False})
        raise SystemExit(2)
