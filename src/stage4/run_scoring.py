"""Stage 4 canonical scoring worker for Qwen, Gemma, and shared E03.

Uses one model selector (`qwen` or `gemma`) for independent CPU runtimes plus a
shared `e03` mode that runs once. Publishes each model's Priority 0 E01
midpoint SCREEN grid and E19 source DEV-C diagonals first, streams exact
completion references and timing/memory telemetry, and resumes strictly from
explicit verified prediction references.
"""

import os

# Keep BLAS/OpenMP single-threaded before importing numerical libraries.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import cProfile
import json
from pathlib import Path
import pstats
import sys
import threading
import time

from src.common.contracts import ContractError
from src.common.storage import phase1_root
from src.stage2.schema import REPO, check, plain
from src.stage4.interfaces import (
    current_producer,
    load_prediction_block,
    load_stage4_plan,
    prediction_index,
    score_and_publish_block,
    scoring_blocks,
)

MODELS = ("qwen", "gemma")
MODES = ("model", "e03")
SCOPES = ("all", "priority", "profile")


def emit(event, **fields):
    print(json.dumps(dict(event=event, **plain(fields)), allow_nan=False), flush=True)


def peak_memory():
    """Actual process high-water RSS on Linux/Colab; None where unavailable."""
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)
    except ImportError:
        return None


def measured(event, operation, report=emit, *, heartbeat_seconds=15.0, **fields):
    """Stream timed heartbeats and cProfile exclusive I/O/hash components."""
    start = time.monotonic()
    report(event + "_start", **fields)
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(heartbeat_seconds):
            report(
                "heartbeat",
                phase=event,
                elapsed_seconds=time.monotonic() - start,
                peak_rss_bytes=peak_memory(),
                **fields,
            )

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    profiler = cProfile.Profile()
    try:
        value = profiler.runcall(operation)
    finally:
        stop.set()
        thread.join()
        elapsed = time.monotonic() - start
        stats = pstats.Stats(profiler).stats
        io_times = {"read": 0.0, "hash": 0.0, "write": 0.0}
        for (_, _, name), (_, _, exclusive, _, _) in stats.items():
            if "hashlib" in name or "_hashlib" in name:
                io_times["hash"] += exclusive
            elif "'read' of '_io." in name or "'readinto' of '_io." in name:
                io_times["read"] += exclusive
            elif "'write' of '_io." in name:
                io_times["write"] += exclusive
        report(
            event + "_end",
            elapsed_seconds=elapsed,
            io_hash_exclusive_seconds=io_times,
            peak_rss_bytes=peak_memory(),
            **fields,
        )
    return value, elapsed, io_times


def resolve_owner(mode, model=None):
    """Resolve explicit scoring owner from one model selector or shared E03 mode."""
    check(mode in MODES, "MODE must be 'model' or 'e03'")
    if mode == "model":
        check(model in MODELS, "MODEL must be 'qwen' or 'gemma' when MODE is 'model'")
        return model
    check(model is None, "MODEL must be omitted/None when MODE is 'e03'")
    return "e03"


def representative_blocks(plan, owner):
    """Deterministic metadata-selected profiling subset for each scoring owner.

    Models select their English E01 midpoint SCREEN decoder block (4 source
    heads on one verified target matrix); shared E03 selects the pooled
    four-view character TF-IDF block.
    """
    ordered = scoring_blocks(plan, owner)
    if owner == "e03":
        selected = [
            bid
            for bid in ordered
            if plan.fit["groups"][plan["blocks"][bid]["target_group"]].get("recipe", {}).get("family")
            == "pooled_char_tfidf"
        ]
    else:
        selected = [
            bid
            for bid in ordered
            if plan["blocks"][bid]["priority"] == 0
            and plan["blocks"][bid]["split"] == "SCREEN"
            and plan.fit["groups"][plan["blocks"][bid]["target_group"]].get("language") == "en"
        ]
    check(len(selected) == 1, f"Expected one deterministic representative block for {owner}")
    return tuple(selected)


def select_blocks(plan, owner, scope="all"):
    """Return ordered owned blocks for the requested execution scope."""
    check(scope in SCOPES, "SCOPE must be 'all', 'priority', or 'profile'")
    ordered = scoring_blocks(plan, owner)
    if scope == "all":
        return ordered
    if scope == "profile":
        return representative_blocks(plan, owner)
    priority_zero = tuple(bid for bid in ordered if plan["blocks"][bid]["priority"] == 0)
    check(priority_zero, f"Owner '{owner}' has no priority-0 blocks; use scope 'all' or 'profile'")
    return priority_zero


def _block_metadata(plan, block_id):
    block = plan["blocks"][block_id]
    target = plan.fit["groups"][block["target_group"]]
    return {
        "block_id": block_id,
        "owner": block["owner"],
        "priority": block["priority"],
        "split": block["split"],
        "target_group": block["target_group"],
        "route_count": len(block["routes"]),
        "condition": target.get("condition", target.get("recipe", {}).get("family")),
        "site": target.get("site"),
        "language": target.get("language"),
    }


def _artifact_bytes(root, ref):
    manifest_path = root / ref["path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    folder = manifest_path.parent
    total = manifest_path.stat().st_size
    for name in manifest["files"]:
        total += (folder / name).stat().st_size
    return int(total)


def load_resume_references(plan, owner, resume_refs=()):
    """Verify explicit resume prediction references and reject foreign/conflicting entries."""
    unique_refs = []
    seen_tokens = set()
    for raw in resume_refs:
        ref = plan.store.ref(raw)
        token = (ref["path"], ref["sha256"])
        if token in seen_tokens:
            continue
        seen_tokens.add(token)
        unique_refs.append(ref)
    if not unique_refs:
        return {}
    indexed = prediction_index(plan, unique_refs)
    verified = {}
    for block_id, routes in indexed.items():
        check(
            plan["blocks"][block_id]["owner"] == owner,
            f"Foreign resume prediction block for owner '{owner}': {block_id}",
        )
        verified[block_id] = plain(next(iter(routes.values())).reference)
    return verified


def _write_json_atomic(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(target)


def _parse_manifest_arg(raw):
    text = raw.strip()
    if text.startswith("{"):
        return json.loads(text)
    return text


def _load_resume_file(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "predictions" in payload and isinstance(payload["predictions"], list):
        return payload["predictions"]
    raise ContractError("Resume file must be a JSON list of references or an object with a 'predictions' list")


def _priority_handoff_payload(plan, owner, priority_zero, completed):
    midpoint_screen = [completed[bid] for bid in priority_zero if plan["blocks"][bid]["split"] == "SCREEN"]
    dev_c_diagonals = [completed[bid] for bid in priority_zero if plan["blocks"][bid]["split"] == "DEV-C"]
    return {
        "owner": owner,
        "priority_blocks": list(priority_zero),
        "midpoint_screen_blocks": len(midpoint_screen),
        "dev_c_diagonal_blocks": len(dev_c_diagonals),
        "midpoint_screen_refs": midpoint_screen,
        "dev_c_diagonal_refs": dev_c_diagonals,
        "predictions": [completed[bid] for bid in priority_zero],
        "ready_for": ["E19", "E25"],
    }


def _build_summary(
    plan,
    owner,
    mode,
    model,
    scope,
    all_owner_blocks,
    scheduled_blocks,
    priority_zero,
    completed,
    published_count,
    reused_count,
    score_seconds,
    readback_seconds,
    published_bytes,
    status="passed",
):
    ordered_completed = [completed[bid] for bid in all_owner_blocks if bid in completed]
    refs_by_block = {bid: completed[bid] for bid in all_owner_blocks if bid in completed}
    scheduled_screen = sum(
        len(plan["blocks"][bid]["routes"])
        for bid in scheduled_blocks
        if plan["blocks"][bid]["split"] == "SCREEN"
    )
    scheduled_dev_c = sum(
        len(plan["blocks"][bid]["routes"])
        for bid in scheduled_blocks
        if plan["blocks"][bid]["split"] == "DEV-C"
    )
    total_owner_screen = sum(
        len(plan["blocks"][bid]["routes"])
        for bid in all_owner_blocks
        if plan["blocks"][bid]["split"] == "SCREEN"
    )
    total_owner_dev_c = sum(
        len(plan["blocks"][bid]["routes"])
        for bid in all_owner_blocks
        if plan["blocks"][bid]["split"] == "DEV-C"
    )
    if published_count > 0:
        avg_total_seconds = (score_seconds + readback_seconds) / published_count
        avg_bytes = published_bytes / published_count
        est_seconds = avg_total_seconds * len(all_owner_blocks)
        est_bytes = int(round(avg_bytes * len(all_owner_blocks)))
    else:
        est_seconds = None
        est_bytes = None
    return {
        "status": status,
        "owner": owner,
        "mode": mode,
        "model": model,
        "scope": scope,
        "plan": plain(plan.reference),
        "plan_identity": plan["identity"],
        "blocks_scheduled": len(scheduled_blocks),
        "blocks_published": published_count,
        "blocks_reused": reused_count,
        "owner_blocks_total": len(all_owner_blocks),
        "owner_blocks_completed": len(ordered_completed),
        "routes_scheduled": {"SCREEN": scheduled_screen, "DEV-C": scheduled_dev_c},
        "routes_owner_total": {"SCREEN": total_owner_screen, "DEV-C": total_owner_dev_c},
        "priority_handoff_ready": bool(priority_zero) and set(priority_zero) <= set(completed),
        "predictions": ordered_completed,
        "references_by_block": refs_by_block,
        "measurements": {
            "score_publish_seconds": score_seconds,
            "readback_seconds": readback_seconds,
            "published_bytes": published_bytes,
            "peak_rss_bytes": peak_memory(),
            "estimated_owner_total_seconds": est_seconds,
            "estimated_owner_total_bytes": est_bytes,
        },
        "stage4_complete": False,
    }


def run_scoring(
    plan,
    mode,
    producer,
    *,
    model=None,
    scope="all",
    resume_refs=(),
    output_inventory=None,
    report=emit,
    heartbeat_seconds=15.0,
):
    """Score missing owned canonical prediction blocks in priority order."""
    owner = resolve_owner(mode, model)
    all_owner_blocks = scoring_blocks(plan, owner)
    scheduled_blocks = select_blocks(plan, owner, scope)
    priority_zero = tuple(bid for bid in all_owner_blocks if plan["blocks"][bid]["priority"] == 0)
    existing = load_resume_references(plan, owner, resume_refs)
    completed = dict(existing)

    report(
        "worker_start",
        owner=owner,
        mode=mode,
        model=model,
        scope=scope,
        plan=plain(plan.reference),
        plan_identity=plan["identity"],
        code_commit=producer["code_commit"],
        runtime=plain(producer["software_versions"]),
        storage_scope=producer["storage_scope"],
        blocks_scheduled=len(scheduled_blocks),
        owner_blocks_total=len(all_owner_blocks),
        priority_zero_blocks=len(priority_zero),
        resumed_blocks=len(existing),
        stage4_complete=False,
    )

    published_count = 0
    reused_count = 0
    score_seconds = 0.0
    readback_seconds = 0.0
    published_bytes = 0
    handoff_emitted = False

    def maybe_emit_priority_handoff():
        nonlocal handoff_emitted
        if not handoff_emitted and priority_zero and set(priority_zero) <= set(completed):
            handoff_emitted = True
            report("priority_handoff", **_priority_handoff_payload(plan, owner, priority_zero, completed))

    # If all priority-0 blocks were already resumed, emit the handoff immediately.
    maybe_emit_priority_handoff()

    try:
        for idx, block_id in enumerate(scheduled_blocks, start=1):
            meta = _block_metadata(plan, block_id)
            meta.update(index=idx, total=len(scheduled_blocks))
            if block_id in existing:
                reused_count += 1
                report("block_reused", reference=completed[block_id], **meta)
            else:
                ref, elapsed, _ = measured(
                    "block_score_publish",
                    lambda bid=block_id: score_and_publish_block(plan, owner, bid, producer),
                    report,
                    heartbeat_seconds=heartbeat_seconds,
                    **meta,
                )
                score_seconds += elapsed
                routes, rb_elapsed, _ = measured(
                    "block_readback",
                    lambda bid=block_id, r=ref: load_prediction_block(plan, r, expected_block=bid),
                    report,
                    heartbeat_seconds=heartbeat_seconds,
                    **meta,
                )
                readback_seconds += rb_elapsed
                completed[block_id] = plain(ref)
                size_bytes = _artifact_bytes(plan.root, ref)
                published_bytes += size_bytes
                published_count += 1
                first_pred = next(iter(routes.values()))
                report(
                    "block_published",
                    reference=completed[block_id],
                    route_ids=sorted(routes),
                    rows=len(first_pred.rows),
                    payload_bytes=size_bytes,
                    score_elapsed_seconds=elapsed,
                    readback_elapsed_seconds=rb_elapsed,
                    peak_rss_bytes=peak_memory(),
                    **meta,
                )
            maybe_emit_priority_handoff()
    except KeyboardInterrupt:
        partial = _build_summary(
            plan,
            owner,
            mode,
            model,
            scope,
            all_owner_blocks,
            scheduled_blocks,
            priority_zero,
            completed,
            published_count,
            reused_count,
            score_seconds,
            readback_seconds,
            published_bytes,
            status="interrupted",
        )
        if output_inventory is not None and partial["predictions"]:
            _write_json_atomic(output_inventory, partial)
        report("worker_interrupted", **partial)
        raise

    summary = _build_summary(
        plan,
        owner,
        mode,
        model,
        scope,
        all_owner_blocks,
        scheduled_blocks,
        priority_zero,
        completed,
        published_count,
        reused_count,
        score_seconds,
        readback_seconds,
        published_bytes,
        status="passed",
    )
    if output_inventory is not None:
        _write_json_atomic(output_inventory, summary)
    report("worker_complete", **summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-root", required=True)
    parser.add_argument("--manifest", required=True, help="Stage 4 plan _COMPLETE.json path or JSON ref")
    parser.add_argument("--mode", default="model", choices=MODES)
    parser.add_argument("--model", choices=MODELS, default=None)
    parser.add_argument("--scope", default="all", choices=SCOPES)
    parser.add_argument("--resume-ref", action="append", default=[], help="Explicit prediction _COMPLETE.json or JSON ref")
    parser.add_argument("--resume-file", action="append", default=[], help="JSON file of explicit prediction refs")
    parser.add_argument("--output-inventory", default=None, help="Optional JSON path to save completed prediction refs")
    parser.add_argument(
        "--storage-scope",
        default="persistent-drive",
        choices=("persistent-drive", "local-development"),
    )
    parser.add_argument("--heartbeat-seconds", type=float, default=15.0)
    args = parser.parse_args(argv)

    root = phase1_root(args.phase1_root)
    protocol = json.loads((REPO / "configs/protocol.json").read_text(encoding="utf-8"))
    producer = current_producer(args.storage_scope)
    plan = load_stage4_plan(root, protocol, _parse_manifest_arg(args.manifest))

    resume_refs = [_parse_manifest_arg(item) for item in args.resume_ref]
    for path in args.resume_file:
        resume_refs.extend(_load_resume_file(path))

    run_scoring(
        plan,
        args.mode,
        producer,
        model=args.model,
        scope=args.scope,
        resume_refs=resume_refs,
        output_inventory=args.output_inventory,
        report=emit,
        heartbeat_seconds=args.heartbeat_seconds,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (ContractError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        emit("worker_blocked", status="blocked", blockers=[str(exc)], stage4_complete=False)
        raise SystemExit(2)
