"""CPU-only E07 original-caption coverage audit and extraction manifest builder.

Implements Phase 2 Stage 1 §1.7 against the verified Stage 1 dataset artifact
(`captions/{en,hi,bn,te}.jsonl`, `records.jsonl`, `targets.jsonl`,
`splits.json`, `dataset.json`) with zero raw image IO, zero GPU/model loading,
and zero CONFIRM draw materialization.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Mapping, Optional, Sequence

from src.common.contracts import (
    LANGUAGES,
    ContractError,
    ModelInput,
    TargetRecord,
    canonical_bytes,
    select_caption,
)
from src.common.storage import file_sha256, phase1_root, within
from src.phase2.interfaces import (
    E07_VERDICTS,
    KIND_E07_AUDIT,
    PHASE1_STAGE4_RELEASE,
    Phase1Binding,
    REPO,
    SCHEMA_E07_AUDIT,
    bind_phase1,
    check,
    contract,
    current_producer,
    frozen,
    load_phase2_artifact,
    normalize_ref,
    plain,
    publish_phase2,
    require_open_split,
    stage1_caption_paths,
    verify_resume_ref,
)

E07_DRAW_KEYS = (0, 1, 2)
E07_ALT_DRAW_KEYS = (1, 2)
E07_CHUNK_SIZE = 32
E07_NAMESPACE = "phase2/audits/e07"
E07_EXTRACTION_PLAN_SCHEMA = "phase2.e07_extraction_plan.v1"

# Predeclared Phase 2 §1.7 audit thresholds.
MIN_CHANGED_ANY_PAIR = 30
MIN_PRIMARY_GE10_ANY_PAIR = 6
MIN_CHANGED_LANG_QUORUM = 60
MIN_LANGS_AT_QUORUM = 3
MIN_CHANGED_FULL_PROCEED = 100
UNAVAILABLE_SINGLE_CAPTION_LT = 18
CHANGED_COHORT_MIN_SUPPORT = 10


def _text_sha256(text: str) -> str:
    """UTF-8 SHA-256 of unchanged original caption text (no Unicode normalization)."""
    check(isinstance(text, str) and bool(text.strip()), "Nonempty original caption text required")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _peak_rss_bytes() -> Optional[int]:
    """Return process peak RSS in bytes where available, else None."""
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB; macOS reports bytes.
        return int(usage * 1024) if usage < (1 << 34) else int(usage)
    except Exception:
        return None


def _emit(progress, event: Mapping[str, Any]) -> None:
    if progress is not None:
        progress(dict(event))


def _read_jsonl(path: Path) -> list:
    raw = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_bytes(row) + b"\n" for row in rows)


class _BindingAdapter:
    """Lightweight view compatible with stage1_caption_paths for plan-or-binding inputs."""

    def __init__(
        self,
        *,
        root: Path,
        protocol: Mapping[str, Any],
        stage4_release_ref: Mapping[str, str],
        lineage_identity: str,
        dataset_ref: Mapping[str, str],
        dataset_fingerprint: str,
        protocol_fingerprint: str,
        plan_identity: Optional[str] = None,
    ):
        self.root = Path(root).resolve()
        self.protocol = plain(protocol)
        self.stage4_release_ref = plain(stage4_release_ref)
        self.lineage_identity = str(lineage_identity)
        self.dataset_ref = plain(dataset_ref)
        self.dataset_fingerprint = str(dataset_fingerprint)
        self.protocol_fingerprint = str(protocol_fingerprint)
        self.plan_identity = plan_identity


def _resolve_context(bound_or_plan) -> _BindingAdapter:
    """Normalize either a Phase1Binding or a Phase 2 plan object/mapping."""
    if isinstance(bound_or_plan, Phase1Binding):
        return _BindingAdapter(
            root=bound_or_plan.root,
            protocol=bound_or_plan.protocol,
            stage4_release_ref=bound_or_plan.stage4_release_ref,
            lineage_identity=bound_or_plan.lineage_identity,
            dataset_ref=bound_or_plan.dataset_ref,
            dataset_fingerprint=bound_or_plan.dataset_fingerprint,
            protocol_fingerprint=bound_or_plan.protocol_fingerprint,
            plan_identity=None,
        )

    if hasattr(bound_or_plan, "binding") and isinstance(bound_or_plan.binding, Phase1Binding):
        b = bound_or_plan.binding
        plan_identity = getattr(bound_or_plan, "plan_identity", None)
        if plan_identity is None and isinstance(bound_or_plan, Mapping):
            plan_identity = bound_or_plan.get("plan_identity")
        if plan_identity is None and hasattr(bound_or_plan, "body"):
            plan_identity = bound_or_plan.body.get("plan_identity")
        return _BindingAdapter(
            root=b.root,
            protocol=b.protocol,
            stage4_release_ref=b.stage4_release_ref,
            lineage_identity=b.lineage_identity,
            dataset_ref=b.dataset_ref,
            dataset_fingerprint=b.dataset_fingerprint,
            protocol_fingerprint=b.protocol_fingerprint,
            plan_identity=plan_identity,
        )

    # Duck-typed plan object (attributes or mapping with root/protocol/body).
    body = None
    if hasattr(bound_or_plan, "body") and isinstance(bound_or_plan.body, Mapping):
        body = bound_or_plan.body
    elif isinstance(bound_or_plan, Mapping):
        body = bound_or_plan

    root = getattr(bound_or_plan, "root", None)
    protocol = getattr(bound_or_plan, "protocol", None)
    if root is None and body is not None:
        root = body.get("root")
    if protocol is None and body is not None:
        protocol = body.get("protocol")

    check(root is not None and protocol is not None,
          "bound_or_plan must be a Phase1Binding or a Phase 2 plan carrying root and protocol")

    if body is not None:
        stage4_release_ref = body.get("stage4_release") or body.get("stage4_release_ref")
        dataset_ref = body.get("stage1_dataset") or body.get("dataset_ref")
        lineage_identity = body.get("lineage_identity")
        dataset_fingerprint = body.get("dataset_fingerprint")
        protocol_fingerprint = body.get("protocol_fingerprint")
        plan_identity = body.get("plan_identity")
    else:
        stage4_release_ref = getattr(bound_or_plan, "stage4_release_ref", None)
        dataset_ref = getattr(bound_or_plan, "dataset_ref", None)
        lineage_identity = getattr(bound_or_plan, "lineage_identity", None)
        dataset_fingerprint = getattr(bound_or_plan, "dataset_fingerprint", None)
        protocol_fingerprint = getattr(bound_or_plan, "protocol_fingerprint", None)
        plan_identity = getattr(bound_or_plan, "plan_identity", None)

    for name, val in (
        ("stage4_release_ref", stage4_release_ref),
        ("dataset_ref", dataset_ref),
        ("lineage_identity", lineage_identity),
        ("dataset_fingerprint", dataset_fingerprint),
        ("protocol_fingerprint", protocol_fingerprint),
    ):
        check(val is not None, f"bound_or_plan missing required binding field: {name}")

    return _BindingAdapter(
        root=root,
        protocol=protocol,
        stage4_release_ref=stage4_release_ref,
        lineage_identity=lineage_identity,
        dataset_ref=dataset_ref,
        dataset_fingerprint=dataset_fingerprint,
        protocol_fingerprint=protocol_fingerprint,
        plan_identity=plan_identity,
    )


def _count_distribution(counts: Sequence[int]) -> dict:
    check(len(counts) > 0, "Nonempty count sequence required")
    hist = {}
    for c in counts:
        key = str(int(c))
        hist[key] = hist.get(key, 0) + 1
    ordered_hist = {k: hist[k] for k in sorted(hist, key=int)}
    return {
        "eq_1": sum(1 for c in counts if c == 1),
        "eq_2": sum(1 for c in counts if c == 2),
        "ge_3": sum(1 for c in counts if c >= 3),
        "min": int(min(counts)),
        "median": float(statistics.median(counts)),
        "max": int(max(counts)),
        "histogram": ordered_hist,
    }


def _multiplicity_summary(avail_counts: Sequence[int], distinct_counts: Sequence[int]) -> dict:
    check(len(avail_counts) == len(distinct_counts) and len(avail_counts) > 0,
          "Mismatched multiplicity count lengths")
    return {
        "n_images": len(avail_counts),
        "total_captions": int(sum(avail_counts)),
        "available_caption_count": _count_distribution(avail_counts),
        "distinct_text_count": _count_distribution(distinct_counts),
        "images_with_duplicate_texts": sum(
            1 for a, d in zip(avail_counts, distinct_counts) if d < a
        ),
    }


def _support_table(
    image_ids: Sequence[str],
    targets_by_id: Mapping[str, TargetRecord],
    category_order: Sequence[str],
    primary_categories: Sequence[str],
    min_screen_support: int,
) -> dict:
    primary_set = set(primary_categories)
    categories = {}
    primary_ge_20 = 0
    primary_ge_10 = 0

    for idx, cat in enumerate(category_order):
        pos = 0
        neg = 0
        unk = 0
        for iid in image_ids:
            val = targets_by_id[iid].values[idx]
            if val == 1:
                pos += 1
            elif val == 0:
                neg += 1
            elif val is None:
                unk += 1
            else:
                raise ContractError(f"Unexpected target value {val!r} for {iid}/{cat}")
        min_cls = min(pos, neg)
        is_primary = cat in primary_set
        if is_primary:
            if min_cls >= min_screen_support:
                primary_ge_20 += 1
            if min_cls >= CHANGED_COHORT_MIN_SUPPORT:
                primary_ge_10 += 1
        categories[cat] = {
            "positive": pos,
            "negative": neg,
            "unknown": unk,
            "known": pos + neg,
            "min_class_count": min_cls,
            "is_primary": is_primary,
        }

    return {
        "n_images": len(image_ids),
        "primary_ge_screen_min": primary_ge_20,
        "primary_ge_20": primary_ge_20,
        "primary_ge_10": primary_ge_10,
        "all_primary_ge_screen_min": primary_ge_20 == len(primary_categories),
        "all_primary_ge_20": primary_ge_20 == len(primary_categories),
        "all_primary_ge_10": primary_ge_10 == len(primary_categories),
        "categories": categories,
    }


def _evaluate_verdict(
    *,
    full_screen_support: Mapping[str, Any],
    per_language_changed_support: Mapping[str, Mapping[str, Mapping[str, Any]]],
    primary_count: int,
    min_screen_support: int,
) -> tuple:
    """Evaluate exact mutually exclusive & exhaustive Phase 2 §1.7 verdict rules."""
    blocked_reasons = []
    if not full_screen_support["all_primary_ge_screen_min"]:
        blocked_reasons.append(
            f"full SCREEN failed to preserve all {primary_count} primary categories at "
            f"min(pos, neg) >= {min_screen_support} "
            f"(only {full_screen_support['primary_ge_screen_min']}/{primary_count})"
        )

    best_per_lang = {}
    max_n_changed_any = 0
    max_primary_ge10_any = 0

    for lang in LANGUAGES:
        d1 = per_language_changed_support[lang]["draw_1"]
        d2 = per_language_changed_support[lang]["draw_2"]
        n1, p1 = int(d1["n_images"]), int(d1["primary_ge_10"])
        n2, p2 = int(d2["n_images"]), int(d2["primary_ge_10"])
        max_n_changed_any = max(max_n_changed_any, n1, n2)
        max_primary_ge10_any = max(max_primary_ge10_any, p1, p2)
        # Best-coverage draw for this language: higher changed count, tie-break by primary_ge_10 then draw_1.
        if (n2, p2) > (n1, p1):
            best_draw, best_n, best_p = "draw_2", n2, p2
        else:
            best_draw, best_n, best_p = "draw_1", n1, p1
        best_per_lang[lang] = {
            "best_draw": best_draw,
            "max_n_changed": best_n,
            "best_primary_ge_10": best_p,
        }

    if max_n_changed_any < MIN_CHANGED_ANY_PAIR:
        blocked_reasons.append(
            f"max genuinely changed SCREEN images across all (language, draw_k) pairs is "
            f"{max_n_changed_any} < {MIN_CHANGED_ANY_PAIR}"
        )

    if max_primary_ge10_any < MIN_PRIMARY_GE10_ANY_PAIR:
        blocked_reasons.append(
            f"no (language, draw_k) pair reached {MIN_PRIMARY_GE10_ANY_PAIR}/{primary_count} "
            f"primary categories at min(pos, neg) >= {CHANGED_COHORT_MIN_SUPPORT} "
            f"(max={max_primary_ge10_any})"
        )

    langs_at_quorum = [
        lang for lang in LANGUAGES if best_per_lang[lang]["max_n_changed"] >= MIN_CHANGED_LANG_QUORUM
    ]
    if len(langs_at_quorum) < MIN_LANGS_AT_QUORUM:
        blocked_reasons.append(
            f"only {len(langs_at_quorum)}/4 languages reached >= {MIN_CHANGED_LANG_QUORUM} "
            f"genuinely changed SCREEN images (requires >= {MIN_LANGS_AT_QUORUM})"
        )

    language_status = {}
    for lang in LANGUAGES:
        best_n = best_per_lang[lang]["max_n_changed"]
        best_p = best_per_lang[lang]["best_primary_ge_10"]
        if best_n < UNAVAILABLE_SINGLE_CAPTION_LT:
            flag = "unavailable_single_caption"
        elif best_n < MIN_CHANGED_LANG_QUORUM:
            flag = "low_change_rate"
        elif best_n < MIN_CHANGED_FULL_PROCEED:
            flag = "moderate_change_rate"
        elif best_p < primary_count:
            flag = "reduced_primary_support"
        else:
            flag = "full"
        language_status[lang] = {
            **best_per_lang[lang],
            "status": flag,
        }

    if blocked_reasons:
        return "blocked", blocked_reasons, language_status

    full_proceed_gaps = []
    for lang in LANGUAGES:
        best_n = best_per_lang[lang]["max_n_changed"]
        best_p = best_per_lang[lang]["best_primary_ge_10"]
        if best_n < MIN_CHANGED_FULL_PROCEED:
            full_proceed_gaps.append(
                f"{lang} max_n_changed={best_n} < {MIN_CHANGED_FULL_PROCEED} "
                f"({language_status[lang]['status']})"
            )
        if best_p < primary_count:
            full_proceed_gaps.append(
                f"{lang} best-coverage primary_ge_10={best_p}/{primary_count} < {primary_count}"
            )

    if not full_proceed_gaps:
        return (
            "passed_full",
            [
                f"all 4 languages reached >= {MIN_CHANGED_FULL_PROCEED} genuinely changed SCREEN views "
                f"with all {primary_count}/{primary_count} primary categories at min(pos, neg) >= "
                f"{CHANGED_COHORT_MIN_SUPPORT} on each language's best-coverage changed subset"
            ],
            language_status,
        )

    restricted_reasons = [
        "mandatory dual reporting required: full 360-image SCREEN (with explicit unchanged-view fraction) "
        "plus restricted_changed_cohort with exact matched draw-0 companion",
        *full_proceed_gaps,
    ]
    return "passed_restricted", restricted_reasons, language_status


def _build_extraction_plan(unique_views: Sequence[Mapping[str, Any]], chunk_size: int = E07_CHUNK_SIZE) -> dict:
    """Build immutable 32-image SCREEN-only deduplicated extraction manifest."""
    check(type(chunk_size) is int and chunk_size > 0, "Positive chunk_size required")
    per_lang = {lang: [] for lang in LANGUAGES}
    for view in unique_views:
        check(view["split"] == "SCREEN", "Extraction plan accepts SCREEN rows only")
        check(view["language"] in LANGUAGES, f"Unknown language in extraction plan: {view['language']!r}")
        per_lang[view["language"]].append(dict(view))

    chunks = []
    chunk_index = 0
    for lang in LANGUAGES:
        rows = sorted(
            per_lang[lang],
            key=lambda r: (r["image_id"], r["draw_keys"][0], r["caption_id"]),
        )
        for offset in range(0, len(rows), chunk_size):
            batch = rows[offset : offset + chunk_size]
            check(all(r["split"] == "SCREEN" for r in batch), "Non-SCREEN row in extraction chunk")
            chunks.append({
                "chunk_id": f"chunk_{chunk_index:04d}",
                "language": lang,
                "split": "SCREEN",
                "n_rows": len(batch),
                "rows": batch,
            })
            chunk_index += 1

    return {
        "schema": E07_EXTRACTION_PLAN_SCHEMA,
        "split": "SCREEN",
        "confirm_rows": 0,
        "condition": "image_caption",
        "depth_policy": "fixed_midpoint",
        "poolings": ["last_caption", "caption_mean"],
        "same_forward_both_poolings": True,
        "chunk_size": chunk_size,
        "total_unique_views": sum(len(per_lang[lang]) for lang in LANGUAGES),
        "total_chunks": len(chunks),
        "per_language_counts": {lang: len(per_lang[lang]) for lang in LANGUAGES},
        "chunks": chunks,
    }


@contract
def load_e07_audit(
    root,
    protocol,
    reference,
    *,
    expected_dataset=None,
    expected_lineage_identity=None,
    expected_plan_identity=None,
):
    """Authenticate and load a published phase2_e07_audit artifact."""
    ref, body, marker = load_phase2_artifact(
        root,
        protocol,
        reference,
        KIND_E07_AUDIT,
        expected_schema=SCHEMA_E07_AUDIT,
        expected_dataset=expected_dataset,
        expected_plan_identity=expected_plan_identity,
    )
    check(body.get("draw0_verified") is True, "E07 audit missing draw0_verified=True")
    check(body.get("confirm_excluded") is True, "E07 audit missing confirm_excluded=True")
    check(body.get("verdict") in E07_VERDICTS, f"Invalid E07 verdict: {body.get('verdict')!r}")
    if expected_lineage_identity is not None:
        check(
            body.get("lineage_identity") == expected_lineage_identity,
            "E07 audit lineage_identity mismatch",
        )
    if body["verdict"] == "blocked":
        check(body.get("extraction_plan") is None, "Blocked E07 audit must not emit extraction_plan")
    else:
        ep = body.get("extraction_plan")
        check(isinstance(ep, Mapping) and ep.get("confirm_rows") == 0 and ep.get("split") == "SCREEN",
              "Passed E07 audit missing valid SCREEN-only extraction_plan")
    return ref, body, marker


@contract
def audit_e07_captions(
    bound_or_plan,
    producer,
    *,
    progress=None,
    resume_ref=None,
    run_id=None,
):
    """Run the CPU-only Phase 2 Stage 1 E07 original-caption coverage audit.

    Reads only verified Stage 1 dataset manifests/JSONL metadata (`dataset.json`,
    `records.jsonl`, `targets.jsonl`, `splits.json`, `captions/{en,hi,bn,te}.jsonl`)
    with zero raw image IO, zero GPU/model loading, and zero CONFIRM draw materialization.
    """
    ctx = _resolve_context(bound_or_plan)
    protocol = ctx.protocol
    total_units = 6

    _emit(progress, {
        "event": "e07_audit_start",
        "unit": "init",
        "completed": 0,
        "total": total_units,
        "read_hash_s": 0.0,
        "compute_s": 0.0,
        "write_s": 0.0,
        "peak_rss_bytes": _peak_rss_bytes(),
    })

    t_read0 = time.monotonic()
    caption_meta = stage1_caption_paths(ctx)
    files_meta = caption_meta["files"]
    read_hash_s = time.monotonic() - t_read0

    expected_inputs = {
        "stage4_release": plain(ctx.stage4_release_ref),
        "lineage_identity": ctx.lineage_identity,
        "dataset_ref": plain(ctx.dataset_ref),
        "dataset_fingerprint": ctx.dataset_fingerprint,
        "protocol_fingerprint": ctx.protocol_fingerprint,
        "caption_files": plain(files_meta),
        "draw_keys": list(E07_DRAW_KEYS),
        "chunk_size": E07_CHUNK_SIZE,
    }

    if resume_ref is not None:
        ref, body, _ = verify_resume_ref(
            ctx.root,
            protocol,
            resume_ref,
            KIND_E07_AUDIT,
            expected_schema=SCHEMA_E07_AUDIT,
            expected_dataset=ctx.dataset_fingerprint,
            expected_plan_identity=ctx.plan_identity,
            expected_inputs=expected_inputs,
        )
        check(body.get("lineage_identity") == ctx.lineage_identity, "Resume E07 audit lineage mismatch")
        check(body.get("draw0_verified") is True, "Resume E07 audit draw0_verified must be True")
        check(body.get("confirm_excluded") is True, "Resume E07 audit confirm_excluded must be True")
        check(body.get("verdict") in E07_VERDICTS, "Resume E07 audit has invalid verdict")
        _emit(progress, {
            "event": "e07_audit_resumed",
            "unit": "resume",
            "completed": total_units,
            "total": total_units,
            "read_hash_s": round(read_hash_s, 6),
            "compute_s": 0.0,
            "write_s": 0.0,
            "peak_rss_bytes": _peak_rss_bytes(),
            "verdict": body["verdict"],
            "reference": plain(ref),
        })
        return ref

    # Step 1: Load and verify dataset.json, splits.json, targets.jsonl, records.jsonl (no image IO).
    t_r = time.monotonic()
    dataset_path = within(ctx.root, files_meta["dataset.json"]["path"])
    splits_path = within(ctx.root, files_meta["splits.json"]["path"])
    targets_path = within(ctx.root, files_meta["targets.jsonl"]["path"])
    records_path = within(ctx.root, files_meta["records.jsonl"]["path"])

    dataset_doc = json.loads(dataset_path.read_text(encoding="utf-8"))
    splits_doc = json.loads(splits_path.read_text(encoding="utf-8"))
    target_rows = _read_jsonl(targets_path)
    record_rows = _read_jsonl(records_path)
    read_hash_s += time.monotonic() - t_r

    t_c = time.monotonic()
    category_order = list(protocol["category_order"])
    primary_categories = list(protocol["primary_categories"])
    min_screen_support = int(protocol["minimum_positive_and_negative"]["SCREEN"])
    check(len(category_order) == 25 and len(set(category_order)) == 25, "Protocol requires 25 unique categories")
    check(
        list(dataset_doc.get("category_order", [])) == category_order,
        "Dataset category_order mismatch",
    )
    check(
        dataset_doc.get("dataset_fingerprint") == ctx.dataset_fingerprint
        and dataset_doc.get("protocol_fingerprint") == ctx.protocol_fingerprint,
        "Dataset manifest fingerprint mismatch",
    )

    # Validate splits disjointness and expected sizes; isolate CONFIRM for strict exclusion.
    expected_split_sizes = protocol["split_sizes"]
    split_sets = {}
    seen_all_ids = set()
    for split_name, expected_n in expected_split_sizes.items():
        check(split_name in splits_doc, f"splits.json missing split: {split_name}")
        ids = splits_doc[split_name]
        check(isinstance(ids, list) and len(ids) == int(expected_n),
              f"Split {split_name} size mismatch: expected {expected_n}, got {len(ids) if isinstance(ids, list) else type(ids)}")
        id_set = set(ids)
        check(len(id_set) == len(ids), f"Duplicate image IDs inside split {split_name}")
        check(seen_all_ids.isdisjoint(id_set), f"Overlapping image IDs involving split {split_name}")
        seen_all_ids.update(id_set)
        split_sets[split_name] = id_set

    screen_ids = sorted(split_sets["SCREEN"])
    screen_id_set = split_sets["SCREEN"]
    fit_id_set = split_sets["FIT"]
    dev_s_id_set = split_sets["DEV-S"]
    confirm_id_set = split_sets["CONFIRM"]
    split_by_id = {iid: s for s, id_set in split_sets.items() for iid in id_set}

    # Validate targets for open splits; never materialize CONFIRM targets into audit tables.
    screen_targets = {}
    seen_target_ids = set()
    for row in target_rows:
        iid = row["image_id"]
        split = row["split"]
        check(iid not in seen_target_ids, f"Duplicate target row for {iid}")
        seen_target_ids.add(iid)
        check(split_by_id.get(iid) == split, f"Target split mismatch for {iid}: {split!r}")
        if split == "CONFIRM":
            # Confirm exclusion: do not load CONFIRM targets into audit support structures.
            continue
        require_open_split(split)
        rec = TargetRecord(iid, split, tuple(row["values"]))
        if split == "SCREEN":
            screen_targets[iid] = rec

    check(seen_target_ids == seen_all_ids, "targets.jsonl image population does not match splits.json")
    check(set(screen_targets) == screen_id_set, "Missing SCREEN target records")
    check(confirm_id_set.isdisjoint(screen_targets), "CONFIRM leaked into SCREEN targets")

    # Verify unknown-cell mask preservation (e.g., plant.n.02 on 6a20677eee04fd08).
    for cell in protocol.get("invalid_cells", []):
        if cell.get("partition") == "SCREEN" and cell.get("image_id") in screen_targets:
            cat = cell["category"]
            cat_idx = category_order.index(cat)
            val = screen_targets[cell["image_id"]].values[cat_idx]
            check(
                val is None,
                f"Unknown-cell mask violated: {cell['image_id']} / {cat} expected None, got {val!r}",
            )

    # Index draw-0 SCREEN records from records.jsonl; exclude CONFIRM completely.
    draw0_screen_records = {}
    for row in record_rows:
        iid = row["image_id"]
        split = row["split"]
        check(split_by_id.get(iid) == split, f"Record split mismatch for {iid}: {split!r}")
        if split == "CONFIRM":
            continue
        require_open_split(split)
        if split != "SCREEN":
            continue
        lang = row["language"]
        check(lang in LANGUAGES, f"Unknown language in records.jsonl: {lang!r}")
        key = (iid, lang)
        check(key not in draw0_screen_records, f"Duplicate draw-0 SCREEN record for {key}")
        draw0_screen_records[key] = ModelInput(
            image_id=iid,
            image_path=row["image_path"],
            image_sha256=row["image_sha256"],
            language=lang,
            caption_id=row["caption_id"],
            caption_text=row["caption_text"],
            split=split,
        )

    expected_screen_pairs = len(screen_ids) * len(LANGUAGES)
    check(
        len(draw0_screen_records) == expected_screen_pairs,
        f"Expected {expected_screen_pairs} SCREEN draw-0 records, found {len(draw0_screen_records)}",
    )

    compute_s = time.monotonic() - t_c
    _emit(progress, {
        "event": "e07_audit_progress",
        "unit": "dataset_metadata",
        "completed": 1,
        "total": total_units,
        "read_hash_s": round(read_hash_s, 6),
        "compute_s": round(compute_s, 6),
        "write_s": 0.0,
        "peak_rss_bytes": _peak_rss_bytes(),
    })

    # Step 2: Process captions/{en,hi,bn,te}.jsonl one language at a time.
    screen_multiplicity = {}
    fit_dev_s_multiplicity = {"FIT": {}, "DEV-S": {}}
    draw_buckets = {}
    changed_ids_by_lang_draw = {lang: {"draw_1": set(), "draw_2": set()} for lang in LANGUAGES}
    unique_views_map = {}  # (image_id, language, text_sha256) -> dict
    draw_resolution_rows = []

    total_changed_draw_pairs = 0
    total_reused_same_id = 0
    total_reused_distinct_id_same_text = 0

    for lang_idx, lang in enumerate(LANGUAGES, start=2):
        t_lr = time.monotonic()
        cap_path = within(ctx.root, files_meta[f"captions/{lang}.jsonl"]["path"])
        cap_rows = _read_jsonl(cap_path)
        read_hash_s += time.monotonic() - t_lr

        t_lc = time.monotonic()
        screen_candidates = {iid: [] for iid in screen_ids}
        fit_meta_counts = {iid: [0, set()] for iid in fit_id_set}
        dev_s_meta_counts = {iid: [0, set()] for iid in dev_s_id_set}

        for row in cap_rows:
            iid = row["image_id"]
            split = row["split"]
            row_lang = row["language"]
            check(row_lang == lang, f"Language mismatch in captions/{lang}.jsonl: got {row_lang!r}")
            check(split_by_id.get(iid) == split, f"Caption split mismatch for {iid}: {split!r}")

            # Strictly exclude CONFIRM before ModelInput construction or any draw materialization.
            if split == "CONFIRM":
                check(iid in confirm_id_set, f"Unexpected CONFIRM caption image_id: {iid}")
                continue

            require_open_split(split)
            if split == "SCREEN":
                screen_candidates[iid].append(
                    ModelInput(
                        image_id=iid,
                        image_path=row["image_path"],
                        image_sha256=row["image_sha256"],
                        language=lang,
                        caption_id=row["caption_id"],
                        caption_text=row["caption_text"],
                        split=split,
                    )
                )
            elif split == "FIT":
                entry = fit_meta_counts[iid]
                entry[0] += 1
                entry[1].add(_text_sha256(row["caption_text"]))
            elif split == "DEV-S":
                entry = dev_s_meta_counts[iid]
                entry[0] += 1
                entry[1].add(_text_sha256(row["caption_text"]))
            # DEV-C is open but not part of E07 SCREEN audit or FIT/DEV-S refit multiplicity.

        # Read-only FIT and DEV-S multiplicity summaries (metadata only, no extraction plan).
        fit_avail = [fit_meta_counts[iid][0] for iid in sorted(fit_id_set)]
        fit_dist = [len(fit_meta_counts[iid][1]) for iid in sorted(fit_id_set)]
        check(all(c >= 1 for c in fit_avail), f"Missing FIT captions for language {lang}")
        fit_dev_s_multiplicity["FIT"][lang] = _multiplicity_summary(fit_avail, fit_dist)

        dev_s_avail = [dev_s_meta_counts[iid][0] for iid in sorted(dev_s_id_set)]
        dev_s_dist = [len(dev_s_meta_counts[iid][1]) for iid in sorted(dev_s_id_set)]
        check(all(c >= 1 for c in dev_s_avail), f"Missing DEV-S captions for language {lang}")
        fit_dev_s_multiplicity["DEV-S"][lang] = _multiplicity_summary(dev_s_avail, dev_s_dist)

        # SCREEN multiplicity, draw-0 parity assertion, and draws 1 & 2 bucketing/deduplication.
        screen_avail = []
        screen_dist = []
        buckets = {
            "draw_1": {
                "same_id_as_draw0": 0,
                "distinct_id_same_text_as_draw0": 0,
                "changed_text_vs_draw0": 0,
            },
            "draw_2": {
                "same_id_as_draw0": 0,
                "distinct_id_same_text_as_draw0": 0,
                "changed_text_vs_draw0": 0,
            },
        }

        for iid in screen_ids:
            cands = screen_candidates[iid]
            check(len(cands) >= 1, f"Missing SCREEN captions for ({iid}, {lang})")
            rec0 = draw0_screen_records[(iid, lang)]
            rec0_text_sha = _text_sha256(rec0.caption_text)

            text_hashes = set()
            for c in cands:
                check(
                    c.image_path == rec0.image_path and c.image_sha256 == rec0.image_sha256,
                    f"Image identity mismatch in captions/{lang}.jsonl for {iid}",
                )
                text_hashes.add(_text_sha256(c.caption_text))

            screen_avail.append(len(cands))
            screen_dist.append(len(text_hashes))

            sel0 = select_caption(cands, 0)
            sel0_text_sha = _text_sha256(sel0.caption_text)

            # HARD ASSERTION: draw_key=0 must reproduce records.jsonl identically.
            check(
                sel0.caption_id == rec0.caption_id and sel0_text_sha == rec0_text_sha,
                f"Draw-0 parity mismatch for ({iid}, {lang}): "
                f"selected=({sel0.caption_id}, {sel0_text_sha}) vs "
                f"records.jsonl=({rec0.caption_id}, {rec0_text_sha})",
            )

            for dk in E07_ALT_DRAW_KEYS:
                dk_name = f"draw_{dk}"
                sel_k = select_caption(cands, dk)
                sel_k_text_sha = _text_sha256(sel_k.caption_text)

                if sel_k.caption_id == sel0.caption_id:
                    bucket = "same_id_as_draw0"
                    buckets[dk_name][bucket] += 1
                    total_reused_same_id += 1
                    resolution_action = "reuse_draw0_same_id"
                elif sel_k_text_sha == sel0_text_sha:
                    bucket = "distinct_id_same_text_as_draw0"
                    buckets[dk_name][bucket] += 1
                    total_reused_distinct_id_same_text += 1
                    resolution_action = "reuse_draw0_text_alias"
                else:
                    bucket = "changed_text_vs_draw0"
                    buckets[dk_name][bucket] += 1
                    total_changed_draw_pairs += 1
                    changed_ids_by_lang_draw[lang][dk_name].add(iid)
                    resolution_action = "extract_changed_view"

                    dedup_key = (iid, lang, sel_k_text_sha)
                    if dedup_key not in unique_views_map:
                        unique_views_map[dedup_key] = {
                            "image_id": iid,
                            "image_path": sel_k.image_path,
                            "image_sha256": sel_k.image_sha256,
                            "language": lang,
                            "split": "SCREEN",
                            "caption_id": sel_k.caption_id,
                            "caption_text": sel_k.caption_text,
                            "caption_text_sha256": sel_k_text_sha,
                            "draw_keys": [dk],
                            "aliased_caption_ids": [sel_k.caption_id],
                        }
                    else:
                        existing = unique_views_map[dedup_key]
                        if dk not in existing["draw_keys"]:
                            existing["draw_keys"].append(dk)
                        if sel_k.caption_id not in existing["aliased_caption_ids"]:
                            existing["aliased_caption_ids"].append(sel_k.caption_id)

                draw_resolution_rows.append({
                    "image_id": iid,
                    "language": lang,
                    "draw_key": dk,
                    "split": "SCREEN",
                    "bucket": bucket,
                    "resolution": resolution_action,
                    "selected_caption_id": sel_k.caption_id,
                    "selected_text_sha256": sel_k_text_sha,
                    "draw0_caption_id": sel0.caption_id,
                    "draw0_text_sha256": sel0_text_sha,
                })

        screen_multiplicity[lang] = _multiplicity_summary(screen_avail, screen_dist)

        n_screen = len(screen_ids)
        for dk_name in ("draw_1", "draw_2"):
            b = buckets[dk_name]
            check(
                b["same_id_as_draw0"] + b["distinct_id_same_text_as_draw0"] + b["changed_text_vs_draw0"]
                == n_screen,
                f"Mutually exclusive bucket sum mismatch for {lang}/{dk_name}",
            )
            unchanged = b["same_id_as_draw0"] + b["distinct_id_same_text_as_draw0"]
            b["unchanged_fraction"] = unchanged / float(n_screen)
            b["changed_fraction"] = b["changed_text_vs_draw0"] / float(n_screen)
        draw_buckets[lang] = buckets

        compute_s += time.monotonic() - t_lc
        _emit(progress, {
            "event": "e07_audit_progress",
            "unit": f"captions:{lang}",
            "completed": lang_idx,
            "total": total_units,
            "read_hash_s": round(read_hash_s, 6),
            "compute_s": round(compute_s, 6),
            "write_s": 0.0,
            "peak_rss_bytes": _peak_rss_bytes(),
        })

    # Step 3: Support tables, verdict evaluation, and optional extraction plan.
    t_fc = time.monotonic()
    for view in unique_views_map.values():
        view["draw_keys"] = sorted(view["draw_keys"])
        view["aliased_caption_ids"] = sorted(view["aliased_caption_ids"])

    full_screen_support = _support_table(
        screen_ids, screen_targets, category_order, primary_categories, min_screen_support
    )

    per_language_changed_support = {}
    for lang in LANGUAGES:
        per_language_changed_support[lang] = {}
        for dk_name in ("draw_1", "draw_2"):
            subset_ids = sorted(changed_ids_by_lang_draw[lang][dk_name])
            per_language_changed_support[lang][dk_name] = _support_table(
                subset_ids, screen_targets, category_order, primary_categories, min_screen_support
            )

    four_language_intersection_support = {}
    for dk_name in ("draw_1", "draw_2"):
        inter = set(screen_ids)
        for lang in LANGUAGES:
            inter &= changed_ids_by_lang_draw[lang][dk_name]
        four_language_intersection_support[dk_name] = _support_table(
            sorted(inter), screen_targets, category_order, primary_categories, min_screen_support
        )

    support_tables = {
        "full_screen": full_screen_support,
        "per_language_changed": per_language_changed_support,
        "four_language_intersection": four_language_intersection_support,
    }

    verdict, verdict_reasons, language_coverage_status = _evaluate_verdict(
        full_screen_support=full_screen_support,
        per_language_changed_support=per_language_changed_support,
        primary_count=len(primary_categories),
        min_screen_support=min_screen_support,
    )
    check(verdict in E07_VERDICTS, f"Unexpected E07 verdict: {verdict!r}")

    unique_views_list = list(unique_views_map.values())
    per_lang_unique = {
        lang: sum(1 for v in unique_views_list if v["language"] == lang) for lang in LANGUAGES
    }
    total_unique = sum(per_lang_unique.values())
    unique_changed_views = {
        "per_language": per_lang_unique,
        "total_unique_views": total_unique,
        "gpu_forwards_per_model": total_unique,
        "total_changed_draw_pairs": total_changed_draw_pairs,
        "deduplicated_across_draws_1_and_2": total_changed_draw_pairs - total_unique,
        "reused_from_draw0_same_id": total_reused_same_id,
        "reused_from_draw0_distinct_id_same_text": total_reused_distinct_id_same_text,
    }

    extraction_plan = (
        None if verdict == "blocked" else _build_extraction_plan(unique_views_list, E07_CHUNK_SIZE)
    )
    compute_s += time.monotonic() - t_fc

    body = {
        "schema": SCHEMA_E07_AUDIT,
        "stage4_release": plain(ctx.stage4_release_ref),
        "lineage_identity": ctx.lineage_identity,
        "dataset_ref": plain(ctx.dataset_ref),
        "dataset_fingerprint": ctx.dataset_fingerprint,
        "draw0_verified": True,
        "confirm_excluded": True,
        "verdict": verdict,
        "verdict_reasons": list(verdict_reasons),
        "dual_reporting_required": verdict == "passed_restricted",
        "language_coverage_status": language_coverage_status,
        "screen_multiplicity": screen_multiplicity,
        "fit_dev_s_multiplicity": fit_dev_s_multiplicity,
        "draw_buckets": draw_buckets,
        "support_tables": support_tables,
        "unique_changed_views": unique_changed_views,
        "extraction_plan": extraction_plan,
        "inputs": expected_inputs,
    }
    if ctx.plan_identity is not None:
        body["plan_identity"] = ctx.plan_identity

    t_w = time.monotonic()
    ref = publish_phase2(
        ctx.root,
        protocol,
        ctx.dataset_fingerprint,
        E07_NAMESPACE,
        KIND_E07_AUDIT,
        body,
        producer,
        files={"draw_resolutions.jsonl": _jsonl_bytes(draw_resolution_rows)},
        upstream=[plain(ctx.stage4_release_ref), plain(ctx.dataset_ref)],
        seeds={"caption_draws": list(E07_DRAW_KEYS), "split": protocol["split_seed"]},
        run_id=run_id,
    )
    write_s = time.monotonic() - t_w

    _emit(progress, {
        "event": "e07_audit_complete",
        "unit": "publish",
        "completed": total_units,
        "total": total_units,
        "read_hash_s": round(read_hash_s, 6),
        "compute_s": round(compute_s, 6),
        "write_s": round(write_s, 6),
        "peak_rss_bytes": _peak_rss_bytes(),
        "verdict": verdict,
        "unique_changed_views": total_unique,
        "extraction_chunks": 0 if extraction_plan is None else extraction_plan["total_chunks"],
        "reference": plain(ref),
    })
    return ref


def _emit_stdout(event: Mapping[str, Any]) -> None:
    print(json.dumps(event), flush=True)


def _parse_ref_arg(value: Optional[str]):
    if value is None:
        return None
    stripped = value.strip()
    if stripped.startswith("{"):
        return json.loads(stripped)
    return stripped


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="CPU-only Phase 2 Stage 1 E07 original-caption coverage audit."
    )
    parser.add_argument("--phase1-root", required=True)
    parser.add_argument(
        "--stage4-release",
        default=None,
        help="Explicit Stage 4 release _COMPLETE.json path or JSON {'path','sha256'}; defaults to pinned accepted release.",
    )
    parser.add_argument(
        "--resume-ref",
        default=None,
        help="Optional explicit phase2_e07_audit _COMPLETE.json path or JSON {'path','sha256'} to verify and reuse.",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO / "configs/protocol.json",
    )
    parser.add_argument(
        "--storage-scope",
        default="persistent-drive",
        choices=("persistent-drive", "local-development"),
    )
    args = parser.parse_args(argv)

    root = phase1_root(args.phase1_root)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    producer = current_producer(args.storage_scope)

    if args.stage4_release is None:
        stage4_ref = plain(PHASE1_STAGE4_RELEASE)
    else:
        parsed_s4 = _parse_ref_arg(args.stage4_release)
        stage4_ref = parsed_s4 if isinstance(parsed_s4, dict) else normalize_ref(root, parsed_s4)
    resume_ref = _parse_ref_arg(args.resume_ref)

    binding = bind_phase1(root, protocol, stage4_ref)
    audit_ref = audit_e07_captions(
        binding,
        producer,
        progress=_emit_stdout,
        resume_ref=resume_ref,
    )
    _, body, _ = load_e07_audit(
        root,
        protocol,
        audit_ref,
        expected_dataset=binding.dataset_fingerprint,
        expected_lineage_identity=binding.lineage_identity,
    )
    _emit_stdout({
        "event": "e07_audit_summary",
        "manifest": str(root / audit_ref["path"]),
        "reference": plain(audit_ref),
        "verdict": body["verdict"],
        "verdict_reasons": list(body["verdict_reasons"]),
        "unique_changed_views": plain(body["unique_changed_views"]),
        "extraction_chunks": (
            0 if body["extraction_plan"] is None else body["extraction_plan"]["total_chunks"]
        ),
        "confirm_excluded": body["confirm_excluded"],
        "draw0_verified": body["draw0_verified"],
    })
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, OSError, ValueError, KeyError) as exc:
        _emit_stdout({"status": "blocked", "blockers": [str(exc)]})
        raise SystemExit(2)
