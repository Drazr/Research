"""Phase 2 Stage 4 CPU worker for E07: frozen-head scoring and dual-cohort comparison.

Implements phase2.md §1.3-1.5, §1.7, §2.9, and §4.1-4.3:
- Authenticates `phase2_plan`, explicitly binds `Phase1Binding` via `bind_phase1`,
  verifies `phase2_e07_audit` (refusing `blocked`), `phase2_e07_release` (including
  every referenced `phase2_e07_chunk`, `.npy` hashes/shapes/finiteness, and passing
  repeat checks), and the model's Stage 2 E29 `phase2_analysis_block` (`draw0_reference`).
- Runs in-memory `en->en` draw-0 SCREEN reproduction self-checks (`atol=0, rtol=0`)
  for both `last_caption` and `caption_mean` against accepted Stage 4 SCREEN logits.
- Scores ONLY the verified unique changed views per `(pooling, target_language)` with
  the 4 frozen Stage 3 midpoint `image_caption` heads/scalers via
  `src.stage3.engine.transform` (gracefully handling zero-changed languages such as
  `bn` with `(4, 0, 25)` empty float64 logits), cross-checks mapped draw-1/draw-2
  counts against `audit_body["draw_buckets"]`, publishes 8 compact
  `phase2_prediction_block` artifacts under `phase2/predictions/e07_screen/<model>/<run_id>/`,
  and reuses Stage 4 draw-0 logits bit-for-bit (`atol=0, rtol=0`) on all unchanged /
  byte-identical rows.
- Evaluates BOTH `full_screen_360` (12-route mean + `changed_targets_only` 9-route
  mean, asserting `<= 1e-12` parity with E29 `draw0_reference`) AND
  `restricted_changed_cohort` (per-target `S_changed(t, draw_k)` with exact matched
  draw-0 companion on identical image-category cells and sliced `(2000, 360)`
  bootstrap counts, marking `n_changed == 0` languages `"unavailable"`, reporting
  `supported_primary_ge_10` and 14-primary macros, pooling-reduction shifts, and
  sign preservation) using `protocol["category_order"]` and
  `int(protocol["minimum_positive_and_negative"]["SCREEN"])`, publishing under
  `phase2/results/e07/<model>/<run_id>/`.

Phase 1 directories remain strictly read-only, heads are never refit, and `CONFIRM`
remains closed.
"""

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Optional

from src.common.contracts import LANGUAGES, ContractError, fingerprint
from src.common.storage import file_sha256, new_run_id, phase1_root, within
from src.phase2.e07_audit import CHANGED_COHORT_MIN_SUPPORT, load_e07_audit
from src.phase2.interfaces import (
    KIND_ANALYSIS,
    KIND_E07_CHUNK,
    KIND_E07_RELEASE,
    KIND_PREDICTION,
    MODELS,
    POOLINGS,
    REPO,
    SCHEMA_ANALYSIS,
    SCHEMA_E07_FEATURES,
    SCHEMA_PREDICTION,
    bind_phase1,
    check,
    contract,
    e29_screen_blocks,
    load_frozen_source_head,
    load_phase2_artifact,
    midpoint_depth,
    midpoint_groups,
    normalize_ref,
    plain,
    publish_phase2,
    read_verified_midpoint_features,
    require_open_split,
    verify_resume_ref,
)
from src.phase2.plan import load_phase2_plan

E07_ALLOWED_VERDICTS = ("passed_full", "passed_restricted")
E07_CONDITION = "image_caption"
E07_SPLIT = "SCREEN"
E07_DRAW_KEYS = (1, 2)
E07_DRAW_NAMES = ("draw_1", "draw_2")
DRAW0_PARITY_TOL = 1e-12
LOSS_METRICS = frozenset({"brier", "nll"})
SUPERVISION_DIAGONAL = "within_context_source_fit"
SUPERVISION_OFF_DIAGONAL = "source_only_transfer"
UNAVAILABLE_ZERO_CHANGED_REASON = "unavailable_single_caption_zero_changed_views"
INTERPRETATION_SCOPE = "frozen_draw0_heads_test_view_sensitivity"


def _require_numpy():
    import numpy as np

    return np


def _peak_rss_bytes():
    try:
        import resource

        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return rss if sys.platform == "darwin" else rss * 1024
    except Exception:
        return None


def _emit_progress(
    progress,
    event,
    *,
    unit,
    completed,
    total,
    read_hash_s=0.0,
    compute_s=0.0,
    write_s=0.0,
    **extra,
):
    if progress is None:
        return
    payload = {
        "event": event,
        "unit": unit,
        "completed": int(completed),
        "total": int(total),
        "read_hash_s": round(float(read_hash_s), 4),
        "compute_s": round(float(compute_s), 4),
        "write_s": round(float(write_s), 4),
        "peak_rss_bytes": _peak_rss_bytes(),
    }
    payload.update(extra)
    progress(payload)


def _route_key(source_lang, target_lang):
    return f"{source_lang}->{target_lang}"


def _off_diagonal_pairs(target_languages=LANGUAGES):
    return [(s, t) for s in LANGUAGES for t in target_languages if s != t]


def _gap_value(metric, diag_val, route_val):
    if diag_val is None or route_val is None:
        return None
    if metric in LOSS_METRICS:
        return float(route_val - diag_val)
    return float(diag_val - route_val)


def _gap_draws(metric, diag_draws, route_draws):
    if metric in LOSS_METRICS:
        return route_draws - diag_draws
    return diag_draws - route_draws


def _diag_gain_value(metric, last_val, mean_val):
    if last_val is None or mean_val is None:
        return None
    if metric in LOSS_METRICS:
        return float(last_val - mean_val)
    return float(mean_val - last_val)


def _diag_gain_draws(metric, last_draws, mean_draws):
    if metric in LOSS_METRICS:
        return last_draws - mean_draws
    return mean_draws - last_draws


def _diff_value(a, b):
    if a is None or b is None:
        return None
    return float(a - b)


def _sign_label(value):
    if value is None:
        return None
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _sign_preservation_record(draw0_val, drawk_val):
    s0 = _sign_label(draw0_val)
    sk = _sign_label(drawk_val)
    preserved = bool(s0 is not None and sk is not None and s0 == sk)
    return {
        "draw_0_delta_g_auroc": None if draw0_val is None else float(draw0_val),
        "draw_k_delta_g_auroc": None if drawk_val is None else float(drawk_val),
        "sign_draw_0": s0,
        "sign_draw_k": sk,
        "preserved": preserved,
    }


def _build_draw_indices(target_lang, changed_views, screen_ids, audit_draw_buckets):
    """Map unique changed views onto sorted screen_ids for draw_1 and draw_2."""
    screen_id_to_row = {iid: idx for idx, iid in enumerate(screen_ids)}
    n_screen = len(screen_ids)
    check(n_screen > 0, "Nonempty screen_ids required")
    lang_buckets = audit_draw_buckets[target_lang]
    draw_indices = {}
    for dk in E07_DRAW_KEYS:
        dk_name = f"draw_{dk}"
        expected_changed = int(lang_buckets[dk_name]["changed_text_vs_draw0"])
        changed_row_map = {}
        for view_idx, view in enumerate(changed_views):
            if dk in view["draw_keys"]:
                iid = view["image_id"]
                check(iid in screen_id_to_row, f"Changed view image_id {iid!r} not in SCREEN IDs")
                s_idx = screen_id_to_row[iid]
                key = str(s_idx)
                check(
                    key not in changed_row_map,
                    f"Duplicate changed view mapping for ({target_lang}, {dk_name}, {iid})",
                )
                changed_row_map[key] = {
                    "changed_view_index": int(view_idx),
                    "image_id": iid,
                    "caption_id": view["caption_id"],
                    "caption_text_sha256": view["caption_text_sha256"],
                    "chunk_ref": plain(view["chunk_ref"]),
                }
        check(
            len(changed_row_map) == expected_changed,
            f"Mapped changed count mismatch for ({target_lang}, {dk_name}): "
            f"mapped={len(changed_row_map)} vs audit={expected_changed}",
        )
        unchanged_count = n_screen - expected_changed
        draw_indices[dk_name] = {
            "changed_text_vs_draw0": expected_changed,
            "unchanged_reused_from_draw0": unchanged_count,
            "unchanged_fraction": unchanged_count / float(n_screen),
            "changed_row_map": changed_row_map,
        }
    return draw_indices


def _normalize_resume_prediction_refs(resume_prediction_refs):
    """Validate optional mapping `'{pooling}:{target_lang}' -> ref`."""
    if not resume_prediction_refs:
        return {}
    check(
        isinstance(resume_prediction_refs, Mapping),
        "resume_prediction_refs must be a mapping of '{pooling}:{language}' -> ref",
    )
    valid_keys = {f"{p}:{lang}" for p in POOLINGS for lang in LANGUAGES}
    out = {}
    for key, ref in resume_prediction_refs.items():
        check(key in valid_keys, f"Invalid resume_prediction_refs key: {key!r}")
        out[key] = ref
    return out


def _verify_npy_file(path: Path, expected_sha256: str, expected_dtype: str, expected_shape: tuple):
    np = _require_numpy()
    check(path.is_file(), f"Missing array file: {path.name}")
    check(file_sha256(path) == expected_sha256, f"Array SHA-256 mismatch: {path.name}")
    arr = np.load(path, allow_pickle=False)
    check(
        isinstance(arr, np.ndarray)
        and arr.dtype == np.dtype(expected_dtype)
        and tuple(arr.shape) == tuple(expected_shape)
        and arr.flags.c_contiguous,
        f"Invalid array layout/dtype/shape in {path.name}: "
        f"got dtype={getattr(arr, 'dtype', None)}, shape={getattr(arr, 'shape', None)}, "
        f"expected dtype={expected_dtype}, shape={expected_shape}",
    )
    if arr.size > 0 and arr.dtype.kind == "f":
        check(np.isfinite(arr).all(), f"Nonfinite values in {path.name}")
    return arr


def _load_and_verify_e07_release(
    *,
    root: Path,
    protocol: Mapping[str, Any],
    plan,
    audit_ref,
    audit_body,
    e07_release_ref,
    model: str,
    depth: int,
):
    """Authenticate `phase2_e07_release` and every referenced `phase2_e07_chunk`."""
    rel_ref, rel_body, _ = load_phase2_artifact(
        root,
        protocol,
        e07_release_ref,
        KIND_E07_RELEASE,
        expected_schema=SCHEMA_E07_FEATURES,
        expected_dataset=plan["dataset_fingerprint"],
        expected_plan_identity=plan["plan_identity"],
        expected_owner=model,
    )
    check(rel_body.get("experiment") == "E07", "E07 release experiment mismatch")
    check(rel_body.get("model") == model, "E07 release model mismatch")
    check(int(rel_body.get("midpoint_depth", -1)) == depth, "E07 release midpoint_depth mismatch")
    check(rel_body.get("condition") == E07_CONDITION, "E07 release condition mismatch")
    check(list(rel_body.get("poolings", [])) == list(POOLINGS), "E07 release poolings mismatch")
    check(rel_body.get("split") == E07_SPLIT, "E07 release split mismatch")
    check(rel_body.get("confirm_excluded") is True, "E07 release confirm_excluded must be True")
    check(plain(rel_body.get("e07_audit_ref")) == plain(audit_ref), "E07 release e07_audit_ref mismatch")
    check(rel_body.get("e07_verdict") == audit_body["verdict"], "E07 release verdict mismatch")

    rc_summary = rel_body.get("repeat_check_summary")
    check(
        isinstance(rc_summary, Mapping) and rc_summary.get("all_passed") is True,
        "E07 release repeat_check_summary did not pass",
    )

    ep = audit_body["extraction_plan"]
    ep_chunks = ep["chunks"]
    rel_chunks = rel_body.get("chunks")
    check(
        isinstance(rel_chunks, (list, tuple)) and len(rel_chunks) == len(ep_chunks),
        "E07 release chunk count differs from audit extraction_plan",
    )
    check(
        int(rc_summary.get("chunks_checked", -1)) == len(rel_chunks),
        "E07 release repeat_check_summary chunks_checked mismatch",
    )
    check(
        int(rel_body.get("total_unique_views", -1)) == int(ep["total_unique_views"]),
        "E07 release total_unique_views mismatch",
    )
    check(
        plain(rel_body.get("per_language_counts")) == plain(ep["per_language_counts"]),
        "E07 release per_language_counts mismatch",
    )

    dimension = int(rel_body["dimension"])
    check(dimension > 0, "E07 release dimension must be positive")
    model_fp = rel_body["model_fingerprint"]
    exec_fp = rel_body["execution_fingerprint"]

    views_by_lang = {lang: [] for lang in LANGUAGES}
    vectors_by_pooling_lang = {p: {lang: [] for lang in LANGUAGES} for p in POOLINGS}
    chunk_refs_by_lang = {lang: [] for lang in LANGUAGES}

    for idx, (rel_chunk, ep_chunk) in enumerate(zip(rel_chunks, ep_chunks)):
        check(int(rel_chunk.get("chunk_index", -1)) == idx, f"E07 release chunk_index order mismatch at {idx}")
        chunk_ref, chunk_body, chunk_marker = load_phase2_artifact(
            root,
            protocol,
            rel_chunk["ref"],
            KIND_E07_CHUNK,
            expected_schema=SCHEMA_E07_FEATURES,
            expected_dataset=plan["dataset_fingerprint"],
            expected_plan_identity=plan["plan_identity"],
            expected_owner=model,
        )
        check(chunk_body.get("experiment") == "E07", f"Chunk {idx} experiment mismatch")
        check(chunk_body.get("model") == model, f"Chunk {idx} model mismatch")
        check(int(chunk_body.get("midpoint_depth", -1)) == depth, f"Chunk {idx} midpoint_depth mismatch")
        check(chunk_body.get("condition") == E07_CONDITION, f"Chunk {idx} condition mismatch")
        check(list(chunk_body.get("poolings", [])) == list(POOLINGS), f"Chunk {idx} poolings mismatch")
        check(chunk_body.get("split") == E07_SPLIT, f"Chunk {idx} split mismatch")
        check(chunk_body.get("confirm_excluded") is True, f"Chunk {idx} confirm_excluded must be True")
        check(plain(chunk_body.get("e07_audit_ref")) == plain(audit_ref), f"Chunk {idx} e07_audit_ref mismatch")
        check(int(chunk_body.get("chunk_index", -1)) == idx, f"Chunk {idx} chunk_index mismatch")
        check(int(chunk_body.get("total_chunks", -1)) == len(rel_chunks), f"Chunk {idx} total_chunks mismatch")
        check(int(chunk_body.get("dimension", -1)) == dimension, f"Chunk {idx} dimension mismatch")
        check(chunk_body.get("model_fingerprint") == model_fp, f"Chunk {idx} model_fingerprint mismatch")
        check(chunk_body.get("execution_fingerprint") == exec_fp, f"Chunk {idx} execution_fingerprint mismatch")
        chunk_rc = chunk_body.get("repeat_check")
        check(isinstance(chunk_rc, Mapping) and chunk_rc.get("passed") is True, f"Chunk {idx} repeat_check failed")

        lang = chunk_body.get("language")
        check(lang in LANGUAGES and lang == ep_chunk["language"] == rel_chunk.get("language"),
              f"Chunk {idx} language mismatch")
        n_views = int(chunk_body.get("n_views", -1))
        check(n_views == len(ep_chunk["rows"]) == int(rel_chunk.get("n_views", -1)) and n_views > 0,
              f"Chunk {idx} n_views mismatch")

        chunk_folder = within(root, chunk_ref["path"]).parent
        check("rows.jsonl" in chunk_marker["files"], f"Chunk {idx} missing rows.jsonl")
        rows_path = chunk_folder / "rows.jsonl"
        rows_sha = file_sha256(rows_path)
        check(
            rows_sha == chunk_marker["files"]["rows.jsonl"] == rel_chunk["rows_sha256"],
            f"Chunk {idx} rows.jsonl SHA-256 mismatch",
        )
        if "rows_sha256" in chunk_body:
            check(chunk_body["rows_sha256"] == rows_sha, f"Chunk {idx} body rows_sha256 mismatch")

        chunk_rows = [
            json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        check(len(chunk_rows) == n_views, f"Chunk {idx} rows.jsonl length mismatch")

        for r_i, (row, ep_row) in enumerate(zip(chunk_rows, ep_chunk["rows"])):
            require_open_split(row.get("split"))
            check(row.get("split") == E07_SPLIT, f"Chunk {idx} row {r_i} is not SCREEN")
            check(row.get("language") == lang, f"Chunk {idx} row {r_i} language mismatch")
            check(row.get("image_id") == ep_row["image_id"], f"Chunk {idx} row {r_i} image_id mismatch")
            check(row.get("caption_id") == ep_row["caption_id"], f"Chunk {idx} row {r_i} caption_id mismatch")
            text_sha = hashlib.sha256(row["caption_text"].encode("utf-8")).hexdigest()
            check(
                text_sha == row.get("caption_text_sha256") == ep_row["caption_text_sha256"],
                f"Chunk {idx} row {r_i} caption_text_sha256 mismatch",
            )
            check(list(row.get("draw_keys", [])) == list(ep_row["draw_keys"]),
                  f"Chunk {idx} row {r_i} draw_keys mismatch")
            views_by_lang[lang].append({
                "image_id": row["image_id"],
                "language": lang,
                "caption_id": row["caption_id"],
                "caption_text_sha256": row["caption_text_sha256"],
                "draw_keys": [int(dk) for dk in row["draw_keys"]],
                "chunk_ref": plain(chunk_ref),
                "chunk_index": idx,
                "chunk_row_index": r_i,
            })

        for pooling in POOLINGS:
            fname = f"{pooling}.npy"
            check(fname in chunk_marker["files"], f"Chunk {idx} missing {fname}")
            arr = _verify_npy_file(
                chunk_folder / fname,
                chunk_marker["files"][fname],
                "<f4",
                (n_views, dimension),
            )
            vectors_by_pooling_lang[pooling][lang].append(arr)

        chunk_refs_by_lang[lang].append(plain(chunk_ref))

    np = _require_numpy()
    stacked_by_pooling_lang = {p: {} for p in POOLINGS}
    for lang in LANGUAGES:
        expected_lang_count = int(rel_body["per_language_counts"][lang])
        check(
            len(views_by_lang[lang]) == expected_lang_count,
            f"Collected view count mismatch for {lang}: {len(views_by_lang[lang])} vs {expected_lang_count}",
        )
        for pooling in POOLINGS:
            parts = vectors_by_pooling_lang[pooling][lang]
            if parts:
                stacked_by_pooling_lang[pooling][lang] = np.ascontiguousarray(
                    np.concatenate(parts, axis=0), dtype="<f4"
                )
            else:
                stacked_by_pooling_lang[pooling][lang] = np.empty((0, dimension), dtype="<f4")

    return (
        rel_ref,
        rel_body,
        dimension,
        views_by_lang,
        stacked_by_pooling_lang,
        chunk_refs_by_lang,
    )


def _interval_record(draw_values):
    from src.stage4.metrics import bootstrap_interval

    interval = bootstrap_interval(draw_values)
    if interval["valid_draws"] == 0:
        interval["reason"] = "no_valid_bootstrap_draws"
    return interval


def _category_status_and_reason(value, pos_n, neg_n, minimum, unestimable):
    from src.stage4.metrics import support_status

    status = support_status(value, pos_n, neg_n, minimum)
    if status == "ok":
        return status, None
    if status == "descriptive_low_support":
        return status, f"support_below_minimum_{minimum}"
    if unestimable:
        return status, "unestimable_head"
    return status, "undefined_metric_or_zero_denominator"


def _summarize_macro_and_categories(
    *,
    categories,
    macro_indices,
    cat_values,
    cat_draws,
    cat_supports,
    cat_unestimable,
    minimum,
    include_categories=True,
):
    """Build per-category table (optional) and equal-weight macro over `macro_indices`."""
    np = _require_numpy()
    from src.stage4.metrics import fixed_macro

    per_category = None
    if include_categories:
        per_category = {}
        for col, cat in enumerate(categories):
            known_n, pos_n, neg_n = cat_supports[col]
            val = cat_values[col]
            status, reason = _category_status_and_reason(
                val, pos_n, neg_n, minimum, bool(cat_unestimable[col])
            )
            per_category[cat] = {
                "value": None if val is None else float(val),
                "interval": _interval_record(cat_draws[col]),
                "status": status,
                "reason": reason,
                "support": {"known": int(known_n), "positive": int(pos_n), "negative": int(neg_n)},
            }

    if not macro_indices:
        total_draws = int(len(cat_draws[0]))
        nan_draws = np.full(total_draws, np.nan, dtype="f8")
        macro_summary = {
            "value": None,
            "interval": _interval_record(nan_draws),
            "status": "unavailable",
            "reason": "no_supported_categories_in_cohort",
            "support": {"categories": 0, "defined_categories": 0, "ok_categories": 0},
        }
        return macro_summary, nan_draws, per_category

    macro_vals = [cat_values[idx] for idx in macro_indices]
    defined_count = sum(v is not None for v in macro_vals)
    ok_count = sum(
        cat_values[idx] is not None and min(cat_supports[idx][1], cat_supports[idx][2]) >= minimum
        for idx in macro_indices
    )
    if defined_count < len(macro_indices):
        macro_val = None
        macro_status = "undefined"
        macro_reason = "unresolved_macro_cohort"
    else:
        macro_val = float(fixed_macro([0.0 if v is None else v for v in cat_values], macro_indices))
        if ok_count == len(macro_indices):
            macro_status = "ok"
            macro_reason = None
        else:
            macro_status = "descriptive_low_support"
            macro_reason = f"macro_category_support_below_minimum_{minimum}"

    stacked = np.stack(cat_draws, axis=0)
    macro_draws = fixed_macro(stacked, macro_indices, axis=0)
    macro_summary = {
        "value": macro_val,
        "interval": _interval_record(macro_draws),
        "status": macro_status,
        "reason": macro_reason,
        "support": {
            "categories": len(macro_indices),
            "defined_categories": defined_count,
            "ok_categories": ok_count,
        },
    }
    return macro_summary, macro_draws, per_category


def _evaluate_arm_logits(logits, estimable, y, known, counts, n_cats, total_draws):
    """Evaluate one (n_images, n_cats) logit matrix against (y, known, counts)."""
    np = _require_numpy()
    from src.stage4.metrics import METRICS, binary_metrics, bootstrap_binary_metrics, probabilities

    probs = probabilities(logits)
    est = np.asarray(estimable, dtype=bool)
    check(est.shape == (n_cats,), f"Unexpected estimable shape: {est.shape}")
    nan_draws = np.full(total_draws, np.nan, dtype="f8")

    points = {m: [] for m in METRICS}
    draws = {m: [] for m in METRICS}
    supports = []
    for col in range(n_cats):
        y_col = y[:, col]
        known_col = known[:, col]
        known_n = int(known_col.sum())
        pos_n = int(y_col[known_col].sum())
        neg_n = known_n - pos_n
        supports.append((known_n, pos_n, neg_n))
        if est[col]:
            p_col = probs[:, col]
            bm = binary_metrics(y_col, p_col, known_col)
            bd = bootstrap_binary_metrics(y_col, p_col, known_col, counts)
            for m in METRICS:
                points[m].append(bm["metrics"][m])
                draws[m].append(bd[m])
        else:
            for m in METRICS:
                points[m].append(None)
                draws[m].append(nan_draws)
    return {
        "points": points,
        "draws": draws,
        "supports": supports,
        "unestimable": [not bool(est[col]) for col in range(n_cats)],
    }


def _evaluate_full_screen_draw(
    *,
    logits_by_pooling_target,
    estimable_by_pooling_target,
    y,
    known,
    counts,
    categories,
    primary_indices,
    total_draws,
    minimum,
    eligible_targets,
):
    """Evaluate one full 360-image SCREEN draw across both poolings and all 16 routes."""
    np = _require_numpy()
    from src.stage4.metrics import METRICS

    n_cats = len(categories)
    arms = {}
    for pooling in POOLINGS:
        for t in LANGUAGES:
            t_logits = logits_by_pooling_target[pooling][t]
            t_est = estimable_by_pooling_target[pooling][t]
            for s_idx, s in enumerate(LANGUAGES):
                arms[(pooling, s, t)] = _evaluate_arm_logits(
                    t_logits[s_idx], t_est[s_idx], y, known, counts, n_cats, total_draws
                )

    routes_all = {pooling: {} for pooling in POOLINGS}
    diagonals = {}
    per_route = {}
    twelve_route_means = {}
    changed_targets_only_means = {}

    # Retain compact per-draw arrays in memory for paired draw_k vs draw_0 shifts
    # (never persisted to JSON).
    cache = {
        "route_macro_vals": {},
        "route_macro_draws": {},
        "route_cat_vals": {},
        "route_cat_draws": {},
        "diag_delta_d_macro_vals": {},
        "diag_delta_d_macro_draws": {},
        "diag_delta_d_cat_vals": {},
        "diag_delta_d_cat_draws": {},
        "gap_macro_vals": {},
        "gap_macro_draws": {},
        "gap_cat_vals": {},
        "gap_cat_draws": {},
        "delta_g_macro_vals": {},
        "delta_g_macro_draws": {},
        "delta_g_cat_vals": {},
        "delta_g_cat_draws": {},
    }

    off_12 = _off_diagonal_pairs(LANGUAGES)
    off_changed = _off_diagonal_pairs(eligible_targets)

    def _mean_summary(macro_vals, macro_draws_list, cat_vals_by_c, cat_draws_by_c):
        if not macro_vals:
            nan_draws = np.full(total_draws, np.nan, dtype="f8")
            return {
                "primary_macro": {"value": None, "interval": _interval_record(nan_draws)},
                "categories": {
                    cat: {"value": None, "interval": _interval_record(nan_draws)} for cat in categories
                },
            }
        m_val = None if any(v is None for v in macro_vals) else float(sum(macro_vals) / len(macro_vals))
        m_draws = np.mean(np.stack(macro_draws_list, axis=0), axis=0)
        cat_table = {}
        for c, cat in enumerate(categories):
            c_vals = cat_vals_by_c[c]
            c_val = None if any(v is None for v in c_vals) else float(sum(c_vals) / len(c_vals))
            c_draws = np.mean(np.stack(cat_draws_by_c[c], axis=0), axis=0)
            cat_table[cat] = {"value": c_val, "interval": _interval_record(c_draws)}
        return {
            "primary_macro": {"value": m_val, "interval": _interval_record(m_draws)},
            "categories": cat_table,
        }

    for metric in METRICS:
        # 1. All 16 routes per pooling
        for pooling in POOLINGS:
            for s in LANGUAGES:
                for t in LANGUAGES:
                    rkey = _route_key(s, t)
                    arm = arms[(pooling, s, t)]
                    macro_summary, macro_draws, cat_table = _summarize_macro_and_categories(
                        categories=categories,
                        macro_indices=primary_indices,
                        cat_values=arm["points"][metric],
                        cat_draws=arm["draws"][metric],
                        cat_supports=arm["supports"],
                        cat_unestimable=arm["unestimable"],
                        minimum=minimum,
                    )
                    cache["route_macro_vals"][(metric, pooling, s, t)] = macro_summary["value"]
                    cache["route_macro_draws"][(metric, pooling, s, t)] = macro_draws
                    cache["route_cat_vals"][(metric, pooling, s, t)] = arm["points"][metric]
                    cache["route_cat_draws"][(metric, pooling, s, t)] = arm["draws"][metric]

                    entry = routes_all[pooling].setdefault(
                        rkey,
                        {
                            "route": rkey,
                            "source_language": s,
                            "target_language": t,
                            "supervision_label": SUPERVISION_DIAGONAL if s == t else SUPERVISION_OFF_DIAGONAL,
                            "metrics": {},
                        },
                    )
                    entry["metrics"][metric] = {
                        "primary_macro": macro_summary,
                        "categories": cat_table,
                    }

        # 2. 4 receiving diagonals (t -> t) + delta_d
        for t in LANGUAGES:
            last_arm = arms[("last_caption", t, t)]
            mean_arm = arms[("caption_mean", t, t)]
            gain_vals = [
                _diag_gain_value(metric, last_arm["points"][metric][c], mean_arm["points"][metric][c])
                for c in range(n_cats)
            ]
            gain_draws = [
                _diag_gain_draws(metric, last_arm["draws"][metric][c], mean_arm["draws"][metric][c])
                for c in range(n_cats)
            ]
            gain_unest = [last_arm["unestimable"][c] or mean_arm["unestimable"][c] for c in range(n_cats)]
            gain_macro, gain_mdraws, gain_cats = _summarize_macro_and_categories(
                categories=categories,
                macro_indices=primary_indices,
                cat_values=gain_vals,
                cat_draws=gain_draws,
                cat_supports=mean_arm["supports"],
                cat_unestimable=gain_unest,
                minimum=minimum,
            )
            cache["diag_delta_d_macro_vals"][(metric, t)] = gain_macro["value"]
            cache["diag_delta_d_macro_draws"][(metric, t)] = gain_mdraws
            cache["diag_delta_d_cat_vals"][(metric, t)] = gain_vals
            cache["diag_delta_d_cat_draws"][(metric, t)] = gain_draws

            diag_entry = diagonals.setdefault(
                t,
                {
                    "route": _route_key(t, t),
                    "language": t,
                    "supervision_label": SUPERVISION_DIAGONAL,
                    "metrics": {},
                },
            )
            diag_entry["metrics"][metric] = {
                "last_caption": routes_all["last_caption"][_route_key(t, t)]["metrics"][metric],
                "caption_mean": routes_all["caption_mean"][_route_key(t, t)]["metrics"][metric],
                "delta_d": {"primary_macro": gain_macro, "categories": gain_cats},
            }

        # 3. 12 off-diagonal routes (s != t): g_last, g_res, delta_g
        for s, t in off_12:
            rkey = _route_key(s, t)
            route_last = arms[("last_caption", s, t)]
            diag_last = arms[("last_caption", t, t)]
            route_mean = arms[("caption_mean", s, t)]
            diag_mean = arms[("caption_mean", t, t)]

            g_last_vals = [
                _gap_value(metric, diag_last["points"][metric][c], route_last["points"][metric][c])
                for c in range(n_cats)
            ]
            g_last_draws = [
                _gap_draws(metric, diag_last["draws"][metric][c], route_last["draws"][metric][c])
                for c in range(n_cats)
            ]
            g_last_unest = [diag_last["unestimable"][c] or route_last["unestimable"][c] for c in range(n_cats)]
            g_last_macro, g_last_mdraws, g_last_cats = _summarize_macro_and_categories(
                categories=categories,
                macro_indices=primary_indices,
                cat_values=g_last_vals,
                cat_draws=g_last_draws,
                cat_supports=route_last["supports"],
                cat_unestimable=g_last_unest,
                minimum=minimum,
            )

            g_res_vals = [
                _gap_value(metric, diag_mean["points"][metric][c], route_mean["points"][metric][c])
                for c in range(n_cats)
            ]
            g_res_draws = [
                _gap_draws(metric, diag_mean["draws"][metric][c], route_mean["draws"][metric][c])
                for c in range(n_cats)
            ]
            g_res_unest = [diag_mean["unestimable"][c] or route_mean["unestimable"][c] for c in range(n_cats)]
            g_res_macro, g_res_mdraws, g_res_cats = _summarize_macro_and_categories(
                categories=categories,
                macro_indices=primary_indices,
                cat_values=g_res_vals,
                cat_draws=g_res_draws,
                cat_supports=route_mean["supports"],
                cat_unestimable=g_res_unest,
                minimum=minimum,
            )

            delta_g_vals = [_diff_value(g_last_vals[c], g_res_vals[c]) for c in range(n_cats)]
            delta_g_draws = [g_last_draws[c] - g_res_draws[c] for c in range(n_cats)]
            delta_g_unest = [g_last_unest[c] or g_res_unest[c] for c in range(n_cats)]
            delta_g_macro, delta_g_mdraws, delta_g_cats = _summarize_macro_and_categories(
                categories=categories,
                macro_indices=primary_indices,
                cat_values=delta_g_vals,
                cat_draws=delta_g_draws,
                cat_supports=route_mean["supports"],
                cat_unestimable=delta_g_unest,
                minimum=minimum,
            )

            cache["gap_macro_vals"][(metric, "last_caption", s, t)] = g_last_macro["value"]
            cache["gap_macro_draws"][(metric, "last_caption", s, t)] = g_last_mdraws
            cache["gap_cat_vals"][(metric, "last_caption", s, t)] = g_last_vals
            cache["gap_cat_draws"][(metric, "last_caption", s, t)] = g_last_draws

            cache["gap_macro_vals"][(metric, "caption_mean", s, t)] = g_res_macro["value"]
            cache["gap_macro_draws"][(metric, "caption_mean", s, t)] = g_res_mdraws
            cache["gap_cat_vals"][(metric, "caption_mean", s, t)] = g_res_vals
            cache["gap_cat_draws"][(metric, "caption_mean", s, t)] = g_res_draws

            cache["delta_g_macro_vals"][(metric, s, t)] = delta_g_macro["value"]
            cache["delta_g_macro_draws"][(metric, s, t)] = delta_g_mdraws
            cache["delta_g_cat_vals"][(metric, s, t)] = delta_g_vals
            cache["delta_g_cat_draws"][(metric, s, t)] = delta_g_draws

            route_entry = per_route.setdefault(
                rkey,
                {
                    "route": rkey,
                    "source_language": s,
                    "target_language": t,
                    "supervision_label": SUPERVISION_OFF_DIAGONAL,
                    "metrics": {},
                },
            )
            route_entry["metrics"][metric] = {
                "g_last": {"primary_macro": g_last_macro, "categories": g_last_cats},
                "g_res": {"primary_macro": g_res_macro, "categories": g_res_cats},
                "delta_g": {"primary_macro": delta_g_macro, "categories": delta_g_cats},
            }

        # 4. Equal-weight means for all 12 routes and for changed_targets_only routes
        def _aggregate_pairs(pairs):
            return {
                "g_last_mean": _mean_summary(
                    [cache["gap_macro_vals"][(metric, "last_caption", s, t)] for s, t in pairs],
                    [cache["gap_macro_draws"][(metric, "last_caption", s, t)] for s, t in pairs],
                    [[cache["gap_cat_vals"][(metric, "last_caption", s, t)][c] for s, t in pairs] for c in range(n_cats)],
                    [[cache["gap_cat_draws"][(metric, "last_caption", s, t)][c] for s, t in pairs] for c in range(n_cats)],
                ),
                "g_res_mean": _mean_summary(
                    [cache["gap_macro_vals"][(metric, "caption_mean", s, t)] for s, t in pairs],
                    [cache["gap_macro_draws"][(metric, "caption_mean", s, t)] for s, t in pairs],
                    [[cache["gap_cat_vals"][(metric, "caption_mean", s, t)][c] for s, t in pairs] for c in range(n_cats)],
                    [[cache["gap_cat_draws"][(metric, "caption_mean", s, t)][c] for s, t in pairs] for c in range(n_cats)],
                ),
                "delta_g_mean": _mean_summary(
                    [cache["delta_g_macro_vals"][(metric, s, t)] for s, t in pairs],
                    [cache["delta_g_macro_draws"][(metric, s, t)] for s, t in pairs],
                    [[cache["delta_g_cat_vals"][(metric, s, t)][c] for s, t in pairs] for c in range(n_cats)],
                    [[cache["delta_g_cat_draws"][(metric, s, t)][c] for s, t in pairs] for c in range(n_cats)],
                ),
                "transfer_last_mean": _mean_summary(
                    [cache["route_macro_vals"][(metric, "last_caption", s, t)] for s, t in pairs],
                    [cache["route_macro_draws"][(metric, "last_caption", s, t)] for s, t in pairs],
                    [[cache["route_cat_vals"][(metric, "last_caption", s, t)][c] for s, t in pairs] for c in range(n_cats)],
                    [[cache["route_cat_draws"][(metric, "last_caption", s, t)][c] for s, t in pairs] for c in range(n_cats)],
                ),
                "transfer_mean_mean": _mean_summary(
                    [cache["route_macro_vals"][(metric, "caption_mean", s, t)] for s, t in pairs],
                    [cache["route_macro_draws"][(metric, "caption_mean", s, t)] for s, t in pairs],
                    [[cache["route_cat_vals"][(metric, "caption_mean", s, t)][c] for s, t in pairs] for c in range(n_cats)],
                    [[cache["route_cat_draws"][(metric, "caption_mean", s, t)][c] for s, t in pairs] for c in range(n_cats)],
                ),
            }

        twelve_route_means[metric] = _aggregate_pairs(off_12)
        changed_targets_only_means[metric] = _aggregate_pairs(off_changed)

    summary = {
        "twelve_route_means": twelve_route_means,
        "changed_targets_only": {
            "eligible_target_languages": list(eligible_targets),
            "n_routes": len(off_changed),
            "means": changed_targets_only_means,
        },
        "diagonals": diagonals,
        "per_route": per_route,
        "routes_all": routes_all,
    }
    return summary, cache


def _compute_full_screen_paired_shift(
    *,
    draw0_cache,
    drawk_cache,
    categories,
    total_draws,
    eligible_targets,
):
    """Compute within-draw paired shifts `draw_k - draw_0` on full_screen_360."""
    np = _require_numpy()
    from src.stage4.metrics import METRICS

    n_cats = len(categories)
    off_12 = _off_diagonal_pairs(LANGUAGES)
    off_changed = _off_diagonal_pairs(eligible_targets)

    def _shift_summary(m_val_0, m_val_k, m_draws_0, m_draws_k, c_vals_0, c_vals_k, c_draws_0, c_draws_k):
        cat_table = {}
        for c, cat in enumerate(categories):
            cat_table[cat] = {
                "value": _diff_value(c_vals_k[c], c_vals_0[c]),
                "interval": _interval_record(c_draws_k[c] - c_draws_0[c]),
            }
        return {
            "primary_macro": {
                "value": _diff_value(m_val_k, m_val_0),
                "interval": _interval_record(m_draws_k - m_draws_0),
            },
            "categories": cat_table,
        }

    def _mean_shift_summary(vals_0, vals_k, draws_0_list, draws_k_list, cat_vals_0, cat_vals_k, cat_draws_0, cat_draws_k):
        if not vals_0:
            nan_draws = np.full(total_draws, np.nan, dtype="f8")
            return {
                "primary_macro": {"value": None, "interval": _interval_record(nan_draws)},
                "categories": {
                    cat: {"value": None, "interval": _interval_record(nan_draws)} for cat in categories
                },
            }
        m0 = None if any(v is None for v in vals_0) else float(sum(vals_0) / len(vals_0))
        mk = None if any(v is None for v in vals_k) else float(sum(vals_k) / len(vals_k))
        d0 = np.mean(np.stack(draws_0_list, axis=0), axis=0)
        dk = np.mean(np.stack(draws_k_list, axis=0), axis=0)
        cat_table = {}
        for c, cat in enumerate(categories):
            c0 = None if any(v is None for v in cat_vals_0[c]) else float(sum(cat_vals_0[c]) / len(cat_vals_0[c]))
            ck = None if any(v is None for v in cat_vals_k[c]) else float(sum(cat_vals_k[c]) / len(cat_vals_k[c]))
            cd0 = np.mean(np.stack(cat_draws_0[c], axis=0), axis=0)
            cdk = np.mean(np.stack(cat_draws_k[c], axis=0), axis=0)
            cat_table[cat] = {
                "value": _diff_value(ck, c0),
                "interval": _interval_record(cdk - cd0),
            }
        return {
            "primary_macro": {
                "value": _diff_value(mk, m0),
                "interval": _interval_record(dk - d0),
            },
            "categories": cat_table,
        }

    twelve_route_mean = {}
    changed_targets_only_mean = {}
    per_route = {}
    diagonals = {}

    for metric in METRICS:
        # Diagonals
        for t in LANGUAGES:
            d_entry = diagonals.setdefault(
                t,
                {"route": _route_key(t, t), "language": t, "supervision_label": SUPERVISION_DIAGONAL, "metrics": {}},
            )
            d_entry["metrics"][metric] = {
                "last_caption_shift": _shift_summary(
                    draw0_cache["route_macro_vals"][(metric, "last_caption", t, t)],
                    drawk_cache["route_macro_vals"][(metric, "last_caption", t, t)],
                    draw0_cache["route_macro_draws"][(metric, "last_caption", t, t)],
                    drawk_cache["route_macro_draws"][(metric, "last_caption", t, t)],
                    draw0_cache["route_cat_vals"][(metric, "last_caption", t, t)],
                    drawk_cache["route_cat_vals"][(metric, "last_caption", t, t)],
                    draw0_cache["route_cat_draws"][(metric, "last_caption", t, t)],
                    drawk_cache["route_cat_draws"][(metric, "last_caption", t, t)],
                ),
                "caption_mean_shift": _shift_summary(
                    draw0_cache["route_macro_vals"][(metric, "caption_mean", t, t)],
                    drawk_cache["route_macro_vals"][(metric, "caption_mean", t, t)],
                    draw0_cache["route_macro_draws"][(metric, "caption_mean", t, t)],
                    drawk_cache["route_macro_draws"][(metric, "caption_mean", t, t)],
                    draw0_cache["route_cat_vals"][(metric, "caption_mean", t, t)],
                    drawk_cache["route_cat_vals"][(metric, "caption_mean", t, t)],
                    draw0_cache["route_cat_draws"][(metric, "caption_mean", t, t)],
                    drawk_cache["route_cat_draws"][(metric, "caption_mean", t, t)],
                ),
                "delta_d_shift": _shift_summary(
                    draw0_cache["diag_delta_d_macro_vals"][(metric, t)],
                    drawk_cache["diag_delta_d_macro_vals"][(metric, t)],
                    draw0_cache["diag_delta_d_macro_draws"][(metric, t)],
                    drawk_cache["diag_delta_d_macro_draws"][(metric, t)],
                    draw0_cache["diag_delta_d_cat_vals"][(metric, t)],
                    drawk_cache["diag_delta_d_cat_vals"][(metric, t)],
                    draw0_cache["diag_delta_d_cat_draws"][(metric, t)],
                    drawk_cache["diag_delta_d_cat_draws"][(metric, t)],
                ),
            }

        # Per-route shifts
        for s, t in off_12:
            rkey = _route_key(s, t)
            r_entry = per_route.setdefault(
                rkey,
                {
                    "route": rkey,
                    "source_language": s,
                    "target_language": t,
                    "supervision_label": SUPERVISION_OFF_DIAGONAL,
                    "metrics": {},
                },
            )
            r_entry["metrics"][metric] = {
                "g_last_shift": _shift_summary(
                    draw0_cache["gap_macro_vals"][(metric, "last_caption", s, t)],
                    drawk_cache["gap_macro_vals"][(metric, "last_caption", s, t)],
                    draw0_cache["gap_macro_draws"][(metric, "last_caption", s, t)],
                    drawk_cache["gap_macro_draws"][(metric, "last_caption", s, t)],
                    draw0_cache["gap_cat_vals"][(metric, "last_caption", s, t)],
                    drawk_cache["gap_cat_vals"][(metric, "last_caption", s, t)],
                    draw0_cache["gap_cat_draws"][(metric, "last_caption", s, t)],
                    drawk_cache["gap_cat_draws"][(metric, "last_caption", s, t)],
                ),
                "g_res_shift": _shift_summary(
                    draw0_cache["gap_macro_vals"][(metric, "caption_mean", s, t)],
                    drawk_cache["gap_macro_vals"][(metric, "caption_mean", s, t)],
                    draw0_cache["gap_macro_draws"][(metric, "caption_mean", s, t)],
                    drawk_cache["gap_macro_draws"][(metric, "caption_mean", s, t)],
                    draw0_cache["gap_cat_vals"][(metric, "caption_mean", s, t)],
                    drawk_cache["gap_cat_vals"][(metric, "caption_mean", s, t)],
                    draw0_cache["gap_cat_draws"][(metric, "caption_mean", s, t)],
                    drawk_cache["gap_cat_draws"][(metric, "caption_mean", s, t)],
                ),
                "delta_g_shift": _shift_summary(
                    draw0_cache["delta_g_macro_vals"][(metric, s, t)],
                    drawk_cache["delta_g_macro_vals"][(metric, s, t)],
                    draw0_cache["delta_g_macro_draws"][(metric, s, t)],
                    drawk_cache["delta_g_macro_draws"][(metric, s, t)],
                    draw0_cache["delta_g_cat_vals"][(metric, s, t)],
                    drawk_cache["delta_g_cat_vals"][(metric, s, t)],
                    draw0_cache["delta_g_cat_draws"][(metric, s, t)],
                    drawk_cache["delta_g_cat_draws"][(metric, s, t)],
                ),
                "transfer_last_shift": _shift_summary(
                    draw0_cache["route_macro_vals"][(metric, "last_caption", s, t)],
                    drawk_cache["route_macro_vals"][(metric, "last_caption", s, t)],
                    draw0_cache["route_macro_draws"][(metric, "last_caption", s, t)],
                    drawk_cache["route_macro_draws"][(metric, "last_caption", s, t)],
                    draw0_cache["route_cat_vals"][(metric, "last_caption", s, t)],
                    drawk_cache["route_cat_vals"][(metric, "last_caption", s, t)],
                    draw0_cache["route_cat_draws"][(metric, "last_caption", s, t)],
                    drawk_cache["route_cat_draws"][(metric, "last_caption", s, t)],
                ),
                "transfer_mean_shift": _shift_summary(
                    draw0_cache["route_macro_vals"][(metric, "caption_mean", s, t)],
                    drawk_cache["route_macro_vals"][(metric, "caption_mean", s, t)],
                    draw0_cache["route_macro_draws"][(metric, "caption_mean", s, t)],
                    drawk_cache["route_macro_draws"][(metric, "caption_mean", s, t)],
                    draw0_cache["route_cat_vals"][(metric, "caption_mean", s, t)],
                    drawk_cache["route_cat_vals"][(metric, "caption_mean", s, t)],
                    draw0_cache["route_cat_draws"][(metric, "caption_mean", s, t)],
                    drawk_cache["route_cat_draws"][(metric, "caption_mean", s, t)],
                ),
            }

        def _agg_shift(pairs, diag_langs):
            return {
                "g_last_shift": _mean_shift_summary(
                    [draw0_cache["gap_macro_vals"][(metric, "last_caption", s, t)] for s, t in pairs],
                    [drawk_cache["gap_macro_vals"][(metric, "last_caption", s, t)] for s, t in pairs],
                    [draw0_cache["gap_macro_draws"][(metric, "last_caption", s, t)] for s, t in pairs],
                    [drawk_cache["gap_macro_draws"][(metric, "last_caption", s, t)] for s, t in pairs],
                    [[draw0_cache["gap_cat_vals"][(metric, "last_caption", s, t)][c] for s, t in pairs] for c in range(n_cats)],
                    [[drawk_cache["gap_cat_vals"][(metric, "last_caption", s, t)][c] for s, t in pairs] for c in range(n_cats)],
                    [[draw0_cache["gap_cat_draws"][(metric, "last_caption", s, t)][c] for s, t in pairs] for c in range(n_cats)],
                    [[drawk_cache["gap_cat_draws"][(metric, "last_caption", s, t)][c] for s, t in pairs] for c in range(n_cats)],
                ),
                "g_res_shift": _mean_shift_summary(
                    [draw0_cache["gap_macro_vals"][(metric, "caption_mean", s, t)] for s, t in pairs],
                    [drawk_cache["gap_macro_vals"][(metric, "caption_mean", s, t)] for s, t in pairs],
                    [draw0_cache["gap_macro_draws"][(metric, "caption_mean", s, t)] for s, t in pairs],
                    [drawk_cache["gap_macro_draws"][(metric, "caption_mean", s, t)] for s, t in pairs],
                    [[draw0_cache["gap_cat_vals"][(metric, "caption_mean", s, t)][c] for s, t in pairs] for c in range(n_cats)],
                    [[drawk_cache["gap_cat_vals"][(metric, "caption_mean", s, t)][c] for s, t in pairs] for c in range(n_cats)],
                    [[draw0_cache["gap_cat_draws"][(metric, "caption_mean", s, t)][c] for s, t in pairs] for c in range(n_cats)],
                    [[drawk_cache["gap_cat_draws"][(metric, "caption_mean", s, t)][c] for s, t in pairs] for c in range(n_cats)],
                ),
                "delta_g_shift": _mean_shift_summary(
                    [draw0_cache["delta_g_macro_vals"][(metric, s, t)] for s, t in pairs],
                    [drawk_cache["delta_g_macro_vals"][(metric, s, t)] for s, t in pairs],
                    [draw0_cache["delta_g_macro_draws"][(metric, s, t)] for s, t in pairs],
                    [drawk_cache["delta_g_macro_draws"][(metric, s, t)] for s, t in pairs],
                    [[draw0_cache["delta_g_cat_vals"][(metric, s, t)][c] for s, t in pairs] for c in range(n_cats)],
                    [[drawk_cache["delta_g_cat_vals"][(metric, s, t)][c] for s, t in pairs] for c in range(n_cats)],
                    [[draw0_cache["delta_g_cat_draws"][(metric, s, t)][c] for s, t in pairs] for c in range(n_cats)],
                    [[drawk_cache["delta_g_cat_draws"][(metric, s, t)][c] for s, t in pairs] for c in range(n_cats)],
                ),
                "transfer_last_shift": _mean_shift_summary(
                    [draw0_cache["route_macro_vals"][(metric, "last_caption", s, t)] for s, t in pairs],
                    [drawk_cache["route_macro_vals"][(metric, "last_caption", s, t)] for s, t in pairs],
                    [draw0_cache["route_macro_draws"][(metric, "last_caption", s, t)] for s, t in pairs],
                    [drawk_cache["route_macro_draws"][(metric, "last_caption", s, t)] for s, t in pairs],
                    [[draw0_cache["route_cat_vals"][(metric, "last_caption", s, t)][c] for s, t in pairs] for c in range(n_cats)],
                    [[drawk_cache["route_cat_vals"][(metric, "last_caption", s, t)][c] for s, t in pairs] for c in range(n_cats)],
                    [[draw0_cache["route_cat_draws"][(metric, "last_caption", s, t)][c] for s, t in pairs] for c in range(n_cats)],
                    [[drawk_cache["route_cat_draws"][(metric, "last_caption", s, t)][c] for s, t in pairs] for c in range(n_cats)],
                ),
                "transfer_mean_shift": _mean_shift_summary(
                    [draw0_cache["route_macro_vals"][(metric, "caption_mean", s, t)] for s, t in pairs],
                    [drawk_cache["route_macro_vals"][(metric, "caption_mean", s, t)] for s, t in pairs],
                    [draw0_cache["route_macro_draws"][(metric, "caption_mean", s, t)] for s, t in pairs],
                    [drawk_cache["route_macro_draws"][(metric, "caption_mean", s, t)] for s, t in pairs],
                    [[draw0_cache["route_cat_vals"][(metric, "caption_mean", s, t)][c] for s, t in pairs] for c in range(n_cats)],
                    [[drawk_cache["route_cat_vals"][(metric, "caption_mean", s, t)][c] for s, t in pairs] for c in range(n_cats)],
                    [[draw0_cache["route_cat_draws"][(metric, "caption_mean", s, t)][c] for s, t in pairs] for c in range(n_cats)],
                    [[drawk_cache["route_cat_draws"][(metric, "caption_mean", s, t)][c] for s, t in pairs] for c in range(n_cats)],
                ),
                "diagonal_last_shift": _mean_shift_summary(
                    [draw0_cache["route_macro_vals"][(metric, "last_caption", t, t)] for t in diag_langs],
                    [drawk_cache["route_macro_vals"][(metric, "last_caption", t, t)] for t in diag_langs],
                    [draw0_cache["route_macro_draws"][(metric, "last_caption", t, t)] for t in diag_langs],
                    [drawk_cache["route_macro_draws"][(metric, "last_caption", t, t)] for t in diag_langs],
                    [[draw0_cache["route_cat_vals"][(metric, "last_caption", t, t)][c] for t in diag_langs] for c in range(n_cats)],
                    [[drawk_cache["route_cat_vals"][(metric, "last_caption", t, t)][c] for t in diag_langs] for c in range(n_cats)],
                    [[draw0_cache["route_cat_draws"][(metric, "last_caption", t, t)][c] for t in diag_langs] for c in range(n_cats)],
                    [[drawk_cache["route_cat_draws"][(metric, "last_caption", t, t)][c] for t in diag_langs] for c in range(n_cats)],
                ),
                "diagonal_mean_shift": _mean_shift_summary(
                    [draw0_cache["route_macro_vals"][(metric, "caption_mean", t, t)] for t in diag_langs],
                    [drawk_cache["route_macro_vals"][(metric, "caption_mean", t, t)] for t in diag_langs],
                    [draw0_cache["route_macro_draws"][(metric, "caption_mean", t, t)] for t in diag_langs],
                    [drawk_cache["route_macro_draws"][(metric, "caption_mean", t, t)] for t in diag_langs],
                    [[draw0_cache["route_cat_vals"][(metric, "caption_mean", t, t)][c] for t in diag_langs] for c in range(n_cats)],
                    [[drawk_cache["route_cat_vals"][(metric, "caption_mean", t, t)][c] for t in diag_langs] for c in range(n_cats)],
                    [[draw0_cache["route_cat_draws"][(metric, "caption_mean", t, t)][c] for t in diag_langs] for c in range(n_cats)],
                    [[drawk_cache["route_cat_draws"][(metric, "caption_mean", t, t)][c] for t in diag_langs] for c in range(n_cats)],
                ),
            }

        twelve_route_mean[metric] = _agg_shift(off_12, LANGUAGES)
        changed_targets_only_mean[metric] = _agg_shift(off_changed, eligible_targets)

    # Pooling-reduction sign preservation on primary AUROC
    dg0_12_vals = [draw0_cache["delta_g_macro_vals"][("auroc", s, t)] for s, t in off_12]
    dgk_12_vals = [drawk_cache["delta_g_macro_vals"][("auroc", s, t)] for s, t in off_12]
    dg0_12 = None if any(v is None for v in dg0_12_vals) else float(sum(dg0_12_vals) / len(dg0_12_vals))
    dgk_12 = None if any(v is None for v in dgk_12_vals) else float(sum(dgk_12_vals) / len(dgk_12_vals))

    if off_changed:
        dg0_ch_vals = [draw0_cache["delta_g_macro_vals"][("auroc", s, t)] for s, t in off_changed]
        dgk_ch_vals = [drawk_cache["delta_g_macro_vals"][("auroc", s, t)] for s, t in off_changed]
        dg0_ch = None if any(v is None for v in dg0_ch_vals) else float(sum(dg0_ch_vals) / len(dg0_ch_vals))
        dgk_ch = None if any(v is None for v in dgk_ch_vals) else float(sum(dgk_ch_vals) / len(dgk_ch_vals))
    else:
        dg0_ch, dgk_ch = None, None

    return {
        "twelve_route_mean": twelve_route_mean,
        "changed_targets_only": {
            "eligible_target_languages": list(eligible_targets),
            "n_routes": len(off_changed),
            "means": changed_targets_only_mean,
        },
        "diagonals": diagonals,
        "per_route": per_route,
        "pooling_reduction_sign_preserved": {
            "twelve_route_mean": _sign_preservation_record(dg0_12, dgk_12),
            "changed_targets_only_mean": _sign_preservation_record(dg0_ch, dgk_ch),
        },
    }


def _verify_draw0_parity_against_e29(draw0_summary, e29_draw0_ref, categories):
    """Assert exact (<= 1e-12) parity between recomputed draw-0 metrics and E29 draw0_reference."""
    from src.stage4.metrics import METRICS

    e29_pts = e29_draw0_ref["point_estimates"]
    max_abs_diff = 0.0

    def _cmp(a, b, label):
        nonlocal max_abs_diff
        if a is None or b is None:
            check(a is None and b is None, f"Draw-0 parity None mismatch at {label}: {a!r} vs {b!r}")
            return
        diff = abs(float(a) - float(b))
        if diff > max_abs_diff:
            max_abs_diff = diff
        check(diff <= DRAW0_PARITY_TOL, f"Draw-0 parity exceeded {DRAW0_PARITY_TOL} at {label}: diff={diff}")

    # 1. All 16 routes per pooling (primary macro + 25 categories)
    for pooling in POOLINGS:
        for rkey, e29_metrics in e29_pts["routes"][pooling].items():
            obs_metrics = draw0_summary["routes_all"][pooling][rkey]["metrics"]
            for m in METRICS:
                _cmp(
                    obs_metrics[m]["primary_macro"]["value"],
                    e29_metrics[m]["primary_macro"],
                    f"routes.{pooling}.{rkey}.{m}.primary_macro",
                )
                for cat in categories:
                    _cmp(
                        obs_metrics[m]["categories"][cat]["value"],
                        e29_metrics[m]["categories"][cat],
                        f"routes.{pooling}.{rkey}.{m}.{cat}",
                    )

    # 2. 12 per-route gaps
    for rkey, e29_metrics in e29_pts["per_route"].items():
        obs_metrics = draw0_summary["per_route"][rkey]["metrics"]
        for m in METRICS:
            for field in ("g_last", "g_res", "delta_g"):
                _cmp(
                    obs_metrics[m][field]["primary_macro"]["value"],
                    e29_metrics[m][field],
                    f"per_route.{rkey}.{m}.{field}",
                )

    # 3. 4 diagonals
    for t in LANGUAGES:
        e29_metrics = e29_pts["diagonals"][t]
        obs_metrics = draw0_summary["diagonals"][t]["metrics"]
        for m in METRICS:
            for field in ("last_caption", "caption_mean", "delta_d"):
                _cmp(
                    obs_metrics[m][field]["primary_macro"]["value"],
                    e29_metrics[m][field],
                    f"diagonals.{t}.{m}.{field}",
                )

    # 4. 12-route mean
    for m in METRICS:
        e29_m = e29_pts["twelve_route_mean"][m]
        obs_m = draw0_summary["twelve_route_means"][m]
        for field in ("g_last_mean", "g_res_mean", "delta_g_mean"):
            _cmp(
                obs_m[field]["primary_macro"]["value"],
                e29_m[field],
                f"twelve_route_mean.{m}.{field}",
            )

    return {
        "e29_draw0_max_abs_diff": float(max_abs_diff),
        "tolerance": DRAW0_PARITY_TOL,
        "unchanged_views_atol": 0.0,
        "unchanged_views_rtol": 0.0,
        "passed": True,
    }


def _evaluate_restricted_target_cohort(
    *,
    target_lang,
    draw_name,
    changed_indices,
    screen_ids,
    draw0_logits_by_pooling,
    drawk_logits_by_pooling,
    estimable_by_pooling,
    y,
    known,
    counts,
    categories,
    primary_indices,
    total_draws,
):
    """Evaluate S_changed(t, draw_k) paired with exact matched draw-0 companion."""
    np = _require_numpy()
    from src.stage4.metrics import METRICS

    n_screen = len(screen_ids)
    n_changed = len(changed_indices)
    if n_changed == 0:
        return (
            {
                "status": "unavailable",
                "reason": UNAVAILABLE_ZERO_CHANGED_REASON,
                "target_language": target_lang,
                "draw": draw_name,
                "n_images": 0,
                "n_screen_images": n_screen,
                "unchanged_fraction": 1.0,
            },
            None,
        )

    idx_arr = np.asarray(changed_indices, dtype=np.intp)
    sub_ids = [screen_ids[i] for i in changed_indices]
    sub_y = y[idx_arr]
    sub_known = known[idx_arr]
    sub_counts = counts[:, idx_arr]
    n_cats = len(categories)

    support_denominators = {}
    supported_primary_indices = []
    supported_primary_categories = []
    for col in primary_indices:
        cat = categories[col]
        known_n = int(sub_known[:, col].sum())
        pos_n = int(sub_y[:, col][sub_known[:, col]].sum())
        neg_n = known_n - pos_n
        min_cls = min(pos_n, neg_n)
        support_denominators[cat] = {
            "known": known_n,
            "positive": pos_n,
            "negative": neg_n,
            "min_class_count": min_cls,
            "ge_10": min_cls >= CHANGED_COHORT_MIN_SUPPORT,
        }
        if min_cls >= CHANGED_COHORT_MIN_SUPPORT:
            supported_primary_indices.append(col)
            supported_primary_categories.append(cat)

    # Evaluate both draw_k and matched draw_0 on identical (sub_y, sub_known, sub_counts)
    arms = {"draw_0": {}, "draw_k": {}}
    for arm_name, logits_src in (("draw_0", draw0_logits_by_pooling), ("draw_k", drawk_logits_by_pooling)):
        for pooling in POOLINGS:
            for s_idx, s in enumerate(LANGUAGES):
                arms[arm_name][(pooling, s)] = _evaluate_arm_logits(
                    logits_src[pooling][target_lang][s_idx][idx_arr],
                    estimable_by_pooling[pooling][target_lang][s_idx],
                    sub_y,
                    sub_known,
                    sub_counts,
                    n_cats,
                    total_draws,
                )

    macro_specs = (
        ("supported_primary_ge_10", supported_primary_indices),
        ("primary_14", primary_indices),
    )
    incoming_sources = [s for s in LANGUAGES if s != target_lang]

    macros_out = {}
    agg_cache = {spec_name: {} for spec_name, _ in macro_specs}

    for spec_name, m_indices in macro_specs:
        diagonal_metrics = {}
        per_route_out = {}
        target_mean_metrics = {}

        for metric in METRICS:
            # Diagonal t -> t (draw_0, draw_k, paired shifts)
            diag_vals = {}
            diag_draws = {}
            diag_summaries = {}
            for arm_name in ("draw_0", "draw_k"):
                for pooling in POOLINGS:
                    arm = arms[arm_name][(pooling, target_lang)]
                    m_sum, m_dr, _ = _summarize_macro_and_categories(
                        categories=categories,
                        macro_indices=m_indices,
                        cat_values=arm["points"][metric],
                        cat_draws=arm["draws"][metric],
                        cat_supports=arm["supports"],
                        cat_unestimable=arm["unestimable"],
                        minimum=CHANGED_COHORT_MIN_SUPPORT,
                        include_categories=False,
                    )
                    diag_vals[(arm_name, pooling)] = m_sum["value"]
                    diag_draws[(arm_name, pooling)] = m_dr
                    diag_summaries[(arm_name, pooling)] = m_sum

            diagonal_metrics[metric] = {
                "draw_k": {
                    "last_caption": diag_summaries[("draw_k", "last_caption")],
                    "caption_mean": diag_summaries[("draw_k", "caption_mean")],
                },
                "matched_draw_0": {
                    "last_caption": diag_summaries[("draw_0", "last_caption")],
                    "caption_mean": diag_summaries[("draw_0", "caption_mean")],
                },
                "shifts": {
                    "last_caption_shift": {
                        "value": _diff_value(diag_vals[("draw_k", "last_caption")], diag_vals[("draw_0", "last_caption")]),
                        "interval": _interval_record(diag_draws[("draw_k", "last_caption")] - diag_draws[("draw_0", "last_caption")]),
                    },
                    "caption_mean_shift": {
                        "value": _diff_value(diag_vals[("draw_k", "caption_mean")], diag_vals[("draw_0", "caption_mean")]),
                        "interval": _interval_record(diag_draws[("draw_k", "caption_mean")] - diag_draws[("draw_0", "caption_mean")]),
                    },
                },
            }

            # 3 incoming off-diagonal routes s -> target_lang
            route_records = []
            for s in incoming_sources:
                rkey = _route_key(s, target_lang)
                route_entry = per_route_out.setdefault(
                    rkey,
                    {
                        "route": rkey,
                        "source_language": s,
                        "target_language": target_lang,
                        "supervision_label": SUPERVISION_OFF_DIAGONAL,
                        "metrics": {},
                    },
                )
                arm_stats = {}
                for arm_name in ("draw_0", "draw_k"):
                    r_last = arms[arm_name][("last_caption", s)]
                    d_last = arms[arm_name][("last_caption", target_lang)]
                    r_mean = arms[arm_name][("caption_mean", s)]
                    d_mean = arms[arm_name][("caption_mean", target_lang)]

                    gl_vals = [_gap_value(metric, d_last["points"][metric][c], r_last["points"][metric][c]) for c in range(n_cats)]
                    gl_draws = [_gap_draws(metric, d_last["draws"][metric][c], r_last["draws"][metric][c]) for c in range(n_cats)]
                    gl_unest = [d_last["unestimable"][c] or r_last["unestimable"][c] for c in range(n_cats)]
                    gl_macro, gl_mdraws, _ = _summarize_macro_and_categories(
                        categories=categories,
                        macro_indices=m_indices,
                        cat_values=gl_vals,
                        cat_draws=gl_draws,
                        cat_supports=r_last["supports"],
                        cat_unestimable=gl_unest,
                        minimum=CHANGED_COHORT_MIN_SUPPORT,
                        include_categories=False,
                    )

                    gr_vals = [_gap_value(metric, d_mean["points"][metric][c], r_mean["points"][metric][c]) for c in range(n_cats)]
                    gr_draws = [_gap_draws(metric, d_mean["draws"][metric][c], r_mean["draws"][metric][c]) for c in range(n_cats)]
                    gr_unest = [d_mean["unestimable"][c] or r_mean["unestimable"][c] for c in range(n_cats)]
                    gr_macro, gr_mdraws, _ = _summarize_macro_and_categories(
                        categories=categories,
                        macro_indices=m_indices,
                        cat_values=gr_vals,
                        cat_draws=gr_draws,
                        cat_supports=r_mean["supports"],
                        cat_unestimable=gr_unest,
                        minimum=CHANGED_COHORT_MIN_SUPPORT,
                        include_categories=False,
                    )

                    dg_vals = [_diff_value(gl_vals[c], gr_vals[c]) for c in range(n_cats)]
                    dg_draws = [gl_draws[c] - gr_draws[c] for c in range(n_cats)]
                    dg_unest = [gl_unest[c] or gr_unest[c] for c in range(n_cats)]
                    dg_macro, dg_mdraws, _ = _summarize_macro_and_categories(
                        categories=categories,
                        macro_indices=m_indices,
                        cat_values=dg_vals,
                        cat_draws=dg_draws,
                        cat_supports=r_mean["supports"],
                        cat_unestimable=dg_unest,
                        minimum=CHANGED_COHORT_MIN_SUPPORT,
                        include_categories=False,
                    )
                    arm_stats[arm_name] = {
                        "g_last": (gl_macro, gl_mdraws),
                        "g_res": (gr_macro, gr_mdraws),
                        "delta_g": (dg_macro, dg_mdraws),
                    }

                route_entry["metrics"][metric] = {
                    "draw_k": {
                        "g_last": arm_stats["draw_k"]["g_last"][0],
                        "g_res": arm_stats["draw_k"]["g_res"][0],
                        "delta_g": arm_stats["draw_k"]["delta_g"][0],
                    },
                    "matched_draw_0": {
                        "g_last": arm_stats["draw_0"]["g_last"][0],
                        "g_res": arm_stats["draw_0"]["g_res"][0],
                        "delta_g": arm_stats["draw_0"]["delta_g"][0],
                    },
                    "shifts": {
                        "g_last_shift": {
                            "value": _diff_value(arm_stats["draw_k"]["g_last"][0]["value"], arm_stats["draw_0"]["g_last"][0]["value"]),
                            "interval": _interval_record(arm_stats["draw_k"]["g_last"][1] - arm_stats["draw_0"]["g_last"][1]),
                        },
                        "g_res_shift": {
                            "value": _diff_value(arm_stats["draw_k"]["g_res"][0]["value"], arm_stats["draw_0"]["g_res"][0]["value"]),
                            "interval": _interval_record(arm_stats["draw_k"]["g_res"][1] - arm_stats["draw_0"]["g_res"][1]),
                        },
                        "delta_g_shift": {
                            "value": _diff_value(arm_stats["draw_k"]["delta_g"][0]["value"], arm_stats["draw_0"]["delta_g"][0]["value"]),
                            "interval": _interval_record(arm_stats["draw_k"]["delta_g"][1] - arm_stats["draw_0"]["delta_g"][1]),
                        },
                    },
                }
                route_records.append((s, arm_stats))

            # Target-level 3-route equal-weight mean
            def _tmean(arm_name, field):
                vals = [st[arm_name][field][0]["value"] for _, st in route_records]
                dr_list = [st[arm_name][field][1] for _, st in route_records]
                val = None if any(v is None for v in vals) else float(sum(vals) / len(vals))
                dr = np.mean(np.stack(dr_list, axis=0), axis=0)
                return val, dr

            tm_out = {"draw_k": {}, "matched_draw_0": {}, "shifts": {}}
            for field in ("g_last", "g_res", "delta_g"):
                v0, d0 = _tmean("draw_0", field)
                vk, dk = _tmean("draw_k", field)
                tm_out["draw_k"][field] = {"value": vk, "interval": _interval_record(dk)}
                tm_out["matched_draw_0"][field] = {"value": v0, "interval": _interval_record(d0)}
                tm_out["shifts"][f"{field}_shift"] = {
                    "value": _diff_value(vk, v0),
                    "interval": _interval_record(dk - d0),
                }
            target_mean_metrics[metric] = tm_out

            # Save route-level macro draws for eligible-targets aggregate
            agg_cache[spec_name][metric] = [
                {
                    "route": _route_key(s, target_lang),
                    "draw_0": {f: (st["draw_0"][f][0]["value"], st["draw_0"][f][1]) for f in ("g_last", "g_res", "delta_g")},
                    "draw_k": {f: (st["draw_k"][f][0]["value"], st["draw_k"][f][1]) for f in ("g_last", "g_res", "delta_g")},
                }
                for s, st in route_records
            ]

        auroc_tm = target_mean_metrics["auroc"]
        macros_out[spec_name] = {
            "categories": list(supported_primary_categories if spec_name == "supported_primary_ge_10" else [categories[i] for i in primary_indices]),
            "n_categories": len(m_indices),
            "diagonal": diagonal_metrics,
            "per_route": per_route_out,
            "target_mean": target_mean_metrics,
            "pooling_reduction_sign_preserved": _sign_preservation_record(
                auroc_tm["matched_draw_0"]["delta_g"]["value"],
                auroc_tm["draw_k"]["delta_g"]["value"],
            ),
        }

    overall_status = "ok" if len(supported_primary_indices) == len(primary_indices) else "descriptive_low_support"
    target_report = {
        "status": overall_status,
        "reason": None if overall_status == "ok" else "subset_primary_support_below_10_for_some_categories",
        "target_language": target_lang,
        "draw": draw_name,
        "n_images": n_changed,
        "n_screen_images": n_screen,
        "unchanged_fraction": (n_screen - n_changed) / float(n_screen),
        "image_ids_sha256": fingerprint(sub_ids),
        "supported_primary_ge_10_categories": supported_primary_categories,
        "n_supported_primary_ge_10": len(supported_primary_categories),
        "support_denominators": support_denominators,
        "macros": macros_out,
    }
    return target_report, agg_cache


def _aggregate_restricted_eligible_targets(
    *,
    eligible_targets,
    unavailable_targets,
    agg_caches_by_target,
    total_draws,
):
    """Equal-weight aggregate over all directed routes `s != t` for eligible changed targets `t`."""
    np = _require_numpy()
    from src.stage4.metrics import METRICS

    n_routes = 3 * len(eligible_targets)
    if not eligible_targets:
        return {
            "status": "unavailable",
            "eligible_target_languages": [],
            "unavailable_target_languages": list(unavailable_targets),
            "n_routes": 0,
            "macros": {},
        }

    macros_out = {}
    for spec_name in ("supported_primary_ge_10", "primary_14"):
        by_metric = {}
        for metric in METRICS:
            route_items = []
            for t in eligible_targets:
                route_items.extend(agg_caches_by_target[t][spec_name][metric])
            m_out = {"draw_k": {}, "matched_draw_0": {}, "shifts": {}}
            for field in ("g_last", "g_res", "delta_g"):
                v0_list = [item["draw_0"][field][0] for item in route_items]
                vk_list = [item["draw_k"][field][0] for item in route_items]
                d0_list = [item["draw_0"][field][1] for item in route_items]
                dk_list = [item["draw_k"][field][1] for item in route_items]
                v0 = None if any(v is None for v in v0_list) else float(sum(v0_list) / len(v0_list))
                vk = None if any(v is None for v in vk_list) else float(sum(vk_list) / len(vk_list))
                d0 = np.mean(np.stack(d0_list, axis=0), axis=0)
                dk = np.mean(np.stack(dk_list, axis=0), axis=0)
                m_out["draw_k"][field] = {"value": vk, "interval": _interval_record(dk)}
                m_out["matched_draw_0"][field] = {"value": v0, "interval": _interval_record(d0)}
                m_out["shifts"][f"{field}_shift"] = {
                    "value": _diff_value(vk, v0),
                    "interval": _interval_record(dk - d0),
                }
            by_metric[metric] = m_out
        auroc_m = by_metric["auroc"]
        macros_out[spec_name] = {
            "metrics": by_metric,
            "pooling_reduction_sign_preserved": _sign_preservation_record(
                auroc_m["matched_draw_0"]["delta_g"]["value"],
                auroc_m["draw_k"]["delta_g"]["value"],
            ),
        }

    return {
        "status": "ok",
        "eligible_target_languages": list(eligible_targets),
        "unavailable_target_languages": list(unavailable_targets),
        "n_routes": n_routes,
        "macros": macros_out,
    }


def _intersection_metadata(languages_subset, changed_sets_by_lang, screen_ids, y, known, primary_indices):
    if not languages_subset:
        inter_indices = []
    else:
        inter = set(changed_sets_by_lang[languages_subset[0]])
        for lang in languages_subset[1:]:
            inter &= set(changed_sets_by_lang[lang])
        inter_indices = sorted(inter)
    sub_ids = [screen_ids[i] for i in inter_indices]
    ge_10 = 0
    if inter_indices:
        np = _require_numpy()
        idx_arr = np.asarray(inter_indices, dtype=np.intp)
        sub_y = y[idx_arr]
        sub_known = known[idx_arr]
        for col in primary_indices:
            k_mask = sub_known[:, col]
            pos = int(sub_y[:, col][k_mask].sum())
            neg = int(k_mask.sum()) - pos
            if min(pos, neg) >= CHANGED_COHORT_MIN_SUPPORT:
                ge_10 += 1
    return {
        "languages": list(languages_subset),
        "n_images": len(inter_indices),
        "image_ids_sha256": fingerprint(sub_ids),
        "primary_ge_10_count": ge_10,
    }


@contract
def run_e07_scoring(
    root,
    protocol,
    plan_ref,
    e07_audit_ref,
    e07_release_ref,
    e29_analysis_ref,
    model: str,
    producer: Mapping[str, Any],
    *,
    resume_prediction_refs=None,
    resume_analysis_ref=None,
    run_id: Optional[str] = None,
    progress=None,
):
    """Execute Phase 2 Stage 4 CPU E07 frozen-head scoring and dual-cohort comparison."""
    check(model in MODELS, f"Unsupported MODEL selector {model!r}; expected one of {MODELS}")
    require_open_split(E07_SPLIT)
    resolved_root = Path(root).resolve()
    total_units = 11  # 1 auth/self-check + 8 prediction blocks + 1 eval + 1 publish

    # Step 1: Authenticate plan, Phase 1 binding, E07 audit, E07 release, and E29 analysis
    t0 = time.monotonic()
    plan = load_phase2_plan(resolved_root, protocol, plan_ref)
    check(plan["confirm_excluded"] is True, "phase2_plan must enforce confirm_excluded=True")

    binding = bind_phase1(resolved_root, plain(plan.protocol), plain(plan["stage4_release"]))
    check(binding.lineage_identity == plan["lineage_identity"], "Phase 1 lineage_identity mismatch with phase2_plan")
    check(binding.dataset_fingerprint == plan["dataset_fingerprint"], "Dataset fingerprint mismatch with phase2_plan")
    check(binding.bootstrap_sha256 == plan["bootstrap"]["sha256"], "Bootstrap SHA-256 mismatch with phase2_plan")
    check(list(binding.bootstrap_shape) == list(plan["bootstrap"]["shape"]), "Bootstrap shape mismatch with phase2_plan")

    depth = midpoint_depth(binding, model)
    check(depth == int(plan["midpoints"][model]), f"Midpoint depth mismatch for {model}")

    audit_ref, audit_body, _ = load_e07_audit(
        resolved_root,
        protocol,
        e07_audit_ref,
        expected_dataset=plan["dataset_fingerprint"],
        expected_lineage_identity=plan["lineage_identity"],
    )
    if audit_body.get("plan_identity") is not None:
        check(audit_body["plan_identity"] == plan["plan_identity"], "E07 audit plan_identity mismatch")
    if plan["e07_audit"] is not None:
        check(plain(audit_ref) == plain(plan["e07_audit"]), "E07 audit ref differs from phase2_plan.e07_audit")
    verdict = audit_body.get("verdict")
    check(
        verdict in E07_ALLOWED_VERDICTS,
        f"E07 scoring refused: audit verdict is {verdict!r} (must be in {E07_ALLOWED_VERDICTS})",
    )
    check(audit_body.get("draw0_verified") is True, "E07 audit draw0_verified must be True")
    check(audit_body.get("confirm_excluded") is True, "E07 audit confirm_excluded must be True")

    e29_ref, e29_body, _ = load_phase2_artifact(
        resolved_root,
        protocol,
        e29_analysis_ref,
        KIND_ANALYSIS,
        expected_schema=SCHEMA_ANALYSIS,
        expected_dataset=plan["dataset_fingerprint"],
        expected_plan_identity=plan["plan_identity"],
        expected_owner=model,
    )
    check(e29_body.get("experiment") == "E29", "E29 analysis artifact experiment mismatch")
    check(e29_body.get("model") == model, "E29 analysis artifact model mismatch")
    check(int(e29_body.get("midpoint_depth", -1)) == depth, "E29 analysis midpoint_depth mismatch")
    check(e29_body.get("confirm_excluded") is True, "E29 analysis confirm_excluded must be True")
    check(plain(e29_body.get("bootstrap")) == plain(plan["bootstrap"]), "E29 analysis bootstrap mismatch")
    draw0_ref_obj = e29_body.get("draw0_reference")
    check(isinstance(draw0_ref_obj, Mapping), "E29 analysis missing draw0_reference")
    check(
        draw0_ref_obj.get("model") == model
        and int(draw0_ref_obj.get("midpoint_depth", -1)) == depth
        and draw0_ref_obj.get("condition") == E07_CONDITION
        and draw0_ref_obj.get("split") == E07_SPLIT,
        "E29 draw0_reference metadata mismatch",
    )

    blocks_meta = e29_screen_blocks(binding, model)
    check(plain(blocks_meta) == plain(plan["e29_blocks"][model]), "Resolved E29 blocks differ from phase2_plan")
    image_caption_blocks = blocks_meta[E07_CONDITION]
    check(
        plain(draw0_ref_obj["blocks"]) == plain(image_caption_blocks),
        "E29 draw0_reference blocks differ from bound Stage 4 image_caption blocks",
    )

    normalized_pred_resume = _normalize_resume_prediction_refs(resume_prediction_refs)

    # Verify E07 release + all chunks before resume/scoring
    (
        release_ref,
        release_body,
        dimension,
        views_by_lang,
        stacked_by_pooling_lang,
        chunk_refs_by_lang,
    ) = _load_and_verify_e07_release(
        root=resolved_root,
        protocol=protocol,
        plan=plan,
        audit_ref=audit_ref,
        audit_body=audit_body,
        e07_release_ref=e07_release_ref,
        model=model,
        depth=depth,
    )

    # Handle explicit analysis-level resume
    if resume_analysis_ref is not None:
        t_res = time.monotonic()
        res_ref, res_body, _ = verify_resume_ref(
            resolved_root,
            protocol,
            resume_analysis_ref,
            KIND_ANALYSIS,
            expected_schema=SCHEMA_ANALYSIS,
            expected_dataset=plan["dataset_fingerprint"],
            expected_plan_identity=plan["plan_identity"],
            expected_owner=model,
        )
        check(res_body.get("experiment") == "E07", "Resume analysis artifact is not E07")
        check(res_body.get("confirm_excluded") is True, "Resume analysis missing confirm_excluded=True")
        check(int(res_body.get("midpoint_depth", -1)) == depth, "Resume analysis midpoint_depth mismatch")
        check(res_body.get("condition") == E07_CONDITION, "Resume analysis condition mismatch")
        check(plain(res_body.get("e07_audit_ref")) == plain(audit_ref), "Resume analysis e07_audit_ref mismatch")
        check(plain(res_body.get("e07_release_ref")) == plain(release_ref), "Resume analysis e07_release_ref mismatch")
        check(plain(res_body.get("e29_analysis_ref")) == plain(e29_ref), "Resume analysis e29_analysis_ref mismatch")
        check(res_body.get("e07_verdict") == verdict, "Resume analysis e07_verdict mismatch")
        check(plain(res_body.get("bootstrap")) == plain(plan["bootstrap"]), "Resume analysis bootstrap mismatch")
        pred_refs = res_body.get("prediction_block_refs")
        check(isinstance(pred_refs, Mapping) and set(pred_refs.keys()) == set(POOLINGS),
              "Resume analysis missing prediction_block_refs")
        for pooling in POOLINGS:
            check(isinstance(pred_refs[pooling], Mapping) and set(pred_refs[pooling].keys()) == set(LANGUAGES),
                  f"Resume analysis missing languages for {pooling}")
            for lang in LANGUAGES:
                pref, pbody, _ = verify_resume_ref(
                    resolved_root,
                    protocol,
                    pred_refs[pooling][lang],
                    KIND_PREDICTION,
                    expected_schema=SCHEMA_PREDICTION,
                    expected_dataset=plan["dataset_fingerprint"],
                    expected_plan_identity=plan["plan_identity"],
                    expected_owner=model,
                )
                check(
                    pbody.get("experiment") == "E07"
                    and pbody.get("pooling") == pooling
                    and pbody.get("target_language") == lang
                    and plain(pbody.get("e07_release_ref")) == plain(release_ref),
                    f"Resume analysis linked incompatible prediction block for {pooling}:{lang}",
                )
                rk = f"{pooling}:{lang}"
                if rk in normalized_pred_resume:
                    exp_ref = normalize_ref(resolved_root, normalized_pred_resume[rk])
                    check(plain(pref) == plain(exp_ref), f"Supplied resume_prediction_refs[{rk}] differs from analysis")
        _emit_progress(
            progress,
            "e07_score_resumed",
            unit="resume",
            completed=total_units,
            total=total_units,
            read_hash_s=time.monotonic() - t_res,
            model=model,
            analysis_ref=plain(res_ref),
        )
        return res_ref, res_body, plain(pred_refs)

    np = _require_numpy()
    from src.stage3.engine import transform
    from src.stage4.metrics import bootstrap_counts, paired_image_alignment, targets_for_prediction
    from src.stage4.plan import load_bootstrap
    from src.stage4.predictions import load_prediction_block

    categories = tuple(protocol["category_order"])
    primary_categories = tuple(protocol["primary_categories"])
    screen_min_support = int(protocol["minimum_positive_and_negative"]["SCREEN"])
    check(categories == tuple(binding.stage4_plan["categories"]), "Protocol category_order mismatch with Stage 4 plan")
    n_cats = len(categories)
    cat_index = {cat: idx for idx, cat in enumerate(categories)}
    primary_indices = [cat_index[cat] for cat in primary_categories]

    # Load all 8 Stage 4 draw-0 SCREEN midpoint image_caption prediction blocks
    screen_ids = tuple(binding.stage4_plan["screen_ids"])
    screen_ids_sha256 = fingerprint(list(screen_ids))
    check(screen_ids_sha256 == draw0_ref_obj["screen_ids_sha256"], "SCREEN IDs mismatch with E29 draw0_reference")

    all_preds = []
    pred_keys = []
    stage4_preds = {p: {} for p in POOLINGS}
    for pooling in POOLINGS:
        for t_lang in LANGUAGES:
            entry = image_caption_blocks[pooling][t_lang]
            block_preds = load_prediction_block(
                binding.stage4_plan,
                entry["prediction_ref"],
                expected_block=entry["block_id"],
            )
            check(set(block_preds.keys()) == set(entry["routes"].values()),
                  f"Route mismatch in Stage 4 block {entry['block_id']}")
            stage4_preds[pooling][t_lang] = {}
            for s_lang in LANGUAGES:
                pred = block_preds[entry["routes"][s_lang]]
                stage4_preds[pooling][t_lang][s_lang] = pred
                all_preds.append(pred)
                pred_keys.append((pooling, s_lang, t_lang))

    ids, aligned_tuple = paired_image_alignment(binding.stage4_plan, *all_preds)
    check(ids == screen_ids, "Aligned SCREEN IDs mismatch plan screen_ids")
    aligned_by_key = {key: aligned for key, aligned in zip(pred_keys, aligned_tuple)}

    y_raw, known_raw = targets_for_prediction(binding.stage4_plan, all_preds[0])
    y = y_raw[aligned_tuple[0]]
    known = known_raw[aligned_tuple[0]]
    target_mask_fp = fingerprint({
        "screen_ids": list(ids),
        "y_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
        "known_sha256": hashlib.sha256(known.tobytes()).hexdigest(),
    })
    check(target_mask_fp == draw0_ref_obj["target_mask_fingerprint"],
          "SCREEN targets/mask fingerprint mismatch with E29 draw0_reference")

    # Assemble aligned Stage 4 draw-0 tensors: (4, n_screen, n_cats)
    draw0_logits = {p: {} for p in POOLINGS}
    draw0_estimable = {p: {} for p in POOLINGS}
    draw0_prevalence = {p: {} for p in POOLINGS}
    for pooling in POOLINGS:
        for t_lang in LANGUAGES:
            t_logits = []
            t_est = []
            t_prev = []
            for s_lang in LANGUAGES:
                pred = stage4_preds[pooling][t_lang][s_lang]
                aln = aligned_by_key[(pooling, s_lang, t_lang)]
                t_logits.append(np.asarray(pred.logits[aln], dtype="<f8"))
                t_est.append(np.asarray(pred.estimable, dtype=bool))
                t_prev.append(np.asarray(pred.prevalence, dtype="<f8"))
            draw0_logits[pooling][t_lang] = np.ascontiguousarray(np.stack(t_logits, axis=0), dtype="<f8")
            draw0_estimable[pooling][t_lang] = np.ascontiguousarray(np.stack(t_est, axis=0), dtype=bool)
            draw0_prevalence[pooling][t_lang] = np.ascontiguousarray(np.stack(t_prev, axis=0), dtype="<f8")

    # Load 8 frozen Stage 3 midpoint image_caption heads/scalers
    frozen_heads = {}
    source_system_refs = {}
    for pooling in POOLINGS:
        gids = midpoint_groups(binding, model, condition=E07_CONDITION, pooling=pooling)
        for s_lang in LANGUAGES:
            gid = gids[s_lang]
            coef, intercept, estimable, prevalence, stats = load_frozen_source_head(
                binding, model, E07_CONDITION, pooling, s_lang
            )
            check(coef.shape == (n_cats, dimension), f"Head coef shape mismatch for {model}/{pooling}/{s_lang}")
            check(len(stats["scale"]) == dimension, f"Scaler dimension mismatch for {model}/{pooling}/{s_lang}")
            frozen_heads[(pooling, s_lang)] = (coef, intercept, estimable, prevalence, stats)
            ordinary = binding.stage4_plan["ordinary"][gid]
            source_system_refs[(pooling, s_lang)] = {
                "group_id": gid,
                "selection": plain(ordinary["selection"]),
                "candidate": plain(ordinary["candidate"]),
                "C": float(ordinary["C"]),
            }

    # Step 2: In-memory en->en draw-0 SCREEN reproduction self-checks (atol=0, rtol=0)
    draw0_self_checks = {}
    for pooling in POOLINGS:
        rows_en, X_en = read_verified_midpoint_features(binding, model, E07_CONDITION, pooling, "en", E07_SPLIT)
        pos_en = {row["image_id"]: i for i, row in enumerate(rows_en)}
        check(set(pos_en) == set(screen_ids) and len(rows_en) == len(screen_ids),
              f"Stage 2 en SCREEN feature row mismatch for {model}/{pooling}")
        order_en = np.array([pos_en[iid] for iid in screen_ids], dtype=np.intp)
        coef_en, intercept_en, _, _, stats_en = frozen_heads[(pooling, "en")]
        z_en = np.asarray(transform(X_en[order_en], stats_en) @ coef_en.T, dtype="<f8") + intercept_en
        expected_z = draw0_logits[pooling]["en"][LANGUAGES.index("en")]
        max_abs_diff = float(np.max(np.abs(z_en - expected_z)))
        exact_ok = bool(
            np.array_equal(z_en, expected_z)
            and np.allclose(z_en, expected_z, atol=0.0, rtol=0.0)
            and max_abs_diff == 0.0
        )
        check(
            exact_ok,
            f"en->en draw-0 SCREEN reproduction self-check failed for {model}/{pooling}: max_abs_diff={max_abs_diff}",
        )
        draw0_self_checks[pooling] = {
            "route": "en->en",
            "pooling": pooling,
            "passed": True,
            "max_abs_diff": 0.0,
            "atol": 0.0,
            "rtol": 0.0,
        }

    read_hash_s = time.monotonic() - t0
    _emit_progress(
        progress,
        "e07_score_preflight_passed",
        unit="preflight_and_self_check",
        completed=1,
        total=total_units,
        read_hash_s=read_hash_s,
        model=model,
        midpoint_depth=depth,
        verdict=verdict,
        total_unique_views=int(release_body["total_unique_views"]),
    )

    # Step 3: Score unique changed views, publish 8 compact prediction blocks, assemble draw_1 & draw_2
    batch_run_id = run_id or new_run_id()
    prediction_block_refs = {p: {} for p in POOLINGS}
    draw_indices_by_lang = {
        lang: _build_draw_indices(lang, views_by_lang[lang], screen_ids, audit_body["draw_buckets"])
        for lang in LANGUAGES
    }
    draw1_logits = {p: {} for p in POOLINGS}
    draw2_logits = {p: {} for p in POOLINGS}
    completed_units = 1

    for pooling in POOLINGS:
        for target_lang in LANGUAGES:
            rk = f"{pooling}:{target_lang}"
            changed_views = views_by_lang[target_lang]
            n_changed = len(changed_views)
            lang_draw_indices = draw_indices_by_lang[target_lang]
            draw0_block_ref = plain(image_caption_blocks[pooling][target_lang]["prediction_ref"])

            t_comp = time.monotonic()
            t_w = 0.0
            resumed_block = False

            if rk in normalized_pred_resume:
                pref, pbody, pmarker = verify_resume_ref(
                    resolved_root,
                    protocol,
                    normalized_pred_resume[rk],
                    KIND_PREDICTION,
                    expected_schema=SCHEMA_PREDICTION,
                    expected_dataset=plan["dataset_fingerprint"],
                    expected_plan_identity=plan["plan_identity"],
                    expected_owner=model,
                )
                check(pbody.get("experiment") == "E07", f"Resume prediction {rk} experiment mismatch")
                check(pbody.get("model") == model, f"Resume prediction {rk} model mismatch")
                check(int(pbody.get("midpoint_depth", -1)) == depth, f"Resume prediction {rk} midpoint_depth mismatch")
                check(pbody.get("condition") == E07_CONDITION, f"Resume prediction {rk} condition mismatch")
                check(pbody.get("pooling") == pooling, f"Resume prediction {rk} pooling mismatch")
                check(pbody.get("split") == E07_SPLIT, f"Resume prediction {rk} split mismatch")
                check(pbody.get("target_language") == target_lang, f"Resume prediction {rk} target_language mismatch")
                check(plain(pbody.get("e07_audit_ref")) == plain(audit_ref), f"Resume prediction {rk} e07_audit_ref mismatch")
                check(plain(pbody.get("e07_release_ref")) == plain(release_ref), f"Resume prediction {rk} e07_release_ref mismatch")
                check(plain(pbody.get("draw0_block_ref")) == draw0_block_ref, f"Resume prediction {rk} draw0_block_ref mismatch")
                check(int(pbody.get("n_changed_views", -1)) == n_changed, f"Resume prediction {rk} n_changed_views mismatch")
                check(int(pbody.get("n_screen_images", -1)) == len(screen_ids), f"Resume prediction {rk} n_screen_images mismatch")
                check(pbody.get("screen_ids_sha256") == screen_ids_sha256, f"Resume prediction {rk} screen_ids_sha256 mismatch")
                check(plain(pbody.get("draw_indices")) == plain(lang_draw_indices), f"Resume prediction {rk} draw_indices mismatch")
                check(pbody.get("confirm_excluded") is True, f"Resume prediction {rk} confirm_excluded must be True")

                pfolder = within(resolved_root, pref["path"]).parent
                changed_logits = _verify_npy_file(
                    pfolder / "logits.npy",
                    pmarker["files"]["logits.npy"],
                    "<f8",
                    (len(LANGUAGES), n_changed, n_cats),
                )
                loaded_est = _verify_npy_file(
                    pfolder / "estimable.npy",
                    pmarker["files"]["estimable.npy"],
                    "|b1",
                    (len(LANGUAGES), n_cats),
                )
                loaded_prev = _verify_npy_file(
                    pfolder / "prevalence.npy",
                    pmarker["files"]["prevalence.npy"],
                    "<f8",
                    (len(LANGUAGES), n_cats),
                )
                check(np.array_equal(loaded_est, draw0_estimable[pooling][target_lang]),
                      f"Resume prediction {rk} estimable mismatch")
                check(np.array_equal(loaded_prev, draw0_prevalence[pooling][target_lang]),
                      f"Resume prediction {rk} prevalence mismatch")
                pred_ref = plain(pref)
                resumed_block = True
                compute_s = time.monotonic() - t_comp
            else:
                if n_changed > 0:
                    X_changed = stacked_by_pooling_lang[pooling][target_lang]
                    check(X_changed.shape == (n_changed, dimension),
                          f"Changed feature matrix shape mismatch for {rk}")
                    per_source_logits = []
                    per_source_est = []
                    per_source_prev = []
                    for s_lang in LANGUAGES:
                        coef, intercept, estimable, prevalence, stats = frozen_heads[(pooling, s_lang)]
                        z = np.asarray(transform(X_changed, stats) @ coef.T, dtype="<f8") + intercept
                        check(z.shape == (n_changed, n_cats), f"Scored logits shape mismatch for {rk}/{s_lang}")
                        check(np.isfinite(z[:, estimable]).all(), f"Nonfinite scored logits for {rk}/{s_lang}")
                        if (~estimable).any():
                            check(not np.any(z[:, ~estimable]), f"Nonzero unestimable padding for {rk}/{s_lang}")
                        per_source_logits.append(z)
                        per_source_est.append(np.asarray(estimable, dtype=bool))
                        per_source_prev.append(np.asarray(prevalence, dtype="<f8"))
                    changed_logits = np.ascontiguousarray(np.stack(per_source_logits, axis=0), dtype="<f8")
                    est_arr = np.ascontiguousarray(np.stack(per_source_est, axis=0), dtype=bool)
                    prev_arr = np.ascontiguousarray(np.stack(per_source_prev, axis=0), dtype="<f8")
                else:
                    changed_logits = np.empty((len(LANGUAGES), 0, n_cats), dtype="<f8")
                    est_arr = np.ascontiguousarray(
                        np.stack([np.asarray(frozen_heads[(pooling, s)][2], dtype=bool) for s in LANGUAGES], axis=0),
                        dtype=bool,
                    )
                    prev_arr = np.ascontiguousarray(
                        np.stack([np.asarray(frozen_heads[(pooling, s)][3], dtype="<f8") for s in LANGUAGES], axis=0),
                        dtype="<f8",
                    )

                check(np.array_equal(est_arr, draw0_estimable[pooling][target_lang]),
                      f"Frozen head estimable mismatch with Stage 4 draw-0 for {rk}")

                routes_meta = [
                    {
                        "route_id": f"{model}:{E07_CONDITION}:{pooling}:{s_lang}->{target_lang}",
                        "source_language": s_lang,
                        "row_index": s_idx,
                        "source_system_ref": source_system_refs[(pooling, s_lang)],
                        "supervision_label": SUPERVISION_DIAGONAL if s_lang == target_lang else SUPERVISION_OFF_DIAGONAL,
                    }
                    for s_idx, s_lang in enumerate(LANGUAGES)
                ]
                pred_body = {
                    "schema": SCHEMA_PREDICTION,
                    "experiment": "E07",
                    "plan": plain(plan.reference),
                    "plan_identity": plan["plan_identity"],
                    "e07_audit_ref": plain(audit_ref),
                    "e07_release_ref": plain(release_ref),
                    "e07_verdict": verdict,
                    "model": model,
                    "midpoint_depth": depth,
                    "condition": E07_CONDITION,
                    "pooling": pooling,
                    "split": E07_SPLIT,
                    "target_language": target_lang,
                    "source_languages": list(LANGUAGES),
                    "n_screen_images": len(screen_ids),
                    "n_changed_views": n_changed,
                    "n_categories": n_cats,
                    "category_order": list(categories),
                    "screen_ids_sha256": screen_ids_sha256,
                    "draw0_block_ref": draw0_block_ref,
                    "draw0_self_check": draw0_self_checks[pooling],
                    "draw_indices": lang_draw_indices,
                    "routes": routes_meta,
                    "upstream_chunk_refs": list(chunk_refs_by_lang[target_lang]),
                    "confirm_excluded": True,
                }
                compute_s = time.monotonic() - t_comp

                t_write_start = time.monotonic()
                upstream_refs = [
                    plain(plan.reference),
                    plain(audit_ref),
                    plain(release_ref),
                    draw0_block_ref,
                    *chunk_refs_by_lang[target_lang],
                ]
                published_ref = publish_phase2(
                    resolved_root,
                    protocol,
                    plan["dataset_fingerprint"],
                    f"phase2/predictions/e07_screen/{model}",
                    KIND_PREDICTION,
                    pred_body,
                    producer,
                    arrays={
                        "logits": changed_logits,
                        "estimable": est_arr,
                        "prevalence": prev_arr,
                    },
                    upstream=upstream_refs,
                    model_fingerprint=release_body["model_fingerprint"],
                    run_id=f"{batch_run_id}-{pooling}-{target_lang}",
                )
                pred_ref = plain(published_ref)
                t_w = time.monotonic() - t_write_start

            prediction_block_refs[pooling][target_lang] = pred_ref

            # Assemble full 360-image draw_1 and draw_2 tensors with bit-for-bit draw-0 reuse
            base_draw0 = draw0_logits[pooling][target_lang]
            for dk_name, dest_map in (("draw_1", draw1_logits), ("draw_2", draw2_logits)):
                full_logits = base_draw0.copy()
                row_map = lang_draw_indices[dk_name]["changed_row_map"]
                changed_row_set = set()
                for s_idx_str, info in row_map.items():
                    s_idx = int(s_idx_str)
                    v_idx = int(info["changed_view_index"])
                    full_logits[:, s_idx, :] = changed_logits[:, v_idx, :]
                    changed_row_set.add(s_idx)
                unchanged_mask = np.array([i not in changed_row_set for i in range(len(screen_ids))], dtype=bool)
                if unchanged_mask.any():
                    check(
                        np.array_equal(full_logits[:, unchanged_mask, :], base_draw0[:, unchanged_mask, :])
                        and np.allclose(
                            full_logits[:, unchanged_mask, :],
                            base_draw0[:, unchanged_mask, :],
                            atol=0.0,
                            rtol=0.0,
                        ),
                        f"Bit-for-bit draw-0 reuse check failed on unchanged rows for {rk}/{dk_name}",
                    )
                dest_map[pooling][target_lang] = full_logits

            completed_units += 1
            _emit_progress(
                progress,
                "e07_prediction_block_resumed" if resumed_block else "e07_prediction_block_published",
                unit=f"score:{rk}",
                completed=completed_units,
                total=total_units,
                compute_s=compute_s,
                write_s=t_w,
                model=model,
                pooling=pooling,
                target_language=target_lang,
                n_changed_views=n_changed,
                ref=pred_ref,
            )

    # Step 4: Dual-cohort paired bootstrap evaluation
    t_eval = time.monotonic()
    draws = load_bootstrap(binding.stage4_plan)
    counts = bootstrap_counts(draws, len(screen_ids))
    total_draws = int(draws.shape[0])

    # Coverage summary
    coverage_per_lang = {}
    per_draw_eligible = {}
    per_draw_unavailable = {}
    for dk_name in E07_DRAW_NAMES:
        elig = [lang for lang in LANGUAGES if draw_indices_by_lang[lang][dk_name]["changed_text_vs_draw0"] > 0]
        unavail = [lang for lang in LANGUAGES if lang not in elig]
        per_draw_eligible[dk_name] = elig
        per_draw_unavailable[dk_name] = unavail

    overall_eligible = [
        lang for lang in LANGUAGES if len(views_by_lang[lang]) > 0
    ]
    overall_unavailable = [lang for lang in LANGUAGES if lang not in overall_eligible]

    for lang in LANGUAGES:
        lang_b = audit_body["draw_buckets"][lang]
        coverage_per_lang[lang] = {
            "unique_changed_views": len(views_by_lang[lang]),
            "draw_1": {
                "changed_text_vs_draw0": int(lang_b["draw_1"]["changed_text_vs_draw0"]),
                "same_id_as_draw0": int(lang_b["draw_1"]["same_id_as_draw0"]),
                "distinct_id_same_text_as_draw0": int(lang_b["draw_1"]["distinct_id_same_text_as_draw0"]),
                "unchanged_view_fraction": draw_indices_by_lang[lang]["draw_1"]["unchanged_fraction"],
            },
            "draw_2": {
                "changed_text_vs_draw0": int(lang_b["draw_2"]["changed_text_vs_draw0"]),
                "same_id_as_draw0": int(lang_b["draw_2"]["same_id_as_draw0"]),
                "distinct_id_same_text_as_draw0": int(lang_b["draw_2"]["distinct_id_same_text_as_draw0"]),
                "unchanged_view_fraction": draw_indices_by_lang[lang]["draw_2"]["unchanged_fraction"],
            },
        }

    coverage_summary = {
        "n_screen_images": len(screen_ids),
        "total_unique_views": int(release_body["total_unique_views"]),
        "eligible_changed_languages": overall_eligible,
        "unavailable_languages": overall_unavailable,
        "per_draw_eligible_languages": per_draw_eligible,
        "per_draw_unavailable_languages": per_draw_unavailable,
        "per_language": coverage_per_lang,
    }

    # Cohort A: full_screen_360
    draw0_summary, draw0_cache = _evaluate_full_screen_draw(
        logits_by_pooling_target=draw0_logits,
        estimable_by_pooling_target=draw0_estimable,
        y=y,
        known=known,
        counts=counts,
        categories=categories,
        primary_indices=primary_indices,
        total_draws=total_draws,
        minimum=screen_min_support,
        eligible_targets=overall_eligible,
    )
    draw0_parity_check = _verify_draw0_parity_against_e29(draw0_summary, draw0_ref_obj, categories)

    draw1_summary, draw1_cache = _evaluate_full_screen_draw(
        logits_by_pooling_target=draw1_logits,
        estimable_by_pooling_target=draw0_estimable,
        y=y,
        known=known,
        counts=counts,
        categories=categories,
        primary_indices=primary_indices,
        total_draws=total_draws,
        minimum=screen_min_support,
        eligible_targets=per_draw_eligible["draw_1"],
    )
    draw2_summary, draw2_cache = _evaluate_full_screen_draw(
        logits_by_pooling_target=draw2_logits,
        estimable_by_pooling_target=draw0_estimable,
        y=y,
        known=known,
        counts=counts,
        categories=categories,
        primary_indices=primary_indices,
        total_draws=total_draws,
        minimum=screen_min_support,
        eligible_targets=per_draw_eligible["draw_2"],
    )

    full_screen_360 = {
        "n_images": len(screen_ids),
        "screen_minimum_support": screen_min_support,
        "draws": {
            "draw_0": draw0_summary,
            "draw_1": draw1_summary,
            "draw_2": draw2_summary,
        },
        "paired_shifts": {
            "draw_1_vs_draw_0": _compute_full_screen_paired_shift(
                draw0_cache=draw0_cache,
                drawk_cache=draw1_cache,
                categories=categories,
                total_draws=total_draws,
                eligible_targets=per_draw_eligible["draw_1"],
            ),
            "draw_2_vs_draw_0": _compute_full_screen_paired_shift(
                draw0_cache=draw0_cache,
                drawk_cache=draw2_cache,
                categories=categories,
                total_draws=total_draws,
                eligible_targets=per_draw_eligible["draw_2"],
            ),
        },
    }

    # Cohort B: restricted_changed_cohort
    restricted_draws = {}
    for dk_name, drawk_logits in (("draw_1", draw1_logits), ("draw_2", draw2_logits)):
        per_target = {}
        agg_caches = {}
        changed_sets_by_lang = {}
        for t_lang in LANGUAGES:
            changed_idx = sorted(int(k) for k in draw_indices_by_lang[t_lang][dk_name]["changed_row_map"].keys())
            changed_sets_by_lang[t_lang] = changed_idx
            t_report, t_cache = _evaluate_restricted_target_cohort(
                target_lang=t_lang,
                draw_name=dk_name,
                changed_indices=changed_idx,
                screen_ids=screen_ids,
                draw0_logits_by_pooling=draw0_logits,
                drawk_logits_by_pooling=drawk_logits,
                estimable_by_pooling=draw0_estimable,
                y=y,
                known=known,
                counts=counts,
                categories=categories,
                primary_indices=primary_indices,
                total_draws=total_draws,
            )
            per_target[t_lang] = t_report
            if t_cache is not None:
                agg_caches[t_lang] = t_cache

        elig_t = per_draw_eligible[dk_name]
        unavail_t = per_draw_unavailable[dk_name]
        eligible_agg = _aggregate_restricted_eligible_targets(
            eligible_targets=elig_t,
            unavailable_targets=unavail_t,
            agg_caches_by_target=agg_caches,
            total_draws=total_draws,
        )
        intersections = {
            "eligible_languages_intersection": _intersection_metadata(
                elig_t, changed_sets_by_lang, screen_ids, y, known, primary_indices
            ),
            "four_language_intersection": _intersection_metadata(
                list(LANGUAGES), changed_sets_by_lang, screen_ids, y, known, primary_indices
            ),
        }
        restricted_draws[dk_name] = {
            "per_target": per_target,
            "eligible_targets_aggregate": eligible_agg,
            "intersections": intersections,
        }

    restricted_changed_cohort = {
        "changed_minimum_support": CHANGED_COHORT_MIN_SUPPORT,
        "draws": restricted_draws,
    }

    eval_s = time.monotonic() - t_eval
    completed_units += 1
    _emit_progress(
        progress,
        "e07_dual_cohort_evaluated",
        unit="evaluate_cohorts",
        completed=completed_units,
        total=total_units,
        compute_s=eval_s,
        model=model,
        draw0_parity_max_abs_diff=draw0_parity_check["e29_draw0_max_abs_diff"],
    )

    # Publish analysis block
    t_pub = time.monotonic()
    analysis_body = {
        "schema": SCHEMA_ANALYSIS,
        "experiment": "E07",
        "plan": plain(plan.reference),
        "plan_identity": plan["plan_identity"],
        "e07_audit_ref": plain(audit_ref),
        "e07_release_ref": plain(release_ref),
        "e07_verdict": verdict,
        "e29_analysis_ref": plain(e29_ref),
        "model": model,
        "midpoint_depth": depth,
        "condition": E07_CONDITION,
        "poolings": list(POOLINGS),
        "bootstrap": plain(plan["bootstrap"]),
        "prediction_block_refs": prediction_block_refs,
        "draw0_self_checks": draw0_self_checks,
        "draw0_parity_check": draw0_parity_check,
        "coverage_summary": coverage_summary,
        "full_screen_360": full_screen_360,
        "restricted_changed_cohort": restricted_changed_cohort,
        "supervision_labels": {
            "diagonal": SUPERVISION_DIAGONAL,
            "off_diagonal": SUPERVISION_OFF_DIAGONAL,
        },
        "interpretation_scope": INTERPRETATION_SCOPE,
        "confirm_excluded": True,
    }

    upstream_analysis_refs = [
        plain(plan.reference),
        plain(audit_ref),
        plain(release_ref),
        plain(e29_ref),
    ]
    for pooling in POOLINGS:
        for lang in LANGUAGES:
            upstream_analysis_refs.append(prediction_block_refs[pooling][lang])

    analysis_ref = publish_phase2(
        resolved_root,
        protocol,
        plan["dataset_fingerprint"],
        f"phase2/results/e07/{model}",
        KIND_ANALYSIS,
        analysis_body,
        producer,
        upstream=upstream_analysis_refs,
        model_fingerprint=release_body["model_fingerprint"],
        run_id=batch_run_id,
    )
    _, verified_analysis_body, _ = verify_resume_ref(
        resolved_root,
        protocol,
        analysis_ref,
        KIND_ANALYSIS,
        expected_schema=SCHEMA_ANALYSIS,
        expected_dataset=plan["dataset_fingerprint"],
        expected_plan_identity=plan["plan_identity"],
        expected_owner=model,
    )
    pub_s = time.monotonic() - t_pub
    completed_units += 1

    _emit_progress(
        progress,
        "e07_score_published",
        unit="publish",
        completed=completed_units,
        total=total_units,
        write_s=pub_s,
        model=model,
        analysis_ref=plain(analysis_ref),
        prediction_block_refs=prediction_block_refs,
        draw0_parity_max_abs_diff=draw0_parity_check["e29_draw0_max_abs_diff"],
    )
    return analysis_ref, verified_analysis_body, prediction_block_refs


def _parse_ref_arg(value):
    if value is None:
        return None
    text = value.strip()
    if text.startswith("{"):
        parsed = json.loads(text)
        check(isinstance(parsed, dict), "JSON reference argument must decode to an object")
        return parsed
    return text


def _parse_resume_prediction_refs_arg(value):
    if value is None:
        return None
    text = value.strip()
    if text.startswith("{"):
        parsed = json.loads(text)
        check(isinstance(parsed, dict), "JSON resume-prediction-refs must decode to an object")
        return parsed
    path = Path(text)
    check(path.is_file(), f"Missing resume-prediction-refs JSON file: {text}")
    parsed = json.loads(path.read_text(encoding="utf-8"))
    check(isinstance(parsed, dict), "resume-prediction-refs file must decode to a JSON object")
    return parsed


def _emit_cli(event):
    print(json.dumps(event), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-root", required=True)
    parser.add_argument("--protocol", type=Path, default=REPO / "configs/protocol.json")
    parser.add_argument("--plan", required=True, help="Explicit phase2_plan {'path', 'sha256'} JSON or path")
    parser.add_argument("--e07-audit", required=True, help="Explicit phase2_e07_audit {'path', 'sha256'} JSON or path")
    parser.add_argument("--e07-release", required=True, help="Explicit phase2_e07_release {'path', 'sha256'} JSON or path")
    parser.add_argument("--e29-analysis", required=True, help="Explicit E29 phase2_analysis_block {'path', 'sha256'} JSON or path")
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument(
        "--storage-scope",
        default="persistent-drive",
        choices=("persistent-drive", "local-development"),
    )
    parser.add_argument(
        "--resume-prediction-refs",
        help="Optional JSON object or JSON file path mapping '{pooling}:{lang}' -> {'path', 'sha256'} or path",
    )
    parser.add_argument(
        "--resume-analysis-ref",
        help="Optional explicit E07 phase2_analysis_block {'path', 'sha256'} JSON or path",
    )
    parser.add_argument("--run-id", help="Optional explicit run_id")
    parser.add_argument("--output-ref", type=Path, help="Optional JSON file path for emitted references")
    args = parser.parse_args(argv)

    root = phase1_root(args.phase1_root)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    from src.phase2.interfaces import current_producer

    producer = current_producer(args.storage_scope)
    _emit_cli({
        "event": "e07_score_start",
        "model": args.model,
        "code_commit": producer["code_commit"],
        "storage_scope": producer["storage_scope"],
    })
    analysis_ref, analysis_body, pred_refs = run_e07_scoring(
        root,
        protocol,
        _parse_ref_arg(args.plan),
        _parse_ref_arg(args.e07_audit),
        _parse_ref_arg(args.e07_release),
        _parse_ref_arg(args.e29_analysis),
        args.model,
        producer,
        resume_prediction_refs=_parse_resume_prediction_refs_arg(args.resume_prediction_refs),
        resume_analysis_ref=_parse_ref_arg(args.resume_analysis_ref),
        run_id=args.run_id,
        progress=_emit_cli,
    )
    summary = {
        "event": "e07_score_complete",
        "status": "passed",
        "model": args.model,
        "e07_verdict": analysis_body["e07_verdict"],
        "analysis_ref": plain(analysis_ref),
        "prediction_block_refs": plain(pred_refs),
        "draw0_parity_check": plain(analysis_body["draw0_parity_check"]),
        "eligible_changed_languages": list(analysis_body["coverage_summary"]["eligible_changed_languages"]),
        "unavailable_languages": list(analysis_body["coverage_summary"]["unavailable_languages"]),
    }
    if args.output_ref is not None:
        args.output_ref.parent.mkdir(parents=True, exist_ok=True)
        args.output_ref.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _emit_cli(summary)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        _emit_cli({"status": "blocked", "blockers": [str(exc)], "stage4_e07_score_complete": False})
        raise SystemExit(2)
