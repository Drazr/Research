"""Phase 3 Stage B3 Worker 2: CPU frozen-head scoring and opposite-label swap analysis for E23.

Implements phase3.md §B3.1-B3.3 (Worker 2 - `run_e23`):
1. Enforces `model in MODELS` (`'qwen'` or `'gemma'`) and `require_open_split("SCREEN")`
   before any IO. `CONFIRM` remains strictly closed; Phase 1 and Phase 2 remain strictly
   read-only; `E35` (`G7`) is dropped (`e35_excluded = True`).
2. Authenticates `gpu_plan = load_phase3_gpu_plan(resolved_root, protocol, plan_ref or plain(CERTIFIED_STAGE_B1_GPU_PLAN), bind=True)`,
   verifying `gpu_plan["confirm_excluded"] is True`, `gpu_plan["e35_excluded"] is True`,
   `gpu_plan["e23_plan"]["zero_model_scores_in_pairing"] is True`, and
   `gpu_plan["e23_plan"]["confirm_rows"] == 0`.
3. Resolves `release_ref` defaulting cleanly to `plain(CERTIFIED_STAGE_B2_RELEASES["e23"][model])`
   when `None`, and authenticates via `verify_e23_release(resolved_root, protocol, eff_release_ref,
   expected_plan_identity=gpu_plan["plan_identity"], expected_model=model)`.
4. Verifies the 8 authentic anchor rows in `authentic_parity_last_caption.npy` and
   `authentic_parity_caption_mean.npy` (from `authentic_parity_rows.jsonl`) against Stage 2
   midpoint `image_caption` `SCREEN` features (`read_verified_midpoint_features`), asserting
   `max_abs_diff <= 1e-4` for both poolings, runs in-memory `en->en` matched logit self-checks
   (`atol=0, rtol=0`) against Stage 4 `SCREEN` logits on Stage 2 `SCREEN` features, and checks
   scored authentic parity logits against Stage 4 authentic logits (`max_abs_diff <= 1e-4`).
5. Loads all verified swapped chunks (`phase3_e23_feature_chunk`) from the E23 release, scores
   swapped views across `t in LANGUAGES` and `pool in ("last_caption", "caption_mean")` through
   the 4 frozen Stage 3 midpoint `image_caption` source heads `s in LANGUAGES`
   (`load_frozen_source_head` + `src.stage3.engine.transform`), and publishes swapped predictions
   under `phase3/predictions/e23/<model>/<run_id>/_COMPLETE.json` (`KIND_PREDICTION`,
   `SCHEMA_PREDICTION`, supporting `--resume-predictions`).
6. Evaluates opposite-label paired swaps on the frozen `(2000, 360)` bootstrap (`load_bootstrap`,
   `bootstrap_counts`) with pair multiplicity weight
   `w_{b,k} = 0.5 * (counts[b, screen_idx[pos_k]] + counts[b, screen_idx[neg_k]])` across all
   categories, routes, macro cohorts (`supported_primary_macro`, `fixed_primary_macro`,
   `all_supported_macro`), 12 off-diagonal route means, 4 diagonal route means, and paired
   `caption_mean - last_caption` contrasts on identical bootstrap draws, evaluates
   `PHASE3_GPU_DECISION_RANGES["E23"]`, and publishes `phase3/results/e23/<model>/<run_id>/_COMPLETE.json`
   (`KIND_ANALYSIS`, `SCHEMA_ANALYSIS`, supporting `--resume-analysis`).
"""

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Optional

from src.common.contracts import ContractError, LANGUAGES, fingerprint
from src.common.storage import file_sha256, new_run_id, phase1_root, within
from src.phase2.interfaces import (
    e29_screen_blocks,
    load_frozen_source_head,
    midpoint_depth,
    midpoint_groups,
    read_verified_midpoint_features,
)
from src.phase3.extract_e23 import (
    _expected_authentic_parity_rows,
    _expected_swapped_chunk_rows,
    verify_e23_release,
)
from src.phase3.interfaces import (
    CERTIFIED_STAGE_B1_GPU_PLAN,
    CERTIFIED_STAGE_B2_RELEASES,
    KIND_ANALYSIS,
    KIND_PREDICTION,
    MODELS,
    PHASE3_GPU_DECISION_RANGES,
    POOLINGS,
    REPO,
    SCHEMA_ANALYSIS,
    SCHEMA_PREDICTION,
    check,
    contract,
    current_producer,
    frozen,
    load_phase3_gpu_plan,
    plain,
    publish_phase3,
    require_open_split,
    verify_resume_ref,
)
from src.stage2.schema import jsonl

E23_EXPERIMENT = "E23"
E23_CONDITION = "image_caption"
E23_SPLIT = "SCREEN"
E23_EXPECTED_MIDPOINTS = {"qwen": 18, "gemma": 17}
E23_PREDICTIONS_NAMESPACE = "phase3/predictions/e23"
E23_RESULTS_NAMESPACE = "phase3/results/e23"

AUTHENTIC_PARITY_TOL = 1e-4
EN_MATCHED_SELF_CHECK_TOL = 0.0
RATIO_DENOM_FLOOR = 1e-12

SUPERVISION_DIAGONAL = "within_context_source_fit"
SUPERVISION_OFF_DIAGONAL = "source_only_transfer"

E23_LINEAR_SWAP_METRICS = (
    "delta_auth",
    "shift_neg_caption_to_pos_image",
    "shift_pos_caption_to_neg_image",
    "signed_drop_on_neg_image",
    "signed_shift_toward_shown_image",
)

E23_RATIO_SWAP_METRICS = (
    "visual_grounding_fraction",
    "caption_dominance_fraction",
)

E23_SWAP_METRICS = E23_LINEAR_SWAP_METRICS + E23_RATIO_SWAP_METRICS

E23_MACRO_COHORTS = (
    "supported_primary_macro",
    "fixed_primary_macro",
    "all_supported_macro",
)


def _require_numpy():
    try:
        import numpy as np

        return np
    except ImportError as exc:
        raise ContractError("numpy is required for E23 CPU scoring and bootstrap analysis") from exc


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


def _route_key(source_lang: str, target_lang: str) -> str:
    return f"{source_lang}->{target_lang}"


def _off_diagonal_pairs():
    return [(s, t) for s in LANGUAGES for t in LANGUAGES if s != t]


def _diagonal_pairs():
    return [(t, t) for t in LANGUAGES]


def _all_route_pairs():
    return [(s, t) for s in LANGUAGES for t in LANGUAGES]


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def compute_e23_pair_shifts(
    p_auth_pos,
    p_auth_neg,
    p_swap_pos_shown_neg_cap,
    p_swap_neg_shown_pos_cap,
):
    """Compute per-pair opposite-label authentic separation and signed swap shifts.

    Works on Python scalar floats or 1D numpy arrays:
    - `delta_auth_k = P_auth(pos_k, pos_k) - P_auth(neg_k, neg_k)`
    - `shift_neg_caption_to_pos_image_k = P_swap(shown=pos_k, owner=neg_k) - P_auth(neg_k, neg_k)`
    - `shift_pos_caption_to_neg_image_k = P_auth(pos_k, pos_k) - P_swap(shown=neg_k, owner=pos_k)`
    - `signed_drop_on_neg_image_k = -shift_pos_caption_to_neg_image_k`
    - `signed_shift_toward_shown_image_k = 0.5 * (shift_neg_caption_to_pos_image_k + shift_pos_caption_to_neg_image_k)`
    """
    if isinstance(p_auth_pos, (int, float)) and isinstance(p_auth_neg, (int, float)):
        pa_pos = float(p_auth_pos)
        pa_neg = float(p_auth_neg)
        ps_pos_shown = float(p_swap_pos_shown_neg_cap)
        ps_neg_shown = float(p_swap_neg_shown_pos_cap)
        delta_auth = pa_pos - pa_neg
        shift_neg_to_pos = ps_pos_shown - pa_neg
        shift_pos_to_neg = pa_pos - ps_neg_shown
        signed_drop_on_neg = -shift_pos_to_neg
        signed_shift = 0.5 * (shift_neg_to_pos + shift_pos_to_neg)
        return {
            "delta_auth": delta_auth,
            "shift_neg_caption_to_pos_image": shift_neg_to_pos,
            "shift_pos_caption_to_neg_image": shift_pos_to_neg,
            "signed_drop_on_neg_image": signed_drop_on_neg,
            "signed_shift_toward_shown_image": signed_shift,
        }

    np = _require_numpy()
    pa_pos = np.asarray(p_auth_pos, dtype=np.float64)
    pa_neg = np.asarray(p_auth_neg, dtype=np.float64)
    ps_pos_shown = np.asarray(p_swap_pos_shown_neg_cap, dtype=np.float64)
    ps_neg_shown = np.asarray(p_swap_neg_shown_pos_cap, dtype=np.float64)
    check(
        pa_pos.ndim == 1
        and pa_pos.shape == pa_neg.shape == ps_pos_shown.shape == ps_neg_shown.shape,
        "Mismatched 1D pair probability vector shapes in compute_e23_pair_shifts",
    )
    delta_auth = pa_pos - pa_neg
    shift_neg_to_pos = ps_pos_shown - pa_neg
    shift_pos_to_neg = pa_pos - ps_neg_shown
    signed_drop_on_neg = -shift_pos_to_neg
    signed_shift = 0.5 * (shift_neg_to_pos + shift_pos_to_neg)
    return {
        "delta_auth": delta_auth,
        "shift_neg_caption_to_pos_image": shift_neg_to_pos,
        "shift_pos_caption_to_neg_image": shift_pos_to_neg,
        "signed_drop_on_neg_image": signed_drop_on_neg,
        "signed_shift_toward_shown_image": signed_shift,
    }


def compute_e23_grounding_fractions(
    mean_signed_shift,
    mean_delta_auth,
    *,
    denom_floor: float = RATIO_DENOM_FLOOR,
):
    """Compute `(visual_grounding_fraction, caption_dominance_fraction)` when `|mean(delta_auth)| > 1e-12`."""
    if mean_signed_shift is None or mean_delta_auth is None:
        return None, None
    da = float(mean_delta_auth)
    ss = float(mean_signed_shift)
    if abs(da) <= float(denom_floor):
        return None, None
    vgf = float(ss / da)
    cdf = float(1.0 - vgf)
    return vgf, cdf


def e23_category_support_status(value, k_pairs: int, minimum: int, *, unestimable: bool = False):
    """Determine category support status and reason for an E23 swap metric."""
    if unestimable:
        return "undefined", "unestimable_head"
    if int(k_pairs) <= 0:
        return "undefined", "zero_opposite_label_pairs"
    if value is None:
        return "undefined", "near_zero_delta_auth_denominator"
    if int(k_pairs) >= int(minimum):
        return "ok", None
    return "descriptive_low_support", f"pair_support_below_minimum_{int(minimum)}"


def build_e23_primary_cohort_audit(
    per_category_plan: Mapping[str, Any],
    category_order,
    primary_categories,
    minimum: int,
):
    """Summarize pair support across the 14 fixed primary categories and all 25 categories."""
    cats = list(category_order)
    prim = list(primary_categories)
    min_pairs = int(minimum)

    per_cat_pairs = {}
    per_cat_status = {}
    for cat in cats:
        c_info = per_category_plan[cat]
        k = int(c_info["n_pairs"])
        check(k == len(c_info["pairs"]), f"Category {cat} n_pairs mismatch with pairs list")
        per_cat_pairs[cat] = k
        expected_status = (
            "ok" if k >= min_pairs else ("descriptive_low_support" if k > 0 else "undefined")
        )
        check(
            c_info.get("support_status") == expected_status,
            f"Category {cat} support_status mismatch: {c_info.get('support_status')!r} vs {expected_status!r}",
        )
        per_cat_status[cat] = expected_status

    supported_primary = [c for c in prim if per_cat_pairs[c] >= min_pairs]
    low_support_primary = [c for c in prim if 0 < per_cat_pairs[c] < min_pairs]
    undefined_primary = [c for c in prim if per_cat_pairs[c] == 0]

    supported_all = [c for c in cats if per_cat_pairs[c] >= min_pairs]
    low_support_all = [c for c in cats if 0 < per_cat_pairs[c] < min_pairs]
    undefined_all = [c for c in cats if per_cat_pairs[c] == 0]

    prim_pair_counts = [per_cat_pairs[c] for c in prim]
    min_prim_pairs = min(prim_pair_counts) if prim_pair_counts else 0
    max_prim_pairs = max(prim_pair_counts) if prim_pair_counts else 0

    return {
        "primary_categories": prim,
        "category_order": cats,
        "primary_min_pairs": min_pairs,
        "all_primary_meet_min_pairs": bool(len(supported_primary) == len(prim) and len(prim) > 0),
        "min_primary_pairs": int(min_prim_pairs),
        "max_primary_pairs": int(max_prim_pairs),
        "n_primary_categories": len(prim),
        "n_supported_primary_categories": len(supported_primary),
        "n_descriptive_low_support_primary_categories": len(low_support_primary),
        "n_undefined_primary_categories": len(undefined_primary),
        "supported_primary_categories": supported_primary,
        "descriptive_low_support_primary_categories": low_support_primary,
        "undefined_primary_categories": undefined_primary,
        "n_all_categories": len(cats),
        "n_supported_all_categories": len(supported_all),
        "n_descriptive_low_support_all_categories": len(low_support_all),
        "n_undefined_all_categories": len(undefined_all),
        "supported_all_categories": supported_all,
        "descriptive_low_support_all_categories": low_support_all,
        "undefined_all_categories": undefined_all,
        "per_category_n_pairs": per_cat_pairs,
        "per_category_support_status": per_cat_status,
    }


def _interval_record(draw_values):
    from src.stage4.metrics import bootstrap_interval

    interval = bootstrap_interval(draw_values)
    if interval["valid_draws"] == 0:
        interval["reason"] = "no_valid_bootstrap_draws"
    return interval


def _mean_optional(values):
    if not values or any(v is None for v in values):
        return None
    return float(sum(values) / len(values))


def _diff_optional(a, b):
    if a is None or b is None:
        return None
    return float(a - b)


def _score_matrix(matrix, coef, intercept, stats, estimable):
    """Score feature matrix `(N, D)` in float64 identically to Stage 4."""
    np = _require_numpy()
    from src.stage3.engine import transform

    logits = np.asarray(transform(matrix, stats) @ coef.T, dtype=np.float64) + intercept
    check(logits.dtype == np.float64 and bool(np.isfinite(logits).all()), "Nonfinite E23 logits")
    estimable_bool = np.asarray(estimable, dtype=bool)
    check(logits.shape[1] == estimable_bool.shape[0], "E23 logits/estimable column mismatch")
    check(not bool(np.any(logits[:, ~estimable_bool])), "Nonzero logits on unestimable E23 columns")
    return logits


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
        check(bool(np.isfinite(arr).all()), f"Nonfinite values in {path.name}")
    return arr


def _run_en_matched_self_checks(
    *,
    model: str,
    depth: int,
    stage2_features,
    frozen_heads,
    stage4_preds,
):
    """Verify in-memory `en->en` matched reproduction (`atol=0, rtol=0`) for both poolings."""
    np = _require_numpy()
    results = {}
    for pooling in POOLINGS:
        rows_en, mat_en = stage2_features[pooling]["en"]
        coef_en, intercept_en, estimable_en, _, stats_en = frozen_heads[pooling]["en"]
        computed = _score_matrix(mat_en, coef_en, intercept_en, stats_en, estimable_en)
        accepted = stage4_preds[pooling]["en"]["en"]
        check(
            plain(rows_en) == plain(accepted.rows),
            f"SCREEN en->en row alignment mismatch for {model}/d{depth}/{pooling}",
        )
        check(
            computed.shape == accepted.logits.shape,
            f"SCREEN en->en shape mismatch for {model}/d{depth}/{pooling}: {computed.shape} vs {accepted.logits.shape}",
        )
        max_abs_diff = float(np.max(np.abs(computed - accepted.logits)))
        check(
            bool(np.array_equal(computed, accepted.logits)) and max_abs_diff == 0.0,
            f"SCREEN en->en matched self-check failed for {model}/d{depth}/{pooling}: max_abs_diff={max_abs_diff}",
        )
        results[pooling] = {
            "route": "en->en",
            "depth": int(depth),
            "condition": E23_CONDITION,
            "pooling": pooling,
            "passed": True,
            "max_abs_diff": 0.0,
            "atol": EN_MATCHED_SELF_CHECK_TOL,
            "rtol": EN_MATCHED_SELF_CHECK_TOL,
            "row_count": int(computed.shape[0]),
            "category_count": int(computed.shape[1]),
        }
    return results


def _verify_authentic_anchor_parity(
    *,
    root: Path,
    model: str,
    depth: int,
    dimension: int,
    e23_plan: Mapping[str, Any],
    rel_ref: Mapping[str, str],
    rel_body: Mapping[str, Any],
    rel_marker: Mapping[str, Any],
    stage2_features,
    frozen_heads,
    stage4_preds,
    tol: float,
):
    """Verify the 8 authentic parity rows against Stage 2 features and Stage 4 logits (`<= 1e-4`)."""
    np = _require_numpy()
    from src.stage4.metrics import probabilities

    rel_folder = within(root, rel_ref["path"]).parent
    parity_rows_path = rel_folder / "authentic_parity_rows.jsonl"
    check(parity_rows_path.is_file(), "Missing authentic_parity_rows.jsonl in E23 release")
    check(
        file_sha256(parity_rows_path)
        == rel_marker["files"]["authentic_parity_rows.jsonl"]
        == rel_body["authentic_parity_rows_sha256"],
        "Authentic parity rows.jsonl SHA-256 mismatch",
    )
    parity_rows = _read_jsonl(parity_rows_path)
    expected_parity_rows = _expected_authentic_parity_rows(e23_plan["authentic_parity_sample"])
    check(
        plain(parity_rows) == plain(expected_parity_rows) and len(parity_rows) == 8,
        "Authentic parity rows.jsonl contents mismatch with phase3_gpu_plan authentic_parity_sample",
    )

    parity_arrays = {}
    for pooling in POOLINGS:
        fname = f"authentic_parity_{pooling}.npy"
        parity_arrays[pooling] = _verify_npy_file(
            rel_folder / fname,
            rel_marker["files"][fname],
            "<f4",
            (len(parity_rows), dimension),
        )

    # Index Stage 2 SCREEN rows and Stage 4 SCREEN rows by (pooling, target_lang, image_id)
    s2_index = {p: {} for p in POOLINGS}
    s4_index = {p: {} for p in POOLINGS}
    for pooling in POOLINGS:
        for lang in LANGUAGES:
            s2_rows, _ = stage2_features[pooling][lang]
            s2_index[pooling][lang] = {r["image_id"]: idx for idx, r in enumerate(s2_rows)}
            s4_rows = stage4_preds[pooling][lang]["en"].rows
            s4_index[pooling][lang] = {r["image_id"]: idx for idx, r in enumerate(s4_rows)}

    per_pooling = {}
    overall_max_feature_diff = 0.0
    overall_max_logit_diff = 0.0
    overall_max_prob_diff = 0.0

    for pooling in POOLINGS:
        p_arr = parity_arrays[pooling]
        max_feat_diff_pool = 0.0
        max_logit_diff_pool = 0.0
        max_prob_diff_pool = 0.0
        row_details = []

        for r_idx, row in enumerate(parity_rows):
            require_open_split(row["split"])
            check(row["split"] == E23_SPLIT, f"Authentic parity row {r_idx} split must be {E23_SPLIT!r}")
            iid = row["image_id"]
            t_lang = row["language"]
            check(
                iid in s2_index[pooling][t_lang],
                f"Authentic parity image_id {iid!r} missing from Stage 2 {pooling}/{t_lang} SCREEN features",
            )
            check(
                iid in s4_index[pooling][t_lang],
                f"Authentic parity image_id {iid!r} missing from Stage 4 {pooling}/{t_lang} SCREEN predictions",
            )
            s2_row_idx = s2_index[pooling][t_lang][iid]
            s4_row_idx = s4_index[pooling][t_lang][iid]

            s2_rows, s2_mat = stage2_features[pooling][t_lang]
            s2_row_meta = s2_rows[s2_row_idx]
            if "caption_id" in s2_row_meta:
                check(
                    s2_row_meta["caption_id"] == row["caption_id"],
                    f"Authentic parity caption_id mismatch for {iid}/{t_lang}",
                )

            rel_vec = np.asarray(p_arr[r_idx], dtype=np.float64)
            s2_vec = np.asarray(s2_mat[s2_row_idx], dtype=np.float64)
            feat_diff = float(np.max(np.abs(rel_vec - s2_vec)))
            check(
                np.isfinite(feat_diff) and feat_diff <= float(tol),
                f"Authentic anchor feature parity exceeded tolerance for {model}/{pooling}/{iid}/{t_lang}: "
                f"max_abs_diff={feat_diff} > tol={tol}",
            )
            max_feat_diff_pool = max(max_feat_diff_pool, feat_diff)

            max_logit_diff_row = 0.0
            max_prob_diff_row = 0.0
            for s_lang in LANGUAGES:
                coef, intercept, estimable, _, stats = frozen_heads[pooling][s_lang]
                scored_logits = _score_matrix(p_arr[r_idx : r_idx + 1], coef, intercept, stats, estimable)[0]
                s4_logits = np.asarray(
                    stage4_preds[pooling][t_lang][s_lang].logits[s4_row_idx], dtype=np.float64
                )
                l_diff = float(np.max(np.abs(scored_logits - s4_logits)))
                check(
                    np.isfinite(l_diff) and l_diff <= float(tol),
                    f"Authentic anchor logit parity exceeded tolerance for {model}/{pooling}/{s_lang}->{t_lang}/{iid}: "
                    f"max_abs_diff={l_diff} > tol={tol}",
                )
                p_diff = float(np.max(np.abs(probabilities(scored_logits) - probabilities(s4_logits))))
                check(
                    np.isfinite(p_diff) and p_diff <= float(tol),
                    f"Authentic anchor probability parity exceeded tolerance for {model}/{pooling}/{s_lang}->{t_lang}/{iid}: "
                    f"max_abs_diff={p_diff} > tol={tol}",
                )
                max_logit_diff_row = max(max_logit_diff_row, l_diff)
                max_prob_diff_row = max(max_prob_diff_row, p_diff)

            max_logit_diff_pool = max(max_logit_diff_pool, max_logit_diff_row)
            max_prob_diff_pool = max(max_prob_diff_pool, max_prob_diff_row)
            row_details.append({
                "row_index": int(r_idx),
                "image_id": iid,
                "language": t_lang,
                "caption_id": row["caption_id"],
                "max_feature_abs_diff": feat_diff,
                "max_logit_abs_diff": max_logit_diff_row,
                "max_prob_abs_diff": max_prob_diff_row,
                "passed": True,
            })

        overall_max_feature_diff = max(overall_max_feature_diff, max_feat_diff_pool)
        overall_max_logit_diff = max(overall_max_logit_diff, max_logit_diff_pool)
        overall_max_prob_diff = max(overall_max_prob_diff, max_prob_diff_pool)
        per_pooling[pooling] = {
            "pooling": pooling,
            "rows_checked": len(parity_rows),
            "max_feature_abs_diff": max_feat_diff_pool,
            "max_logit_abs_diff": max_logit_diff_pool,
            "max_prob_abs_diff": max_prob_diff_pool,
            "tol": float(tol),
            "passed": True,
            "rows": row_details,
        }

    return {
        "model": model,
        "midpoint_depth": int(depth),
        "condition": E23_CONDITION,
        "split": E23_SPLIT,
        "rows_checked": len(parity_rows),
        "tol": float(tol),
        "max_feature_abs_diff": overall_max_feature_diff,
        "max_logit_abs_diff": overall_max_logit_diff,
        "max_prob_abs_diff": overall_max_prob_diff,
        "passed": True,
        "per_pooling": per_pooling,
    }


def _load_and_index_swapped_chunks(
    *,
    root: Path,
    protocol: Mapping[str, Any],
    gpu_plan,
    rel_body: Mapping[str, Any],
    model: str,
    depth: int,
    dimension: int,
    screen_id_set: set,
    categories,
):
    """Load all verified swapped E23 chunks and index `(shown_image_id, caption_owner_image_id, language)`."""
    np = _require_numpy()
    e23_plan = gpu_plan["e23_plan"]
    plan_chunks = e23_plan["chunks"]
    rel_chunks = rel_body["chunks"]
    check(len(rel_chunks) == len(plan_chunks), "E23 release chunk count differs from phase3_gpu_plan")

    swapped_rows = []
    vecs_by_pool = {p: [] for p in POOLINGS}
    swap_view_index = {}
    directed_pairs_by_lang = {lang: set() for lang in LANGUAGES}
    global_idx = 0

    for chunk_index, (rel_chunk, plan_chunk) in enumerate(zip(rel_chunks, plan_chunks)):
        check(int(rel_chunk["chunk_index"]) == chunk_index, f"Chunk ordering mismatch at {chunk_index}")
        chunk_folder = within(root, rel_chunk["ref"]["path"]).parent
        rows_path = chunk_folder / "rows.jsonl"
        check(rows_path.is_file(), f"Chunk {chunk_index} missing rows.jsonl")
        check(file_sha256(rows_path) == rel_chunk["rows_sha256"], f"Chunk {chunk_index} rows.jsonl SHA-256 mismatch")

        chunk_rows = _read_jsonl(rows_path)
        expected_rows = _expected_swapped_chunk_rows(plan_chunk)
        check(
            plain(chunk_rows) == plain(expected_rows),
            f"Chunk {chunk_index} rows.jsonl contents mismatch with phase3_gpu_plan",
        )
        n_views = len(chunk_rows)

        for r_i, row in enumerate(chunk_rows):
            require_open_split(row["split"])
            check(row["split"] == E23_SPLIT, f"Chunk {chunk_index} row {r_i} split must be {E23_SPLIT!r}")
            shown_id = row["shown_image_id"]
            owner_id = row["caption_owner_image_id"]
            lang = row["language"]
            check(shown_id != owner_id, f"Chunk {chunk_index} row {r_i} shown_image_id == caption_owner_image_id")
            check(
                shown_id in screen_id_set and owner_id in screen_id_set,
                f"Chunk {chunk_index} row {r_i} contains non-SCREEN image ID",
            )
            check(lang in LANGUAGES, f"Chunk {chunk_index} row {r_i} invalid language {lang!r}")
            key = (shown_id, owner_id, lang)
            check(key not in swap_view_index, f"Duplicate swapped view key in E23 release: {key!r}")
            swap_view_index[key] = global_idx
            directed_pairs_by_lang[lang].add((shown_id, owner_id))
            swapped_rows.append({
                "global_row_index": int(global_idx),
                "chunk_index": int(chunk_index),
                "chunk_id": plan_chunk["chunk_id"],
                "chunk_row_index": int(r_i),
                "shown_image_id": shown_id,
                "shown_image_path": row["shown_image_path"],
                "shown_image_sha256": row["shown_image_sha256"],
                "caption_owner_image_id": owner_id,
                "language": lang,
                "split": E23_SPLIT,
                "caption_id": row["caption_id"],
                "caption_draw_key": int(row["caption_draw_key"]),
                "caption_text_sha256": row["caption_text_sha256"],
            })
            global_idx += 1

        for pooling in POOLINGS:
            arr = np.load(chunk_folder / f"{pooling}.npy", allow_pickle=False)
            check(
                arr.dtype == np.dtype("<f4")
                and arr.shape == (n_views, dimension)
                and bool(np.isfinite(arr).all()),
                f"Invalid chunk {chunk_index} {pooling}.npy array",
            )
            vecs_by_pool[pooling].append(arr)

    check(
        len(swapped_rows) == int(e23_plan["total_swapped_views"]),
        "Total loaded swapped views mismatch with e23_plan",
    )

    # Verify every category pair's two directed swaps exist for all 4 target languages
    per_category_plan = e23_plan["per_category"]
    for cat in categories:
        cat_info = per_category_plan[cat]
        for pair in cat_info["pairs"]:
            pos_id = pair["positive_image_id"]
            neg_id = pair["negative_image_id"]
            check(pos_id != neg_id, f"Category {cat} pair has identical positive and negative image IDs")
            for lang in LANGUAGES:
                check(
                    (pos_id, neg_id, lang) in swap_view_index,
                    f"Missing swapped view (shown={pos_id}, owner={neg_id}, lang={lang}) for category {cat}",
                )
                check(
                    (neg_id, pos_id, lang) in swap_view_index,
                    f"Missing swapped view (shown={neg_id}, owner={pos_id}, lang={lang}) for category {cat}",
                )

    stacked_features = {
        pooling: np.ascontiguousarray(np.concatenate(vecs_by_pool[pooling], axis=0), dtype="<f4")
        for pooling in POOLINGS
    }
    for pooling in POOLINGS:
        check(
            stacked_features[pooling].shape == (len(swapped_rows), dimension),
            f"Concatenated swapped feature matrix shape mismatch for {pooling}",
        )

    return swapped_rows, stacked_features, swap_view_index, directed_pairs_by_lang


def _score_or_resume_swapped_predictions(
    *,
    root: Path,
    protocol: Mapping[str, Any],
    gpu_plan,
    rel_ref: Mapping[str, str],
    model: str,
    depth: int,
    categories,
    swapped_rows,
    swapped_features,
    frozen_heads,
    gids_by_pool,
    matched_self_checks,
    authentic_parity_report,
    expected_inputs: Mapping[str, Any],
    producer: Mapping[str, Any],
    resume_predictions_ref=None,
    run_id: str,
):
    """Score all swapped views through frozen Stage 3 heads or verify resumed predictions artifact."""
    np = _require_numpy()
    n_views = len(swapped_rows)
    n_cats = len(categories)
    n_src = len(LANGUAGES)
    rows_bytes = jsonl(swapped_rows)
    rows_sha256 = hashlib.sha256(rows_bytes).hexdigest()

    expected_estimable = {}
    expected_prevalence = {}
    for pooling in POOLINGS:
        expected_estimable[pooling] = np.asarray(
            [frozen_heads[pooling][s][2] for s in LANGUAGES], dtype=bool
        )
        expected_prevalence[pooling] = np.asarray(
            [frozen_heads[pooling][s][3] for s in LANGUAGES], dtype=np.float64
        )

    if resume_predictions_ref is not None:
        pred_ref, pred_body, pred_marker = verify_resume_ref(
            root,
            protocol,
            resume_predictions_ref,
            KIND_PREDICTION,
            expected_schema=SCHEMA_PREDICTION,
            expected_dataset=gpu_plan["dataset_fingerprint"],
            expected_plan_identity=gpu_plan["plan_identity"],
            expected_owner=model,
            expected_inputs=expected_inputs,
        )
        check(pred_body.get("experiment") == E23_EXPERIMENT, "Resumed E23 predictions experiment mismatch")
        check(pred_body.get("confirm_excluded") is True, "Resumed E23 predictions confirm_excluded must be True")
        check(pred_body.get("e35_excluded") is True, "Resumed E23 predictions e35_excluded must be True")
        check(pred_body.get("rows_sha256") == rows_sha256, "Resumed E23 predictions rows_sha256 mismatch")

        pred_folder = within(root, pred_ref["path"]).parent
        check("rows.jsonl" in pred_marker["files"], "Resumed E23 predictions missing rows.jsonl")
        check(
            pred_marker["files"]["rows.jsonl"] == rows_sha256
            and file_sha256(pred_folder / "rows.jsonl") == rows_sha256,
            "Resumed E23 predictions rows.jsonl SHA-256 mismatch on disk",
        )

        logits_by_pool = {}
        estimable_by_pool = {}
        prevalence_by_pool = {}
        for pooling in POOLINGS:
            l_name = f"{pooling}_logits.npy"
            e_name = f"{pooling}_estimable.npy"
            p_name = f"{pooling}_prevalence.npy"
            for fname in (l_name, e_name, p_name):
                check(fname in pred_marker["files"], f"Resumed E23 predictions missing {fname}")
            logits_arr = _verify_npy_file(
                pred_folder / l_name, pred_marker["files"][l_name], "<f8", (n_src, n_views, n_cats)
            )
            est_arr = _verify_npy_file(
                pred_folder / e_name, pred_marker["files"][e_name], "|b1", (n_src, n_cats)
            )
            prev_arr = _verify_npy_file(
                pred_folder / p_name, pred_marker["files"][p_name], "<f8", (n_src, n_cats)
            )
            check(
                not bool(np.any(logits_arr * (~est_arr[:, None, :]))),
                f"Nonzero unestimable logits in resumed E23 predictions for {pooling}",
            )
            check(
                bool(np.array_equal(est_arr, expected_estimable[pooling])),
                f"Resumed E23 predictions estimable mismatch with frozen Stage 3 heads for {pooling}",
            )
            check(
                bool(np.allclose(prev_arr, expected_prevalence[pooling], rtol=0.0, atol=1e-12)),
                f"Resumed E23 predictions prevalence mismatch with frozen Stage 3 heads for {pooling}",
            )
            logits_by_pool[pooling] = logits_arr
            estimable_by_pool[pooling] = est_arr
            prevalence_by_pool[pooling] = prev_arr

        return plain(pred_ref), logits_by_pool, estimable_by_pool, prevalence_by_pool, True

    ordinary = gpu_plan.binding.stage4_plan["ordinary"]
    logits_by_pool = {}
    estimable_by_pool = {}
    prevalence_by_pool = {}
    source_heads_meta = {p: {} for p in POOLINGS}
    upstream = [plain(gpu_plan.reference), plain(rel_ref)]

    for pooling in POOLINGS:
        x_swap = swapped_features[pooling]
        logits_list = []
        valid_list = []
        prev_list = []
        for s_idx, s_lang in enumerate(LANGUAGES):
            src_gid = gids_by_pool[pooling][s_lang]
            coef, intercept, estimable, prevalence, stats = frozen_heads[pooling][s_lang]
            logits_s = _score_matrix(x_swap, coef, intercept, stats, estimable)
            logits_list.append(logits_s)
            valid_list.append(estimable)
            prev_list.append(prevalence)

            src_sel_ref = plain(ordinary[src_gid]["selection"])
            src_cand_ref = plain(ordinary[src_gid]["candidate"])
            src_stats_ref = plain(stats["reference"])
            upstream.extend([src_sel_ref, src_cand_ref, src_stats_ref])
            source_heads_meta[pooling][s_lang] = {
                "source_language": s_lang,
                "row_index": int(s_idx),
                "source_group": src_gid,
                "selection_ref": src_sel_ref,
                "candidate_ref": src_cand_ref,
                "statistics_ref": src_stats_ref,
                "C": float(ordinary[src_gid]["C"]),
            }

        logits_arr = np.ascontiguousarray(np.asarray(logits_list, dtype=np.float64), dtype="<f8")
        est_arr = np.ascontiguousarray(np.asarray(valid_list, dtype=bool))
        prev_arr = np.ascontiguousarray(np.asarray(prev_list, dtype=np.float64), dtype="<f8")
        check(
            logits_arr.shape == (n_src, n_views, n_cats) and bool(np.isfinite(logits_arr).all()),
            f"Invalid E23 swapped logits array for {pooling}",
        )
        check(
            not bool(np.any(logits_arr * (~est_arr[:, None, :]))),
            f"Nonzero logits on unestimable E23 columns for {pooling}",
        )
        logits_by_pool[pooling] = logits_arr
        estimable_by_pool[pooling] = est_arr
        prevalence_by_pool[pooling] = prev_arr

    pred_body = {
        "schema": SCHEMA_PREDICTION,
        "experiment": E23_EXPERIMENT,
        "plan": plain(gpu_plan.reference),
        "plan_identity": gpu_plan["plan_identity"],
        "release_ref": plain(rel_ref),
        "model": model,
        "midpoint_depth": int(depth),
        "condition": E23_CONDITION,
        "poolings": list(POOLINGS),
        "split": E23_SPLIT,
        "source_languages": list(LANGUAGES),
        "target_languages": list(LANGUAGES),
        "n_swapped_views": int(n_views),
        "n_unique_directed_pairs": int(gpu_plan["e23_plan"]["total_unique_directed_pairs"]),
        "n_categories": int(n_cats),
        "category_order": list(categories),
        "rows_sha256": rows_sha256,
        "inputs": plain(expected_inputs),
        "source_heads": source_heads_meta,
        "matched_self_checks": matched_self_checks,
        "authentic_anchor_parity_summary": {
            "passed": authentic_parity_report["passed"],
            "max_feature_abs_diff": authentic_parity_report["max_feature_abs_diff"],
            "max_logit_abs_diff": authentic_parity_report["max_logit_abs_diff"],
            "tol": authentic_parity_report["tol"],
        },
        "zero_model_scores_in_pairing": True,
        "e35_excluded": True,
        "confirm_excluded": True,
    }

    pred_ref = publish_phase3(
        root,
        protocol,
        gpu_plan["dataset_fingerprint"],
        f"{E23_PREDICTIONS_NAMESPACE}/{model}",
        KIND_PREDICTION,
        pred_body,
        producer,
        arrays={
            "last_caption_logits": logits_by_pool["last_caption"],
            "last_caption_estimable": estimable_by_pool["last_caption"],
            "last_caption_prevalence": prevalence_by_pool["last_caption"],
            "caption_mean_logits": logits_by_pool["caption_mean"],
            "caption_mean_estimable": estimable_by_pool["caption_mean"],
            "caption_mean_prevalence": prevalence_by_pool["caption_mean"],
        },
        files={"rows.jsonl": rows_bytes},
        upstream=upstream,
        run_id=run_id,
    )
    return plain(pred_ref), logits_by_pool, estimable_by_pool, prevalence_by_pool, False


def compute_e23_category_point_and_draws(
    *,
    p_auth_pos,
    p_auth_neg,
    p_swap_pos_shown_neg_cap,
    p_swap_neg_shown_pos_cap,
    pair_weights_b,
    pair_weight_sums_b,
    total_draws: int,
    estimable: bool,
    denom_floor: float = RATIO_DENOM_FLOOR,
):
    """Compute point estimates and `(B,)` bootstrap draw arrays for one `(route, pooling, category)`."""
    np = _require_numpy()
    nan_draws = np.full(int(total_draws), np.nan, dtype=np.float64)
    k_pairs = int(len(p_auth_pos)) if p_auth_pos is not None else 0

    if not estimable or k_pairs == 0:
        return (
            {m: None for m in E23_SWAP_METRICS},
            {m: nan_draws.copy() for m in E23_SWAP_METRICS},
        )

    shifts = compute_e23_pair_shifts(
        p_auth_pos,
        p_auth_neg,
        p_swap_pos_shown_neg_cap,
        p_swap_neg_shown_pos_cap,
    )

    points = {}
    draws = {}
    positive_weight_mask = pair_weight_sums_b > 0.0

    for m in E23_LINEAR_SWAP_METRICS:
        vec = np.asarray(shifts[m], dtype=np.float64)
        points[m] = float(np.mean(vec))
        weighted_sums = pair_weights_b @ vec
        draws[m] = np.divide(
            weighted_sums,
            pair_weight_sums_b,
            out=np.full(int(total_draws), np.nan, dtype=np.float64),
            where=positive_weight_mask,
        )

    vgf_point, cdf_point = compute_e23_grounding_fractions(
        points["signed_shift_toward_shown_image"],
        points["delta_auth"],
        denom_floor=denom_floor,
    )
    points["visual_grounding_fraction"] = vgf_point
    points["caption_dominance_fraction"] = cdf_point

    d_shift = draws["signed_shift_toward_shown_image"]
    d_delta = draws["delta_auth"]
    valid_ratio = (
        np.isfinite(d_shift)
        & np.isfinite(d_delta)
        & (np.abs(d_delta) > float(denom_floor))
    )
    d_vgf = np.divide(
        d_shift,
        d_delta,
        out=np.full(int(total_draws), np.nan, dtype=np.float64),
        where=valid_ratio,
    )
    d_cdf = np.where(np.isfinite(d_vgf), 1.0 - d_vgf, np.nan)
    draws["visual_grounding_fraction"] = d_vgf
    draws["caption_dominance_fraction"] = d_cdf
    return points, draws


def _summarize_cohort_macro(
    *,
    cohort_indices,
    cat_values,
    cat_draws,
    cat_k_pairs,
    minimum: int,
    total_draws: int,
):
    """Build equal-weight macro summary and `(B,)` draw array across `cohort_indices`."""
    np = _require_numpy()
    from src.stage4.metrics import fixed_macro

    nan_draws = np.full(int(total_draws), np.nan, dtype=np.float64)
    if not cohort_indices:
        return (
            {
                "value": None,
                "interval": _interval_record(nan_draws),
                "status": "unavailable",
                "reason": "no_categories_in_cohort",
                "support": {
                    "categories": 0,
                    "defined_categories": 0,
                    "ok_categories": 0,
                    "min_pairs": 0,
                    "total_pairs": 0,
                },
            },
            nan_draws,
        )

    cohort_vals = [cat_values[idx] for idx in cohort_indices]
    defined_count = sum(v is not None for v in cohort_vals)
    ok_count = sum(
        cat_values[idx] is not None and int(cat_k_pairs[idx]) >= int(minimum)
        for idx in cohort_indices
    )
    min_k = int(min(cat_k_pairs[idx] for idx in cohort_indices))
    total_k = int(sum(cat_k_pairs[idx] for idx in cohort_indices))

    if defined_count < len(cohort_indices):
        macro_val = None
        macro_draws = nan_draws
        macro_status = "undefined"
        macro_reason = "unresolved_macro_cohort"
    else:
        macro_val = float(fixed_macro([0.0 if v is None else v for v in cat_values], cohort_indices))
        stacked = np.stack(cat_draws, axis=0)
        macro_draws = fixed_macro(stacked, cohort_indices, axis=0)
        if ok_count == len(cohort_indices):
            macro_status = "ok"
            macro_reason = None
        else:
            macro_status = "descriptive_low_support"
            macro_reason = f"category_pair_support_below_minimum_{int(minimum)}"

    summary = {
        "value": macro_val,
        "interval": _interval_record(macro_draws),
        "status": macro_status,
        "reason": macro_reason,
        "support": {
            "categories": len(cohort_indices),
            "defined_categories": int(defined_count),
            "ok_categories": int(ok_count),
            "min_pairs": min_k,
            "total_pairs": total_k,
        },
    }
    return summary, macro_draws


def _summarize_metric_across_categories(
    *,
    categories,
    primary_indices,
    supported_primary_indices,
    all_supported_indices,
    cat_values,
    cat_draws,
    cat_k_pairs,
    cat_support_details,
    cat_unestimable,
    minimum: int,
    total_draws: int,
):
    """Build per-category table and all 3 macro cohorts for one `(route/aggregate, metric)`."""
    per_category = {}
    for col, cat in enumerate(categories):
        val = cat_values[col]
        k_c = int(cat_k_pairs[col])
        status, reason = e23_category_support_status(
            val, k_c, minimum, unestimable=bool(cat_unestimable[col])
        )
        per_category[cat] = {
            "value": None if val is None else float(val),
            "interval": _interval_record(cat_draws[col]),
            "status": status,
            "reason": reason,
            "is_primary": col in primary_indices,
            "n_pairs": k_c,
            "support": dict(cat_support_details[col]),
        }

    sup_prim_summary, sup_prim_draws = _summarize_cohort_macro(
        cohort_indices=supported_primary_indices,
        cat_values=cat_values,
        cat_draws=cat_draws,
        cat_k_pairs=cat_k_pairs,
        minimum=minimum,
        total_draws=total_draws,
    )
    fix_prim_summary, fix_prim_draws = _summarize_cohort_macro(
        cohort_indices=primary_indices,
        cat_values=cat_values,
        cat_draws=cat_draws,
        cat_k_pairs=cat_k_pairs,
        minimum=minimum,
        total_draws=total_draws,
    )
    all_sup_summary, all_sup_draws = _summarize_cohort_macro(
        cohort_indices=all_supported_indices,
        cat_values=cat_values,
        cat_draws=cat_draws,
        cat_k_pairs=cat_k_pairs,
        minimum=minimum,
        total_draws=total_draws,
    )

    # Use supported_primary_macro as primary_macro when non-empty (falling back to fixed_primary_macro if empty).
    primary_macro_summary = (
        sup_prim_summary if supported_primary_indices else fix_prim_summary
    )

    metric_out = {
        "primary_macro": primary_macro_summary,
        "supported_primary_macro": sup_prim_summary,
        "fixed_primary_macro": fix_prim_summary,
        "all_supported_macro": all_sup_summary,
        "categories": per_category,
    }
    macro_draws_out = {
        "supported_primary_macro": sup_prim_draws,
        "fixed_primary_macro": fix_prim_draws,
        "all_supported_macro": all_sup_draws,
    }
    return metric_out, macro_draws_out


def _aggregate_route_group(
    *,
    route_pairs,
    route_cat_vals,
    route_cat_draws,
    route_unestimable,
    categories,
    primary_indices,
    supported_primary_indices,
    all_supported_indices,
    cat_k_pairs,
    cat_support_details,
    minimum: int,
    total_draws: int,
):
    """Average category values and within-draw arrays across `route_pairs`, then compute macros."""
    np = _require_numpy()
    n_cats = len(categories)
    metrics_out = {}

    for m in E23_SWAP_METRICS:
        agg_cat_vals = [
            _mean_optional([route_cat_vals[(s, t)][m][c] for s, t in route_pairs])
            for c in range(n_cats)
        ]
        agg_cat_draws = [
            np.mean(np.stack([route_cat_draws[(s, t)][m][c] for s, t in route_pairs], axis=0), axis=0)
            for c in range(n_cats)
        ]
        agg_unest = [
            any(route_unestimable[(s, t)][c] for s, t in route_pairs)
            for c in range(n_cats)
        ]
        metric_summary, _ = _summarize_metric_across_categories(
            categories=categories,
            primary_indices=primary_indices,
            supported_primary_indices=supported_primary_indices,
            all_supported_indices=all_supported_indices,
            cat_values=agg_cat_vals,
            cat_draws=agg_cat_draws,
            cat_k_pairs=cat_k_pairs,
            cat_support_details=cat_support_details,
            cat_unestimable=agg_unest,
            minimum=minimum,
            total_draws=total_draws,
        )
        metrics_out[m] = metric_summary

    return {
        "route_count": len(route_pairs),
        "routes": [_route_key(s, t) for s, t in route_pairs],
        "metrics": metrics_out,
    }


def _evaluate_swap_analysis(
    *,
    auth_probs,
    swap_probs,
    estimable_by_pool,
    category_pair_specs,
    categories,
    primary_indices,
    supported_primary_indices,
    all_supported_indices,
    cat_k_pairs,
    cat_support_details,
    minimum: int,
    total_draws: int,
):
    """Evaluate all 16 routes, 3 macro cohorts, route-group means, and paired pooling contrasts."""
    n_cats = len(categories)
    off_pairs = _off_diagonal_pairs()
    diag_pairs = _diagonal_pairs()
    all_pairs = _all_route_pairs()

    raw_cat_vals = {p: {} for p in POOLINGS}
    raw_cat_draws = {p: {} for p in POOLINGS}
    raw_unest = {p: {} for p in POOLINGS}
    poolings_out = {}

    for pooling in POOLINGS:
        est_mat = estimable_by_pool[pooling]
        per_route = {}

        for s_idx, s_lang in enumerate(LANGUAGES):
            p_swap_s = swap_probs[(pooling, s_lang)]
            est_s = [bool(est_mat[s_idx, c]) for c in range(n_cats)]

            for t_lang in LANGUAGES:
                rkey = _route_key(s_lang, t_lang)
                p_auth_st = auth_probs[(pooling, s_lang, t_lang)]

                cat_vals_by_m = {m: [] for m in E23_SWAP_METRICS}
                cat_draws_by_m = {m: [] for m in E23_SWAP_METRICS}
                unest_flags = [not est_s[c] for c in range(n_cats)]

                for c_idx in range(n_cats):
                    spec = category_pair_specs[c_idx]
                    if spec["k_pairs"] == 0 or not est_s[c_idx]:
                        pts, drs = compute_e23_category_point_and_draws(
                            p_auth_pos=None,
                            p_auth_neg=None,
                            p_swap_pos_shown_neg_cap=None,
                            p_swap_neg_shown_pos_cap=None,
                            pair_weights_b=None,
                            pair_weight_sums_b=None,
                            total_draws=total_draws,
                            estimable=est_s[c_idx],
                        )
                    else:
                        p_auth_pos = p_auth_st[spec["pos_screen_idx"], c_idx]
                        p_auth_neg = p_auth_st[spec["neg_screen_idx"], c_idx]
                        p_swap_pos_shown = p_swap_s[spec["swap_pos_shown_idx"][t_lang], c_idx]
                        p_swap_neg_shown = p_swap_s[spec["swap_neg_shown_idx"][t_lang], c_idx]
                        pts, drs = compute_e23_category_point_and_draws(
                            p_auth_pos=p_auth_pos,
                            p_auth_neg=p_auth_neg,
                            p_swap_pos_shown_neg_cap=p_swap_pos_shown,
                            p_swap_neg_shown_pos_cap=p_swap_neg_shown,
                            pair_weights_b=spec["pair_weights_b"],
                            pair_weight_sums_b=spec["pair_weight_sums_b"],
                            total_draws=total_draws,
                            estimable=True,
                        )
                    for m in E23_SWAP_METRICS:
                        cat_vals_by_m[m].append(pts[m])
                        cat_draws_by_m[m].append(drs[m])

                raw_cat_vals[pooling][(s_lang, t_lang)] = cat_vals_by_m
                raw_cat_draws[pooling][(s_lang, t_lang)] = cat_draws_by_m
                raw_unest[pooling][(s_lang, t_lang)] = unest_flags

                route_metrics = {}
                for m in E23_SWAP_METRICS:
                    m_summary, _ = _summarize_metric_across_categories(
                        categories=categories,
                        primary_indices=primary_indices,
                        supported_primary_indices=supported_primary_indices,
                        all_supported_indices=all_supported_indices,
                        cat_values=cat_vals_by_m[m],
                        cat_draws=cat_draws_by_m[m],
                        cat_k_pairs=cat_k_pairs,
                        cat_support_details=cat_support_details,
                        cat_unestimable=unest_flags,
                        minimum=minimum,
                        total_draws=total_draws,
                    )
                    route_metrics[m] = m_summary

                per_route[rkey] = {
                    "route": rkey,
                    "pooling": pooling,
                    "source_language": s_lang,
                    "target_language": t_lang,
                    "is_diagonal": bool(s_lang == t_lang),
                    "supervision_label": (
                        SUPERVISION_DIAGONAL if s_lang == t_lang else SUPERVISION_OFF_DIAGONAL
                    ),
                    "metrics": route_metrics,
                }

        twelve_off = _aggregate_route_group(
            route_pairs=off_pairs,
            route_cat_vals=raw_cat_vals[pooling],
            route_cat_draws=raw_cat_draws[pooling],
            route_unestimable=raw_unest[pooling],
            categories=categories,
            primary_indices=primary_indices,
            supported_primary_indices=supported_primary_indices,
            all_supported_indices=all_supported_indices,
            cat_k_pairs=cat_k_pairs,
            cat_support_details=cat_support_details,
            minimum=minimum,
            total_draws=total_draws,
        )
        four_diag = _aggregate_route_group(
            route_pairs=diag_pairs,
            route_cat_vals=raw_cat_vals[pooling],
            route_cat_draws=raw_cat_draws[pooling],
            route_unestimable=raw_unest[pooling],
            categories=categories,
            primary_indices=primary_indices,
            supported_primary_indices=supported_primary_indices,
            all_supported_indices=all_supported_indices,
            cat_k_pairs=cat_k_pairs,
            cat_support_details=cat_support_details,
            minimum=minimum,
            total_draws=total_draws,
        )
        sixteen_all = _aggregate_route_group(
            route_pairs=all_pairs,
            route_cat_vals=raw_cat_vals[pooling],
            route_cat_draws=raw_cat_draws[pooling],
            route_unestimable=raw_unest[pooling],
            categories=categories,
            primary_indices=primary_indices,
            supported_primary_indices=supported_primary_indices,
            all_supported_indices=all_supported_indices,
            cat_k_pairs=cat_k_pairs,
            cat_support_details=cat_support_details,
            minimum=minimum,
            total_draws=total_draws,
        )

        poolings_out[pooling] = {
            "pooling": pooling,
            "twelve_off_diagonal_mean": twelve_off,
            "four_diagonal_mean": four_diag,
            "sixteen_all_route_mean": sixteen_all,
            "per_route": per_route,
        }

    # Compute paired `caption_mean - last_caption` contrasts on identical (2000, 360) draws
    diff_cat_vals = {}
    diff_cat_draws = {}
    diff_unest = {}
    contrast_per_route = {}

    for s_lang, t_lang in all_pairs:
        rkey = _route_key(s_lang, t_lang)
        diff_cat_vals[(s_lang, t_lang)] = {}
        diff_cat_draws[(s_lang, t_lang)] = {}
        unest_pair = [
            raw_unest["caption_mean"][(s_lang, t_lang)][c]
            or raw_unest["last_caption"][(s_lang, t_lang)][c]
            for c in range(n_cats)
        ]
        diff_unest[(s_lang, t_lang)] = unest_pair

        route_diff_metrics = {}
        for m in E23_SWAP_METRICS:
            vals_mean = raw_cat_vals["caption_mean"][(s_lang, t_lang)][m]
            vals_last = raw_cat_vals["last_caption"][(s_lang, t_lang)][m]
            draws_mean = raw_cat_draws["caption_mean"][(s_lang, t_lang)][m]
            draws_last = raw_cat_draws["last_caption"][(s_lang, t_lang)][m]

            c_vals = [_diff_optional(vals_mean[c], vals_last[c]) for c in range(n_cats)]
            c_draws = [draws_mean[c] - draws_last[c] for c in range(n_cats)]
            diff_cat_vals[(s_lang, t_lang)][m] = c_vals
            diff_cat_draws[(s_lang, t_lang)][m] = c_draws

            m_summary, _ = _summarize_metric_across_categories(
                categories=categories,
                primary_indices=primary_indices,
                supported_primary_indices=supported_primary_indices,
                all_supported_indices=all_supported_indices,
                cat_values=c_vals,
                cat_draws=c_draws,
                cat_k_pairs=cat_k_pairs,
                cat_support_details=cat_support_details,
                cat_unestimable=unest_pair,
                minimum=minimum,
                total_draws=total_draws,
            )
            route_diff_metrics[m] = m_summary

        contrast_per_route[rkey] = {
            "route": rkey,
            "contrast": "caption_mean_minus_last_caption",
            "source_language": s_lang,
            "target_language": t_lang,
            "is_diagonal": bool(s_lang == t_lang),
            "supervision_label": (
                SUPERVISION_DIAGONAL if s_lang == t_lang else SUPERVISION_OFF_DIAGONAL
            ),
            "metrics": route_diff_metrics,
        }

    pooling_contrast = {
        "contrast": "caption_mean_minus_last_caption",
        "description": "Paired caption_mean - last_caption swap contrast on identical (2000, 360) bootstrap draws",
        "twelve_off_diagonal_mean": _aggregate_route_group(
            route_pairs=off_pairs,
            route_cat_vals=diff_cat_vals,
            route_cat_draws=diff_cat_draws,
            route_unestimable=diff_unest,
            categories=categories,
            primary_indices=primary_indices,
            supported_primary_indices=supported_primary_indices,
            all_supported_indices=all_supported_indices,
            cat_k_pairs=cat_k_pairs,
            cat_support_details=cat_support_details,
            minimum=minimum,
            total_draws=total_draws,
        ),
        "four_diagonal_mean": _aggregate_route_group(
            route_pairs=diag_pairs,
            route_cat_vals=diff_cat_vals,
            route_cat_draws=diff_cat_draws,
            route_unestimable=diff_unest,
            categories=categories,
            primary_indices=primary_indices,
            supported_primary_indices=supported_primary_indices,
            all_supported_indices=all_supported_indices,
            cat_k_pairs=cat_k_pairs,
            cat_support_details=cat_support_details,
            minimum=minimum,
            total_draws=total_draws,
        ),
        "sixteen_all_route_mean": _aggregate_route_group(
            route_pairs=all_pairs,
            route_cat_vals=diff_cat_vals,
            route_cat_draws=diff_cat_draws,
            route_unestimable=diff_unest,
            categories=categories,
            primary_indices=primary_indices,
            supported_primary_indices=supported_primary_indices,
            all_supported_indices=all_supported_indices,
            cat_k_pairs=cat_k_pairs,
            cat_support_details=cat_support_details,
            minimum=minimum,
            total_draws=total_draws,
        ),
        "per_route": contrast_per_route,
    }

    return poolings_out, pooling_contrast


def evaluate_e23_decision_summary(
    *,
    poolings_summary: Mapping[str, Any],
    pooling_contrast: Mapping[str, Any],
    primary_cohort_audit: Mapping[str, Any],
    decision_ranges: Mapping[str, Any],
    zero_model_scores_in_pairing: bool,
    authentic_anchor_parity_passed: bool,
    max_authentic_parity_diff: float,
    en_matched_self_check_passed: bool,
    min_valid_draws: int = 1900,
):
    """Evaluate `PHASE3_GPU_DECISION_RANGES['E23']` thresholds and sanity checks."""
    shift_min = float(decision_ranges["signed_shift_toward_positive_min"])
    primary_min_pairs = int(decision_ranges["primary_min_pairs"])
    neg_upper_lt = float(decision_ranges["negligible_shift_ci_upper_lt"])
    checks_cfg = decision_ranges.get("checks", {})
    parity_tol = float(checks_cfg.get("authentic_anchor_parity_tol", AUTHENTIC_PARITY_TOL))
    require_zero_scores = bool(checks_cfg.get("zero_model_scores_in_pairing", True))

    def _eval_shift_entry(entry):
        val = entry.get("value")
        iv = entry.get("interval", {})
        lo = iv.get("lower")
        hi = iv.get("upper")
        vd = int(iv.get("valid_draws", 0))
        valid_ok = vd >= int(min_valid_draws)
        ci_pos = bool(lo is not None and lo > 0.0 and valid_ok)
        consequential = bool(val is not None and float(val) >= shift_min and ci_pos)
        negligible = bool(hi is not None and float(hi) < neg_upper_lt and valid_ok)
        return {
            "value": None if val is None else float(val),
            "interval": plain(iv),
            "status": entry.get("status"),
            "ci_strictly_positive": ci_pos,
            "consequential_ge_min": consequential,
            "negligible_ci_upper_lt_bound": negligible,
        }

    def _eval_contrast_entry(entry):
        val = entry.get("value")
        iv = entry.get("interval", {})
        lo = iv.get("lower")
        hi = iv.get("upper")
        vd = int(iv.get("valid_draws", 0))
        valid_ok = vd >= int(min_valid_draws)
        ci_excludes_zero = bool(
            lo is not None and hi is not None and (float(lo) > 0.0 or float(hi) < 0.0) and valid_ok
        )
        return {
            "value": None if val is None else float(val),
            "interval": plain(iv),
            "status": entry.get("status"),
            "ci_excludes_zero": ci_excludes_zero,
        }

    per_pooling = {}
    for pooling in POOLINGS:
        p_block = poolings_summary[pooling]
        p_eval = {}
        for rg_key in ("twelve_off_diagonal_mean", "four_diagonal_mean", "sixteen_all_route_mean"):
            rg_metrics = p_block[rg_key]["metrics"]
            ss_sup = _eval_shift_entry(rg_metrics["signed_shift_toward_shown_image"]["supported_primary_macro"])
            ss_fix = _eval_shift_entry(rg_metrics["signed_shift_toward_shown_image"]["fixed_primary_macro"])
            ss_all = _eval_shift_entry(rg_metrics["signed_shift_toward_shown_image"]["all_supported_macro"])
            np_sup = _eval_shift_entry(rg_metrics["shift_neg_caption_to_pos_image"]["supported_primary_macro"])
            pn_sup = _eval_shift_entry(rg_metrics["shift_pos_caption_to_neg_image"]["supported_primary_macro"])
            vg_sup = rg_metrics["visual_grounding_fraction"]["supported_primary_macro"]
            cd_sup = rg_metrics["caption_dominance_fraction"]["supported_primary_macro"]
            da_sup = rg_metrics["delta_auth"]["supported_primary_macro"]
            p_eval[rg_key] = {
                "signed_shift_toward_shown_image": {
                    "supported_primary_macro": ss_sup,
                    "fixed_primary_macro": ss_fix,
                    "all_supported_macro": ss_all,
                },
                "shift_neg_caption_to_pos_image": {
                    "supported_primary_macro": np_sup,
                },
                "shift_pos_caption_to_neg_image": {
                    "supported_primary_macro": pn_sup,
                },
                "visual_grounding_fraction": {
                    "supported_primary_macro": plain(vg_sup),
                },
                "caption_dominance_fraction": {
                    "supported_primary_macro": plain(cd_sup),
                },
                "delta_auth": {
                    "supported_primary_macro": plain(da_sup),
                },
                "consequential_visual_grounding": bool(
                    ss_sup["consequential_ge_min"] or np_sup["consequential_ge_min"]
                ),
                "negligible_visual_shift": bool(ss_sup["negligible_ci_upper_lt_bound"]),
            }
        per_pooling[pooling] = p_eval

    contrast_eval = {}
    for rg_key in ("twelve_off_diagonal_mean", "four_diagonal_mean", "sixteen_all_route_mean"):
        c_metrics = pooling_contrast[rg_key]["metrics"]
        ss_diff = _eval_contrast_entry(c_metrics["signed_shift_toward_shown_image"]["supported_primary_macro"])
        np_diff = _eval_contrast_entry(c_metrics["shift_neg_caption_to_pos_image"]["supported_primary_macro"])
        vg_diff = _eval_contrast_entry(c_metrics["visual_grounding_fraction"]["supported_primary_macro"])
        cd_diff = _eval_contrast_entry(c_metrics["caption_dominance_fraction"]["supported_primary_macro"])
        contrast_eval[rg_key] = {
            "signed_shift_toward_shown_image": ss_diff,
            "shift_neg_caption_to_pos_image": np_diff,
            "visual_grounding_fraction": vg_diff,
            "caption_dominance_fraction": cd_diff,
            "pooling_changes_signed_shift": ss_diff["ci_excludes_zero"],
            "pooling_changes_visual_grounding": vg_diff["ci_excludes_zero"],
        }

    any_consequential = any(
        per_pooling[p]["twelve_off_diagonal_mean"]["consequential_visual_grounding"]
        or per_pooling[p]["four_diagonal_mean"]["consequential_visual_grounding"]
        for p in POOLINGS
    )
    all_negligible = all(
        per_pooling[p]["twelve_off_diagonal_mean"]["negligible_visual_shift"]
        and per_pooling[p]["four_diagonal_mean"]["negligible_visual_shift"]
        for p in POOLINGS
    )
    if any_consequential:
        outcome = "visual_grounding_consequential"
    elif all_negligible:
        outcome = "caption_dominated_negligible_visual_shift"
    else:
        outcome = "partial_visual_sensitivity"

    sanity_checks = {
        "zero_model_scores_in_pairing": bool(zero_model_scores_in_pairing is True and require_zero_scores),
        "authentic_anchor_parity_passed": bool(
            authentic_anchor_parity_passed and float(max_authentic_parity_diff) <= parity_tol
        ),
        "max_authentic_anchor_abs_diff": float(max_authentic_parity_diff),
        "authentic_anchor_parity_tol": parity_tol,
        "en_matched_self_check_passed": bool(en_matched_self_check_passed),
        "confirm_excluded": True,
        "e35_excluded": True,
    }
    sanity_checks["all_passed"] = bool(
        sanity_checks["zero_model_scores_in_pairing"]
        and sanity_checks["authentic_anchor_parity_passed"]
        and sanity_checks["en_matched_self_check_passed"]
    )

    return {
        "thresholds": {
            "signed_shift_toward_positive_min": shift_min,
            "primary_min_pairs": primary_min_pairs,
            "negligible_shift_ci_upper_lt": neg_upper_lt,
            "min_valid_draws": int(min_valid_draws),
        },
        "sanity_checks": sanity_checks,
        "primary_cohort_support": {
            "all_primary_meet_min_pairs": bool(primary_cohort_audit["all_primary_meet_min_pairs"]),
            "min_primary_pairs": int(primary_cohort_audit["min_primary_pairs"]),
            "n_supported_primary_categories": int(primary_cohort_audit["n_supported_primary_categories"]),
            "supported_primary_categories": list(primary_cohort_audit["supported_primary_categories"]),
            "descriptive_low_support_primary_categories": list(
                primary_cohort_audit["descriptive_low_support_primary_categories"]
            ),
        },
        "per_pooling": per_pooling,
        "caption_mean_minus_last_caption": contrast_eval,
        "any_consequential_visual_grounding": bool(any_consequential),
        "all_negligible_visual_shift": bool(all_negligible),
        "outcome": outcome,
    }


@contract
def run_e23(
    root,
    protocol,
    model: str,
    producer: Mapping[str, Any],
    *,
    plan_ref=None,
    release_ref=None,
    resume_predictions=None,
    resume_analysis=None,
    resume_predictions_ref=None,
    resume_analysis_ref=None,
    run_id: Optional[str] = None,
    progress=None,
):
    """Execute Phase 3 Stage B3 Worker 2 (`run_e23`) for one model (`qwen` or `gemma`)."""
    require_open_split(E23_SPLIT)
    check(model in MODELS, f"Unsupported MODEL selector {model!r}; expected one of {MODELS}")

    resolved_root = Path(root).resolve()
    effective_run_id = run_id or new_run_id()
    eff_resume_preds = resume_predictions if resume_predictions is not None else resume_predictions_ref
    eff_resume_analysis = resume_analysis if resume_analysis is not None else resume_analysis_ref

    total_units = 4
    t_bind = time.monotonic()

    # 1. Authenticate phase3_gpu_plan (bind=True) and E23 Stage B2 release
    effective_plan_ref = plain(plan_ref) if plan_ref is not None else plain(CERTIFIED_STAGE_B1_GPU_PLAN)
    gpu_plan = load_phase3_gpu_plan(resolved_root, protocol, effective_plan_ref, bind=True)
    check(gpu_plan["confirm_excluded"] is True, "phase3_gpu_plan must enforce confirm_excluded=True")
    check(gpu_plan["e35_excluded"] is True, "phase3_gpu_plan must enforce e35_excluded=True (E35 dropped)")

    e23_plan = gpu_plan["e23_plan"]
    check(
        e23_plan.get("zero_model_scores_in_pairing") is True,
        "e23_plan must certify zero_model_scores_in_pairing=True",
    )
    check(
        e23_plan.get("confirm_rows") == 0,
        f"e23_plan must have confirm_rows == 0; got {e23_plan.get('confirm_rows')!r}",
    )
    require_open_split(e23_plan.get("split"))
    check(e23_plan.get("split") == E23_SPLIT, f"e23_plan split must be {E23_SPLIT!r}")

    binding = gpu_plan.binding
    check(binding is not None, "load_phase3_gpu_plan(bind=True) must attach Phase1Binding")
    depth = int(midpoint_depth(binding, model))
    expected_depth = E23_EXPECTED_MIDPOINTS[model]
    check(
        depth == expected_depth == int(gpu_plan["models"][model]["midpoint_depth"]),
        f"Midpoint depth mismatch for {model}: got {depth}, expected {expected_depth}",
    )

    effective_release_ref = (
        plain(release_ref)
        if release_ref is not None
        else plain(CERTIFIED_STAGE_B2_RELEASES["e23"][model])
    )
    rel_ref, rel_body, rel_marker = verify_e23_release(
        resolved_root,
        protocol,
        effective_release_ref,
        expected_plan_identity=gpu_plan["plan_identity"],
        expected_model=model,
    )
    check(int(rel_body["midpoint_depth"]) == depth, "E23 release midpoint_depth mismatch")
    check(rel_body["condition"] == E23_CONDITION, "E23 release condition mismatch")
    check(list(rel_body["poolings"]) == list(POOLINGS), "E23 release poolings mismatch")
    check(rel_body["split"] == E23_SPLIT, "E23 release split mismatch")
    check(rel_body["confirm_excluded"] is True, "E23 release confirm_excluded must be True")
    check(rel_body["e35_excluded"] is True, "E23 release e35_excluded must be True")
    check(rel_body["confirm_rows"] == 0, "E23 release confirm_rows must be 0")
    check(rel_body["zero_model_scores_in_pairing"] is True, "E23 release zero_model_scores_in_pairing must be True")
    check(
        int(rel_body["total_unique_directed_pairs"]) == int(e23_plan["total_unique_directed_pairs"])
        and int(rel_body["total_swapped_views"]) == int(e23_plan["total_swapped_views"])
        and int(rel_body["total_chunks"]) == int(e23_plan["total_chunks"]),
        "E23 release pair/view/chunk counts mismatch with phase3_gpu_plan",
    )

    categories = list(protocol["category_order"])
    primary_categories = list(protocol["primary_categories"])
    primary_indices = [categories.index(c) for c in primary_categories]
    screen_minimum = int(protocol["minimum_positive_and_negative"]["SCREEN"])
    n_cats = len(categories)

    decision_ranges = plain(
        gpu_plan.get("decision_ranges", {}).get("E23", PHASE3_GPU_DECISION_RANGES["E23"])
    )
    parity_tol = float(
        decision_ranges.get("checks", {}).get("authentic_anchor_parity_tol", AUTHENTIC_PARITY_TOL)
    )

    primary_cohort_audit = build_e23_primary_cohort_audit(
        e23_plan["per_category"],
        categories,
        primary_categories,
        screen_minimum,
    )
    supported_primary_indices = [
        categories.index(c) for c in primary_cohort_audit["supported_primary_categories"]
    ]
    all_supported_indices = [
        categories.index(c) for c in primary_cohort_audit["supported_all_categories"]
    ]

    expected_inputs = {
        "experiment": E23_EXPERIMENT,
        "model": model,
        "midpoint_depth": int(depth),
        "condition": E23_CONDITION,
        "poolings": list(POOLINGS),
        "split": E23_SPLIT,
        "plan_ref": plain(gpu_plan.reference),
        "release_ref": plain(rel_ref),
        "total_unique_directed_pairs": int(e23_plan["total_unique_directed_pairs"]),
        "total_swapped_views": int(e23_plan["total_swapped_views"]),
        "total_chunks": int(e23_plan["total_chunks"]),
    }

    if eff_resume_analysis is not None:
        a_ref, a_body, _ = verify_resume_ref(
            resolved_root,
            protocol,
            eff_resume_analysis,
            KIND_ANALYSIS,
            expected_schema=SCHEMA_ANALYSIS,
            expected_dataset=gpu_plan["dataset_fingerprint"],
            expected_plan_identity=gpu_plan["plan_identity"],
            expected_owner=model,
            expected_inputs=expected_inputs,
        )
        check(a_body.get("experiment") == E23_EXPERIMENT, "Resumed E23 analysis experiment mismatch")
        check(a_body.get("confirm_excluded") is True, "Resumed E23 analysis confirm_excluded must be True")
        check(a_body.get("e35_excluded") is True, "Resumed E23 analysis e35_excluded must be True")
        check(
            a_body.get("authentic_anchor_parity", {}).get("passed") is True,
            "Resumed E23 analysis authentic_anchor_parity did not pass",
        )
        _emit_progress(
            progress,
            "e23_analysis_resumed",
            unit=f"{model}:resume_analysis",
            completed=total_units,
            total=total_units,
            read_hash_s=time.monotonic() - t_bind,
            model=model,
            analysis_ref=plain(a_ref),
            predictions_ref=plain(a_body["predictions_ref"]),
        )
        return frozen({
            "status": "passed",
            "model": model,
            "run_id": effective_run_id,
            "midpoint_depth": int(depth),
            "plan_ref": plain(gpu_plan.reference),
            "release_ref": plain(rel_ref),
            "predictions_ref": plain(a_body["predictions_ref"]),
            "analysis_ref": plain(a_ref),
            "authentic_anchor_parity": plain(a_body["authentic_anchor_parity"]),
            "matched_self_checks": plain(a_body["matched_self_checks"]),
            "primary_cohort_audit": plain(a_body["primary_cohort_audit"]),
            "decision_summary": plain(a_body["decision_summary"]),
            "resumed_predictions": True,
            "resumed_analysis": True,
            "e35_excluded": True,
            "confirm_excluded": True,
        })

    np = _require_numpy()
    from src.stage4.metrics import bootstrap_counts, paired_image_alignment, probabilities
    from src.stage4.plan import load_bootstrap
    from src.stage4.predictions import load_prediction_block

    screen_ids = tuple(binding.stage4_plan["screen_ids"])
    screen_id_set = set(screen_ids)
    screen_idx = {iid: idx for idx, iid in enumerate(screen_ids)}
    dimension = int(rel_body["dimension"])

    # Load Stage 2 SCREEN features, frozen Stage 3 heads, and Stage 4 SCREEN prediction blocks
    gids_by_pool = {}
    stage2_features = {p: {} for p in POOLINGS}
    frozen_heads = {p: {} for p in POOLINGS}
    stage4_preds = {p: {t: {} for t in LANGUAGES} for p in POOLINGS}
    s4_blocks_meta = e29_screen_blocks(binding, model)[E23_CONDITION]

    for pooling in POOLINGS:
        gids_by_pool[pooling] = midpoint_groups(
            binding, model, condition=E23_CONDITION, pooling=pooling
        )
        for lang in LANGUAGES:
            s2_rows, s2_mat = read_verified_midpoint_features(
                binding, model, E23_CONDITION, pooling, lang, E23_SPLIT
            )
            check(
                len(s2_rows) == len(screen_ids)
                and all(r["split"] == E23_SPLIT for r in s2_rows)
                and sorted(r["image_id"] for r in s2_rows) == list(screen_ids),
                f"Stage 2 SCREEN row population mismatch for {model}/{pooling}/{lang}",
            )
            check(
                s2_mat.shape == (len(screen_ids), dimension),
                f"Stage 2 SCREEN feature shape mismatch for {model}/{pooling}/{lang}: {s2_mat.shape}",
            )
            stage2_features[pooling][lang] = (s2_rows, s2_mat)
            frozen_heads[pooling][lang] = load_frozen_source_head(
                binding, model, E23_CONDITION, pooling, lang
            )

            b_info = s4_blocks_meta[pooling][lang]
            loaded_block = load_prediction_block(
                binding.stage4_plan,
                b_info["prediction_ref"],
                expected_block=b_info["block_id"],
            )
            for s_lang in LANGUAGES:
                stage4_preds[pooling][lang][s_lang] = loaded_block[b_info["routes"][s_lang]]

    matched_self_checks = _run_en_matched_self_checks(
        model=model,
        depth=depth,
        stage2_features=stage2_features,
        frozen_heads=frozen_heads,
        stage4_preds=stage4_preds,
    )

    authentic_parity_report = _verify_authentic_anchor_parity(
        root=resolved_root,
        model=model,
        depth=depth,
        dimension=dimension,
        e23_plan=e23_plan,
        rel_ref=rel_ref,
        rel_body=rel_body,
        rel_marker=rel_marker,
        stage2_features=stage2_features,
        frozen_heads=frozen_heads,
        stage4_preds=stage4_preds,
        tol=parity_tol,
    )

    swapped_rows, swapped_features, swap_view_index, _ = _load_and_index_swapped_chunks(
        root=resolved_root,
        protocol=protocol,
        gpu_plan=gpu_plan,
        rel_body=rel_body,
        model=model,
        depth=depth,
        dimension=dimension,
        screen_id_set=screen_id_set,
        categories=categories,
    )
    read_s = time.monotonic() - t_bind

    _emit_progress(
        progress,
        "e23_parity_and_chunks_verified",
        unit=f"{model}:verify_and_load",
        completed=1,
        total=total_units,
        read_hash_s=read_s,
        model=model,
        midpoint_depth=depth,
        max_authentic_feature_abs_diff=authentic_parity_report["max_feature_abs_diff"],
        max_authentic_logit_abs_diff=authentic_parity_report["max_logit_abs_diff"],
        total_swapped_views=len(swapped_rows),
    )

    # 2. Score swapped features through frozen Stage 3 heads and publish phase3/predictions/e23/<model>
    t_score = time.monotonic()
    predictions_ref, logits_by_pool, estimable_by_pool, _, resumed_predictions = (
        _score_or_resume_swapped_predictions(
            root=resolved_root,
            protocol=protocol,
            gpu_plan=gpu_plan,
            rel_ref=rel_ref,
            model=model,
            depth=depth,
            categories=categories,
            swapped_rows=swapped_rows,
            swapped_features=swapped_features,
            frozen_heads=frozen_heads,
            gids_by_pool=gids_by_pool,
            matched_self_checks=matched_self_checks,
            authentic_parity_report=authentic_parity_report,
            expected_inputs=expected_inputs,
            producer=producer,
            resume_predictions_ref=eff_resume_preds,
            run_id=effective_run_id,
        )
    )
    score_s = time.monotonic() - t_score
    _emit_progress(
        progress,
        "e23_predictions_resumed" if resumed_predictions else "e23_predictions_published",
        unit=f"{model}:predictions",
        completed=2,
        total=total_units,
        compute_s=0.0 if resumed_predictions else score_s,
        read_hash_s=score_s if resumed_predictions else 0.0,
        model=model,
        predictions_ref=predictions_ref,
    )

    # 3. Paired opposite-label swap analysis on frozen (2000, 360) bootstrap
    t_eval = time.monotonic()
    all_auth_preds = []
    auth_keys = []
    for pooling in POOLINGS:
        for s_lang in LANGUAGES:
            for t_lang in LANGUAGES:
                all_auth_preds.append(stage4_preds[pooling][t_lang][s_lang])
                auth_keys.append((pooling, s_lang, t_lang))

    ids, aligned_tuple = paired_image_alignment(binding.stage4_plan, *all_auth_preds)
    check(ids == screen_ids, "Aligned SCREEN IDs mismatch stage4_plan screen_ids")

    auth_probs = {}
    for (pooling, s_lang, t_lang), pred, aligned in zip(auth_keys, all_auth_preds, aligned_tuple):
        auth_probs[(pooling, s_lang, t_lang)] = probabilities(pred.logits[aligned])

    swap_probs = {}
    for pooling in POOLINGS:
        for s_idx, s_lang in enumerate(LANGUAGES):
            swap_probs[(pooling, s_lang)] = probabilities(logits_by_pool[pooling][s_idx])

    draws = load_bootstrap(binding.stage4_plan)
    check(
        hashlib.sha256(draws.tobytes()).hexdigest() == gpu_plan["bootstrap"]["sha256"],
        "Loaded bootstrap SHA-256 mismatch with phase3_gpu_plan",
    )
    counts = bootstrap_counts(draws, len(screen_ids))
    total_draws = int(draws.shape[0])

    # Verify unknown cell 6a20677eee04fd08 / plant.n.02 = None exclusion
    targets_map = binding.stage4_plan.fit["targets"]
    if "6a20677eee04fd08" in screen_idx and "plant.n.02" in categories:
        plant_idx = categories.index("plant.n.02")
        check(
            targets_map["6a20677eee04fd08"]["values"][plant_idx] is None,
            "Expected 6a20677eee04fd08 / plant.n.02 to be unknown (None)",
        )
        plant_pairs = e23_plan["per_category"]["plant.n.02"]["pairs"]
        check(
            all(
                p["positive_image_id"] != "6a20677eee04fd08"
                and p["negative_image_id"] != "6a20677eee04fd08"
                for p in plant_pairs
            ),
            "Unknown cell 6a20677eee04fd08 / plant.n.02 leaked into E23 pairs",
        )

    category_pair_specs = []
    cat_k_pairs = []
    cat_support_details = []

    for c_idx, cat in enumerate(categories):
        cat_plan = e23_plan["per_category"][cat]
        pairs = cat_plan["pairs"]
        k_pairs = len(pairs)
        cat_k_pairs.append(k_pairs)
        cat_support_details.append({
            "n_pairs": int(k_pairs),
            "n_positive": int(cat_plan["n_positive"]),
            "n_negative": int(cat_plan["n_negative"]),
            "n_unknown": int(cat_plan["n_unknown"]),
        })

        seen_pos = set()
        seen_neg = set()
        for p in pairs:
            pos_id = p["positive_image_id"]
            neg_id = p["negative_image_id"]
            check(
                pos_id not in seen_pos and neg_id not in seen_neg and pos_id != neg_id,
                f"Non-disjoint opposite-label pairs in category {cat}",
            )
            seen_pos.add(pos_id)
            seen_neg.add(neg_id)
            check(
                targets_map[pos_id]["values"][c_idx] == 1
                and targets_map[neg_id]["values"][c_idx] == 0,
                f"Ground-truth opposite-label mismatch for pair ({pos_id}, {neg_id}) in {cat}",
            )

        if k_pairs == 0:
            category_pair_specs.append({"k_pairs": 0})
            continue

        pos_scr_idx = np.array([screen_idx[p["positive_image_id"]] for p in pairs], dtype=np.intp)
        neg_scr_idx = np.array([screen_idx[p["negative_image_id"]] for p in pairs], dtype=np.intp)
        w_b = 0.5 * (
            counts[:, pos_scr_idx].astype(np.float64)
            + counts[:, neg_scr_idx].astype(np.float64)
        )
        w_sum = np.sum(w_b, axis=1)

        swap_pos_shown_idx = {}
        swap_neg_shown_idx = {}
        for t_lang in LANGUAGES:
            swap_pos_shown_idx[t_lang] = np.array(
                [swap_view_index[(p["positive_image_id"], p["negative_image_id"], t_lang)] for p in pairs],
                dtype=np.intp,
            )
            swap_neg_shown_idx[t_lang] = np.array(
                [swap_view_index[(p["negative_image_id"], p["positive_image_id"], t_lang)] for p in pairs],
                dtype=np.intp,
            )

        category_pair_specs.append({
            "k_pairs": k_pairs,
            "pos_screen_idx": pos_scr_idx,
            "neg_screen_idx": neg_scr_idx,
            "pair_weights_b": w_b,
            "pair_weight_sums_b": w_sum,
            "swap_pos_shown_idx": swap_pos_shown_idx,
            "swap_neg_shown_idx": swap_neg_shown_idx,
        })

    poolings_summary, pooling_contrast = _evaluate_swap_analysis(
        auth_probs=auth_probs,
        swap_probs=swap_probs,
        estimable_by_pool=estimable_by_pool,
        category_pair_specs=category_pair_specs,
        categories=categories,
        primary_indices=primary_indices,
        supported_primary_indices=supported_primary_indices,
        all_supported_indices=all_supported_indices,
        cat_k_pairs=cat_k_pairs,
        cat_support_details=cat_support_details,
        minimum=screen_minimum,
        total_draws=total_draws,
    )

    max_auth_diff = max(
        authentic_parity_report["max_feature_abs_diff"],
        authentic_parity_report["max_logit_abs_diff"],
    )
    decision_summary = evaluate_e23_decision_summary(
        poolings_summary=poolings_summary,
        pooling_contrast=pooling_contrast,
        primary_cohort_audit=primary_cohort_audit,
        decision_ranges=decision_ranges,
        zero_model_scores_in_pairing=bool(
            e23_plan.get("zero_model_scores_in_pairing") is True
            and rel_body.get("zero_model_scores_in_pairing") is True
        ),
        authentic_anchor_parity_passed=bool(authentic_parity_report["passed"]),
        max_authentic_parity_diff=max_auth_diff,
        en_matched_self_check_passed=bool(
            all(sc["passed"] and sc["max_abs_diff"] == 0.0 for sc in matched_self_checks.values())
        ),
    )
    eval_s = time.monotonic() - t_eval

    _emit_progress(
        progress,
        "e23_bootstrap_evaluated",
        unit=f"{model}:bootstrap_analysis",
        completed=3,
        total=total_units,
        compute_s=eval_s,
        model=model,
        outcome=decision_summary["outcome"],
    )

    # 4. Publish E23 analysis artifact under phase3/results/e23/<model>
    t_pub = time.monotonic()
    stage4_block_refs = {
        pooling: {
            t_lang: plain(s4_blocks_meta[pooling][t_lang]["prediction_ref"])
            for t_lang in LANGUAGES
        }
        for pooling in POOLINGS
    }
    analysis_body = {
        "schema": SCHEMA_ANALYSIS,
        "experiment": E23_EXPERIMENT,
        "plan": plain(gpu_plan.reference),
        "plan_identity": gpu_plan["plan_identity"],
        "release_ref": plain(rel_ref),
        "predictions_ref": plain(predictions_ref),
        "stage4_authentic_block_refs": stage4_block_refs,
        "model": model,
        "midpoint_depth": int(depth),
        "condition": E23_CONDITION,
        "poolings": list(POOLINGS),
        "split": E23_SPLIT,
        "bootstrap": plain(gpu_plan["bootstrap"]),
        "inputs": plain(expected_inputs),
        "zero_model_scores_in_pairing": True,
        "confirm_rows": 0,
        "matched_self_checks": matched_self_checks,
        "authentic_anchor_parity": authentic_parity_report,
        "primary_cohort_audit": primary_cohort_audit,
        "poolings_summary": poolings_summary,
        "caption_mean_minus_last_caption": pooling_contrast,
        "decision_ranges": decision_ranges,
        "decision_summary": decision_summary,
        "supervision_labels": {
            "diagonal": SUPERVISION_DIAGONAL,
            "off_diagonal": SUPERVISION_OFF_DIAGONAL,
        },
        "e35_excluded": True,
        "confirm_excluded": True,
    }

    analysis_upstream = [plain(gpu_plan.reference), plain(rel_ref), plain(predictions_ref)]
    for pooling in POOLINGS:
        for t_lang in LANGUAGES:
            analysis_upstream.append(stage4_block_refs[pooling][t_lang])

    analysis_ref = publish_phase3(
        resolved_root,
        protocol,
        gpu_plan["dataset_fingerprint"],
        f"{E23_RESULTS_NAMESPACE}/{model}",
        KIND_ANALYSIS,
        analysis_body,
        producer,
        upstream=analysis_upstream,
        run_id=effective_run_id,
    )
    verify_resume_ref(
        resolved_root,
        protocol,
        analysis_ref,
        KIND_ANALYSIS,
        expected_schema=SCHEMA_ANALYSIS,
        expected_dataset=gpu_plan["dataset_fingerprint"],
        expected_plan_identity=gpu_plan["plan_identity"],
        expected_owner=model,
        expected_inputs=expected_inputs,
    )
    write_s = time.monotonic() - t_pub

    _emit_progress(
        progress,
        "e23_analysis_published",
        unit=f"{model}:publish_analysis",
        completed=total_units,
        total=total_units,
        write_s=write_s,
        model=model,
        predictions_ref=plain(predictions_ref),
        analysis_ref=plain(analysis_ref),
    )

    return frozen({
        "status": "passed",
        "model": model,
        "run_id": effective_run_id,
        "midpoint_depth": int(depth),
        "plan_ref": plain(gpu_plan.reference),
        "release_ref": plain(rel_ref),
        "predictions_ref": plain(predictions_ref),
        "analysis_ref": plain(analysis_ref),
        "authentic_anchor_parity": plain(authentic_parity_report),
        "matched_self_checks": plain(matched_self_checks),
        "primary_cohort_audit": plain(primary_cohort_audit),
        "decision_summary": plain(decision_summary),
        "resumed_predictions": bool(resumed_predictions),
        "resumed_analysis": False,
        "e35_excluded": True,
        "confirm_excluded": True,
    })


def _parse_ref_arg(value):
    if value is None:
        return None
    text = str(value).strip()
    if text.startswith("{"):
        parsed = json.loads(text)
        check(isinstance(parsed, dict), "JSON reference argument must decode to an object")
        return parsed
    return text


def _emit_cli(event):
    print(json.dumps(plain(event), sort_keys=True), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--phase1-root", default=None)
    parser.add_argument(
        "--plan",
        default=None,
        help="Optional explicit phase3_gpu_plan {'path', 'sha256'} JSON or path (defaults to CERTIFIED_STAGE_B1_GPU_PLAN)",
    )
    parser.add_argument(
        "--release",
        default=None,
        help="Optional explicit E23 release {'path', 'sha256'} JSON or path (defaults to CERTIFIED_STAGE_B2_RELEASES['e23'][model])",
    )
    parser.add_argument("--protocol", type=Path, default=REPO / "configs/protocol.json")
    parser.add_argument(
        "--resume-predictions",
        "--resume-predictions-ref",
        dest="resume_predictions",
        default=None,
        help="Optional explicit E23 predictions {'path', 'sha256'} JSON or path",
    )
    parser.add_argument(
        "--resume-analysis",
        "--resume-analysis-ref",
        dest="resume_analysis",
        default=None,
        help="Optional explicit E23 analysis {'path', 'sha256'} JSON or path",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--storage-scope",
        default="persistent-drive",
        choices=("persistent-drive", "local-development"),
    )
    parser.add_argument("--output-ref", type=Path, default=None)
    args = parser.parse_args(argv)

    root = phase1_root(args.phase1_root)
    check(root.is_dir(), "PHASE1_ROOT must exist and be mounted before launch")
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    producer = current_producer(args.storage_scope)

    _emit_cli({
        "event": "e23_start",
        "model": args.model,
        "code_commit": producer["code_commit"],
        "storage_scope": producer["storage_scope"],
    })
    result = run_e23(
        root,
        protocol,
        args.model,
        producer,
        plan_ref=_parse_ref_arg(args.plan),
        release_ref=_parse_ref_arg(args.release),
        resume_predictions=_parse_ref_arg(args.resume_predictions),
        resume_analysis=_parse_ref_arg(args.resume_analysis),
        run_id=args.run_id,
        progress=_emit_cli,
    )
    summary = {
        "event": "e23_complete",
        "status": "passed",
        "model": args.model,
        "midpoint_depth": result["midpoint_depth"],
        "plan_ref": plain(result["plan_ref"]),
        "release_ref": plain(result["release_ref"]),
        "predictions_ref": plain(result["predictions_ref"]),
        "analysis_ref": plain(result["analysis_ref"]),
        "authentic_anchor_parity": {
            "passed": result["authentic_anchor_parity"]["passed"],
            "max_feature_abs_diff": result["authentic_anchor_parity"]["max_feature_abs_diff"],
            "max_logit_abs_diff": result["authentic_anchor_parity"]["max_logit_abs_diff"],
            "tol": result["authentic_anchor_parity"]["tol"],
        },
        "decision_summary": plain(result["decision_summary"]),
        "e35_excluded": True,
        "confirm_excluded": True,
    }
    if args.output_ref is not None:
        args.output_ref.parent.mkdir(parents=True, exist_ok=True)
        args.output_ref.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _emit_cli(summary)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, OSError, ValueError, KeyError, ImportError, json.JSONDecodeError) as exc:
        _emit_cli({"status": "blocked", "blockers": [str(exc)], "phase3_e23_complete": False})
        raise SystemExit(2)
