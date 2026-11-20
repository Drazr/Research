"""Phase 3 Stage A2 CPU worker for E40: cross-depth + caption_only E30 2x2 head x pooling swap.

Implements phase3.md §A2.1-A2.2 (Worker 1: E40):
1. Loads `phase3_cpu_plan` via `load_phase3_cpu_plan(root, protocol, plan_ref, bind=True)`.
2. Reuses Phase 2 E30's 8 midpoint `image_caption` crossed blocks and all 64 Stage 4 matched
   SCREEN blocks across all 4 depths (`qwen`: [9, 18, 27, 36]; `gemma`: [9, 17, 26, 34])
   and both conditions (`image_caption`, `caption_only`).
3. Verifies identical depth, hook (`decoder:<depth>`), dimension, `model_plan`, and 360-image
   SCREEN row keys across `last_caption` and `caption_mean` at every `(depth, condition)`,
   runs in-memory `en->en` matched self-checks (`atol=0, rtol=0`) for both poolings at all
   8 `(depth, condition)` cells, and verifies midpoint `image_caption` parity (`<= 1e-12`)
   against Phase 2 E30.
4. Scores the remaining 7 `(depth, condition)` cells = 56 new crossed blocks (224 routes per
   model: `head_last_on_mean` and `head_mean_on_last`, including crossed `t->t` diagonals)
   via `src.stage3.engine.transform` and publishes under
   `phase3/predictions/e40_crossed/<model>/<run_id>/`.
5. Evaluates all 8 `(depth, condition)` cells on the frozen `(2000, 360)` bootstrap via
   `src.stage4.metrics`, computing per-`(depth, condition)` 4-arm summaries,
   `feature_gain_fixed_head_last`, `feature_gain_fixed_head_mean`, `interaction`,
   `off_distribution_collapse`, AND the paired `image_caption - caption_only` swap contrast
   inside every `(2000, 360)` bootstrap draw before taking linear 2.5th/97.5th percentiles.
   Publishes analysis under `phase3/results/e40/<model>/<run_id>/`.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from src.common.contracts import LANGUAGES, ContractError, fingerprint
from src.common.storage import new_run_id, phase1_root, within
from src.phase2.interfaces import (
    KIND_ANALYSIS as P2_KIND_ANALYSIS,
    KIND_PREDICTION as P2_KIND_PREDICTION,
    SCHEMA_ANALYSIS as P2_SCHEMA_ANALYSIS,
    SCHEMA_PREDICTION as P2_SCHEMA_PREDICTION,
    load_phase2_artifact,
)
from src.phase3.interfaces import (
    CONDITIONS,
    KIND_ANALYSIS,
    KIND_PREDICTION,
    MODELS,
    POOLINGS,
    REPO,
    SCHEMA_ANALYSIS,
    SCHEMA_PREDICTION,
    check,
    contract,
    current_producer,
    load_phase3_cpu_plan,
    normalize_ref,
    plain,
    publish_phase3,
    require_open_split,
    verify_resume_ref,
)

E40_SPLIT = "SCREEN"
E40_PREDICTIONS_NAMESPACE = "phase3/predictions/e40_crossed"
E40_RESULTS_NAMESPACE = "phase3/results/e40"

MATCHED_ARMS = ("last_on_last", "mean_on_mean")
CROSSED_ARMS = ("head_last_on_mean", "head_mean_on_last")
ALL_ARMS = MATCHED_ARMS + CROSSED_ARMS

CROSSED_ARM_SPECS = {
    "head_last_on_mean": ("last_caption", "caption_mean"),
    "head_mean_on_last": ("caption_mean", "last_caption"),
}

CONTRAST_ORDER = (
    "feature_gain_fixed_head_last",
    "feature_gain_fixed_head_mean",
    "head_gain_fixed_feature_last",
    "head_gain_fixed_feature_mean",
    "interaction",
    "total_matched_reduction",
)

LOSS_METRICS = frozenset({"brier", "nll"})
SUPERVISION_DIAGONAL = "within_context_source_fit"
SUPERVISION_OFF_DIAGONAL = "source_only_transfer"
COLLAPSE_DROP_THRESHOLD = 0.015
MIDPOINT_PARITY_TOL = 1e-12


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


def _off_diagonal_pairs():
    return [(s, t) for s in LANGUAGES for t in LANGUAGES if s != t]


def _new_cells(depths, midpoint):
    return [
        (int(depth), cond)
        for depth in depths
        for cond in CONDITIONS
        if not (int(depth) == int(midpoint) and cond == "image_caption")
    ]


def _valid_crossed_resume_keys(depths, midpoint):
    return {
        f"{depth}:{cond}:{arm}:{lang}"
        for depth, cond in _new_cells(depths, midpoint)
        for arm in CROSSED_ARMS
        for lang in LANGUAGES
    }


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


def _improvement_value(metric, base_val, alt_val):
    """Signed improvement of `alt` over `base` (positive always means better)."""
    if base_val is None or alt_val is None:
        return None
    if metric in LOSS_METRICS:
        return float(base_val - alt_val)
    return float(alt_val - base_val)


def _improvement_draws(metric, base_draws, alt_draws):
    if metric in LOSS_METRICS:
        return base_draws - alt_draws
    return alt_draws - base_draws


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
        return status, f"screen_support_below_minimum_{minimum}"
    if unestimable:
        return status, "unestimable_head"
    return status, "undefined_metric_or_zero_denominator"


def _summarize_macro_and_categories(
    *,
    categories,
    primary_indices,
    cat_values,
    cat_draws,
    cat_supports,
    cat_unestimable,
    minimum,
):
    """Build ordered category table and 14-primary macro summary from aligned category arrays."""
    import numpy as np
    from src.stage4.metrics import fixed_macro

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

    primary_vals = [cat_values[idx] for idx in primary_indices]
    defined_count = sum(v is not None for v in primary_vals)
    ok_count = sum(
        cat_values[idx] is not None and min(cat_supports[idx][1], cat_supports[idx][2]) >= minimum
        for idx in primary_indices
    )
    if defined_count < len(primary_indices):
        macro_val = None
        macro_status = "undefined"
        macro_reason = "unresolved_primary_macro"
    else:
        macro_val = float(fixed_macro([0.0 if v is None else v for v in cat_values], primary_indices))
        if ok_count == len(primary_indices):
            macro_status = "ok"
            macro_reason = None
        else:
            macro_status = "descriptive_low_support"
            macro_reason = f"primary_category_support_below_minimum_{minimum}"

    stacked = np.stack(cat_draws, axis=0)
    macro_draws = fixed_macro(stacked, primary_indices, axis=0)
    primary_macro = {
        "value": macro_val,
        "interval": _interval_record(macro_draws),
        "status": macro_status,
        "reason": macro_reason,
        "support": {
            "categories": len(primary_indices),
            "defined_categories": defined_count,
            "ok_categories": ok_count,
        },
    }
    return primary_macro, macro_draws, per_category


def _mean_optional(values):
    if not values or any(v is None for v in values):
        return None
    return float(sum(values) / len(values))


def _summarize_aggregate(point_val, draw_array, *, categories_count, defined_count, ok_count, minimum):
    if point_val is None:
        status = "undefined"
        reason = "unresolved_aggregate"
    elif ok_count == categories_count:
        status = "ok"
        reason = None
    else:
        status = "descriptive_low_support"
        reason = f"primary_category_support_below_minimum_{minimum}"
    return {
        "value": None if point_val is None else float(point_val),
        "interval": _interval_record(draw_array),
        "status": status,
        "reason": reason,
        "support": {
            "categories": int(categories_count),
            "defined_categories": int(defined_count),
            "ok_categories": int(ok_count),
        },
    }


def _diff_opt(a, b):
    return None if (a is None or b is None) else float(a - b)


def _parse_hook_from_site(site, expected_pooling):
    """Extract residual-stream hook (`decoder:<depth>`) from a Stage 2 site string."""
    parts = str(site).split(":")
    check(
        len(parts) == 3 and parts[0] == "decoder" and parts[2] == expected_pooling,
        f"Unexpected site format: {site!r} (expected decoder:<depth>:{expected_pooling})",
    )
    return f"{parts[0]}:{parts[1]}", int(parts[1])


@contract
def verify_e40_cell_provenance(binding, model, depth, condition, gids_by_pooling, rows_by_pooling, matrices_by_pooling):
    """Verify identical depth, hook, dimension, model_plan, and SCREEN row keys across poolings."""
    fit_groups = binding.stage4_plan.fit["groups"]
    screen_ids = tuple(binding.stage4_plan["screen_ids"])
    expected_hook = f"decoder:{depth}"
    expected_model_plan = plain(binding.model_plan_refs[model])

    shared_dimension = None
    per_language = {}

    for lang in LANGUAGES:
        gid_last = gids_by_pooling["last_caption"][lang]
        gid_mean = gids_by_pooling["caption_mean"][lang]
        g_last = fit_groups[gid_last]
        g_mean = fit_groups[gid_mean]

        check(
            g_last["model"] == model and g_mean["model"] == model,
            f"Model mismatch in E40 groups for {model}/{depth}/{condition}/{lang}",
        )
        check(
            g_last["condition"] == condition and g_mean["condition"] == condition,
            f"Condition mismatch in E40 groups for {model}/{depth}/{condition}/{lang}",
        )
        hook_last, depth_last = _parse_hook_from_site(g_last["site"], "last_caption")
        hook_mean, depth_mean = _parse_hook_from_site(g_mean["site"], "caption_mean")
        check(
            depth_last == depth and depth_mean == depth,
            f"Depth mismatch for {model}/{condition}/{lang}: last={depth_last}, mean={depth_mean}, expected={depth}",
        )
        check(
            hook_last == hook_mean == expected_hook,
            f"Hook mismatch for {model}/{condition}/{lang}: last={hook_last!r}, mean={hook_mean!r}, expected={expected_hook!r}",
        )

        dim_last = int(g_last["dimension"])
        dim_mean = int(g_mean["dimension"])
        check(
            dim_last == dim_mean and dim_last > 0,
            f"Feature dimension mismatch across poolings for {model}/{depth}/{condition}/{lang}: last={dim_last}, mean={dim_mean}",
        )
        if shared_dimension is None:
            shared_dimension = dim_last
        else:
            check(
                dim_last == shared_dimension,
                f"Feature dimension mismatch across languages for {model}/{depth}/{condition}/{lang}: {dim_last} vs {shared_dimension}",
            )

        mp_last = plain(g_last["model_plan"])
        mp_mean = plain(g_mean["model_plan"])
        check(
            mp_last == mp_mean == expected_model_plan,
            f"Stage 2 model_plan mismatch across poolings for {model}/{depth}/{condition}/{lang}",
        )

        rows_last = rows_by_pooling["last_caption"][lang]
        rows_mean = rows_by_pooling["caption_mean"][lang]
        mat_last = matrices_by_pooling["last_caption"][lang]
        mat_mean = matrices_by_pooling["caption_mean"][lang]

        check(
            plain(rows_last) == plain(rows_mean),
            f"SCREEN row keys mismatch between last_caption and caption_mean for {model}/{depth}/{condition}/{lang}",
        )
        check(
            len(rows_last) == len(screen_ids),
            f"Unexpected SCREEN row count for {model}/{depth}/{condition}/{lang}: {len(rows_last)} vs {len(screen_ids)}",
        )
        check(
            all(r["split"] == E40_SPLIT for r in rows_last),
            f"Non-SCREEN split detected in E40 features for {model}/{depth}/{condition}/{lang}",
        )
        row_image_ids = [r["image_id"] for r in rows_last]
        check(
            sorted(row_image_ids) == list(screen_ids) and len(set(row_image_ids)) == len(screen_ids),
            f"SCREEN image_id population mismatch for {model}/{depth}/{condition}/{lang}",
        )
        check(
            mat_last.shape == (len(rows_last), shared_dimension)
            and mat_mean.shape == (len(rows_mean), shared_dimension),
            f"Feature matrix shape mismatch for {model}/{depth}/{condition}/{lang}: last={mat_last.shape}, mean={mat_mean.shape}",
        )

        per_language[lang] = {
            "last_caption_group": gid_last,
            "caption_mean_group": gid_mean,
            "dimension": shared_dimension,
            "n_rows": len(rows_last),
            "image_ids_sha256": fingerprint(row_image_ids),
        }

    return {
        "model": model,
        "depth": int(depth),
        "condition": condition,
        "split": E40_SPLIT,
        "depth_match": True,
        "hook": expected_hook,
        "hook_match": True,
        "dimension": int(shared_dimension),
        "model_plan_ref": expected_model_plan,
        "model_plan_match": True,
        "row_keys_match": True,
        "n_screen_images": len(screen_ids),
        "screen_ids_sha256": fingerprint(list(screen_ids)),
        "per_language": per_language,
    }


def _score_matrix(matrix, coef, intercept, stats, estimable):
    """Score verified SCREEN features in float64 identically to Stage 4."""
    import numpy as np
    from src.stage3.engine import transform

    logits = np.asarray(transform(matrix, stats) @ coef.T, dtype=np.float64) + intercept
    check(logits.dtype == np.float64 and np.isfinite(logits).all(), "Nonfinite E40 logits")
    estimable_bool = np.asarray(estimable, dtype=bool)
    check(logits.shape[1] == estimable_bool.shape[0], "E40 logits/estimable column mismatch")
    check(not np.any(logits[:, ~estimable_bool]), "Nonzero logits on unestimable E40 columns")
    return logits


def _run_en_matched_self_checks(
    *,
    model,
    depth,
    condition,
    rows_by_pooling,
    matrices_by_pooling,
    source_heads_by_pooling,
    matched_preds_by_pooling,
):
    """Run mandatory in-memory en->en matched reproduction checks (atol=0, rtol=0) for both poolings."""
    import numpy as np

    results = {}
    for pooling in POOLINGS:
        rows_en = rows_by_pooling[pooling]["en"]
        mat_en = matrices_by_pooling[pooling]["en"]
        coef_en, intercept_en, estimable_en, _, stats_en = source_heads_by_pooling[pooling]["en"]
        computed = _score_matrix(mat_en, coef_en, intercept_en, stats_en, estimable_en)
        accepted = matched_preds_by_pooling[pooling]["en"]["en"]
        check(
            plain(rows_en) == plain(accepted.rows),
            f"SCREEN en->en row alignment mismatch for {model}/{depth}/{condition}/{pooling}",
        )
        check(
            computed.shape == accepted.logits.shape,
            f"SCREEN en->en shape mismatch for {model}/{depth}/{condition}/{pooling}: {computed.shape} vs {accepted.logits.shape}",
        )
        max_abs_diff = float(np.max(np.abs(computed - accepted.logits)))
        check(
            np.array_equal(computed, accepted.logits) and max_abs_diff == 0.0,
            f"SCREEN en->en matched self-check failed for {model}/{depth}/{condition}/{pooling}: max_abs_diff={max_abs_diff}",
        )
        results[pooling] = {
            "route": "en->en",
            "depth": int(depth),
            "condition": condition,
            "pooling": pooling,
            "passed": True,
            "max_abs_diff": 0.0,
            "atol": 0.0,
            "rtol": 0.0,
            "row_count": int(computed.shape[0]),
            "category_count": int(computed.shape[1]),
        }
    return results


def _load_crossed_block_arrays(root, ref, expected_n_images, expected_n_cats):
    """Load and validate logits, estimable, prevalence from a published crossed block."""
    import numpy as np

    marker_dir = within(root, ref["path"]).parent
    logits = np.load(marker_dir / "logits.npy", allow_pickle=False)
    estimable = np.load(marker_dir / "estimable.npy", allow_pickle=False)
    prevalence = np.load(marker_dir / "prevalence.npy", allow_pickle=False)
    n_src = len(LANGUAGES)
    check(
        logits.shape == (n_src, expected_n_images, expected_n_cats) and logits.dtype == np.float64,
        f"Invalid crossed logits shape/dtype: {logits.shape}/{logits.dtype}",
    )
    check(np.isfinite(logits).all(), "Nonfinite logits in crossed block")
    check(
        estimable.shape == (n_src, expected_n_cats) and estimable.dtype == bool,
        f"Invalid crossed estimable shape/dtype: {estimable.shape}/{estimable.dtype}",
    )
    check(
        prevalence.shape == (n_src, expected_n_cats) and prevalence.dtype == np.float64,
        f"Invalid crossed prevalence shape/dtype: {prevalence.shape}/{prevalence.dtype}",
    )
    check(not np.any(logits * (~estimable[:, None, :])), "Nonzero unestimable logits in crossed block")
    return logits, estimable, prevalence


def _evaluate_route(pred, aligned, y, known, counts, n_cats, total_draws):
    """Evaluate one route's 25-category point estimates, bootstrap draws, and support."""
    import numpy as np
    from src.stage4.metrics import METRICS, binary_metrics, bootstrap_binary_metrics, probabilities

    probs = probabilities(pred.logits[aligned])
    estimable = np.asarray(pred.estimable, dtype=bool)
    check(estimable.shape == (n_cats,), f"Unexpected estimable shape: {estimable.shape}")
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
        if estimable[col]:
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
        "unestimable": [not bool(estimable[col]) for col in range(n_cats)],
    }


def evaluate_off_distribution_collapse(
    *,
    last_transfer_val,
    last_transfer_draws,
    last_diag_val,
    last_diag_draws,
    crossed_transfer_vals,
    crossed_transfer_draws,
    crossed_diag_vals,
    crossed_diag_draws,
    drop_threshold=COLLAPSE_DROP_THRESHOLD,
):
    """Evaluate off-distribution collapse flags against matched `last_on_last`."""
    per_arm = {}
    for arm in CROSSED_ARMS:
        t_val = crossed_transfer_vals[arm]
        d_val = crossed_diag_vals[arm]
        t_delta = None if (t_val is None or last_transfer_val is None) else float(t_val - last_transfer_val)
        d_delta = None if (d_val is None or last_diag_val is None) else float(d_val - last_diag_val)
        t_delta_draws = crossed_transfer_draws[arm] - last_transfer_draws
        d_delta_draws = crossed_diag_draws[arm] - last_diag_draws
        t_drop = None if t_delta is None else float(-t_delta)
        d_drop = None if d_delta is None else float(-d_delta)
        per_arm[arm] = {
            "transfer_12route_auroc": {
                "crossed_value": None if t_val is None else float(t_val),
                "matched_last_on_last_value": None if last_transfer_val is None else float(last_transfer_val),
                "delta": t_delta,
                "drop": t_drop,
                "interval": _interval_record(t_delta_draws),
                "dropped_ge_threshold": bool(t_drop is not None and t_drop >= drop_threshold),
            },
            "diagonal_4lang_auroc": {
                "crossed_value": None if d_val is None else float(d_val),
                "matched_last_on_last_value": None if last_diag_val is None else float(last_diag_val),
                "delta": d_delta,
                "drop": d_drop,
                "interval": _interval_record(d_delta_draws),
                "dropped_ge_threshold": bool(d_drop is not None and d_drop >= drop_threshold),
            },
        }

    transfer_flagged = all(
        per_arm[arm]["transfer_12route_auroc"]["dropped_ge_threshold"] for arm in CROSSED_ARMS
    )
    diagonal_flagged = all(
        per_arm[arm]["diagonal_4lang_auroc"]["dropped_ge_threshold"] for arm in CROSSED_ARMS
    )
    return {
        "drop_threshold": float(drop_threshold),
        "transfer_collapse_flagged": bool(transfer_flagged),
        "diagonal_collapse_flagged": bool(diagonal_flagged),
        "flagged": bool(transfer_flagged or diagonal_flagged),
        "head_last_on_mean_vs_last_on_last": per_arm["head_last_on_mean"],
        "head_mean_on_last_vs_last_on_last": per_arm["head_mean_on_last"],
    }


def _evaluate_cell_arms_and_contrasts(
    *,
    preds_by_arm,
    aligned_by_key,
    y,
    known,
    counts,
    categories,
    primary_indices,
    total_draws,
    minimum,
):
    """Compute per-arm tables, within-draw 2x2 paired contrasts, collapse flags, and raw contrast cache."""
    import numpy as np
    from src.stage4.metrics import METRICS

    n_cats = len(categories)
    off_pairs = _off_diagonal_pairs()

    raw_routes = {}
    for arm in ALL_ARMS:
        for s in LANGUAGES:
            for t in LANGUAGES:
                raw_routes[(arm, s, t)] = _evaluate_route(
                    preds_by_arm[arm][t][s],
                    aligned_by_key[(arm, s, t)],
                    y,
                    known,
                    counts,
                    n_cats,
                    total_draws,
                )

    arms_out = {}
    arm_gap_mean_vals = {arm: {} for arm in ALL_ARMS}
    arm_gap_mean_draws = {arm: {} for arm in ALL_ARMS}
    arm_gap_cat_vals = {arm: {} for arm in ALL_ARMS}
    arm_gap_cat_draws = {arm: {} for arm in ALL_ARMS}
    arm_route_gap_vals = {arm: {} for arm in ALL_ARMS}
    arm_route_gap_draws = {arm: {} for arm in ALL_ARMS}
    arm_transfer_mean_vals = {arm: {} for arm in ALL_ARMS}
    arm_transfer_mean_draws = {arm: {} for arm in ALL_ARMS}
    arm_diag_mean_vals = {arm: {} for arm in ALL_ARMS}
    arm_diag_mean_draws = {arm: {} for arm in ALL_ARMS}
    arm_route_supports = {}

    for arm in ALL_ARMS:
        is_crossed = arm in CROSSED_ARMS
        head_pool, feat_pool = (
            ("last_caption", "last_caption")
            if arm == "last_on_last"
            else ("caption_mean", "caption_mean")
            if arm == "mean_on_mean"
            else CROSSED_ARM_SPECS[arm]
        )
        per_route_all = {}
        twelve_route_gap = {}
        twelve_route_transfer = {}
        four_diagonal = {}

        for metric in METRICS:
            route_abs_macro_vals = {}
            route_abs_macro_draws = {}
            route_abs_cat_vals = {}
            route_abs_cat_draws = {}
            route_supports_ref = None
            for s in LANGUAGES:
                for t in LANGUAGES:
                    rkey = _route_key(s, t)
                    rd = raw_routes[(arm, s, t)]
                    if route_supports_ref is None:
                        route_supports_ref = rd["supports"]
                    macro_s, macro_d, cat_t = _summarize_macro_and_categories(
                        categories=categories,
                        primary_indices=primary_indices,
                        cat_values=rd["points"][metric],
                        cat_draws=rd["draws"][metric],
                        cat_supports=rd["supports"],
                        cat_unestimable=rd["unestimable"],
                        minimum=minimum,
                    )
                    route_abs_macro_vals[(s, t)] = macro_s["value"]
                    route_abs_macro_draws[(s, t)] = macro_d
                    route_abs_cat_vals[(s, t)] = rd["points"][metric]
                    route_abs_cat_draws[(s, t)] = rd["draws"][metric]

                    entry = per_route_all.setdefault(
                        rkey,
                        {
                            "route": rkey,
                            "arm": arm,
                            "is_crossed": is_crossed,
                            "source_language": s,
                            "target_language": t,
                            "supervision_label": SUPERVISION_DIAGONAL if s == t else SUPERVISION_OFF_DIAGONAL,
                            "metrics": {},
                        },
                    )
                    entry["metrics"][metric] = {
                        "primary_macro": macro_s,
                        "categories": cat_t,
                    }

            arm_route_supports[arm] = route_supports_ref
            ok_primary = sum(
                min(route_supports_ref[idx][1], route_supports_ref[idx][2]) >= minimum
                for idx in primary_indices
            )

            diag_macro_vals = [route_abs_macro_vals[(t, t)] for t in LANGUAGES]
            diag_macro_draws = np.mean(
                np.stack([route_abs_macro_draws[(t, t)] for t in LANGUAGES], axis=0), axis=0
            )
            diag_mean_val = _mean_optional(diag_macro_vals)
            diag_cat_vals = [
                _mean_optional([route_abs_cat_vals[(t, t)][c] for t in LANGUAGES])
                for c in range(n_cats)
            ]
            diag_cat_draws = [
                np.mean(np.stack([route_abs_cat_draws[(t, t)][c] for t in LANGUAGES], axis=0), axis=0)
                for c in range(n_cats)
            ]
            diag_unest = [
                any(raw_routes[(arm, t, t)]["unestimable"][c] for t in LANGUAGES)
                for c in range(n_cats)
            ]
            _, _, diag_cats_table = _summarize_macro_and_categories(
                categories=categories,
                primary_indices=primary_indices,
                cat_values=diag_cat_vals,
                cat_draws=diag_cat_draws,
                cat_supports=route_supports_ref,
                cat_unestimable=diag_unest,
                minimum=minimum,
            )
            four_diagonal[metric] = {
                "primary_macro": _summarize_aggregate(
                    diag_mean_val,
                    diag_macro_draws,
                    categories_count=len(primary_indices),
                    defined_count=len(primary_indices) if diag_mean_val is not None else 0,
                    ok_count=ok_primary if diag_mean_val is not None else 0,
                    minimum=minimum,
                ),
                "categories": diag_cats_table,
            }
            arm_diag_mean_vals[arm][metric] = diag_mean_val
            arm_diag_mean_draws[arm][metric] = diag_macro_draws

            xfer_macro_vals = [route_abs_macro_vals[(s, t)] for s, t in off_pairs]
            xfer_macro_draws = np.mean(
                np.stack([route_abs_macro_draws[(s, t)] for s, t in off_pairs], axis=0), axis=0
            )
            xfer_mean_val = _mean_optional(xfer_macro_vals)
            xfer_cat_vals = [
                _mean_optional([route_abs_cat_vals[(s, t)][c] for s, t in off_pairs])
                for c in range(n_cats)
            ]
            xfer_cat_draws = [
                np.mean(np.stack([route_abs_cat_draws[(s, t)][c] for s, t in off_pairs], axis=0), axis=0)
                for c in range(n_cats)
            ]
            xfer_unest = [
                any(raw_routes[(arm, s, t)]["unestimable"][c] for s, t in off_pairs)
                for c in range(n_cats)
            ]
            _, _, xfer_cats_table = _summarize_macro_and_categories(
                categories=categories,
                primary_indices=primary_indices,
                cat_values=xfer_cat_vals,
                cat_draws=xfer_cat_draws,
                cat_supports=route_supports_ref,
                cat_unestimable=xfer_unest,
                minimum=minimum,
            )
            twelve_route_transfer[metric] = {
                "primary_macro": _summarize_aggregate(
                    xfer_mean_val,
                    xfer_macro_draws,
                    categories_count=len(primary_indices),
                    defined_count=len(primary_indices) if xfer_mean_val is not None else 0,
                    ok_count=ok_primary if xfer_mean_val is not None else 0,
                    minimum=minimum,
                ),
                "categories": xfer_cats_table,
            }
            arm_transfer_mean_vals[arm][metric] = xfer_mean_val
            arm_transfer_mean_draws[arm][metric] = xfer_macro_draws

            gap_macro_vals = []
            gap_macro_draws_list = []
            gap_cat_vals_by_c = [[] for _ in range(n_cats)]
            gap_cat_draws_by_c = [[] for _ in range(n_cats)]
            arm_route_gap_vals[arm][metric] = {}
            arm_route_gap_draws[arm][metric] = {}

            for s, t in off_pairs:
                rkey = _route_key(s, t)
                rd_route = raw_routes[(arm, s, t)]
                rd_diag = raw_routes[(arm, t, t)]
                g_vals = [
                    _gap_value(metric, rd_diag["points"][metric][c], rd_route["points"][metric][c])
                    for c in range(n_cats)
                ]
                g_draws = [
                    _gap_draws(metric, rd_diag["draws"][metric][c], rd_route["draws"][metric][c])
                    for c in range(n_cats)
                ]
                g_unest = [rd_diag["unestimable"][c] or rd_route["unestimable"][c] for c in range(n_cats)]
                g_macro, g_mdraws, g_cats = _summarize_macro_and_categories(
                    categories=categories,
                    primary_indices=primary_indices,
                    cat_values=g_vals,
                    cat_draws=g_draws,
                    cat_supports=rd_route["supports"],
                    cat_unestimable=g_unest,
                    minimum=minimum,
                )
                per_route_all[rkey]["metrics"][metric]["gap"] = {
                    "primary_macro": g_macro,
                    "categories": g_cats,
                }
                gap_macro_vals.append(g_macro["value"])
                gap_macro_draws_list.append(g_mdraws)
                arm_route_gap_vals[arm][metric][rkey] = g_macro["value"]
                arm_route_gap_draws[arm][metric][rkey] = g_mdraws
                for c in range(n_cats):
                    gap_cat_vals_by_c[c].append(g_vals[c])
                    gap_cat_draws_by_c[c].append(g_draws[c])

            gap_mean_val = _mean_optional(gap_macro_vals)
            gap_mean_draws = np.mean(np.stack(gap_macro_draws_list, axis=0), axis=0)
            gap_mean_cat_vals = [_mean_optional(gap_cat_vals_by_c[c]) for c in range(n_cats)]
            gap_mean_cat_draws = [
                np.mean(np.stack(gap_cat_draws_by_c[c], axis=0), axis=0) for c in range(n_cats)
            ]
            gap_unest = [
                any(
                    raw_routes[(arm, s, t)]["unestimable"][c] or raw_routes[(arm, t, t)]["unestimable"][c]
                    for s, t in off_pairs
                )
                for c in range(n_cats)
            ]
            _, _, gap_cats_table = _summarize_macro_and_categories(
                categories=categories,
                primary_indices=primary_indices,
                cat_values=gap_mean_cat_vals,
                cat_draws=gap_mean_cat_draws,
                cat_supports=route_supports_ref,
                cat_unestimable=gap_unest,
                minimum=minimum,
            )
            twelve_route_gap[metric] = {
                "primary_macro": _summarize_aggregate(
                    gap_mean_val,
                    gap_mean_draws,
                    categories_count=len(primary_indices),
                    defined_count=len(primary_indices) if gap_mean_val is not None else 0,
                    ok_count=ok_primary if gap_mean_val is not None else 0,
                    minimum=minimum,
                ),
                "categories": gap_cats_table,
            }
            arm_gap_mean_vals[arm][metric] = gap_mean_val
            arm_gap_mean_draws[arm][metric] = gap_mean_draws
            arm_gap_cat_vals[arm][metric] = gap_mean_cat_vals
            arm_gap_cat_draws[arm][metric] = gap_mean_cat_draws

        arms_out[arm] = {
            "arm": arm,
            "is_crossed": is_crossed,
            "head_pooling": head_pool,
            "feature_pooling": feat_pool,
            "twelve_route_mean_gap": twelve_route_gap,
            "twelve_route_mean_transfer": twelve_route_transfer,
            "four_diagonal_mean": four_diagonal,
            "per_route": per_route_all,
        }

    supports_ref = arm_route_supports["last_on_last"]
    ok_primary_ref = sum(
        min(supports_ref[idx][1], supports_ref[idx][2]) >= minimum for idx in primary_indices
    )
    contrast_cache = {}

    def _build_pair_contrast(cname, arm_high_gap, arm_low_gap, description):
        out = {"description": description, "metrics": {}}
        contrast_cache[cname] = {}
        for metric in METRICS:
            g_val = _diff_opt(arm_gap_mean_vals[arm_high_gap][metric], arm_gap_mean_vals[arm_low_gap][metric])
            g_draws = arm_gap_mean_draws[arm_high_gap][metric] - arm_gap_mean_draws[arm_low_gap][metric]

            t_val = _improvement_value(
                metric, arm_transfer_mean_vals[arm_high_gap][metric], arm_transfer_mean_vals[arm_low_gap][metric]
            )
            t_draws = _improvement_draws(
                metric, arm_transfer_mean_draws[arm_high_gap][metric], arm_transfer_mean_draws[arm_low_gap][metric]
            )

            d_val = _improvement_value(
                metric, arm_diag_mean_vals[arm_high_gap][metric], arm_diag_mean_vals[arm_low_gap][metric]
            )
            d_draws = _improvement_draws(
                metric, arm_diag_mean_draws[arm_high_gap][metric], arm_diag_mean_draws[arm_low_gap][metric]
            )

            cat_vals = [
                _diff_opt(arm_gap_cat_vals[arm_high_gap][metric][c], arm_gap_cat_vals[arm_low_gap][metric][c])
                for c in range(n_cats)
            ]
            cat_draws = [
                arm_gap_cat_draws[arm_high_gap][metric][c] - arm_gap_cat_draws[arm_low_gap][metric][c]
                for c in range(n_cats)
            ]
            cat_unest = [v is None for v in cat_vals]
            _, _, cats_table = _summarize_macro_and_categories(
                categories=categories,
                primary_indices=primary_indices,
                cat_values=cat_vals,
                cat_draws=cat_draws,
                cat_supports=supports_ref,
                cat_unestimable=cat_unest,
                minimum=minimum,
            )

            per_route = {}
            route_vals_map = {}
            route_draws_map = {}
            for s, t in off_pairs:
                rkey = _route_key(s, t)
                rv = _diff_opt(
                    arm_route_gap_vals[arm_high_gap][metric][rkey],
                    arm_route_gap_vals[arm_low_gap][metric][rkey],
                )
                rd = arm_route_gap_draws[arm_high_gap][metric][rkey] - arm_route_gap_draws[arm_low_gap][metric][rkey]
                route_vals_map[rkey] = rv
                route_draws_map[rkey] = rd
                per_route[rkey] = _summarize_aggregate(
                    rv,
                    rd,
                    categories_count=len(primary_indices),
                    defined_count=len(primary_indices) if rv is not None else 0,
                    ok_count=ok_primary_ref if rv is not None else 0,
                    minimum=minimum,
                )

            contrast_cache[cname][metric] = {
                "g_val": g_val,
                "g_draws": g_draws,
                "t_val": t_val,
                "t_draws": t_draws,
                "d_val": d_val,
                "d_draws": d_draws,
                "cat_vals": cat_vals,
                "cat_draws": cat_draws,
                "route_vals": route_vals_map,
                "route_draws": route_draws_map,
            }

            out["metrics"][metric] = {
                "gap_reduction": {
                    "primary_macro": _summarize_aggregate(
                        g_val,
                        g_draws,
                        categories_count=len(primary_indices),
                        defined_count=len(primary_indices) if g_val is not None else 0,
                        ok_count=ok_primary_ref if g_val is not None else 0,
                        minimum=minimum,
                    ),
                    "categories": cats_table,
                    "per_route": per_route,
                },
                "transfer_shift": _summarize_aggregate(
                    t_val,
                    t_draws,
                    categories_count=len(primary_indices),
                    defined_count=len(primary_indices) if t_val is not None else 0,
                    ok_count=ok_primary_ref if t_val is not None else 0,
                    minimum=minimum,
                ),
                "diagonal_shift": _summarize_aggregate(
                    d_val,
                    d_draws,
                    categories_count=len(primary_indices),
                    defined_count=len(primary_indices) if d_val is not None else 0,
                    ok_count=ok_primary_ref if d_val is not None else 0,
                    minimum=minimum,
                ),
            }
        return out

    def _build_interaction_contrast():
        out = {
            "description": "I = DeltaG_feat|last - DeltaG_feat|mean = (G(last_on_last) - G(head_last_on_mean)) - (G(head_mean_on_last) - G(mean_on_mean))",
            "metrics": {},
        }
        contrast_cache["interaction"] = {}
        for metric in METRICS:
            fg_last_val = _diff_opt(
                arm_gap_mean_vals["last_on_last"][metric],
                arm_gap_mean_vals["head_last_on_mean"][metric],
            )
            fg_mean_val = _diff_opt(
                arm_gap_mean_vals["head_mean_on_last"][metric],
                arm_gap_mean_vals["mean_on_mean"][metric],
            )
            inter_val = _diff_opt(fg_last_val, fg_mean_val)
            inter_draws = (
                (arm_gap_mean_draws["last_on_last"][metric] - arm_gap_mean_draws["head_last_on_mean"][metric])
                - (arm_gap_mean_draws["head_mean_on_last"][metric] - arm_gap_mean_draws["mean_on_mean"][metric])
            )

            t_last_val = _improvement_value(
                metric, arm_transfer_mean_vals["last_on_last"][metric], arm_transfer_mean_vals["head_last_on_mean"][metric]
            )
            t_mean_val = _improvement_value(
                metric, arm_transfer_mean_vals["head_mean_on_last"][metric], arm_transfer_mean_vals["mean_on_mean"][metric]
            )
            t_inter_val = _diff_opt(t_last_val, t_mean_val)
            t_inter_draws = (
                _improvement_draws(
                    metric, arm_transfer_mean_draws["last_on_last"][metric], arm_transfer_mean_draws["head_last_on_mean"][metric]
                )
                - _improvement_draws(
                    metric, arm_transfer_mean_draws["head_mean_on_last"][metric], arm_transfer_mean_draws["mean_on_mean"][metric]
                )
            )

            d_last_val = _improvement_value(
                metric, arm_diag_mean_vals["last_on_last"][metric], arm_diag_mean_vals["head_last_on_mean"][metric]
            )
            d_mean_val = _improvement_value(
                metric, arm_diag_mean_vals["head_mean_on_last"][metric], arm_diag_mean_vals["mean_on_mean"][metric]
            )
            d_inter_val = _diff_opt(d_last_val, d_mean_val)
            d_inter_draws = (
                _improvement_draws(
                    metric, arm_diag_mean_draws["last_on_last"][metric], arm_diag_mean_draws["head_last_on_mean"][metric]
                )
                - _improvement_draws(
                    metric, arm_diag_mean_draws["head_mean_on_last"][metric], arm_diag_mean_draws["mean_on_mean"][metric]
                )
            )

            cat_vals = [
                _diff_opt(
                    _diff_opt(arm_gap_cat_vals["last_on_last"][metric][c], arm_gap_cat_vals["head_last_on_mean"][metric][c]),
                    _diff_opt(arm_gap_cat_vals["head_mean_on_last"][metric][c], arm_gap_cat_vals["mean_on_mean"][metric][c]),
                )
                for c in range(n_cats)
            ]
            cat_draws = [
                (arm_gap_cat_draws["last_on_last"][metric][c] - arm_gap_cat_draws["head_last_on_mean"][metric][c])
                - (arm_gap_cat_draws["head_mean_on_last"][metric][c] - arm_gap_cat_draws["mean_on_mean"][metric][c])
                for c in range(n_cats)
            ]
            cat_unest = [v is None for v in cat_vals]
            _, _, cats_table = _summarize_macro_and_categories(
                categories=categories,
                primary_indices=primary_indices,
                cat_values=cat_vals,
                cat_draws=cat_draws,
                cat_supports=supports_ref,
                cat_unestimable=cat_unest,
                minimum=minimum,
            )

            per_route = {}
            route_vals_map = {}
            route_draws_map = {}
            for s, t in off_pairs:
                rkey = _route_key(s, t)
                rv = _diff_opt(
                    _diff_opt(arm_route_gap_vals["last_on_last"][metric][rkey], arm_route_gap_vals["head_last_on_mean"][metric][rkey]),
                    _diff_opt(arm_route_gap_vals["head_mean_on_last"][metric][rkey], arm_route_gap_vals["mean_on_mean"][metric][rkey]),
                )
                rd = (
                    (arm_route_gap_draws["last_on_last"][metric][rkey] - arm_route_gap_draws["head_last_on_mean"][metric][rkey])
                    - (arm_route_gap_draws["head_mean_on_last"][metric][rkey] - arm_route_gap_draws["mean_on_mean"][metric][rkey])
                )
                route_vals_map[rkey] = rv
                route_draws_map[rkey] = rd
                per_route[rkey] = _summarize_aggregate(
                    rv,
                    rd,
                    categories_count=len(primary_indices),
                    defined_count=len(primary_indices) if rv is not None else 0,
                    ok_count=ok_primary_ref if rv is not None else 0,
                    minimum=minimum,
                )

            contrast_cache["interaction"][metric] = {
                "g_val": inter_val,
                "g_draws": inter_draws,
                "t_val": t_inter_val,
                "t_draws": t_inter_draws,
                "d_val": d_inter_val,
                "d_draws": d_inter_draws,
                "cat_vals": cat_vals,
                "cat_draws": cat_draws,
                "route_vals": route_vals_map,
                "route_draws": route_draws_map,
            }

            out["metrics"][metric] = {
                "gap_reduction": {
                    "primary_macro": _summarize_aggregate(
                        inter_val,
                        inter_draws,
                        categories_count=len(primary_indices),
                        defined_count=len(primary_indices) if inter_val is not None else 0,
                        ok_count=ok_primary_ref if inter_val is not None else 0,
                        minimum=minimum,
                    ),
                    "categories": cats_table,
                    "per_route": per_route,
                },
                "transfer_shift": _summarize_aggregate(
                    t_inter_val,
                    t_inter_draws,
                    categories_count=len(primary_indices),
                    defined_count=len(primary_indices) if t_inter_val is not None else 0,
                    ok_count=ok_primary_ref if t_inter_val is not None else 0,
                    minimum=minimum,
                ),
                "diagonal_shift": _summarize_aggregate(
                    d_inter_val,
                    d_inter_draws,
                    categories_count=len(primary_indices),
                    defined_count=len(primary_indices) if d_inter_val is not None else 0,
                    ok_count=ok_primary_ref if d_inter_val is not None else 0,
                    minimum=minimum,
                ),
            }
        return out

    contrasts = {
        "feature_gain_fixed_head_last": _build_pair_contrast(
            "feature_gain_fixed_head_last",
            "last_on_last",
            "head_last_on_mean",
            "DeltaG_feat|last = G_mean(last_on_last) - G_mean(head_last_on_mean)",
        ),
        "feature_gain_fixed_head_mean": _build_pair_contrast(
            "feature_gain_fixed_head_mean",
            "head_mean_on_last",
            "mean_on_mean",
            "DeltaG_feat|mean = G_mean(head_mean_on_last) - G_mean(mean_on_mean)",
        ),
        "head_gain_fixed_feature_last": _build_pair_contrast(
            "head_gain_fixed_feature_last",
            "last_on_last",
            "head_mean_on_last",
            "DeltaG_head|last = G_mean(last_on_last) - G_mean(head_mean_on_last)",
        ),
        "head_gain_fixed_feature_mean": _build_pair_contrast(
            "head_gain_fixed_feature_mean",
            "head_last_on_mean",
            "mean_on_mean",
            "DeltaG_head|mean = G_mean(head_last_on_mean) - G_mean(mean_on_mean)",
        ),
        "interaction": _build_interaction_contrast(),
        "total_matched_reduction": _build_pair_contrast(
            "total_matched_reduction",
            "last_on_last",
            "mean_on_mean",
            "DeltaG_total = G_mean(last_on_last) - G_mean(mean_on_mean)",
        ),
    }

    collapse = evaluate_off_distribution_collapse(
        last_transfer_val=arm_transfer_mean_vals["last_on_last"]["auroc"],
        last_transfer_draws=arm_transfer_mean_draws["last_on_last"]["auroc"],
        last_diag_val=arm_diag_mean_vals["last_on_last"]["auroc"],
        last_diag_draws=arm_diag_mean_draws["last_on_last"]["auroc"],
        crossed_transfer_vals={arm: arm_transfer_mean_vals[arm]["auroc"] for arm in CROSSED_ARMS},
        crossed_transfer_draws={arm: arm_transfer_mean_draws[arm]["auroc"] for arm in CROSSED_ARMS},
        crossed_diag_vals={arm: arm_diag_mean_vals[arm]["auroc"] for arm in CROSSED_ARMS},
        crossed_diag_draws={arm: arm_diag_mean_draws[arm]["auroc"] for arm in CROSSED_ARMS},
        drop_threshold=COLLAPSE_DROP_THRESHOLD,
    )

    return arms_out, contrasts, collapse, contrast_cache, supports_ref


def compute_condition_swap_contrast(
    *,
    ic_cache,
    co_cache,
    supports_ref,
    categories,
    primary_indices,
    minimum,
):
    """Compute paired `image_caption - caption_only` swap contrast inside every bootstrap draw."""
    from src.stage4.metrics import METRICS

    n_cats = len(categories)
    off_pairs = _off_diagonal_pairs()
    ok_primary_ref = sum(
        min(supports_ref[idx][1], supports_ref[idx][2]) >= minimum for idx in primary_indices
    )
    out = {}
    for cname in CONTRAST_ORDER:
        metrics_out = {}
        for metric in METRICS:
            ic_m = ic_cache[cname][metric]
            co_m = co_cache[cname][metric]

            g_val = _diff_opt(ic_m["g_val"], co_m["g_val"])
            g_draws = ic_m["g_draws"] - co_m["g_draws"]
            t_val = _diff_opt(ic_m["t_val"], co_m["t_val"])
            t_draws = ic_m["t_draws"] - co_m["t_draws"]
            d_val = _diff_opt(ic_m["d_val"], co_m["d_val"])
            d_draws = ic_m["d_draws"] - co_m["d_draws"]

            cat_vals = [_diff_opt(ic_m["cat_vals"][c], co_m["cat_vals"][c]) for c in range(n_cats)]
            cat_draws = [ic_m["cat_draws"][c] - co_m["cat_draws"][c] for c in range(n_cats)]
            cat_unest = [v is None for v in cat_vals]
            _, _, cats_table = _summarize_macro_and_categories(
                categories=categories,
                primary_indices=primary_indices,
                cat_values=cat_vals,
                cat_draws=cat_draws,
                cat_supports=supports_ref,
                cat_unestimable=cat_unest,
                minimum=minimum,
            )

            per_route = {}
            for s, t in off_pairs:
                rkey = _route_key(s, t)
                rv = _diff_opt(ic_m["route_vals"][rkey], co_m["route_vals"][rkey])
                rd = ic_m["route_draws"][rkey] - co_m["route_draws"][rkey]
                per_route[rkey] = _summarize_aggregate(
                    rv,
                    rd,
                    categories_count=len(primary_indices),
                    defined_count=len(primary_indices) if rv is not None else 0,
                    ok_count=ok_primary_ref if rv is not None else 0,
                    minimum=minimum,
                )

            metrics_out[metric] = {
                "gap_reduction": {
                    "primary_macro": _summarize_aggregate(
                        g_val,
                        g_draws,
                        categories_count=len(primary_indices),
                        defined_count=len(primary_indices) if g_val is not None else 0,
                        ok_count=ok_primary_ref if g_val is not None else 0,
                        minimum=minimum,
                    ),
                    "categories": cats_table,
                    "per_route": per_route,
                },
                "transfer_shift": _summarize_aggregate(
                    t_val,
                    t_draws,
                    categories_count=len(primary_indices),
                    defined_count=len(primary_indices) if t_val is not None else 0,
                    ok_count=ok_primary_ref if t_val is not None else 0,
                    minimum=minimum,
                ),
                "diagonal_shift": _summarize_aggregate(
                    d_val,
                    d_draws,
                    categories_count=len(primary_indices),
                    defined_count=len(primary_indices) if d_val is not None else 0,
                    ok_count=ok_primary_ref if d_val is not None else 0,
                    minimum=minimum,
                ),
            }
        out[cname] = {"metrics": metrics_out}
    return out


def _verify_midpoint_e30_analysis_parity(e40_cell, e30_body, tol=MIDPOINT_PARITY_TOL):
    """Verify midpoint image_caption 4-arm & contrast AUROC/AP/Brier/NLL match Phase 2 E30."""
    from src.stage4.metrics import METRICS

    max_diff = 0.0
    for arm in ALL_ARMS:
        for summary_key in ("twelve_route_mean_gap", "twelve_route_mean_transfer", "four_diagonal_mean"):
            for metric in METRICS:
                v_e40 = e40_cell["arms"][arm][summary_key][metric]["primary_macro"]["value"]
                v_e30 = e30_body["arms"][arm][summary_key][metric]["primary_macro"]["value"]
                if v_e40 is None or v_e30 is None:
                    check(
                        v_e40 is None and v_e30 is None,
                        f"Midpoint E30 parity None mismatch for {arm}/{summary_key}/{metric}",
                    )
                    continue
                diff = abs(float(v_e40) - float(v_e30))
                max_diff = max(max_diff, diff)
                check(
                    diff <= tol,
                    f"Midpoint E30 parity failed for {arm}/{summary_key}/{metric}: diff={diff} > {tol}",
                )

    for cname in CONTRAST_ORDER:
        for metric in METRICS:
            v_e40 = e40_cell["contrasts"][cname]["metrics"][metric]["gap_reduction"]["primary_macro"]["value"]
            v_e30 = e30_body["contrasts"][cname]["metrics"][metric]["gap_reduction"]["primary_macro"]["value"]
            if v_e40 is None or v_e30 is None:
                check(
                    v_e40 is None and v_e30 is None,
                    f"Midpoint E30 contrast parity None mismatch for {cname}/{metric}",
                )
                continue
            diff = abs(float(v_e40) - float(v_e30))
            max_diff = max(max_diff, diff)
            check(
                diff <= tol,
                f"Midpoint E30 contrast parity failed for {cname}/{metric}: diff={diff} > {tol}",
            )
    return float(max_diff)


def _verify_reused_e30_crossed_block(
    *,
    root,
    protocol,
    ref,
    plan,
    model,
    midpoint,
    arm,
    head_pooling,
    feature_pooling,
    target_lang,
    expected_feature_ref,
    expected_image_ids_sha256,
    expected_n_images,
    expected_n_cats,
):
    """Authenticate one reused Phase 2 E30 midpoint crossed prediction block."""
    res_ref, res_body, _ = load_phase2_artifact(
        root,
        protocol,
        ref,
        P2_KIND_PREDICTION,
        expected_schema=P2_SCHEMA_PREDICTION,
        expected_dataset=plan["dataset_fingerprint"],
        expected_owner=model,
    )
    check(res_body.get("experiment") == "E30", "Reused E30 crossed block experiment mismatch")
    check(res_body.get("confirm_excluded") is True, "Reused E30 block missing confirm_excluded=True")
    check(int(res_body.get("midpoint_depth", -1)) == midpoint, "Reused E30 block midpoint_depth mismatch")
    check(res_body.get("condition") == "image_caption", "Reused E30 block condition mismatch")
    check(res_body.get("split") == E40_SPLIT, "Reused E30 block split mismatch")
    check(res_body.get("arm") == arm, "Reused E30 block arm mismatch")
    check(res_body.get("head_pooling") == head_pooling, "Reused E30 block head_pooling mismatch")
    check(res_body.get("feature_pooling") == feature_pooling, "Reused E30 block feature_pooling mismatch")
    check(res_body.get("target_language") == target_lang, "Reused E30 block target_language mismatch")
    check(list(res_body.get("source_languages", ())) == list(LANGUAGES), "Reused E30 block sources mismatch")
    check(
        plain(res_body.get("upstream_feature_ref")) == plain(expected_feature_ref),
        "Reused E30 block upstream_feature_ref mismatch",
    )
    check(
        res_body.get("image_ids_sha256") == expected_image_ids_sha256,
        "Reused E30 block image_ids_sha256 mismatch",
    )
    arrays = _load_crossed_block_arrays(root, res_ref, expected_n_images, expected_n_cats)
    return res_ref, res_body, arrays


def _verify_e40_crossed_block_ref(
    *,
    root,
    protocol,
    ref,
    plan,
    model,
    depth,
    condition,
    arm,
    head_pooling,
    feature_pooling,
    target_lang,
    expected_feature_ref,
    expected_image_ids_sha256,
    expected_n_images,
    expected_n_cats,
):
    """Authenticate one resumed E40 crossed prediction block and return `(ref, body, arrays)`."""
    res_ref, res_body, _ = verify_resume_ref(
        root,
        protocol,
        ref,
        KIND_PREDICTION,
        expected_schema=SCHEMA_PREDICTION,
        expected_dataset=plan["dataset_fingerprint"],
        expected_plan_identity=plan["plan_identity"],
        expected_owner=model,
    )
    check(res_body.get("experiment") == "E40", "Resumed E40 crossed block experiment mismatch")
    check(res_body.get("confirm_excluded") is True, "Resumed E40 crossed block missing confirm_excluded=True")
    check(int(res_body.get("depth", -1)) == int(depth), "Resumed E40 crossed block depth mismatch")
    check(res_body.get("condition") == condition, "Resumed E40 crossed block condition mismatch")
    check(res_body.get("split") == E40_SPLIT, "Resumed E40 crossed block split mismatch")
    check(res_body.get("arm") == arm, "Resumed E40 crossed block arm mismatch")
    check(res_body.get("head_pooling") == head_pooling, "Resumed E40 crossed block head_pooling mismatch")
    check(res_body.get("feature_pooling") == feature_pooling, "Resumed E40 crossed block feature_pooling mismatch")
    check(res_body.get("target_language") == target_lang, "Resumed E40 crossed block target_language mismatch")
    check(list(res_body.get("source_languages", ())) == list(LANGUAGES), "Resumed E40 crossed block sources mismatch")
    check(
        plain(res_body.get("upstream_feature_ref")) == plain(expected_feature_ref),
        "Resumed E40 crossed block upstream_feature_ref mismatch",
    )
    check(
        res_body.get("image_ids_sha256") == expected_image_ids_sha256,
        "Resumed E40 crossed block image_ids_sha256 mismatch",
    )
    arrays = _load_crossed_block_arrays(root, res_ref, expected_n_images, expected_n_cats)
    return res_ref, res_body, arrays


def _build_decision_summary(cells_out, depths, midpoint, decision_ranges):
    """Evaluate predeclared E40 decision ranges from phase3_cpu_plan."""
    lo, hi = decision_ranges["negligible_ci_within"]
    final_depth = int(depths[-1])
    per_cell = {}
    for depth in depths:
        dkey = str(depth)
        per_cell[dkey] = {}
        for cond in CONDITIONS:
            contrasts = cells_out[dkey][cond]["contrasts"]
            fg_last = contrasts["feature_gain_fixed_head_last"]["metrics"]["auroc"]["gap_reduction"]["primary_macro"]
            fg_mean = contrasts["feature_gain_fixed_head_mean"]["metrics"]["auroc"]["gap_reduction"]["primary_macro"]
            inter = contrasts["interaction"]["metrics"]["auroc"]["gap_reduction"]["primary_macro"]

            def _within_negligible(item):
                iv = item["interval"]
                return (
                    iv["valid_draws"] >= 1900
                    and iv["lower"] is not None
                    and iv["upper"] is not None
                    and iv["lower"] >= lo
                    and iv["upper"] <= hi
                )

            per_cell[dkey][cond] = {
                "feature_gain_fixed_head_last_auroc": fg_last["value"],
                "feature_gain_fixed_head_mean_auroc": fg_mean["value"],
                "interaction_auroc": inter["value"],
                "swap_asymmetry_absent": bool(
                    _within_negligible(fg_last) and _within_negligible(fg_mean) and _within_negligible(inter)
                ),
            }

    mid_ic = cells_out[str(midpoint)]["image_caption"]["contrasts"]["feature_gain_fixed_head_last"]["metrics"]["auroc"]["gap_reduction"]["primary_macro"]
    mid_co = cells_out[str(midpoint)]["caption_only"]["contrasts"]["feature_gain_fixed_head_last"]["metrics"]["auroc"]["gap_reduction"]["primary_macro"]
    ratio = None
    if mid_ic["value"] is not None and mid_co["value"] is not None and mid_ic["value"] != 0.0:
        ratio = float(mid_co["value"] / mid_ic["value"])
    co_iv = mid_co["interval"]
    co_ci_excl_zero = bool(co_iv["lower"] is not None and co_iv["lower"] > 0.0)
    final_ic_inter = cells_out[str(final_depth)]["image_caption"]["contrasts"]["interaction"]["metrics"]["auroc"]["gap_reduction"]["primary_macro"]["value"]

    return {
        "per_cell": per_cell,
        "midpoint_caption_only_replication": {
            "ratio_caption_only_to_image_caption": ratio,
            "threshold_min": float(decision_ranges["caption_only_replication_ratio_min"]),
            "caption_only_ci_excludes_zero": co_ci_excl_zero,
            "replicates": bool(
                ratio is not None
                and ratio >= float(decision_ranges["caption_only_replication_ratio_min"])
                and co_ci_excl_zero
            ),
        },
        "final_depth_image_caption_interaction_abs": None if final_ic_inter is None else abs(float(final_ic_inter)),
        "final_depth_localized": bool(
            final_ic_inter is not None
            and abs(float(final_ic_inter)) < float(decision_ranges["final_depth_interaction_absent_lt"])
        ),
    }


@contract
def run_e40(
    root,
    protocol,
    plan_ref,
    model,
    producer,
    *,
    resume_crossed_refs=None,
    resume_analysis_ref=None,
    run_id=None,
    progress=None,
):
    """Execute Phase 3 E40 CPU cross-depth + caption_only head x pooling swap for one model."""
    check(model in MODELS, f"Unknown model selector: {model!r}")
    require_open_split(E40_SPLIT)
    resolved_root = Path(root).resolve()

    plan = load_phase3_cpu_plan(resolved_root, protocol, plan_ref, bind=True)
    check(plan["confirm_excluded"] is True, "phase3_cpu_plan must enforce confirm_excluded=True")
    plan_protocol = plain(plan.protocol)
    binding = plan.binding
    check(binding is not None, "load_phase3_cpu_plan(..., bind=True) must return bound Phase1Binding")

    inv = plain(plan["e40_inventory"][model])
    depths = [int(d) for d in inv["depths"]]
    midpoint = int(plan["midpoints"][model])
    check(midpoint in depths, f"Midpoint {midpoint} not in E40 depths {depths} for {model}")

    resume_crossed_map = dict(resume_crossed_refs or {})
    valid_resume_keys = _valid_crossed_resume_keys(depths, midpoint)
    unknown_keys = set(resume_crossed_map.keys()) - valid_resume_keys
    check(not unknown_keys, f"Unknown resume_crossed_refs keys: {sorted(unknown_keys)}")

    import numpy as np
    from src.stage3.data import _read_verified_group
    from src.stage4.metrics import bootstrap_counts, paired_image_alignment, targets_for_prediction
    from src.stage4.plan import load_bootstrap
    from src.stage4.predictions import Prediction, _selected_parameters, load_prediction_block

    categories = tuple(plan_protocol["category_order"])
    n_cats = len(categories)
    if "categories" in binding.stage4_plan:
        check(
            tuple(binding.stage4_plan["categories"]) == categories,
            "Stage 4 plan categories mismatch protocol['category_order']",
        )
    primary_categories = tuple(plan_protocol["primary_categories"])
    check(len(primary_categories) == 14, f"Expected 14 fixed primary categories; got {len(primary_categories)}")
    cat_index = {cat: idx for idx, cat in enumerate(categories)}
    primary_indices = [cat_index[cat] for cat in primary_categories]
    screen_minimum = int(plan_protocol["minimum_positive_and_negative"]["SCREEN"])

    total_units = 2 + 56 + 2  # init + provenance/self-checks + 56 new crossed blocks + evaluate + publish

    # Load and authenticate Phase 2 E30 analysis & reused midpoint crossed refs.
    t0 = time.monotonic()
    p2_e30_ref, e30_body, _ = load_phase2_artifact(
        resolved_root,
        plan_protocol,
        plain(plan["phase2_refs"]["models"][model]["e30_analysis"]),
        P2_KIND_ANALYSIS,
        expected_schema=P2_SCHEMA_ANALYSIS,
        expected_dataset=plan["dataset_fingerprint"],
        expected_plan_identity=plan["phase2_refs"]["phase2_plan_identity"],
        expected_owner=model,
    )
    reused_midpoint_refs = plain(inv["reused_midpoint_crossed_refs"])
    check(
        reused_midpoint_refs == plain(e30_body["crossed_block_refs"]),
        f"e40_inventory reused_midpoint_crossed_refs mismatch with Phase 2 E30 for {model}",
    )

    matched_block_refs = {}
    for depth in depths:
        dkey = str(depth)
        matched_block_refs[dkey] = {}
        for cond in CONDITIONS:
            matched_block_refs[dkey][cond] = {
                "last_on_last": {
                    lang: plain(inv["matched_blocks"][dkey][cond]["last_caption"][lang]["prediction_ref"])
                    for lang in LANGUAGES
                },
                "mean_on_mean": {
                    lang: plain(inv["matched_blocks"][dkey][cond]["caption_mean"][lang]["prediction_ref"])
                    for lang in LANGUAGES
                },
            }

    _emit_progress(
        progress,
        "e40_plan_verified",
        unit="init",
        completed=1,
        total=total_units,
        read_hash_s=time.monotonic() - t0,
        model=model,
        depths=depths,
        midpoint_depth=midpoint,
    )

    # Fast-path explicit resume_analysis_ref verification.
    if resume_analysis_ref is not None:
        t_res = time.monotonic()
        res_ref, res_body, _ = verify_resume_ref(
            resolved_root,
            plan_protocol,
            resume_analysis_ref,
            KIND_ANALYSIS,
            expected_schema=SCHEMA_ANALYSIS,
            expected_dataset=plan["dataset_fingerprint"],
            expected_plan_identity=plan["plan_identity"],
            expected_owner=model,
        )
        check(res_body.get("experiment") == "E40", "Resume analysis artifact is not E40")
        check(res_body.get("confirm_excluded") is True, "Resume analysis missing confirm_excluded=True")
        check(list(res_body.get("depths", ())) == depths, "Resume analysis depths mismatch")
        check(int(res_body.get("midpoint_depth", -1)) == midpoint, "Resume analysis midpoint_depth mismatch")
        check(plain(res_body.get("bootstrap")) == plain(plan["bootstrap"]), "Resume analysis bootstrap mismatch")
        check(plain(res_body.get("matched_block_refs")) == matched_block_refs, "Resume analysis matched_block_refs mismatch")
        check(
            plain(res_body.get("reused_midpoint_crossed_refs")) == reused_midpoint_refs,
            "Resume analysis reused_midpoint_crossed_refs mismatch",
        )
        new_refs_in_body = plain(res_body.get("new_crossed_block_refs", {}))
        for depth, cond in _new_cells(depths, midpoint):
            dkey = str(depth)
            check(dkey in new_refs_in_body and cond in new_refs_in_body[dkey], f"Missing {dkey}/{cond} in resumed analysis")
            for arm in CROSSED_ARMS:
                for lang in LANGUAGES:
                    cref = new_refs_in_body[dkey][cond][arm][lang]
                    verify_resume_ref(
                        resolved_root,
                        plan_protocol,
                        cref,
                        KIND_PREDICTION,
                        expected_schema=SCHEMA_PREDICTION,
                        expected_dataset=plan["dataset_fingerprint"],
                        expected_plan_identity=plan["plan_identity"],
                        expected_owner=model,
                    )
                    rkey = f"{depth}:{cond}:{arm}:{lang}"
                    if rkey in resume_crossed_map and resume_crossed_map[rkey] is not None:
                        explicit_cref = normalize_ref(resolved_root, resume_crossed_map[rkey])
                        check(
                            plain(cref) == plain(explicit_cref),
                            f"Explicit resume_crossed_refs[{rkey!r}] differs from resumed analysis block",
                        )
        _emit_progress(
            progress,
            "e40_resumed",
            unit="resume",
            completed=total_units,
            total=total_units,
            read_hash_s=time.monotonic() - t_res,
            model=model,
            analysis_ref=plain(res_ref),
            crossed_block_refs=plain(res_body.get("crossed_block_refs")),
        )
        return res_ref, res_body, plain(res_body.get("crossed_block_refs"))

    # 2. Load Stage 2 SCREEN features, Stage 3 frozen heads, and Stage 4 matched blocks across all 8 cells;
    #    run provenance checks and en->en matched self-checks (atol=0, rtol=0) at every (depth, condition).
    t_prov = time.monotonic()
    fit = binding.stage4_plan.fit
    store = binding.stage4_plan.store
    fit_groups = fit["groups"]
    ordinary = binding.stage4_plan["ordinary"]

    cell_data = {}
    provenance_checks = {}
    matched_self_checks = {}

    for depth in depths:
        dkey = str(depth)
        provenance_checks[dkey] = {}
        matched_self_checks[dkey] = {}
        for cond in CONDITIONS:
            gids_by_pooling = {p: {} for p in POOLINGS}
            rows_by_pooling = {p: {} for p in POOLINGS}
            matrices_by_pooling = {p: {} for p in POOLINGS}
            source_heads_by_pooling = {p: {} for p in POOLINGS}
            matched_preds_by_pooling = {p: {} for p in POOLINGS}

            for pooling in POOLINGS:
                for lang in LANGUAGES:
                    feat_meta = inv["screen_feature_releases"][dkey][cond][pooling][lang]
                    gid = feat_meta["group_id"]
                    gids_by_pooling[pooling][lang] = gid
                    rows, matrix = _read_verified_group(fit, store, gid, E40_SPLIT)
                    rows_by_pooling[pooling][lang] = rows
                    matrices_by_pooling[pooling][lang] = matrix
                    source_heads_by_pooling[pooling][lang] = _selected_parameters(binding.stage4_plan, gid)

                    entry = inv["matched_blocks"][dkey][cond][pooling][lang]
                    block_preds = load_prediction_block(
                        binding.stage4_plan,
                        entry["prediction_ref"],
                        expected_block=entry["block_id"],
                    )
                    check(
                        set(block_preds.keys()) == set(entry["routes"].values()),
                        f"Route mismatch in Stage 4 matched block {entry['block_id']}",
                    )
                    matched_preds_by_pooling[pooling][lang] = {
                        src_lang: block_preds[rid] for src_lang, rid in entry["routes"].items()
                    }

            cell_prov = verify_e40_cell_provenance(
                binding, model, depth, cond, gids_by_pooling, rows_by_pooling, matrices_by_pooling
            )
            cell_self = _run_en_matched_self_checks(
                model=model,
                depth=depth,
                condition=cond,
                rows_by_pooling=rows_by_pooling,
                matrices_by_pooling=matrices_by_pooling,
                source_heads_by_pooling=source_heads_by_pooling,
                matched_preds_by_pooling=matched_preds_by_pooling,
            )
            provenance_checks[dkey][cond] = cell_prov
            matched_self_checks[dkey][cond] = cell_self
            cell_data[(depth, cond)] = {
                "gids_by_pooling": gids_by_pooling,
                "rows_by_pooling": rows_by_pooling,
                "matrices_by_pooling": matrices_by_pooling,
                "source_heads_by_pooling": source_heads_by_pooling,
                "matched_preds_by_pooling": matched_preds_by_pooling,
            }

    _emit_progress(
        progress,
        "e40_provenance_and_self_checks_passed",
        unit="provenance_and_self_check",
        completed=2,
        total=total_units,
        read_hash_s=time.monotonic() - t_prov,
        model=model,
        n_cells=len(depths) * len(CONDITIONS),
        en_self_check_max_abs_diff=0.0,
    )

    # 3. Reuse Phase 2 E30's 8 midpoint image_caption crossed blocks (and verify logit parity <= 1e-12),
    #    then score (or resume) the 56 new crossed blocks across the remaining 7 (depth, condition) cells.
    base_run_id = run_id or new_run_id()
    crossed_block_refs = {str(d): {c: {arm: {} for arm in CROSSED_ARMS} for c in CONDITIONS} for d in depths}
    new_crossed_block_refs = {}
    crossed_preds_by_cell = {(d, c): {arm: {} for arm in CROSSED_ARMS} for d in depths for c in CONDITIONS}

    # 3a. Reused midpoint image_caption crossed blocks + logit parity check vs Phase 2 E30.
    mid_dkey = str(midpoint)
    mid_ic = cell_data[(midpoint, "image_caption")]
    mid_prov = provenance_checks[mid_dkey]["image_caption"]
    max_reused_logit_diff = 0.0
    for arm in CROSSED_ARMS:
        head_pooling, feature_pooling = CROSSED_ARM_SPECS[arm]
        for target_lang in LANGUAGES:
            rows_target = mid_ic["rows_by_pooling"][feature_pooling][target_lang]
            matrix_target = mid_ic["matrices_by_pooling"][feature_pooling][target_lang]
            target_gid = mid_ic["gids_by_pooling"][feature_pooling][target_lang]
            target_feature_ref = plain(fit_groups[target_gid]["releases"][E40_SPLIT])
            image_ids_sha256 = mid_prov["per_language"][target_lang]["image_ids_sha256"]
            ref = reused_midpoint_refs[arm][target_lang]

            block_ref, _, (logits_arr, estimable_arr, prevalence_arr) = _verify_reused_e30_crossed_block(
                root=resolved_root,
                protocol=plan_protocol,
                ref=ref,
                plan=plan,
                model=model,
                midpoint=midpoint,
                arm=arm,
                head_pooling=head_pooling,
                feature_pooling=feature_pooling,
                target_lang=target_lang,
                expected_feature_ref=target_feature_ref,
                expected_image_ids_sha256=image_ids_sha256,
                expected_n_images=len(rows_target),
                expected_n_cats=n_cats,
            )
            # Check midpoint crossed logit parity <= 1e-12 against Stage 3 transform.
            for s_idx, src_lang in enumerate(LANGUAGES):
                coef, intercept, estimable, _, stats = mid_ic["source_heads_by_pooling"][head_pooling][src_lang]
                recomputed = _score_matrix(matrix_target, coef, intercept, stats, estimable)
                diff = float(np.max(np.abs(recomputed - logits_arr[s_idx])))
                max_reused_logit_diff = max(max_reused_logit_diff, diff)
                check(
                    diff <= MIDPOINT_PARITY_TOL,
                    f"Midpoint E30 crossed logit parity failed for {model}/{arm}/{src_lang}->{target_lang}: {diff}",
                )

            crossed_block_refs[mid_dkey]["image_caption"][arm][target_lang] = plain(block_ref)
            crossed_preds_by_cell[(midpoint, "image_caption")][arm][target_lang] = {
                src_lang: Prediction(
                    reference=plain(block_ref),
                    route={
                        "route_id": f"{model}:e30:{arm}:{src_lang}->{target_lang}",
                        "source_language": src_lang,
                        "target_language": target_lang,
                        "arm": arm,
                    },
                    split=E40_SPLIT,
                    rows=tuple(rows_target),
                    logits=logits_arr[s_idx],
                    estimable=estimable_arr[s_idx],
                    prevalence=prevalence_arr[s_idx],
                )
                for s_idx, src_lang in enumerate(LANGUAGES)
            }

    # 3b. Score or resume the 56 new crossed blocks across the remaining 7 (depth, condition) cells.
    completed_units = 2
    for depth, cond in _new_cells(depths, midpoint):
        dkey = str(depth)
        new_crossed_block_refs.setdefault(dkey, {})[cond] = {arm: {} for arm in CROSSED_ARMS}
        cdata = cell_data[(depth, cond)]
        cprov = provenance_checks[dkey][cond]
        cself = matched_self_checks[dkey][cond]

        for arm in CROSSED_ARMS:
            head_pooling, feature_pooling = CROSSED_ARM_SPECS[arm]
            for target_lang in LANGUAGES:
                completed_units += 1
                rkey = f"{depth}:{cond}:{arm}:{target_lang}"
                rows_target = cdata["rows_by_pooling"][feature_pooling][target_lang]
                matrix_target = cdata["matrices_by_pooling"][feature_pooling][target_lang]
                target_gid = cdata["gids_by_pooling"][feature_pooling][target_lang]
                target_feature_ref = plain(fit_groups[target_gid]["releases"][E40_SPLIT])
                image_ids_sha256 = cprov["per_language"][target_lang]["image_ids_sha256"]

                if rkey in resume_crossed_map and resume_crossed_map[rkey] is not None:
                    t_res = time.monotonic()
                    block_ref, _, (logits_arr, estimable_arr, prevalence_arr) = _verify_e40_crossed_block_ref(
                        root=resolved_root,
                        protocol=plan_protocol,
                        ref=resume_crossed_map[rkey],
                        plan=plan,
                        model=model,
                        depth=depth,
                        condition=cond,
                        arm=arm,
                        head_pooling=head_pooling,
                        feature_pooling=feature_pooling,
                        target_lang=target_lang,
                        expected_feature_ref=target_feature_ref,
                        expected_image_ids_sha256=image_ids_sha256,
                        expected_n_images=len(rows_target),
                        expected_n_cats=n_cats,
                    )
                    crossed_block_refs[dkey][cond][arm][target_lang] = plain(block_ref)
                    new_crossed_block_refs[dkey][cond][arm][target_lang] = plain(block_ref)
                    crossed_preds_by_cell[(depth, cond)][arm][target_lang] = {
                        src_lang: Prediction(
                            reference=plain(block_ref),
                            route={
                                "route_id": f"{model}:e40:d{depth}:{cond}:{arm}:{src_lang}->{target_lang}",
                                "source_language": src_lang,
                                "target_language": target_lang,
                                "arm": arm,
                            },
                            split=E40_SPLIT,
                            rows=tuple(rows_target),
                            logits=logits_arr[s_idx],
                            estimable=estimable_arr[s_idx],
                            prevalence=prevalence_arr[s_idx],
                        )
                        for s_idx, src_lang in enumerate(LANGUAGES)
                    }
                    _emit_progress(
                        progress,
                        "e40_crossed_block_resumed",
                        unit=f"crossed:{rkey}",
                        completed=completed_units,
                        total=total_units,
                        read_hash_s=time.monotonic() - t_res,
                        model=model,
                        block_ref=plain(block_ref),
                    )
                    continue

                t_comp = time.monotonic()
                logits_list = []
                valid_list = []
                prev_list = []
                routes_meta = []
                upstream = [plain(plan.reference), target_feature_ref]

                for s_idx, src_lang in enumerate(LANGUAGES):
                    src_gid = cdata["gids_by_pooling"][head_pooling][src_lang]
                    coef, intercept, estimable, prevalence, stats = cdata["source_heads_by_pooling"][head_pooling][src_lang]
                    logits_s = _score_matrix(matrix_target, coef, intercept, stats, estimable)
                    logits_list.append(logits_s)
                    valid_list.append(estimable)
                    prev_list.append(prevalence)

                    src_sel_ref = plain(ordinary[src_gid]["selection"])
                    src_cand_ref = plain(ordinary[src_gid]["candidate"])
                    src_stats_ref = plain(stats["reference"])
                    upstream.extend([src_sel_ref, src_cand_ref, src_stats_ref])

                    routes_meta.append({
                        "route_id": f"{model}:e40:d{depth}:{cond}:{arm}:{src_lang}->{target_lang}",
                        "source_language": src_lang,
                        "row_index": s_idx,
                        "source_group": src_gid,
                        "target_group": target_gid,
                        "source_system_ref": src_sel_ref,
                        "source_candidate_ref": src_cand_ref,
                        "source_statistics_ref": src_stats_ref,
                        "supervision_label": SUPERVISION_DIAGONAL if src_lang == target_lang else SUPERVISION_OFF_DIAGONAL,
                        "arm": arm,
                    })

                logits_arr = np.asarray(logits_list, dtype=np.float64)
                estimable_arr = np.asarray(valid_list, dtype=bool)
                prevalence_arr = np.asarray(prev_list, dtype=np.float64)
                compute_s = time.monotonic() - t_comp

                block_body = {
                    "schema": SCHEMA_PREDICTION,
                    "experiment": "E40",
                    "plan": plain(plan.reference),
                    "plan_identity": plan["plan_identity"],
                    "model": model,
                    "depth": int(depth),
                    "midpoint_depth": midpoint,
                    "condition": cond,
                    "split": E40_SPLIT,
                    "arm": arm,
                    "head_pooling": head_pooling,
                    "feature_pooling": feature_pooling,
                    "target_language": target_lang,
                    "source_languages": list(LANGUAGES),
                    "n_images": len(rows_target),
                    "n_categories": n_cats,
                    "category_order": list(categories),
                    "image_ids_sha256": image_ids_sha256,
                    "provenance_checks": cprov,
                    "matched_self_check": cself[head_pooling],
                    "routes": routes_meta,
                    "upstream_feature_ref": target_feature_ref,
                    "confirm_excluded": True,
                }

                t_write = time.monotonic()
                block_ref = publish_phase3(
                    resolved_root,
                    plan_protocol,
                    plan["dataset_fingerprint"],
                    f"{E40_PREDICTIONS_NAMESPACE}/{model}",
                    KIND_PREDICTION,
                    block_body,
                    producer,
                    arrays={
                        "logits": logits_arr,
                        "estimable": estimable_arr,
                        "prevalence": prevalence_arr,
                    },
                    upstream=upstream,
                    run_id=f"{base_run_id}_d{depth}_{cond}_{arm}_{target_lang}",
                )
                write_s = time.monotonic() - t_write

                crossed_block_refs[dkey][cond][arm][target_lang] = plain(block_ref)
                new_crossed_block_refs[dkey][cond][arm][target_lang] = plain(block_ref)
                crossed_preds_by_cell[(depth, cond)][arm][target_lang] = {
                    src_lang: Prediction(
                        reference=plain(block_ref),
                        route=routes_meta[s_idx],
                        split=E40_SPLIT,
                        rows=tuple(rows_target),
                        logits=logits_arr[s_idx],
                        estimable=estimable_arr[s_idx],
                        prevalence=prevalence_arr[s_idx],
                    )
                    for s_idx, src_lang in enumerate(LANGUAGES)
                }
                _emit_progress(
                    progress,
                    "e40_crossed_block_published",
                    unit=f"crossed:{rkey}",
                    completed=completed_units,
                    total=total_units,
                    compute_s=compute_s,
                    write_s=write_s,
                    model=model,
                    block_ref=plain(block_ref),
                )

    # 4. Evaluate all 8 (depth, condition) cells on the identical frozen (2000, 360) bootstrap,
    #    compute paired `image_caption - caption_only` swap contrasts inside every draw,
    #    and verify midpoint image_caption parity <= 1e-12 vs Phase 2 E30.
    t_eval = time.monotonic()
    all_preds = []
    pred_keys = []
    preds_by_cell_arm = {}

    for depth in depths:
        for cond in CONDITIONS:
            preds_by_arm = {
                "last_on_last": cell_data[(depth, cond)]["matched_preds_by_pooling"]["last_caption"],
                "mean_on_mean": cell_data[(depth, cond)]["matched_preds_by_pooling"]["caption_mean"],
                "head_last_on_mean": crossed_preds_by_cell[(depth, cond)]["head_last_on_mean"],
                "head_mean_on_last": crossed_preds_by_cell[(depth, cond)]["head_mean_on_last"],
            }
            preds_by_cell_arm[(depth, cond)] = preds_by_arm
            for arm in ALL_ARMS:
                for s in LANGUAGES:
                    for t in LANGUAGES:
                        all_preds.append(preds_by_arm[arm][t][s])
                        pred_keys.append((depth, cond, arm, s, t))

    screen_ids = tuple(binding.stage4_plan["screen_ids"])
    ids, aligned_tuple = paired_image_alignment(binding.stage4_plan, *all_preds)
    check(ids == screen_ids, "Aligned SCREEN IDs mismatch plan screen_ids")
    aligned_by_cell = {(d, c): {} for d in depths for c in CONDITIONS}
    for (depth, cond, arm, s, t), aligned in zip(pred_keys, aligned_tuple):
        aligned_by_cell[(depth, cond)][(arm, s, t)] = aligned

    y_raw, known_raw = targets_for_prediction(binding.stage4_plan, all_preds[0])
    aligned_0 = aligned_tuple[0]
    y = y_raw[aligned_0]
    known = known_raw[aligned_0]

    draws = load_bootstrap(binding.stage4_plan)
    check(
        hashlib.sha256(draws.tobytes()).hexdigest() == plan["bootstrap"]["sha256"],
        "Loaded bootstrap SHA-256 mismatch with phase3_cpu_plan",
    )
    counts = bootstrap_counts(draws, len(screen_ids))
    total_draws = int(draws.shape[0])

    cells_out = {}
    contrast_caches = {}
    supports_ref_shared = None

    for depth in depths:
        dkey = str(depth)
        cells_out[dkey] = {}
        for cond in CONDITIONS:
            arms_res, contrasts_res, collapse_res, c_cache, supports_ref = _evaluate_cell_arms_and_contrasts(
                preds_by_arm=preds_by_cell_arm[(depth, cond)],
                aligned_by_key=aligned_by_cell[(depth, cond)],
                y=y,
                known=known,
                counts=counts,
                categories=categories,
                primary_indices=primary_indices,
                total_draws=total_draws,
                minimum=screen_minimum,
            )
            if supports_ref_shared is None:
                supports_ref_shared = supports_ref
            cells_out[dkey][cond] = {
                "depth": int(depth),
                "condition": cond,
                "arms": arms_res,
                "contrasts": contrasts_res,
                "off_distribution_collapse": collapse_res,
            }
            contrast_caches[(depth, cond)] = c_cache

    max_metric_parity_diff = _verify_midpoint_e30_analysis_parity(
        cells_out[mid_dkey]["image_caption"], e30_body, tol=MIDPOINT_PARITY_TOL
    )

    condition_swap_contrasts = {}
    for depth in depths:
        dkey = str(depth)
        condition_swap_contrasts[dkey] = {
            "depth": int(depth),
            "description": "Paired image_caption - caption_only swap contrast on identical (2000, 360) draws",
            "contrasts": compute_condition_swap_contrast(
                ic_cache=contrast_caches[(depth, "image_caption")],
                co_cache=contrast_caches[(depth, "caption_only")],
                supports_ref=supports_ref_shared,
                categories=categories,
                primary_indices=primary_indices,
                minimum=screen_minimum,
            ),
        }

    decision_summary = _build_decision_summary(
        cells_out, depths, midpoint, plain(plan["decision_ranges"]["E40"])
    )
    compute_eval_s = time.monotonic() - t_eval
    _emit_progress(
        progress,
        "e40_contrasts_computed",
        unit="evaluate_8cells",
        completed=total_units - 1,
        total=total_units,
        compute_s=compute_eval_s,
        model=model,
        midpoint_e30_logit_diff=max_reused_logit_diff,
        midpoint_e30_metric_diff=max_metric_parity_diff,
    )

    # 5. Publish E40 analysis block under phase3/results/e40/<model>/<run_id>/.
    t_pub = time.monotonic()
    analysis_body = {
        "schema": SCHEMA_ANALYSIS,
        "experiment": "E40",
        "plan": plain(plan.reference),
        "plan_identity": plan["plan_identity"],
        "model": model,
        "depths": depths,
        "midpoint_depth": midpoint,
        "conditions": list(CONDITIONS),
        "bootstrap": plain(plan["bootstrap"]),
        "reused_e30_analysis_ref": plain(p2_e30_ref),
        "reused_midpoint_crossed_refs": reused_midpoint_refs,
        "matched_block_refs": matched_block_refs,
        "crossed_block_refs": crossed_block_refs,
        "new_crossed_block_refs": new_crossed_block_refs,
        "provenance_checks": provenance_checks,
        "matched_self_checks": matched_self_checks,
        "midpoint_e30_parity_check": {
            "passed": True,
            "max_logit_abs_diff": max_reused_logit_diff,
            "max_metric_abs_diff": max_metric_parity_diff,
            "tol": MIDPOINT_PARITY_TOL,
        },
        "cells": cells_out,
        "condition_swap_contrasts": condition_swap_contrasts,
        "decision_summary": decision_summary,
        "supervision_labels": {
            "diagonal": SUPERVISION_DIAGONAL,
            "off_diagonal": SUPERVISION_OFF_DIAGONAL,
            "crossed_diagonal_note": (
                "Crossed receiving diagonals carry supervision_label='within_context_source_fit' "
                "and are distinguished from matched diagonals by arm ('head_last_on_mean' / 'head_mean_on_last')."
            ),
        },
        "confirm_excluded": True,
    }
    analysis_upstream = [plain(plan.reference), plain(p2_e30_ref)]
    for depth in depths:
        dkey = str(depth)
        for cond in CONDITIONS:
            for arm in MATCHED_ARMS:
                for lang in LANGUAGES:
                    analysis_upstream.append(matched_block_refs[dkey][cond][arm][lang])
            for arm in CROSSED_ARMS:
                for lang in LANGUAGES:
                    analysis_upstream.append(crossed_block_refs[dkey][cond][arm][lang])

    analysis_ref = publish_phase3(
        resolved_root,
        plan_protocol,
        plan["dataset_fingerprint"],
        f"{E40_RESULTS_NAMESPACE}/{model}",
        KIND_ANALYSIS,
        analysis_body,
        producer,
        upstream=analysis_upstream,
        run_id=base_run_id,
    )
    _, verified_analysis_body, _ = verify_resume_ref(
        resolved_root,
        plan_protocol,
        analysis_ref,
        KIND_ANALYSIS,
        expected_schema=SCHEMA_ANALYSIS,
        expected_dataset=plan["dataset_fingerprint"],
        expected_plan_identity=plan["plan_identity"],
        expected_owner=model,
    )
    write_pub_s = time.monotonic() - t_pub
    _emit_progress(
        progress,
        "e40_published",
        unit="publish",
        completed=total_units,
        total=total_units,
        write_s=write_pub_s,
        model=model,
        analysis_ref=plain(analysis_ref),
        crossed_block_refs=crossed_block_refs,
    )
    return analysis_ref, verified_analysis_body, crossed_block_refs


def _parse_ref_arg(value):
    if value is None:
        return None
    text = value.strip()
    if text.startswith("{"):
        parsed = json.loads(text)
        check(isinstance(parsed, dict), "JSON reference argument must decode to an object")
        return parsed
    return text


def _parse_mapping_arg(value):
    if value is None:
        return None
    text = value.strip()
    if text.startswith("{"):
        parsed = json.loads(text)
    else:
        parsed = json.loads(Path(text).read_text(encoding="utf-8"))
    check(isinstance(parsed, dict), "Mapping argument must decode to a JSON object")
    return parsed


def _emit_cli(event):
    print(json.dumps(event), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-root", required=True)
    parser.add_argument("--protocol", type=Path, default=REPO / "configs/protocol.json")
    parser.add_argument("--plan", required=True, help="Explicit phase3_cpu_plan {'path', 'sha256'} JSON or path")
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument(
        "--storage-scope",
        default="persistent-drive",
        choices=("persistent-drive", "local-development"),
    )
    parser.add_argument(
        "--resume-crossed-refs",
        help="Optional JSON object (or path) mapping '{depth}:{condition}:{arm}:{target_lang}' -> {'path', 'sha256'}",
    )
    parser.add_argument(
        "--resume-analysis-ref",
        help="Optional explicit E40 analysis {'path', 'sha256'} JSON or path",
    )
    parser.add_argument("--run-id", help="Optional explicit run_id")
    parser.add_argument("--output-ref", type=Path, help="Optional JSON file path for emitted references")
    args = parser.parse_args(argv)

    root = phase1_root(args.phase1_root)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    producer = current_producer(args.storage_scope)
    _emit_cli({
        "event": "e40_start",
        "model": args.model,
        "code_commit": producer["code_commit"],
        "storage_scope": producer["storage_scope"],
    })
    analysis_ref, analysis_body, crossed_block_refs = run_e40(
        root,
        protocol,
        _parse_ref_arg(args.plan),
        args.model,
        producer,
        resume_crossed_refs=_parse_mapping_arg(args.resume_crossed_refs),
        resume_analysis_ref=_parse_ref_arg(args.resume_analysis_ref),
        run_id=args.run_id,
        progress=_emit_cli,
    )
    summary = {
        "event": "e40_complete",
        "status": "passed",
        "model": args.model,
        "analysis_ref": plain(analysis_ref),
        "crossed_block_refs": plain(crossed_block_refs),
        "midpoint_e30_parity_check": plain(analysis_body["midpoint_e30_parity_check"]),
        "decision_summary": plain(analysis_body["decision_summary"]),
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
        _emit_cli({"status": "blocked", "blockers": [str(exc)], "phase3_e40_complete": False})
        raise SystemExit(2)
