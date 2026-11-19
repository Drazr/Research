"""Phase 2 Stage 3 CPU worker for E30: head x pooling swap at fixed midpoint.

Implements phase2.md §1.3-1.5 and §3.1-3.3:
1. Loads `phase2_plan`, explicitly binds `Phase1Binding` via `bind_phase1`, loads
   `phase2_e29_gate` via `load_e29_gate`, and refuses to run unless
   `decisions["E30"]["decision"] == "OPEN"`.
2. Verifies identical midpoint depth, hook (`decoder:<depth>`), dimension,
   `model_plan`, and 360-image SCREEN row keys across `last_caption` and
   `caption_mean` for all 4 languages, and asserts exact `en->en` matched-arm
   reproduction (`atol=0, rtol=0`) against Stage 4 for both poolings.
3. Scores ONLY the 2 crossed arms (8 blocks / 32 routes per model:
   `head_last_on_mean` and `head_mean_on_last`, including crossed `t->t`
   diagonals) via `src.stage3.engine.transform` and publishes under
   `phase2/predictions/e30_crossed/<model>/<run_id>/`.
4. Evaluates all 4 arms (reusing 8 Stage 4 matched blocks) on the frozen
   `(2000, 360)` bootstrap via `src.stage4.metrics` (using
   `protocol["category_order"]` and `int(protocol["minimum_positive_and_negative"]["SCREEN"])`),
   computing within-draw paired feature gains, head-swap contrasts, interaction,
   total matched reduction, and off-distribution collapse flags (`>= 0.015`
   AUROC drop vs matched `last_on_last`), publishing under
   `phase2/results/e30/<model>/<run_id>/`.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from src.common.contracts import LANGUAGES, ContractError, fingerprint
from src.common.storage import new_run_id, phase1_root, within
from src.phase2.gates import load_e29_gate
from src.phase2.interfaces import (
    KIND_ANALYSIS,
    KIND_PREDICTION,
    MODELS,
    POOLINGS,
    REPO,
    SCHEMA_ANALYSIS,
    SCHEMA_PREDICTION,
    bind_phase1,
    check,
    contract,
    e29_screen_blocks,
    load_frozen_source_head,
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

E30_CONDITION = "image_caption"
E30_SPLIT = "SCREEN"
E30_PREDICTIONS_NAMESPACE = "phase2/predictions/e30_crossed"
E30_RESULTS_NAMESPACE = "phase2/results/e30"

MATCHED_ARMS = ("last_on_last", "mean_on_mean")
CROSSED_ARMS = ("head_last_on_mean", "head_mean_on_last")
ALL_ARMS = MATCHED_ARMS + CROSSED_ARMS

CROSSED_ARM_SPECS = {
    "head_last_on_mean": ("last_caption", "caption_mean"),
    "head_mean_on_last": ("caption_mean", "last_caption"),
}

LOSS_METRICS = frozenset({"brier", "nll"})
SUPERVISION_DIAGONAL = "within_context_source_fit"
SUPERVISION_OFF_DIAGONAL = "source_only_transfer"
COLLAPSE_DROP_THRESHOLD = 0.015


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


def _valid_crossed_resume_keys():
    return {f"{arm}:{lang}" for arm in CROSSED_ARMS for lang in LANGUAGES}


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
    """Build 25-category table and 14-primary macro summary from aligned category arrays."""
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


def _parse_hook_from_site(site, expected_pooling):
    """Extract residual-stream hook (`decoder:<depth>`) from a Stage 2 site string."""
    parts = str(site).split(":")
    check(
        len(parts) == 3 and parts[0] == "decoder" and parts[2] == expected_pooling,
        f"Unexpected midpoint site format: {site!r} (expected decoder:<depth>:{expected_pooling})",
    )
    return f"{parts[0]}:{parts[1]}", int(parts[1])


@contract
def check_e30_gate_open(gate_body, model):
    """Assert `phase2_e29_gate` authorizes E30 (`decisions['E30']['decision'] == 'OPEN'`)."""
    check(gate_body.get("model") == model, f"E29 gate model mismatch: expected {model!r}, got {gate_body.get('model')!r}")
    decisions = gate_body.get("decisions")
    check(isinstance(decisions, dict) or hasattr(decisions, "get"), "E29 gate missing decisions")
    e30 = decisions.get("E30")
    check(e30 is not None, "E29 gate missing E30 decision")
    decision = e30.get("decision")
    reason = e30.get("reason")
    check(
        decision == "OPEN",
        f"E30 gate is closed for {model}: decision={decision!r}, reason={reason!r} (requires 'OPEN')",
    )
    return {"decision": decision, "reason": reason}


@contract
def verify_e30_provenance(binding, model, depth, gids_by_pooling, rows_by_pooling, matrices_by_pooling):
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
            f"Model mismatch in midpoint groups for {lang}",
        )
        check(
            g_last["condition"] == E30_CONDITION and g_mean["condition"] == E30_CONDITION,
            f"Condition mismatch in midpoint groups for {lang}",
        )
        hook_last, depth_last = _parse_hook_from_site(g_last["site"], "last_caption")
        hook_mean, depth_mean = _parse_hook_from_site(g_mean["site"], "caption_mean")
        check(
            depth_last == depth and depth_mean == depth,
            f"Midpoint depth mismatch for {model}/{lang}: last={depth_last}, mean={depth_mean}, expected={depth}",
        )
        check(
            hook_last == hook_mean == expected_hook,
            f"Hook mismatch for {model}/{lang}: last={hook_last!r}, mean={hook_mean!r}, expected={expected_hook!r}",
        )

        dim_last = int(g_last["dimension"])
        dim_mean = int(g_mean["dimension"])
        check(
            dim_last == dim_mean and dim_last > 0,
            f"Feature dimension mismatch across poolings for {model}/{lang}: last={dim_last}, mean={dim_mean}",
        )
        if shared_dimension is None:
            shared_dimension = dim_last
        else:
            check(
                dim_last == shared_dimension,
                f"Feature dimension mismatch across languages for {model}/{lang}: {dim_last} vs {shared_dimension}",
            )

        mp_last = plain(g_last["model_plan"])
        mp_mean = plain(g_mean["model_plan"])
        check(
            mp_last == mp_mean == expected_model_plan,
            f"Stage 2 model_plan mismatch across poolings for {model}/{lang}",
        )

        rows_last = rows_by_pooling["last_caption"][lang]
        rows_mean = rows_by_pooling["caption_mean"][lang]
        mat_last = matrices_by_pooling["last_caption"][lang]
        mat_mean = matrices_by_pooling["caption_mean"][lang]

        check(
            plain(rows_last) == plain(rows_mean),
            f"SCREEN row keys mismatch between last_caption and caption_mean for {model}/{lang}",
        )
        check(
            len(rows_last) == len(screen_ids),
            f"Unexpected SCREEN row count for {model}/{lang}: {len(rows_last)} vs {len(screen_ids)}",
        )
        check(
            all(r["split"] == E30_SPLIT for r in rows_last),
            f"Non-SCREEN split detected in E30 features for {model}/{lang}",
        )
        row_image_ids = [r["image_id"] for r in rows_last]
        check(
            sorted(row_image_ids) == list(screen_ids) and len(set(row_image_ids)) == len(screen_ids),
            f"SCREEN image_id population mismatch for {model}/{lang}",
        )
        check(
            mat_last.shape == (len(rows_last), shared_dimension)
            and mat_mean.shape == (len(rows_mean), shared_dimension),
            f"Feature matrix shape mismatch for {model}/{lang}: last={mat_last.shape}, mean={mat_mean.shape}",
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
        "condition": E30_CONDITION,
        "split": E30_SPLIT,
        "midpoint_depth": int(depth),
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
    check(logits.dtype == np.float64 and np.isfinite(logits).all(), "Nonfinite E30 logits")
    estimable_bool = np.asarray(estimable, dtype=bool)
    check(logits.shape[1] == estimable_bool.shape[0], "E30 logits/estimable column mismatch")
    check(not np.any(logits[:, ~estimable_bool]), "Nonzero logits on unestimable E30 columns")
    return logits


def _run_en_matched_self_checks(
    *,
    model,
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
            f"SCREEN en->en row alignment mismatch for {model}/{pooling}",
        )
        check(
            computed.shape == accepted.logits.shape,
            f"SCREEN en->en shape mismatch for {model}/{pooling}: {computed.shape} vs {accepted.logits.shape}",
        )
        max_abs_diff = float(np.max(np.abs(computed - accepted.logits)))
        check(
            np.array_equal(computed, accepted.logits) and max_abs_diff == 0.0,
            f"SCREEN en->en matched self-check failed for {model}/{pooling}: max_abs_diff={max_abs_diff}",
        )
        results[pooling] = {
            "route": "en->en",
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
    """Load and validate logits, estimable, prevalence from a published E30 crossed block."""
    import numpy as np

    marker_dir = within(root, ref["path"]).parent
    logits = np.load(marker_dir / "logits.npy", allow_pickle=False)
    estimable = np.load(marker_dir / "estimable.npy", allow_pickle=False)
    prevalence = np.load(marker_dir / "prevalence.npy", allow_pickle=False)
    n_src = len(LANGUAGES)
    check(
        logits.shape == (n_src, expected_n_images, expected_n_cats) and logits.dtype == np.float64,
        f"Invalid E30 crossed logits shape/dtype: {logits.shape}/{logits.dtype}",
    )
    check(np.isfinite(logits).all(), "Nonfinite logits in resumed E30 crossed block")
    check(
        estimable.shape == (n_src, expected_n_cats) and estimable.dtype == bool,
        f"Invalid E30 crossed estimable shape/dtype: {estimable.shape}/{estimable.dtype}",
    )
    check(
        prevalence.shape == (n_src, expected_n_cats) and prevalence.dtype == np.float64,
        f"Invalid E30 crossed prevalence shape/dtype: {prevalence.shape}/{prevalence.dtype}",
    )
    check(not np.any(logits * (~estimable[:, None, :])), "Nonzero unestimable logits in resumed E30 block")
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
    """Evaluate §1.5 / §3.3 off-distribution collapse flags against matched `last_on_last`.

    Flags collapse when BOTH crossed arms (`head_last_on_mean` and `head_mean_on_last`)
    fall `>= drop_threshold` (0.015 AUROC) below matched `last_on_last`.
    """
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


def _evaluate_all_arms_and_contrasts(
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
    """Compute per-arm tables, within-draw 2x2 paired contrasts, and collapse flags."""
    import numpy as np
    from src.stage4.metrics import METRICS

    n_cats = len(categories)
    off_pairs = _off_diagonal_pairs()

    # 1. Evaluate all 64 routes (4 arms x 16 routes) once.
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

    # 2. Build per-arm summaries and cache draw arrays needed for paired contrasts.
    arms_out = {}
    # Caches indexed by [arm][metric]
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
            # 16 per-route absolute metrics
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

            # 4-diagonal mean D_mean(A)
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

            # 12-route mean absolute transfer T_mean(A)
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

            # 12 off-diagonal route gaps G(s->t; A) = M(t->t; A) - M(s->t; A) (reversed for losses)
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

    # 3. Within-draw paired 2x2 contrasts across all metrics.
    supports_ref = arm_route_supports["last_on_last"]
    ok_primary_ref = sum(
        min(supports_ref[idx][1], supports_ref[idx][2]) >= minimum for idx in primary_indices
    )

    def _diff_opt(a, b):
        return None if (a is None or b is None) else float(a - b)

    def _build_pair_contrast(arm_high_gap, arm_low_gap, description):
        out = {"description": description, "metrics": {}}
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
            for s, t in off_pairs:
                rkey = _route_key(s, t)
                rv = _diff_opt(
                    arm_route_gap_vals[arm_high_gap][metric][rkey],
                    arm_route_gap_vals[arm_low_gap][metric][rkey],
                )
                rd = arm_route_gap_draws[arm_high_gap][metric][rkey] - arm_route_gap_draws[arm_low_gap][metric][rkey]
                per_route[rkey] = _summarize_aggregate(
                    rv,
                    rd,
                    categories_count=len(primary_indices),
                    defined_count=len(primary_indices) if rv is not None else 0,
                    ok_count=ok_primary_ref if rv is not None else 0,
                    minimum=minimum,
                )

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
                per_route[rkey] = _summarize_aggregate(
                    rv,
                    rd,
                    categories_count=len(primary_indices),
                    defined_count=len(primary_indices) if rv is not None else 0,
                    ok_count=ok_primary_ref if rv is not None else 0,
                    minimum=minimum,
                )

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
            "last_on_last",
            "head_last_on_mean",
            "DeltaG_feat|last = G_mean(last_on_last) - G_mean(head_last_on_mean)",
        ),
        "feature_gain_fixed_head_mean": _build_pair_contrast(
            "head_mean_on_last",
            "mean_on_mean",
            "DeltaG_feat|mean = G_mean(head_mean_on_last) - G_mean(mean_on_mean)",
        ),
        "head_gain_fixed_feature_last": _build_pair_contrast(
            "last_on_last",
            "head_mean_on_last",
            "DeltaG_head|last = G_mean(last_on_last) - G_mean(head_mean_on_last)",
        ),
        "head_gain_fixed_feature_mean": _build_pair_contrast(
            "head_last_on_mean",
            "mean_on_mean",
            "DeltaG_head|mean = G_mean(head_last_on_mean) - G_mean(mean_on_mean)",
        ),
        "interaction": _build_interaction_contrast(),
        "total_matched_reduction": _build_pair_contrast(
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

    return arms_out, contrasts, collapse


def _verify_crossed_block_ref(
    *,
    root,
    protocol,
    ref,
    plan,
    gate_ref,
    model,
    depth,
    arm,
    head_pooling,
    feature_pooling,
    target_lang,
    expected_feature_ref,
    expected_image_ids_sha256,
    expected_n_images,
    expected_n_cats,
):
    """Authenticate one resumed E30 crossed prediction block and return `(ref, body, arrays)`."""
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
    check(res_body.get("experiment") == "E30", "Resumed crossed block experiment mismatch")
    check(res_body.get("confirm_excluded") is True, "Resumed crossed block missing confirm_excluded=True")
    check(int(res_body.get("midpoint_depth", -1)) == depth, "Resumed crossed block midpoint_depth mismatch")
    check(res_body.get("condition") == E30_CONDITION, "Resumed crossed block condition mismatch")
    check(res_body.get("split") == E30_SPLIT, "Resumed crossed block split mismatch")
    check(res_body.get("arm") == arm, "Resumed crossed block arm mismatch")
    check(res_body.get("head_pooling") == head_pooling, "Resumed crossed block head_pooling mismatch")
    check(res_body.get("feature_pooling") == feature_pooling, "Resumed crossed block feature_pooling mismatch")
    check(res_body.get("target_language") == target_lang, "Resumed crossed block target_language mismatch")
    check(list(res_body.get("source_languages", ())) == list(LANGUAGES), "Resumed crossed block sources mismatch")
    check(plain(res_body.get("gate_ref")) == plain(gate_ref), "Resumed crossed block gate_ref mismatch")
    check(
        plain(res_body.get("upstream_feature_ref")) == plain(expected_feature_ref),
        "Resumed crossed block upstream_feature_ref mismatch",
    )
    check(
        res_body.get("image_ids_sha256") == expected_image_ids_sha256,
        "Resumed crossed block image_ids_sha256 mismatch",
    )
    arrays = _load_crossed_block_arrays(root, res_ref, expected_n_images, expected_n_cats)
    return res_ref, res_body, arrays


@contract
def run_e30(
    root,
    protocol,
    plan_ref,
    gate_ref,
    model,
    producer,
    *,
    resume_crossed_refs=None,
    resume_analysis_ref=None,
    run_id=None,
    progress=None,
):
    """Execute E30 CPU head x pooling swap at fixed midpoint for one model."""
    check(model in MODELS, f"Unknown model selector: {model!r}")
    require_open_split(E30_SPLIT)
    resolved_root = Path(root).resolve()

    resume_crossed_map = dict(resume_crossed_refs or {})
    unknown_keys = set(resume_crossed_map.keys()) - _valid_crossed_resume_keys()
    check(not unknown_keys, f"Unknown resume_crossed_refs keys: {sorted(unknown_keys)}")

    import numpy as np
    from src.stage4.metrics import bootstrap_counts, paired_image_alignment, targets_for_prediction
    from src.stage4.plan import load_bootstrap
    from src.stage4.predictions import Prediction, load_prediction_block

    total_units = 12  # init + provenance/self-check + 8 crossed blocks + evaluate + publish

    # 1. Load phase2_plan, explicitly bind Phase1Binding, load & enforce OPEN E29 gate.
    t0 = time.monotonic()
    plan = load_phase2_plan(resolved_root, protocol, plan_ref)
    check(plan["confirm_excluded"] is True, "phase2_plan must enforce confirm_excluded=True")
    plan_protocol = plain(plan.protocol)
    binding = bind_phase1(resolved_root, plan_protocol, plain(plan["stage4_release"]))
    check(binding.lineage_identity == plan["lineage_identity"], "Phase 1 lineage_identity mismatch with phase2_plan")
    check(binding.dataset_fingerprint == plan["dataset_fingerprint"], "Dataset fingerprint mismatch with phase2_plan")
    check(binding.bootstrap_sha256 == plan["bootstrap"]["sha256"], "Bootstrap SHA-256 mismatch with phase2_plan")
    check(list(binding.bootstrap_shape) == list(plan["bootstrap"]["shape"]), "Bootstrap shape mismatch with phase2_plan")

    verified_gate_ref, gate_body, _ = load_e29_gate(
        resolved_root,
        plan_protocol,
        gate_ref,
        plan=plan,
        expected_model=model,
    )
    e30_gate_decision = check_e30_gate_open(gate_body, model)

    depth = midpoint_depth(binding, model)
    check(depth == int(plan["midpoints"][model]), f"Midpoint depth mismatch for {model}")

    e29_blocks_all = e29_screen_blocks(binding, model)
    matched_meta = e29_blocks_all[E30_CONDITION]
    check(
        plain(matched_meta) == plain(plan["e29_blocks"][model][E30_CONDITION]),
        "Resolved Stage 4 matched midpoint blocks differ from phase2_plan",
    )
    matched_block_refs = {
        "last_on_last": {lang: plain(matched_meta["last_caption"][lang]["prediction_ref"]) for lang in LANGUAGES},
        "mean_on_mean": {lang: plain(matched_meta["caption_mean"][lang]["prediction_ref"]) for lang in LANGUAGES},
    }

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

    _emit_progress(
        progress,
        "e30_plan_and_gate_verified",
        unit="init",
        completed=1,
        total=total_units,
        read_hash_s=time.monotonic() - t0,
        model=model,
        midpoint_depth=depth,
        e30_gate=e30_gate_decision,
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
        check(res_body.get("experiment") == "E30", "Resume analysis artifact is not E30")
        check(res_body.get("confirm_excluded") is True, "Resume analysis missing confirm_excluded=True")
        check(int(res_body.get("midpoint_depth", -1)) == depth, "Resume analysis midpoint_depth mismatch")
        check(res_body.get("condition") == E30_CONDITION, "Resume analysis condition mismatch")
        check(plain(res_body.get("gate_ref")) == plain(verified_gate_ref), "Resume analysis gate_ref mismatch")
        check(plain(res_body.get("bootstrap")) == plain(plan["bootstrap"]), "Resume analysis bootstrap mismatch")
        check(
            plain(res_body.get("matched_block_refs")) == matched_block_refs,
            "Resume analysis matched_block_refs mismatch",
        )
        crossed_refs_in_body = plain(res_body.get("crossed_block_refs", {}))
        check(set(crossed_refs_in_body.keys()) == set(CROSSED_ARMS), "Resume analysis missing crossed_block_refs arms")
        for arm in CROSSED_ARMS:
            check(
                set(crossed_refs_in_body[arm].keys()) == set(LANGUAGES),
                f"Resume analysis crossed_block_refs[{arm}] missing languages",
            )
            for lang in LANGUAGES:
                cref = crossed_refs_in_body[arm][lang]
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
                rkey = f"{arm}:{lang}"
                if rkey in resume_crossed_map and resume_crossed_map[rkey] is not None:
                    explicit_cref = normalize_ref(resolved_root, resume_crossed_map[rkey])
                    check(
                        plain(cref) == plain(explicit_cref),
                        f"Explicit resume_crossed_refs[{rkey!r}] differs from resumed analysis block",
                    )
        _emit_progress(
            progress,
            "e30_resumed",
            unit="resume",
            completed=total_units,
            total=total_units,
            read_hash_s=time.monotonic() - t_res,
            model=model,
            analysis_ref=plain(res_ref),
            crossed_block_refs=crossed_refs_in_body,
        )
        return res_ref, res_body, crossed_refs_in_body

    # 2. Load 8 Stage 2 SCREEN feature groups, 8 Stage 3 frozen heads, and 8 Stage 4 matched blocks.
    t_prov = time.monotonic()
    gids_by_pooling = {
        pooling: midpoint_groups(binding, model, condition=E30_CONDITION, pooling=pooling)
        for pooling in POOLINGS
    }
    rows_by_pooling = {pooling: {} for pooling in POOLINGS}
    matrices_by_pooling = {pooling: {} for pooling in POOLINGS}
    source_heads_by_pooling = {pooling: {} for pooling in POOLINGS}
    matched_preds_by_pooling = {pooling: {} for pooling in POOLINGS}

    for pooling in POOLINGS:
        for lang in LANGUAGES:
            rows, matrix = read_verified_midpoint_features(
                binding, model, E30_CONDITION, pooling, lang, E30_SPLIT
            )
            rows_by_pooling[pooling][lang] = rows
            matrices_by_pooling[pooling][lang] = matrix
            source_heads_by_pooling[pooling][lang] = load_frozen_source_head(
                binding, model, E30_CONDITION, pooling, lang
            )

            entry = matched_meta[pooling][lang]
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

    provenance_checks = verify_e30_provenance(
        binding, model, depth, gids_by_pooling, rows_by_pooling, matrices_by_pooling
    )
    matched_self_checks = _run_en_matched_self_checks(
        model=model,
        rows_by_pooling=rows_by_pooling,
        matrices_by_pooling=matrices_by_pooling,
        source_heads_by_pooling=source_heads_by_pooling,
        matched_preds_by_pooling=matched_preds_by_pooling,
    )
    _emit_progress(
        progress,
        "e30_provenance_and_self_check_passed",
        unit="provenance_and_self_check",
        completed=2,
        total=total_units,
        read_hash_s=time.monotonic() - t_prov,
        model=model,
        dimension=provenance_checks["dimension"],
        en_self_check_max_abs_diff=0.0,
    )

    # 3. Score (or resume via explicit refs) ONLY the 2 crossed arms (8 blocks / 32 routes).
    base_run_id = run_id or new_run_id()
    fit_groups = binding.stage4_plan.fit["groups"]
    ordinary = binding.stage4_plan["ordinary"]

    crossed_block_refs = {arm: {} for arm in CROSSED_ARMS}
    crossed_preds_by_arm = {arm: {} for arm in CROSSED_ARMS}
    completed_units = 2

    for arm in CROSSED_ARMS:
        head_pooling, feature_pooling = CROSSED_ARM_SPECS[arm]
        for target_lang in LANGUAGES:
            completed_units += 1
            rkey = f"{arm}:{target_lang}"
            rows_target = rows_by_pooling[feature_pooling][target_lang]
            matrix_target = matrices_by_pooling[feature_pooling][target_lang]
            target_gid = gids_by_pooling[feature_pooling][target_lang]
            target_feature_ref = plain(fit_groups[target_gid]["releases"][E30_SPLIT])
            image_ids_sha256 = provenance_checks["per_language"][target_lang]["image_ids_sha256"]

            if rkey in resume_crossed_map and resume_crossed_map[rkey] is not None:
                t_res = time.monotonic()
                block_ref, _, (logits_arr, estimable_arr, prevalence_arr) = _verify_crossed_block_ref(
                    root=resolved_root,
                    protocol=plan_protocol,
                    ref=resume_crossed_map[rkey],
                    plan=plan,
                    gate_ref=verified_gate_ref,
                    model=model,
                    depth=depth,
                    arm=arm,
                    head_pooling=head_pooling,
                    feature_pooling=feature_pooling,
                    target_lang=target_lang,
                    expected_feature_ref=target_feature_ref,
                    expected_image_ids_sha256=image_ids_sha256,
                    expected_n_images=len(rows_target),
                    expected_n_cats=n_cats,
                )
                crossed_block_refs[arm][target_lang] = plain(block_ref)
                crossed_preds_by_arm[arm][target_lang] = {
                    src_lang: Prediction(
                        reference=plain(block_ref),
                        route={
                            "route_id": f"{model}:e30:{arm}:{src_lang}->{target_lang}",
                            "source_language": src_lang,
                            "target_language": target_lang,
                            "arm": arm,
                        },
                        split=E30_SPLIT,
                        rows=tuple(rows_target),
                        logits=logits_arr[s_idx],
                        estimable=estimable_arr[s_idx],
                        prevalence=prevalence_arr[s_idx],
                    )
                    for s_idx, src_lang in enumerate(LANGUAGES)
                }
                _emit_progress(
                    progress,
                    "e30_crossed_block_resumed",
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
            upstream = [plain(plan.reference), plain(verified_gate_ref), target_feature_ref]

            for s_idx, src_lang in enumerate(LANGUAGES):
                src_gid = gids_by_pooling[head_pooling][src_lang]
                coef, intercept, estimable, prevalence, stats = source_heads_by_pooling[head_pooling][src_lang]
                logits_s = _score_matrix(matrix_target, coef, intercept, stats, estimable)
                logits_list.append(logits_s)
                valid_list.append(estimable)
                prev_list.append(prevalence)

                src_sel_ref = plain(ordinary[src_gid]["selection"])
                src_cand_ref = plain(ordinary[src_gid]["candidate"])
                src_stats_ref = plain(stats["reference"])
                upstream.extend([src_sel_ref, src_cand_ref, src_stats_ref])

                routes_meta.append({
                    "route_id": f"{model}:e30:{arm}:{src_lang}->{target_lang}",
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
                "experiment": "E30",
                "plan": plain(plan.reference),
                "plan_identity": plan["plan_identity"],
                "gate_ref": plain(verified_gate_ref),
                "model": model,
                "midpoint_depth": depth,
                "condition": E30_CONDITION,
                "split": E30_SPLIT,
                "arm": arm,
                "head_pooling": head_pooling,
                "feature_pooling": feature_pooling,
                "target_language": target_lang,
                "source_languages": list(LANGUAGES),
                "n_images": len(rows_target),
                "n_categories": n_cats,
                "category_order": list(categories),
                "image_ids_sha256": image_ids_sha256,
                "provenance_checks": provenance_checks,
                "matched_self_check": matched_self_checks[head_pooling],
                "routes": routes_meta,
                "upstream_feature_ref": target_feature_ref,
                "confirm_excluded": True,
            }

            t_write = time.monotonic()
            block_ref = publish_phase2(
                resolved_root,
                plan_protocol,
                plan["dataset_fingerprint"],
                f"{E30_PREDICTIONS_NAMESPACE}/{model}",
                KIND_PREDICTION,
                block_body,
                producer,
                arrays={
                    "logits": logits_arr,
                    "estimable": estimable_arr,
                    "prevalence": prevalence_arr,
                },
                upstream=upstream,
                run_id=f"{base_run_id}_{arm}_{target_lang}",
            )
            write_s = time.monotonic() - t_write

            crossed_block_refs[arm][target_lang] = plain(block_ref)
            crossed_preds_by_arm[arm][target_lang] = {
                src_lang: Prediction(
                    reference=plain(block_ref),
                    route=routes_meta[s_idx],
                    split=E30_SPLIT,
                    rows=tuple(rows_target),
                    logits=logits_arr[s_idx],
                    estimable=estimable_arr[s_idx],
                    prevalence=prevalence_arr[s_idx],
                )
                for s_idx, src_lang in enumerate(LANGUAGES)
            }
            _emit_progress(
                progress,
                "e30_crossed_block_published",
                unit=f"crossed:{rkey}",
                completed=completed_units,
                total=total_units,
                compute_s=compute_s,
                write_s=write_s,
                model=model,
                block_ref=plain(block_ref),
            )

    # 4. Evaluate all 4 arms on the frozen (2000, 360) bootstrap.
    t_eval = time.monotonic()
    preds_by_arm = {
        "last_on_last": matched_preds_by_pooling["last_caption"],
        "mean_on_mean": matched_preds_by_pooling["caption_mean"],
        "head_last_on_mean": crossed_preds_by_arm["head_last_on_mean"],
        "head_mean_on_last": crossed_preds_by_arm["head_mean_on_last"],
    }

    all_preds = []
    pred_keys = []
    for arm in ALL_ARMS:
        for s in LANGUAGES:
            for t in LANGUAGES:
                all_preds.append(preds_by_arm[arm][t][s])
                pred_keys.append((arm, s, t))

    screen_ids = tuple(binding.stage4_plan["screen_ids"])
    ids, aligned_tuple = paired_image_alignment(binding.stage4_plan, *all_preds)
    check(ids == screen_ids, "Aligned SCREEN IDs mismatch plan screen_ids")
    aligned_by_key = {key: aligned for key, aligned in zip(pred_keys, aligned_tuple)}

    y_raw, known_raw = targets_for_prediction(binding.stage4_plan, all_preds[0])
    aligned_0 = aligned_tuple[0]
    y = y_raw[aligned_0]
    known = known_raw[aligned_0]

    draws = load_bootstrap(binding.stage4_plan)
    check(
        hashlib.sha256(draws.tobytes()).hexdigest() == plan["bootstrap"]["sha256"],
        "Loaded bootstrap SHA-256 mismatch with phase2_plan",
    )
    counts = bootstrap_counts(draws, len(screen_ids))
    total_draws = int(draws.shape[0])

    arms_results, contrasts, collapse = _evaluate_all_arms_and_contrasts(
        preds_by_arm=preds_by_arm,
        aligned_by_key=aligned_by_key,
        y=y,
        known=known,
        counts=counts,
        categories=categories,
        primary_indices=primary_indices,
        total_draws=total_draws,
        minimum=screen_minimum,
    )
    compute_eval_s = time.monotonic() - t_eval
    _emit_progress(
        progress,
        "e30_contrasts_computed",
        unit="evaluate_4arms",
        completed=11,
        total=total_units,
        compute_s=compute_eval_s,
        model=model,
        collapse_flagged=collapse["flagged"],
    )

    # 5. Publish E30 analysis block under phase2/results/e30/<model>/<run_id>/.
    t_pub = time.monotonic()
    analysis_body = {
        "schema": SCHEMA_ANALYSIS,
        "experiment": "E30",
        "plan": plain(plan.reference),
        "plan_identity": plan["plan_identity"],
        "gate_ref": plain(verified_gate_ref),
        "model": model,
        "midpoint_depth": depth,
        "condition": E30_CONDITION,
        "bootstrap": plain(plan["bootstrap"]),
        "matched_block_refs": matched_block_refs,
        "crossed_block_refs": crossed_block_refs,
        "provenance_checks": provenance_checks,
        "matched_self_checks": matched_self_checks,
        "arms": arms_results,
        "contrasts": contrasts,
        "off_distribution_collapse": collapse,
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
    analysis_upstream = [plain(plan.reference), plain(verified_gate_ref)]
    for arm in MATCHED_ARMS:
        for lang in LANGUAGES:
            analysis_upstream.append(matched_block_refs[arm][lang])
    for arm in CROSSED_ARMS:
        for lang in LANGUAGES:
            analysis_upstream.append(crossed_block_refs[arm][lang])

    analysis_ref = publish_phase2(
        resolved_root,
        plan_protocol,
        plan["dataset_fingerprint"],
        f"{E30_RESULTS_NAMESPACE}/{model}",
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
        "e30_published",
        unit="publish",
        completed=total_units,
        total=total_units,
        write_s=write_pub_s,
        model=model,
        analysis_ref=plain(analysis_ref),
        crossed_block_refs=crossed_block_refs,
        collapse_flagged=collapse["flagged"],
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
    parser.add_argument("--plan", required=True, help="Explicit phase2_plan {'path', 'sha256'} JSON or path")
    parser.add_argument("--e29-gate", required=True, help="Explicit phase2_e29_gate {'path', 'sha256'} JSON or path")
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument(
        "--storage-scope",
        default="persistent-drive",
        choices=("persistent-drive", "local-development"),
    )
    parser.add_argument(
        "--resume-crossed-refs",
        help="Optional JSON object (or path) mapping '{arm}:{target_lang}' -> {'path', 'sha256'}",
    )
    parser.add_argument(
        "--resume-analysis-ref",
        help="Optional explicit E30 analysis {'path', 'sha256'} JSON or path",
    )
    parser.add_argument("--run-id", help="Optional explicit run_id")
    parser.add_argument("--output-ref", type=Path, help="Optional JSON file path for emitted references")
    args = parser.parse_args(argv)

    root = phase1_root(args.phase1_root)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    from src.phase2.interfaces import current_producer

    producer = current_producer(args.storage_scope)
    _emit_cli({
        "event": "e30_start",
        "model": args.model,
        "code_commit": producer["code_commit"],
        "storage_scope": producer["storage_scope"],
    })
    analysis_ref, analysis_body, crossed_block_refs = run_e30(
        root,
        protocol,
        _parse_ref_arg(args.plan),
        _parse_ref_arg(args.e29_gate),
        args.model,
        producer,
        resume_crossed_refs=_parse_mapping_arg(args.resume_crossed_refs),
        resume_analysis_ref=_parse_ref_arg(args.resume_analysis_ref),
        run_id=args.run_id,
        progress=_emit_cli,
    )
    auroc_contrasts = analysis_body["contrasts"]
    summary = {
        "event": "e30_complete",
        "status": "passed",
        "model": args.model,
        "analysis_ref": plain(analysis_ref),
        "crossed_block_refs": plain(crossed_block_refs),
        "feature_gain_fixed_head_last_auroc": auroc_contrasts["feature_gain_fixed_head_last"]["metrics"]["auroc"]["gap_reduction"]["primary_macro"]["value"],
        "feature_gain_fixed_head_mean_auroc": auroc_contrasts["feature_gain_fixed_head_mean"]["metrics"]["auroc"]["gap_reduction"]["primary_macro"]["value"],
        "interaction_auroc": auroc_contrasts["interaction"]["metrics"]["auroc"]["gap_reduction"]["primary_macro"]["value"],
        "off_distribution_collapse": plain(analysis_body["off_distribution_collapse"]),
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
        _emit_cli({"status": "blocked", "blockers": [str(exc)], "stage3_e30_complete": False})
        raise SystemExit(2)
