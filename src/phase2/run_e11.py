"""Phase 2 Stage 3 CPU worker for E11: FIT-only mean/scale covariate adjustment on caption_mean.

Implements phase2.md §1.3-1.5, §3.1-3.2, and §3.4:
1. Loads `phase2_plan`, explicitly binds `Phase1Binding` via `bind_phase1`, loads the
   model's `phase2_e29_gate` via `load_e29_gate`, and refuses to run unless
   `decisions["E11"]["decision"] == "OPEN"`.
2. Resolves `e11_fit_statistics_refs`, verifies all 4 `caption_mean` Stage 3 FIT
   `statistics` artifacts use the exact same 2,160 paired FIT `image_id`s with the
   `1e-12` variance floor (zero FIT feature matrix reloads), and asserts `en->en`
   `caption_mean` SCREEN diagonal reproduction (`atol=0, rtol=0`) against Stage 4.
3. Applies FIT-only `mean_shift` (`h_t - mu_t + mu_s`) and `mean_scale`
   (`mu_s + sigma_s * (h_t - mu_t) / sigma_t`) to verified `caption_mean` SCREEN
   features, scores through the unchanged source scaler (`src.stage3.engine.transform`)
   and frozen source head, and publishes 8 adapted blocks (24 `s != t` routes/model)
   under `phase2/predictions/e11_adapted/<model>/<run_id>/` with
   `"target_fit_input_adapted"` supervision labels.
4. Evaluates `identity` (reused 4 Stage 4 `caption_mean` SCREEN blocks), `mean_shift`,
   and `mean_scale` against unchanged `t->t` diagonals on the frozen `(2000, 360)`
   bootstrap via `src.stage4.metrics` (using `protocol["category_order"]` and
   `int(protocol["minimum_positive_and_negative"]["SCREEN"])`), hard-failing if
   `mean_shift` per-category AUROC/AP differs from `identity` by `> 1e-12` or logit
   offset std `> 1e-10`, and publishes `phase2_analysis_block` under
   `phase2/results/e11/<model>/<run_id>/`.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from src.common.contracts import ContractError, LANGUAGES, fingerprint
from src.common.storage import new_run_id, phase1_root, within
from src.phase2.gates import STAGE5_DECISION_RANGES, load_e29_gate
from src.phase2.interfaces import (
    KIND_ANALYSIS,
    KIND_PREDICTION,
    MODELS,
    REPO,
    SCHEMA_ANALYSIS,
    SCHEMA_PREDICTION,
    bind_phase1,
    check,
    contract,
    current_producer,
    e11_fit_statistics_refs,
    frozen,
    load_frozen_source_head,
    midpoint_depth,
    normalize_ref,
    plain,
    publish_phase2,
    read_verified_midpoint_features,
    require_open_split,
    verify_resume_ref,
)
from src.phase2.plan import load_phase2_plan
from src.stage3.engine import transform
from src.stage3.readouts import expected_rows, load_statistics
from src.stage4.metrics import (
    METRICS,
    binary_metrics,
    bootstrap_binary_metrics,
    bootstrap_counts,
    bootstrap_interval,
    fixed_macro,
    paired_image_alignment,
    probabilities,
    support_status,
    targets_for_prediction,
)
from src.stage4.plan import load_bootstrap
from src.stage4.predictions import Prediction, load_prediction_block

E11_CONDITION = "image_caption"
E11_POOLING = "caption_mean"
E11_SPLIT = "SCREEN"
E11_VARIANTS = ("mean_shift", "mean_scale")
E11_ADAPTED_NAMESPACE = "phase2/predictions/e11_adapted"
E11_RESULTS_NAMESPACE = "phase2/results/e11"

EXPECTED_FIT_IMAGES = 2160
VARIANCE_FLOOR = 1e-12
MEAN_SHIFT_RANK_TOL = 1e-12
MEAN_SHIFT_LOGIT_OFFSET_STD_TOL = 1e-10

LOSS_METRICS = frozenset({"brier", "nll"})
SUPERVISION_DIAGONAL = "within_context_source_fit"
SUPERVISION_IDENTITY = "source_only_transfer"
SUPERVISION_ADAPTED = "target_fit_input_adapted"

CAPTION_ROBUSTNESS_NOTE = (
    "E11 informs draw-0 caption_mean covariate statistics only; "
    "claims about robustness across alternate captions wait for Stage 4 E07."
)


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
    """Positive always indicates `alt` improves over `base`."""
    if base_val is None or alt_val is None:
        return None
    if metric in LOSS_METRICS:
        return float(base_val - alt_val)
    return float(alt_val - base_val)


def _improvement_draws(metric, base_draws, alt_draws):
    """Positive always indicates `alt` improves over `base`."""
    if metric in LOSS_METRICS:
        return base_draws - alt_draws
    return alt_draws - base_draws


def _interval_record(draw_values):
    interval = bootstrap_interval(draw_values)
    if interval["valid_draws"] == 0:
        interval["reason"] = "no_valid_bootstrap_draws"
    return interval


def _category_status_and_reason(value, pos_n, neg_n, minimum, unestimable):
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


def _mean_across_routes(route_summaries, route_draws_list, *, label_count):
    """Equal-weight mean across routes within every bootstrap draw."""
    values = [r["value"] for r in route_summaries]
    statuses = [r["status"] for r in route_summaries]
    if any(v is None for v in values):
        point = None
        status = "undefined"
        reason = "undefined_route_component"
    else:
        point = float(np.mean(values))
        if all(s == "ok" for s in statuses):
            status = "ok"
            reason = None
        else:
            status = "descriptive_low_support"
            reason = "route_component_low_support"
    stacked = np.stack(route_draws_list, axis=0)
    mean_draws = np.mean(stacked, axis=0)
    return (
        {
            "value": point,
            "interval": _interval_record(mean_draws),
            "status": status,
            "reason": reason,
            "route_count": int(label_count),
        },
        mean_draws,
    )


def _mean_per_category_across_routes(categories, per_route_cat_tables, per_route_cat_draws):
    """Equal-weight 25-category mean across the 12 directed routes within every draw."""
    out = {}
    for col, cat in enumerate(categories):
        vals = [tbl[cat]["value"] for tbl in per_route_cat_tables]
        statuses = [tbl[cat]["status"] for tbl in per_route_cat_tables]
        if any(v is None for v in vals):
            val = None
            status = "undefined"
            reason = "undefined_route_component"
        else:
            val = float(np.mean(vals))
            if all(s == "ok" for s in statuses):
                status = "ok"
                reason = None
            else:
                status = "descriptive_low_support"
                reason = "route_component_low_support"
        mean_draws = np.mean(np.stack([d[col] for d in per_route_cat_draws], axis=0), axis=0)
        out[cat] = {
            "value": val,
            "interval": _interval_record(mean_draws),
            "status": status,
            "reason": reason,
        }
    return out


def _evaluate_arm(logits, estimable, aligned, y, known, counts, categories, total_draws):
    """Evaluate one route's 25-category point estimates, bootstrap draws, and support."""
    n_cats = len(categories)
    probs = probabilities(logits[aligned])
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


def _apply_fit_transform(matrix_f64, stats_source, stats_target, variant):
    """Apply FIT-only mean_shift or mean_scale transform in float64."""
    mu_s = np.asarray(stats_source["mean"], dtype=np.float64)
    sigma_s = np.asarray(stats_source["scale"], dtype=np.float64)
    mu_t = np.asarray(stats_target["mean"], dtype=np.float64)
    sigma_t = np.asarray(stats_target["scale"], dtype=np.float64)
    if variant == "mean_shift":
        adapted = matrix_f64 - mu_t + mu_s
    elif variant == "mean_scale":
        adapted = mu_s + sigma_s * ((matrix_f64 - mu_t) / sigma_t)
    else:
        raise ContractError(f"Unknown E11 variant: {variant!r}")
    check(np.isfinite(adapted).all(), f"Nonfinite adapted features for variant {variant!r}")
    return adapted


def _score_through_source_head(features_f64, coef, intercept, estimable, stats_source):
    """Score features through the unchanged Stage 3 source scaler and frozen linear head."""
    standardized = transform(features_f64, stats_source)
    logits = np.asarray(standardized @ coef.T, dtype=np.float64) + intercept
    check(np.isfinite(logits).all(), "Nonfinite E11 scored logits")
    est = np.asarray(estimable, dtype=bool)
    check(not np.any(logits * (~est[None, :])), "Unestimable E11 logit padding mismatch")
    return logits


def _verify_fit_statistics_and_heads(binding, model, e11_refs):
    """Verify all 4 caption_mean FIT statistics share 2,160 paired FIT IDs (0 FIT matrix reloads)."""
    fit = binding.stage4_plan.fit
    store = binding.stage4_plan.store
    depth = midpoint_depth(binding, model)
    expected_site = f"decoder:{depth}:{E11_POOLING}"

    dimensions = set()
    canonical_fit_ids = None
    stats_by_lang = {}
    heads_by_lang = {}

    for lang in LANGUAGES:
        rec = e11_refs[lang]
        gid = rec["group_id"]
        check(
            rec["site"] == expected_site,
            f"Unexpected E11 site for {model}/{lang}: expected {expected_site!r}, got {rec['site']!r}",
        )
        dimensions.add(int(rec["dimension"]))

        fit_rows = expected_rows(fit, gid, "FIT", store)
        check(
            len(fit_rows) == EXPECTED_FIT_IMAGES,
            f"Expected {EXPECTED_FIT_IMAGES} paired FIT rows for {model}/{lang}, got {len(fit_rows)}",
        )
        fit_ids = []
        for row in fit_rows:
            check(
                row.get("split") == "FIT" and row.get("language") == lang,
                f"Invalid FIT row metadata for {model}/{lang}: {row!r}",
            )
            fit_ids.append(row["image_id"])
        check(len(set(fit_ids)) == len(fit_ids), f"Duplicate FIT image_id in {model}/{lang}")

        if canonical_fit_ids is None:
            canonical_fit_ids = fit_ids
        else:
            check(
                fit_ids == canonical_fit_ids,
                f"Unpaired FIT image_id ordering between en and {lang} for {model}",
            )

        # Load verified FIT statistics artifact without reading FIT .npy matrices.
        stats = load_statistics(fit, rec["statistics_ref"], gid, fit_rows, store)
        check(stats["center"] is True, f"Expected centered dense statistics for {model}/{lang}")
        mean_arr = np.asarray(stats["mean"], dtype=np.float64)
        var_arr = np.asarray(stats["variance"], dtype=np.float64)
        scale_arr = np.asarray(stats["scale"], dtype=np.float64)
        check(
            np.isfinite(mean_arr).all()
            and np.isfinite(var_arr).all()
            and np.isfinite(scale_arr).all(),
            f"Nonfinite FIT statistics for {model}/{lang}",
        )
        check(np.all(var_arr >= 0.0) and np.all(scale_arr > 0.0), f"Invalid FIT variance/scale for {model}/{lang}")
        expected_scale = np.where(var_arr <= VARIANCE_FLOOR, 1.0, np.sqrt(var_arr))
        check(
            np.array_equal(scale_arr, expected_scale),
            f"FIT variance floor ({VARIANCE_FLOOR}) mismatch for {model}/{lang}",
        )

        # Load frozen source head (without FIT matrices) and verify statistics identity matches.
        coef, intercept, estimable, prevalence, head_stats = load_frozen_source_head(
            binding, model, E11_CONDITION, E11_POOLING, lang
        )
        check(
            plain(head_stats["reference"]) == plain(rec["statistics_ref"]),
            f"Frozen head statistics ref mismatch for {model}/{lang}",
        )
        check(
            np.array_equal(head_stats["mean"], stats["mean"])
            and np.array_equal(head_stats["scale"], stats["scale"]),
            f"Frozen head statistics payload mismatch for {model}/{lang}",
        )

        stats_by_lang[lang] = stats
        heads_by_lang[lang] = {
            "coef": coef,
            "intercept": intercept,
            "estimable": estimable,
            "prevalence": prevalence,
            "stats": head_stats,
        }

    check(len(dimensions) == 1, f"Inconsistent feature dimensions across languages for {model}: {dimensions}")
    dimension = next(iter(dimensions))

    fit_pairing_check = {
        "paired_fit_images": int(len(canonical_fit_ids)),
        "fit_image_ids_sha256": fingerprint(canonical_fit_ids),
        "fit_rows_match_across_languages": True,
        "variance_floor": VARIANCE_FLOOR,
        "fit_matrices_reloaded": 0,
    }
    return dimension, fit_pairing_check, stats_by_lang, heads_by_lang


def _verify_en_identity_self_check(binding, model, e11_refs, stage4_blocks, heads_by_lang):
    """Run mandatory in-memory en->en caption_mean SCREEN diagonal reproduction check (atol=0, rtol=0)."""
    rows_en, matrix_en = read_verified_midpoint_features(
        binding, model, E11_CONDITION, E11_POOLING, "en", E11_SPLIT
    )
    head_en = heads_by_lang["en"]
    computed_en = _score_through_source_head(
        np.asarray(matrix_en, dtype=np.float64),
        head_en["coef"],
        head_en["intercept"],
        head_en["estimable"],
        head_en["stats"],
    )
    en_route_id = e11_refs["en"]["screen_routes"]["en"]
    accepted_en = stage4_blocks["en"][en_route_id]
    check(
        plain(rows_en) == plain(accepted_en.rows),
        f"SCREEN en->en caption_mean row alignment mismatch for {model}",
    )
    check(
        computed_en.shape == accepted_en.logits.shape,
        f"SCREEN en->en caption_mean logit shape mismatch for {model}",
    )
    check(
        np.array_equal(head_en["estimable"], accepted_en.estimable),
        f"SCREEN en->en caption_mean estimable mask mismatch for {model}",
    )
    max_abs_diff = float(np.max(np.abs(computed_en - accepted_en.logits)))
    check(
        np.allclose(computed_en, accepted_en.logits, atol=0.0, rtol=0.0) and max_abs_diff == 0.0,
        f"SCREEN en->en caption_mean diagonal self-check failed for {model}: max_abs_diff={max_abs_diff}",
    )
    self_check = {
        "route": "en->en",
        "passed": True,
        "max_abs_diff": 0.0,
        "atol": 0.0,
        "rtol": 0.0,
        "row_count": int(len(rows_en)),
        "category_count": int(computed_en.shape[1]),
    }
    return self_check, rows_en, matrix_en


def _load_adapted_block_arrays(root, ref, expected_sources, expected_n_images, expected_n_cats):
    """Load and validate logits, estimable, prevalence arrays from an E11 adapted prediction block."""
    marker_dir = within(root, ref["path"]).parent
    logits = np.load(marker_dir / "logits.npy", allow_pickle=False)
    estimable = np.load(marker_dir / "estimable.npy", allow_pickle=False)
    prevalence = np.load(marker_dir / "prevalence.npy", allow_pickle=False)
    n_src = len(expected_sources)
    check(
        logits.shape == (n_src, expected_n_images, expected_n_cats) and logits.dtype == np.float64,
        f"Invalid E11 adapted logits shape/dtype: {logits.shape}/{logits.dtype}",
    )
    check(np.isfinite(logits).all(), "E11 adapted logits must be finite float64")
    check(
        estimable.shape == (n_src, expected_n_cats) and estimable.dtype == bool,
        f"Invalid E11 adapted estimable shape/dtype: {estimable.shape}/{estimable.dtype}",
    )
    check(
        prevalence.shape == (n_src, expected_n_cats) and prevalence.dtype == np.float64,
        f"Invalid E11 adapted prevalence shape/dtype: {prevalence.shape}/{prevalence.dtype}",
    )
    check(
        not np.any(logits * (~estimable[:, None, :])),
        "E11 adapted logits must be zero on unestimable columns",
    )
    return logits, estimable, prevalence


def _verify_resume_adapted_block(
    root,
    protocol,
    plan,
    gate_ref,
    model,
    variant,
    target_lang,
    sorted_sources,
    e11_refs,
    image_ids_sha256,
    fit_pairing_check,
    n_images,
    n_cats,
    reference,
):
    """Verify an explicit E11 adapted prediction block resume reference."""
    res_ref, res_body, _ = verify_resume_ref(
        root,
        protocol,
        reference,
        KIND_PREDICTION,
        expected_schema=SCHEMA_PREDICTION,
        expected_dataset=plan["dataset_fingerprint"],
        expected_plan_identity=plan["plan_identity"],
        expected_owner=model,
    )
    check(res_body.get("experiment") == "E11", "Resume adapted block experiment mismatch")
    check(plain(res_body.get("gate_ref")) == plain(gate_ref), "Resume adapted block gate_ref mismatch")
    check(res_body.get("condition") == E11_CONDITION, "Resume adapted block condition mismatch")
    check(res_body.get("pooling") == E11_POOLING, "Resume adapted block pooling mismatch")
    check(res_body.get("split") == E11_SPLIT, "Resume adapted block split mismatch")
    check(res_body.get("variant") == variant, "Resume adapted block variant mismatch")
    check(res_body.get("target_language") == target_lang, "Resume adapted block target_language mismatch")
    check(
        list(res_body.get("source_languages", ())) == sorted_sources,
        "Resume adapted block source_languages mismatch",
    )
    check(
        plain(res_body.get("upstream_feature_ref")) == plain(e11_refs[target_lang]["screen_feature_release_ref"]),
        "Resume adapted block upstream_feature_ref mismatch",
    )
    check(
        res_body.get("image_ids_sha256") == image_ids_sha256,
        "Resume adapted block image_ids_sha256 mismatch",
    )
    check(
        plain(res_body.get("fit_pairing_check")) == plain(fit_pairing_check),
        "Resume adapted block fit_pairing_check mismatch",
    )
    self_check = res_body.get("identity_self_check") or {}
    check(
        self_check.get("passed") is True and float(self_check.get("max_abs_diff", -1.0)) == 0.0,
        "Resume adapted block identity_self_check invalid",
    )
    check(res_body.get("confirm_excluded") is True, "Resume adapted block confirm_excluded must be True")
    routes = res_body.get("routes") or ()
    check(len(routes) == len(sorted_sources), "Resume adapted block route count mismatch")
    for s_idx, (src_lang, route_rec) in enumerate(zip(sorted_sources, routes)):
        check(
            route_rec.get("source_language") == src_lang
            and int(route_rec.get("row_index", -1)) == s_idx
            and route_rec.get("supervision_label") == SUPERVISION_ADAPTED
            and plain(route_rec.get("source_fit_stats_ref")) == plain(e11_refs[src_lang]["statistics_ref"])
            and plain(route_rec.get("target_fit_stats_ref")) == plain(e11_refs[target_lang]["statistics_ref"]),
            f"Resume adapted block route metadata mismatch for {variant}:{src_lang}->{target_lang}",
        )
    logits, estimable, prevalence = _load_adapted_block_arrays(
        root, res_ref, sorted_sources, n_images, n_cats
    )
    return plain(res_ref), logits, estimable, prevalence


def _assert_mean_shift_invariance(
    *,
    identity_arms,
    mean_shift_arms,
    identity_logits_by_route,
    mean_shift_logits_by_route,
    aligned_by_target,
    categories,
):
    """Hard-fail if mean_shift AUROC/AP differs from identity by >1e-12 or logit offset std >1e-10."""
    max_abs_auroc_diff = 0.0
    max_abs_ap_diff = 0.0
    max_logit_offset_std = 0.0
    checked_pairs = 0

    for s, t in _off_diagonal_pairs():
        id_arm = identity_arms[(s, t)]
        sh_arm = mean_shift_arms[(s, t)]
        align_idx = aligned_by_target[t]
        z_id = identity_logits_by_route[(s, t)][align_idx]
        z_sh = mean_shift_logits_by_route[(s, t)][align_idx]
        diff_z = z_sh - z_id

        for col, cat in enumerate(categories):
            unest_id = id_arm["unestimable"][col]
            unest_sh = sh_arm["unestimable"][col]
            check(
                unest_id == unest_sh,
                f"Estimability mismatch between identity and mean_shift on {s}->{t}/{cat}",
            )
            if unest_id:
                continue

            col_std = float(np.std(diff_z[:, col]))
            check(
                np.isfinite(col_std) and col_std <= MEAN_SHIFT_LOGIT_OFFSET_STD_TOL,
                f"mean_shift logit offset not constant on {s}->{t}/{cat}: "
                f"std={col_std:.3e} > {MEAN_SHIFT_LOGIT_OFFSET_STD_TOL:.1e}",
            )
            if col_std > max_logit_offset_std:
                max_logit_offset_std = col_std

            auroc_id = id_arm["points"]["auroc"][col]
            auroc_sh = sh_arm["points"]["auroc"][col]
            check(
                (auroc_id is None) == (auroc_sh is None),
                f"mean_shift AUROC definedness mismatch on {s}->{t}/{cat}",
            )
            if auroc_id is not None:
                auroc_diff = abs(float(auroc_sh) - float(auroc_id))
                check(
                    auroc_diff <= MEAN_SHIFT_RANK_TOL,
                    f"mean_shift AUROC invariance violated on {s}->{t}/{cat}: "
                    f"|diff|={auroc_diff:.3e} > {MEAN_SHIFT_RANK_TOL:.1e}",
                )
                if auroc_diff > max_abs_auroc_diff:
                    max_abs_auroc_diff = auroc_diff

            ap_id = id_arm["points"]["ap"][col]
            ap_sh = sh_arm["points"]["ap"][col]
            check(
                (ap_id is None) == (ap_sh is None),
                f"mean_shift AP definedness mismatch on {s}->{t}/{cat}",
            )
            if ap_id is not None:
                ap_diff = abs(float(ap_sh) - float(ap_id))
                check(
                    ap_diff <= MEAN_SHIFT_RANK_TOL,
                    f"mean_shift AP invariance violated on {s}->{t}/{cat}: "
                    f"|diff|={ap_diff:.3e} > {MEAN_SHIFT_RANK_TOL:.1e}",
                )
                if ap_diff > max_abs_ap_diff:
                    max_abs_ap_diff = ap_diff

            checked_pairs += 1

    return {
        "tolerance": MEAN_SHIFT_RANK_TOL,
        "logit_offset_std_tolerance": MEAN_SHIFT_LOGIT_OFFSET_STD_TOL,
        "mean_shift_max_abs_auroc_diff": float(max_abs_auroc_diff),
        "mean_shift_max_abs_ap_diff": float(max_abs_ap_diff),
        "mean_shift_logit_offset_max_std": float(max_logit_offset_std),
        "checked_route_categories": int(checked_pairs),
        "passed": True,
    }


def _recovered_fraction_summary(g_id_point, g_var_point, g_id_draws, g_var_draws):
    """Compute fraction of residual gap G_res_mean recovered: (G_id - G_var) / G_id."""
    if g_id_point is None or g_var_point is None or g_id_point <= 0.0:
        point = None
        reason = "non_positive_or_undefined_identity_gap"
    else:
        point = float((g_id_point - g_var_point) / g_id_point)
        reason = None
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_draws = np.where(
            np.isfinite(g_id_draws) & np.isfinite(g_var_draws) & (g_id_draws > 0.0),
            (g_id_draws - g_var_draws) / g_id_draws,
            np.nan,
        )
    return {
        "value": point,
        "interval": _interval_record(ratio_draws),
        "reason": reason,
    }


def _evaluate_e11_arms(
    *,
    diag_arms,
    arm_evals,
    categories,
    primary_indices,
    minimum,
):
    """Compute all E11 arm summaries, paired contrasts vs identity, and recovered gap fractions."""
    off_pairs = _off_diagonal_pairs()
    arm_names = ("identity", "mean_shift", "mean_scale")

    arms_output = {
        "target_diagonal_ref": {"supervision_label": SUPERVISION_DIAGONAL, "metrics": {}},
        "identity": {"supervision_label": SUPERVISION_IDENTITY, "metrics": {}},
        "mean_shift": {"supervision_label": SUPERVISION_ADAPTED, "metrics": {}},
        "mean_scale": {"supervision_label": SUPERVISION_ADAPTED, "metrics": {}},
    }
    summaries = {
        "target_diagonal_4_mean": {},
        "twelve_route_mean_gap": {name: {} for name in arm_names},
        "twelve_route_mean_absolute": {name: {} for name in arm_names},
        "per_route": {name: {} for name in arm_names},
        "paired_gains_vs_identity": {"mean_shift": {}, "mean_scale": {}},
        "mean_scale_vs_mean_shift": {},
        "recovered_fraction_of_g_res_mean": {"mean_shift": {}, "mean_scale": {}},
    }

    for metric in METRICS:
        # 1. Unchanged t->t receiving-context diagonals
        diag_per_lang = {}
        diag_macros = []
        diag_macro_draws = []
        diag_cat_tables = []
        diag_cat_draws_list = []
        for t in LANGUAGES:
            arm = diag_arms[t]
            macro_sum, macro_dr, cat_tbl = _summarize_macro_and_categories(
                categories=categories,
                primary_indices=primary_indices,
                cat_values=arm["points"][metric],
                cat_draws=arm["draws"][metric],
                cat_supports=arm["supports"],
                cat_unestimable=arm["unestimable"],
                minimum=minimum,
            )
            diag_per_lang[_route_key(t, t)] = {
                "target_language": t,
                "supervision_label": SUPERVISION_DIAGONAL,
                "primary_macro": macro_sum,
                "per_category": cat_tbl,
            }
            diag_macros.append(macro_sum)
            diag_macro_draws.append(macro_dr)
            diag_cat_tables.append(cat_tbl)
            diag_cat_draws_list.append(arm["draws"][metric])

        diag_4_mean, _ = _mean_across_routes(diag_macros, diag_macro_draws, label_count=len(LANGUAGES))
        diag_4_per_cat = _mean_per_category_across_routes(categories, diag_cat_tables, diag_cat_draws_list)
        arms_output["target_diagonal_ref"]["metrics"][metric] = {
            "four_diagonal_mean": diag_4_mean,
            "four_diagonal_per_category": diag_4_per_cat,
            "per_route": diag_per_lang,
        }
        summaries["target_diagonal_4_mean"][metric] = diag_4_mean

        # 2. Each transfer arm (identity, mean_shift, mean_scale): absolute + gap G vs t->t
        arm_route_abs_draws = {name: {} for name in arm_names}
        arm_route_gap_draws = {name: {} for name in arm_names}
        arm_route_abs_points = {name: {} for name in arm_names}
        arm_route_gap_points = {name: {} for name in arm_names}
        arm_mean_gap_summary = {}
        arm_mean_gap_draws = {}
        arm_mean_abs_summary = {}
        arm_mean_abs_draws = {}

        for name in arm_names:
            sup_label = SUPERVISION_IDENTITY if name == "identity" else SUPERVISION_ADAPTED
            per_route_abs_macros = []
            per_route_abs_draws_list = []
            per_route_abs_cat_tables = []
            per_route_abs_cat_draws = []

            per_route_gap_macros = []
            per_route_gap_draws_list = []
            per_route_gap_cat_tables = []
            per_route_gap_cat_draws = []
            per_route_rows = {}

            for s, t in off_pairs:
                rkey = _route_key(s, t)
                arm = arm_evals[name][(s, t)]
                d_arm = diag_arms[t]

                # Absolute metric
                abs_macro, abs_dr, abs_cats = _summarize_macro_and_categories(
                    categories=categories,
                    primary_indices=primary_indices,
                    cat_values=arm["points"][metric],
                    cat_draws=arm["draws"][metric],
                    cat_supports=arm["supports"],
                    cat_unestimable=arm["unestimable"],
                    minimum=minimum,
                )
                per_route_abs_macros.append(abs_macro)
                per_route_abs_draws_list.append(abs_dr)
                per_route_abs_cat_tables.append(abs_cats)
                per_route_abs_cat_draws.append(arm["draws"][metric])
                arm_route_abs_points[name][rkey] = abs_macro["value"]
                arm_route_abs_draws[name][rkey] = abs_dr

                # Gap G(s->t) vs unchanged t->t diagonal
                gap_cat_vals = [
                    _gap_value(metric, d_arm["points"][metric][c], arm["points"][metric][c])
                    for c in range(len(categories))
                ]
                gap_cat_dr = [
                    _gap_draws(metric, d_arm["draws"][metric][c], arm["draws"][metric][c])
                    for c in range(len(categories))
                ]
                gap_unest = [
                    bool(d_arm["unestimable"][c] or arm["unestimable"][c])
                    for c in range(len(categories))
                ]
                gap_macro, gap_dr, gap_cats = _summarize_macro_and_categories(
                    categories=categories,
                    primary_indices=primary_indices,
                    cat_values=gap_cat_vals,
                    cat_draws=gap_cat_dr,
                    cat_supports=arm["supports"],
                    cat_unestimable=gap_unest,
                    minimum=minimum,
                )
                per_route_gap_macros.append(gap_macro)
                per_route_gap_draws_list.append(gap_dr)
                per_route_gap_cat_tables.append(gap_cats)
                per_route_gap_cat_draws.append(gap_cat_dr)
                arm_route_gap_points[name][rkey] = gap_macro["value"]
                arm_route_gap_draws[name][rkey] = gap_dr

                per_route_rows[rkey] = {
                    "source_language": s,
                    "target_language": t,
                    "supervision_label": sup_label,
                    "absolute": {"primary_macro": abs_macro, "per_category": abs_cats},
                    "gap": {"primary_macro": gap_macro, "per_category": gap_cats},
                }

            mean_abs, mean_abs_dr = _mean_across_routes(
                per_route_abs_macros, per_route_abs_draws_list, label_count=len(off_pairs)
            )
            mean_abs_per_cat = _mean_per_category_across_routes(
                categories, per_route_abs_cat_tables, per_route_abs_cat_draws
            )
            mean_gap, mean_gap_dr = _mean_across_routes(
                per_route_gap_macros, per_route_gap_draws_list, label_count=len(off_pairs)
            )
            mean_gap_per_cat = _mean_per_category_across_routes(
                categories, per_route_gap_cat_tables, per_route_gap_cat_draws
            )

            arm_mean_abs_summary[name] = mean_abs
            arm_mean_abs_draws[name] = mean_abs_dr
            arm_mean_gap_summary[name] = mean_gap
            arm_mean_gap_draws[name] = mean_gap_dr

            arms_output[name]["metrics"][metric] = {
                "twelve_route_mean_absolute": mean_abs,
                "twelve_route_mean_absolute_per_category": mean_abs_per_cat,
                "twelve_route_mean_gap": mean_gap,
                "twelve_route_mean_gap_per_category": mean_gap_per_cat,
                "per_route": per_route_rows,
            }
            summaries["twelve_route_mean_gap"][name][metric] = mean_gap
            summaries["twelve_route_mean_absolute"][name][metric] = mean_abs
            summaries["per_route"][name][metric] = {
                rkey: {
                    "source_language": per_route_rows[rkey]["source_language"],
                    "target_language": per_route_rows[rkey]["target_language"],
                    "supervision_label": per_route_rows[rkey]["supervision_label"],
                    "absolute_primary_macro": per_route_rows[rkey]["absolute"]["primary_macro"],
                    "gap_primary_macro": per_route_rows[rkey]["gap"]["primary_macro"],
                }
                for rkey in per_route_rows
            }

        # 3. Paired gains vs identity (mean_shift and mean_scale)
        for variant in E11_VARIANTS:
            per_route_gains = {}
            for s, t in off_pairs:
                rkey = _route_key(s, t)
                g_id = arm_route_gap_points["identity"][rkey]
                g_var = arm_route_gap_points[variant][rkey]
                gain_pt = None if (g_id is None or g_var is None) else float(g_id - g_var)
                gain_dr = arm_route_gap_draws["identity"][rkey] - arm_route_gap_draws[variant][rkey]
                per_route_gains[rkey] = {
                    "source_language": s,
                    "target_language": t,
                    "supervision_label": SUPERVISION_ADAPTED,
                    "gap_reduction": {
                        "value": gain_pt,
                        "interval": _interval_record(gain_dr),
                    },
                    "absolute_improvement": {
                        "value": _improvement_value(
                            metric,
                            arm_route_abs_points["identity"][rkey],
                            arm_route_abs_points[variant][rkey],
                        ),
                        "interval": _interval_record(
                            _improvement_draws(
                                metric,
                                arm_route_abs_draws["identity"][rkey],
                                arm_route_abs_draws[variant][rkey],
                            )
                        ),
                    },
                }

            g_id_mean = arm_mean_gap_summary["identity"]["value"]
            g_var_mean = arm_mean_gap_summary[variant]["value"]
            mean_gain_pt = None if (g_id_mean is None or g_var_mean is None) else float(g_id_mean - g_var_mean)
            mean_gain_dr = arm_mean_gap_draws["identity"] - arm_mean_gap_draws[variant]

            summaries["paired_gains_vs_identity"][variant][metric] = {
                "twelve_route_mean_gap_reduction": {
                    "value": mean_gain_pt,
                    "interval": _interval_record(mean_gain_dr),
                },
                "twelve_route_mean_absolute_improvement": {
                    "value": _improvement_value(
                        metric,
                        arm_mean_abs_summary["identity"]["value"],
                        arm_mean_abs_summary[variant]["value"],
                    ),
                    "interval": _interval_record(
                        _improvement_draws(
                            metric,
                            arm_mean_abs_draws["identity"],
                            arm_mean_abs_draws[variant],
                        )
                    ),
                },
                "per_route": per_route_gains,
            }
            summaries["recovered_fraction_of_g_res_mean"][variant][metric] = _recovered_fraction_summary(
                g_id_mean,
                g_var_mean,
                arm_mean_gap_draws["identity"],
                arm_mean_gap_draws[variant],
            )

        # 4. Paired contrast: mean_scale minus mean_shift (isolates diagonal scaling beyond constant offset)
        per_route_scale_vs_shift = {}
        for s, t in off_pairs:
            rkey = _route_key(s, t)
            g_sh = arm_route_gap_points["mean_shift"][rkey]
            g_sc = arm_route_gap_points["mean_scale"][rkey]
            delta_pt = None if (g_sh is None or g_sc is None) else float(g_sh - g_sc)
            delta_dr = arm_route_gap_draws["mean_shift"][rkey] - arm_route_gap_draws["mean_scale"][rkey]
            per_route_scale_vs_shift[rkey] = {
                "source_language": s,
                "target_language": t,
                "gap_reduction": {
                    "value": delta_pt,
                    "interval": _interval_record(delta_dr),
                },
            }
        g_sh_mean = arm_mean_gap_summary["mean_shift"]["value"]
        g_sc_mean = arm_mean_gap_summary["mean_scale"]["value"]
        mean_delta_pt = None if (g_sh_mean is None or g_sc_mean is None) else float(g_sh_mean - g_sc_mean)
        mean_delta_dr = arm_mean_gap_draws["mean_shift"] - arm_mean_gap_draws["mean_scale"]
        summaries["mean_scale_vs_mean_shift"][metric] = {
            "twelve_route_mean_gap_reduction": {
                "value": mean_delta_pt,
                "interval": _interval_record(mean_delta_dr),
            },
            "twelve_route_mean_absolute_improvement": {
                "value": _improvement_value(
                    metric,
                    arm_mean_abs_summary["mean_shift"]["value"],
                    arm_mean_abs_summary["mean_scale"]["value"],
                ),
                "interval": _interval_record(
                    _improvement_draws(
                        metric,
                        arm_mean_abs_draws["mean_shift"],
                        arm_mean_abs_draws["mean_scale"],
                    )
                ),
            },
            "per_route": per_route_scale_vs_shift,
        }

    # Explicit ranking vs proper-loss summary (phase2.md §1.5 & §3.4)
    summaries["ranking_vs_proper_loss_summary"] = {
        "g_res_mean_auroc_identity": summaries["twelve_route_mean_gap"]["identity"]["auroc"],
        "mean_shift_auroc_gap_reduction": summaries["paired_gains_vs_identity"]["mean_shift"]["auroc"][
            "twelve_route_mean_gap_reduction"
        ],
        "mean_shift_ap_gap_reduction": summaries["paired_gains_vs_identity"]["mean_shift"]["ap"][
            "twelve_route_mean_gap_reduction"
        ],
        "mean_shift_brier_gain": summaries["paired_gains_vs_identity"]["mean_shift"]["brier"][
            "twelve_route_mean_absolute_improvement"
        ],
        "mean_shift_nll_gain": summaries["paired_gains_vs_identity"]["mean_shift"]["nll"][
            "twelve_route_mean_absolute_improvement"
        ],
        "mean_scale_auroc_gap_reduction": summaries["paired_gains_vs_identity"]["mean_scale"]["auroc"][
            "twelve_route_mean_gap_reduction"
        ],
        "mean_scale_ap_gap_reduction": summaries["paired_gains_vs_identity"]["mean_scale"]["ap"][
            "twelve_route_mean_gap_reduction"
        ],
        "mean_scale_brier_gain": summaries["paired_gains_vs_identity"]["mean_scale"]["brier"][
            "twelve_route_mean_absolute_improvement"
        ],
        "mean_scale_nll_gain": summaries["paired_gains_vs_identity"]["mean_scale"]["nll"][
            "twelve_route_mean_absolute_improvement"
        ],
        "mean_scale_recovered_fraction_auroc": summaries["recovered_fraction_of_g_res_mean"]["mean_scale"]["auroc"],
        "stage5_reference_ranges": plain(STAGE5_DECISION_RANGES["E11"]),
    }

    return arms_output, summaries


def _verify_resume_analysis_ref(
    root,
    protocol,
    plan,
    gate_ref,
    model,
    e11_refs,
    resume_analysis_ref,
):
    """Verify an explicit E11 analysis resume reference and its 8 referenced adapted blocks."""
    ref, body, _ = verify_resume_ref(
        root,
        protocol,
        resume_analysis_ref,
        KIND_ANALYSIS,
        expected_schema=SCHEMA_ANALYSIS,
        expected_dataset=plan["dataset_fingerprint"],
        expected_plan_identity=plan["plan_identity"],
        expected_owner=model,
    )
    check(body.get("experiment") == "E11", "Resume E11 analysis experiment mismatch")
    check(plain(body.get("gate_ref")) == plain(gate_ref), "Resume E11 analysis gate_ref mismatch")
    check(body.get("condition") == E11_CONDITION, "Resume E11 analysis condition mismatch")
    check(body.get("pooling") == E11_POOLING, "Resume E11 analysis pooling mismatch")
    check(body.get("confirm_excluded") is True, "Resume E11 analysis confirm_excluded must be True")

    inv = body.get("invariance_checks") or {}
    check(
        inv.get("passed") is True
        and float(inv.get("mean_shift_max_abs_auroc_diff", 1.0)) <= MEAN_SHIFT_RANK_TOL
        and float(inv.get("mean_shift_max_abs_ap_diff", 1.0)) <= MEAN_SHIFT_RANK_TOL
        and float(inv.get("mean_shift_logit_offset_max_std", 1.0)) <= MEAN_SHIFT_LOGIT_OFFSET_STD_TOL,
        "Resume E11 analysis failed invariance check verification",
    )

    expected_fit_stats = {lang: plain(e11_refs[lang]["statistics_ref"]) for lang in LANGUAGES}
    expected_identity_blocks = {lang: plain(e11_refs[lang]["screen_prediction_ref"]) for lang in LANGUAGES}
    check(
        plain(body.get("fit_statistics_refs")) == expected_fit_stats,
        "Resume E11 analysis fit_statistics_refs mismatch",
    )
    check(
        plain(body.get("identity_block_refs")) == expected_identity_blocks,
        "Resume E11 analysis identity_block_refs mismatch",
    )

    adapted_refs = plain(body.get("adapted_block_refs"))
    check(
        isinstance(adapted_refs, dict) and set(adapted_refs.keys()) == set(E11_VARIANTS),
        "Resume E11 analysis adapted_block_refs must cover mean_shift and mean_scale",
    )
    for variant in E11_VARIANTS:
        by_lang = adapted_refs[variant]
        check(
            isinstance(by_lang, dict) and set(by_lang.keys()) == set(LANGUAGES),
            f"Resume E11 analysis adapted_block_refs[{variant}] must cover all 4 languages",
        )
        for lang in LANGUAGES:
            verify_resume_ref(
                root,
                protocol,
                by_lang[lang],
                KIND_PREDICTION,
                expected_schema=SCHEMA_PREDICTION,
                expected_dataset=plan["dataset_fingerprint"],
                expected_plan_identity=plan["plan_identity"],
                expected_owner=model,
            )
    return plain(ref), body, adapted_refs


@contract
def run_e11(
    root,
    protocol,
    plan_ref,
    e29_gate_ref,
    model,
    producer,
    *,
    resume_adapted_refs=None,
    resume_analysis_ref=None,
    run_id=None,
    progress=None,
):
    """Run the Phase 2 Stage 3 E11 CPU worker for one model (`qwen` or `gemma`)."""
    check(model in MODELS, f"Unknown model selector: {model!r}")
    require_open_split(E11_SPLIT)
    resolved_root = Path(root).resolve()

    total_units = 11  # 1 bind/gate + 1 fit/self-check + 8 adapted blocks + 1 evaluate/publish
    t_bind = time.monotonic()

    # 1. Authenticate phase2_plan, bind Phase 1 explicitly, and enforce E29 -> E11 gate == OPEN.
    plan = load_phase2_plan(resolved_root, protocol, plan_ref)
    check(plan["confirm_excluded"] is True, "phase2_plan must exclude CONFIRM")
    binding = bind_phase1(plan.root, plain(plan.protocol), plain(plan["stage4_release"]))
    check(binding.lineage_identity == plan["lineage_identity"], "Phase 1 binding lineage mismatch")
    check(binding.dataset_fingerprint == plan["dataset_fingerprint"], "Phase 1 binding dataset mismatch")
    check(binding.bootstrap_sha256 == plan["bootstrap"]["sha256"], "Bootstrap SHA-256 mismatch")
    check(list(binding.bootstrap_shape) == list(plan["bootstrap"]["shape"]), "Bootstrap shape mismatch")

    gate_ref_norm, gate_body, _ = load_e29_gate(
        resolved_root,
        protocol,
        e29_gate_ref,
        plan=plan,
        expected_model=model,
    )
    e11_gate = gate_body["decisions"]["E11"]
    check(
        e11_gate["decision"] == "OPEN",
        f"E11 gate is closed for {model}: decision={e11_gate['decision']!r}, reason={e11_gate.get('reason')!r}",
    )

    categories = list(protocol["category_order"])
    primary_categories = list(protocol["primary_categories"])
    check(len(primary_categories) == 14, f"Expected 14 primary categories, got {len(primary_categories)}")
    primary_indices = [categories.index(cat) for cat in primary_categories]
    screen_minimum = int(protocol["minimum_positive_and_negative"]["SCREEN"])
    depth = midpoint_depth(binding, model)
    n_cats = len(categories)

    # Resolve E11 FIT statistics and caption_mean SCREEN refs.
    e11_refs = e11_fit_statistics_refs(binding, model)
    plan_e11 = plan.get("e11_fit_statistics")
    if plan_e11 is not None and model in plan_e11:
        for lang in LANGUAGES:
            check(
                plain(plan_e11[model][lang]["statistics_ref"]) == plain(e11_refs[lang]["statistics_ref"]),
                f"phase2_plan e11_fit_statistics mismatch for {model}/{lang}",
            )

    bind_s = time.monotonic() - t_bind
    _emit_progress(
        progress,
        "e11_bound_and_gated",
        unit="bind_and_gate",
        completed=1,
        total=total_units,
        read_hash_s=bind_s,
        model=model,
        gate_decision=e11_gate["decision"],
        gate_trigger=e11_gate.get("trigger"),
    )

    # Optional full-analysis resume check.
    if resume_analysis_ref is not None:
        t_res = time.monotonic()
        res_ref, res_body, adapted_refs = _verify_resume_analysis_ref(
            resolved_root,
            protocol,
            plan,
            gate_ref_norm,
            model,
            e11_refs,
            resume_analysis_ref,
        )
        _emit_progress(
            progress,
            "e11_resumed_analysis",
            unit="resume_analysis",
            completed=total_units,
            total=total_units,
            read_hash_s=time.monotonic() - t_res,
            model=model,
            analysis_ref=res_ref,
        )
        return frozen({
            "model": model,
            "run_id": None,
            "gate_ref": plain(gate_ref_norm),
            "fit_pairing_check": plain(res_body["fit_pairing_check"]),
            "identity_self_check": plain(res_body["identity_self_check"]),
            "invariance_checks": plain(res_body["invariance_checks"]),
            "adapted_block_refs": adapted_refs,
            "analysis_ref": res_ref,
            "resumed": True,
        })

    # Validate resume_adapted_refs keys upfront.
    valid_resume_keys = {f"{v}:{t}" for v in E11_VARIANTS for t in LANGUAGES}
    resume_map = {}
    if resume_adapted_refs is not None:
        check(isinstance(resume_adapted_refs, dict), "resume_adapted_refs must be a mapping")
        for key, val in resume_adapted_refs.items():
            check(key in valid_resume_keys, f"Invalid resume_adapted_refs key: {key!r}")
            if val is not None:
                resume_map[key] = val

    # 2. Verify 2,160 paired FIT statistics (0 FIT matrix reloads) and en->en SCREEN identity self-check.
    t_check = time.monotonic()
    dimension, fit_pairing_check, stats_by_lang, heads_by_lang = _verify_fit_statistics_and_heads(
        binding, model, e11_refs
    )

    # Load reused Stage 4 caption_mean SCREEN prediction blocks once.
    stage4_blocks = {}
    for target_lang in LANGUAGES:
        rec = e11_refs[target_lang]
        stage4_blocks[target_lang] = load_prediction_block(
            binding.stage4_plan,
            rec["screen_prediction_ref"],
            expected_block=rec["screen_block_id"],
        )

    identity_self_check, rows_en_cached, matrix_en_cached = _verify_en_identity_self_check(
        binding, model, e11_refs, stage4_blocks, heads_by_lang
    )
    check_s = time.monotonic() - t_check
    _emit_progress(
        progress,
        "e11_fit_and_self_check_passed",
        unit="fit_pairing_and_self_check",
        completed=2,
        total=total_units,
        read_hash_s=check_s,
        model=model,
        paired_fit_images=fit_pairing_check["paired_fit_images"],
        en_self_check_max_abs_diff=0.0,
    )

    # 3. Score (or resume) the 8 adapted SCREEN blocks (4 mean_shift + 4 mean_scale; 24 s != t routes).
    effective_run_id = run_id or new_run_id()
    adapted_block_refs = {variant: {} for variant in E11_VARIANTS}
    adapted_predictions = {variant: {} for variant in E11_VARIANTS}
    completed_units = 2

    for target_lang in LANGUAGES:
        t_feat = time.monotonic()
        if target_lang == "en":
            rows_t, matrix_t = rows_en_cached, matrix_en_cached
        else:
            rows_t, matrix_t = read_verified_midpoint_features(
                binding, model, E11_CONDITION, E11_POOLING, target_lang, E11_SPLIT
            )
        diag_rid = e11_refs[target_lang]["screen_routes"][target_lang]
        diag_pred = stage4_blocks[target_lang][diag_rid]
        check(
            plain(rows_t) == plain(diag_pred.rows),
            f"SCREEN caption_mean feature row alignment mismatch for {model}/{target_lang}",
        )
        H_t = np.asarray(matrix_t, dtype=np.float64)
        check(
            H_t.ndim == 2 and H_t.shape[1] == dimension and np.isfinite(H_t).all(),
            f"Invalid SCREEN caption_mean features for {model}/{target_lang}",
        )
        n_images = int(H_t.shape[0])
        image_ids_t = [row["image_id"] for row in rows_t]
        image_ids_sha256_t = fingerprint(image_ids_t)
        sorted_sources = sorted(s for s in LANGUAGES if s != target_lang)
        feat_read_s = time.monotonic() - t_feat

        for variant in E11_VARIANTS:
            rkey_resume = f"{variant}:{target_lang}"
            if rkey_resume in resume_map:
                t_res = time.monotonic()
                block_ref, logits_arr, est_arr, prev_arr = _verify_resume_adapted_block(
                    resolved_root,
                    protocol,
                    plan,
                    gate_ref_norm,
                    model,
                    variant,
                    target_lang,
                    sorted_sources,
                    e11_refs,
                    image_ids_sha256_t,
                    fit_pairing_check,
                    n_images,
                    n_cats,
                    resume_map[rkey_resume],
                )
                adapted_block_refs[variant][target_lang] = block_ref
                for s_idx, src_lang in enumerate(sorted_sources):
                    adapted_predictions[variant][(src_lang, target_lang)] = {
                        "logits": logits_arr[s_idx],
                        "estimable": est_arr[s_idx],
                        "prevalence": prev_arr[s_idx],
                        "rows": diag_pred.rows,
                        "block_ref": block_ref,
                    }
                completed_units += 1
                _emit_progress(
                    progress,
                    "e11_adapted_block_resumed",
                    unit=f"adapted:{variant}:{target_lang}",
                    completed=completed_units,
                    total=total_units,
                    read_hash_s=feat_read_s + (time.monotonic() - t_res),
                    model=model,
                    variant=variant,
                    target_language=target_lang,
                    block_ref=block_ref,
                )
                continue

            t_comp = time.monotonic()
            logits_list = []
            est_list = []
            prev_list = []
            route_records = []
            upstream_refs = [
                plain(plan.reference),
                plain(gate_ref_norm),
                plain(e11_refs[target_lang]["screen_feature_release_ref"]),
                plain(e11_refs[target_lang]["statistics_ref"]),
            ]

            stats_t = stats_by_lang[target_lang]
            mu_t = np.asarray(stats_t["mean"], dtype=np.float64)
            sigma_t = np.asarray(stats_t["scale"], dtype=np.float64)

            for s_idx, src_lang in enumerate(sorted_sources):
                stats_s = stats_by_lang[src_lang]
                head_s = heads_by_lang[src_lang]
                H_adapted = _apply_fit_transform(H_t, stats_s, stats_t, variant)
                Z = _score_through_source_head(
                    H_adapted,
                    head_s["coef"],
                    head_s["intercept"],
                    head_s["estimable"],
                    head_s["stats"],
                )
                logits_list.append(Z)
                est_list.append(np.asarray(head_s["estimable"], dtype=bool))
                prev_list.append(np.asarray(head_s["prevalence"], dtype=np.float64))

                mu_s = np.asarray(stats_s["mean"], dtype=np.float64)
                sigma_s = np.asarray(stats_s["scale"], dtype=np.float64)
                scale_ratio = sigma_s / sigma_t
                route_records.append({
                    "route_id": f"{model}:{E11_CONDITION}:{E11_POOLING}:{variant}:{src_lang}->{target_lang}",
                    "source_language": src_lang,
                    "row_index": int(s_idx),
                    "source_system_ref": {
                        "group_id": e11_refs[src_lang]["group_id"],
                        "selection_ref": plain(e11_refs[src_lang]["selection_ref"]),
                        "candidate_ref": plain(e11_refs[src_lang]["candidate_ref"]),
                        "C": float(e11_refs[src_lang]["C"]),
                    },
                    "source_fit_stats_ref": plain(e11_refs[src_lang]["statistics_ref"]),
                    "target_fit_stats_ref": plain(e11_refs[target_lang]["statistics_ref"]),
                    "delta_mean_l2": float(np.linalg.norm(mu_s - mu_t)),
                    "scale_ratio_summary": {
                        "min": float(np.min(scale_ratio)),
                        "median": float(np.median(scale_ratio)),
                        "max": float(np.max(scale_ratio)),
                        "mean": float(np.mean(scale_ratio)),
                    },
                    "supervision_label": SUPERVISION_ADAPTED,
                })
                upstream_refs.extend([
                    plain(e11_refs[src_lang]["selection_ref"]),
                    plain(e11_refs[src_lang]["candidate_ref"]),
                    plain(e11_refs[src_lang]["statistics_ref"]),
                ])

            logits_arr = np.asarray(logits_list, dtype="<f8")
            est_arr = np.asarray(est_list, dtype=bool)
            prev_arr = np.asarray(prev_list, dtype="<f8")
            compute_s = time.monotonic() - t_comp

            block_body = {
                "schema": SCHEMA_PREDICTION,
                "experiment": "E11",
                "plan": plain(plan.reference),
                "plan_identity": plan["plan_identity"],
                "gate_ref": plain(gate_ref_norm),
                "model": model,
                "midpoint_depth": depth,
                "condition": E11_CONDITION,
                "pooling": E11_POOLING,
                "split": E11_SPLIT,
                "variant": variant,
                "target_language": target_lang,
                "source_languages": sorted_sources,
                "n_images": n_images,
                "n_categories": n_cats,
                "category_order": categories,
                "image_ids_sha256": image_ids_sha256_t,
                "fit_pairing_check": fit_pairing_check,
                "identity_self_check": identity_self_check,
                "routes": route_records,
                "upstream_feature_ref": plain(e11_refs[target_lang]["screen_feature_release_ref"]),
                "confirm_excluded": True,
            }

            t_write = time.monotonic()
            block_ref = plain(
                publish_phase2(
                    resolved_root,
                    protocol,
                    plan["dataset_fingerprint"],
                    f"{E11_ADAPTED_NAMESPACE}/{model}",
                    KIND_PREDICTION,
                    block_body,
                    producer,
                    arrays={
                        "logits": logits_arr,
                        "estimable": est_arr,
                        "prevalence": prev_arr,
                    },
                    upstream=upstream_refs,
                    run_id=f"{effective_run_id}-{variant}-{target_lang}",
                )
            )
            write_s = time.monotonic() - t_write

            adapted_block_refs[variant][target_lang] = block_ref
            for s_idx, src_lang in enumerate(sorted_sources):
                adapted_predictions[variant][(src_lang, target_lang)] = {
                    "logits": logits_arr[s_idx],
                    "estimable": est_arr[s_idx],
                    "prevalence": prev_arr[s_idx],
                    "rows": diag_pred.rows,
                    "block_ref": block_ref,
                }
            completed_units += 1
            _emit_progress(
                progress,
                "e11_adapted_block_published",
                unit=f"adapted:{variant}:{target_lang}",
                completed=completed_units,
                total=total_units,
                read_hash_s=feat_read_s if variant == E11_VARIANTS[0] else 0.0,
                compute_s=compute_s,
                write_s=write_s,
                model=model,
                variant=variant,
                target_language=target_lang,
                block_ref=block_ref,
            )

    # 4. Evaluate identity, mean_shift, and mean_scale against unchanged t->t diagonals on frozen bootstrap.
    t_eval = time.monotonic()
    draws = load_bootstrap(binding.stage4_plan)
    check(
        list(draws.shape) == list(plan["bootstrap"]["shape"])
        and hashlib.sha256(draws.tobytes()).hexdigest() == plan["bootstrap"]["sha256"],
        "Loaded bootstrap matrix does not match phase2_plan",
    )
    total_draws = int(draws.shape[0])

    # Align all 4 target-language Stage 4 diagonal predictions to sorted screen_ids.
    diag_preds_list = [
        stage4_blocks[t][e11_refs[t]["screen_routes"][t]] for t in LANGUAGES
    ]
    screen_ids, aligned_tuple = paired_image_alignment(binding.stage4_plan, *diag_preds_list)
    aligned_by_target = {t: aligned_tuple[idx] for idx, t in enumerate(LANGUAGES)}
    counts = bootstrap_counts(draws, len(screen_ids))

    # Extract SCREEN targets & unknown-cell mask once and assert exact agreement across languages.
    y_ref, known_ref = targets_for_prediction(binding.stage4_plan, diag_preds_list[0])
    y = y_ref[aligned_by_target[LANGUAGES[0]]]
    known = known_ref[aligned_by_target[LANGUAGES[0]]]
    for idx, t in enumerate(LANGUAGES[1:], start=1):
        y_t, known_t = targets_for_prediction(binding.stage4_plan, diag_preds_list[idx])
        check(
            np.array_equal(y, y_t[aligned_by_target[t]])
            and np.array_equal(known, known_t[aligned_by_target[t]]),
            f"SCREEN target/mask mismatch across languages: {t}",
        )

    # Evaluate diagonals (t->t), identity (s!=t), mean_shift (s!=t), mean_scale (s!=t).
    diag_arms = {}
    for t in LANGUAGES:
        pred = stage4_blocks[t][e11_refs[t]["screen_routes"][t]]
        diag_arms[t] = _evaluate_arm(
            pred.logits,
            pred.estimable,
            aligned_by_target[t],
            y,
            known,
            counts,
            categories,
            total_draws,
        )

    arm_evals = {name: {} for name in ("identity", "mean_shift", "mean_scale")}
    identity_logits_by_route = {}
    mean_shift_logits_by_route = {}

    for s, t in _off_diagonal_pairs():
        id_rid = e11_refs[t]["screen_routes"][s]
        id_pred = stage4_blocks[t][id_rid]
        identity_logits_by_route[(s, t)] = id_pred.logits
        arm_evals["identity"][(s, t)] = _evaluate_arm(
            id_pred.logits,
            id_pred.estimable,
            aligned_by_target[t],
            y,
            known,
            counts,
            categories,
            total_draws,
        )
        for variant in E11_VARIANTS:
            ap = adapted_predictions[variant][(s, t)]
            if variant == "mean_shift":
                mean_shift_logits_by_route[(s, t)] = ap["logits"]
            arm_evals[variant][(s, t)] = _evaluate_arm(
                ap["logits"],
                ap["estimable"],
                aligned_by_target[t],
                y,
                known,
                counts,
                categories,
                total_draws,
            )

    # Mandatory hard automatic invariance check on mean_shift (HARD FAIL if violated).
    invariance_checks = _assert_mean_shift_invariance(
        identity_arms=arm_evals["identity"],
        mean_shift_arms=arm_evals["mean_shift"],
        identity_logits_by_route=identity_logits_by_route,
        mean_shift_logits_by_route=mean_shift_logits_by_route,
        aligned_by_target=aligned_by_target,
        categories=categories,
    )

    arms_output, summaries = _evaluate_e11_arms(
        diag_arms=diag_arms,
        arm_evals=arm_evals,
        categories=categories,
        primary_indices=primary_indices,
        minimum=screen_minimum,
    )
    eval_s = time.monotonic() - t_eval

    identity_block_refs = {lang: plain(e11_refs[lang]["screen_prediction_ref"]) for lang in LANGUAGES}
    fit_statistics_refs = {lang: plain(e11_refs[lang]["statistics_ref"]) for lang in LANGUAGES}

    analysis_body = {
        "schema": SCHEMA_ANALYSIS,
        "experiment": "E11",
        "plan": plain(plan.reference),
        "plan_identity": plan["plan_identity"],
        "gate_ref": plain(gate_ref_norm),
        "model": model,
        "midpoint_depth": depth,
        "condition": E11_CONDITION,
        "pooling": E11_POOLING,
        "bootstrap": plain(plan["bootstrap"]),
        "identity_block_refs": identity_block_refs,
        "adapted_block_refs": adapted_block_refs,
        "fit_statistics_refs": fit_statistics_refs,
        "fit_pairing_check": fit_pairing_check,
        "identity_self_check": identity_self_check,
        "arms": arms_output,
        "summaries": summaries,
        "invariance_checks": invariance_checks,
        "supervision_labels": {
            "target_diagonal_ref": SUPERVISION_DIAGONAL,
            "identity": SUPERVISION_IDENTITY,
            "mean_shift": SUPERVISION_ADAPTED,
            "mean_scale": SUPERVISION_ADAPTED,
        },
        "caption_robustness_note": CAPTION_ROBUSTNESS_NOTE,
        "confirm_excluded": True,
    }

    upstream_analysis = [plain(plan.reference), plain(gate_ref_norm)]
    upstream_analysis.extend(identity_block_refs.values())
    upstream_analysis.extend(fit_statistics_refs.values())
    for variant in E11_VARIANTS:
        upstream_analysis.extend(adapted_block_refs[variant].values())

    t_write = time.monotonic()
    analysis_ref = plain(
        publish_phase2(
            resolved_root,
            protocol,
            plan["dataset_fingerprint"],
            f"{E11_RESULTS_NAMESPACE}/{model}",
            KIND_ANALYSIS,
            analysis_body,
            producer,
            upstream=upstream_analysis,
            run_id=effective_run_id,
        )
    )
    verify_resume_ref(
        resolved_root,
        protocol,
        analysis_ref,
        KIND_ANALYSIS,
        expected_schema=SCHEMA_ANALYSIS,
        expected_dataset=plan["dataset_fingerprint"],
        expected_plan_identity=plan["plan_identity"],
        expected_owner=model,
    )
    write_s = time.monotonic() - t_write

    _emit_progress(
        progress,
        "e11_published",
        unit="evaluate_and_publish",
        completed=total_units,
        total=total_units,
        compute_s=eval_s,
        write_s=write_s,
        model=model,
        analysis_ref=analysis_ref,
        invariance_passed=True,
    )

    return frozen({
        "model": model,
        "run_id": effective_run_id,
        "gate_ref": plain(gate_ref_norm),
        "fit_pairing_check": fit_pairing_check,
        "identity_self_check": identity_self_check,
        "invariance_checks": invariance_checks,
        "adapted_block_refs": adapted_block_refs,
        "analysis_ref": analysis_ref,
        "resumed": False,
    })


def _parse_ref_arg(value):
    if value is None:
        return None
    text = value.strip()
    if text.startswith("{"):
        parsed = json.loads(text)
        check(isinstance(parsed, dict), "JSON reference argument must decode to an object")
        return parsed
    return text


def _parse_ref_map_arg(value):
    if value is None:
        return None
    text = value.strip()
    if text.startswith("{"):
        parsed = json.loads(text)
        check(isinstance(parsed, dict), "JSON reference map argument must decode to an object")
        return parsed
    path = Path(text)
    check(path.is_file(), f"Missing resume adapted refs JSON file: {path}")
    parsed = json.loads(path.read_text(encoding="utf-8"))
    check(isinstance(parsed, dict), "Resume adapted refs file must contain a JSON object")
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
        "--resume-adapted-refs",
        help="Optional JSON object or file mapping '{variant}:{target_lang}' -> {'path', 'sha256'} or path",
    )
    parser.add_argument(
        "--resume-analysis-ref",
        help="Optional explicit E11 analysis {'path', 'sha256'} JSON or path",
    )
    parser.add_argument("--run-id", help="Optional explicit run_id")
    parser.add_argument("--output-ref", type=Path, help="Optional JSON file path for emitted references")
    args = parser.parse_args(argv)

    root = phase1_root(args.phase1_root)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    producer = current_producer(args.storage_scope)
    _emit_cli({
        "event": "e11_start",
        "model": args.model,
        "code_commit": producer["code_commit"],
        "storage_scope": producer["storage_scope"],
    })
    result = plain(
        run_e11(
            root,
            protocol,
            _parse_ref_arg(args.plan),
            _parse_ref_arg(args.e29_gate),
            args.model,
            producer,
            resume_adapted_refs=_parse_ref_map_arg(args.resume_adapted_refs),
            resume_analysis_ref=_parse_ref_arg(args.resume_analysis_ref),
            run_id=args.run_id,
            progress=_emit_cli,
        )
    )
    summary = {
        "event": "e11_complete",
        "status": "passed",
        **result,
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
        _emit_cli({"status": "blocked", "blockers": [str(exc)], "stage3_e11_complete": False})
        raise SystemExit(2)
