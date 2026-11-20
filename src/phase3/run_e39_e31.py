"""Phase 3 Stage B3 Worker 1 (`run_e39_e31`): CPU scoring & mechanistic analysis for E39+E31.

Implements phase3.md §B3.2-B3.3 (Worker 1: `run_e39_e31`):
1. Authenticates `phase3_gpu_plan` (`bind=True`, `confirm_excluded=True`,
   `e35_excluded=True`) and Stage B2 `E39+E31` release (`verify_e39_e31_release`),
   and reassembles all 12 `SCREEN` chunks into `(360, dim)` / `(360, 3)` matrices
   aligned to `stage4_plan["screen_ids"]`.
2. Runs the Stage 2 draw-0 feature parity gate (`max_abs_diff <= 1e-4`) across all
   `4 depths x 2 conditions x 2 poolings x 4 languages = 64` cells plus in-memory
   `en->en` matched logit self-checks (`<= 1e-4`) against accepted Stage 4 logits.
3. E39 sub-caption analysis: scores all 8 `E39_SITES` across `4 depths x 2 conditions
   x 16 routes` through both frozen Stage 3 source heads (`last_caption` and
   `caption_mean`), evaluates `(2000, 360)` bootstrap metrics (`auroc, ap, brier, nll`),
   paired gains vs `last_caption`, recovery fractions under `fixed_head_last`,
   `fixed_head_mean`, and `matched_head`, paired `image_caption - caption_only`
   profiles, token-selection audit, and `PHASE3_GPU_DECISION_RANGES["E39"]`.
4. E31 decoder visual-token & attention-mass analysis: cross-language cosine/diff
   invariance, deterministic 5-fold CV L2 logistic probe (`fold = image_idx % 5`,
   `C = 1.0`) across `encoder -> projector -> decoder:d1..d4`, attention-mass
   summaries (`[visual_mass, caption_mass, system_mass]`), Pearson `r` & Spearman
   `rho` correlations against `G_last`, `G_mean`, and `G1` gap reduction, and
   `PHASE3_GPU_DECISION_RANGES["E31"]`.
5. Publishes immutable analysis artifact under `phase3/results/e39_e31/<model>/`.
"""

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import statistics as py_stats
import sys
import time
from typing import Any, Optional

from src.common.contracts import LANGUAGES, ContractError, fingerprint
from src.common.storage import file_sha256, new_run_id, phase1_root, within
from src.phase3.extract_e39_e31 import (
    E31_ATTENTION_SITES,
    E39_SINGLE_TOKEN_SITES,
    E39_SPAN_MEAN_SITES,
    expected_array_specs,
    verify_e39_e31_release,
)
from src.phase3.interfaces import (
    CERTIFIED_STAGE_B1_GPU_PLAN,
    CERTIFIED_STAGE_B2_RELEASES,
    CONDITIONS,
    E31_SITES,
    E39_SITES,
    KIND_ANALYSIS,
    KIND_E39_E31_RELEASE,
    MODELS,
    PHASE3_GPU_DECISION_RANGES,
    POOLINGS,
    REPO,
    SCHEMA_ANALYSIS,
    SCHEMA_E39_E31_FEATURES,
    check,
    contract,
    current_producer,
    frozen,
    load_phase3_artifact,
    load_phase3_gpu_plan,
    normalize_ref,
    plain,
    publish_phase3,
    require_open_split,
    verify_resume_ref,
)

E39_E31_SPLIT = "SCREEN"
E39_E31_RESULTS_NAMESPACE = "phase3/results/e39_e31"
REPORT_METRICS = ("auroc", "ap", "brier", "nll")
LOSS_METRICS = frozenset({"brier", "nll"})
HEAD_MODES = ("fixed_head_last", "fixed_head_mean", "matched_head")
MATCHED_HEAD_FOR_SITE = frozen({
    "first_caption": "last_caption",
    "middle_caption": "last_caption",
    "penultimate_caption": "last_caption",
    "last_non_punct_caption": "last_caption",
    "last_caption": "last_caption",
    "first_half_mean": "caption_mean",
    "second_half_mean": "caption_mean",
    "caption_mean": "caption_mean",
})
CV_FOLDS = 5
CV_C = 1.0
SUPERVISION_DIAGONAL = "within_context_source_fit"
SUPERVISION_OFF_DIAGONAL = "source_only_transfer"
SUPERVISION_SCREEN_CV = "screen_5fold_cv_probe"
ATTENTION_SLOTS = ("visual_mass", "caption_mass", "system_mass")


def _peak_rss_bytes() -> Optional[int]:
    try:
        import resource

        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return rss if sys.platform == "darwin" else rss * 1024
    except Exception:
        return None


def _emit_progress(
    progress,
    event: str,
    *,
    unit: str,
    completed: int,
    total: int,
    read_hash_s: float = 0.0,
    compute_s: float = 0.0,
    write_s: float = 0.0,
    **extra,
) -> None:
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


def _default_emit(event: Mapping[str, Any]) -> None:
    print(json.dumps(plain(event), sort_keys=True), flush=True)


def _route_key(source_lang: str, target_lang: str) -> str:
    return f"{source_lang}->{target_lang}"


def _off_diagonal_pairs():
    return [(s, t) for s in LANGUAGES for t in LANGUAGES if s != t]


def _gap_value(metric: str, diag_val: Optional[float], route_val: Optional[float]) -> Optional[float]:
    if diag_val is None or route_val is None:
        return None
    if metric in LOSS_METRICS:
        return float(route_val - diag_val)
    return float(diag_val - route_val)


def _gap_draws(metric: str, diag_draws, route_draws):
    if metric in LOSS_METRICS:
        return route_draws - diag_draws
    return diag_draws - route_draws


def _improvement_value(metric: str, base_val: Optional[float], alt_val: Optional[float]) -> Optional[float]:
    """Signed improvement of `alt` over `base` (positive always means better)."""
    if base_val is None or alt_val is None:
        return None
    if metric in LOSS_METRICS:
        return float(base_val - alt_val)
    return float(alt_val - base_val)


def _improvement_draws(metric: str, base_draws, alt_draws):
    if metric in LOSS_METRICS:
        return base_draws - alt_draws
    return alt_draws - base_draws


def _diff_opt(a: Optional[float], b: Optional[float]) -> Optional[float]:
    return None if (a is None or b is None) else float(a - b)


def _mean_optional(values) -> Optional[float]:
    if not values or any(v is None for v in values):
        return None
    return float(sum(values) / len(values))


def _interval_record(draw_values) -> dict:
    from src.stage4.metrics import bootstrap_interval

    interval = bootstrap_interval(draw_values)
    if interval["valid_draws"] == 0:
        interval["reason"] = "no_valid_bootstrap_draws"
    return interval


def _category_status_and_reason(value, pos_n: int, neg_n: int, minimum: int, unestimable: bool):
    from src.stage4.metrics import support_status

    status = support_status(value, pos_n, neg_n, minimum)
    if status == "ok":
        return status, None
    if status == "descriptive_low_support":
        return status, f"screen_support_below_minimum_{minimum}"
    if unestimable:
        return status, "unestimable_head"
    return status, "undefined_metric_or_zero_denominator"


def _macro_summary(point_val, draw_array, *, categories_count: int, defined_count: int, ok_count: int, minimum: int) -> dict:
    if point_val is None:
        status = "undefined"
        reason = "unresolved_macro"
    elif ok_count == categories_count:
        status = "ok"
        reason = None
    else:
        status = "descriptive_low_support"
        reason = f"category_support_below_minimum_{minimum}"
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


def _summarize_macros_and_categories(
    *,
    categories,
    primary_indices,
    all_indices,
    cat_values,
    cat_draws,
    cat_supports,
    cat_unestimable,
    minimum: int,
    include_categories: bool = True,
):
    """Build primary 14-category macro, all-category macro, and optional per-category table."""
    import numpy as np
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

    stacked = np.stack(cat_draws, axis=0)
    float_vals = [0.0 if v is None else float(v) for v in cat_values]

    def _cohort(indices):
        defined_n = sum(cat_values[idx] is not None for idx in indices)
        ok_n = sum(
            cat_values[idx] is not None and min(cat_supports[idx][1], cat_supports[idx][2]) >= minimum
            for idx in indices
        )
        m_val = None if defined_n < len(indices) else float(fixed_macro(float_vals, indices))
        m_draws = fixed_macro(stacked, indices, axis=0)
        summary = _macro_summary(
            m_val,
            m_draws,
            categories_count=len(indices),
            defined_count=defined_n,
            ok_count=ok_n,
            minimum=minimum,
        )
        return summary, m_draws

    primary_macro, primary_draws = _cohort(primary_indices)
    all_macro, all_draws = _cohort(all_indices)
    return primary_macro, primary_draws, all_macro, all_draws, per_category


def _compute_recovery_summary(
    *,
    num_val: Optional[float],
    num_draws,
    den_val: Optional[float],
    den_draws,
    categories_count: int,
    ok_count: int,
    minimum: int,
) -> dict:
    """Compute within-draw recovery ratio `(site - last) / (mean - last)` safely."""
    import numpy as np

    if num_val is None or den_val is None or abs(float(den_val)) <= 1e-12:
        point_rec = None
    else:
        point_rec = float(num_val / den_val)

    valid_mask = np.isfinite(num_draws) & np.isfinite(den_draws) & (np.abs(den_draws) > 1e-12)
    safe_den = np.where(valid_mask, den_draws, 1.0)
    rec_draws = np.where(valid_mask, num_draws / safe_den, np.nan)
    return _macro_summary(
        point_rec,
        rec_draws,
        categories_count=categories_count,
        defined_count=categories_count if point_rec is not None else 0,
        ok_count=ok_count if point_rec is not None else 0,
        minimum=minimum,
    )


def assign_screen_cv_folds(n_images: int, n_folds: int = CV_FOLDS) -> list:
    """Deterministic `fold = image_idx % n_folds` assignment over `screen_ids`."""
    check(type(n_images) is int and n_images >= n_folds, f"Need at least {n_folds} images for {n_folds}-fold CV")
    check(type(n_folds) is int and n_folds >= 2, "n_folds must be >= 2")
    return [idx % n_folds for idx in range(n_images)]


def _average_ranks(values: list) -> list:
    """Compute 1-based average ranks with exact tie averaging."""
    n = len(values)
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i + 1
        while j < n and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def pearson_and_spearman(x_vals, y_vals) -> dict:
    """Compute exact Pearson `r` and tie-averaged Spearman `rho` in float64."""
    pairs = []
    for xv, yv in zip(x_vals, y_vals):
        if xv is None or yv is None:
            continue
        xf = float(xv)
        yf = float(yv)
        if math.isfinite(xf) and math.isfinite(yf):
            pairs.append((xf, yf))

    n = len(pairs)
    if n < 2:
        return {"n": n, "pearson_r": None, "spearman_rho": None, "reason": "insufficient_pairs"}

    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]

    def _pearson(a, b):
        mean_a = sum(a) / len(a)
        mean_b = sum(b) / len(b)
        da = [v - mean_a for v in a]
        db = [v - mean_b for v in b]
        ss_a = sum(v * v for v in da)
        ss_b = sum(v * v for v in db)
        if ss_a <= 1e-24 or ss_b <= 1e-24:
            return None
        r = sum(u * v for u, v in zip(da, db)) / math.sqrt(ss_a * ss_b)
        return max(-1.0, min(1.0, float(r)))

    r = _pearson(xs, ys)
    rho = _pearson(_average_ranks(xs), _average_ranks(ys))
    return {
        "n": n,
        "pearson_r": r,
        "spearman_rho": rho,
        "reason": None if (r is not None and rho is not None) else "zero_variance",
    }


def _validate_token_selection_entry(entry: Any) -> None:
    check(isinstance(entry, Mapping), "e39_token_selection entry must be a mapping")
    n_tok = entry.get("token_count")
    check(type(n_tok) is int and n_tok >= 1, "Invalid e39_token_selection token_count")
    for site in E39_SINGLE_TOKEN_SITES:
        val = entry.get(site)
        check(type(val) is int and 0 <= val < n_tok, f"Invalid e39_token_selection index for {site}")
    check(entry.get("last_caption") == n_tok - 1, "e39_token_selection last_caption must equal token_count - 1")
    check(
        type(entry.get("last_non_punct_fallback_used")) is bool,
        "e39_token_selection missing boolean last_non_punct_fallback_used",
    )
    for span_key in ("first_half_span", "second_half_span", "full_span"):
        span = entry.get(span_key)
        check(
            isinstance(span, (list, tuple))
            and len(span) == 2
            and type(span[0]) is int
            and type(span[1]) is int
            and 0 <= span[0] < span[1] <= n_tok,
            f"Invalid e39_token_selection span {span_key}",
        )


def _summarize_selection_slice(entries: list) -> dict:
    n_views = len(entries)
    check(n_views > 0, "Empty token_selection entries")
    counts = [int(e["token_count"]) for e in entries]
    diff_count = sum(1 for e in entries if int(e["last_non_punct_caption"]) != int(e["last_caption"]))
    fallback_count = sum(1 for e in entries if bool(e["last_non_punct_fallback_used"]))
    rel_pos = {}
    for site in E39_SINGLE_TOKEN_SITES:
        vals = [
            float(int(e[site]) / max(1, int(e["token_count"]) - 1)) if int(e["token_count"]) > 1 else 0.0
            for e in entries
        ]
        rel_pos[site] = float(sum(vals) / n_views)
    return {
        "n_views": n_views,
        "token_count": {
            "mean": float(sum(counts) / n_views),
            "median": float(py_stats.median(counts)),
            "min": int(min(counts)),
            "max": int(max(counts)),
        },
        "last_non_punct_differs_from_last_count": int(diff_count),
        "last_non_punct_differs_from_last_fraction": float(diff_count / n_views),
        "last_non_punct_fallback_used_count": int(fallback_count),
        "last_non_punct_fallback_used_fraction": float(fallback_count / n_views),
        "mean_relative_position": rel_pos,
    }


def summarize_e39_token_selection_audit(token_selections_by_cond: Mapping[str, Any], screen_ids) -> dict:
    """Summarize E39 sub-caption token-selection statistics across conditions and languages."""
    n_screen = len(screen_ids)
    check(n_screen > 0, "Empty screen_ids in token selection audit")
    by_condition = {}
    for cond in CONDITIONS:
        check(cond in token_selections_by_cond, f"Missing condition {cond!r} in token selections")
        all_entries = []
        per_language = {}
        for lang in LANGUAGES:
            lang_entries = list(token_selections_by_cond[cond][lang])
            check(
                len(lang_entries) == n_screen,
                f"Expected {n_screen} token selections for {cond}/{lang}; got {len(lang_entries)}",
            )
            for entry in lang_entries:
                _validate_token_selection_entry(entry)
            per_language[lang] = _summarize_selection_slice(lang_entries)
            all_entries.extend(lang_entries)
        by_condition[cond] = {
            "overall": _summarize_selection_slice(all_entries),
            "per_language": per_language,
        }

    # Cross-condition selection parity across all 4 * n_screen views
    identical_total = 0
    total_views = n_screen * len(LANGUAGES)
    per_lang_parity = {}
    for lang in LANGUAGES:
        same_lang = sum(
            1
            for a, b in zip(token_selections_by_cond["image_caption"][lang], token_selections_by_cond["caption_only"][lang])
            if plain(a) == plain(b)
        )
        identical_total += same_lang
        per_lang_parity[lang] = {
            "n_views": n_screen,
            "identical_selection_count": int(same_lang),
            "identical_selection_fraction": float(same_lang / n_screen),
        }

    return {
        "by_condition": by_condition,
        "cross_condition_parity": {
            "n_views": int(total_views),
            "identical_selection_count": int(identical_total),
            "identical_selection_fraction": float(identical_total / total_views),
            "per_language": per_lang_parity,
        },
    }


def evaluate_e39_decisions(e39_summary: Mapping[str, Any], depths, midpoint_depth: int, decision_ranges: Mapping[str, Any]) -> dict:
    """Evaluate predeclared E39 decision thresholds from `PHASE3_GPU_DECISION_RANGES["E39"]`."""
    rec_min = float(decision_ranges["terminal_token_recovery_fraction_min"])
    gain_min = float(decision_ranges["auroc_gain_min"])
    neg_lo, neg_hi = [float(x) for x in decision_ranges["negligible_ci_within"]]

    def _eval_site_record(site_rec: Mapping[str, Any]) -> dict:
        gain_pm = site_rec["gain_vs_last_caption"]["twelve_route_mean_transfer"]["auroc"]["primary_macro"]
        rec_pm = site_rec["recovery_fraction_of_caption_mean"]["twelve_route_mean_transfer"]["auroc"]["primary_macro"]
        g_val = gain_pm["value"]
        g_iv = gain_pm["interval"]
        r_val = rec_pm["value"]
        r_iv = rec_pm["interval"]
        g_lo, g_hi = g_iv["lower"], g_iv["upper"]
        r_lo = r_iv["lower"]

        ci_excl_zero = bool(g_lo is not None and g_lo > 0.0)
        meets_gain = bool(g_val is not None and g_val >= gain_min and ci_excl_zero)
        meets_rec = bool(r_val is not None and r_val >= rec_min)
        negligible = bool(
            g_lo is not None
            and g_hi is not None
            and g_lo >= neg_lo
            and g_hi <= neg_hi
            and int(g_iv.get("valid_draws", 0)) >= 1900
        )
        return {
            "auroc_gain_vs_last": g_val,
            "auroc_gain_ci": [g_lo, g_hi],
            "auroc_gain_ci_excludes_zero": ci_excl_zero,
            "meets_auroc_gain_min": meets_gain,
            "recovery_fraction": r_val,
            "recovery_fraction_ci": [r_lo, r_iv["upper"]],
            "meets_recovery_fraction_min": meets_rec,
            "negligible_vs_last": negligible,
            "consequential": bool(meets_gain and meets_rec),
        }

    per_cell = {}
    for depth in depths:
        dkey = str(depth)
        per_cell[dkey] = {}
        for cond in CONDITIONS:
            cell = e39_summary["cells"][dkey][cond]
            mode_summary = {}
            for mode in HEAD_MODES:
                sites_eval = {
                    site: _eval_site_record(cell["head_modes"][mode]["sites"][site])
                    for site in E39_SITES
                }
                terminal_alt_conseq = any(
                    sites_eval[s]["consequential"] for s in ("last_non_punct_caption", "penultimate_caption")
                )
                half_span_conseq = any(
                    sites_eval[s]["consequential"] for s in ("first_half_mean", "second_half_mean")
                )
                mode_summary[mode] = {
                    "sites": sites_eval,
                    "terminal_token_artifact_supported": bool(terminal_alt_conseq),
                    "half_span_consequential": bool(half_span_conseq),
                }
            per_cell[dkey][cond] = mode_summary

    mid_ic_fixed = per_cell[str(midpoint_depth)]["image_caption"]["fixed_head_last"]
    mid_ic_matched = per_cell[str(midpoint_depth)]["image_caption"]["matched_head"]
    mid_co_fixed = per_cell[str(midpoint_depth)]["caption_only"]["fixed_head_last"]

    return {
        "thresholds": {
            "terminal_token_recovery_fraction_min": rec_min,
            "auroc_gain_min": gain_min,
            "negligible_ci_within": [neg_lo, neg_hi],
        },
        "midpoint_summary": {
            "depth": int(midpoint_depth),
            "image_caption_fixed_head_last": mid_ic_fixed,
            "image_caption_matched_head": mid_ic_matched,
            "caption_only_fixed_head_last": mid_co_fixed,
            "terminal_token_artifact_supported": bool(
                mid_ic_fixed["terminal_token_artifact_supported"]
                or mid_ic_matched["terminal_token_artifact_supported"]
            ),
            "last_non_punct_negligible_under_fixed_head_last": bool(
                mid_ic_fixed["sites"]["last_non_punct_caption"]["negligible_vs_last"]
            ),
            "span_mean_pooling_required": bool(
                not mid_ic_fixed["terminal_token_artifact_supported"]
                and not mid_ic_matched["terminal_token_artifact_supported"]
                and (mid_ic_fixed["half_span_consequential"] or mid_ic_matched["half_span_consequential"])
            ),
        },
        "per_cell": per_cell,
    }


def evaluate_e31_decisions(e31_summary: Mapping[str, Any], depths, midpoint_depth: int, decision_ranges: Mapping[str, Any]) -> dict:
    """Evaluate predeclared E31 decision thresholds from `PHASE3_GPU_DECISION_RANGES["E31"]`."""
    vis_min = float(decision_ranges["decoder_visual_auroc_min"])
    corr_min = float(decision_ranges["attention_gap_corr_min"])
    sum_tol = float(decision_ranges["checks"]["attention_mass_sum_tol"])

    probes = e31_summary["visual_token_probe"]["decoder_visual_mean"]
    per_depth_en = {}
    per_depth_lang_mean = {}
    for depth in depths:
        dkey = str(depth)
        en_val = probes[dkey]["per_language"]["en"]["metrics"]["auroc"]["primary_macro"]["value"]
        mean_val = probes[dkey]["four_language_mean"]["metrics"]["auroc"]["primary_macro"]["value"]
        per_depth_en[dkey] = en_val
        per_depth_lang_mean[dkey] = mean_val

    mid_key = str(midpoint_depth)
    mid_en_auroc = per_depth_en[mid_key]
    mid_lang_mean_auroc = per_depth_lang_mean[mid_key]
    defined_en = [v for v in per_depth_en.values() if v is not None]
    max_en_auroc = max(defined_en) if defined_en else None

    corr_table = e31_summary["attention_gap_correlations"]
    best_abs_r = 0.0
    best_abs_rho = 0.0
    best_entry = None
    any_corr_ge_min = False

    def _scan_corr(obj, prefix):
        nonlocal best_abs_r, best_abs_rho, best_entry, any_corr_ge_min
        if not isinstance(obj, Mapping):
            return
        if "pearson_r" in obj and "spearman_rho" in obj:
            r_val = obj["pearson_r"]
            rho_val = obj["spearman_rho"]
            abs_r = abs(float(r_val)) if r_val is not None else 0.0
            abs_rho = abs(float(rho_val)) if rho_val is not None else 0.0
            if max(abs_r, abs_rho) >= corr_min:
                any_corr_ge_min = True
            if max(abs_r, abs_rho) >= max(best_abs_r, best_abs_rho):
                best_abs_r = abs_r
                best_abs_rho = abs_rho
                best_entry = {
                    "key": prefix,
                    "n": obj.get("n"),
                    "pearson_r": r_val,
                    "spearman_rho": rho_val,
                }
            return
        for k, v in obj.items():
            _scan_corr(v, f"{prefix}.{k}" if prefix else str(k))

    _scan_corr(corr_table, "")

    return {
        "thresholds": {
            "decoder_visual_auroc_min": vis_min,
            "attention_gap_corr_min": corr_min,
            "attention_mass_sum_tol": sum_tol,
        },
        "attention_mass_sum_check_passed": bool(e31_summary["attention_mass_check"]["passed"]),
        "decoder_visual_probe": {
            "midpoint_depth": int(midpoint_depth),
            "midpoint_en_auroc": mid_en_auroc,
            "midpoint_four_language_mean_auroc": mid_lang_mean_auroc,
            "max_en_auroc_across_depths": max_en_auroc,
            "per_depth_en_auroc": per_depth_en,
            "per_depth_four_language_mean_auroc": per_depth_lang_mean,
            "visual_tokens_retain_semantics_at_midpoint": bool(
                mid_en_auroc is not None and mid_en_auroc >= vis_min
            ),
            "visual_tokens_retain_semantics_any_depth": bool(
                max_en_auroc is not None and max_en_auroc >= vis_min
            ),
        },
        "attention_gap_correlation": {
            "attention_correlates_with_gap": bool(any_corr_ge_min),
            "strongest_correlation": best_entry,
        },
    }


def _read_jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@contract
def reassemble_e39_e31_release(
    resolved_root: Path,
    protocol: Mapping[str, Any],
    gpu_plan,
    model: str,
    release_ref,
    *,
    attention_sum_tol: float = 1e-4,
):
    """Authenticate Stage B2 `E39+E31` release and reassemble all chunks in `screen_ids` order."""
    import numpy as np

    require_open_split(E39_E31_SPLIT)
    check(model in MODELS, f"Unsupported MODEL selector {model!r}")

    rel_ref, rel_body, rel_marker = verify_e39_e31_release(
        resolved_root,
        protocol,
        release_ref,
        expected_plan_identity=gpu_plan["plan_identity"],
        expected_model=model,
    )
    check(
        rel_marker["dataset_fingerprint"] == gpu_plan["dataset_fingerprint"],
        "Stage B2 release dataset_fingerprint mismatch with phase3_gpu_plan",
    )
    check(plain(rel_body["plan"]) == plain(gpu_plan.reference), "Release plan ref mismatch with phase3_gpu_plan")

    screen_ids = tuple(gpu_plan.binding.stage4_plan["screen_ids"])
    n_screen = len(screen_ids)
    check(n_screen == int(rel_body["total_images"]), "Release total_images mismatch with stage4_plan screen_ids")
    id_to_pos = {iid: idx for idx, iid in enumerate(screen_ids)}

    depths = [int(d) for d in rel_body["depths"]]
    expected_depths = [int(d) for d in gpu_plan["models"][model]["depths"]]
    check(depths == expected_depths, f"Release depths {depths} mismatch gpu_plan depths {expected_depths}")
    dimension = int(rel_body["dimension"])
    check(dimension > 0, "Release dimension must be positive")
    specs = expected_array_specs(depths, dimension)

    e39_features = {
        cond: {
            d: {
                site: {lang: np.empty((n_screen, dimension), dtype=np.float32) for lang in LANGUAGES}
                for site in E39_SITES
            }
            for d in depths
        }
        for cond in CONDITIONS
    }
    e31_visual = {
        d: {lang: np.empty((n_screen, dimension), dtype=np.float32) for lang in LANGUAGES}
        for d in depths
    }
    e31_attention = {
        att_site: {
            d: {lang: np.empty((n_screen, 3), dtype=np.float32) for lang in LANGUAGES}
            for d in depths
        }
        for att_site in E31_ATTENTION_SITES
    }
    token_selections = {
        cond: {lang: [None] * n_screen for lang in LANGUAGES}
        for cond in CONDITIONS
    }
    rows_by_lang = {lang: [None] * n_screen for lang in LANGUAGES}
    seen_positions = {lang: set() for lang in LANGUAGES}

    for entry in rel_body["chunks"]:
        chunk_dir = within(resolved_root, entry["ref"]["path"]).parent
        chunk_rows = _read_jsonl(chunk_dir / "rows.jsonl")
        n_views = int(entry["n_views_per_condition"])
        check(len(chunk_rows) == n_views, f"Chunk {entry['chunk_id']} rows.jsonl length mismatch")

        src_indices = {lang: [] for lang in LANGUAGES}
        dst_positions = {lang: [] for lang in LANGUAGES}
        for r_idx, row in enumerate(chunk_rows):
            row_split = row.get("split", E39_E31_SPLIT)
            require_open_split(row_split)
            check(row_split == E39_E31_SPLIT, f"Non-SCREEN row split in chunk {entry['chunk_id']}: {row_split!r}")
            iid = row["image_id"]
            lang = row["language"]
            check(iid in id_to_pos, f"Unknown SCREEN image_id {iid!r} in chunk {entry['chunk_id']}")
            check(lang in LANGUAGES, f"Invalid language {lang!r} in chunk {entry['chunk_id']}")
            check(int(row.get("caption_draw_key", 0)) == 0, "Only caption_draw_key == 0 is allowed")
            pos = id_to_pos[iid]
            check(pos not in seen_positions[lang], f"Duplicate view ({iid}, {lang}) across chunks")
            seen_positions[lang].add(pos)

            src_indices[lang].append(r_idx)
            dst_positions[lang].append(pos)
            rows_by_lang[lang][pos] = {
                "image_id": iid,
                "language": lang,
                "split": row_split,
                "caption_id": row["caption_id"],
                "caption_draw_key": 0,
            }
            sel_map = row.get("e39_token_selection")
            check(isinstance(sel_map, Mapping), f"Missing e39_token_selection in chunk {entry['chunk_id']}")
            for cond in CONDITIONS:
                _validate_token_selection_entry(sel_map[cond])
                token_selections[cond][lang][pos] = plain(sel_map[cond])

        src_np = {lang: np.asarray(src_indices[lang], dtype=np.intp) for lang in LANGUAGES}
        dst_np = {lang: np.asarray(dst_positions[lang], dtype=np.intp) for lang in LANGUAGES}

        for cond in CONDITIONS:
            for d in depths:
                for site in E39_SITES:
                    arr = np.load(chunk_dir / f"{cond}__d{d}__{site}.npy", allow_pickle=False)
                    for lang in LANGUAGES:
                        if src_np[lang].size:
                            e39_features[cond][d][site][lang][dst_np[lang]] = arr[src_np[lang]]

        for d in depths:
            vis_arr = np.load(chunk_dir / f"image_caption__d{d}__decoder_visual_mean.npy", allow_pickle=False)
            for lang in LANGUAGES:
                if src_np[lang].size:
                    e31_visual[d][lang][dst_np[lang]] = vis_arr[src_np[lang]]
            for att_site in E31_ATTENTION_SITES:
                att_arr = np.load(chunk_dir / f"image_caption__d{d}__{att_site}.npy", allow_pickle=False)
                for lang in LANGUAGES:
                    if src_np[lang].size:
                        e31_attention[att_site][d][lang][dst_np[lang]] = att_arr[src_np[lang]]

    expected_set = set(range(n_screen))
    for lang in LANGUAGES:
        check(seen_positions[lang] == expected_set, f"Incomplete SCREEN coverage after reassembly for {lang}")

    max_att_sum_err = 0.0
    for att_site in E31_ATTENTION_SITES:
        for d in depths:
            for lang in LANGUAGES:
                arr = e31_attention[att_site][d][lang]
                check(np.isfinite(arr).all() and bool((arr >= 0.0).all()), f"Invalid attention array {att_site}/{d}/{lang}")
                err = float(np.max(np.abs(arr.astype(np.float64).sum(axis=1) - 1.0)))
                max_att_sum_err = max(max_att_sum_err, err)
                check(
                    err <= attention_sum_tol,
                    f"Attention mass row sum tolerance exceeded for {att_site}/d{d}/{lang}: {err} > {attention_sum_tol}",
                )

    return {
        "release_ref": plain(rel_ref),
        "release_body": plain(rel_body),
        "screen_ids": screen_ids,
        "depths": depths,
        "midpoint_depth": int(rel_body["midpoint_depth"]),
        "dimension": dimension,
        "n_arrays_per_chunk": len(specs),
        "e39_features": e39_features,
        "e31_visual": e31_visual,
        "e31_attention": e31_attention,
        "token_selections": token_selections,
        "rows_by_lang": rows_by_lang,
        "attention_mass_check": {
            "passed": True,
            "max_row_sum_abs_error": float(max_att_sum_err),
            "tol": float(attention_sum_tol),
        },
    }


def _resolve_vlm_groups_and_blocks(binding, model: str, depths):
    """Resolve Stage 2 group IDs and Stage 4 SCREEN prediction block entries for one model."""
    fit_groups = binding.stage4_plan.fit["groups"]
    stage4_blocks = binding.stage4_plan["blocks"]
    stage4_preds = binding.stage4_release["predictions"]

    decoder_groups = {}
    image_only_groups = {}
    for gid, group in fit_groups.items():
        if group.get("source") != "vlm" or group.get("model") != model:
            continue
        cond = group.get("condition")
        site = group.get("site")
        lang = group.get("language")
        if cond in CONDITIONS and lang in LANGUAGES:
            for depth in depths:
                for pool in POOLINGS:
                    if site == f"decoder:{depth}:{pool}":
                        key = (int(depth), cond, pool, lang)
                        check(key not in decoder_groups, f"Duplicate Stage 2 group for {key}")
                        decoder_groups[key] = gid
        elif cond == "image_only" and lang is None:
            for stage in ("encoder", "projector"):
                if site == f"{stage}:None:visual_mean":
                    check(stage not in image_only_groups, f"Duplicate Stage 2 image_only group for {model}/{stage}")
                    image_only_groups[stage] = gid

    for depth in depths:
        for cond in CONDITIONS:
            for pool in POOLINGS:
                for lang in LANGUAGES:
                    check(
                        (int(depth), cond, pool, lang) in decoder_groups,
                        f"Missing Stage 2 VLM group for {model}/{depth}/{cond}/{pool}/{lang}",
                    )
    for stage in ("encoder", "projector"):
        check(stage in image_only_groups, f"Missing Stage 2 VLM image_only group for {model}/{stage}")

    gid_to_block = {}
    for bid, block in stage4_blocks.items():
        if block.get("split") != E39_E31_SPLIT or block.get("owner") != model:
            continue
        tgid = block["target_group"]
        check(tgid not in gid_to_block, f"Duplicate Stage 4 SCREEN block for target_group {tgid}")
        routes = {}
        for r in block["routes"]:
            src_gid = r["source_group"]
            src_lang = fit_groups[src_gid]["language"]
            routes[src_lang] = r["route_id"]
        gid_to_block[tgid] = {
            "block_id": bid,
            "prediction_ref": plain(stage4_preds[bid]),
            "routes": routes,
        }

    for gid in list(decoder_groups.values()) + list(image_only_groups.values()):
        check(gid in gid_to_block, f"Missing Stage 4 SCREEN prediction block for group {gid}")

    return decoder_groups, image_only_groups, gid_to_block


def _align_rows_to_screen_ids(rows, screen_ids: tuple, label: str):
    import numpy as np

    id_to_row_idx = {}
    for idx, row in enumerate(rows):
        check(row["split"] == E39_E31_SPLIT, f"Non-SCREEN row in {label}: {row['split']!r}")
        iid = row["image_id"]
        check(iid not in id_to_row_idx, f"Duplicate image_id {iid!r} in {label}")
        id_to_row_idx[iid] = idx
    check(set(id_to_row_idx.keys()) == set(screen_ids), f"SCREEN image_id mismatch in {label}")
    return np.asarray([id_to_row_idx[iid] for iid in screen_ids], dtype=np.intp)


def _score_feature_matrix(matrix, coef, intercept, stats, estimable):
    """Score float64 features identically to Stage 4."""
    import numpy as np
    from src.stage3.engine import transform

    logits = np.asarray(transform(matrix, stats) @ coef.T, dtype=np.float64) + intercept
    check(logits.dtype == np.float64 and bool(np.isfinite(logits).all()), "Nonfinite scored logits")
    est_bool = np.asarray(estimable, dtype=bool)
    check(logits.shape[1] == est_bool.shape[0], "Logits/estimable column mismatch")
    check(not bool(np.any(logits[:, ~est_bool])), "Nonzero logits on unestimable columns")
    return logits


@contract
def verify_stage2_draw0_parity(
    binding,
    model: str,
    depths,
    screen_ids: tuple,
    e39_features,
    *,
    tol: float = 1e-4,
):
    """Verify Stage 2 SCREEN feature parity and `en->en` Stage 4 logit parity (`<= 1e-4`)."""
    import numpy as np
    from src.stage3.data import _read_verified_group
    from src.stage4.metrics import paired_image_alignment
    from src.stage4.predictions import _selected_parameters, load_prediction_block

    decoder_groups, image_only_groups, gid_to_block = _resolve_vlm_groups_and_blocks(binding, model, depths)
    fit = binding.stage4_plan.fit
    store = binding.stage4_plan.store

    stage3_heads = {}
    stage4_screen_blocks = {}
    feature_cells = {}
    logit_cells = {}
    max_feat_diff = 0.0
    max_logit_diff = 0.0

    for depth in depths:
        dkey = str(depth)
        feature_cells[dkey] = {}
        logit_cells[dkey] = {}
        for cond in CONDITIONS:
            feature_cells[dkey][cond] = {}
            logit_cells[dkey][cond] = {}
            for pool in POOLINGS:
                feature_cells[dkey][cond][pool] = {}
                for lang in LANGUAGES:
                    gid = decoder_groups[(int(depth), cond, pool, lang)]
                    s2_rows, s2_mat = _read_verified_group(fit, store, gid, E39_E31_SPLIT)
                    align_idx = _align_rows_to_screen_ids(
                        s2_rows, screen_ids, f"stage2/{model}/d{depth}/{cond}/{pool}/{lang}"
                    )
                    s2_aligned = np.asarray(s2_mat[align_idx], dtype=np.float64)
                    e39_mat = np.asarray(e39_features[cond][int(depth)][pool][lang], dtype=np.float64)
                    check(
                        e39_mat.shape == s2_aligned.shape,
                        f"Feature shape mismatch for d{depth}/{cond}/{pool}/{lang}: {e39_mat.shape} vs {s2_aligned.shape}",
                    )
                    diff = float(np.max(np.abs(e39_mat - s2_aligned)))
                    max_feat_diff = max(max_feat_diff, diff)
                    check(
                        diff <= tol,
                        f"Stage 2 draw-0 feature parity failed for {model}/d{depth}/{cond}/{pool}/{lang}: "
                        f"max_abs_diff={diff} > {tol}",
                    )
                    feature_cells[dkey][cond][pool][lang] = {
                        "group_id": gid,
                        "max_abs_diff": diff,
                        "passed": True,
                    }

                    stage3_heads[(int(depth), cond, pool, lang)] = _selected_parameters(binding.stage4_plan, gid)
                    blk_entry = gid_to_block[gid]
                    loaded_preds = load_prediction_block(
                        binding.stage4_plan,
                        blk_entry["prediction_ref"],
                        expected_block=blk_entry["block_id"],
                    )
                    stage4_screen_blocks[(int(depth), cond, pool, lang)] = {
                        "entry": blk_entry,
                        "preds": {src_l: loaded_preds[rid] for src_l, rid in blk_entry["routes"].items()},
                    }

                # In-memory en->en matched logit self-check against accepted Stage 4 SCREEN logits.
                coef_en, intercept_en, est_en, _, stats_en = stage3_heads[(int(depth), cond, pool, "en")]
                e39_en = e39_features[cond][int(depth)][pool]["en"]
                scored_en = _score_feature_matrix(e39_en, coef_en, intercept_en, stats_en, est_en)

                pred_en = stage4_screen_blocks[(int(depth), cond, pool, "en")]["preds"]["en"]
                _, (aligned_pred,) = paired_image_alignment(binding.stage4_plan, pred_en)
                accepted_logits = np.asarray(pred_en.logits[aligned_pred], dtype=np.float64)
                check(
                    scored_en.shape == accepted_logits.shape,
                    f"en->en logit shape mismatch for d{depth}/{cond}/{pool}",
                )
                ldiff = float(np.max(np.abs(scored_en - accepted_logits)))
                max_logit_diff = max(max_logit_diff, ldiff)
                check(
                    ldiff <= tol,
                    f"en->en matched logit self-check failed for {model}/d{depth}/{cond}/{pool}: "
                    f"max_abs_diff={ldiff} > {tol}",
                )
                logit_cells[dkey][cond][pool] = {
                    "route": "en->en",
                    "max_abs_diff": ldiff,
                    "passed": True,
                }

    return {
        "parity_report": {
            "tol": float(tol),
            "feature_parity_cells_checked": len(depths) * len(CONDITIONS) * len(POOLINGS) * len(LANGUAGES),
            "feature_parity_all_passed": True,
            "max_feature_abs_diff": float(max_feat_diff),
            "feature_cells": feature_cells,
            "logit_self_checks_checked": len(depths) * len(CONDITIONS) * len(POOLINGS),
            "logit_self_checks_all_passed": True,
            "max_logit_abs_diff": float(max_logit_diff),
            "logit_cells": logit_cells,
        },
        "stage3_heads": stage3_heads,
        "stage4_screen_blocks": stage4_screen_blocks,
        "image_only_groups": image_only_groups,
        "gid_to_block": gid_to_block,
    }


def _evaluate_prediction_arrays(logits, estimable, y, known, counts, n_cats: int, total_draws: int) -> dict:
    """Evaluate 25-category point metrics & bootstrap draws for logits already aligned to `screen_ids`."""
    import numpy as np
    from src.stage4.metrics import binary_metrics, bootstrap_binary_metrics, probabilities

    probs = probabilities(logits)
    est_bool = np.asarray(estimable, dtype=bool)
    check(est_bool.shape == (n_cats,), f"Unexpected estimable shape: {est_bool.shape}")
    nan_draws = np.full(total_draws, np.nan, dtype="f8")

    points = {m: [] for m in REPORT_METRICS}
    draws = {m: [] for m in REPORT_METRICS}
    supports = []
    for col in range(n_cats):
        y_col = y[:, col]
        known_col = known[:, col]
        known_n = int(known_col.sum())
        pos_n = int(y_col[known_col].sum())
        neg_n = known_n - pos_n
        supports.append((known_n, pos_n, neg_n))
        if est_bool[col]:
            p_col = probs[:, col]
            bm = binary_metrics(y_col, p_col, known_col)
            bd = bootstrap_binary_metrics(y_col, p_col, known_col, counts)
            for m in REPORT_METRICS:
                points[m].append(bm["metrics"][m])
                draws[m].append(bd[m])
        else:
            for m in REPORT_METRICS:
                points[m].append(None)
                draws[m].append(nan_draws)

    return {
        "points": points,
        "draws": draws,
        "supports": supports,
        "unestimable": [not bool(est_bool[col]) for col in range(n_cats)],
    }


def _build_route_summary_bundle(
    *,
    raw_routes: Mapping[tuple, dict],
    categories,
    primary_indices,
    all_indices,
    minimum: int,
):
    """Aggregate 16 routes into 4-diagonal, 12-transfer, 12-gap, and per-route summaries + draw cache."""
    import numpy as np

    n_cats = len(categories)
    off_pairs = _off_diagonal_pairs()
    per_route = {}
    twelve_transfer = {}
    four_diagonal = {}
    twelve_gap = {}

    cache = {
        "twelve_transfer": {},
        "four_diagonal": {},
        "twelve_gap": {},
        "per_route_transfer": {},
        "per_route_gap": {},
        "supports_ref": None,
        "ok_counts": {},
    }

    supports_ref = None
    for metric in REPORT_METRICS:
        route_pm_vals = {}
        route_pm_draws = {}
        route_am_vals = {}
        route_am_draws = {}
        route_cat_vals = {}
        route_cat_draws = {}

        for s in LANGUAGES:
            for t in LANGUAGES:
                rkey = _route_key(s, t)
                rd = raw_routes[(s, t)]
                if supports_ref is None:
                    supports_ref = rd["supports"]
                pm, pm_d, am, am_d, _ = _summarize_macros_and_categories(
                    categories=categories,
                    primary_indices=primary_indices,
                    all_indices=all_indices,
                    cat_values=rd["points"][metric],
                    cat_draws=rd["draws"][metric],
                    cat_supports=rd["supports"],
                    cat_unestimable=rd["unestimable"],
                    minimum=minimum,
                    include_categories=False,
                )
                route_pm_vals[(s, t)] = pm["value"]
                route_pm_draws[(s, t)] = pm_d
                route_am_vals[(s, t)] = am["value"]
                route_am_draws[(s, t)] = am_d
                route_cat_vals[(s, t)] = rd["points"][metric]
                route_cat_draws[(s, t)] = rd["draws"][metric]

                r_entry = per_route.setdefault(
                    rkey,
                    {
                        "route": rkey,
                        "source_language": s,
                        "target_language": t,
                        "supervision_label": SUPERVISION_DIAGONAL if s == t else SUPERVISION_OFF_DIAGONAL,
                        "metrics": {},
                    },
                )
                r_entry["metrics"][metric] = {
                    "primary_macro": pm,
                    "all_categories_macro": am,
                }
                cache["per_route_transfer"].setdefault(metric, {})[rkey] = {
                    "pm_val": pm["value"],
                    "pm_draws": pm_d,
                    "am_val": am["value"],
                    "am_draws": am_d,
                }

        ok_primary = sum(
            min(supports_ref[idx][1], supports_ref[idx][2]) >= minimum for idx in primary_indices
        )
        ok_all = sum(
            min(supports_ref[idx][1], supports_ref[idx][2]) >= minimum for idx in all_indices
        )
        cache["supports_ref"] = supports_ref
        cache["ok_counts"] = {"primary": ok_primary, "all": ok_all}

        def _agg_subset(pairs, *, is_gap: bool = False):
            if not is_gap:
                pm_vals = [route_pm_vals[p] for p in pairs]
                pm_draws = np.mean(np.stack([route_pm_draws[p] for p in pairs], axis=0), axis=0)
                am_vals = [route_am_vals[p] for p in pairs]
                am_draws = np.mean(np.stack([route_am_draws[p] for p in pairs], axis=0), axis=0)
                c_vals = [_mean_optional([route_cat_vals[p][c] for p in pairs]) for c in range(n_cats)]
                c_draws = [
                    np.mean(np.stack([route_cat_draws[p][c] for p in pairs], axis=0), axis=0)
                    for c in range(n_cats)
                ]
                c_unest = [any(raw_routes[p]["unestimable"][c] for p in pairs) for c in range(n_cats)]
            else:
                pm_vals = []
                pm_draws_list = []
                am_vals = []
                am_draws_list = []
                c_vals_by_c = [[] for _ in range(n_cats)]
                c_draws_by_c = [[] for _ in range(n_cats)]
                for s, t in pairs:
                    rkey = _route_key(s, t)
                    rd_r = raw_routes[(s, t)]
                    rd_d = raw_routes[(t, t)]
                    g_c_vals = [
                        _gap_value(metric, rd_d["points"][metric][c], rd_r["points"][metric][c])
                        for c in range(n_cats)
                    ]
                    g_c_draws = [
                        _gap_draws(metric, rd_d["draws"][metric][c], rd_r["draws"][metric][c])
                        for c in range(n_cats)
                    ]
                    g_unest = [rd_d["unestimable"][c] or rd_r["unestimable"][c] for c in range(n_cats)]
                    g_pm, g_pm_d, g_am, g_am_d, _ = _summarize_macros_and_categories(
                        categories=categories,
                        primary_indices=primary_indices,
                        all_indices=all_indices,
                        cat_values=g_c_vals,
                        cat_draws=g_c_draws,
                        cat_supports=rd_r["supports"],
                        cat_unestimable=g_unest,
                        minimum=minimum,
                        include_categories=False,
                    )
                    per_route[rkey]["metrics"][metric]["gap"] = {
                        "primary_macro": g_pm,
                        "all_categories_macro": g_am,
                    }
                    cache["per_route_gap"].setdefault(metric, {})[rkey] = {
                        "pm_val": g_pm["value"],
                        "pm_draws": g_pm_d,
                        "am_val": g_am["value"],
                        "am_draws": g_am_d,
                    }
                    pm_vals.append(g_pm["value"])
                    pm_draws_list.append(g_pm_d)
                    am_vals.append(g_am["value"])
                    am_draws_list.append(g_am_d)
                    for c in range(n_cats):
                        c_vals_by_c[c].append(g_c_vals[c])
                        c_draws_by_c[c].append(g_c_draws[c])
                pm_draws = np.mean(np.stack(pm_draws_list, axis=0), axis=0)
                am_draws = np.mean(np.stack(am_draws_list, axis=0), axis=0)
                c_vals = [_mean_optional(c_vals_by_c[c]) for c in range(n_cats)]
                c_draws = [
                    np.mean(np.stack(c_draws_by_c[c], axis=0), axis=0) for c in range(n_cats)
                ]
                c_unest = [
                    any(raw_routes[p]["unestimable"][c] or raw_routes[(p[1], p[1])]["unestimable"][c] for p in pairs)
                    for c in range(n_cats)
                ]

            pm_val = _mean_optional(pm_vals)
            am_val = _mean_optional(am_vals)
            _, _, _, _, cats_table = _summarize_macros_and_categories(
                categories=categories,
                primary_indices=primary_indices,
                all_indices=all_indices,
                cat_values=c_vals,
                cat_draws=c_draws,
                cat_supports=supports_ref,
                cat_unestimable=c_unest,
                minimum=minimum,
                include_categories=True,
            )
            pm_summary = _macro_summary(
                pm_val,
                pm_draws,
                categories_count=len(primary_indices),
                defined_count=len(primary_indices) if pm_val is not None else 0,
                ok_count=ok_primary if pm_val is not None else 0,
                minimum=minimum,
            )
            am_summary = _macro_summary(
                am_val,
                am_draws,
                categories_count=len(all_indices),
                defined_count=len(all_indices) if am_val is not None else 0,
                ok_count=ok_all if am_val is not None else 0,
                minimum=minimum,
            )
            bundle = {
                "primary_macro": pm_summary,
                "all_categories_macro": am_summary,
                "categories": cats_table,
            }
            raw_info = {
                "pm_val": pm_val,
                "pm_draws": pm_draws,
                "am_val": am_val,
                "am_draws": am_draws,
            }
            return bundle, raw_info

        diag_bundle, diag_raw = _agg_subset([(t, t) for t in LANGUAGES], is_gap=False)
        xfer_bundle, xfer_raw = _agg_subset(off_pairs, is_gap=False)
        gap_bundle, gap_raw = _agg_subset(off_pairs, is_gap=True)

        four_diagonal[metric] = diag_bundle
        twelve_transfer[metric] = xfer_bundle
        twelve_gap[metric] = gap_bundle

        cache["four_diagonal"][metric] = diag_raw
        cache["twelve_transfer"][metric] = xfer_raw
        cache["twelve_gap"][metric] = gap_raw

    return {
        "twelve_route_mean_transfer": twelve_transfer,
        "four_diagonal_mean": four_diagonal,
        "twelve_route_mean_gap": twelve_gap,
        "per_route": per_route,
    }, cache


def _paired_gain_and_recovery(
    *,
    site_cache,
    base_cache,
    upper_cache,
    primary_count: int,
    all_count: int,
    minimum: int,
):
    """Compute paired gain vs `last_caption` and recovery fraction of `(caption_mean - last_caption)`."""
    ok_primary = site_cache["ok_counts"]["primary"]
    ok_all = site_cache["ok_counts"]["all"]
    off_pairs = _off_diagonal_pairs()

    def _pair_macros(metric, site_entry, base_entry, *, is_gap: bool = False):
        if is_gap:
            # Lower gap is better; gap_reduction = base_gap - site_gap
            p_val = _diff_opt(base_entry["pm_val"], site_entry["pm_val"])
            p_draws = base_entry["pm_draws"] - site_entry["pm_draws"]
            a_val = _diff_opt(base_entry["am_val"], site_entry["am_val"])
            a_draws = base_entry["am_draws"] - site_entry["am_draws"]
        else:
            p_val = _improvement_value(metric, base_entry["pm_val"], site_entry["pm_val"])
            p_draws = _improvement_draws(metric, base_entry["pm_draws"], site_entry["pm_draws"])
            a_val = _improvement_value(metric, base_entry["am_val"], site_entry["am_val"])
            a_draws = _improvement_draws(metric, base_entry["am_draws"], site_entry["am_draws"])
        return {
            "primary_macro": _macro_summary(
                p_val,
                p_draws,
                categories_count=primary_count,
                defined_count=primary_count if p_val is not None else 0,
                ok_count=ok_primary if p_val is not None else 0,
                minimum=minimum,
            ),
            "all_categories_macro": _macro_summary(
                a_val,
                a_draws,
                categories_count=all_count,
                defined_count=all_count if a_val is not None else 0,
                ok_count=ok_all if a_val is not None else 0,
                minimum=minimum,
            ),
            "_draws": (p_val, p_draws, a_val, a_draws),
        }

    gain_out = {
        "twelve_route_mean_transfer": {},
        "four_diagonal_mean": {},
        "twelve_route_mean_gap_reduction": {},
        "per_route_transfer": {},
        "per_route_gap_reduction": {},
    }
    rec_out = {
        "twelve_route_mean_transfer": {},
        "four_diagonal_mean": {},
        "twelve_route_mean_gap_reduction": {},
        "per_route_transfer": {},
    }
    gain_draws_cache = {
        "twelve_route_mean_transfer": {},
        "four_diagonal_mean": {},
        "twelve_route_mean_gap_reduction": {},
    }

    for metric in REPORT_METRICS:
        # Transfer
        g_xfer = _pair_macros(metric, site_cache["twelve_transfer"][metric], base_cache["twelve_transfer"][metric])
        u_xfer = _pair_macros(metric, upper_cache["twelve_transfer"][metric], base_cache["twelve_transfer"][metric])
        gain_draws_cache["twelve_route_mean_transfer"][metric] = g_xfer.pop("_draws")
        u_pm_val, u_pm_d, u_am_val, u_am_d = u_xfer.pop("_draws")
        g_pm_val, g_pm_d, g_am_val, g_am_d = gain_draws_cache["twelve_route_mean_transfer"][metric]
        gain_out["twelve_route_mean_transfer"][metric] = g_xfer
        rec_out["twelve_route_mean_transfer"][metric] = {
            "primary_macro": _compute_recovery_summary(
                num_val=g_pm_val,
                num_draws=g_pm_d,
                den_val=u_pm_val,
                den_draws=u_pm_d,
                categories_count=primary_count,
                ok_count=ok_primary,
                minimum=minimum,
            ),
            "all_categories_macro": _compute_recovery_summary(
                num_val=g_am_val,
                num_draws=g_am_d,
                den_val=u_am_val,
                den_draws=u_am_d,
                categories_count=all_count,
                ok_count=ok_all,
                minimum=minimum,
            ),
        }

        # Diagonal
        g_diag = _pair_macros(metric, site_cache["four_diagonal"][metric], base_cache["four_diagonal"][metric])
        u_diag = _pair_macros(metric, upper_cache["four_diagonal"][metric], base_cache["four_diagonal"][metric])
        gain_draws_cache["four_diagonal_mean"][metric] = g_diag.pop("_draws")
        ud_pm_val, ud_pm_d, ud_am_val, ud_am_d = u_diag.pop("_draws")
        gd_pm_val, gd_pm_d, gd_am_val, gd_am_d = gain_draws_cache["four_diagonal_mean"][metric]
        gain_out["four_diagonal_mean"][metric] = g_diag
        rec_out["four_diagonal_mean"][metric] = {
            "primary_macro": _compute_recovery_summary(
                num_val=gd_pm_val,
                num_draws=gd_pm_d,
                den_val=ud_pm_val,
                den_draws=ud_pm_d,
                categories_count=primary_count,
                ok_count=ok_primary,
                minimum=minimum,
            ),
            "all_categories_macro": _compute_recovery_summary(
                num_val=gd_am_val,
                num_draws=gd_am_d,
                den_val=ud_am_val,
                den_draws=ud_am_d,
                categories_count=all_count,
                ok_count=ok_all,
                minimum=minimum,
            ),
        }

        # Gap reduction
        g_gap = _pair_macros(
            metric, site_cache["twelve_gap"][metric], base_cache["twelve_gap"][metric], is_gap=True
        )
        u_gap = _pair_macros(
            metric, upper_cache["twelve_gap"][metric], base_cache["twelve_gap"][metric], is_gap=True
        )
        gain_draws_cache["twelve_route_mean_gap_reduction"][metric] = g_gap.pop("_draws")
        ug_pm_val, ug_pm_d, ug_am_val, ug_am_d = u_gap.pop("_draws")
        gg_pm_val, gg_pm_d, gg_am_val, gg_am_d = gain_draws_cache["twelve_route_mean_gap_reduction"][metric]
        gain_out["twelve_route_mean_gap_reduction"][metric] = g_gap
        rec_out["twelve_route_mean_gap_reduction"][metric] = {
            "primary_macro": _compute_recovery_summary(
                num_val=gg_pm_val,
                num_draws=gg_pm_d,
                den_val=ug_pm_val,
                den_draws=ug_pm_d,
                categories_count=primary_count,
                ok_count=ok_primary,
                minimum=minimum,
            ),
            "all_categories_macro": _compute_recovery_summary(
                num_val=gg_am_val,
                num_draws=gg_am_d,
                den_val=ug_am_val,
                den_draws=ug_am_d,
                categories_count=all_count,
                ok_count=ok_all,
                minimum=minimum,
            ),
        }

        # Per-route transfer & gap
        gain_out["per_route_transfer"][metric] = {}
        rec_out["per_route_transfer"][metric] = {}
        for s in LANGUAGES:
            for t in LANGUAGES:
                rkey = _route_key(s, t)
                gr = _pair_macros(
                    metric,
                    site_cache["per_route_transfer"][metric][rkey],
                    base_cache["per_route_transfer"][metric][rkey],
                )
                ur = _pair_macros(
                    metric,
                    upper_cache["per_route_transfer"][metric][rkey],
                    base_cache["per_route_transfer"][metric][rkey],
                )
                gr_draws = gr.pop("_draws")
                ur_draws = ur.pop("_draws")
                gain_out["per_route_transfer"][metric][rkey] = gr
                rec_out["per_route_transfer"][metric][rkey] = {
                    "primary_macro": _compute_recovery_summary(
                        num_val=gr_draws[0],
                        num_draws=gr_draws[1],
                        den_val=ur_draws[0],
                        den_draws=ur_draws[1],
                        categories_count=primary_count,
                        ok_count=ok_primary,
                        minimum=minimum,
                    ),
                    "all_categories_macro": _compute_recovery_summary(
                        num_val=gr_draws[2],
                        num_draws=gr_draws[3],
                        den_val=ur_draws[2],
                        den_draws=ur_draws[3],
                        categories_count=all_count,
                        ok_count=ok_all,
                        minimum=minimum,
                    ),
                }

        gain_out["per_route_gap_reduction"][metric] = {}
        for s, t in off_pairs:
            rkey = _route_key(s, t)
            ggr = _pair_macros(
                metric,
                site_cache["per_route_gap"][metric][rkey],
                base_cache["per_route_gap"][metric][rkey],
                is_gap=True,
            )
            ggr.pop("_draws")
            gain_out["per_route_gap_reduction"][metric][rkey] = ggr

    return gain_out, rec_out, gain_draws_cache


def compute_e39_analysis(
    *,
    depths,
    midpoint_depth: int,
    e39_features,
    token_selections,
    screen_ids: tuple,
    stage3_heads,
    y,
    known,
    counts,
    categories,
    primary_indices,
    all_indices,
    minimum: int,
    decision_ranges: Mapping[str, Any],
) -> dict:
    """Run full E39 sub-caption scoring & paired bootstrap analysis."""
    n_cats = len(categories)
    total_draws = int(counts.shape[0])
    primary_count = len(primary_indices)
    all_count = len(all_indices)

    cells_out = {}
    caches_by_cell = {}

    for depth in depths:
        dkey = str(depth)
        cells_out[dkey] = {}
        for cond in CONDITIONS:
            # 1. Score all 8 sites through both head pools across all 16 routes.
            head_pools_out = {}
            head_pool_caches = {}
            for head_pool in POOLINGS:
                sites_out = {}
                site_caches = {}
                for site in E39_SITES:
                    raw_routes = {}
                    for t in LANGUAGES:
                        x_t = e39_features[cond][int(depth)][site][t]
                        for s in LANGUAGES:
                            coef, intercept, est, _, stats = stage3_heads[(int(depth), cond, head_pool, s)]
                            logits = _score_feature_matrix(x_t, coef, intercept, stats, est)
                            raw_routes[(s, t)] = _evaluate_prediction_arrays(
                                logits, est, y, known, counts, n_cats, total_draws
                            )
                    bundle, cache = _build_route_summary_bundle(
                        raw_routes=raw_routes,
                        categories=categories,
                        primary_indices=primary_indices,
                        all_indices=all_indices,
                        minimum=minimum,
                    )
                    sites_out[site] = bundle
                    site_caches[site] = cache
                head_pools_out[head_pool] = {"sites": sites_out}
                head_pool_caches[head_pool] = site_caches

            # 2. Compute paired gain vs last_caption & recovery fraction under all 3 head modes.
            head_modes_out = {}
            mode_site_caches = {}
            mode_gain_caches = {}
            for mode in HEAD_MODES:
                if mode == "fixed_head_last":
                    base_c = head_pool_caches["last_caption"]["last_caption"]
                    upper_c = head_pool_caches["last_caption"]["caption_mean"]
                    site_map = {site: head_pool_caches["last_caption"][site] for site in E39_SITES}
                elif mode == "fixed_head_mean":
                    base_c = head_pool_caches["caption_mean"]["last_caption"]
                    upper_c = head_pool_caches["caption_mean"]["caption_mean"]
                    site_map = {site: head_pool_caches["caption_mean"][site] for site in E39_SITES}
                else:
                    base_c = head_pool_caches["last_caption"]["last_caption"]
                    upper_c = head_pool_caches["caption_mean"]["caption_mean"]
                    site_map = {
                        site: head_pool_caches[MATCHED_HEAD_FOR_SITE[site]][site] for site in E39_SITES
                    }
                mode_site_caches[mode] = site_map
                mode_gain_caches[mode] = {}

                sites_contrasts = {}
                for site in E39_SITES:
                    gain_out, rec_out, g_cache = _paired_gain_and_recovery(
                        site_cache=site_map[site],
                        base_cache=base_c,
                        upper_cache=upper_c,
                        primary_count=primary_count,
                        all_count=all_count,
                        minimum=minimum,
                    )
                    sites_contrasts[site] = {
                        "head_pool": (
                            "last_caption"
                            if mode == "fixed_head_last"
                            else "caption_mean"
                            if mode == "fixed_head_mean"
                            else MATCHED_HEAD_FOR_SITE[site]
                        ),
                        "gain_vs_last_caption": gain_out,
                        "recovery_fraction_of_caption_mean": rec_out,
                    }
                    mode_gain_caches[mode][site] = g_cache
                head_modes_out[mode] = {"sites": sites_contrasts}

            cells_out[dkey][cond] = {
                "depth": int(depth),
                "condition": cond,
                "head_pools": head_pools_out,
                "head_modes": head_modes_out,
            }
            caches_by_cell[(int(depth), cond)] = {
                "site_caches": mode_site_caches,
                "gain_caches": mode_gain_caches,
                "ok_counts": head_pool_caches["last_caption"]["last_caption"]["ok_counts"],
            }

    # 3. Paired `image_caption - caption_only` sub-caption profile difference across all 4 depths and 8 sites.
    condition_swap = {}
    for depth in depths:
        dkey = str(depth)
        ic_c = caches_by_cell[(int(depth), "image_caption")]
        co_c = caches_by_cell[(int(depth), "caption_only")]
        ok_primary = ic_c["ok_counts"]["primary"]
        ok_all = ic_c["ok_counts"]["all"]

        modes_diff = {}
        for mode in HEAD_MODES:
            sites_diff = {}
            for site in E39_SITES:
                abs_diff = {"twelve_route_mean_transfer": {}, "four_diagonal_mean": {}, "twelve_route_mean_gap": {}}
                gain_diff = {"twelve_route_mean_transfer": {}, "four_diagonal_mean": {}, "twelve_route_mean_gap_reduction": {}}
                for metric in REPORT_METRICS:
                    for key_out, cache_key in (
                        ("twelve_route_mean_transfer", "twelve_transfer"),
                        ("four_diagonal_mean", "four_diagonal"),
                        ("twelve_route_mean_gap", "twelve_gap"),
                    ):
                        e_ic = ic_c["site_caches"][mode][site][cache_key][metric]
                        e_co = co_c["site_caches"][mode][site][cache_key][metric]
                        p_val = _diff_opt(e_ic["pm_val"], e_co["pm_val"])
                        p_draws = e_ic["pm_draws"] - e_co["pm_draws"]
                        a_val = _diff_opt(e_ic["am_val"], e_co["am_val"])
                        a_draws = e_ic["am_draws"] - e_co["am_draws"]
                        abs_diff[key_out][metric] = {
                            "primary_macro": _macro_summary(
                                p_val,
                                p_draws,
                                categories_count=primary_count,
                                defined_count=primary_count if p_val is not None else 0,
                                ok_count=ok_primary if p_val is not None else 0,
                                minimum=minimum,
                            ),
                            "all_categories_macro": _macro_summary(
                                a_val,
                                a_draws,
                                categories_count=all_count,
                                defined_count=all_count if a_val is not None else 0,
                                ok_count=ok_all if a_val is not None else 0,
                                minimum=minimum,
                            ),
                        }

                    for g_key in ("twelve_route_mean_transfer", "four_diagonal_mean", "twelve_route_mean_gap_reduction"):
                        g_ic = ic_c["gain_caches"][mode][site][g_key][metric]
                        g_co = co_c["gain_caches"][mode][site][g_key][metric]
                        gp_val = _diff_opt(g_ic[0], g_co[0])
                        gp_draws = g_ic[1] - g_co[1]
                        ga_val = _diff_opt(g_ic[2], g_co[2])
                        ga_draws = g_ic[3] - g_co[3]
                        gain_diff[g_key][metric] = {
                            "primary_macro": _macro_summary(
                                gp_val,
                                gp_draws,
                                categories_count=primary_count,
                                defined_count=primary_count if gp_val is not None else 0,
                                ok_count=ok_primary if gp_val is not None else 0,
                                minimum=minimum,
                            ),
                            "all_categories_macro": _macro_summary(
                                ga_val,
                                ga_draws,
                                categories_count=all_count,
                                defined_count=all_count if ga_val is not None else 0,
                                ok_count=ok_all if ga_val is not None else 0,
                                minimum=minimum,
                            ),
                        }
                sites_diff[site] = {
                    "score_difference": abs_diff,
                    "gain_vs_last_caption_difference": gain_diff,
                }
            modes_diff[mode] = {"sites": sites_diff}
        condition_swap[dkey] = {
            "depth": int(depth),
            "description": "Paired image_caption - caption_only sub-caption profile difference on identical draws",
            "head_modes": modes_diff,
        }

    token_audit = summarize_e39_token_selection_audit(token_selections, screen_ids)
    summary_out = {
        "e39_sites": list(E39_SITES),
        "head_modes": list(HEAD_MODES),
        "matched_head_mapping": plain(MATCHED_HEAD_FOR_SITE),
        "cells": cells_out,
        "condition_swap_profiles": condition_swap,
        "token_selection_audit": token_audit,
    }
    summary_out["decision_summary"] = evaluate_e39_decisions(
        summary_out, depths, midpoint_depth, decision_ranges
    )
    return summary_out, caches_by_cell


def _compute_visual_token_invariance(depths, e31_visual, counts, n_screen: int) -> dict:
    """Measure cross-language cosine similarity & abs/L2 diffs of `decoder_visual_mean`."""
    import numpy as np

    lang_pairs = [("en", lang) for lang in LANGUAGES if lang != "en"] + [
        ("hi", "bn"),
        ("hi", "te"),
        ("bn", "te"),
    ]
    row_den = counts.sum(axis=1)
    by_depth = {}
    for depth in depths:
        dkey = str(depth)
        pairs_out = {}
        en_vs_non_en_cos = []
        en_vs_non_en_l2 = []
        en_vs_non_en_max_abs = 0.0

        for la, lb in lang_pairs:
            xa = np.asarray(e31_visual[int(depth)][la], dtype=np.float64)
            xb = np.asarray(e31_visual[int(depth)][lb], dtype=np.float64)
            na = np.linalg.norm(xa, axis=1)
            nb = np.linalg.norm(xb, axis=1)
            den = na * nb
            dots = np.sum(xa * xb, axis=1)
            cos = np.where(den > 1e-24, dots / np.where(den > 1e-24, den, 1.0), np.nan)
            valid = np.isfinite(cos)
            abs_diff = np.abs(xa - xb)
            l2_diff = np.linalg.norm(xa - xb, axis=1)
            max_abs = float(np.max(abs_diff))
            mean_l2 = float(np.mean(l2_diff))

            if bool(valid.all()):
                cos_draws = (counts @ cos) / row_den
                cos_mean = float(np.mean(cos))
                cos_std = float(np.std(cos))
                cos_med = float(np.median(cos))
                cos_min = float(np.min(cos))
                cos_max = float(np.max(cos))
                cos_iv = _interval_record(cos_draws)
            else:
                finite_cos = cos[valid]
                cos_mean = float(np.mean(finite_cos)) if finite_cos.size else None
                cos_std = float(np.std(finite_cos)) if finite_cos.size else None
                cos_med = float(np.median(finite_cos)) if finite_cos.size else None
                cos_min = float(np.min(finite_cos)) if finite_cos.size else None
                cos_max = float(np.max(finite_cos)) if finite_cos.size else None
                cos_iv = {"lower": None, "upper": None, "valid_draws": 0, "total_draws": int(counts.shape[0])}

            l2_draws = (counts @ l2_diff) / row_den
            pkey = f"{la}_vs_{lb}"
            pairs_out[pkey] = {
                "n_images": int(n_screen),
                "cosine_similarity": {
                    "mean": cos_mean,
                    "std": cos_std,
                    "min": cos_min,
                    "median": cos_med,
                    "max": cos_max,
                    "interval": cos_iv,
                },
                "max_abs_diff": max_abs,
                "mean_abs_diff": float(np.mean(abs_diff)),
                "mean_l2_diff": {
                    "mean": mean_l2,
                    "interval": _interval_record(l2_draws),
                },
            }
            if la == "en" and lb != "en":
                en_vs_non_en_cos.append(cos)
                en_vs_non_en_l2.append(l2_diff)
                en_vs_non_en_max_abs = max(en_vs_non_en_max_abs, max_abs)

        avg_cos = np.mean(np.stack(en_vs_non_en_cos, axis=0), axis=0)
        avg_l2 = np.mean(np.stack(en_vs_non_en_l2, axis=0), axis=0)
        by_depth[dkey] = {
            "depth": int(depth),
            "en_vs_non_en_mean": {
                "mean_cosine_similarity": float(np.mean(avg_cos)),
                "cosine_interval": _interval_record((counts @ avg_cos) / row_den),
                "max_abs_diff": float(en_vs_non_en_max_abs),
                "mean_l2_diff": float(np.mean(avg_l2)),
                "l2_interval": _interval_record((counts @ avg_l2) / row_den),
            },
            "pairs": pairs_out,
        }
    return by_depth


def _run_5fold_cv_probe(
    *,
    X,
    y,
    known,
    counts,
    categories,
    primary_indices,
    all_indices,
    minimum: int,
    fold_ids: list,
    n_folds: int = CV_FOLDS,
    C: float = CV_C,
):
    """Run deterministic 5-fold CV L2 logistic probe (`fold = image_idx % 5`, `C=1.0`)."""
    import numpy as np
    from src.stage3.engine import fit_heads, statistics, transform

    n_samples, _dim = X.shape
    n_cats = len(categories)
    total_draws = int(counts.shape[0])
    folds_np = np.asarray(fold_ids, dtype=np.int64)
    check(folds_np.shape == (n_samples,), "Fold assignment length mismatch")

    oof_logits = np.zeros((n_samples, n_cats), dtype=np.float64)
    fold_valid = np.zeros((n_folds, n_cats), dtype=bool)
    fold_sizes = []

    for k in range(n_folds):
        test_mask = folds_np == k
        train_mask = ~test_mask
        n_train = int(train_mask.sum())
        n_test = int(test_mask.sum())
        check(n_train > 0 and n_test > 0, f"Empty train/test fold {k} in 5-fold CV")
        fold_sizes.append({"fold": k, "n_train": n_train, "n_test": n_test})

        ones_train = np.ones(n_train, dtype=np.float64)
        stats_k = statistics(X[train_mask], ones_train)
        z_train = transform(X[train_mask], stats_k)
        z_test = transform(X[test_mask], stats_k)

        coef_k, intercept_k, valid_k, _ = fit_heads(z_train, y[train_mask], known[train_mask], ones_train, C)
        fold_valid[k] = valid_k
        fold_logits = np.asarray(z_test @ coef_k.T, dtype=np.float64) + intercept_k
        fold_logits[:, ~valid_k] = 0.0
        check(np.isfinite(fold_logits).all(), f"Nonfinite OOF logits in fold {k}")
        oof_logits[test_mask] = fold_logits

    # A category's OOF logits are estimable iff every fold's training split was estimable.
    oof_estimable = np.all(fold_valid, axis=0)
    oof_logits[:, ~oof_estimable] = 0.0

    raw = _evaluate_prediction_arrays(oof_logits, oof_estimable, y, known, counts, n_cats, total_draws)
    metrics_out = {}
    macro_cache = {}
    for metric in REPORT_METRICS:
        pm, pm_d, am, am_d, cats_table = _summarize_macros_and_categories(
            categories=categories,
            primary_indices=primary_indices,
            all_indices=all_indices,
            cat_values=raw["points"][metric],
            cat_draws=raw["draws"][metric],
            cat_supports=raw["supports"],
            cat_unestimable=raw["unestimable"],
            minimum=minimum,
            include_categories=True,
        )
        metrics_out[metric] = {
            "primary_macro": pm,
            "all_categories_macro": am,
            "categories": cats_table,
        }
        macro_cache[metric] = {
            "pm_val": pm["value"],
            "pm_draws": pm_d,
            "am_val": am["value"],
            "am_draws": am_d,
            "ok_primary": pm["support"]["ok_categories"],
            "ok_all": am["support"]["ok_categories"],
        }

    return {
        "supervision_label": SUPERVISION_SCREEN_CV,
        "n_folds": n_folds,
        "C": float(C),
        "fold_sizes": fold_sizes,
        "fold_valid_counts": {cat: int(fold_valid[:, idx].sum()) for idx, cat in enumerate(categories)},
        "metrics": metrics_out,
    }, macro_cache


def _compute_attention_summary(depths, e31_attention, counts, n_screen: int) -> dict:
    """Summarize `[visual_mass, caption_mass, system_mass]` and `caption_mean - last_caption` shift."""
    import numpy as np

    row_den = counts.sum(axis=1)
    by_depth = {}
    cache = {}  # (att_key, depth, lang, slot_idx) -> (point_mean, draw_means)

    def _stat_entry(vec):
        d_arr = (counts @ vec) / row_den
        return {
            "mean": float(np.mean(vec)),
            "std": float(np.std(vec)),
            "min": float(np.min(vec)),
            "median": float(np.median(vec)),
            "max": float(np.max(vec)),
            "interval": _interval_record(d_arr),
        }, float(np.mean(vec)), d_arr

    for depth in depths:
        dkey = str(depth)
        per_lang = {}
        for lang in LANGUAGES:
            att_last = np.asarray(e31_attention["attention_from_last_caption"][int(depth)][lang], dtype=np.float64)
            att_mean = np.asarray(e31_attention["attention_from_caption_mean"][int(depth)][lang], dtype=np.float64)
            att_shift = att_mean - att_last

            last_slots = {}
            mean_slots = {}
            shift_slots = {}
            for s_idx, slot_name in enumerate(ATTENTION_SLOTS):
                s_last, p_l, d_l = _stat_entry(att_last[:, s_idx])
                s_mean, p_m, d_m = _stat_entry(att_mean[:, s_idx])
                s_shift, p_s, d_s = _stat_entry(att_shift[:, s_idx])
                last_slots[slot_name] = s_last
                mean_slots[slot_name] = s_mean
                shift_slots[slot_name] = s_shift
                cache[("last", int(depth), lang, s_idx)] = (p_l, d_l)
                cache[("mean", int(depth), lang, s_idx)] = (p_m, d_m)
                cache[("shift", int(depth), lang, s_idx)] = (p_s, d_s)

            per_lang[lang] = {
                "n_images": int(n_screen),
                "attention_from_last_caption": last_slots,
                "attention_from_caption_mean": mean_slots,
                "caption_mean_minus_last_caption_shift": shift_slots,
            }

        lang_avg = {}
        for arm_key, label_out in (
            ("last", "attention_from_last_caption"),
            ("mean", "attention_from_caption_mean"),
            ("shift", "caption_mean_minus_last_caption_shift"),
        ):
            lang_avg[label_out] = {}
            for s_idx, slot_name in enumerate(ATTENTION_SLOTS):
                pts = [cache[(arm_key, int(depth), lang, s_idx)][0] for lang in LANGUAGES]
                drws = np.mean(
                    np.stack([cache[(arm_key, int(depth), lang, s_idx)][1] for lang in LANGUAGES], axis=0),
                    axis=0,
                )
                lang_avg[label_out][slot_name] = {
                    "mean": float(sum(pts) / len(pts)),
                    "interval": _interval_record(drws),
                }

        by_depth[dkey] = {
            "depth": int(depth),
            "four_language_mean": lang_avg,
            "per_language": per_lang,
        }
    return by_depth, cache


def _compute_attention_gap_correlations(depths, e39_caches, att_cache) -> dict:
    """Compute Pearson `r` & Spearman `rho` between visual attention mass and E01/G1 pooling gaps."""
    off_pairs = _off_diagonal_pairs()

    def _build_pair_table(metric: str):
        # 1. Across all 48 (depth, source_lang -> target_lang) off-diagonal pairs
        route_rows = []
        for depth in depths:
            ic_c = e39_caches[(int(depth), "image_caption")]["site_caches"]
            last_routes = ic_c["fixed_head_last"]["last_caption"]["per_route_gap"][metric]
            mean_routes = ic_c["fixed_head_mean"]["caption_mean"]["per_route_gap"][metric]
            fg_last_routes = ic_c["fixed_head_last"]["caption_mean"]["per_route_gap"][metric]

            for s, t in off_pairs:
                rkey = _route_key(s, t)
                g_last = last_routes[rkey]["pm_val"]
                g_mean = mean_routes[rkey]["pm_val"]
                g_fg_last = fg_last_routes[rkey]["pm_val"]
                g1_red = _diff_opt(g_last, g_mean)
                feat_gain = _diff_opt(g_last, g_fg_last)

                v_last_t = att_cache[("last", int(depth), t, 0)][0]
                v_mean_t = att_cache[("mean", int(depth), t, 0)][0]
                v_shift_t = att_cache[("shift", int(depth), t, 0)][0]
                v_last_s = att_cache[("last", int(depth), s, 0)][0]
                v_mean_s = att_cache[("mean", int(depth), s, 0)][0]
                v_shift_s = att_cache[("shift", int(depth), s, 0)][0]
                route_rows.append({
                    "depth": int(depth),
                    "source_language": s,
                    "target_language": t,
                    "G_last": g_last,
                    "G_mean": g_mean,
                    "G1_reduction": g1_red,
                    "feature_gain_fixed_head_last": feat_gain,
                    "vis_last_target": v_last_t,
                    "vis_mean_target": v_mean_t,
                    "vis_shift_target": v_shift_t,
                    "vis_last_source": v_last_s,
                    "vis_mean_source": v_mean_s,
                    "vis_shift_source": v_shift_s,
                    "vis_last_target_minus_source": v_last_t - v_last_s,
                    "vis_mean_target_minus_source": v_mean_t - v_mean_s,
                })

        # 2. Across 16 (depth, target_lang) cells (averaging the 3 incoming routes to target_lang)
        target_rows = []
        for depth in depths:
            for t in LANGUAGES:
                incoming = [r for r in route_rows if r["depth"] == int(depth) and r["target_language"] == t]
                target_rows.append({
                    "depth": int(depth),
                    "target_language": t,
                    "G_last": _mean_optional([r["G_last"] for r in incoming]),
                    "G_mean": _mean_optional([r["G_mean"] for r in incoming]),
                    "G1_reduction": _mean_optional([r["G1_reduction"] for r in incoming]),
                    "feature_gain_fixed_head_last": _mean_optional([r["feature_gain_fixed_head_last"] for r in incoming]),
                    "vis_last_target": att_cache[("last", int(depth), t, 0)][0],
                    "vis_mean_target": att_cache[("mean", int(depth), t, 0)][0],
                    "vis_shift_target": att_cache[("shift", int(depth), t, 0)][0],
                })

        def _corr_grid(rows, att_fields, gap_fields):
            grid = {}
            for af in att_fields:
                grid[af] = {}
                for gf in gap_fields:
                    grid[af][gf] = pearson_and_spearman([r[af] for r in rows], [r[gf] for r in rows])
            return grid

        gap_fields = ("G_last", "G_mean", "G1_reduction", "feature_gain_fixed_head_last")
        route_att_fields = (
            "vis_last_target",
            "vis_mean_target",
            "vis_shift_target",
            "vis_last_source",
            "vis_mean_source",
            "vis_shift_source",
            "vis_last_target_minus_source",
            "vis_mean_target_minus_source",
        )
        target_att_fields = ("vis_last_target", "vis_mean_target", "vis_shift_target")

        non_en_target_rows = [r for r in target_rows if r["target_language"] != "en"]
        return {
            "by_depth_and_route": {
                "n_points": len(route_rows),
                "correlations": _corr_grid(route_rows, route_att_fields, gap_fields),
            },
            "by_depth_and_target_lang": {
                "n_points": len(target_rows),
                "correlations": _corr_grid(target_rows, target_att_fields, gap_fields),
            },
            "by_depth_and_non_en_target_lang": {
                "n_points": len(non_en_target_rows),
                "correlations": _corr_grid(non_en_target_rows, target_att_fields, gap_fields),
            },
        }

    return {
        "auroc": _build_pair_table("auroc"),
        "nll": _build_pair_table("nll"),
    }


def compute_e31_analysis(
    *,
    binding,
    model: str,
    depths,
    midpoint_depth: int,
    screen_ids: tuple,
    e31_visual,
    e31_attention,
    attention_mass_check: dict,
    image_only_groups,
    gid_to_block,
    e39_caches,
    y,
    known,
    counts,
    categories,
    primary_indices,
    all_indices,
    minimum: int,
    decision_ranges: Mapping[str, Any],
) -> dict:
    """Run full E31 cross-language invariance, 5-fold CV probe, and attention-gap analysis."""
    import numpy as np
    from src.stage3.data import _read_verified_group
    from src.stage4.metrics import paired_image_alignment
    from src.stage4.predictions import load_prediction_block

    n_screen = len(screen_ids)
    n_cats = len(categories)
    total_draws = int(counts.shape[0])
    fit = binding.stage4_plan.fit
    store = binding.stage4_plan.store
    fold_ids = assign_screen_cv_folds(n_screen, CV_FOLDS)

    # 1. Cross-language decoder_visual_mean invariance
    invariance = _compute_visual_token_invariance(depths, e31_visual, counts, n_screen)

    # 2. 5-fold CV Semantic Probe on encoder -> projector -> decoder:d1..d4
    stage2_image_only_cv = {}
    stage4_image_only_ref = {}
    stage2_cv_caches = {}

    for stage in ("encoder", "projector"):
        gid = image_only_groups[stage]
        rows_s, mat_s = _read_verified_group(fit, store, gid, E39_E31_SPLIT)
        align_idx = _align_rows_to_screen_ids(rows_s, screen_ids, f"stage2/{model}/{stage}:None:visual_mean")
        x_aligned = np.asarray(mat_s[align_idx], dtype=np.float64)
        cv_res, cv_cache = _run_5fold_cv_probe(
            X=x_aligned,
            y=y,
            known=known,
            counts=counts,
            categories=categories,
            primary_indices=primary_indices,
            all_indices=all_indices,
            minimum=minimum,
            fold_ids=fold_ids,
        )
        stage2_image_only_cv[stage] = cv_res
        stage2_cv_caches[stage] = cv_cache

        # Stage 4 FIT-trained reference
        blk_entry = gid_to_block[gid]
        preds = load_prediction_block(
            binding.stage4_plan,
            blk_entry["prediction_ref"],
            expected_block=blk_entry["block_id"],
        )
        ref_pred = next(iter(preds.values()))
        _, (aligned_pred,) = paired_image_alignment(binding.stage4_plan, ref_pred)
        raw_ref = _evaluate_prediction_arrays(
            ref_pred.logits[aligned_pred], ref_pred.estimable, y, known, counts, n_cats, total_draws
        )
        ref_metrics = {}
        for metric in REPORT_METRICS:
            pm, _, am, _, cats_table = _summarize_macros_and_categories(
                categories=categories,
                primary_indices=primary_indices,
                all_indices=all_indices,
                cat_values=raw_ref["points"][metric],
                cat_draws=raw_ref["draws"][metric],
                cat_supports=raw_ref["supports"],
                cat_unestimable=raw_ref["unestimable"],
                minimum=minimum,
                include_categories=True,
            )
            ref_metrics[metric] = {
                "primary_macro": pm,
                "all_categories_macro": am,
                "categories": cats_table,
            }
        stage4_image_only_ref[stage] = {
            "supervision_label": "within_context_source_fit",
            "block_id": blk_entry["block_id"],
            "prediction_ref": blk_entry["prediction_ref"],
            "metrics": ref_metrics,
        }

    decoder_cv_by_depth = {}
    decoder_cv_caches = {}
    for depth in depths:
        dkey = str(depth)
        per_lang_cv = {}
        lang_caches = {}
        for lang in LANGUAGES:
            x_dec = np.asarray(e31_visual[int(depth)][lang], dtype=np.float64)
            cv_res, cv_cache = _run_5fold_cv_probe(
                X=x_dec,
                y=y,
                known=known,
                counts=counts,
                categories=categories,
                primary_indices=primary_indices,
                all_indices=all_indices,
                minimum=minimum,
                fold_ids=fold_ids,
            )
            per_lang_cv[lang] = cv_res
            lang_caches[lang] = cv_cache
        decoder_cv_caches[int(depth)] = lang_caches

        # 4-language mean summary on identical bootstrap draws
        four_lang_metrics = {}
        for metric in REPORT_METRICS:
            pm_vals = [lang_caches[lang][metric]["pm_val"] for lang in LANGUAGES]
            pm_draws = np.mean(np.stack([lang_caches[lang][metric]["pm_draws"] for lang in LANGUAGES], axis=0), axis=0)
            am_vals = [lang_caches[lang][metric]["am_val"] for lang in LANGUAGES]
            am_draws = np.mean(np.stack([lang_caches[lang][metric]["am_draws"] for lang in LANGUAGES], axis=0), axis=0)
            pm_val = _mean_optional(pm_vals)
            am_val = _mean_optional(am_vals)
            ok_p = min(lang_caches[lang][metric]["ok_primary"] for lang in LANGUAGES)
            ok_a = min(lang_caches[lang][metric]["ok_all"] for lang in LANGUAGES)
            four_lang_metrics[metric] = {
                "primary_macro": _macro_summary(
                    pm_val,
                    pm_draws,
                    categories_count=len(primary_indices),
                    defined_count=len(primary_indices) if pm_val is not None else 0,
                    ok_count=ok_p if pm_val is not None else 0,
                    minimum=minimum,
                ),
                "all_categories_macro": _macro_summary(
                    am_val,
                    am_draws,
                    categories_count=len(all_indices),
                    defined_count=len(all_indices) if am_val is not None else 0,
                    ok_count=ok_a if am_val is not None else 0,
                    minimum=minimum,
                ),
            }
        decoder_cv_by_depth[dkey] = {
            "depth": int(depth),
            "four_language_mean": {"metrics": four_lang_metrics},
            "per_language": per_lang_cv,
        }

    # Paired 5-fold CV trajectory contrasts on identical bootstrap draws
    def _pair_cv(cache_a, cache_b):
        out = {}
        for metric in REPORT_METRICS:
            ea = cache_a[metric]
            eb = cache_b[metric]
            p_val = _diff_opt(ea["pm_val"], eb["pm_val"])
            p_draws = ea["pm_draws"] - eb["pm_draws"]
            a_val = _diff_opt(ea["am_val"], eb["am_val"])
            a_draws = ea["am_draws"] - eb["am_draws"]
            ok_p = min(ea["ok_primary"], eb["ok_primary"])
            ok_a = min(ea["ok_all"], eb["ok_all"])
            out[metric] = {
                "primary_macro": _macro_summary(
                    p_val,
                    p_draws,
                    categories_count=len(primary_indices),
                    defined_count=len(primary_indices) if p_val is not None else 0,
                    ok_count=ok_p if p_val is not None else 0,
                    minimum=minimum,
                ),
                "all_categories_macro": _macro_summary(
                    a_val,
                    a_draws,
                    categories_count=len(all_indices),
                    defined_count=len(all_indices) if a_val is not None else 0,
                    ok_count=ok_a if a_val is not None else 0,
                    minimum=minimum,
                ),
            }
        return out

    trajectory_contrasts = {
        "projector_minus_encoder": _pair_cv(stage2_cv_caches["projector"], stage2_cv_caches["encoder"]),
        "decoder_en_minus_projector": {
            str(d): _pair_cv(decoder_cv_caches[int(d)]["en"], stage2_cv_caches["projector"])
            for d in depths
        },
        "decoder_en_minus_encoder": {
            str(d): _pair_cv(decoder_cv_caches[int(d)]["en"], stage2_cv_caches["encoder"])
            for d in depths
        },
        "decoder_final_minus_first_en": _pair_cv(
            decoder_cv_caches[int(depths[-1])]["en"],
            decoder_cv_caches[int(depths[0])]["en"],
        ),
    }

    # 3. Attention mass summaries & gap correlations
    att_summary, att_cache = _compute_attention_summary(depths, e31_attention, counts, n_screen)
    att_corrs = _compute_attention_gap_correlations(depths, e39_caches, att_cache)

    e31_out = {
        "e31_sites": list(E31_SITES),
        "attention_mass_check": attention_mass_check,
        "visual_token_invariance": invariance,
        "visual_token_probe": {
            "protocol": f"{CV_FOLDS}fold_cv_fold_eq_idx_mod_{CV_FOLDS}_C_{CV_C}",
            "n_folds": CV_FOLDS,
            "C": CV_C,
            "fold_assignment_sample": fold_ids[:min(15, len(fold_ids))],
            "stage2_5fold_cv": stage2_image_only_cv,
            "stage4_fit_reference": stage4_image_only_ref,
            "decoder_visual_mean": decoder_cv_by_depth,
            "trajectory_contrasts": trajectory_contrasts,
        },
        "attention_mass": att_summary,
        "attention_gap_correlations": att_corrs,
    }
    e31_out["decision_summary"] = evaluate_e31_decisions(
        e31_out, depths, midpoint_depth, decision_ranges
    )
    return e31_out


@contract
def run_e39_e31(
    root,
    protocol,
    model: str,
    producer: Mapping[str, Any],
    *,
    plan_ref=None,
    release_ref=None,
    resume_analysis_ref=None,
    run_id: Optional[str] = None,
    progress=None,
):
    """Execute Phase 3 Stage B3 Worker 1 (`run_e39_e31`) for `qwen` or `gemma`."""
    check(model in MODELS, f"Unsupported MODEL selector {model!r}; expected one of {MODELS}")
    require_open_split(E39_E31_SPLIT)
    resolved_root = Path(root).resolve()
    check(resolved_root.is_dir(), f"PHASE1_ROOT must exist and be a directory: {resolved_root}")

    eff_plan_ref = plain(plan_ref) if plan_ref is not None else plain(CERTIFIED_STAGE_B1_GPU_PLAN)
    eff_release_ref = (
        plain(release_ref) if release_ref is not None else plain(CERTIFIED_STAGE_B2_RELEASES["e39_e31"][model])
    )

    total_units = 6
    t0 = time.monotonic()
    gpu_plan = load_phase3_gpu_plan(resolved_root, protocol, eff_plan_ref, bind=True)
    check(gpu_plan["confirm_excluded"] is True, "phase3_gpu_plan must enforce confirm_excluded=True")
    check(gpu_plan["e35_excluded"] is True, "phase3_gpu_plan must enforce e35_excluded=True")
    plan_protocol = plain(gpu_plan.protocol)
    binding = gpu_plan.binding
    check(binding is not None, "load_phase3_gpu_plan(..., bind=True) must bind Phase 1")

    categories = tuple(plan_protocol["category_order"])
    primary_categories = tuple(plan_protocol["primary_categories"])
    check(len(categories) == 25, f"Expected 25 ordered categories; got {len(categories)}")
    check(len(primary_categories) == 14, f"Expected 14 primary categories; got {len(primary_categories)}")
    cat_index = {cat: idx for idx, cat in enumerate(categories)}
    primary_indices = [cat_index[cat] for cat in primary_categories]
    all_indices = list(range(len(categories)))
    screen_minimum = int(plan_protocol["minimum_positive_and_negative"]["SCREEN"])

    _emit_progress(
        progress,
        "e39_e31_plan_verified",
        unit=f"{model}:init",
        completed=1,
        total=total_units,
        read_hash_s=time.monotonic() - t0,
        model=model,
        plan_ref=plain(gpu_plan.reference),
    )

    # Fast-path explicit resume_analysis_ref verification.
    if resume_analysis_ref is not None:
        t_res = time.monotonic()
        norm_rel_ref, _, _ = load_phase3_artifact(
            resolved_root,
            plan_protocol,
            eff_release_ref,
            KIND_E39_E31_RELEASE,
            expected_schema=SCHEMA_E39_E31_FEATURES,
            expected_dataset=gpu_plan["dataset_fingerprint"],
            expected_plan_identity=gpu_plan["plan_identity"],
            expected_owner=model,
        )
        res_ref, res_body, _ = verify_resume_ref(
            resolved_root,
            plan_protocol,
            resume_analysis_ref,
            KIND_ANALYSIS,
            expected_schema=SCHEMA_ANALYSIS,
            expected_dataset=gpu_plan["dataset_fingerprint"],
            expected_plan_identity=gpu_plan["plan_identity"],
            expected_owner=model,
        )
        check(res_body.get("experiment") == "E39+E31", "Resume analysis artifact is not E39+E31")
        check(res_body.get("confirm_excluded") is True, "Resume analysis missing confirm_excluded=True")
        check(res_body.get("e35_excluded") is True, "Resume analysis missing e35_excluded=True")
        check(plain(res_body.get("plan")) == plain(gpu_plan.reference), "Resume analysis plan ref mismatch")
        check(plain(res_body.get("release")) == plain(norm_rel_ref), "Resume analysis release ref mismatch")
        check(plain(res_body.get("bootstrap")) == plain(gpu_plan["bootstrap"]), "Resume analysis bootstrap mismatch")
        _emit_progress(
            progress,
            "e39_e31_resumed",
            unit=f"{model}:resume",
            completed=total_units,
            total=total_units,
            read_hash_s=time.monotonic() - t_res,
            model=model,
            analysis_ref=plain(res_ref),
        )
        return res_ref, res_body

    import numpy as np
    from src.stage4.metrics import bootstrap_counts
    from src.stage4.plan import DEPTHS, load_bootstrap

    decision_ranges = plain(gpu_plan.get("decision_ranges") or PHASE3_GPU_DECISION_RANGES)
    e39_decision_cfg = decision_ranges["E39"]
    e31_decision_cfg = decision_ranges["E31"]
    parity_tol = float(e39_decision_cfg["checks"]["stage2_draw0_parity_tol"])
    att_sum_tol = float(e31_decision_cfg["checks"]["attention_mass_sum_tol"])

    # 1. Authenticate Stage B2 release and reassemble chunks in screen_ids order.
    t_reasm = time.monotonic()
    reasm = reassemble_e39_e31_release(
        resolved_root,
        plan_protocol,
        gpu_plan,
        model,
        eff_release_ref,
        attention_sum_tol=att_sum_tol,
    )
    depths = reasm["depths"]
    check(
        tuple(depths) == tuple(DEPTHS[model]),
        f"Release depths {depths} mismatch Stage 4 DEPTHS[{model!r}]={DEPTHS[model]}",
    )
    midpoint_depth = reasm["midpoint_depth"]
    screen_ids = reasm["screen_ids"]
    _emit_progress(
        progress,
        "e39_e31_release_reassembled",
        unit=f"{model}:reassemble",
        completed=2,
        total=total_units,
        read_hash_s=time.monotonic() - t_reasm,
        model=model,
        depths=depths,
        dimension=reasm["dimension"],
        total_images=len(screen_ids),
    )

    # 2. Verify Stage 2 draw-0 feature parity & en->en Stage 4 logit self-checks (<= 1e-4).
    t_par = time.monotonic()
    parity = verify_stage2_draw0_parity(
        binding,
        model,
        depths,
        screen_ids,
        reasm["e39_features"],
        tol=parity_tol,
    )
    _emit_progress(
        progress,
        "e39_e31_stage2_parity_passed",
        unit=f"{model}:parity_gate",
        completed=3,
        total=total_units,
        read_hash_s=time.monotonic() - t_par,
        model=model,
        max_feature_abs_diff=parity["parity_report"]["max_feature_abs_diff"],
        max_logit_abs_diff=parity["parity_report"]["max_logit_abs_diff"],
    )

    # Load frozen targets (in screen_ids order, preserving unknown cell) & frozen bootstrap.
    fit_targets = binding.stage4_plan.fit["targets"]
    y_rows = []
    known_rows = []
    for iid in screen_ids:
        trec = fit_targets[iid]
        check(trec["split"] == E39_E31_SPLIT, f"Cross-split target join on {iid}")
        vals = trec["values"]
        y_rows.append([0 if v is None else int(v) for v in vals])
        known_rows.append([v is not None for v in vals])
    y = np.asarray(y_rows, dtype=bool)
    known = np.asarray(known_rows, dtype=bool)
    if "6a20677eee04fd08" in screen_ids and "plant.n.02" in cat_index:
        check(
            not bool(known[screen_ids.index("6a20677eee04fd08"), cat_index["plant.n.02"]]),
            "Unknown cell 6a20677eee04fd08 / plant.n.02 must stay masked",
        )

    draws = load_bootstrap(binding.stage4_plan)
    check(
        list(draws.shape) == list(gpu_plan["bootstrap"]["shape"])
        and hashlib.sha256(draws.tobytes()).hexdigest() == gpu_plan["bootstrap"]["sha256"],
        "Loaded bootstrap mismatch with phase3_gpu_plan",
    )
    counts = bootstrap_counts(draws, len(screen_ids))

    # 3. E39 sub-caption scoring & bootstrap evaluation.
    t_e39 = time.monotonic()
    e39_summary, e39_caches = compute_e39_analysis(
        depths=depths,
        midpoint_depth=midpoint_depth,
        e39_features=reasm["e39_features"],
        token_selections=reasm["token_selections"],
        screen_ids=screen_ids,
        stage3_heads=parity["stage3_heads"],
        y=y,
        known=known,
        counts=counts,
        categories=categories,
        primary_indices=primary_indices,
        all_indices=all_indices,
        minimum=screen_minimum,
        decision_ranges=e39_decision_cfg,
    )
    _emit_progress(
        progress,
        "e39_analysis_complete",
        unit=f"{model}:e39_analysis",
        completed=4,
        total=total_units,
        compute_s=time.monotonic() - t_e39,
        model=model,
        terminal_token_artifact_supported=e39_summary["decision_summary"]["midpoint_summary"][
            "terminal_token_artifact_supported"
        ],
    )

    # 4. E31 visual-token probe & attention-gap analysis.
    t_e31 = time.monotonic()
    e31_summary = compute_e31_analysis(
        binding=binding,
        model=model,
        depths=depths,
        midpoint_depth=midpoint_depth,
        screen_ids=screen_ids,
        e31_visual=reasm["e31_visual"],
        e31_attention=reasm["e31_attention"],
        attention_mass_check=reasm["attention_mass_check"],
        image_only_groups=parity["image_only_groups"],
        gid_to_block=parity["gid_to_block"],
        e39_caches=e39_caches,
        y=y,
        known=known,
        counts=counts,
        categories=categories,
        primary_indices=primary_indices,
        all_indices=all_indices,
        minimum=screen_minimum,
        decision_ranges=e31_decision_cfg,
    )
    _emit_progress(
        progress,
        "e31_analysis_complete",
        unit=f"{model}:e31_analysis",
        completed=5,
        total=total_units,
        compute_s=time.monotonic() - t_e31,
        model=model,
        midpoint_en_visual_auroc=e31_summary["decision_summary"]["decoder_visual_probe"]["midpoint_en_auroc"],
        attention_correlates_with_gap=e31_summary["decision_summary"]["attention_gap_correlation"][
            "attention_correlates_with_gap"
        ],
    )

    # 5. Publish immutable analysis artifact.
    t_pub = time.monotonic()
    base_run_id = run_id or new_run_id()
    analysis_body = {
        "schema": SCHEMA_ANALYSIS,
        "experiment": "E39+E31",
        "plan": plain(gpu_plan.reference),
        "plan_identity": gpu_plan["plan_identity"],
        "release": plain(reasm["release_ref"]),
        "model": model,
        "depths": list(depths),
        "midpoint_depth": int(midpoint_depth),
        "dimension": int(reasm["dimension"]),
        "conditions": list(CONDITIONS),
        "split": E39_E31_SPLIT,
        "n_screen_images": len(screen_ids),
        "category_order": list(categories),
        "primary_categories": list(primary_categories),
        "bootstrap": plain(gpu_plan["bootstrap"]),
        "stage2_parity_gate": parity["parity_report"],
        "e39": e39_summary,
        "e31": e31_summary,
        "e35_excluded": True,
        "confirm_excluded": True,
    }
    upstream = [
        plain(gpu_plan.reference),
        plain(reasm["release_ref"]),
        plain(gpu_plan["phase1_stage4_release"]),
    ]
    seen_up = {json.dumps(u, sort_keys=True) for u in upstream}
    for blk in parity["gid_to_block"].values():
        ref = plain(blk["prediction_ref"])
        key = json.dumps(ref, sort_keys=True)
        if key not in seen_up:
            seen_up.add(key)
            upstream.append(ref)

    analysis_ref = publish_phase3(
        resolved_root,
        plan_protocol,
        gpu_plan["dataset_fingerprint"],
        f"{E39_E31_RESULTS_NAMESPACE}/{model}",
        KIND_ANALYSIS,
        analysis_body,
        producer,
        upstream=upstream,
        model_fingerprint=reasm["release_body"].get("model_fingerprint"),
        run_id=base_run_id,
    )
    verified_ref, verified_body, _ = verify_resume_ref(
        resolved_root,
        plan_protocol,
        analysis_ref,
        KIND_ANALYSIS,
        expected_schema=SCHEMA_ANALYSIS,
        expected_dataset=gpu_plan["dataset_fingerprint"],
        expected_plan_identity=gpu_plan["plan_identity"],
        expected_owner=model,
    )
    _emit_progress(
        progress,
        "e39_e31_published",
        unit=f"{model}:publish",
        completed=total_units,
        total=total_units,
        write_s=time.monotonic() - t_pub,
        model=model,
        analysis_ref=plain(verified_ref),
    )
    return verified_ref, verified_body


def _parse_ref_arg(value: Optional[str]):
    if value is None:
        return None
    text = value.strip()
    if text.startswith("{"):
        parsed = json.loads(text)
        check(isinstance(parsed, dict), "JSON reference argument must decode to an object")
        return parsed
    return text


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--phase1-root", default=None)
    parser.add_argument(
        "--plan",
        default=None,
        help="Optional phase3_gpu_plan {'path', 'sha256'} JSON or path (defaults to CERTIFIED_STAGE_B1_GPU_PLAN)",
    )
    parser.add_argument(
        "--release",
        default=None,
        help="Optional Stage B2 E39+E31 release {'path', 'sha256'} JSON or path (defaults to CERTIFIED_STAGE_B2_RELEASES['e39_e31'][model])",
    )
    parser.add_argument("--protocol", type=Path, default=REPO / "configs/protocol.json")
    parser.add_argument(
        "--resume-analysis",
        "--resume-analysis-ref",
        dest="resume_analysis",
        default=None,
        help="Optional explicit E39+E31 analysis {'path', 'sha256'} JSON or path",
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
    check(root.is_dir(), f"PHASE1_ROOT must exist and be mounted before launch: {root}")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    producer = current_producer(args.storage_scope)
    _default_emit({
        "event": "e39_e31_start",
        "model": args.model,
        "code_commit": producer["code_commit"],
        "storage_scope": producer["storage_scope"],
    })
    analysis_ref, analysis_body = run_e39_e31(
        root,
        protocol,
        args.model,
        producer,
        plan_ref=_parse_ref_arg(args.plan),
        release_ref=_parse_ref_arg(args.release),
        resume_analysis_ref=_parse_ref_arg(args.resume_analysis),
        run_id=args.run_id,
        progress=_default_emit,
    )
    summary = {
        "event": "e39_e31_complete",
        "status": "passed",
        "model": args.model,
        "analysis_ref": plain(analysis_ref),
        "stage2_parity_gate": {
            "feature_parity_all_passed": analysis_body["stage2_parity_gate"]["feature_parity_all_passed"],
            "max_feature_abs_diff": analysis_body["stage2_parity_gate"]["max_feature_abs_diff"],
            "logit_self_checks_all_passed": analysis_body["stage2_parity_gate"]["logit_self_checks_all_passed"],
            "max_logit_abs_diff": analysis_body["stage2_parity_gate"]["max_logit_abs_diff"],
        },
        "e39_decision_summary": plain(analysis_body["e39"]["decision_summary"]["midpoint_summary"]),
        "e31_decision_summary": plain(analysis_body["e31"]["decision_summary"]),
        "e35_excluded": True,
        "confirm_excluded": True,
    }
    if args.output_ref is not None:
        args.output_ref.parent.mkdir(parents=True, exist_ok=True)
        args.output_ref.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _default_emit(summary)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        _default_emit({"status": "blocked", "blockers": [str(exc)], "phase3_e39_e31_complete": False})
        raise SystemExit(2)
