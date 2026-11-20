"""Phase 3 Stage A2 Worker 3 (E44): selective-confidence coverage & retained risk.

Implements phase3.md §A2.1–A2.2 (`E44`):
- Binds `phase3_cpu_plan` with `bind=True` (never touches `CONFIRM`, never mutates
  Phase 1 or Phase 2 artifacts, writes only under `PHASE1_ROOT/phase3/`).
- Evaluates the E19 selective-confidence protocol at nominal coverages
  `{50%, 75%, 90%}` using `c = max(p, 1 - p)` with exact tie handling at
  midpoint `image_caption` across 5 arms:
  1. `last_caption:raw` (reproduces Phase 1 E19 to `<= 1e-12`),
  2. `last_caption:source_dev_c` (Phase 2 E17 `source_dev_c:temp_intercept`),
  3. `last_caption:target_dev_c` (Phase 2 E17 `target_dev_c:temp_intercept`),
  4. `caption_mean:raw` (cutoffs fitted on Stage 2 `caption_mean` DEV-C diagonals),
  5. `caption_mean:e11_mean_scale` (cutoffs fitted on `caption_mean` DEV-C
     diagonals/adapted routes, evaluated on Phase 2 E11 `mean_scale` SCREEN).
- Computes actual coverage, `|coverage - nominal|` drift, retained risk vs base
  error vs score-blind risk at matched coverage, positive/negative retention,
  explicit support status (`>= 20` per outcome class), and paired differences vs
  `last_caption:raw` inside every `(2000, 360)` bootstrap draw (cutoffs held
  fixed during resampling) before taking linear 2.5th/97.5th percentiles.
- Publishes `KIND_ANALYSIS` / `SCHEMA_ANALYSIS` under `phase3/results/e44/<model>/<run_id>/`.
"""

import argparse
import json
from pathlib import Path
import sys
import time

from src.common.contracts import ContractError, LANGUAGES, fingerprint
from src.common.storage import phase1_root, within
from src.phase2.interfaces import (
    KIND_CALIBRATORS as P2_KIND_CALIBRATORS,
    KIND_PREDICTION as P2_KIND_PREDICTION,
    SCHEMA_CALIBRATORS as P2_SCHEMA_CALIBRATORS,
    SCHEMA_PREDICTION as P2_SCHEMA_PREDICTION,
    e29_screen_blocks,
    load_frozen_source_head,
    load_phase2_artifact,
    read_verified_midpoint_features,
)
from src.phase3.interfaces import (
    KIND_ANALYSIS,
    MODELS,
    REPO,
    SCHEMA_ANALYSIS,
    check,
    contract,
    current_producer,
    load_phase3_cpu_plan,
    plain,
    publish_phase3,
    require_open_split,
    verify_resume_ref,
)

E44_RESULTS_NAMESPACE = "phase3/results/e44"
E44_CONDITION = "image_caption"
E44_NOMINALS = (0.5, 0.75, 0.9)
E44_PRIMARY_NOMINAL = 0.75

E44_ARMS = (
    "last_caption:raw",
    "last_caption:source_dev_c",
    "last_caption:target_dev_c",
    "caption_mean:raw",
    "caption_mean:e11_mean_scale",
)

E44_ARM_SUPERVISION = {
    "last_caption:raw": "source_only_transfer",
    "last_caption:source_dev_c": "source_dev_c_calibrated",
    "last_caption:target_dev_c": "target_dev_c_supervised_calibration",
    "caption_mean:raw": "source_only_transfer",
    "caption_mean:e11_mean_scale": "target_fit_input_adapted",
}

E19_SELECTIVE_METRICS = (
    "target_coverage",
    "retained_risk",
    "base_risk",
    "score_blind_risk",
    "positive_retention",
    "negative_retention",
    "retained_sensitivity",
    "retained_specificity",
    "overall_retained_positive_correct",
    "overall_retained_negative_correct",
)

E44_SELECTIVE_METRICS = E19_SELECTIVE_METRICS + (
    "coverage_drift",
    "retained_risk_vs_base_risk",
    "retained_risk_vs_score_blind_risk",
)


def _peak_rss_bytes():
    """Return process high-water RSS in bytes on Linux/macOS, or None on Windows."""
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)
    except Exception:
        return None


def _emit(progress, event):
    if progress is not None:
        progress(event)


def _default_emit(event):
    print(json.dumps(plain(event), allow_nan=False, sort_keys=True), flush=True)


def _progress_event(
    event_name,
    unit,
    completed,
    total,
    *,
    read_hash_s=0.0,
    compute_s=0.0,
    write_s=0.0,
    extra=None,
):
    payload = {
        "event": event_name,
        "unit": unit,
        "completed": int(completed),
        "total": int(total),
        "read_hash_s": round(float(read_hash_s), 6),
        "compute_s": round(float(compute_s), 6),
        "write_s": round(float(write_s), 6),
        "peak_rss_bytes": _peak_rss_bytes(),
    }
    if extra:
        payload.update(extra)
    return payload


def _route_key(source_lang, target_lang):
    return f"{source_lang}->{target_lang}"


def _all_route_pairs():
    return [(s, t) for s in LANGUAGES for t in LANGUAGES]


def _off_diagonal_pairs():
    return [(s, t) for s in LANGUAGES for t in LANGUAGES if s != t]


def _diagonal_pairs():
    return [(s, s) for s in LANGUAGES]


def _e44_expected_inputs(plan, model, inv):
    return {
        "plan_ref": plain(plan.reference),
        "model": model,
        "e44_inventory": plain(inv),
        "nominals": list(E44_NOMINALS),
        "arms": list(E44_ARMS),
    }


def _support_status_for_metric(metric, value, n_pos, n_neg, point_counts, screen_min, subgroup_min):
    """Apply frozen split support and subgroup/correctness support rules (`>= 20`)."""
    if value is None:
        return "undefined", f"Zero denominator for {metric}."
    if min(n_pos, n_neg) < screen_min:
        return "descriptive_low_support", "Below SCREEN minimum positive/negative support."
    if metric in ("retained_risk", "retained_risk_vs_base_risk", "retained_risk_vs_score_blind_risk"):
        r_err = point_counts["retained_errors"]
        r_cor = point_counts["retained_correct"]
        if min(r_err, r_cor) < subgroup_min:
            return "descriptive_low_support", "Below minimum retained correctness outcome support."
    if metric in (
        "base_risk",
        "score_blind_risk",
        "retained_risk_vs_base_risk",
        "retained_risk_vs_score_blind_risk",
    ):
        err = point_counts["errors"]
        cor = point_counts["correct"]
        if min(err, cor) < subgroup_min:
            return "descriptive_low_support", "Below minimum correctness outcome support."
    elif metric in (
        "retained_sensitivity",
        "retained_specificity",
        "overall_retained_positive_correct",
        "overall_retained_negative_correct",
    ):
        outcome = (
            "positive"
            if metric in ("retained_sensitivity", "overall_retained_positive_correct")
            else "negative"
        )
        successes = point_counts[f"retained_true_{outcome}"]
        if min(successes, point_counts["denominator"] - successes) < subgroup_min:
            return "descriptive_low_support", f"Below minimum {outcome} subgroup outcome support."
    return "ok", None


def _metric_support(metric, counts_row):
    """Build machine-readable support dictionary from 10 selective indicator sums."""
    n, n_pos, n_neg, r, r_pos, r_neg, e, e_ret, tp_ret, tn_ret = (int(v) for v in counts_row)
    base = {"known": n, "positive": n_pos, "negative": n_neg}
    if metric in ("target_coverage", "coverage_drift"):
        return {**base, "retained": r, "denominator": n}
    if metric == "retained_risk":
        return {**base, "retained": r, "retained_errors": e_ret, "retained_correct": r - e_ret, "denominator": r}
    if metric == "base_risk":
        return {**base, "errors": e, "correct": n - e, "denominator": n}
    if metric == "score_blind_risk":
        return {**base, "retained": r, "errors": e, "correct": n - e, "denominator": n if r > 0 else 0}
    if metric in ("retained_risk_vs_base_risk", "retained_risk_vs_score_blind_risk"):
        return {
            **base,
            "retained": r,
            "retained_errors": e_ret,
            "retained_correct": r - e_ret,
            "errors": e,
            "correct": n - e,
            "denominator": r if (r > 0 and n > 0) else 0,
        }
    if metric == "positive_retention":
        return {**base, "retained_positive": r_pos, "denominator": n_pos}
    if metric == "negative_retention":
        return {**base, "retained_negative": r_neg, "denominator": n_neg}
    if metric == "retained_sensitivity":
        return {**base, "retained_positive": r_pos, "retained_true_positive": tp_ret, "denominator": r_pos}
    if metric == "retained_specificity":
        return {**base, "retained_negative": r_neg, "retained_true_negative": tn_ret, "denominator": r_neg}
    if metric == "overall_retained_positive_correct":
        return {**base, "retained_positive": r_pos, "retained_true_positive": tp_ret, "denominator": n_pos}
    if metric == "overall_retained_negative_correct":
        return {**base, "retained_negative": r_neg, "retained_true_negative": tn_ret, "denominator": n_neg}
    raise ContractError(f"Unknown E44 selective metric: {metric}")


def _selective_indicators(y, p, known, cutoff):
    """Build exact integer indicator columns for one category under a fixed DEV-C cutoff."""
    import numpy as np

    d = p >= 0.5
    conf = np.maximum(p, 1.0 - p)
    ret = conf >= cutoff
    err = d != y
    k = known
    return np.column_stack((
        k,                  # 0: N (known)
        k & y,              # 1: N_pos
        k & ~y,             # 2: N_neg
        k & ret,            # 3: R (retained)
        k & ret & y,        # 4: R_pos
        k & ret & ~y,       # 5: R_neg
        k & err,            # 6: E (base errors)
        k & ret & err,      # 7: E_ret (retained errors)
        k & ret & y & d,    # 8: TP_ret
        k & ret & ~y & ~d,  # 9: TN_ret
    )).astype(np.int64, copy=False)


def _safe_ratio(numerator, denominator):
    import numpy as np

    numer, denom = np.broadcast_arrays(numerator, denominator)
    return np.divide(numer, denom, out=np.full(numer.shape, np.nan, dtype="f8"), where=denom > 0)


def _selective_ratios(sums, nominal):
    """Compute all E44 selective SCREEN metrics from indicator sums of shape (..., 10)."""
    import numpy as np

    n, n_pos, n_neg, r, r_pos, r_neg, e, e_ret, tp_ret, tn_ret = (sums[..., i] for i in range(10))
    target_cov = _safe_ratio(r, n)
    retained_risk = _safe_ratio(e_ret, r)
    base_risk = _safe_ratio(e, n)
    score_blind = np.where(r > 0, base_risk, np.nan)
    return {
        "target_coverage": target_cov,
        "coverage_drift": np.abs(target_cov - float(nominal)),
        "retained_risk": retained_risk,
        "base_risk": base_risk,
        "score_blind_risk": score_blind,
        "retained_risk_vs_base_risk": retained_risk - base_risk,
        "retained_risk_vs_score_blind_risk": retained_risk - score_blind,
        "positive_retention": _safe_ratio(r_pos, n_pos),
        "negative_retention": _safe_ratio(r_neg, n_neg),
        "retained_sensitivity": _safe_ratio(tp_ret, r_pos),
        "retained_specificity": _safe_ratio(tn_ret, r_neg),
        "overall_retained_positive_correct": _safe_ratio(tp_ret, n_pos),
        "overall_retained_negative_correct": _safe_ratio(tn_ret, n_neg),
    }


def _compute_cutoffs_for_logits(logits, estimable, categories, nominals):
    """Fit E19 DEV-C cutoffs at `nominals` using `c = max(p, 1-p)` with exact tie handling."""
    from src.stage4.metrics import confidence_cutoff, probabilities

    probs = probabilities(logits)
    cutoffs = {}
    for idx, cat in enumerate(categories):
        scores = probs[:, idx] if estimable[idx] else probs[:0, idx]
        for nominal in nominals:
            res = confidence_cutoff(scores, nominal)
            cutoffs[(cat, nominal)] = {
                "category": cat,
                "nominal": float(nominal),
                "eligible": int(res["eligible"]),
                "retained": int(res["retained"]),
                "cutoff": res["cutoff"],
            }
    return cutoffs


def _load_e17_calibrators(plan, inv, model):
    """Load and validate Phase 2 E17 source_dev_c and target_dev_c temp_intercept calibrators."""
    cal_ref, cal_body, _ = load_phase2_artifact(
        plan.root,
        plain(plan.protocol),
        inv["reused_e17_calibrators_ref"],
        P2_KIND_CALIBRATORS,
        expected_schema=P2_SCHEMA_CALIBRATORS,
        expected_dataset=plan["dataset_fingerprint"],
        expected_owner=model,
    )
    check(cal_body.get("experiment") == "E17", "Expected E17 calibrators artifact")
    check(cal_body.get("confirm_excluded") is True, "E17 calibrators must exclude CONFIRM")

    source_cals = {}
    for s in LANGUAGES:
        entry = cal_body["source_dev_c"][s]["temp_intercept"]
        t_val, b_val = float(entry["T"]), float(entry["b"])
        check(entry.get("converged") is True and t_val > 0.0, f"Invalid E17 source calibrator for {s}")
        source_cals[s] = (t_val, b_val)

    target_cals = {}
    for s, t in _off_diagonal_pairs():
        entry = cal_body["target_dev_c"][_route_key(s, t)]["temp_intercept"]
        t_val, b_val = float(entry["T"]), float(entry["b"])
        check(entry.get("converged") is True and t_val > 0.0, f"Invalid E17 target calibrator for {s}->{t}")
        target_cals[(s, t)] = (t_val, b_val)
    for s in LANGUAGES:
        target_cals[(s, s)] = source_cals[s]

    return plain(cal_ref), source_cals, target_cals


def _load_e17_off_diag_dev_c(plan, inv, model, n_cats):
    """Load Phase 2 E17 off-diagonal DEV-C logits and estimable masks."""
    import numpy as np

    refs = []
    by_route = {}
    for t in LANGUAGES:
        ref_raw = inv["reused_e17_off_diag_dev_c_blocks"][t]
        ref, body, _ = load_phase2_artifact(
            plan.root,
            plain(plan.protocol),
            ref_raw,
            P2_KIND_PREDICTION,
            expected_schema=P2_SCHEMA_PREDICTION,
            expected_dataset=plan["dataset_fingerprint"],
            expected_owner=model,
        )
        check(body.get("split") == "DEV-C" and body.get("target_language") == t, "Invalid E17 DEV-C block metadata")
        sorted_sources = list(body["source_languages"])
        check(set(sorted_sources) == set(LANGUAGES) - {t}, f"Invalid E17 DEV-C sources for {t}")
        marker_dir = within(plan.root, ref["path"]).parent
        logits = np.load(marker_dir / "logits.npy", allow_pickle=False)
        estimable = np.load(marker_dir / "estimable.npy", allow_pickle=False)
        check(logits.ndim == 3 and logits.shape[0] == len(sorted_sources) and logits.shape[2] == n_cats,
              "Invalid E17 DEV-C logits shape")
        check(np.isfinite(logits).all(), "Nonfinite E17 DEV-C logits")
        check(estimable.shape == (len(sorted_sources), n_cats) and estimable.dtype == bool,
              "Invalid E17 DEV-C estimable shape")
        refs.append(plain(ref))
        for idx, s in enumerate(sorted_sources):
            by_route[(s, t)] = (np.asarray(logits[idx], dtype=np.float64), np.asarray(estimable[idx], dtype=bool))
    return refs, by_route


def _load_e11_screen_adapted(plan, inv, model, cm_screen_blocks, n_cats):
    """Load Phase 2 E11 mean_scale SCREEN logits aligned to stage4_plan.screen_ids."""
    import numpy as np
    from src.stage4.predictions import load_prediction_block
    from src.stage4.metrics import paired_image_alignment

    refs = []
    by_route = {}
    stage4_plan = plan.binding.stage4_plan
    n_screen = len(stage4_plan["screen_ids"])
    mean_scale_refs = inv["reused_e11_screen_adapted_refs"]["mean_scale"]

    for t in LANGUAGES:
        # Reference Stage 4 block for row alignment & image_ids_sha256 verification.
        blk_info = cm_screen_blocks[t]
        s4_block = load_prediction_block(stage4_plan, blk_info["prediction_ref"], expected_block=blk_info["block_id"])
        s4_diag = s4_block[blk_info["routes"][t]]
        _, (aligned,) = paired_image_alignment(stage4_plan, s4_diag)
        expected_ids_sha = fingerprint([row["image_id"] for row in s4_diag.rows])

        ref, body, _ = load_phase2_artifact(
            plan.root,
            plain(plan.protocol),
            mean_scale_refs[t],
            P2_KIND_PREDICTION,
            expected_schema=P2_SCHEMA_PREDICTION,
            expected_dataset=plan["dataset_fingerprint"],
            expected_owner=model,
        )
        check(body.get("experiment") == "E11" and body.get("variant") == "mean_scale", "Invalid E11 block variant")
        check(body.get("split") == "SCREEN" and body.get("target_language") == t, "Invalid E11 block split/target")
        check(body.get("image_ids_sha256") == expected_ids_sha, f"E11 SCREEN image_ids_sha256 mismatch for {t}")
        sorted_sources = list(body["source_languages"])
        check(set(sorted_sources) == set(LANGUAGES) - {t}, f"Invalid E11 sources for {t}")

        marker_dir = within(plan.root, ref["path"]).parent
        logits = np.load(marker_dir / "logits.npy", allow_pickle=False)
        estimable = np.load(marker_dir / "estimable.npy", allow_pickle=False)
        check(logits.shape == (len(sorted_sources), n_screen, n_cats), "Invalid E11 SCREEN logits shape")
        check(np.isfinite(logits).all(), "Nonfinite E11 SCREEN logits")
        check(estimable.shape == (len(sorted_sources), n_cats) and estimable.dtype == bool,
              "Invalid E11 SCREEN estimable shape")
        refs.append(plain(ref))
        for idx, s in enumerate(sorted_sources):
            by_route[(s, t)] = (
                np.asarray(logits[idx][aligned], dtype=np.float64),
                np.asarray(estimable[idx], dtype=bool),
            )
    return refs, by_route


def _score_caption_mean_dev_c(plan, model, categories):
    """Score Stage 2 caption_mean DEV-C features on diagonals (raw) and off-diagonals (E11 mean_scale)."""
    import numpy as np
    from src.stage3.engine import transform

    binding = plan.binding
    heads = {
        s: load_frozen_source_head(binding, model, E44_CONDITION, "caption_mean", s)
        for s in LANGUAGES
    }

    raw_cutoffs = {}
    e11_cutoffs = {}

    for t in LANGUAGES:
        _, x_dev_t = read_verified_midpoint_features(
            binding, model, E44_CONDITION, "caption_mean", t, "DEV-C"
        )
        x_dev_t = np.asarray(x_dev_t, dtype=np.float64)
        _, _, _, _, stats_t = heads[t]
        mu_t = np.asarray(stats_t["mean"], dtype=np.float64)
        sigma_t = np.asarray(stats_t["scale"], dtype=np.float64)

        for s in LANGUAGES:
            coef_s, intercept_s, estimable_s, _, stats_s = heads[s]
            est_s = np.asarray(estimable_s, dtype=bool)
            if s == t:
                logits_raw = np.asarray(transform(x_dev_t, stats_s) @ coef_s.T, dtype=np.float64) + intercept_s
                logits_raw = np.where(est_s[None, :], logits_raw, 0.0)
                cut_raw = _compute_cutoffs_for_logits(logits_raw, est_s, categories, E44_NOMINALS)
                raw_cutoffs[(s, s)] = cut_raw
                e11_cutoffs[(s, s)] = cut_raw
            else:
                mu_s = np.asarray(stats_s["mean"], dtype=np.float64)
                sigma_s = np.asarray(stats_s["scale"], dtype=np.float64)
                x_adapted = mu_s + sigma_s * ((x_dev_t - mu_t) / sigma_t)
                logits_e11 = np.asarray(transform(x_adapted, stats_s) @ coef_s.T, dtype=np.float64) + intercept_s
                logits_e11 = np.where(est_s[None, :], logits_e11, 0.0)
                e11_cutoffs[(s, t)] = _compute_cutoffs_for_logits(logits_e11, est_s, categories, E44_NOMINALS)

    return raw_cutoffs, e11_cutoffs


def _evaluate_route_selective(
    *,
    y_aligned,
    known_aligned,
    probs_aligned,
    estimable,
    cutoffs_by_cat_nom,
    categories,
    primary_indices,
    counts,
    screen_min,
):
    """Evaluate all E44 selective metrics and fixed-cutoff bootstrap draws for one route."""
    import numpy as np
    from src.stage4.metrics import bootstrap_interval, fixed_macro

    total_draws = int(counts.shape[0])
    n_cats = len(categories)
    results_by_nominal = {}

    for nominal in E44_NOMINALS:
        nom_key = f"{nominal:.2f}"
        cat_values = {m: np.full(n_cats, np.nan, dtype="f8") for m in E44_SELECTIVE_METRICS}
        cat_draws = {m: np.full((n_cats, total_draws), np.nan, dtype="f8") for m in E44_SELECTIVE_METRICS}
        cat_statuses = {m: [] for m in E44_SELECTIVE_METRICS}
        per_category = {}

        for idx, cat in enumerate(categories):
            c_rec = cutoffs_by_cat_nom[(cat, nominal)]
            cutoff = c_rec["cutoff"]
            y_col = y_aligned[:, idx]
            k_col = known_aligned[:, idx]
            kn = int(k_col.sum())
            pos = int((k_col & y_col).sum())
            neg = kn - pos
            unestimable = (not bool(estimable[idx])) or (cutoff is None)

            per_cat_metrics = {}
            if unestimable:
                empty_draws = np.full(total_draws, np.nan, dtype="f8")
                empty_interval = bootstrap_interval(empty_draws)
                base_sup = {"known": kn, "positive": pos, "negative": neg, "denominator": 0}
                for metric in E44_SELECTIVE_METRICS:
                    per_cat_metrics[metric] = {
                        "value": None,
                        "status": "undefined",
                        "reason": "Unestimable source head.",
                        "support": dict(base_sup),
                        "interval": dict(empty_interval),
                    }
                    cat_statuses[metric].append("undefined")
            else:
                indicators = _selective_indicators(y_col, probs_aligned[:, idx], k_col, cutoff)
                point_sums = indicators.sum(axis=0)
                boot_sums = counts @ indicators
                point_ratios = _selective_ratios(point_sums, nominal)
                boot_ratios = _selective_ratios(boot_sums, nominal)

                for metric in E44_SELECTIVE_METRICS:
                    raw_val = point_ratios[metric]
                    val = float(raw_val) if np.isfinite(raw_val) else None
                    sup = _metric_support(metric, point_sums)
                    status, reason = _support_status_for_metric(
                        metric, val, pos, neg, sup, screen_min, screen_min
                    )
                    draws_arr = boot_ratios[metric]
                    per_cat_metrics[metric] = {
                        "value": val,
                        "status": status,
                        "reason": reason,
                        "support": sup,
                        "interval": bootstrap_interval(draws_arr),
                    }
                    if val is not None:
                        cat_values[metric][idx] = val
                    cat_draws[metric][idx] = draws_arr
                    cat_statuses[metric].append(status)

            per_category[cat] = {
                "cutoff": c_rec,
                "metrics": per_cat_metrics,
            }

        primary_macro = {}
        macro_draws_by_metric = {}
        for metric in E44_SELECTIVE_METRICS:
            prim_vals = cat_values[metric][primary_indices]
            prim_statuses = [cat_statuses[metric][i] for i in primary_indices]
            defined_count = int(np.isfinite(prim_vals).sum())
            ok_count = sum(1 for st in prim_statuses if st == "ok")
            macro_raw = float(fixed_macro(cat_values[metric], primary_indices))
            if np.isfinite(macro_raw):
                macro_val = macro_raw
                if ok_count == len(primary_indices):
                    macro_status, macro_reason = "ok", None
                else:
                    macro_status = "descriptive_low_support"
                    macro_reason = "One or more primary categories have low subgroup or correctness support."
            else:
                macro_val = None
                macro_status = "undefined"
                macro_reason = "Unresolved primary macro: one or more primary categories are undefined."
            macro_draws = fixed_macro(cat_draws[metric], primary_indices, axis=0)
            macro_draws_by_metric[metric] = macro_draws
            primary_macro[metric] = {
                "value": macro_val,
                "status": macro_status,
                "reason": macro_reason,
                "support": {
                    "categories": len(primary_indices),
                    "defined_categories": defined_count,
                    "ok_categories": ok_count,
                },
                "interval": bootstrap_interval(macro_draws),
            }

        results_by_nominal[nom_key] = {
            "nominal": float(nominal),
            "primary_macro": primary_macro,
            "per_category": per_category,
            "_macro_draws": macro_draws_by_metric,
        }
    return results_by_nominal


def _aggregate_routes_within_draws(route_results, route_pairs):
    """Equal-weight mean across `route_pairs` inside every bootstrap draw."""
    import numpy as np
    from src.stage4.metrics import bootstrap_interval

    agg_by_nom = {}
    draws_by_nom = {}
    for nominal in E44_NOMINALS:
        nom_key = f"{nominal:.2f}"
        agg_by_nom[nom_key] = {}
        draws_by_nom[nom_key] = {}
        for metric in E44_SELECTIVE_METRICS:
            entries = [route_results[_route_key(s, t)][nom_key]["primary_macro"][metric] for s, t in route_pairs]
            vals = [e["value"] for e in entries]
            statuses = [e["status"] for e in entries]
            if any(v is None for v in vals):
                point = None
                status = "undefined"
                reason = "One or more routes have undefined primary_macro."
            else:
                point = float(np.mean(vals))
                if all(st == "ok" for st in statuses):
                    status, reason = "ok", None
                else:
                    status = "descriptive_low_support"
                    reason = "One or more routes have low subgroup or correctness support."
            stacked = np.stack(
                [route_results[_route_key(s, t)][nom_key]["_macro_draws"][metric] for s, t in route_pairs],
                axis=0,
            )
            mean_draws = np.mean(stacked, axis=0)
            draws_by_nom[nom_key][metric] = mean_draws
            agg_by_nom[nom_key][metric] = {
                "value": point,
                "status": status,
                "reason": reason,
                "route_count": len(route_pairs),
                "interval": bootstrap_interval(mean_draws),
            }
    return agg_by_nom, draws_by_nom


def _compute_paired_vs_raw(arm_results, raw_results, arm_mean_draws, raw_mean_draws, route_pairs):
    """Compute paired differences (arm - last_caption:raw) inside every bootstrap draw."""
    import numpy as np
    from src.stage4.metrics import bootstrap_interval

    paired_mean = {}
    paired_per_route = {}
    for nominal in E44_NOMINALS:
        nom_key = f"{nominal:.2f}"
        paired_mean[nom_key] = {}
        paired_per_route[nom_key] = {}
        for metric in E44_SELECTIVE_METRICS:
            # 12-route mean delta inside every draw
            arm_d = arm_mean_draws[nom_key][metric]
            raw_d = raw_mean_draws[nom_key][metric]
            delta_d = arm_d - raw_d
            arm_vals = [arm_results[_route_key(s, t)][nom_key]["primary_macro"][metric]["value"] for s, t in route_pairs]
            raw_vals = [raw_results[_route_key(s, t)][nom_key]["primary_macro"][metric]["value"] for s, t in route_pairs]
            if any(v is None for v in arm_vals) or any(v is None for v in raw_vals):
                mean_pt = None
            else:
                mean_pt = float(np.mean(arm_vals) - np.mean(raw_vals))
            paired_mean[nom_key][metric] = {
                "value": mean_pt,
                "interval": bootstrap_interval(delta_d),
            }

            # Per-route delta inside every draw
            for s, t in route_pairs:
                rkey = _route_key(s, t)
                r_bucket = paired_per_route[nom_key].setdefault(rkey, {})
                a_pt = arm_results[rkey][nom_key]["primary_macro"][metric]["value"]
                r_pt = raw_results[rkey][nom_key]["primary_macro"][metric]["value"]
                a_dr = arm_results[rkey][nom_key]["_macro_draws"][metric]
                r_dr = raw_results[rkey][nom_key]["_macro_draws"][metric]
                r_bucket[metric] = {
                    "value": None if (a_pt is None or r_pt is None) else float(a_pt - r_pt),
                    "interval": bootstrap_interval(a_dr - r_dr),
                }
    return {"twelve_route_mean": paired_mean, "per_route": paired_per_route}


def _verify_phase1_e19_parity(plan, model, raw_route_results, tol=1e-12):
    """Verify `last_caption:raw` reproduces Phase 1 E19 cutoffs, point estimates, and CIs to `<= tol`."""
    from src.stage4.analysis import MACRO, analysis_units, load_analysis_block

    stage4_plan = plan.binding.stage4_plan
    units = analysis_units(stage4_plan)
    max_diff = 0.0
    checked_values = 0

    for s in LANGUAGES:
        uid = f"E19/{model}/{s}"
        e19_ref = plan.binding.stage4_release["analyses"][uid]
        block = load_analysis_block(stage4_plan, e19_ref, units=units)
        comp_map = units[uid]["comparisons"]

        # Map (target_lang, nominal) -> comparison_id for baseline == "probe"
        target_by_block_id = {}
        for t in LANGUAGES:
            blk = plan["e44_inventory"][model]["last_caption_screen_blocks"][t]
            target_by_block_id[blk["block_id"]] = t

        cid_lookup = {}
        for cid, comp in comp_map.items():
            if comp["baseline"] != "probe":
                continue
            t = target_by_block_id[comp["inputs"][1]["block_id"]]
            cid_lookup[(cid)] = (t, float(comp["nominal"]))

        # Verify cutoffs
        for c_rec in block["cutoffs"]:
            if c_rec["baseline"] != "probe":
                continue
            nom_key = f"{float(c_rec['nominal']):.2f}"
            ours = raw_route_results[_route_key(s, s)][nom_key]["per_category"][c_rec["category"]]["cutoff"]
            check(ours["eligible"] == c_rec["eligible"] and ours["retained"] == c_rec["retained"],
                  f"Phase 1 E19 cutoff count mismatch for {model}/{s}/{c_rec['category']}/{nom_key}")
            if c_rec["cutoff"] is not None:
                diff = abs(float(ours["cutoff"]) - float(c_rec["cutoff"]))
                max_diff = max(max_diff, diff)
                checked_values += 1

        # Verify rows (all 10 E19 selective metrics across 25 categories + primary_macro)
        for row in block["rows"]:
            cid = row["comparison_id"]
            if cid not in cid_lookup:
                continue
            stat = row["statistic"]
            if stat not in E19_SELECTIVE_METRICS:
                continue
            t, nominal = cid_lookup[cid]
            nom_key = f"{nominal:.2f}"
            rkey = _route_key(s, t)
            if row["category"] == MACRO:
                ours = raw_route_results[rkey][nom_key]["primary_macro"][stat]
            else:
                ours = raw_route_results[rkey][nom_key]["per_category"][row["category"]]["metrics"][stat]
            check((ours["value"] is None) == (row["value"] is None),
                  f"Phase 1 E19 definedness mismatch: {rkey}/{nom_key}/{row['category']}/{stat}")
            if row["value"] is not None:
                max_diff = max(max_diff, abs(float(ours["value"]) - float(row["value"])))
                checked_values += 1
            for bound in ("lower", "upper"):
                ov, rv = ours["interval"][bound], row["interval"][bound]
                check((ov is None) == (rv is None),
                      f"Phase 1 E19 CI definedness mismatch ({bound}): {rkey}/{nom_key}/{row['category']}/{stat}")
                if rv is not None:
                    max_diff = max(max_diff, abs(float(ov) - float(rv)))
                    checked_values += 1

    check(max_diff <= tol, f"Phase 1 E19 parity failed for {model}: max_abs_diff={max_diff:.3e} > {tol:.1e}")
    return {"passed": True, "max_abs_diff": float(max_diff), "tol": float(tol), "checked_values": int(checked_values)}


def evaluate_e44_decision_summary(twelve_route_means, paired_vs_raw, decision_ranges):
    """Evaluate all non-baseline arms against frozen Track A E44 decision thresholds at nominal 0.75."""
    nom_key = f"{float(decision_ranges['primary_nominal']):.2f}"
    drift_max = float(decision_ranges["coverage_drift_max"])
    risk_red_min = float(decision_ranges["retained_risk_reduction_min"])
    pos_ret_min = float(decision_ranges["positive_retention_min"])
    neg_ci_lt = float(decision_ranges["negligible_risk_ci_upper_lt"])

    per_arm = {}
    for arm in E44_ARMS:
        mean_nom = twelve_route_means[arm][nom_key]
        drift_val = mean_nom["coverage_drift"]["value"]
        pos_ret_val = mean_nom["positive_retention"]["value"]
        ret_risk_val = mean_nom["retained_risk"]["value"]
        base_risk_val = mean_nom["base_risk"]["value"]
        blind_risk_val = mean_nom["score_blind_risk"]["value"]

        if arm == "last_caption:raw":
            risk_red_val = 0.0
            risk_red_ci = {"lower": 0.0, "upper": 0.0}
        else:
            delta_entry = paired_vs_raw[arm]["twelve_route_mean"][nom_key]["retained_risk"]
            # delta is (arm - raw); reduction is (raw - arm) = -delta
            risk_red_val = None if delta_entry["value"] is None else -float(delta_entry["value"])
            lo, hi = delta_entry["interval"]["lower"], delta_entry["interval"]["upper"]
            risk_red_ci = {
                "lower": None if hi is None else -float(hi),
                "upper": None if lo is None else -float(lo),
            }

        consequential = (
            drift_val is not None
            and drift_val <= drift_max
            and risk_red_val is not None
            and risk_red_val >= risk_red_min
            and pos_ret_val is not None
            and pos_ret_val >= pos_ret_min
        )
        negligible = (
            arm != "last_caption:raw"
            and risk_red_ci["upper"] is not None
            and risk_red_ci["upper"] < neg_ci_lt
        )
        per_arm[arm] = {
            "coverage_drift": drift_val,
            "retained_risk": ret_risk_val,
            "base_risk": base_risk_val,
            "score_blind_risk": blind_risk_val,
            "retained_risk_reduction_vs_raw": risk_red_val,
            "retained_risk_reduction_ci": risk_red_ci,
            "positive_retention": pos_ret_val,
            "consequential_recovery": bool(consequential),
            "practically_negligible": bool(negligible),
        }
    return {"primary_nominal": float(decision_ranges["primary_nominal"]), "arms": per_arm}


def _strip_private_draws(route_results):
    """Remove transient `_macro_draws` numpy arrays before JSON serialization."""
    clean = {}
    for rkey, by_nom in route_results.items():
        clean[rkey] = {}
        for nom_key, payload in by_nom.items():
            clean[rkey][nom_key] = {
                "nominal": payload["nominal"],
                "primary_macro": payload["primary_macro"],
                "per_category": payload["per_category"],
            }
    return clean


@contract
def run_e44(
    root,
    protocol,
    plan_ref,
    model,
    producer,
    *,
    resume_analysis_ref=None,
    run_id=None,
    progress=None,
):
    """Run Phase 3 Stage A2 Worker 3 (E44) for one model (`qwen` or `gemma`)."""
    check(model in MODELS, f"MODEL must be one of {MODELS}; got {model!r}")
    require_open_split("DEV-C")
    require_open_split("SCREEN")
    resolved_root = Path(root).resolve()

    total_units = 7  # bind, cutoffs, 5 arms, verify/publish
    t_start = time.monotonic()

    t_read0 = time.perf_counter()
    plan = load_phase3_cpu_plan(resolved_root, protocol, plan_ref, bind=True)
    check(plan["confirm_excluded"] is True, "phase3_cpu_plan must exclude CONFIRM")
    inv = plan["e44_inventory"][model]
    expected_inputs = _e44_expected_inputs(plan, model, inv)

    if resume_analysis_ref is not None:
        ref, body, _ = verify_resume_ref(
            resolved_root,
            protocol,
            resume_analysis_ref,
            KIND_ANALYSIS,
            expected_schema=SCHEMA_ANALYSIS,
            expected_dataset=plan["dataset_fingerprint"],
            expected_plan_identity=plan["plan_identity"],
            expected_owner=model,
            expected_inputs=expected_inputs,
        )
        check(body.get("experiment") == "E44", "Resume E44 experiment mismatch")
        check(body.get("confirm_excluded") is True, "Resume E44 confirm_excluded must be True")
        check((body.get("phase1_e19_parity") or {}).get("passed") is True, "Resume E44 parity check missing")
        _emit(
            progress,
            _progress_event(
                "e44_resumed",
                "resume",
                total_units,
                total_units,
                read_hash_s=time.perf_counter() - t_read0,
                extra={"model": model, "analysis_ref": plain(ref)},
            ),
        )
        summary = {
            "event": "complete",
            "status": "passed",
            "experiment": "E44",
            "model": model,
            "resumed": True,
            "analysis_ref": plain(ref),
            "phase1_e19_parity": plain(body["phase1_e19_parity"]),
            "decision_summary": plain(body["decision_summary"]),
            "elapsed_s": round(time.monotonic() - t_start, 3),
        }
        _emit(progress, summary)
        return summary

    import numpy as np
    from src.stage4.metrics import (
        bootstrap_counts,
        paired_image_alignment,
        probabilities,
        targets_for_prediction,
    )
    from src.stage4.plan import load_bootstrap
    from src.stage4.predictions import load_prediction_block

    binding = plan.binding
    stage4_plan = binding.stage4_plan
    categories = tuple(protocol["category_order"])
    primary = tuple(protocol["primary_categories"])
    check(len(primary) == 14, f"Expected 14 primary categories; got {len(primary)}")
    primary_indices = [categories.index(cat) for cat in primary]
    screen_min = int(protocol["minimum_positive_and_negative"]["SCREEN"])
    n_cats = len(categories)

    draws = load_bootstrap(stage4_plan)
    counts = bootstrap_counts(draws, len(stage4_plan["screen_ids"]))

    cal_ref, source_cals, target_cals = _load_e17_calibrators(plan, inv, model)
    e17_dev_c_refs, e17_off_diag_dev_c = _load_e17_off_diag_dev_c(plan, inv, model, n_cats)
    cm_screen_blocks = e29_screen_blocks(binding, model)[E44_CONDITION]["caption_mean"]
    e11_screen_refs, e11_screen_adapted = _load_e11_screen_adapted(plan, inv, model, cm_screen_blocks, n_cats)
    read_s = time.perf_counter() - t_read0
    _emit(
        progress,
        _progress_event("progress", "bind_and_load", 1, total_units, read_hash_s=read_s, extra={"model": model}),
    )

    # 1. Fit DEV-C cutoffs strictly on DEV-C for all 5 arms.
    t_cut0 = time.perf_counter()
    cutoffs_by_arm = {arm: {} for arm in E44_ARMS}
    for s in LANGUAGES:
        diag = inv["last_caption_dev_c_diagonals"][s]
        dev_pred = load_prediction_block(stage4_plan, diag["prediction_ref"], expected_block=diag["block_id"])[
            diag["route_id"]
        ]
        check(dev_pred.split == "DEV-C", "E44 last_caption DEV-C source must be DEV-C")
        cut_raw = _compute_cutoffs_for_logits(dev_pred.logits, dev_pred.estimable, categories, E44_NOMINALS)
        t_src, b_src = source_cals[s]
        cut_src_cal = _compute_cutoffs_for_logits(
            dev_pred.logits / t_src + b_src, dev_pred.estimable, categories, E44_NOMINALS
        )
        for t in LANGUAGES:
            cutoffs_by_arm["last_caption:raw"][(s, t)] = cut_raw
            cutoffs_by_arm["last_caption:source_dev_c"][(s, t)] = cut_src_cal
        cutoffs_by_arm["last_caption:target_dev_c"][(s, s)] = cut_src_cal

    for s, t in _off_diagonal_pairs():
        logits_off, est_off = e17_off_diag_dev_c[(s, t)]
        t_tgt, b_tgt = target_cals[(s, t)]
        cutoffs_by_arm["last_caption:target_dev_c"][(s, t)] = _compute_cutoffs_for_logits(
            logits_off / t_tgt + b_tgt, est_off, categories, E44_NOMINALS
        )

    cm_raw_cutoffs, cm_e11_cutoffs = _score_caption_mean_dev_c(plan, model, categories)
    for s in LANGUAGES:
        for t in LANGUAGES:
            cutoffs_by_arm["caption_mean:raw"][(s, t)] = cm_raw_cutoffs[(s, s)]
    for s, t in _all_route_pairs():
        cutoffs_by_arm["caption_mean:e11_mean_scale"][(s, t)] = cm_e11_cutoffs[(s, t)]

    cut_s = time.perf_counter() - t_cut0
    _emit(
        progress,
        _progress_event("progress", "dev_c_cutoffs", 2, total_units, compute_s=cut_s, extra={"model": model}),
    )

    # 2. Load aligned SCREEN ground truth and logits per pooling.
    lc_screen = {}
    cm_screen = {}
    targets_by_target = {}
    for t in LANGUAGES:
        lc_blk = inv["last_caption_screen_blocks"][t]
        lc_preds = load_prediction_block(stage4_plan, lc_blk["prediction_ref"], expected_block=lc_blk["block_id"])
        cm_blk = cm_screen_blocks[t]
        cm_preds = load_prediction_block(stage4_plan, cm_blk["prediction_ref"], expected_block=cm_blk["block_id"])

        for s in LANGUAGES:
            p_lc = lc_preds[lc_blk["routes"][s]]
            _, (aligned_lc,) = paired_image_alignment(stage4_plan, p_lc)
            if t not in targets_by_target:
                y_all, known_all = targets_for_prediction(stage4_plan, p_lc)
                targets_by_target[t] = (y_all[aligned_lc], known_all[aligned_lc])
            lc_screen[(s, t)] = (np.asarray(p_lc.logits[aligned_lc], dtype=np.float64), np.asarray(p_lc.estimable, dtype=bool))

            p_cm = cm_preds[cm_blk["routes"][s]]
            _, (aligned_cm,) = paired_image_alignment(stage4_plan, p_cm)
            cm_screen[(s, t)] = (np.asarray(p_cm.logits[aligned_cm], dtype=np.float64), np.asarray(p_cm.estimable, dtype=bool))

    # 3. Evaluate all 5 arms across all 16 routes (4 diagonals + 12 off-diagonals).
    off_pairs = _off_diagonal_pairs()
    diag_pairs = _diagonal_pairs()
    all_pairs = _all_route_pairs()

    arm_route_results = {}
    twelve_route_means = {}
    twelve_route_mean_draws = {}
    four_diag_means = {}

    for arm_idx, arm in enumerate(E44_ARMS, start=1):
        t_arm0 = time.perf_counter()
        route_res = {}
        for s, t in all_pairs:
            y_aligned, known_aligned = targets_by_target[t]
            if arm == "last_caption:raw":
                logits, est = lc_screen[(s, t)]
            elif arm == "last_caption:source_dev_c":
                raw_z, est = lc_screen[(s, t)]
                t_val, b_val = source_cals[s]
                logits = raw_z / t_val + b_val
            elif arm == "last_caption:target_dev_c":
                raw_z, est = lc_screen[(s, t)]
                t_val, b_val = target_cals[(s, t)]
                logits = raw_z / t_val + b_val
            elif arm == "caption_mean:raw":
                logits, est = cm_screen[(s, t)]
            elif arm == "caption_mean:e11_mean_scale":
                if s == t:
                    logits, est = cm_screen[(s, s)]
                else:
                    logits, est = e11_screen_adapted[(s, t)]
            else:
                raise ContractError(f"Unknown E44 arm: {arm!r}")

            probs_aligned = probabilities(logits)
            route_res[_route_key(s, t)] = _evaluate_route_selective(
                y_aligned=y_aligned,
                known_aligned=known_aligned,
                probs_aligned=probs_aligned,
                estimable=est,
                cutoffs_by_cat_nom=cutoffs_by_arm[arm][(s, t)],
                categories=categories,
                primary_indices=primary_indices,
                counts=counts,
                screen_min=screen_min,
            )

        arm_route_results[arm] = route_res
        twelve_route_means[arm], twelve_route_mean_draws[arm] = _aggregate_routes_within_draws(route_res, off_pairs)
        four_diag_means[arm], _ = _aggregate_routes_within_draws(route_res, diag_pairs)

        _emit(
            progress,
            _progress_event(
                "progress",
                f"arm:{arm}",
                2 + arm_idx,
                total_units,
                compute_s=time.perf_counter() - t_arm0,
                extra={"model": model, "arm": arm},
            ),
        )

    # 4. Mandatory Phase 1 E19 parity check (<= 1e-12) and paired contrasts vs last_caption:raw.
    t_fin0 = time.perf_counter()
    parity_tol = float(plan["decision_ranges"]["E44"]["checks"]["phase1_e19_repro_tol"])
    parity = _verify_phase1_e19_parity(plan, model, arm_route_results["last_caption:raw"], tol=parity_tol)

    paired_vs_raw = {}
    for arm in E44_ARMS:
        if arm == "last_caption:raw":
            continue
        paired_vs_raw[arm] = _compute_paired_vs_raw(
            arm_route_results[arm],
            arm_route_results["last_caption:raw"],
            twelve_route_mean_draws[arm],
            twelve_route_mean_draws["last_caption:raw"],
            off_pairs,
        )

    decision_summary = evaluate_e44_decision_summary(
        twelve_route_means, paired_vs_raw, plan["decision_ranges"]["E44"]
    )

    body = {
        "schema": SCHEMA_ANALYSIS,
        "experiment": "E44",
        "plan": plain(plan.reference),
        "plan_identity": plan["plan_identity"],
        "model": model,
        "midpoint_depth": int(inv["midpoint_depth"]),
        "condition": E44_CONDITION,
        "nominals": list(E44_NOMINALS),
        "arms": list(E44_ARMS),
        "arm_supervision": dict(E44_ARM_SUPERVISION),
        "inputs": expected_inputs,
        "phase1_e19_parity": parity,
        "twelve_route_mean": twelve_route_means,
        "four_diagonal_mean": four_diag_means,
        "paired_vs_raw": paired_vs_raw,
        "per_route": {arm: _strip_private_draws(arm_route_results[arm]) for arm in E44_ARMS},
        "decision_summary": decision_summary,
        "confirm_excluded": True,
    }

    upstream = [plain(plan.reference), plain(binding.stage4_release_ref), cal_ref, *e17_dev_c_refs, *e11_screen_refs]
    t_write0 = time.perf_counter()
    analysis_ref = publish_phase3(
        resolved_root,
        protocol,
        plan["dataset_fingerprint"],
        f"{E44_RESULTS_NAMESPACE}/{model}",
        KIND_ANALYSIS,
        body,
        producer,
        upstream=upstream,
        run_id=run_id,
    )
    write_s = time.perf_counter() - t_write0
    _emit(
        progress,
        _progress_event(
            "progress",
            "publish",
            total_units,
            total_units,
            compute_s=time.perf_counter() - t_fin0 - write_s,
            write_s=write_s,
            extra={"model": model, "analysis_ref": plain(analysis_ref)},
        ),
    )

    summary = {
        "event": "complete",
        "status": "passed",
        "experiment": "E44",
        "model": model,
        "resumed": False,
        "analysis_ref": plain(analysis_ref),
        "phase1_e19_parity": parity,
        "decision_summary": decision_summary,
        "elapsed_s": round(time.monotonic() - t_start, 3),
    }
    _emit(progress, summary)
    return summary


def _parse_ref_arg(raw):
    text = raw.strip()
    return json.loads(text) if text.startswith("{") else text


def main(argv=None):
    parser = argparse.ArgumentParser(description="Phase 3 Stage A2 Worker 3 (E44): selective confidence.")
    parser.add_argument("--phase1-root", default=None)
    parser.add_argument("--protocol", default=str(REPO / "configs/protocol.json"))
    parser.add_argument("--plan", required=True, help="Explicit phase3_cpu_plan _COMPLETE.json path or JSON ref")
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--resume", default=None, help="Optional explicit completed E44 analysis ref")
    parser.add_argument("--storage-scope", default="persistent-drive",
                        choices=("persistent-drive", "local-development"))
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)

    root = phase1_root(args.phase1_root)
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    producer = current_producer(scope=args.storage_scope)
    _default_emit({"event": "e44_start", "model": args.model, "code_commit": producer["code_commit"]})
    run_e44(
        root,
        protocol,
        _parse_ref_arg(args.plan),
        args.model,
        producer,
        resume_analysis_ref=_parse_ref_arg(args.resume) if args.resume else None,
        run_id=args.run_id,
        progress=_default_emit,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        _default_emit({"status": "blocked", "blockers": [str(exc)]})
        raise SystemExit(2)
