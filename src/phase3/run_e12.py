"""Phase 3 Stage A2 CPU worker for E12: PCA-64 Procrustes vs ridge linear maps on caption_mean G_res.

Implements phase3.md §A2.1-A2.2 (Worker 4 - E12):
1. Binds `phase3_cpu_plan` via `load_phase3_cpu_plan(root, protocol, plan_ref, bind=True)`.
   Enforces `require_open_split`, `protocol["category_order"]`, `protocol["primary_categories"]`,
   and `int(protocol["minimum_positive_and_negative"]["SCREEN"])`. Never touches `CONFIRM`
   or mutates Phase 1 / Phase 2 directories.
2. Reads verified midpoint `image_caption` / `caption_mean` `FIT`, `DEV-S`, and `SCREEN`
   features across all 4 languages (`en`, `hi`, `bn`, `ta`) and verifies exact 2,160 paired
   `FIT`, 180 paired `DEV-S`, and 360 paired `SCREEN` image IDs.
3. For each source language `s`, fits a mean-centered source-`FIT` PCA basis `r = min(64, source_rank)`
   and fits an `r=64` source linear category head (`C in {0.01, 0.1, 1.0, 10.0}` selected on
   source `DEV-S` primary macro AUROC with deterministic tie-break favoring stronger regularization).
   Reports full-space Stage 4 `caption_mean` performance separately as reference.
4. For each off-diagonal route `s != t` (12 directed routes), projects target `t` features into
   source `s`'s `r=64` basis (centered by target `FIT` mean) and compares 3 arms:
   - `identity_centered`
   - `orthogonal_procrustes` (SVD orthogonal `Q in O(r)` on paired 2,160 `FIT` anchors, zero target labels)
   - `ridge_linear` (`alpha in {0.01, 0.1, 1.0, 10.0, 100.0}` selected strictly by paired 180-image
     `DEV-S` feature reconstruction MSE, zero target semantic labels)
5. Evaluates paired `SCREEN` `AUROC`/`AP`/`Brier`/`NLL` gap reductions, `ridge - procrustes`
   constraint contrast, recovered fraction of `r=64` `G_res`, and paired `SCREEN` feature
   reconstruction MSE inside every `(2000, 360)` bootstrap draw before taking linear 2.5th/97.5th
   percentiles.
6. Publishes under `phase3/maps/e12/<model>/<run_id>/`, `phase3/predictions/e12/<model>/<run_id>/`,
   and `phase3/results/e12/<model>/<run_id>/` with explicit-ref resume via `verify_resume_ref`.
"""

import argparse
import json
from pathlib import Path
import sys
import time

from src.common.contracts import ContractError, LANGUAGES, fingerprint
from src.common.storage import new_run_id, phase1_root, within
from src.phase3.interfaces import (
    KIND_ANALYSIS,
    KIND_MAPS,
    KIND_PREDICTION,
    MODELS,
    PHASE3_CPU_DECISION_RANGES,
    REPO,
    SCHEMA_ANALYSIS,
    SCHEMA_MAPS,
    SCHEMA_PREDICTION,
    check,
    contract,
    current_producer,
    frozen,
    load_phase3_cpu_plan,
    normalize_ref,
    plain,
    publish_phase3,
    require_open_split,
    verify_resume_ref,
)

E12_CONDITION = "image_caption"
E12_POOLING = "caption_mean"
E12_ARMS = ("identity_centered", "orthogonal_procrustes", "ridge_linear")
E12_METRICS = ("auroc", "ap", "brier", "nll")
LOSS_METRICS = frozenset({"brier", "nll"})

EXPECTED_FIT_IMAGES = 2160
EXPECTED_DEV_S_IMAGES = 180
EXPECTED_SCREEN_IMAGES = 360
PCA_MAX_RANK = 64
C_GRID = (0.01, 0.1, 1.0, 10.0)
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)

SUPERVISION_DIAGONAL = "within_context_source_fit"
SUPERVISION_IDENTITY = "source_only_transfer"
SUPERVISION_ADAPTED = "paired_image_adapted"

E12_MAPS_NAMESPACE = "phase3/maps/e12"
E12_PREDICTIONS_NAMESPACE = "phase3/predictions/e12"
E12_RESULTS_NAMESPACE = "phase3/results/e12"


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


def _mean_across_routes(route_summaries, route_draws_list, *, route_count):
    """Equal-weight mean across routes within every bootstrap draw."""
    import numpy as np

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
            "route_count": int(route_count),
        },
        mean_draws,
    )


def _evaluate_logits(logits, estimable, aligned, y, known, counts, categories, total_draws):
    """Evaluate one route's 25-category point estimates, bootstrap draws, and support."""
    import numpy as np
    from src.stage4.metrics import binary_metrics, bootstrap_binary_metrics, probabilities

    n_cats = len(categories)
    probs = probabilities(logits[aligned])
    est = np.asarray(estimable, dtype=bool)
    check(est.shape == (n_cats,), f"Unexpected estimable shape: {est.shape}")
    nan_draws = np.full(total_draws, np.nan, dtype=np.float64)

    points = {m: [] for m in E12_METRICS}
    draws = {m: [] for m in E12_METRICS}
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
            for m in E12_METRICS:
                points[m].append(bm["metrics"][m])
                draws[m].append(bd[m])
        else:
            for m in E12_METRICS:
                points[m].append(None)
                draws[m].append(nan_draws)
    return {
        "points": points,
        "draws": draws,
        "supports": supports,
        "unestimable": [not bool(est[col]) for col in range(n_cats)],
    }


def _summarize_eval(eval_res, categories, primary_indices, minimum):
    summary = {}
    macro_draws = {}
    for m in E12_METRICS:
        macro, m_draws, per_cat = _summarize_macro_and_categories(
            categories=categories,
            primary_indices=primary_indices,
            cat_values=eval_res["points"][m],
            cat_draws=eval_res["draws"][m],
            cat_supports=eval_res["supports"],
            cat_unestimable=eval_res["unestimable"],
            minimum=minimum,
        )
        summary[m] = {"primary_macro": macro, "per_category": per_cat}
        macro_draws[m] = m_draws
    return summary, macro_draws


def _fit_pca_basis(X_fit_s, max_rank=PCA_MAX_RANK):
    """Fit mean-centered source-FIT PCA basis of rank r = min(max_rank, source_rank)."""
    import numpy as np

    X_f64 = np.asarray(X_fit_s, dtype=np.float64)
    check(X_f64.ndim == 2 and np.isfinite(X_f64).all(), "FIT features must be finite 2D array")
    mu_s = np.mean(X_f64, axis=0)
    centered = X_f64 - mu_s
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    if S.size > 0 and S[0] > 0.0:
        tol = np.finfo(np.float64).eps * max(centered.shape) * S[0]
        source_rank = int(np.sum(S > tol))
    else:
        source_rank = 0
    r = min(int(max_rank), source_rank)
    check(r >= 1, f"Source FIT rank must be >= 1; got {source_rank}")
    V_s = np.asarray(Vt[:r, :].T, dtype=np.float64)
    check(
        np.allclose(V_s.T @ V_s, np.eye(r, dtype=np.float64), atol=1e-10),
        "Source PCA basis is not orthonormal",
    )
    return mu_s, V_s, source_rank, r


def _fit_pca_source_head(
    Z_fit_s,
    y_fit,
    known_fit,
    Z_dev_s,
    y_dev_s,
    known_dev_s,
    dev_contexts,
    primary_indices,
):
    """Fit r=64 source linear head on source FIT and select C on source DEV-S primary macro AUROC."""
    import numpy as np
    from src.stage3.engine import choose, fit_heads, metrics, statistics, transform

    weights_fit = np.ones(Z_fit_s.shape[0], dtype=np.float64)
    pca_stats = statistics(Z_fit_s, weights_fit)
    Z_fit_std = transform(Z_fit_s, pca_stats)
    Z_dev_std = transform(Z_dev_s, pca_stats)

    scores = {}
    candidates = {}
    for c_val in C_GRID:
        coef, intercept, valid, records = fit_heads(Z_fit_std, y_fit, known_fit, weights_fit, c_val)
        dev_logits = np.asarray(Z_dev_std @ coef.T, dtype=np.float64) + intercept
        rep = metrics(dev_logits, y_dev_s, known_dev_s, valid, dev_contexts, primary_indices)
        check(rep["selection_score"] is not None, f"Undefined DEV-S primary macro AUROC for C={c_val}")
        scores[c_val] = float(rep["selection_score"])
        candidates[c_val] = {
            "coef": coef,
            "intercept": intercept,
            "estimable": valid,
            "dev_s_selection_score": float(rep["selection_score"]),
            "converged_heads": int(sum(1 for r in records if r["status"] == "fitted")),
        }

    best_c = choose(scores)
    sel = candidates[best_c]
    return {
        "pca_stats": pca_stats,
        "selected_C": float(best_c),
        "c_scores": {str(c): float(v) for c, v in scores.items()},
        "coef": sel["coef"],
        "intercept": sel["intercept"],
        "estimable": sel["estimable"],
        "dev_s_selection_score": sel["dev_s_selection_score"],
    }


def _fit_orthogonal_procrustes(Z_fit_t, Z_fit_s):
    """Fit SVD orthogonal Procrustes Q in O(r) minimizing ||Z_fit_t Q - Z_fit_s||_F^2 (zero target labels)."""
    import numpy as np

    M = Z_fit_t.T @ Z_fit_s
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    Q = np.asarray(U @ Vt, dtype=np.float64)
    r = Q.shape[0]
    check(Q.shape == (r, r) and np.isfinite(Q).all(), "Invalid Procrustes matrix shape/values")
    check(
        np.allclose(Q.T @ Q, np.eye(r, dtype=np.float64), atol=1e-10),
        "Procrustes solution failed orthogonality check",
    )
    return Q


def _fit_ridge_linear(Z_fit_t, Z_fit_s, Z_dev_t, Z_dev_s, alphas=RIDGE_ALPHAS):
    """Fit ridge W on paired FIT and select alpha strictly by paired DEV-S feature MSE (zero target labels)."""
    import numpy as np

    r = Z_fit_t.shape[1]
    G = Z_fit_t.T @ Z_fit_t
    C_mat = Z_fit_t.T @ Z_fit_s
    eye = np.eye(r, dtype=np.float64)

    w_by_alpha = {}
    mse_by_alpha = {}
    for alpha in alphas:
        a = float(alpha)
        W_a = np.linalg.solve(G + a * eye, C_mat)
        check(np.isfinite(W_a).all(), f"Nonfinite ridge solution for alpha={a}")
        dev_pred = Z_dev_t @ W_a
        mse = float(np.mean((dev_pred - Z_dev_s) ** 2))
        check(np.isfinite(mse), f"Nonfinite DEV-S MSE for alpha={a}")
        w_by_alpha[a] = np.asarray(W_a, dtype=np.float64)
        mse_by_alpha[a] = mse

    best_mse = min(mse_by_alpha.values())
    # Deterministic tie-break favoring stronger regularization (larger alpha).
    best_alpha = max(a for a, v in mse_by_alpha.items() if v - best_mse <= 1e-12)
    return w_by_alpha[best_alpha], float(best_alpha), {str(a): float(v) for a, v in mse_by_alpha.items()}


def _score_pca_head(Z_adapted, pca_stats, coef, intercept, estimable):
    """Score r=64 adapted features through the fitted source PCA head."""
    import numpy as np
    from src.stage3.engine import transform

    Z_std = transform(np.asarray(Z_adapted, dtype=np.float64), pca_stats)
    logits = np.asarray(Z_std @ coef.T, dtype=np.float64) + intercept
    check(np.isfinite(logits).all(), "Nonfinite E12 PCA logits")
    est = np.asarray(estimable, dtype=bool)
    check(not np.any(logits * (~est[None, :])), "Unestimable E12 logit padding mismatch")
    return logits


def _read_and_verify_features(binding, model):
    """Read midpoint caption_mean FIT/DEV-S/SCREEN features and verify exact cross-language pairing."""
    import numpy as np
    from src.phase2.interfaces import read_verified_midpoint_features

    for split in ("FIT", "DEV-S", "SCREEN"):
        require_open_split(split)

    rows_by_split = {s: {} for s in ("FIT", "DEV-S", "SCREEN")}
    feats_by_split = {s: {} for s in ("FIT", "DEV-S", "SCREEN")}

    for lang in LANGUAGES:
        for split in ("FIT", "DEV-S", "SCREEN"):
            rows, mat = read_verified_midpoint_features(
                binding, model, E12_CONDITION, E12_POOLING, lang, split
            )
            arr = np.asarray(mat, dtype=np.float64)
            check(arr.ndim == 2 and np.isfinite(arr).all(), f"Invalid {split} features for {model}/{lang}")
            rows_by_split[split][lang] = rows
            feats_by_split[split][lang] = arr

    fit_ids_ref = None
    dev_s_ids_ref = None
    dims = set()
    for lang in LANGUAGES:
        dims.add(int(feats_by_split["FIT"][lang].shape[1]))
        dims.add(int(feats_by_split["DEV-S"][lang].shape[1]))
        dims.add(int(feats_by_split["SCREEN"][lang].shape[1]))

        f_rows = rows_by_split["FIT"][lang]
        check(len(f_rows) == EXPECTED_FIT_IMAGES, f"Expected {EXPECTED_FIT_IMAGES} FIT rows for {lang}")
        f_ids = [r["image_id"] for r in f_rows]
        check(len(set(f_ids)) == len(f_ids), f"Duplicate FIT image_id for {lang}")
        if fit_ids_ref is None:
            fit_ids_ref = f_ids
        else:
            check(f_ids == fit_ids_ref, f"Unpaired FIT image_ids for {lang}")

        d_rows = rows_by_split["DEV-S"][lang]
        check(len(d_rows) == EXPECTED_DEV_S_IMAGES, f"Expected {EXPECTED_DEV_S_IMAGES} DEV-S rows for {lang}")
        d_ids = [r["image_id"] for r in d_rows]
        check(len(set(d_ids)) == len(d_ids), f"Duplicate DEV-S image_id for {lang}")
        if dev_s_ids_ref is None:
            dev_s_ids_ref = d_ids
        else:
            check(d_ids == dev_s_ids_ref, f"Unpaired DEV-S image_ids for {lang}")

    check(len(dims) == 1, f"Inconsistent feature dimensions across languages/splits: {dims}")
    dimension = next(iter(dims))

    # Align SCREEN rows and matrices to canonical sorted stage4_plan["screen_ids"].
    screen_ids = tuple(binding.stage4_plan["screen_ids"])
    check(len(screen_ids) == EXPECTED_SCREEN_IMAGES, f"Expected {EXPECTED_SCREEN_IMAGES} SCREEN IDs")
    aligned_screen_feats = {}
    aligned_screen_rows = {}
    for lang in LANGUAGES:
        s_rows = rows_by_split["SCREEN"][lang]
        check(len(s_rows) == EXPECTED_SCREEN_IMAGES, f"Expected {EXPECTED_SCREEN_IMAGES} SCREEN rows for {lang}")
        pos = {r["image_id"]: i for i, r in enumerate(s_rows)}
        check(set(pos) == set(screen_ids), f"SCREEN image_id population mismatch for {lang}")
        idx = np.array([pos[iid] for iid in screen_ids], dtype=np.intp)
        aligned_screen_feats[lang] = feats_by_split["SCREEN"][lang][idx]
        aligned_screen_rows[lang] = tuple(s_rows[i] for i in idx)

    pairing_check = {
        "paired_fit_images": int(len(fit_ids_ref)),
        "fit_image_ids_sha256": fingerprint(fit_ids_ref),
        "paired_dev_s_images": int(len(dev_s_ids_ref)),
        "dev_s_image_ids_sha256": fingerprint(dev_s_ids_ref),
        "paired_screen_images": int(len(screen_ids)),
        "screen_image_ids_sha256": fingerprint(list(screen_ids)),
        "dimension": int(dimension),
        "passed": True,
    }
    return {
        "dimension": dimension,
        "screen_ids": screen_ids,
        "pairing_check": pairing_check,
        "fit_rows": rows_by_split["FIT"],
        "dev_s_rows": rows_by_split["DEV-S"],
        "screen_rows": aligned_screen_rows,
        "fit": feats_by_split["FIT"],
        "dev_s": feats_by_split["DEV-S"],
        "screen": aligned_screen_feats,
    }


def _evaluate_full_space_reference(
    binding, inv, screen_ids, y, known, counts, categories, primary_indices, minimum, total_draws
):
    """Evaluate Stage 4 full-space caption_mean SCREEN performance separately as reference."""
    from src.stage4.metrics import paired_image_alignment
    from src.stage4.predictions import load_prediction_block

    diag_summaries = {}
    diag_draws = {}
    off_summaries = {}
    off_draws = {}
    g_res_summaries = {}
    g_res_draws = {}

    blocks = {}
    aligned_by_lang = {}
    for lang in LANGUAGES:
        blk_info = inv["screen_blocks"][lang]
        pred = load_prediction_block(
            binding.stage4_plan, blk_info["prediction_ref"], expected_block=blk_info["block_id"]
        )
        ids, (aligned,) = paired_image_alignment(binding.stage4_plan, pred)
        check(ids == screen_ids, "Full-space Stage 4 SCREEN ID alignment mismatch")
        blocks[lang] = pred
        aligned_by_lang[lang] = aligned

        diag_idx = list(blk_info["routes"].keys()).index(lang)
        # Map route_id explicitly using plan["blocks"]
        route_ids = [r["route_id"] for r in binding.stage4_plan["blocks"][blk_info["block_id"]]["routes"]]
        diag_idx = route_ids.index(blk_info["routes"][lang])
        ev = _evaluate_logits(
            pred.logits[diag_idx],
            pred.estimable[diag_idx],
            aligned,
            y,
            known,
            counts,
            categories,
            total_draws,
        )
        diag_summaries[lang], diag_draws[lang] = _summarize_eval(ev, categories, primary_indices, minimum)

    for s, t in _off_diagonal_pairs():
        blk_info = inv["screen_blocks"][t]
        route_ids = [r["route_id"] for r in binding.stage4_plan["blocks"][blk_info["block_id"]]["routes"]]
        src_idx = route_ids.index(blk_info["routes"][s])
        pred = blocks[t]
        ev = _evaluate_logits(
            pred.logits[src_idx],
            pred.estimable[src_idx],
            aligned_by_lang[t],
            y,
            known,
            counts,
            categories,
            total_draws,
        )
        rk = _route_key(s, t)
        off_summaries[rk], off_draws[rk] = _summarize_eval(ev, categories, primary_indices, minimum)
        g_res_summaries[rk] = {}
        g_res_draws[rk] = {}
        for m in E12_METRICS:
            d_macro = diag_summaries[t][m]["primary_macro"]
            r_macro = off_summaries[rk][m]["primary_macro"]
            g_val = _gap_value(m, d_macro["value"], r_macro["value"])
            g_dr = _gap_draws(m, diag_draws[t][m], off_draws[rk][m])
            g_res_draws[rk][m] = g_dr
            g_res_summaries[rk][m] = {
                "value": g_val,
                "interval": _interval_record(g_dr),
                "status": "ok" if (d_macro["status"] == "ok" and r_macro["status"] == "ok") else "descriptive_low_support",
            }

    mean_diag = {}
    mean_off = {}
    mean_g_res = {}
    for m in E12_METRICS:
        mean_diag[m], _ = _mean_across_routes(
            [diag_summaries[t][m]["primary_macro"] for t in LANGUAGES],
            [diag_draws[t][m] for t in LANGUAGES],
            route_count=len(LANGUAGES),
        )
        mean_off[m], _ = _mean_across_routes(
            [off_summaries[_route_key(s, t)][m]["primary_macro"] for s, t in _off_diagonal_pairs()],
            [off_draws[_route_key(s, t)][m] for s, t in _off_diagonal_pairs()],
            route_count=12,
        )
        mean_g_res[m], _ = _mean_across_routes(
            [g_res_summaries[_route_key(s, t)][m] for s, t in _off_diagonal_pairs()],
            [g_res_draws[_route_key(s, t)][m] for s, t in _off_diagonal_pairs()],
            route_count=12,
        )

    return {
        "note": "Full-space Stage 4 caption_mean reference reported separately from r=64 PCA subspace arms.",
        "diagonal_mean": mean_diag,
        "off_diagonal_mean": mean_off,
        "g_res_mean": mean_g_res,
    }


def _evaluate_decision_summary(arms_summary, contrasts_summary, decision_ranges):
    """Evaluate frozen Phase 3 E12 decision thresholds."""
    min_gap_red = float(decision_ranges["auroc_gap_reduction_min"])
    min_frac = float(decision_ranges["or_fraction_of_g_res_min"])
    min_contrast = float(decision_ranges["constraint_diff_min"])
    neg_upper_lt = float(decision_ranges["negligible_ci_upper_lt"])

    def _is_consequential(arm_name):
        gr = arms_summary[arm_name]["gap_reduction"]["auroc"]
        fr = arms_summary[arm_name]["recovered_fraction_of_g_res"]["auroc"]
        gr_ci_pos = (
            gr["interval"]["lower"] is not None
            and gr["interval"]["lower"] > 0.0
            and gr["interval"]["valid_draws"] >= 1900
        )
        fr_ci_pos = (
            fr["interval"]["lower"] is not None
            and fr["interval"]["lower"] > 0.0
            and fr["interval"]["valid_draws"] >= 1900
        )
        by_abs = gr["value"] is not None and gr["value"] >= min_gap_red and gr_ci_pos
        by_frac = fr["value"] is not None and fr["value"] >= min_frac and fr_ci_pos
        return bool(by_abs or by_frac)

    def _is_negligible(arm_name):
        upper = arms_summary[arm_name]["gap_reduction"]["auroc"]["interval"]["upper"]
        return upper is not None and upper < neg_upper_lt

    procrustes_conseq = _is_consequential("orthogonal_procrustes")
    ridge_conseq = _is_consequential("ridge_linear")

    c_auroc = contrasts_summary["ridge_minus_procrustes"]["auroc"]
    lo, hi = c_auroc["interval"]["lower"], c_auroc["interval"]["upper"]
    ci_excludes_zero = lo is not None and hi is not None and (lo > 0.0 or hi < 0.0)
    constraint_matters = bool(
        c_auroc["value"] is not None
        and abs(c_auroc["value"]) >= min_contrast
        and ci_excludes_zero
        and c_auroc["interval"]["valid_draws"] >= 1900
    )
    negligible = bool(_is_negligible("orthogonal_procrustes") and _is_negligible("ridge_linear"))

    if procrustes_conseq or ridge_conseq:
        outcome = "consequential_recovery_with_constraint_difference" if constraint_matters else "consequential_recovery"
    elif negligible:
        outcome = "negligible"
    else:
        outcome = "inconclusive"

    return {
        "procrustes_consequential": procrustes_conseq,
        "ridge_consequential": ridge_conseq,
        "constraint_matters": constraint_matters,
        "negligible": negligible,
        "outcome": outcome,
    }


@contract
def run_e12(
    root,
    protocol,
    plan_ref,
    model,
    producer,
    *,
    resume_maps_ref=None,
    resume_predictions_ref=None,
    resume_analysis_ref=None,
    run_id=None,
    progress=None,
):
    """Execute Phase 3 Stage A2 E12 for one model and publish maps, predictions, and analysis."""
    import numpy as np
    from src.stage3.readouts import targets
    from src.stage4.metrics import bootstrap_counts
    from src.stage4.plan import load_bootstrap

    check(model in MODELS, f"Unknown model selector: {model!r}")
    resolved_root = Path(root).resolve()
    effective_run_id = run_id or new_run_id()

    t0 = time.monotonic()
    plan = load_phase3_cpu_plan(resolved_root, protocol, plan_ref, bind=True)
    binding = plan.binding
    inv = plan["e12_inventory"][model]
    check(inv["condition"] == E12_CONDITION and inv["pooling"] == E12_POOLING, "E12 inventory mismatch")
    decision_ranges = plain(plan["decision_ranges"]["E12"])

    categories = list(protocol["category_order"])
    primary_categories = list(protocol["primary_categories"])
    primary_indices = [categories.index(c) for c in primary_categories]
    screen_minimum = int(protocol["minimum_positive_and_negative"]["SCREEN"])
    n_cats = len(categories)

    expected_inputs = {
        "experiment": "E12",
        "model": model,
        "midpoint_depth": int(inv["midpoint_depth"]),
        "condition": E12_CONDITION,
        "pooling": E12_POOLING,
        "pca_max_rank": int(inv["pca_rank"]),
        "feature_releases": plain(inv["feature_releases"]),
        "screen_blocks": plain(inv["screen_blocks"]),
    }

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
        _emit_progress(progress, "e12_resumed_analysis", unit="resume", completed=4, total=4, model=model)
        return frozen({
            "model": model,
            "run_id": effective_run_id,
            "maps_ref": plain(body["maps_ref"]),
            "predictions_ref": plain(body["predictions_ref"]),
            "analysis_ref": plain(ref),
            "resumed": True,
        })

    _emit_progress(progress, "e12_load_features", unit="load_features", completed=0, total=4, model=model)
    t_read = time.monotonic()
    data = _read_and_verify_features(binding, model)
    draws = load_bootstrap(binding.stage4_plan)
    total_draws = int(draws.shape[0])
    counts = bootstrap_counts(draws, len(data["screen_ids"]))

    # Ground-truth targets (used ONLY for source head fitting and final SCREEN evaluation).
    fit_y, fit_known = targets(binding.stage4_plan.fit, data["fit_rows"]["en"], "true")
    dev_y, dev_known = targets(binding.stage4_plan.fit, data["dev_s_rows"]["en"], "true")
    scr_rows_en = data["screen_rows"]["en"]
    scr_y, scr_known = targets(binding.stage4_plan.fit, list(scr_rows_en), "true")
    read_s = time.monotonic() - t_read

    t_comp = time.monotonic()
    # Step 1: Maps & source r=64 heads (or resume from verified maps_ref).
    if resume_maps_ref is not None:
        maps_ref_obj, maps_body, _ = verify_resume_ref(
            resolved_root,
            protocol,
            resume_maps_ref,
            KIND_MAPS,
            expected_schema=SCHEMA_MAPS,
            expected_dataset=plan["dataset_fingerprint"],
            expected_plan_identity=plan["plan_identity"],
            expected_owner=model,
            expected_inputs=expected_inputs,
        )
        maps_ref = plain(maps_ref_obj)
        maps_dir = within(resolved_root, maps_ref["path"]).parent
        pca_means_arr = np.load(maps_dir / "pca_means.npy", allow_pickle=False)
        pca_bases_arr = np.load(maps_dir / "pca_bases.npy", allow_pickle=False)
        pca_scales_arr = np.load(maps_dir / "pca_stats_scale.npy", allow_pickle=False)
        head_coefs_arr = np.load(maps_dir / "head_coefs.npy", allow_pickle=False)
        head_intercepts_arr = np.load(maps_dir / "head_intercepts.npy", allow_pickle=False)
        head_estimable_arr = np.load(maps_dir / "head_estimable.npy", allow_pickle=False)
        procrustes_q_arr = np.load(maps_dir / "procrustes_q.npy", allow_pickle=False)
        ridge_w_arr = np.load(maps_dir / "ridge_w.npy", allow_pickle=False)

        source_pca = {}
        for s_idx, s in enumerate(LANGUAGES):
            r_s = int(maps_body["source_pca"][s]["rank"])
            source_pca[s] = {
                "mu": pca_means_arr[s_idx],
                "V": pca_bases_arr[s_idx, :, :r_s],
                "rank": r_s,
                "source_rank": int(maps_body["source_pca"][s]["source_rank"]),
                "head": {
                    "pca_stats": {
                        "mean": np.zeros(r_s, dtype=np.float64),
                        "scale": pca_scales_arr[s_idx, :r_s],
                        "center": True,
                    },
                    "coef": head_coefs_arr[s_idx, :, :r_s],
                    "intercept": head_intercepts_arr[s_idx],
                    "estimable": head_estimable_arr[s_idx],
                    "selected_C": float(maps_body["source_pca"][s]["selected_C"]),
                    "c_scores": maps_body["source_pca"][s]["c_scores"],
                    "dev_s_selection_score": float(maps_body["source_pca"][s]["dev_s_selection_score"]),
                },
            }
        route_maps = {}
        for r_idx, (s, t) in enumerate(_off_diagonal_pairs()):
            rk = _route_key(s, t)
            r_s = source_pca[s]["rank"]
            route_maps[(s, t)] = {
                "Q": procrustes_q_arr[r_idx, :r_s, :r_s],
                "W": ridge_w_arr[r_idx, :r_s, :r_s],
                "selected_alpha": float(maps_body["route_maps"][rk]["selected_alpha"]),
                "dev_s_mse_by_alpha": maps_body["route_maps"][rk]["dev_s_mse_by_alpha"],
            }
    else:
        source_pca = {}
        for s in LANGUAGES:
            mu_s, V_s, src_rank, r_s = _fit_pca_basis(data["fit"][s], max_rank=int(inv["pca_rank"]))
            Z_fit_s = (data["fit"][s] - mu_s) @ V_s
            Z_dev_s = (data["dev_s"][s] - mu_s) @ V_s
            dev_contexts = [s] * EXPECTED_DEV_S_IMAGES
            head = _fit_pca_source_head(
                Z_fit_s, fit_y, fit_known, Z_dev_s, dev_y, dev_known, dev_contexts, primary_indices
            )
            source_pca[s] = {
                "mu": mu_s,
                "V": V_s,
                "rank": r_s,
                "source_rank": src_rank,
                "head": head,
            }

        route_maps = {}
        for s, t in _off_diagonal_pairs():
            V_s = source_pca[s]["V"]
            mu_s = source_pca[s]["mu"]
            mu_t = np.mean(data["fit"][t], axis=0)
            Z_fit_s = (data["fit"][s] - mu_s) @ V_s
            Z_dev_s = (data["dev_s"][s] - mu_s) @ V_s
            Z_fit_t = (data["fit"][t] - mu_t) @ V_s
            Z_dev_t = (data["dev_s"][t] - mu_t) @ V_s

            Q = _fit_orthogonal_procrustes(Z_fit_t, Z_fit_s)
            W, best_alpha, mse_by_alpha = _fit_ridge_linear(Z_fit_t, Z_fit_s, Z_dev_t, Z_dev_s)
            route_maps[(s, t)] = {
                "Q": Q,
                "W": W,
                "selected_alpha": best_alpha,
                "dev_s_mse_by_alpha": mse_by_alpha,
            }

        max_r = max(source_pca[s]["rank"] for s in LANGUAGES)
        d = data["dimension"]
        pca_means_arr = np.stack([source_pca[s]["mu"] for s in LANGUAGES], axis=0)
        pca_bases_arr = np.zeros((4, d, max_r), dtype=np.float64)
        pca_scales_arr = np.ones((4, max_r), dtype=np.float64)
        head_coefs_arr = np.zeros((4, n_cats, max_r), dtype=np.float64)
        head_intercepts_arr = np.zeros((4, n_cats), dtype=np.float64)
        head_estimable_arr = np.zeros((4, n_cats), dtype=bool)
        for s_idx, s in enumerate(LANGUAGES):
            r_s = source_pca[s]["rank"]
            pca_bases_arr[s_idx, :, :r_s] = source_pca[s]["V"]
            pca_scales_arr[s_idx, :r_s] = source_pca[s]["head"]["pca_stats"]["scale"]
            head_coefs_arr[s_idx, :, :r_s] = source_pca[s]["head"]["coef"]
            head_intercepts_arr[s_idx] = source_pca[s]["head"]["intercept"]
            head_estimable_arr[s_idx] = source_pca[s]["head"]["estimable"]

        procrustes_q_arr = np.zeros((12, max_r, max_r), dtype=np.float64)
        ridge_w_arr = np.zeros((12, max_r, max_r), dtype=np.float64)
        for r_idx, (s, t) in enumerate(_off_diagonal_pairs()):
            r_s = source_pca[s]["rank"]
            procrustes_q_arr[r_idx, :r_s, :r_s] = route_maps[(s, t)]["Q"]
            ridge_w_arr[r_idx, :r_s, :r_s] = route_maps[(s, t)]["W"]

        maps_body = {
            "schema": SCHEMA_MAPS,
            "experiment": "E12",
            "plan": plain(plan.reference),
            "plan_identity": plan["plan_identity"],
            "model": model,
            "inputs": expected_inputs,
            "pairing_check": data["pairing_check"],
            "zero_target_semantic_labels": True,
            "source_pca": {
                s: {
                    "rank": source_pca[s]["rank"],
                    "source_rank": source_pca[s]["source_rank"],
                    "selected_C": source_pca[s]["head"]["selected_C"],
                    "c_scores": source_pca[s]["head"]["c_scores"],
                    "dev_s_selection_score": source_pca[s]["head"]["dev_s_selection_score"],
                }
                for s in LANGUAGES
            },
            "route_maps": {
                _route_key(s, t): {
                    "source_language": s,
                    "target_language": t,
                    "selected_alpha": route_maps[(s, t)]["selected_alpha"],
                    "dev_s_mse_by_alpha": route_maps[(s, t)]["dev_s_mse_by_alpha"],
                }
                for s, t in _off_diagonal_pairs()
            },
            "confirm_excluded": True,
        }
        maps_ref = plain(
            publish_phase3(
                resolved_root,
                protocol,
                plan["dataset_fingerprint"],
                f"{E12_MAPS_NAMESPACE}/{model}",
                KIND_MAPS,
                maps_body,
                producer,
                arrays={
                    "pca_means": pca_means_arr,
                    "pca_bases": pca_bases_arr,
                    "pca_stats_scale": pca_scales_arr,
                    "head_coefs": head_coefs_arr,
                    "head_intercepts": head_intercepts_arr,
                    "head_estimable": head_estimable_arr,
                    "procrustes_q": procrustes_q_arr,
                    "ridge_w": ridge_w_arr,
                },
                upstream=[plain(plan.reference)],
                run_id=effective_run_id,
            )
        )
    _emit_progress(progress, "e12_maps_ready", unit="fit_maps", completed=1, total=4, model=model, maps_ref=maps_ref)

    # Step 2: Project SCREEN features and score (or resume from verified predictions_ref).
    n_scr = len(data["screen_ids"])
    screen_mse_per_image = {}
    for s, t in _off_diagonal_pairs():
        V_s = source_pca[s]["V"]
        mu_s = source_pca[s]["mu"]
        mu_t = np.mean(data["fit"][t], axis=0)
        Z_scr_s = (data["screen"][s] - mu_s) @ V_s
        Z_scr_t = (data["screen"][t] - mu_t) @ V_s
        rm = route_maps[(s, t)]
        z_arms = {
            "identity_centered": Z_scr_t,
            "orthogonal_procrustes": Z_scr_t @ rm["Q"],
            "ridge_linear": Z_scr_t @ rm["W"],
        }
        screen_mse_per_image[(s, t)] = {
            arm: np.mean((z_arms[arm] - Z_scr_s) ** 2, axis=1) for arm in E12_ARMS
        }

    if resume_predictions_ref is not None:
        pred_ref_obj, pred_body, _ = verify_resume_ref(
            resolved_root,
            protocol,
            resume_predictions_ref,
            KIND_PREDICTION,
            expected_schema=SCHEMA_PREDICTION,
            expected_dataset=plan["dataset_fingerprint"],
            expected_plan_identity=plan["plan_identity"],
            expected_owner=model,
            expected_inputs=expected_inputs,
        )
        predictions_ref = plain(pred_ref_obj)
        pred_dir = within(resolved_root, predictions_ref["path"]).parent
        diag_logits_arr = np.load(pred_dir / "diag_logits.npy", allow_pickle=False)
        diag_estimable_arr = np.load(pred_dir / "diag_estimable.npy", allow_pickle=False)
        arm_logits_arr = np.load(pred_dir / "arm_logits.npy", allow_pickle=False)
        arm_estimable_arr = np.load(pred_dir / "arm_estimable.npy", allow_pickle=False)
        check(diag_logits_arr.shape == (4, n_scr, n_cats), "Resume diag_logits shape mismatch")
        check(arm_logits_arr.shape == (len(E12_ARMS), 12, n_scr, n_cats), "Resume arm_logits shape mismatch")
    else:
        diag_logits_arr = np.zeros((4, n_scr, n_cats), dtype=np.float64)
        diag_estimable_arr = np.zeros((4, n_cats), dtype=bool)
        for s_idx, s in enumerate(LANGUAGES):
            Z_scr_s = (data["screen"][s] - source_pca[s]["mu"]) @ source_pca[s]["V"]
            h = source_pca[s]["head"]
            diag_logits_arr[s_idx] = _score_pca_head(
                Z_scr_s, h["pca_stats"], h["coef"], h["intercept"], h["estimable"]
            )
            diag_estimable_arr[s_idx] = h["estimable"]

        arm_logits_arr = np.zeros((len(E12_ARMS), 12, n_scr, n_cats), dtype=np.float64)
        arm_estimable_arr = np.zeros((len(E12_ARMS), 12, n_cats), dtype=bool)
        for r_idx, (s, t) in enumerate(_off_diagonal_pairs()):
            V_s = source_pca[s]["V"]
            mu_t = np.mean(data["fit"][t], axis=0)
            Z_scr_t = (data["screen"][t] - mu_t) @ V_s
            rm = route_maps[(s, t)]
            h = source_pca[s]["head"]
            z_arms = {
                "identity_centered": Z_scr_t,
                "orthogonal_procrustes": Z_scr_t @ rm["Q"],
                "ridge_linear": Z_scr_t @ rm["W"],
            }
            for a_idx, arm in enumerate(E12_ARMS):
                arm_logits_arr[a_idx, r_idx] = _score_pca_head(
                    z_arms[arm], h["pca_stats"], h["coef"], h["intercept"], h["estimable"]
                )
                arm_estimable_arr[a_idx, r_idx] = h["estimable"]

        pred_body = {
            "schema": SCHEMA_PREDICTION,
            "experiment": "E12",
            "plan": plain(plan.reference),
            "plan_identity": plan["plan_identity"],
            "model": model,
            "inputs": expected_inputs,
            "maps_ref": maps_ref,
            "arms": list(E12_ARMS),
            "routes": [_route_key(s, t) for s, t in _off_diagonal_pairs()],
            "diagonal_languages": list(LANGUAGES),
            "supervision_labels": {
                "diagonal": SUPERVISION_DIAGONAL,
                "identity_centered": SUPERVISION_IDENTITY,
                "orthogonal_procrustes": SUPERVISION_ADAPTED,
                "ridge_linear": SUPERVISION_ADAPTED,
            },
            "image_ids_sha256": data["pairing_check"]["screen_image_ids_sha256"],
            "confirm_excluded": True,
        }
        predictions_ref = plain(
            publish_phase3(
                resolved_root,
                protocol,
                plan["dataset_fingerprint"],
                f"{E12_PREDICTIONS_NAMESPACE}/{model}",
                KIND_PREDICTION,
                pred_body,
                producer,
                arrays={
                    "diag_logits": diag_logits_arr,
                    "diag_estimable": diag_estimable_arr,
                    "arm_logits": arm_logits_arr,
                    "arm_estimable": arm_estimable_arr,
                },
                upstream=[plain(plan.reference), maps_ref],
                run_id=effective_run_id,
            )
        )
    _emit_progress(
        progress,
        "e12_predictions_ready",
        unit="score_predictions",
        completed=2,
        total=4,
        model=model,
        predictions_ref=predictions_ref,
    )

    # Step 3: Bootstrap evaluation (full-space reference + r=64 arms + MSE + contrasts).
    identity_align = np.arange(n_scr, dtype=np.intp)
    full_space_ref = _evaluate_full_space_reference(
        binding, inv, data["screen_ids"], scr_y, scr_known, counts, categories, primary_indices, screen_minimum, total_draws
    )

    pca_diag_summaries = {}
    pca_diag_draws = {}
    for s_idx, s in enumerate(LANGUAGES):
        ev = _evaluate_logits(
            diag_logits_arr[s_idx],
            diag_estimable_arr[s_idx],
            identity_align,
            scr_y,
            scr_known,
            counts,
            categories,
            total_draws,
        )
        pca_diag_summaries[s], pca_diag_draws[s] = _summarize_eval(
            ev, categories, primary_indices, screen_minimum
        )

    arm_route_summaries = {arm: {} for arm in E12_ARMS}
    arm_route_draws = {arm: {} for arm in E12_ARMS}
    for r_idx, (s, t) in enumerate(_off_diagonal_pairs()):
        rk = _route_key(s, t)
        for a_idx, arm in enumerate(E12_ARMS):
            ev = _evaluate_logits(
                arm_logits_arr[a_idx, r_idx],
                arm_estimable_arr[a_idx, r_idx],
                identity_align,
                scr_y,
                scr_known,
                counts,
                categories,
                total_draws,
            )
            arm_route_summaries[arm][rk], arm_route_draws[arm][rk] = _summarize_eval(
                ev, categories, primary_indices, screen_minimum
            )

    # r=64 G_res per route & 12-route mean
    g_res_route_summaries = {m: {} for m in E12_METRICS}
    g_res_route_draws = {m: {} for m in E12_METRICS}
    for s, t in _off_diagonal_pairs():
        rk = _route_key(s, t)
        for m in E12_METRICS:
            d_macro = pca_diag_summaries[t][m]["primary_macro"]
            id_macro = arm_route_summaries["identity_centered"][rk][m]["primary_macro"]
            g_dr = _gap_draws(m, pca_diag_draws[t][m], arm_route_draws["identity_centered"][rk][m])
            g_res_route_draws[m][rk] = g_dr
            g_res_route_summaries[m][rk] = {
                "value": _gap_value(m, d_macro["value"], id_macro["value"]),
                "interval": _interval_record(g_dr),
                "status": "ok" if (d_macro["status"] == "ok" and id_macro["status"] == "ok") else "descriptive_low_support",
            }

    g_res_mean = {}
    g_res_mean_draws = {}
    for m in E12_METRICS:
        g_res_mean[m], g_res_mean_draws[m] = _mean_across_routes(
            [g_res_route_summaries[m][_route_key(s, t)] for s, t in _off_diagonal_pairs()],
            [g_res_route_draws[m][_route_key(s, t)] for s, t in _off_diagonal_pairs()],
            route_count=12,
        )

    # Arm summaries (12-route mean performance, gap_reduction, recovered_fraction_of_g_res)
    arms_summary = {}
    arm_mean_draws = {arm: {} for arm in E12_ARMS}
    for arm in E12_ARMS:
        mean_perf = {}
        gap_red_mean = {}
        rec_frac_mean = {}
        for m in E12_METRICS:
            mean_perf[m], arm_mean_draws[arm][m] = _mean_across_routes(
                [arm_route_summaries[arm][_route_key(s, t)][m]["primary_macro"] for s, t in _off_diagonal_pairs()],
                [arm_route_draws[arm][_route_key(s, t)][m] for s, t in _off_diagonal_pairs()],
                route_count=12,
            )
            if arm == "identity_centered":
                continue
            route_gr = []
            route_gr_draws = []
            for s, t in _off_diagonal_pairs():
                rk = _route_key(s, t)
                base_m = arm_route_summaries["identity_centered"][rk][m]["primary_macro"]
                alt_m = arm_route_summaries[arm][rk][m]["primary_macro"]
                gr_dr = _improvement_draws(
                    m, arm_route_draws["identity_centered"][rk][m], arm_route_draws[arm][rk][m]
                )
                route_gr_draws.append(gr_dr)
                route_gr.append({
                    "value": _improvement_value(m, base_m["value"], alt_m["value"]),
                    "interval": _interval_record(gr_dr),
                    "status": "ok" if (base_m["status"] == "ok" and alt_m["status"] == "ok") else "descriptive_low_support",
                })
            gap_red_mean[m], gr_mean_dr = _mean_across_routes(route_gr, route_gr_draws, route_count=12)

            g_pt = g_res_mean[m]["value"]
            gr_pt = gap_red_mean[m]["value"]
            frac_pt = None if (g_pt is None or gr_pt is None or abs(g_pt) <= 1e-12) else float(gr_pt / g_pt)
            frac_dr = np.where(np.abs(g_res_mean_draws[m]) > 1e-12, gr_mean_dr / g_res_mean_draws[m], np.nan)
            rec_frac_mean[m] = {
                "value": frac_pt,
                "interval": _interval_record(frac_dr),
                "status": gap_red_mean[m]["status"],
            }

        arms_summary[arm] = {
            "supervision_label": SUPERVISION_IDENTITY if arm == "identity_centered" else SUPERVISION_ADAPTED,
            "mean_12_routes": mean_perf,
            "by_route": {
                _route_key(s, t): {
                    m: arm_route_summaries[arm][_route_key(s, t)][m]["primary_macro"] for m in E12_METRICS
                }
                for s, t in _off_diagonal_pairs()
            },
        }
        if arm != "identity_centered":
            arms_summary[arm]["gap_reduction"] = gap_red_mean
            arms_summary[arm]["recovered_fraction_of_g_res"] = rec_frac_mean

    # Constraint contrast: ridge_linear - orthogonal_procrustes
    contrasts_summary = {"ridge_minus_procrustes": {}}
    for m in E12_METRICS:
        r_pt = arms_summary["ridge_linear"]["mean_12_routes"][m]["value"]
        p_pt = arms_summary["orthogonal_procrustes"]["mean_12_routes"][m]["value"]
        c_pt = None if (r_pt is None or p_pt is None) else float(r_pt - p_pt)
        c_dr = arm_mean_draws["ridge_linear"][m] - arm_mean_draws["orthogonal_procrustes"][m]
        contrasts_summary["ridge_minus_procrustes"][m] = {
            "value": c_pt,
            "interval": _interval_record(c_dr),
            "status": (
                "ok"
                if (
                    arms_summary["ridge_linear"]["mean_12_routes"][m]["status"] == "ok"
                    and arms_summary["orthogonal_procrustes"]["mean_12_routes"][m]["status"] == "ok"
                )
                else "descriptive_low_support"
            ),
        }

    # Paired SCREEN feature reconstruction MSE (with bootstrap draws via counts)
    count_sums = counts.sum(axis=1).astype(np.float64)
    mse_by_arm = {}
    mse_draws_by_arm = {}
    for arm in E12_ARMS:
        per_route_mse = {}
        per_route_draws = []
        for s, t in _off_diagonal_pairs():
            rk = _route_key(s, t)
            vec = screen_mse_per_image[(s, t)][arm]
            pt = float(np.mean(vec))
            dr = (counts @ vec) / count_sums
            per_route_draws.append(dr)
            per_route_mse[rk] = {"value": pt, "interval": _interval_record(dr)}
        mean_dr = np.mean(np.stack(per_route_draws, axis=0), axis=0)
        mse_draws_by_arm[arm] = mean_dr
        mse_by_arm[arm] = {
            "mean_12_routes": {
                "value": float(np.mean([v["value"] for v in per_route_mse.values()])),
                "interval": _interval_record(mean_dr),
            },
            "by_route": per_route_mse,
        }

    mse_ridge_minus_procrustes_dr = mse_draws_by_arm["ridge_linear"] - mse_draws_by_arm["orthogonal_procrustes"]
    mse_summary = {
        "arms": mse_by_arm,
        "ridge_minus_procrustes": {
            "value": float(
                mse_by_arm["ridge_linear"]["mean_12_routes"]["value"]
                - mse_by_arm["orthogonal_procrustes"]["mean_12_routes"]["value"]
            ),
            "interval": _interval_record(mse_ridge_minus_procrustes_dr),
        },
    }

    decision_summary = _evaluate_decision_summary(arms_summary, contrasts_summary, decision_ranges)
    compute_s = time.monotonic() - t_comp

    t_write = time.monotonic()
    analysis_body = {
        "schema": SCHEMA_ANALYSIS,
        "experiment": "E12",
        "plan": plain(plan.reference),
        "plan_identity": plan["plan_identity"],
        "model": model,
        "midpoint_depth": int(inv["midpoint_depth"]),
        "condition": E12_CONDITION,
        "pooling": E12_POOLING,
        "inputs": expected_inputs,
        "maps_ref": maps_ref,
        "predictions_ref": predictions_ref,
        "bootstrap": plain(plan["bootstrap"]),
        "sanity_checks": {
            "pairing_check": data["pairing_check"],
            "zero_target_semantic_labels_in_maps_and_ridge_selection": True,
            "pca_orthonormality_passed": True,
            "procrustes_orthogonality_passed": True,
            "full_space_reference_reported_separately": True,
            "confirm_excluded": True,
        },
        "full_space_reference": full_space_ref,
        "pca_source_summary": {
            "diagonal_mean_12_lang": {
                m: _mean_across_routes(
                    [pca_diag_summaries[s][m]["primary_macro"] for s in LANGUAGES],
                    [pca_diag_draws[s][m] for s in LANGUAGES],
                    route_count=len(LANGUAGES),
                )[0]
                for m in E12_METRICS
            },
            "by_source": {
                s: {
                    "rank": source_pca[s]["rank"],
                    "source_rank": source_pca[s]["source_rank"],
                    "selected_C": source_pca[s]["head"]["selected_C"],
                    "dev_s_selection_score": source_pca[s]["head"]["dev_s_selection_score"],
                    "screen_diagonal": {
                        m: pca_diag_summaries[s][m]["primary_macro"] for m in E12_METRICS
                    },
                }
                for s in LANGUAGES
            },
            "r64_g_res_mean": g_res_mean,
        },
        "arms": arms_summary,
        "contrasts": contrasts_summary,
        "mse_summary": mse_summary,
        "decision_summary": decision_summary,
        "supervision_labels": {
            "diagonal": SUPERVISION_DIAGONAL,
            "identity_centered": SUPERVISION_IDENTITY,
            "orthogonal_procrustes": SUPERVISION_ADAPTED,
            "ridge_linear": SUPERVISION_ADAPTED,
        },
        "confirm_excluded": True,
    }
    analysis_ref = plain(
        publish_phase3(
            resolved_root,
            protocol,
            plan["dataset_fingerprint"],
            f"{E12_RESULTS_NAMESPACE}/{model}",
            KIND_ANALYSIS,
            analysis_body,
            producer,
            upstream=[plain(plan.reference), maps_ref, predictions_ref],
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
        expected_inputs=expected_inputs,
    )
    write_s = time.monotonic() - t_write

    _emit_progress(
        progress,
        "e12_published",
        unit="publish",
        completed=4,
        total=4,
        read_hash_s=read_s,
        compute_s=compute_s,
        write_s=write_s,
        elapsed_s=round(time.monotonic() - t0, 3),
        model=model,
        maps_ref=maps_ref,
        predictions_ref=predictions_ref,
        analysis_ref=analysis_ref,
        outcome=decision_summary["outcome"],
    )
    return frozen({
        "model": model,
        "run_id": effective_run_id,
        "maps_ref": maps_ref,
        "predictions_ref": predictions_ref,
        "analysis_ref": analysis_ref,
        "decision_summary": decision_summary,
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


def _emit_cli(event):
    print(json.dumps(event, sort_keys=True), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Phase 3 Stage A2 E12 CPU worker.")
    parser.add_argument("--phase1-root", required=True)
    parser.add_argument("--protocol", type=Path, default=REPO / "configs/protocol.json")
    parser.add_argument("--plan", required=True, help="Explicit phase3_cpu_plan {'path','sha256'} JSON or path")
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument(
        "--storage-scope",
        default="persistent-drive",
        choices=("persistent-drive", "local-development"),
    )
    parser.add_argument("--resume-maps", default=None)
    parser.add_argument("--resume-predictions", default=None)
    parser.add_argument("--resume-analysis", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-ref", type=Path, default=None)
    args = parser.parse_args(argv)

    root = phase1_root(args.phase1_root)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    producer = current_producer(args.storage_scope)
    _emit_cli({
        "event": "e12_start",
        "model": args.model,
        "code_commit": producer["code_commit"],
        "storage_scope": producer["storage_scope"],
    })
    result = plain(
        run_e12(
            root,
            protocol,
            _parse_ref_arg(args.plan),
            args.model,
            producer,
            resume_maps_ref=_parse_ref_arg(args.resume_maps),
            resume_predictions_ref=_parse_ref_arg(args.resume_predictions),
            resume_analysis_ref=_parse_ref_arg(args.resume_analysis),
            run_id=args.run_id,
            progress=_emit_cli,
        )
    )
    summary = {"event": "e12_complete", "status": "passed", **result}
    if args.output_ref is not None:
        args.output_ref.parent.mkdir(parents=True, exist_ok=True)
        args.output_ref.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _emit_cli(summary)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, ImportError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        _emit_cli({"status": "blocked", "blockers": [str(exc)], "phase3_e12_complete": False})
        raise SystemExit(2)
