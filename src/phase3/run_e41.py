"""Phase 3 Stage A2 CPU worker for E41: 7-arm supervision ladder + PCA-CORAL whitening.

Implements phase3.md §A1.2.2, §A1.3, §A2.1, and §A2.2 Worker 2:
1. Binds `phase3_cpu_plan` via `load_phase3_cpu_plan(root, protocol, plan_ref, bind=True)`.
   Never touches `CONFIRM` (`require_open_split`), never mutates Phase 1 or Phase 2 dirs,
   and publishes only under `PHASE1_ROOT/phase3/` via `publish_phase3`.
2. Covers the 4 `(depth, pooling)` cells: `depth in {midpoint, final}` (`qwen`: `[18, 36]`,
   `gemma`: `[17, 34]`) x `pooling in {last_caption, caption_mean}` on `image_caption`,
   evaluating the 7-arm supervision ladder:
     `raw`, `source_dev_c`, `E11_diag`, `E11_cov`, `E11_diag+source_dev_c`,
     `target_dev_c`, `E11_diag+target_dev_c`.
3. Reuses Phase 2 E17 midpoint `last_caption` DEV-C blocks/calibrators and Phase 2 E11
   midpoint `caption_mean` diagonal adapted blocks.
4. For newly covered `(depth, pooling)` cells, fits `temp` and `temp_intercept` BFGS
   calibrators (exact analytic gradient, `gtol=1e-10`, `maxiter=500`, with L-BFGS-B polish
   when SciPy BFGS hits `status == 2` precision loss), verifying `T > 0` and `<= 1e-12`
   AUROC/AP invariance.
5. Implements `E11_diag` (`mean_shift` + `mean_scale` from Stage 3 FIT `statistics`,
   verifying `mean_shift` `<= 1e-12` AUROC/AP invariance) and `E11_cov` (regularized
   full-covariance CORAL whitening in source-FIT PCA subspace `r in {64, 256}` using
   unlabeled `FIT` features only — zero target labels — projecting `H_t` into source-FIT
   PCA, whitening by shrunk `Sigma_t^{-1/2}`, re-coloring by `Sigma_s^{1/2}`,
   reconstructing back to raw space, and passing through unchanged source scaler + head).
6. Evaluates all 7 arms on reused Stage 4 `SCREEN` logits/features vs `raw`, vs `FIT-prior`,
   and vs `target_dev_c` inside every `(2000, 360)` bootstrap draw before taking linear
   2.5th/97.5th percentiles.
7. Publishes under `phase3/predictions/e41_cov/<model>/<run_id>/`,
   `phase3/calibrators/e41/<model>/<run_id>/`, and `phase3/results/e41/<model>/<run_id>/`,
   with explicit-ref resume via `verify_resume_ref`.
"""

import argparse
import json
from pathlib import Path
import sys
import time

from src.common.contracts import ContractError, LANGUAGES, fingerprint
from src.common.storage import new_run_id, phase1_root, within
from src.phase2.interfaces import (
    KIND_CALIBRATORS as P2_KIND_CALIBRATORS,
    KIND_PREDICTION as P2_KIND_PREDICTION,
    SCHEMA_CALIBRATORS as P2_SCHEMA_CALIBRATORS,
    SCHEMA_PREDICTION as P2_SCHEMA_PREDICTION,
    load_phase2_artifact,
)
from src.phase3.interfaces import (
    KIND_ANALYSIS,
    KIND_CALIBRATORS,
    KIND_PREDICTION,
    MODELS,
    POOLINGS,
    REPO,
    SCHEMA_ANALYSIS,
    SCHEMA_CALIBRATORS,
    SCHEMA_PREDICTION,
    check,
    contract,
    current_producer,
    frozen,
    load_phase3_cpu_plan,
    plain,
    publish_phase3,
    require_open_split,
    verify_resume_ref,
)

E41_CONDITION = "image_caption"
E41_FIT_SPLIT = "FIT"
E41_DEV_C_SPLIT = "DEV-C"
E41_SCREEN_SPLIT = "SCREEN"

E41_PREDICTIONS_NAMESPACE = "phase3/predictions/e41_cov"
E41_CALIBRATORS_NAMESPACE = "phase3/calibrators/e41"
E41_RESULTS_NAMESPACE = "phase3/results/e41"

E41_ARMS = (
    "raw",
    "source_dev_c",
    "E11_diag",
    "E11_cov",
    "E11_diag+source_dev_c",
    "target_dev_c",
    "E11_diag+target_dev_c",
)
E41_COV_RANKS = (64, 256)
E41_PRIMARY_COV_RANK = 256
E41_DIAG_VARIANTS = ("mean_shift", "mean_scale")
E41_CALIBRATOR_FORMS = ("temp", "temp_intercept")

BFGS_GTOL = 1e-10
BFGS_MAXITER = 500
INVARIANCE_TOL = 1e-12
EIG_FLOOR = 1e-12

SUPERVISION_LABELS = frozen({
    "within_context_source_fit": "within_context_source_fit",
    "source_only_transfer": "source_only_transfer",
    "source_dev_c_calibrated": "source_dev_c_calibrated",
    "target_fit_input_adapted": "target_fit_input_adapted",
    "target_fit_input_adapted_plus_source_dev_c": "target_fit_input_adapted_plus_source_dev_c",
    "target_dev_c_supervised_calibration": "target_dev_c_supervised_calibration",
})


def _peak_rss_bytes():
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(usage * (1024 if sys.platform != "darwin" else 1))
    except Exception:
        return None


def _emit(progress, event):
    if progress is not None:
        progress(event)


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


def _primary_indices(protocol):
    categories = list(protocol["category_order"])
    primary = list(protocol["primary_categories"])
    check(len(primary) == 14, f"Expected 14 fixed primary categories; got {len(primary)}")
    return [categories.index(cat) for cat in primary]


def _read_verified_group_features(binding, gid, split):
    """Read verified Stage 2 features for a group_id and open split."""
    from src.stage3.data import _read_verified_group

    require_open_split(split)
    return _read_verified_group(binding.stage4_plan.fit, binding.stage4_plan.store, gid, split)


def _load_group_head(binding, gid):
    """Load frozen Stage 3 ordinary (coef, intercept, estimable, prevalence, stats)."""
    from src.stage4.predictions import _selected_parameters

    return _selected_parameters(binding.stage4_plan, gid)


def _score_through_source_head(features_f64, coef, intercept, estimable, stats_source):
    """Score float64 features through unchanged Stage 3 source scaler + frozen head."""
    import numpy as np
    from src.stage3.engine import transform

    standardized = transform(np.asarray(features_f64, dtype=np.float64), stats_source)
    logits = np.asarray(standardized @ np.asarray(coef, dtype=np.float64).T, dtype=np.float64) + np.asarray(
        intercept, dtype=np.float64
    )
    check(np.isfinite(logits).all(), "Nonfinite scored logits")
    est = np.asarray(estimable, dtype=bool)
    check(not np.any(logits * (~est[None, :])), "Unestimable logit padding mismatch")
    return logits


def _extract_targets_for_rows(stage4_plan, rows, expected_split):
    """Build aligned (y, known) float64/bool arrays for rows in an open split."""
    import numpy as np

    require_open_split(expected_split)
    values = []
    for row in rows:
        target = stage4_plan.fit["targets"][row["image_id"]]
        check(
            target["split"] == row["split"] == expected_split,
            f"Cross-split target join: expected {expected_split!r}, got {target['split']!r}/{row['split']!r}",
        )
        values.append(target["values"])
    known = np.array([[v is not None for v in row] for row in values], dtype=bool)
    y = np.array([[0 if v is None else int(v) for v in row] for row in values], dtype=np.float64)
    return y, known


def fit_macro_bce_calibrator(logits, y, known, primary_indices, form, *, estimable=None):
    """Fit global float64 `temp` or `temp_intercept` calibrator via deterministic BFGS.

    Matches src/phase2/run_e17.py: exact analytic gradient, gtol=1e-10, maxiter=500,
    zero initialization, with L-BFGS-B polish when SciPy BFGS reports status == 2.
    """
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import expit

    check(form in E41_CALIBRATOR_FORMS, f"Unsupported calibrator form: {form!r}")
    z = np.asarray(logits, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    known_arr = np.asarray(known, dtype=bool)
    check(z.ndim == 2 and z.shape == y_arr.shape == known_arr.shape, "Calibrator input shape mismatch")
    check(np.isfinite(z).all(), "Nonfinite logits passed to calibrator")
    check(len(primary_indices) == 14, f"Expected 14 primary category indices; got {len(primary_indices)}")

    if estimable is not None:
        est = np.asarray(estimable, dtype=bool)
        for k in primary_indices:
            check(bool(est[k]), f"Primary category index {k} is unestimable")

    cat_slices = []
    for k in primary_indices:
        mask = known_arr[:, k]
        n_k = int(mask.sum())
        check(n_k > 0, f"Primary category index {k} has zero known DEV-C observations")
        cat_slices.append((z[mask, k], y_arr[mask, k], float(n_k)))

    n_primary = float(len(cat_slices))

    def _objective_and_grad(params):
        log_t = float(params[0])
        if log_t > 50.0 or log_t < -50.0:
            return np.inf, np.full_like(params, np.nan)
        inv_t = float(np.exp(-log_t))
        b = float(params[1]) if form == "temp_intercept" else 0.0

        total_loss = 0.0
        grad_log_t = 0.0
        grad_b = 0.0
        for z_k, y_k, n_k in cat_slices:
            z_prime = inv_t * z_k + b
            loss_k = float(np.mean(np.logaddexp(0.0, z_prime) - y_k * z_prime))
            p_k = expit(z_prime)
            diff_k = p_k - y_k
            total_loss += loss_k
            grad_log_t += float(np.dot(diff_k, -inv_t * z_k)) / n_k
            if form == "temp_intercept":
                grad_b += float(np.sum(diff_k)) / n_k

        loss = total_loss / n_primary
        if form == "temp":
            grad = np.array([grad_log_t / n_primary], dtype=np.float64)
        else:
            grad = np.array([grad_log_t / n_primary, grad_b / n_primary], dtype=np.float64)
        return float(loss), grad

    x0 = np.zeros(1 if form == "temp" else 2, dtype=np.float64)
    raw_loss, _ = _objective_and_grad(x0)
    res = minimize(
        _objective_and_grad,
        x0,
        method="BFGS",
        jac=True,
        options={"gtol": BFGS_GTOL, "maxiter": BFGS_MAXITER},
    )
    total_nit = int(res.nit)
    if (not res.success) and getattr(res, "status", None) == 2 and np.isfinite(res.x).all() and np.isfinite(res.fun):
        polish = minimize(
            _objective_and_grad,
            np.asarray(res.x, dtype=np.float64),
            method="L-BFGS-B",
            jac=True,
            options={"gtol": BFGS_GTOL, "ftol": 1e-15, "maxiter": BFGS_MAXITER},
        )
        total_nit += int(polish.nit)
        if np.isfinite(polish.x).all() and np.isfinite(polish.fun) and float(polish.fun) <= float(res.fun):
            res = polish

    grad_norm_check = float(np.max(np.abs(res.jac))) if np.isfinite(res.jac).all() else float("inf")
    converged = bool(res.success) or (
        getattr(res, "status", None) == 2
        and np.isfinite(res.x).all()
        and np.isfinite(res.fun)
        and grad_norm_check <= 1e-6
        and float(res.fun) <= float(raw_loss) + 1e-12
    )
    check(converged, f"BFGS calibrator ({form}) failed to converge: {res.message}")
    check(np.isfinite(res.x).all() and np.isfinite(res.fun), f"Nonfinite BFGS solution for {form}")

    log_t = float(res.x[0])
    t_val = float(np.exp(log_t))
    b_val = float(res.x[1]) if form == "temp_intercept" else 0.0
    check(np.isfinite(t_val) and t_val > 0.0, f"Invalid fitted temperature T={t_val!r}")
    check(np.isfinite(b_val), f"Invalid fitted intercept b={b_val!r}")
    grad_norm = float(np.max(np.abs(res.jac)))
    check(np.isfinite(grad_norm), "Nonfinite BFGS gradient norm")

    return {
        "log_T": log_t,
        "T": t_val,
        "b": b_val,
        "converged": bool(converged),
        "nit": total_nit,
        "grad_norm": grad_norm,
        "dev_c_loss_raw": float(raw_loss),
        "dev_c_loss_cal": float(res.fun),
    }


def apply_calibrator(logits, cal_params):
    """Apply float64 z' = z / T + b."""
    import numpy as np

    z = np.asarray(logits, dtype=np.float64)
    t_val = float(cal_params["T"])
    b_val = float(cal_params["b"])
    check(np.isfinite(t_val) and t_val > 0.0, f"Invalid temperature T={t_val!r}")
    check(np.isfinite(b_val), f"Invalid intercept b={b_val!r}")
    return z / t_val + b_val


def apply_e11_diag_transform(matrix_f64, stats_source, stats_target, variant):
    """Apply FIT-only mean_shift or mean_scale transform in float64."""
    import numpy as np

    x = np.asarray(matrix_f64, dtype=np.float64)
    mu_s = np.asarray(stats_source["mean"], dtype=np.float64)
    sigma_s = np.asarray(stats_source["scale"], dtype=np.float64)
    mu_t = np.asarray(stats_target["mean"], dtype=np.float64)
    sigma_t = np.asarray(stats_target["scale"], dtype=np.float64)
    if variant == "mean_shift":
        adapted = x - mu_t + mu_s
    elif variant == "mean_scale":
        adapted = mu_s + sigma_s * ((x - mu_t) / sigma_t)
    else:
        raise ContractError(f"Unknown E11_diag variant: {variant!r}")
    check(np.isfinite(adapted).all(), f"Nonfinite adapted features for variant {variant!r}")
    return adapted


def _ledoit_wolf_cov(centered_z):
    """Closed-form Ledoit-Wolf shrinkage covariance on centered (N, r) unlabeled features."""
    import numpy as np

    z = np.asarray(centered_z, dtype=np.float64)
    n, r = z.shape
    check(n >= 2 and r >= 1, f"Invalid shape for Ledoit-Wolf covariance: {z.shape}")
    sample_cov = (z.T @ z) / float(n)
    sample_cov = 0.5 * (sample_cov + sample_cov.T)
    mu = float(np.trace(sample_cov)) / float(r)
    target = mu * np.eye(r, dtype=np.float64)
    delta_sq = float(np.sum((sample_cov - target) ** 2)) / float(r)
    if delta_sq <= 0.0:
        return sample_cov, 0.0
    row_norm_sq = np.sum(z * z, axis=1)
    beta_num = float(np.sum(row_norm_sq ** 2)) - float(n) * float(np.sum(sample_cov ** 2))
    beta_sq = max(0.0, beta_num / (float(n) ** 2 * float(r)))
    shrinkage = min(1.0, max(0.0, beta_sq / delta_sq))
    shrunk = (1.0 - shrinkage) * sample_cov + shrinkage * target
    shrunk = 0.5 * (shrunk + shrunk.T)
    return shrunk, float(shrinkage)


def fit_pca_coral_operator(fit_source_f64, fit_target_f64, rank, *, eig_floor=EIG_FLOOR, basis_v_max=None):
    """Fit unlabeled PCA-CORAL whitening+recoloring operator from source/target FIT features.

    Uses zero target labels. Returns (mu_s, mu_t, v_r, coral_matrix_r, metadata).
    """
    import numpy as np

    xs = np.asarray(fit_source_f64, dtype=np.float64)
    xt = np.asarray(fit_target_f64, dtype=np.float64)
    check(xs.ndim == 2 and xt.ndim == 2 and xs.shape[1] == xt.shape[1], "FIT feature dimension mismatch")
    n_s, d = xs.shape
    n_t, _ = xt.shape
    check(n_s >= 2 and n_t >= 2, "Need at least 2 FIT samples for covariance")

    mu_s = np.mean(xs, axis=0)
    mu_t = np.mean(xt, axis=0)
    xs_c = xs - mu_s
    xt_c = xt - mu_t

    eff_rank = min(int(rank), d, n_s - 1, n_t - 1)
    check(eff_rank >= 1, f"Invalid effective PCA rank {eff_rank}")

    if basis_v_max is None:
        _, _, vt = np.linalg.svd(xs_c, full_matrices=False)
        v_r = vt[:eff_rank, :].T
    else:
        check(basis_v_max.shape[0] == d and basis_v_max.shape[1] >= eff_rank, "Invalid precomputed PCA basis")
        v_r = basis_v_max[:, :eff_rank]

    zs = xs_c @ v_r
    zt = xt_c @ v_r

    cov_s, shrink_s = _ledoit_wolf_cov(zs)
    cov_t, shrink_t = _ledoit_wolf_cov(zt)

    evals_s, evecs_s = np.linalg.eigh(cov_s)
    evals_t, evecs_t = np.linalg.eigh(cov_t)
    evals_s_clamped = np.maximum(evals_s, eig_floor)
    evals_t_clamped = np.maximum(evals_t, eig_floor)

    sigma_s_half = (evecs_s * np.sqrt(evals_s_clamped)[None, :]) @ evecs_s.T
    sigma_t_inv_half = (evecs_t * (1.0 / np.sqrt(evals_t_clamped))[None, :]) @ evecs_t.T

    # For row vectors z_t: z_adapted = z_t @ Sigma_t^{-1/2} @ Sigma_s^{1/2}.
    coral_r = sigma_t_inv_half @ sigma_s_half
    check(np.isfinite(coral_r).all(), "Nonfinite CORAL operator")

    meta = {
        "requested_rank": int(rank),
        "effective_rank": int(eff_rank),
        "shrinkage_source": float(shrink_s),
        "shrinkage_target": float(shrink_t),
        "min_eig_source": float(np.min(evals_s)),
        "min_eig_target": float(np.min(evals_t)),
        "eig_floor": float(eig_floor),
        "zero_target_labels": True,
    }
    return mu_s, mu_t, v_r, coral_r, meta


def apply_pca_coral_operator(target_features_f64, mu_s, mu_t, v_r, coral_r):
    """Project H_t into source-FIT PCA, whiten+recolor, and reconstruct back to raw space."""
    import numpy as np

    ht = np.asarray(target_features_f64, dtype=np.float64)
    zt = (ht - mu_t) @ v_r
    zt_adapted = zt @ coral_r
    reconstructed = mu_s + zt_adapted @ v_r.T
    check(np.isfinite(reconstructed).all(), "Nonfinite PCA-CORAL reconstructed features")
    return reconstructed


def _load_block_arrays(root, ref, expected_sources, expected_n_cats):
    import numpy as np

    marker_dir = within(root, ref["path"]).parent
    logits = np.load(marker_dir / "logits.npy", allow_pickle=False)
    estimable = np.load(marker_dir / "estimable.npy", allow_pickle=False)
    prevalence = np.load(marker_dir / "prevalence.npy", allow_pickle=False)
    n_src = len(expected_sources)
    check(
        logits.ndim == 3 and logits.shape[0] == n_src and logits.shape[2] == expected_n_cats,
        f"Invalid prediction block logits shape: {logits.shape}",
    )
    check(logits.dtype == np.float64 and np.isfinite(logits).all(), "Prediction block logits must be finite float64")
    check(estimable.shape == (n_src, expected_n_cats) and estimable.dtype == bool, "Invalid estimable shape")
    check(prevalence.shape == (n_src, expected_n_cats) and prevalence.dtype == np.float64, "Invalid prevalence shape")
    return logits, estimable, prevalence


def _summarize_draws(point_val, draw_vals):
    import numpy as np
    from src.stage4.metrics import bootstrap_interval

    arr = np.asarray(draw_vals, dtype=np.float64)
    ci = bootstrap_interval(arr)
    pt = None if (point_val is None or not np.isfinite(point_val)) else float(point_val)
    return {
        "point": pt,
        "lower": ci["lower"],
        "upper": ci["upper"],
        "valid_draws": ci["valid_draws"],
        "total_draws": ci["total_draws"],
    }


def _run_en_screen_self_check(plan, cell_inv, depth, pooling):
    """In-memory en->en SCREEN diagonal reproduction check (atol=0, rtol=0) against Stage 4."""
    import numpy as np
    from src.stage4.predictions import load_prediction_block

    gid_en = cell_inv["group_ids"]["en"]
    rows_en, x_en = _read_verified_group_features(plan.binding, gid_en, E41_SCREEN_SPLIT)
    coef, intercept, estimable, _, stats = _load_group_head(plan.binding, gid_en)
    computed = _score_through_source_head(x_en, coef, intercept, estimable, stats)

    blk = cell_inv["screen_blocks"]["en"]
    accepted = load_prediction_block(plan.binding.stage4_plan, blk["prediction_ref"], expected_block=blk["block_id"])[
        blk["routes"]["en"]
    ]
    check(plain(rows_en) == plain(accepted.rows), f"SCREEN en->en row alignment mismatch at {depth}:{pooling}")
    diff = float(np.max(np.abs(computed - accepted.logits)))
    check(diff == 0.0, f"SCREEN en->en diagonal mismatch at {depth}:{pooling}: {diff}")
    return {"route": "en->en", "depth": int(depth), "pooling": pooling, "max_abs_diff": diff, "passed": True}


def _evaluate_cell(
    *,
    plan,
    model,
    depth,
    depth_role,
    pooling,
    cell_inv,
    cal_cell,
    adapted_screen_by_variant_route,
    cov_screen_by_rank_route,
    y_screen,
    known_screen,
    screen_ids,
    counts,
    n_draws,
    categories,
    primary_idx,
    screen_min_support,
):
    """Evaluate all 7 canonical arms + auxiliary variants for one (depth, pooling) cell."""
    import numpy as np
    from src.stage4.metrics import (
        METRICS,
        binary_metrics,
        bootstrap_binary_metrics,
        fixed_macro,
        paired_image_alignment,
        probabilities,
        support_status,
    )
    from src.stage4.predictions import load_prediction_block

    stage4_plan = plan.binding.stage4_plan
    n_cats = len(categories)
    n_screen = len(screen_ids)

    # Load Stage 4 SCREEN predictions for all 16 routes and align to canonical screen_ids.
    screen_preds = {}
    all_pred_objs = []
    ordered_keys = []
    for t_lang in LANGUAGES:
        blk = cell_inv["screen_blocks"][t_lang]
        blk_preds = load_prediction_block(stage4_plan, blk["prediction_ref"], expected_block=blk["block_id"])
        for s_lang in LANGUAGES:
            pred = blk_preds[blk["routes"][s_lang]]
            screen_preds[(s_lang, t_lang)] = pred
            all_pred_objs.append(pred)
            ordered_keys.append((s_lang, t_lang))

    _, aligned_indices = paired_image_alignment(stage4_plan, *all_pred_objs)
    aligned_map = dict(zip(ordered_keys, aligned_indices))

    # Build canonical row-alignment map for adapted blocks (which follow target feature release order).
    # Verify target row IDs against screen_ids via Stage 4 diagonal prediction rows.
    diag_id_to_pos = {}
    for t_lang in LANGUAGES:
        diag_pred = screen_preds[(t_lang, t_lang)]
        row_pos = {r["image_id"]: i for i, r in enumerate(diag_pred.rows)}
        diag_id_to_pos[t_lang] = np.array([row_pos[img_id] for img_id in screen_ids], dtype=np.intp)

    # Define canonical 7 arms + auxiliary arms to evaluate identically.
    aux_arms = (
        "E11_cov_r64",
        "E11_cov_r256",
        "E11_diag:mean_shift",
        "source_dev_c:temp",
        "target_dev_c:temp",
        "E11_diag+source_dev_c:temp",
        "E11_diag+target_dev_c:temp",
    )
    all_eval_arms = list(E41_ARMS) + list(aux_arms)

    route_metrics = {arm: {} for arm in all_eval_arms}
    route_fit_prior = {}
    route_flips = {arm: {} for arm in all_eval_arms}
    route_supervision = {arm: {} for arm in all_eval_arms}

    max_cal_auroc_diff = 0.0
    max_cal_ap_diff = 0.0
    max_shift_auroc_diff = 0.0
    max_shift_ap_diff = 0.0
    temp_only_flips = 0

    def _score_metric_arrays(probs_mat):
        pts = {m: np.full(n_cats, np.nan, dtype=np.float64) for m in METRICS}
        drs = {m: np.full((n_draws, n_cats), np.nan, dtype=np.float64) for m in METRICS}
        for k in range(n_cats):
            pt = binary_metrics(y_screen[:, k], probs_mat[:, k], known_screen[:, k])["metrics"]
            dr = bootstrap_binary_metrics(y_screen[:, k], probs_mat[:, k], known_screen[:, k], counts)
            for m in METRICS:
                if pt[m] is not None:
                    pts[m][k] = pt[m]
                drs[m][:, k] = dr[m]
        return pts, drs

    def _arm_logits_and_label(arm, s_lang, t_lang, raw_logits):
        is_diag = s_lang == t_lang
        route_key = f"{s_lang}->{t_lang}"
        if arm == "raw":
            lbl = SUPERVISION_LABELS["within_context_source_fit"] if is_diag else SUPERVISION_LABELS["source_only_transfer"]
            return raw_logits, lbl
        if arm in ("source_dev_c", "source_dev_c:temp"):
            form = "temp" if arm.endswith(":temp") else "temp_intercept"
            lbl = SUPERVISION_LABELS["within_context_source_fit"] if is_diag else SUPERVISION_LABELS["source_dev_c_calibrated"]
            return apply_calibrator(raw_logits, cal_cell["source_dev_c"][s_lang][form]), lbl
        if arm == "target_dev_c" or arm == "target_dev_c:temp":
            form = "temp" if arm.endswith(":temp") else "temp_intercept"
            if is_diag:
                return apply_calibrator(raw_logits, cal_cell["source_dev_c"][s_lang][form]), SUPERVISION_LABELS[
                    "within_context_source_fit"
                ]
            return apply_calibrator(raw_logits, cal_cell["target_dev_c"][route_key][form]), SUPERVISION_LABELS[
                "target_dev_c_supervised_calibration"
            ]
        if arm in ("E11_diag", "E11_diag:mean_shift"):
            variant = "mean_shift" if arm.endswith(":mean_shift") else "mean_scale"
            if is_diag:
                return raw_logits, SUPERVISION_LABELS["within_context_source_fit"]
            z_ad = adapted_screen_by_variant_route[variant][(s_lang, t_lang)][diag_id_to_pos[t_lang]]
            return z_ad, SUPERVISION_LABELS["target_fit_input_adapted"]
        if arm in ("E11_cov", "E11_cov_r64", "E11_cov_r256"):
            rank = 64 if arm == "E11_cov_r64" else E41_PRIMARY_COV_RANK
            if is_diag:
                return raw_logits, SUPERVISION_LABELS["within_context_source_fit"]
            z_cov = cov_screen_by_rank_route[rank][(s_lang, t_lang)][diag_id_to_pos[t_lang]]
            return z_cov, SUPERVISION_LABELS["target_fit_input_adapted"]
        if arm in ("E11_diag+source_dev_c", "E11_diag+source_dev_c:temp"):
            form = "temp" if arm.endswith(":temp") else "temp_intercept"
            base = raw_logits if is_diag else adapted_screen_by_variant_route["mean_scale"][(s_lang, t_lang)][diag_id_to_pos[t_lang]]
            lbl = (
                SUPERVISION_LABELS["within_context_source_fit"]
                if is_diag
                else SUPERVISION_LABELS["target_fit_input_adapted_plus_source_dev_c"]
            )
            return apply_calibrator(base, cal_cell["source_dev_c"][s_lang][form]), lbl
        if arm in ("E11_diag+target_dev_c", "E11_diag+target_dev_c:temp"):
            form = "temp" if arm.endswith(":temp") else "temp_intercept"
            if is_diag:
                return apply_calibrator(raw_logits, cal_cell["source_dev_c"][s_lang][form]), SUPERVISION_LABELS[
                    "within_context_source_fit"
                ]
            base = adapted_screen_by_variant_route["mean_scale"][(s_lang, t_lang)][diag_id_to_pos[t_lang]]
            return apply_calibrator(base, cal_cell["target_dev_c_on_e11_diag"][route_key][form]), SUPERVISION_LABELS[
                "target_dev_c_supervised_calibration"
            ]
        raise ContractError(f"Unknown E41 arm: {arm!r}")

    known_total = int(np.sum(known_screen))

    for t_lang in LANGUAGES:
        for s_lang in LANGUAGES:
            pred = screen_preds[(s_lang, t_lang)]
            idx_arr = aligned_map[(s_lang, t_lang)]
            raw_logits = np.asarray(pred.logits[idx_arr], dtype=np.float64)
            raw_probs = probabilities(raw_logits)
            raw_dec = raw_probs >= 0.5

            prior_probs = np.broadcast_to(np.asarray(pred.prevalence, dtype=np.float64)[None, :], (n_screen, n_cats))
            route_fit_prior[(s_lang, t_lang)] = _score_metric_arrays(prior_probs)

            for arm in all_eval_arms:
                arm_logits, sup_lbl = _arm_logits_and_label(arm, s_lang, t_lang, raw_logits)
                route_supervision[arm][(s_lang, t_lang)] = sup_lbl
                arm_probs = probabilities(arm_logits)
                arm_dec = arm_probs >= 0.5
                flips = int(np.sum((arm_dec != raw_dec) & known_screen))
                if arm in ("source_dev_c:temp", "target_dev_c:temp"):
                    temp_only_flips += flips
                route_flips[arm][(s_lang, t_lang)] = {
                    "flips": flips,
                    "known_cells": known_total,
                    "flip_rate": float(flips / known_total) if known_total > 0 else 0.0,
                }
                route_metrics[arm][(s_lang, t_lang)] = _score_metric_arrays(arm_probs)

            # Enforce invariances on this route:
            # 1) Pure calibration arms vs raw must preserve AUROC/AP <= 1e-12.
            raw_pts, _ = route_metrics["raw"][(s_lang, t_lang)]
            for cal_arm in ("source_dev_c", "source_dev_c:temp", "target_dev_c", "target_dev_c:temp"):
                cal_pts, _ = route_metrics[cal_arm][(s_lang, t_lang)]
                for k in range(n_cats):
                    for inv_m in ("auroc", "ap"):
                        rv, cv = raw_pts[inv_m][k], cal_pts[inv_m][k]
                        if np.isnan(rv) and np.isnan(cv):
                            continue
                        diff = abs(float(cv) - float(rv))
                        if inv_m == "auroc":
                            max_cal_auroc_diff = max(max_cal_auroc_diff, diff)
                        else:
                            max_cal_ap_diff = max(max_cal_ap_diff, diff)
                        check(diff <= INVARIANCE_TOL, f"Calibrator {inv_m} invariance violated: {diff}")

            # 2) E11_diag+source_dev_c / E11_diag+target_dev_c vs E11_diag must preserve AUROC/AP <= 1e-12.
            diag_pts, _ = route_metrics["E11_diag"][(s_lang, t_lang)]
            for cal_arm in (
                "E11_diag+source_dev_c",
                "E11_diag+source_dev_c:temp",
                "E11_diag+target_dev_c",
                "E11_diag+target_dev_c:temp",
            ):
                cal_pts, _ = route_metrics[cal_arm][(s_lang, t_lang)]
                for k in range(n_cats):
                    for inv_m in ("auroc", "ap"):
                        dv, cv = diag_pts[inv_m][k], cal_pts[inv_m][k]
                        if np.isnan(dv) and np.isnan(cv):
                            continue
                        diff = abs(float(cv) - float(dv))
                        if inv_m == "auroc":
                            max_cal_auroc_diff = max(max_cal_auroc_diff, diff)
                        else:
                            max_cal_ap_diff = max(max_cal_ap_diff, diff)
                        check(diff <= INVARIANCE_TOL, f"E11_diag calibrator {inv_m} invariance violated: {diff}")

            # 3) E11_diag:mean_shift vs raw must preserve AUROC/AP <= 1e-12.
            shift_pts, _ = route_metrics["E11_diag:mean_shift"][(s_lang, t_lang)]
            for k in range(n_cats):
                for inv_m in ("auroc", "ap"):
                    rv, sv = raw_pts[inv_m][k], shift_pts[inv_m][k]
                    if np.isnan(rv) and np.isnan(sv):
                        continue
                    diff = abs(float(sv) - float(rv))
                    if inv_m == "auroc":
                        max_shift_auroc_diff = max(max_shift_auroc_diff, diff)
                    else:
                        max_shift_ap_diff = max(max_shift_ap_diff, diff)
                    check(diff <= INVARIANCE_TOL, f"mean_shift {inv_m} invariance violated: {diff}")

    check(temp_only_flips == 0, f"temp-only calibrators produced {temp_only_flips} 0.5-decision flips")

    eval_metrics = ("brier", "nll", "auroc", "ap")
    dec_metrics = ("sensitivity", "specificity", "f1", "balanced_accuracy")
    off_diag_routes = [(s, t) for t in LANGUAGES for s in LANGUAGES if s != t]
    diag_routes = [(t, t) for t in LANGUAGES]

    def _macro_pt_dr(pts_arr, drs_arr):
        return float(fixed_macro(pts_arr, primary_idx, axis=0)), fixed_macro(drs_arr, primary_idx, axis=1)

    def _agg_mean(pairs):
        pt_mean = float(np.mean([p for p, _ in pairs]))
        dr_mean = np.mean(np.stack([d for _, d in pairs], axis=0), axis=0)
        return _summarize_draws(pt_mean, dr_mean)

    def _summarize_arm_set(arm_list):
        summaries = {}
        dec_out = {}
        for arm in arm_list:
            routes_eval = []
            routes_dec = []
            off_m = {m: [] for m in eval_metrics + dec_metrics}
            diag_m = {m: [] for m in eval_metrics + dec_metrics}
            off_gain_raw = {m: [] for m in eval_metrics}
            diag_gain_raw = {m: [] for m in eval_metrics}
            off_gain_fp = {m: [] for m in ("brier", "nll")}
            diag_gain_fp = {m: [] for m in ("brier", "nll")}
            off_gap_vs_tdc = {m: [] for m in ("brier", "nll")}
            off_gain_vs_tdc = {m: [] for m in eval_metrics}
            off_auroc_gap_red = []

            for s_lang, t_lang in [(s, t) for t in LANGUAGES for s in LANGUAGES]:
                pts, drs = route_metrics[arm][(s_lang, t_lang)]
                raw_pts, raw_drs = route_metrics["raw"][(s_lang, t_lang)]
                diag_raw_pts, diag_raw_drs = route_metrics["raw"][(t_lang, t_lang)]
                fp_pts, fp_drs = route_fit_prior[(s_lang, t_lang)]
                tdc_pts, tdc_drs = route_metrics["target_dev_c"][(s_lang, t_lang)]

                row_eval = {
                    "route": f"{s_lang}->{t_lang}",
                    "source_language": s_lang,
                    "target_language": t_lang,
                    "is_diagonal": s_lang == t_lang,
                    "supervision_label": route_supervision[arm][(s_lang, t_lang)],
                    "primary_macro": {},
                    "gain_vs_raw": {},
                    "gain_vs_fit_prior": {},
                    "gain_vs_target_dev_c": {},
                    "loss_gap_to_target_dev_c": {},
                    "per_category": {},
                }
                for m in eval_metrics:
                    pt_m, dr_m = _macro_pt_dr(pts[m], drs[m])
                    raw_pt_m, raw_dr_m = _macro_pt_dr(raw_pts[m], raw_drs[m])
                    tdc_pt_m, tdc_dr_m = _macro_pt_dr(tdc_pts[m], tdc_drs[m])

                    row_eval["primary_macro"][m] = _summarize_draws(pt_m, dr_m)
                    row_eval["per_category"][m] = [_summarize_draws(pts[m][k], drs[m][:, k]) for k in range(n_cats)]

                    if m in ("brier", "nll"):
                        g_raw_pt, g_raw_dr = raw_pt_m - pt_m, raw_dr_m - dr_m
                        g_tdc_pt, g_tdc_dr = tdc_pt_m - pt_m, tdc_dr_m - dr_m
                        gap_tdc_pt, gap_tdc_dr = pt_m - tdc_pt_m, dr_m - tdc_dr_m

                        fp_pt_m, fp_dr_m = _macro_pt_dr(fp_pts[m], fp_drs[m])
                        g_fp_pt, g_fp_dr = fp_pt_m - pt_m, fp_dr_m - dr_m
                        row_eval["gain_vs_fit_prior"][m] = _summarize_draws(g_fp_pt, g_fp_dr)
                        row_eval["loss_gap_to_target_dev_c"][m] = _summarize_draws(gap_tdc_pt, gap_tdc_dr)
                        if s_lang != t_lang:
                            off_gain_fp[m].append((g_fp_pt, g_fp_dr))
                            off_gap_vs_tdc[m].append((gap_tdc_pt, gap_tdc_dr))
                        else:
                            diag_gain_fp[m].append((g_fp_pt, g_fp_dr))
                    else:
                        g_raw_pt, g_raw_dr = pt_m - raw_pt_m, dr_m - raw_dr_m
                        g_tdc_pt, g_tdc_dr = pt_m - tdc_pt_m, dr_m - tdc_dr_m

                    row_eval["gain_vs_raw"][m] = _summarize_draws(g_raw_pt, g_raw_dr)
                    row_eval["gain_vs_target_dev_c"][m] = _summarize_draws(g_tdc_pt, g_tdc_dr)

                    if s_lang != t_lang:
                        off_m[m].append((pt_m, dr_m))
                        off_gain_raw[m].append((g_raw_pt, g_raw_dr))
                        off_gain_vs_tdc[m].append((g_tdc_pt, g_tdc_dr))
                    else:
                        diag_m[m].append((pt_m, dr_m))
                        diag_gain_raw[m].append((g_raw_pt, g_raw_dr))

                if s_lang != t_lang:
                    diag_raw_auc_pt, diag_raw_auc_dr = _macro_pt_dr(diag_raw_pts["auroc"], diag_raw_drs["auroc"])
                    raw_auc_pt, raw_auc_dr = _macro_pt_dr(raw_pts["auroc"], raw_drs["auroc"])
                    arm_auc_pt, arm_auc_dr = _macro_pt_dr(pts["auroc"], drs["auroc"])
                    # Gap reduction = (diag_raw - raw) - (diag_raw - arm) = arm - raw
                    red_pt = arm_auc_pt - raw_auc_pt
                    red_dr = arm_auc_dr - raw_auc_dr
                    row_eval["auroc_gap_reduction_vs_raw"] = _summarize_draws(red_pt, red_dr)
                    off_auroc_gap_red.append((red_pt, red_dr, diag_raw_auc_pt - raw_auc_pt, diag_raw_auc_dr - raw_auc_dr))

                routes_eval.append(row_eval)

                row_dec = {
                    "route": f"{s_lang}->{t_lang}",
                    "source_language": s_lang,
                    "target_language": t_lang,
                    "is_diagonal": s_lang == t_lang,
                    "supervision_label": route_supervision[arm][(s_lang, t_lang)],
                    "primary_macro": {},
                    "flip_vs_raw": route_flips[arm][(s_lang, t_lang)],
                }
                for m in dec_metrics:
                    pt_m, dr_m = _macro_pt_dr(pts[m], drs[m])
                    row_dec["primary_macro"][m] = _summarize_draws(pt_m, dr_m)
                    if s_lang != t_lang:
                        off_m[m].append((pt_m, dr_m))
                    else:
                        diag_m[m].append((pt_m, dr_m))
                routes_dec.append(row_dec)

            off_flips = sum(route_flips[arm][r]["flips"] for r in off_diag_routes)
            off_known = sum(route_flips[arm][r]["known_cells"] for r in off_diag_routes)
            diag_flips = sum(route_flips[arm][r]["flips"] for r in diag_routes)
            diag_known = sum(route_flips[arm][r]["known_cells"] for r in diag_routes)

            raw_gap_pt = float(np.mean([g for _, _, g, _ in off_auroc_gap_red]))
            raw_gap_dr = np.mean(np.stack([gd for _, _, _, gd in off_auroc_gap_red], axis=0), axis=0)
            red_pt_mean = float(np.mean([p for p, _, _, _ in off_auroc_gap_red]))
            red_dr_mean = np.mean(np.stack([d for _, d, _, _ in off_auroc_gap_red], axis=0), axis=0)
            frac_pt = (red_pt_mean / raw_gap_pt) if raw_gap_pt > 0.0 else None
            with np.errstate(divide="ignore", invalid="ignore"):
                frac_dr = np.where(raw_gap_dr > 0.0, red_dr_mean / raw_gap_dr, np.nan)

            summaries[arm] = {
                "off_diagonal_12_route_mean": {
                    "primary_macro": {m: _agg_mean(off_m[m]) for m in eval_metrics},
                    "gain_vs_raw": {m: _agg_mean(off_gain_raw[m]) for m in eval_metrics},
                    "gain_vs_fit_prior": {m: _agg_mean(off_gain_fp[m]) for m in ("brier", "nll")},
                    "gain_vs_target_dev_c": {m: _agg_mean(off_gain_vs_tdc[m]) for m in eval_metrics},
                    "loss_gap_to_target_dev_c": {m: _agg_mean(off_gap_vs_tdc[m]) for m in ("brier", "nll")},
                    "auroc_gap_reduction_vs_raw": _summarize_draws(red_pt_mean, red_dr_mean),
                    "auroc_gap_reduction_fraction_of_g_res": _summarize_draws(frac_pt, frac_dr),
                },
                "diagonal_4_route_mean": {
                    "primary_macro": {m: _agg_mean(diag_m[m]) for m in eval_metrics},
                    "gain_vs_raw": {m: _agg_mean(diag_gain_raw[m]) for m in eval_metrics},
                    "gain_vs_fit_prior": {m: _agg_mean(diag_gain_fp[m]) for m in ("brier", "nll")},
                },
                "routes": routes_eval,
            }
            dec_out[arm] = {
                "off_diagonal_12_route_mean": {
                    "primary_macro": {m: _agg_mean(off_m[m]) for m in dec_metrics},
                    "flips": off_flips,
                    "known_cells": off_known,
                    "flip_rate": float(off_flips / off_known) if off_known > 0 else 0.0,
                },
                "diagonal_4_route_mean": {
                    "primary_macro": {m: _agg_mean(diag_m[m]) for m in dec_metrics},
                    "flips": diag_flips,
                    "known_cells": diag_known,
                    "flip_rate": float(diag_flips / diag_known) if diag_known > 0 else 0.0,
                },
                "routes": routes_dec,
            }
        return summaries, dec_out

    summaries_main, dec_main = _summarize_arm_set(E41_ARMS)
    summaries_aux, dec_aux = _summarize_arm_set(aux_arms)

    # Summarize FIT-prior reference.
    fp_off = {m: [] for m in ("brier", "nll")}
    fp_diag = {m: [] for m in ("brier", "nll")}
    fp_routes = []
    for s_lang, t_lang in [(s, t) for t in LANGUAGES for s in LANGUAGES]:
        fp_pts, fp_drs = route_fit_prior[(s_lang, t_lang)]
        row_fp = {"route": f"{s_lang}->{t_lang}", "primary_macro": {}}
        for m in ("brier", "nll"):
            pt_m, dr_m = _macro_pt_dr(fp_pts[m], fp_drs[m])
            row_fp["primary_macro"][m] = _summarize_draws(pt_m, dr_m)
            if s_lang != t_lang:
                fp_off[m].append((pt_m, dr_m))
            else:
                fp_diag[m].append((pt_m, dr_m))
        fp_routes.append(row_fp)

    fit_prior_reference = {
        "off_diagonal_12_route_mean": {m: _agg_mean(fp_off[m]) for m in ("brier", "nll")},
        "diagonal_4_route_mean": {m: _agg_mean(fp_diag[m]) for m in ("brier", "nll")},
        "routes": fp_routes,
    }

    cell_body = {
        "depth": int(depth),
        "depth_role": depth_role,
        "pooling": pooling,
        "primary_cov_rank": E41_PRIMARY_COV_RANK,
        "fit_prior_reference": fit_prior_reference,
        "summaries": summaries_main,
        "auxiliary_arms": summaries_aux,
        "decision_0p5": dec_main,
        "decision_0p5_auxiliary": dec_aux,
        "invariance_checks": {
            "calibrator_max_abs_auroc_diff": float(max_cal_auroc_diff),
            "calibrator_max_abs_ap_diff": float(max_cal_ap_diff),
            "mean_shift_max_abs_auroc_diff": float(max_shift_auroc_diff),
            "mean_shift_max_abs_ap_diff": float(max_shift_ap_diff),
            "temp_only_0p5_flips": int(temp_only_flips),
            "tolerance": INVARIANCE_TOL,
        },
    }
    return cell_body


@contract
def run_e41(
    phase1_root_dir,
    protocol,
    plan_ref,
    *,
    model,
    producer=None,
    storage_scope="local-development",
    run_id=None,
    resume_prediction_refs=None,
    resume_calibrators_ref=None,
    resume_analysis_ref=None,
    progress=None,
):
    """Run the end-to-end Phase 3 CPU E41 worker for one MODEL ('qwen' or 'gemma')."""
    import numpy as np
    from src.stage4.metrics import bootstrap_counts, paired_image_alignment, support_status
    from src.stage4.plan import load_bootstrap
    from src.stage4.predictions import load_prediction_block

    check(model in MODELS, f"E41 requires MODEL in {MODELS!r}; got {model!r}")
    t_init0 = time.perf_counter()
    plan = load_phase3_cpu_plan(phase1_root_dir, protocol, plan_ref, bind=True)
    resolved_protocol = plain(plan.protocol)
    inv = plan["e41_inventory"][model]
    prod = current_producer(scope=storage_scope) if producer is None else producer
    effective_run_id = run_id or new_run_id()

    depths = [int(d) for d in inv["depths"]]
    check(len(depths) == 2, f"Expected 2 depths (midpoint, final) in e41_inventory; got {depths}")
    depth_roles = {depths[0]: "midpoint", depths[1]: "final"}
    categories = list(resolved_protocol["category_order"])
    n_cats = len(categories)
    primary_idx = _primary_indices(resolved_protocol)
    screen_min_support = int(resolved_protocol["minimum_positive_and_negative"]["SCREEN"])

    total_units = 10
    _emit(
        progress,
        _progress_event(
            "start",
            "init",
            0,
            total_units,
            read_hash_s=time.perf_counter() - t_init0,
            extra={"model": model, "run_id": effective_run_id, "depths": depths},
        ),
    )

    if resume_analysis_ref is not None:
        t_res0 = time.perf_counter()
        res_ref, res_body, _ = verify_resume_ref(
            plan.root,
            resolved_protocol,
            resume_analysis_ref,
            KIND_ANALYSIS,
            expected_schema=SCHEMA_ANALYSIS,
            expected_dataset=plan["dataset_fingerprint"],
            expected_plan_identity=plan["plan_identity"],
            expected_owner=model,
        )
        check(res_body.get("experiment") == "E41", "Resume E41 analysis experiment mismatch")
        inv_chk = res_body.get("invariance_checks", {})
        check(
            inv_chk.get("all_T_positive") is True
            and inv_chk.get("temp_only_0p5_flips") == 0
            and inv_chk.get("e11_cov_zero_target_labels") is True
            and float(inv_chk.get("calibrator_max_abs_auroc_diff", 1.0)) <= INVARIANCE_TOL
            and float(inv_chk.get("calibrator_max_abs_ap_diff", 1.0)) <= INVARIANCE_TOL
            and float(inv_chk.get("mean_shift_max_abs_auroc_diff", 1.0)) <= INVARIANCE_TOL
            and float(inv_chk.get("mean_shift_max_abs_ap_diff", 1.0)) <= INVARIANCE_TOL,
            "Resume E41 analysis failed invariance checks",
        )
        _emit(
            progress,
            _progress_event(
                "complete",
                "analysis:resumed",
                total_units,
                total_units,
                read_hash_s=time.perf_counter() - t_res0,
                extra={"ref": plain(res_ref)},
            ),
        )
        return frozen({
            "model": model,
            "run_id": effective_run_id,
            "prediction_refs": plain(res_body["prediction_refs"]),
            "calibrators_ref": plain(res_body["calibrators_ref"]),
            "analysis_ref": plain(res_ref),
            "resumed": True,
        })

    # Authenticate reused Phase 2 E17 and Phase 2 E11 artifacts (read-only).
    reused_e17_cal_ref = inv["reused_e17_midpoint_calibrators_ref"]
    _, e17_cal_body, _ = load_phase2_artifact(
        plan.root,
        resolved_protocol,
        reused_e17_cal_ref,
        P2_KIND_CALIBRATORS,
        expected_schema=P2_SCHEMA_CALIBRATORS,
        expected_dataset=plan["dataset_fingerprint"],
        expected_owner=model,
    )
    reused_e17_devc_refs = plain(inv["reused_e17_midpoint_dev_c_blocks"])
    for t_lang in LANGUAGES:
        load_phase2_artifact(
            plan.root,
            resolved_protocol,
            reused_e17_devc_refs[t_lang],
            P2_KIND_PREDICTION,
            expected_schema=P2_SCHEMA_PREDICTION,
            expected_dataset=plan["dataset_fingerprint"],
            expected_owner=model,
        )

    reused_e11_refs = plain(inv["reused_e11_midpoint_adapted_refs"])
    for variant in E41_DIAG_VARIANTS:
        for t_lang in LANGUAGES:
            load_phase2_artifact(
                plan.root,
                resolved_protocol,
                reused_e11_refs[variant][t_lang],
                P2_KIND_PREDICTION,
                expected_schema=P2_SCHEMA_PREDICTION,
                expected_dataset=plan["dataset_fingerprint"],
                expected_owner=model,
            )

    resume_pred_map = dict(resume_prediction_refs or {})

    # Optional calibrators resume.
    resumed_cal_body = None
    resumed_cal_ref = None
    if resume_calibrators_ref is not None:
        cal_ref_obj, cal_body_obj, _ = verify_resume_ref(
            plan.root,
            resolved_protocol,
            resume_calibrators_ref,
            KIND_CALIBRATORS,
            expected_schema=SCHEMA_CALIBRATORS,
            expected_dataset=plan["dataset_fingerprint"],
            expected_plan_identity=plan["plan_identity"],
            expected_owner=model,
        )
        check(cal_body_obj.get("experiment") == "E41", "Resume E41 calibrators experiment mismatch")
        resumed_cal_ref = plain(cal_ref_obj)
        resumed_cal_body = plain(cal_body_obj)

    # Canonical SCREEN IDs and bootstrap matrix.
    stage4_plan = plan.binding.stage4_plan
    mid_last_inv = inv["by_depth"][str(depths[0])]["last_caption"]
    sample_preds = [
        load_prediction_block(
            stage4_plan,
            mid_last_inv["screen_blocks"][t_lang]["prediction_ref"],
            expected_block=mid_last_inv["screen_blocks"][t_lang]["block_id"],
        )[mid_last_inv["screen_blocks"][t_lang]["routes"][t_lang]]
        for t_lang in LANGUAGES
    ]
    screen_ids, _ = paired_image_alignment(stage4_plan, *sample_preds)
    n_screen = len(screen_ids)
    y_screen = np.zeros((n_screen, n_cats), dtype=bool)
    known_screen = np.zeros((n_screen, n_cats), dtype=bool)
    for i, img_id in enumerate(screen_ids):
        target = stage4_plan.fit["targets"][img_id]
        check(target["split"] == "SCREEN", f"Unexpected split in SCREEN targets: {target['split']!r}")
        for k, val in enumerate(target["values"]):
            if val is not None:
                known_screen[i, k] = True
                y_screen[i, k] = bool(val)

    draws = load_bootstrap(stage4_plan)
    n_draws = int(draws.shape[0])
    counts = bootstrap_counts(draws, n_screen)

    cat_support = []
    for k, cat_name in enumerate(categories):
        known_k = int(known_screen[:, k].sum())
        pos_k = int(y_screen[known_screen[:, k], k].sum())
        neg_k = known_k - pos_k
        cat_support.append({
            "category": cat_name,
            "is_primary": k in primary_idx,
            "known": known_k,
            "positive": pos_k,
            "negative": neg_k,
            "status": support_status(0.0, pos_k, neg_k, screen_min_support),
        })

    # Per-cell execution: self-check, E11_diag, E11_cov, DEV-C blocks + calibrators.
    published_prediction_refs = {}
    calibrators_by_cell = {}
    cell_results = {}
    self_checks = []
    upstream_all = [
        plain(plan.reference),
        plain(reused_e17_cal_ref),
        *[plain(reused_e17_devc_refs[t_lang]) for t_lang in LANGUAGES],
        *[plain(reused_e11_refs[v][t_lang]) for v in E41_DIAG_VARIANTS for t_lang in LANGUAGES],
    ]

    completed_units = 1
    for depth in depths:
        dkey = str(depth)
        depth_role = depth_roles[depth]
        for pooling in POOLINGS:
            cell_key = f"{depth}:{pooling}"
            cell_inv = inv["by_depth"][dkey][pooling]
            self_checks.append(_run_en_screen_self_check(plan, cell_inv, depth, pooling))

            # Load frozen source heads for all 4 source languages in this cell.
            heads_by_lang = {lang: _load_group_head(plan.binding, cell_inv["group_ids"][lang]) for lang in LANGUAGES}

            # Load SCREEN and DEV-C features per language once for this cell.
            screen_rows_by_lang = {}
            screen_x_by_lang = {}
            devc_rows_by_lang = {}
            devc_x_by_lang = {}
            for lang in LANGUAGES:
                gid = cell_inv["group_ids"][lang]
                s_rows, s_x = _read_verified_group_features(plan.binding, gid, E41_SCREEN_SPLIT)
                d_rows, d_x = _read_verified_group_features(plan.binding, gid, E41_DEV_C_SPLIT)
                screen_rows_by_lang[lang] = s_rows
                screen_x_by_lang[lang] = np.asarray(s_x, dtype=np.float64)
                devc_rows_by_lang[lang] = d_rows
                devc_x_by_lang[lang] = np.asarray(d_x, dtype=np.float64)

            # 1) E11_diag on SCREEN (and DEV-C mean_scale for E11_diag+target_dev_c):
            # Reuse Phase 2 E11 for (midpoint, caption_mean) SCREEN; compute & publish for other 3 cells.
            adapted_screen_by_variant_route = {v: {} for v in E41_DIAG_VARIANTS}
            is_mid_mean = (depth == depths[0] and pooling == "caption_mean")

            if is_mid_mean:
                for variant in E41_DIAG_VARIANTS:
                    for t_lang in LANGUAGES:
                        sorted_sources = sorted(s for s in LANGUAGES if s != t_lang)
                        logits_arr, _, _ = _load_block_arrays(
                            plan.root, reused_e11_refs[variant][t_lang], sorted_sources, n_cats
                        )
                        for s_idx, s_lang in enumerate(sorted_sources):
                            adapted_screen_by_variant_route[variant][(s_lang, t_lang)] = logits_arr[s_idx]
            else:
                for variant in E41_DIAG_VARIANTS:
                    for t_lang in LANGUAGES:
                        sorted_sources = sorted(s for s in LANGUAGES if s != t_lang)
                        pred_key = f"diag:d{depth}:{pooling}:{variant}:{t_lang}"
                        image_ids_sha = fingerprint([r["image_id"] for r in screen_rows_by_lang[t_lang]])
                        if pred_key in resume_pred_map and resume_pred_map[pred_key] is not None:
                            ref_obj, body_obj, _ = verify_resume_ref(
                                plan.root,
                                resolved_protocol,
                                resume_pred_map[pred_key],
                                KIND_PREDICTION,
                                expected_schema=SCHEMA_PREDICTION,
                                expected_dataset=plan["dataset_fingerprint"],
                                expected_plan_identity=plan["plan_identity"],
                                expected_owner=model,
                            )
                            check(
                                body_obj.get("experiment") == "E41"
                                and body_obj.get("variant") == f"E11_diag:{variant}"
                                and int(body_obj.get("depth", -1)) == depth
                                and body_obj.get("pooling") == pooling
                                and body_obj.get("target_language") == t_lang
                                and body_obj.get("image_ids_sha256") == image_ids_sha,
                                f"Incompatible resume prediction block for {pred_key}",
                            )
                            logits_arr, _, _ = _load_block_arrays(plan.root, ref_obj, sorted_sources, n_cats)
                            published_prediction_refs[pred_key] = plain(ref_obj)
                        else:
                            stats_t = heads_by_lang[t_lang][4]
                            logits_list, est_list, prev_list, route_meta = [], [], [], []
                            for s_idx, s_lang in enumerate(sorted_sources):
                                coef_s, int_s, est_s, prev_s, stats_s = heads_by_lang[s_lang]
                                h_ad = apply_e11_diag_transform(screen_x_by_lang[t_lang], stats_s, stats_t, variant)
                                z_ad = _score_through_source_head(h_ad, coef_s, int_s, est_s, stats_s)
                                logits_list.append(z_ad)
                                est_list.append(np.asarray(est_s, dtype=bool))
                                prev_list.append(np.asarray(prev_s, dtype=np.float64))
                                route_meta.append({
                                    "route_id": f"{model}:{E41_CONDITION}:{pooling}:E11_diag:{variant}:{s_lang}->{t_lang}",
                                    "source_language": s_lang,
                                    "row_index": s_idx,
                                    "supervision_label": SUPERVISION_LABELS["target_fit_input_adapted"],
                                })
                            logits_arr = np.asarray(logits_list, dtype=np.float64)
                            block_body = {
                                "schema": SCHEMA_PREDICTION,
                                "experiment": "E41",
                                "plan": plain(plan.reference),
                                "plan_identity": plan["plan_identity"],
                                "model": model,
                                "depth": int(depth),
                                "condition": E41_CONDITION,
                                "pooling": pooling,
                                "split": E41_SCREEN_SPLIT,
                                "variant": f"E11_diag:{variant}",
                                "target_language": t_lang,
                                "source_languages": sorted_sources,
                                "image_ids_sha256": image_ids_sha,
                                "n_images": len(screen_rows_by_lang[t_lang]),
                                "n_categories": n_cats,
                                "category_order": categories,
                                "routes": route_meta,
                                "upstream_feature_ref": plain(cell_inv["feature_releases"][t_lang]["SCREEN"]),
                                "confirm_excluded": True,
                            }
                            ref_obj = publish_phase3(
                                plan.root,
                                resolved_protocol,
                                plan["dataset_fingerprint"],
                                f"{E41_PREDICTIONS_NAMESPACE}/{model}",
                                KIND_PREDICTION,
                                block_body,
                                prod,
                                arrays={
                                    "logits": logits_arr,
                                    "estimable": np.asarray(est_list, dtype=bool),
                                    "prevalence": np.asarray(prev_list, dtype=np.float64),
                                },
                                upstream=[plain(plan.reference), plain(cell_inv["feature_releases"][t_lang]["SCREEN"])],
                                run_id=f"{effective_run_id}-d{depth}-{pooling}-diag-{variant}-{t_lang}",
                            )
                            published_prediction_refs[pred_key] = plain(ref_obj)
                        upstream_all.append(published_prediction_refs[pred_key])
                        for s_idx, s_lang in enumerate(sorted_sources):
                            adapted_screen_by_variant_route[variant][(s_lang, t_lang)] = logits_arr[s_idx]

            # 2) E11_cov on SCREEN (r in {64, 256}, using unlabeled FIT features only — zero target labels!).
            cov_screen_by_rank_route = {r: {} for r in E41_COV_RANKS}
            need_fit_for_cov = False
            for r in E41_COV_RANKS:
                for t_lang in LANGUAGES:
                    pred_key = f"cov:d{depth}:{pooling}:r{r}:{t_lang}"
                    if pred_key not in resume_pred_map or resume_pred_map[pred_key] is None:
                        need_fit_for_cov = True
                        break

            fit_x_by_lang = {}
            pca_v_max_by_source = {}
            if need_fit_for_cov:
                for lang in LANGUAGES:
                    gid = cell_inv["group_ids"][lang]
                    _, f_x = _read_verified_group_features(plan.binding, gid, E41_FIT_SPLIT)
                    fit_x_by_lang[lang] = np.asarray(f_x, dtype=np.float64)
                max_rank = max(E41_COV_RANKS)
                for s_lang in LANGUAGES:
                    xs_c = fit_x_by_lang[s_lang] - np.mean(fit_x_by_lang[s_lang], axis=0)
                    _, _, vt = np.linalg.svd(xs_c, full_matrices=False)
                    eff_max = min(max_rank, vt.shape[0])
                    pca_v_max_by_source[s_lang] = vt[:eff_max, :].T

            for r in E41_COV_RANKS:
                for t_lang in LANGUAGES:
                    sorted_sources = sorted(s for s in LANGUAGES if s != t_lang)
                    pred_key = f"cov:d{depth}:{pooling}:r{r}:{t_lang}"
                    image_ids_sha = fingerprint([r_row["image_id"] for r_row in screen_rows_by_lang[t_lang]])
                    if pred_key in resume_pred_map and resume_pred_map[pred_key] is not None:
                        ref_obj, body_obj, _ = verify_resume_ref(
                            plan.root,
                            resolved_protocol,
                            resume_pred_map[pred_key],
                            KIND_PREDICTION,
                            expected_schema=SCHEMA_PREDICTION,
                            expected_dataset=plan["dataset_fingerprint"],
                            expected_plan_identity=plan["plan_identity"],
                            expected_owner=model,
                        )
                        check(
                            body_obj.get("experiment") == "E41"
                            and body_obj.get("variant") == f"E11_cov:r{r}"
                            and int(body_obj.get("depth", -1)) == depth
                            and body_obj.get("pooling") == pooling
                            and body_obj.get("target_language") == t_lang
                            and body_obj.get("image_ids_sha256") == image_ids_sha,
                            f"Incompatible resume prediction block for {pred_key}",
                        )
                        logits_arr, _, _ = _load_block_arrays(plan.root, ref_obj, sorted_sources, n_cats)
                        published_prediction_refs[pred_key] = plain(ref_obj)
                    else:
                        logits_list, est_list, prev_list, route_meta = [], [], [], []
                        for s_idx, s_lang in enumerate(sorted_sources):
                            coef_s, int_s, est_s, prev_s, stats_s = heads_by_lang[s_lang]
                            mu_s, mu_t, v_r, coral_r, coral_meta = fit_pca_coral_operator(
                                fit_x_by_lang[s_lang],
                                fit_x_by_lang[t_lang],
                                r,
                                basis_v_max=pca_v_max_by_source[s_lang],
                            )
                            h_cov = apply_pca_coral_operator(screen_x_by_lang[t_lang], mu_s, mu_t, v_r, coral_r)
                            z_cov = _score_through_source_head(h_cov, coef_s, int_s, est_s, stats_s)
                            logits_list.append(z_cov)
                            est_list.append(np.asarray(est_s, dtype=bool))
                            prev_list.append(np.asarray(prev_s, dtype=np.float64))
                            route_meta.append({
                                "route_id": f"{model}:{E41_CONDITION}:{pooling}:E11_cov:r{r}:{s_lang}->{t_lang}",
                                "source_language": s_lang,
                                "row_index": s_idx,
                                "coral": coral_meta,
                                "supervision_label": SUPERVISION_LABELS["target_fit_input_adapted"],
                            })
                        logits_arr = np.asarray(logits_list, dtype=np.float64)
                        block_body = {
                            "schema": SCHEMA_PREDICTION,
                            "experiment": "E41",
                            "plan": plain(plan.reference),
                            "plan_identity": plan["plan_identity"],
                            "model": model,
                            "depth": int(depth),
                            "condition": E41_CONDITION,
                            "pooling": pooling,
                            "split": E41_SCREEN_SPLIT,
                            "variant": f"E11_cov:r{r}",
                            "cov_rank": int(r),
                            "zero_target_labels": True,
                            "target_language": t_lang,
                            "source_languages": sorted_sources,
                            "image_ids_sha256": image_ids_sha,
                            "n_images": len(screen_rows_by_lang[t_lang]),
                            "n_categories": n_cats,
                            "category_order": categories,
                            "routes": route_meta,
                            "upstream_feature_ref": plain(cell_inv["feature_releases"][t_lang]["SCREEN"]),
                            "confirm_excluded": True,
                        }
                        ref_obj = publish_phase3(
                            plan.root,
                            resolved_protocol,
                            plan["dataset_fingerprint"],
                            f"{E41_PREDICTIONS_NAMESPACE}/{model}",
                            KIND_PREDICTION,
                            block_body,
                            prod,
                            arrays={
                                "logits": logits_arr,
                                "estimable": np.asarray(est_list, dtype=bool),
                                "prevalence": np.asarray(prev_list, dtype=np.float64),
                            },
                            upstream=[
                                plain(plan.reference),
                                plain(cell_inv["feature_releases"][t_lang]["SCREEN"]),
                                plain(cell_inv["feature_releases"][t_lang]["FIT"]),
                                *[plain(cell_inv["feature_releases"][s]["FIT"]) for s in sorted_sources],
                            ],
                            run_id=f"{effective_run_id}-d{depth}-{pooling}-cov-r{r}-{t_lang}",
                        )
                        published_prediction_refs[pred_key] = plain(ref_obj)
                    upstream_all.append(published_prediction_refs[pred_key])
                    for s_idx, s_lang in enumerate(sorted_sources):
                        cov_screen_by_rank_route[r][(s_lang, t_lang)] = logits_arr[s_idx]

            # Free FIT matrices before calibration/evaluation.
            fit_x_by_lang.clear()
            pca_v_max_by_source.clear()

            # 3) DEV-C off-diagonal blocks & calibrators for this cell.
            is_mid_last = (depth == depths[0] and pooling == "last_caption")
            devc_off_logits = {}
            devc_off_block_refs = {}

            if is_mid_last:
                for t_lang in LANGUAGES:
                    sorted_sources = sorted(s for s in LANGUAGES if s != t_lang)
                    ref_p2 = reused_e17_devc_refs[t_lang]
                    devc_off_block_refs[t_lang] = plain(ref_p2)
                    logits_arr, est_arr, _ = _load_block_arrays(plan.root, ref_p2, sorted_sources, n_cats)
                    for s_idx, s_lang in enumerate(sorted_sources):
                        devc_off_logits[(s_lang, t_lang)] = (logits_arr[s_idx], est_arr[s_idx])
            else:
                for t_lang in LANGUAGES:
                    sorted_sources = sorted(s for s in LANGUAGES if s != t_lang)
                    pred_key = f"devc:d{depth}:{pooling}:{t_lang}"
                    image_ids_sha = fingerprint([r_row["image_id"] for r_row in devc_rows_by_lang[t_lang]])
                    if pred_key in resume_pred_map and resume_pred_map[pred_key] is not None:
                        ref_obj, body_obj, _ = verify_resume_ref(
                            plan.root,
                            resolved_protocol,
                            resume_pred_map[pred_key],
                            KIND_PREDICTION,
                            expected_schema=SCHEMA_PREDICTION,
                            expected_dataset=plan["dataset_fingerprint"],
                            expected_plan_identity=plan["plan_identity"],
                            expected_owner=model,
                        )
                        check(
                            body_obj.get("experiment") == "E41"
                            and body_obj.get("split") == E41_DEV_C_SPLIT
                            and int(body_obj.get("depth", -1)) == depth
                            and body_obj.get("pooling") == pooling
                            and body_obj.get("target_language") == t_lang
                            and body_obj.get("image_ids_sha256") == image_ids_sha,
                            f"Incompatible resume DEV-C block for {pred_key}",
                        )
                        logits_arr, est_arr, _ = _load_block_arrays(plan.root, ref_obj, sorted_sources, n_cats)
                        published_prediction_refs[pred_key] = plain(ref_obj)
                        devc_off_block_refs[t_lang] = plain(ref_obj)
                    else:
                        logits_list, est_list, prev_list, route_meta = [], [], [], []
                        for s_idx, s_lang in enumerate(sorted_sources):
                            coef_s, int_s, est_s, prev_s, stats_s = heads_by_lang[s_lang]
                            z_devc = _score_through_source_head(devc_x_by_lang[t_lang], coef_s, int_s, est_s, stats_s)
                            logits_list.append(z_devc)
                            est_list.append(np.asarray(est_s, dtype=bool))
                            prev_list.append(np.asarray(prev_s, dtype=np.float64))
                            route_meta.append({
                                "route_id": f"{model}:{E41_CONDITION}:{pooling}:dev_c:{s_lang}->{t_lang}",
                                "source_language": s_lang,
                                "row_index": s_idx,
                                "supervision_label": SUPERVISION_LABELS["source_only_transfer"],
                            })
                        logits_arr = np.asarray(logits_list, dtype=np.float64)
                        est_arr = np.asarray(est_list, dtype=bool)
                        block_body = {
                            "schema": SCHEMA_PREDICTION,
                            "experiment": "E41",
                            "plan": plain(plan.reference),
                            "plan_identity": plan["plan_identity"],
                            "model": model,
                            "depth": int(depth),
                            "condition": E41_CONDITION,
                            "pooling": pooling,
                            "split": E41_DEV_C_SPLIT,
                            "variant": "dev_c_off_diagonal",
                            "target_language": t_lang,
                            "source_languages": sorted_sources,
                            "image_ids_sha256": image_ids_sha,
                            "n_images": len(devc_rows_by_lang[t_lang]),
                            "n_categories": n_cats,
                            "category_order": categories,
                            "routes": route_meta,
                            "upstream_feature_ref": plain(cell_inv["feature_releases"][t_lang]["DEV-C"]),
                            "confirm_excluded": True,
                        }
                        ref_obj = publish_phase3(
                            plan.root,
                            resolved_protocol,
                            plan["dataset_fingerprint"],
                            f"{E41_PREDICTIONS_NAMESPACE}/{model}",
                            KIND_PREDICTION,
                            block_body,
                            prod,
                            arrays={
                                "logits": logits_arr,
                                "estimable": est_arr,
                                "prevalence": np.asarray(prev_list, dtype=np.float64),
                            },
                            upstream=[plain(plan.reference), plain(cell_inv["feature_releases"][t_lang]["DEV-C"])],
                            run_id=f"{effective_run_id}-d{depth}-{pooling}-devc-{t_lang}",
                        )
                        published_prediction_refs[pred_key] = plain(ref_obj)
                        devc_off_block_refs[t_lang] = plain(ref_obj)
                    upstream_all.append(devc_off_block_refs[t_lang])
                    for s_idx, s_lang in enumerate(sorted_sources):
                        devc_off_logits[(s_lang, t_lang)] = (logits_arr[s_idx], est_arr[s_idx])

            if resumed_cal_body is not None:
                cal_cell = resumed_cal_body["cells"][cell_key]
            else:
                targets_devc = {
                    lang: _extract_targets_for_rows(stage4_plan, devc_rows_by_lang[lang], E41_DEV_C_SPLIT)
                    for lang in LANGUAGES
                }
                if is_mid_last:
                    src_cal = plain(e17_cal_body["source_dev_c"])
                    tgt_cal = plain(e17_cal_body["target_dev_c"])
                else:
                    src_cal = {}
                    for s_lang in LANGUAGES:
                        coef_s, int_s, est_s, _, stats_s = heads_by_lang[s_lang]
                        z_diag = _score_through_source_head(devc_x_by_lang[s_lang], coef_s, int_s, est_s, stats_s)
                        y_d, known_d = targets_devc[s_lang]
                        src_cal[s_lang] = {
                            form: {
                                **fit_macro_bce_calibrator(
                                    z_diag, y_d, known_d, primary_idx, form, estimable=est_s
                                ),
                                "source_language": s_lang,
                                "target_language": s_lang,
                                "dev_c_input_ref": plain(cell_inv["feature_releases"][s_lang]["DEV-C"]),
                            }
                            for form in E41_CALIBRATOR_FORMS
                        }

                    tgt_cal = {}
                    for t_lang in LANGUAGES:
                        y_d, known_d = targets_devc[t_lang]
                        for s_lang in LANGUAGES:
                            if s_lang == t_lang:
                                continue
                            logits_st, est_st = devc_off_logits[(s_lang, t_lang)]
                            route_key = f"{s_lang}->{t_lang}"
                            tgt_cal[route_key] = {
                                form: {
                                    **fit_macro_bce_calibrator(
                                        logits_st, y_d, known_d, primary_idx, form, estimable=est_st
                                    ),
                                    "source_language": s_lang,
                                    "target_language": t_lang,
                                    "dev_c_block_ref": devc_off_block_refs[t_lang],
                                }
                                for form in E41_CALIBRATOR_FORMS
                            }

                # Fit target_dev_c_on_e11_diag for E11_diag+target_dev_c in every cell.
                tgt_cal_e11_diag = {}
                for t_lang in LANGUAGES:
                    y_d, known_d = targets_devc[t_lang]
                    stats_t = heads_by_lang[t_lang][4]
                    for s_lang in LANGUAGES:
                        if s_lang == t_lang:
                            continue
                        coef_s, int_s, est_s, _, stats_s = heads_by_lang[s_lang]
                        h_devc_ad = apply_e11_diag_transform(devc_x_by_lang[t_lang], stats_s, stats_t, "mean_scale")
                        z_devc_ad = _score_through_source_head(h_devc_ad, coef_s, int_s, est_s, stats_s)
                        route_key = f"{s_lang}->{t_lang}"
                        tgt_cal_e11_diag[route_key] = {
                            form: {
                                **fit_macro_bce_calibrator(z_devc_ad, y_d, known_d, primary_idx, form, estimable=est_s),
                                "source_language": s_lang,
                                "target_language": t_lang,
                            }
                            for form in E41_CALIBRATOR_FORMS
                        }

                cal_cell = {
                    "depth": int(depth),
                    "depth_role": depth_role,
                    "pooling": pooling,
                    "reused_e17_source_and_target": bool(is_mid_last),
                    "source_dev_c": src_cal,
                    "target_dev_c": tgt_cal,
                    "target_dev_c_on_e11_diag": tgt_cal_e11_diag,
                }

            calibrators_by_cell[cell_key] = cal_cell

            # 4) Evaluate all 7 arms + auxiliary arms on SCREEN.
            cell_results[cell_key] = _evaluate_cell(
                plan=plan,
                model=model,
                depth=depth,
                depth_role=depth_role,
                pooling=pooling,
                cell_inv=cell_inv,
                cal_cell=cal_cell,
                adapted_screen_by_variant_route=adapted_screen_by_variant_route,
                cov_screen_by_rank_route=cov_screen_by_rank_route,
                y_screen=y_screen,
                known_screen=known_screen,
                screen_ids=screen_ids,
                counts=counts,
                n_draws=n_draws,
                categories=categories,
                primary_idx=primary_idx,
                screen_min_support=screen_min_support,
            )

            completed_units += 2
            _emit(
                progress,
                _progress_event(
                    "progress",
                    f"cell:{cell_key}",
                    min(completed_units, total_units - 1),
                    total_units,
                    extra={"model": model, "depth": depth, "pooling": pooling},
                ),
            )

    # Verify all fitted T > 0 across all cells.
    all_t_positive = True
    for cell_key, cal_cell in calibrators_by_cell.items():
        for scope in ("source_dev_c", "target_dev_c", "target_dev_c_on_e11_diag"):
            for _, forms in cal_cell[scope].items():
                for form_name in E41_CALIBRATOR_FORMS:
                    if not (float(forms[form_name]["T"]) > 0.0):
                        all_t_positive = False
    check(all_t_positive, "All fitted calibrator temperatures T must be strictly positive")

    # Publish calibrators artifact (or reuse verified resume_calibrators_ref).
    if resumed_cal_ref is not None:
        cal_ref = resumed_cal_ref
    else:
        cal_body = {
            "schema": SCHEMA_CALIBRATORS,
            "experiment": "E41",
            "plan": plain(plan.reference),
            "plan_identity": plan["plan_identity"],
            "model": model,
            "depths": depths,
            "poolings": list(POOLINGS),
            "reused_e17_calibrators_ref": plain(reused_e17_cal_ref),
            "objective": "equal_weight_14_primary_macro_bce",
            "optimizer": {
                "method": "BFGS",
                "polish": "L-BFGS-B_on_status2",
                "jac": True,
                "gtol": BFGS_GTOL,
                "maxiter": BFGS_MAXITER,
                "init": {"log_T": 0.0, "b": 0.0},
            },
            "cells": calibrators_by_cell,
            "confirm_excluded": True,
        }
        cal_ref = plain(
            publish_phase3(
                plan.root,
                resolved_protocol,
                plan["dataset_fingerprint"],
                f"{E41_CALIBRATORS_NAMESPACE}/{model}",
                KIND_CALIBRATORS,
                cal_body,
                prod,
                upstream=upstream_all,
                run_id=effective_run_id,
            )
        )

    # Aggregate global invariance checks across all 4 cells.
    max_cal_auroc_diff = max(c["invariance_checks"]["calibrator_max_abs_auroc_diff"] for c in cell_results.values())
    max_cal_ap_diff = max(c["invariance_checks"]["calibrator_max_abs_ap_diff"] for c in cell_results.values())
    max_shift_auroc_diff = max(c["invariance_checks"]["mean_shift_max_abs_auroc_diff"] for c in cell_results.values())
    max_shift_ap_diff = max(c["invariance_checks"]["mean_shift_max_abs_ap_diff"] for c in cell_results.values())
    total_temp_flips = sum(c["invariance_checks"]["temp_only_0p5_flips"] for c in cell_results.values())

    analysis_body = {
        "schema": SCHEMA_ANALYSIS,
        "experiment": "E41",
        "plan": plain(plan.reference),
        "plan_identity": plan["plan_identity"],
        "model": model,
        "depths": depths,
        "midpoint_depth": depths[0],
        "final_depth": depths[1],
        "poolings": list(POOLINGS),
        "arms": list(E41_ARMS),
        "cov_ranks": list(E41_COV_RANKS),
        "primary_cov_rank": E41_PRIMARY_COV_RANK,
        "bootstrap": plain(plan["bootstrap"]),
        "category_order": categories,
        "primary_categories": list(resolved_protocol["primary_categories"]),
        "category_support": cat_support,
        "reused_phase2_refs": {
            "e17_calibrators_ref": plain(reused_e17_cal_ref),
            "e17_dev_c_block_refs": reused_e17_devc_refs,
            "e11_adapted_block_refs": reused_e11_refs,
        },
        "calibrators_ref": cal_ref,
        "prediction_refs": published_prediction_refs,
        "en_diagonal_self_checks": self_checks,
        "invariance_checks": {
            "all_T_positive": all_t_positive,
            "temp_only_0p5_flips": int(total_temp_flips),
            "calibrator_max_abs_auroc_diff": float(max_cal_auroc_diff),
            "calibrator_max_abs_ap_diff": float(max_cal_ap_diff),
            "mean_shift_max_abs_auroc_diff": float(max_shift_auroc_diff),
            "mean_shift_max_abs_ap_diff": float(max_shift_ap_diff),
            "e11_cov_zero_target_labels": True,
            "tolerance": INVARIANCE_TOL,
        },
        "cells": cell_results,
        "confirm_excluded": True,
    }

    analysis_ref = plain(
        publish_phase3(
            plan.root,
            resolved_protocol,
            plan["dataset_fingerprint"],
            f"{E41_RESULTS_NAMESPACE}/{model}",
            KIND_ANALYSIS,
            analysis_body,
            prod,
            upstream=upstream_all + [cal_ref],
            run_id=effective_run_id,
        )
    )

    result = {
        "model": model,
        "run_id": effective_run_id,
        "prediction_refs": published_prediction_refs,
        "calibrators_ref": cal_ref,
        "analysis_ref": analysis_ref,
        "resumed": False,
    }
    _emit(
        progress,
        _progress_event(
            "complete",
            "e41_complete",
            total_units,
            total_units,
            extra={"model": model, "analysis_ref": analysis_ref},
        ),
    )
    return frozen(result)


def _parse_ref_arg(value):
    if value is None:
        return None
    text = value.strip()
    if text.startswith("{"):
        return json.loads(text)
    path = Path(text)
    if path.is_file() and path.suffix == ".json" and path.name != "_COMPLETE.json":
        return json.loads(path.read_text(encoding="utf-8"))
    return text


def _parse_ref_map_arg(value):
    if value is None:
        return None
    text = value.strip()
    if text.startswith("{"):
        return json.loads(text)
    path = Path(text)
    check(path.is_file(), f"Missing resume prediction refs JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _cli(argv=None):
    parser = argparse.ArgumentParser(description="Phase 3 Stage A2 CPU E41 worker")
    parser.add_argument("--phase1-root", type=Path, default=None)
    parser.add_argument("--protocol", type=Path, default=REPO / "configs/protocol.json")
    parser.add_argument("--plan-ref", required=True)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--storage-scope", default="local-development")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume-prediction-refs", default=None)
    parser.add_argument("--resume-calibrators-ref", default=None)
    parser.add_argument("--resume-analysis-ref", default=None)
    parser.add_argument("--output-ref", type=Path, default=None)
    args = parser.parse_args(argv)
    resolved_root = args.phase1_root if args.phase1_root is not None else phase1_root()
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))

    def _stdout_progress(event):
        print(json.dumps(event, sort_keys=True), flush=True)

    try:
        result = plain(
            run_e41(
                resolved_root,
                protocol,
                _parse_ref_arg(args.plan_ref),
                model=args.model,
                storage_scope=args.storage_scope,
                run_id=args.run_id,
                resume_prediction_refs=_parse_ref_map_arg(args.resume_prediction_refs),
                resume_calibrators_ref=_parse_ref_arg(args.resume_calibrators_ref),
                resume_analysis_ref=_parse_ref_arg(args.resume_analysis_ref),
                progress=_stdout_progress,
            )
        )
        if args.output_ref is not None:
            args.output_ref.parent.mkdir(parents=True, exist_ok=True)
            args.output_ref.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"status": "passed", **result}, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        print(json.dumps({"status": "blocked", "blockers": [str(exc)]}, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
