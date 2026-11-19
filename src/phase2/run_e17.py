"""Phase 2 Stage 2 CPU E17 worker: DEV-C off-diagonal scoring, calibration, and SCREEN evaluation.

Implements phase2.md §1.3-1.6, §2.1-2.2, and §2.4:
1. Authenticates `phase2_plan` and `phase2_e17_auth` (`phase2.e17_dev_c_transfer`) and runs
   the mandatory in-memory `en -> en` DEV-C diagonal reproduction check (`atol=0, rtol=0`)
   before writing any output.
2. Scores and publishes the 4 target-context off-diagonal DEV-C blocks (12 frozen-source-head
   `s != t` routes per model) under `phase2/predictions/e17_dev_c/<model>/<run_id>/`.
3. Fits global float64 `temp` and `temp_intercept` calibrators (equal-weight 14-primary-category
   macro BCE, deterministic BFGS with analytic gradient, `gtol=1e-10`, `maxiter=500`, zero init)
   separately on `source_dev_c` (reused Stage 4 diagonals) and `target_dev_c` (frozen source head
   on target DEV-C, never target-trained heads), publishing `phase2_e17_calibrators`.
4. Evaluates all 5 arms (`raw`, `source_dev_c:temp`, `source_dev_c:temp_intercept`,
   `target_dev_c:temp`, `target_dev_c:temp_intercept`) on reused Stage 4 SCREEN logits against
   `raw` and fixed FIT-prior references with paired `(2000, 360)` bootstrap intervals, enforcing
   per-category AUROC/AP invariance (`<= 1e-12`), `T > 0`, `temp`-only 0.5-decision invariance,
   and separate `decision_0p5` reporting.
"""

import argparse
import json
from pathlib import Path
import sys
import time

from src.common.contracts import ContractError, LANGUAGES, fingerprint
from src.common.storage import new_run_id, phase1_root, within
from src.phase2.e17_auth import (
    E17_BFGS_GTOL,
    E17_BFGS_MAXITER,
    E17_CALIBRATOR_FORMS,
    E17_CALIBRATOR_SCOPES,
    E17_CALIBRATORS_NAMESPACE,
    E17_CONDITION,
    E17_DEV_C_PERMISSION,
    E17_DEV_C_PREDICTIONS_NAMESPACE,
    E17_DEV_C_SPLIT,
    E17_INVARIANCE_TOL,
    E17_POOLING,
    E17_SUPERVISION_LABELS,
    authorize_e17_operation,
    enumerate_e17_model_inventory,
    load_e17_auth,
    score_frozen_head,
    validate_target_calibrator_input,
    verify_diagonal_reproduction,
)
from src.phase2.interfaces import (
    KIND_ANALYSIS,
    KIND_CALIBRATORS,
    KIND_PREDICTION,
    MODELS,
    SCHEMA_ANALYSIS,
    SCHEMA_CALIBRATORS,
    SCHEMA_PREDICTION,
    bind_phase1,
    check,
    contract,
    current_producer,
    frozen,
    load_frozen_source_head,
    normalize_ref,
    plain,
    publish_phase2,
    read_verified_midpoint_features,
    require_open_split,
    verify_resume_ref,
)
from src.phase2.plan import Phase2Plan, load_phase2_plan

E17_RESULTS_NAMESPACE = "phase2/results/e17"
E17_ARMS = (
    "raw",
    "source_dev_c:temp",
    "source_dev_c:temp_intercept",
    "target_dev_c:temp",
    "target_dev_c:temp_intercept",
)


def _peak_rss_bytes():
    """Return process peak RSS in bytes where available, else None."""
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


def _extract_targets_for_rows(stage4_plan, rows, expected_split):
    """Build aligned boolean (y, known) arrays of shape (n_rows, n_cats) from plan targets."""
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


def _primary_indices(protocol):
    """Return ordered integer indices of the 14 fixed primary categories."""
    categories = list(protocol["category_order"])
    primary = list(protocol["primary_categories"])
    check(len(primary) == 14, f"Expected 14 fixed primary categories; got {len(primary)}")
    return [categories.index(cat) for cat in primary]


def _run_en_diagonal_self_check(plan, inv, model):
    """Run mandatory in-memory en->en DEV-C diagonal reproduction check (atol=0, rtol=0)."""
    from src.stage4.predictions import load_prediction_block

    authorize_e17_operation(
        split=E17_DEV_C_SPLIT,
        operation="diagonal_self_check",
        source_lang="en",
        target_lang="en",
        head_source_lang="en",
        allow_diagonal_self_check=True,
    )
    rows_en, matrix_en = read_verified_midpoint_features(
        plan.binding, model, E17_CONDITION, E17_POOLING, "en", E17_DEV_C_SPLIT
    )
    coef_en, intercept_en, estimable_en, _, stats_en = load_frozen_source_head(
        plan.binding, model, E17_CONDITION, E17_POOLING, "en"
    )
    computed_en = score_frozen_head(matrix_en, coef_en, intercept_en, stats_en, estimable=estimable_en)
    diag_en = inv["dev_c_diagonals"]["en"]
    accepted_pred = load_prediction_block(
        plan.binding.stage4_plan, diag_en["prediction_ref"], expected_block=diag_en["block_id"]
    )[diag_en["route_id"]]
    check(plain(rows_en) == plain(accepted_pred.rows), f"DEV-C en->en row alignment mismatch for {model}")
    diag_result = verify_diagonal_reproduction(
        computed_en, accepted_pred.logits, atol=0.0, rtol=0.0, model=model
    )
    return {
        "route": "en->en",
        "passed": True,
        "max_abs_diff": 0.0,
        "atol": 0.0,
        "rtol": 0.0,
        "row_count": int(diag_result["row_count"]),
        "category_count": int(diag_result["category_count"]),
    }


def _load_dev_c_block_arrays(root, ref, expected_sources, expected_n_cats):
    """Load and validate logits, estimable, prevalence from a published E17 DEV-C block."""
    import numpy as np

    marker_dir = within(root, ref["path"]).parent
    logits = np.load(marker_dir / "logits.npy", allow_pickle=False)
    estimable = np.load(marker_dir / "estimable.npy", allow_pickle=False)
    prevalence = np.load(marker_dir / "prevalence.npy", allow_pickle=False)
    n_src = len(expected_sources)
    check(
        logits.ndim == 3 and logits.shape[0] == n_src and logits.shape[2] == expected_n_cats,
        f"Invalid E17 DEV-C logits shape: {logits.shape}",
    )
    check(logits.dtype == np.float64 and np.isfinite(logits).all(), "E17 DEV-C logits must be finite float64")
    check(estimable.shape == (n_src, expected_n_cats) and estimable.dtype == bool, "Invalid DEV-C estimable shape")
    check(prevalence.shape == (n_src, expected_n_cats) and prevalence.dtype == np.float64, "Invalid DEV-C prevalence")
    return logits, estimable, prevalence


def _score_or_resume_dev_c_blocks(
    plan,
    auth_ref,
    inv,
    model,
    producer,
    *,
    run_id,
    resume_dev_c_refs=None,
    progress=None,
    completed_base=1,
    total_units=10,
):
    """Score (or resume via explicit refs) the 4 off-diagonal target-context DEV-C blocks."""
    import numpy as np

    root = plan.root
    protocol = plain(plan.protocol)
    categories = list(protocol["category_order"])
    n_cats = len(categories)
    resume_map = dict(resume_dev_c_refs or {})

    # Cache frozen source heads once per source language (no FIT feature matrix reads).
    source_heads = {}
    for src_lang in LANGUAGES:
        source_heads[src_lang] = load_frozen_source_head(
            plan.binding, model, E17_CONDITION, E17_POOLING, src_lang
        )

    diag_check = _run_en_diagonal_self_check(plan, inv, model)
    _emit(
        progress,
        _progress_event(
            "progress",
            "diagonal_self_check",
            completed_base,
            total_units,
            extra={"model": model, "max_abs_diff": 0.0},
        ),
    )

    block_refs = {}
    dev_c_logits_by_route = {}
    dev_c_rows_by_target = {}

    for idx, target_lang in enumerate(LANGUAGES, start=1):
        t_read0 = time.perf_counter()
        off_block = inv["off_diagonal_blocks"][target_lang]
        sorted_sources = sorted(src for src in LANGUAGES if src != target_lang)
        expected_routes_by_src = {r["source_language"]: r for r in off_block["routes"]}
        rows, matrix = read_verified_midpoint_features(
            plan.binding, model, E17_CONDITION, E17_POOLING, target_lang, E17_DEV_C_SPLIT
        )
        check(len(rows) > 0, f"Empty DEV-C feature matrix for {model}/{target_lang}")
        image_ids = [row["image_id"] for row in rows]
        image_ids_sha256 = fingerprint(image_ids)
        dev_c_rows_by_target[target_lang] = rows
        read_s = time.perf_counter() - t_read0

        if target_lang in resume_map and resume_map[target_lang] is not None:
            t_res0 = time.perf_counter()
            res_ref, res_body, _ = verify_resume_ref(
                root,
                protocol,
                resume_map[target_lang],
                KIND_PREDICTION,
                expected_schema=SCHEMA_PREDICTION,
                expected_dataset=plan["dataset_fingerprint"],
                expected_plan_identity=plan["plan_identity"],
                expected_owner=model,
            )
            check(res_body.get("experiment") == "E17", "Resume DEV-C block experiment mismatch")
            check(res_body.get("split") == E17_DEV_C_SPLIT, "Resume DEV-C block split mismatch")
            check(res_body.get("target_language") == target_lang, "Resume DEV-C target_language mismatch")
            check(list(res_body.get("source_languages", ())) == sorted_sources, "Resume DEV-C sources mismatch")
            check(plain(res_body.get("e17_auth_ref")) == plain(auth_ref), "Resume DEV-C e17_auth_ref mismatch")
            check(
                plain(res_body.get("upstream_feature_ref")) == plain(off_block["target_release"]),
                "Resume DEV-C feature ref mismatch",
            )
            check(res_body.get("image_ids_sha256") == image_ids_sha256, "Resume DEV-C image_ids_sha256 mismatch")
            logits_arr, estimable_arr, _ = _load_dev_c_block_arrays(root, res_ref, sorted_sources, n_cats)
            check(logits_arr.shape[1] == len(rows), "Resume DEV-C row count mismatch")
            block_refs[target_lang] = plain(res_ref)
            for s_idx, src_lang in enumerate(sorted_sources):
                dev_c_logits_by_route[(src_lang, target_lang)] = {
                    "logits": logits_arr[s_idx],
                    "estimable": estimable_arr[s_idx],
                    "route_meta": expected_routes_by_src[src_lang],
                    "block_ref": plain(res_ref),
                }
            _emit(
                progress,
                _progress_event(
                    "progress",
                    f"dev_c_block:{target_lang}:resumed",
                    completed_base + idx,
                    total_units,
                    read_hash_s=read_s + (time.perf_counter() - t_res0),
                ),
            )
            continue

        t_comp0 = time.perf_counter()
        logits_list = []
        valid_list = []
        prev_list = []
        routes_meta = []
        for s_idx, src_lang in enumerate(sorted_sources):
            authorize_e17_operation(
                split=E17_DEV_C_SPLIT,
                operation="score_off_diagonal",
                source_lang=src_lang,
                target_lang=target_lang,
                head_source_lang=src_lang,
            )
            coef, intercept, estimable, prevalence, stats = source_heads[src_lang]
            logits_s = score_frozen_head(matrix, coef, intercept, stats, estimable=estimable)
            logits_list.append(logits_s)
            valid_list.append(estimable)
            prev_list.append(prevalence)
            route_info = expected_routes_by_src[src_lang]
            routes_meta.append({
                "route_id": route_info["route_id"],
                "source_language": src_lang,
                "row_index": s_idx,
                "source_group": route_info["source_group"],
                "target_group": route_info["target_group"],
                "source_system_ref": plain(route_info["source_selection_ref"]),
                "source_candidate_ref": plain(route_info["source_candidate_ref"]),
                "source_statistics_ref": plain(route_info["source_statistics_ref"]),
                "supervision_label": E17_SUPERVISION_LABELS["source_only_transfer"],
            })

        logits_arr = np.asarray(logits_list, dtype=np.float64)
        estimable_arr = np.asarray(valid_list, dtype=bool)
        prevalence_arr = np.asarray(prev_list, dtype=np.float64)
        compute_s = time.perf_counter() - t_comp0

        body = {
            "schema": SCHEMA_PREDICTION,
            "experiment": "E17",
            "plan": plain(plan.reference),
            "plan_identity": plan["plan_identity"],
            "e17_auth_ref": plain(auth_ref),
            "model": model,
            "midpoint_depth": int(inv["midpoint_depth"]),
            "condition": E17_CONDITION,
            "pooling": E17_POOLING,
            "split": E17_DEV_C_SPLIT,
            "target_language": target_lang,
            "source_languages": sorted_sources,
            "image_ids_sha256": image_ids_sha256,
            "n_images": len(rows),
            "n_categories": n_cats,
            "category_order": categories,
            "diagonal_self_check": diag_check,
            "routes": routes_meta,
            "upstream_feature_ref": plain(off_block["target_release"]),
            "confirm_excluded": True,
        }
        upstream = [
            plain(plan.reference),
            plain(auth_ref),
            plain(off_block["target_release"]),
        ]
        for r_meta in routes_meta:
            upstream.extend([
                r_meta["source_system_ref"],
                r_meta["source_candidate_ref"],
                r_meta["source_statistics_ref"],
            ])

        t_write0 = time.perf_counter()
        block_ref = publish_phase2(
            root,
            protocol,
            plan["dataset_fingerprint"],
            f"{E17_DEV_C_PREDICTIONS_NAMESPACE}/{model}",
            KIND_PREDICTION,
            body,
            producer,
            arrays={
                "logits": logits_arr,
                "estimable": estimable_arr,
                "prevalence": prevalence_arr,
            },
            upstream=upstream,
            run_id=f"{run_id}_{target_lang}",
        )
        write_s = time.perf_counter() - t_write0
        block_refs[target_lang] = plain(block_ref)
        for s_idx, src_lang in enumerate(sorted_sources):
            dev_c_logits_by_route[(src_lang, target_lang)] = {
                "logits": logits_arr[s_idx],
                "estimable": estimable_arr[s_idx],
                "route_meta": expected_routes_by_src[src_lang],
                "block_ref": plain(block_ref),
            }
        _emit(
            progress,
            _progress_event(
                "progress",
                f"dev_c_block:{target_lang}",
                completed_base + idx,
                total_units,
                read_hash_s=read_s,
                compute_s=compute_s,
                write_s=write_s,
                extra={"ref": plain(block_ref)},
            ),
        )

    return block_refs, dev_c_logits_by_route, dev_c_rows_by_target, diag_check


def fit_macro_bce_calibrator(logits, y, known, primary_indices, form, *, estimable=None):
    """Fit global float64 `temp` or `temp_intercept` calibrator via deterministic BFGS.

    Objective (phase2.md §1.6 / §2.4):
        L = (1/14) * sum_{k in primary} mean_{i in known(DEV-C, k)} [logaddexp(0, z'_{ik}) - y_{ik} z'_{ik}]
    with exact analytic gradient, gtol=1e-10, maxiter=500, zero initialization.
    """
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import expit

    check(form in ("temp", "temp_intercept"), f"Unsupported calibrator form: {form!r}")
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

    # Pre-slice known cells per primary category for fast exact macro objective & gradient.
    cat_slices = []
    for k in primary_indices:
        mask = known_arr[:, k]
        n_k = int(mask.sum())
        check(n_k > 0, f"Primary category index {k} has zero known DEV-C observations")
        cat_slices.append((z[mask, k], y_arr[mask, k], float(n_k)))

    n_primary = float(len(cat_slices))

    def _objective_and_grad(params):
        log_t = float(params[0])
        # Guard against extreme overflow during BFGS line search while keeping finite gradient.
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
        options={"gtol": E17_BFGS_GTOL, "maxiter": E17_BFGS_MAXITER},
    )
    total_nit = int(res.nit)
    if (not res.success) and getattr(res, "status", None) == 2 and np.isfinite(res.x).all() and np.isfinite(res.fun):
        # SciPy BFGS can report status=2 (precision loss) when already at the float64 minimum;
        # polish with L-BFGS-B from res.x and verify small gradient norm.
        polish = minimize(
            _objective_and_grad,
            np.asarray(res.x, dtype=np.float64),
            method="L-BFGS-B",
            jac=True,
            options={"gtol": E17_BFGS_GTOL, "ftol": 1e-15, "maxiter": E17_BFGS_MAXITER},
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


def _fit_or_resume_calibrators(
    plan,
    auth_ref,
    inv,
    model,
    producer,
    dev_c_block_refs,
    dev_c_logits_by_route,
    dev_c_rows_by_target,
    *,
    run_id,
    resume_calibrators_ref=None,
    progress=None,
    completed_base=5,
    total_units=10,
):
    """Fit (or resume via explicit ref) source_dev_c and target_dev_c calibrators."""
    from src.stage4.predictions import load_prediction_block

    root = plan.root
    protocol = plain(plan.protocol)
    primary_idx = _primary_indices(protocol)

    if resume_calibrators_ref is not None:
        t_res0 = time.perf_counter()
        cal_ref, cal_body, _ = verify_resume_ref(
            root,
            protocol,
            resume_calibrators_ref,
            KIND_CALIBRATORS,
            expected_schema=SCHEMA_CALIBRATORS,
            expected_dataset=plan["dataset_fingerprint"],
            expected_plan_identity=plan["plan_identity"],
            expected_owner=model,
        )
        check(cal_body.get("experiment") == "E17", "Resume calibrators experiment mismatch")
        check(plain(cal_body.get("e17_auth_ref")) == plain(auth_ref), "Resume calibrators e17_auth_ref mismatch")
        check(set(cal_body.get("source_dev_c", {}).keys()) == set(LANGUAGES), "Resume calibrators missing source_dev_c")
        check(len(cal_body.get("target_dev_c", {})) == 12, "Resume calibrators missing target_dev_c")
        for s_lang, forms in cal_body["source_dev_c"].items():
            for form_name in ("temp", "temp_intercept"):
                entry = forms[form_name]
                check(entry["converged"] and float(entry["T"]) > 0.0, f"Invalid resumed source calibrator {s_lang}/{form_name}")
        for route_key, forms in cal_body["target_dev_c"].items():
            for form_name in ("temp", "temp_intercept"):
                entry = forms[form_name]
                check(entry["converged"] and float(entry["T"]) > 0.0, f"Invalid resumed target calibrator {route_key}/{form_name}")
                check(
                    plain(entry["dev_c_block_ref"]) == plain(dev_c_block_refs[entry["target_language"]]),
                    f"Resume calibrator {route_key} DEV-C block ref mismatch",
                )
        _emit(
            progress,
            _progress_event(
                "progress",
                "calibrators:resumed",
                completed_base + 2,
                total_units,
                read_hash_s=time.perf_counter() - t_res0,
            ),
        )
        return plain(cal_ref), plain(cal_body)

    t_comp0 = time.perf_counter()
    targets_by_lang = {
        lang: _extract_targets_for_rows(plan.binding.stage4_plan, dev_c_rows_by_target[lang], E17_DEV_C_SPLIT)
        for lang in LANGUAGES
    }

    # Scope 1: source_dev_c (4 reused Stage 4 s -> s diagonal blocks).
    source_dev_c = {}
    upstream_refs = [plain(plan.reference), plain(auth_ref)]
    for lang in LANGUAGES:
        authorize_e17_operation(
            split=E17_DEV_C_SPLIT,
            operation="fit_source_calibrator",
            source_lang=lang,
            target_lang=lang,
            head_source_lang=lang,
        )
        diag_info = inv["dev_c_diagonals"][lang]
        pred = load_prediction_block(
            plan.binding.stage4_plan, diag_info["prediction_ref"], expected_block=diag_info["block_id"]
        )[diag_info["route_id"]]
        check(plain(pred.rows) == plain(dev_c_rows_by_target[lang]), f"Stage 4 DEV-C diagonal row mismatch for {lang}")
        y_dev, known_dev = targets_by_lang[lang]
        diag_ref_plain = plain(diag_info["prediction_ref"])
        upstream_refs.append(diag_ref_plain)
        source_dev_c[lang] = {}
        for form in ("temp", "temp_intercept"):
            fitted = fit_macro_bce_calibrator(
                pred.logits, y_dev, known_dev, primary_idx, form, estimable=pred.estimable
            )
            source_dev_c[lang][form] = {
                **fitted,
                "source_language": lang,
                "target_language": lang,
                "dev_c_input_ref": diag_ref_plain,
                "supervision_label": E17_SUPERVISION_LABELS["source_dev_c_calibrated"],
                "diagonal_note": E17_SUPERVISION_LABELS["within_context_source_fit"],
            }

    # Scope 2: target_dev_c (12 newly scored off-diagonal s -> t blocks; never target-trained heads).
    target_dev_c = {}
    for target_lang in LANGUAGES:
        y_dev, known_dev = targets_by_lang[target_lang]
        target_gid = inv["off_diagonal_blocks"][target_lang]["target_group"]
        upstream_refs.append(plain(dev_c_block_refs[target_lang]))
        for source_lang in LANGUAGES:
            if source_lang == target_lang:
                continue
            route_entry = dev_c_logits_by_route[(source_lang, target_lang)]
            route_meta = route_entry["route_meta"]
            validate_target_calibrator_input(
                source_lang,
                target_lang,
                logits_source_group=route_meta["source_group"],
                source_group_id=inv["source_systems"][source_lang]["group_id"],
                target_group_id=target_gid,
                split=E17_DEV_C_SPLIT,
            )
            route_key = f"{source_lang}->{target_lang}"
            target_dev_c[route_key] = {}
            for form in ("temp", "temp_intercept"):
                fitted = fit_macro_bce_calibrator(
                    route_entry["logits"],
                    y_dev,
                    known_dev,
                    primary_idx,
                    form,
                    estimable=route_entry["estimable"],
                )
                target_dev_c[route_key][form] = {
                    **fitted,
                    "source_language": source_lang,
                    "target_language": target_lang,
                    "dev_c_input_ref": plain(route_entry["block_ref"]),
                    "dev_c_block_ref": plain(route_entry["block_ref"]),
                    "supervision_label": E17_SUPERVISION_LABELS["target_dev_c_supervised_calibration"],
                }

    compute_s = time.perf_counter() - t_comp0
    cal_body = {
        "schema": SCHEMA_CALIBRATORS,
        "experiment": "E17",
        "plan": plain(plan.reference),
        "plan_identity": plan["plan_identity"],
        "e17_auth_ref": plain(auth_ref),
        "model": model,
        "midpoint_depth": int(inv["midpoint_depth"]),
        "objective": "equal_weight_14_primary_macro_bce",
        "optimizer": {
            "method": "BFGS",
            "jac": True,
            "gtol": E17_BFGS_GTOL,
            "maxiter": E17_BFGS_MAXITER,
            "init": {"log_T": 0.0, "b": 0.0},
        },
        "source_dev_c": source_dev_c,
        "target_dev_c": target_dev_c,
        "confirm_excluded": True,
    }
    t_write0 = time.perf_counter()
    cal_ref = publish_phase2(
        root,
        protocol,
        plan["dataset_fingerprint"],
        f"{E17_CALIBRATORS_NAMESPACE}/{model}",
        KIND_CALIBRATORS,
        cal_body,
        producer,
        upstream=upstream_refs,
        run_id=run_id,
    )
    write_s = time.perf_counter() - t_write0
    _emit(
        progress,
        _progress_event(
            "progress",
            "calibrators",
            completed_base + 2,
            total_units,
            compute_s=compute_s,
            write_s=write_s,
            extra={"ref": plain(cal_ref)},
        ),
    )
    return plain(cal_ref), plain(cal_body)


def _apply_calibrator(logits, cal_params):
    """Apply float64 z' = z / T + b."""
    import numpy as np

    z = np.asarray(logits, dtype=np.float64)
    t_val = float(cal_params["T"])
    b_val = float(cal_params["b"])
    check(np.isfinite(t_val) and t_val > 0.0, f"Invalid temperature T={t_val!r}")
    check(np.isfinite(b_val), f"Invalid intercept b={b_val!r}")
    return z / t_val + b_val


def _select_arm_calibrator(arm, source_lang, target_lang, cal_body):
    """Return (cal_params_or_None, supervision_label) for an arm and route."""
    if arm == "raw":
        label = (
            E17_SUPERVISION_LABELS["within_context_source_fit"]
            if source_lang == target_lang
            else E17_SUPERVISION_LABELS["source_only_transfer"]
        )
        return None, label
    scope, form = arm.split(":", 1)
    if scope == "source_dev_c":
        label = (
            E17_SUPERVISION_LABELS["within_context_source_fit"]
            if source_lang == target_lang
            else E17_SUPERVISION_LABELS["source_dev_c_calibrated"]
        )
        return cal_body["source_dev_c"][source_lang][form], label
    if scope == "target_dev_c":
        if source_lang == target_lang:
            return cal_body["source_dev_c"][source_lang][form], E17_SUPERVISION_LABELS["within_context_source_fit"]
        return (
            cal_body["target_dev_c"][f"{source_lang}->{target_lang}"][form],
            E17_SUPERVISION_LABELS["target_dev_c_supervised_calibration"],
        )
    raise ContractError(f"Unknown E17 arm: {arm!r}")


def _summarize_draws(point_val, draw_vals):
    """Summarize scalar point estimate and 1D bootstrap draw vector into JSON-safe dict."""
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


def _evaluate_screen(
    plan,
    auth_ref,
    inv,
    model,
    cal_ref,
    cal_body,
    dev_c_block_refs,
    producer,
    *,
    run_id,
    progress=None,
    completed_base=7,
    total_units=10,
):
    """Evaluate all 5 arms on reused Stage 4 SCREEN logits with paired bootstrap & invariances."""
    import numpy as np
    from src.stage4.metrics import (
        METRICS,
        binary_metrics,
        bootstrap_binary_metrics,
        bootstrap_counts,
        fixed_macro,
        paired_image_alignment,
        probabilities,
        support_status,
    )
    from src.stage4.plan import load_bootstrap
    from src.stage4.predictions import load_prediction_block

    t_read0 = time.perf_counter()
    stage4_plan = plan.binding.stage4_plan
    protocol = plain(plan.protocol)
    categories = list(protocol["category_order"])
    n_cats = len(categories)
    primary_idx = _primary_indices(protocol)
    screen_min_support = int(protocol["minimum_positive_and_negative"]["SCREEN"])

    # Load all 4 Stage 4 SCREEN blocks (16 routes) and verify identical image alignment.
    screen_preds = {}
    all_pred_objs = []
    for target_lang in LANGUAGES:
        blk_info = inv["screen_blocks"][target_lang]
        blk_preds = load_prediction_block(
            stage4_plan, blk_info["prediction_ref"], expected_block=blk_info["block_id"]
        )
        for src_lang in LANGUAGES:
            pred = blk_preds[blk_info["routes"][src_lang]]
            screen_preds[(src_lang, target_lang)] = pred
            all_pred_objs.append(pred)

    screen_ids, aligned_indices = paired_image_alignment(stage4_plan, *all_pred_objs)
    n_screen = len(screen_ids)
    # Reorder logits/rows to canonical sorted screen_ids order if needed (Stage 4 is already sorted).
    aligned_map = {}
    for (src_lang, target_lang), idx_arr in zip(
        [(s, t) for t in LANGUAGES for s in LANGUAGES], aligned_indices
    ):
        aligned_map[(src_lang, target_lang)] = idx_arr

    # Build canonical SCREEN target (y, known) aligned to screen_ids.
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
    counts = bootstrap_counts(draws, n_screen)
    read_s = time.perf_counter() - t_read0

    t_comp0 = time.perf_counter()
    n_draws = int(draws.shape[0])

    # Precompute per-route, per-arm, per-category point & bootstrap draw arrays.
    # Also enforce AUROC/AP invariance and temp-only 0.5-decision invariance.
    max_abs_auroc_diff = 0.0
    max_abs_ap_diff = 0.0
    temp_only_0p5_flips = 0

    all_t_positive = True
    for scope_name in ("source_dev_c", "target_dev_c"):
        for _, forms in cal_body[scope_name].items():
            for form_name in ("temp", "temp_intercept"):
                if not (float(forms[form_name]["T"]) > 0.0):
                    all_t_positive = False
    check(all_t_positive, "All fitted calibrator temperatures T must be strictly positive")

    # Storage: arm -> route (s, t) -> metric -> (point_cats [25], draws_cats [2000, 25])
    route_arm_metrics = {arm: {} for arm in E17_ARMS}
    route_fit_prior = {}
    route_flip_counts = {arm: {} for arm in E17_ARMS}
    route_supervision = {arm: {} for arm in E17_ARMS}

    for target_lang in LANGUAGES:
        for source_lang in LANGUAGES:
            authorize_e17_operation(
                split="SCREEN",
                operation="evaluate_screen",
                source_lang=source_lang,
                target_lang=target_lang,
                head_source_lang=source_lang,
            )
            pred = screen_preds[(source_lang, target_lang)]
            idx_arr = aligned_map[(source_lang, target_lang)]
            raw_logits = np.asarray(pred.logits[idx_arr], dtype=np.float64)
            raw_probs = probabilities(raw_logits)
            raw_dec = raw_probs >= 0.5

            # Fixed FIT-prior reference for this source language.
            prior_probs = np.broadcast_to(
                np.asarray(pred.prevalence, dtype=np.float64)[None, :], (n_screen, n_cats)
            )
            fp_pts = {m: np.full(n_cats, np.nan, dtype=np.float64) for m in METRICS}
            fp_draws = {m: np.full((n_draws, n_cats), np.nan, dtype=np.float64) for m in METRICS}
            for k in range(n_cats):
                pt = binary_metrics(y_screen[:, k], prior_probs[:, k], known_screen[:, k])["metrics"]
                dr = bootstrap_binary_metrics(y_screen[:, k], prior_probs[:, k], known_screen[:, k], counts)
                for m in METRICS:
                    if pt[m] is not None:
                        fp_pts[m][k] = pt[m]
                    fp_draws[m][:, k] = dr[m]
            route_fit_prior[(source_lang, target_lang)] = (fp_pts, fp_draws)

            for arm in E17_ARMS:
                cal_params, sup_label = _select_arm_calibrator(arm, source_lang, target_lang, cal_body)
                route_supervision[arm][(source_lang, target_lang)] = sup_label
                if cal_params is None:
                    arm_logits = raw_logits
                else:
                    arm_logits = _apply_calibrator(raw_logits, cal_params)
                arm_probs = probabilities(arm_logits)
                arm_dec = arm_probs >= 0.5

                flips = int(np.sum((arm_dec != raw_dec) & known_screen))
                known_total = int(np.sum(known_screen))
                if arm.endswith(":temp"):
                    temp_only_0p5_flips += flips
                route_flip_counts[arm][(source_lang, target_lang)] = {
                    "flips": flips,
                    "known_cells": known_total,
                    "flip_rate": float(flips / known_total) if known_total > 0 else 0.0,
                }

                pts = {m: np.full(n_cats, np.nan, dtype=np.float64) for m in METRICS}
                drs = {m: np.full((n_draws, n_cats), np.nan, dtype=np.float64) for m in METRICS}
                for k in range(n_cats):
                    pt = binary_metrics(y_screen[:, k], arm_probs[:, k], known_screen[:, k])["metrics"]
                    dr = bootstrap_binary_metrics(y_screen[:, k], arm_probs[:, k], known_screen[:, k], counts)
                    for m in METRICS:
                        if pt[m] is not None:
                            pts[m][k] = pt[m]
                        drs[m][:, k] = dr[m]

                if arm != "raw":
                    raw_pts, _ = route_arm_metrics["raw"][(source_lang, target_lang)]
                    for k in range(n_cats):
                        for inv_m in ("auroc", "ap"):
                            rv = raw_pts[inv_m][k]
                            av = pts[inv_m][k]
                            if np.isnan(rv) and np.isnan(av):
                                continue
                            check(
                                np.isfinite(rv) and np.isfinite(av),
                                f"Invariance NaN mismatch on {source_lang}->{target_lang} {arm} {categories[k]} {inv_m}",
                            )
                            diff = abs(float(av) - float(rv))
                            if inv_m == "auroc":
                                max_abs_auroc_diff = max(max_abs_auroc_diff, diff)
                            else:
                                max_abs_ap_diff = max(max_abs_ap_diff, diff)
                            check(
                                diff <= E17_INVARIANCE_TOL,
                                f"E17 {inv_m} invariance violated on {source_lang}->{target_lang} {arm} "
                                f"{categories[k]}: diff={diff!r} > {E17_INVARIANCE_TOL}",
                            )

                route_arm_metrics[arm][(source_lang, target_lang)] = (pts, drs)

    check(
        temp_only_0p5_flips == 0,
        f"temp-only calibrators produced {temp_only_0p5_flips} 0.5-decision flips (expected 0)",
    )

    # Category support on SCREEN (shared across all routes).
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

    off_diag_routes = [(s, t) for t in LANGUAGES for s in LANGUAGES if s != t]
    diag_routes = [(t, t) for t in LANGUAGES]
    eval_metrics = ("brier", "nll", "auroc", "ap")
    dec_metrics = ("sensitivity", "specificity", "f1", "balanced_accuracy")

    def _macro_point_and_draws(pts_arr, drs_arr):
        pt_m = float(fixed_macro(pts_arr, primary_idx, axis=0))
        dr_m = fixed_macro(drs_arr, primary_idx, axis=1)
        return pt_m, dr_m

    # Build summaries for proper-loss + rank metrics and separate decision_0p5.
    summaries_by_arm = {}
    decision_0p5_by_arm = {}

    for arm in E17_ARMS:
        arm_routes_summary = []
        arm_routes_dec = []

        # Collect per-route 14-primary macro arrays for within-draw 12-route and 4-diagonal means.
        off_macros = {m: [] for m in eval_metrics + dec_metrics}
        off_raw_gains = {m: [] for m in ("brier", "nll")}
        off_fp_gains = {m: [] for m in ("brier", "nll")}
        diag_macros = {m: [] for m in eval_metrics + dec_metrics}
        diag_raw_gains = {m: [] for m in ("brier", "nll")}
        diag_fp_gains = {m: [] for m in ("brier", "nll")}

        for s_lang, t_lang in [(s, t) for t in LANGUAGES for s in LANGUAGES]:
            pts, drs = route_arm_metrics[arm][(s_lang, t_lang)]
            raw_pts, raw_drs = route_arm_metrics["raw"][(s_lang, t_lang)]
            fp_pts, fp_drs = route_fit_prior[(s_lang, t_lang)]

            row_eval = {
                "route": f"{s_lang}->{t_lang}",
                "source_language": s_lang,
                "target_language": t_lang,
                "is_diagonal": s_lang == t_lang,
                "supervision_label": route_supervision[arm][(s_lang, t_lang)],
                "primary_macro": {},
                "gain_vs_raw": {},
                "gain_vs_fit_prior": {},
                "per_category": {},
            }
            for m in eval_metrics:
                pt_m, dr_m = _macro_point_and_draws(pts[m], drs[m])
                row_eval["primary_macro"][m] = _summarize_draws(pt_m, dr_m)
                row_eval["per_category"][m] = [
                    _summarize_draws(pts[m][k], drs[m][:, k]) for k in range(n_cats)
                ]
                if m in ("brier", "nll"):
                    raw_pt_m, raw_dr_m = _macro_point_and_draws(raw_pts[m], raw_drs[m])
                    fp_pt_m, fp_dr_m = _macro_point_and_draws(fp_pts[m], fp_drs[m])
                    gain_raw_pt = raw_pt_m - pt_m
                    gain_raw_dr = raw_dr_m - dr_m
                    gain_fp_pt = fp_pt_m - pt_m
                    gain_fp_dr = fp_dr_m - dr_m
                    row_eval["gain_vs_raw"][m] = _summarize_draws(gain_raw_pt, gain_raw_dr)
                    row_eval["gain_vs_fit_prior"][m] = _summarize_draws(gain_fp_pt, gain_fp_dr)
                    if s_lang != t_lang:
                        off_raw_gains[m].append((gain_raw_pt, gain_raw_dr))
                        off_fp_gains[m].append((gain_fp_pt, gain_fp_dr))
                    else:
                        diag_raw_gains[m].append((gain_raw_pt, gain_raw_dr))
                        diag_fp_gains[m].append((gain_fp_pt, gain_fp_dr))
                if s_lang != t_lang:
                    off_macros[m].append((pt_m, dr_m))
                else:
                    diag_macros[m].append((pt_m, dr_m))
            arm_routes_summary.append(row_eval)

            row_dec = {
                "route": f"{s_lang}->{t_lang}",
                "source_language": s_lang,
                "target_language": t_lang,
                "is_diagonal": s_lang == t_lang,
                "supervision_label": route_supervision[arm][(s_lang, t_lang)],
                "primary_macro": {},
                "flip_vs_raw": route_flip_counts[arm][(s_lang, t_lang)],
            }
            for m in dec_metrics:
                pt_m, dr_m = _macro_point_and_draws(pts[m], drs[m])
                row_dec["primary_macro"][m] = _summarize_draws(pt_m, dr_m)
                if s_lang != t_lang:
                    off_macros[m].append((pt_m, dr_m))
                else:
                    diag_macros[m].append((pt_m, dr_m))
            arm_routes_dec.append(row_dec)

        def _agg_mean(pairs):
            pt_mean = float(np.mean([p for p, _ in pairs]))
            dr_mean = np.mean(np.stack([d for _, d in pairs], axis=0), axis=0)
            return _summarize_draws(pt_mean, dr_mean)

        off_flips = sum(route_flip_counts[arm][r]["flips"] for r in off_diag_routes)
        off_known = sum(route_flip_counts[arm][r]["known_cells"] for r in off_diag_routes)
        diag_flips = sum(route_flip_counts[arm][r]["flips"] for r in diag_routes)
        diag_known = sum(route_flip_counts[arm][r]["known_cells"] for r in diag_routes)

        summaries_by_arm[arm] = {
            "off_diagonal_12_route_mean": {
                "primary_macro": {m: _agg_mean(off_macros[m]) for m in eval_metrics},
                "gain_vs_raw": {m: _agg_mean(off_raw_gains[m]) for m in ("brier", "nll")},
                "gain_vs_fit_prior": {m: _agg_mean(off_fp_gains[m]) for m in ("brier", "nll")},
            },
            "diagonal_4_route_mean": {
                "primary_macro": {m: _agg_mean(diag_macros[m]) for m in eval_metrics},
                "gain_vs_raw": {m: _agg_mean(diag_raw_gains[m]) for m in ("brier", "nll")},
                "gain_vs_fit_prior": {m: _agg_mean(diag_fp_gains[m]) for m in ("brier", "nll")},
            },
            "routes": arm_routes_summary,
        }
        decision_0p5_by_arm[arm] = {
            "off_diagonal_12_route_mean": {
                "primary_macro": {m: _agg_mean(off_macros[m]) for m in dec_metrics},
                "flips": off_flips,
                "known_cells": off_known,
                "flip_rate": float(off_flips / off_known) if off_known > 0 else 0.0,
            },
            "diagonal_4_route_mean": {
                "primary_macro": {m: _agg_mean(diag_macros[m]) for m in dec_metrics},
                "flips": diag_flips,
                "known_cells": diag_known,
                "flip_rate": float(diag_flips / diag_known) if diag_known > 0 else 0.0,
            },
            "routes": arm_routes_dec,
        }

    # Summarize FIT-prior reference (12-route mean, 4-diagonal mean, per-route).
    fp_off = {m: [] for m in ("brier", "nll")}
    fp_diag = {m: [] for m in ("brier", "nll")}
    fp_routes = []
    for s_lang, t_lang in [(s, t) for t in LANGUAGES for s in LANGUAGES]:
        fp_pts, fp_drs = route_fit_prior[(s_lang, t_lang)]
        row_fp = {"route": f"{s_lang}->{t_lang}", "primary_macro": {}}
        for m in ("brier", "nll"):
            pt_m, dr_m = _macro_point_and_draws(fp_pts[m], fp_drs[m])
            row_fp["primary_macro"][m] = _summarize_draws(pt_m, dr_m)
            if s_lang != t_lang:
                fp_off[m].append((pt_m, dr_m))
            else:
                fp_diag[m].append((pt_m, dr_m))
        fp_routes.append(row_fp)

    fit_prior_reference = {
        "off_diagonal_12_route_mean": {
            m: _summarize_draws(
                float(np.mean([p for p, _ in fp_off[m]])),
                np.mean(np.stack([d for _, d in fp_off[m]], axis=0), axis=0),
            )
            for m in ("brier", "nll")
        },
        "diagonal_4_route_mean": {
            m: _summarize_draws(
                float(np.mean([p for p, _ in fp_diag[m]])),
                np.mean(np.stack([d for _, d in fp_diag[m]], axis=0), axis=0),
            )
            for m in ("brier", "nll")
        },
        "routes": fp_routes,
    }
    compute_s = time.perf_counter() - t_comp0

    screen_block_refs = [plain(inv["screen_blocks"][t_lang]["prediction_ref"]) for t_lang in LANGUAGES]
    upstream_refs = [
        plain(plan.reference),
        plain(auth_ref),
        plain(cal_ref),
        *[plain(dev_c_block_refs[t_lang]) for t_lang in LANGUAGES],
        *screen_block_refs,
    ]
    analysis_body = {
        "schema": SCHEMA_ANALYSIS,
        "experiment": "E17",
        "plan": plain(plan.reference),
        "plan_identity": plan["plan_identity"],
        "e17_auth_ref": plain(auth_ref),
        "calibrators_ref": plain(cal_ref),
        "dev_c_block_refs": plain(dev_c_block_refs),
        "screen_block_refs": screen_block_refs,
        "model": model,
        "midpoint_depth": int(inv["midpoint_depth"]),
        "bootstrap": plain(plan["bootstrap"]),
        "arms": list(E17_ARMS),
        "category_order": categories,
        "primary_categories": list(protocol["primary_categories"]),
        "category_support": cat_support,
        "fit_prior_reference": fit_prior_reference,
        "summaries": summaries_by_arm,
        "decision_0p5": decision_0p5_by_arm,
        "invariance_checks": {
            "max_abs_auroc_diff": float(max_abs_auroc_diff),
            "max_abs_ap_diff": float(max_abs_ap_diff),
            "tolerance": E17_INVARIANCE_TOL,
            "all_T_positive": all_t_positive,
            "temp_only_0p5_flips": temp_only_0p5_flips,
        },
        "e19_cutoffs_untouched": True,
        "confirm_excluded": True,
    }
    t_write0 = time.perf_counter()
    analysis_ref = publish_phase2(
        plan.root,
        protocol,
        plan["dataset_fingerprint"],
        f"{E17_RESULTS_NAMESPACE}/{model}",
        KIND_ANALYSIS,
        analysis_body,
        producer,
        upstream=upstream_refs,
        run_id=run_id,
    )
    write_s = time.perf_counter() - t_write0
    _emit(
        progress,
        _progress_event(
            "progress",
            "screen_evaluation",
            total_units,
            total_units,
            read_hash_s=read_s,
            compute_s=compute_s,
            write_s=write_s,
            extra={"ref": plain(analysis_ref)},
        ),
    )
    return plain(analysis_ref), plain(analysis_body)


@contract
def run_e17(
    phase1_root_dir,
    protocol,
    plan_ref,
    *,
    model,
    e17_auth_ref=None,
    producer=None,
    storage_scope="local-development",
    run_id=None,
    resume_dev_c_refs=None,
    resume_calibrators_ref=None,
    resume_analysis_ref=None,
    progress=None,
):
    """Run the end-to-end CPU E17 worker for one MODEL ('qwen' or 'gemma')."""
    check(model in MODELS, f"E17 requires MODEL in {MODELS!r}; got {model!r}")
    t_init0 = time.perf_counter()
    plan = load_phase2_plan(phase1_root_dir, protocol, plan_ref)
    if plan.binding is None:
        binding = bind_phase1(plan.root, plain(plan.protocol), plain(plan["stage4_release"]))
        plan = Phase2Plan(plan._body, plan.reference, plan.root, plain(plan.protocol), binding=binding)
    resolved_auth_ref = e17_auth_ref if e17_auth_ref is not None else plan["e17_auth"]
    check(
        resolved_auth_ref is not None,
        "E17 requires an explicit phase2_e17_auth reference (or phase2_plan['e17_auth'])",
    )
    if plan["e17_auth"] is not None and e17_auth_ref is not None:
        check(
            plain(normalize_ref(plan.root, e17_auth_ref)) == plain(normalize_ref(plan.root, plan["e17_auth"])),
            "Explicit e17_auth_ref does not match phase2_plan['e17_auth']",
        )

    auth_ref, auth_body, _ = load_e17_auth(
        plan.root, plain(plan.protocol), resolved_auth_ref, binding=plan.binding, models=(model,)
    )
    check(auth_body["permission"] == E17_DEV_C_PERMISSION, "Invalid E17 DEV-C permission")
    check(auth_body["lineage_identity"] == plan["lineage_identity"], "E17 auth lineage_identity mismatch")
    check(auth_body["confirm_excluded"] is True, "CONFIRM must remain excluded")

    inv = enumerate_e17_model_inventory(plan.binding, model)
    prod = current_producer(scope=storage_scope) if producer is None else producer
    effective_run_id = run_id or new_run_id()
    total_units = 8

    _emit(
        progress,
        _progress_event(
            "start",
            "init",
            0,
            total_units,
            read_hash_s=time.perf_counter() - t_init0,
            extra={"model": model, "run_id": effective_run_id},
        ),
    )

    if resume_analysis_ref is not None:
        t_res0 = time.perf_counter()
        res_ref, res_body, _ = verify_resume_ref(
            plan.root,
            plain(plan.protocol),
            resume_analysis_ref,
            KIND_ANALYSIS,
            expected_schema=SCHEMA_ANALYSIS,
            expected_dataset=plan["dataset_fingerprint"],
            expected_plan_identity=plan["plan_identity"],
            expected_owner=model,
        )
        check(res_body.get("experiment") == "E17", "Resume analysis experiment mismatch")
        check(plain(res_body.get("e17_auth_ref")) == plain(auth_ref), "Resume analysis e17_auth_ref mismatch")
        invariance = res_body.get("invariance_checks", {})
        check(
            invariance.get("all_T_positive") is True
            and invariance.get("temp_only_0p5_flips") == 0
            and float(invariance.get("max_abs_auroc_diff", 1.0)) <= E17_INVARIANCE_TOL
            and float(invariance.get("max_abs_ap_diff", 1.0)) <= E17_INVARIANCE_TOL,
            "Resume analysis failed invariance checks",
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
            "dev_c_block_refs": plain(res_body["dev_c_block_refs"]),
            "calibrators_ref": plain(res_body["calibrators_ref"]),
            "analysis_ref": plain(res_ref),
            "resumed": True,
        })

    dev_c_block_refs, dev_c_logits_by_route, dev_c_rows_by_target, diag_check = _score_or_resume_dev_c_blocks(
        plan,
        auth_ref,
        inv,
        model,
        prod,
        run_id=effective_run_id,
        resume_dev_c_refs=resume_dev_c_refs,
        progress=progress,
        completed_base=1,
        total_units=total_units,
    )

    cal_ref, cal_body = _fit_or_resume_calibrators(
        plan,
        auth_ref,
        inv,
        model,
        prod,
        dev_c_block_refs,
        dev_c_logits_by_route,
        dev_c_rows_by_target,
        run_id=effective_run_id,
        resume_calibrators_ref=resume_calibrators_ref,
        progress=progress,
        completed_base=5,
        total_units=total_units,
    )

    analysis_ref, _ = _evaluate_screen(
        plan,
        auth_ref,
        inv,
        model,
        cal_ref,
        cal_body,
        dev_c_block_refs,
        prod,
        run_id=effective_run_id,
        progress=progress,
        completed_base=7,
        total_units=total_units,
    )

    result = {
        "model": model,
        "run_id": effective_run_id,
        "diagonal_self_check": diag_check,
        "dev_c_block_refs": dev_c_block_refs,
        "calibrators_ref": cal_ref,
        "analysis_ref": analysis_ref,
        "resumed": False,
    }
    _emit(
        progress,
        _progress_event(
            "complete",
            "e17_complete",
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
    check(path.is_file(), f"Missing resume DEV-C refs JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _cli(argv=None):
    parser = argparse.ArgumentParser(description="Phase 2 Stage 2 CPU E17 worker")
    parser.add_argument("--phase1-root", type=Path, default=None)
    parser.add_argument("--protocol", type=Path, default=Path("configs/protocol.json"))
    parser.add_argument("--plan-ref", required=True)
    parser.add_argument("--e17-auth-ref", default=None)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--storage-scope", default="local-development")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume-dev-c-refs", default=None)
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
            run_e17(
                resolved_root,
                protocol,
                _parse_ref_arg(args.plan_ref),
                model=args.model,
                e17_auth_ref=_parse_ref_arg(args.e17_auth_ref),
                storage_scope=args.storage_scope,
                run_id=args.run_id,
                resume_dev_c_refs=_parse_ref_map_arg(args.resume_dev_c_refs),
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
