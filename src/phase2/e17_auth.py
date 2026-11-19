"""Scoped E17 DEV-C transfer permission layer, CPU preflight, and authorization publisher.

Implements Phase 2 Stage 1 §1.6:
- Scoped `phase2.e17_dev_c_transfer` permission over accepted Stage 2 midpoint
  `image_caption` / `last_caption` DEV-C feature releases (Qwen depth 18, Gemma depth 17).
- Exact per-model inventory:
  - 4 reused accepted Stage 4 DEV-C source diagonals (`s == t`, read-only, never duplicated),
  - 4 newly authorized target-context DEV-C score blocks per model (3 off-diagonal
    frozen-source-head transfer routes `s != t` per block; 12/model, 24 total),
  - 4 reused accepted Stage 4 SCREEN blocks per model (16 routes, read-only).
- CPU preflight verifying without opening FIT feature matrices or CONFIRM:
  (a) all 8 Stage 2 DEV-C midpoint feature releases authenticate cleanly;
  (b) all 8 Stage 3 selected ordinary heads + source-FIT scalers load cleanly via
      `_selected_parameters`;
  (c) in-memory `en -> en` DEV-C diagonal scoring matches Stage 4's accepted
      `en -> en` DEV-C logits to `atol=0, rtol=0` for each model.
- Hard guardrails rejecting target-trained head misuse, unauthorized splits
  (CONFIRM, FIT, DEV-S, non-DEV-C scoring), base-head/scaler refitting, SCREEN tuning,
  and Stage 4 E19 cutoff modification.
"""

import argparse
import json
import math
from pathlib import Path
import sys
import time

from src.common.contracts import ContractError, LANGUAGES, fingerprint
from src.common.storage import phase1_root
from src.phase2.interfaces import (
    KIND_E17_AUTH,
    MODELS,
    PHASE1_STAGE4_RELEASE,
    PHASE2_VERSION,
    REPO,
    SCHEMA_E17_AUTH,
    SUPERVISION_LABELS,
    Phase1Binding,
    bind_phase1,
    check,
    contract,
    current_producer,
    e17_inputs,
    frozen,
    init_phase2_dirs,
    load_frozen_source_head,
    load_phase2_artifact,
    plain,
    publish_phase2,
    read_verified_midpoint_features,
    require_open_split,
    verify_resume_ref,
)

E17_DEV_C_PERMISSION = "phase2.e17_dev_c_transfer"
E17_CONDITION = "image_caption"
E17_POOLING = "last_caption"
E17_DEV_C_SPLIT = "DEV-C"
E17_AUTH_NAMESPACE = "phase2/plans"
E17_DEV_C_PREDICTIONS_NAMESPACE = "phase2/predictions/e17_dev_c"
E17_CALIBRATORS_NAMESPACE = "phase2/calibrators/e17"

E17_CALIBRATOR_FORMS = ("raw", "temp", "temp_intercept")
E17_CALIBRATOR_SCOPES = ("source_dev_c", "target_dev_c")
E17_INVARIANCE_TOL = 1e-12
E17_BFGS_GTOL = 1e-10
E17_BFGS_MAXITER = 500

E17_DEV_C_PURPOSES = (
    "score_off_diagonal",
    "diagonal_self_check",
    "fit_source_calibrator",
    "fit_target_calibrator",
)
E17_ALLOWED_OPERATIONS = E17_DEV_C_PURPOSES + ("evaluate_screen",)
E17_FORBIDDEN_OPERATIONS = (
    "refit_head",
    "fit_base_head",
    "fit_head",
    "refit_scaler",
    "fit_scaler",
    "reselect_c",
    "select_c",
    "tune_on_screen",
    "fit_on_screen",
    "modify_e19_cutoffs",
    "modify_e19",
    "score_confirm",
)

E17_SUPERVISION_LABELS = frozen({
    "within_context_source_fit": "within_context_source_fit",
    "source_only_transfer": "source_only_transfer",
    "source_dev_c_calibrated": "source_dev_c_calibrated",
    "target_dev_c_supervised_calibration": "target_dev_c_supervised_calibration",
})


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


def _normalize_models(models):
    if isinstance(models, str):
        if models == "both":
            return MODELS
        check(models in MODELS, f"Unknown E17 model selector: {models!r}")
        return (models,)
    seq = tuple(models)
    check(bool(seq), "At least one model must be selected for E17 preflight")
    seen = set()
    for model in seq:
        check(model in MODELS, f"Unknown E17 model selector: {model!r}")
        check(model not in seen, f"Duplicate model selector: {model!r}")
        seen.add(model)
    return seq


def _resolve_binding(bound_or_plan):
    if isinstance(bound_or_plan, Phase1Binding):
        return bound_or_plan
    binding = getattr(bound_or_plan, "binding", None)
    if isinstance(binding, Phase1Binding):
        return binding
    raise ContractError("Expected Phase1Binding or a Phase2Plan exposing .binding")


def e17_calibrator_recipe(protocol):
    """Return the frozen float64 E17 calibrator recipe defined in phase2.md §1.6."""
    primary = tuple(protocol["primary_categories"])
    check(len(primary) == 14, f"Expected 14 fixed primary categories; got {len(primary)}")
    for label in E17_SUPERVISION_LABELS.values():
        check(label in SUPERVISION_LABELS, f"Unknown supervision label: {label!r}")
    return frozen({
        "dtype": "float64",
        "forms": {
            "raw": "z",
            "temp": "z / exp(log_T)",
            "temp_intercept": "z / exp(log_T) + b",
        },
        "form_names": list(E17_CALIBRATOR_FORMS),
        "scopes": {
            "source_dev_c": "Reused Stage 4 s->s DEV-C diagonal logits (read-only)",
            "target_dev_c": (
                "Newly authorized frozen source s head scored on target t DEV-C features "
                "(s != t; coincides with source_dev_c when s == t and labeled once)"
            ),
        },
        "objective": {
            "name": "equal_weight_primary_macro_bce",
            "split": E17_DEV_C_SPLIT,
            "primary_category_count": 14,
            "primary_categories": list(primary),
            "known_cells_only": True,
            "prevalence_weighting": "natural_within_category",
            "formula": (
                "L = (1/14) * sum_{k in primary} mean_{i in known(DEV-C, k)} "
                "[logaddexp(0, z'_{ik}) - y_{ik} * z'_{ik}]"
            ),
        },
        "optimizer": {
            "library": "scipy.optimize.minimize",
            "method": "BFGS",
            "jac": True,
            "analytic_gradient": True,
            "gtol": E17_BFGS_GTOL,
            "maxiter": E17_BFGS_MAXITER,
            "init": {"log_T": 0.0, "b": 0.0},
            "deterministic": True,
            "require_converged": True,
            "require_finite": True,
            "require_positive_temperature": True,
        },
        "invariance_tol": E17_INVARIANCE_TOL,
        "diagonal_self_check": {"route": "en->en", "atol": 0.0, "rtol": 0.0},
        "supervision_labels": plain(E17_SUPERVISION_LABELS),
        "calibrator_namespace": E17_CALIBRATORS_NAMESPACE,
    })


@contract
def require_e17_dev_c_split(split):
    """Enforce that E17 DEV-C transfer permission accepts ONLY DEV-C."""
    require_open_split(split)
    check(
        split == E17_DEV_C_SPLIT,
        f"Permission {E17_DEV_C_PERMISSION!r} authorizes only {E17_DEV_C_SPLIT!r}; rejected split {split!r}",
    )
    return split


@contract
def authorize_e17_dev_c_route(
    source_lang,
    target_lang,
    *,
    head_source_lang=None,
    split=E17_DEV_C_SPLIT,
    purpose="score_off_diagonal",
    allow_diagonal_self_check=False,
):
    """Enforce scoped `phase2.e17_dev_c_transfer` guardrails for a DEV-C route."""
    require_e17_dev_c_split(split)
    check(
        purpose not in E17_FORBIDDEN_OPERATIONS and purpose in E17_DEV_C_PURPOSES,
        f"Forbidden or unauthorized E17 DEV-C purpose: {purpose!r}",
    )
    check(source_lang in LANGUAGES, f"Unknown source language: {source_lang!r}")
    check(target_lang in LANGUAGES, f"Unknown target language: {target_lang!r}")

    effective_head = source_lang if head_source_lang is None else head_source_lang
    check(effective_head in LANGUAGES, f"Unknown head source language: {effective_head!r}")

    if source_lang != target_lang and effective_head == target_lang:
        raise ContractError(
            f"Target calibrator/scoring for route {source_lang}->{target_lang} must consume "
            f"frozen source head {source_lang!r}, never target {target_lang!r}'s own trained head"
        )
    check(
        effective_head == source_lang,
        f"Route {source_lang}->{target_lang} requires frozen source head {source_lang!r}; got {effective_head!r}",
    )

    if purpose == "score_off_diagonal":
        check(
            source_lang != target_lang,
            f"Stage 4 DEV-C diagonal {source_lang}->{target_lang} is reused read-only and must never be duplicated",
        )
    elif purpose == "diagonal_self_check":
        check(
            allow_diagonal_self_check and source_lang == "en" and target_lang == "en",
            "In-memory diagonal self-check is restricted to en->en with allow_diagonal_self_check=True",
        )
    elif purpose == "fit_target_calibrator":
        check(
            source_lang != target_lang,
            "Target DEV-C calibrator applies to off-diagonal routes s != t (s == t coincides with source_dev_c)",
        )
    elif purpose == "fit_source_calibrator":
        check(
            source_lang == target_lang,
            "Source DEV-C calibrator must fit on reused Stage 4 source diagonal s == t logits",
        )

    return frozen({
        "permission": E17_DEV_C_PERMISSION,
        "split": split,
        "purpose": purpose,
        "source_language": source_lang,
        "target_language": target_lang,
        "head_source_language": effective_head,
    })


@contract
def authorize_e17_operation(
    *,
    split,
    operation,
    source_lang=None,
    target_lang=None,
    head_source_lang=None,
    modifies_e19=False,
    refits_head=False,
    refits_scaler=False,
    tunes_on_screen=False,
    allow_diagonal_self_check=False,
):
    """Central hard-guardrail gate for every E17 operation."""
    require_open_split(split)
    check(not modifies_e19, "E17 must never modify Stage 4 E19 cutoffs or claims")
    check(not refits_head, "E17 must never fit or refit base heads")
    check(not refits_scaler, "E17 must never fit or refit feature scalers")
    check(not tunes_on_screen, "E17 must never tune calibrators on SCREEN")
    check(
        operation not in E17_FORBIDDEN_OPERATIONS and operation in E17_ALLOWED_OPERATIONS,
        f"Forbidden or unauthorized E17 operation: {operation!r}",
    )

    if operation == "evaluate_screen":
        check(split == "SCREEN", f"Read-only SCREEN evaluation requires split='SCREEN'; got {split!r}")
        if source_lang is not None:
            check(source_lang in LANGUAGES, f"Unknown source language: {source_lang!r}")
        if target_lang is not None:
            check(target_lang in LANGUAGES, f"Unknown target language: {target_lang!r}")
        if head_source_lang is not None and source_lang is not None:
            check(
                head_source_lang == source_lang,
                f"SCREEN evaluation for {source_lang}->{target_lang} must use frozen source head {source_lang!r}",
            )
        return frozen({"permission": E17_DEV_C_PERMISSION, "split": "SCREEN", "operation": operation, "read_only": True})

    return authorize_e17_dev_c_route(
        source_lang,
        target_lang,
        head_source_lang=head_source_lang,
        split=split,
        purpose=operation,
        allow_diagonal_self_check=allow_diagonal_self_check,
    )


@contract
def validate_target_calibrator_input(
    source_lang,
    target_lang,
    *,
    logits_source_group,
    source_group_id,
    target_group_id,
    split=E17_DEV_C_SPLIT,
):
    """Verify at group-ID level that target DEV-C calibration uses frozen source s head, never target t head."""
    authorize_e17_dev_c_route(
        source_lang,
        target_lang,
        head_source_lang=source_lang,
        split=split,
        purpose="fit_target_calibrator",
    )
    check(source_group_id != target_group_id, "Off-diagonal route must have distinct source and target groups")
    if logits_source_group == target_group_id:
        raise ContractError(
            f"Target calibrator for {source_lang}->{target_lang} cannot consume target {target_lang!r}'s own trained head"
        )
    check(
        logits_source_group == source_group_id,
        f"Target calibrator for {source_lang}->{target_lang} requires frozen source group {source_group_id!r}; "
        f"got {logits_source_group!r}",
    )
    return True


@contract
def enumerate_e17_model_inventory(binding, model):
    """Enumerate the exact E17 DEV-C and SCREEN score inventory for one model."""
    check(model in MODELS, f"Unknown model selector: {model!r}")
    raw = e17_inputs(binding, model)
    check(
        raw["condition"] == E17_CONDITION and raw["pooling"] == E17_POOLING,
        "E17 inventory requires fixed midpoint image_caption / last_caption",
    )

    screen_blocks = {}
    screen_route_count = 0
    for target_lang in LANGUAGES:
        entry = plain(raw["screen_blocks"][target_lang])
        check(set(entry["routes"]) == set(LANGUAGES), f"Incomplete 4-source SCREEN block for {model}/{target_lang}")
        screen_route_count += len(entry["routes"])
        screen_blocks[target_lang] = {
            **entry,
            "split": "SCREEN",
            "reused_from": "stage4",
            "access": "read_only",
        }
    check(len(screen_blocks) == 4 and screen_route_count == 16, "Expected 4 reused SCREEN blocks (16 routes) per model")

    dev_c_diagonals = {}
    for lang in LANGUAGES:
        entry = plain(raw["dev_c_diagonals"][lang])
        dev_c_diagonals[lang] = {
            **entry,
            "source_language": lang,
            "target_language": lang,
            "split": E17_DEV_C_SPLIT,
            "reused_from": "stage4",
            "access": "read_only",
            "never_duplicate": True,
        }
    check(len(dev_c_diagonals) == 4, "Expected 4 reused Stage 4 DEV-C diagonals per model")

    source_systems = {lang: plain(raw["source_systems"][lang]) for lang in LANGUAGES}
    dev_c_feature_releases = {lang: plain(raw["dev_c_feature_releases"][lang]) for lang in LANGUAGES}

    off_diagonal_blocks = {}
    off_diagonal_route_count = 0
    for target_lang in LANGUAGES:
        target_release = dev_c_feature_releases[target_lang]
        target_gid = target_release["group_id"]
        expected_sources = tuple(raw["off_diagonal_targets"][target_lang])
        check(
            expected_sources == tuple(src for src in LANGUAGES if src != target_lang),
            f"Unexpected off-diagonal source set for {model}/{target_lang}",
        )
        block_id = fingerprint([
            SCHEMA_E17_AUTH,
            E17_DEV_C_PERMISSION,
            model,
            target_gid,
            E17_DEV_C_SPLIT,
            "off_diagonal",
        ])
        routes = []
        for source_lang in expected_sources:
            authorize_e17_dev_c_route(
                source_lang,
                target_lang,
                head_source_lang=source_lang,
                split=E17_DEV_C_SPLIT,
                purpose="score_off_diagonal",
            )
            src_sys = source_systems[source_lang]
            source_gid = src_sys["group_id"]
            check(source_gid != target_gid, "Off-diagonal route cannot reuse target group as source")
            route_id = fingerprint([
                SCHEMA_E17_AUTH,
                E17_DEV_C_PERMISSION,
                source_gid,
                target_gid,
                E17_DEV_C_SPLIT,
            ])
            routes.append({
                "route_id": route_id,
                "source_language": source_lang,
                "target_language": target_lang,
                "source_group": source_gid,
                "target_group": target_gid,
                "source_selection_ref": src_sys["selection_ref"],
                "source_candidate_ref": src_sys["candidate_ref"],
                "source_statistics_ref": src_sys["statistics_ref"],
                "C": src_sys["C"],
                "split": E17_DEV_C_SPLIT,
                "frozen_source_head_only": True,
                "supervision_label": E17_SUPERVISION_LABELS["target_dev_c_supervised_calibration"],
            })
        check(len(routes) == 3, f"Expected 3 off-diagonal routes for {model}/{target_lang}")
        off_diagonal_route_count += len(routes)
        off_diagonal_blocks[target_lang] = {
            "block_id": block_id,
            "owner": model,
            "permission": E17_DEV_C_PERMISSION,
            "split": E17_DEV_C_SPLIT,
            "condition": E17_CONDITION,
            "pooling": E17_POOLING,
            "midpoint_depth": int(raw["midpoint_depth"]),
            "target_language": target_lang,
            "target_group": target_gid,
            "target_release": target_release["release_ref"],
            "dimension": target_release["dimension"],
            "site": target_release["site"],
            "routes": routes,
        }
    check(
        len(off_diagonal_blocks) == 4 and off_diagonal_route_count == 12,
        "Expected 4 off-diagonal DEV-C blocks (12 routes) per model",
    )

    return frozen({
        "model": model,
        "midpoint_depth": int(raw["midpoint_depth"]),
        "condition": E17_CONDITION,
        "pooling": E17_POOLING,
        "screen_blocks": screen_blocks,
        "dev_c_diagonals": dev_c_diagonals,
        "dev_c_feature_releases": dev_c_feature_releases,
        "off_diagonal_blocks": off_diagonal_blocks,
        "source_systems": source_systems,
    })


@contract
def enumerate_e17_inventory(bound_or_plan, *, models=MODELS):
    """Enumerate the E17 DEV-C and SCREEN score inventory across requested models."""
    binding = _resolve_binding(bound_or_plan)
    selected = _normalize_models(models)
    per_model = {model: plain(enumerate_e17_model_inventory(binding, model)) for model in selected}
    return frozen({
        "permission": E17_DEV_C_PERMISSION,
        "condition": E17_CONDITION,
        "pooling": E17_POOLING,
        "models": per_model,
        "counts": {
            "models": len(selected),
            "reused_screen_blocks": 4 * len(selected),
            "reused_screen_routes": 16 * len(selected),
            "reused_dev_c_diagonal_blocks": 4 * len(selected),
            "reused_dev_c_diagonal_routes": 4 * len(selected),
            "authorized_off_diagonal_blocks": 4 * len(selected),
            "authorized_off_diagonal_routes": 12 * len(selected),
            "dev_c_feature_releases": 4 * len(selected),
            "source_systems": 4 * len(selected),
        },
    })


@contract
def score_frozen_head(matrix, coef, intercept, stats, *, estimable=None):
    """Score one verified feature matrix through a frozen Stage 3 head + source-FIT scaler in float64."""
    import numpy as np
    from src.stage3.engine import transform

    transformed = transform(matrix, stats)
    coef_arr = np.asarray(coef, dtype=np.float64)
    intercept_arr = np.asarray(intercept, dtype=np.float64)
    logits = np.asarray(transformed @ coef_arr.T, dtype=np.float64) + intercept_arr
    check(np.isfinite(logits).all(), "Nonfinite Phase 2 E17 logits")
    if estimable is not None:
        valid = np.asarray(estimable, dtype=bool)
        check(
            not np.any(logits * (~valid[None, :])),
            "Nonzero logits on unestimable categories",
        )
    return logits


@contract
def verify_diagonal_reproduction(computed_logits, accepted_logits, *, atol=0.0, rtol=0.0, model=None):
    """Assert exact float64 reproduction of Stage 4's accepted en->en DEV-C diagonal logits."""
    check(float(atol) == 0.0 and float(rtol) == 0.0, "E17 diagonal self-check requires exact atol=0.0, rtol=0.0")
    label = f" for {model}" if model else ""
    try:
        import numpy as np
    except ImportError:
        np = None

    if np is not None and (isinstance(computed_logits, np.ndarray) or isinstance(accepted_logits, np.ndarray)):
        comp = np.asarray(computed_logits, dtype=np.float64)
        acc = np.asarray(accepted_logits, dtype=np.float64)
        check(comp.ndim == 2 and comp.shape == acc.shape and comp.size > 0, f"Diagonal logit shape mismatch{label}")
        check(np.isfinite(comp).all() and np.isfinite(acc).all(), f"Nonfinite diagonal logits{label}")
        max_abs_diff = float(np.max(np.abs(comp - acc)))
        exact = bool(np.allclose(comp, acc, atol=atol, rtol=rtol) and np.array_equal(comp, acc))
        check(
            exact and max_abs_diff == 0.0,
            f"E17 en->en DEV-C diagonal reproduction failed{label}: max_abs_diff={max_abs_diff!r} (atol=0, rtol=0)",
        )
        rows, cats = int(comp.shape[0]), int(comp.shape[1])
    else:
        check(
            len(computed_logits) > 0 and len(computed_logits) == len(accepted_logits),
            f"Diagonal logit row count mismatch{label}",
        )
        max_abs_diff = 0.0
        cats = len(computed_logits[0])
        check(cats > 0, f"Empty diagonal logit categories{label}")
        for r_idx, (c_row, a_row) in enumerate(zip(computed_logits, accepted_logits)):
            check(len(c_row) == cats and len(a_row) == cats, f"Diagonal logit category count mismatch at row {r_idx}")
            for c_val, a_val in zip(c_row, a_row):
                cf, af = float(c_val), float(a_val)
                check(math.isfinite(cf) and math.isfinite(af), f"Nonfinite diagonal logits{label}")
                diff = abs(cf - af)
                if diff > max_abs_diff:
                    max_abs_diff = diff
                if cf != af:
                    raise ContractError(
                        f"E17 en->en DEV-C diagonal reproduction failed{label}: "
                        f"max_abs_diff={diff!r} (atol=0, rtol=0)"
                    )
        rows = len(computed_logits)

    return frozen({
        "passed": True,
        "max_abs_diff": 0.0,
        "atol": 0.0,
        "rtol": 0.0,
        "row_count": rows,
        "category_count": cats,
    })


def _load_stage4_prediction_block(stage4_plan, prediction_ref, *, expected_block):
    from src.stage4.predictions import load_prediction_block

    return load_prediction_block(stage4_plan, prediction_ref, expected_block=expected_block)


def _check_matrix_shape(matrix, expected_rows, expected_dim):
    shape = getattr(matrix, "shape", None)
    if shape is not None:
        check(
            tuple(int(x) for x in shape) == (expected_rows, expected_dim),
            f"DEV-C feature shape mismatch: expected {(expected_rows, expected_dim)}, got {shape}",
        )
    else:
        check(len(matrix) == expected_rows, f"DEV-C row count mismatch: expected {expected_rows}, got {len(matrix)}")
        if expected_rows > 0:
            check(len(matrix[0]) == expected_dim, "DEV-C feature dimension mismatch")


def _check_head_shapes(coef, intercept, estimable, prevalence, stats, *, expected_cats, expected_dim, expected_stats_ref):
    c_shape = getattr(coef, "shape", None)
    if c_shape is not None:
        check(tuple(int(x) for x in c_shape) == (expected_cats, expected_dim), "Frozen head coef shape mismatch")
    else:
        check(len(coef) == expected_cats and len(coef[0]) == expected_dim, "Frozen head coef shape mismatch")
    check(len(intercept) == expected_cats, "Frozen head intercept length mismatch")
    check(len(estimable) == expected_cats, "Frozen head estimable length mismatch")
    check(len(prevalence) == expected_cats, "Frozen head prevalence length mismatch")
    check(len(stats["scale"]) == expected_dim, "Frozen source-FIT scaler dimension mismatch")
    check(stats.get("center") is True, "Dense VLM source-FIT scaler must enable centering")
    check(all(float(v) > 0.0 for v in stats["scale"]), "Frozen source-FIT scaler must have positive scales")
    check(all(0.0 <= float(p) <= 1.0 for p in prevalence), "Frozen FIT prevalence out of [0, 1]")
    if "reference" in stats:
        check(
            plain(stats["reference"]) == plain(expected_stats_ref),
            "Loaded source-FIT statistics reference disagrees with Stage 3 candidate lineage",
        )


def _arrays_equal(a, b):
    try:
        import numpy as np

        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            return bool(np.array_equal(a, b))
    except ImportError:
        pass
    return list(a) == list(b)


def _dedupe_refs(refs):
    seen = set()
    out = []
    for ref in refs:
        p = plain(ref)
        key = (p["path"], p["sha256"])
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _expected_auth_inputs(binding, models):
    return {
        "stage4_release": plain(binding.stage4_release_ref),
        "lineage_identity": binding.lineage_identity,
        "models": list(models),
        "permission": E17_DEV_C_PERMISSION,
    }


@contract
def preflight_e17_dev_c(bound_or_plan, producer, *, models=MODELS, progress=None):
    """Run CPU E17 DEV-C preflight and publish an immutable `phase2_e17_auth` artifact.

    Verifies without opening FIT feature matrices or CONFIRM:
      (a) all Stage 2 DEV-C midpoint feature releases authenticate cleanly;
      (b) all Stage 3 selected ordinary heads + source-FIT scalers load cleanly
          via `_selected_parameters`;
      (c) in-memory `en -> en` DEV-C diagonal scoring matches Stage 4's accepted
          `en -> en` DEV-C logits to `atol=0, rtol=0` for each requested model.
    """
    binding = _resolve_binding(bound_or_plan)
    selected_models = _normalize_models(models)
    require_e17_dev_c_split(E17_DEV_C_SPLIT)

    expected_dev_c_rows = int(binding.protocol["split_sizes"][E17_DEV_C_SPLIT])
    expected_cats = len(binding.protocol["category_order"])
    recipe = plain(e17_calibrator_recipe(binding.protocol))

    # 9 work units per model: 4 DEV-C feature releases + 4 frozen source heads + 1 en->en diagonal check.
    total_units = len(selected_models) * 9 + 1
    completed_units = 0

    inventory = plain(enumerate_e17_inventory(binding, models=selected_models))
    verified_models = {}
    upstream_refs = [plain(binding.stage4_release_ref)]

    for model in selected_models:
        model_inv = inventory["models"][model]
        en_rows = None
        en_matrix = None
        en_head = None

        # (a) Authenticate all 4 Stage 2 DEV-C midpoint feature releases (never opening FIT or CONFIRM).
        for lang in LANGUAGES:
            t0 = time.monotonic()
            rows, matrix = read_verified_midpoint_features(
                binding, model, E17_CONDITION, E17_POOLING, lang, E17_DEV_C_SPLIT
            )
            read_s = time.monotonic() - t0
            check(
                len(rows) == expected_dev_c_rows,
                f"DEV-C row count mismatch for {model}/{lang}: expected {expected_dev_c_rows}, got {len(rows)}",
            )
            for row in rows:
                check(
                    row["split"] == E17_DEV_C_SPLIT and row["language"] == lang,
                    f"DEV-C row split/language mismatch for {model}/{lang}",
                )
            expected_dim = int(model_inv["dev_c_feature_releases"][lang]["dimension"])
            _check_matrix_shape(matrix, len(rows), expected_dim)
            model_inv["dev_c_feature_releases"][lang]["row_count"] = len(rows)
            model_inv["dev_c_feature_releases"][lang]["verified"] = True
            upstream_refs.append(model_inv["dev_c_feature_releases"][lang]["release_ref"])
            if lang == "en":
                en_rows, en_matrix = rows, matrix
            completed_units += 1
            _emit(progress, {
                "event": "preflight_unit",
                "stage": "verify_dev_c_features",
                "unit": f"{model}:dev_c_features:{lang}",
                "model": model,
                "language": lang,
                "completed": completed_units,
                "total": total_units,
                "read_hash_s": round(read_s, 6),
                "compute_s": 0.0,
                "write_s": 0.0,
                "peak_rss_bytes": _peak_rss_bytes(),
            })

        # (b) Load and verify all 4 Stage 3 selected ordinary heads + source-FIT scalers via _selected_parameters.
        for lang in LANGUAGES:
            t0 = time.monotonic()
            coef, intercept, estimable, prevalence, stats = load_frozen_source_head(
                binding, model, E17_CONDITION, E17_POOLING, lang
            )
            read_s = time.monotonic() - t0
            sys_rec = model_inv["source_systems"][lang]
            _check_head_shapes(
                coef,
                intercept,
                estimable,
                prevalence,
                stats,
                expected_cats=expected_cats,
                expected_dim=int(sys_rec["dimension"]),
                expected_stats_ref=sys_rec["statistics_ref"],
            )
            sys_rec["estimable_count"] = int(sum(bool(v) for v in estimable))
            sys_rec["verified"] = True
            upstream_refs.extend([
                sys_rec["selection_ref"],
                sys_rec["candidate_ref"],
                sys_rec["statistics_ref"],
            ])
            if lang == "en":
                en_head = (coef, intercept, estimable, prevalence, stats)
            completed_units += 1
            _emit(progress, {
                "event": "preflight_unit",
                "stage": "verify_source_head",
                "unit": f"{model}:source_head:{lang}",
                "model": model,
                "language": lang,
                "completed": completed_units,
                "total": total_units,
                "read_hash_s": round(read_s, 6),
                "compute_s": 0.0,
                "write_s": 0.0,
                "peak_rss_bytes": _peak_rss_bytes(),
            })

        # (c) In-memory en -> en DEV-C diagonal reproduction check against Stage 4 (atol=0, rtol=0).
        authorize_e17_dev_c_route(
            "en",
            "en",
            head_source_lang="en",
            split=E17_DEV_C_SPLIT,
            purpose="diagonal_self_check",
            allow_diagonal_self_check=True,
        )
        diag = model_inv["dev_c_diagonals"]["en"]
        t_read = time.monotonic()
        pred_map = _load_stage4_prediction_block(
            binding.stage4_plan,
            diag["prediction_ref"],
            expected_block=diag["block_id"],
        )
        read_s = time.monotonic() - t_read
        check(diag["route_id"] in pred_map, f"Missing en->en route in Stage 4 DEV-C diagonal block for {model}")
        accepted = pred_map[diag["route_id"]]
        check(accepted.split == E17_DEV_C_SPLIT, f"Accepted diagonal split mismatch for {model}")
        check(plain(accepted.rows) == plain(en_rows), f"Accepted en->en DEV-C row identity mismatch for {model}")

        coef, intercept, estimable, prevalence, stats = en_head
        check(_arrays_equal(accepted.estimable, estimable), f"Accepted en->en estimable mask mismatch for {model}")
        check(_arrays_equal(accepted.prevalence, prevalence), f"Accepted en->en prevalence mismatch for {model}")

        t_comp = time.monotonic()
        computed_logits = score_frozen_head(en_matrix, coef, intercept, stats, estimable=estimable)
        diag_check = plain(verify_diagonal_reproduction(
            computed_logits, accepted.logits, atol=0.0, rtol=0.0, model=model
        ))
        compute_s = time.monotonic() - t_comp

        # Add Stage 4 reused block refs to upstream provenance.
        for lang in LANGUAGES:
            upstream_refs.append(model_inv["dev_c_diagonals"][lang]["prediction_ref"])
            upstream_refs.append(model_inv["screen_blocks"][lang]["prediction_ref"])

        model_inv["diagonal_reproduction_check"] = {
            "route": "en->en",
            "source_language": "en",
            "target_language": "en",
            "split": E17_DEV_C_SPLIT,
            "stage4_block_id": diag["block_id"],
            "stage4_route_id": diag["route_id"],
            "stage4_prediction_ref": diag["prediction_ref"],
            **diag_check,
        }
        verified_models[model] = model_inv

        completed_units += 1
        _emit(progress, {
            "event": "preflight_unit",
            "stage": "verify_diagonal_reproduction",
            "unit": f"{model}:en_en_diagonal_check",
            "model": model,
            "completed": completed_units,
            "total": total_units,
            "read_hash_s": round(read_s, 6),
            "compute_s": round(compute_s, 6),
            "write_s": 0.0,
            "max_abs_diff": 0.0,
            "passed": True,
            "peak_rss_bytes": _peak_rss_bytes(),
        })

    body = {
        "schema": SCHEMA_E17_AUTH,
        "phase2_version": PHASE2_VERSION,
        "stage4_release": plain(binding.stage4_release_ref),
        "lineage_identity": binding.lineage_identity,
        "dataset_fingerprint": binding.dataset_fingerprint,
        "protocol_fingerprint": binding.protocol_fingerprint,
        "permission": E17_DEV_C_PERMISSION,
        "authorized_split": E17_DEV_C_SPLIT,
        "condition": E17_CONDITION,
        "pooling": E17_POOLING,
        "confirm_excluded": True,
        "fit_matrices_opened": False,
        "inputs": _expected_auth_inputs(binding, selected_models),
        "guardrails": {
            "authorized_scoring_split": E17_DEV_C_SPLIT,
            "frozen_source_head_only": True,
            "forbid_target_trained_head_for_target_calibrator": True,
            "forbid_base_head_refit": True,
            "forbid_scaler_refit": True,
            "forbid_screen_tuning": True,
            "forbid_e19_cutoff_modification": True,
            "forbid_confirm_access": True,
        },
        "inventory_summary": {
            "models": list(selected_models),
            "reused_screen_blocks_total": 4 * len(selected_models),
            "reused_screen_routes_total": 16 * len(selected_models),
            "reused_dev_c_diagonal_blocks_total": 4 * len(selected_models),
            "reused_dev_c_diagonal_routes_total": 4 * len(selected_models),
            "authorized_off_diagonal_blocks_total": 4 * len(selected_models),
            "authorized_off_diagonal_routes_total": 12 * len(selected_models),
            "dev_c_feature_releases_verified": 4 * len(selected_models),
            "source_heads_verified": 4 * len(selected_models),
        },
        "models": verified_models,
        "calibrator_recipe": recipe,
    }

    t_write = time.monotonic()
    auth_ref = publish_phase2(
        binding.root,
        plain(binding.protocol),
        binding.dataset_fingerprint,
        E17_AUTH_NAMESPACE,
        KIND_E17_AUTH,
        body,
        producer,
        upstream=_dedupe_refs(upstream_refs),
    )
    write_s = time.monotonic() - t_write

    completed_units += 1
    _emit(progress, {
        "event": "preflight_unit",
        "stage": "publish_e17_auth",
        "unit": "publish:phase2_e17_auth",
        "completed": completed_units,
        "total": total_units,
        "read_hash_s": 0.0,
        "compute_s": 0.0,
        "write_s": round(write_s, 6),
        "ref": plain(auth_ref),
        "peak_rss_bytes": _peak_rss_bytes(),
    })

    return auth_ref, frozen(body)


def _validate_e17_auth_body(body, protocol, *, binding=None, models=None):
    check(body.get("schema") == SCHEMA_E17_AUTH, "Invalid phase2_e17_auth schema")
    check(body.get("permission") == E17_DEV_C_PERMISSION, "Invalid E17 permission identifier")
    check(body.get("confirm_excluded") is True, "phase2_e17_auth must assert confirm_excluded=True")
    check(body.get("fit_matrices_opened") is False, "phase2_e17_auth must not open FIT feature matrices")
    check(
        plain(body.get("calibrator_recipe")) == plain(e17_calibrator_recipe(protocol)),
        "phase2_e17_auth calibrator_recipe does not match frozen §1.6 recipe",
    )

    if binding is not None:
        check(
            body.get("lineage_identity") == binding.lineage_identity,
            "phase2_e17_auth lineage_identity mismatch",
        )
        check(
            plain(body.get("stage4_release")) == plain(binding.stage4_release_ref),
            "phase2_e17_auth stage4_release mismatch",
        )
        check(
            body.get("dataset_fingerprint") == binding.dataset_fingerprint,
            "phase2_e17_auth dataset_fingerprint mismatch",
        )

    present_models = body.get("models")
    check(isinstance(present_models, (dict,)) or hasattr(present_models, "keys"), "Missing models mapping")
    required_models = _normalize_models(models) if models is not None else tuple(present_models.keys())
    check(bool(required_models), "No models present in phase2_e17_auth")
    for model in required_models:
        check(model in present_models, f"phase2_e17_auth missing required model: {model!r}")
        entry = present_models[model]
        diag = entry.get("diagonal_reproduction_check", {})
        check(
            diag.get("passed") is True
            and diag.get("max_abs_diff") == 0.0
            and diag.get("atol") == 0.0
            and diag.get("rtol") == 0.0,
            f"phase2_e17_auth failed diagonal reproduction check for {model!r}",
        )
        check(len(entry.get("screen_blocks", {})) == 4, f"Expected 4 SCREEN blocks for {model!r}")
        check(len(entry.get("dev_c_diagonals", {})) == 4, f"Expected 4 DEV-C diagonals for {model!r}")
        off_blocks = entry.get("off_diagonal_blocks", {})
        check(len(off_blocks) == 4, f"Expected 4 off-diagonal DEV-C blocks for {model!r}")
        route_count = 0
        for t_lang, blk in off_blocks.items():
            routes = blk.get("routes", ())
            check(len(routes) == 3, f"Expected 3 off-diagonal routes for {model}/{t_lang}")
            for route in routes:
                check(
                    route["source_language"] != route["target_language"] == t_lang
                    and route["source_group"] != route["target_group"]
                    and route.get("frozen_source_head_only") is True,
                    f"Invalid off-diagonal route in {model}/{t_lang}",
                )
                route_count += 1
        check(route_count == 12, f"Expected 12 off-diagonal DEV-C routes for {model!r}")


@contract
def load_e17_auth(root, protocol, reference, *, binding=None, models=None):
    """Authenticate an existing `phase2_e17_auth` artifact on disk."""
    expected_dataset = binding.dataset_fingerprint if binding is not None else None
    plain_protocol = plain(protocol)
    ref, body, marker = load_phase2_artifact(
        root,
        plain_protocol,
        reference,
        KIND_E17_AUTH,
        expected_schema=SCHEMA_E17_AUTH,
        expected_dataset=expected_dataset,
    )
    _validate_e17_auth_body(body, plain_protocol, binding=binding, models=models)
    return ref, body, marker


@contract
def verify_e17_auth_resume(bound_or_plan, reference, *, models=MODELS):
    """Explicit-reference resume verification for `phase2_e17_auth`."""
    binding = _resolve_binding(bound_or_plan)
    selected_models = _normalize_models(models)
    plain_protocol = plain(binding.protocol)
    ref, body, marker = verify_resume_ref(
        binding.root,
        plain_protocol,
        reference,
        KIND_E17_AUTH,
        expected_schema=SCHEMA_E17_AUTH,
        expected_dataset=binding.dataset_fingerprint,
        expected_inputs=_expected_auth_inputs(binding, selected_models),
    )
    _validate_e17_auth_body(body, plain_protocol, binding=binding, models=selected_models)
    return ref, body, marker


def _parse_ref_arg(value):
    if value is None:
        return None
    text = value.strip()
    if text.startswith("{"):
        return json.loads(text)
    return text


def _cli(argv=None):
    parser = argparse.ArgumentParser(description="Phase 2 Stage 1 E17 DEV-C preflight and authorization.")
    parser.add_argument("--phase1-root", required=True, help="Mounted PHASE1_ROOT directory.")
    parser.add_argument(
        "--model",
        choices=("qwen", "gemma", "both"),
        default="both",
        help="Model selection ('qwen', 'gemma', or 'both').",
    )
    parser.add_argument(
        "--stage4-release",
        default=None,
        help="Optional explicit Stage 4 release path or JSON {'path','sha256'} dict (defaults to pinned Phase 1 release).",
    )
    parser.add_argument("--protocol", type=Path, default=REPO / "configs/protocol.json")
    parser.add_argument(
        "--storage-scope",
        choices=("local-development", "persistent-drive"),
        default="local-development",
    )
    parser.add_argument(
        "--resume-ref",
        default=None,
        help="Optional explicit phase2_e17_auth _COMPLETE.json path or JSON {'path','sha256'} dict to verify and reuse.",
    )
    parser.add_argument(
        "--output-ref",
        type=Path,
        default=None,
        help="Optional path to write the published/resumed {'path','sha256'} JSON reference.",
    )
    args = parser.parse_args(argv)

    def stream(event):
        print(json.dumps(event, sort_keys=True), flush=True)

    root = phase1_root(args.phase1_root)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    models = _normalize_models(args.model)
    release_ref = _parse_ref_arg(args.stage4_release) if args.stage4_release else plain(PHASE1_STAGE4_RELEASE)

    stream({
        "event": "e17_preflight_start",
        "phase1_root": str(root),
        "models": list(models),
        "permission": E17_DEV_C_PERMISSION,
        "resuming": args.resume_ref is not None,
    })

    binding = bind_phase1(root, protocol, stage4_release_ref=release_ref)

    if args.resume_ref is not None:
        resume_target = _parse_ref_arg(args.resume_ref)
        auth_ref, _body, _marker = verify_e17_auth_resume(binding, resume_target, models=models)
        resumed = True
    else:
        init_phase2_dirs(root)
        producer = current_producer(scope=args.storage_scope)
        auth_ref, _body = preflight_e17_dev_c(binding, producer, models=models, progress=stream)
        resumed = False

    ref_plain = plain(auth_ref)
    if args.output_ref is not None:
        args.output_ref.parent.mkdir(parents=True, exist_ok=True)
        args.output_ref.write_text(json.dumps(ref_plain, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    stream({
        "event": "e17_auth_complete",
        "status": "passed",
        "resumed": resumed,
        "models": list(models),
        "permission": E17_DEV_C_PERMISSION,
        "ref": ref_plain,
        "peak_rss_bytes": _peak_rss_bytes(),
    })
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_cli())
    except (ContractError, OSError, ValueError, KeyError, RuntimeError) as exc:
        print(json.dumps({"event": "blocked", "status": "blocked", "blockers": [str(exc)]}, sort_keys=True), flush=True)
        raise SystemExit(2)
