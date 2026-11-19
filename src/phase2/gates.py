"""Frozen E29 -> E30 / E11 gate evaluator, Stage 5 decision ranges, and gate I/O.

Implements phase2.md §1.5 and the `phase2_e29_gate` schema in
docs/phase2_stage1_interfaces.md. All thresholds are frozen before running E29
or inspecting any Phase 2 SCREEN outputs.
"""

import math
from typing import Any, Mapping

from src.common.contracts import LANGUAGES
from src.phase2.interfaces import (
    GATE_DECISIONS,
    KIND_E29_GATE,
    MODELS,
    SCHEMA_E29_GATE,
    check,
    contract,
    frozen,
    load_phase2_artifact,
    normalize_ref,
    plain,
    publish_phase2,
)

# ---------------------------------------------------------------------------
# Frozen E29 -> E30 and E29 -> E11 gate constants (phase2.md §1.5)
# ---------------------------------------------------------------------------

ACCESSIBILITY_DIAGONAL_FLOOR = 0.750
BOOTSTRAP_TOTAL_DRAWS = 2000
BOOTSTRAP_MIN_VALID_DRAWS = 1900

E30_POINT_MIN = 0.015
E30_CI_LOWER_MIN = 0.005
E30_PARK_CI_UPPER_MAX = 0.015

E11_BROAD_POINT_MIN = 0.015
E11_CI_LOWER_MIN = 0.005
E11_PARK_CI_UPPER_MAX = 0.015
E11_DIRECTIONAL_TOTAL_ROUTES = 12
E11_DIRECTIONAL_MIN_ROUTES = 3
E11_DIRECTIONAL_ROUTE_POINT_MIN = 0.025
E11_DIRECTIONAL_ROUTE_CI_LOWER_MIN = 0.010

E29_GATE_THRESHOLDS = frozen({
    "accessibility_diagonal_floor": ACCESSIBILITY_DIAGONAL_FLOOR,
    "bootstrap_total_draws": BOOTSTRAP_TOTAL_DRAWS,
    "bootstrap_min_valid_draws": BOOTSTRAP_MIN_VALID_DRAWS,
    "E30": {
        "open_point_min": E30_POINT_MIN,
        "open_ci_lower_gt": E30_CI_LOWER_MIN,
        "park_ci_upper_lt": E30_PARK_CI_UPPER_MAX,
    },
    "E11": {
        "broad_open_point_min": E11_BROAD_POINT_MIN,
        "open_ci_lower_gt": E11_CI_LOWER_MIN,
        "park_ci_upper_lt": E11_PARK_CI_UPPER_MAX,
        "directional_total_routes": E11_DIRECTIONAL_TOTAL_ROUTES,
        "directional_min_routes": E11_DIRECTIONAL_MIN_ROUTES,
        "directional_route_point_min": E11_DIRECTIONAL_ROUTE_POINT_MIN,
        "directional_route_ci_lower_gt": E11_DIRECTIONAL_ROUTE_CI_LOWER_MIN,
    },
})

# ---------------------------------------------------------------------------
# Frozen Stage 5 decision-range constants for E17, E07, E30, E11 (§1.5)
# ---------------------------------------------------------------------------

STAGE5_DECISION_RANGES = frozen({
    "E17": {
        "consequential": {
            "nll_gain_min": 0.015,
            "brier_gain_min": 0.005,
            "ci_excludes_zero_favorably": True,
            "separate_source_and_target_dev_c_verdicts": True,
        },
        "negligible": {
            "nll_ci_upper_lt": 0.005,
            "brier_ci_upper_lt": 0.002,
        },
        "checks": {
            "ranking_invariance_tol": 1e-12,
            "positive_temperature_required": True,
            "temp_only_preserves_0p5_decisions": True,
        },
    },
    "E07": {
        "consequential": {
            "abs_gap_shift_min": 0.015,
            "ci_excludes_zero": True,
            "or_pooling_reduction_sign_reversal": True,
        },
        "negligible": {
            "abs_gap_shift_lt": 0.015,
            "ci_within": (-0.025, 0.025),
            "pooling_reduction_sign_preserved": True,
        },
        "checks": {
            "draw0_exact_reproduction": True,
            "unchanged_and_identical_text_atol": 0.0,
            "unchanged_and_identical_text_rtol": 0.0,
            "restricted_cohort_matched_draw0_required": True,
        },
    },
    "E30": {
        "consequential": {
            "fixed_head_feature_gain_min": 0.015,
            "abs_interaction_min": 0.015,
            "ci_excludes_zero": True,
        },
        "negligible": {
            "fixed_head_feature_gain_ci_within": (-0.015, 0.015),
            "interaction_ci_within": (-0.015, 0.015),
        },
        "checks": {
            "identical_depth_hook_width_row_keys": True,
            "off_distribution_collapse_drop": 0.015,
        },
    },
    "E11": {
        "consequential": {
            "mean_scale_auroc_gap_reduction_min": 0.010,
            "or_fraction_of_g_res_mean_min": 0.25,
            "ci_excludes_zero": True,
            "loss_only_nll_gain_min": 0.015,
            "loss_only_brier_gain_min": 0.005,
        },
        "negligible": {
            "mean_scale_auroc_gap_reduction_ci_upper_lt": 0.005,
        },
        "checks": {
            "mean_shift_ranking_invariance_tol": 1e-12,
            "fit_paired_images": 2160,
            "variance_floor": 1e-12,
        },
    },
})


def _is_finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _coerce_optional_float(value, label):
    if value is None:
        return None
    check(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} must be numeric or None")
    fval = float(value)
    return fval if math.isfinite(fval) else None


def _coerce_ci(ci, label):
    if ci is None:
        return None
    check(isinstance(ci, (list, tuple)) and len(ci) == 2, f"{label} must be a 2-element [low, high] sequence or None")
    low = _coerce_optional_float(ci[0], f"{label}[0]")
    high = _coerce_optional_float(ci[1], f"{label}[1]")
    if low is None or high is None:
        return None
    check(low <= high, f"{label} lower bound ({low}) exceeds upper bound ({high})")
    return [low, high]


def _normalize_diagonals(diagonals):
    check(isinstance(diagonals, Mapping), "caption_mean diagonals must be a mapping by language")
    check(set(diagonals.keys()) == set(LANGUAGES), f"Expected all 4 receiving diagonals {LANGUAGES}")
    out = {}
    for lang in LANGUAGES:
        out[lang] = _coerce_optional_float(diagonals[lang], f"diagonals[{lang!r}]")
    return out


def _normalize_route_residuals(route_residuals):
    """Normalize 12 directed off-diagonal route residuals into sorted (route_key, point, ci_low)."""
    if isinstance(route_residuals, Mapping):
        items = list(route_residuals.items())
    elif isinstance(route_residuals, (list, tuple)):
        items = []
        for idx, rec in enumerate(route_residuals):
            check(isinstance(rec, Mapping), f"Route residual entry {idx} must be a mapping")
            key = rec.get("route", rec.get("route_id"))
            check(isinstance(key, str) and bool(key), f"Route residual entry {idx} missing route identifier")
            items.append((key, rec))
    else:
        check(False, "route_residuals must be a mapping or 12-element sequence")
        return []

    check(
        len(items) == E11_DIRECTIONAL_TOTAL_ROUTES,
        f"Expected exactly {E11_DIRECTIONAL_TOTAL_ROUTES} directed off-diagonal routes, got {len(items)}",
    )
    seen = set()
    normalized = []
    for key, rec in items:
        check(isinstance(key, str) and bool(key), "Route identifier must be a nonempty string")
        check(key not in seen, f"Duplicate route identifier in route_residuals: {key!r}")
        seen.add(key)
        check(isinstance(rec, Mapping), f"Route residual {key!r} must be a mapping")
        point = _coerce_optional_float(rec.get("point", rec.get("g_res")), f"route_residuals[{key!r}].point")
        if "ci" in rec and rec["ci"] is not None:
            ci = _coerce_ci(rec["ci"], f"route_residuals[{key!r}].ci")
            ci_low = ci[0] if ci is not None else None
        else:
            ci_low = _coerce_optional_float(rec.get("ci_low"), f"route_residuals[{key!r}].ci_low")
        normalized.append((key, point, ci_low))
    normalized.sort(key=lambda item: item[0])
    return normalized


@contract
def evaluate_e29_gates(
    model,
    *,
    diagonals,
    valid_draws,
    delta_g_mean,
    delta_g_ci,
    g_res_mean,
    g_res_ci,
    route_residuals,
):
    """Evaluate frozen E29 -> E30 and E29 -> E11 gates for one model.

    Parameters
    ----------
    model:
        Explicit model selector (`"qwen"` or `"gemma"`).
    diagonals:
        Mapping `{lang: auroc}` for all 4 `caption_mean` receiving diagonals
        (`en`, `hi`, `bn`, `te`) on the 14-category primary macro.
    valid_draws:
        Integer count of valid bootstrap draws out of 2000 on the primary
        12-route macro contrast.
    delta_g_mean, delta_g_ci:
        Point estimate and 95% CI `[L_Delta, U_Delta]` for the 12-route mean
        pooling gap reduction `G(last_caption) - G(caption_mean)`.
    g_res_mean, g_res_ci:
        Point estimate and 95% CI `[L_res, U_res]` for the 12-route mean
        residual gap `G(caption_mean)`.
    route_residuals:
        Mapping `{route_key: {"point": ..., "ci": [low, high]}}` (or
        `{"point": ..., "ci_low": ...}`) or 12-item sequence with `route` /
        `route_id` keys for the 12 directed off-diagonal routes (`s != t`).
    """
    check(model in MODELS, f"Unknown model selector: {model!r}")
    check(
        type(valid_draws) is int and 0 <= valid_draws <= BOOTSTRAP_TOTAL_DRAWS,
        f"valid_draws must be an int in [0, {BOOTSTRAP_TOTAL_DRAWS}]; got {valid_draws!r}",
    )

    diag_map = _normalize_diagonals(diagonals)
    diag_finite = [v for v in diag_map.values() if v is not None]
    diagonal_min = min(diag_finite) if len(diag_finite) == len(LANGUAGES) else None
    diagonal_floor_met = (
        diagonal_min is not None and diagonal_min >= ACCESSIBILITY_DIAGONAL_FLOOR
    )

    dg_point = _coerce_optional_float(delta_g_mean, "delta_g_mean")
    dg_ci = _coerce_ci(delta_g_ci, "delta_g_ci")
    gres_point = _coerce_optional_float(g_res_mean, "g_res_mean")
    gres_ci = _coerce_ci(g_res_ci, "g_res_ci")
    routes = _normalize_route_residuals(route_residuals)

    directional_triggered = [
        key
        for key, pt, ci_low in routes
        if pt is not None
        and ci_low is not None
        and pt >= E11_DIRECTIONAL_ROUTE_POINT_MIN
        and ci_low > E11_DIRECTIONAL_ROUTE_CI_LOWER_MIN
    ]
    n_directional = len(directional_triggered)

    valid_ok = valid_draws >= BOOTSTRAP_MIN_VALID_DRAWS

    # --- E30 gate evaluation ---
    if not valid_ok:
        e30_decision = {
            "decision": "UNRESOLVED",
            "reason": "insufficient_valid_draws",
        }
    elif not diagonal_floor_met:
        e30_decision = {
            "decision": "PARK",
            "reason": "accessibility_floor_failed",
        }
    elif dg_point is None or dg_ci is None:
        e30_decision = {
            "decision": "UNRESOLVED",
            "reason": "non_finite_estimate",
        }
    elif dg_point >= E30_POINT_MIN and dg_ci[0] > E30_CI_LOWER_MIN:
        e30_decision = {
            "decision": "OPEN",
            "reason": "consequential_pooling_reduction",
        }
    elif dg_ci[1] < E30_PARK_CI_UPPER_MAX:
        e30_decision = {
            "decision": "PARK",
            "reason": "upper_ci_below_threshold",
        }
    else:
        e30_decision = {
            "decision": "UNRESOLVED",
            "reason": "inconclusive_interval",
        }

    # --- E11 gate evaluation ---
    if not valid_ok:
        e11_decision = {
            "decision": "UNRESOLVED",
            "reason": "insufficient_valid_draws",
            "trigger": None,
        }
    elif not diagonal_floor_met:
        e11_decision = {
            "decision": "PARK",
            "reason": "accessibility_floor_failed",
            "trigger": None,
        }
    elif gres_point is None or gres_ci is None:
        e11_decision = {
            "decision": "UNRESOLVED",
            "reason": "non_finite_estimate",
            "trigger": None,
        }
    else:
        broad_open = (gres_point >= E11_BROAD_POINT_MIN) and (gres_ci[0] > E11_CI_LOWER_MIN)
        directional_open = (gres_ci[0] > E11_CI_LOWER_MIN) and (n_directional >= E11_DIRECTIONAL_MIN_ROUTES)
        if broad_open or directional_open:
            if broad_open and directional_open:
                trigger = "broad_and_directional_residual"
                reason = "broad_and_directional_residual_trigger"
            elif broad_open:
                trigger = "broad_residual"
                reason = "broad_residual_trigger"
            else:
                trigger = "directional_residual_trigger"
                reason = "directional_residual_trigger"
            e11_decision = {
                "decision": "OPEN",
                "reason": reason,
                "trigger": trigger,
            }
        elif gres_ci[1] < E11_PARK_CI_UPPER_MAX and n_directional < E11_DIRECTIONAL_MIN_ROUTES:
            e11_decision = {
                "decision": "PARK",
                "reason": "residual_below_threshold",
                "trigger": None,
            }
        else:
            e11_decision = {
                "decision": "UNRESOLVED",
                "reason": "inconclusive_residual",
                "trigger": None,
            }

    check(e30_decision["decision"] in GATE_DECISIONS, "Invalid E30 gate decision")
    check(e11_decision["decision"] in GATE_DECISIONS, "Invalid E11 gate decision")

    return frozen({
        "model": model,
        "thresholds": plain(E29_GATE_THRESHOLDS),
        "observed": {
            "diagonals": diag_map,
            "diagonal_min_auroc": diagonal_min,
            "diagonal_floor_met": diagonal_floor_met,
            "valid_draws": valid_draws,
            "delta_g_mean": dg_point,
            "delta_g_ci": dg_ci,
            "g_res_mean": gres_point,
            "g_res_ci": gres_ci,
            "directional_routes_triggered": directional_triggered,
            "directional_routes_count": n_directional,
        },
        "decisions": {
            "E30": e30_decision,
            "E11": e11_decision,
        },
    })


def _extract_plan_binding(plan):
    """Extract (plan_ref, plan_identity, dataset_fingerprint) from a Phase2Plan or mapping."""
    if hasattr(plan, "reference") and isinstance(plan, Mapping):
        plan_ref = plain(plan.reference)
        plan_identity = plan["plan_identity"]
        dataset_fp = plan["dataset_fingerprint"]
    else:
        check(isinstance(plan, Mapping), "plan must be a Phase2Plan or plan mapping")
        plan_ref = plain(plan["reference"])
        plan_identity = plan["plan_identity"]
        dataset_fp = plan["dataset_fingerprint"]
    return plan_ref, plan_identity, dataset_fp


def _validate_gate_body(body):
    required = {
        "schema",
        "plan",
        "plan_identity",
        "model",
        "e29_analysis_ref",
        "thresholds",
        "observed",
        "decisions",
    }
    missing = required - set(body.keys())
    check(not missing, f"phase2_e29_gate missing required keys: {sorted(missing)}")
    check(body["schema"] == SCHEMA_E29_GATE, "Wrong phase2_e29_gate schema")
    check(body["model"] in MODELS, f"Unknown model in phase2_e29_gate: {body['model']!r}")
    check(plain(body["thresholds"]) == plain(E29_GATE_THRESHOLDS), "E29 gate thresholds mismatch frozen constants")
    observed = body["observed"]
    for key in (
        "diagonal_min_auroc",
        "valid_draws",
        "delta_g_mean",
        "delta_g_ci",
        "g_res_mean",
        "g_res_ci",
        "directional_routes_triggered",
    ):
        check(key in observed, f"phase2_e29_gate observed missing: {key}")
    decisions = body["decisions"]
    check(set(decisions.keys()) == {"E30", "E11"}, "phase2_e29_gate decisions must cover E30 and E11")
    check(decisions["E30"].get("decision") in GATE_DECISIONS and isinstance(decisions["E30"].get("reason"), str),
          "Invalid E30 decision record")
    check(
        decisions["E11"].get("decision") in GATE_DECISIONS
        and isinstance(decisions["E11"].get("reason"), str)
        and "trigger" in decisions["E11"],
        "Invalid E11 decision record",
    )


@contract
def publish_e29_gate(
    root,
    protocol,
    plan,
    model,
    e29_analysis_ref,
    evaluation,
    producer,
    *,
    run_id=None,
):
    """Publish an immutable `phase2_e29_gate` artifact under `phase2/results/e29_gates/`."""
    check(model in MODELS, f"Unknown model selector: {model!r}")
    plan_ref, plan_identity, dataset_fp = _extract_plan_binding(plan)
    e29_ref = normalize_ref(root, e29_analysis_ref)
    eval_plain = plain(evaluation)
    check(eval_plain.get("model") == model, "Gate evaluation model does not match target model")
    check(
        eval_plain.get("thresholds") == plain(E29_GATE_THRESHOLDS),
        "Gate evaluation thresholds do not match frozen E29_GATE_THRESHOLDS",
    )

    body = {
        "schema": SCHEMA_E29_GATE,
        "plan": plan_ref,
        "plan_identity": plan_identity,
        "model": model,
        "e29_analysis_ref": e29_ref,
        "thresholds": eval_plain["thresholds"],
        "observed": eval_plain["observed"],
        "decisions": eval_plain["decisions"],
    }
    _validate_gate_body(body)

    return publish_phase2(
        root,
        protocol,
        dataset_fp,
        "phase2/results/e29_gates",
        KIND_E29_GATE,
        body,
        producer,
        upstream=[plan_ref, e29_ref],
        run_id=run_id,
    )


@contract
def load_e29_gate(root, protocol, reference, *, plan=None, expected_model=None):
    """Authenticate and return `(ref, body, marker)` for a published `phase2_e29_gate`."""
    expected_dataset = None
    expected_plan_identity = None
    expected_plan_ref = None
    if plan is not None:
        expected_plan_ref, expected_plan_identity, expected_dataset = _extract_plan_binding(plan)
    if expected_model is not None:
        check(expected_model in MODELS, f"Unknown model selector: {expected_model!r}")

    ref, body, marker = load_phase2_artifact(
        root,
        protocol,
        reference,
        KIND_E29_GATE,
        expected_schema=SCHEMA_E29_GATE,
        expected_dataset=expected_dataset,
        expected_plan_identity=expected_plan_identity,
        expected_owner=expected_model,
    )
    _validate_gate_body(body)
    if expected_plan_ref is not None:
        check(plain(body["plan"]) == expected_plan_ref, "E29 gate plan reference mismatch")
    return ref, body, marker
