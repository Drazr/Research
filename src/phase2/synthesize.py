"""Phase 2 Stage 5 synthesis, decision register, Markdown review, and independent readback.

Authenticates all 15 certified Stage 1-4 Phase 2 artifacts on mounted PHASE1_ROOT
(zero new model scoring, zero feature reads, zero bootstrap recomputation, zero
Phase 1 mutation, CONFIRM strictly closed), evaluates every direction against the
frozen `STAGE5_DECISION_RANGES`, publishes `phase2_release` (`body.json` +
`phase2_review.md`) under `phase2/releases/<run_id>/`, and verifies the release
via a fresh-subprocess independent readback.
"""

import argparse
from collections.abc import Mapping
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Optional

from src.common.contracts import ContractError, fingerprint
from src.common.storage import file_sha256, new_run_id, phase1_root, verify, within
from src.phase2.e07_audit import load_e07_audit
from src.phase2.e17_auth import E17_DEV_C_PERMISSION, E17_INVARIANCE_TOL
from src.phase2.gates import (
    BOOTSTRAP_MIN_VALID_DRAWS,
    E29_GATE_THRESHOLDS,
    STAGE5_DECISION_RANGES,
    load_e29_gate,
)
from src.phase2.interfaces import (
    KIND_ANALYSIS,
    KIND_E07_AUDIT,
    KIND_E07_RELEASE,
    KIND_E17_AUTH,
    KIND_E29_GATE,
    KIND_PLAN,
    KIND_RELEASE,
    MODELS,
    PHASE2_VERSION,
    REPO,
    SCHEMA_ANALYSIS,
    SCHEMA_E07_FEATURES,
    SCHEMA_E17_AUTH,
    SCHEMA_RELEASE,
    check,
    contract,
    current_producer,
    frozen,
    load_phase2_artifact,
    normalize_ref,
    plain,
    publish_phase2,
)
from src.phase2.plan import load_phase2_plan

STAGE5_VERSION = "phase2-stage5.v1"
RELEASE_NAMESPACE = "phase2/releases"
DECISION_OUTCOMES = (
    "retain",
    "revise",
    "park within tested scope",
    "insufficient evidence",
)

# Certified Stage 1-4 completion ledger (phase2.md §2.9 & §4.7).
CERTIFIED_PHASE2_REFS = frozen({
    "phase2_plan": {
        "path": "phase2/plans/20261119T163039Z-52eb4650cec3436bb4bf989afe3cc303/_COMPLETE.json",
        "sha256": "c2d8e153802ebfefd1caddcb7b0cfc9e681bd1b978a422e4475c3187effbd62d",
    },
    "e07_audit": {
        "path": "phase2/audits/e07/20261119T161858Z-a6b9ac3da1324696adf82048b0661bd4/_COMPLETE.json",
        "sha256": "cdee16135af47f595a278316769b5bf26193fde172851df0a4f40275c23cbc38",
    },
    "e17_auth": {
        "path": "phase2/plans/20261119T162232Z-710a83aae9064c1aa26086b34b17bf45/_COMPLETE.json",
        "sha256": "129cba3ef774dc4c69ecfeabc8547e2aa2186567ec1ee77fb047fc39ba96ddff",
    },
    "models": {
        "qwen": {
            "e29_analysis": {
                "path": "phase2/results/e29/qwen/20261119T173548Z-73ad41f4722e42fa931817b6f2b28681/_COMPLETE.json",
                "sha256": "77a8dc8ff50f8886cb706e09e7540952a66c6d87fa7f1278a830f196016c9926",
            },
            "e29_gate": {
                "path": "phase2/results/e29_gates/20261119T173548Z-73ad41f4722e42fa931817b6f2b28681-gate/_COMPLETE.json",
                "sha256": "3975fedbd35e63b2e21f5f94b196782f5642ec9e35ec99bb84d7b547c40341ef",
            },
            "e17_analysis": {
                "path": "phase2/results/e17/qwen/20261119T174402Z-317626df13c944b188686ba32d9b92cb/_COMPLETE.json",
                "sha256": "fcb7c5b1e31f1dfdb43572e077879fdacba18cc8384f77b647d195c04227d771",
            },
            "e07_release": {
                "path": "phase2/features/e07/qwen/releases/20261119T185419Z-7dd0387228af4589b14b976c6b569043/_COMPLETE.json",
                "sha256": "b05a325cae38c89633d237e57c8cd0f7b40ad30a48bd60b40d0d895bad4764d3",
            },
            "e30_analysis": {
                "path": "phase2/results/e30/qwen/20261119T230416Z-8c601fa863604e3395624e37382fdedb/_COMPLETE.json",
                "sha256": "0dbe25f3fe6659b6e18419153c354b97ee4b124d8b29bd5096e7652c7a0b1f9f",
            },
            "e11_analysis": {
                "path": "phase2/results/e11/qwen/20261119T230112Z-af0550ca6a0c494e8a4f9d1728af213e/_COMPLETE.json",
                "sha256": "552d0f4f59bf0760ff7ee8da62b64197550d9087071becd2a260080f203f4c1b",
            },
            "e07_analysis": {
                "path": "phase2/results/e07/qwen/20261119T230606Z-f629f48ed8e7431b930fce6b47d8fd53/_COMPLETE.json",
                "sha256": "f8bda6f082862a07a15b42d497d605894ab4ceed7403dfdb257d9ff013f3ee09",
            },
        },
        "gemma": {
            "e29_analysis": {
                "path": "phase2/results/e29/gemma/20261119T173446Z-35a50762bbf445439faff5951c00eb80/_COMPLETE.json",
                "sha256": "ae3b5183ce126c7e2f6545384949798a4831756da460294585abf980fa12ec16",
            },
            "e29_gate": {
                "path": "phase2/results/e29_gates/20261119T173446Z-35a50762bbf445439faff5951c00eb80-gate/_COMPLETE.json",
                "sha256": "7434bee02f3875af67287a520e0731fd1fa8d066c189159a85fc13b0e65fe512",
            },
            "e17_analysis": {
                "path": "phase2/results/e17/gemma/20261119T220802Z-167de08008b74942813cdf81ca061e43/_COMPLETE.json",
                "sha256": "ab3d81143f917c2133d9beb888f64b38330995bcb4f325b7c33ed629e5b05c50",
            },
            "e07_release": {
                "path": "phase2/features/e07/gemma/releases/20261119T185440Z-bb5649a5ecdb418eab80a50a8bf748c8/_COMPLETE.json",
                "sha256": "ff5c8d2d2bb04374c735136c5e059390001ff4acdaff7a68ef5c4cc803b0b55e",
            },
            "e30_analysis": {
                "path": "phase2/results/e30/gemma/20261119T225928Z-682c54d91984488aacffba40ab0f6b2c/_COMPLETE.json",
                "sha256": "0de28dc4d6d4592e4312c3b64c86c8810267d1a4d10f9f36bb319a32b5382c6a",
            },
            "e11_analysis": {
                "path": "phase2/results/e11/gemma/20261119T230113Z-af983c1752eb4861a0c9390d53c096e5/_COMPLETE.json",
                "sha256": "09fa67ee3d4aed49726e57403077f47aeca47bab6fe4a67559ce2def0173dbbc",
            },
            "e07_analysis": {
                "path": "phase2/results/e07/gemma/20261119T230653Z-7ee10a1674f245ea97d5e50a6a6645ae/_COMPLETE.json",
                "sha256": "4603c8d8120d816ce14a7c299e561248a1f63fa670ac6c776e7a6034f09bdd95",
            },
        },
    },
})


def _emit(progress, event):
    if progress is not None:
        progress(event)


def _num(val):
    if val is None or isinstance(val, bool):
        return None
    f = float(val)
    return f if math.isfinite(f) else None


def _extract_stat(node):
    """Normalize a metric summary dict across E29/E17/E30/E11/E07 schemas.

    Returns dict: {"value": float|None, "lower": float|None, "upper": float|None, "valid_draws": int|None}.
    """
    if not isinstance(node, Mapping):
        return {"value": None, "lower": None, "upper": None, "valid_draws": None}
    if "primary_macro" in node and isinstance(node["primary_macro"], Mapping):
        node = node["primary_macro"]
    val = _num(node.get("value", node.get("point")))
    interval = node.get("interval")
    if isinstance(interval, Mapping):
        lower = _num(interval.get("lower"))
        upper = _num(interval.get("upper"))
        vd = interval.get("valid_draws")
    else:
        lower = _num(node.get("lower", node.get("ci_low")))
        upper = _num(node.get("upper", node.get("ci_high")))
        vd = node.get("valid_draws")
        ci = node.get("ci")
        if lower is None and isinstance(ci, (list, tuple)) and len(ci) == 2:
            lower = _num(ci[0])
            upper = _num(ci[1])
    return {
        "value": val,
        "lower": lower,
        "upper": upper,
        "valid_draws": int(vd) if vd is not None else None,
    }


def _fmt(val, digits=4):
    if val is None:
        return "n/a"
    return f"{float(val):+.{digits}f}" if digits == 4 else f"{float(val):.{digits}f}"


def _fmt_stat(stat, signed=True):
    v = stat.get("value")
    lo = stat.get("lower")
    hi = stat.get("upper")
    if v is None:
        return "n/a"
    v_s = f"{v:+.4f}" if signed else f"{v:.4f}"
    if lo is None or hi is None:
        return v_s
    lo_s = f"{lo:+.4f}" if signed else f"{lo:.4f}"
    hi_s = f"{hi:+.4f}" if signed else f"{hi:.4f}"
    return f"{v_s} [{lo_s}, {hi_s}]"


def _ci_excludes_zero_positive(stat):
    return stat["lower"] is not None and stat["lower"] > 0.0


def _ci_excludes_zero_either(stat):
    if stat["lower"] is None or stat["upper"] is None:
        return False
    return stat["lower"] > 0.0 or stat["upper"] < 0.0


def _ci_within(stat, low_bound, high_bound):
    if stat["lower"] is None or stat["upper"] is None:
        return False
    return stat["lower"] >= low_bound and stat["upper"] <= high_bound


def _has_valid_draws(stat, floor=BOOTSTRAP_MIN_VALID_DRAWS):
    vd = stat.get("valid_draws")
    return vd is not None and vd >= floor


def _load_and_verify_all_inputs(root: Path, protocol: Mapping[str, Any], refs: Mapping[str, Any]):
    """Authenticate all 15 Stage 1-4 artifacts and enforce cross-stage lineage consistency."""
    plan = load_phase2_plan(root, protocol, refs["phase2_plan"])
    check(plan["confirm_excluded"] is True, "phase2_plan must exclude CONFIRM")
    plan_id = plan["plan_identity"]
    ds_fp = plan["dataset_fingerprint"]
    lineage_id = plan["lineage_identity"]

    e07_audit_ref, e07_audit_body, _ = load_e07_audit(
        root,
        protocol,
        refs["e07_audit"],
        expected_dataset=ds_fp,
        expected_lineage_identity=lineage_id,
    )
    if plan["e07_audit"] is not None:
        check(plain(e07_audit_ref) == plain(plan["e07_audit"]), "e07_audit ref mismatch with phase2_plan")
    check(e07_audit_body["confirm_excluded"] is True, "e07_audit confirm_excluded must be True")

    e17_auth_ref, e17_auth_body, _ = load_phase2_artifact(
        root,
        protocol,
        refs["e17_auth"],
        KIND_E17_AUTH,
        expected_schema=SCHEMA_E17_AUTH,
        expected_dataset=ds_fp,
    )
    if plan["e17_auth"] is not None:
        check(plain(e17_auth_ref) == plain(plan["e17_auth"]), "e17_auth ref mismatch with phase2_plan")
    check(e17_auth_body["permission"] == E17_DEV_C_PERMISSION, "Invalid E17 permission")
    check(e17_auth_body["lineage_identity"] == lineage_id, "e17_auth lineage_identity mismatch")
    check(e17_auth_body["confirm_excluded"] is True, "e17_auth confirm_excluded must be True")
    for m in MODELS:
        diag = e17_auth_body["models"][m]["diagonal_reproduction_check"]
        check(diag.get("passed") is True and float(diag.get("max_abs_diff", -1.0)) == 0.0, f"e17_auth diagonal check failed for {m}")

    models_loaded = {}
    integrity = {
        "plan_identity": plan_id,
        "dataset_fingerprint": ds_fp,
        "confirm_excluded_all": True,
        "e07_verdict": e07_audit_body["verdict"],
        "models": {},
    }

    for model in MODELS:
        mrefs = refs["models"][model]

        # E29 gate & analysis
        gate_ref, gate_body, _ = load_e29_gate(root, protocol, mrefs["e29_gate"], plan=plan, expected_model=model)
        e29_ref, e29_body, _ = load_phase2_artifact(
            root, protocol, mrefs["e29_analysis"], KIND_ANALYSIS,
            expected_schema=SCHEMA_ANALYSIS, expected_dataset=ds_fp,
            expected_plan_identity=plan_id, expected_owner=model,
        )
        check(e29_body.get("experiment") == "E29", f"Expected E29 analysis for {model}")
        check(e29_body.get("confirm_excluded") is True, f"E29 confirm_excluded failed for {model}")
        check(plain(e29_body.get("gate_ref")) == plain(gate_ref), f"E29 gate_ref mismatch for {model}")

        # E17 analysis
        e17_ref, e17_body, _ = load_phase2_artifact(
            root, protocol, mrefs["e17_analysis"], KIND_ANALYSIS,
            expected_schema=SCHEMA_ANALYSIS, expected_dataset=ds_fp,
            expected_plan_identity=plan_id, expected_owner=model,
        )
        check(e17_body.get("experiment") == "E17", f"Expected E17 analysis for {model}")
        check(e17_body.get("confirm_excluded") is True, f"E17 confirm_excluded failed for {model}")
        check(plain(e17_body.get("e17_auth_ref")) == plain(e17_auth_ref), f"E17 auth_ref mismatch for {model}")
        check(e17_body.get("e19_cutoffs_untouched") is True, f"E17 modified E19 cutoffs for {model}")
        e17_inv = e17_body["invariance_checks"]
        check(
            e17_inv.get("all_T_positive") is True
            and int(e17_inv.get("temp_only_0p5_flips", -1)) == 0
            and float(e17_inv.get("max_abs_auroc_diff", 1.0)) <= E17_INVARIANCE_TOL
            and float(e17_inv.get("max_abs_ap_diff", 1.0)) <= E17_INVARIANCE_TOL,
            f"E17 invariance check failed for {model}: {e17_inv}",
        )

        # E07 release
        e07_rel_ref, e07_rel_body, _ = load_phase2_artifact(
            root, protocol, mrefs["e07_release"], KIND_E07_RELEASE,
            expected_schema=SCHEMA_E07_FEATURES, expected_dataset=ds_fp,
            expected_plan_identity=plan_id, expected_owner=model,
        )
        check(e07_rel_body.get("confirm_excluded") is True, f"E07 release confirm_excluded failed for {model}")
        check(plain(e07_rel_body.get("e07_audit_ref")) == plain(e07_audit_ref), f"E07 release audit_ref mismatch for {model}")
        check(e07_rel_body.get("repeat_check_summary", {}).get("all_passed") is True, f"E07 repeat check failed for {model}")

        # E30 analysis
        e30_ref, e30_body, _ = load_phase2_artifact(
            root, protocol, mrefs["e30_analysis"], KIND_ANALYSIS,
            expected_schema=SCHEMA_ANALYSIS, expected_dataset=ds_fp,
            expected_plan_identity=plan_id, expected_owner=model,
        )
        check(e30_body.get("experiment") == "E30", f"Expected E30 analysis for {model}")
        check(e30_body.get("confirm_excluded") is True, f"E30 confirm_excluded failed for {model}")
        check(plain(e30_body.get("gate_ref")) == plain(gate_ref), f"E30 gate_ref mismatch for {model}")
        check(gate_body["decisions"]["E30"]["decision"] == "OPEN", f"E30 ran on non-OPEN gate for {model}")

        # E11 analysis
        e11_ref, e11_body, _ = load_phase2_artifact(
            root, protocol, mrefs["e11_analysis"], KIND_ANALYSIS,
            expected_schema=SCHEMA_ANALYSIS, expected_dataset=ds_fp,
            expected_plan_identity=plan_id, expected_owner=model,
        )
        check(e11_body.get("experiment") == "E11", f"Expected E11 analysis for {model}")
        check(e11_body.get("confirm_excluded") is True, f"E11 confirm_excluded failed for {model}")
        check(plain(e11_body.get("gate_ref")) == plain(gate_ref), f"E11 gate_ref mismatch for {model}")
        check(gate_body["decisions"]["E11"]["decision"] == "OPEN", f"E11 ran on non-OPEN gate for {model}")
        e11_inv = e11_body["invariance_checks"]
        check(e11_inv.get("passed") is True, f"E11 mean_shift invariance failed for {model}")
        check(e11_body["identity_self_check"].get("passed") is True, f"E11 identity self-check failed for {model}")
        check(
            e11_body["fit_pairing_check"].get("paired_fit_images") == 2160
            and e11_body["fit_pairing_check"].get("fit_rows_match_across_languages") is True,
            f"E11 FIT pairing check failed for {model}",
        )

        # E07 analysis
        e07_ref, e07_body, _ = load_phase2_artifact(
            root, protocol, mrefs["e07_analysis"], KIND_ANALYSIS,
            expected_schema=SCHEMA_ANALYSIS, expected_dataset=ds_fp,
            expected_plan_identity=plan_id, expected_owner=model,
        )
        check(e07_body.get("experiment") == "E07", f"Expected E07 analysis for {model}")
        check(e07_body.get("confirm_excluded") is True, f"E07 confirm_excluded failed for {model}")
        check(plain(e07_body.get("e07_audit_ref")) == plain(e07_audit_ref), f"E07 analysis audit_ref mismatch for {model}")
        check(plain(e07_body.get("e07_release_ref")) == plain(e07_rel_ref), f"E07 analysis release_ref mismatch for {model}")
        check(plain(e07_body.get("e29_analysis_ref")) == plain(e29_ref), f"E07 analysis e29_ref mismatch for {model}")
        check(e07_body["draw0_parity_check"].get("passed") is True, f"E07 draw0 parity check failed for {model}")

        models_loaded[model] = {
            "e29_gate_ref": plain(gate_ref),
            "e29_gate": plain(gate_body),
            "e29_ref": plain(e29_ref),
            "e29": plain(e29_body),
            "e17_ref": plain(e17_ref),
            "e17": plain(e17_body),
            "e07_release_ref": plain(e07_rel_ref),
            "e07_release": plain(e07_rel_body),
            "e30_ref": plain(e30_ref),
            "e30": plain(e30_body),
            "e11_ref": plain(e11_ref),
            "e11": plain(e11_body),
            "e07_ref": plain(e07_ref),
            "e07": plain(e07_body),
        }
        integrity["models"][model] = {
            "e29_valid_draws": int(gate_body["observed"]["valid_draws"]),
            "e29_gates": plain(gate_body["decisions"]),
            "e17_invariance_passed": True,
            "e17_max_abs_auroc_diff": float(e17_inv["max_abs_auroc_diff"]),
            "e17_max_abs_ap_diff": float(e17_inv["max_abs_ap_diff"]),
            "e17_temp_only_0p5_flips": int(e17_inv["temp_only_0p5_flips"]),
            "e07_release_repeat_passed": True,
            "e07_draw0_parity_max_abs_diff": float(e07_body["draw0_parity_check"]["e29_draw0_max_abs_diff"]),
            "e11_invariance_passed": True,
            "e11_mean_shift_max_abs_auroc_diff": float(e11_inv["mean_shift_max_abs_auroc_diff"]),
            "e11_paired_fit_images": int(e11_body["fit_pairing_check"]["paired_fit_images"]),
            "e30_collapse_flagged": bool(e30_body["off_distribution_collapse"]["flagged"]),
        }

    return {
        "plan": plan,
        "e07_audit_ref": plain(e07_audit_ref),
        "e07_audit": plain(e07_audit_body),
        "e17_auth_ref": plain(e17_auth_ref),
        "e17_auth": plain(e17_auth_body),
        "models": models_loaded,
        "integrity": integrity,
    }


def _summarize_model_experiments(model: str, mdata: Mapping[str, Any]):
    """Extract compact, uniform summaries from the 5 verified analysis blocks for one model."""
    e29 = mdata["e29"]
    e29_ic = e29["image_caption"]["twelve_route_means"]["auroc"]
    e29_co = e29["caption_only"]["twelve_route_means"]["auroc"]
    e29_summary = {
        "image_caption_g_last": _extract_stat(e29_ic["g_last_mean"]),
        "image_caption_g_res": _extract_stat(e29_ic["g_res_mean"]),
        "image_caption_delta_g": _extract_stat(e29_ic["delta_g_mean"]),
        "caption_only_g_last": _extract_stat(e29_co["g_last_mean"]),
        "caption_only_g_res": _extract_stat(e29_co["g_res_mean"]),
        "caption_only_delta_g": _extract_stat(e29_co["delta_g_mean"]),
        "image_caption_nll_delta_g": _extract_stat(e29["image_caption"]["twelve_route_means"]["nll"]["delta_g_mean"]),
        "image_caption_brier_delta_g": _extract_stat(e29["image_caption"]["twelve_route_means"]["brier"]["delta_g_mean"]),
        "gates": plain(mdata["e29_gate"]["decisions"]),
    }

    e30 = mdata["e30"]
    c30 = e30["contrasts"]
    e30_summary = {
        "feature_gain_fixed_head_last_auroc": _extract_stat(
            c30["feature_gain_fixed_head_last"]["metrics"]["auroc"]["gap_reduction"]
        ),
        "feature_gain_fixed_head_mean_auroc": _extract_stat(
            c30["feature_gain_fixed_head_mean"]["metrics"]["auroc"]["gap_reduction"]
        ),
        "head_gain_fixed_feature_last_auroc": _extract_stat(
            c30["head_gain_fixed_feature_last"]["metrics"]["auroc"]["gap_reduction"]
        ),
        "head_gain_fixed_feature_mean_auroc": _extract_stat(
            c30["head_gain_fixed_feature_mean"]["metrics"]["auroc"]["gap_reduction"]
        ),
        "interaction_auroc": _extract_stat(c30["interaction"]["metrics"]["auroc"]["gap_reduction"]),
        "total_matched_reduction_auroc": _extract_stat(
            c30["total_matched_reduction"]["metrics"]["auroc"]["gap_reduction"]
        ),
        "off_distribution_collapse": plain(e30["off_distribution_collapse"]),
    }

    e11 = mdata["e11"]
    s11 = e11["summaries"]
    e11_summary = {
        "mean_shift_auroc_gap_reduction": _extract_stat(
            s11["paired_gains_vs_identity"]["mean_shift"]["auroc"]["twelve_route_mean_gap_reduction"]
        ),
        "mean_scale_auroc_gap_reduction": _extract_stat(
            s11["paired_gains_vs_identity"]["mean_scale"]["auroc"]["twelve_route_mean_gap_reduction"]
        ),
        "mean_scale_recovered_fraction": _extract_stat(
            s11["recovered_fraction_of_g_res_mean"]["mean_scale"]["auroc"]
        ),
        "mean_shift_nll_gain": _extract_stat(
            s11["paired_gains_vs_identity"]["mean_shift"]["nll"]["twelve_route_mean_absolute_improvement"]
        ),
        "mean_shift_brier_gain": _extract_stat(
            s11["paired_gains_vs_identity"]["mean_shift"]["brier"]["twelve_route_mean_absolute_improvement"]
        ),
        "mean_scale_nll_gain": _extract_stat(
            s11["paired_gains_vs_identity"]["mean_scale"]["nll"]["twelve_route_mean_absolute_improvement"]
        ),
        "mean_scale_brier_gain": _extract_stat(
            s11["paired_gains_vs_identity"]["mean_scale"]["brier"]["twelve_route_mean_absolute_improvement"]
        ),
        "invariance_checks": plain(e11["invariance_checks"]),
    }

    e17 = mdata["e17"]
    s17 = e17["summaries"]
    d17 = e17["decision_0p5"]
    e17_arms = {}
    for arm in ("source_dev_c:temp", "source_dev_c:temp_intercept", "target_dev_c:temp", "target_dev_c:temp_intercept"):
        off = s17[arm]["off_diagonal_12_route_mean"]
        e17_arms[arm] = {
            "nll_gain_vs_raw": _extract_stat(off["gain_vs_raw"]["nll"]),
            "brier_gain_vs_raw": _extract_stat(off["gain_vs_raw"]["brier"]),
            "nll_gain_vs_fit_prior": _extract_stat(off["gain_vs_fit_prior"]["nll"]),
            "brier_gain_vs_fit_prior": _extract_stat(off["gain_vs_fit_prior"]["brier"]),
            "flip_rate_0p5": float(d17[arm]["off_diagonal_12_route_mean"]["flip_rate"]),
        }
    e17_summary = {
        "raw_nll": _extract_stat(s17["raw"]["off_diagonal_12_route_mean"]["primary_macro"]["nll"]),
        "raw_brier": _extract_stat(s17["raw"]["off_diagonal_12_route_mean"]["primary_macro"]["brier"]),
        "fit_prior_nll": _extract_stat(e17["fit_prior_reference"]["off_diagonal_12_route_mean"]["nll"]),
        "fit_prior_brier": _extract_stat(e17["fit_prior_reference"]["off_diagonal_12_route_mean"]["brier"]),
        "arms": e17_arms,
        "invariance_checks": plain(e17["invariance_checks"]),
    }

    e07 = mdata["e07"]
    fs_shifts = e07["full_screen_360"]["paired_shifts"]
    rc_draws = e07["restricted_changed_cohort"]["draws"]
    e07_draws = {}
    for dk in ("draw_1", "draw_2"):
        shift_key = f"{dk}_vs_draw_0"
        fs = fs_shifts[shift_key]
        rc_agg = rc_draws[dk]["eligible_targets_aggregate"]
        rc_macros = rc_agg.get("macros", {})
        rc_primary = rc_macros.get("primary_14") or rc_macros.get("supported_primary_ge_10") or {}
        rc_auroc = rc_primary.get("metrics", {}).get("auroc", {}) if isinstance(rc_primary, Mapping) else {}
        e07_draws[dk] = {
            "full_12_g_last_shift": _extract_stat(fs["twelve_route_mean"]["auroc"]["g_last_shift"]),
            "full_12_g_res_shift": _extract_stat(fs["twelve_route_mean"]["auroc"]["g_res_shift"]),
            "full_12_delta_g_shift": _extract_stat(fs["twelve_route_mean"]["auroc"]["delta_g_shift"]),
            "changed_9_g_last_shift": _extract_stat(fs["changed_targets_only"]["means"]["auroc"]["g_last_shift"]),
            "changed_9_g_res_shift": _extract_stat(fs["changed_targets_only"]["means"]["auroc"]["g_res_shift"]),
            "changed_9_delta_g_shift": _extract_stat(fs["changed_targets_only"]["means"]["auroc"]["delta_g_shift"]),
            "sign_preserved_12": bool(
                fs["pooling_reduction_sign_preserved"]["twelve_route_mean"].get(
                    "preserved", fs["pooling_reduction_sign_preserved"]["twelve_route_mean"].get("sign_preserved")
                )
            ),
            "sign_preserved_9": bool(
                fs["pooling_reduction_sign_preserved"]["changed_targets_only_mean"].get(
                    "preserved", fs["pooling_reduction_sign_preserved"]["changed_targets_only_mean"].get("sign_preserved")
                )
            ),
            "restricted_g_last_shift": _extract_stat(rc_auroc.get("g_last_shift")),
            "restricted_g_res_shift": _extract_stat(rc_auroc.get("g_res_shift")),
            "restricted_delta_g_shift": _extract_stat(rc_auroc.get("delta_g_shift")),
            "restricted_n_routes": int(rc_agg.get("n_routes", 0)),
        }
    e07_summary = {
        "verdict": e07["e07_verdict"],
        "eligible_changed_languages": list(e07["coverage_summary"]["eligible_changed_languages"]),
        "unavailable_languages": list(e07["coverage_summary"]["unavailable_languages"]),
        "draw0_parity_check": plain(e07["draw0_parity_check"]),
        "draws": e07_draws,
    }

    return {
        "model": model,
        "E29": e29_summary,
        "E30": e30_summary,
        "E11": e11_summary,
        "E17": e17_summary,
        "E07": e07_summary,
    }


def evaluate_model_decisions(model: str, summary: Mapping[str, Any]):
    """Evaluate all Phase 2 directions for one model against frozen STAGE5_DECISION_RANGES."""
    decisions = {}

    # 1. E29 — Paired pooling reduction (context for E30/E11)
    e29 = summary["E29"]
    dg = e29["image_caption_delta_g"]
    dg_co = e29["caption_only_delta_g"]
    g_res = e29["image_caption_g_res"]
    co_mirror = (
        dg["value"] is not None
        and dg_co["value"] is not None
        and dg["value"] > 0
        and dg_co["value"] >= 0.5 * dg["value"]
    )
    if (
        _has_valid_draws(dg)
        and dg["value"] is not None
        and dg["value"] >= E29_GATE_THRESHOLDS["E30"]["open_point_min"]
        and dg["lower"] is not None
        and dg["lower"] > E29_GATE_THRESHOLDS["E30"]["open_ci_lower_gt"]
    ):
        e29_dec = "revise" if co_mirror else "retain"
    elif _has_valid_draws(dg) and dg["upper"] is not None and dg["upper"] < E29_GATE_THRESHOLDS["E30"]["park_ci_upper_lt"]:
        e29_dec = "park within tested scope"
    else:
        e29_dec = "insufficient evidence"
    decisions["E29_pooling_gap_reduction"] = {
        "experiment": "E29",
        "model": model,
        "decision": e29_dec,
        "exact_comparison": "image_caption midpoint 12-route mean AUROC gap reduction Delta_G = G(last_caption) - G(caption_mean)",
        "effect": dg["value"],
        "ci_95": [dg["lower"], dg["upper"]],
        "valid_draws": dg["valid_draws"],
        "denominators": {"n_images": 360, "n_routes": 12, "n_primary_categories": 14},
        "invariance_and_sanity_checks": {"gates": e29["gates"]},
        "strongest_contrary_evidence": (
            f"caption_only diagnostic Delta_G = {_fmt_stat(dg_co)}; "
            f"residual image_caption G_res = {_fmt_stat(g_res)} remains above zero."
        ),
        "remaining_alternative_explanation": (
            "Independent head fits at last_caption vs caption_mean (tested by E30); "
            "caption-only lexical pooling contribution if caption_only mirrors image_caption."
        ),
    }

    # 2. E30 — Head x pooling swap
    e30 = summary["E30"]
    r30 = STAGE5_DECISION_RANGES["E30"]
    fg_last = e30["feature_gain_fixed_head_last_auroc"]
    fg_mean = e30["feature_gain_fixed_head_mean_auroc"]
    inter = e30["interaction_auroc"]
    collapse = e30["off_distribution_collapse"]
    fg_min = r30["consequential"]["fixed_head_feature_gain_min"]
    inter_min = r30["consequential"]["abs_interaction_min"]
    neg_lo, neg_hi = r30["negligible"]["fixed_head_feature_gain_ci_within"]

    fg_last_conseq = (
        _has_valid_draws(fg_last)
        and fg_last["value"] is not None
        and fg_last["value"] >= fg_min
        and _ci_excludes_zero_positive(fg_last)
    )
    fg_mean_conseq = (
        _has_valid_draws(fg_mean)
        and fg_mean["value"] is not None
        and fg_mean["value"] >= fg_min
        and _ci_excludes_zero_positive(fg_mean)
    )
    inter_conseq = (
        _has_valid_draws(inter)
        and inter["value"] is not None
        and abs(inter["value"]) >= inter_min
        and _ci_excludes_zero_either(inter)
    )
    all_negligible = (
        _has_valid_draws(fg_last)
        and _has_valid_draws(fg_mean)
        and _has_valid_draws(inter)
        and _ci_within(fg_last, neg_lo, neg_hi)
        and _ci_within(fg_mean, neg_lo, neg_hi)
        and _ci_within(inter, neg_lo, neg_hi)
    )
    if collapse.get("flagged"):
        e30_dec = "revise"
    elif fg_last_conseq and fg_mean_conseq and not inter_conseq:
        e30_dec = "retain"
    elif fg_last_conseq or fg_mean_conseq or inter_conseq:
        e30_dec = "revise" if inter_conseq else "retain"
    elif all_negligible:
        e30_dec = "park within tested scope"
    else:
        e30_dec = "insufficient evidence"
    decisions["E30_pooling_feature_vs_head"] = {
        "experiment": "E30",
        "model": model,
        "decision": e30_dec,
        "exact_comparison": "2x2 head x pooling swap at midpoint image_caption (fixed-head feature gains & paired interaction on 12-route mean AUROC gap)",
        "effect": fg_last["value"],
        "ci_95": [fg_last["lower"], fg_last["upper"]],
        "valid_draws": fg_last["valid_draws"],
        "secondary_effects": {
            "feature_gain_fixed_head_mean": fg_mean,
            "interaction": inter,
        },
        "denominators": {"n_images": 360, "n_routes": 12, "n_primary_categories": 14},
        "invariance_and_sanity_checks": {
            "off_distribution_collapse_flagged": bool(collapse.get("flagged")),
            "head_mean_on_last_dropped": bool(
                collapse.get("head_mean_on_last_vs_last_on_last", {})
                .get("transfer_12route_auroc", {})
                .get("dropped_ge_threshold", False)
            ),
        },
        "strongest_contrary_evidence": (
            f"Interaction = {_fmt_stat(inter)} and fixed-head-mean feature gain = {_fmt_stat(fg_mean)}; "
            f"asymmetric head_mean_on_last drop (collapse_flagged={collapse.get('flagged')})."
        ),
        "remaining_alternative_explanation": (
            "Asymmetric feature-distribution compatibility: last_caption heads benefit directly from caption_mean "
            "features without refitting, whereas caption_mean heads degrade when evaluated on last_caption features."
        ),
    }

    # 3. E11 — FIT-only mean/scale covariate adjustment
    e11 = summary["E11"]
    r11 = STAGE5_DECISION_RANGES["E11"]
    ms_auroc = e11["mean_scale_auroc_gap_reduction"]
    ms_frac = e11["mean_scale_recovered_fraction"]
    ms_nll = e11["mean_scale_nll_gain"]
    ms_brier = e11["mean_scale_brier_gain"]
    auroc_conseq = (
        _has_valid_draws(ms_auroc)
        and ms_auroc["value"] is not None
        and _ci_excludes_zero_positive(ms_auroc)
        and (
            ms_auroc["value"] >= r11["consequential"]["mean_scale_auroc_gap_reduction_min"]
            or (
                ms_frac["value"] is not None
                and ms_frac["value"] >= r11["consequential"]["or_fraction_of_g_res_mean_min"]
            )
        )
    )
    loss_only_conseq = (
        (
            _has_valid_draws(ms_nll)
            and ms_nll["value"] is not None
            and ms_nll["value"] >= r11["consequential"]["loss_only_nll_gain_min"]
            and _ci_excludes_zero_positive(ms_nll)
        )
        or (
            _has_valid_draws(ms_brier)
            and ms_brier["value"] is not None
            and ms_brier["value"] >= r11["consequential"]["loss_only_brier_gain_min"]
            and _ci_excludes_zero_positive(ms_brier)
        )
    )
    auroc_negligible = (
        _has_valid_draws(ms_auroc)
        and ms_auroc["upper"] is not None
        and ms_auroc["upper"] < r11["negligible"]["mean_scale_auroc_gap_reduction_ci_upper_lt"]
    )
    if auroc_conseq:
        e11_dec = "retain"
    elif loss_only_conseq:
        e11_dec = "revise"
    elif auroc_negligible:
        e11_dec = "park within tested scope"
    else:
        e11_dec = "insufficient evidence"
    decisions["E11_fit_covariate_scale"] = {
        "experiment": "E11",
        "model": model,
        "decision": e11_dec,
        "exact_comparison": "target_fit_input_adapted mean_scale vs identity on caption_mean midpoint (12-route mean AUROC gap reduction; NLL/Brier companions)",
        "effect": ms_auroc["value"],
        "ci_95": [ms_auroc["lower"], ms_auroc["upper"]],
        "valid_draws": ms_auroc["valid_draws"],
        "secondary_effects": {
            "recovered_fraction_of_g_res": ms_frac,
            "mean_scale_nll_gain": ms_nll,
            "mean_scale_brier_gain": ms_brier,
        },
        "denominators": {"n_images": 360, "n_fit_paired_images": 2160, "n_routes": 12, "n_primary_categories": 14},
        "invariance_and_sanity_checks": e11["invariance_checks"],
        "strongest_contrary_evidence": (
            f"mean_scale AUROC gap reduction = {_fmt_stat(ms_auroc)} (recovered fraction = {_fmt_stat(ms_frac, signed=False)}); "
            f"NLL gain = {_fmt_stat(ms_nll)}, Brier gain = {_fmt_stat(ms_brier)}."
        ),
        "remaining_alternative_explanation": (
            "Diagonal coordinate scaling cannot correct cross-lingual rotation/subspace misalignment (E12 candidate); "
            "loss-only shifts belong with probability calibration (E17); draw-0 only until paired with E07."
        ),
    }

    # 4 & 5. E17 — Source DEV-C calibration and Target DEV-C calibration (separate verdicts per §1.5)
    e17 = summary["E17"]
    r17 = STAGE5_DECISION_RANGES["E17"]

    def _eval_e17_scope(scope_prefix, dir_id, label):
        best_arm = None
        best_score = -1e9
        for form in ("temp", "temp_intercept"):
            arm = f"{scope_prefix}:{form}"
            info = e17["arms"][arm]
            nll_g = info["nll_gain_vs_raw"]["value"] or -1e9
            if nll_g > best_score:
                best_score = nll_g
                best_arm = arm
        info = e17["arms"][best_arm]
        nll_raw = info["nll_gain_vs_raw"]
        brier_raw = info["brier_gain_vs_raw"]
        nll_fp = info["nll_gain_vs_fit_prior"]
        brier_fp = info["brier_gain_vs_fit_prior"]

        conseq_raw = (
            (_has_valid_draws(nll_raw) and nll_raw["value"] is not None and nll_raw["value"] >= r17["consequential"]["nll_gain_min"] and _ci_excludes_zero_positive(nll_raw))
            or (_has_valid_draws(brier_raw) and brier_raw["value"] is not None and brier_raw["value"] >= r17["consequential"]["brier_gain_min"] and _ci_excludes_zero_positive(brier_raw))
        )
        beats_fp = (
            (_has_valid_draws(nll_fp) and _ci_excludes_zero_positive(nll_fp))
            or (_has_valid_draws(brier_fp) and _ci_excludes_zero_positive(brier_fp))
        )
        negligible = (
            _has_valid_draws(nll_raw)
            and _has_valid_draws(brier_raw)
            and nll_raw["upper"] is not None
            and brier_raw["upper"] is not None
            and nll_raw["upper"] < r17["negligible"]["nll_ci_upper_lt"]
            and brier_raw["upper"] < r17["negligible"]["brier_ci_upper_lt"]
        )
        if conseq_raw and beats_fp:
            dec = "retain" if scope_prefix == "source_dev_c" else "revise"
        elif conseq_raw:
            dec = "revise"
        elif negligible:
            dec = "park within tested scope"
        else:
            dec = "insufficient evidence"
        decisions[dir_id] = {
            "experiment": "E17",
            "model": model,
            "decision": dec,
            "exact_comparison": f"{label} ({best_arm}) vs raw & FIT-prior on midpoint last_caption SCREEN (12-route mean NLL & Brier gain)",
            "best_arm": best_arm,
            "effect": nll_raw["value"],
            "ci_95": [nll_raw["lower"], nll_raw["upper"]],
            "valid_draws": nll_raw["valid_draws"],
            "secondary_effects": {
                "brier_gain_vs_raw": brier_raw,
                "nll_gain_vs_fit_prior": nll_fp,
                "brier_gain_vs_fit_prior": brier_fp,
                "flip_rate_0p5": info["flip_rate_0p5"],
            },
            "denominators": {"n_images": 360, "n_dev_c_images": 180, "n_routes": 12, "n_primary_categories": 14},
            "invariance_and_sanity_checks": e17["invariance_checks"],
            "strongest_contrary_evidence": (
                f"Ranking (AUROC/AP) is strictly invariant (max_diff={e17['invariance_checks']['max_abs_auroc_diff']:.2e}); "
                f"0.5-decision flip rate on {best_arm} = {info['flip_rate_0p5']:.4f}; "
                f"NLL gain vs FIT-prior = {_fmt_stat(nll_fp)}."
            ),
            "remaining_alternative_explanation": (
                "Global 1- or 2-parameter logit scaling cannot fix per-category or ranking transfer errors; "
                + ("target DEV-C requires 180 labeled target examples." if scope_prefix == "target_dev_c" else "source DEV-C may under-correct target-specific logit shifts.")
            ),
        }

    _eval_e17_scope("source_dev_c", "E17_source_calibration", "source_dev_c_calibrated")
    _eval_e17_scope("target_dev_c", "E17_target_calibration", "target_dev_c_supervised_calibration")

    # 6. E07 — Alternate caption view stability (dual cohort: full_screen_360 + restricted_changed_cohort)
    e07 = summary["E07"]
    r07 = STAGE5_DECISION_RANGES["E07"]
    abs_min = r07["consequential"]["abs_gap_shift_min"]
    neg_lo7, neg_hi7 = r07["negligible"]["ci_within"]

    # Inspect worst-case shift across draws 1 & 2 on both full_12 and changed_9 (plus restricted cohort)
    checks_e07 = []
    any_sign_reversed = False
    for dk in ("draw_1", "draw_2"):
        d = e07["draws"][dk]
        if not d["sign_preserved_12"] or not d["sign_preserved_9"]:
            any_sign_reversed = True
        for key in ("full_12_g_last_shift", "full_12_g_res_shift", "changed_9_g_last_shift", "changed_9_g_res_shift"):
            checks_e07.append((dk, key, d[key]))

    worst_dk, worst_key, worst_stat = max(
        checks_e07,
        key=lambda item: abs(item[2]["value"]) if item[2]["value"] is not None else -1.0,
    )
    sensitive = any_sign_reversed or any(
        _has_valid_draws(st)
        and st["value"] is not None
        and abs(st["value"]) >= abs_min
        and _ci_excludes_zero_either(st)
        for _, _, st in checks_e07
    )
    all_stable = (not any_sign_reversed) and all(
        _has_valid_draws(st)
        and st["value"] is not None
        and abs(st["value"]) < r07["negligible"]["abs_gap_shift_lt"]
        and _ci_within(st, neg_lo7, neg_hi7)
        for _, _, st in checks_e07
    )
    restricted_coverage = e07["verdict"] == "passed_restricted" or bool(e07["unavailable_languages"])
    if sensitive:
        e07_dec = "revise"
    elif all_stable:
        e07_dec = "revise" if restricted_coverage else "retain"
    else:
        e07_dec = "insufficient evidence"

    decisions["E07_caption_view_stability"] = {
        "experiment": "E07",
        "model": model,
        "decision": e07_dec,
        "exact_comparison": f"Frozen-head alternate-caption draws 1 & 2 vs draw 0 ({worst_dk} {worst_key} worst-case AUROC gap shift)",
        "effect": worst_stat["value"],
        "ci_95": [worst_stat["lower"], worst_stat["upper"]],
        "valid_draws": worst_stat["valid_draws"],
        "denominators": {
            "n_images": 360,
            "eligible_changed_languages": e07["eligible_changed_languages"],
            "unavailable_languages": e07["unavailable_languages"],
            "n_changed_target_routes": 9,
            "n_primary_categories": 14,
        },
        "invariance_and_sanity_checks": e07["draw0_parity_check"],
        "strongest_contrary_evidence": (
            f"Verdict={e07['verdict']}: unavailable single-caption languages={e07['unavailable_languages']} "
            f"(bn has 0 alternate captions in XM3600 SCREEN); sign_reversed={any_sign_reversed}; "
            f"draw_1 restricted g_last_shift={_fmt_stat(e07['draws']['draw_1']['restricted_g_last_shift'])}."
        ),
        "remaining_alternative_explanation": (
            "Frozen-head E07 tests evaluation-view sensitivity with heads trained on draw 0; "
            "it does not test training-view refit sensitivity or separate wording from caption content."
        ),
    }

    return decisions


def _combine_decisions(per_model_decisions: Mapping[str, Mapping[str, Any]]):
    """Combine Qwen and Gemma verdicts per direction conservatively."""
    direction_ids = list(next(iter(per_model_decisions.values())).keys())
    combined = {}
    for did in direction_ids:
        q_dec = per_model_decisions["qwen"][did]["decision"]
        g_dec = per_model_decisions["gemma"][did]["decision"]
        if q_dec == g_dec:
            overall = q_dec
        elif "retain" in (q_dec, g_dec) or "revise" in (q_dec, g_dec):
            overall = "revise"
        elif "insufficient evidence" in (q_dec, g_dec):
            overall = "insufficient evidence"
        else:
            overall = "park within tested scope"
        combined[did] = {
            "decision": overall,
            "qwen": q_dec,
            "gemma": g_dec,
            "experiment": per_model_decisions["qwen"][did]["experiment"],
        }
    return combined


def _build_post_prelim_recommendation(combined: Mapping[str, Mapping[str, Any]]):
    surviving = [k for k, v in combined.items() if v["decision"] in ("retain", "revise")]
    parked = [k for k, v in combined.items() if v["decision"] == "park within tested scope"]
    unresolved = [k for k, v in combined.items() if v["decision"] == "insufficient evidence"]
    ready = len(surviving) > 0 and len(unresolved) <= 1
    return {
        "recommendation": "proceed_to_post_prelim" if ready else "optional_phase3_needed",
        "surviving_directions": surviving,
        "parked_directions": parked,
        "unresolved_directions": unresolved,
        "confirm_status": "CLOSED (untouched throughout Phase 1 and Phase 2)",
    }


def _render_markdown_review(body: Mapping[str, Any]) -> str:
    lines = [
        "# Phase 2 Results Review & Decision Register — G9–Q6",
        "",
        f"- **Release schema:** `{body['schema']}` (`{body['stage5_version']}`)",
        f"- **Plan identity:** `{body['plan_identity']}`",
        f"- **Dataset fingerprint:** `{body['dataset_fingerprint']}`",
        f"- **CONFIRM status:** `CLOSED` (`confirm_excluded = {body['confirm_excluded']}`)",
        f"- **E07 coverage verdict:** `{body['integrity_checks']['e07_verdict']}`",
        "",
        "## 1. Combined Decision Register",
        "",
        "| Direction | Experiment | Combined Decision | Qwen | Gemma |",
        "|---|---|---|---|---|",
    ]
    for did, rec in body["decision_register"]["combined"].items():
        lines.append(
            f"| `{did}` | {rec['experiment']} | **{rec['decision']}** | `{rec['qwen']}` | `{rec['gemma']}` |"
        )

    lines += [
        "",
        "## 2. Per-Model Quantitative Summary (Primary 14-Category Macro, 95% Paired Bootstrap CI)",
        "",
    ]
    for model in MODELS:
        s = body["experiment_summaries"][model]
        e29, e30, e11, e17, e07 = s["E29"], s["E30"], s["E11"], s["E17"], s["E07"]
        lines += [
            f"### {model.upper()}",
            "",
            "| Experiment | Key Contrast / Metric | Estimate [95% CI] |",
            "|---|---|---|",
            f"| **E29** | `image_caption` ΔG = G(last) − G(mean) (AUROC) | {_fmt_stat(e29['image_caption_delta_g'])} |",
            f"| **E29** | `image_caption` residual G(mean) (AUROC) | {_fmt_stat(e29['image_caption_g_res'])} |",
            f"| **E29** | `caption_only` diagnostic ΔG (AUROC) | {_fmt_stat(e29['caption_only_delta_g'])} |",
            f"| **E30** | Fixed `last` head feature gain (AUROC) | {_fmt_stat(e30['feature_gain_fixed_head_last_auroc'])} |",
            f"| **E30** | Fixed `mean` head feature gain (AUROC) | {_fmt_stat(e30['feature_gain_fixed_head_mean_auroc'])} |",
            f"| **E30** | Head × pooling interaction (AUROC) | {_fmt_stat(e30['interaction_auroc'])} |",
            f"| **E11** | `mean_scale` AUROC gap reduction | {_fmt_stat(e11['mean_scale_auroc_gap_reduction'])} |",
            f"| **E11** | `mean_scale` recovered fraction of G_res | {_fmt_stat(e11['mean_scale_recovered_fraction'], signed=False)} |",
            f"| **E11** | `mean_scale` NLL / Brier gain vs identity | {_fmt_stat(e11['mean_scale_nll_gain'])} / {_fmt_stat(e11['mean_scale_brier_gain'])} |",
            f"| **E17** | `source_dev_c:temp` NLL / Brier gain vs raw | {_fmt_stat(e17['arms']['source_dev_c:temp']['nll_gain_vs_raw'])} / {_fmt_stat(e17['arms']['source_dev_c:temp']['brier_gain_vs_raw'])} |",
            f"| **E17** | `source_dev_c:temp_intercept` NLL / Brier gain vs raw | {_fmt_stat(e17['arms']['source_dev_c:temp_intercept']['nll_gain_vs_raw'])} / {_fmt_stat(e17['arms']['source_dev_c:temp_intercept']['brier_gain_vs_raw'])} |",
            f"| **E17** | `target_dev_c:temp` NLL / Brier gain vs raw | {_fmt_stat(e17['arms']['target_dev_c:temp']['nll_gain_vs_raw'])} / {_fmt_stat(e17['arms']['target_dev_c:temp']['brier_gain_vs_raw'])} |",
            f"| **E17** | `target_dev_c:temp_intercept` NLL / Brier gain vs raw | {_fmt_stat(e17['arms']['target_dev_c:temp_intercept']['nll_gain_vs_raw'])} / {_fmt_stat(e17['arms']['target_dev_c:temp_intercept']['brier_gain_vs_raw'])} |",
            f"| **E07** | Draw 1 vs 0 full-12 / changed-9 ΔG_last (AUROC) | {_fmt_stat(e07['draws']['draw_1']['full_12_g_last_shift'])} / {_fmt_stat(e07['draws']['draw_1']['changed_9_g_last_shift'])} |",
            f"| **E07** | Draw 2 vs 0 full-12 / changed-9 ΔG_last (AUROC) | {_fmt_stat(e07['draws']['draw_2']['full_12_g_last_shift'])} / {_fmt_stat(e07['draws']['draw_2']['changed_9_g_last_shift'])} |",
            f"| **E07** | Draw 1 / Draw 2 restricted-cohort ΔG_last (AUROC) | {_fmt_stat(e07['draws']['draw_1']['restricted_g_last_shift'])} / {_fmt_stat(e07['draws']['draw_2']['restricted_g_last_shift'])} |",
            "",
        ]

    lines += [
        "## 3. Detailed Decision Notes (Strongest Contrary Evidence & Alternative Explanations)",
        "",
    ]
    for model in MODELS:
        lines.append(f"### {model.upper()}")
        for did, d in body["decision_register"]["per_model"][model].items():
            lines.append(f"- **`{did}` → `{d['decision']}`**: {d['exact_comparison']}.")
            lines.append(f"  - Effect: `{_fmt_stat({'value': d['effect'], 'lower': d['ci_95'][0], 'upper': d['ci_95'][1]})}` (valid draws: `{d['valid_draws']}`).")
            lines.append(f"  - Strongest contrary evidence: {d['strongest_contrary_evidence']}")
            lines.append(f"  - Remaining alternative explanation: {d['remaining_alternative_explanation']}")
        lines.append("")

    rec = body["post_prelim_recommendation"]
    lines += [
        "## 4. Next-Step Recommendation",
        "",
        f"- **Recommendation:** `{rec['recommendation']}`",
        f"- **Surviving directions:** {', '.join(f'`{d}`' for d in rec['surviving_directions']) or 'none'}",
        f"- **Parked directions:** {', '.join(f'`{d}`' for d in rec['parked_directions']) or 'none'}",
        f"- **Unresolved directions:** {', '.join(f'`{d}`' for d in rec['unresolved_directions']) or 'none'}",
        f"- **CONFIRM:** {rec['confirm_status']}",
        "",
    ]
    return "\n".join(lines)


@contract
def verify_phase2_release(root, protocol, release_ref):
    """Independently re-authenticate a published `phase2_release` and all upstream markers."""
    resolved_root = Path(root).resolve()
    ref, body, marker = load_phase2_artifact(
        resolved_root,
        protocol,
        release_ref,
        KIND_RELEASE,
        expected_schema=SCHEMA_RELEASE,
    )
    check(body.get("confirm_excluded") is True, "phase2_release must enforce confirm_excluded=True")
    check("phase2_review.md" in marker["files"], "phase2_release missing phase2_review.md")
    review_path = within(resolved_root, ref["path"]).parent / "phase2_review.md"
    check(review_path.is_file(), "Missing phase2_review.md on disk")
    check(file_sha256(review_path) == marker["files"]["phase2_review.md"], "phase2_review.md SHA-256 mismatch")

    # Re-authenticate all 15 upstream Stage 1-4 refs from scratch.
    refs = {
        "phase2_plan": body["stage1_refs"]["phase2_plan"],
        "e07_audit": body["stage1_refs"]["e07_audit"],
        "e17_auth": body["stage1_refs"]["e17_auth"],
        "models": {
            m: {
                "e29_analysis": body["stage2_refs"]["e29"][m],
                "e29_gate": body["stage2_refs"]["e29_gates"][m],
                "e17_analysis": body["stage2_refs"]["e17"][m],
                "e07_release": body["stage2_refs"]["e07_releases"][m],
                "e30_analysis": body["stage3_refs"]["e30"][m],
                "e11_analysis": body["stage3_refs"]["e11"][m],
                "e07_analysis": body["stage4_refs"]["e07"][m],
            }
            for m in MODELS
        },
    }
    loaded = _load_and_verify_all_inputs(resolved_root, protocol, refs)
    check(loaded["plan"]["plan_identity"] == body["plan_identity"], "Readback plan_identity mismatch")
    check(loaded["plan"]["dataset_fingerprint"] == body["dataset_fingerprint"], "Readback dataset_fingerprint mismatch")

    return frozen({
        "status": "passed",
        "verified": True,
        "release_ref": plain(ref),
        "plan_identity": body["plan_identity"],
        "verified_artifact_count": 16,  # 15 Stage 1-4 refs + 1 phase2_release
        "review_sha256": marker["files"]["phase2_review.md"],
        "confirm_excluded": True,
    })


def _run_fresh_subprocess_readback(root: Path, protocol_path: Path, release_ref: Mapping[str, Any]):
    cmd = [
        sys.executable, "-u", "-m", "src.phase2.synthesize",
        "verify-release",
        "--phase1-root", str(root),
        "--protocol", str(protocol_path),
        "--release", json.dumps(plain(release_ref)),
    ]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, check=False)
    check(
        proc.returncode == 0,
        f"Fresh-subprocess independent readback failed (exit={proc.returncode}): {proc.stdout.strip()} {proc.stderr.strip()}",
    )
    last_json = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            last_json = json.loads(line)
        except json.JSONDecodeError:
            pass
    check(isinstance(last_json, dict) and last_json.get("verified") is True, "Readback subprocess did not emit verified=True")
    return last_json


@contract
def synthesize_phase2(
    root,
    protocol,
    refs,
    producer,
    *,
    run_id: Optional[str] = None,
    run_subprocess_readback: bool = True,
    protocol_path: Optional[Path] = None,
    progress=None,
):
    """Execute Phase 2 Stage 5 synthesis, publish `phase2_release`, and run fresh-subprocess readback."""
    resolved_root = Path(root).resolve()
    t0 = time.monotonic()
    _emit(progress, {"event": "stage5_start", "unit": "verify_inputs", "completed": 0, "total": 4})

    loaded = _load_and_verify_all_inputs(resolved_root, protocol, refs)
    read_s = time.monotonic() - t0
    _emit(progress, {"event": "stage5_inputs_verified", "unit": "verify_inputs", "completed": 1, "total": 4, "read_hash_s": round(read_s, 4)})

    t1 = time.monotonic()
    summaries = {m: _summarize_model_experiments(m, loaded["models"][m]) for m in MODELS}
    per_model_decisions = {m: evaluate_model_decisions(m, summaries[m]) for m in MODELS}
    combined_decisions = _combine_decisions(per_model_decisions)
    recommendation = _build_post_prelim_recommendation(combined_decisions)
    compute_s = time.monotonic() - t1
    _emit(progress, {"event": "stage5_decisions_evaluated", "unit": "evaluate_decisions", "completed": 2, "total": 4, "compute_s": round(compute_s, 4)})

    plan = loaded["plan"]
    body = {
        "schema": SCHEMA_RELEASE,
        "stage5_version": STAGE5_VERSION,
        "phase2_version": PHASE2_VERSION,
        "plan": plain(plan.reference),
        "plan_identity": plan["plan_identity"],
        "dataset_fingerprint": plan["dataset_fingerprint"],
        "protocol_fingerprint": plan["protocol_fingerprint"],
        "confirm_excluded": True,
        "stage1_refs": {
            "phase2_plan": plain(plan.reference),
            "e07_audit": loaded["e07_audit_ref"],
            "e17_auth": loaded["e17_auth_ref"],
        },
        "stage2_refs": {
            "e29": {m: loaded["models"][m]["e29_ref"] for m in MODELS},
            "e29_gates": {m: loaded["models"][m]["e29_gate_ref"] for m in MODELS},
            "e17": {m: loaded["models"][m]["e17_ref"] for m in MODELS},
            "e07_releases": {m: loaded["models"][m]["e07_release_ref"] for m in MODELS},
        },
        "stage3_refs": {
            "e30": {m: loaded["models"][m]["e30_ref"] for m in MODELS},
            "e11": {m: loaded["models"][m]["e11_ref"] for m in MODELS},
        },
        "stage4_refs": {
            "e07": {m: loaded["models"][m]["e07_ref"] for m in MODELS},
        },
        "integrity_checks": loaded["integrity"],
        "experiment_summaries": summaries,
        "decision_register": {
            "per_model": per_model_decisions,
            "combined": combined_decisions,
        },
        "post_prelim_recommendation": recommendation,
    }
    review_md = _render_markdown_review(body)

    upstream = [
        plain(plan.reference),
        loaded["e07_audit_ref"],
        loaded["e17_auth_ref"],
    ]
    for m in MODELS:
        md = loaded["models"][m]
        upstream.extend([
            md["e29_ref"], md["e29_gate_ref"], md["e17_ref"],
            md["e07_release_ref"], md["e30_ref"], md["e11_ref"], md["e07_ref"],
        ])

    t2 = time.monotonic()
    release_ref = publish_phase2(
        resolved_root,
        protocol,
        plan["dataset_fingerprint"],
        RELEASE_NAMESPACE,
        KIND_RELEASE,
        body,
        producer,
        files={"phase2_review.md": review_md.encode("utf-8")},
        upstream=upstream,
        run_id=run_id or new_run_id(),
    )
    write_s = time.monotonic() - t2
    _emit(progress, {"event": "stage5_release_published", "unit": "publish_release", "completed": 3, "total": 4, "write_s": round(write_s, 4), "release_ref": plain(release_ref)})

    readback = None
    if run_subprocess_readback:
        proto_file = protocol_path if protocol_path is not None else (REPO / "configs/protocol.json")
        readback = _run_fresh_subprocess_readback(resolved_root, proto_file, release_ref)
    else:
        readback = plain(verify_phase2_release(resolved_root, protocol, release_ref))

    _emit(progress, {
        "event": "stage5_complete",
        "status": "passed",
        "unit": "independent_readback",
        "completed": 4,
        "total": 4,
        "release_ref": plain(release_ref),
        "independent_readback": readback,
        "combined_decisions": combined_decisions,
        "post_prelim_recommendation": recommendation,
    })
    return frozen({
        "status": "passed",
        "release_ref": plain(release_ref),
        "independent_readback": readback,
        "body": body,
        "review_markdown": review_md,
    })


def _parse_json_or_file(raw: str):
    p = Path(raw)
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return json.loads(raw)


def _default_emit(event):
    print(json.dumps(event, sort_keys=True), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Phase 2 Stage 5 synthesis and independent readback.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Synthesize Stage 1-4 outputs and publish phase2_release.")
    p_run.add_argument("--phase1-root", default=None)
    p_run.add_argument("--protocol", default=str(REPO / "configs/protocol.json"))
    p_run.add_argument("--refs", default=None, help="Optional JSON string or path overriding CERTIFIED_PHASE2_REFS.")
    p_run.add_argument("--storage-scope", default="persistent-drive")
    p_run.add_argument("--run-id", default=None)
    p_run.add_argument("--print-review", action="store_true")

    p_ver = sub.add_parser("verify-release", help="Fresh-subprocess independent readback of phase2_release.")
    p_ver.add_argument("--phase1-root", default=None)
    p_ver.add_argument("--protocol", default=str(REPO / "configs/protocol.json"))
    p_ver.add_argument("--release", required=True, help="JSON {path, sha256} or path to phase2_release _COMPLETE.json.")

    args = parser.parse_args(argv)
    root = phase1_root(args.phase1_root)
    proto_path = Path(args.protocol).resolve()
    protocol = json.loads(proto_path.read_text(encoding="utf-8"))

    if args.command == "verify-release":
        rel_ref = _parse_json_or_file(args.release) if args.release.strip().startswith("{") else args.release
        out = verify_phase2_release(root, protocol, rel_ref)
        _default_emit(plain(out))
        return 0

    refs = _parse_json_or_file(args.refs) if args.refs else plain(CERTIFIED_PHASE2_REFS)
    producer = current_producer(scope=args.storage_scope)
    result = synthesize_phase2(
        root,
        protocol,
        refs,
        producer,
        run_id=args.run_id,
        run_subprocess_readback=True,
        protocol_path=proto_path,
        progress=_default_emit,
    )
    if args.print_review:
        print("\n=== PHASE 2 REVIEW MARKDOWN ===\n", flush=True)
        print(result["review_markdown"], flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        _default_emit({"status": "blocked", "blockers": [str(exc)]})
        raise SystemExit(2)
