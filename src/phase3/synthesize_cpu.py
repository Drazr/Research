"""Phase 3 CPU Track Stage A3 synthesis, decision register, Markdown review, and independent readback.

Authenticates the Stage A1 `phase3_cpu_plan` and all 8 Stage A2 CPU Track
artifacts (`E40`, `E41`, `E44`, `E12` for `qwen` and `gemma`) on mounted
`PHASE1_ROOT` (zero new scoring, zero feature reads, zero bootstrap
recomputation, zero Phase 1/2 mutation, `CONFIRM` strictly closed), evaluates
combined CPU Track decisions against the frozen `decision_ranges`, publishes
`phase3_release` (`body.json` + `phase3_cpu_review.md`) under
`phase3/releases/<run_id>/`, and verifies the release via a fresh-subprocess
independent readback.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Optional

from src.common.contracts import ContractError
from src.common.storage import new_run_id, phase1_root, within
from src.phase3.interfaces import (
    KIND_ANALYSIS,
    KIND_CALIBRATORS,
    KIND_MAPS,
    KIND_PREDICTION,
    KIND_RELEASE,
    MODELS,
    PHASE3_CPU_VERSION,
    POOLINGS,
    REPO,
    SCHEMA_ANALYSIS,
    SCHEMA_CALIBRATORS,
    SCHEMA_MAPS,
    SCHEMA_PREDICTION,
    SCHEMA_RELEASE,
    check,
    contract,
    current_producer,
    frozen,
    load_phase3_artifact,
    load_phase3_cpu_plan,
    plain,
    publish_phase3,
)

STAGE_A3_VERSION = "phase3-cpu-stageA3.v1"
RELEASE_NAMESPACE = "phase3/releases"

# Authoritative Stage A1 + Stage A2 CPU Track completion ledger.
CERTIFIED_PHASE3_CPU_REFS = frozen({
    "phase3_cpu_plan": {
        "path": "phase3/plans/20261120T020039Z-0d82ded3b7c3497e999d70a33f35c3a8/_COMPLETE.json",
        "sha256": "93cbf480e9d8c13401aaa98315444f3a4d2b3b57ec769b458a799dc269c40439",
    },
    "models": {
        "qwen": {
            "e40_analysis": {
                "path": "phase3/results/e40/qwen/20261120T022710Z-dcb5f379d6c945699384cc5c6a28271a/_COMPLETE.json",
                "sha256": "391b0827d76133b15a3124cf43a78dd86dc3a1e6d8aaceeb1800c81159bcf4b1",
            },
            "e41_calibrators": {
                "path": "phase3/calibrators/e41/qwen/20261120T030513Z-45f4432f92b74707aef02aa682fabe22/_COMPLETE.json",
                "sha256": "2d265bb4319333948109ec560233bcf9312cf624be6e583278b49c248f0dbb2a",
            },
            "e41_analysis": {
                "path": "phase3/results/e41/qwen/20261120T030513Z-45f4432f92b74707aef02aa682fabe22/_COMPLETE.json",
                "sha256": "85a591b018bf56e9cfad6a6421d96be7a173a630678a8ba301ddb201bc2a65d8",
            },
            "e44_analysis": {
                "path": "phase3/results/e44/qwen/20261120T022040Z-3b1fead173304b02baf2e7ac9231e207/_COMPLETE.json",
                "sha256": "bedef6271d1947ebbe6067cc18e3f6bfdcf440e117376e11abc82b1abad6ee76",
            },
            "e12_maps": {
                "path": "phase3/maps/e12/qwen/20261120T021916Z-7821583467f14ee68302e946d6bfa7f4/_COMPLETE.json",
                "sha256": "a1ecaf369eaa67ec9e4096cf2a4bafb9322a07bdbf2bc6d42a56dfb0f0588198",
            },
            "e12_predictions": {
                "path": "phase3/predictions/e12/qwen/20261120T021916Z-7821583467f14ee68302e946d6bfa7f4/_COMPLETE.json",
                "sha256": "06d572d33e5a149242eeeb62e8d444418ea0a11f7baed8af2e9d09a153ad80ae",
            },
            "e12_analysis": {
                "path": "phase3/results/e12/qwen/20261120T021916Z-7821583467f14ee68302e946d6bfa7f4/_COMPLETE.json",
                "sha256": "aac43985ac655c79cca064a362cdaead57c09e69168ce39e33171a7a87cdfcd5",
            },
        },
        "gemma": {
            "e40_analysis": {
                "path": "phase3/results/e40/gemma/20261120T022510Z-e37b359207654356888bc8e97647989e/_COMPLETE.json",
                "sha256": "e837d79d84f30433177fcd73775848262dbe178dc230ffc4cb44d8947c289255",
            },
            "e41_calibrators": {
                "path": "phase3/calibrators/e41/gemma/20261120T021429Z-5d8aff3bb4534b588e7f2706736330d5/_COMPLETE.json",
                "sha256": "a4b63ec484139db78e9fb3a6544106179879f9470dd7f4ea4764920083381107",
            },
            "e41_analysis": {
                "path": "phase3/results/e41/gemma/20261120T021429Z-5d8aff3bb4534b588e7f2706736330d5/_COMPLETE.json",
                "sha256": "3da3e86ca48a4ed1410fcac3febe3fb11fe3f4eabedbaff53fd0b5a0349c0775",
            },
            "e44_analysis": {
                "path": "phase3/results/e44/gemma/20261120T022654Z-96c6d0c92958462ea4f4167ea9b4d458/_COMPLETE.json",
                "sha256": "76b39534a20736a38ff27b4d8d62b405f5e0bfd14a13068af2f97bd71e6d337d",
            },
            "e12_maps": {
                "path": "phase3/maps/e12/gemma/20261120T021930Z-06357c1000a943aaaa5b21ee875baba0/_COMPLETE.json",
                "sha256": "38afaa17f780d4733d337e32c2e711bb9c587e0c6e19f978346fb293132498e2",
            },
            "e12_predictions": {
                "path": "phase3/predictions/e12/gemma/20261120T021930Z-06357c1000a943aaaa5b21ee875baba0/_COMPLETE.json",
                "sha256": "c8e866a71f660dae04e432ad0edb1da39d00e9c47f28e916082fe6b502cc492e",
            },
            "e12_analysis": {
                "path": "phase3/results/e12/gemma/20261120T021930Z-06357c1000a943aaaa5b21ee875baba0/_COMPLETE.json",
                "sha256": "cb734055c3492ae18d5d038e4aadd060e38bce1d08b79418fb3a54fb1ad2fb29",
            },
        },
    },
})


def _emit(progress, event):
    if progress is not None:
        progress(event)


def _fmt_num(val, signed=True, decimals=4):
    if val is None:
        return "n/a"
    fmt = f"{float(val):+.{decimals}f}" if signed else f"{float(val):.{decimals}f}"
    return fmt


def _fmt_interval(iv, *, signed=True, decimals=4):
    if not isinstance(iv, dict):
        return "n/a"
    val = iv.get("point", iv.get("value"))
    lo = iv.get("lower")
    hi = iv.get("upper")
    if val is None or lo is None or hi is None:
        return "n/a"
    return f"{_fmt_num(val, signed=signed, decimals=decimals)} [{_fmt_num(lo, signed=signed, decimals=decimals)}, {_fmt_num(hi, signed=signed, decimals=decimals)}]"


def _load_and_verify_all_inputs(root: Path, protocol, refs):
    plan = load_phase3_cpu_plan(root, protocol, refs["phase3_cpu_plan"], bind=False)
    check(plan["confirm_excluded"] is True, "phase3_cpu_plan must enforce confirm_excluded=True")
    ds_fp = plan["dataset_fingerprint"]
    plan_id = plan["plan_identity"]

    loaded_models = {}
    integrity = {}
    for model in MODELS:
        mrefs = refs["models"][model]

        # 1. E40
        e40_ref, e40_body, _ = load_phase3_artifact(
            root, protocol, mrefs["e40_analysis"], KIND_ANALYSIS,
            expected_schema=SCHEMA_ANALYSIS, expected_dataset=ds_fp,
            expected_plan_identity=plan_id, expected_owner=model,
        )
        check(e40_body.get("experiment") == "E40", f"Invalid E40 experiment tag for {model}")
        check(e40_body.get("confirm_excluded") is True, f"E40 confirm_excluded violated for {model}")
        check((e40_body.get("midpoint_e30_parity_check") or {}).get("passed") is True, f"E40 parity check failed for {model}")

        # 2. E41
        e41_cal_ref, e41_cal_body, _ = load_phase3_artifact(
            root, protocol, mrefs["e41_calibrators"], KIND_CALIBRATORS,
            expected_schema=SCHEMA_CALIBRATORS, expected_dataset=ds_fp,
            expected_plan_identity=plan_id, expected_owner=model,
        )
        check(e41_cal_body.get("experiment") == "E41", f"Invalid E41 calibrators experiment tag for {model}")
        check(e41_cal_body.get("confirm_excluded") is True, f"E41 calibrators confirm_excluded violated for {model}")

        e41_ref, e41_body, _ = load_phase3_artifact(
            root, protocol, mrefs["e41_analysis"], KIND_ANALYSIS,
            expected_schema=SCHEMA_ANALYSIS, expected_dataset=ds_fp,
            expected_plan_identity=plan_id, expected_owner=model,
        )
        check(e41_body.get("experiment") == "E41", f"Invalid E41 analysis experiment tag for {model}")
        check(e41_body.get("confirm_excluded") is True, f"E41 confirm_excluded violated for {model}")
        check(plain(e41_body.get("calibrators_ref")) == plain(e41_cal_ref), f"E41 calibrators_ref mismatch for {model}")
        inv41 = e41_body.get("invariance_checks") or {}
        tol41 = float(inv41.get("tolerance", 1e-12))
        check(
            inv41.get("all_T_positive") is True
            and inv41.get("temp_only_0p5_flips") == 0
            and inv41.get("e11_cov_zero_target_labels") is True
            and float(inv41.get("calibrator_max_abs_auroc_diff", 1.0)) <= tol41
            and float(inv41.get("calibrator_max_abs_ap_diff", 1.0)) <= tol41
            and float(inv41.get("mean_shift_max_abs_auroc_diff", 1.0)) <= tol41
            and float(inv41.get("mean_shift_max_abs_ap_diff", 1.0)) <= tol41,
            f"E41 invariance checks failed for {model}",
        )

        # 3. E44
        e44_ref, e44_body, _ = load_phase3_artifact(
            root, protocol, mrefs["e44_analysis"], KIND_ANALYSIS,
            expected_schema=SCHEMA_ANALYSIS, expected_dataset=ds_fp,
            expected_plan_identity=plan_id, expected_owner=model,
        )
        check(e44_body.get("experiment") == "E44", f"Invalid E44 experiment tag for {model}")
        check(e44_body.get("confirm_excluded") is True, f"E44 confirm_excluded violated for {model}")
        check((e44_body.get("phase1_e19_parity") or {}).get("passed") is True, f"E44 Phase 1 E19 parity failed for {model}")

        # 4. E12
        e12_maps_ref, e12_maps_body, _ = load_phase3_artifact(
            root, protocol, mrefs["e12_maps"], KIND_MAPS,
            expected_schema=SCHEMA_MAPS, expected_dataset=ds_fp,
            expected_plan_identity=plan_id, expected_owner=model,
        )
        check(e12_maps_body.get("experiment") == "E12", f"Invalid E12 maps experiment tag for {model}")
        check(e12_maps_body.get("zero_target_semantic_labels") is True, f"E12 maps label guard failed for {model}")

        e12_pred_ref, e12_pred_body, _ = load_phase3_artifact(
            root, protocol, mrefs["e12_predictions"], KIND_PREDICTION,
            expected_schema=SCHEMA_PREDICTION, expected_dataset=ds_fp,
            expected_plan_identity=plan_id, expected_owner=model,
        )
        check(e12_pred_body.get("experiment") == "E12", f"Invalid E12 predictions experiment tag for {model}")
        check(plain(e12_pred_body.get("maps_ref")) == plain(e12_maps_ref), f"E12 predictions maps_ref mismatch for {model}")

        e12_ref, e12_body, _ = load_phase3_artifact(
            root, protocol, mrefs["e12_analysis"], KIND_ANALYSIS,
            expected_schema=SCHEMA_ANALYSIS, expected_dataset=ds_fp,
            expected_plan_identity=plan_id, expected_owner=model,
        )
        check(e12_body.get("experiment") == "E12", f"Invalid E12 analysis experiment tag for {model}")
        check(e12_body.get("confirm_excluded") is True, f"E12 confirm_excluded violated for {model}")
        check(plain(e12_body.get("maps_ref")) == plain(e12_maps_ref), f"E12 analysis maps_ref mismatch for {model}")
        check(plain(e12_body.get("predictions_ref")) == plain(e12_pred_ref), f"E12 analysis predictions_ref mismatch for {model}")
        san12 = e12_body.get("sanity_checks") or {}
        check(
            san12.get("zero_target_semantic_labels_in_maps_and_ridge_selection") is True
            and san12.get("pca_orthonormality_passed") is True
            and san12.get("procrustes_orthogonality_passed") is True
            and san12.get("full_space_reference_reported_separately") is True,
            f"E12 sanity checks failed for {model}",
        )

        loaded_models[model] = {
            "e40_ref": plain(e40_ref), "e40_body": plain(e40_body),
            "e41_cal_ref": plain(e41_cal_ref), "e41_ref": plain(e41_ref), "e41_body": plain(e41_body),
            "e44_ref": plain(e44_ref), "e44_body": plain(e44_body),
            "e12_maps_ref": plain(e12_maps_ref), "e12_pred_ref": plain(e12_pred_ref),
            "e12_ref": plain(e12_ref), "e12_body": plain(e12_body),
        }
        integrity[model] = {
            "e40_midpoint_parity_max_logit_diff": e40_body["midpoint_e30_parity_check"]["max_logit_abs_diff"],
            "e40_midpoint_parity_max_metric_diff": e40_body["midpoint_e30_parity_check"]["max_metric_abs_diff"],
            "e41_invariance_checks": plain(inv41),
            "e44_phase1_e19_parity": plain(e44_body["phase1_e19_parity"]),
            "e12_sanity_checks": plain(san12),
        }

    return {"plan": plan, "models": loaded_models, "integrity": integrity}


def _iv_from_e40_macro(macro_obj):
    iv = macro_obj["interval"]
    return {
        "point": macro_obj["value"],
        "lower": iv["lower"],
        "upper": iv["upper"],
        "valid_draws": iv["valid_draws"],
        "total_draws": iv["total_draws"],
    }


def _evaluate_e41_decisions(e41_body, decision_ranges):
    r41 = decision_ranges["E41"]
    mid_depth = str(e41_body["midpoint_depth"])
    fin_depth = str(e41_body["final_depth"])
    nll_gap_max = float(r41["label_free_nll_gap_to_target_max"])
    cov_red_min = float(r41["cov_auroc_gap_reduction_min"])
    cov_frac_min = float(r41["cov_or_fraction_of_g_res_min"])
    cov_neg_lt = float(r41["cov_negligible_ci_upper_lt"])

    per_cell = {}
    for dkey in (mid_depth, fin_depth):
        for pooling in POOLINGS:
            ckey = f"{dkey}:{pooling}"
            cell = e41_body["cells"][ckey]
            sums = cell["summaries"]
            comb = sums["E11_diag+source_dev_c"]["off_diagonal_12_route_mean"]
            beats_fp = (
                comb["gain_vs_fit_prior"]["nll"]["lower"] is not None
                and comb["gain_vs_fit_prior"]["nll"]["lower"] > 0.0
                and comb["gain_vs_fit_prior"]["nll"]["valid_draws"] >= 1900
            )
            nll_gap_tdc = comb["loss_gap_to_target_dev_c"]["nll"]["point"]
            parity_ok = bool(beats_fp and nll_gap_tdc is not None and nll_gap_tdc < nll_gap_max)

            cov = sums["E11_cov"]["off_diagonal_12_route_mean"]
            cov_gr = cov["auroc_gap_reduction_vs_raw"]
            cov_fr = cov["auroc_gap_reduction_fraction_of_g_res"]
            cov_ci_pos = (
                cov_gr["lower"] is not None
                and cov_gr["lower"] > 0.0
                and cov_gr["valid_draws"] >= 1900
            )
            cov_conseq = bool(
                cov_ci_pos
                and (
                    (cov_gr["point"] is not None and cov_gr["point"] >= cov_red_min)
                    or (cov_fr["point"] is not None and cov_fr["point"] >= cov_frac_min)
                )
            )
            cov_negligible = bool(cov_gr["upper"] is not None and cov_gr["upper"] < cov_neg_lt)
            per_cell[ckey] = {
                "label_free_beats_fit_prior": beats_fp,
                "label_free_nll_gap_to_target_dev_c": nll_gap_tdc,
                "label_free_nll_parity": parity_ok,
                "e11_cov_auroc_gap_reduction": cov_gr,
                "e11_cov_recovered_fraction": cov_fr,
                "e11_cov_consequential": cov_conseq,
                "e11_cov_negligible": cov_negligible,
            }

    mid_last = per_cell[f"{mid_depth}:last_caption"]
    mid_mean = per_cell[f"{mid_depth}:caption_mean"]
    return {
        "per_cell": per_cell,
        "midpoint_caption_mean_label_free_parity": mid_mean["label_free_nll_parity"],
        "midpoint_last_caption_label_free_parity": mid_last["label_free_nll_parity"],
        "any_cell_cov_consequential": any(v["e11_cov_consequential"] for v in per_cell.values()),
        "all_cells_cov_negligible": all(v["e11_cov_negligible"] for v in per_cell.values()),
    }


def _summarize_model(model, mdata, decision_ranges):
    e40 = mdata["e40_body"]
    e41 = mdata["e41_body"]
    e44 = mdata["e44_body"]
    e12 = mdata["e12_body"]

    # E40 summary
    e40_cells = {}
    for dkey in [str(d) for d in e40["depths"]]:
        e40_cells[dkey] = {}
        for cond in e40["conditions"]:
            cobj = e40["cells"][dkey][cond]
            e40_cells[dkey][cond] = {
                "feature_gain_fixed_head_last_auroc": _iv_from_e40_macro(
                    cobj["contrasts"]["feature_gain_fixed_head_last"]["metrics"]["auroc"]["gap_reduction"]["primary_macro"]
                ),
                "feature_gain_fixed_head_mean_auroc": _iv_from_e40_macro(
                    cobj["contrasts"]["feature_gain_fixed_head_mean"]["metrics"]["auroc"]["gap_reduction"]["primary_macro"]
                ),
                "interaction_auroc": _iv_from_e40_macro(
                    cobj["contrasts"]["interaction"]["metrics"]["auroc"]["gap_reduction"]["primary_macro"]
                ),
                "off_distribution_collapse_flagged": bool(cobj["off_distribution_collapse"]["flagged"]),
                "head_mean_on_last_drop_auroc": cobj["off_distribution_collapse"]["head_mean_on_last_vs_last_on_last"]["transfer_12route_auroc"]["delta"],
            }

    # E41 summary
    e41_cells = {}
    for ckey, cobj in e41["cells"].items():
        arms_out = {}
        for arm in ("raw", "source_dev_c", "E11_diag", "E11_cov", "E11_diag+source_dev_c", "target_dev_c", "E11_diag+target_dev_c"):
            off = cobj["summaries"][arm]["off_diagonal_12_route_mean"]
            arms_out[arm] = {
                "auroc": off["primary_macro"]["auroc"],
                "nll": off["primary_macro"]["nll"],
                "brier": off["primary_macro"]["brier"],
                "nll_gain_vs_raw": off["gain_vs_raw"]["nll"],
                "nll_gain_vs_fit_prior": off["gain_vs_fit_prior"]["nll"],
                "nll_gap_to_target_dev_c": off["loss_gap_to_target_dev_c"]["nll"],
                "auroc_gap_reduction_vs_raw": off["auroc_gap_reduction_vs_raw"],
                "auroc_gap_reduction_fraction_of_g_res": off["auroc_gap_reduction_fraction_of_g_res"],
                "flip_rate_0p5": cobj["decision_0p5"][arm]["off_diagonal_12_route_mean"]["flip_rate"],
            }
        e41_cells[ckey] = {
            "depth": cobj["depth"],
            "depth_role": cobj["depth_role"],
            "pooling": cobj["pooling"],
            "fit_prior_nll": cobj["fit_prior_reference"]["off_diagonal_12_route_mean"]["nll"],
            "arms": arms_out,
        }
    e41_dec = _evaluate_e41_decisions(e41, decision_ranges)

    # E44 summary
    e44_arms = {}
    for arm in e44["arms"]:
        e44_arms[arm] = {}
        for nom_key in ("0.50", "0.75", "0.90"):
            mobj = e44["twelve_route_mean"][arm][nom_key]
            e44_arms[arm][nom_key] = {
                "target_coverage": _iv_from_e40_macro(mobj["target_coverage"]),
                "coverage_drift": _iv_from_e40_macro(mobj["coverage_drift"]),
                "retained_risk": _iv_from_e40_macro(mobj["retained_risk"]),
                "base_risk": _iv_from_e40_macro(mobj["base_risk"]),
                "score_blind_risk": _iv_from_e40_macro(mobj["score_blind_risk"]),
                "positive_retention": _iv_from_e40_macro(mobj["positive_retention"]),
                "negative_retention": _iv_from_e40_macro(mobj["negative_retention"]),
            }

    # E12 summary
    e12_arms = {}
    for arm in ("identity_centered", "orthogonal_procrustes", "ridge_linear"):
        aobj = e12["arms"][arm]
        entry = {
            "auroc": _iv_from_e40_macro(aobj["mean_12_routes"]["auroc"]),
            "nll": _iv_from_e40_macro(aobj["mean_12_routes"]["nll"]),
            "mse": {
                "point": e12["mse_summary"]["arms"][arm]["mean_12_routes"]["value"],
                **e12["mse_summary"]["arms"][arm]["mean_12_routes"]["interval"],
            },
        }
        if arm != "identity_centered":
            entry["auroc_gap_reduction"] = _iv_from_e40_macro(aobj["gap_reduction"]["auroc"])
            entry["recovered_fraction_of_g_res"] = _iv_from_e40_macro(aobj["recovered_fraction_of_g_res"]["auroc"])
        e12_arms[arm] = entry

    return {
        "E40": {
            "depths": e40["depths"],
            "midpoint_depth": e40["midpoint_depth"],
            "cells": e40_cells,
            "decision_summary": e40["decision_summary"],
        },
        "E41": {
            "depths": e41["depths"],
            "midpoint_depth": e41["midpoint_depth"],
            "final_depth": e41["final_depth"],
            "cells": e41_cells,
            "decision_summary": e41_dec,
        },
        "E44": {
            "midpoint_depth": e44["midpoint_depth"],
            "arms": e44_arms,
            "decision_summary": e44["decision_summary"],
        },
        "E12": {
            "midpoint_depth": e12["midpoint_depth"],
            "full_space_g_res_auroc": _iv_from_e40_macro(e12["full_space_reference"]["g_res_mean"]["auroc"]),
            "r64_g_res_auroc": _iv_from_e40_macro(e12["pca_source_summary"]["r64_g_res_mean"]["auroc"]),
            "arms": e12_arms,
            "ridge_minus_procrustes_auroc": _iv_from_e40_macro(e12["contrasts"]["ridge_minus_procrustes"]["auroc"]),
            "ridge_minus_procrustes_mse": {
                "point": e12["mse_summary"]["ridge_minus_procrustes"]["value"],
                **e12["mse_summary"]["ridge_minus_procrustes"]["interval"],
            },
            "decision_summary": e12["decision_summary"],
        },
    }


def _evaluate_decisions(summaries):
    per_model = {}
    for m in MODELS:
        s = summaries[m]
        d40 = s["E40"]["decision_summary"]
        d41 = s["E41"]["decision_summary"]
        d44 = s["E44"]["decision_summary"]
        d12 = s["E12"]["decision_summary"]

        e40_verdict = (
            "replicates_caption_only_non_localized"
            if (d40["midpoint_caption_only_replication"]["replicates"] and not d40["final_depth_localized"])
            else ("replicates_caption_only_localized" if d40["midpoint_caption_only_replication"]["replicates"] else "revise")
        )
        e41_verdict = (
            "label_free_nll_parity_with_cov_gain"
            if (d41["midpoint_caption_mean_label_free_parity"] and d41["any_cell_cov_consequential"])
            else ("label_free_nll_parity_cov_negligible" if d41["midpoint_caption_mean_label_free_parity"] else "revise")
        )
        conseq_44 = [arm for arm, info in d44["arms"].items() if info["consequential_recovery"]]
        e44_verdict = "consequential_recovery" if conseq_44 else "revise_or_negligible"
        e12_verdict = d12["outcome"]

        per_model[m] = {
            "E40": e40_verdict,
            "E41": e41_verdict,
            "E44": {"verdict": e44_verdict, "consequential_arms_at_0p75": conseq_44},
            "E12": e12_verdict,
        }

    combined = {
        "E40": {
            "qwen": per_model["qwen"]["E40"],
            "gemma": per_model["gemma"]["E40"],
        },
        "E41": {
            "qwen": per_model["qwen"]["E41"],
            "gemma": per_model["gemma"]["E41"],
        },
        "E44": {
            "qwen": per_model["qwen"]["E44"],
            "gemma": per_model["gemma"]["E44"],
        },
        "E12": {
            "qwen": per_model["qwen"]["E12"],
            "gemma": per_model["gemma"]["E12"],
        },
    }
    return {"per_model": per_model, "combined": combined}


def _render_markdown_review(body):
    lines = [
        "# Phase 3 CPU Track Stage A3 Synthesis & Decision Register",
        "",
        f"- **Schema:** `{body['schema']}` (`{body['stage_a3_version']}`)",
        f"- **Plan identity:** `{body['plan_identity']}`",
        f"- **Dataset fingerprint:** `{body['dataset_fingerprint']}`",
        f"- **CONFIRM status:** `CLOSED` (`confirm_excluded = {body['confirm_excluded']}`)",
        "",
        "## 1. CPU Track Decision Summary",
        "",
        "| Experiment | Qwen | Gemma |",
        "|---|---|---|",
    ]
    comb = body["decision_register"]["combined"]
    lines.append(f"| **E40** (Cross-depth & `caption_only` swap) | `{comb['E40']['qwen']}` | `{comb['E40']['gemma']}` |")
    lines.append(f"| **E41** (7-arm ladder + PCA-CORAL) | `{comb['E41']['qwen']}` | `{comb['E41']['gemma']}` |")
    lines.append(
        f"| **E44** (Selective risk @ 75%) | `{comb['E44']['qwen']['verdict']}` ({', '.join(comb['E44']['qwen']['consequential_arms_at_0p75']) or 'none'}) "
        f"| `{comb['E44']['gemma']['verdict']}` ({', '.join(comb['E44']['gemma']['consequential_arms_at_0p75']) or 'none'}) |"
    )
    lines.append(f"| **E12** (PCA-64 Procrustes vs Ridge) | `{comb['E12']['qwen']}` | `{comb['E12']['gemma']}` |")

    for model in MODELS:
        ms = body["experiment_summaries"][model]
        lines.extend(["", f"## 2. {model.upper()} Quantitative Summary", ""])

        # E40
        lines.extend([
            "### E40 — Cross-Depth & `caption_only` Head × Pooling Swap (AUROC)",
            "",
            "| Depth | Condition | `feature_gain\\|last` [95% CI] | `feature_gain\\|mean` [95% CI] | Interaction [95% CI] | Collapse (`mean_on_last`) |",
            "|---|---|---|---|---|---|",
        ])
        for dkey in [str(d) for d in ms["E40"]["depths"]]:
            for cond in ("image_caption", "caption_only"):
                c = ms["E40"]["cells"][dkey][cond]
                lines.append(
                    f"| `{dkey}` | `{cond}` | {_fmt_interval(c['feature_gain_fixed_head_last_auroc'])} "
                    f"| {_fmt_interval(c['feature_gain_fixed_head_mean_auroc'])} "
                    f"| {_fmt_interval(c['interaction_auroc'])} "
                    f"| `{c['off_distribution_collapse_flagged']}` ({_fmt_num(c['head_mean_on_last_drop_auroc'])}) |"
                )

        # E41
        lines.extend([
            "",
            "### E41 — 7-Arm Supervision Ladder + PCA-CORAL (`r=256`, 12-route mean)",
            "",
            "| Cell | Arm | AUROC [95% CI] | NLL [95% CI] | NLL gap to `target_dev_c` | `E11_cov` ΔG_AUROC |",
            "|---|---|---|---|---|---|",
        ])
        for ckey, cell in ms["E41"]["cells"].items():
            for arm in ("raw", "source_dev_c", "E11_diag", "E11_cov", "E11_diag+source_dev_c", "target_dev_c", "E11_diag+target_dev_c"):
                a = cell["arms"][arm]
                lines.append(
                    f"| `{ckey}` | `{arm}` | {_fmt_interval(a['auroc'], signed=False)} "
                    f"| {_fmt_interval(a['nll'], signed=False)} "
                    f"| {_fmt_interval(a['nll_gap_to_target_dev_c'])} "
                    f"| {_fmt_interval(a['auroc_gap_reduction_vs_raw'])} |"
                )

        # E44
        lines.extend([
            "",
            "### E44 — Selective-Confidence Risk-Coverage (Nominal `0.75`, 12-route mean)",
            "",
            "| Arm | Coverage [95% CI] | Drift | Retained Risk [95% CI] | Base Risk | Pos. Retention |",
            "|---|---|---|---|---|---|",
        ])
        for arm, by_nom in ms["E44"]["arms"].items():
            r75 = by_nom["0.75"]
            lines.append(
                f"| `{arm}` | {_fmt_interval(r75['target_coverage'], signed=False)} "
                f"| {_fmt_num(r75['coverage_drift']['point'], signed=False)} "
                f"| {_fmt_interval(r75['retained_risk'], signed=False)} "
                f"| {_fmt_num(r75['base_risk']['point'], signed=False)} "
                f"| {_fmt_num(r75['positive_retention']['point'], signed=False)} |"
            )

        # E12
        e12 = ms["E12"]
        lines.extend([
            "",
            "### E12 — Image-Anchored PCA-64 Procrustes vs Ridge (Midpoint `caption_mean`)",
            "",
            f"- Full-space `G_res` AUROC: {_fmt_interval(e12['full_space_g_res_auroc'])} | `r=64` `G_res` AUROC: {_fmt_interval(e12['r64_g_res_auroc'])}",
            f"- Constraint contrast (`ridge - procrustes` AUROC): {_fmt_interval(e12['ridge_minus_procrustes_auroc'])} | MSE Δ: {_fmt_interval(e12['ridge_minus_procrustes_mse'])}",
            "",
            "| Arm | AUROC [95% CI] | AUROC ΔG [95% CI] | Recovered `G_res` Frac | Feature MSE [95% CI] |",
            "|---|---|---|---|---|",
        ])
        for arm in ("identity_centered", "orthogonal_procrustes", "ridge_linear"):
            a = e12["arms"][arm]
            gr_str = "—" if arm == "identity_centered" else _fmt_interval(a["auroc_gap_reduction"])
            fr_str = "—" if arm == "identity_centered" else _fmt_interval(a["recovered_fraction_of_g_res"], signed=False)
            lines.append(
                f"| `{arm}` | {_fmt_interval(a['auroc'], signed=False)} | {gr_str} | {fr_str} | {_fmt_interval(a['mse'], signed=False)} |"
            )

    lines.append("")
    return "\n".join(lines)


@contract
def verify_phase3_cpu_release(root, protocol, release_ref):
    """Independent verification of published `phase3_release` and all upstream artifacts."""
    resolved_root = Path(root).resolve()
    ref, body, marker = load_phase3_artifact(
        resolved_root, protocol, release_ref, KIND_RELEASE, expected_schema=SCHEMA_RELEASE
    )
    check(body.get("confirm_excluded") is True, "phase3_release must enforce confirm_excluded=True")
    check("phase3_cpu_review.md" in marker["files"], "phase3_release missing phase3_cpu_review.md")
    review_path = within(resolved_root, ref["path"]).parent / "phase3_cpu_review.md"
    check(review_path.is_file() and review_path.stat().st_size > 0, "Empty or missing phase3_cpu_review.md")

    refs = {
        "phase3_cpu_plan": plain(body["stage_a1_ref"]),
        "models": {
            m: {
                "e40_analysis": plain(body["stage_a2_refs"]["e40"][m]),
                "e41_calibrators": plain(body["stage_a2_refs"]["e41_calibrators"][m]),
                "e41_analysis": plain(body["stage_a2_refs"]["e41"][m]),
                "e44_analysis": plain(body["stage_a2_refs"]["e44"][m]),
                "e12_maps": plain(body["stage_a2_refs"]["e12_maps"][m]),
                "e12_predictions": plain(body["stage_a2_refs"]["e12_predictions"][m]),
                "e12_analysis": plain(body["stage_a2_refs"]["e12"][m]),
            }
            for m in MODELS
        },
    }
    _load_and_verify_all_inputs(resolved_root, protocol, refs)
    return frozen({
        "verified": True,
        "release_ref": plain(ref),
        "plan_identity": body["plan_identity"],
        "dataset_fingerprint": body["dataset_fingerprint"],
        "confirm_excluded": True,
        "combined_decisions": plain(body["decision_register"]["combined"]),
    })


def _run_fresh_subprocess_readback(root: Path, protocol_path: Path, release_ref):
    cmd = [
        sys.executable, "-u", "-m", "src.phase3.synthesize_cpu",
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
def synthesize_phase3_cpu(
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
    """Execute Phase 3 CPU Track Stage A3 synthesis and publish `phase3_release`."""
    resolved_root = Path(root).resolve()
    t0 = time.monotonic()
    _emit(progress, {"event": "stage_a3_start", "unit": "verify_inputs", "completed": 0, "total": 4})

    loaded = _load_and_verify_all_inputs(resolved_root, protocol, refs)
    read_s = time.monotonic() - t0
    _emit(progress, {"event": "stage_a3_inputs_verified", "unit": "verify_inputs", "completed": 1, "total": 4, "read_hash_s": round(read_s, 4)})

    plan = loaded["plan"]
    t1 = time.monotonic()
    summaries = {m: _summarize_model(m, loaded["models"][m], plain(plan["decision_ranges"])) for m in MODELS}
    decision_register = _evaluate_decisions(summaries)
    compute_s = time.monotonic() - t1
    _emit(progress, {"event": "stage_a3_decisions_evaluated", "unit": "evaluate_decisions", "completed": 2, "total": 4, "compute_s": round(compute_s, 4)})

    body = {
        "schema": SCHEMA_RELEASE,
        "track": "cpu_track",
        "stage_a3_version": STAGE_A3_VERSION,
        "phase3_version": PHASE3_CPU_VERSION,
        "plan": plain(plan.reference),
        "plan_identity": plan["plan_identity"],
        "dataset_fingerprint": plan["dataset_fingerprint"],
        "protocol_fingerprint": plan["protocol_fingerprint"],
        "confirm_excluded": True,
        "stage_a1_ref": plain(plan.reference),
        "stage_a2_refs": {
            "e40": {m: loaded["models"][m]["e40_ref"] for m in MODELS},
            "e41_calibrators": {m: loaded["models"][m]["e41_cal_ref"] for m in MODELS},
            "e41": {m: loaded["models"][m]["e41_ref"] for m in MODELS},
            "e44": {m: loaded["models"][m]["e44_ref"] for m in MODELS},
            "e12_maps": {m: loaded["models"][m]["e12_maps_ref"] for m in MODELS},
            "e12_predictions": {m: loaded["models"][m]["e12_pred_ref"] for m in MODELS},
            "e12": {m: loaded["models"][m]["e12_ref"] for m in MODELS},
        },
        "integrity_checks": loaded["integrity"],
        "experiment_summaries": summaries,
        "decision_register": decision_register,
    }
    review_md = _render_markdown_review(body)

    upstream = [plain(plan.reference)]
    for m in MODELS:
        md = loaded["models"][m]
        upstream.extend([
            md["e40_ref"], md["e41_cal_ref"], md["e41_ref"],
            md["e44_ref"], md["e12_maps_ref"], md["e12_pred_ref"], md["e12_ref"],
        ])

    t2 = time.monotonic()
    release_ref = publish_phase3(
        resolved_root,
        protocol,
        plan["dataset_fingerprint"],
        RELEASE_NAMESPACE,
        KIND_RELEASE,
        body,
        producer,
        files={"phase3_cpu_review.md": review_md.encode("utf-8")},
        upstream=upstream,
        run_id=run_id or new_run_id(),
    )
    write_s = time.monotonic() - t2
    _emit(progress, {"event": "stage_a3_release_published", "unit": "publish_release", "completed": 3, "total": 4, "write_s": round(write_s, 4), "release_ref": plain(release_ref)})

    if run_subprocess_readback:
        proto_file = protocol_path if protocol_path is not None else (REPO / "configs/protocol.json")
        readback = _run_fresh_subprocess_readback(resolved_root, proto_file, release_ref)
    else:
        readback = plain(verify_phase3_cpu_release(resolved_root, protocol, release_ref))

    _emit(progress, {
        "event": "stage_a3_complete",
        "status": "passed",
        "unit": "independent_readback",
        "completed": 4,
        "total": 4,
        "release_ref": plain(release_ref),
        "independent_readback": readback,
        "combined_decisions": decision_register["combined"],
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
    print(json.dumps(plain(event), sort_keys=True), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Phase 3 CPU Track Stage A3 synthesis and independent readback.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Synthesize Stage A1-A2 outputs and publish phase3_release.")
    p_run.add_argument("--phase1-root", default=None)
    p_run.add_argument("--protocol", default=str(REPO / "configs/protocol.json"))
    p_run.add_argument("--refs", default=None, help="Optional JSON string or path overriding CERTIFIED_PHASE3_CPU_REFS.")
    p_run.add_argument("--storage-scope", default="persistent-drive")
    p_run.add_argument("--run-id", default=None)
    p_run.add_argument("--print-review", action="store_true")

    p_ver = sub.add_parser("verify-release", help="Fresh-subprocess independent readback of phase3_release.")
    p_ver.add_argument("--phase1-root", default=None)
    p_ver.add_argument("--protocol", default=str(REPO / "configs/protocol.json"))
    p_ver.add_argument("--release", required=True, help="JSON {path, sha256} or path to phase3_release _COMPLETE.json.")

    args = parser.parse_args(argv)
    root = phase1_root(args.phase1_root)
    proto_path = Path(args.protocol).resolve()
    protocol = json.loads(proto_path.read_text(encoding="utf-8"))

    if args.command == "verify-release":
        rel_ref = _parse_json_or_file(args.release) if args.release.strip().startswith("{") else args.release
        out = verify_phase3_cpu_release(root, protocol, rel_ref)
        _default_emit(plain(out))
        return 0

    refs = _parse_json_or_file(args.refs) if args.refs else plain(CERTIFIED_PHASE3_CPU_REFS)
    producer = current_producer(scope=args.storage_scope)
    result = synthesize_phase3_cpu(
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
        print("\n=== PHASE 3 CPU TRACK REVIEW MARKDOWN ===\n", flush=True)
        print(result["review_markdown"], flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        _default_emit({"status": "blocked", "blockers": [str(exc)]})
        raise SystemExit(2)
