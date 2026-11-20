"""Unit and contract tests for Phase 3 CPU Track Stage A3 synthesis."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

from src.common.contracts import ContractError, fingerprint
from src.phase3.cpu_plan import PHASE3_CPU_DECISION_RANGES
from src.phase3.interfaces import (
    KIND_ANALYSIS,
    KIND_CALIBRATORS,
    KIND_CPU_PLAN,
    KIND_MAPS,
    KIND_PREDICTION,
    KIND_PROBE,
    KIND_PROBE_PAYLOAD,
    MODELS,
    PHASE3_CPU_VERSION,
    POOLINGS,
    REPO,
    SCHEMA_ANALYSIS,
    SCHEMA_CALIBRATORS,
    SCHEMA_CPU_PLAN,
    SCHEMA_MAPS,
    SCHEMA_PREDICTION,
    SCHEMA_PROBE,
    init_phase3_dirs,
    plain,
    publish_phase3,
)
import src.phase3.synthesize_cpu as synth


def _load_protocol():
    return json.loads((REPO / "configs/protocol.json").read_text(encoding="utf-8"))


def _fake_producer():
    return {
        "code_commit": "0" * 40,
        "storage_scope": "local-development",
        "software_versions": {"python": "3.12.14", "numpy": "2.1.3"},
    }


def _iv(point=0.05, lower=0.02, upper=0.08, valid=2000):
    return {"point": point, "lower": lower, "upper": upper, "valid_draws": valid, "total_draws": 2000}


def _macro(value=0.05, lower=0.02, upper=0.08, valid=2000):
    return {"value": value, "interval": {"lower": lower, "upper": upper, "valid_draws": valid, "total_draws": 2000}}


def _build_synthetic_stage_a3_root(root: Path, protocol):
    init_phase3_dirs(root)
    ds_fp = fingerprint({"synthetic": "stage_a3"})
    proto_fp = fingerprint(protocol)
    prod = _fake_producer()

    # Two-process-compatible probe receipt (using synthetic distinct PIDs)
    payload_ref = publish_phase3(
        root, protocol, ds_fp, "phase3/probes", KIND_PROBE_PAYLOAD,
        {"schema": SCHEMA_PROBE, "nonce": "n1"}, prod,
    )
    probe_ref = publish_phase3(
        root, protocol, ds_fp, "phase3/probes", KIND_PROBE,
        {"schema": SCHEMA_PROBE, "payload": plain(payload_ref), "independent_process_readback": True, "reader_pid": 1001, "writer_pid": 1002},
        prod, upstream=[payload_ref],
    )
    core = {
        "schema": SCHEMA_CPU_PLAN,
        "phase3_version": PHASE3_CPU_VERSION,
        "dataset_fingerprint": ds_fp,
        "protocol_fingerprint": proto_fp,
        "storage_probe": plain(probe_ref),
        "decision_ranges": plain(PHASE3_CPU_DECISION_RANGES),
        "confirm_excluded": True,
    }
    plan_id = fingerprint(core)
    plan_ref = publish_phase3(
        root, protocol, ds_fp, "phase3/plans", KIND_CPU_PLAN,
        dict(core, plan_identity=plan_id), prod, upstream=[probe_ref],
    )

    refs = {"phase3_cpu_plan": plain(plan_ref), "models": {}}
    for m in MODELS:
        depths = [9, 18, 27, 36] if m == "qwen" else [9, 17, 26, 34]
        mid, fin = depths[1], depths[3]

        # E40
        e40_cells = {}
        for d in depths:
            e40_cells[str(d)] = {}
            for cond in ("image_caption", "caption_only"):
                e40_cells[str(d)][cond] = {
                    "contrasts": {
                        "feature_gain_fixed_head_last": {"metrics": {"auroc": {"gap_reduction": {"primary_macro": _macro(0.08, 0.06, 0.10)}}}},
                        "feature_gain_fixed_head_mean": {"metrics": {"auroc": {"gap_reduction": {"primary_macro": _macro(0.00, -0.01, 0.01)}}}},
                        "interaction": {"metrics": {"auroc": {"gap_reduction": {"primary_macro": _macro(0.08, 0.06, 0.10)}}}},
                    },
                    "off_distribution_collapse": {
                        "flagged": True,
                        "head_mean_on_last_vs_last_on_last": {"transfer_12route_auroc": {"delta": -0.10}},
                    },
                }
        e40_ref = publish_phase3(
            root, protocol, ds_fp, f"phase3/results/e40/{m}", KIND_ANALYSIS,
            {
                "schema": SCHEMA_ANALYSIS, "experiment": "E40", "plan_identity": plan_id, "model": m,
                "depths": depths, "midpoint_depth": mid, "conditions": ["image_caption", "caption_only"],
                "midpoint_e30_parity_check": {"passed": True, "max_logit_abs_diff": 0.0, "max_metric_abs_diff": 0.0},
                "cells": e40_cells,
                "decision_summary": {"final_depth_localized": False, "midpoint_caption_only_replication": {"replicates": True}},
                "confirm_excluded": True,
            },
            prod, upstream=[plan_ref],
        )

        # E41
        e41_cal_ref = publish_phase3(
            root, protocol, ds_fp, f"phase3/calibrators/e41/{m}", KIND_CALIBRATORS,
            {"schema": SCHEMA_CALIBRATORS, "experiment": "E41", "plan_identity": plan_id, "model": m, "confirm_excluded": True},
            prod, upstream=[plan_ref],
        )
        e41_cells = {}
        for d, role in ((mid, "midpoint"), (fin, "final")):
            for p in POOLINGS:
                ck = f"{d}:{p}"
                arms = {}
                for arm in ("raw", "source_dev_c", "E11_diag", "E11_cov", "E11_diag+source_dev_c", "target_dev_c", "E11_diag+target_dev_c"):
                    arms[arm] = {
                        "off_diagonal_12_route_mean": {
                            "primary_macro": {"auroc": _iv(0.80, 0.78, 0.82), "nll": _iv(0.30, 0.28, 0.32), "brier": _iv(0.08, 0.07, 0.09)},
                            "gain_vs_raw": {"nll": _iv(0.10, 0.08, 0.12)},
                            "gain_vs_fit_prior": {"nll": _iv(0.08, 0.06, 0.10)},
                            "loss_gap_to_target_dev_c": {"nll": _iv(0.005, 0.001, 0.009)},
                            "auroc_gap_reduction_vs_raw": _iv(0.001, -0.001, 0.003),
                            "auroc_gap_reduction_fraction_of_g_res": _iv(0.03, -0.02, 0.08),
                        }
                    }
                e41_cells[ck] = {
                    "depth": d, "depth_role": role, "pooling": p,
                    "fit_prior_reference": {"off_diagonal_12_route_mean": {"nll": _iv(0.38, 0.36, 0.40)}},
                    "summaries": arms,
                    "decision_0p5": {arm: {"off_diagonal_12_route_mean": {"flip_rate": 0.01}} for arm in arms},
                }
        e41_ref = publish_phase3(
            root, protocol, ds_fp, f"phase3/results/e41/{m}", KIND_ANALYSIS,
            {
                "schema": SCHEMA_ANALYSIS, "experiment": "E41", "plan_identity": plan_id, "model": m,
                "depths": [mid, fin], "midpoint_depth": mid, "final_depth": fin,
                "calibrators_ref": plain(e41_cal_ref),
                "invariance_checks": {
                    "all_T_positive": True, "temp_only_0p5_flips": 0, "e11_cov_zero_target_labels": True,
                    "calibrator_max_abs_auroc_diff": 0.0, "calibrator_max_abs_ap_diff": 0.0,
                    "mean_shift_max_abs_auroc_diff": 0.0, "mean_shift_max_abs_ap_diff": 0.0, "tolerance": 1e-12,
                },
                "cells": e41_cells, "confirm_excluded": True,
            },
            prod, upstream=[plan_ref, e41_cal_ref],
        )

        # E44
        e44_arms = ("last_caption:raw", "last_caption:source_dev_c", "last_caption:target_dev_c", "caption_mean:raw", "caption_mean:e11_mean_scale")
        twelve = {}
        for arm in e44_arms:
            twelve[arm] = {
                nom: {
                    "target_coverage": _macro(float(nom), float(nom) - 0.01, float(nom) + 0.01),
                    "coverage_drift": _macro(0.01, 0.00, 0.02),
                    "retained_risk": _macro(0.05, 0.04, 0.06),
                    "base_risk": _macro(0.08, 0.07, 0.09),
                    "score_blind_risk": _macro(0.08, 0.07, 0.09),
                    "positive_retention": _macro(0.70, 0.65, 0.75),
                    "negative_retention": _macro(0.75, 0.70, 0.80),
                }
                for nom in ("0.50", "0.75", "0.90")
            }
        e44_ref = publish_phase3(
            root, protocol, ds_fp, f"phase3/results/e44/{m}", KIND_ANALYSIS,
            {
                "schema": SCHEMA_ANALYSIS, "experiment": "E44", "plan_identity": plan_id, "model": m,
                "midpoint_depth": mid, "arms": list(e44_arms),
                "phase1_e19_parity": {"passed": True, "max_abs_diff": 0.0},
                "twelve_route_mean": twelve,
                "decision_summary": {"arms": {arm: {"consequential_recovery": arm == "caption_mean:e11_mean_scale"} for arm in e44_arms}},
                "confirm_excluded": True,
            },
            prod, upstream=[plan_ref],
        )

        # E12
        e12_maps_ref = publish_phase3(
            root, protocol, ds_fp, f"phase3/maps/e12/{m}", KIND_MAPS,
            {"schema": SCHEMA_MAPS, "experiment": "E12", "plan_identity": plan_id, "model": m, "zero_target_semantic_labels": True, "confirm_excluded": True},
            prod, upstream=[plan_ref],
        )
        e12_pred_ref = publish_phase3(
            root, protocol, ds_fp, f"phase3/predictions/e12/{m}", KIND_PREDICTION,
            {"schema": SCHEMA_PREDICTION, "experiment": "E12", "plan_identity": plan_id, "model": m, "maps_ref": plain(e12_maps_ref), "confirm_excluded": True},
            prod, upstream=[plan_ref, e12_maps_ref],
        )
        e12_ref = publish_phase3(
            root, protocol, ds_fp, f"phase3/results/e12/{m}", KIND_ANALYSIS,
            {
                "schema": SCHEMA_ANALYSIS, "experiment": "E12", "plan_identity": plan_id, "model": m,
                "midpoint_depth": mid, "maps_ref": plain(e12_maps_ref), "predictions_ref": plain(e12_pred_ref),
                "sanity_checks": {
                    "zero_target_semantic_labels_in_maps_and_ridge_selection": True,
                    "pca_orthonormality_passed": True, "procrustes_orthogonality_passed": True,
                    "full_space_reference_reported_separately": True, "confirm_excluded": True,
                },
                "full_space_reference": {"g_res_mean": {"auroc": _macro(0.02, 0.015, 0.025)}},
                "pca_source_summary": {"r64_g_res_mean": {"auroc": _macro(0.02, 0.015, 0.025)}},
                "arms": {
                    "identity_centered": {"mean_12_routes": {"auroc": _macro(0.80, 0.78, 0.82), "nll": _macro(0.35, 0.33, 0.37)}},
                    "orthogonal_procrustes": {
                        "mean_12_routes": {"auroc": _macro(0.81, 0.79, 0.83), "nll": _macro(0.33, 0.31, 0.35)},
                        "gap_reduction": {"auroc": _macro(0.01, 0.005, 0.015)},
                        "recovered_fraction_of_g_res": {"auroc": _macro(0.50, 0.25, 0.75)},
                    },
                    "ridge_linear": {
                        "mean_12_routes": {"auroc": _macro(0.815, 0.795, 0.835), "nll": _macro(0.32, 0.30, 0.34)},
                        "gap_reduction": {"auroc": _macro(0.015, 0.01, 0.02)},
                        "recovered_fraction_of_g_res": {"auroc": _macro(0.75, 0.50, 1.00)},
                    },
                },
                "contrasts": {"ridge_minus_procrustes": {"auroc": _macro(0.005, 0.001, 0.009)}},
                "mse_summary": {
                    "arms": {a: {"mean_12_routes": {"value": 0.1, "interval": {"lower": 0.09, "upper": 0.11, "valid_draws": 2000, "total_draws": 2000}}}
                             for a in ("identity_centered", "orthogonal_procrustes", "ridge_linear")},
                    "ridge_minus_procrustes": {"value": -0.02, "interval": {"lower": -0.03, "upper": -0.01, "valid_draws": 2000, "total_draws": 2000}},
                },
                "decision_summary": {"outcome": "consequential_recovery"},
                "confirm_excluded": True,
            },
            prod, upstream=[plan_ref, e12_maps_ref, e12_pred_ref],
        )

        refs["models"][m] = {
            "e40_analysis": plain(e40_ref),
            "e41_calibrators": plain(e41_cal_ref),
            "e41_analysis": plain(e41_ref),
            "e44_analysis": plain(e44_ref),
            "e12_maps": plain(e12_maps_ref),
            "e12_predictions": plain(e12_pred_ref),
            "e12_analysis": plain(e12_ref),
        }
    return refs


class Phase3StageA3Tests(unittest.TestCase):
    def test_lazy_imports_and_certified_refs(self):
        self.assertNotIn("numpy", sys.modules)
        self.assertIn("phase3_cpu_plan", synth.CERTIFIED_PHASE3_CPU_REFS)
        for m in MODELS:
            self.assertEqual(
                set(synth.CERTIFIED_PHASE3_CPU_REFS["models"][m].keys()),
                {"e40_analysis", "e41_calibrators", "e41_analysis", "e44_analysis", "e12_maps", "e12_predictions", "e12_analysis"},
            )

    def test_synthesize_and_independent_subprocess_readback(self):
        protocol = _load_protocol()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            refs = _build_synthetic_stage_a3_root(root, protocol)
            res = synth.synthesize_phase3_cpu(
                root, protocol, refs, _fake_producer(), run_subprocess_readback=True
            )
            self.assertEqual(res["status"], "passed")
            self.assertTrue(res["independent_readback"]["verified"])
            self.assertTrue(res["independent_readback"]["confirm_excluded"])
            self.assertIn("CPU Track Decision Summary", res["review_markdown"])

    def test_notebook_contract(self):
        nb_path = REPO / "scripts/phase3/synthesize_phase3_cpu.ipynb"
        self.assertTrue(nb_path.is_file())
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertEqual(len(nb["cells"]), 4)
        full = "\n".join("".join(c["source"]) for c in nb["cells"])
        self.assertIn("CODE_REF = 'phase3'", full)
        self.assertIn("src.phase3.synthesize_cpu", full)
        self.assertIn("93cbf480e9d8c13401aaa98315444f3a4d2b3b57ec769b458a799dc269c40439", full)


if __name__ == "__main__":
    unittest.main()
