"""Unit and contract tests for Phase 3 Stage A2 Worker 3 (E44).

Runs with Python 3.9 stdlib only (no numpy/scipy required).
"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from src.common.contracts import ContractError, fingerprint
from src.phase3.interfaces import (
    KIND_ANALYSIS,
    PHASE3_CPU_DECISION_RANGES,
    REPO,
    SCHEMA_ANALYSIS,
    init_phase3_dirs,
    plain,
    publish_phase3,
    require_open_split,
)
from src.phase3.run_e44 import (
    E19_SELECTIVE_METRICS,
    E44_ARMS,
    E44_ARM_SUPERVISION,
    E44_NOMINALS,
    E44_PRIMARY_NOMINAL,
    E44_RESULTS_NAMESPACE,
    E44_SELECTIVE_METRICS,
    _e44_expected_inputs,
    _metric_support,
    _support_status_for_metric,
    evaluate_e44_decision_summary,
    run_e44,
)


def _load_protocol():
    return json.loads((REPO / "configs/protocol.json").read_text(encoding="utf-8"))


def _fake_producer():
    return {
        "code_commit": "0" * 40,
        "storage_scope": "local-development",
        "software_versions": {"python": "3.12.14", "numpy": "2.1.3"},
    }


class _FakePlan(dict):
    def __init__(self, body, reference, root, protocol):
        super().__init__(body)
        self.reference = reference
        self.root = Path(root)
        self.protocol = protocol
        self.binding = None


class Phase3E44Tests(unittest.TestCase):
    def test_lazy_imports_and_cli_help_stdlib_only(self):
        self.assertNotIn("numpy", sys.modules)
        self.assertNotIn("scipy", sys.modules)
        proc = subprocess.run(
            [sys.executable, "-m", "src.phase3.run_e44", "--help"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--plan", proc.stdout)
        self.assertIn("--model", proc.stdout)
        self.assertIn("--resume", proc.stdout)

    def test_constants_and_confirm_guard(self):
        self.assertEqual(
            E44_ARMS,
            (
                "last_caption:raw",
                "last_caption:source_dev_c",
                "last_caption:target_dev_c",
                "caption_mean:raw",
                "caption_mean:e11_mean_scale",
            ),
        )
        self.assertEqual(E44_NOMINALS, (0.5, 0.75, 0.9))
        self.assertEqual(E44_PRIMARY_NOMINAL, 0.75)
        self.assertEqual(set(E44_ARM_SUPERVISION.keys()), set(E44_ARMS))
        for metric in E19_SELECTIVE_METRICS:
            self.assertIn(metric, E44_SELECTIVE_METRICS)
        self.assertIn("coverage_drift", E44_SELECTIVE_METRICS)
        self.assertIn("retained_risk_vs_base_risk", E44_SELECTIVE_METRICS)
        self.assertIn("retained_risk_vs_score_blind_risk", E44_SELECTIVE_METRICS)
        with self.assertRaises(ContractError):
            require_open_split("CONFIRM")

    def test_support_status_minimum_20_rules(self):
        # [N, N_pos, N_neg, R, R_pos, R_neg, E, E_ret, TP_ret, TN_ret]
        counts_ok = [360, 100, 260, 270, 80, 190, 60, 25, 65, 180]
        sup = _metric_support("retained_risk", counts_ok)
        self.assertEqual(sup["retained_errors"], 25)
        self.assertEqual(sup["retained_correct"], 245)
        status, reason = _support_status_for_metric("retained_risk", 25 / 270, 100, 260, sup, 20, 20)
        self.assertEqual((status, reason), ("ok", None))

        # Retained errors < 20 -> descriptive_low_support
        counts_low_ret_err = [360, 100, 260, 270, 80, 190, 60, 19, 70, 181]
        sup_low = _metric_support("retained_risk", counts_low_ret_err)
        status_low, _ = _support_status_for_metric("retained_risk", 19 / 270, 100, 260, sup_low, 20, 20)
        self.assertEqual(status_low, "descriptive_low_support")

        # SCREEN pos/neg < 20 -> descriptive_low_support
        status_scr, _ = _support_status_for_metric("target_coverage", 0.75, 19, 341, sup, 20, 20)
        self.assertEqual(status_scr, "descriptive_low_support")

        # Subgroup outcome < 20 -> descriptive_low_support
        counts_low_tp = [360, 100, 260, 270, 80, 190, 60, 25, 15, 180]
        sup_tp = _metric_support("retained_sensitivity", counts_low_tp)
        status_tp, _ = _support_status_for_metric("retained_sensitivity", 15 / 80, 100, 260, sup_tp, 20, 20)
        self.assertEqual(status_tp, "descriptive_low_support")

        # Zero denominator -> undefined
        status_undef, _ = _support_status_for_metric("retained_risk", None, 100, 260, sup, 20, 20)
        self.assertEqual(status_undef, "undefined")

    def test_decision_summary_thresholds(self):
        ranges = plain(PHASE3_CPU_DECISION_RANGES["E44"])
        twelve_route_means = {}
        for arm in E44_ARMS:
            drift = 0.10 if arm == "last_caption:raw" else 0.020 if arm == "caption_mean:e11_mean_scale" else 0.08
            ret_risk = 0.14 if arm == "last_caption:raw" else 0.11 if arm == "caption_mean:e11_mean_scale" else 0.138
            pos_ret = 0.70 if arm != "last_caption:target_dev_c" else 0.45
            twelve_route_means[arm] = {
                "0.75": {
                    "coverage_drift": {"value": drift},
                    "retained_risk": {"value": ret_risk},
                    "base_risk": {"value": 0.18},
                    "score_blind_risk": {"value": 0.18},
                    "positive_retention": {"value": pos_ret},
                }
            }
        paired_vs_raw = {
            "last_caption:source_dev_c": {
                "twelve_route_mean": {
                    "0.75": {"retained_risk": {"value": -0.002, "interval": {"lower": -0.004, "upper": 0.001}}}
                }
            },
            "last_caption:target_dev_c": {
                "twelve_route_mean": {
                    "0.75": {"retained_risk": {"value": -0.002, "interval": {"lower": -0.004, "upper": 0.000}}}
                }
            },
            "caption_mean:raw": {
                "twelve_route_mean": {
                    "0.75": {"retained_risk": {"value": -0.015, "interval": {"lower": -0.025, "upper": -0.005}}}
                }
            },
            "caption_mean:e11_mean_scale": {
                "twelve_route_mean": {
                    "0.75": {"retained_risk": {"value": -0.030, "interval": {"lower": -0.040, "upper": -0.020}}}
                }
            },
        }
        summary = evaluate_e44_decision_summary(twelve_route_means, paired_vs_raw, ranges)
        self.assertTrue(summary["arms"]["caption_mean:e11_mean_scale"]["consequential_recovery"])
        self.assertFalse(summary["arms"]["caption_mean:e11_mean_scale"]["practically_negligible"])
        self.assertFalse(summary["arms"]["last_caption:source_dev_c"]["consequential_recovery"])
        self.assertTrue(summary["arms"]["last_caption:source_dev_c"]["practically_negligible"])

    def test_explicit_resume_verification_and_rejections(self):
        protocol = _load_protocol()
        ds_fp = fingerprint({"test": "e44_resume"})
        plan_id = fingerprint({"plan": "e44_test"})
        inv = {"midpoint_depth": 18, "dummy": True}
        plan_ref = {"path": "phase3/plans/run1/_COMPLETE.json", "sha256": "a" * 64}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            init_phase3_dirs(root)
            fake_plan = _FakePlan(
                {
                    "plan_identity": plan_id,
                    "dataset_fingerprint": ds_fp,
                    "confirm_excluded": True,
                    "e44_inventory": {"qwen": inv, "gemma": {"midpoint_depth": 17, "dummy": True}},
                },
                plan_ref,
                root,
                protocol,
            )
            expected_inputs = _e44_expected_inputs(fake_plan, "qwen", inv)
            body = {
                "schema": SCHEMA_ANALYSIS,
                "experiment": "E44",
                "plan_identity": plan_id,
                "model": "qwen",
                "inputs": expected_inputs,
                "phase1_e19_parity": {"passed": True, "max_abs_diff": 0.0, "tol": 1e-12},
                "decision_summary": {"primary_nominal": 0.75, "arms": {}},
                "confirm_excluded": True,
            }
            ref = publish_phase3(
                root,
                protocol,
                ds_fp,
                f"{E44_RESULTS_NAMESPACE}/qwen",
                KIND_ANALYSIS,
                body,
                _fake_producer(),
            )

            events = []
            with mock.patch("src.phase3.run_e44.load_phase3_cpu_plan", return_value=fake_plan):
                res = run_e44(
                    root,
                    protocol,
                    plan_ref,
                    "qwen",
                    _fake_producer(),
                    resume_analysis_ref=ref,
                    progress=events.append,
                )
            self.assertEqual(res["status"], "passed")
            self.assertTrue(res["resumed"])
            self.assertEqual(plain(res["analysis_ref"]), plain(ref))
            self.assertTrue(any(e.get("event") == "e44_resumed" for e in events))

            # Foreign owner rejection
            with mock.patch("src.phase3.run_e44.load_phase3_cpu_plan", return_value=fake_plan):
                with self.assertRaises(ContractError):
                    run_e44(root, protocol, plan_ref, "gemma", _fake_producer(), resume_analysis_ref=ref)

            # Tampered input rejection
            bad_plan = _FakePlan(
                {
                    "plan_identity": plan_id,
                    "dataset_fingerprint": ds_fp,
                    "confirm_excluded": True,
                    "e44_inventory": {"qwen": {"midpoint_depth": 18, "dummy": False}},
                },
                plan_ref,
                root,
                protocol,
            )
            with mock.patch("src.phase3.run_e44.load_phase3_cpu_plan", return_value=bad_plan):
                with self.assertRaises(ContractError):
                    run_e44(root, protocol, plan_ref, "qwen", _fake_producer(), resume_analysis_ref=ref)

    def test_notebook_contract(self):
        nb_path = REPO / "scripts/phase3/run_e44.ipynb"
        self.assertTrue(nb_path.is_file())
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertEqual(len(nb["cells"]), 4)
        cell3 = "".join(nb["cells"][2]["source"])
        self.assertIn("MODEL = 'qwen'", cell3)
        self.assertIn("PLAN_REF", cell3)
        full = "\n".join("".join(c["source"]) for c in nb["cells"])
        self.assertIn("CODE_REF = 'phase3'", full)
        self.assertIn("OPENBLAS_NUM_THREADS", full)
        self.assertIn("requirements-stage2-cpu.txt", full)
        self.assertIn("src.phase3.run_e44", full)


if __name__ == "__main__":
    unittest.main()
