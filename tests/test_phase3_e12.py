"""Unit and contract tests for Phase 3 Stage A2 Worker 4 (E12)."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.common.contracts import ContractError, LANGUAGES, fingerprint
from src.phase3.interfaces import (
    KIND_ANALYSIS,
    KIND_CPU_PLAN,
    KIND_MAPS,
    KIND_PREDICTION,
    PHASE3_CPU_DECISION_RANGES,
    REPO,
    SCHEMA_ANALYSIS,
    SCHEMA_CPU_PLAN,
    SCHEMA_MAPS,
    SCHEMA_PREDICTION,
    frozen,
    init_phase3_dirs,
    load_phase3_artifact,
    plain,
    publish_phase3,
    require_open_split,
    verify_resume_ref,
)
from src.phase3.run_e12 import (
    C_GRID,
    E12_ARMS,
    E12_CONDITION,
    E12_METRICS,
    E12_POOLING,
    EXPECTED_DEV_S_IMAGES,
    EXPECTED_FIT_IMAGES,
    EXPECTED_SCREEN_IMAGES,
    RIDGE_ALPHAS,
    SUPERVISION_ADAPTED,
    SUPERVISION_DIAGONAL,
    SUPERVISION_IDENTITY,
    _evaluate_decision_summary,
    _gap_value,
    _improvement_value,
    _off_diagonal_pairs,
    _parse_ref_arg,
    _route_key,
)

try:
    import numpy as np
    import scipy  # noqa: F401
    from src.phase3.run_e12 import (
        _fit_orthogonal_procrustes,
        _fit_pca_basis,
        _fit_ridge_linear,
        run_e12,
    )

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def _load_protocol():
    return json.loads((REPO / "configs/protocol.json").read_text(encoding="utf-8"))


def _fake_producer():
    return {
        "code_commit": "0" * 40,
        "storage_scope": "local-development",
        "software_versions": {"python": "3.12.14", "numpy": "2.1.3"},
    }


def _make_interval(val, lo, hi, valid_draws=2000, total_draws=2000):
    return {
        "value": val,
        "interval": {"lower": lo, "upper": hi, "valid_draws": valid_draws, "total_draws": total_draws},
        "status": "ok",
    }


class Phase3E12StdlibTests(unittest.TestCase):
    def test_lazy_import_and_constants(self):
        self.assertEqual(E12_CONDITION, "image_caption")
        self.assertEqual(E12_POOLING, "caption_mean")
        self.assertEqual(E12_ARMS, ("identity_centered", "orthogonal_procrustes", "ridge_linear"))
        self.assertEqual(E12_METRICS, ("auroc", "ap", "brier", "nll"))
        self.assertEqual(EXPECTED_FIT_IMAGES, 2160)
        self.assertEqual(EXPECTED_DEV_S_IMAGES, 180)
        self.assertEqual(EXPECTED_SCREEN_IMAGES, 360)
        self.assertEqual(C_GRID, (0.01, 0.1, 1.0, 10.0))
        self.assertEqual(RIDGE_ALPHAS, (0.01, 0.1, 1.0, 10.0, 100.0))
        self.assertEqual(SUPERVISION_ADAPTED, "paired_image_adapted")
        self.assertEqual(len(_off_diagonal_pairs()), 12)
        self.assertEqual(_route_key("en", "hi"), "en->hi")

    def test_gap_and_improvement_sign_conventions(self):
        # Higher-is-better metrics
        self.assertAlmostEqual(_gap_value("auroc", 0.80, 0.70), 0.10)
        self.assertAlmostEqual(_improvement_value("auroc", 0.70, 0.75), 0.05)
        self.assertAlmostEqual(_gap_value("ap", 0.60, 0.50), 0.10)
        self.assertAlmostEqual(_improvement_value("ap", 0.50, 0.54), 0.04)
        # Lower-is-better loss metrics
        self.assertAlmostEqual(_gap_value("brier", 0.10, 0.15), 0.05)
        self.assertAlmostEqual(_improvement_value("brier", 0.15, 0.12), 0.03)
        self.assertAlmostEqual(_gap_value("nll", 0.30, 0.45), 0.15)
        self.assertAlmostEqual(_improvement_value("nll", 0.45, 0.38), 0.07)
        self.assertIsNone(_gap_value("auroc", None, 0.70))
        self.assertIsNone(_improvement_value("nll", 0.40, None))

    def test_decision_summary_rules(self):
        ranges = plain(PHASE3_CPU_DECISION_RANGES["E12"])

        def _arm(gap_val, gap_lo, gap_hi, frac_val, frac_lo, frac_hi):
            return {
                "gap_reduction": {"auroc": _make_interval(gap_val, gap_lo, gap_hi)},
                "recovered_fraction_of_g_res": {"auroc": _make_interval(frac_val, frac_lo, frac_hi)},
            }

        # Case 1: consequential recovery + constraint matters
        arms1 = {
            "orthogonal_procrustes": _arm(0.015, 0.005, 0.025, 0.40, 0.15, 0.65),
            "ridge_linear": _arm(0.030, 0.015, 0.045, 0.70, 0.35, 0.95),
        }
        contrasts1 = {"ridge_minus_procrustes": {"auroc": _make_interval(0.015, 0.004, 0.026)}}
        d1 = _evaluate_decision_summary(arms1, contrasts1, ranges)
        self.assertTrue(d1["procrustes_consequential"])
        self.assertTrue(d1["ridge_consequential"])
        self.assertTrue(d1["constraint_matters"])
        self.assertFalse(d1["negligible"])
        self.assertEqual(d1["outcome"], "consequential_recovery_with_constraint_difference")

        # Case 2: negligible
        arms2 = {
            "orthogonal_procrustes": _arm(0.001, -0.002, 0.004, 0.02, -0.05, 0.09),
            "ridge_linear": _arm(0.001, -0.002, 0.004, 0.02, -0.05, 0.09),
        }
        contrasts2 = {"ridge_minus_procrustes": {"auroc": _make_interval(0.0, -0.003, 0.003)}}
        d2 = _evaluate_decision_summary(arms2, contrasts2, ranges)
        self.assertFalse(d2["procrustes_consequential"])
        self.assertFalse(d2["ridge_consequential"])
        self.assertFalse(d2["constraint_matters"])
        self.assertTrue(d2["negligible"])
        self.assertEqual(d2["outcome"], "negligible")

    def test_parse_ref_arg_and_confirm_guard(self):
        self.assertIsNone(_parse_ref_arg(None))
        self.assertEqual(_parse_ref_arg('{"path": "a", "sha256": "b"}'), {"path": "a", "sha256": "b"})
        self.assertEqual(_parse_ref_arg("phase3/plans/x/_COMPLETE.json"), "phase3/plans/x/_COMPLETE.json")
        with self.assertRaises(ContractError):
            require_open_split("CONFIRM")

    def test_cli_help_and_blocked_exit(self):
        help_res = subprocess.run(
            [sys.executable, "-m", "src.phase3.run_e12", "--help"],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        self.assertEqual(help_res.returncode, 0)
        self.assertIn("--plan", help_res.stdout)
        self.assertIn("--resume-maps", help_res.stdout)
        self.assertIn("--resume-predictions", help_res.stdout)
        self.assertIn("--resume-analysis", help_res.stdout)

        with tempfile.TemporaryDirectory() as tmp:
            bad_res = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.phase3.run_e12",
                    "--phase1-root",
                    tmp,
                    "--plan",
                    "phase3/plans/missing/_COMPLETE.json",
                    "--model",
                    "qwen",
                    "--storage-scope",
                    "local-development",
                ],
                cwd=REPO,
                capture_output=True,
                text=True,
            )
            self.assertEqual(bad_res.returncode, 2)
            lines = [json.loads(line) for line in bad_res.stdout.strip().splitlines() if line.strip()]
            self.assertEqual(lines[-1]["status"], "blocked")
            self.assertFalse(lines[-1]["phase3_e12_complete"])

    def test_notebook_contract(self):
        nb_path = REPO / "scripts/phase3/run_e12.ipynb"
        self.assertTrue(nb_path.is_file())
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertEqual(len(nb["cells"]), 4)
        self.assertEqual(nb["cells"][0]["cell_type"], "markdown")
        for idx in (1, 2, 3):
            self.assertEqual(nb["cells"][idx]["cell_type"], "code")
        full = "\n".join("".join(c["source"]) for c in nb["cells"])
        self.assertIn("CODE_REF = 'phase3'", full)
        self.assertIn("MODEL = 'qwen'", full)
        self.assertIn("PLAN_REF", full)
        self.assertIn("OPENBLAS_NUM_THREADS", full)
        self.assertIn("requirements-stage2-cpu.txt", full)
        self.assertIn("src.phase3.run_e12", full)
        self.assertIn("e12_complete", full)


@unittest.skipUnless(HAS_NUMPY, "numpy/scipy required for numeric E12 tests")
class Phase3E12NumericTests(unittest.TestCase):
    def test_pca_basis_procrustes_and_ridge(self):
        rng = np.random.default_rng(42)
        # Low-rank matrix: rank 12 < 64
        A = rng.normal(size=(100, 12))
        B = rng.normal(size=(12, 20))
        X = A @ B
        mu, V, src_rank, r = _fit_pca_basis(X, max_rank=64)
        self.assertEqual(mu.shape, (20,))
        self.assertEqual(src_rank, 12)
        self.assertEqual(r, 12)
        self.assertTrue(np.allclose(V.T @ V, np.eye(12), atol=1e-10))

        # Orthogonal Procrustes recovers exact orthogonal rotation
        Z_s = rng.normal(size=(200, 12))
        Q_true, _ = np.linalg.qr(rng.normal(size=(12, 12)))
        Z_t = Z_s @ Q_true.T
        Q_hat = _fit_orthogonal_procrustes(Z_t, Z_s)
        self.assertTrue(np.allclose(Q_hat, Q_true, atol=1e-10))
        self.assertTrue(np.allclose(Q_hat.T @ Q_hat, np.eye(12), atol=1e-10))

        # Ridge linear selects alpha strictly by DEV-S MSE
        Z_dev_t = rng.normal(size=(50, 12))
        Z_dev_s = Z_dev_t @ Q_true + 0.05 * rng.normal(size=(50, 12))
        W_hat, best_alpha, mse_map = _fit_ridge_linear(Z_t, Z_s, Z_dev_t, Z_dev_s)
        self.assertIn(best_alpha, RIDGE_ALPHAS)
        self.assertEqual(set(mse_map.keys()), {str(float(a)) for a in RIDGE_ALPHAS})
        self.assertEqual(W_hat.shape, (12, 12))


if __name__ == "__main__":
    unittest.main()
