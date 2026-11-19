"""Unit and fixture tests for Phase 2 Stage 2 CPU E17 worker."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.common.contracts import ContractError, LANGUAGES
from src.phase2.interfaces import (
    KIND_ANALYSIS,
    KIND_CALIBRATORS,
    KIND_PREDICTION,
    REPO,
    SCHEMA_ANALYSIS,
    SCHEMA_CALIBRATORS,
    SCHEMA_PREDICTION,
    load_phase2_artifact,
    plain,
)
from src.phase2.run_e17 import (
    E17_ARMS,
    _select_arm_calibrator,
    fit_macro_bce_calibrator,
    run_e17,
)

HAS_SCIENTIFIC = all(importlib.util.find_spec(pkg) is not None for pkg in ("numpy", "scipy", "sklearn"))


def _load_protocol():
    return json.loads((REPO / "configs/protocol.json").read_text(encoding="utf-8"))


class E17NotebookAndContractTests(unittest.TestCase):
    """Tests that run on stdlib Python without numpy/scipy/sklearn."""

    def test_notebook_structure_and_code_ref(self):
        nb_path = REPO / "scripts/phase2/run_e17.ipynb"
        self.assertTrue(nb_path.is_file())
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertEqual(nb["nbformat"], 4)
        self.assertEqual(len(nb["cells"]), 4)
        setup_src = "".join(nb["cells"][1]["source"])
        cfg_src = "".join(nb["cells"][2]["source"])
        exec_src = "".join(nb["cells"][3]["source"])
        self.assertIn("CODE_REF = 'phase2'", setup_src)
        self.assertIn("OPENBLAS_NUM_THREADS", setup_src)
        self.assertIn("requirements-stage2-cpu.txt", setup_src)
        self.assertIn("MODEL = 'qwen'", cfg_src)
        self.assertIn("PLAN_REF = None", cfg_src)
        self.assertIn("E17_AUTH_REF = None", cfg_src)
        self.assertIn("RESUME_DEV_C_REFS = None", cfg_src)
        self.assertIn("RESUME_CALIBRATORS_REF = None", cfg_src)
        self.assertIn("RESUME_ANALYSIS_REF = None", cfg_src)
        self.assertIn("src.phase2.run_e17", exec_src)
        self.assertIn("MODEL not in ('qwen', 'gemma')", exec_src)

    def test_arm_calibrator_selection_separates_source_and_target_scopes(self):
        cal_body = {
            "source_dev_c": {
                lang: {
                    "temp": {"T": 1.2, "b": 0.0},
                    "temp_intercept": {"T": 1.3, "b": -0.1},
                }
                for lang in LANGUAGES
            },
            "target_dev_c": {
                f"{s}->{t}": {
                    "temp": {"T": 2.0, "b": 0.0},
                    "temp_intercept": {"T": 2.1, "b": 0.25},
                }
                for t in LANGUAGES
                for s in LANGUAGES
                if s != t
            },
        }
        raw_cal, raw_lbl = _select_arm_calibrator("raw", "en", "hi", cal_body)
        self.assertIsNone(raw_cal)
        self.assertEqual(raw_lbl, "source_only_transfer")

        src_cal, src_lbl = _select_arm_calibrator("source_dev_c:temp", "en", "hi", cal_body)
        self.assertEqual(src_cal["T"], 1.2)
        self.assertEqual(src_lbl, "source_dev_c_calibrated")

        tgt_cal, tgt_lbl = _select_arm_calibrator("target_dev_c:temp_intercept", "en", "hi", cal_body)
        self.assertEqual(tgt_cal["T"], 2.1)
        self.assertEqual(tgt_cal["b"], 0.25)
        self.assertEqual(tgt_lbl, "target_dev_c_supervised_calibration")

        diag_cal, diag_lbl = _select_arm_calibrator("target_dev_c:temp", "hi", "hi", cal_body)
        self.assertEqual(diag_cal["T"], 1.2)
        self.assertEqual(diag_lbl, "within_context_source_fit")

    def test_invalid_model_selector_rejected(self):
        protocol = _load_protocol()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ContractError):
                run_e17(
                    Path(tmp),
                    protocol,
                    {"path": "phase2/plans/x/_COMPLETE.json", "sha256": "0" * 64},
                    model="both",
                )


@unittest.skipUnless(HAS_SCIENTIFIC, "numpy/scipy/sklearn required for full E17 fixture tests")
class E17EndToEndFixtureTests(unittest.TestCase):
    """End-to-end E17 tests against a complete synthetic Phase 1 + Stage 1 fixture."""

    def _build_fixture(self, tmp_dir):
        import numpy as np
        from tests.test_phase2_gates_plan import _build_full_phase1_fixture
        from src.phase2.e17_auth import preflight_e17_dev_c
        from src.phase2.plan import load_phase2_plan, prepare_phase2

        root, protocol, stage4_ref, producer = _build_full_phase1_fixture(Path(tmp_dir))
        plan_ref, _ = prepare_phase2(root, protocol, stage4_release_ref=stage4_ref, producer=producer)
        plan = load_phase2_plan(root, protocol, plan_ref)
        auth_ref, _ = preflight_e17_dev_c(plan, producer, models=("qwen", "gemma"))
        return root, protocol, plan_ref, auth_ref, producer, np

    def test_calibrator_analytic_gradient_matches_finite_difference_and_bfgs_failure_blocks(self):
        import numpy as np

        rng = np.random.default_rng(20261119)
        z = rng.normal(0.0, 2.0, size=(60, 25))
        y = (rng.uniform(size=(60, 25)) < 0.3).astype(np.float64)
        known = np.ones((60, 25), dtype=bool)
        known[0, 5] = False
        primary_indices = list(range(14))

        for form in ("temp", "temp_intercept"):
            res = fit_macro_bce_calibrator(z, y, known, primary_indices, form)
            self.assertTrue(res["converged"])
            self.assertGreater(res["T"], 0.0)
            self.assertLessEqual(res["grad_norm"], 1e-10)
            self.assertLessEqual(res["dev_c_loss_cal"], res["dev_c_loss_raw"] + 1e-12)

        # Force non-convergence via maxiter=0 / mocked BFGS failure.
        fake_fail = type("FakeRes", (), {"success": False, "x": np.zeros(1), "fun": 1.0, "nit": 0, "jac": np.ones(1), "message": "forced"})()
        with patch("scipy.optimize.minimize", return_value=fake_fail):
            with self.assertRaises(ContractError):
                fit_macro_bce_calibrator(z, y, known, primary_indices, "temp")

    def test_end_to_end_e17_and_resume_and_diagonal_mismatch_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, protocol, plan_ref, auth_ref, producer, np = self._build_fixture(tmp)
            events = []
            res = plain(
                run_e17(
                    root,
                    protocol,
                    plan_ref,
                    model="qwen",
                    e17_auth_ref=auth_ref,
                    producer=producer,
                    run_id="e17_test_run",
                    progress=events.append,
                )
            )
            self.assertFalse(res["resumed"])
            self.assertTrue(res["diagonal_self_check"]["passed"])
            self.assertEqual(res["diagonal_self_check"]["max_abs_diff"], 0.0)
            self.assertEqual(set(res["dev_c_block_refs"].keys()), set(LANGUAGES))
            self.assertTrue(any(e.get("event") == "complete" for e in events))

            # Verify 4 DEV-C prediction blocks (3 sorted s != t routes each).
            for t_lang in LANGUAGES:
                _, pred_body, _ = load_phase2_artifact(
                    root,
                    protocol,
                    res["dev_c_block_refs"][t_lang],
                    KIND_PREDICTION,
                    expected_schema=SCHEMA_PREDICTION,
                    expected_owner="qwen",
                )
                expected_sources = sorted(s for s in LANGUAGES if s != t_lang)
                self.assertEqual(pred_body["source_languages"], expected_sources)
                self.assertEqual(len(pred_body["routes"]), 3)
                self.assertTrue(pred_body["confirm_excluded"])

            # Verify calibrators artifact.
            _, cal_body, _ = load_phase2_artifact(
                root,
                protocol,
                res["calibrators_ref"],
                KIND_CALIBRATORS,
                expected_schema=SCHEMA_CALIBRATORS,
                expected_owner="qwen",
            )
            self.assertEqual(set(cal_body["source_dev_c"].keys()), set(LANGUAGES))
            self.assertEqual(len(cal_body["target_dev_c"]), 12)
            for s_entry in cal_body["source_dev_c"].values():
                for form in ("temp", "temp_intercept"):
                    self.assertTrue(s_entry[form]["converged"])
                    self.assertGreater(s_entry[form]["T"], 0.0)
            for t_entry in cal_body["target_dev_c"].values():
                for form in ("temp", "temp_intercept"):
                    self.assertTrue(t_entry[form]["converged"])
                    self.assertGreater(t_entry[form]["T"], 0.0)

            # Verify SCREEN analysis artifact & invariances.
            _, ana_body, _ = load_phase2_artifact(
                root,
                protocol,
                res["analysis_ref"],
                KIND_ANALYSIS,
                expected_schema=SCHEMA_ANALYSIS,
                expected_owner="qwen",
            )
            self.assertEqual(tuple(ana_body["arms"]), E17_ARMS)
            self.assertTrue(ana_body["e19_cutoffs_untouched"])
            self.assertTrue(ana_body["confirm_excluded"])
            inv = ana_body["invariance_checks"]
            self.assertTrue(inv["all_T_positive"])
            self.assertEqual(inv["temp_only_0p5_flips"], 0)
            self.assertLessEqual(inv["max_abs_auroc_diff"], 1e-12)
            self.assertLessEqual(inv["max_abs_ap_diff"], 1e-12)
            self.assertIn("decision_0p5", ana_body)
            self.assertIn("fit_prior_reference", ana_body)

            # Explicit-ref resume at DEV-C + calibrator level and at analysis level.
            resumed_ana = plain(
                run_e17(
                    root,
                    protocol,
                    plan_ref,
                    model="qwen",
                    e17_auth_ref=auth_ref,
                    producer=producer,
                    resume_analysis_ref=res["analysis_ref"],
                )
            )
            self.assertTrue(resumed_ana["resumed"])
            self.assertEqual(resumed_ana["analysis_ref"], res["analysis_ref"])

            resumed_mid = plain(
                run_e17(
                    root,
                    protocol,
                    plan_ref,
                    model="qwen",
                    e17_auth_ref=auth_ref,
                    producer=producer,
                    run_id="e17_resume_mid",
                    resume_dev_c_refs=res["dev_c_block_refs"],
                    resume_calibrators_ref=res["calibrators_ref"],
                )
            )
            self.assertFalse(resumed_mid["resumed"])
            self.assertEqual(resumed_mid["calibrators_ref"], res["calibrators_ref"])

            # Wrong-model resume ref must fail.
            with self.assertRaises(ContractError):
                run_e17(
                    root,
                    protocol,
                    plan_ref,
                    model="gemma",
                    e17_auth_ref=auth_ref,
                    producer=producer,
                    resume_analysis_ref=res["analysis_ref"],
                )

            # Tampered diagonal reproduction must block before writing.
            with patch("src.phase2.run_e17.score_frozen_head", side_effect=lambda m, c, i, s, estimable=None: np.zeros((len(m), 25))):
                with self.assertRaises(ContractError):
                    run_e17(
                        root,
                        protocol,
                        plan_ref,
                        model="qwen",
                        e17_auth_ref=auth_ref,
                        producer=producer,
                        run_id="e17_should_fail",
                    )


if __name__ == "__main__":
    unittest.main()
