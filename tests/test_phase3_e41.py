"""Unit and contract tests for Phase 3 Stage A2 CPU Worker 2 (E41)."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

from src.common.contracts import ContractError, fingerprint
from src.phase3.interfaces import (
    KIND_ANALYSIS,
    KIND_CALIBRATORS,
    KIND_PREDICTION,
    REPO,
    SCHEMA_ANALYSIS,
    SCHEMA_CALIBRATORS,
    SCHEMA_PREDICTION,
    init_phase3_dirs,
    load_phase3_artifact,
    publish_phase3,
    require_open_split,
    require_phase3_namespace,
    verify_resume_ref,
)
import src.phase3.run_e41 as run_e41


def _load_protocol():
    return json.loads((REPO / "configs/protocol.json").read_text(encoding="utf-8"))


def _fake_producer():
    return {
        "code_commit": "0" * 40,
        "storage_scope": "local-development",
        "software_versions": {"python": "3.12.14", "numpy": "2.1.3"},
    }


class Phase3E41Tests(unittest.TestCase):
    def test_lazy_imports_and_constants(self):
        # Top-level import of src.phase3.run_e41 must not eagerly import numpy/scipy/stage4.
        self.assertNotIn("numpy", sys.modules)
        self.assertNotIn("scipy", sys.modules)
        self.assertEqual(
            run_e41.E41_ARMS,
            (
                "raw",
                "source_dev_c",
                "E11_diag",
                "E11_cov",
                "E11_diag+source_dev_c",
                "target_dev_c",
                "E11_diag+target_dev_c",
            ),
        )
        self.assertEqual(run_e41.E41_COV_RANKS, (64, 256))
        self.assertEqual(run_e41.BFGS_GTOL, 1e-10)
        self.assertEqual(run_e41.BFGS_MAXITER, 500)
        self.assertEqual(run_e41.INVARIANCE_TOL, 1e-12)
        self.assertEqual(
            require_phase3_namespace(run_e41.E41_PREDICTIONS_NAMESPACE),
            "phase3/predictions/e41_cov",
        )
        self.assertEqual(
            require_phase3_namespace(run_e41.E41_CALIBRATORS_NAMESPACE),
            "phase3/calibrators/e41",
        )
        self.assertEqual(
            require_phase3_namespace(run_e41.E41_RESULTS_NAMESPACE),
            "phase3/results/e41",
        )

    def test_protocol_and_confirm_guard(self):
        protocol = _load_protocol()
        self.assertEqual(len(protocol["category_order"]), 25)
        self.assertEqual(len(protocol["primary_categories"]), 14)
        self.assertEqual(int(protocol["minimum_positive_and_negative"]["SCREEN"]), 20)
        self.assertEqual(len(run_e41._primary_indices(protocol)), 14)

        with self.assertRaises(ContractError):
            require_open_split("CONFIRM")

    def test_e41_publish_and_resume_ref_contracts(self):
        protocol = _load_protocol()
        ds_fp = fingerprint({"test": "e41_resume"})
        plan_id = "plan_abc123"
        producer = _fake_producer()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            init_phase3_dirs(root)

            pred_ref = publish_phase3(
                root,
                protocol,
                ds_fp,
                f"{run_e41.E41_PREDICTIONS_NAMESPACE}/qwen",
                KIND_PREDICTION,
                {
                    "schema": SCHEMA_PREDICTION,
                    "experiment": "E41",
                    "plan_identity": plan_id,
                    "model": "qwen",
                    "depth": 18,
                    "pooling": "last_caption",
                    "variant": "E11_cov:r256",
                    "target_language": "hi",
                    "zero_target_labels": True,
                    "confirm_excluded": True,
                },
                producer,
            )
            cal_ref = publish_phase3(
                root,
                protocol,
                ds_fp,
                f"{run_e41.E41_CALIBRATORS_NAMESPACE}/qwen",
                KIND_CALIBRATORS,
                {
                    "schema": SCHEMA_CALIBRATORS,
                    "experiment": "E41",
                    "plan_identity": plan_id,
                    "model": "qwen",
                    "confirm_excluded": True,
                },
                producer,
                upstream=[pred_ref],
            )
            res_ref = publish_phase3(
                root,
                protocol,
                ds_fp,
                f"{run_e41.E41_RESULTS_NAMESPACE}/qwen",
                KIND_ANALYSIS,
                {
                    "schema": SCHEMA_ANALYSIS,
                    "experiment": "E41",
                    "plan_identity": plan_id,
                    "model": "qwen",
                    "arms": list(run_e41.E41_ARMS),
                    "calibrators_ref": cal_ref,
                    "confirm_excluded": True,
                },
                producer,
                upstream=[pred_ref, cal_ref],
            )

            _, pred_body, _ = verify_resume_ref(
                root,
                protocol,
                pred_ref,
                KIND_PREDICTION,
                expected_schema=SCHEMA_PREDICTION,
                expected_dataset=ds_fp,
                expected_plan_identity=plan_id,
                expected_owner="qwen",
            )
            self.assertTrue(pred_body["zero_target_labels"])
            self.assertTrue(pred_body["confirm_excluded"])

            _, cal_body, _ = verify_resume_ref(
                root,
                protocol,
                cal_ref,
                KIND_CALIBRATORS,
                expected_schema=SCHEMA_CALIBRATORS,
                expected_dataset=ds_fp,
                expected_plan_identity=plan_id,
                expected_owner="qwen",
            )
            self.assertEqual(cal_body["experiment"], "E41")

            _, res_body, _ = load_phase3_artifact(
                root,
                protocol,
                res_ref,
                KIND_ANALYSIS,
                expected_schema=SCHEMA_ANALYSIS,
                expected_dataset=ds_fp,
                expected_plan_identity=plan_id,
                expected_owner="qwen",
            )
            self.assertEqual(list(res_body["arms"]), list(run_e41.E41_ARMS))

            with self.assertRaises(ContractError):
                verify_resume_ref(
                    root,
                    protocol,
                    res_ref,
                    KIND_ANALYSIS,
                    expected_schema=SCHEMA_ANALYSIS,
                    expected_dataset=ds_fp,
                    expected_plan_identity=plan_id,
                    expected_owner="gemma",
                )

    def test_notebook_contract(self):
        nb_path = REPO / "scripts/phase3/run_e41.ipynb"
        self.assertTrue(nb_path.is_file())
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        cells = nb["cells"]
        self.assertEqual(len(cells), 4)
        self.assertEqual(cells[0]["cell_type"], "markdown")
        self.assertTrue(all(c["cell_type"] == "code" for c in cells[1:]))

        setup_src = "".join(cells[1]["source"])
        cfg_src = "".join(cells[2]["source"])
        run_src = "".join(cells[3]["source"])

        self.assertIn("CODE_REF = 'phase3'", setup_src)
        self.assertIn("OPENBLAS_NUM_THREADS", setup_src)
        self.assertIn("requirements-stage2-cpu.txt", setup_src)

        self.assertIn("PHASE1_ROOT = Path('/content/drive/MyDrive/phase1')", cfg_src)
        self.assertIn("MODEL = 'qwen'", cfg_src)
        self.assertIn("PLAN_REF = None", cfg_src)

        self.assertIn("src.phase3.run_e41", run_src)
        self.assertIn("subprocess.Popen", run_src)
        self.assertIn("status", run_src)

    def test_cli_parse_helpers(self):
        self.assertEqual(run_e41._parse_ref_arg('{"path": "x", "sha256": "y"}'), {"path": "x", "sha256": "y"})
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "refs.json"
            p.write_text('{"k": {"path": "p", "sha256": "s"}}', encoding="utf-8")
            self.assertEqual(run_e41._parse_ref_map_arg(str(p)), {"k": {"path": "p", "sha256": "s"}})
        with self.assertRaises(ContractError):
            run_e41._parse_ref_map_arg("/nonexistent/refs.json")

    def test_weighted_rank_metrics_avoids_expit_saturation(self):
        self.assertTrue(callable(run_e41._weighted_rank_metrics))
        try:
            import numpy as np
        except ImportError:
            return
        y = np.array([True, False, True, False], dtype=bool)
        z_raw = np.array([50.0, 42.0, 38.0, -5.0], dtype=np.float64)
        z_cal = z_raw / 3.0 - 0.5
        mass = np.ones((1, 4), dtype=np.int64)
        m_raw = run_e41._weighted_rank_metrics(y, z_raw, mass)
        m_cal = run_e41._weighted_rank_metrics(y, z_cal, mass)
        self.assertEqual(float(m_raw["auroc"][0]), float(m_cal["auroc"][0]))
        self.assertEqual(float(m_raw["ap"][0]), float(m_cal["ap"][0]))


if __name__ == "__main__":
    unittest.main()
