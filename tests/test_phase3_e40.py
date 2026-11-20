"""Unit and contract tests for Phase 3 Stage A2 CPU E40 worker and notebook."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from src.common.contracts import LANGUAGES, ContractError, fingerprint
from src.phase2.interfaces import POOLINGS, Phase1Binding
from src.phase3.interfaces import (
    CONDITIONS,
    MODELS,
    PHASE3_CPU_DECISION_RANGES,
    REPO,
    frozen,
)
from src.phase3.run_e40 import (
    ALL_ARMS,
    CONTRAST_ORDER,
    CROSSED_ARMS,
    MATCHED_ARMS,
    MIDPOINT_PARITY_TOL,
    _build_decision_summary,
    _gap_value,
    _improvement_value,
    _new_cells,
    _parse_hook_from_site,
    _valid_crossed_resume_keys,
    _verify_midpoint_e30_analysis_parity,
    run_e40,
    verify_e40_cell_provenance,
)

HAS_SCIENTIFIC = all(importlib.util.find_spec(pkg) is not None for pkg in ("numpy", "scipy", "sklearn"))


def _load_protocol():
    return json.loads((REPO / "configs/protocol.json").read_text(encoding="utf-8"))


class _DummyMatrix:
    """Minimal shape-only stub for stdlib provenance tests without numpy."""

    def __init__(self, shape):
        self.shape = tuple(shape)


class _PlanDict(dict):
    pass


def _make_provenance_fixture(*, n_images=12, dim=8, depth=9, condition="caption_only", model="qwen"):
    screen_ids = tuple(f"img-{i:04d}" for i in range(n_images))
    model_plan_ref = {"path": f"shared/mp_{model}/_COMPLETE.json", "sha256": "a" * 64}
    groups = {}
    gids_by_pooling = {p: {} for p in POOLINGS}
    rows_by_pooling = {p: {} for p in POOLINGS}
    matrices_by_pooling = {p: {} for p in POOLINGS}

    for pool in POOLINGS:
        for lang in LANGUAGES:
            gid = f"grp-{model}-d{depth}-{condition}-{pool}-{lang}"
            gids_by_pooling[pool][lang] = gid
            groups[gid] = {
                "group_id": gid,
                "source": "vlm",
                "model": model,
                "condition": condition,
                "site": f"decoder:{depth}:{pool}",
                "language": lang,
                "dimension": dim,
                "model_plan": model_plan_ref,
                "releases": {
                    "SCREEN": {
                        "path": f"features/{model}/{condition}/SCREEN/_COMPLETE.json",
                        "sha256": fingerprint([model, condition, "SCREEN"]),
                    }
                },
            }
            rows = tuple(
                {"image_id": iid, "split": "SCREEN", "language": lang, "row_index": idx}
                for idx, iid in enumerate(screen_ids)
            )
            rows_by_pooling[pool][lang] = rows
            matrices_by_pooling[pool][lang] = _DummyMatrix((n_images, dim))

    stage4_plan = _PlanDict({"screen_ids": list(screen_ids)})
    stage4_plan.fit = {"groups": groups}

    binding = Phase1Binding(
        root=Path("."),
        protocol=frozen({}),
        stage4_release_ref=frozen({"path": "results/s4/_COMPLETE.json", "sha256": "1" * 64}),
        stage4_release=frozen({}),
        stage4_plan=stage4_plan,
        bootstrap_shape=(2000, n_images),
        bootstrap_sha256="2" * 64,
        midpoints=frozen({"qwen": 18, "gemma": 17}),
        dataset_ref=frozen({"path": "shared/ds/_COMPLETE.json", "sha256": "3" * 64}),
        data_plan_ref=frozen({"path": "shared/dp/_COMPLETE.json", "sha256": "4" * 64}),
        model_plan_refs=frozen({
            "qwen": model_plan_ref,
            "gemma": {"path": "shared/mp_gemma/_COMPLETE.json", "sha256": "b" * 64},
        }),
        suite_ref=frozen({"path": "results/suite/_COMPLETE.json", "sha256": "5" * 64}),
        stage3_registry_ref=frozen({"path": "results/s3/_COMPLETE.json", "sha256": "6" * 64}),
        g0_gate_ref=frozen({"path": "results/g0/_COMPLETE.json", "sha256": "7" * 64}),
        g0_p_value=0.362,
        dataset_fingerprint="8" * 64,
        protocol_fingerprint="9" * 64,
        lineage_identity="0" * 64,
    )
    return binding, gids_by_pooling, rows_by_pooling, matrices_by_pooling


class _FakeCpuPlan(dict):
    def __init__(self, body, protocol, binding):
        super().__init__(body)
        self.protocol = frozen(protocol)
        self.binding = binding
        self.reference = frozen({"path": "phase3/plans/test/_COMPLETE.json", "sha256": "a" * 64})


class E40ContractAndNotebookTests(unittest.TestCase):
    """Stdlib-only tests verifying lazy imports, notebook, cell grid, provenance, and helpers."""

    def test_lazy_imports_without_scientific_stack(self):
        # Importing src.phase3.run_e40 must not force-import numpy/scipy/stage4 at module import time.
        for mod_name in ("numpy", "scipy", "src.stage4.metrics", "src.stage3.engine"):
            if not HAS_SCIENTIFIC:
                self.assertNotIn(mod_name, sys.modules)

    def test_notebook_structure_and_code_ref(self):
        nb_path = REPO / "scripts/phase3/run_e40.ipynb"
        self.assertTrue(nb_path.is_file(), "Missing scripts/phase3/run_e40.ipynb")
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertEqual(nb.get("nbformat"), 4)
        self.assertEqual(len(nb.get("cells", [])), 4)
        setup_src = "".join(nb["cells"][1]["source"])
        cfg_src = "".join(nb["cells"][2]["source"])
        exec_src = "".join(nb["cells"][3]["source"])

        self.assertIn("CODE_REF = 'phase3'", setup_src)
        self.assertIn("OPENBLAS_NUM_THREADS", setup_src)
        self.assertIn("requirements-stage2-cpu.txt", setup_src)
        self.assertIn("MODEL = 'qwen'", cfg_src)
        self.assertIn("PLAN_REF = None", cfg_src)
        self.assertIn("RESUME_CROSSED_REFS = None", cfg_src)
        self.assertIn("RESUME_ANALYSIS_REF = None", cfg_src)
        self.assertIn("OUTPUT_REF = None", cfg_src)
        self.assertIn("src.phase3.run_e40", exec_src)
        self.assertIn("MODEL not in ('qwen', 'gemma')", exec_src)

    def test_new_cells_and_resume_keys(self):
        qwen_depths = [9, 18, 27, 36]
        gemma_depths = [9, 17, 26, 34]

        qwen_new = _new_cells(qwen_depths, 18)
        self.assertEqual(len(qwen_new), 7)
        self.assertNotIn((18, "image_caption"), qwen_new)
        self.assertIn((18, "caption_only"), qwen_new)

        qwen_keys = _valid_crossed_resume_keys(qwen_depths, 18)
        self.assertEqual(len(qwen_keys), 56)  # 7 cells * 2 crossed arms * 4 languages
        self.assertFalse(any(k.startswith("18:image_caption:") for k in qwen_keys))
        self.assertIn("18:caption_only:head_last_on_mean:en", qwen_keys)
        self.assertIn("36:image_caption:head_mean_on_last:te", qwen_keys)

        gemma_keys = _valid_crossed_resume_keys(gemma_depths, 17)
        self.assertEqual(len(gemma_keys), 56)
        self.assertFalse(any(k.startswith("17:image_caption:") for k in gemma_keys))
        self.assertIn("17:caption_only:head_last_on_mean:hi", gemma_keys)

    def test_invalid_model_and_resume_keys_rejected(self):
        protocol = _load_protocol()
        dummy_ref = {"path": "phase3/plans/x/_COMPLETE.json", "sha256": "0" * 64}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ContractError):
                run_e40(Path(tmp), protocol, dummy_ref, "both", {})

            binding, _, _, _ = _make_provenance_fixture()
            fake_plan = _FakeCpuPlan(
                {
                    "confirm_excluded": True,
                    "midpoints": {"qwen": 18, "gemma": 17},
                    "e40_inventory": {"qwen": {"depths": [9, 18, 27, 36]}},
                },
                protocol,
                binding,
            )
            with patch("src.phase3.run_e40.load_phase3_cpu_plan", return_value=fake_plan):
                # Reused midpoint image_caption key must be rejected as resume key.
                with self.assertRaises(ContractError):
                    run_e40(
                        Path(tmp),
                        protocol,
                        dummy_ref,
                        "qwen",
                        {},
                        resume_crossed_refs={"18:image_caption:head_last_on_mean:en": dummy_ref},
                    )
                # Matched arm key must be rejected.
                with self.assertRaises(ContractError):
                    run_e40(
                        Path(tmp),
                        protocol,
                        dummy_ref,
                        "qwen",
                        {},
                        resume_crossed_refs={"9:image_caption:last_on_last:en": dummy_ref},
                    )

    def test_provenance_checks_pass_and_reject_mismatches(self):
        binding, gids, rows, mats = _make_provenance_fixture(depth=9, condition="caption_only")
        prov = verify_e40_cell_provenance(binding, "qwen", 9, "caption_only", gids, rows, mats)
        self.assertTrue(prov["depth_match"])
        self.assertTrue(prov["hook_match"])
        self.assertTrue(prov["model_plan_match"])
        self.assertTrue(prov["row_keys_match"])
        self.assertEqual(prov["hook"], "decoder:9")
        self.assertEqual(prov["dimension"], 8)

        # Depth mismatch
        binding_d, gids_d, rows_d, mats_d = _make_provenance_fixture(depth=9, condition="caption_only")
        binding_d.stage4_plan.fit["groups"][gids_d["caption_mean"]["hi"]]["site"] = "decoder:18:caption_mean"
        with self.assertRaises(ContractError):
            verify_e40_cell_provenance(binding_d, "qwen", 9, "caption_only", gids_d, rows_d, mats_d)

        # Condition mismatch
        binding_c, gids_c, rows_c, mats_c = _make_provenance_fixture(depth=9, condition="caption_only")
        binding_c.stage4_plan.fit["groups"][gids_c["caption_mean"]["en"]]["condition"] = "image_caption"
        with self.assertRaises(ContractError):
            verify_e40_cell_provenance(binding_c, "qwen", 9, "caption_only", gids_c, rows_c, mats_c)

        # Row key mismatch across poolings
        binding_r, gids_r, rows_r, mats_r = _make_provenance_fixture(depth=9, condition="caption_only")
        rows_r["caption_mean"]["bn"] = tuple(reversed(rows_r["caption_mean"]["bn"]))
        with self.assertRaises(ContractError):
            verify_e40_cell_provenance(binding_r, "qwen", 9, "caption_only", gids_r, rows_r, mats_r)

        # CONFIRM split row rejection
        binding_s, gids_s, rows_s, mats_s = _make_provenance_fixture(depth=9, condition="caption_only")
        bad_row = dict(rows_s["last_caption"]["te"][0], split="CONFIRM")
        rows_s["last_caption"]["te"] = (bad_row,) + rows_s["last_caption"]["te"][1:]
        rows_s["caption_mean"]["te"] = (bad_row,) + rows_s["caption_mean"]["te"][1:]
        with self.assertRaises(ContractError):
            verify_e40_cell_provenance(binding_s, "qwen", 9, "caption_only", gids_s, rows_s, mats_s)

    def test_gap_improvement_and_hook_helpers(self):
        self.assertAlmostEqual(_gap_value("auroc", 0.85, 0.75), 0.10)
        self.assertAlmostEqual(_gap_value("nll", 0.30, 0.45), 0.15)
        self.assertAlmostEqual(_improvement_value("auroc", 0.70, 0.80), 0.10)
        self.assertAlmostEqual(_improvement_value("brier", 0.25, 0.18), 0.07)
        self.assertEqual(_parse_hook_from_site("decoder:27:caption_mean", "caption_mean"), ("decoder:27", 27))
        with self.assertRaises(ContractError):
            _parse_hook_from_site("decoder:27:last_caption", "caption_mean")

    def test_midpoint_e30_parity_check(self):
        fake_metrics_mod = types.ModuleType("src.stage4.metrics")
        fake_metrics_mod.METRICS = ("auroc", "ap", "brier", "nll")
        fake_stage4_pkg = types.ModuleType("src.stage4")

        def _make_cell(offset=0.0):
            arms = {}
            for arm in ALL_ARMS:
                arms[arm] = {
                    sk: {m: {"primary_macro": {"value": 0.5 + offset}} for m in fake_metrics_mod.METRICS}
                    for sk in ("twelve_route_mean_gap", "twelve_route_mean_transfer", "four_diagonal_mean")
                }
            contrasts = {
                cname: {
                    "metrics": {
                        m: {"gap_reduction": {"primary_macro": {"value": 0.1 + offset}}}
                        for m in fake_metrics_mod.METRICS
                    }
                }
                for cname in CONTRAST_ORDER
            }
            return {"arms": arms, "contrasts": contrasts}

        with patch.dict(sys.modules, {"src.stage4": fake_stage4_pkg, "src.stage4.metrics": fake_metrics_mod}):
            diff = _verify_midpoint_e30_analysis_parity(_make_cell(0.0), _make_cell(1e-13), tol=MIDPOINT_PARITY_TOL)
            self.assertLessEqual(diff, MIDPOINT_PARITY_TOL)
            with self.assertRaises(ContractError):
                _verify_midpoint_e30_analysis_parity(_make_cell(0.0), _make_cell(1e-9), tol=MIDPOINT_PARITY_TOL)

    def test_decision_summary_logic(self):
        depths = [9, 18, 27, 36]
        cells_out = {}
        for d in depths:
            cells_out[str(d)] = {}
            for cond in CONDITIONS:
                val = 0.04 if (d == 18 and cond == "image_caption") else (0.03 if (d == 18 and cond == "caption_only") else 0.002)
                lo, hi = (val - 0.005, val + 0.005)
                entry = {"value": val, "interval": {"valid_draws": 2000, "lower": lo, "upper": hi}}
                cells_out[str(d)][cond] = {
                    "contrasts": {
                        "feature_gain_fixed_head_last": {"metrics": {"auroc": {"gap_reduction": {"primary_macro": entry}}}},
                        "feature_gain_fixed_head_mean": {"metrics": {"auroc": {"gap_reduction": {"primary_macro": entry}}}},
                        "interaction": {"metrics": {"auroc": {"gap_reduction": {"primary_macro": entry}}}},
                    }
                }
        summary = _build_decision_summary(cells_out, depths, 18, dict(PHASE3_CPU_DECISION_RANGES["E40"]))
        self.assertTrue(summary["midpoint_caption_only_replication"]["replicates"])
        self.assertAlmostEqual(summary["midpoint_caption_only_replication"]["ratio_caption_only_to_image_caption"], 0.75)
        self.assertTrue(summary["final_depth_localized"])
        self.assertTrue(summary["per_cell"]["36"]["image_caption"]["swap_asymmetry_absent"])
        self.assertFalse(summary["per_cell"]["18"]["image_caption"]["swap_asymmetry_absent"])


if __name__ == "__main__":
    unittest.main()
