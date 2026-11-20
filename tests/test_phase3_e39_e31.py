"""Unit, contract, and synthetic-fixture tests for Phase 3 Stage B3 Worker 1 (`run_e39_e31`)."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HAS_SCIENTIFIC = all(
    importlib.util.find_spec(pkg) is not None for pkg in ("numpy", "scipy", "sklearn")
)

from src.common.contracts import ContractError, LANGUAGES, canonical_bytes, fingerprint
from src.common.storage import MARKER, file_sha256, publish
from src.phase3.extract_e39_e31 import expected_array_specs
from src.phase3.interfaces import (
    CERTIFIED_STAGE_B1_GPU_PLAN,
    CERTIFIED_STAGE_B2_RELEASES,
    CONDITIONS,
    E31_SITES,
    E39_SITES,
    KIND_ANALYSIS,
    KIND_E39_E31_CHUNK,
    KIND_E39_E31_RELEASE,
    KIND_GPU_PLAN,
    KIND_PROBE,
    KIND_PROBE_PAYLOAD,
    MODELS,
    PHASE3_GPU_DECISION_RANGES,
    PHASE3_GPU_VERSION,
    PHASE3_PARENT_DIRS,
    POOLINGS,
    REPO,
    SCHEMA_ANALYSIS,
    SCHEMA_E39_E31_FEATURES,
    SCHEMA_GPU_PLAN,
    SCHEMA_PROBE,
    frozen,
    init_phase3_dirs,
    plain,
    publish_phase3,
    select_e39_token_indices,
)
import src.phase3.run_e39_e31 as run_e39_e31_mod
from src.phase3.run_e39_e31 import (
    ATTENTION_SLOTS,
    CV_C,
    CV_FOLDS,
    E39_E31_RESULTS_NAMESPACE,
    E39_E31_SPLIT,
    HEAD_MODES,
    MATCHED_HEAD_FOR_SITE,
    REPORT_METRICS,
    _resolve_vlm_groups_and_blocks,
    assign_screen_cv_folds,
    evaluate_e31_decisions,
    evaluate_e39_decisions,
    pearson_and_spearman,
    reassemble_e39_e31_release,
    run_e39_e31,
    summarize_e39_token_selection_audit,
    verify_stage2_draw0_parity,
)


def _load_protocol():
    return json.loads((REPO / "configs/protocol.json").read_text(encoding="utf-8"))


def _publish_probe_without_subprocess(root: Path, protocol: dict, dataset_fp: str, producer: dict):
    payload_ref = publish_phase3(
        root,
        protocol,
        dataset_fp,
        "phase3/probes",
        KIND_PROBE_PAYLOAD,
        {"schema": SCHEMA_PROBE, "nonce": "probe-nonce", "remote_sync_guarantee": False},
        producer,
    )
    return publish_phase3(
        root,
        protocol,
        dataset_fp,
        "phase3/probes",
        KIND_PROBE,
        {
            "schema": SCHEMA_PROBE,
            "payload": plain(payload_ref),
            "independent_process_readback": True,
            "reader_pid": os.getpid() + 1,
            "writer_pid": os.getpid(),
            "remote_sync_guarantee": False,
        },
        producer,
        upstream=[plain(payload_ref)],
    )


class _FakeGpuPlan(dict):
    def __init__(self, body, reference, root, protocol, binding=None):
        super().__init__(body)
        self.reference = frozen(reference)
        self.root = Path(root).resolve()
        self.protocol = frozen(protocol)
        self.binding = binding


class RunE39E31StdlibContractTests(unittest.TestCase):
    """Stdlib-only contract tests running cleanly without numpy, scipy, or sklearn."""

    def test_lazy_imports_and_constants(self):
        if not HAS_SCIENTIFIC:
            for mod_name in ("numpy", "scipy", "sklearn", "src.stage3.engine", "src.stage4.metrics"):
                self.assertNotIn(mod_name, sys.modules)

        self.assertEqual(E39_E31_SPLIT, "SCREEN")
        self.assertEqual(E39_E31_RESULTS_NAMESPACE, "phase3/results/e39_e31")
        self.assertEqual(REPORT_METRICS, ("auroc", "ap", "brier", "nll"))
        self.assertEqual(HEAD_MODES, ("fixed_head_last", "fixed_head_mean", "matched_head"))
        self.assertEqual(ATTENTION_SLOTS, ("visual_mass", "caption_mass", "system_mass"))
        self.assertEqual(CV_FOLDS, 5)
        self.assertEqual(CV_C, 1.0)
        self.assertEqual(set(MATCHED_HEAD_FOR_SITE.keys()), set(E39_SITES))
        for single_site in (
            "first_caption",
            "middle_caption",
            "penultimate_caption",
            "last_non_punct_caption",
            "last_caption",
        ):
            self.assertEqual(MATCHED_HEAD_FOR_SITE[single_site], "last_caption")
        for span_site in ("first_half_mean", "second_half_mean", "caption_mean"):
            self.assertEqual(MATCHED_HEAD_FOR_SITE[span_site], "caption_mean")

        self.assertEqual(set(CERTIFIED_STAGE_B2_RELEASES["e39_e31"].keys()), {"qwen", "gemma"})
        for m in MODELS:
            self.assertEqual(len(CERTIFIED_STAGE_B2_RELEASES["e39_e31"][m]["sha256"]), 64)

    def test_cli_help_and_blocked_exit(self):
        help_res = subprocess.run(
            [sys.executable, "-m", "src.phase3.run_e39_e31", "--help"],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        self.assertEqual(help_res.returncode, 0, help_res.stderr)
        for flag in (
            "--model",
            "--phase1-root",
            "--plan",
            "--release",
            "--protocol",
            "--resume-analysis",
            "--run-id",
            "--storage-scope",
        ):
            self.assertIn(flag, help_res.stdout)

        with tempfile.TemporaryDirectory() as tmp:
            missing_proto = Path(tmp) / "missing_protocol.json"
            blocked_res = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.phase3.run_e39_e31",
                    "--model",
                    "qwen",
                    "--phase1-root",
                    tmp,
                    "--protocol",
                    str(missing_proto),
                ],
                cwd=REPO,
                capture_output=True,
                text=True,
            )
            self.assertEqual(blocked_res.returncode, 2)
            payload = json.loads(blocked_res.stdout.strip().splitlines()[-1])
            self.assertEqual(payload["status"], "blocked")
            self.assertFalse(payload["phase3_e39_e31_complete"])
            self.assertTrue(len(payload["blockers"]) >= 1)

    def test_notebook_contract(self):
        nb_path = REPO / "scripts/phase3/run_e39_e31.ipynb"
        self.assertTrue(nb_path.is_file(), "Missing scripts/phase3/run_e39_e31.ipynb")
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertEqual(nb.get("nbformat"), 4)
        self.assertEqual(len(nb.get("cells", [])), 4)
        self.assertEqual(nb["cells"][0]["cell_type"], "markdown")

        for idx, cell in enumerate(nb["cells"][1:], start=1):
            self.assertEqual(cell["cell_type"], "code")
            self.assertEqual(cell["outputs"], [])
            self.assertIsNone(cell["execution_count"])
            compile("".join(cell["source"]), f"cell_{idx}", "exec")

        setup_src = "".join(nb["cells"][1]["source"])
        cfg_src = "".join(nb["cells"][2]["source"])
        exec_src = "".join(nb["cells"][3]["source"])

        self.assertIn("CODE_REF = 'phase3'", setup_src)
        self.assertIn("OPENBLAS_NUM_THREADS", setup_src)
        self.assertIn("requirements-stage2-cpu.txt", setup_src)
        self.assertIn("MODEL = 'qwen'", cfg_src)
        self.assertIn(CERTIFIED_STAGE_B1_GPU_PLAN["path"], cfg_src)
        self.assertIn(CERTIFIED_STAGE_B1_GPU_PLAN["sha256"], cfg_src)
        self.assertIn("RELEASE_REF = None", cfg_src)
        self.assertIn("RESUME_ANALYSIS_REF = None", cfg_src)
        self.assertIn("OUTPUT_REF = None", cfg_src)
        self.assertIn("src.phase3.run_e39_e31", exec_src)
        self.assertIn("MODEL not in ('qwen', 'gemma')", exec_src)

    def test_model_and_plan_safety_flag_rejections(self):
        protocol = _load_protocol()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ContractError, "Unsupported MODEL"):
                run_e39_e31(root, protocol, "llama", {})

            for bad_body in (
                {"confirm_excluded": False, "e35_excluded": True},
                {"confirm_excluded": True, "e35_excluded": False},
            ):
                fake_plan = _FakeGpuPlan(
                    bad_body,
                    CERTIFIED_STAGE_B1_GPU_PLAN,
                    root,
                    protocol,
                    binding=SimpleNamespace(),
                )
                with mock.patch("src.phase3.run_e39_e31.load_phase3_gpu_plan", return_value=fake_plan):
                    with self.assertRaises(ContractError):
                        run_e39_e31(root, protocol, "qwen", {})

    def test_token_selection_audit_and_validation(self):
        screen_ids = ("img_0", "img_1")
        sel_en = select_e39_token_indices(["A", "brown", "dog", "."])
        sel_hi = select_e39_token_indices(["एक", "कुत्ता", "।"])
        sel_bn = select_e39_token_indices(["কুকুর"])
        sel_te_punct = select_e39_token_indices([".", "!"])

        by_cond = {
            "image_caption": {
                "en": [sel_en, sel_en],
                "hi": [sel_hi, sel_hi],
                "bn": [sel_bn, sel_bn],
                "te": [sel_te_punct, sel_en],
            },
            "caption_only": {
                "en": [sel_en, sel_en],
                "hi": [sel_hi, sel_hi],
                "bn": [sel_bn, sel_bn],
                "te": [sel_te_punct, sel_hi],  # 1 view differs from image_caption
            },
        }
        audit = summarize_e39_token_selection_audit(by_cond, screen_ids)
        ic_overall = audit["by_condition"]["image_caption"]["overall"]
        self.assertEqual(ic_overall["n_views"], 8)
        self.assertEqual(ic_overall["token_count"]["min"], 1)
        self.assertEqual(ic_overall["token_count"]["max"], 4)
        # en (2), hi (2), and te[1] (1) have last_non_punct != last -> 5 of 8
        self.assertEqual(ic_overall["last_non_punct_differs_from_last_count"], 5)
        self.assertAlmostEqual(ic_overall["last_non_punct_differs_from_last_fraction"], 5 / 8)
        # te[0] used all-punct fallback -> 1 of 8
        self.assertEqual(ic_overall["last_non_punct_fallback_used_count"], 1)
        self.assertAlmostEqual(ic_overall["last_non_punct_fallback_used_fraction"], 1 / 8)

        # 7 of 8 views match between image_caption and caption_only
        self.assertEqual(audit["cross_condition_parity"]["identical_selection_count"], 7)
        self.assertAlmostEqual(audit["cross_condition_parity"]["identical_selection_fraction"], 7 / 8)

        bad_sel = dict(sel_en, last_caption=0)
        bad_cond = dict(by_cond, image_caption=dict(by_cond["image_caption"], en=[bad_sel, sel_en]))
        with self.assertRaises(ContractError):
            summarize_e39_token_selection_audit(bad_cond, screen_ids)

    def test_assign_screen_cv_folds_and_correlations(self):
        folds = assign_screen_cv_folds(12, 5)
        self.assertEqual(folds, [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1])
        with self.assertRaises(ContractError):
            assign_screen_cv_folds(4, 5)

        pos = pearson_and_spearman([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0])
        self.assertEqual(pos["n"], 4)
        self.assertAlmostEqual(pos["pearson_r"], 1.0, places=12)
        self.assertAlmostEqual(pos["spearman_rho"], 1.0, places=12)

        neg = pearson_and_spearman([1.0, 2.0, 3.0, 4.0], [40.0, 30.0, 20.0, 10.0])
        self.assertAlmostEqual(neg["pearson_r"], -1.0, places=12)
        self.assertAlmostEqual(neg["spearman_rho"], -1.0, places=12)

        # Tie handling and None filtering
        tied = pearson_and_spearman([1.0, 2.0, 2.0, 4.0, None], [1.0, 3.0, 2.0, 4.0, 99.0])
        self.assertEqual(tied["n"], 4)
        self.assertTrue(0.8 < tied["spearman_rho"] <= 1.0)

        const = pearson_and_spearman([5.0, 5.0, 5.0], [1.0, 2.0, 3.0])
        self.assertIsNone(const["pearson_r"])
        self.assertEqual(const["reason"], "zero_variance")

        short = pearson_and_spearman([1.0], [2.0])
        self.assertIsNone(short["pearson_r"])
        self.assertEqual(short["reason"], "insufficient_pairs")

    def test_e39_and_e31_decision_rule_evaluators(self):
        depths = [9, 18, 27, 36]
        midpoint = 18

        def _make_site_entry(gain_val, gain_lo, gain_hi, rec_val, rec_lo=0.4, rec_hi=0.8):
            return {
                "gain_vs_last_caption": {
                    "twelve_route_mean_transfer": {
                        "auroc": {
                            "primary_macro": {
                                "value": gain_val,
                                "interval": {"lower": gain_lo, "upper": gain_hi, "valid_draws": 2000},
                            }
                        }
                    }
                },
                "recovery_fraction_of_caption_mean": {
                    "twelve_route_mean_transfer": {
                        "auroc": {
                            "primary_macro": {
                                "value": rec_val,
                                "interval": {"lower": rec_lo, "upper": rec_hi, "valid_draws": 2000},
                            }
                        }
                    }
                },
            }

        cells = {}
        for d in depths:
            cells[str(d)] = {}
            for cond in CONDITIONS:
                modes = {}
                for mode in HEAD_MODES:
                    sites = {}
                    for site in E39_SITES:
                        if site == "last_non_punct_caption":
                            # Negligible within (-0.005, 0.005)
                            sites[site] = _make_site_entry(0.001, -0.002, 0.003, 0.03)
                        elif site == "penultimate_caption":
                            sites[site] = _make_site_entry(0.004, -0.001, 0.008, 0.12)
                        elif site in ("first_half_mean", "second_half_mean", "caption_mean"):
                            sites[site] = _make_site_entry(0.030, 0.018, 0.042, 0.85)
                        elif site == "last_caption":
                            sites[site] = _make_site_entry(0.0, 0.0, 0.0, 0.0)
                        else:
                            sites[site] = _make_site_entry(0.008, 0.001, 0.014, 0.25)
                    modes[mode] = {"sites": sites}
                cells[str(d)][cond] = {"head_modes": modes}

        e39_dec = evaluate_e39_decisions(
            {"cells": cells}, depths, midpoint, plain(PHASE3_GPU_DECISION_RANGES["E39"])
        )
        mid_s = e39_dec["midpoint_summary"]
        self.assertFalse(mid_s["terminal_token_artifact_supported"])
        self.assertTrue(mid_s["last_non_punct_negligible_under_fixed_head_last"])
        self.assertTrue(mid_s["span_mean_pooling_required"])

        # Now flip last_non_punct_caption to consequential (>= 0.015 AUROC gain and >= 0.50 recovery)
        cells["18"]["image_caption"]["head_modes"]["fixed_head_last"]["sites"]["last_non_punct_caption"] = (
            _make_site_entry(0.022, 0.010, 0.034, 0.65)
        )
        e39_dec2 = evaluate_e39_decisions(
            {"cells": cells}, depths, midpoint, plain(PHASE3_GPU_DECISION_RANGES["E39"])
        )
        self.assertTrue(e39_dec2["midpoint_summary"]["terminal_token_artifact_supported"])

        # E31 decision evaluator
        e31_mock = {
            "attention_mass_check": {"passed": True, "max_row_sum_abs_error": 1e-6, "tol": 1e-4},
            "visual_token_probe": {
                "decoder_visual_mean": {
                    str(d): {
                        "per_language": {
                            "en": {"metrics": {"auroc": {"primary_macro": {"value": 0.92 if d == 18 else 0.88}}}}
                        },
                        "four_language_mean": {
                            "metrics": {"auroc": {"primary_macro": {"value": 0.91 if d == 18 else 0.87}}}
                        },
                    }
                    for d in depths
                }
            },
            "attention_gap_correlations": {
                "auroc": {
                    "by_depth_and_route": {
                        "correlations": {
                            "vis_shift_target": {
                                "G1_reduction": {"n": 48, "pearson_r": 0.42, "spearman_rho": 0.39}
                            }
                        }
                    }
                }
            },
        }
        e31_dec = evaluate_e31_decisions(
            e31_mock, depths, midpoint, plain(PHASE3_GPU_DECISION_RANGES["E31"])
        )
        self.assertTrue(e31_dec["attention_mass_sum_check_passed"])
        self.assertTrue(e31_dec["decoder_visual_probe"]["visual_tokens_retain_semantics_at_midpoint"])
        self.assertTrue(e31_dec["decoder_visual_probe"]["visual_tokens_retain_semantics_any_depth"])
        self.assertTrue(e31_dec["attention_gap_correlation"]["attention_correlates_with_gap"])
        self.assertAlmostEqual(
            e31_dec["attention_gap_correlation"]["strongest_correlation"]["pearson_r"], 0.42
        )

    def test_resolve_vlm_groups_and_blocks_and_resume_fast_path(self):
        protocol = _load_protocol()
        depths = [9, 18, 27, 36]
        groups = {}
        blocks = {}
        preds = {}

        for d in depths:
            for cond in CONDITIONS:
                for pool in POOLINGS:
                    for lang in LANGUAGES:
                        gid = f"g_qwen_{cond}_d{d}_{pool}_{lang}"
                        groups[gid] = {
                            "group_id": gid,
                            "source": "vlm",
                            "model": "qwen",
                            "condition": cond,
                            "site": f"decoder:{d}:{pool}",
                            "language": lang,
                        }
        for d in depths:
            for cond in CONDITIONS:
                for pool in POOLINGS:
                    src_gids = [f"g_qwen_{cond}_d{d}_{pool}_{s}" for s in LANGUAGES]
                    for lang in LANGUAGES:
                        tgid = f"g_qwen_{cond}_d{d}_{pool}_{lang}"
                        bid = f"blk_{tgid}"
                        blocks[bid] = {
                            "target_group": tgid,
                            "split": "SCREEN",
                            "owner": "qwen",
                            "routes": [{"source_group": sg, "route_id": f"r_{sg}_to_{tgid}"} for sg in src_gids],
                        }
                        preds[bid] = {"path": f"predictions/stage4/qwen/{bid}/_COMPLETE.json", "sha256": "a" * 64}

        for stage in ("encoder", "projector"):
            gid = f"g_qwen_image_only_{stage}"
            groups[gid] = {
                "group_id": gid,
                "source": "vlm",
                "model": "qwen",
                "condition": "image_only",
                "site": f"{stage}:None:visual_mean",
                "language": None,
            }
            bid = f"blk_{gid}"
            blocks[bid] = {
                "target_group": gid,
                "split": "SCREEN",
                "owner": "qwen",
                "routes": [{"source_group": gid, "route_id": f"r_{gid}"}],
            }
            preds[bid] = {"path": f"predictions/stage4/qwen/{bid}/_COMPLETE.json", "sha256": "b" * 64}

        class _PlanStub(dict):
            pass

        stage4_plan = _PlanStub({"blocks": blocks})
        stage4_plan.fit = {"groups": groups}
        binding = SimpleNamespace(stage4_plan=stage4_plan, stage4_release={"predictions": preds})

        dec_g, img_g, gid_to_blk = _resolve_vlm_groups_and_blocks(binding, "qwen", depths)
        self.assertEqual(len(dec_g), 64)
        self.assertEqual(set(img_g.keys()), {"encoder", "projector"})
        self.assertEqual(len(gid_to_blk), 66)

        # Missing projector group must raise ContractError
        bad_groups = dict(groups)
        bad_groups.pop("g_qwen_image_only_projector")
        bad_s4 = _PlanStub({"blocks": blocks})
        bad_s4.fit = {"groups": bad_groups}
        with self.assertRaisesRegex(ContractError, "Missing Stage 2 VLM image_only group"):
            _resolve_vlm_groups_and_blocks(
                SimpleNamespace(stage4_plan=bad_s4, stage4_release={"predictions": preds}),
                "qwen",
                depths,
            )

        # Test resume_analysis_ref fast-path authentication & tamper rejection
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            init_phase3_dirs(root)
            dataset_fp = "d" * 64
            plan_id = "p" * 64
            producer = {
                "code_commit": "0" * 40,
                "storage_scope": "local-development",
                "software_versions": {"python": "3.12.0"},
            }
            plan_ref = {"path": "phase3/plans/fake_plan/_COMPLETE.json", "sha256": "1" * 64}
            rel_ref = publish_phase3(
                root,
                protocol,
                dataset_fp,
                "phase3/features/e39_e31/qwen",
                KIND_E39_E31_RELEASE,
                {
                    "schema": SCHEMA_E39_E31_FEATURES,
                    "experiment": "E39+E31",
                    "plan": plan_ref,
                    "plan_identity": plan_id,
                    "model": "qwen",
                },
                producer,
                run_id="rel_01",
            )
            boot_meta = {"shape": [2000, 360], "sha256": "2" * 64}
            ana_ref = publish_phase3(
                root,
                protocol,
                dataset_fp,
                "phase3/results/e39_e31/qwen",
                KIND_ANALYSIS,
                {
                    "schema": SCHEMA_ANALYSIS,
                    "experiment": "E39+E31",
                    "plan": plan_ref,
                    "plan_identity": plan_id,
                    "release": plain(rel_ref),
                    "model": "qwen",
                    "bootstrap": boot_meta,
                    "confirm_excluded": True,
                    "e35_excluded": True,
                },
                producer,
                run_id="ana_01",
            )
            fake_gpu_plan = _FakeGpuPlan(
                {
                    "dataset_fingerprint": dataset_fp,
                    "plan_identity": plan_id,
                    "bootstrap": boot_meta,
                    "confirm_excluded": True,
                    "e35_excluded": True,
                },
                plan_ref,
                root,
                protocol,
                binding=SimpleNamespace(),
            )
            with mock.patch("src.phase3.run_e39_e31.load_phase3_gpu_plan", return_value=fake_gpu_plan):
                r_ref, r_body = run_e39_e31(
                    root,
                    protocol,
                    "qwen",
                    producer,
                    plan_ref=plan_ref,
                    release_ref=rel_ref,
                    resume_analysis_ref=ana_ref,
                )
                self.assertEqual(plain(r_ref), plain(ana_ref))
                self.assertEqual(r_body["experiment"], "E39+E31")

                # Mismatched release_ref against resumed analysis must fail
                other_rel = publish_phase3(
                    root,
                    protocol,
                    dataset_fp,
                    "phase3/features/e39_e31/qwen",
                    KIND_E39_E31_RELEASE,
                    {
                        "schema": SCHEMA_E39_E31_FEATURES,
                        "experiment": "E39+E31",
                        "plan": plan_ref,
                        "plan_identity": plan_id,
                        "model": "qwen",
                    },
                    producer,
                    run_id="rel_02",
                )
                with self.assertRaisesRegex(ContractError, "release ref mismatch"):
                    run_e39_e31(
                        root,
                        protocol,
                        "qwen",
                        producer,
                        plan_ref=plan_ref,
                        release_ref=other_rel,
                        resume_analysis_ref=ana_ref,
                    )


@unittest.skipUnless(HAS_SCIENTIFIC, "numpy, scipy, and scikit-learn required for end-to-end E39+E31 fixture tests")
class RunE39E31EndToEndScientificTests(unittest.TestCase):
    """End-to-end synthetic fixture tests exercising reassembly, parity gates, E39, E31, and resume."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _build_end_to_end_fixture(self, *, perturb_feature_diff=0.0, perturb_logit_diff=0.0, bad_att_sum=False):
        import numpy as np
        from src.stage4.predictions import Prediction

        protocol = _load_protocol()
        proto = json.loads(json.dumps(protocol))
        n_screen = 10
        n_chunks = 2
        images_per_chunk = 5
        dim = 6
        depths = [9, 18, 27, 36]
        midpoint = 18
        proto["split_sizes"] = {"FIT": 10, "DEV-S": 5, "DEV-C": 5, "SCREEN": n_screen, "CONFIRM": 2}
        proto["minimum_positive_and_negative"]["SCREEN"] = 2
        categories = list(proto["category_order"])
        n_cats = len(categories)
        plant_idx = categories.index("plant.n.02")

        producer = {
            "code_commit": "5" * 40,
            "storage_scope": "local-development",
            "software_versions": {"python": "3.12.0"},
        }
        init_phase3_dirs(self.root)

        screen_ids = sorted([f"{1000 + i:016x}" for i in range(n_screen - 1)] + ["6a20677eee04fd08"])
        dataset_fp = fingerprint({"screen_ids": screen_ids})
        probe_ref = _publish_probe_without_subprocess(self.root, proto, dataset_fp, producer)

        # Targets with balanced binary classes across all 25 categories & all 5 CV folds (0,1,0,1...)
        targets = {}
        for idx, iid in enumerate(screen_ids):
            vals = [int((idx + c_idx) % 2) for c_idx in range(n_cats)]
            if iid == "6a20677eee04fd08":
                vals[plant_idx] = None
            targets[iid] = {"image_id": iid, "split": "SCREEN", "values": vals}

        rng = np.random.Generator(np.random.PCG64(20261120))
        draws = rng.integers(0, n_screen, size=(2000, n_screen), dtype=np.int64)
        boot_sha = hashlib.sha256(draws.tobytes()).hexdigest()

        # Synthetic features in screen_ids order for all 8 E39 sites and 3 E31 sites
        true_e39 = {
            cond: {
                d: {
                    site: {
                        lang: rng.normal(0.0, 1.0, size=(n_screen, dim)).astype("<f4")
                        for lang in LANGUAGES
                    }
                    for site in E39_SITES
                }
                for d in depths
            }
            for cond in CONDITIONS
        }
        true_e31_vis = {
            d: {lang: rng.normal(0.0, 1.0, size=(n_screen, dim)).astype("<f4") for lang in LANGUAGES}
            for d in depths
        }
        true_e31_att = {}
        for att_site in ("attention_from_last_caption", "attention_from_caption_mean"):
            true_e31_att[att_site] = {}
            for d in depths:
                true_e31_att[att_site][d] = {}
                for l_idx, lang in enumerate(LANGUAGES):
                    vis_w = 0.25 + 0.05 * l_idx + (0.08 if att_site == "attention_from_caption_mean" else 0.0)
                    cap_w = 0.50 - 0.03 * l_idx
                    sys_w = 1.0 - vis_w - cap_w
                    arr = np.tile(np.array([vis_w, cap_w, sys_w], dtype="<f4"), (n_screen, 1))
                    if bad_att_sum and d == depths[0] and lang == "en":
                        arr[0, 0] += 0.05
                    true_e31_att[att_site][d][lang] = arr

        # Stage 2 groups, Stage 3 frozen heads, Stage 4 prediction blocks
        groups = {}
        blocks = {}
        preds_meta = {}
        s2_store_data = {}
        s3_head_data = {}
        s4_block_data = {}

        for d in depths:
            for cond in CONDITIONS:
                for pool in POOLINGS:
                    for lang in LANGUAGES:
                        gid = f"g_qwen_{cond}_d{d}_{pool}_{lang}"
                        groups[gid] = {
                            "group_id": gid,
                            "source": "vlm",
                            "model": "qwen",
                            "condition": cond,
                            "site": f"decoder:{d}:{pool}",
                            "language": lang,
                            "dimension": dim,
                        }
                        rows_lang = tuple(
                            {"image_id": iid, "split": "SCREEN", "language": lang}
                            for iid in screen_ids
                        )
                        s2_mat = np.array(true_e39[cond][d][pool][lang], dtype="<f4", copy=True)
                        if perturb_feature_diff and d == depths[0] and cond == "image_caption" and pool == "last_caption" and lang == "en":
                            s2_mat[0, 0] += float(perturb_feature_diff)
                        s2_store_data[gid] = (rows_lang, s2_mat)

                        coef = rng.normal(0.0, 0.4, size=(n_cats, dim)).astype(np.float64)
                        intercept = rng.normal(0.0, 0.1, size=(n_cats,)).astype(np.float64)
                        valid = np.ones(n_cats, dtype=bool)
                        prior = np.full(n_cats, 0.5, dtype=np.float64)
                        stats = {
                            "mean": np.zeros(dim, dtype=np.float64),
                            "variance": np.ones(dim, dtype=np.float64),
                            "scale": np.ones(dim, dtype=np.float64),
                            "center": True,
                            "reference": {"path": f"cache/stats/{gid}/_COMPLETE.json", "sha256": "c" * 64},
                        }
                        s3_head_data[gid] = (coef, intercept, valid, prior, stats)

        for d in depths:
            for cond in CONDITIONS:
                for pool in POOLINGS:
                    for t_lang in LANGUAGES:
                        tgid = f"g_qwen_{cond}_d{d}_{pool}_{t_lang}"
                        bid = f"blk_{tgid}"
                        routes = []
                        block_preds = {}
                        t_rows, _ = s2_store_data[tgid]
                        t_mat = true_e39[cond][d][pool][t_lang].astype(np.float64)
                        for s_lang in LANGUAGES:
                            sgid = f"g_qwen_{cond}_d{d}_{pool}_{s_lang}"
                            rid = f"r_{sgid}_to_{tgid}"
                            routes.append({"source_group": sgid, "route_id": rid})
                            coef, intercept, valid, prior, _ = s3_head_data[sgid]
                            logits = (t_mat @ coef.T) + intercept
                            if perturb_logit_diff and d == depths[0] and cond == "image_caption" and pool == "last_caption" and s_lang == "en" and t_lang == "en":
                                logits = logits.copy()
                                logits[0, 0] += float(perturb_logit_diff)
                            ref = {"path": f"predictions/stage4/qwen/{bid}/_COMPLETE.json", "sha256": "a" * 64}
                            block_preds[rid] = Prediction(
                                reference=ref,
                                route={"route_id": rid, "source_group": sgid},
                                split="SCREEN",
                                rows=t_rows,
                                logits=logits,
                                estimable=valid,
                                prevalence=prior,
                            )
                        blocks[bid] = {
                            "target_group": tgid,
                            "split": "SCREEN",
                            "owner": "qwen",
                            "routes": routes,
                        }
                        preds_meta[bid] = {"path": f"predictions/stage4/qwen/{bid}/_COMPLETE.json", "sha256": "a" * 64}
                        s4_block_data[bid] = block_preds

        for stage in ("encoder", "projector"):
            gid = f"g_qwen_image_only_{stage}"
            groups[gid] = {
                "group_id": gid,
                "source": "vlm",
                "model": "qwen",
                "condition": "image_only",
                "site": f"{stage}:None:visual_mean",
                "language": None,
                "dimension": dim,
            }
            rows_img = tuple({"image_id": iid, "split": "SCREEN", "language": None} for iid in screen_ids)
            mat_img = rng.normal(0.0, 1.0, size=(n_screen, dim)).astype("<f4")
            s2_store_data[gid] = (rows_img, mat_img)
            bid = f"blk_{gid}"
            rid = f"r_{gid}"
            blocks[bid] = {
                "target_group": gid,
                "split": "SCREEN",
                "owner": "qwen",
                "routes": [{"source_group": gid, "route_id": rid}],
            }
            ref = {"path": f"predictions/stage4/qwen/{bid}/_COMPLETE.json", "sha256": "b" * 64}
            preds_meta[bid] = ref
            s4_block_data[bid] = {
                rid: Prediction(
                    reference=ref,
                    route={"route_id": rid, "source_group": gid},
                    split="SCREEN",
                    rows=rows_img,
                    logits=rng.normal(0.0, 1.0, size=(n_screen, n_cats)).astype(np.float64),
                    estimable=np.ones(n_cats, dtype=bool),
                    prevalence=np.full(n_cats, 0.5, dtype=np.float64),
                )
            }

        # Publish phase3_gpu_plan
        plan_chunks = []
        for c_idx in range(n_chunks):
            batch_ids = screen_ids[c_idx * images_per_chunk : (c_idx + 1) * images_per_chunk]
            c_rows = []
            for iid in batch_ids:
                for lang in LANGUAGES:
                    txt = f"एक कुत्ता {iid} ।" if lang == "hi" else f"A dog {iid} ."
                    c_rows.append({
                        "image_id": iid,
                        "image_path": f"images/{iid}.jpg",
                        "image_sha256": "e" * 64,
                        "language": lang,
                        "caption_id": f"{iid}_{lang}_0",
                        "caption_draw_key": 0,
                        "caption_text": txt,
                        "caption_text_sha256": hashlib.sha256(txt.encode("utf-8")).hexdigest(),
                    })
            plan_chunks.append({
                "chunk_id": f"SCREEN_{c_idx:04d}",
                "split": "SCREEN",
                "image_ids": batch_ids,
                "n_images": len(batch_ids),
                "n_views_per_condition": len(c_rows),
                "rows": c_rows,
            })

        e39_e31_plan = {
            "schema": "phase3.e39_e31_extraction_plan.v1",
            "split": "SCREEN",
            "confirm_rows": 0,
            "conditions": list(CONDITIONS),
            "e39_sites": list(E39_SITES),
            "e31_sites": list(E31_SITES),
            "chunk_size_images": 5,
            "total_chunks": n_chunks,
            "total_images": n_screen,
            "views_per_condition": n_screen * len(LANGUAGES),
            "chunks": plan_chunks,
        }
        gpu_plan_core = {
            "schema": SCHEMA_GPU_PLAN,
            "phase3_gpu_version": PHASE3_GPU_VERSION,
            "lineage_identity": "1" * 64,
            "dataset_fingerprint": dataset_fp,
            "protocol_fingerprint": fingerprint(proto),
            "phase1_stage4_release": {"path": "results/stage4/rel/_COMPLETE.json", "sha256": "9" * 64},
            "phase3_cpu_plan": None,
            "bootstrap": {"shape": [2000, n_screen], "sha256": boot_sha},
            "storage_probe": plain(probe_ref),
            "initialized_dirs": list(PHASE3_PARENT_DIRS),
            "models": {
                "qwen": {
                    "depths": depths,
                    "midpoint_depth": midpoint,
                    "model_plan_ref": {"path": "shared/mp_qwen/_COMPLETE.json", "sha256": "8" * 64},
                    "midpoint_groups": {},
                    "source_systems": {},
                }
            },
            "e39_e31_plan": e39_e31_plan,
            "e23_plan": {"schema": "phase3.e23_pair_plan.v1", "chunks": [], "authentic_parity_sample": []},
            "decision_ranges": plain(PHASE3_GPU_DECISION_RANGES),
            "e35_excluded": True,
            "confirm_excluded": True,
        }
        plan_id = fingerprint(gpu_plan_core)
        gpu_plan_body = dict(gpu_plan_core, plan_identity=plan_id)
        gpu_plan_ref = publish_phase3(
            self.root, proto, dataset_fp, "phase3/plans", KIND_GPU_PLAN, gpu_plan_body, producer
        )

        # Publish Stage B2 E39+E31 chunks & release
        specs = expected_array_specs(depths, dim)
        release_chunks = []
        for c_idx, p_chunk in enumerate(plan_chunks):
            batch_ids = p_chunk["image_ids"]
            pos_list = [screen_ids.index(iid) for iid in batch_ids]
            chunk_rows = []
            for row_idx, r in enumerate(p_chunk["rows"]):
                toks = ["एक", "कुत्ता", r["image_id"], "।"] if r["language"] == "hi" else ["A", "dog", r["image_id"], "."]
                sel = select_e39_token_indices(toks)
                chunk_rows.append({
                    "row_index": row_idx,
                    "image_id": r["image_id"],
                    "image_path": r["image_path"],
                    "image_sha256": r["image_sha256"],
                    "language": r["language"],
                    "split": "SCREEN",
                    "caption_id": r["caption_id"],
                    "caption_draw_key": 0,
                    "caption_text": r["caption_text"],
                    "caption_text_sha256": r["caption_text_sha256"],
                    "e39_token_selection": {"image_caption": sel, "caption_only": sel},
                })
            rows_bytes = b"".join(canonical_bytes(cr) + b"\n" for cr in chunk_rows)
            rows_sha = hashlib.sha256(rows_bytes).hexdigest()

            arrays = {}
            for cond in CONDITIONS:
                for d in depths:
                    for site in E39_SITES:
                        rows_arr = []
                        for pos in pos_list:
                            for lang in LANGUAGES:
                                rows_arr.append(true_e39[cond][d][site][lang][pos])
                        arrays[f"{cond}__d{d}__{site}"] = np.ascontiguousarray(np.stack(rows_arr, axis=0), dtype="<f4")
            for d in depths:
                vis_rows = [true_e31_vis[d][lang][pos] for pos in pos_list for lang in LANGUAGES]
                arrays[f"image_caption__d{d}__decoder_visual_mean"] = np.ascontiguousarray(
                    np.stack(vis_rows, axis=0), dtype="<f4"
                )
                for att_site in ("attention_from_last_caption", "attention_from_caption_mean"):
                    att_rows = [true_e31_att[att_site][d][lang][pos] for pos in pos_list for lang in LANGUAGES]
                    arrays[f"image_caption__d{d}__{att_site}"] = np.ascontiguousarray(
                        np.stack(att_rows, axis=0), dtype="<f4"
                    )

            repeat_check = {
                "first_index": 0,
                "last_index": len(chunk_rows) - 1,
                "max_abs_diff_hidden": 0.0,
                "max_abs_diff_attention": 0.0,
                "passed": True,
            }
            c_body = {
                "schema": SCHEMA_E39_E31_FEATURES,
                "experiment": "E39+E31",
                "plan": plain(gpu_plan_ref),
                "plan_identity": plan_id,
                "model": "qwen",
                "depths": depths,
                "midpoint_depth": midpoint,
                "conditions": list(CONDITIONS),
                "e39_sites": list(E39_SITES),
                "e31_sites": list(E31_SITES),
                "split": "SCREEN",
                "chunk_index": c_idx,
                "chunk_id": p_chunk["chunk_id"],
                "total_chunks": n_chunks,
                "n_images": len(batch_ids),
                "n_views_per_condition": len(chunk_rows),
                "dimension": dim,
                "array_count": len(specs),
                "model_fingerprint": "f" * 64,
                "execution_fingerprint": "e" * 64,
                "site_specs": {},
                "evidence_bridge_sha256": "b" * 64,
                "repeat_check": repeat_check,
                "rows_sha256": rows_sha,
                "e35_excluded": True,
                "confirm_excluded": True,
            }
            c_ref = publish_phase3(
                self.root,
                proto,
                dataset_fp,
                "phase3/features/e39_e31/qwen/batch_test",
                KIND_E39_E31_CHUNK,
                c_body,
                producer,
                arrays=arrays,
                files={
                    "rows.jsonl": rows_bytes,
                    "token_evidence_image_caption.jsonl": b"{}\n",
                    "token_evidence_caption_only.jsonl": b"{}\n",
                    "audits.json": canonical_bytes({"repeat_check": repeat_check}) + b"\n",
                },
                upstream=[plain(gpu_plan_ref)],
                model_fingerprint="f" * 64,
                run_id=f"chunk_{c_idx:04d}",
            )
            release_chunks.append({
                "chunk_index": c_idx,
                "chunk_id": p_chunk["chunk_id"],
                "n_images": len(batch_ids),
                "n_views_per_condition": len(chunk_rows),
                "ref": plain(c_ref),
                "rows_sha256": rows_sha,
            })

        rel_body = {
            "schema": SCHEMA_E39_E31_FEATURES,
            "experiment": "E39+E31",
            "plan": plain(gpu_plan_ref),
            "plan_identity": plan_id,
            "batch_run_id": "batch_test",
            "model": "qwen",
            "depths": depths,
            "midpoint_depth": midpoint,
            "conditions": list(CONDITIONS),
            "e39_sites": list(E39_SITES),
            "e31_sites": list(E31_SITES),
            "split": "SCREEN",
            "confirm_rows": 0,
            "total_images": n_screen,
            "views_per_condition": n_screen * len(LANGUAGES),
            "total_chunks": n_chunks,
            "dimension": dim,
            "array_count_per_chunk": len(specs),
            "model_fingerprint": "f" * 64,
            "execution_fingerprint": "e" * 64,
            "site_specs": {},
            "evidence_bridge_sha256": "b" * 64,
            "chunks": release_chunks,
            "repeat_check_summary": {
                "chunks_checked": n_chunks,
                "all_passed": True,
                "max_abs_diff_hidden": 0.0,
                "max_abs_diff_attention": 0.0,
            },
            "e35_excluded": True,
            "confirm_excluded": True,
        }
        rel_ref = publish_phase3(
            self.root,
            proto,
            dataset_fp,
            "phase3/features/e39_e31/qwen",
            KIND_E39_E31_RELEASE,
            rel_body,
            producer,
            upstream=[plain(gpu_plan_ref)] + [c["ref"] for c in release_chunks],
            model_fingerprint="f" * 64,
            run_id="batch_test_release",
        )

        class _FakeStage4Plan(dict):
            pass

        s4_plan = _FakeStage4Plan({"screen_ids": screen_ids, "blocks": blocks, "categories": categories})
        s4_plan.fit = {"groups": groups, "targets": targets}
        s4_plan.store = SimpleNamespace()
        binding = SimpleNamespace(
            stage4_plan=s4_plan,
            stage4_release={"predictions": preds_meta},
        )
        bound_gpu_plan = _FakeGpuPlan(gpu_plan_body, gpu_plan_ref, self.root, proto, binding=binding)
        return (
            proto,
            producer,
            gpu_plan_ref,
            rel_ref,
            bound_gpu_plan,
            draws,
            s2_store_data,
            s3_head_data,
            s4_block_data,
        )

    def test_end_to_end_run_e39_e31_parity_5fold_cv_and_resume(self):
        (
            proto,
            producer,
            gpu_plan_ref,
            rel_ref,
            bound_gpu_plan,
            draws,
            s2_store_data,
            s3_head_data,
            s4_block_data,
        ) = self._build_end_to_end_fixture()

        events = []
        with mock.patch("src.phase3.run_e39_e31.load_phase3_gpu_plan", return_value=bound_gpu_plan), \
             mock.patch("src.stage3.data._read_verified_group", side_effect=lambda fit, store, gid, sp: s2_store_data[gid]), \
             mock.patch("src.stage4.predictions._selected_parameters", side_effect=lambda plan, gid: s3_head_data[gid]), \
             mock.patch("src.stage4.predictions.load_prediction_block", side_effect=lambda plan, ref, expected_block=None: s4_block_data[expected_block]), \
             mock.patch("src.stage4.plan.load_bootstrap", return_value=draws):
            ana_ref, ana_body = run_e39_e31(
                self.root,
                proto,
                "qwen",
                producer,
                plan_ref=gpu_plan_ref,
                release_ref=rel_ref,
                run_id="e39_e31_test_run",
                progress=events.append,
            )

            self.assertEqual(ana_body["schema"], SCHEMA_ANALYSIS)
            self.assertEqual(ana_body["experiment"], "E39+E31")
            self.assertEqual(ana_body["model"], "qwen")
            self.assertTrue(ana_body["confirm_excluded"])
            self.assertTrue(ana_body["e35_excluded"])
            self.assertEqual(ana_body["stage2_parity_gate"]["feature_parity_cells_checked"], 64)
            self.assertTrue(ana_body["stage2_parity_gate"]["feature_parity_all_passed"])
            self.assertEqual(ana_body["stage2_parity_gate"]["max_feature_abs_diff"], 0.0)
            self.assertEqual(ana_body["stage2_parity_gate"]["logit_self_checks_checked"], 16)
            self.assertTrue(ana_body["stage2_parity_gate"]["logit_self_checks_all_passed"])
            self.assertEqual(ana_body["stage2_parity_gate"]["max_logit_abs_diff"], 0.0)

            # E39 checks: 8 sites, 3 head modes, recovery fraction of caption_mean == 1.0 and last_caption == 0.0
            mid_ic = ana_body["e39"]["cells"]["18"]["image_caption"]["head_modes"]["fixed_head_last"]["sites"]
            self.assertEqual(set(mid_ic.keys()), set(E39_SITES))
            self.assertAlmostEqual(
                mid_ic["last_caption"]["gain_vs_last_caption"]["twelve_route_mean_transfer"]["auroc"]["primary_macro"]["value"],
                0.0,
                places=12,
            )
            self.assertAlmostEqual(
                mid_ic["caption_mean"]["recovery_fraction_of_caption_mean"]["twelve_route_mean_transfer"]["auroc"]["primary_macro"]["value"],
                1.0,
                places=12,
            )

            # E31 checks: 5-fold CV trajectory encoder -> projector -> decoder:d1..d4 & attention correlations
            e31 = ana_body["e31"]
            self.assertTrue(e31["attention_mass_check"]["passed"])
            self.assertIn("encoder", e31["visual_token_probe"]["stage2_5fold_cv"])
            self.assertIn("projector", e31["visual_token_probe"]["stage2_5fold_cv"])
            self.assertEqual(set(e31["visual_token_probe"]["decoder_visual_mean"].keys()), {"9", "18", "27", "36"})
            self.assertIn("by_depth_and_route", e31["attention_gap_correlations"]["auroc"])
            self.assertEqual(e31["attention_gap_correlations"]["auroc"]["by_depth_and_route"]["n_points"], 48)

            # Resume fast-path roundtrip
            res_events = []
            res_ref, res_body = run_e39_e31(
                self.root,
                proto,
                "qwen",
                producer,
                plan_ref=gpu_plan_ref,
                release_ref=rel_ref,
                resume_analysis_ref=ana_ref,
                progress=res_events.append,
            )
            self.assertEqual(plain(res_ref), plain(ana_ref))
            self.assertEqual(res_body["plan_identity"], ana_body["plan_identity"])
            self.assertTrue(any(e["event"] == "e39_e31_resumed" for e in res_events))

    def test_stage2_feature_and_logit_parity_gate_rejections(self):
        # 1. Perturb Stage 2 feature by 5e-4 > 1e-4 -> must fail parity gate
        (
            proto,
            producer,
            gpu_plan_ref,
            rel_ref,
            bound_gpu_plan,
            draws,
            s2_store_data,
            s3_head_data,
            s4_block_data,
        ) = self._build_end_to_end_fixture(perturb_feature_diff=5e-4)

        with mock.patch("src.phase3.run_e39_e31.load_phase3_gpu_plan", return_value=bound_gpu_plan), \
             mock.patch("src.stage3.data._read_verified_group", side_effect=lambda fit, store, gid, sp: s2_store_data[gid]), \
             mock.patch("src.stage4.predictions._selected_parameters", side_effect=lambda plan, gid: s3_head_data[gid]), \
             mock.patch("src.stage4.predictions.load_prediction_block", side_effect=lambda plan, ref, expected_block=None: s4_block_data[expected_block]), \
             mock.patch("src.stage4.plan.load_bootstrap", return_value=draws):
            with self.assertRaisesRegex(ContractError, "Stage 2 draw-0 feature parity failed"):
                run_e39_e31(
                    self.root,
                    proto,
                    "qwen",
                    producer,
                    plan_ref=gpu_plan_ref,
                    release_ref=rel_ref,
                )

        # 2. Perturb Stage 4 en->en logit by 5e-4 > 1e-4 -> must fail logit parity gate
        self.setUp()
        (
            proto2,
            producer2,
            gpu_plan_ref2,
            rel_ref2,
            bound_gpu_plan2,
            draws2,
            s2_store_data2,
            s3_head_data2,
            s4_block_data2,
        ) = self._build_end_to_end_fixture(perturb_logit_diff=5e-4)

        with mock.patch("src.phase3.run_e39_e31.load_phase3_gpu_plan", return_value=bound_gpu_plan2), \
             mock.patch("src.stage3.data._read_verified_group", side_effect=lambda fit, store, gid, sp: s2_store_data2[gid]), \
             mock.patch("src.stage4.predictions._selected_parameters", side_effect=lambda plan, gid: s3_head_data2[gid]), \
             mock.patch("src.stage4.predictions.load_prediction_block", side_effect=lambda plan, ref, expected_block=None: s4_block_data2[expected_block]), \
             mock.patch("src.stage4.plan.load_bootstrap", return_value=draws2):
            with self.assertRaisesRegex(ContractError, "en->en matched logit self-check failed"):
                run_e39_e31(
                    self.root,
                    proto2,
                    "qwen",
                    producer2,
                    plan_ref=gpu_plan_ref2,
                    release_ref=rel_ref2,
                )


if __name__ == "__main__":
    unittest.main()
