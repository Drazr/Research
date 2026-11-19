"""Unit and fixture tests for Phase 2 Stage 5 synthesis, decision register, and independent readback."""

import json
from pathlib import Path
import tempfile
import unittest

from src.common.contracts import ContractError, LANGUAGES, fingerprint
from src.phase2.e17_auth import E17_DEV_C_PERMISSION
from src.phase2.gates import E29_GATE_THRESHOLDS, STAGE5_DECISION_RANGES, evaluate_e29_gates, publish_e29_gate
from src.phase2.interfaces import (
    KIND_ANALYSIS,
    KIND_E07_AUDIT,
    KIND_E07_RELEASE,
    KIND_E17_AUTH,
    KIND_PLAN,
    MODELS,
    PHASE2_PARENT_DIRS,
    PHASE2_VERSION,
    REPO,
    SCHEMA_ANALYSIS,
    SCHEMA_E07_AUDIT,
    SCHEMA_E07_FEATURES,
    SCHEMA_E17_AUTH,
    SCHEMA_PLAN,
    init_phase2_dirs,
    plain,
    publish_phase2,
    storage_probe,
)
from src.phase2.plan import load_phase2_plan
from src.phase2.synthesize import (
    CERTIFIED_PHASE2_REFS,
    DECISION_OUTCOMES,
    evaluate_model_decisions,
    synthesize_phase2,
    verify_phase2_release,
)


def _load_protocol():
    return json.loads((REPO / "configs/protocol.json").read_text(encoding="utf-8"))


def _fake_producer():
    return {
        "code_commit": "0" * 40,
        "storage_scope": "local-development",
        "software_versions": {"python": "3.12.14", "numpy": "2.1.3"},
    }


def _stat(val, lo, hi, vd=2000):
    return {"value": val, "interval": {"lower": lo, "upper": hi, "valid_draws": vd}}


def _stat_e17(val, lo, hi, vd=2000):
    return {"point": val, "lower": lo, "upper": hi, "valid_draws": vd}


def _build_synthetic_stage1_to_4_tree(root: Path, protocol: dict):
    """Publish a complete synthetic 15-artifact Phase 2 Stage 1-4 tree on disk."""
    init_phase2_dirs(root)
    prod = _fake_producer()
    ds_fp = fingerprint({"synthetic": "phase2_stage5"})
    proto_fp = fingerprint(protocol)
    stage4_ref = {"path": "results/stage4/releases/fake/_COMPLETE.json", "sha256": "a" * 64}
    lineage_id = fingerprint({"lineage": "stage5_test"})

    # Stage 1 e07_audit
    audit_body = {
        "schema": SCHEMA_E07_AUDIT,
        "stage4_release": stage4_ref,
        "lineage_identity": lineage_id,
        "dataset_ref": {"path": "shared/ds/_COMPLETE.json", "sha256": "b" * 64},
        "dataset_fingerprint": ds_fp,
        "draw0_verified": True,
        "confirm_excluded": True,
        "verdict": "passed_restricted",
        "verdict_reasons": ["bn unavailable"],
        "screen_multiplicity": {},
        "fit_dev_s_multiplicity": {},
        "draw_buckets": {},
        "support_tables": {},
        "unique_changed_views": {"en": 267, "hi": 327, "bn": 0, "te": 264, "total": 858},
        "extraction_plan": {"split": "SCREEN", "confirm_rows": 0, "total_unique_views": 858, "total_chunks": 29, "chunks": []},
    }
    audit_ref = publish_phase2(root, protocol, ds_fp, "phase2/audits/e07", KIND_E07_AUDIT, audit_body, prod)

    # Stage 1 e17_auth
    auth_models = {
        m: {
            "midpoint_depth": 18 if m == "qwen" else 17,
            "screen_blocks": {},
            "dev_c_diagonals": {},
            "off_diagonal_blocks": {},
            "source_systems": {},
            "diagonal_reproduction_check": {"passed": True, "max_abs_diff": 0.0, "atol": 0.0, "rtol": 0.0},
        }
        for m in MODELS
    }
    auth_body = {
        "schema": SCHEMA_E17_AUTH,
        "stage4_release": stage4_ref,
        "lineage_identity": lineage_id,
        "dataset_fingerprint": ds_fp,
        "permission": E17_DEV_C_PERMISSION,
        "confirm_excluded": True,
        "models": auth_models,
        "calibrator_recipe": {},
    }
    auth_ref = publish_phase2(root, protocol, ds_fp, "phase2/plans", KIND_E17_AUTH, auth_body, prod)

    # Stage 1 storage probe + phase2_plan
    probe_ref = storage_probe(root, protocol, ds_fp, prod)
    core_plan = {
        "schema": SCHEMA_PLAN,
        "phase2_version": PHASE2_VERSION,
        "lineage_identity": lineage_id,
        "dataset_fingerprint": ds_fp,
        "protocol_fingerprint": proto_fp,
        "stage4_release": stage4_ref,
        "stage4_plan": {"path": "results/stage4/plan/_COMPLETE.json", "sha256": "c" * 64},
        "stage3_registry": {"path": "results/stage3/reg/_COMPLETE.json", "sha256": "d" * 64},
        "g0_gate": {"path": "results/stage3/g0/_COMPLETE.json", "sha256": "e" * 64},
        "g0_p_value": 0.362,
        "stage2_suite": {"path": "results/stage2/suite/_COMPLETE.json", "sha256": "f" * 64},
        "stage1_dataset": {"path": "shared/ds/_COMPLETE.json", "sha256": "b" * 64},
        "bootstrap": {"shape": [2000, 360], "sha256": "1" * 64},
        "midpoints": {"qwen": 18, "gemma": 17},
        "storage_probe": plain(probe_ref),
        "initialized_dirs": list(PHASE2_PARENT_DIRS),
        "e29_blocks": {
            m: {
                c: {
                    p: {
                        l: {"block_id": f"{m}_{c}_{p}_{l}", "prediction_ref": {"path": f"p/{m}/{c}/{p}/{l}/_COMPLETE.json", "sha256": "4" * 64}, "routes": {s: f"{s}->{l}" for s in LANGUAGES}}
                        for l in LANGUAGES
                    }
                    for p in ("last_caption", "caption_mean")
                }
                for c in ("image_caption", "caption_only")
            }
            for m in MODELS
        },
        "e11_fit_statistics": {m: {l: {"statistics_ref": {"path": f"s/{m}/{l}/_COMPLETE.json", "sha256": "5" * 64}} for l in LANGUAGES} for m in MODELS},
        "gate_thresholds": plain(E29_GATE_THRESHOLDS),
        "stage5_decision_ranges": plain(STAGE5_DECISION_RANGES),
        "e07_audit": plain(audit_ref),
        "e17_auth": plain(auth_ref),
        "confirm_excluded": True,
    }
    plan_id = fingerprint(core_plan)
    plan_body = dict(core_plan, plan_identity=plan_id)
    plan_ref = publish_phase2(root, protocol, ds_fp, "phase2/plans", KIND_PLAN, plan_body, prod)
    plan = load_phase2_plan(root, protocol, plan_ref)

    model_refs = {}
    for m in MODELS:
        # E29 gate + analysis
        route_res = {f"{s}->{t}": {"point": 0.030, "ci": [0.015, 0.045]} for s in LANGUAGES for t in LANGUAGES if s != t}
        ev = evaluate_e29_gates(
            m,
            diagonals={l: 0.90 for l in LANGUAGES},
            valid_draws=2000,
            delta_g_mean=0.070,
            delta_g_ci=[0.055, 0.085],
            g_res_mean=0.030,
            g_res_ci=[0.015, 0.045],
            route_residuals=route_res,
        )
        gate_ref = publish_e29_gate(
            root, protocol, plan, m, {"path": f"phase2/results/e29/{m}/x/_COMPLETE.json", "sha256": "3" * 64}, ev, prod
        )
        e29_body = {
            "schema": SCHEMA_ANALYSIS,
            "experiment": "E29",
            "plan": plain(plan_ref),
            "plan_identity": plan_id,
            "gate_ref": plain(gate_ref),
            "model": m,
            "confirm_excluded": True,
            "image_caption": {
                "twelve_route_means": {
                    "auroc": {
                        "g_last_mean": {"primary_macro": _stat(0.10, 0.08, 0.12)},
                        "g_res_mean": {"primary_macro": _stat(0.03, 0.015, 0.045)},
                        "delta_g_mean": {"primary_macro": _stat(0.07, 0.055, 0.085)},
                    },
                    "nll": {"delta_g_mean": {"primary_macro": _stat(0.05, 0.03, 0.07)}},
                    "brier": {"delta_g_mean": {"primary_macro": _stat(0.02, 0.01, 0.03)}},
                }
            },
            "caption_only": {
                "twelve_route_means": {
                    "auroc": {
                        "g_last_mean": {"primary_macro": _stat(0.06, 0.04, 0.08)},
                        "g_res_mean": {"primary_macro": _stat(0.04, 0.02, 0.06)},
                        "delta_g_mean": {"primary_macro": _stat(0.02, 0.01, 0.03)},
                    }
                }
            },
        }
        e29_ref = publish_phase2(root, protocol, ds_fp, f"phase2/results/e29/{m}", KIND_ANALYSIS, e29_body, prod)

        # E17 analysis
        arm_src = {
            "off_diagonal_12_route_mean": {
                "primary_macro": {"nll": _stat_e17(0.30, 0.28, 0.32), "brier": _stat_e17(0.09, 0.08, 0.10)},
                "gain_vs_raw": {"nll": _stat_e17(0.04, 0.02, 0.06), "brier": _stat_e17(0.01, 0.006, 0.014)},
                "gain_vs_fit_prior": {"nll": _stat_e17(0.08, 0.06, 0.10), "brier": _stat_e17(0.03, 0.02, 0.04)},
            }
        }
        arm_raw = {"off_diagonal_12_route_mean": {"primary_macro": {"nll": _stat_e17(0.34, 0.32, 0.36), "brier": _stat_e17(0.10, 0.09, 0.11)}}}
        e17_body = {
            "schema": SCHEMA_ANALYSIS,
            "experiment": "E17",
            "plan": plain(plan_ref),
            "plan_identity": plan_id,
            "e17_auth_ref": plain(auth_ref),
            "model": m,
            "confirm_excluded": True,
            "e19_cutoffs_untouched": True,
            "fit_prior_reference": {"off_diagonal_12_route_mean": {"nll": _stat_e17(0.38, 0.36, 0.40), "brier": _stat_e17(0.12, 0.11, 0.13)}},
            "summaries": {
                "raw": arm_raw,
                "source_dev_c:temp": arm_src,
                "source_dev_c:temp_intercept": arm_src,
                "target_dev_c:temp": arm_src,
                "target_dev_c:temp_intercept": arm_src,
            },
            "decision_0p5": {
                arm: {"off_diagonal_12_route_mean": {"flip_rate": 0.0 if arm.endswith(":temp") else 0.01}}
                for arm in ("raw", "source_dev_c:temp", "source_dev_c:temp_intercept", "target_dev_c:temp", "target_dev_c:temp_intercept")
            },
            "invariance_checks": {"max_abs_auroc_diff": 0.0, "max_abs_ap_diff": 0.0, "all_T_positive": True, "temp_only_0p5_flips": 0},
        }
        e17_ref = publish_phase2(root, protocol, ds_fp, f"phase2/results/e17/{m}", KIND_ANALYSIS, e17_body, prod)

        # E07 release
        e07_rel_body = {
            "schema": SCHEMA_E07_FEATURES,
            "experiment": "E07",
            "plan": plain(plan_ref),
            "plan_identity": plan_id,
            "e07_audit_ref": plain(audit_ref),
            "model": m,
            "confirm_excluded": True,
            "repeat_check_summary": {"all_passed": True, "chunks_checked": 29},
        }
        e07_rel_ref = publish_phase2(root, protocol, ds_fp, f"phase2/features/e07/{m}/releases", KIND_E07_RELEASE, e07_rel_body, prod)

        # E30 analysis
        e30_body = {
            "schema": SCHEMA_ANALYSIS,
            "experiment": "E30",
            "plan": plain(plan_ref),
            "plan_identity": plan_id,
            "gate_ref": plain(gate_ref),
            "model": m,
            "confirm_excluded": True,
            "contrasts": {
                "feature_gain_fixed_head_last": {"metrics": {"auroc": {"gap_reduction": {"primary_macro": _stat(0.06, 0.05, 0.07)}}}},
                "feature_gain_fixed_head_mean": {"metrics": {"auroc": {"gap_reduction": {"primary_macro": _stat(0.005, -0.002, 0.012)}}}},
                "head_gain_fixed_feature_last": {"metrics": {"auroc": {"gap_reduction": {"primary_macro": _stat(0.01, 0.002, 0.018)}}}},
                "head_gain_fixed_feature_mean": {"metrics": {"auroc": {"gap_reduction": {"primary_macro": _stat(-0.02, -0.03, -0.01)}}}},
                "interaction": {"metrics": {"auroc": {"gap_reduction": {"primary_macro": _stat(0.055, 0.045, 0.065)}}}},
                "total_matched_reduction": {"metrics": {"auroc": {"gap_reduction": {"primary_macro": _stat(0.07, 0.055, 0.085)}}}},
            },
            "off_distribution_collapse": {"flagged": False},
        }
        e30_ref = publish_phase2(root, protocol, ds_fp, f"phase2/results/e30/{m}", KIND_ANALYSIS, e30_body, prod)

        # E11 analysis
        e11_body = {
            "schema": SCHEMA_ANALYSIS,
            "experiment": "E11",
            "plan": plain(plan_ref),
            "plan_identity": plan_id,
            "gate_ref": plain(gate_ref),
            "model": m,
            "confirm_excluded": True,
            "fit_pairing_check": {"paired_fit_images": 2160, "fit_rows_match_across_languages": True, "fit_matrices_reloaded": 0},
            "identity_self_check": {"passed": True, "max_abs_diff": 0.0},
            "invariance_checks": {"passed": True, "mean_shift_max_abs_auroc_diff": 0.0, "mean_shift_max_abs_ap_diff": 0.0},
            "summaries": {
                "paired_gains_vs_identity": {
                    "mean_shift": {
                        "auroc": {"twelve_route_mean_gap_reduction": _stat(0.0, 0.0, 0.0)},
                        "nll": {"twelve_route_mean_absolute_improvement": _stat(0.002, -0.001, 0.005)},
                        "brier": {"twelve_route_mean_absolute_improvement": _stat(0.001, -0.001, 0.003)},
                    },
                    "mean_scale": {
                        "auroc": {"twelve_route_mean_gap_reduction": _stat(0.001, -0.002, 0.003)},
                        "nll": {"twelve_route_mean_absolute_improvement": _stat(0.003, -0.001, 0.007)},
                        "brier": {"twelve_route_mean_absolute_improvement": _stat(0.001, -0.001, 0.003)},
                    },
                },
                "recovered_fraction_of_g_res_mean": {
                    "mean_scale": {"auroc": _stat(0.03, -0.05, 0.10)},
                },
            },
        }
        e11_ref = publish_phase2(root, protocol, ds_fp, f"phase2/results/e11/{m}", KIND_ANALYSIS, e11_body, prod)

        # E07 analysis
        shift_block = {
            "twelve_route_mean": {
                "auroc": {
                    "g_last_shift": {"primary_macro": _stat(0.003, -0.005, 0.011)},
                    "g_res_shift": {"primary_macro": _stat(0.001, -0.004, 0.006)},
                    "delta_g_shift": {"primary_macro": _stat(0.002, -0.005, 0.009)},
                }
            },
            "changed_targets_only": {
                "means": {
                    "auroc": {
                        "g_last_shift": {"primary_macro": _stat(0.004, -0.006, 0.014)},
                        "g_res_shift": {"primary_macro": _stat(0.001, -0.005, 0.007)},
                        "delta_g_shift": {"primary_macro": _stat(0.003, -0.006, 0.012)},
                    }
                }
            },
            "pooling_reduction_sign_preserved": {
                "twelve_route_mean": {"sign_preserved": True},
                "changed_targets_only_mean": {"sign_preserved": True},
            },
        }
        rc_draw = {
            "eligible_targets_aggregate": {
                "n_routes": 9,
                "macros": {
                    "primary_14": {
                        "metrics": {
                            "auroc": {
                                "g_last_shift": _stat(0.005, -0.008, 0.018),
                                "g_res_shift": _stat(0.002, -0.007, 0.011),
                                "delta_g_shift": _stat(0.003, -0.009, 0.015),
                            }
                        }
                    }
                },
            }
        }
        e07_body = {
            "schema": SCHEMA_ANALYSIS,
            "experiment": "E07",
            "plan": plain(plan_ref),
            "plan_identity": plan_id,
            "e07_audit_ref": plain(audit_ref),
            "e07_release_ref": plain(e07_rel_ref),
            "e07_verdict": "passed_restricted",
            "e29_analysis_ref": plain(e29_ref),
            "model": m,
            "confirm_excluded": True,
            "draw0_parity_check": {"passed": True, "e29_draw0_max_abs_diff": 0.0},
            "coverage_summary": {"eligible_changed_languages": ["en", "hi", "te"], "unavailable_languages": ["bn"]},
            "full_screen_360": {"paired_shifts": {"draw_1_vs_draw_0": shift_block, "draw_2_vs_draw_0": shift_block}},
            "restricted_changed_cohort": {"draws": {"draw_1": rc_draw, "draw_2": rc_draw}},
        }
        e07_ref = publish_phase2(root, protocol, ds_fp, f"phase2/results/e07/{m}", KIND_ANALYSIS, e07_body, prod)

        model_refs[m] = {
            "e29_analysis": plain(e29_ref),
            "e29_gate": plain(gate_ref),
            "e17_analysis": plain(e17_ref),
            "e07_release": plain(e07_rel_ref),
            "e30_analysis": plain(e30_ref),
            "e11_analysis": plain(e11_ref),
            "e07_analysis": plain(e07_ref),
        }

    return {
        "phase2_plan": plain(plan_ref),
        "e07_audit": plain(audit_ref),
        "e17_auth": plain(auth_ref),
        "models": model_refs,
    }


class Stage5SynthesizeTests(unittest.TestCase):
    def test_end_to_end_synthesis_and_fresh_subprocess_readback(self):
        protocol = _load_protocol()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            refs = _build_synthetic_stage1_to_4_tree(root, protocol)
            events = []
            result = synthesize_phase2(
                root,
                protocol,
                refs,
                _fake_producer(),
                run_id="stage5_test_run",
                run_subprocess_readback=True,
                progress=events.append,
            )
            self.assertEqual(result["status"], "passed")
            self.assertTrue(result["independent_readback"]["verified"])
            self.assertEqual(result["independent_readback"]["verified_artifact_count"], 16)
            self.assertIn("# Phase 2 Results Review", result["review_markdown"])
            combined = result["body"]["decision_register"]["combined"]
            for did, rec in combined.items():
                self.assertIn(rec["decision"], DECISION_OUTCOMES, f"Bad decision for {did}")
            # E11 was synthetic negligible -> parked; E17 source was consequential -> retain; E07 restricted -> revise
            self.assertEqual(combined["E11_fit_covariate_scale"]["decision"], "park within tested scope")
            self.assertEqual(combined["E17_source_calibration"]["decision"], "retain")
            self.assertEqual(combined["E07_caption_view_stability"]["decision"], "revise")
            self.assertEqual(combined["E30_pooling_feature_vs_head"]["decision"], "revise")

            # Tampering with phase2_review.md must fail independent readback.
            rel_dir = (root / result["release_ref"]["path"]).parent
            (rel_dir / "phase2_review.md").write_text("tampered", encoding="utf-8")
            with self.assertRaises(ContractError):
                verify_phase2_release(root, protocol, result["release_ref"])

    def test_lineage_mismatch_and_tampered_upstream_rejected(self):
        protocol = _load_protocol()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            refs = _build_synthetic_stage1_to_4_tree(root, protocol)
            bad_refs = plain(refs)
            bad_refs["models"]["qwen"]["e11_analysis"]["sha256"] = "9" * 64
            with self.assertRaises(ContractError):
                synthesize_phase2(root, protocol, bad_refs, _fake_producer(), run_subprocess_readback=False)

    def test_all_four_decision_outcomes_reachable(self):
        # Verify retain, revise, park within tested scope, and insufficient evidence on E11.
        base = {
            "E29": {
                "image_caption_g_last": {"value": 0.1, "lower": 0.08, "upper": 0.12, "valid_draws": 2000},
                "image_caption_g_res": {"value": 0.03, "lower": 0.02, "upper": 0.04, "valid_draws": 2000},
                "image_caption_delta_g": {"value": 0.07, "lower": 0.05, "upper": 0.09, "valid_draws": 2000},
                "caption_only_delta_g": {"value": 0.01, "lower": 0.0, "upper": 0.02, "valid_draws": 2000},
                "gates": {},
            },
            "E30": {
                "feature_gain_fixed_head_last_auroc": {"value": 0.03, "lower": 0.02, "upper": 0.04, "valid_draws": 2000},
                "feature_gain_fixed_head_mean_auroc": {"value": 0.02, "lower": 0.01, "upper": 0.03, "valid_draws": 2000},
                "interaction_auroc": {"value": 0.005, "lower": -0.005, "upper": 0.012, "valid_draws": 2000},
                "off_distribution_collapse": {"flagged": False},
            },
            "E11": {
                "mean_scale_auroc_gap_reduction": {"value": 0.015, "lower": 0.005, "upper": 0.025, "valid_draws": 2000},
                "mean_scale_recovered_fraction": {"value": 0.40, "lower": 0.20, "upper": 0.60, "valid_draws": 2000},
                "mean_scale_nll_gain": {"value": 0.001, "lower": -0.001, "upper": 0.003, "valid_draws": 2000},
                "mean_scale_brier_gain": {"value": 0.001, "lower": -0.001, "upper": 0.003, "valid_draws": 2000},
                "invariance_checks": {"passed": True},
            },
            "E17": {
                "arms": {
                    arm: {
                        "nll_gain_vs_raw": {"value": 0.001, "lower": -0.001, "upper": 0.003, "valid_draws": 2000},
                        "brier_gain_vs_raw": {"value": 0.0005, "lower": -0.0005, "upper": 0.001, "valid_draws": 2000},
                        "nll_gain_vs_fit_prior": {"value": 0.01, "lower": 0.005, "upper": 0.015, "valid_draws": 2000},
                        "brier_gain_vs_fit_prior": {"value": 0.005, "lower": 0.002, "upper": 0.008, "valid_draws": 2000},
                        "flip_rate_0p5": 0.0,
                    }
                    for arm in ("source_dev_c:temp", "source_dev_c:temp_intercept", "target_dev_c:temp", "target_dev_c:temp_intercept")
                },
                "invariance_checks": {"max_abs_auroc_diff": 0.0, "max_abs_ap_diff": 0.0, "all_T_positive": True, "temp_only_0p5_flips": 0},
            },
            "E07": {
                "verdict": "passed_full",
                "eligible_changed_languages": ["en", "hi", "bn", "te"],
                "unavailable_languages": [],
                "draw0_parity_check": {"passed": True},
                "draws": {
                    dk: {
                        "full_12_g_last_shift": {"value": 0.002, "lower": -0.01, "upper": 0.014, "valid_draws": 2000},
                        "full_12_g_res_shift": {"value": 0.001, "lower": -0.008, "upper": 0.010, "valid_draws": 2000},
                        "changed_9_g_last_shift": {"value": 0.002, "lower": -0.01, "upper": 0.014, "valid_draws": 2000},
                        "changed_9_g_res_shift": {"value": 0.001, "lower": -0.008, "upper": 0.010, "valid_draws": 2000},
                        "sign_preserved_12": True,
                        "sign_preserved_9": True,
                        "restricted_g_last_shift": {"value": 0.002, "lower": -0.01, "upper": 0.014, "valid_draws": 2000},
                    }
                    for dk in ("draw_1", "draw_2")
                },
            },
        }
        d_retain = evaluate_model_decisions("qwen", base)
        self.assertEqual(d_retain["E11_fit_covariate_scale"]["decision"], "retain")
        self.assertEqual(d_retain["E17_source_calibration"]["decision"], "park within tested scope")
        self.assertEqual(d_retain["E07_caption_view_stability"]["decision"], "retain")

        # Insufficient evidence via wide CI straddling threshold
        base["E11"]["mean_scale_auroc_gap_reduction"] = {"value": 0.008, "lower": -0.005, "upper": 0.020, "valid_draws": 2000}
        d_insuff = evaluate_model_decisions("qwen", base)
        self.assertEqual(d_insuff["E11_fit_covariate_scale"]["decision"], "insufficient evidence")

        # Loss-only gain -> revise
        base["E11"]["mean_scale_nll_gain"] = {"value": 0.020, "lower": 0.010, "upper": 0.030, "valid_draws": 2000}
        d_revise = evaluate_model_decisions("qwen", base)
        self.assertEqual(d_revise["E11_fit_covariate_scale"]["decision"], "revise")

    def test_notebook_structure_and_prefilled_refs(self):
        nb_path = REPO / "scripts/phase2/synthesize_phase2.ipynb"
        self.assertTrue(nb_path.is_file())
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertEqual(len(nb["cells"]), 4)
        full = "\n".join("".join(c["source"]) for c in nb["cells"])
        self.assertIn("CODE_REF = 'phase2'", full)
        self.assertIn(CERTIFIED_PHASE2_REFS["phase2_plan"]["sha256"], full)
        self.assertIn(CERTIFIED_PHASE2_REFS["models"]["qwen"]["e07_analysis"]["sha256"], full)
        self.assertIn(CERTIFIED_PHASE2_REFS["models"]["gemma"]["e07_analysis"]["sha256"], full)


if __name__ == "__main__":
    unittest.main()
