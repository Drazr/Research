"""Exhaustive boundary, roundtrip, tamper, and bundle consistency tests for Phase 2 gates & plan."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from src.common.contracts import LANGUAGES, ContractError, fingerprint
from src.phase2.gates import (
    ACCESSIBILITY_DIAGONAL_FLOOR,
    BOOTSTRAP_MIN_VALID_DRAWS,
    E29_GATE_THRESHOLDS,
    STAGE5_DECISION_RANGES,
    evaluate_e29_gates,
    load_e29_gate,
    publish_e29_gate,
)
from src.phase2.interfaces import (
    CONDITIONS,
    KIND_E07_AUDIT,
    KIND_E17_AUTH,
    KIND_PLAN,
    MODELS,
    PHASE2_PARENT_DIRS,
    PHASE2_VERSION,
    POOLINGS,
    REPO,
    SCHEMA_E07_AUDIT,
    SCHEMA_E17_AUTH,
    SCHEMA_PLAN,
    frozen,
    plain,
    publish_phase2,
)
from src.phase2.plan import load_phase2_plan, prepare_phase2, verify_stage1_bundle

try:
    import numpy as np
except ImportError:
    np = None


def _all_12_routes(triggered_count=0, *, triggered_point=0.025, triggered_ci_low=0.0101):
    """Build 12 directed off-diagonal route residuals with `triggered_count` meeting E11 directional criteria."""
    routes = {}
    pairs = [(s, t) for s in LANGUAGES for t in LANGUAGES if s != t]
    for idx, (s, t) in enumerate(pairs):
        key = f"{s}->{t}"
        if idx < triggered_count:
            routes[key] = {"point": triggered_point, "ci": [triggered_ci_low, triggered_point + 0.015]}
        else:
            routes[key] = {"point": 0.004, "ci": [-0.002, 0.010]}
    return routes


def _good_diagonals(value=0.800):
    return {lang: value for lang in LANGUAGES}


class GateBoundaryTests(unittest.TestCase):
    def test_frozen_constants_immutable_and_complete(self):
        self.assertEqual(E29_GATE_THRESHOLDS["accessibility_diagonal_floor"], 0.750)
        self.assertEqual(E29_GATE_THRESHOLDS["bootstrap_min_valid_draws"], 1900)
        self.assertEqual(E29_GATE_THRESHOLDS["bootstrap_total_draws"], 2000)
        self.assertEqual(set(STAGE5_DECISION_RANGES.keys()), {"E17", "E07", "E30", "E11"})
        with self.assertRaises(TypeError):
            E29_GATE_THRESHOLDS["accessibility_diagonal_floor"] = 0.5
        with self.assertRaises(TypeError):
            STAGE5_DECISION_RANGES["E17"]["negligible"] = {}

    def test_valid_draws_1899_vs_1900(self):
        # At 1899 valid draws, both E30 and E11 must be UNRESOLVED even if effects are large or diagonals fail.
        res_1899 = evaluate_e29_gates(
            "qwen",
            diagonals=_good_diagonals(0.85),
            valid_draws=1899,
            delta_g_mean=0.050,
            delta_g_ci=[0.030, 0.070],
            g_res_mean=0.050,
            g_res_ci=[0.030, 0.070],
            route_residuals=_all_12_routes(6),
        )
        self.assertEqual(res_1899["decisions"]["E30"]["decision"], "UNRESOLVED")
        self.assertEqual(res_1899["decisions"]["E11"]["decision"], "UNRESOLVED")

        # Even if diagonal floor fails, valid_draws=1899 yields UNRESOLVED (not PARK).
        res_1899_low_diag = evaluate_e29_gates(
            "qwen",
            diagonals=_good_diagonals(0.60),
            valid_draws=1899,
            delta_g_mean=0.001,
            delta_g_ci=[-0.005, 0.005],
            g_res_mean=0.001,
            g_res_ci=[-0.005, 0.005],
            route_residuals=_all_12_routes(0),
        )
        self.assertEqual(res_1899_low_diag["decisions"]["E30"]["decision"], "UNRESOLVED")
        self.assertEqual(res_1899_low_diag["decisions"]["E11"]["decision"], "UNRESOLVED")

        # At exactly 1900 valid draws, OPEN holds cleanly.
        res_1900 = evaluate_e29_gates(
            "qwen",
            diagonals=_good_diagonals(0.85),
            valid_draws=1900,
            delta_g_mean=0.015,
            delta_g_ci=[0.005001, 0.025],
            g_res_mean=0.015,
            g_res_ci=[0.005001, 0.025],
            route_residuals=_all_12_routes(0),
        )
        self.assertEqual(res_1900["decisions"]["E30"]["decision"], "OPEN")
        self.assertEqual(res_1900["decisions"]["E11"]["decision"], "OPEN")
        self.assertEqual(res_1900["decisions"]["E11"]["trigger"], "broad_residual")

    def test_diagonal_accessibility_floor_boundaries(self):
        # Exact 0.750 on all 4 diagonals meets the floor.
        res_exact = evaluate_e29_gates(
            "gemma",
            diagonals=_good_diagonals(0.750),
            valid_draws=2000,
            delta_g_mean=0.020,
            delta_g_ci=[0.010, 0.030],
            g_res_mean=0.020,
            g_res_ci=[0.010, 0.030],
            route_residuals=_all_12_routes(0),
        )
        self.assertTrue(res_exact["observed"]["diagonal_floor_met"])
        self.assertEqual(res_exact["decisions"]["E30"]["decision"], "OPEN")
        self.assertEqual(res_exact["decisions"]["E11"]["decision"], "OPEN")

        # One diagonal at 0.749999 fails the floor -> PARK for both E30 and E11.
        diags = _good_diagonals(0.850)
        diags["te"] = 0.749999
        res_fail = evaluate_e29_gates(
            "gemma",
            diagonals=diags,
            valid_draws=1900,
            delta_g_mean=0.040,
            delta_g_ci=[0.020, 0.060],
            g_res_mean=0.040,
            g_res_ci=[0.020, 0.060],
            route_residuals=_all_12_routes(5),
        )
        self.assertFalse(res_fail["observed"]["diagonal_floor_met"])
        self.assertEqual(res_fail["decisions"]["E30"]["decision"], "PARK")
        self.assertEqual(res_fail["decisions"]["E30"]["reason"], "accessibility_floor_failed")
        self.assertEqual(res_fail["decisions"]["E11"]["decision"], "PARK")
        self.assertEqual(res_fail["decisions"]["E11"]["reason"], "accessibility_floor_failed")

        # NaN or None diagonal fails the floor -> PARK when valid_draws >= 1900.
        diags_nan = _good_diagonals(0.850)
        diags_nan["bn"] = math.nan
        res_nan = evaluate_e29_gates(
            "qwen",
            diagonals=diags_nan,
            valid_draws=2000,
            delta_g_mean=0.030,
            delta_g_ci=[0.010, 0.050],
            g_res_mean=0.030,
            g_res_ci=[0.010, 0.050],
            route_residuals=_all_12_routes(4),
        )
        self.assertFalse(res_nan["observed"]["diagonal_floor_met"])
        self.assertEqual(res_nan["decisions"]["E30"]["decision"], "PARK")
        self.assertEqual(res_nan["decisions"]["E11"]["decision"], "PARK")

    def test_e30_open_park_unresolved_boundaries(self):
        # Exact L_Delta == 0.005 is NOT strictly > 0.005 -> UNRESOLVED (since U_Delta >= 0.015).
        r1 = evaluate_e29_gates(
            "qwen",
            diagonals=_good_diagonals(),
            valid_draws=2000,
            delta_g_mean=0.015,
            delta_g_ci=[0.005, 0.025],
            g_res_mean=0.002,
            g_res_ci=[-0.002, 0.010],
            route_residuals=_all_12_routes(0),
        )
        self.assertEqual(r1["decisions"]["E30"]["decision"], "UNRESOLVED")

        # Point just below 0.015 with L_Delta > 0.005 and U_Delta >= 0.015 -> UNRESOLVED.
        r2 = evaluate_e29_gates(
            "qwen",
            diagonals=_good_diagonals(),
            valid_draws=2000,
            delta_g_mean=0.014999,
            delta_g_ci=[0.006, 0.020],
            g_res_mean=0.002,
            g_res_ci=[-0.002, 0.010],
            route_residuals=_all_12_routes(0),
        )
        self.assertEqual(r2["decisions"]["E30"]["decision"], "UNRESOLVED")

        # U_Delta strictly < 0.015 -> PARK.
        r3 = evaluate_e29_gates(
            "qwen",
            diagonals=_good_diagonals(),
            valid_draws=2000,
            delta_g_mean=0.008,
            delta_g_ci=[0.002, 0.014999],
            g_res_mean=0.002,
            g_res_ci=[-0.002, 0.010],
            route_residuals=_all_12_routes(0),
        )
        self.assertEqual(r3["decisions"]["E30"]["decision"], "PARK")

        # Exact U_Delta == 0.015 is NOT strictly < 0.015 -> UNRESOLVED.
        r4 = evaluate_e29_gates(
            "qwen",
            diagonals=_good_diagonals(),
            valid_draws=2000,
            delta_g_mean=0.008,
            delta_g_ci=[0.002, 0.015],
            g_res_mean=0.002,
            g_res_ci=[-0.002, 0.010],
            route_residuals=_all_12_routes(0),
        )
        self.assertEqual(r4["decisions"]["E30"]["decision"], "UNRESOLVED")

    def test_e11_directional_residual_2_vs_3_routes_and_exact_boundaries(self):
        # 2 directional routes with U_res < 0.015 -> PARK.
        r_2routes = evaluate_e29_gates(
            "gemma",
            diagonals=_good_diagonals(),
            valid_draws=2000,
            delta_g_mean=0.002,
            delta_g_ci=[-0.002, 0.006],
            g_res_mean=0.010,
            g_res_ci=[0.006, 0.014],
            route_residuals=_all_12_routes(2, triggered_point=0.025, triggered_ci_low=0.0101),
        )
        self.assertEqual(r_2routes["observed"]["directional_routes_count"], 2)
        self.assertEqual(r_2routes["decisions"]["E11"]["decision"], "PARK")
        self.assertIsNone(r_2routes["decisions"]["E11"]["trigger"])

        # 3 directional routes with L_res > 0.005 (even with G_res_mean < 0.015 and U_res < 0.015) -> OPEN!
        r_3routes = evaluate_e29_gates(
            "gemma",
            diagonals=_good_diagonals(),
            valid_draws=2000,
            delta_g_mean=0.002,
            delta_g_ci=[-0.002, 0.006],
            g_res_mean=0.010,
            g_res_ci=[0.005001, 0.014],
            route_residuals=_all_12_routes(3, triggered_point=0.025, triggered_ci_low=0.010001),
        )
        self.assertEqual(r_3routes["observed"]["directional_routes_count"], 3)
        self.assertEqual(r_3routes["decisions"]["E11"]["decision"], "OPEN")
        self.assertEqual(r_3routes["decisions"]["E11"]["trigger"], "directional_residual_trigger")

        # Exact route ci_low == 0.010 does NOT trigger (must be strictly > 0.010) -> 0 routes -> PARK.
        r_ci_eq = evaluate_e29_gates(
            "gemma",
            diagonals=_good_diagonals(),
            valid_draws=2000,
            delta_g_mean=0.002,
            delta_g_ci=[-0.002, 0.006],
            g_res_mean=0.010,
            g_res_ci=[0.006, 0.014],
            route_residuals=_all_12_routes(4, triggered_point=0.030, triggered_ci_low=0.010),
        )
        self.assertEqual(r_ci_eq["observed"]["directional_routes_count"], 0)
        self.assertEqual(r_ci_eq["decisions"]["E11"]["decision"], "PARK")

        # Exact route point == 0.024999 does NOT trigger -> 0 routes -> PARK.
        r_pt_low = evaluate_e29_gates(
            "gemma",
            diagonals=_good_diagonals(),
            valid_draws=2000,
            delta_g_mean=0.002,
            delta_g_ci=[-0.002, 0.006],
            g_res_mean=0.010,
            g_res_ci=[0.006, 0.014],
            route_residuals=_all_12_routes(4, triggered_point=0.024999, triggered_ci_low=0.015),
        )
        self.assertEqual(r_pt_low["observed"]["directional_routes_count"], 0)
        self.assertEqual(r_pt_low["decisions"]["E11"]["decision"], "PARK")

        # 3 directional routes with L_res == 0.005 and U_res < 0.015 -> neither OPEN nor PARK -> UNRESOLVED!
        r_unres = evaluate_e29_gates(
            "gemma",
            diagonals=_good_diagonals(),
            valid_draws=2000,
            delta_g_mean=0.002,
            delta_g_ci=[-0.002, 0.006],
            g_res_mean=0.010,
            g_res_ci=[0.005, 0.014],
            route_residuals=_all_12_routes(3, triggered_point=0.030, triggered_ci_low=0.015),
        )
        self.assertEqual(r_unres["observed"]["directional_routes_count"], 3)
        self.assertEqual(r_unres["decisions"]["E11"]["decision"], "UNRESOLVED")

        # Both broad and directional open -> OPEN with broad_and_directional_residual.
        r_both = evaluate_e29_gates(
            "qwen",
            diagonals=_good_diagonals(),
            valid_draws=2000,
            delta_g_mean=0.002,
            delta_g_ci=[-0.002, 0.006],
            g_res_mean=0.020,
            g_res_ci=[0.010, 0.030],
            route_residuals=_all_12_routes(3, triggered_point=0.030, triggered_ci_low=0.015),
        )
        self.assertEqual(r_both["decisions"]["E11"]["decision"], "OPEN")
        self.assertEqual(r_both["decisions"]["E11"]["trigger"], "broad_and_directional_residual")

        # Sequence input format is also accepted; wrong count (!= 12) rejected.
        seq = [{"route": k, "point": v["point"], "ci_low": v["ci"][0]} for k, v in _all_12_routes(3).items()]
        r_seq = evaluate_e29_gates(
            "qwen",
            diagonals=_good_diagonals(),
            valid_draws=2000,
            delta_g_mean=0.002,
            delta_g_ci=[-0.002, 0.006],
            g_res_mean=0.010,
            g_res_ci=[0.006, 0.014],
            route_residuals=seq,
        )
        self.assertEqual(r_seq["decisions"]["E11"]["decision"], "OPEN")
        with self.assertRaises(ContractError):
            evaluate_e29_gates(
                "qwen",
                diagonals=_good_diagonals(),
                valid_draws=2000,
                delta_g_mean=0.002,
                delta_g_ci=[-0.002, 0.006],
                g_res_mean=0.010,
                g_res_ci=[0.006, 0.014],
                route_residuals=seq[:11],
            )

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ContractError):
            evaluate_e29_gates(
                "unknown_model",
                diagonals=_good_diagonals(),
                valid_draws=2000,
                delta_g_mean=0.0,
                delta_g_ci=[0.0, 0.0],
                g_res_mean=0.0,
                g_res_ci=[0.0, 0.0],
                route_residuals=_all_12_routes(0),
            )
        with self.assertRaises(ContractError):
            evaluate_e29_gates(
                "qwen",
                diagonals=_good_diagonals(),
                valid_draws=True,
                delta_g_mean=0.0,
                delta_g_ci=[0.0, 0.0],
                g_res_mean=0.0,
                g_res_ci=[0.0, 0.0],
                route_residuals=_all_12_routes(0),
            )
        with self.assertRaises(ContractError):
            evaluate_e29_gates(
                "qwen",
                diagonals=_good_diagonals(),
                valid_draws=2001,
                delta_g_mean=0.0,
                delta_g_ci=[0.0, 0.0],
                g_res_mean=0.0,
                g_res_ci=[0.0, 0.0],
                route_residuals=_all_12_routes(0),
            )


@dataclass(frozen=True)
class _FakeStage4Plan:
    reference: dict


@dataclass(frozen=True)
class _FakeBinding:
    root: Path
    protocol: dict
    stage4_release_ref: dict
    stage4_release: dict
    stage4_plan: _FakeStage4Plan
    bootstrap_shape: tuple
    bootstrap_sha256: str
    midpoints: dict
    dataset_ref: dict
    data_plan_ref: dict
    model_plan_refs: dict
    suite_ref: dict
    stage3_registry_ref: dict
    g0_gate_ref: dict
    g0_p_value: float
    dataset_fingerprint: str
    protocol_fingerprint: str
    lineage_identity: str


def _make_fake_binding(root, protocol, *, lineage_seed="v1"):
    dataset_fp = fingerprint(["dataset", lineage_seed])
    protocol_fp = fingerprint(protocol)
    stage4_release_ref = {
        "path": f"results/stage4/releases/{lineage_seed}/_COMPLETE.json",
        "sha256": fingerprint(["stage4_release", lineage_seed]),
    }
    stage4_plan_ref = {
        "path": f"cache/shared/stage4/plans/{lineage_seed}/_COMPLETE.json",
        "sha256": fingerprint(["stage4_plan", lineage_seed]),
    }
    registry_ref = {
        "path": f"results/stage3-validation/{lineage_seed}/_COMPLETE.json",
        "sha256": fingerprint(["stage3_registry", lineage_seed]),
    }
    g0_ref = {
        "path": f"results/stage3-g0/{lineage_seed}/_COMPLETE.json",
        "sha256": fingerprint(["g0_gate", lineage_seed]),
    }
    suite_ref = {
        "path": f"results/stage2-validation/{lineage_seed}/_COMPLETE.json",
        "sha256": fingerprint(["stage2_suite", lineage_seed]),
    }
    dataset_ref = {
        "path": f"shared/{protocol_fp}/manifests/{dataset_fp}/_COMPLETE.json",
        "sha256": fingerprint(["stage1_dataset", lineage_seed]),
    }
    data_plan_ref = {
        "path": f"cache/shared/stage2/data_plan/{lineage_seed}/_COMPLETE.json",
        "sha256": fingerprint(["data_plan", lineage_seed]),
    }
    model_plan_refs = {
        m: {"path": f"cache/shared/stage2/models/{m}/{lineage_seed}/_COMPLETE.json",
            "sha256": fingerprint(["model_plan", m, lineage_seed])}
        for m in MODELS
    }
    bootstrap_sha = fingerprint(["bootstrap", lineage_seed])
    midpoints = {"qwen": 18, "gemma": 17}
    lineage_identity = fingerprint(["lineage", lineage_seed, dataset_fp, protocol_fp])
    return _FakeBinding(
        root=root,
        protocol=protocol,
        stage4_release_ref=stage4_release_ref,
        stage4_release={},
        stage4_plan=_FakeStage4Plan(reference=stage4_plan_ref),
        bootstrap_shape=(2000, 360),
        bootstrap_sha256=bootstrap_sha,
        midpoints=midpoints,
        dataset_ref=dataset_ref,
        data_plan_ref=data_plan_ref,
        model_plan_refs=model_plan_refs,
        suite_ref=suite_ref,
        stage3_registry_ref=registry_ref,
        g0_gate_ref=g0_ref,
        g0_p_value=0.362,
        dataset_fingerprint=dataset_fp,
        protocol_fingerprint=protocol_fp,
        lineage_identity=lineage_identity,
    )


def _fake_e29_blocks(_binding, model):
    out = {}
    for cond in CONDITIONS:
        out[cond] = {}
        for pool in POOLINGS:
            out[cond][pool] = {}
            for lang in LANGUAGES:
                bid = fingerprint(["block", model, cond, pool, lang])
                out[cond][pool][lang] = {
                    "block_id": bid,
                    "target_group": fingerprint(["grp", model, cond, pool, lang]),
                    "prediction_ref": {
                        "path": f"predictions/stage4/{model}/{bid}/_COMPLETE.json",
                        "sha256": fingerprint(["pred", bid]),
                    },
                    "routes": {src: fingerprint(["route", bid, src]) for src in LANGUAGES},
                }
    return frozen(out)


def _fake_e11_stats(_binding, model):
    out = {}
    for lang in LANGUAGES:
        gid = fingerprint(["grp", model, "image_caption", "caption_mean", lang])
        out[lang] = {
            "group_id": gid,
            "selection_ref": {"path": f"results/stage3/sel/{gid}/_COMPLETE.json", "sha256": fingerprint(["sel", gid])},
            "candidate_ref": {"path": f"results/stage3/cand/{gid}/_COMPLETE.json", "sha256": fingerprint(["cand", gid])},
            "C": 1.0,
            "statistics_ref": {"path": f"results/stage3/stats/{gid}/_COMPLETE.json", "sha256": fingerprint(["stats", gid])},
            "dimension": 2048,
            "site": f"decoder:{18 if model == 'qwen' else 17}:caption_mean",
            "model_plan_ref": {"path": f"cache/shared/stage2/models/{model}/_COMPLETE.json", "sha256": fingerprint(["mp", model])},
            "screen_block_id": fingerprint(["sb", gid]),
            "screen_prediction_ref": {"path": f"predictions/stage4/{model}/{gid}/_COMPLETE.json", "sha256": fingerprint(["sp", gid])},
            "screen_routes": {src: fingerprint(["sr", gid, src]) for src in LANGUAGES},
            "screen_feature_release_ref": {"path": f"features/stage2/{model}/{gid}/_COMPLETE.json", "sha256": fingerprint(["fr", gid])},
        }
    return frozen(out)


def _publish_fake_e07_audit(root, protocol, binding, producer, *, verdict="passed_full", lineage_override=None, draw0_verified=True, confirm_excluded=True):
    body = {
        "schema": SCHEMA_E07_AUDIT,
        "stage4_release": plain(binding.stage4_release_ref),
        "lineage_identity": lineage_override or binding.lineage_identity,
        "dataset_ref": plain(binding.dataset_ref),
        "dataset_fingerprint": binding.dataset_fingerprint,
        "draw0_verified": draw0_verified,
        "confirm_excluded": confirm_excluded,
        "verdict": verdict,
        "verdict_reasons": ["synthetic_test"],
        "screen_multiplicity": {lang: {"min": 1, "median": 3, "max": 5} for lang in LANGUAGES},
        "fit_dev_s_multiplicity": {lang: {"min": 1, "median": 3, "max": 5} for lang in LANGUAGES},
        "draw_buckets": {lang: {"draw1": {"changed_text_vs_draw0": 150}} for lang in LANGUAGES},
        "support_tables": {"full_screen": {}, "per_language_changed": {}, "four_language_intersection": {}},
        "unique_changed_views": {"per_language": {lang: 150 for lang in LANGUAGES}, "total": 600},
        "extraction_plan": None if verdict == "blocked" else {"chunk_size": 32, "chunks": []},
    }
    return publish_phase2(
        root,
        protocol,
        binding.dataset_fingerprint,
        "phase2/audits/e07",
        KIND_E07_AUDIT,
        body,
        producer,
        upstream=[plain(binding.stage4_release_ref)],
    )


def _publish_fake_e17_auth(root, protocol, binding, producer, *, lineage_override=None, diag_passed=True, max_abs_diff=0.0):
    models_body = {}
    for m in MODELS:
        models_body[m] = {
            "midpoint_depth": binding.midpoints[m],
            "screen_blocks": {},
            "dev_c_diagonals": {},
            "off_diagonal_blocks": {},
            "source_systems": {},
            "diagonal_reproduction_check": {
                "passed": diag_passed,
                "max_abs_diff": max_abs_diff,
                "atol": 0.0,
                "rtol": 0.0,
            },
        }
    body = {
        "schema": SCHEMA_E17_AUTH,
        "stage4_release": plain(binding.stage4_release_ref),
        "lineage_identity": lineage_override or binding.lineage_identity,
        "dataset_fingerprint": binding.dataset_fingerprint,
        "permission": "phase2.e17_dev_c_transfer",
        "confirm_excluded": True,
        "models": models_body,
        "calibrator_recipe": {
            "forms": ["raw", "temp", "temp_intercept"],
            "optimizer": "BFGS",
            "invariance_tol": 1e-12,
        },
    }
    return publish_phase2(
        root,
        protocol,
        binding.dataset_fingerprint,
        "phase2/plans",
        KIND_E17_AUTH,
        body,
        producer,
        upstream=[plain(binding.stage4_release_ref)],
    )


class PlanAndBundleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.protocol = json.loads((REPO / "configs/protocol.json").read_text(encoding="utf-8"))
        self.producer = {
            "code_commit": "4" * 40,
            "storage_scope": "local-development",
            "software_versions": {"python": "3.12.0"},
        }
        self.binding = _make_fake_binding(self.root, self.protocol)

    def _prepare_with_mocked_binding(self, *, e07_audit_ref=None, e17_auth_ref=None, progress=None):
        with (
            mock.patch("src.phase2.plan.bind_phase1", return_value=self.binding),
            mock.patch("src.phase2.plan.e29_screen_blocks", side_effect=_fake_e29_blocks),
            mock.patch("src.phase2.plan.e11_fit_statistics_refs", side_effect=_fake_e11_stats),
        ):
            return prepare_phase2(
                self.root,
                self.protocol,
                self.binding.stage4_release_ref,
                self.producer,
                e07_audit_ref=e07_audit_ref,
                e17_auth_ref=e17_auth_ref,
                progress=progress,
            )

    def test_prepare_load_roundtrip_gate_io_and_tamper_rejection(self):
        events = []
        plan = self._prepare_with_mocked_binding(progress=events.append)
        self.assertEqual(len(events), 5)
        self.assertEqual(plan["phase2_version"], PHASE2_VERSION)
        self.assertTrue(plan["confirm_excluded"])
        self.assertEqual(tuple(plan["initialized_dirs"]), PHASE2_PARENT_DIRS)
        for rel in PHASE2_PARENT_DIRS:
            self.assertTrue((self.root / rel).is_dir())
        self.assertEqual(plain(plan["gate_thresholds"]), plain(E29_GATE_THRESHOLDS))
        self.assertEqual(plain(plan["stage5_decision_ranges"]), plain(STAGE5_DECISION_RANGES))
        for model in MODELS:
            count = sum(
                len(plan["e29_blocks"][model][c][p])
                for c in CONDITIONS
                for p in POOLINGS
            )
            self.assertEqual(count, 16)

        # Fresh load_phase2_plan roundtrip succeeds.
        reloaded = load_phase2_plan(self.root, self.protocol, plan.reference)
        self.assertEqual(plain(reloaded), plain(plan))
        self.assertEqual(plain(reloaded.reference), plain(plan.reference))

        # Gate publish + load roundtrip bound to this plan.
        eval_qwen = evaluate_e29_gates(
            "qwen",
            diagonals=_good_diagonals(0.82),
            valid_draws=2000,
            delta_g_mean=0.025,
            delta_g_ci=[0.015, 0.035],
            g_res_mean=0.012,
            g_res_ci=[0.006, 0.014],
            route_residuals=_all_12_routes(3),
        )
        e29_ref = {
            "path": "phase2/results/e29/qwen/run1/_COMPLETE.json",
            "sha256": "e" * 64,
        }
        gate_ref = publish_e29_gate(
            self.root, self.protocol, reloaded, "qwen", e29_ref, eval_qwen, self.producer
        )
        loaded_gate_ref, gate_body, _ = load_e29_gate(
            self.root, self.protocol, gate_ref, plan=reloaded, expected_model="qwen"
        )
        self.assertEqual(plain(loaded_gate_ref), plain(gate_ref))
        self.assertEqual(gate_body["decisions"]["E30"]["decision"], "OPEN")
        self.assertEqual(gate_body["decisions"]["E11"]["decision"], "OPEN")
        self.assertEqual(gate_body["decisions"]["E11"]["trigger"], "directional_residual_trigger")
        with self.assertRaisesRegex(ContractError, "Foreign Phase 2 owner"):
            load_e29_gate(self.root, self.protocol, gate_ref, plan=reloaded, expected_model="gemma")

        # Tampered checksum rejected.
        with self.assertRaisesRegex(ContractError, "checksum mismatch"):
            load_phase2_plan(
                self.root,
                self.protocol,
                {"path": plan.reference["path"], "sha256": "0" * 64},
            )

        # Tampered body.json on disk rejected.
        body_path = (self.root / plan.reference["path"]).parent / "body.json"
        original_bytes = body_path.read_bytes()
        tampered = json.loads(original_bytes.decode("utf-8"))
        tampered["confirm_excluded"] = False
        body_path.write_bytes((json.dumps(tampered) + "\n").encode("utf-8"))
        with self.assertRaises(ContractError):
            load_phase2_plan(self.root, self.protocol, plan.reference)

        # Restore body.json and verify CLI readback works in a fresh subprocess.
        body_path.write_bytes(original_bytes)
        child = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.phase2.operations",
                "readback",
                "--phase1-root",
                str(self.root),
                "--manifest",
                json.dumps(plain(plan.reference)),
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        )
        out = json.loads(child.stdout.strip())
        self.assertEqual(out["status"], "passed")
        self.assertEqual(out["plan_identity"], plan["plan_identity"])

    def test_verify_stage1_bundle_and_cross_artifact_consistency(self):
        e07_ref = _publish_fake_e07_audit(self.root, self.protocol, self.binding, self.producer, verdict="passed_restricted")
        e17_ref = _publish_fake_e17_auth(self.root, self.protocol, self.binding, self.producer)

        # Prepare plan binding both completed Stage 1 branch refs.
        plan = self._prepare_with_mocked_binding(e07_audit_ref=e07_ref, e17_auth_ref=e17_ref)
        self.assertEqual(plain(plan["e07_audit"]), plain(e07_ref))
        self.assertEqual(plain(plan["e17_auth"]), plain(e17_ref))

        summary = verify_stage1_bundle(self.root, self.protocol, plan.reference, e07_ref, e17_ref)
        self.assertTrue(summary["stage1_complete"])
        self.assertEqual(summary["e07_verdict"], "passed_restricted")
        self.assertTrue(summary["parallel_stage2_ready"]["E29"])
        self.assertTrue(summary["parallel_stage2_ready"]["E17"])
        self.assertTrue(summary["parallel_stage2_ready"]["E07_GPU"])

        # Blocked E07 audit still completes Stage 1 cleanly but keeps E07_GPU=False.
        plan_unbound = self._prepare_with_mocked_binding()
        e07_blocked = _publish_fake_e07_audit(self.root, self.protocol, self.binding, self.producer, verdict="blocked")
        summary_blocked = verify_stage1_bundle(self.root, self.protocol, plan_unbound.reference, e07_blocked, e17_ref)
        self.assertTrue(summary_blocked["stage1_complete"])
        self.assertFalse(summary_blocked["parallel_stage2_ready"]["E07_GPU"])

        # CLI verify-stage1-bundle succeeds on consistent bundle.
        child = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.phase2.operations",
                "verify-stage1-bundle",
                "--phase1-root",
                str(self.root),
                "--plan",
                json.dumps(plain(plan.reference)),
                "--e07-audit",
                json.dumps(plain(e07_ref)),
                "--e17-auth",
                json.dumps(plain(e17_ref)),
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        )
        cli_out = json.loads(child.stdout.strip())
        self.assertEqual(cli_out["status"], "passed")
        self.assertTrue(cli_out["stage1_complete"])

        # Mismatched lineage in E07 or E17 is rejected by both prepare_phase2 and verify_stage1_bundle.
        bad_e07 = _publish_fake_e07_audit(
            self.root, self.protocol, self.binding, self.producer, lineage_override="f" * 64
        )
        with self.assertRaisesRegex(ContractError, "lineage_identity mismatch"):
            verify_stage1_bundle(self.root, self.protocol, plan_unbound.reference, bad_e07, e17_ref)
        with self.assertRaisesRegex(ContractError, "lineage_identity mismatch"):
            self._prepare_with_mocked_binding(e07_audit_ref=bad_e07)

        # Failed E17 diagonal reproduction check is rejected.
        bad_e17 = _publish_fake_e17_auth(
            self.root, self.protocol, self.binding, self.producer, diag_passed=False, max_abs_diff=1e-6
        )
        with self.assertRaisesRegex(ContractError, "diagonal reproduction check failed"):
            verify_stage1_bundle(self.root, self.protocol, plan_unbound.reference, e07_ref, bad_e17)

        # CLI returns exit code 2 and JSON blocker on mismatch.
        failed_cli = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.phase2.operations",
                "verify-stage1-bundle",
                "--phase1-root",
                str(self.root),
                "--plan",
                json.dumps(plain(plan_unbound.reference)),
                "--e07-audit",
                json.dumps(plain(bad_e07)),
                "--e17-auth",
                json.dumps(plain(e17_ref)),
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        self.assertEqual(failed_cli.returncode, 2)
        err_out = json.loads(failed_cli.stdout.strip())
        self.assertEqual(err_out["status"], "blocked")
        self.assertFalse(err_out["stage1_complete"])
        self.assertTrue(err_out["blockers"])

    def test_prepare_phase2_notebook_contract(self):
        nb_path = REPO / "scripts/phase2/prepare_phase2.ipynb"
        self.assertTrue(nb_path.is_file())
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertEqual(nb["nbformat"], 4)
        self.assertGreaterEqual(len(nb["cells"]), 4)
        all_source = "\n".join("".join(cell["source"]) for cell in nb["cells"])
        self.assertIn("requirements-stage2-cpu.txt", all_source)
        self.assertIn("OPENBLAS_NUM_THREADS", all_source)
        self.assertIn("OMP_NUM_THREADS", all_source)
        self.assertIn("prepare-phase2", all_source)
        self.assertIn("readback", all_source)
        self.assertIn("verify-stage1-bundle", all_source)
        self.assertIn("flush_and_unmount", all_source)


@unittest.skipIf(np is None, "numpy/scipy/sklearn required for full Phase 1 end-to-end fixture test")
class FullPhase1FixturePlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tests_dir = str(REPO / "tests")
        if tests_dir not in sys.path:
            sys.path.insert(0, tests_dir)
        import test_stage5_foundation as stage5

        stage5.ReportingTests.setUpClass()
        cls.addClassCleanup(stage5.ReportingTests.doClassCleanups)
        cls.plan = stage5.ReportingTests.plan
        cls.release = stage5.ReportingTests.release
        cls.root = cls.plan.root
        cls.protocol = cls.plan.protocol

    def test_end_to_end_prepare_and_load_phase2_plan(self):
        producer = {
            "code_commit": "4" * 40,
            "storage_scope": "local-development",
            "software_versions": {"python": "3.12.0"},
        }
        p2 = prepare_phase2(self.root, self.protocol, self.release, producer)
        reloaded = load_phase2_plan(self.root, self.protocol, p2.reference)
        self.assertEqual(plain(reloaded), plain(p2))
        self.assertEqual(reloaded["bootstrap"]["shape"][0], 2000)


if __name__ == "__main__":
    unittest.main()
