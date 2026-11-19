"""Unit and fixture tests for Phase 2 Stage 3 CPU E30 worker and notebook."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.common.contracts import LANGUAGES, ContractError, fingerprint
from src.phase2.gates import evaluate_e29_gates, publish_e29_gate
from src.phase2.interfaces import (
    CONDITIONS,
    KIND_ANALYSIS,
    KIND_PREDICTION,
    POOLINGS,
    Phase1Binding,
    REPO,
    SCHEMA_ANALYSIS,
    SCHEMA_PREDICTION,
    frozen,
    load_phase2_artifact,
    plain,
)
from src.phase2.plan import prepare_phase2
from src.phase2.run_e30 import (
    ALL_ARMS,
    COLLAPSE_DROP_THRESHOLD,
    CROSSED_ARMS,
    MATCHED_ARMS,
    _gap_value,
    _improvement_value,
    _parse_hook_from_site,
    _valid_crossed_resume_keys,
    check_e30_gate_open,
    evaluate_off_distribution_collapse,
    run_e30,
    verify_e30_provenance,
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


def _make_provenance_fixture(*, n_images=12, dim=8, depth=18, model="qwen"):
    screen_ids = tuple(f"img-{i:04d}" for i in range(n_images))
    model_plan_ref = {"path": f"shared/mp_{model}/_COMPLETE.json", "sha256": "a" * 64}
    groups = {}
    gids_by_pooling = {p: {} for p in POOLINGS}
    rows_by_pooling = {p: {} for p in POOLINGS}
    matrices_by_pooling = {p: {} for p in POOLINGS}

    for pool in POOLINGS:
        for lang in LANGUAGES:
            gid = f"grp-{model}-image_caption-{pool}-{lang}"
            gids_by_pooling[pool][lang] = gid
            groups[gid] = {
                "group_id": gid,
                "source": "vlm",
                "model": model,
                "condition": "image_caption",
                "site": f"decoder:{depth}:{pool}",
                "language": lang,
                "dimension": dim,
                "model_plan": model_plan_ref,
                "releases": {
                    "SCREEN": {
                        "path": f"features/{model}/image_caption/SCREEN/_COMPLETE.json",
                        "sha256": fingerprint([model, "SCREEN"]),
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
        bootstrap_shape=(100, n_images),
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


class E30ContractAndNotebookTests(unittest.TestCase):
    """Stdlib-only tests verifying notebook, gate checks, provenance guards, and helpers."""

    def test_notebook_structure_and_code_ref(self):
        nb_path = REPO / "scripts/phase2/run_e30.ipynb"
        self.assertTrue(nb_path.is_file(), "Missing scripts/phase2/run_e30.ipynb")
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertEqual(nb.get("nbformat"), 4)
        self.assertEqual(len(nb.get("cells", [])), 4)
        setup_src = "".join(nb["cells"][1]["source"])
        cfg_src = "".join(nb["cells"][2]["source"])
        exec_src = "".join(nb["cells"][3]["source"])

        self.assertIn("CODE_REF = 'phase2'", setup_src)
        self.assertIn("OPENBLAS_NUM_THREADS", setup_src)
        self.assertIn("requirements-stage2-cpu.txt", setup_src)
        self.assertIn("MODEL = 'qwen'", cfg_src)
        self.assertIn("PLAN_REF = None", cfg_src)
        self.assertIn("E29_GATE_REF = None", cfg_src)
        self.assertIn("RESUME_CROSSED_REFS = None", cfg_src)
        self.assertIn("RESUME_ANALYSIS_REF = None", cfg_src)
        self.assertIn("src.phase2.run_e30", exec_src)
        self.assertIn("MODEL not in ('qwen', 'gemma')", exec_src)
        self.assertIn("--e29-gate", exec_src)

    def test_gate_open_and_closed_refusal(self):
        open_body = {
            "model": "qwen",
            "decisions": {"E30": {"decision": "OPEN", "reason": "consequential_pooling_reduction"}},
        }
        res = check_e30_gate_open(open_body, "qwen")
        self.assertEqual(res["decision"], "OPEN")

        for bad_decision in ("PARK", "UNRESOLVED"):
            closed_body = {
                "model": "qwen",
                "decisions": {"E30": {"decision": bad_decision, "reason": "test_closed"}},
            }
            with self.assertRaises(ContractError):
                check_e30_gate_open(closed_body, "qwen")

        with self.assertRaises(ContractError):
            check_e30_gate_open(open_body, "gemma")

    def test_invalid_model_and_resume_keys_rejected(self):
        protocol = _load_protocol()
        dummy_ref = {"path": "phase2/plans/x/_COMPLETE.json", "sha256": "0" * 64}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ContractError):
                run_e30(Path(tmp), protocol, dummy_ref, dummy_ref, "both", {})
            with self.assertRaises(ContractError):
                run_e30(
                    Path(tmp),
                    protocol,
                    dummy_ref,
                    dummy_ref,
                    "qwen",
                    {},
                    resume_crossed_refs={"last_on_last:en": dummy_ref},
                )

    def test_provenance_checks_pass_and_reject_mismatches(self):
        binding, gids, rows, mats = _make_provenance_fixture()
        prov = verify_e30_provenance(binding, "qwen", 18, gids, rows, mats)
        self.assertTrue(prov["depth_match"])
        self.assertTrue(prov["hook_match"])
        self.assertTrue(prov["model_plan_match"])
        self.assertTrue(prov["row_keys_match"])
        self.assertEqual(prov["hook"], "decoder:18")
        self.assertEqual(prov["dimension"], 8)

        # Depth mismatch
        binding_d, gids_d, rows_d, mats_d = _make_provenance_fixture()
        binding_d.stage4_plan.fit["groups"][gids_d["caption_mean"]["hi"]]["site"] = "decoder:17:caption_mean"
        with self.assertRaises(ContractError):
            verify_e30_provenance(binding_d, "qwen", 18, gids_d, rows_d, mats_d)

        # Dimension mismatch
        binding_dim, gids_dim, rows_dim, mats_dim = _make_provenance_fixture()
        binding_dim.stage4_plan.fit["groups"][gids_dim["caption_mean"]["en"]]["dimension"] = 16
        with self.assertRaises(ContractError):
            verify_e30_provenance(binding_dim, "qwen", 18, gids_dim, rows_dim, mats_dim)

        # model_plan mismatch
        binding_mp, gids_mp, rows_mp, mats_mp = _make_provenance_fixture()
        binding_mp.stage4_plan.fit["groups"][gids_mp["caption_mean"]["bn"]]["model_plan"] = {
            "path": "shared/wrong/_COMPLETE.json",
            "sha256": "f" * 64,
        }
        with self.assertRaises(ContractError):
            verify_e30_provenance(binding_mp, "qwen", 18, gids_mp, rows_mp, mats_mp)

        # Row keys mismatch across poolings
        binding_r, gids_r, rows_r, mats_r = _make_provenance_fixture()
        rows_r["caption_mean"]["te"] = tuple(reversed(rows_r["caption_mean"]["te"]))
        with self.assertRaises(ContractError):
            verify_e30_provenance(binding_r, "qwen", 18, gids_r, rows_r, mats_r)

        # CONFIRM split leakage rejected
        binding_c, gids_c, rows_c, mats_c = _make_provenance_fixture()
        bad_rows = tuple(dict(r, split="CONFIRM") for r in rows_c["last_caption"]["en"])
        rows_c["last_caption"]["en"] = bad_rows
        rows_c["caption_mean"]["en"] = bad_rows
        with self.assertRaises(ContractError):
            verify_e30_provenance(binding_c, "qwen", 18, gids_c, rows_c, mats_c)

    def test_parse_hook_and_direction_helpers(self):
        self.assertEqual(_parse_hook_from_site("decoder:18:last_caption", "last_caption"), ("decoder:18", 18))
        with self.assertRaises(ContractError):
            _parse_hook_from_site("decoder:18:caption_mean", "last_caption")

        self.assertAlmostEqual(_gap_value("auroc", 0.90, 0.80), 0.10)
        self.assertAlmostEqual(_gap_value("nll", 0.30, 0.50), 0.20)
        self.assertAlmostEqual(_improvement_value("auroc", 0.80, 0.85), 0.05)
        self.assertAlmostEqual(_improvement_value("brier", 0.20, 0.15), 0.05)
        self.assertEqual(len(_valid_crossed_resume_keys()), 8)


# ---------------------------------------------------------------------------
# Full numeric fixture tests (require numpy, scipy, sklearn)
# ---------------------------------------------------------------------------


def _fake_e29_blocks(model):
    out = {}
    for cond in CONDITIONS:
        out[cond] = {}
        for pool in POOLINGS:
            out[cond][pool] = {}
            for lang in LANGUAGES:
                out[cond][pool][lang] = {
                    "block_id": f"{model}-{cond}-{pool}-{lang}",
                    "target_group": f"grp-{model}-{cond}-{pool}-{lang}",
                    "prediction_ref": {
                        "path": f"predictions/stage4/{model}/{cond}_{pool}_{lang}/_COMPLETE.json",
                        "sha256": fingerprint([model, cond, pool, lang]),
                    },
                    "routes": {
                        src: f"{model}:{cond}:{pool}:{src}->{lang}" for src in LANGUAGES
                    },
                }
    return out


def _fake_e11_stats(model):
    return {
        lang: {
            "group_id": f"grp-{model}-image_caption-caption_mean-{lang}",
            "statistics_ref": {
                "path": f"checkpoints/stage3/{model}_{lang}_stats/_COMPLETE.json",
                "sha256": fingerprint(["e11", model, lang]),
            },
        }
        for lang in LANGUAGES
    }


def _build_synthetic_e30_world(
    root,
    protocol,
    *,
    model="qwen",
    n_images=40,
    n_draws=100,
    dim=6,
    seed=20261120,
    collapse_crossed=False,
):
    """Build deterministic Stage 2 features, Stage 3 heads, and Stage 4 matched blocks."""
    import numpy as np
    from src.stage3.engine import transform
    from src.stage4.predictions import Prediction

    categories = tuple(protocol["category_order"])
    n_cats = len(categories)
    screen_ids = tuple(f"img-{i:04d}" for i in range(n_images))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_images, size=(n_draws, n_images), dtype=np.int64)
    bootstrap_sha256 = hashlib.sha256(draws.tobytes()).hexdigest()

    # Targets: 50% positive per category, preserving unknown plant.n.02 on img-0000.
    targets = {}
    y_sign = np.ones((n_images, n_cats), dtype=np.float64)
    for i, iid in enumerate(screen_ids):
        vals = [int((i + c) % 2 == 0) for c in range(n_cats)]
        if i == 0 and "plant.n.02" in categories:
            vals[categories.index("plant.n.02")] = None
        targets[iid] = {"split": "SCREEN", "values": vals}
        for c, v in enumerate(vals):
            y_sign[i, c] = 1.0 if v == 1 else -1.0

    depth = 18 if model == "qwen" else 17
    model_plan_ref = {"path": f"shared/mp_{model}/_COMPLETE.json", "sha256": fingerprint(["mp", model])}

    groups = {}
    ordinary = {}
    features = {}  # (pool, lang) -> (rows, matrix)
    heads = {}     # (pool, lang) -> (coef, intercept, estimable, prevalence, stats)

    # Construct orthogonal feature subspaces so crossed arms have well-defined behavior:
    # - last_caption features put signal in dims 0..2; caption_mean features put stronger signal in dims 0..2 (or 3..5 if collapse_crossed).
    for pool in POOLINGS:
        for lang_idx, lang in enumerate(LANGUAGES):
            gid = f"grp-{model}-image_caption-{pool}-{lang}"
            groups[gid] = {
                "group_id": gid,
                "source": "vlm",
                "model": model,
                "condition": "image_caption",
                "site": f"decoder:{depth}:{pool}",
                "language": lang,
                "dimension": dim,
                "model_plan": model_plan_ref,
                "releases": {
                    "SCREEN": {
                        "path": f"features/{model}/image_caption/SCREEN/_COMPLETE.json",
                        "sha256": fingerprint(["feat", model, "SCREEN"]),
                    }
                },
            }
            ordinary[gid] = {
                "selection": {"path": f"checkpoints/s3/{gid}/sel/_COMPLETE.json", "sha256": fingerprint(["sel", gid])},
                "candidate": {"path": f"checkpoints/s3/{gid}/cand/_COMPLETE.json", "sha256": fingerprint(["cand", gid])},
                "C": 1.0,
            }

            rows = tuple(
                {"image_id": iid, "split": "SCREEN", "language": lang, "row_index": idx}
                for idx, iid in enumerate(screen_ids)
            )
            noise = rng.normal(0.0, 0.35, size=(n_images, dim))
            # Base signal from first 3 categories cycled across dims
            sig = np.zeros((n_images, dim), dtype=np.float64)
            for d_idx in range(dim):
                sig[:, d_idx] = y_sign[:, d_idx % n_cats]

            if collapse_crossed:
                # Put last_caption signal in dims 0..2, caption_mean signal in dims 3..5 (anti-correlated on crossed)
                if pool == "last_caption":
                    mat = np.concatenate([sig[:, :3] * 2.0 + noise[:, :3], -sig[:, 3:] * 2.0 + noise[:, 3:]], axis=1)
                else:
                    mat = np.concatenate([-sig[:, :3] * 2.0 + noise[:, :3], sig[:, 3:] * 2.0 + noise[:, 3:]], axis=1)
            else:
                # caption_mean has cleaner signal (less noise) than last_caption on shared dims
                scale_sig = 2.0 if pool == "caption_mean" else 1.2
                # Slight language shift so off-diagonal has nonzero gap
                lang_shift = 0.25 * (lang_idx - 1.5)
                mat = sig * scale_sig + noise + lang_shift
                if pool == "last_caption":
                    # Flip a few rows per language to create realistic transfer penalty
                    mat[: (4 + lang_idx), :] *= -0.5

            mat_f32 = mat.astype(np.float32)
            features[(pool, lang)] = (rows, mat_f32)

            # Frozen head & FIT stats
            coef = np.zeros((n_cats, dim), dtype=np.float64)
            for c in range(n_cats):
                if collapse_crossed:
                    if pool == "last_caption":
                        coef[c, c % 3] = 1.0
                    else:
                        coef[c, 3 + (c % 3)] = 1.0
                else:
                    coef[c, c % dim] = 1.0 + 0.05 * lang_idx
            intercept = np.zeros((n_cats,), dtype=np.float64)
            estimable = np.ones((n_cats,), dtype=bool)
            prevalence = np.full((n_cats,), 0.5, dtype=np.float64)
            stats = {
                "mean": np.zeros((dim,), dtype=np.float64),
                "variance": np.ones((dim,), dtype=np.float64),
                "scale": np.ones((dim,), dtype=np.float64),
                "center": True,
                "reference": {
                    "path": f"checkpoints/s3/{gid}/stats/_COMPLETE.json",
                    "sha256": fingerprint(["stats", gid]),
                },
            }
            heads[(pool, lang)] = (coef, intercept, estimable, prevalence, stats)

    # Build Stage 4 matched prediction blocks by exact scoring via transform(mat, stats) @ coef.T + intercept
    blocks_meta = _fake_e29_blocks(model)
    matched_by_block_id = {}
    for pool in POOLINGS:
        for t_lang in LANGUAGES:
            entry = blocks_meta["image_caption"][pool][t_lang]
            block_id = entry["block_id"]
            rows_t, mat_t = features[(pool, t_lang)]
            route_preds = {}
            for s_lang, rid in entry["routes"].items():
                coef, intercept, estimable, prevalence, stats = heads[(pool, s_lang)]
                logits = np.asarray(transform(mat_t, stats) @ coef.T, dtype=np.float64) + intercept
                route_preds[rid] = Prediction(
                    reference=dict(entry["prediction_ref"]),
                    route={"route_id": rid, "source_group": f"grp-{model}-image_caption-{pool}-{s_lang}", "target_group": entry["target_group"]},
                    split="SCREEN",
                    rows=rows_t,
                    logits=logits,
                    estimable=estimable.copy(),
                    prevalence=prevalence.copy(),
                )
            matched_by_block_id[block_id] = route_preds

    fit_dict = {"targets": targets, "groups": groups}
    stage4_plan = _PlanDict({
        "screen_ids": list(screen_ids),
        "categories": list(categories),
        "ordinary": ordinary,
    })
    stage4_plan.fit = fit_dict
    stage4_plan.store = None

    dataset_fp = fingerprint(["e30-fixture-dataset", model, n_images, seed])
    stage4_rel = {
        "path": "results/stage4/releases/fixture/_COMPLETE.json",
        "sha256": fingerprint(["stage4-rel", dataset_fp]),
    }
    binding = Phase1Binding(
        root=root,
        protocol=frozen(protocol),
        stage4_release_ref=frozen(stage4_rel),
        stage4_release=frozen({"predictions": {}}),
        stage4_plan=stage4_plan,
        bootstrap_shape=(n_draws, n_images),
        bootstrap_sha256=bootstrap_sha256,
        midpoints=frozen({"qwen": 18, "gemma": 17}),
        dataset_ref=frozen({"path": "shared/ds/_COMPLETE.json", "sha256": dataset_fp}),
        data_plan_ref=frozen({"path": "shared/dp/_COMPLETE.json", "sha256": "a" * 64}),
        model_plan_refs=frozen({
            "qwen": model_plan_ref if model == "qwen" else {"path": "shared/mp_q/_COMPLETE.json", "sha256": "b" * 64},
            "gemma": model_plan_ref if model == "gemma" else {"path": "shared/mp_g/_COMPLETE.json", "sha256": "c" * 64},
        }),
        suite_ref=frozen({"path": "results/suite/_COMPLETE.json", "sha256": "d" * 64}),
        stage3_registry_ref=frozen({"path": "results/s3/_COMPLETE.json", "sha256": "e" * 64}),
        g0_gate_ref=frozen({"path": "results/g0/_COMPLETE.json", "sha256": "f" * 64}),
        g0_p_value=0.362,
        dataset_fingerprint=dataset_fp,
        protocol_fingerprint=fingerprint(protocol),
        lineage_identity=fingerprint(["e30-lineage", dataset_fp, bootstrap_sha256]),
    )
    return binding, draws, features, heads, matched_by_block_id


@unittest.skipUnless(HAS_SCIENTIFIC, "numpy/scipy/sklearn required for E30 numeric fixture tests")
class E30NumericFixtureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.protocol = _load_protocol()
        self.producer = {
            "code_commit": "4" * 40,
            "storage_scope": "local-development",
            "software_versions": {"python": "3.12.0", "numpy": "2.1.3", "scipy": "1.18.1"},
        }

    def _prepare_plan_and_gate(self, binding, model, *, gate_mode="OPEN"):
        with (
            patch("src.phase2.plan.bind_phase1", return_value=binding),
            patch("src.phase2.plan.e29_screen_blocks", side_effect=lambda b, m: _fake_e29_blocks(m)),
            patch("src.phase2.plan.e11_fit_statistics_refs", side_effect=lambda b, m: _fake_e11_stats(m)),
        ):
            plan = prepare_phase2(self.root, self.protocol, binding.stage4_release_ref, self.producer)

        diags = {lang: 0.88 for lang in LANGUAGES}
        routes_low = {
            f"{s}->{t}": {"point": 0.005, "ci": [0.001, 0.009]}
            for s in LANGUAGES
            for t in LANGUAGES
            if s != t
        }
        if gate_mode == "OPEN":
            ev = evaluate_e29_gates(
                model,
                diagonals=diags,
                valid_draws=2000,
                delta_g_mean=0.030,
                delta_g_ci=[0.015, 0.045],
                g_res_mean=0.020,
                g_res_ci=[0.008, 0.032],
                route_residuals=routes_low,
            )
        elif gate_mode == "PARK":
            ev = evaluate_e29_gates(
                model,
                diagonals=diags,
                valid_draws=2000,
                delta_g_mean=0.004,
                delta_g_ci=[0.000, 0.010],
                g_res_mean=0.004,
                g_res_ci=[0.000, 0.010],
                route_residuals=routes_low,
            )
        else:
            ev = evaluate_e29_gates(
                model,
                diagonals=diags,
                valid_draws=1800,
                delta_g_mean=0.030,
                delta_g_ci=[0.015, 0.045],
                g_res_mean=0.020,
                g_res_ci=[0.008, 0.032],
                route_residuals=routes_low,
            )
        dummy_e29_ref = {
            "path": f"phase2/results/e29/{model}/fixture/_COMPLETE.json",
            "sha256": fingerprint(["e29", model, gate_mode]),
        }
        gate_ref = publish_e29_gate(
            self.root, self.protocol, plan, model, dummy_e29_ref, ev, self.producer
        )
        return plan, gate_ref

    def _run_e30_mocked(self, binding, draws, features, heads, matched_by_block_id, plan, gate_ref, model, **kwargs):
        with (
            patch("src.phase2.run_e30.bind_phase1", return_value=binding),
            patch("src.phase2.run_e30.e29_screen_blocks", side_effect=lambda b, m: _fake_e29_blocks(m)),
            patch("src.phase2.run_e30.read_verified_midpoint_features", side_effect=lambda b, m, c, p, lang, s: features[(p, lang)]),
            patch("src.phase2.run_e30.load_frozen_source_head", side_effect=lambda b, m, c, p, lang: heads[(p, lang)]),
            patch("src.stage4.plan.load_bootstrap", return_value=draws),
            patch(
                "src.stage4.predictions.load_prediction_block",
                side_effect=lambda s4plan, ref, expected_block=None: matched_by_block_id[expected_block],
            ),
        ):
            return run_e30(
                self.root,
                self.protocol,
                plan.reference,
                gate_ref,
                model,
                self.producer,
                **kwargs,
            )

    def test_refuses_park_and_unresolved_gate(self):
        binding, draws, features, heads, matched = _build_synthetic_e30_world(self.root, self.protocol)
        for mode in ("PARK", "UNRESOLVED"):
            plan, gate_ref = self._prepare_plan_and_gate(binding, "qwen", gate_mode=mode)
            with self.assertRaises(ContractError):
                self._run_e30_mocked(binding, draws, features, heads, matched, plan, gate_ref, "qwen")

    def test_end_to_end_2x2_contrasts_invariants_and_mask_preservation(self):
        binding, draws, features, heads, matched = _build_synthetic_e30_world(self.root, self.protocol)
        plan, gate_ref = self._prepare_plan_and_gate(binding, "qwen", gate_mode="OPEN")
        events = []
        analysis_ref, body, crossed_refs = self._run_e30_mocked(
            binding, draws, features, heads, matched, plan, gate_ref, "qwen", progress=events.append
        )

        # Verify 8 crossed prediction blocks published (no matched blocks duplicated)
        self.assertEqual(set(crossed_refs.keys()), set(CROSSED_ARMS))
        for arm in CROSSED_ARMS:
            self.assertEqual(set(crossed_refs[arm].keys()), set(LANGUAGES))
            for lang in LANGUAGES:
                _, pbody, _ = load_phase2_artifact(
                    self.root, self.protocol, crossed_refs[arm][lang], KIND_PREDICTION, expected_schema=SCHEMA_PREDICTION
                )
                self.assertEqual(pbody["experiment"], "E30")
                self.assertEqual(pbody["arm"], arm)
                self.assertEqual(pbody["target_language"], lang)
                self.assertEqual(pbody["source_languages"], list(LANGUAGES))
                self.assertTrue(pbody["confirm_excluded"])
                self.assertEqual(pbody["matched_self_check"]["max_abs_diff"], 0.0)

        # Verify analysis block schema & 2x2 algebraic identities
        self.assertEqual(body["experiment"], "E30")
        self.assertEqual(body["schema"], SCHEMA_ANALYSIS)
        self.assertTrue(body["confirm_excluded"])
        self.assertEqual(set(body["arms"].keys()), set(ALL_ARMS))
        self.assertEqual(set(body["matched_block_refs"].keys()), set(MATCHED_ARMS))

        c = body["contrasts"]
        fg_last = c["feature_gain_fixed_head_last"]["metrics"]["auroc"]["gap_reduction"]["primary_macro"]["value"]
        fg_mean = c["feature_gain_fixed_head_mean"]["metrics"]["auroc"]["gap_reduction"]["primary_macro"]["value"]
        hg_last = c["head_gain_fixed_feature_last"]["metrics"]["auroc"]["gap_reduction"]["primary_macro"]["value"]
        hg_mean = c["head_gain_fixed_feature_mean"]["metrics"]["auroc"]["gap_reduction"]["primary_macro"]["value"]
        inter = c["interaction"]["metrics"]["auroc"]["gap_reduction"]["primary_macro"]["value"]
        total = c["total_matched_reduction"]["metrics"]["auroc"]["gap_reduction"]["primary_macro"]["value"]

        self.assertAlmostEqual(total, fg_last + hg_mean, places=12)
        self.assertAlmostEqual(total, hg_last + fg_mean, places=12)
        self.assertAlmostEqual(inter, fg_last - fg_mean, places=12)

        # Verify unknown cell plant.n.02 preserved (known count = 39 vs 40)
        en_en_cats = body["arms"]["last_on_last"]["per_route"]["en->en"]["metrics"]["auroc"]["categories"]
        self.assertEqual(en_en_cats["plant.n.02"]["support"]["known"], 39)
        self.assertEqual(en_en_cats["person.n.01"]["support"]["known"], 40)

        # Verify streamed progress events
        self.assertTrue(any(e["event"] == "e30_published" for e in events))

    def test_en_matched_self_check_hard_fails_on_drift(self):
        binding, draws, features, heads, matched = _build_synthetic_e30_world(self.root, self.protocol)
        plan, gate_ref = self._prepare_plan_and_gate(binding, "qwen", gate_mode="OPEN")
        # Perturb accepted Stage 4 en->en logit by 1e-9
        en_block = matched["qwen-image_caption-last_caption-en"]
        en_pred = en_block["qwen:image_caption:last_caption:en->en"]
        en_pred.logits[0, 0] += 1e-9

        with self.assertRaises(ContractError):
            self._run_e30_mocked(binding, draws, features, heads, matched, plan, gate_ref, "qwen")

    def test_off_distribution_collapse_detection(self):
        import numpy as np

        zeros = np.zeros(50, dtype=np.float64)
        # Only one crossed arm drops >= 0.015 -> NOT flagged
        res_one = evaluate_off_distribution_collapse(
            last_transfer_val=0.80,
            last_transfer_draws=zeros + 0.80,
            last_diag_val=0.85,
            last_diag_draws=zeros + 0.85,
            crossed_transfer_vals={"head_last_on_mean": 0.75, "head_mean_on_last": 0.795},
            crossed_transfer_draws={"head_last_on_mean": zeros + 0.75, "head_mean_on_last": zeros + 0.795},
            crossed_diag_vals={"head_last_on_mean": 0.80, "head_mean_on_last": 0.845},
            crossed_diag_draws={"head_last_on_mean": zeros + 0.80, "head_mean_on_last": zeros + 0.845},
        )
        self.assertFalse(res_one["transfer_collapse_flagged"])
        self.assertFalse(res_one["diagonal_collapse_flagged"])
        self.assertFalse(res_one["flagged"])

        # Both crossed arms drop >= 0.015 -> flagged
        res_both = evaluate_off_distribution_collapse(
            last_transfer_val=0.80,
            last_transfer_draws=zeros + 0.80,
            last_diag_val=0.85,
            last_diag_draws=zeros + 0.85,
            crossed_transfer_vals={"head_last_on_mean": 0.75, "head_mean_on_last": 0.78},
            crossed_transfer_draws={"head_last_on_mean": zeros + 0.75, "head_mean_on_last": zeros + 0.78},
            crossed_diag_vals={"head_last_on_mean": 0.80, "head_mean_on_last": 0.82},
            crossed_diag_draws={"head_last_on_mean": zeros + 0.80, "head_mean_on_last": zeros + 0.82},
        )
        self.assertTrue(res_both["transfer_collapse_flagged"])
        self.assertTrue(res_both["diagonal_collapse_flagged"])
        self.assertTrue(res_both["flagged"])

        # End-to-end collapse fixture
        binding, draws, features, heads, matched = _build_synthetic_e30_world(
            self.root, self.protocol, collapse_crossed=True
        )
        plan, gate_ref = self._prepare_plan_and_gate(binding, "qwen", gate_mode="OPEN")
        _, body, _ = self._run_e30_mocked(binding, draws, features, heads, matched, plan, gate_ref, "qwen")
        self.assertTrue(body["off_distribution_collapse"]["flagged"])
        self.assertEqual(body["off_distribution_collapse"]["drop_threshold"], COLLAPSE_DROP_THRESHOLD)

    def test_explicit_ref_resume_and_tamper_rejection(self):
        binding, draws, features, heads, matched = _build_synthetic_e30_world(self.root, self.protocol)
        plan, gate_ref = self._prepare_plan_and_gate(binding, "qwen", gate_mode="OPEN")
        analysis_ref, body, crossed_refs = self._run_e30_mocked(
            binding, draws, features, heads, matched, plan, gate_ref, "qwen"
        )

        flat_crossed = {
            f"{arm}:{lang}": crossed_refs[arm][lang] for arm in CROSSED_ARMS for lang in LANGUAGES
        }

        # Resume full analysis ref
        res_ref, res_body, res_crossed = self._run_e30_mocked(
            binding,
            draws,
            features,
            heads,
            matched,
            plan,
            gate_ref,
            "qwen",
            resume_analysis_ref=analysis_ref,
            resume_crossed_refs=flat_crossed,
        )
        self.assertEqual(plain(res_ref), plain(analysis_ref))
        self.assertEqual(plain(res_crossed), plain(crossed_refs))

        # Resume crossed blocks only -> publishes new analysis reusing all 8 crossed blocks
        res2_ref, _, res2_crossed = self._run_e30_mocked(
            binding,
            draws,
            features,
            heads,
            matched,
            plan,
            gate_ref,
            "qwen",
            resume_crossed_refs=flat_crossed,
        )
        self.assertEqual(plain(res2_crossed), plain(crossed_refs))
        self.assertNotEqual(plain(res2_ref)["path"], plain(analysis_ref)["path"])

        # Tamper a crossed block payload -> resume must reject with ContractError
        tampered_path = self.root / crossed_refs["head_last_on_mean"]["en"]["path"]
        logits_file = tampered_path.parent / "logits.npy"
        logits_file.write_bytes(logits_file.read_bytes() + b"\x00")
        with self.assertRaises(ContractError):
            self._run_e30_mocked(
                binding,
                draws,
                features,
                heads,
                matched,
                plan,
                gate_ref,
                "qwen",
                resume_crossed_refs=flat_crossed,
            )


if __name__ == "__main__":
    unittest.main()
