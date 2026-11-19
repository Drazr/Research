"""Unit and fixture tests for Phase 2 Stage 3 E11 CPU worker and notebook."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.common.contracts import ContractError, LANGUAGES, fingerprint
from src.phase2.gates import evaluate_e29_gates, publish_e29_gate
from src.phase2.interfaces import (
    CONDITIONS,
    KIND_ANALYSIS,
    KIND_PREDICTION,
    POOLINGS,
    Phase1Binding,
    SCHEMA_ANALYSIS,
    SCHEMA_PREDICTION,
    frozen,
    init_phase2_dirs,
    load_phase2_artifact,
    plain,
)
from src.phase2.plan import prepare_phase2
from src.stage2.schema import REPO

try:
    import numpy as np
    import scipy  # noqa: F401
    from src.phase2.run_e11 import (
        EXPECTED_FIT_IMAGES,
        MEAN_SHIFT_LOGIT_OFFSET_STD_TOL,
        MEAN_SHIFT_RANK_TOL,
        SUPERVISION_ADAPTED,
        VARIANCE_FLOOR,
        _assert_mean_shift_invariance,
        main as e11_main,
        run_e11,
    )
    from src.stage3.engine import transform
    from src.stage4.predictions import Prediction

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def _fake_producer():
    return {
        "code_commit": "0" * 40,
        "storage_scope": "local-development",
        "software_versions": {"python": "3.12.14", "numpy": "2.1.3"},
    }


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
                    "routes": {src: f"{model}:{cond}:{pool}:{src}->{lang}" for src in LANGUAGES},
                }
    return out


def _fake_e11_refs(model, *, dimension=8):
    depth = 18 if model == "qwen" else 17
    out = {}
    for lang in LANGUAGES:
        out[lang] = {
            "group_id": f"grp-{model}-image_caption-caption_mean-{lang}",
            "selection_ref": {
                "path": f"checkpoints/stage3/{model}_{lang}_sel/_COMPLETE.json",
                "sha256": fingerprint(["sel", model, lang]),
            },
            "candidate_ref": {
                "path": f"checkpoints/stage3/{model}_{lang}_cand/_COMPLETE.json",
                "sha256": fingerprint(["cand", model, lang]),
            },
            "C": 1.0,
            "statistics_ref": {
                "path": f"cache/stage3/{model}_{lang}_stats/_COMPLETE.json",
                "sha256": fingerprint(["e11_stats", model, lang]),
            },
            "dimension": dimension,
            "site": f"decoder:{depth}:caption_mean",
            "model_plan_ref": {
                "path": f"shared/mp_{model}/_COMPLETE.json",
                "sha256": fingerprint(["mp", model]),
            },
            "screen_block_id": f"{model}-image_caption-caption_mean-{lang}",
            "screen_prediction_ref": {
                "path": f"predictions/stage4/{model}/image_caption_caption_mean_{lang}/_COMPLETE.json",
                "sha256": fingerprint([model, "image_caption", "caption_mean", lang]),
            },
            "screen_routes": {
                src: f"{model}:image_caption:caption_mean:{src}->{lang}" for src in LANGUAGES
            },
            "screen_feature_release_ref": {
                "path": f"features/{model}/screen_caption_mean_{lang}/_COMPLETE.json",
                "sha256": fingerprint(["feat_screen", model, lang]),
            },
        }
    return out


def _make_e11_fixture(
    root,
    protocol,
    model="qwen",
    *,
    n_screen=48,
    n_draws=200,
    dimension=8,
    n_fit_images=EXPECTED_FIT_IMAGES,
    unestimable_cat=None,
    seed=20261120,
):
    """Build a complete synthetic E11 fixture (binding, FIT stats, heads, SCREEN features & Stage 4 logits)."""
    categories = list(protocol["category_order"])
    n_cats = len(categories)
    screen_ids = [f"scr-{i:04d}" for i in range(n_screen)]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_screen, size=(n_draws, n_screen), dtype=np.int64)
    bootstrap_sha256 = hashlib.sha256(draws.tobytes()).hexdigest()

    # Latent clean signal per SCREEN image so linear probes achieve realistic high AUROC
    # while covariate shift across languages creates a real transferable gap.
    latent = rng.normal(0.0, 1.0, size=(n_screen, dimension)).astype(np.float64)
    # Keep coordinate 0 near-constant to exercise the 1e-12 variance floor in mean_scale!
    latent[:, 0] = 0.0

    # Generate binary targets from latent coordinates (with balanced pos/neg counts)
    # and preserve unknown cell on scr-0000 for plant.n.02.
    true_w = rng.normal(0.0, 1.0, size=(n_cats, dimension)).astype(np.float64)
    true_w[:, 0] = 0.0
    scores_clean = latent @ true_w.T
    targets = {}
    for i, iid in enumerate(screen_ids):
        vals = [int(scores_clean[i, c] >= np.median(scores_clean[:, c])) for c in range(n_cats)]
        if i == 0 and "plant.n.02" in categories:
            vals[categories.index("plant.n.02")] = None
        targets[iid] = {"split": "SCREEN", "values": vals}

    fit_dict = {"targets": targets}

    class _PlanObj(dict):
        pass

    plan_obj = _PlanObj({
        "screen_ids": list(screen_ids),
        "categories": list(categories),
        "fit": fit_dict,
    })
    plan_obj.fit = fit_dict
    plan_obj.store = None

    dataset_fp = fingerprint(["e11-fixture-dataset", model, n_screen, n_draws])
    plan_obj.reference = frozen({
        "path": "results/stage4/plan/_COMPLETE.json",
        "sha256": fingerprint(["stage4-plan", dataset_fp]),
    })
    stage4_rel = {
        "path": "results/stage4/releases/fixture/_COMPLETE.json",
        "sha256": fingerprint(["stage4-rel", dataset_fp]),
    }
    binding = Phase1Binding(
        root=root,
        protocol=frozen(protocol),
        stage4_release_ref=frozen(stage4_rel),
        stage4_release=frozen({"predictions": {}}),
        stage4_plan=plan_obj,
        bootstrap_shape=(n_draws, n_screen),
        bootstrap_sha256=bootstrap_sha256,
        midpoints=frozen({"qwen": 18, "gemma": 17}),
        dataset_ref=frozen({"path": "shared/ds/_COMPLETE.json", "sha256": dataset_fp}),
        data_plan_ref=frozen({"path": "shared/dp/_COMPLETE.json", "sha256": "a" * 64}),
        model_plan_refs=frozen({
            "qwen": {"path": "shared/mp_q/_COMPLETE.json", "sha256": "b" * 64},
            "gemma": {"path": "shared/mp_g/_COMPLETE.json", "sha256": "c" * 64},
        }),
        suite_ref=frozen({"path": "results/suite/_COMPLETE.json", "sha256": "d" * 64}),
        stage3_registry_ref=frozen({"path": "results/s3/_COMPLETE.json", "sha256": "e" * 64}),
        g0_gate_ref=frozen({"path": "results/g0/_COMPLETE.json", "sha256": "f" * 64}),
        g0_p_value=0.362,
        dataset_fingerprint=dataset_fp,
        protocol_fingerprint=fingerprint(protocol),
        lineage_identity=fingerprint(["e11-lineage", dataset_fp, bootstrap_sha256]),
    )

    e11_refs = _fake_e11_refs(model, dimension=dimension)

    # Paired FIT metadata rows (zero FIT .npy matrices!)
    fit_ids = [f"fit-{i:04d}" for i in range(n_fit_images)]
    fit_rows_by_gid = {
        e11_refs[lang]["group_id"]: [
            {"image_id": iid, "split": "FIT", "language": lang, "row_index": idx}
            for idx, iid in enumerate(fit_ids)
        ]
        for lang in LANGUAGES
    }

    # Language-specific affine covariate shifts (mean + anisotropic diagonal scale).
    # Coordinate 0 has variance 0.0 <= 1e-12 -> scale = 1.0 (variance floor test).
    lang_shifts = {
        "en": np.array([0.0, 0.0, 0.2, -0.1, 0.3, -0.2, 0.1, 0.0], dtype=np.float64),
        "hi": np.array([0.0, 1.5, -1.0, 0.8, -0.9, 1.2, -0.7, 0.6], dtype=np.float64),
        "bn": np.array([0.0, -1.2, 1.3, -0.7, 1.1, -0.8, 0.9, -1.0], dtype=np.float64),
        "te": np.array([0.0, 0.7, 0.9, -1.4, -0.5, 0.6, 1.3, -0.8], dtype=np.float64),
    }
    lang_scales = {
        "en": np.array([1.0, 1.0, 1.2, 0.8, 1.1, 0.9, 1.0, 1.3], dtype=np.float64),
        "hi": np.array([1.0, 2.8, 0.4, 2.2, 0.5, 2.5, 0.45, 1.9], dtype=np.float64),
        "bn": np.array([1.0, 0.35, 2.6, 0.5, 2.4, 0.4, 2.3, 0.6], dtype=np.float64),
        "te": np.array([1.0, 2.1, 0.55, 2.7, 0.4, 1.8, 0.5, 2.5], dtype=np.float64),
    }

    stats_by_gid = {}
    heads_by_lang = {}
    screen_features_by_lang = {}
    screen_rows_by_lang = {}

    estimable = np.ones((n_cats,), dtype=bool)
    if unestimable_cat is not None and unestimable_cat in categories:
        estimable[categories.index(unestimable_cat)] = False
    prevalence = np.full((n_cats,), 0.5, dtype=np.float64)

    for lang in LANGUAGES:
        mu = lang_shifts[lang]
        sigma = lang_scales[lang]
        var = sigma ** 2
        var[0] = 5e-13  # Below 1e-12 variance floor -> scale[0] MUST be 1.0
        scale = np.where(var <= VARIANCE_FLOOR, 1.0, np.sqrt(var))

        gid = e11_refs[lang]["group_id"]
        stats = {
            "mean": mu.copy(),
            "variance": var.copy(),
            "scale": scale.copy(),
            "center": True,
            "reference": dict(e11_refs[lang]["statistics_ref"]),
        }
        stats_by_gid[gid] = stats

        # SCREEN features H_t = mu_t + sigma_t * latent + small noise
        noise = rng.normal(0.0, 0.12, size=(n_screen, dimension)).astype(np.float64)
        noise[:, 0] = 0.0
        H_t = mu[None, :] + sigma[None, :] * (latent + noise)
        screen_features_by_lang[lang] = H_t.astype(np.float32)
        screen_rows_by_lang[lang] = [
            {"image_id": iid, "split": "SCREEN", "language": lang, "row_index": idx}
            for idx, iid in enumerate(screen_ids)
        ]

        # Frozen head aligned with standardized latent features (realistic logit magnitude)
        coef = (true_w * 0.45).astype(np.float64)
        intercept = np.zeros((n_cats,), dtype=np.float64)
        if not estimable.all():
            coef[~estimable] = 0.0
            intercept[~estimable] = 0.0
        heads_by_lang[lang] = (coef, intercept, estimable.copy(), prevalence.copy(), stats)

    # Compute exact Stage 4 SCREEN prediction blocks via transform(H_t, stats_s) @ coef_s.T + intercept_s
    stage4_blocks_by_id = {}
    for t_lang in LANGUAGES:
        rec = e11_refs[t_lang]
        block_id = rec["screen_block_id"]
        H_t = screen_features_by_lang[t_lang]
        rows_t = tuple(screen_rows_by_lang[t_lang])
        preds = {}
        for s_lang in LANGUAGES:
            rid = rec["screen_routes"][s_lang]
            coef_s, intercept_s, est_s, prev_s, stats_s = heads_by_lang[s_lang]
            logits = np.asarray(transform(H_t, stats_s) @ coef_s.T, dtype=np.float64) + intercept_s
            preds[rid] = Prediction(
                reference=dict(rec["screen_prediction_ref"]),
                route={"route_id": rid, "source_group": e11_refs[s_lang]["group_id"], "target_group": rec["group_id"]},
                split="SCREEN",
                rows=rows_t,
                logits=logits,
                estimable=est_s.copy(),
                prevalence=prev_s.copy(),
            )
        stage4_blocks_by_id[block_id] = preds

    return {
        "binding": binding,
        "draws": draws,
        "e11_refs": e11_refs,
        "fit_rows_by_gid": fit_rows_by_gid,
        "stats_by_gid": stats_by_gid,
        "heads_by_lang": heads_by_lang,
        "screen_features_by_lang": screen_features_by_lang,
        "screen_rows_by_lang": screen_rows_by_lang,
        "stage4_blocks_by_id": stage4_blocks_by_id,
    }


def _publish_plan_and_gate(root, protocol, fixture, model, *, gate_mode="OPEN"):
    """Publish a verified phase2_plan and phase2_e29_gate under root."""
    init_phase2_dirs(root)
    producer = _fake_producer()
    binding = fixture["binding"]

    def _fake_probe(r, proto, dfp, prod):
        from src.phase2.interfaces import KIND_PROBE, KIND_PROBE_PAYLOAD, SCHEMA_PROBE, publish_phase2

        payload_ref = publish_phase2(
            r,
            proto,
            dfp,
            "phase2/probes",
            KIND_PROBE_PAYLOAD,
            {"schema": SCHEMA_PROBE, "nonce": "fixture-nonce", "remote_sync_guarantee": False},
            prod,
        )
        return publish_phase2(
            r,
            proto,
            dfp,
            "phase2/probes",
            KIND_PROBE,
            {
                "schema": SCHEMA_PROBE,
                "payload": plain(payload_ref),
                "independent_process_readback": True,
                "reader_pid": 1001,
                "writer_pid": 1002,
                "remote_sync_guarantee": False,
            },
            prod,
            upstream=[plain(payload_ref)],
        )

    with patch("src.phase2.plan.bind_phase1", return_value=binding), \
         patch("src.phase2.plan.storage_probe", side_effect=_fake_probe), \
         patch("src.phase2.plan.e29_screen_blocks", side_effect=lambda b, m: frozen(_fake_e29_blocks(m))), \
         patch(
             "src.phase2.plan.e11_fit_statistics_refs",
             side_effect=lambda b, m: frozen(fixture["e11_refs"] if m == model else _fake_e11_refs(m)),
         ):
        plan = prepare_phase2(root, protocol, plain(binding.stage4_release_ref), producer)
        plan_ref = plan.reference

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
            delta_g_mean=0.025,
            delta_g_ci=[0.010, 0.040],
            g_res_mean=0.025,
            g_res_ci=[0.010, 0.040],
            route_residuals=routes_low,
        )
    elif gate_mode == "PARK":
        ev = evaluate_e29_gates(
            model,
            diagonals=diags,
            valid_draws=2000,
            delta_g_mean=0.004,
            delta_g_ci=[0.001, 0.009],
            g_res_mean=0.004,
            g_res_ci=[0.001, 0.009],
            route_residuals=routes_low,
        )
    elif gate_mode == "UNRESOLVED":
        ev = evaluate_e29_gates(
            model,
            diagonals=diags,
            valid_draws=1800,
            delta_g_mean=0.025,
            delta_g_ci=[0.010, 0.040],
            g_res_mean=0.025,
            g_res_ci=[0.010, 0.040],
            route_residuals=routes_low,
        )
    else:
        raise ValueError(gate_mode)

    fake_e29_analysis_ref = {
        "path": f"phase2/results/e29/{model}/fake/_COMPLETE.json",
        "sha256": fingerprint(["e29-analysis", model, gate_mode]),
    }
    gate_ref = publish_e29_gate(
        root,
        protocol,
        plan,
        model,
        fake_e29_analysis_ref,
        ev,
        producer,
    )
    return plain(plan_ref), plain(gate_ref), plan


class E11NotebookTests(unittest.TestCase):
    def test_notebook_structure_and_code_ref(self):
        nb_path = REPO / "scripts/phase2/run_e11.ipynb"
        self.assertTrue(nb_path.is_file(), "Missing scripts/phase2/run_e11.ipynb")
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertEqual(nb.get("nbformat"), 4)
        sources = "\n".join("".join(c.get("source", [])) for c in nb.get("cells", []))
        self.assertIn("CODE_REF = 'phase2'", sources)
        self.assertIn("src.phase2.run_e11", sources)
        self.assertIn("MODEL", sources)
        self.assertIn("PLAN_REF", sources)
        self.assertIn("E29_GATE_REF", sources)
        self.assertIn("RESUME_ADAPTED_REFS", sources)
        self.assertIn("RESUME_ANALYSIS_REF", sources)
        self.assertIn("OPENBLAS_NUM_THREADS", sources)
        self.assertIn("requirements-stage2-cpu.txt", sources)


@unittest.skipUnless(HAS_NUMPY, "numpy/scipy required for E11 numeric engine tests")
class E11WorkerFixtureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.protocol = json.loads((REPO / "configs/protocol.json").read_text(encoding="utf-8"))
        # Lower SCREEN minimum support slightly for fast 48-image synthetic fixtures
        self.protocol["minimum_positive_and_negative"]["SCREEN"] = 10

    def _patch_e11_io(self, fixture, *, model="qwen"):
        """Return a context manager patching Phase 1 read-only loaders for E11."""
        from contextlib import ExitStack

        stack = ExitStack()
        stack.enter_context(patch("src.phase2.run_e11.bind_phase1", return_value=fixture["binding"]))
        stack.enter_context(
            patch(
                "src.phase2.run_e11.e11_fit_statistics_refs",
                side_effect=lambda b, m: frozen(fixture["e11_refs"]),
            )
        )
        stack.enter_context(
            patch(
                "src.phase2.run_e11.expected_rows",
                side_effect=lambda fit, gid, split, store: fixture["fit_rows_by_gid"][gid],
            )
        )
        stack.enter_context(
            patch(
                "src.phase2.run_e11.load_statistics",
                side_effect=lambda fit, ref, gid, rows, store: fixture["stats_by_gid"][gid],
            )
        )
        stack.enter_context(
            patch(
                "src.phase2.run_e11.load_frozen_source_head",
                side_effect=lambda b, m, cond, pool, lang: fixture["heads_by_lang"][lang],
            )
        )
        stack.enter_context(
            patch(
                "src.phase2.run_e11.read_verified_midpoint_features",
                side_effect=lambda b, m, cond, pool, lang, split: (
                    fixture["screen_rows_by_lang"][lang],
                    fixture["screen_features_by_lang"][lang],
                ),
            )
        )
        stack.enter_context(
            patch(
                "src.phase2.run_e11.load_prediction_block",
                side_effect=lambda plan, ref, expected_block=None: fixture["stage4_blocks_by_id"][expected_block],
            )
        )
        stack.enter_context(
            patch("src.phase2.run_e11.load_bootstrap", return_value=fixture["draws"])
        )
        return stack

    def test_refuses_closed_park_and_unresolved_gates(self):
        fixture = _make_e11_fixture(self.root, self.protocol, model="qwen")
        for mode in ("PARK", "UNRESOLVED"):
            plan_ref, gate_ref, _ = _publish_plan_and_gate(
                self.root, self.protocol, fixture, "qwen", gate_mode=mode
            )
            with self._patch_e11_io(fixture, model="qwen"):
                with self.assertRaises(ContractError) as ctx:
                    run_e11(self.root, self.protocol, plan_ref, gate_ref, "qwen", _fake_producer())
                self.assertIn("gate is closed", str(ctx.exception))

        # Foreign-model gate must also be refused
        plan_ref, qwen_gate_ref, _ = _publish_plan_and_gate(
            self.root, self.protocol, fixture, "qwen", gate_mode="OPEN"
        )
        with self._patch_e11_io(fixture, model="gemma"):
            with self.assertRaises(ContractError):
                run_e11(self.root, self.protocol, plan_ref, qwen_gate_ref, "gemma", _fake_producer())

    def test_end_to_end_e11_happy_path_and_invariance(self):
        primary_set = set(self.protocol["primary_categories"])
        unest_cat = next(c for c in self.protocol["category_order"] if c not in primary_set)
        fixture = _make_e11_fixture(
            self.root,
            self.protocol,
            model="qwen",
            unestimable_cat=unest_cat,
        )
        plan_ref, gate_ref, _ = _publish_plan_and_gate(
            self.root, self.protocol, fixture, "qwen", gate_mode="OPEN"
        )

        events = []
        with self._patch_e11_io(fixture, model="qwen"):
            result = plain(
                run_e11(
                    self.root,
                    self.protocol,
                    plan_ref,
                    gate_ref,
                    "qwen",
                    _fake_producer(),
                    progress=events.append,
                )
            )

        self.assertFalse(result["resumed"])
        self.assertEqual(result["model"], "qwen")
        self.assertEqual(result["fit_pairing_check"]["paired_fit_images"], EXPECTED_FIT_IMAGES)
        self.assertEqual(result["fit_pairing_check"]["fit_matrices_reloaded"], 0)
        self.assertEqual(result["fit_pairing_check"]["variance_floor"], VARIANCE_FLOOR)
        self.assertTrue(result["identity_self_check"]["passed"])
        self.assertEqual(result["identity_self_check"]["max_abs_diff"], 0.0)

        # Invariance checks must pass strictly
        inv = result["invariance_checks"]
        self.assertTrue(inv["passed"])
        self.assertLessEqual(inv["mean_shift_max_abs_auroc_diff"], MEAN_SHIFT_RANK_TOL)
        self.assertLessEqual(inv["mean_shift_max_abs_ap_diff"], MEAN_SHIFT_RANK_TOL)
        self.assertLessEqual(inv["mean_shift_logit_offset_max_std"], MEAN_SHIFT_LOGIT_OFFSET_STD_TOL)

        # Verify all 8 adapted prediction blocks
        adapted_refs = result["adapted_block_refs"]
        self.assertEqual(set(adapted_refs.keys()), {"mean_shift", "mean_scale"})
        for variant in ("mean_shift", "mean_scale"):
            self.assertEqual(set(adapted_refs[variant].keys()), set(LANGUAGES))
            for lang in LANGUAGES:
                _, body, _ = load_phase2_artifact(
                    self.root,
                    self.protocol,
                    adapted_refs[variant][lang],
                    KIND_PREDICTION,
                    expected_schema=SCHEMA_PREDICTION,
                    expected_owner="qwen",
                )
                self.assertEqual(body["experiment"], "E11")
                self.assertEqual(body["variant"], variant)
                self.assertEqual(body["target_language"], lang)
                self.assertEqual(len(body["routes"]), 3)
                self.assertTrue(body["confirm_excluded"])
                for r in body["routes"]:
                    self.assertEqual(r["supervision_label"], SUPERVISION_ADAPTED)
                    self.assertNotEqual(r["source_language"], lang)

        # Verify analysis block schema & content
        _, analysis_body, _ = load_phase2_artifact(
            self.root,
            self.protocol,
            result["analysis_ref"],
            KIND_ANALYSIS,
            expected_schema=SCHEMA_ANALYSIS,
            expected_owner="qwen",
        )
        self.assertEqual(analysis_body["experiment"], "E11")
        self.assertEqual(analysis_body["condition"], "image_caption")
        self.assertEqual(analysis_body["pooling"], "caption_mean")
        self.assertTrue(analysis_body["confirm_excluded"])
        self.assertIn("caption_robustness_note", analysis_body)
        self.assertEqual(set(analysis_body["identity_block_refs"].keys()), set(LANGUAGES))
        self.assertEqual(set(analysis_body["fit_statistics_refs"].keys()), set(LANGUAGES))
        self.assertEqual(
            set(analysis_body["arms"].keys()),
            {"identity", "mean_shift", "mean_scale", "target_diagonal_ref"},
        )

        # Unknown cell mask preserved on plant.n.02 (known = n_screen - 1 = 47)
        if "plant.n.02" in self.protocol["category_order"]:
            en_diag_plant = analysis_body["arms"]["target_diagonal_ref"]["metrics"]["auroc"]["per_route"]["en->en"][
                "per_category"
            ]["plant.n.02"]
            self.assertEqual(en_diag_plant["support"]["known"], 47)

        # Unestimable category is marked undefined
        unest_rec = analysis_body["arms"]["identity"]["metrics"]["auroc"]["per_route"]["hi->en"]["absolute"][
            "per_category"
        ][unest_cat]
        self.assertEqual(unest_rec["status"], "undefined")
        self.assertIsNone(unest_rec["value"])

        # mean_scale should recover positive AUROC gap on this anisotropic-scale synthetic fixture
        scale_auroc_gain = analysis_body["summaries"]["paired_gains_vs_identity"]["mean_scale"]["auroc"][
            "twelve_route_mean_gap_reduction"
        ]["value"]
        self.assertIsNotNone(scale_auroc_gain)
        self.assertGreater(scale_auroc_gain, 0.0)

        # mean_shift AUROC gap reduction must be ~0 (<= 1e-12)
        shift_auroc_gain = analysis_body["summaries"]["paired_gains_vs_identity"]["mean_shift"]["auroc"][
            "twelve_route_mean_gap_reduction"
        ]["value"]
        self.assertLessEqual(abs(shift_auroc_gain), MEAN_SHIFT_RANK_TOL)

        # Streamed progress events present
        self.assertGreaterEqual(len(events), 10)
        for ev in events:
            for k in ("event", "unit", "completed", "total", "read_hash_s", "compute_s", "write_s"):
                self.assertIn(k, ev)

    def test_fit_pairing_and_variance_floor_failures(self):
        # 1. Misaligned FIT image_id ordering across languages
        fixture = _make_e11_fixture(self.root, self.protocol, model="qwen")
        plan_ref, gate_ref, _ = _publish_plan_and_gate(
            self.root, self.protocol, fixture, "qwen", gate_mode="OPEN"
        )
        hi_gid = fixture["e11_refs"]["hi"]["group_id"]
        fixture["fit_rows_by_gid"][hi_gid] = list(reversed(fixture["fit_rows_by_gid"][hi_gid]))
        with self._patch_e11_io(fixture, model="qwen"):
            with self.assertRaises(ContractError) as ctx:
                run_e11(self.root, self.protocol, plan_ref, gate_ref, "qwen", _fake_producer())
            self.assertIn("Unpaired FIT image_id", str(ctx.exception))

        # 2. Wrong FIT row count (2159 != 2160)
        fixture2 = _make_e11_fixture(self.root, self.protocol, model="qwen", n_fit_images=2159)
        plan_ref2, gate_ref2, _ = _publish_plan_and_gate(
            self.root, self.protocol, fixture2, "qwen", gate_mode="OPEN"
        )
        with self._patch_e11_io(fixture2, model="qwen"):
            with self.assertRaises(ContractError) as ctx:
                run_e11(self.root, self.protocol, plan_ref2, gate_ref2, "qwen", _fake_producer())
            self.assertIn("2160", str(ctx.exception))

        # 3. Variance floor violation (scale != 1.0 when variance <= 1e-12)
        fixture3 = _make_e11_fixture(self.root, self.protocol, model="qwen")
        plan_ref3, gate_ref3, _ = _publish_plan_and_gate(
            self.root, self.protocol, fixture3, "qwen", gate_mode="OPEN"
        )
        en_gid = fixture3["e11_refs"]["en"]["group_id"]
        fixture3["stats_by_gid"][en_gid]["scale"][0] = 1e-6  # Violates 1e-12 floor -> 1.0 rule
        with self._patch_e11_io(fixture3, model="qwen"):
            with self.assertRaises(ContractError) as ctx:
                run_e11(self.root, self.protocol, plan_ref3, gate_ref3, "qwen", _fake_producer())
            self.assertIn("variance floor", str(ctx.exception))

    def test_en_diagonal_self_check_failure(self):
        fixture = _make_e11_fixture(self.root, self.protocol, model="gemma")
        plan_ref, gate_ref, _ = _publish_plan_and_gate(
            self.root, self.protocol, fixture, "gemma", gate_mode="OPEN"
        )
        en_bid = fixture["e11_refs"]["en"]["screen_block_id"]
        en_rid = fixture["e11_refs"]["en"]["screen_routes"]["en"]
        orig = fixture["stage4_blocks_by_id"][en_bid][en_rid]
        perturbed = orig.logits.copy()
        perturbed[0, 0] += 1e-9
        fixture["stage4_blocks_by_id"][en_bid][en_rid] = Prediction(
            reference=orig.reference,
            route=orig.route,
            split=orig.split,
            rows=orig.rows,
            logits=perturbed,
            estimable=orig.estimable,
            prevalence=orig.prevalence,
        )
        with self._patch_e11_io(fixture, model="gemma"):
            with self.assertRaises(ContractError) as ctx:
                run_e11(self.root, self.protocol, plan_ref, gate_ref, "gemma", _fake_producer())
            self.assertIn("diagonal self-check failed", str(ctx.exception))

    def test_mean_shift_invariance_hard_failure(self):
        categories = list(self.protocol["category_order"])
        n_cats = len(categories)
        n_imgs = 10
        align = {t: np.arange(n_imgs, dtype=np.intp) for t in LANGUAGES}
        id_arms = {}
        sh_arms = {}
        id_logits = {}
        sh_logits = {}
        for s in LANGUAGES:
            for t in LANGUAGES:
                if s == t:
                    continue
                id_arms[(s, t)] = {
                    "points": {"auroc": [0.8] * n_cats, "ap": [0.75] * n_cats},
                    "unestimable": [False] * n_cats,
                }
                sh_arms[(s, t)] = {
                    "points": {"auroc": [0.8] * n_cats, "ap": [0.75] * n_cats},
                    "unestimable": [False] * n_cats,
                }
                base_z = np.zeros((n_imgs, n_cats), dtype=np.float64)
                id_logits[(s, t)] = base_z
                sh_logits[(s, t)] = base_z + 0.5  # exact constant offset

        # 1. AUROC violation (> 1e-12) must hard-fail
        sh_arms[("hi", "en")]["points"]["auroc"][0] = 0.8 + 1e-9
        with self.assertRaises(ContractError) as ctx:
            _assert_mean_shift_invariance(
                identity_arms=id_arms,
                mean_shift_arms=sh_arms,
                identity_logits_by_route=id_logits,
                mean_shift_logits_by_route=sh_logits,
                aligned_by_target=align,
                categories=categories,
            )
        self.assertIn("AUROC invariance violated", str(ctx.exception))

        # 2. Logit offset std violation (> 1e-10) must hard-fail
        sh_arms[("hi", "en")]["points"]["auroc"][0] = 0.8
        bad_z = sh_logits[("hi", "en")].copy()
        bad_z[0, 0] += 1e-6
        sh_logits[("hi", "en")] = bad_z
        with self.assertRaises(ContractError) as ctx:
            _assert_mean_shift_invariance(
                identity_arms=id_arms,
                mean_shift_arms=sh_arms,
                identity_logits_by_route=id_logits,
                mean_shift_logits_by_route=sh_logits,
                aligned_by_target=align,
                categories=categories,
            )
        self.assertIn("logit offset not constant", str(ctx.exception))

    def test_explicit_ref_resume_and_rejection(self):
        fixture = _make_e11_fixture(self.root, self.protocol, model="qwen")
        plan_ref, gate_ref, _ = _publish_plan_and_gate(
            self.root, self.protocol, fixture, "qwen", gate_mode="OPEN"
        )
        with self._patch_e11_io(fixture, model="qwen"):
            first = plain(
                run_e11(self.root, self.protocol, plan_ref, gate_ref, "qwen", _fake_producer())
            )

            # 1. Partial resume of 2 adapted blocks succeeds
            partial_res = plain(
                run_e11(
                    self.root,
                    self.protocol,
                    plan_ref,
                    gate_ref,
                    "qwen",
                    _fake_producer(),
                    resume_adapted_refs={
                        "mean_shift:en": first["adapted_block_refs"]["mean_shift"]["en"],
                        "mean_scale:hi": first["adapted_block_refs"]["mean_scale"]["hi"],
                    },
                )
            )
            self.assertEqual(
                partial_res["adapted_block_refs"]["mean_shift"]["en"],
                first["adapted_block_refs"]["mean_shift"]["en"],
            )
            self.assertEqual(
                partial_res["adapted_block_refs"]["mean_scale"]["hi"],
                first["adapted_block_refs"]["mean_scale"]["hi"],
            )

            # 2. Full resume via resume_analysis_ref succeeds
            full_res = plain(
                run_e11(
                    self.root,
                    self.protocol,
                    plan_ref,
                    gate_ref,
                    "qwen",
                    _fake_producer(),
                    resume_analysis_ref=first["analysis_ref"],
                )
            )
            self.assertTrue(full_res["resumed"])
            self.assertEqual(full_res["analysis_ref"], first["analysis_ref"])

            # 3. Unknown resume_adapted_refs key rejected
            with self.assertRaises(ContractError):
                run_e11(
                    self.root,
                    self.protocol,
                    plan_ref,
                    gate_ref,
                    "qwen",
                    _fake_producer(),
                    resume_adapted_refs={"bad_variant:en": first["adapted_block_refs"]["mean_shift"]["en"]},
                )

            # 4. Wrong variant passed to resume_adapted_refs rejected
            with self.assertRaises(ContractError):
                run_e11(
                    self.root,
                    self.protocol,
                    plan_ref,
                    gate_ref,
                    "qwen",
                    _fake_producer(),
                    resume_adapted_refs={"mean_scale:en": first["adapted_block_refs"]["mean_shift"]["en"]},
                )

            # 5. Tampered checksum rejected
            bad_ref = dict(first["analysis_ref"], sha256="f" * 64)
            with self.assertRaises(ContractError):
                run_e11(
                    self.root,
                    self.protocol,
                    plan_ref,
                    gate_ref,
                    "qwen",
                    _fake_producer(),
                    resume_analysis_ref=bad_ref,
                )

    def test_cli_smoke(self):
        fixture = _make_e11_fixture(self.root, self.protocol, model="qwen")
        plan_ref, gate_ref, _ = _publish_plan_and_gate(
            self.root, self.protocol, fixture, "qwen", gate_mode="OPEN"
        )
        proto_path = self.root / "protocol_test.json"
        proto_path.write_text(json.dumps(self.protocol), encoding="utf-8")
        out_path = self.root / "e11_out.json"
        with self._patch_e11_io(fixture, model="qwen"), \
             patch("src.phase2.run_e11.current_producer", return_value=_fake_producer()):
            rc = e11_main([
                "--phase1-root", str(self.root),
                "--protocol", str(proto_path),
                "--plan", json.dumps(plan_ref),
                "--e29-gate", json.dumps(gate_ref),
                "--model", "qwen",
                "--storage-scope", "local-development",
                "--output-ref", str(out_path),
            ])
        self.assertEqual(rc, 0)
        self.assertTrue(out_path.is_file())
        emitted = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertEqual(emitted["event"], "e11_complete")
        self.assertEqual(emitted["status"], "passed")
        self.assertEqual(emitted["model"], "qwen")


if __name__ == "__main__":
    unittest.main()
