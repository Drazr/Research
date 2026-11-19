"""Unit and fixture tests for Phase 2 Stage 2 E29 CPU worker and notebook."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.common.contracts import LANGUAGES, ContractError, fingerprint
from src.phase2.gates import evaluate_e29_gates, load_e29_gate
from src.phase2.interfaces import (
    CONDITIONS,
    KIND_ANALYSIS,
    MODELS,
    PHASE2_PARENT_DIRS,
    POOLINGS,
    Phase1Binding,
    SCHEMA_ANALYSIS,
    frozen,
    plain,
)
from src.phase2.plan import load_phase2_plan, prepare_phase2
from src.stage2.schema import REPO

try:
    import numpy as np
    import scipy  # noqa: F401
    from src.phase2.run_e29 import main as e29_main, run_e29
    from src.stage4.predictions import Prediction

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


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


def _make_mock_binding(root, protocol, *, n_images=40, n_draws=2000, seed=12345):
    """Create a synthetic Phase1Binding + Stage 4 plan dict for E29 fixture tests."""
    categories = tuple(protocol["categories"])
    screen_ids = tuple(f"img-{i:04d}" for i in range(n_images))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_images, size=(n_draws, n_images), dtype=np.int64)
    bootstrap_sha256 = hashlib.sha256(draws.tobytes()).hexdigest()

    # Build synthetic targets: 50% positive per category, with one unknown cell
    # on img-0000 for plant.n.02 (matching SCREEN unknown-cell mask semantics).
    targets = {}
    for i, iid in enumerate(screen_ids):
        vals = [int((i + c) % 2 == 0) for c in range(len(categories))]
        if i == 0 and "plant.n.02" in categories:
            vals[categories.index("plant.n.02")] = None
        targets[iid] = {"split": "SCREEN", "values": vals}

    fit_dict = {"targets": targets}
    stage4_plan = {
        "screen_ids": list(screen_ids),
        "categories": list(categories),
        "fit": fit_dict,
    }

    # Wrap stage4_plan in a lightweight dict subclass with .fit attribute
    class _PlanObj(dict):
        pass

    plan_obj = _PlanObj(stage4_plan)
    plan_obj.fit = fit_dict
    plan_obj.store = None

    dataset_fp = fingerprint(["e29-fixture-dataset", n_images])
    stage4_rel = {
        "path": "results/stage4/releases/fixture/_COMPLETE.json",
        "sha256": fingerprint(["stage4-rel", dataset_fp]),
    }
    return (
        Phase1Binding(
            root=root,
            protocol=frozen(protocol),
            stage4_release_ref=frozen(stage4_rel),
            stage4_release=frozen({"predictions": {}}),
            stage4_plan=plan_obj,
            bootstrap_shape=(n_draws, n_images),
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
            lineage_identity=fingerprint(["e29-lineage", dataset_fp, bootstrap_sha256]),
        ),
        draws,
    )


def _build_predictions_map(binding, model, *, diag_logit=3.0, last_off_logit=0.2, mean_off_logit=2.5, unestimable_cat=None):
    """Build deterministic Prediction objects for all 16 E29 midpoint blocks."""
    categories = binding.stage4_plan["categories"]
    screen_ids = binding.stage4_plan["screen_ids"]
    targets = binding.stage4_plan.fit["targets"]
    n_images = len(screen_ids)
    n_cats = len(categories)

    # Signed labels (+1 / -1) with known=0 mapped to +1 arbitrarily (masked out by known)
    y_sign = np.ones((n_images, n_cats), dtype=np.float64)
    for i, iid in enumerate(screen_ids):
        for c, v in enumerate(targets[iid]["values"]):
            y_sign[i, c] = 1.0 if (v == 1) else -1.0

    rows = tuple({"image_id": iid, "split": "SCREEN", "language": "en"} for iid in screen_ids)
    estimable = np.ones((n_cats,), dtype=bool)
    if unestimable_cat is not None and unestimable_cat in categories:
        estimable[categories.index(unestimable_cat)] = False
    prevalence = np.full((n_cats,), 0.5, dtype=np.float64)

    blocks_meta = _fake_e29_blocks(model)
    by_block_id = {}
    for cond in CONDITIONS:
        for pool in POOLINGS:
            for t_lang in LANGUAGES:
                entry = blocks_meta[cond][pool][t_lang]
                block_id = entry["block_id"]
                route_preds = {}
                for s_lang, rid in entry["routes"].items():
                    if s_lang == t_lang:
                        mag = diag_logit
                    elif pool == "last_caption":
                        mag = last_off_logit
                    else:
                        mag = mean_off_logit
                    logits = y_sign * mag
                    if not estimable.all():
                        logits[:, ~estimable] = 0.0
                    route_preds[rid] = Prediction(
                        reference=dict(entry["prediction_ref"]),
                        route={"route_id": rid, "source_group": f"src-{s_lang}", "target_group": f"tgt-{t_lang}"},
                        split="SCREEN",
                        rows=rows,
                        logits=logits,
                        estimable=estimable.copy(),
                        prevalence=prevalence.copy(),
                    )
                by_block_id[block_id] = route_preds
    return by_block_id


class E29NotebookAndGateTests(unittest.TestCase):
    def test_notebook_structure_and_code_ref(self):
        nb_path = REPO / "scripts/phase2/run_e29.ipynb"
        self.assertTrue(nb_path.is_file(), "Missing scripts/phase2/run_e29.ipynb")
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertEqual(nb.get("nbformat"), 4)
        sources = "\n".join("".join(c.get("source", [])) for c in nb.get("cells", []))
        self.assertIn("CODE_REF = 'phase2'", sources)
        self.assertIn("src.phase2.run_e29", sources)
        self.assertIn("MODEL", sources)
        self.assertIn("OPENBLAS_NUM_THREADS", sources)
        self.assertIn("requirements-stage2-cpu.txt", sources)

    def test_gate_open_park_unresolved_cases(self):
        diags = {lang: 0.85 for lang in LANGUAGES}
        routes_low = {f"{s}->{t}": {"point": 0.005, "ci": [0.001, 0.009]} for s in LANGUAGES for t in LANGUAGES if s != t}
        # OPEN both E30 and E11
        ev_open = evaluate_e29_gates(
            "qwen",
            diagonals=diags,
            valid_draws=1950,
            delta_g_mean=0.025,
            delta_g_ci=[0.010, 0.040],
            g_res_mean=0.020,
            g_res_ci=[0.008, 0.032],
            route_residuals=routes_low,
        )
        self.assertEqual(ev_open["decisions"]["E30"]["decision"], "OPEN")
        self.assertEqual(ev_open["decisions"]["E11"]["decision"], "OPEN")

        # PARK both when upper CI < 0.015
        ev_park = evaluate_e29_gates(
            "gemma",
            diagonals=diags,
            valid_draws=2000,
            delta_g_mean=0.005,
            delta_g_ci=[0.001, 0.010],
            g_res_mean=0.004,
            g_res_ci=[0.000, 0.010],
            route_residuals=routes_low,
        )
        self.assertEqual(ev_park["decisions"]["E30"]["decision"], "PARK")
        self.assertEqual(ev_park["decisions"]["E11"]["decision"], "PARK")

        # UNRESOLVED when valid_draws < 1900
        ev_unres = evaluate_e29_gates(
            "qwen",
            diagonals=diags,
            valid_draws=1899,
            delta_g_mean=0.030,
            delta_g_ci=[0.015, 0.045],
            g_res_mean=0.030,
            g_res_ci=[0.015, 0.045],
            route_residuals=routes_low,
        )
        self.assertEqual(ev_unres["decisions"]["E30"]["decision"], "UNRESOLVED")
        self.assertEqual(ev_unres["decisions"]["E11"]["decision"], "UNRESOLVED")


@unittest.skipUnless(HAS_NUMPY, "numpy/scipy required for E29 numeric engine tests")
class E29WorkerFixtureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.protocol = json.loads((REPO / "configs/protocol.json").read_text(encoding="utf-8"))
        self.producer = {
            "code_commit": "4" * 40,
            "storage_scope": "local-development",
            "software_versions": {"python": "3.12.0", "numpy": "2.1.3", "scipy": "1.18.1"},
        }

    def _prepare_plan(self, binding, draws):
        with (
            patch("src.phase2.plan.bind_phase1", return_value=binding),
            patch("src.phase2.plan.e29_screen_blocks", side_effect=lambda b, m: _fake_e29_blocks(m)),
            patch("src.phase2.plan.e11_fit_statistics_refs", side_effect=lambda b, m: _fake_e11_stats(m)),
        ):
            return prepare_phase2(self.root, self.protocol, binding.stage4_release_ref, self.producer)

    def _run_with_mocks(self, binding, draws, plan, model, by_block_id, **kwargs):
        with (
            patch("src.phase2.run_e29.bind_phase1", return_value=binding),
            patch("src.phase2.run_e29.e29_screen_blocks", side_effect=lambda b, m: _fake_e29_blocks(m)),
            patch("src.phase2.run_e29.load_bootstrap", return_value=draws),
            patch(
                "src.phase2.run_e29.load_prediction_block",
                side_effect=lambda s4plan, ref, expected_block=None: by_block_id[expected_block],
            ),
        ):
            return run_e29(self.root, self.protocol, plan.reference, model, self.producer, **kwargs)

    def test_paired_draw_arithmetic_mask_and_open_gate(self):
        binding, draws = _make_mock_binding(self.root, self.protocol, n_images=40, n_draws=200)
        plan = self._prepare_plan(binding, draws)
        # diag_logit=3.0 (AUROC=1.0), last_off=0.0 (AUROC=0.5 -> G_last=0.5), mean_off=1.0 (AUROC=1.0 -> G_res=0.0)
        #Wait: with sign*mag where mag>0, AUROC is 1.0 even for 0.2! Use negative/mixed logits for imperfect AUROC.
        by_block_id = _build_predictions_map(binding, "qwen", diag_logit=3.0, last_off_logit=0.0, mean_off_logit=0.0)
        # Inject controlled imperfect predictions for last_off (flip 10 rows -> lower AUROC) and mean_off (flip 3 rows -> higher AUROC, still < 1.0)
        for t_lang in LANGUAGES:
            for s_lang in LANGUAGES:
                if s_lang == t_lang:
                    continue
                rid_last = f"qwen:image_caption:last_caption:{s_lang}->{t_lang}"
                p_last = by_block_id[f"qwen-image_caption-last_caption-{t_lang}"][rid_last]
                p_last.logits[:10, :] *= -1.0  # flipped on first 10 images -> G_last > 0
                p_last.logits[10:, :] += 2.0

                rid_mean = f"qwen:image_caption:caption_mean:{s_lang}->{t_lang}"
                p_mean = by_block_id[f"qwen-image_caption-caption_mean-{t_lang}"][rid_mean]
                p_mean.logits[:4, :] *= -1.0  # fewer flips -> G_res smaller than G_last, still > 0
                p_mean.logits[4:, :] += 2.0

        events = []
        analysis_ref, analysis_body, gate_ref, gate_body = self._run_with_mocks(
            binding, draws, plan, "qwen", by_block_id, progress=events.append
        )

        # Verify streamed progress events
        self.assertTrue(any(e["event"] == "e29_published" for e in events))
        self.assertEqual(analysis_body["schema"], SCHEMA_ANALYSIS)
        self.assertEqual(analysis_body["experiment"], "E29")
        self.assertTrue(analysis_body["confirm_excluded"])
        self.assertEqual(analysis_body["midpoint_depth"], 18)

        # Verify unknown-cell mask preserved: plant.n.02 has 39 known images (not 40)
        en_diag = analysis_body["image_caption"]["routes_all"]["last_caption"]["en->en"]
        self.assertEqual(en_diag["metrics"]["auroc"]["categories"]["plant.n.02"]["support"]["known"], 39)
        self.assertEqual(en_diag["metrics"]["auroc"]["categories"]["person.n.01"]["support"]["known"], 40)

        # Verify paired draw identity: delta_g == g_last - g_res and loss sign convention
        means_auroc = analysis_body["image_caption"]["twelve_route_means"]["auroc"]
        dg = means_auroc["delta_g_mean"]["primary_macro"]["value"]
        gl = means_auroc["g_last_mean"]["primary_macro"]["value"]
        gr = means_auroc["g_res_mean"]["primary_macro"]["value"]
        self.assertAlmostEqual(dg, gl - gr, places=12)
        self.assertGreater(dg, 0.015)
        self.assertGreater(gr, 0.015)

        # Loss metric gap: G_loss = M(s->t) - M(t->t) > 0 since off-diagonal has higher loss
        means_nll = analysis_body["image_caption"]["twelve_route_means"]["nll"]
        self.assertGreater(means_nll["g_last_mean"]["primary_macro"]["value"], 0.0)

        # Gates should both be OPEN
        self.assertEqual(gate_body["decisions"]["E30"]["decision"], "OPEN")
        self.assertEqual(gate_body["decisions"]["E11"]["decision"], "OPEN")
        self.assertEqual(plain(analysis_body["gate_ref"]), plain(gate_ref))

        # draw0_reference present and complete for Stage 4 score_e07
        d0 = analysis_body["draw0_reference"]
        self.assertEqual(d0["model"], "qwen")
        self.assertEqual(d0["n_images"], 40)
        self.assertIn("target_mask_fingerprint", d0)
        self.assertIn("twelve_route_mean", d0["point_estimates"])

        # Explicit-ref resume succeeds and returns identical refs
        res_a, _, res_g, _ = self._run_with_mocks(
            binding, draws, plan, "qwen", by_block_id,
            resume_analysis_ref=analysis_ref,
            resume_gate_ref=gate_ref,
        )
        self.assertEqual(plain(res_a), plain(analysis_ref))
        self.assertEqual(plain(res_g), plain(gate_ref))

    def test_gate_park_and_unresolved_with_unestimable_category(self):
        binding, draws = _make_mock_binding(self.root, self.protocol, n_images=40, n_draws=200)
        plan = self._prepare_plan(binding, draws)

        # Case 1: PARK (all routes identical perfect logits -> delta_g = 0, g_res = 0)
        by_block_park = _build_predictions_map(binding, "gemma", diag_logit=3.0, last_off_logit=3.0, mean_off_logit=3.0)
        # Note: valid_draws=200 < 1900 in this tiny 200-draw fixture, so gate is UNRESOLVED due to valid_draws < 1900.
        _, _, _, gate_unres = self._run_with_mocks(binding, draws, plan, "gemma", by_block_park)
        self.assertEqual(gate_unres["decisions"]["E30"]["decision"], "UNRESOLVED")
        self.assertEqual(gate_unres["decisions"]["E30"]["reason"], "insufficient_valid_draws")

        # Case 2: Full 2000 draws -> PARK when delta_g=0 and g_res=0
        binding_2k, draws_2k = _make_mock_binding(self.root, self.protocol, n_images=40, n_draws=2000)
        plan_2k = self._prepare_plan(binding_2k, draws_2k)
        by_block_2k = _build_predictions_map(binding_2k, "gemma", diag_logit=3.0, last_off_logit=3.0, mean_off_logit=3.0)
        _, _, _, gate_park = self._run_with_mocks(binding_2k, draws_2k, plan_2k, "gemma", by_block_2k)
        self.assertEqual(gate_park["decisions"]["E30"]["decision"], "PARK")
        self.assertEqual(gate_park["decisions"]["E11"]["decision"], "PARK")

        # Case 3: Unestimable primary category propagates None / valid_draws=0 -> UNRESOLVED
        primary_cat = self.protocol["primary_categories"][0]
        by_block_unest = _build_predictions_map(
            binding_2k, "qwen", diag_logit=3.0, last_off_logit=3.0, mean_off_logit=3.0, unestimable_cat=primary_cat
        )
        _, body_unest, _, gate_unest = self._run_with_mocks(binding_2k, draws_2k, plan_2k, "qwen", by_block_unest)
        auroc_macro = body_unest["image_caption"]["twelve_route_means"]["auroc"]["delta_g_mean"]["primary_macro"]
        self.assertIsNone(auroc_macro["value"])
        self.assertEqual(auroc_macro["interval"]["valid_draws"], 0)
        self.assertEqual(auroc_macro["interval"]["reason"], "no_valid_bootstrap_draws")
        self.assertEqual(gate_unest["decisions"]["E30"]["decision"], "UNRESOLVED")

    def test_resume_rejection_on_tamper_foreign_model_and_wrong_plan(self):
        binding, draws = _make_mock_binding(self.root, self.protocol, n_images=40, n_draws=200)
        plan = self._prepare_plan(binding, draws)
        by_block_id = _build_predictions_map(binding, "qwen")
        analysis_ref, _, gate_ref, _ = self._run_with_mocks(binding, draws, plan, "qwen", by_block_id)

        # 1. Foreign model rejection
        with self.assertRaises(ContractError):
            self._run_with_mocks(binding, draws, plan, "gemma", by_block_id, resume_analysis_ref=analysis_ref)

        # 2. Tampered sha256 rejection
        bad_ref = {"path": analysis_ref["path"], "sha256": "0" * 64}
        with self.assertRaises(ContractError):
            self._run_with_mocks(binding, draws, plan, "qwen", by_block_id, resume_analysis_ref=bad_ref)

        # 3. Wrong plan_identity rejection
        binding2, draws2 = _make_mock_binding(self.root, self.protocol, n_images=40, n_draws=200, seed=99999)
        plan2 = self._prepare_plan(binding2, draws2)
        with self.assertRaises(ContractError):
            self._run_with_mocks(binding2, draws2, plan2, "qwen", by_block_id, resume_analysis_ref=analysis_ref)


if __name__ == "__main__":
    unittest.main()
