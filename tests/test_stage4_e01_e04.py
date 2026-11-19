"""CPU fixture tests for the Stage 4 E01-E04 analysis worker."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from src.common.contracts import ContractError, fingerprint
from src.stage2.schema import REPO, plain
from src.stage3.artifacts import write
from src.stage3.data import CS
from src.stage3.readouts import expected_rows, stats_identity
from src.stage4.interfaces import (
    MACRO,
    METRICS,
    analysis_units,
    binary_metrics,
    bootstrap_binary_metrics,
    bootstrap_counts,
    bootstrap_interval,
    fixed_macro,
    load_analysis_block,
    load_bootstrap,
    load_prediction_block,
    paired_image_alignment,
    prepare_stage4,
    probabilities,
    score_and_publish_block,
    scoring_blocks,
    targets_for_prediction,
)
from src.stage4.run_e01_e04 import (
    E04_UNAVAILABLE_ARMS,
    LOSS_METRICS,
    compute_e01_e04_unit,
    main as cli_main,
    run_e01_e04,
    select_e01_e04_units,
)
from src.stage4.validation import _validate_analysis_support, _validate_screen_support
from test_stage3_foundation import Fixture


class Stage4E01E04Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        f = cls.fixture = Fixture(Path(cls.tmp.name))
        f.protocol["minimum_positive_and_negative"] = dict.fromkeys(("FIT", "DEV-S", "DEV-C", "SCREEN"), 1)
        cls.fit = f.build()
        depths = patch.dict("src.stage4.plan.DEPTHS", {"qwen": (1, 2, 3, 4), "gemma": (1, 2, 3, 4)})
        depths.start()
        cls.addClassCleanup(depths.stop)

        # Populate accepted fitted candidates for:
        # - all qwen depth-2 decoder groups (both conditions, both poolings) + qwen encoder/projector
        # - all gemma depth-2 caption_only last_caption groups (for shared E03 comparison reuse)
        # - all 13 E03 groups
        cls.scored = set()
        for gid, g in cls.fit["groups"].items():
            if g["source"] == "e03":
                cls.scored.add(gid)
            elif g["model"] == "qwen" and (
                g["condition"] == "image_only" or g["site"].startswith("decoder:2:")
            ):
                cls.scored.add(gid)
            elif g["model"] == "gemma" and g["condition"] == "caption_only" and g["site"] == "decoder:2:last_caption":
                cls.scored.add(gid)

        cls.registry = cls._build_registry()
        cls.plan = prepare_stage4(f.root, f.protocol, cls.registry, f.producer)
        cls.pred_refs = cls._score_needed_blocks()

    @classmethod
    def _build_registry(cls):
        f, fit = cls.fixture, cls.fit
        selections, candidates = [], []
        for idx, gid in enumerate(sorted(fit["groups"])):
            for assignment in ("seed0", "seed1", "seed2", "true"):
                def placeholder(label):
                    return {"path": f"fixture/{gid}/{assignment}/{label}/_COMPLETE.json", "sha256": "0" * 64}

                block = [placeholder(str(c)) for c in CS]
                selected = placeholder("selection")
                if assignment == "true":
                    if gid in cls.scored:
                        rows = expected_rows(fit, gid, "FIT", fit.store)
                        dense = (fit["groups"][gid]["source"] == "vlm"
                                 or fit["groups"][gid].get("recipe", {}).get("family") == "length")
                        stats = write(
                            fit, "fixture/scalers", "statistics",
                            {"group_id": gid, "input_identity": stats_identity(fit, gid, rows), "center": dense},
                            {"mean": np.array([.5, 1.]), "variance": np.array([.25, 0.]), "scale": np.array([.5, 1.])},
                            f.producer,
                        )
                        # Distinct per-group coefficients so paired contrasts are non-trivial.
                        c0 = 1.2 + 0.17 * (idx % 7)
                        c1 = -0.8 + 0.23 * (idx % 5)
                        b0 = -0.3 + 0.09 * (idx % 6)
                        records = [{"status": "fitted", "attempts": [{"converged": True}]} for _ in range(25)]
                        block[0] = plain(write(
                            fit, "fixture/fits", "candidate",
                            {"group_id": gid, "assignment_id": "true", "C": CS[0],
                             "fit_rows": fingerprint(rows), "statistics": plain(stats), "records": records},
                            {"coefficients": np.tile([c0, c1], (25, 1)), "intercepts": np.full(25, b0),
                             "estimable": np.ones(25, dtype=bool)},
                            f.producer,
                        ))
                    selected = plain(write(
                        fit, "fixture/selections", "selection",
                        {"group_id": gid, "assignment_id": "true", "candidates": block,
                         "selected": block[0], "selected_C": CS[0]},
                        {}, f.producer,
                    ))
                candidates.extend(block)
                selections.append(selected)
        gate = write(
            fit, "fixture/gate", "g0_gate",
            {"flagged": False, "p": .362, "structural_checks": "passed",
             "groups": {gid: None for gid in fit["groups"]}},
            {}, f.producer,
        )
        body = {
            "suite": plain(fit["suite"]), "candidates": candidates, "selections": selections,
            "g0_gate": plain(gate), "stage2_groups": plain(fit["groups"]),
            "ordinary_systems": 145, "stage3_complete": False,
        }
        pending = write(fit, "fixture/registry", "registry_readback", body, {}, f.producer)
        return write(
            fit, "fixture/registry", "system_registry",
            dict(body, stage3_complete=True, independent_readback=plain(pending)),
            {}, f.producer,
        )

    @classmethod
    def _score_needed_blocks(cls):
        p, f = cls.plan, cls.fixture
        refs = {}
        for owner in ("e03", "qwen", "gemma"):
            for bid in scoring_blocks(p, owner):
                block = p["blocks"][bid]
                if block["split"] == "SCREEN" and block["target_group"] in cls.scored:
                    refs[bid] = score_and_publish_block(p, owner, bid, f.producer)
        return refs

    def test_unit_selection_and_ownership(self):
        p = self.plan
        self.assertEqual(select_e01_e04_units(p, owner="e03"), ("E03/e03/shared",))
        qwen_units = select_e01_e04_units(p, owner="qwen")
        gemma_units = select_e01_e04_units(p, owner="gemma")
        all_units = select_e01_e04_units(p, owner="all")
        self.assertEqual(len(qwen_units), 14)
        self.assertEqual(len(gemma_units), 14)
        self.assertEqual(len(all_units), 29)
        self.assertIn("E02/qwen/visual", qwen_units)
        self.assertIn("E03/qwen/comparison", qwen_units)
        self.assertIn("E03/gemma/comparison", gemma_units)
        with self.assertRaisesRegex(ContractError, "Unknown E01-E04 owner"):
            select_e01_e04_units(p, owner="other")
        with self.assertRaisesRegex(ContractError, "Experiments must be"):
            select_e01_e04_units(p, owner="qwen", experiments=("E19",))
        with self.assertRaisesRegex(ContractError, "not in the selected"):
            select_e01_e04_units(p, owner="qwen", unit_ids=("E03/e03/shared",))

    def test_e01_e02_e03_e04_end_to_end_math_and_validator_support(self):
        p, f = self.plan, self.fixture
        units = analysis_units(p)
        screen_support = _validate_screen_support(p)

        # 1. Shared E03 runs once without needing VLM predictions.
        e03_only_preds = [ref for bid, ref in self.pred_refs.items() if p["blocks"][bid]["owner"] == "e03"]
        e03_summary = run_e01_e04(p, e03_only_preds, f.producer, owner="e03")
        self.assertEqual(e03_summary["computed_units"], ["E03/e03/shared"])
        self.assertEqual(e03_summary["optional_coverage"], list(E04_UNAVAILABLE_ARMS))
        e03_block = load_analysis_block(p, e03_summary["units"]["E03/e03/shared"], units=units)
        _validate_analysis_support(units["E03/e03/shared"], e03_block["rows"], screen_support)

        # Verify pooled_context_mean equals equal-weight average of the 4 language views.
        by_key = {(r["comparison_id"], r["category"], r["statistic"]): r for r in e03_block["rows"]}
        pcm_cid = next(cid for cid, c in units["E03/e03/shared"]["comparisons"].items()
                       if c["kind"] == "pooled_context_mean")
        pooled_bid = units["E03/e03/shared"]["comparisons"][pcm_cid]["inputs"][0]["block_id"]
        view_cids = {
            c["view"]: cid for cid, c in units["E03/e03/shared"]["comparisons"].items()
            if c["kind"] == "absolute" and c["inputs"][0]["block_id"] == pooled_bid
        }
        for cat in (p["categories"][0], "plant.n.02", MACRO):
            for metric in METRICS:
                expected_mean = sum(by_key[(view_cids[v], cat, metric)]["value"] for v in ("en", "hi", "bn", "te")) / 4.0
                assert_allclose(by_key[(pcm_cid, cat, metric)]["value"], expected_mean, rtol=1e-12, atol=1e-12)
        # Verify unknown plant.n.02 cell on SCREEN is preserved (3 known instead of 4).
        plant_row = by_key[(pcm_cid, "plant.n.02", "brier")]
        self.assertEqual(plant_row["support"]["known"], 3)
        self.assertEqual(plant_row["support"]["views"], 4)

        # 2. Run Qwen depth-2 E01, E02 (depth 2 + visual), E03 comparison, E04 (depth 2).
        qwen_target_units = (
            "E01/qwen/2",
            "E02/qwen/2",
            "E02/qwen/visual",
            "E03/qwen/comparison",
            "E04/qwen/2",
        )
        qwen_summary = run_e01_e04(
            p, list(self.pred_refs.values()), f.producer, owner="qwen", unit_ids=qwen_target_units
        )
        self.assertEqual(set(qwen_summary["computed_units"]), set(qwen_target_units))

        # 3. Run Gemma E03 comparison reusing the exact same shared E03 predictions.
        gemma_summary = run_e01_e04(
            p, list(self.pred_refs.values()), f.producer, owner="gemma", unit_ids=("E03/gemma/comparison",)
        )
        self.assertEqual(gemma_summary["computed_units"], ["E03/gemma/comparison"])

        # Validate every published block against coordinator load_analysis_block & support check.
        loaded = {}
        for uid, ref in {**qwen_summary["units"], **gemma_summary["units"]}.items():
            block = load_analysis_block(p, ref, units=units)
            _validate_analysis_support(units[uid], block["rows"], screen_support)
            loaded[uid] = block

        # Verify E01 transfer_gap sign convention (diag - base for discrimination, base - diag for loss).
        e01_unit = units["E01/qwen/2"]
        e01_rows = {(r["comparison_id"], r["category"], r["statistic"]): r for r in loaded["E01/qwen/2"]["rows"]}
        tg_cid, tg_comp = next((cid, c) for cid, c in e01_unit["comparisons"].items() if c["kind"] == "transfer_gap")
        diag_inp, base_inp = tg_comp["inputs"]
        diag_abs = next(cid for cid, c in e01_unit["comparisons"].items()
                        if c["kind"] == "absolute" and tuple(c["inputs"]) == (diag_inp,))
        base_abs = next(cid for cid, c in e01_unit["comparisons"].items()
                        if c["kind"] == "absolute" and tuple(c["inputs"]) == (base_inp,))
        cat0 = p["categories"][0]
        for metric in METRICS:
            dv = e01_rows[(diag_abs, cat0, metric)]["value"]
            bv = e01_rows[(base_abs, cat0, metric)]["value"]
            expected = (bv - dv) if metric in LOSS_METRICS else (dv - bv)
            assert_allclose(e01_rows[(tg_cid, cat0, metric)]["value"], expected, rtol=1e-12, atol=1e-12)
            self.assertIsNotNone(e01_rows[(tg_cid, cat0, metric)]["interval"])

        # Verify E01 fit_prior & always_absent baselines omit interval and match independent binary_metrics.
        fp_cid, fp_comp = next((cid, c) for cid, c in e01_unit["comparisons"].items() if c["kind"] == "fit_prior")
        aa_cid, _ = next((cid, c) for cid, c in e01_unit["comparisons"].items()
                         if c["kind"] == "always_absent" and c["inputs"] == fp_comp["inputs"])
        self.assertNotIn("interval", e01_rows[(fp_cid, cat0, "brier")])
        self.assertNotIn("interval", e01_rows[(aa_cid, cat0, "brier")])
        pred_fp = load_prediction_block(p, self.pred_refs[fp_comp["inputs"][0]["block_id"]])[fp_comp["inputs"][0]["route_id"]]
        y_fp, known_fp = targets_for_prediction(p, pred_fp)
        _, (al_fp,) = paired_image_alignment(p, pred_fp)
        expected_aa = binary_metrics(y_fp[al_fp, 0], np.zeros(len(al_fp)), known_fp[al_fp, 0])["metrics"]
        assert_allclose(e01_rows[(aa_cid, cat0, "brier")]["value"], expected_aa["brier"], rtol=1e-12, atol=1e-12)

        # Verify E02 transfer_gap_change = combined_gap - caption_only_gap (with loss sign convention).
        e02_unit = units["E02/qwen/2"]
        e02_rows = {(r["comparison_id"], r["category"], r["statistic"]): r for r in loaded["E02/qwen/2"]["rows"]}
        tgc_cid, tgc_comp = next((cid, c) for cid, c in e02_unit["comparisons"].items()
                                 if c["kind"] == "transfer_gap_change")
        d_comb, b_comb, d_text, b_text = tgc_comp["inputs"]
        e02_t_diag = next(cid for cid, c in e02_unit["comparisons"].items()
                          if c["kind"] == "caption_only" and tuple(c["inputs"]) == (d_text,))
        e02_t_base = next(cid for cid, c in e02_unit["comparisons"].items()
                          if c["kind"] == "caption_only" and tuple(c["inputs"]) == (b_text,))
        e01_c_gap = next(cid for cid, c in e01_unit["comparisons"].items()
                         if c["kind"] == "transfer_gap" and tuple(c["inputs"]) == (d_comb, b_comb))
        for metric in METRICS:
            comb_gap = e01_rows[(e01_c_gap, cat0, metric)]["value"]
            td_v = e02_rows[(e02_t_diag, cat0, metric)]["value"]
            tb_v = e02_rows[(e02_t_base, cat0, metric)]["value"]
            text_gap = (tb_v - td_v) if metric in LOSS_METRICS else (td_v - tb_v)
            assert_allclose(e02_rows[(tgc_cid, cat0, metric)]["value"], comb_gap - text_gap, rtol=1e-12, atol=1e-12)

        # Verify E02 visual_delta = neural - visual.
        vis_unit = units["E02/qwen/visual"]
        vis_rows = {(r["comparison_id"], r["category"], r["statistic"]): r for r in loaded["E02/qwen/visual"]["rows"]}
        vd_cid, vd_comp = next((cid, c) for cid, c in vis_unit["comparisons"].items() if c["kind"] == "visual_delta")
        img_cid = next(cid for cid, c in vis_unit["comparisons"].items()
                       if c["kind"] == "image_only" and tuple(c["inputs"]) == (vd_comp["inputs"][1],))
        neural_pred = load_prediction_block(p, self.pred_refs[vd_comp["inputs"][0]["block_id"]])[vd_comp["inputs"][0]["route_id"]]
        y_n, k_n = targets_for_prediction(p, neural_pred)
        _, (al_n,) = paired_image_alignment(p, neural_pred)
        neural_brier = binary_metrics(y_n[al_n, 0], probabilities(neural_pred.logits[al_n])[:, 0], k_n[al_n, 0])["metrics"]["brier"]
        assert_allclose(
            vis_rows[(vd_cid, cat0, "brier")]["value"],
            neural_brier - vis_rows[(img_cid, cat0, "brier")]["value"],
            rtol=1e-12, atol=1e-12,
        )

        # Verify E04 pool_delta = caption_mean - last_caption, including paired bootstrap interval.
        e04_unit = units["E04/qwen/2"]
        e04_rows = {(r["comparison_id"], r["category"], r["statistic"]): r for r in loaded["E04/qwen/2"]["rows"]}
        pd_cid, pd_comp = next((cid, c) for cid, c in e04_unit["comparisons"].items() if c["kind"] == "pool_delta")
        mean_inp, last_inp = pd_comp["inputs"]
        mean_pred = load_prediction_block(p, self.pred_refs[mean_inp["block_id"]])[mean_inp["route_id"]]
        last_pred = load_prediction_block(p, self.pred_refs[last_inp["block_id"]])[last_inp["route_id"]]
        _, (am, al) = paired_image_alignment(p, mean_pred, last_pred)
        ym, km = targets_for_prediction(p, mean_pred)
        counts = bootstrap_counts(load_bootstrap(p), len(p["screen_ids"]))
        dm = bootstrap_binary_metrics(ym[am, 0], probabilities(mean_pred.logits[am])[:, 0], km[am, 0], counts)["brier"]
        dl = bootstrap_binary_metrics(ym[al, 0], probabilities(last_pred.logits[al])[:, 0], km[al, 0], counts)["brier"]
        expected_iv = bootstrap_interval(dm - dl)
        self.assertEqual(e04_rows[(pd_cid, cat0, "brier")]["interval"], expected_iv)

    def test_unestimable_head_and_low_support_and_unresolved_macro(self):
        p = self.plan
        unit_id = "E03/e03/shared"
        unit = analysis_units(p)[unit_id]
        blocks = {bid: load_prediction_block(p, self.pred_refs[bid], expected_block=bid)
                  for bid in unit["required_blocks"]}
        counts = bootstrap_counts(load_bootstrap(p), len(p["screen_ids"]))

        # Clone one route with category 0 (primary) marked unestimable and zero-padded logits.
        first_bid = unit["required_blocks"][0]
        first_rid, orig_pred = next(iter(blocks[first_bid].items()))
        est = orig_pred.estimable.copy()
        est[0] = False
        logits = orig_pred.logits.copy()
        logits[:, 0] = 0.0
        patched_pred = type(orig_pred)(
            orig_pred.reference, orig_pred.route, orig_pred.split, orig_pred.rows,
            logits, est, orig_pred.prevalence,
        )
        blocks[first_bid] = dict(blocks[first_bid], **{first_rid: patched_pred})

        # Raise minimum_positive_and_negative["SCREEN"] to 2 so categories with min(pos,neg)==1 are low-support.
        orig_min = p.protocol["minimum_positive_and_negative"]["SCREEN"]
        p.protocol["minimum_positive_and_negative"]["SCREEN"] = 2
        try:
            rows = compute_e01_e04_unit(p, unit_id, blocks, counts)
        finally:
            p.protocol["minimum_positive_and_negative"]["SCREEN"] = orig_min

        target_cid = next(cid for cid, c in unit["comparisons"].items()
                          if c["inputs"][0] == {"block_id": first_bid, "route_id": first_rid})
        by_key = {(r["comparison_id"], r["category"], r["statistic"]): r for r in rows}

        # Category 0 is unestimable -> undefined with reason 'unestimable_head', valid_draws=0, intact SCREEN support.
        r0 = by_key[(target_cid, p["categories"][0], "brier")]
        self.assertIsNone(r0["value"])
        self.assertEqual(r0["status"], "undefined")
        self.assertEqual(r0["reason"], "unestimable_head")
        self.assertEqual(r0["interval"]["valid_draws"], 0)
        self.assertEqual(r0["support"]["known"], 4)

        # Primary macro is unresolved because category 0 is in plan["primary"].
        rm = by_key[(target_cid, MACRO, "brier")]
        self.assertIsNone(rm["value"])
        self.assertEqual(rm["status"], "undefined")
        self.assertEqual(rm["reason"], "unresolved_primary_macro")
        self.assertEqual(rm["support"]["defined_categories"], len(p["primary"]) - 1)

        # plant.n.02 has 3 known cells -> min(pos, neg) == 1 < 2 -> descriptive_low_support.
        r_plant = by_key[(target_cid, "plant.n.02", "brier")]
        self.assertIsNotNone(r_plant["value"])
        self.assertEqual(r_plant["status"], "descriptive_low_support")
        self.assertEqual(r_plant["reason"], "screen_support_below_minimum_2")

    def test_checkpoint_resume_mismatch_and_cli_and_notebook(self):
        p, f = self.plan, self.fixture
        e03_preds = [ref for bid, ref in self.pred_refs.items() if p["blocks"][bid]["owner"] == "e03"]

        # First run computes and checkpoints E03/e03/shared.
        first = run_e01_e04(p, e03_preds, f.producer, owner="e03")
        ref1 = first["units"]["E03/e03/shared"]
        self.assertEqual(first["computed_units"], ["E03/e03/shared"])
        self.assertEqual(first["reused_units"], [])

        # Second run with exact matching resume ref reuses without recomputing.
        second = run_e01_e04(p, e03_preds, f.producer, owner="e03", resume_refs=[ref1])
        self.assertEqual(second["reused_units"], ["E03/e03/shared"])
        self.assertEqual(second["computed_units"], [])
        self.assertEqual(second["units"]["E03/e03/shared"], ref1)

        # Re-publish one E03 prediction block -> new prediction ref -> resume ref must NOT be reused.
        one_e03_bid = next(bid for bid in self.pred_refs if p["blocks"][bid]["owner"] == "e03")
        new_pred_ref = score_and_publish_block(p, "e03", one_e03_bid, f.producer)
        updated_e03_preds = [
            new_pred_ref if bid == one_e03_bid else ref
            for bid, ref in self.pred_refs.items() if p["blocks"][bid]["owner"] == "e03"
        ]
        events = []
        third = run_e01_e04(p, updated_e03_preds, f.producer, owner="e03",
                            resume_refs=[ref1], progress=events.append)
        self.assertEqual(third["computed_units"], ["E03/e03/shared"])
        self.assertEqual(third["reused_units"], [])
        self.assertNotEqual(third["units"]["E03/e03/shared"], ref1)
        self.assertTrue(any(e.get("event") == "analysis_unit_resume_skipped" for e in events))

        # Missing required prediction block raises ContractError.
        with self.assertRaisesRegex(ContractError, "Missing required prediction blocks"):
            run_e01_e04(p, e03_preds[:-1], f.producer, owner="e03")

        # CLI end-to-end with --predictions, --resume-analyses, and --output-index.
        with tempfile.TemporaryDirectory() as td:
            pred_file = Path(td) / "preds.json"
            resume_file = Path(td) / "resume.json"
            out_file = Path(td) / "out.json"
            pred_file.write_text(json.dumps({"predictions": e03_preds}), encoding="utf-8")
            resume_file.write_text(json.dumps({"analyses": [ref1]}), encoding="utf-8")
            with patch("src.stage4.run_e01_e04.current_producer", return_value=f.producer), \
                 patch("src.stage4.run_e01_e04.load_stage4_plan", return_value=p):
                summary = cli_main([
                    "--phase1-root", str(f.root),
                    "--manifest", str(f.root / p.reference["path"]),
                    "--owner", "e03",
                    "--predictions", str(pred_file),
                    "--resume-analyses", str(resume_file),
                    "--output-index", str(out_file),
                    "--storage-scope", "local-development",
                ])
            self.assertEqual(summary["reused_units"], ["E03/e03/shared"])
            written = json.loads(out_file.read_text(encoding="utf-8"))
            self.assertEqual(written["optional_coverage"], list(E04_UNAVAILABLE_ARMS))
            self.assertEqual(written["units"]["E03/e03/shared"], ref1)

        # Notebook structure verification.
        nb_path = REPO / "scripts/stage4/run_e01_e04.ipynb"
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertEqual(nb["nbformat"], 4)
        self.assertEqual(len(nb["cells"]), 4)
        code_text = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        self.assertIn("MODEL = 'qwen'", code_text)
        self.assertIn("src.stage4.run_e01_e04", code_text)
        self.assertIn("OPENBLAS_NUM_THREADS", code_text)


if __name__ == "__main__":
    unittest.main()
