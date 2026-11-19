"""CPU fixture and math verification tests for the Stage 4 E19 worker."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from src.common.contracts import ContractError, LANGUAGES, fingerprint
from src.stage2.schema import REPO, plain
from src.stage3.artifacts import write
from src.stage3.data import CS
from src.stage3.readouts import expected_rows, stats_identity
from src.stage4.interfaces import (
    MACRO, analysis_units, binary_metrics, bootstrap_counts, bootstrap_interval,
    confidence_cutoff, load_analysis_block, load_bootstrap, load_prediction_block,
    prepare_stage4, probabilities, score_and_publish_block, scoring_blocks,
)
from src.stage4.run_e19 import (
    E19_SELECTIVE_METRICS, _category_status, _metric_support, _selective_indicators,
    _selective_ratios, compute_e19_cutoffs, e19_units, evaluate_e19_unit, run_e19,
)
from src.stage4.validation import _validate_analysis_support, _validate_e19, _validate_screen_support
from test_stage3_foundation import Fixture


def _independent_selective(y, p, known, cutoff):
    """Independent scalar reference implementation for one E19 SCREEN sample."""
    y = np.asarray(y, dtype=bool)
    p = np.asarray(p, dtype="f8")
    known = np.asarray(known, dtype=bool)
    yk, pk = y[known], p[known]
    n = len(yk)
    if n == 0:
        return dict.fromkeys(E19_SELECTIVE_METRICS, None)
    d = pk >= 0.5
    conf = np.maximum(pk, 1.0 - pk)
    ret = conf >= cutoff
    err = d != yk
    n_pos = int(yk.sum())
    n_neg = n - n_pos
    r = int(ret.sum())
    r_pos = int((ret & yk).sum())
    r_neg = int((ret & ~yk).sum())
    e = int(err.sum())
    e_ret = int((ret & err).sum())
    tp_ret = int((ret & yk & d).sum())
    tn_ret = int((ret & ~yk & ~d).sum())
    assert tp_ret + tn_ret + e_ret == r
    base_risk = e / n
    return {
        "target_coverage": r / n,
        "retained_risk": None if r == 0 else e_ret / r,
        "base_risk": base_risk,
        "score_blind_risk": None if r == 0 else base_risk,
        "positive_retention": None if n_pos == 0 else r_pos / n_pos,
        "negative_retention": None if n_neg == 0 else r_neg / n_neg,
        "retained_sensitivity": None if r_pos == 0 else tp_ret / r_pos,
        "retained_specificity": None if r_neg == 0 else tn_ret / r_neg,
        "overall_retained_positive_correct": None if n_pos == 0 else tp_ret / n_pos,
        "overall_retained_negative_correct": None if n_neg == 0 else tn_ret / n_neg,
    }


class E19MathTests(unittest.TestCase):
    def test_selective_metrics_and_fixed_cutoff_bootstrap_match_independent_loop(self):
        # DEV-C probabilities with ties at the cutoff (0.25 and 0.75 both have conf=0.75)
        dev_p = np.array([0.25, 0.75, 0.50, 0.90, 0.60, 0.40])
        cut50 = confidence_cutoff(dev_p, 0.5)
        self.assertEqual(cut50, {"eligible": 6, "retained": 3, "cutoff": 0.75})

        # SCREEN observations with an unknown cell, ties at cutoff=0.75, and mixed errors
        y = np.array([1, 0, 1, 0, 1, 0, 1])
        p = np.array([0.90, 0.75, 0.25, 0.50, 0.80, 0.10, 0.20])
        known = np.array([True, True, True, True, False, True, True])
        draws = np.array([
            [0, 1, 2, 3, 4, 5, 6],
            [0, 0, 1, 1, 2, 5, 6],
            [3, 3, 3, 3, 4, 4, 3],  # no retained items in this draw (conf=0.5 < 0.75)
            [0, 0, 0, 0, 0, 0, 0],  # all positive, no negatives
        ], dtype=np.int64)
        counts = bootstrap_counts(draws, len(y))

        indicators = _selective_indicators(y.astype(bool), p, known, cut50["cutoff"])
        point_ratios = _selective_ratios(indicators.sum(axis=0))
        boot_ratios = _selective_ratios(counts @ indicators)

        expected_point = _independent_selective(y, p, known, cut50["cutoff"])
        for metric, expected in expected_point.items():
            if expected is None:
                self.assertTrue(np.isnan(point_ratios[metric]))
            else:
                self.assertAlmostEqual(point_ratios[metric], expected, places=12)
        # Identity: overall_retained_positive_correct == positive_retention * retained_sensitivity
        self.assertAlmostEqual(
            point_ratios["overall_retained_positive_correct"],
            point_ratios["positive_retention"] * point_ratios["retained_sensitivity"],
            places=12,
        )
        self.assertAlmostEqual(
            point_ratios["overall_retained_negative_correct"],
            point_ratios["negative_retention"] * point_ratios["retained_specificity"],
            places=12,
        )

        # Verify every bootstrap draw against independent row expansion with FIXED DEV-C cutoff
        for i, draw in enumerate(draws):
            expected_draw = _independent_selective(y[draw], p[draw], known[draw], cut50["cutoff"])
            for metric, expected in expected_draw.items():
                got = boot_ratios[metric][i]
                if expected is None:
                    self.assertTrue(np.isnan(got), f"Expected NaN for {metric} on draw {i}, got {got}")
                else:
                    self.assertAlmostEqual(got, expected, places=12)

        # Draw 2 has zero retained items -> retained_risk and score_blind_risk must be NaN
        self.assertTrue(np.isnan(boot_ratios["retained_risk"][2]))
        self.assertTrue(np.isnan(boot_ratios["score_blind_risk"][2]))
        self.assertEqual(bootstrap_interval(boot_ratios["retained_risk"])["valid_draws"], 3)

    def test_constant_references_tie_rule_and_support_statuses(self):
        # Constant always_absent (p=0.0 -> conf=1.0) and fit_prior (p=0.3 -> conf=0.7)
        for prior in (0.0, 0.3, 0.7):
            dev_p = np.full(10, prior)
            for nominal in (0.5, 0.75, 0.9):
                c = confidence_cutoff(dev_p, nominal)
                self.assertEqual(c["eligible"], 10)
                self.assertEqual(c["retained"], 10)
                self.assertAlmostEqual(c["cutoff"], max(prior, 1.0 - prior))

        # Support status checks: split minimum and subgroup/correctness minimum
        y = np.array([1] * 25 + [0] * 25, dtype=bool)
        p = np.array([0.9] * 24 + [0.1] + [0.1] * 25, dtype="f8")  # only 1 error total
        known = np.ones(50, dtype=bool)
        ind = _selective_indicators(y, p, known, 0.8)
        sums = ind.sum(axis=0)
        ratios = _selective_ratios(sums)
        # retained_risk has 1 error (< 20) -> descriptive_low_support, not ok and not undefined
        sup = _metric_support("retained_risk", sums)
        status, reason = _category_status("retained_risk", float(ratios["retained_risk"]), 25, 25, sup, 20, 20)
        self.assertEqual(status, "descriptive_low_support")
        self.assertTrue(reason)
        # A large denominator still leaves only one incorrect positive decision.
        sup_sens = _metric_support("retained_sensitivity", sums)
        status_s, reason_s = _category_status(
            "retained_sensitivity", float(ratios["retained_sensitivity"]), 25, 25, sup_sens, 20, 20
        )
        self.assertEqual(status_s, "descriptive_low_support")
        self.assertTrue(reason_s)


class E19PipelineTests(unittest.TestCase):
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
        # Score all E19 midpoint groups for both qwen and gemma (image_caption, decoder:2:last_caption)
        cls.scored = {
            gid for gid, g in cls.fit["groups"].items()
            if g.get("model") in ("qwen", "gemma")
            and g["condition"] == "image_caption"
            and g["site"] == "decoder:2:last_caption"
        }
        # Make one secondary category (index 1, timepiece.n.01) unestimable on one source to test padding/undefined
        cls.unestimable_group = sorted(
            gid for gid in cls.scored
            if cls.fit["groups"][gid]["model"] == "qwen" and cls.fit["groups"][gid]["language"] == "hi"
        )[0]
        cls.registry = cls._registry()
        cls.plan = prepare_stage4(f.root, f.protocol, cls.registry, f.producer)
        # Score all priority-0 blocks (4 DEV-C + 4 midpoint SCREEN) for both qwen and gemma
        cls.pred_refs = {}
        for model in ("qwen", "gemma"):
            model_refs = []
            for bid in scoring_blocks(cls.plan, model):
                if cls.plan["blocks"][bid]["priority"] == 0:
                    model_refs.append(score_and_publish_block(cls.plan, model, bid, f.producer))
            cls.pred_refs[model] = model_refs

    @classmethod
    def _registry(cls):
        f, fit = cls.fixture, cls.fit
        selections, candidates = [], []
        for gid in sorted(fit["groups"]):
            for assignment in ("seed0", "seed1", "seed2", "true"):
                def placeholder(label):
                    return {"path": f"fixture/{gid}/{assignment}/{label}/_COMPLETE.json", "sha256": "0" * 64}

                block = [placeholder(str(c)) for c in CS]
                selected = placeholder("selection")
                if assignment == "true":
                    if gid in cls.scored:
                        rows = expected_rows(fit, gid, "FIT", fit.store)
                        stats = write(
                            fit, "fixture/scalers", "statistics",
                            {"group_id": gid, "input_identity": stats_identity(fit, gid, rows), "center": True},
                            {"mean": np.array([0.5, 1.0]), "variance": np.array([0.25, 0.0]),
                             "scale": np.array([0.5, 1.0])},
                            f.producer,
                        )
                        coef = np.tile([1.5, -0.5], (25, 1))
                        intercept = np.linspace(-0.4, 0.4, 25)
                        estimable = np.ones(25, dtype=bool)
                        records = [{"status": "fitted", "attempts": [{"converged": True}]} for _ in range(25)]
                        if gid == cls.unestimable_group:
                            estimable[1] = False
                            coef[1] = 0.0
                            intercept[1] = 0.0
                            records[1] = {"status": "unestimable", "attempts": []}
                        block[0] = plain(write(
                            fit, "fixture/fits", "candidate",
                            {"group_id": gid, "assignment_id": "true", "C": CS[0],
                             "fit_rows": fingerprint(rows), "statistics": plain(stats), "records": records},
                            {"coefficients": coef, "intercepts": intercept, "estimable": estimable},
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
            {"flagged": False, "p": 0.362, "structural_checks": "passed",
             "groups": {gid: None for gid in fit["groups"]}},
            {}, f.producer,
        )
        body = {
            "suite": plain(fit["suite"]), "candidates": candidates, "selections": selections,
            "g0_gate": plain(gate), "stage2_groups": plain(fit["groups"]), "ordinary_systems": 145,
            "stage3_complete": False,
        }
        pending = write(fit, "fixture/registry", "registry_readback", body, {}, f.producer)
        return write(
            fit, "fixture/registry", "system_registry",
            dict(body, stage3_complete=True, independent_readback=plain(pending)),
            {}, f.producer,
        )

    def test_full_16_route_qwen_and_gemma_validate_against_coordinator_checks(self):
        p, f = self.plan, self.fixture
        support = _validate_screen_support(p)
        units = analysis_units(p)
        for model in ("qwen", "gemma"):
            events = []
            result = run_e19(p, model, self.pred_refs[model], f.producer,
                             progress=lambda ev, **kw: events.append((ev, kw)))
            self.assertEqual(len(result["units"]), 4)
            self.assertEqual(len(result["published"]), 4)
            self.assertEqual(result["reused"], [])
            # Build block_id -> ref map for coordinator _validate_e19 check
            pred_map = {
                p.store.json(ref, "body.json")["block_id"]: ref for ref in self.pred_refs[model]
            }
            for lang in LANGUAGES:
                uid = f"E19/{model}/{lang}"
                loaded = load_analysis_block(p, result["units"][uid], units=units)
                self.assertEqual(len(loaded["cutoffs"]), 225)
                # Verify coordinator's independent DEV-C cutoff & cohort support validators
                _validate_analysis_support(units[uid], loaded["rows"], support)
                _validate_e19(p, units[uid], loaded, pred_map)

            # Check unestimable head handling on qwen/hi category index 1 (timepiece.n.01)
            if model == "qwen":
                hi_block = load_analysis_block(p, result["units"]["E19/qwen/hi"], units=units)
                unest_cat = p["categories"][1]
                probe_cutoffs = [
                    c for c in hi_block["cutoffs"]
                    if c["baseline"] == "probe" and c["category"] == unest_cat
                ]
                self.assertEqual(len(probe_cutoffs), 3)
                for c in probe_cutoffs:
                    self.assertEqual((c["eligible"], c["retained"], c["cutoff"]), (0, 0, None))
                # Constant references remain defined even when probe head is unestimable
                const_cutoffs = [
                    c for c in hi_block["cutoffs"]
                    if c["baseline"] != "probe" and c["category"] == unest_cat
                ]
                self.assertTrue(all(c["cutoff"] is not None and c["retained"] == c["eligible"] > 0
                                    for c in const_cutoffs))

    def test_resume_reuse_and_rejection_guards(self):
        p, f = self.plan, self.fixture
        # 1. Run single representative source 'en'
        partial = run_e19(p, "qwen", self.pred_refs["qwen"], f.producer,
                          sources=["en"], progress=lambda *a, **k: None)
        self.assertEqual(partial["published"], ["E19/qwen/en"])
        en_ref = partial["units"]["E19/qwen/en"]

        # 2. Resume with all 4 sources: 'en' must be reused verbatim, other 3 published
        full = run_e19(p, "qwen", self.pred_refs["qwen"], f.producer,
                       resume_refs=[en_ref], progress=lambda *a, **k: None)
        self.assertEqual(full["reused"], ["E19/qwen/en"])
        self.assertEqual(full["published"], ["E19/qwen/hi", "E19/qwen/bn", "E19/qwen/te"])
        self.assertEqual(full["units"]["E19/qwen/en"], en_ref)

        # 3. Foreign-model resume reference must be rejected
        with self.assertRaisesRegex(ContractError, "not an owned E19 unit"):
            run_e19(p, "gemma", self.pred_refs["gemma"], f.producer,
                    resume_refs=[en_ref], progress=lambda *a, **k: None)

        # 4. Missing required prediction block must be rejected
        with self.assertRaisesRegex(ContractError, "Missing required E19 prediction blocks"):
            run_e19(p, "qwen", self.pred_refs["qwen"][:-1], f.producer,
                    progress=lambda *a, **k: None)

        # 5. Mismatched prediction inputs on resume (re-published prediction block -> new run_id) must reject
        re_scored = list(self.pred_refs["qwen"])
        bid0 = p.store.json(re_scored[0], "body.json")["block_id"]
        re_scored[0] = score_and_publish_block(p, "qwen", bid0, f.producer)
        with self.assertRaisesRegex(ContractError, "does not match current prediction inputs"):
            run_e19(p, "qwen", re_scored, f.producer,
                    sources=["en"], resume_refs=[en_ref], progress=lambda *a, **k: None)

    def test_notebook_structure(self):
        nb_path = REPO / "scripts" / "stage4" / "run_e19.ipynb"
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertEqual(nb["nbformat"], 4)
        self.assertEqual(len(nb["cells"]), 4)
        joined = "\n".join("".join(cell["source"]) for cell in nb["cells"])
        self.assertIn("MODEL = 'qwen'", joined)
        self.assertIn("src.stage4.run_e19", joined)
        self.assertIn("OPENBLAS_NUM_THREADS", joined)
        self.assertIn("RESUME_REFS", joined)


if __name__ == "__main__":
    unittest.main()
