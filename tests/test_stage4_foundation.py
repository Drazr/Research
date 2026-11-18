"""CPU checks for the Stage 4 worker handoff on synthetic accepted inputs."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.metrics import average_precision_score, roc_auc_score

from src.common.contracts import ContractError, fingerprint
from src.stage2.schema import plain
from src.stage3.artifacts import write
from src.stage3.data import CS
from src.stage3.readouts import expected_rows, stats_identity
from src.stage4.analysis import _validate_rows
from src.stage4.interfaces import (
    analysis_units, association_orders, binary_metrics, bootstrap_binary_metrics, bootstrap_counts,
    bootstrap_interval, confidence_cutoff, load_analysis_block, load_bootstrap,
    load_prediction_block, load_stage4_plan, paired_image_alignment, prepare_stage4,
    probabilities, publish_analysis_block, score_and_publish_block, scoring_blocks, targets_for_prediction,
)
from src.stage4.plan import _publish
from src.stage4.validation import PENDING, assemble_release, independent_readback
from test_stage3_foundation import Fixture


class MathTests(unittest.TestCase):
    def test_ties_mask_and_bootstrap_against_independent_row_expansion(self):
        y = np.array([0, 1, 1, 0, 1, 0])
        p = np.array([.1, .5, .5, .5, .8, .9])
        mask = np.array([1, 1, 0, 1, 1, 1], dtype=bool)
        point = binary_metrics(y, p, mask)
        self.assertEqual((point["positive"], point["negative"]), (2, 3))
        self.assertAlmostEqual(point["metrics"]["auroc"], roc_auc_score(y[mask], p[mask]))
        self.assertAlmostEqual(point["metrics"]["ap"], average_precision_score(y[mask], p[mask]))
        draws = np.array([[0, 1, 1, 2, 3, 4], [0, 0, 3, 3, 5, 5], [2, 2, 2, 2, 2, 2]])
        values = bootstrap_binary_metrics(y, p, mask, bootstrap_counts(draws, len(y)))
        for i, draw in enumerate(draws):
            expected = binary_metrics(y[draw], p[draw], mask[draw])["metrics"]
            for metric, value in expected.items():
                if value is None:
                    self.assertTrue(np.isnan(values[metric][i]))
                else:
                    self.assertAlmostEqual(value, values[metric][i])
        self.assertEqual(bootstrap_interval(values["auroc"])["valid_draws"], 1)

    def test_cutoff_and_association_seed(self):
        self.assertEqual(confidence_cutoff([.25, .75, .5], .5),
                         {"eligible": 3, "retained": 2, "cutoff": .75})
        self.assertIsNone(confidence_cutoff([], .9)["cutoff"])
        ids = ["a", "b", "c", "d"]
        got = association_orders("building.n.01", 1, ids)
        seed = int.from_bytes(hashlib.sha256(b"G9Q6-E25-association-v1|building.n.01|1").digest()[:8], "big")
        expected = np.random.Generator(np.random.PCG64(seed))
        assert_array_equal(got[0], expected.permutation(4))
        assert_array_equal(got[1], expected.permutation(4))
        self.assertEqual(got.shape, (1000, 4))
        with self.assertRaisesRegex(ContractError, "Invalid E25 stratum"):
            association_orders("building.n.01", 1, ids[::-1])


class AcceptedInputTests(unittest.TestCase):
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
        cls.scored = {gid for gid, g in cls.fit["groups"].items()
                      if g.get("model") == "qwen" and g["condition"] == "image_caption"
                      and g["site"] == "decoder:2:last_caption" or g.get("recipe", {}).get("family") == "pooled_char_tfidf"}
        cls.registry = cls._registry()
        cls.plan = prepare_stage4(f.root, f.protocol, cls.registry, f.producer)

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
                        dense = fit["groups"][gid]["source"] == "vlm"
                        stats = write(fit, "fixture/scalers", "statistics",
                                      {"group_id": gid, "input_identity": stats_identity(fit, gid, rows),
                                       "center": dense},
                                      {"mean": np.array([.5, 1.]), "variance": np.array([.25, 0.]),
                                       "scale": np.array([.5, 1.])}, f.producer)
                        records = [{"status": "fitted", "attempts": [{"converged": True}]} for _ in range(25)]
                        block[0] = plain(write(
                            fit, "fixture/fits", "candidate",
                            {"group_id": gid, "assignment_id": "true", "C": CS[0],
                             "fit_rows": fingerprint(rows), "statistics": plain(stats), "records": records},
                            {"coefficients": np.tile([2., 3.], (25, 1)), "intercepts": np.full(25, .25),
                             "estimable": np.ones(25, dtype=bool)}, f.producer))
                    selected = plain(write(fit, "fixture/selections", "selection",
                                           {"group_id": gid, "assignment_id": "true", "candidates": block,
                                            "selected": block[0], "selected_C": CS[0]}, {}, f.producer))
                candidates.extend(block)
                selections.append(selected)
        gate = write(fit, "fixture/gate", "g0_gate",
                     {"flagged": False, "p": .362, "structural_checks": "passed",
                      "groups": {gid: None for gid in fit["groups"]}}, {}, f.producer)
        body = {"suite": plain(fit["suite"]), "candidates": candidates, "selections": selections,
                "g0_gate": plain(gate), "stage2_groups": plain(fit["groups"]), "ordinary_systems": 145,
                "stage3_complete": False}
        pending = write(fit, "fixture/registry", "registry_readback", body, {}, f.producer)
        return write(fit, "fixture/registry", "system_registry",
                     dict(body, stage3_complete=True, independent_readback=plain(pending)), {}, f.producer)

    def _find(self, owner, split, *, language=None):
        plan = self.plan
        return next(bid for bid in scoring_blocks(plan, owner)
                    if plan["blocks"][bid]["split"] == split
                    and plan["blocks"][bid]["target_group"] in self.scored
                    and self.fit["groups"][plan["blocks"][bid]["target_group"]]["language"] == language)

    def test_gating_and_work_inventory(self):
        p = load_stage4_plan(self.fixture.root, self.fixture.protocol, self.plan.reference)
        self.assertEqual(p["identity"], self.plan["identity"])
        self.assertEqual([len(scoring_blocks(p, owner)) for owner in ("qwen", "gemma", "e03")], [70, 70, 13])
        self.assertEqual(len(analysis_units(p)), 47)
        draws = np.random.Generator(np.random.PCG64(2026111804)).integers(0, 4, (2000, 4), dtype=np.int64)
        assert_array_equal(load_bootstrap(p), draws)
        with self.assertRaisesRegex(ContractError, "Wrong artifact kind"):
            prepare_stage4(p.root, p.protocol, p["g0_gate"], self.fixture.producer)
        body = p.store.json(self.registry, "body.json")
        body["stage3_complete"] = False
        incomplete = self.fixture.put("stage3_system_registry", "body.json", body)
        with self.assertRaisesRegex(ContractError, "complete Stage 3"):
            prepare_stage4(p.root, p.protocol, incomplete, self.fixture.producer)
        with self.assertRaisesRegex(ContractError, "Incomplete canonical"):
            assemble_release(p, [], [])

    def test_dense_sparse_reuse_and_fresh_readback(self):
        p, f = self.plan, self.fixture
        en = self._find("qwen", "SCREEN", language="en")
        hi = self._find("qwen", "SCREEN", language="hi")
        dev = self._find("qwen", "DEV-C", language="en")
        pooled = self._find("e03", "SCREEN")
        with self.assertRaisesRegex(ContractError, "Foreign scoring block"):
            score_and_publish_block(p, "gemma", en, f.producer)
        refs = {bid: score_and_publish_block(p, p["blocks"][bid]["owner"], bid, f.producer)
                for bid in (en, hi, dev, pooled)}
        views = {bid: load_prediction_block(p, ref, expected_block=bid) for bid, ref in refs.items()}
        self.assertEqual(len(views[en]), 4)
        self.assertEqual(len(views[dev]), 1)
        dense = next(iter(views[en].values()))
        native = np.array([int(r["image_id"], 16) % 2 for r in dense.rows])
        assert_allclose(dense.logits[:, 0], (native - .5) / .5 * 2 + .25)
        sparse = next(iter(views[pooled].values()))
        assert_allclose(sparse.logits, (2 / .5 + 3) * np.float32(1 / np.sqrt(2)) + .25)
        _, mask = targets_for_prediction(p, dense)
        self.assertEqual(int((~mask[:, 16]).sum()), 1)
        _, (a, b) = paired_image_alignment(p, dense, sparse, contexts=(None, "bn"))
        self.assertEqual([dense.rows[i]["image_id"] for i in a], [sparse.rows[i]["image_id"] for i in b])
        with self.assertRaisesRegex(ContractError, "Select exactly one view"):
            paired_image_alignment(p, dense, sparse)
        with self.assertRaisesRegex(ContractError, "Wrong prediction block"):
            load_prediction_block(p, refs[en], expected_block=hi)

        pending = _publish(p.store, p["dataset_fingerprint"], "fixture/stage4", PENDING,
                           {"schema": "stage4.release.v1", "stage4_complete": False,
                            "dataset_fingerprint": p["dataset_fingerprint"],
                            "plan": p.reference, "predictions": {"dense": refs[en]}, "analyses": {}},
                           {}, [refs[en]], f.producer)
        self.assertEqual(independent_readback(p.root, p.protocol, pending), pending)
        (p.root / refs[en]["path"]).with_name("logits.npy").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ContractError, "Changed/corrupt payload"):
            independent_readback(p.root, p.protocol, pending)

    def test_missing_analysis_coverage(self):
        unit = analysis_units(self.plan)["E03/e03/shared"]
        cid = next(iter(unit["comparisons"]))
        row = {"comparison_id": cid, "category": self.plan["categories"][0], "statistic": "brier",
               "value": .25, "status": "ok", "reason": None, "support": {"known": 4, "positive": 2, "negative": 2}}
        with self.assertRaisesRegex(ContractError, "Incomplete required analysis rows"):
            _validate_rows(self.plan, unit, [row])

    def test_analysis_checkpoint_uses_exact_prediction_reference(self):
        p, f = self.plan, self.fixture
        bid = self._find("e03", "SCREEN")
        ref = score_and_publish_block(p, "e03", bid, f.producer)
        prediction = next(iter(load_prediction_block(p, ref).values()))
        uid = "E03/e03/shared"
        full = analysis_units(p)[uid]
        cid, comparison = next((key, value) for key, value in full["comparisons"].items()
                               if value.get("view") == "en" and value["inputs"][0]["block_id"] == bid)
        unit = dict(full, comparisons={cid: comparison}, required_blocks=[bid])
        index = [i for i, row in enumerate(prediction.rows) if row["language"] == "en"]
        y, known = targets_for_prediction(p, prediction)
        ps = probabilities(prediction.logits)
        rows, macros = [], {}
        for col, category in enumerate(p["categories"]):
            result = binary_metrics(y[index, col], ps[index, col], known[index, col])
            metrics = result.pop("metrics")
            for key, value in metrics.items():
                rows.append({"comparison_id": cid, "category": category, "statistic": key, "value": value,
                             "status": "ok", "reason": None, "support": result})
                if category in p["primary"]:
                    macros.setdefault(key, []).append(value)
        for key, values in macros.items():
            rows.append({"comparison_id": cid, "category": "primary_macro", "statistic": key,
                         "value": sum(values) / len(values), "status": "ok", "reason": None,
                         "support": {"categories": len(values)}})
        with patch("src.stage4.analysis.analysis_units", return_value={uid: unit}):
            with self.assertRaisesRegex(ContractError, "Wrong analysis prediction input coverage"):
                publish_analysis_block(p, uid, {}, rows, f.producer)
            saved = publish_analysis_block(p, uid, {bid: ref}, rows, f.producer)
            loaded = load_analysis_block(p, saved)
        self.assertEqual(plain(loaded["predictions"]), {bid: ref})
        self.assertEqual(plain(loaded["rows"]), rows)


if __name__ == "__main__":
    unittest.main()
