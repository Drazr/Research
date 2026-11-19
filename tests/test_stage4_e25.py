"""CPU fixture tests for the Stage 4 E25 paired-error and association-null worker."""

import hashlib
import json
import os
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
    MACRO, PERMUTATION_DRAWS, analysis_units, bootstrap_counts, e25_memberships,
    load_analysis_block, load_bootstrap, load_prediction_block, paired_image_alignment,
    prepare_stage4, probabilities, required_rows, score_and_publish_block,
    scoring_blocks, targets_for_prediction,
)
from src.stage4.run_e25 import E25_PARTS, _eval_stratified, compute_e25_unit, e25_units, run_e25
from src.stage4.validation import _validate_analysis_support, _validate_screen_support
from test_stage3_foundation import Fixture


def _independent_e25_point_and_null(y, known, d_a, d_b, ids, category, label):
    """Independent reference implementation for one category and label stratum."""
    if label is None:
        m = known
        n = int(m.sum())
        if n == 0:
            return dict.fromkeys(("disagreement", "common_wrong", "error_a", "error_b")), None
        ea = d_a[m] != y[m]
        eb = d_b[m] != y[m]
        return {
            "disagreement": float((d_a[m] != d_b[m]).mean()),
            "common_wrong": float((ea & eb).mean()),
            "error_a": float(ea.mean()),
            "error_b": float(eb.mean()),
        }, None

    m = known & (y == bool(label))
    n = int(m.sum())
    if n == 0:
        keys = ("error_a", "error_b", "common_wrong", "independence", "excess",
                "error_b_given_a", "error_a_given_b", "null_lower", "null_upper")
        return dict.fromkeys(keys), np.full(PERMUTATION_DRAWS, np.nan)
    ea = d_a[m] != bool(label)
    eb = d_b[m] != bool(label)
    cw = ea & eb
    pa, pb, pab = float(ea.mean()), float(eb.mean()), float(cw.mean())
    ka, kb, kab = int(ea.sum()), int(eb.sum()), int(cw.sum())
    seed = int.from_bytes(
        hashlib.sha256(f"G9Q6-E25-association-v1|{category}|{label}".encode("utf-8")).digest()[:8],
        "big",
    )
    rng = np.random.Generator(np.random.PCG64(seed))
    null_overlaps = np.array([
        float((ea & eb[rng.permutation(n)]).mean()) for _ in range(PERMUTATION_DRAWS)
    ], dtype="f8")
    q_low, q_high = np.quantile(null_overlaps, [0.025, 0.975], method="linear")
    return {
        "error_a": pa,
        "error_b": pb,
        "common_wrong": pab,
        "independence": pa * pb,
        "excess": pab - pa * pb,
        "error_b_given_a": None if ka == 0 else float(kab / ka),
        "error_a_given_b": None if kb == 0 else float(kab / kb),
        "null_lower": float(q_low),
        "null_upper": float(q_high),
    }, null_overlaps


class E25SupportTests(unittest.TestCase):
    def test_large_conditional_denominator_cannot_hide_a_sparse_outcome(self):
        n = 25
        category = "building.n.01"
        ids = tuple(f"{i:02d}" for i in range(2 * n))
        y = np.array([False] * n + [True] * n, dtype=bool)[:, None]
        decision_a = np.zeros_like(y)
        decision_b = y.copy()
        decision_b[n] = False
        rows = _eval_stratified(
            "example", 1, ids, [category], [0], y, np.ones_like(y),
            decision_a, decision_b, np.array([True]), np.ones((4, 2 * n), dtype=np.int64),
            20, {category: {"1": fingerprint(list(ids[n:]))}}, {},
        )
        row = next(r for r in rows if r["category"] == category and r["statistic"] == "error_b_given_a")
        self.assertEqual(row["status"], "descriptive_low_support")
        self.assertEqual(row["value"], 1 / n)
        self.assertEqual((row["support"]["numerator"], row["support"]["denominator"]), (1, n))


class E25FixtureTests(unittest.TestCase):
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

        cls.midpoint_groups = {
            gid for gid, g in cls.fit["groups"].items()
            if g.get("source") == "vlm"
            and g["condition"] == "image_caption"
            and g["site"] == "decoder:2:last_caption"
        }
        cls.registry = cls._build_registry()
        cls.plan = prepare_stage4(f.root, f.protocol, cls.registry, f.producer)
        cls.pred_refs = {}
        for model in ("qwen", "gemma"):
            refs = {}
            for bid in scoring_blocks(cls.plan, model):
                block = cls.plan["blocks"][bid]
                if block["split"] == "SCREEN" and block["target_group"] in cls.midpoint_groups:
                    refs[bid] = score_and_publish_block(cls.plan, model, bid, f.producer)
            cls.pred_refs[model] = refs

    @classmethod
    def _build_registry(cls):
        f, fit = cls.fixture, cls.fit
        selections, candidates = [], []
        lang_offset = {"en": -0.35, "hi": 0.15, "bn": -0.05, "te": 0.45}
        for gid in sorted(fit["groups"]):
            group = fit["groups"][gid]
            for assignment in ("seed0", "seed1", "seed2", "true"):
                def placeholder(label):
                    return {"path": f"fixture/{gid}/{assignment}/{label}/_COMPLETE.json", "sha256": "0" * 64}

                block = [placeholder(str(c)) for c in CS]
                selected = placeholder("selection")
                if assignment == "true":
                    if gid in cls.midpoint_groups:
                        rows = expected_rows(fit, gid, "FIT", fit.store)
                        stats = write(
                            fit, "fixture/scalers", "statistics",
                            {"group_id": gid, "input_identity": stats_identity(fit, gid, rows), "center": True},
                            {"mean": np.array([0.5, 1.0]), "variance": np.array([0.25, 0.25]),
                             "scale": np.array([0.5, 0.5])},
                            f.producer,
                        )
                        records = [{"status": "fitted", "attempts": [{"converged": True}]} for _ in range(25)]
                        # Distinct per-language/per-category weights so contexts produce varied errors and disagreements.
                        offset = lang_offset[group["language"]]
                        coefs = np.array([
                            [1.2 if (c % 2 == 0) else -0.9, 0.4 * ((c % 3) - 1) + offset]
                            for c in range(25)
                        ], dtype="f8")
                        intercepts = np.array([offset + 0.1 * ((c % 5) - 2) for c in range(25)], dtype="f8")
                        block[0] = plain(write(
                            fit, "fixture/fits", "candidate",
                            {"group_id": gid, "assignment_id": "true", "C": CS[0],
                             "fit_rows": fingerprint(rows), "statistics": plain(stats), "records": records},
                            {"coefficients": coefs, "intercepts": intercepts,
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
            {"flagged": False, "p": 0.362, "structural_checks": "passed",
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

    def test_01_full_model_run_coverage_validation_and_math_equivalence(self):
        plan = self.plan
        support = _validate_screen_support(plan)
        units_all = analysis_units(plan)

        for model in ("qwen", "gemma"):
            events = []
            summary = run_e25(
                plan, model, list(self.pred_refs[model].values()),
                self.fixture.producer, progress=events.append,
            )
            self.assertEqual(summary["status"], "passed")
            self.assertEqual(len(summary["units"]), 5)
            self.assertEqual(list(summary["units"]), [f"E25/{model}/{p}" for p in E25_PARTS])
            self.assertEqual(len(summary["published"]), 5)
            self.assertEqual(summary["reused"], [])

            for uid, ref in summary["units"].items():
                loaded = load_analysis_block(plan, ref, units=units_all)
                unit = units_all[uid]
                self.assertEqual(plain(loaded["association_memberships"]), e25_memberships(plan))
                self.assertEqual(plain(loaded["cutoffs"]), [])
                _validate_analysis_support(unit, loaded["rows"], support)
                row_keys = {(r["comparison_id"], r["category"], r["statistic"]) for r in loaded["rows"]}
                self.assertEqual(row_keys, required_rows(plan, unit))

        # Verify exact numerical equivalence against independent reference on E25/qwen/en and E25/qwen/diagonal.
        blocks = {bid: load_prediction_block(plan, ref, expected_block=bid)
                  for bid, ref in self.pred_refs["qwen"].items()}
        draws = load_bootstrap(plan)
        counts = bootstrap_counts(draws, len(plan["screen_ids"]))
        categories = tuple(plan["categories"])
        primary_indices = [categories.index(c) for c in plan["primary"]]

        for part in ("en", "diagonal"):
            uid = f"E25/qwen/{part}"
            unit = units_all[uid]
            rows = compute_e25_unit(plan, uid, unit, blocks, counts=counts)
            by_key = {(r["comparison_id"], r["category"], r["statistic"]): r for r in rows}

            # Check all 6 target pairs exist and diagonal uses distinct source heads while fixed source does not.
            probe_unstrat = [
                (cid, comp) for cid, comp in unit["comparisons"].items()
                if comp["baseline"] == "probe" and comp["label"] is None
            ]
            self.assertEqual(len(probe_unstrat), 6)
            for cid, comp in probe_unstrat:
                pa = blocks[comp["inputs"][0]["block_id"]][comp["inputs"][0]["route_id"]]
                pb = blocks[comp["inputs"][1]["block_id"]][comp["inputs"][1]["route_id"]]
                src_a = plan.fit["groups"][pa.route["source_group"]]["language"]
                src_b = plan.fit["groups"][pb.route["source_group"]]["language"]
                tgt_a = plan.fit["groups"][plan["blocks"][comp["inputs"][0]["block_id"]]["target_group"]]["language"]
                tgt_b = plan.fit["groups"][plan["blocks"][comp["inputs"][1]["block_id"]]["target_group"]]["language"]
                self.assertNotEqual(tgt_a, tgt_b)
                if part == "diagonal":
                    self.assertEqual(src_a, tgt_a)
                    self.assertEqual(src_b, tgt_b)
                else:
                    self.assertEqual(src_a, part)
                    self.assertEqual(src_b, part)

            # Deep mathematical verification on one comparison pair across all three strata & baselines.
            first_pair = probe_unstrat[0][1]["inputs"]
            matching = [(cid, comp) for cid, comp in unit["comparisons"].items()
                        if comp["inputs"] == first_pair]
            self.assertEqual(len(matching), 9)  # 3 baselines x 3 strata

            pred_a = blocks[first_pair[0]["block_id"]][first_pair[0]["route_id"]]
            pred_b = blocks[first_pair[1]["block_id"]][first_pair[1]["route_id"]]
            ids, (idx_a, idx_b) = paired_image_alignment(plan, pred_a, pred_b)
            y_all, known_all = targets_for_prediction(plan, pred_a)
            y_al, known_al = y_all[idx_a], known_all[idx_a]

            for cid, comp in matching:
                baseline, label = comp["baseline"], comp["label"]
                if baseline == "probe":
                    d_a = probabilities(pred_a.logits[idx_a]) >= 0.5
                    d_b = probabilities(pred_b.logits[idx_b]) >= 0.5
                elif baseline == "fit_prior":
                    d_a = np.broadcast_to(pred_a.prevalence >= 0.5, y_al.shape)
                    d_b = np.broadcast_to(pred_b.prevalence >= 0.5, y_al.shape)
                else:
                    d_a = np.zeros(y_al.shape, dtype=bool)
                    d_b = np.zeros(y_al.shape, dtype=bool)

                null_mat = []
                for col, cat in enumerate(categories):
                    expected_vals, null_vec = _independent_e25_point_and_null(
                        y_al[:, col], known_al[:, col], d_a[:, col], d_b[:, col], ids, cat, label
                    )
                    if null_vec is not None:
                        null_mat.append(null_vec)
                    for metric, exp_val in expected_vals.items():
                        row = by_key[(cid, cat, metric)]
                        if exp_val is None:
                            self.assertIsNone(row["value"])
                            self.assertEqual(row["status"], "undefined")
                        else:
                            self.assertAlmostEqual(row["value"], exp_val, places=12)
                        # Verify plant.n.02 (index 16) excludes the unknown cell on SCREEN.
                        if cat == "plant.n.02" and label is None:
                            self.assertEqual(row["support"]["known"], len(ids) - 1)

                # Verify primary_macro null quantiles average across primary categories per permutation BEFORE quantiles.
                if label in (0, 1):
                    macro_null = np.mean(np.stack([null_mat[i] for i in primary_indices], axis=0), axis=0)
                    exp_lo, exp_hi = np.quantile(macro_null, [0.025, 0.975], method="linear")
                    self.assertAlmostEqual(by_key[(cid, MACRO, "null_lower")]["value"], float(exp_lo), places=12)
                    self.assertAlmostEqual(by_key[(cid, MACRO, "null_upper")]["value"], float(exp_hi), places=12)
                    self.assertNotIn("interval", by_key[(cid, MACRO, "null_lower")])

                # Verify bootstrap interval on category 0 against explicit draw expansion.
                y0, k0, da0, db0 = y_al[:, 0], known_al[:, 0], d_a[:, 0], d_b[:, 0]
                exp_draws = []
                for draw in draws:
                    if label is None:
                        m = k0[draw]
                        val = float((da0[draw][m] != db0[draw][m]).mean()) if m.any() else np.nan
                    else:
                        m = k0[draw] & (y0[draw] == bool(label))
                        if not m.any():
                            val = np.nan
                        else:
                            ea = da0[draw][m] != bool(label)
                            eb = db0[draw][m] != bool(label)
                            val = float((ea & eb).mean() - ea.mean() * eb.mean())
                    exp_draws.append(val)
                exp_arr = np.asarray(exp_draws, dtype="f8")
                finite = exp_arr[np.isfinite(exp_arr)]
                check_metric = "disagreement" if label is None else "excess"
                got_iv = by_key[(cid, categories[0], check_metric)]["interval"]
                self.assertEqual(got_iv["valid_draws"], int(finite.size))
                self.assertEqual(got_iv["total_draws"], len(draws))
                if finite.size:
                    lo, hi = np.quantile(finite, [0.025, 0.975], method="linear")
                    assert_allclose([got_iv["lower"], got_iv["upper"]], [lo, hi], rtol=1e-12, atol=1e-12)

    def test_02_unresolved_low_support_constant_and_unestimable_handling(self):
        plan = self.plan
        uid = "E25/qwen/en"
        unit = analysis_units(plan)[uid]
        blocks = {bid: load_prediction_block(plan, ref, expected_block=bid)
                  for bid, ref in self.pred_refs["qwen"].items()}

        # Case A: always_absent on label=0 has zero errors -> conditionals undefined, macro undefined.
        rows = compute_e25_unit(plan, uid, unit, blocks)
        by_key = {(r["comparison_id"], r["category"], r["statistic"]): r for r in rows}
        aa_zero = next(cid for cid, comp in unit["comparisons"].items()
                       if comp["baseline"] == "always_absent" and comp["label"] == 0)
        for metric in ("error_b_given_a", "error_a_given_b"):
            row_cat = by_key[(aa_zero, plan["categories"][0], metric)]
            self.assertIsNone(row_cat["value"])
            self.assertEqual(row_cat["status"], "undefined")
            self.assertEqual(row_cat["reason"], "zero_conditional_denominator")
            self.assertEqual(row_cat["support"]["denominator"], 0)
            row_macro = by_key[(aa_zero, MACRO, metric)]
            self.assertIsNone(row_macro["value"])
            self.assertEqual(row_macro["status"], "undefined")
            self.assertEqual(row_macro["reason"], "unresolved_primary_category")
            self.assertEqual(row_macro["support"]["undefined_categories"], len(plan["primary"]))
        # Meanwhile error_a/common_wrong/excess/null are finite 0.0.
        for metric in ("error_a", "error_b", "common_wrong", "independence", "excess", "null_lower", "null_upper"):
            self.assertEqual(by_key[(aa_zero, plan["categories"][0], metric)]["value"], 0.0)

        # Case B: higher minimum support triggers descriptive_low_support while keeping finite values and intervals.
        counts = bootstrap_counts(load_bootstrap(plan), len(plan["screen_ids"]))
        with patch.dict(plan.protocol["minimum_positive_and_negative"], {"SCREEN": 20}):
            low_rows = compute_e25_unit(plan, uid, unit, blocks, counts=counts)
        low_by_key = {(r["comparison_id"], r["category"], r["statistic"]): r for r in low_rows}
        probe_unstrat = next(cid for cid, comp in unit["comparisons"].items()
                             if comp["baseline"] == "probe" and comp["label"] is None)
        row_low = low_by_key[(probe_unstrat, plan["categories"][0], "disagreement")]
        self.assertEqual(row_low["status"], "descriptive_low_support")
        self.assertIsNotNone(row_low["value"])
        self.assertIn("interval", row_low)
        self.assertEqual(low_by_key[(probe_unstrat, MACRO, "disagreement")]["status"], "descriptive_low_support")

        # Case C: unestimable head on a primary category makes probe undefined (ignoring padding) but keeps baselines.
        primary_col = plan["categories"].index(plan["primary"][0])
        patched_blocks = {}
        for bid, routes in blocks.items():
            patched_routes = {}
            for rid, pred in routes.items():
                est = pred.estimable.copy()
                logits = pred.logits.copy()
                est[primary_col] = False
                logits[:, primary_col] = 999.0  # Poison padding that must never be read.
                patched_routes[rid] = type(pred)(
                    pred.reference, pred.route, pred.split, pred.rows, logits, est, pred.prevalence
                )
            patched_blocks[bid] = patched_routes

        unest_rows = compute_e25_unit(plan, uid, unit, patched_blocks)
        u_key = {(r["comparison_id"], r["category"], r["statistic"]): r for r in unest_rows}
        probe_l1 = next(cid for cid, comp in unit["comparisons"].items()
                        if comp["baseline"] == "probe" and comp["label"] == 1)
        prior_l1 = next(cid for cid, comp in unit["comparisons"].items()
                        if comp["baseline"] == "fit_prior" and comp["label"] == 1)
        self.assertEqual(u_key[(probe_l1, plan["primary"][0], "common_wrong")]["status"], "undefined")
        self.assertEqual(u_key[(probe_l1, plan["primary"][0], "common_wrong")]["reason"], "unestimable_head")
        self.assertEqual(u_key[(probe_l1, MACRO, "common_wrong")]["status"], "undefined")
        self.assertEqual(u_key[(probe_l1, MACRO, "null_lower")]["status"], "undefined")
        self.assertEqual(u_key[(probe_l1, MACRO, "common_wrong")]["support"]["undefined_categories"], 1)
        self.assertIsNotNone(u_key[(prior_l1, plan["primary"][0], "common_wrong")]["value"])

    def test_03_checkpoint_resume_and_guard_rejections(self):
        plan = self.plan
        qwen_preds = list(self.pred_refs["qwen"].values())
        gemma_preds = list(self.pred_refs["gemma"].values())

        # Partial run: checkpoint only en and hi.
        partial = run_e25(plan, "qwen", qwen_preds, self.fixture.producer, parts=["en", "hi"])
        self.assertEqual(partial["published"], ["E25/qwen/en", "E25/qwen/hi"])
        self.assertEqual(partial["reused"], [])

        # Resume full run: reuse en and hi verbatim, publish remaining 3 units.
        resumed = run_e25(plan, "qwen", qwen_preds, self.fixture.producer, resume_refs=partial["analyses"])
        self.assertEqual(resumed["reused"], ["E25/qwen/en", "E25/qwen/hi"])
        self.assertEqual(resumed["published"], ["E25/qwen/bn", "E25/qwen/te", "E25/qwen/diagonal"])
        self.assertEqual(resumed["units"]["E25/qwen/en"], partial["units"]["E25/qwen/en"])
        self.assertEqual(resumed["units"]["E25/qwen/hi"], partial["units"]["E25/qwen/hi"])

        # Foreign model resume ref must be rejected.
        gemma_en = run_e25(plan, "gemma", gemma_preds, self.fixture.producer, parts=["en"])
        with self.assertRaisesRegex(ContractError, "Foreign resumed analysis unit"):
            run_e25(plan, "qwen", qwen_preds, self.fixture.producer, resume_refs=gemma_en["analyses"])

        # Duplicate resume ref must be rejected.
        with self.assertRaisesRegex(ContractError, "Duplicate resumed E25 unit"):
            run_e25(plan, "qwen", qwen_preds, self.fixture.producer,
                    resume_refs=[partial["analyses"][0], partial["analyses"][0]])

        # Missing required prediction block must be rejected.
        with self.assertRaisesRegex(ContractError, "Missing required prediction blocks"):
            run_e25(plan, "qwen", qwen_preds[:-1], self.fixture.producer)

        # Mismatched prediction reference on resume (freshly re-scored block) must be rejected.
        one_bid = next(iter(self.pred_refs["qwen"]))
        fresh_ref = score_and_publish_block(plan, "qwen", one_bid, self.fixture.producer)
        swapped = [fresh_ref if bid == one_bid else ref for bid, ref in self.pred_refs["qwen"].items()]
        with self.assertRaisesRegex(ContractError, "does not match current prediction references"):
            run_e25(plan, "qwen", swapped, self.fixture.producer, resume_refs=partial["analyses"])

    def test_04_cli_and_notebook_checks(self):
        # Verify notebook JSON structure and compile all code cells.
        nb_path = REPO / "scripts" / "stage4" / "run_e25.ipynb"
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertEqual(nb["nbformat"], 4)
        code_cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        self.assertEqual(len(code_cells), 3)
        for idx, source in enumerate(code_cells):
            compile(source, f"run_e25.ipynb#cell{idx}", "exec")
        combined = "\n".join(code_cells)
        self.assertIn("OPENBLAS_NUM_THREADS", combined)
        self.assertIn("MODEL = 'qwen'", combined)
        self.assertIn("src.stage4.run_e25", combined)

        # Verify CLI end-to-end execution with --predictions-file, --part, --output-manifest, and --resume-file.
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            preds_file = tmp_path / "preds.json"
            preds_file.write_text(json.dumps({"predictions": list(self.pred_refs["qwen"].values())}), encoding="utf-8")
            out1 = tmp_path / "out1.json"
            out2 = tmp_path / "out2.json"

            env = os.environ.copy()
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                env[name] = "1"
            # Patch DEPTHS in subprocess via a tiny -c wrapper calling run_e25.main() with fixture protocol.
            protocol_path = tmp_path / "protocol.json"
            protocol_path.write_text(json.dumps(self.fixture.protocol), encoding="utf-8")
            producer_path = tmp_path / "producer.json"
            producer_path.write_text(json.dumps(self.fixture.producer), encoding="utf-8")
            runner = (
                "import json, sys\n"
                "from pathlib import Path\n"
                "from unittest.mock import patch\n"
                "import src.stage4.run_e25 as mod\n"
                f"proto = json.loads(Path({str(protocol_path)!r}).read_text(encoding='utf-8'))\n"
                f"producer = json.loads(Path({str(producer_path)!r}).read_text(encoding='utf-8'))\n"
                "orig_read = Path.read_text\n"
                "def fake_read(self, *a, **kw):\n"
                "    if self.name == 'protocol.json':\n"
                "        return json.dumps(proto)\n"
                "    return orig_read(self, *a, **kw)\n"
                "with patch.dict('src.stage4.plan.DEPTHS', {'qwen': (1, 2, 3, 4), 'gemma': (1, 2, 3, 4)}), "
                "patch.object(Path, 'read_text', fake_read), "
                "patch.object(mod, 'current_producer', return_value=producer):\n"
                "    mod.main()\n"
            )
            cmd1 = [
                sys.executable, "-c", runner,
                "--phase1-root", str(self.fixture.root),
                "--manifest", str(self.fixture.root / self.plan.reference["path"]),
                "--model", "qwen",
                "--predictions-file", str(preds_file),
                "--part", "diagonal",
                "--output-manifest", str(out1),
                "--storage-scope", "local-development",
            ]
            proc1 = subprocess.run(cmd1, cwd=REPO, env=env, capture_output=True, text=True, check=True)
            events1 = [json.loads(line) for line in proc1.stdout.splitlines() if line.strip()]
            self.assertEqual(events1[-1]["event"], "e25_worker_complete")
            self.assertEqual(events1[-1]["published"], ["E25/qwen/diagonal"])

            cmd2 = [
                sys.executable, "-c", runner,
                "--phase1-root", str(self.fixture.root),
                "--manifest", str(self.fixture.root / self.plan.reference["path"]),
                "--model", "qwen",
                "--predictions-file", str(preds_file),
                "--resume-file", str(out1),
                "--part", "diagonal",
                "--output-manifest", str(out2),
                "--storage-scope", "local-development",
            ]
            proc2 = subprocess.run(cmd2, cwd=REPO, env=env, capture_output=True, text=True, check=True)
            events2 = [json.loads(line) for line in proc2.stdout.splitlines() if line.strip()]
            self.assertEqual(events2[-1]["reused"], ["E25/qwen/diagonal"])
            self.assertEqual(events2[-1]["published"], [])


if __name__ == "__main__":
    unittest.main()
