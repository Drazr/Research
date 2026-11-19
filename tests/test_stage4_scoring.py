"""Fixture tests for the Stage 4 canonical scoring worker and CPU notebook."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from src.common.contracts import ContractError, fingerprint
from src.stage2.schema import REPO, plain
from src.stage3.artifacts import write
from src.stage3.data import CS
from src.stage3.readouts import expected_rows, stats_identity
from src.stage4.interfaces import (
    load_prediction_block,
    load_stage4_plan,
    prediction_index,
    prepare_stage4,
    scoring_blocks,
)
from src.stage4.run_scoring import (
    load_resume_references,
    main,
    representative_blocks,
    resolve_owner,
    run_scoring,
    select_blocks,
)
from test_stage3_foundation import Fixture


class FullStage4Fixture:
    """Synthetic accepted Stage 3 registry with all 145 ordinary systems scored."""

    def __init__(self, tmpdir):
        self.fixture = Fixture(Path(tmpdir))
        self.fixture.protocol["minimum_positive_and_negative"] = dict.fromkeys(
            ("FIT", "DEV-S", "DEV-C", "SCREEN"), 1
        )
        self.fit = self.fixture.build()
        self.unestimable_col = 24
        self.registry = self._build_registry()

    def _build_registry(self):
        f, fit = self.fixture, self.fit
        selections, candidates = [], []
        for g_idx, gid in enumerate(sorted(fit["groups"])):
            group = fit["groups"][gid]
            dim = group["dimension"]
            dense = group["source"] == "vlm" or group.get("recipe", {}).get("family") == "length"
            for assignment in ("seed0", "seed1", "seed2", "true"):
                def placeholder(label):
                    return {"path": f"fixture/{gid}/{assignment}/{label}/_COMPLETE.json", "sha256": "0" * 64}

                block = [placeholder(str(c)) for c in CS]
                selected = placeholder("selection")
                if assignment == "true":
                    rows = expected_rows(fit, gid, "FIT", fit.store)
                    mean = np.linspace(0.25, 0.75, dim, dtype=np.float64)
                    scale = np.linspace(0.5, 1.0, dim, dtype=np.float64)
                    variance = np.where(scale > 0, scale ** 2, 0.0)
                    stats = write(
                        fit,
                        "fixture/scalers",
                        "statistics",
                        {"group_id": gid, "input_identity": stats_identity(fit, gid, rows), "center": dense},
                        {"mean": mean, "variance": variance, "scale": scale},
                        f.producer,
                    )
                    coef = np.tile(np.linspace(1.0, 2.0, dim, dtype=np.float64) + 0.01 * (g_idx % 7), (25, 1))
                    intercept = np.full(25, 0.25, dtype=np.float64)
                    estimable = np.ones(25, dtype=bool)
                    records = [{"status": "fitted", "attempts": [{"converged": True}]} for _ in range(25)]
                    # Exercise unestimable-head masking and zero-padding on one category.
                    coef[self.unestimable_col, :] = 0.0
                    intercept[self.unestimable_col] = 0.0
                    estimable[self.unestimable_col] = False
                    records[self.unestimable_col] = {"status": "unestimable", "attempts": []}
                    block[0] = plain(
                        write(
                            fit,
                            "fixture/fits",
                            "candidate",
                            {
                                "group_id": gid,
                                "assignment_id": "true",
                                "C": CS[0],
                                "fit_rows": fingerprint(rows),
                                "statistics": plain(stats),
                                "records": records,
                            },
                            {"coefficients": coef, "intercepts": intercept, "estimable": estimable},
                            f.producer,
                        )
                    )
                    selected = plain(
                        write(
                            fit,
                            "fixture/selections",
                            "selection",
                            {
                                "group_id": gid,
                                "assignment_id": "true",
                                "candidates": block,
                                "selected": block[0],
                                "selected_C": CS[0],
                            },
                            {},
                            f.producer,
                        )
                    )
                candidates.extend(block)
                selections.append(selected)
        gate = write(
            fit,
            "fixture/gate",
            "g0_gate",
            {"flagged": False, "p": 0.362, "structural_checks": "passed", "groups": {g: None for g in fit["groups"]}},
            {},
            f.producer,
        )
        body = {
            "suite": plain(fit["suite"]),
            "candidates": candidates,
            "selections": selections,
            "g0_gate": plain(gate),
            "stage2_groups": plain(fit["groups"]),
            "ordinary_systems": 145,
            "stage3_complete": False,
        }
        pending = write(fit, "fixture/registry", "registry_readback", body, {}, f.producer)
        return write(
            fit,
            "fixture/registry",
            "system_registry",
            dict(body, stage3_complete=True, independent_readback=plain(pending)),
            {},
            f.producer,
        )


class Stage4ScoringWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.depths = patch.dict("src.stage4.plan.DEPTHS", {"qwen": (1, 2, 3, 4), "gemma": (1, 2, 3, 4)})
        cls.depths.start()
        cls.addClassCleanup(cls.depths.stop)
        cls.bundle = FullStage4Fixture(cls.tmp.name)
        cls.fixture = cls.bundle.fixture
        cls.plan = prepare_stage4(cls.fixture.root, cls.fixture.protocol, cls.bundle.registry, cls.fixture.producer)

    def test_owner_selector_and_scope_validation(self):
        self.assertEqual(resolve_owner("model", "qwen"), "qwen")
        self.assertEqual(resolve_owner("model", "gemma"), "gemma")
        self.assertEqual(resolve_owner("e03", None), "e03")
        with self.assertRaisesRegex(ContractError, "MODEL must be 'qwen' or 'gemma'"):
            resolve_owner("model", None)
        with self.assertRaisesRegex(ContractError, "MODEL must be omitted/None"):
            resolve_owner("e03", "qwen")
        with self.assertRaisesRegex(ContractError, "MODE must be"):
            resolve_owner("invalid", "qwen")
        with self.assertRaisesRegex(ContractError, "no priority-0 blocks"):
            select_blocks(self.plan, "e03", "priority")

        # Representative blocks are deterministic metadata selections.
        for model in ("qwen", "gemma"):
            rep = representative_blocks(self.plan, model)
            self.assertEqual(len(rep), 1)
            blk = self.plan["blocks"][rep[0]]
            self.assertEqual((blk["owner"], blk["priority"], blk["split"]), (model, 0, "SCREEN"))
            self.assertEqual(self.plan.fit["groups"][blk["target_group"]]["language"], "en")
        e03_rep = representative_blocks(self.plan, "e03")
        self.assertEqual(len(e03_rep), 1)
        self.assertEqual(
            self.plan.fit["groups"][self.plan["blocks"][e03_rep[0]]["target_group"]]["recipe"]["family"],
            "pooled_char_tfidf",
        )

    def test_full_inventory_priority_order_and_handoff(self):
        all_refs = []
        for mode, model, expected_blocks, expected_screen, expected_devc in (
            ("model", "qwen", 70, 258, 4),
            ("model", "gemma", 70, 258, 4),
            ("e03", None, 13, 13, 0),
        ):
            events = []
            summary = run_scoring(
                self.plan,
                mode,
                self.fixture.producer,
                model=model,
                scope="all",
                report=lambda ev, **kw: events.append((ev, kw)),
                heartbeat_seconds=60.0,
            )
            self.assertEqual(summary["status"], "passed")
            self.assertFalse(summary["stage4_complete"])
            self.assertEqual(summary["blocks_published"], expected_blocks)
            self.assertEqual(summary["blocks_reused"], 0)
            self.assertEqual(summary["routes_scheduled"], {"SCREEN": expected_screen, "DEV-C": expected_devc})
            self.assertEqual(len(summary["predictions"]), expected_blocks)
            all_refs.extend(summary["predictions"])

            published_events = [kw for ev, kw in events if ev == "block_published"]
            self.assertEqual(len(published_events), expected_blocks)
            if mode == "model":
                # First 8 published blocks are strictly Priority 0 (4 midpoint SCREEN + 4 DEV-C diagonals).
                first_eight = published_events[:8]
                self.assertTrue(all(item["priority"] == 0 for item in first_eight))
                self.assertEqual(sum(1 for item in first_eight if item["split"] == "SCREEN"), 4)
                self.assertEqual(sum(1 for item in first_eight if item["split"] == "DEV-C"), 4)
                self.assertTrue(all(item["priority"] == 1 for item in published_events[8:]))

                # priority_handoff must fire immediately after the 8th block, before any Priority 1 block.
                handoff_idx = next(i for i, (ev, _) in enumerate(events) if ev == "priority_handoff")
                first_p1_idx = next(
                    i for i, (ev, kw) in enumerate(events)
                    if ev == "block_score_publish_start" and kw["priority"] == 1
                )
                self.assertLess(handoff_idx, first_p1_idx)
                handoff = events[handoff_idx][1]
                self.assertEqual(handoff["midpoint_screen_blocks"], 4)
                self.assertEqual(handoff["dev_c_diagonal_blocks"], 4)
                self.assertEqual(handoff["ready_for"], ["E19", "E25"])
                self.assertTrue(summary["priority_handoff_ready"])
            else:
                self.assertFalse(any(ev == "priority_handoff" for ev, _ in events))
                self.assertFalse(summary["priority_handoff_ready"])

        # Combined inventory covers all 153 blocks: 529 SCREEN + 8 DEV-C slots.
        indexed = prediction_index(self.plan, all_refs, required_blocks=self.plan["blocks"])
        self.assertEqual(len(indexed), 153)
        screen_slots = sum(
            len(routes) for bid, routes in indexed.items() if self.plan["blocks"][bid]["split"] == "SCREEN"
        )
        devc_slots = sum(
            len(routes) for bid, routes in indexed.items() if self.plan["blocks"][bid]["split"] == "DEV-C"
        )
        self.assertEqual(screen_slots, 529)
        self.assertEqual(devc_slots, 8)

    def test_dense_sparse_arithmetic_row_joins_and_single_target_read(self):
        # Fresh plan instance so plan.store.reads starts at zero.
        fresh_plan = load_stage4_plan(self.fixture.root, self.fixture.protocol, self.plan.reference)
        qwen_rep = representative_blocks(fresh_plan, "qwen")[0]
        e03_rep = representative_blocks(fresh_plan, "e03")[0]

        reads_before = dict(fresh_plan.store.reads)
        qwen_summary = run_scoring(
            fresh_plan, "model", self.fixture.producer, model="qwen", scope="profile", report=lambda *a, **k: None
        )
        # Verify the target feature array for the 4-source-head block was read from disk exactly once.
        new_reads = {
            k: v - reads_before.get(k, 0)
            for k, v in fresh_plan.store.reads.items()
            if v > reads_before.get(k, 0)
        }
        vec_reads = {k: v for k, v in new_reads.items() if Path(k).name.startswith("site_") and k.endswith(".npy")}
        self.assertEqual(len(vec_reads), 1, f"Expected one target site .npy read, got: {vec_reads}")
        self.assertTrue(all(count == 1 for count in vec_reads.values()), f"Expected single read: {vec_reads}")

        # Verify dense decoder logits, exact row order, and unestimable padding.
        target_gid = fresh_plan["blocks"][qwen_rep]["target_group"]
        qwen_ref = qwen_summary["references_by_block"][qwen_rep]
        qwen_views = load_prediction_block(fresh_plan, qwen_ref, expected_block=qwen_rep)
        self.assertEqual(len(qwen_views), 4)
        exp_rows = expected_rows(fresh_plan.fit, target_gid, "SCREEN", fresh_plan.store)
        for route in fresh_plan["blocks"][qwen_rep]["routes"]:
            pred = qwen_views[route["route_id"]]
            self.assertEqual(plain(pred.rows), exp_rows)
            self.assertFalse(bool(pred.estimable[self.bundle.unestimable_col]))
            assert_array_equal(pred.logits[:, self.bundle.unestimable_col], np.zeros(len(exp_rows)))
            self.assertTrue(np.isfinite(pred.logits).all())
            # Independent dense calculation on fixture synthetic vectors: [id%2, 1.0].
            src_gid = route["source_group"]
            src_idx = sorted(fresh_plan.fit["groups"]).index(src_gid)
            c0 = 1.0 + 0.01 * (src_idx % 7)
            c1 = 2.0 + 0.01 * (src_idx % 7)
            x0 = np.array([int(r["image_id"], 16) % 2 for r in pred.rows], dtype=np.float64)
            x1 = np.ones(len(pred.rows), dtype=np.float64)
            expected_logit = ((x0 - 0.25) / 0.5) * c0 + ((x1 - 0.75) / 1.0) * c1 + 0.25
            assert_allclose(pred.logits[:, 0], expected_logit)

        # Verify sparse pooled E03 scoring and exact row join (4 views per SCREEN image).
        e03_summary = run_scoring(
            fresh_plan, "e03", self.fixture.producer, scope="profile", report=lambda *a, **k: None
        )
        e03_ref = e03_summary["references_by_block"][e03_rep]
        e03_views = load_prediction_block(fresh_plan, e03_ref, expected_block=e03_rep)
        self.assertEqual(len(e03_views), 1)
        e03_pred = next(iter(e03_views.values()))
        e03_gid = fresh_plan["blocks"][e03_rep]["target_group"]
        self.assertEqual(plain(e03_pred.rows), expected_rows(fresh_plan.fit, e03_gid, "SCREEN", fresh_plan.store))
        e03_idx = sorted(fresh_plan.fit["groups"]).index(e03_gid)
        ec0 = 1.0 + 0.01 * (e03_idx % 7)
        ec1 = 2.0 + 0.01 * (e03_idx % 7)
        val = np.float64(np.float32(1.0 / np.sqrt(2.0)))
        expected_sparse = (val / 0.5) * ec0 + (val / 1.0) * ec1 + 0.25
        assert_allclose(e03_pred.logits[:, 0], expected_sparse)
        assert_array_equal(e03_pred.logits[:, self.bundle.unestimable_col], np.zeros(len(e03_pred.rows)))

    def test_interruption_resume_deduplication_and_guards(self):
        with tempfile.TemporaryDirectory() as out_dir:
            inv_path = Path(out_dir) / "qwen_partial.json"
            # Step 1: Run profile scope (1 block), then interrupt during priority scope after 3 more blocks.
            prof = run_scoring(
                self.plan, "model", self.fixture.producer, model="qwen", scope="profile", report=lambda *a, **k: None
            )
            calls = []
            from src.stage4 import run_scoring as mod

            real_score = mod.score_and_publish_block

            def interrupt_on_fourth(plan, owner, block_id, producer):
                calls.append(block_id)
                if len(calls) == 4:
                    raise KeyboardInterrupt("simulated interrupt")
                return real_score(plan, owner, block_id, producer)

            events = []
            with patch.object(mod, "score_and_publish_block", side_effect=interrupt_on_fourth):
                with self.assertRaises(KeyboardInterrupt):
                    run_scoring(
                        self.plan,
                        "model",
                        self.fixture.producer,
                        model="qwen",
                        scope="priority",
                        resume_refs=prof["predictions"],
                        output_inventory=inv_path,
                        report=lambda ev, **kw: events.append((ev, kw)),
                    )
            self.assertTrue(any(ev == "worker_interrupted" for ev, _ in events))
            self.assertTrue(inv_path.is_file())
            saved = json.loads(inv_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "interrupted")
            self.assertEqual(len(saved["predictions"]), 4)  # 1 reused from profile + 3 newly published

            # Step 2: Resume priority scope passing overlapping duplicate identical refs; verify only remaining 4 run.
            resume_calls = []

            def track_score(plan, owner, block_id, producer):
                resume_calls.append(block_id)
                return real_score(plan, owner, block_id, producer)

            with patch.object(mod, "score_and_publish_block", side_effect=track_score):
                resumed = run_scoring(
                    self.plan,
                    "model",
                    self.fixture.producer,
                    model="qwen",
                    scope="priority",
                    resume_refs=saved["predictions"] + prof["predictions"],  # overlapping exact refs
                    output_inventory=inv_path,
                    report=lambda *a, **k: None,
                )
            self.assertEqual(resumed["blocks_reused"], 4)
            self.assertEqual(resumed["blocks_published"], 4)
            self.assertEqual(len(resume_calls), 4)
            self.assertTrue(resumed["priority_handoff_ready"])

            # Step 3: Guard against foreign ownership in resume refs.
            e03_prof = run_scoring(
                self.plan, "e03", self.fixture.producer, scope="profile", report=lambda *a, **k: None
            )
            with self.assertRaisesRegex(ContractError, "Foreign resume prediction block"):
                load_resume_references(self.plan, "qwen", e03_prof["predictions"])

            # Step 4: Guard against conflicting duplicate block IDs (two different runs for the same block).
            qwen_rep_2 = run_scoring(
                self.plan, "model", self.fixture.producer, model="qwen", scope="profile", report=lambda *a, **k: None
            )
            with self.assertRaisesRegex(ContractError, "Duplicate prediction block"):
                load_resume_references(self.plan, "qwen", prof["predictions"] + qwen_rep_2["predictions"])

            # Step 5: Guard against corrupt payload bytes on fresh-process resume.
            corrupt_ref = qwen_rep_2["predictions"][0]
            logits_file = (self.fixture.root / corrupt_ref["path"]).with_name("logits.npy")
            original_bytes = logits_file.read_bytes()
            try:
                logits_file.write_bytes(b"corrupt")
                fresh_plan = load_stage4_plan(self.fixture.root, self.fixture.protocol, self.plan.reference)
                with self.assertRaisesRegex(ContractError, "Changed/corrupt payload"):
                    load_resume_references(fresh_plan, "qwen", [corrupt_ref])
            finally:
                logits_file.write_bytes(original_bytes)

    def test_cli_entry_point_and_notebook_contract(self):
        with tempfile.TemporaryDirectory() as out_dir:
            inv_path = Path(out_dir) / "e03_profile.json"
            buf = io.StringIO()
            with patch("src.stage4.run_scoring.REPO", self.fixture.root):
                # Write fixture protocol where CLI expects REPO/configs/protocol.json.
                proto_dir = self.fixture.root / "configs"
                proto_dir.mkdir(parents=True, exist_ok=True)
                (proto_dir / "protocol.json").write_text(
                    json.dumps(self.fixture.protocol), encoding="utf-8"
                )
                with patch("src.stage4.run_scoring.current_producer", return_value=self.fixture.producer):
                    with redirect_stdout(buf):
                        main(
                            [
                                "--phase1-root",
                                str(self.fixture.root),
                                "--manifest",
                                json.dumps(plain(self.plan.reference)),
                                "--mode",
                                "e03",
                                "--scope",
                                "profile",
                                "--output-inventory",
                                str(inv_path),
                                "--storage-scope",
                                "local-development",
                            ]
                        )
            lines = [json.loads(line) for line in buf.getvalue().splitlines() if line.strip()]
            self.assertTrue(any(item["event"] == "worker_complete" for item in lines))
            saved = json.loads(inv_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["owner"], "e03")
            self.assertEqual(saved["blocks_published"], 1)

            # Resume all E03 blocks via --resume-file and verify the 1 profiled block is reused.
            buf2 = io.StringIO()
            with patch("src.stage4.run_scoring.REPO", self.fixture.root):
                with patch("src.stage4.run_scoring.current_producer", return_value=self.fixture.producer):
                    with redirect_stdout(buf2):
                        main(
                            [
                                "--phase1-root",
                                str(self.fixture.root),
                                "--manifest",
                                str(self.fixture.root / self.plan.reference["path"]),
                                "--mode",
                                "e03",
                                "--scope",
                                "all",
                                "--resume-file",
                                str(inv_path),
                                "--output-inventory",
                                str(inv_path),
                                "--storage-scope",
                                "local-development",
                            ]
                        )
            completed = json.loads(inv_path.read_text(encoding="utf-8"))
            self.assertEqual(completed["blocks_reused"], 1)
            self.assertEqual(completed["blocks_published"], 12)
            self.assertEqual(len(completed["predictions"]), 13)

        # Verify CPU notebook structure and contract.
        nb_path = REPO / "scripts/stage4/run_scoring.ipynb"
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertEqual(nb["nbformat"], 4)
        self.assertEqual(len(nb["cells"]), 4)
        joined = "\n".join("".join(cell["source"]) for cell in nb["cells"])
        for token in (
            "OPENBLAS_NUM_THREADS",
            "STAGE4_PLAN",
            "MODE",
            "MODEL",
            "SCOPE",
            "RESUME_REFS",
            "RESUME_FILES",
            "OUTPUT_INVENTORY",
            "src.stage4.run_scoring",
        ):
            self.assertIn(token, joined)


if __name__ == "__main__":
    unittest.main()
