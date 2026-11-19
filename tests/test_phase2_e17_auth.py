"""Fixture and contract tests for Phase 2 Stage 1 E17 DEV-C preflight & authorization."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.common.contracts import ContractError, LANGUAGES, fingerprint
from src.phase2.e17_auth import (
    E17_BFGS_GTOL,
    E17_BFGS_MAXITER,
    E17_DEV_C_PERMISSION,
    E17_INVARIANCE_TOL,
    authorize_e17_dev_c_route,
    authorize_e17_operation,
    e17_calibrator_recipe,
    enumerate_e17_inventory,
    load_e17_auth,
    preflight_e17_dev_c,
    require_e17_dev_c_split,
    validate_target_calibrator_input,
    verify_diagonal_reproduction,
    verify_e17_auth_resume,
)
from src.phase2.interfaces import (
    KIND_E17_AUTH,
    MODELS,
    PHASE2_VERSION,
    Phase1Binding,
    REPO,
    SCHEMA_E17_AUTH,
    frozen,
    init_phase2_dirs,
    plain,
)

try:
    import numpy as np
except ImportError:
    np = None


def _stub_ref(rel_path):
    return {"path": rel_path, "sha256": fingerprint(rel_path)}


def _build_mock_binding(root, protocol, *, lineage_tag="v1"):
    """Build a lightweight Phase1Binding fixture with production midpoints (Qwen 18, Gemma 17)."""
    midpoints = {"qwen": 18, "gemma": 17}
    groups = {}
    ordinary = {}
    blocks = {}
    predictions = {}
    candidates = {}

    for model in MODELS:
        depth = midpoints[model]
        site = f"decoder:{depth}:last_caption"
        dim = 4 if model == "qwen" else 6
        model_plan_ref = _stub_ref(f"cache/stage2/{model}/model_plan/_COMPLETE.json")
        gids = {}
        for lang in LANGUAGES:
            gid = fingerprint(f"{model}|image_caption|{site}|{lang}")
            gids[lang] = gid
            groups[gid] = {
                "group_id": gid,
                "name": f"{model}|image_caption|{site}|{lang}",
                "source": "vlm",
                "model": model,
                "condition": "image_caption",
                "site": site,
                "language": lang,
                "dimension": dim,
                "model_plan": model_plan_ref,
                "releases": {
                    split: _stub_ref(f"features/{model}/image_caption/{split}/{lang}/_COMPLETE.json")
                    for split in ("FIT", "DEV-S", "DEV-C", "SCREEN")
                },
            }
            cand_ref = _stub_ref(f"checkpoints/stage3/{model}/{lang}/candidate/_COMPLETE.json")
            sel_ref = _stub_ref(f"checkpoints/stage3/{model}/{lang}/selection/_COMPLETE.json")
            stat_ref = _stub_ref(f"checkpoints/stage3/{model}/{lang}/statistics/_COMPLETE.json")
            ordinary[gid] = {"selection": sel_ref, "candidate": cand_ref, "C": 1.0}
            candidates[cand_ref["path"]] = {
                "group_id": gid,
                "assignment_id": "true",
                "C": 1.0,
                "statistics": stat_ref,
            }

        for target_lang in LANGUAGES:
            target_gid = gids[target_lang]
            # Stage 4 SCREEN block (all 4 source languages)
            scr_bid = fingerprint(["stage4.plan.v1", target_gid, "SCREEN"])
            blocks[scr_bid] = {
                "target_group": target_gid,
                "split": "SCREEN",
                "owner": model,
                "priority": 0,
                "routes": [
                    {
                        "source_group": gids[src_lang],
                        "route_id": fingerprint(["stage4.plan.v1", gids[src_lang], target_gid, "SCREEN"]),
                    }
                    for src_lang in LANGUAGES
                ],
            }
            predictions[scr_bid] = _stub_ref(f"predictions/stage4/{model}/screen_{target_lang}/_COMPLETE.json")

            # Stage 4 DEV-C diagonal block (source == target only)
            dev_bid = fingerprint(["stage4.plan.v1", target_gid, "DEV-C"])
            blocks[dev_bid] = {
                "target_group": target_gid,
                "split": "DEV-C",
                "owner": model,
                "priority": 0,
                "routes": [
                    {
                        "source_group": target_gid,
                        "route_id": fingerprint(["stage4.plan.v1", target_gid, target_gid, "DEV-C"]),
                    }
                ],
            }
            predictions[dev_bid] = _stub_ref(f"predictions/stage4/{model}/dev_c_{target_lang}/_COMPLETE.json")

    class _StubStore:
        def json(self, ref, filename, expected_kind):
            assert expected_kind == "stage3_candidate" and filename == "body.json"
            return candidates[ref["path"]]

    fit_plan = {"groups": groups}
    stage4_plan = SimpleNamespace(fit=fit_plan, store=_StubStore(), reference=_stub_ref("cache/shared/stage4/plan/_COMPLETE.json"))
    stage4_plan_dict = {"blocks": blocks, "ordinary": ordinary}

    class _PlanProxy:
        def __init__(self):
            self.fit = fit_plan
            self.store = stage4_plan.store
            self.reference = stage4_plan.reference

        def __getitem__(self, key):
            return stage4_plan_dict[key]

    stage4_release_ref = _stub_ref(f"results/stage4/releases/{lineage_tag}/_COMPLETE.json")
    dataset_fp = fingerprint(f"dataset-{lineage_tag}")
    protocol_fp = fingerprint(protocol)
    return Phase1Binding(
        root=Path(root).resolve(),
        protocol=frozen(protocol),
        stage4_release_ref=frozen(stage4_release_ref),
        stage4_release=frozen({"predictions": predictions}),
        stage4_plan=_PlanProxy(),
        bootstrap_shape=(2000, protocol["split_sizes"]["SCREEN"]),
        bootstrap_sha256=fingerprint("bootstrap"),
        midpoints=frozen(midpoints),
        dataset_ref=frozen(_stub_ref("shared/dataset/_COMPLETE.json")),
        data_plan_ref=frozen(_stub_ref("cache/stage2/data_plan/_COMPLETE.json")),
        model_plan_refs=frozen({m: _stub_ref(f"cache/stage2/{m}/model_plan/_COMPLETE.json") for m in MODELS}),
        suite_ref=frozen(_stub_ref("results/stage2-validation/suite/_COMPLETE.json")),
        stage3_registry_ref=frozen(_stub_ref("results/stage3-validation/registry/_COMPLETE.json")),
        g0_gate_ref=frozen(_stub_ref("results/stage3-validation/g0/_COMPLETE.json")),
        g0_p_value=0.362,
        dataset_fingerprint=dataset_fp,
        protocol_fingerprint=protocol_fp,
        lineage_identity=fingerprint(f"lineage-{lineage_tag}"),
    )


def _pure_python_score(matrix, coef, intercept, stats, *, estimable=None):
    """Exact float64 matmul matching transform(matrix, stats) @ coef.T + intercept."""
    mean = stats["mean"]
    scale = stats["scale"]
    out = []
    for row in matrix:
        norm = [(float(row[j]) - float(mean[j])) / float(scale[j]) for j in range(len(row))]
        logits_row = []
        for k in range(len(coef)):
            val = sum(norm[j] * float(coef[k][j]) for j in range(len(norm))) + float(intercept[k])
            if estimable is not None and not estimable[k]:
                val = 0.0
            logits_row.append(val)
        out.append(logits_row)
    return out


class E17AuthContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        init_phase2_dirs(self.root)
        self.protocol = json.loads((REPO / "configs/protocol.json").read_text(encoding="utf-8"))
        # Use a compact DEV-C row count in the lightweight mock protocol for fast deterministic tests.
        self.mock_protocol = dict(self.protocol, split_sizes=dict(self.protocol["split_sizes"], **{"DEV-C": 6}))
        self.binding = _build_mock_binding(self.root, self.mock_protocol)
        self.producer = {
            "code_commit": "4" * 40,
            "storage_scope": "local-development",
            "software_versions": {"python": "3.12.0"},
        }

    def test_exact_24_route_inventory_and_calibrator_recipe(self):
        inv = enumerate_e17_inventory(self.binding, models=MODELS)
        self.assertEqual(inv["permission"], E17_DEV_C_PERMISSION)
        self.assertEqual(inv["counts"]["models"], 2)
        self.assertEqual(inv["counts"]["reused_screen_blocks"], 8)
        self.assertEqual(inv["counts"]["reused_screen_routes"], 32)
        self.assertEqual(inv["counts"]["reused_dev_c_diagonal_blocks"], 8)
        self.assertEqual(inv["counts"]["reused_dev_c_diagonal_routes"], 8)
        self.assertEqual(inv["counts"]["authorized_off_diagonal_blocks"], 8)
        self.assertEqual(inv["counts"]["authorized_off_diagonal_routes"], 24)
        self.assertEqual(inv["counts"]["dev_c_feature_releases"], 8)
        self.assertEqual(inv["counts"]["source_systems"], 8)

        self.assertEqual(inv["models"]["qwen"]["midpoint_depth"], 18)
        self.assertEqual(inv["models"]["gemma"]["midpoint_depth"], 17)

        all_off_route_ids = set()
        for model in MODELS:
            m = inv["models"][model]
            self.assertEqual(len(m["screen_blocks"]), 4)
            self.assertEqual(sum(len(b["routes"]) for b in m["screen_blocks"].values()), 16)
            self.assertEqual(len(m["dev_c_diagonals"]), 4)
            for lang, diag in m["dev_c_diagonals"].items():
                self.assertEqual(diag["source_language"], lang)
                self.assertEqual(diag["target_language"], lang)
                self.assertEqual(diag["access"], "read_only")
                self.assertTrue(diag["never_duplicate"])

            self.assertEqual(len(m["off_diagonal_blocks"]), 4)
            model_pairs = set()
            for target_lang, blk in m["off_diagonal_blocks"].items():
                self.assertEqual(blk["split"], "DEV-C")
                self.assertEqual(blk["owner"], model)
                self.assertEqual(len(blk["routes"]), 3)
                for route in blk["routes"]:
                    self.assertNotEqual(route["source_language"], route["target_language"])
                    self.assertEqual(route["target_language"], target_lang)
                    self.assertNotEqual(route["source_group"], route["target_group"])
                    self.assertTrue(route["frozen_source_head_only"])
                    model_pairs.add((route["source_language"], route["target_language"]))
                    all_off_route_ids.add(route["route_id"])
            self.assertEqual(len(model_pairs), 12)
            self.assertTrue(all(s != t for s, t in model_pairs))
        self.assertEqual(len(all_off_route_ids), 24)

        recipe = e17_calibrator_recipe(self.protocol)
        self.assertEqual(recipe["dtype"], "float64")
        self.assertEqual(recipe["forms"]["raw"], "z")
        self.assertEqual(recipe["forms"]["temp"], "z / exp(log_T)")
        self.assertEqual(recipe["forms"]["temp_intercept"], "z / exp(log_T) + b")
        self.assertEqual(recipe["objective"]["primary_category_count"], 14)
        self.assertEqual(recipe["optimizer"]["method"], "BFGS")
        self.assertTrue(recipe["optimizer"]["jac"])
        self.assertEqual(recipe["optimizer"]["gtol"], E17_BFGS_GTOL)
        self.assertEqual(recipe["optimizer"]["maxiter"], E17_BFGS_MAXITER)
        self.assertEqual(recipe["optimizer"]["init"], {"log_T": 0.0, "b": 0.0})
        self.assertEqual(recipe["invariance_tol"], E17_INVARIANCE_TOL)
        self.assertIn("target_dev_c_supervised_calibration", recipe["supervision_labels"])
        self.assertIn("source_dev_c_calibrated", recipe["supervision_labels"])

    def test_guardrails_reject_unauthorized_splits_target_head_and_forbidden_ops(self):
        # Split guardrails: only DEV-C authorized for E17 DEV-C scoring/calibration.
        self.assertEqual(require_e17_dev_c_split("DEV-C"), "DEV-C")
        with self.assertRaisesRegex(ContractError, "CONFIRM is closed"):
            require_e17_dev_c_split("CONFIRM")
        for bad_split in ("FIT", "DEV-S", "SCREEN", "TEST"):
            with self.assertRaises(ContractError):
                require_e17_dev_c_split(bad_split)

        # Off-diagonal route authorization succeeds for frozen source head s != t.
        auth = authorize_e17_dev_c_route("en", "hi", head_source_lang="en", split="DEV-C", purpose="score_off_diagonal")
        self.assertEqual(auth["permission"], E17_DEV_C_PERMISSION)

        # Reject target-trained head misuse for target calibrator / off-diagonal scoring.
        with self.assertRaisesRegex(ContractError, "never target 'hi'"):
            authorize_e17_dev_c_route("en", "hi", head_source_lang="hi", split="DEV-C", purpose="fit_target_calibrator")
        with self.assertRaisesRegex(ContractError, "never target 'bn'"):
            authorize_e17_dev_c_route("te", "bn", head_source_lang="bn", split="DEV-C", purpose="score_off_diagonal")
        with self.assertRaisesRegex(ContractError, "requires frozen source head 'en'"):
            authorize_e17_dev_c_route("en", "hi", head_source_lang="bn", split="DEV-C", purpose="fit_target_calibrator")

        # Group-ID level validation rejects target t's own trained head.
        self.assertTrue(
            validate_target_calibrator_input(
                "en", "hi", logits_source_group="gid_en", source_group_id="gid_en", target_group_id="gid_hi"
            )
        )
        with self.assertRaisesRegex(ContractError, "own trained head"):
            validate_target_calibrator_input(
                "en", "hi", logits_source_group="gid_hi", source_group_id="gid_en", target_group_id="gid_hi"
            )

        # Reject duplicating Stage 4 DEV-C diagonals.
        with self.assertRaisesRegex(ContractError, "never be duplicated"):
            authorize_e17_dev_c_route("en", "en", head_source_lang="en", split="DEV-C", purpose="score_off_diagonal")

        # Reject base-head refit, scaler refit, SCREEN tuning, E19 modification, CONFIRM/FIT/DEV-S splits.
        with self.assertRaisesRegex(ContractError, "E19"):
            authorize_e17_operation(split="DEV-C", operation="fit_target_calibrator", source_lang="en", target_lang="hi", modifies_e19=True)
        with self.assertRaisesRegex(ContractError, "refit base heads"):
            authorize_e17_operation(split="DEV-C", operation="fit_target_calibrator", source_lang="en", target_lang="hi", refits_head=True)
        with self.assertRaisesRegex(ContractError, "refit feature scalers"):
            authorize_e17_operation(split="DEV-C", operation="fit_target_calibrator", source_lang="en", target_lang="hi", refits_scaler=True)
        with self.assertRaisesRegex(ContractError, "tune calibrators on SCREEN"):
            authorize_e17_operation(split="SCREEN", operation="evaluate_screen", tunes_on_screen=True)
        for bad_op in ("refit_head", "refit_scaler", "tune_on_screen", "modify_e19_cutoffs", "reselect_c"):
            with self.assertRaises(ContractError):
                authorize_e17_operation(split="DEV-C", operation=bad_op, source_lang="en", target_lang="hi")
        for bad_split in ("CONFIRM", "FIT", "DEV-S"):
            with self.assertRaises(ContractError):
                authorize_e17_operation(split=bad_split, operation="score_off_diagonal", source_lang="en", target_lang="hi")

        # Read-only SCREEN evaluation is allowed when no forbidden flags are set.
        ev = authorize_e17_operation(split="SCREEN", operation="evaluate_screen", source_lang="en", target_lang="hi", head_source_lang="en")
        self.assertTrue(ev["read_only"])

    def test_diagonal_reproduction_pass_and_fail(self):
        exact = [[0.25, -1.5, 0.0], [2.75, 0.125, -0.5]]
        res = verify_diagonal_reproduction(exact, exact, atol=0.0, rtol=0.0, model="qwen")
        self.assertTrue(res["passed"])
        self.assertEqual(res["max_abs_diff"], 0.0)
        self.assertEqual(res["atol"], 0.0)
        self.assertEqual(res["rtol"], 0.0)

        perturbed = [[0.25, -1.5 + 1e-15, 0.0], [2.75, 0.125, -0.5]]
        with self.assertRaisesRegex(ContractError, "diagonal reproduction failed"):
            verify_diagonal_reproduction(perturbed, exact, atol=0.0, rtol=0.0, model="qwen")

        with self.assertRaisesRegex(ContractError, "exact atol=0.0, rtol=0.0"):
            verify_diagonal_reproduction(exact, exact, atol=1e-9, rtol=0.0, model="qwen")

    def _run_mocked_preflight(self, *, perturb_gemma=0.0, bound_or_plan=None, models=MODELS):
        n_rows = self.mock_protocol["split_sizes"]["DEV-C"]
        n_cats = len(self.mock_protocol["category_order"])
        seen_splits = []

        def fake_read_features(binding, model, condition, pooling, target_lang, split):
            seen_splits.append((model, target_lang, split))
            dim = 4 if model == "qwen" else 6
            rows = tuple(
                {"image_id": f"img_{i:03d}", "split": split, "language": target_lang, "row_index": i}
                for i in range(n_rows)
            )
            matrix = [[float((i + 1) * (j + 1)) * 0.25 for j in range(dim)] for i in range(n_rows)]
            return rows, matrix

        def fake_load_head(binding, model, condition, pooling, source_lang):
            dim = 4 if model == "qwen" else 6
            coef = [[0.5 if (k % 2 == 0) else -0.25 for _ in range(dim)] for k in range(n_cats)]
            intercept = [0.1 * (k - 12) for k in range(n_cats)]
            estimable = [True] * n_cats
            prevalence = [0.25] * n_cats
            stats_ref = _stub_ref(f"checkpoints/stage3/{model}/{source_lang}/statistics/_COMPLETE.json")
            stats = {
                "mean": [0.5] * dim,
                "variance": [4.0] * dim,
                "scale": [2.0] * dim,
                "center": True,
                "reference": stats_ref,
            }
            return coef, intercept, estimable, prevalence, stats

        def fake_load_pred(plan, pred_ref, *, expected_block):
            model = "qwen" if "/qwen/" in pred_ref["path"] else "gemma"
            rows, matrix = fake_read_features(self.binding, model, "image_caption", "last_caption", "en", "DEV-C")
            # Remove the helper call from seen_splits tracking
            seen_splits.pop()
            coef, intercept, estimable, prevalence, stats = fake_load_head(
                self.binding, model, "image_caption", "last_caption", "en"
            )
            logits = _pure_python_score(matrix, coef, intercept, stats, estimable=estimable)
            if model == "gemma" and perturb_gemma != 0.0:
                logits[0][0] += perturb_gemma
            inv = enumerate_e17_inventory(self.binding, models=(model,))
            route_id = inv["models"][model]["dev_c_diagonals"]["en"]["route_id"]
            pred = SimpleNamespace(
                reference=pred_ref,
                split="DEV-C",
                rows=rows,
                logits=logits,
                estimable=estimable,
                prevalence=prevalence,
            )
            return {route_id: pred}

        events = []
        target = self.binding if bound_or_plan is None else bound_or_plan
        with (
            patch("src.phase2.e17_auth.read_verified_midpoint_features", side_effect=fake_read_features),
            patch("src.phase2.e17_auth.load_frozen_source_head", side_effect=fake_load_head),
            patch("src.phase2.e17_auth._load_stage4_prediction_block", side_effect=fake_load_pred),
            patch("src.phase2.e17_auth.score_frozen_head", side_effect=_pure_python_score),
        ):
            ref, body = preflight_e17_dev_c(target, self.producer, models=models, progress=events.append)
        return ref, body, events, seen_splits

    def test_preflight_end_to_end_publish_and_resume_verification(self):
        # Accept a plan wrapper exposing .binding as well as a direct Phase1Binding.
        plan_wrapper = SimpleNamespace(binding=self.binding)
        ref, body, events, seen_splits = self._run_mocked_preflight(bound_or_plan=plan_wrapper)

        # Verify ONLY DEV-C feature releases were ever opened (8 total: 4 languages x 2 models), never FIT or CONFIRM.
        self.assertEqual(len(seen_splits), 8)
        self.assertTrue(all(split == "DEV-C" for _, _, split in seen_splits))
        self.assertEqual(body["schema"], SCHEMA_E17_AUTH)
        self.assertEqual(body["permission"], E17_DEV_C_PERMISSION)
        self.assertTrue(body["confirm_excluded"])
        self.assertFalse(body["fit_matrices_opened"])
        self.assertEqual(body["inventory_summary"]["authorized_off_diagonal_routes_total"], 24)
        self.assertEqual(len(events), 19)  # 2 models * 9 units + 1 publish unit
        self.assertTrue(ref["path"].startswith("phase2/plans/"))

        # Load and resume verification succeed on exact match.
        loaded_ref, loaded_body, marker = load_e17_auth(
            self.root, self.mock_protocol, ref, binding=self.binding, models=MODELS
        )
        self.assertEqual(plain(loaded_ref), plain(ref))
        self.assertEqual(marker["kind"], KIND_E17_AUTH)
        self.assertEqual(loaded_body["phase2_version"], PHASE2_VERSION)

        resumed_ref, _, _ = verify_e17_auth_resume(self.binding, ref, models=MODELS)
        self.assertEqual(plain(resumed_ref), plain(ref))

        # Resume rejects wrong model set (e.g. single-model request against dual-model inputs mismatch).
        with self.assertRaisesRegex(ContractError, "inputs do not match"):
            verify_e17_auth_resume(self.binding, ref, models=("qwen",))

        # Resume rejects foreign lineage.
        other_binding = _build_mock_binding(self.root, self.mock_protocol, lineage_tag="v2")
        with self.assertRaises(ContractError):
            verify_e17_auth_resume(other_binding, ref, models=MODELS)

        # Resume rejects tampered SHA-256 and tampered body.json.
        with self.assertRaisesRegex(ContractError, "checksum mismatch"):
            verify_e17_auth_resume(self.binding, {"path": ref["path"], "sha256": "0" * 64}, models=MODELS)

        body_path = (self.root / ref["path"]).parent / "body.json"
        body_path.write_text('{"tampered": true}\n', encoding="utf-8")
        with self.assertRaises(ContractError):
            verify_e17_auth_resume(self.binding, ref, models=MODELS)

    def test_preflight_blocks_publication_on_diagonal_mismatch(self):
        before = list((self.root / "phase2/plans").iterdir())
        with self.assertRaisesRegex(ContractError, "diagonal reproduction failed for gemma"):
            self._run_mocked_preflight(perturb_gemma=1e-12)
        after = list((self.root / "phase2/plans").iterdir())
        self.assertEqual(before, after)

    def test_notebook_structure(self):
        nb_path = REPO / "scripts/phase2/preflight_e17.ipynb"
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertEqual(nb["nbformat"], 4)
        self.assertEqual(len(nb["cells"]), 4)
        self.assertEqual(nb["cells"][0]["cell_type"], "markdown")
        for idx in (1, 2, 3):
            self.assertEqual(nb["cells"][idx]["cell_type"], "code")
        settings_src = "".join(nb["cells"][2]["source"])
        exec_src = "".join(nb["cells"][3]["source"])
        self.assertIn("MODEL = 'both'", settings_src)
        self.assertIn("('qwen', 'gemma', 'both')", exec_src)
        self.assertIn("src.phase2.e17_auth", exec_src)
        self.assertIn("flush=True", exec_src)


@unittest.skipIf(np is None, "numpy/scipy/sklearn required for full Phase 1 Stage 4 fixture tests")
class FullPhase1E17PreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys

        tests_dir = str(REPO / "tests")
        if tests_dir not in sys.path:
            sys.path.insert(0, tests_dir)
        import test_stage5_foundation as stage5
        from src.phase2.interfaces import bind_phase1
        from src.stage4.predictions import score_and_publish_block
        from src.stage4.validation import PENDING, RELEASE, _publish

        stage5.ReportingTests.setUpClass()
        cls.addClassCleanup(stage5.ReportingTests.doClassCleanups)
        cls.plan = stage5.ReportingTests.plan
        cls.producer = stage5.ReportingTests.producer
        cls.root = cls.plan.root
        cls.protocol = cls.plan.protocol

        # Materialize real Stage 4 en->en DEV-C diagonal prediction blocks for both qwen and gemma.
        release_body = plain(cls.plan.store.json(stage5.ReportingTests.release, "body.json", RELEASE))
        preds = dict(release_body["predictions"])
        for model in MODELS:
            for bid, blk in cls.plan["blocks"].items():
                if blk["owner"] == model and blk["split"] == "DEV-C":
                    gid = blk["target_group"]
                    if cls.plan.fit["groups"][gid]["language"] == "en":
                        preds[bid] = plain(score_and_publish_block(cls.plan, model, bid, cls.producer))
        base = {k: v for k, v in release_body.items() if k != "independent_readback"}
        base["predictions"] = preds
        base["stage4_complete"] = False
        pending = _publish(cls.plan.store, cls.plan["dataset_fingerprint"], "fixture/e17_stage4", PENDING, base, {}, [cls.plan.reference], cls.producer)
        cls.release = _publish(
            cls.plan.store,
            cls.plan["dataset_fingerprint"],
            "fixture/e17_stage4",
            RELEASE,
            dict(base, stage4_complete=True, independent_readback=pending),
            {},
            [cls.plan.reference],
            cls.producer,
        )
        cls.binding = bind_phase1(cls.root, cls.protocol, cls.release)
        init_phase2_dirs(cls.root)

    def test_real_stage4_diagonal_atol_zero_and_resume(self):
        ref, body = preflight_e17_dev_c(self.binding, self.producer, models=MODELS)
        self.assertEqual(body["inventory_summary"]["authorized_off_diagonal_routes_total"], 24)
        for model in MODELS:
            diag = body["models"][model]["diagonal_reproduction_check"]
            self.assertTrue(diag["passed"])
            self.assertEqual(diag["max_abs_diff"], 0.0)
            self.assertEqual(diag["atol"], 0.0)
            self.assertEqual(diag["rtol"], 0.0)
        resumed_ref, _, _ = verify_e17_auth_resume(self.binding, ref, models=MODELS)
        self.assertEqual(plain(resumed_ref), plain(ref))


if __name__ == "__main__":
    unittest.main()
