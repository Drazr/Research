"""End-to-end Stage 1 cross-worker integration tests.

Verifies that Chat 1 (foundation), Chat 2 (e07_audit), Chat 3 (e17_auth), and
Chat 4 (gates/plan/operations) interoperate cleanly with zero Phase 1 writes,
zero CONFIRM processing, exact cross-artifact lineage consistency, and valid
Colab notebooks.
"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.common.contracts import ContractError, LANGUAGES, fingerprint
from src.phase2.e07_audit import audit_e07_captions, load_e07_audit
from src.phase2.e17_auth import enumerate_e17_inventory, preflight_e17_dev_c
from src.phase2.interfaces import MODELS, PHASE2_PARENT_DIRS, Phase1Binding, frozen, plain
from src.phase2.plan import load_phase2_plan, prepare_phase2, verify_stage1_bundle
from src.stage2.schema import REPO

# Reuse synthetic builders from worker test suites without duplicating fixture math.
_TESTS_DIR = str(REPO / "tests")
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
from test_phase2_e07_audit import _build_synthetic_stage1_dataset
from test_phase2_e17_auth import _build_mock_binding, _pure_python_score


def _snapshot_phase1_tree(root: Path) -> dict:
    """Record relative paths and byte sizes of all non-phase2 files under root."""
    snap = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        if rel == "phase2" or rel.startswith("phase2/"):
            continue
        if p.is_file():
            snap[rel] = p.stat().st_size
    return snap


class Stage1CrossWorkerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.protocol = json.loads((REPO / "configs/protocol.json").read_text(encoding="utf-8"))
        self.producer = {
            "code_commit": "4" * 40,
            "storage_scope": "local-development",
            "software_versions": {"python": "3.12.0", "numpy": "2.1.3", "scipy": "1.18.1", "scikit-learn": "1.6.1"},
        }

    def test_full_stage1_pipeline_and_bundle_verification(self):
        # 1. Build a synthetic Stage 1 dataset with real captions/{en,hi,bn,te}.jsonl + unknown mask.
        e07_binding, proto, _ = _build_synthetic_stage1_dataset(
            self.root,
            self.protocol,
            screen_caption_factory=lambda iid, pos, lang: [f"cap-{lang}-{iid}-{j}" for j in range(5)],
            run_tag="intg",
        )
        # 2. Build a matching mock Phase1Binding sharing the exact dataset_ref / fingerprints / lineage.
        base = _build_mock_binding(self.root, proto, lineage_tag="intg")
        unified = Phase1Binding(
            root=self.root,
            protocol=frozen(proto),
            stage4_release_ref=base.stage4_release_ref,
            stage4_release=base.stage4_release,
            stage4_plan=base.stage4_plan,
            bootstrap_shape=base.bootstrap_shape,
            bootstrap_sha256=base.bootstrap_sha256,
            midpoints=base.midpoints,
            dataset_ref=e07_binding.dataset_ref,
            data_plan_ref=base.data_plan_ref,
            model_plan_refs=base.model_plan_refs,
            suite_ref=base.suite_ref,
            stage3_registry_ref=base.stage3_registry_ref,
            g0_gate_ref=base.g0_gate_ref,
            g0_p_value=base.g0_p_value,
            dataset_fingerprint=e07_binding.dataset_fingerprint,
            protocol_fingerprint=e07_binding.protocol_fingerprint,
            lineage_identity=base.lineage_identity,
        )

        # Snapshot all Phase 1 files before any Phase 2 Stage 1 code runs.
        before_phase1 = _snapshot_phase1_tree(self.root)

        # 3. Step 1A: prepare_phase2 (common bind + parent dir pre-init + two-process storage_probe).
        with (
            patch("src.phase2.plan.bind_phase1", return_value=unified),
            patch("src.phase2.plan.e29_screen_blocks", side_effect=lambda b, m: _fake_e29_blocks(m)),
            patch("src.phase2.plan.e11_fit_statistics_refs", side_effect=lambda b, m: _fake_e11_stats(m)),
        ):
            plan = prepare_phase2(self.root, proto, unified.stage4_release_ref, self.producer)

        for rel in PHASE2_PARENT_DIRS:
            self.assertTrue((self.root / rel).is_dir())
        reloaded = load_phase2_plan(self.root, proto, plan.reference)
        self.assertEqual(reloaded["plan_identity"], plan["plan_identity"])

        # 4. Step 1C (E07 audit) directly from the real synthetic stage1_dataset captions.
        e07_ref = audit_e07_captions(plan, self.producer)
        _, e07_body, _ = load_e07_audit(self.root, proto, e07_ref, expected_plan_identity=plan["plan_identity"])
        self.assertEqual(e07_body["verdict"], "passed_full")
        self.assertTrue(e07_body["draw0_verified"])
        self.assertTrue(e07_body["confirm_excluded"])
        self.assertEqual(e07_body["extraction_plan"]["confirm_rows"], 0)

        # 5. Step 1B (E17 DEV-C preflight) for both models with exact atol=0, rtol=0 diagonal check.
        seen_splits = []
        e17_ref, e17_body = _run_mocked_e17(unified, plan, self.producer, seen_splits, models=MODELS)
        self.assertEqual(len(seen_splits), 8)
        self.assertTrue(all(s == "DEV-C" for s in seen_splits))
        self.assertEqual(e17_body["inventory_summary"]["authorized_off_diagonal_routes_total"], 24)

        # 6. Cross-worker verify_stage1_bundle passes with exact lineage agreement.
        bundle = verify_stage1_bundle(
            self.root,
            proto,
            plan.reference,
            e07_audit_ref=e07_ref,
            e17_auth_ref=e17_ref,
        )
        self.assertTrue(bundle["stage1_complete"])
        self.assertEqual(bundle["e07_verdict"], "passed_full")
        self.assertEqual(bundle["lineage_identity"], unified.lineage_identity)

        # 7. CLI verify-stage1-bundle subprocess also passes cleanly.
        proto_file = Path(self.tmp.name).parent / f"phase2_intg_proto_{id(self)}.json"
        proto_file.write_text(json.dumps(plain(proto), sort_keys=True) + "\n", encoding="utf-8")
        self.addCleanup(lambda: proto_file.exists() and proto_file.unlink())
        cli = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.phase2.operations",
                "verify-stage1-bundle",
                "--phase1-root",
                str(self.root),
                "--protocol",
                str(proto_file),
                "--plan",
                json.dumps(plain(plan.reference)),
                "--e07-audit",
                json.dumps(plain(e07_ref)),
                "--e17-auth",
                json.dumps(plain(e17_ref)),
            ],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        )
        cli_out = json.loads(cli.stdout.strip().splitlines()[-1])
        self.assertTrue(cli_out["stage1_complete"])
        self.assertEqual(cli_out["e07_verdict"], "passed_full")

        # 8. Single-model E17 auth is rejected by verify_stage1_bundle (both models required).
        single_e17_ref, _ = _run_mocked_e17(unified, plan, self.producer, [], models=("qwen",))
        with self.assertRaisesRegex(ContractError, "cover qwen and gemma"):
            verify_stage1_bundle(
                self.root,
                proto,
                plan.reference,
                e07_audit_ref=e07_ref,
                e17_auth_ref=single_e17_ref,
            )

        # 9. Zero Phase 1 files were created, modified, or deleted during Stage 1.
        self.assertEqual(_snapshot_phase1_tree(self.root), before_phase1)

    def test_all_three_stage1_notebooks_consistent_and_valid(self):
        notebooks = (
            ("scripts/phase2/prepare_phase2.ipynb", "src.phase2.operations"),
            ("scripts/phase2/audit_e07.ipynb", "src.phase2.e07_audit"),
            ("scripts/phase2/preflight_e17.ipynb", "src.phase2.e17_auth"),
        )
        for rel, module_name in notebooks:
            nb = json.loads((REPO / rel).read_text(encoding="utf-8"))
            self.assertEqual(nb["nbformat"], 4, rel)
            self.assertEqual(len(nb["cells"]), 4, rel)
            self.assertEqual(nb["cells"][0]["cell_type"], "markdown", rel)
            setup_src = "".join(nb["cells"][1]["source"])
            settings_src = "".join(nb["cells"][2]["source"])
            exec_src = "".join(nb["cells"][3]["source"])
            self.assertIn("requirements-stage2-cpu.txt", setup_src, rel)
            self.assertIn("OPENBLAS_NUM_THREADS", setup_src, rel)
            self.assertIn("PHASE1_ROOT = Path('/content/drive/MyDrive/phase1')", settings_src, rel)
            self.assertIn(module_name, exec_src, rel)
            self.assertIn("blockers", exec_src, rel)
            self.assertIn("flush=True", exec_src, rel)
            self.assertNotIn("capture_output=True", exec_src, rel)


def _fake_e29_blocks(model):
    out = {}
    for cond in ("image_caption", "caption_only"):
        out[cond] = {}
        for pool in ("last_caption", "caption_mean"):
            out[cond][pool] = {
                lang: {
                    "block_id": fingerprint(["e29", model, cond, pool, lang]),
                    "target_group": fingerprint(["gid", model, cond, pool, lang]),
                    "prediction_ref": {
                        "path": f"predictions/stage4/{model}/{cond}_{pool}_{lang}/_COMPLETE.json",
                        "sha256": fingerprint(["pred", model, cond, pool, lang]),
                    },
                    "routes": {src: fingerprint(["route", model, cond, pool, src, lang]) for src in LANGUAGES},
                }
                for lang in LANGUAGES
            }
    return out


def _fake_e11_stats(model):
    return {
        lang: {
            "group_id": fingerprint(["gid", model, "image_caption", "caption_mean", lang]),
            "statistics_ref": {
                "path": f"cache/stage3/{model}_{lang}/_COMPLETE.json",
                "sha256": fingerprint(["stats", model, lang]),
            },
            "candidate_ref": {
                "path": f"checkpoints/stage3/{model}_{lang}/_COMPLETE.json",
                "sha256": fingerprint(["cand", model, lang]),
            },
            "selection_ref": {
                "path": f"checkpoints/stage3/sel_{model}_{lang}/_COMPLETE.json",
                "sha256": fingerprint(["sel", model, lang]),
            },
            "C": 1.0,
            "dimension": 4,
            "site": "decoder:18:caption_mean" if model == "qwen" else "decoder:17:caption_mean",
            "model_plan_ref": {
                "path": f"cache/shared/stage2-plans/{model}/_COMPLETE.json",
                "sha256": fingerprint(["mp", model]),
            },
            "screen_block_id": fingerprint(["e29", model, "image_caption", "caption_mean", lang]),
            "screen_prediction_ref": {
                "path": f"predictions/stage4/{model}/image_caption_caption_mean_{lang}/_COMPLETE.json",
                "sha256": fingerprint(["pred", model, "image_caption", "caption_mean", lang]),
            },
            "screen_routes": {src: fingerprint(["route", model, "image_caption", "caption_mean", src, lang]) for src in LANGUAGES},
            "screen_feature_release_ref": {
                "path": f"features/{model}/SCREEN_{lang}/_COMPLETE.json",
                "sha256": fingerprint(["feat", model, lang]),
            },
        }
        for lang in LANGUAGES
    }


def _run_mocked_e17(binding, plan, producer, seen_splits, *, models):
    n_rows = int(binding.protocol["split_sizes"]["DEV-C"])
    n_cats = len(binding.protocol["category_order"])

    def _dim_for(b, model, lang):
        inv = enumerate_e17_inventory(b, models=(model,))
        return int(inv["models"][model]["dev_c_feature_releases"][lang]["dimension"])

    def fake_read_features(b, model, condition, pooling, lang, split):
        seen_splits.append(split)
        dim = _dim_for(b, model, lang)
        rows = tuple(
            {"image_id": f"{i:016x}", "split": split, "language": lang, "caption_id": f"c:{lang}:{i}"}
            for i in range(n_rows)
        )
        base_val = 0.25 if model == "qwen" else -0.5
        matrix = [[base_val + 0.01 * i + 0.1 * d for d in range(dim)] for i in range(n_rows)]
        return rows, matrix

    def fake_load_head(b, model, condition, pooling, lang):
        inv = enumerate_e17_inventory(b, models=(model,))
        dim = int(inv["models"][model]["source_systems"][lang]["dimension"])
        stats_ref = inv["models"][model]["source_systems"][lang]["statistics_ref"]
        coef = [[0.05 * (k + 1) + 0.01 * d for d in range(dim)] for k in range(n_cats)]
        intercept = [0.01 * k for k in range(n_cats)]
        estimable = [True] * n_cats
        prevalence = [0.25] * n_cats
        stats = {
            "mean": [0.1] * dim,
            "variance": [0.25] * dim,
            "scale": [0.5] * dim,
            "center": True,
            "reference": stats_ref,
        }
        return coef, intercept, estimable, prevalence, stats

    def fake_load_pred(stage4_plan, pred_ref, *, expected_block):
        model = "qwen" if "qwen" in pred_ref["path"] else "gemma"
        rows, matrix = fake_read_features(binding, model, "image_caption", "last_caption", "en", "DEV-C")
        seen_splits.pop()
        coef, intercept, estimable, prevalence, stats = fake_load_head(
            binding, model, "image_caption", "last_caption", "en"
        )
        logits = _pure_python_score(matrix, coef, intercept, stats, estimable=estimable)
        inv = enumerate_e17_inventory(binding, models=(model,))
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

    with (
        patch("src.phase2.e17_auth.read_verified_midpoint_features", side_effect=fake_read_features),
        patch("src.phase2.e17_auth.load_frozen_source_head", side_effect=fake_load_head),
        patch("src.phase2.e17_auth._load_stage4_prediction_block", side_effect=fake_load_pred),
        patch("src.phase2.e17_auth.score_frozen_head", side_effect=_pure_python_score),
    ):
        return preflight_e17_dev_c(plan, producer, models=models)


if __name__ == "__main__":
    unittest.main()
