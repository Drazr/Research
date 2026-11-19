"""Unit and fixture tests for Phase 2 Stage 4 E07 CPU scorer and notebook."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.common.contracts import LANGUAGES, ContractError, fingerprint
from src.common.storage import file_sha256, within
from src.phase2.interfaces import (
    CONDITIONS,
    KIND_ANALYSIS,
    KIND_E07_AUDIT,
    KIND_E07_CHUNK,
    KIND_E07_RELEASE,
    KIND_PREDICTION,
    MODELS,
    POOLINGS,
    Phase1Binding,
    SCHEMA_ANALYSIS,
    SCHEMA_E07_AUDIT,
    SCHEMA_E07_FEATURES,
    SCHEMA_PREDICTION,
    frozen,
    plain,
    publish_phase2,
)
from src.phase2.plan import prepare_phase2
from src.phase2.score_e07 import (
    UNAVAILABLE_ZERO_CHANGED_REASON,
    _build_draw_indices,
    _load_and_verify_e07_release,
    _normalize_resume_prediction_refs,
    _sign_preservation_record,
    run_e07_scoring,
)
from src.stage2.schema import REPO, json_bytes, jsonl

try:
    import numpy as np
    import scipy  # noqa: F401
    from src.phase2.run_e29 import run_e29
    from src.stage3.engine import transform
    from src.stage4.predictions import Prediction

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def _producer():
    return {
        "code_commit": "0" * 40,
        "storage_scope": "local-development",
        "software_versions": {"python": "3.12.0", "numpy": "2.1.3"},
    }


def _protocol():
    return json.loads((REPO / "configs/protocol.json").read_text(encoding="utf-8"))


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


def _publish_minimal_plan_and_blocked_audit(root, protocol):
    dataset_fp = fingerprint(["e07-score-pure-ds"])
    stage4_rel = {"path": "results/stage4/releases/fix/_COMPLETE.json", "sha256": fingerprint(["s4", dataset_fp])}
    lineage_id = fingerprint(["lineage", dataset_fp])
    bootstrap_sha = fingerprint(["boot", dataset_fp])

    class _PlanStub(dict):
        pass

    s4_plan = _PlanStub({"screen_ids": ["img-0000"], "categories": list(protocol["category_order"])})
    s4_plan.reference = {"path": "results/stage4/plan/_COMPLETE.json", "sha256": "1" * 64}
    s4_plan.fit = {"targets": {"img-0000": {"split": "SCREEN", "values": [0] * 25}}}
    s4_plan.store = None

    binding = Phase1Binding(
        root=root,
        protocol=frozen(protocol),
        stage4_release_ref=frozen(stage4_rel),
        stage4_release=frozen({"predictions": {}}),
        stage4_plan=s4_plan,
        bootstrap_shape=(20, 1),
        bootstrap_sha256=bootstrap_sha,
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
        lineage_identity=lineage_id,
    )

    with (
        patch("src.phase2.plan.bind_phase1", return_value=binding),
        patch("src.phase2.plan.e29_screen_blocks", side_effect=lambda b, m: frozen(_fake_e29_blocks(m))),
        patch("src.phase2.plan.e11_fit_statistics_refs", side_effect=lambda b, m: frozen(_fake_e11_stats(m))),
    ):
        plan = prepare_phase2(root, protocol, stage4_rel, _producer())

    audit_body = {
        "schema": SCHEMA_E07_AUDIT,
        "stage4_release": stage4_rel,
        "lineage_identity": lineage_id,
        "dataset_ref": plain(binding.dataset_ref),
        "dataset_fingerprint": dataset_fp,
        "draw0_verified": True,
        "confirm_excluded": True,
        "verdict": "blocked",
        "verdict_reasons": ["insufficient changed views"],
        "screen_multiplicity": {},
        "fit_dev_s_multiplicity": {},
        "draw_buckets": {},
        "support_tables": {},
        "unique_changed_views": {"total": 0},
        "extraction_plan": None,
    }
    audit_ref = publish_phase2(
        root,
        protocol,
        dataset_fp,
        "phase2/audits/e07",
        KIND_E07_AUDIT,
        audit_body,
        _producer(),
    )
    return binding, plan, audit_ref


class ScoreE07PurePythonTests(unittest.TestCase):
    """Pure-Python unit tests that run without numpy/scipy."""

    def test_notebook_structure_and_code_ref(self):
        nb_path = REPO / "scripts/phase2/score_e07.ipynb"
        self.assertTrue(nb_path.is_file(), "Missing scripts/phase2/score_e07.ipynb")
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertEqual(nb["nbformat"], 4)
        self.assertEqual(len(nb["cells"]), 4)
        full_src = "\n".join("".join(cell["source"]) for cell in nb["cells"])
        self.assertIn("CODE_REF = 'phase2'", full_src)
        self.assertIn("MODEL = 'qwen'", full_src)
        self.assertIn("OPENBLAS_NUM_THREADS", full_src)
        self.assertIn("requirements-stage2-cpu.txt", full_src)
        self.assertIn("src.phase2.score_e07", full_src)
        for ref_var in ("PLAN_REF", "E07_AUDIT_REF", "E07_RELEASE_REF", "E29_ANALYSIS_REF"):
            self.assertIn(ref_var, full_src)
        self.assertIn("RESUME_PREDICTION_REFS", full_src)
        self.assertIn("RESUME_ANALYSIS_REF", full_src)

    def test_build_draw_indices_zero_changed_and_bucket_crosscheck(self):
        screen_ids = ("img-0000", "img-0001", "img-0002", "img-0003")
        chunk_ref = {"path": "phase2/features/e07/qwen/r1/chunk_0000/_COMPLETE.json", "sha256": "a" * 64}
        audit_buckets = {
            "bn": {
                "draw_1": {"same_id_as_draw0": 4, "distinct_id_same_text_as_draw0": 0, "changed_text_vs_draw0": 0},
                "draw_2": {"same_id_as_draw0": 4, "distinct_id_same_text_as_draw0": 0, "changed_text_vs_draw0": 0},
            },
            "en": {
                "draw_1": {"same_id_as_draw0": 2, "distinct_id_same_text_as_draw0": 0, "changed_text_vs_draw0": 2},
                "draw_2": {"same_id_as_draw0": 3, "distinct_id_same_text_as_draw0": 0, "changed_text_vs_draw0": 1},
            },
        }
        # Zero-changed language (bn)
        bn_idx = _build_draw_indices("bn", [], screen_ids, audit_buckets)
        self.assertEqual(bn_idx["draw_1"]["changed_text_vs_draw0"], 0)
        self.assertEqual(bn_idx["draw_1"]["unchanged_reused_from_draw0"], 4)
        self.assertEqual(bn_idx["draw_1"]["unchanged_fraction"], 1.0)
        self.assertEqual(bn_idx["draw_1"]["changed_row_map"], {})

        # Deduplicated views across draws 1 and 2 (en)
        en_views = [
            {
                "image_id": "img-0000",
                "caption_id": "c1",
                "caption_text_sha256": "b" * 64,
                "draw_keys": [1, 2],
                "chunk_ref": chunk_ref,
            },
            {
                "image_id": "img-0002",
                "caption_id": "c2",
                "caption_text_sha256": "c" * 64,
                "draw_keys": [1],
                "chunk_ref": chunk_ref,
            },
        ]
        en_idx = _build_draw_indices("en", en_views, screen_ids, audit_buckets)
        self.assertEqual(sorted(en_idx["draw_1"]["changed_row_map"].keys()), ["0", "2"])
        self.assertEqual(list(en_idx["draw_2"]["changed_row_map"].keys()), ["0"])
        self.assertEqual(en_idx["draw_2"]["changed_row_map"]["0"]["changed_view_index"], 0)

        # Bucket count mismatch rejected
        bad_buckets = {
            "en": {
                "draw_1": {"same_id_as_draw0": 1, "distinct_id_same_text_as_draw0": 0, "changed_text_vs_draw0": 3},
                "draw_2": {"same_id_as_draw0": 3, "distinct_id_same_text_as_draw0": 0, "changed_text_vs_draw0": 1},
            }
        }
        with self.assertRaises(ContractError):
            _build_draw_indices("en", en_views, screen_ids, bad_buckets)

        # Duplicate (image_id, draw_k) collision rejected
        dup_views = [
            en_views[0],
            dict(en_views[0], caption_id="c1_dup", caption_text_sha256="d" * 64),
        ]
        with self.assertRaises(ContractError):
            _build_draw_indices("en", dup_views, screen_ids, audit_buckets)

    def test_sign_preservation_and_resume_key_validation(self):
        same = _sign_preservation_record(0.03, 0.025)
        self.assertTrue(same["preserved"])
        self.assertEqual(same["sign_draw_0"], 1)
        self.assertEqual(same["sign_draw_k"], 1)

        flipped = _sign_preservation_record(0.03, -0.01)
        self.assertFalse(flipped["preserved"])
        self.assertEqual(flipped["sign_draw_k"], -1)

        missing = _sign_preservation_record(None, 0.02)
        self.assertFalse(missing["preserved"])

        valid = _normalize_resume_prediction_refs({"last_caption:en": "x", "caption_mean:bn": "y"})
        self.assertEqual(set(valid.keys()), {"last_caption:en", "caption_mean:bn"})
        with self.assertRaises(ContractError):
            _normalize_resume_prediction_refs({"invalid_pool:en": "x"})
        with self.assertRaises(ContractError):
            _normalize_resume_prediction_refs({"last_caption:fr": "x"})

    def test_blocked_audit_refusal_and_invalid_model(self):
        protocol = _protocol()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binding, plan, audit_ref = _publish_minimal_plan_and_blocked_audit(root, protocol)
            dummy_ref = {"path": "phase2/results/e29/qwen/r/_COMPLETE.json", "sha256": "a" * 64}

            with self.assertRaises(ContractError):
                run_e07_scoring(
                    root, protocol, plan.reference, audit_ref, dummy_ref, dummy_ref, "unknown_model", _producer()
                )

            with patch("src.phase2.score_e07.bind_phase1", return_value=binding):
                with self.assertRaises(ContractError) as ctx:
                    run_e07_scoring(
                        root, protocol, plan.reference, audit_ref, dummy_ref, dummy_ref, "qwen", _producer()
                    )
            self.assertIn("blocked", str(ctx.exception).lower())

    def test_failed_repeat_check_and_foreign_model_release_rejection_pure_python(self):
        protocol = _protocol()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binding, plan, blocked_audit_ref = _publish_minimal_plan_and_blocked_audit(root, protocol)
            dataset_fp = plan["dataset_fingerprint"]
            ep = {
                "split": "SCREEN",
                "confirm_rows": 0,
                "condition": "image_caption",
                "poolings": list(POOLINGS),
                "total_unique_views": 0,
                "per_language_counts": {l: 0 for l in LANGUAGES},
                "chunks": [],
            }
            audit_body = {
                "verdict": "passed_restricted",
                "extraction_plan": ep,
            }
            rel_body = {
                "schema": SCHEMA_E07_FEATURES,
                "experiment": "E07",
                "plan": plain(plan.reference),
                "plan_identity": plan["plan_identity"],
                "e07_audit_ref": plain(blocked_audit_ref),
                "e07_verdict": "passed_restricted",
                "model": "qwen",
                "midpoint_depth": 18,
                "condition": "image_caption",
                "poolings": list(POOLINGS),
                "split": "SCREEN",
                "total_unique_views": 0,
                "per_language_counts": {l: 0 for l in LANGUAGES},
                "dimension": 6,
                "model_fingerprint": "a" * 64,
                "execution_fingerprint": "b" * 64,
                "chunks": [],
                "repeat_check_summary": {
                    "chunks_checked": 0,
                    "all_passed": False,
                    "max_abs_diff_last_caption": 0.1,
                    "max_abs_diff_caption_mean": 0.0,
                },
                "confirm_excluded": True,
            }
            bad_rel_ref = publish_phase2(
                root,
                protocol,
                dataset_fp,
                "phase2/features/e07/qwen/releases",
                KIND_E07_RELEASE,
                rel_body,
                _producer(),
            )
            with self.assertRaises(ContractError) as ctx:
                _load_and_verify_e07_release(
                    root=root,
                    protocol=protocol,
                    plan=plan,
                    audit_ref=blocked_audit_ref,
                    audit_body=audit_body,
                    e07_release_ref=bad_rel_ref,
                    model="qwen",
                    depth=18,
                )
            self.assertIn("repeat_check_summary", str(ctx.exception))

            # Foreign model rejection (release owned by qwen, loaded for gemma)
            with self.assertRaises(ContractError):
                _load_and_verify_e07_release(
                    root=root,
                    protocol=protocol,
                    plan=plan,
                    audit_ref=blocked_audit_ref,
                    audit_body=audit_body,
                    e07_release_ref=bad_rel_ref,
                    model="gemma",
                    depth=17,
                )


@unittest.skipUnless(HAS_NUMPY, "numpy/scipy required for E07 numeric scoring fixture tests")
class ScoreE07FixtureTests(unittest.TestCase):
    """End-to-end numeric/bootstrap fixture tests for Stage 4 E07 CPU scoring."""

    def _build_full_fixture(self, root: Path, protocol, *, model="qwen", n_images=40, n_draws=60, dim=6):
        categories = tuple(protocol["category_order"])
        n_cats = len(categories)
        screen_ids = tuple(f"img-{i:04d}" for i in range(n_images))
        rng = np.random.default_rng(20261120)
        draws = rng.integers(0, n_images, size=(n_draws, n_images), dtype=np.int64)
        bootstrap_sha256 = hashlib.sha256(draws.tobytes()).hexdigest()

        # Balanced targets with one unknown cell on img-0000 for plant.n.02
        targets = {}
        for i, iid in enumerate(screen_ids):
            vals = [int((i + c) % 2 == 0) for c in range(n_cats)]
            if i == 0 and "plant.n.02" in categories:
                vals[categories.index("plant.n.02")] = None
            targets[iid] = {"split": "SCREEN", "values": vals}

        # Synthetic Stage 3 frozen heads and scalers for both poolings × 4 source languages
        frozen_heads = {}
        ordinary = {}
        fit_groups = {}
        for p_idx, pooling in enumerate(POOLINGS):
            for l_idx, lang in enumerate(LANGUAGES):
                gid = f"grp-{model}-image_caption-{pooling}-{lang}"
                coef = rng.normal(0.0, 0.35, size=(n_cats, dim)).astype("<f8")
                intercept = rng.normal(0.0, 0.05, size=(n_cats,)).astype("<f8")
                estimable = np.ones((n_cats,), dtype=bool)
                prevalence = np.full((n_cats,), 0.5, dtype="<f8")
                stats = {
                    "mean": rng.normal(0.0, 0.1, size=(dim,)).astype("<f8"),
                    "variance": np.full((dim,), 1.0, dtype="<f8"),
                    "scale": np.full((dim,), 1.0, dtype="<f8"),
                    "center": True,
                }
                frozen_heads[(pooling, lang)] = (coef, intercept, estimable, prevalence, stats)
                ordinary[gid] = {
                    "selection": {"path": f"results/stage3/{gid}_sel/_COMPLETE.json", "sha256": fingerprint(["sel", gid])},
                    "candidate": {"path": f"results/stage3/{gid}_cand/_COMPLETE.json", "sha256": fingerprint(["cand", gid])},
                    "C": 1.0,
                }
                fit_groups[gid] = {
                    "source": "vlm",
                    "model": model,
                    "condition": "image_caption",
                    "site": f"decoder:{18 if model == 'qwen' else 17}:{pooling}",
                    "language": lang,
                    "dimension": dim,
                }

        # Synthetic Stage 2 draw-0 SCREEN features per (pooling, target_lang)
        draw0_features = {}
        y_sign = np.ones((n_images, n_cats), dtype="<f8")
        for i, iid in enumerate(screen_ids):
            for c, v in enumerate(targets[iid]["values"]):
                y_sign[i, c] = 1.0 if (v == 1) else -1.0

        # Build features so en->en and all Stage 4 draw-0 logits match transform(X, stats) @ coef.T + intercept
        for p_idx, pooling in enumerate(POOLINGS):
            for l_idx, t_lang in enumerate(LANGUAGES):
                base_x = rng.normal(0.0, 1.0, size=(n_images, dim)).astype("<f4")
                # Inject class-correlated signal into feature 0..dim-1
                draw0_features[(pooling, t_lang)] = base_x

        rows_template = tuple({"image_id": iid, "split": "SCREEN", "language": "en"} for iid in screen_ids)
        blocks_meta = _fake_e29_blocks(model)
        stage4_pred_map = {}
        for cond in CONDITIONS:
            for pooling in POOLINGS:
                for t_lang in LANGUAGES:
                    entry = blocks_meta[cond][pooling][t_lang]
                    b_preds = {}
                    for s_lang in LANGUAGES:
                        rid = entry["routes"][s_lang]
                        if cond == "image_caption":
                            coef, intercept, est, prev, stats = frozen_heads[(pooling, s_lang)]
                            x_t = draw0_features[(pooling, t_lang)]
                            z = np.asarray(transform(x_t, stats) @ coef.T, dtype="<f8") + intercept
                        else:
                            est = np.ones((n_cats,), dtype=bool)
                            prev = np.full((n_cats,), 0.5, dtype="<f8")
                            z = (1.5 if s_lang == t_lang else 0.5) * y_sign
                        b_preds[rid] = Prediction(
                            reference=entry["prediction_ref"],
                            route={"route_id": rid},
                            split="SCREEN",
                            rows=rows_template,
                            logits=z,
                            estimable=est,
                            prevalence=prev,
                        )
                    stage4_pred_map[entry["block_id"]] = b_preds

        fit_dict = {"targets": targets, "groups": fit_groups}

        class _PlanObj(dict):
            pass

        s4_plan = _PlanObj({
            "screen_ids": list(screen_ids),
            "categories": list(categories),
            "ordinary": ordinary,
            "fit": fit_dict,
        })
        s4_plan.reference = {"path": "results/stage4/plan/_COMPLETE.json", "sha256": "1" * 64}
        s4_plan.fit = fit_dict
        s4_plan.store = None

        dataset_fp = fingerprint(["e07-score-fixture", model, n_images])
        stage4_rel = {"path": "results/stage4/releases/fix/_COMPLETE.json", "sha256": fingerprint(["s4", dataset_fp])}
        binding = Phase1Binding(
            root=root,
            protocol=frozen(protocol),
            stage4_release_ref=frozen(stage4_rel),
            stage4_release=frozen({"predictions": {}}),
            stage4_plan=s4_plan,
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
            lineage_identity=fingerprint(["e07-score-lineage", dataset_fp, bootstrap_sha256]),
        )

        # Build E07 audit (passed_restricted: en/hi/te changed, bn = 0 changed!)
        # For en, hi, te: 24 changed in draw_1, 20 changed in draw_2 (12 shared across draws 1 & 2)
        per_lang_views = {"en": [], "hi": [], "bn": [], "te": []}
        draw_buckets = {}
        for lang in LANGUAGES:
            if lang == "bn":
                draw_buckets[lang] = {
                    "draw_1": {"same_id_as_draw0": n_images, "distinct_id_same_text_as_draw0": 0, "changed_text_vs_draw0": 0},
                    "draw_2": {"same_id_as_draw0": n_images, "distinct_id_same_text_as_draw0": 0, "changed_text_vs_draw0": 0},
                }
                continue
            # Indices 0..23 changed in draw_1; indices 12..31 changed in draw_2
            # Indices 12..23 share the exact same changed caption across draw_1 & draw_2 (dedup!)
            for i in range(0, 12):
                txt = f"{lang}-alt-d1-{i}"
                per_lang_views[lang].append({
                    "image_id": screen_ids[i],
                    "image_path": f"datasets/xm3600/img/{screen_ids[i]}.jpg",
                    "image_sha256": fingerprint(["img", screen_ids[i]]),
                    "language": lang,
                    "split": "SCREEN",
                    "caption_id": f"cap:{screen_ids[i]}:{lang}:1",
                    "caption_text": txt,
                    "caption_text_sha256": hashlib.sha256(txt.encode("utf-8")).hexdigest(),
                    "draw_keys": [1],
                    "aliased_caption_ids": [f"cap:{screen_ids[i]}:{lang}:1"],
                })
            for i in range(12, 24):
                txt = f"{lang}-alt-d12-{i}"
                per_lang_views[lang].append({
                    "image_id": screen_ids[i],
                    "image_path": f"datasets/xm3600/img/{screen_ids[i]}.jpg",
                    "image_sha256": fingerprint(["img", screen_ids[i]]),
                    "language": lang,
                    "split": "SCREEN",
                    "caption_id": f"cap:{screen_ids[i]}:{lang}:12",
                    "caption_text": txt,
                    "caption_text_sha256": hashlib.sha256(txt.encode("utf-8")).hexdigest(),
                    "draw_keys": [1, 2],
                    "aliased_caption_ids": [f"cap:{screen_ids[i]}:{lang}:12"],
                })
            for i in range(24, 32):
                txt = f"{lang}-alt-d2-{i}"
                per_lang_views[lang].append({
                    "image_id": screen_ids[i],
                    "image_path": f"datasets/xm3600/img/{screen_ids[i]}.jpg",
                    "image_sha256": fingerprint(["img", screen_ids[i]]),
                    "language": lang,
                    "split": "SCREEN",
                    "caption_id": f"cap:{screen_ids[i]}:{lang}:2",
                    "caption_text": txt,
                    "caption_text_sha256": hashlib.sha256(txt.encode("utf-8")).hexdigest(),
                    "draw_keys": [2],
                    "aliased_caption_ids": [f"cap:{screen_ids[i]}:{lang}:2"],
                })
            draw_buckets[lang] = {
                "draw_1": {"same_id_as_draw0": n_images - 24, "distinct_id_same_text_as_draw0": 0, "changed_text_vs_draw0": 24},
                "draw_2": {"same_id_as_draw0": n_images - 20, "distinct_id_same_text_as_draw0": 0, "changed_text_vs_draw0": 20},
            }

        ep_chunks = []
        c_idx = 0
        for lang in LANGUAGES:
            rows = per_lang_views[lang]
            if not rows:
                continue
            ep_chunks.append({
                "chunk_id": f"chunk_{c_idx:04d}",
                "language": lang,
                "split": "SCREEN",
                "n_rows": len(rows),
                "rows": rows,
            })
            c_idx += 1

        extraction_plan = {
            "schema": "phase2.e07_extraction_plan.v1",
            "split": "SCREEN",
            "confirm_rows": 0,
            "condition": "image_caption",
            "depth_policy": "fixed_midpoint",
            "poolings": list(POOLINGS),
            "same_forward_both_poolings": True,
            "chunk_size": 32,
            "total_unique_views": sum(len(per_lang_views[l]) for l in LANGUAGES),
            "total_chunks": len(ep_chunks),
            "per_language_counts": {l: len(per_lang_views[l]) for l in LANGUAGES},
            "chunks": ep_chunks,
        }

        audit_body = {
            "schema": SCHEMA_E07_AUDIT,
            "stage4_release": stage4_rel,
            "lineage_identity": binding.lineage_identity,
            "dataset_ref": plain(binding.dataset_ref),
            "dataset_fingerprint": dataset_fp,
            "draw0_verified": True,
            "confirm_excluded": True,
            "verdict": "passed_restricted",
            "verdict_reasons": ["bn has zero changed views"],
            "screen_multiplicity": {},
            "fit_dev_s_multiplicity": {},
            "draw_buckets": draw_buckets,
            "support_tables": {},
            "unique_changed_views": {
                "per_language": {l: len(per_lang_views[l]) for l in LANGUAGES},
                "total": extraction_plan["total_unique_views"],
            },
            "extraction_plan": extraction_plan,
        }
        audit_ref = publish_phase2(
            root, protocol, dataset_fp, "phase2/audits/e07", KIND_E07_AUDIT, audit_body, _producer()
        )

        # Publish phase2_plan binding audit_ref
        with (
            patch("src.phase2.plan.bind_phase1", return_value=binding),
            patch("src.phase2.plan.e29_screen_blocks", side_effect=lambda b, m: frozen(_fake_e29_blocks(m))),
            patch("src.phase2.plan.e11_fit_statistics_refs", side_effect=lambda b, m: frozen(_fake_e11_stats(m))),
        ):
            plan = prepare_phase2(root, protocol, stage4_rel, _producer(), e07_audit_ref=audit_ref)

        # Publish real Stage 2 E29 analysis block (produces authentic draw0_reference!)
        with (
            patch("src.phase2.run_e29.bind_phase1", return_value=binding),
            patch("src.phase2.run_e29.e29_screen_blocks", return_value=frozen(blocks_meta)),
            patch("src.phase2.run_e29.load_bootstrap", return_value=draws),
            patch("src.phase2.run_e29.load_prediction_block", side_effect=lambda p, ref, expected_block=None: stage4_pred_map[expected_block]),
        ):
            e29_ref, e29_body, _, _ = run_e29(root, protocol, plan.reference, model, _producer())

        # Publish Stage 2 E07 chunks and release
        depth = 18 if model == "qwen" else 17
        model_fp = fingerprint(["model-fp", model])
        exec_fp = fingerprint(["exec-fp", model])
        rel_chunks = []
        for idx, ep_chunk in enumerate(ep_chunks):
            lang = ep_chunk["language"]
            rows = ep_chunk["rows"]
            n_v = len(rows)
            chunk_rows = [
                {
                    "row_index": r_i,
                    "image_id": r["image_id"],
                    "language": lang,
                    "split": "SCREEN",
                    "caption_id": r["caption_id"],
                    "caption_text": r["caption_text"],
                    "caption_text_sha256": r["caption_text_sha256"],
                    "draw_keys": list(r["draw_keys"]),
                    "aliased_caption_ids": list(r["aliased_caption_ids"]),
                    "image_path": r["image_path"],
                    "image_sha256": r["image_sha256"],
                }
                for r_i, r in enumerate(rows)
            ]
            rows_bytes = jsonl(chunk_rows)
            rows_sha = hashlib.sha256(rows_bytes).hexdigest()
            arr_last = np.ascontiguousarray(rng.normal(0.0, 1.0, size=(n_v, dim)), dtype="<f4")
            arr_mean = np.ascontiguousarray(rng.normal(0.0, 1.0, size=(n_v, dim)), dtype="<f4")
            rc = {
                "first_index": 0,
                "last_index": n_v - 1,
                "max_abs_diff_last_caption": 0.0,
                "max_abs_diff_caption_mean": 0.0,
                "passed": True,
            }
            c_body = {
                "schema": SCHEMA_E07_FEATURES,
                "experiment": "E07",
                "plan": plain(plan.reference),
                "plan_identity": plan["plan_identity"],
                "e07_audit_ref": plain(audit_ref),
                "model": model,
                "midpoint_depth": depth,
                "condition": "image_caption",
                "poolings": list(POOLINGS),
                "split": "SCREEN",
                "chunk_index": idx,
                "chunk_id": ep_chunk["chunk_id"],
                "language": lang,
                "total_chunks": len(ep_chunks),
                "n_views": n_v,
                "dimension": dim,
                "model_fingerprint": model_fp,
                "execution_fingerprint": exec_fp,
                "site_specs": {},
                "evidence_bridge_sha256": "a" * 64,
                "repeat_check": rc,
                "rows_sha256": rows_sha,
                "confirm_excluded": True,
            }
            c_ref = publish_phase2(
                root,
                protocol,
                dataset_fp,
                f"phase2/features/e07/{model}/run1",
                KIND_E07_CHUNK,
                c_body,
                _producer(),
                arrays={"last_caption": arr_last, "caption_mean": arr_mean},
                files={"rows.jsonl": rows_bytes, "token_evidence.jsonl": b"", "audits.json": json_bytes({"repeat_check": rc})},
                upstream=[plain(plan.reference), plain(audit_ref)],
                model_fingerprint=model_fp,
                run_id=f"chunk_{idx:04d}",
            )
            rel_chunks.append({
                "chunk_index": idx,
                "chunk_id": ep_chunk["chunk_id"],
                "language": lang,
                "n_views": n_v,
                "ref": plain(c_ref),
                "rows_sha256": rows_sha,
            })

        rel_body = {
            "schema": SCHEMA_E07_FEATURES,
            "experiment": "E07",
            "plan": plain(plan.reference),
            "plan_identity": plan["plan_identity"],
            "e07_audit_ref": plain(audit_ref),
            "e07_verdict": "passed_restricted",
            "model": model,
            "midpoint_depth": depth,
            "condition": "image_caption",
            "poolings": list(POOLINGS),
            "split": "SCREEN",
            "total_unique_views": extraction_plan["total_unique_views"],
            "per_language_counts": extraction_plan["per_language_counts"],
            "dimension": dim,
            "model_fingerprint": model_fp,
            "execution_fingerprint": exec_fp,
            "chunks": rel_chunks,
            "repeat_check_summary": {
                "chunks_checked": len(rel_chunks),
                "all_passed": True,
                "max_abs_diff_last_caption": 0.0,
                "max_abs_diff_caption_mean": 0.0,
            },
            "confirm_excluded": True,
        }
        release_ref = publish_phase2(
            root,
            protocol,
            dataset_fp,
            f"phase2/features/e07/{model}/releases",
            KIND_E07_RELEASE,
            rel_body,
            _producer(),
            upstream=[plain(plan.reference), plain(audit_ref)],
            model_fingerprint=model_fp,
            run_id="rel1",
        )

        return {
            "binding": binding,
            "plan": plan,
            "audit_ref": audit_ref,
            "release_ref": release_ref,
            "e29_ref": e29_ref,
            "e29_body": e29_body,
            "draws": draws,
            "blocks_meta": blocks_meta,
            "stage4_pred_map": stage4_pred_map,
            "frozen_heads": frozen_heads,
            "draw0_features": draw0_features,
            "screen_ids": screen_ids,
        }

    def _run_with_patches(self, root, protocol, fx, model="qwen", **kwargs):
        def _fake_read_features(binding, m, cond, pool, t_lang, split):
            rows = tuple({"image_id": iid, "split": split, "language": t_lang} for iid in fx["screen_ids"])
            return rows, fx["draw0_features"][(pool, t_lang)]

        def _fake_load_head(binding, m, cond, pool, s_lang):
            return fx["frozen_heads"][(pool, s_lang)]

        def _fake_midpoint_groups(binding, m, *, condition="image_caption", pooling="last_caption"):
            return frozen({lang: f"grp-{m}-{condition}-{pooling}-{lang}" for lang in LANGUAGES})

        with (
            patch("src.phase2.score_e07.bind_phase1", return_value=fx["binding"]),
            patch("src.phase2.score_e07.e29_screen_blocks", return_value=frozen(fx["blocks_meta"])),
            patch("src.phase2.score_e07.midpoint_groups", side_effect=_fake_midpoint_groups),
            patch("src.phase2.score_e07.load_frozen_source_head", side_effect=_fake_load_head),
            patch("src.phase2.score_e07.read_verified_midpoint_features", side_effect=_fake_read_features),
            patch("src.stage4.plan.load_bootstrap", return_value=fx["draws"]),
            patch(
                "src.stage4.predictions.load_prediction_block",
                side_effect=lambda p, ref, expected_block=None: fx["stage4_pred_map"][expected_block],
            ),
        ):
            return run_e07_scoring(
                root,
                protocol,
                fx["plan"].reference,
                fx["audit_ref"],
                fx["release_ref"],
                fx["e29_ref"],
                model,
                _producer(),
                **kwargs,
            )

    def test_end_to_end_zero_changed_bn_unchanged_reuse_draw0_parity_and_restricted_cohort(self):
        protocol = _protocol()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fx = self._build_full_fixture(root, protocol, model="qwen")
            events = []
            analysis_ref, analysis_body, pred_refs = self._run_with_patches(
                root, protocol, fx, model="qwen", progress=events.append
            )

            # 1. Streamed progress emitted
            self.assertTrue(any(e.get("event") == "e07_score_preflight_passed" for e in events))
            self.assertTrue(any(e.get("event") == "e07_score_published" for e in events))

            # 2. All 8 prediction blocks published; bn has (4, 0, 25) empty float64 logits
            for pooling in POOLINGS:
                for lang in LANGUAGES:
                    pref = pred_refs[pooling][lang]
                    pfolder = within(root, pref["path"]).parent
                    pbody = json.loads((pfolder / "body.json").read_text(encoding="utf-8"))
                    logits = np.load(pfolder / "logits.npy", allow_pickle=False)
                    self.assertEqual(logits.dtype, np.dtype("<f8"))
                    if lang == "bn":
                        self.assertEqual(pbody["n_changed_views"], 0)
                        self.assertEqual(logits.shape, (4, 0, 25))
                    else:
                        self.assertEqual(pbody["n_changed_views"], 32)
                        self.assertEqual(logits.shape, (4, 32, 25))

            # 3. Draw-0 self-checks and E29 draw0_reference parity
            for pooling in POOLINGS:
                self.assertTrue(analysis_body["draw0_self_checks"][pooling]["passed"])
                self.assertEqual(analysis_body["draw0_self_checks"][pooling]["max_abs_diff"], 0.0)
            self.assertTrue(analysis_body["draw0_parity_check"]["passed"])
            self.assertLessEqual(analysis_body["draw0_parity_check"]["e29_draw0_max_abs_diff"], 1e-12)
            self.assertEqual(analysis_body["draw0_parity_check"]["unchanged_views_atol"], 0.0)
            self.assertEqual(analysis_body["draw0_parity_check"]["unchanged_views_rtol"], 0.0)

            # 4. Unchanged bn rows match draw-0 bit-for-bit => zero shift on s->bn routes
            fs = analysis_body["full_screen_360"]
            for shift_key in ("draw_1_vs_draw_0", "draw_2_vs_draw_0"):
                shifts = fs["paired_shifts"][shift_key]
                self.assertEqual(shifts["changed_targets_only"]["eligible_target_languages"], ["en", "hi", "te"])
                self.assertEqual(shifts["changed_targets_only"]["n_routes"], 9)
                for src in ("en", "hi", "te"):
                    r_shift = shifts["per_route"][f"{src}->bn"]["metrics"]["auroc"]["g_last_shift"]["primary_macro"]["value"]
                    self.assertEqual(r_shift, 0.0)
                self.assertIn("pooling_reduction_sign_preserved", shifts)

            # 5. Restricted changed cohort: bn unavailable, en/hi/te paired with matched draw-0
            rc = analysis_body["restricted_changed_cohort"]
            for dk in ("draw_1", "draw_2"):
                d_rc = rc["draws"][dk]
                bn_rc = d_rc["per_target"]["bn"]
                self.assertEqual(bn_rc["status"], "unavailable")
                self.assertEqual(bn_rc["reason"], UNAVAILABLE_ZERO_CHANGED_REASON)
                self.assertEqual(bn_rc["n_images"], 0)
                for lang in ("en", "hi", "te"):
                    l_rc = d_rc["per_target"][lang]
                    self.assertIn(l_rc["status"], ("ok", "descriptive_low_support"))
                    self.assertGreater(l_rc["n_images"], 0)
                    self.assertIn("supported_primary_ge_10", l_rc["macros"])
                    self.assertIn("primary_14", l_rc["macros"])
                    self.assertIn("matched_draw_0", l_rc["macros"]["primary_14"]["target_mean"]["auroc"])
                    self.assertIn("pooling_reduction_sign_preserved", l_rc["macros"]["primary_14"])
                agg = d_rc["eligible_targets_aggregate"]
                self.assertEqual(agg["n_routes"], 9)
                self.assertEqual(agg["eligible_target_languages"], ["en", "hi", "te"])
                self.assertEqual(agg["unavailable_target_languages"], ["bn"])
                self.assertEqual(d_rc["intersections"]["four_language_intersection"]["n_images"], 0)
                self.assertGreater(d_rc["intersections"]["eligible_languages_intersection"]["n_images"], 0)

            # 6. Unknown-cell mask preserved on plant.n.02 (img-0000 is None => known == 39)
            plant_sup = fs["draws"]["draw_0"]["routes_all"]["last_caption"]["en->en"]["metrics"]["auroc"]["categories"]["plant.n.02"]["support"]
            self.assertEqual(plant_sup["known"], 39)

    def test_draw0_parity_failure_on_tampered_e29_reference(self):
        protocol = _protocol()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fx = self._build_full_fixture(root, protocol, model="qwen")
            # Publish a tampered E29 analysis block with a perturbed draw0_reference point estimate
            bad_e29 = plain(fx["e29_body"])
            bad_e29["draw0_reference"]["point_estimates"]["twelve_route_mean"]["auroc"]["g_last_mean"] += 1e-4
            bad_e29_ref = publish_phase2(
                root,
                protocol,
                fx["plan"]["dataset_fingerprint"],
                "phase2/results/e29/qwen",
                KIND_ANALYSIS,
                bad_e29,
                _producer(),
            )
            fx["e29_ref"] = bad_e29_ref
            with self.assertRaises(ContractError) as ctx:
                self._run_with_patches(root, protocol, fx, model="qwen")
            self.assertIn("parity", str(ctx.exception).lower())

    def test_en_en_self_check_failure_rejection(self):
        protocol = _protocol()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fx = self._build_full_fixture(root, protocol, model="qwen")
            # Perturb Stage 2 en features so en->en reproduction fails atol=0, rtol=0
            fx["draw0_features"][("last_caption", "en")] = fx["draw0_features"][("last_caption", "en")] + np.float32(1e-3)
            with self.assertRaises(ContractError) as ctx:
                self._run_with_patches(root, protocol, fx, model="qwen")
            self.assertIn("reproduction", str(ctx.exception).lower())

    def test_explicit_resume_success_and_rejection(self):
        protocol = _protocol()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fx = self._build_full_fixture(root, protocol, model="qwen")
            analysis_ref, analysis_body, pred_refs = self._run_with_patches(root, protocol, fx, model="qwen")

            # 1. Valid analysis-level resume
            res_ref, res_body, res_preds = self._run_with_patches(
                root,
                protocol,
                fx,
                model="qwen",
                resume_analysis_ref=analysis_ref,
                resume_prediction_refs={"last_caption:en": pred_refs["last_caption"]["en"]},
            )
            self.assertEqual(plain(res_ref), plain(analysis_ref))
            self.assertEqual(plain(res_preds), plain(pred_refs))

            # 2. Valid prediction-block-level resume (partial blocks)
            events = []
            _, _, pred_refs_2 = self._run_with_patches(
                root,
                protocol,
                fx,
                model="qwen",
                resume_prediction_refs={
                    "last_caption:en": pred_refs["last_caption"]["en"],
                    "caption_mean:bn": pred_refs["caption_mean"]["bn"],
                },
                progress=events.append,
            )
            resumed_events = [e for e in events if e.get("event") == "e07_prediction_block_resumed"]
            self.assertEqual(len(resumed_events), 2)
            self.assertEqual(pred_refs_2["last_caption"]["en"], pred_refs["last_caption"]["en"])

            # 3. Reject mismatched resume_prediction_refs vs analysis_ref
            with self.assertRaises(ContractError):
                self._run_with_patches(
                    root,
                    protocol,
                    fx,
                    model="qwen",
                    resume_analysis_ref=analysis_ref,
                    resume_prediction_refs={"last_caption:en": pred_refs["last_caption"]["hi"]},
                )

            # 4. Reject tampered prediction block on disk
            pfolder = within(root, pred_refs["last_caption"]["en"]["path"]).parent
            (pfolder / "logits.npy").write_bytes(b"corrupted")
            with self.assertRaises(ContractError):
                self._run_with_patches(
                    root,
                    protocol,
                    fx,
                    model="qwen",
                    resume_prediction_refs={"last_caption:en": pred_refs["last_caption"]["en"]},
                )

    def test_tampered_chunk_array_rejection(self):
        protocol = _protocol()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fx = self._build_full_fixture(root, protocol, model="qwen")
            rel_folder = within(root, fx["release_ref"]["path"]).parent
            rel_body = json.loads((rel_folder / "body.json").read_text(encoding="utf-8"))
            chunk0_folder = within(root, rel_body["chunks"][0]["ref"]["path"]).parent
            (chunk0_folder / "last_caption.npy").write_bytes(b"corrupted")
            with self.assertRaises(ContractError):
                self._run_with_patches(root, protocol, fx, model="qwen")


if __name__ == "__main__":
    unittest.main()
