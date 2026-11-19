"""Fixture and unit tests for Phase 2 Stage 1 E07 original-caption coverage audit."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from src.common.contracts import (
    LANGUAGES,
    ContractError,
    ModelInput,
    canonical_bytes,
    fingerprint,
    select_caption,
)
from src.common.storage import MARKER, file_sha256, publish
from src.phase2.e07_audit import (
    E07_CHUNK_SIZE,
    audit_e07_captions,
    load_e07_audit,
)
from src.phase2.interfaces import (
    KIND_E07_AUDIT,
    REPO,
    SCHEMA_E07_AUDIT,
    Phase1Binding,
    frozen,
    plain,
)


def _jsonl_bytes(rows):
    return b"".join(canonical_bytes(r) + b"\n" for r in rows)


def _json_bytes(obj):
    return canonical_bytes(obj) + b"\n"


def _make_id(idx: int) -> str:
    return f"{idx:016x}"


def _build_synthetic_stage1_dataset(
    root: Path,
    protocol: dict,
    *,
    screen_caption_factory,
    target_factory=None,
    tamper_draw0=None,
    leak_confirm_in_captions=False,
    run_tag="fixture",
):
    """Publish a verified Stage 1 dataset artifact on disk with ZERO raw image files.

    Uses a lightweight protocol copy whose split_sizes keep SCREEN=360 (matching
    production) and small FIT/DEV-S/DEV-C/CONFIRM splits so tests run in milliseconds.
    """
    proto = json.loads(json.dumps(protocol))
    proto["split_sizes"] = {
        "FIT": 8,
        "DEV-S": 4,
        "DEV-C": 4,
        "SCREEN": 360,
        "CONFIRM": 6,
    }
    protocol_fp = fingerprint(proto)
    category_order = proto["category_order"]
    primary_set = set(proto["primary_categories"])
    plant_idx = category_order.index("plant.n.02")

    # Reserve exact invalid_cell image_id 6a20677eee04fd08 in SCREEN.
    special_screen_id = "6a20677eee04fd08"
    screen_ids = [special_screen_id] + [_make_id(1000 + i) for i in range(359)]
    fit_ids = [_make_id(10 + i) for i in range(proto["split_sizes"]["FIT"])]
    dev_s_ids = [_make_id(100 + i) for i in range(proto["split_sizes"]["DEV-S"])]
    dev_c_ids = [_make_id(200 + i) for i in range(proto["split_sizes"]["DEV-C"])]
    confirm_ids = [_make_id(9000 + i) for i in range(proto["split_sizes"]["CONFIRM"])]

    partitions = {
        "FIT": fit_ids,
        "DEV-S": dev_s_ids,
        "DEV-C": dev_c_ids,
        "SCREEN": screen_ids,
        "CONFIRM": confirm_ids,
    }
    split_by_id = {iid: s for s, ids in partitions.items() for iid in ids}
    all_ids = sorted(split_by_id)

    # Targets: balanced across all 25 categories, preserving None on 6a20677eee04fd08 / plant.n.02.
    targets = {}
    screen_pos_map = {iid: idx for idx, iid in enumerate(screen_ids)}
    for iid in all_ids:
        if target_factory is not None:
            vals = list(target_factory(iid, split_by_id[iid], screen_pos_map.get(iid)))
        else:
            rank = screen_pos_map.get(iid, int(iid, 16))
            vals = [1 if ((rank + k) % 2 == 0) else 0 for k in range(25)]
            if iid == special_screen_id:
                vals[plant_idx] = None
        targets[iid] = vals

    # Captions & draw-0 records.
    records = []
    views = {lang: [] for lang in LANGUAGES}
    dummy_img_sha = "1" * 64
    release_sha = "ad44c380c86cb1240666361caabaf5684507d0ef"

    for iid in all_ids:
        split = split_by_id[iid]
        img_path = f"datasets/xm3600/{'2' * 64}/images/{iid}.jpg"
        for lang in LANGUAGES:
            if split == "SCREEN":
                texts = screen_caption_factory(iid, screen_pos_map[iid], lang)
            elif split == "FIT":
                # Mix of multiplicity 1, 2, 3 and duplicate texts to exercise FIT summary.
                idx_num = int(iid, 16) % 3
                if idx_num == 0:
                    texts = [f"fit-{lang}-{iid}-only"]
                elif idx_num == 1:
                    texts = [f"fit-{lang}-{iid}-dup", f"fit-{lang}-{iid}-dup"]
                else:
                    texts = [f"fit-{lang}-{iid}-a", f"fit-{lang}-{iid}-b", f"fit-{lang}-{iid}-c"]
            elif split == "DEV-S":
                texts = [f"devs-{lang}-{iid}-a", f"devs-{lang}-{iid}-b"]
            elif split == "CONFIRM":
                # Multi-caption CONFIRM rows: must never enter select_caption or audit outputs.
                texts = [f"confirm-{lang}-{iid}-0", f"confirm-{lang}-{iid}-1", f"confirm-{lang}-{iid}-2"]
            else:
                texts = [f"devc-{lang}-{iid}-0"]

            candidates = []
            for j, text in enumerate(texts):
                cid = f"xm3600:{release_sha}:{iid}:{lang}:{j}"
                mi = ModelInput(iid, img_path, dummy_img_sha, lang, cid, text, split)
                candidates.append(mi)
                row_split = "SCREEN" if (leak_confirm_in_captions and split == "CONFIRM" and j == 0) else split
                views[lang].append({
                    "image_id": iid,
                    "image_path": img_path,
                    "image_sha256": dummy_img_sha,
                    "language": lang,
                    "caption_id": cid,
                    "caption_text": text,
                    "split": row_split,
                    "source_index": j,
                    "source_field": f"{lang}.caption[{j}]",
                })
            chosen = select_caption(candidates, 0)
            rec = {
                "image_id": chosen.image_id,
                "image_path": chosen.image_path,
                "image_sha256": chosen.image_sha256,
                "language": chosen.language,
                "caption_id": chosen.caption_id,
                "caption_text": chosen.caption_text,
                "split": chosen.split,
            }
            if tamper_draw0 is not None and split == "SCREEN":
                rec = tamper_draw0(rec, screen_pos_map[iid])
            records.append(rec)

    record_bytes = _jsonl_bytes(records)
    target_bytes = _jsonl_bytes(
        [{"image_id": iid, "split": split_by_id[iid], "values": targets[iid]} for iid in all_ids]
    )
    split_bytes = _json_bytes(partitions)
    identity = {
        "sources": {"fixture": run_tag},
        "category_order": category_order,
        "split_manifest_sha256": hashlib.sha256(split_bytes).hexdigest(),
        "caption_recipe": "select_caption stage1.v1; draw_key=0",
        "records_sha256": hashlib.sha256(record_bytes).hexdigest(),
        "targets_sha256": hashlib.sha256(target_bytes).hexdigest(),
    }
    dfp = fingerprint(identity)
    dataset_doc = dict(
        identity,
        schema_version="stage1.v1",
        dataset_fingerprint=dfp,
        protocol_fingerprint=protocol_fp,
        source_artifact="datasets/xm3600/" + "2" * 64,
    )
    files = {
        "dataset.json": _json_bytes(dataset_doc),
        "records.jsonl": record_bytes,
        "targets.jsonl": target_bytes,
        "splits.json": split_bytes,
        **{f"captions/{lang}.jsonl": _jsonl_bytes(views[lang]) for lang in LANGUAGES},
    }
    meta = {
        "kind": "stage1_dataset",
        "status": "passed",
        "run_id": f"20261119T000000Z-{run_tag}",
        "code_commit": "4" * 40,
        "protocol_fingerprint": protocol_fp,
        "dataset_fingerprint": dfp,
        "model_fingerprint": None,
        "seeds": {"caption_draw": 0, "split": proto["split_seed"]},
        "software_versions": {"python": "3.12.0"},
        "upstream_artifacts": [],
        "storage_scope": "local-development",
    }
    ns = f"shared/{protocol_fp}/manifests/{dfp}"
    folder = publish(root, ns, meta, files)
    marker_path = folder / MARKER
    dataset_ref = {
        "path": marker_path.relative_to(root).as_posix(),
        "sha256": file_sha256(marker_path),
    }

    stage4_ref = {
        "path": "results/stage4/releases/fixture/_COMPLETE.json",
        "sha256": "9" * 64,
    }
    binding = Phase1Binding(
        root=root.resolve(),
        protocol=frozen(proto),
        stage4_release_ref=frozen(stage4_ref),
        stage4_release=frozen({}),
        stage4_plan=None,
        bootstrap_shape=(2000, 360),
        bootstrap_sha256="8" * 64,
        midpoints=frozen({"qwen": 18, "gemma": 17}),
        dataset_ref=frozen(dataset_ref),
        data_plan_ref=frozen({"path": "results/stage2/data_plan/_COMPLETE.json", "sha256": "7" * 64}),
        model_plan_refs=frozen({}),
        suite_ref=frozen({"path": "results/stage2/suite/_COMPLETE.json", "sha256": "6" * 64}),
        stage3_registry_ref=frozen({"path": "results/stage3/reg/_COMPLETE.json", "sha256": "5" * 64}),
        g0_gate_ref=frozen({"path": "results/stage3/g0/_COMPLETE.json", "sha256": "4" * 64}),
        g0_p_value=0.362,
        dataset_fingerprint=dfp,
        protocol_fingerprint=protocol_fp,
        lineage_identity=fingerprint({"dataset": dfp, "stage4": stage4_ref}),
    )
    return binding, proto, partitions


class E07AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base_protocol = json.loads((REPO / "configs/protocol.json").read_text(encoding="utf-8"))
        self.producer = {
            "code_commit": "4" * 40,
            "storage_scope": "local-development",
            "software_versions": {"python": "3.12.0"},
        }

    def test_passed_full_dedup_unknown_mask_confirm_exclusion_and_resume(self):
        # 5 distinct captions per (image, lang) -> high change rate across all 4 languages (>250/360).
        # Also inject explicit byte-identical duplicate texts on specific rows to verify:
        #   (a) distinct_id_same_text_as_draw0 reuse (0 GPU forwards),
        #   (b) deduplication across draws 1 and 2 when both select identical changed text.
        def caption_factory(iid, pos, lang):
            if pos == 0:
                # Special image 6a20677eee04fd08: 5 distinct captions so it enters changed subsets too.
                return [f"special-{lang}-{j}" for j in range(5)]
            if pos < 20:
                # All 4 candidates have byte-identical text -> distinct_id_same_text_as_draw0 or same_id_as_draw0!
                return [f"identical-to-draw0-{lang}-{iid}"] * 4
            if pos < 40:
                # Construct 2 distinct texts across 4 caption IDs so draw 1 and draw 2 often collide on the same changed text.
                return [
                    f"pairA-{lang}-{iid}",
                    f"pairB-{lang}-{iid}",
                    f"pairB-{lang}-{iid}",
                    f"pairB-{lang}-{iid}",
                ]
            return [f"cap-{lang}-{iid}-{j}" for j in range(5)]

        binding, proto, partitions = _build_synthetic_stage1_dataset(
            self.root, self.base_protocol, screen_caption_factory=caption_factory, run_tag="full"
        )
        confirm_set = set(partitions["CONFIRM"])
        events = []

        # Wrap select_caption to assert CONFIRM is NEVER passed to select_caption.
        real_select = select_caption
        def guarded_select(candidates, draw_key=0):
            for c in candidates:
                if c.split == "CONFIRM" or c.image_id in confirm_set:
                    raise AssertionError("CONFIRM reached select_caption!")
            return real_select(candidates, draw_key)

        with mock.patch("src.phase2.e07_audit.select_caption", side_effect=guarded_select):
            audit_ref = audit_e07_captions(binding, self.producer, progress=events.append)

        # Verify zero raw images exist anywhere under root (proving zero raw image IO).
        self.assertEqual(list(self.root.rglob("*.jpg")), [])

        ref, body, marker = load_e07_audit(
            self.root,
            proto,
            audit_ref,
            expected_dataset=binding.dataset_fingerprint,
            expected_lineage_identity=binding.lineage_identity,
        )
        self.assertEqual(plain(ref), plain(audit_ref))
        self.assertEqual(marker["kind"], KIND_E07_AUDIT)
        self.assertEqual(body["schema"], SCHEMA_E07_AUDIT)
        self.assertTrue(body["draw0_verified"])
        self.assertTrue(body["confirm_excluded"])
        self.assertEqual(body["verdict"], "passed_full")
        self.assertFalse(body["dual_reporting_required"])

        # Verify byte-identical text reuse across caption_ids (pos 1..19).
        self.assertGreater(body["unique_changed_views"]["reused_from_draw0_distinct_id_same_text"], 0)
        # Verify deduplication across draws 1 & 2 (same changed text selected in both draws).
        self.assertGreater(body["unique_changed_views"]["deduplicated_across_draws_1_and_2"], 0)
        self.assertEqual(
            body["unique_changed_views"]["total_unique_views"],
            body["unique_changed_views"]["total_changed_draw_pairs"]
            - body["unique_changed_views"]["deduplicated_across_draws_1_and_2"],
        )

        # Verify mutually exclusive buckets sum to 360 for every (lang, draw_k).
        for lang in LANGUAGES:
            for dk in ("draw_1", "draw_2"):
                b = body["draw_buckets"][lang][dk]
                self.assertEqual(
                    b["same_id_as_draw0"] + b["distinct_id_same_text_as_draw0"] + b["changed_text_vs_draw0"],
                    360,
                )

        # Verify FIT and DEV-S read-only multiplicity summaries exist and exclude CONFIRM.
        self.assertEqual(set(body["fit_dev_s_multiplicity"]), {"FIT", "DEV-S"})
        for lang in LANGUAGES:
            self.assertEqual(body["fit_dev_s_multiplicity"]["FIT"][lang]["n_images"], 8)
            self.assertEqual(body["fit_dev_s_multiplicity"]["DEV-S"][lang]["n_images"], 4)
            self.assertEqual(body["screen_multiplicity"][lang]["n_images"], 360)

        # Verify unknown-mask preservation on 6a20677eee04fd08 / plant.n.02.
        plant_full = body["support_tables"]["full_screen"]["categories"]["plant.n.02"]
        self.assertEqual(plant_full["unknown"], 1)
        self.assertEqual(plant_full["known"], 359)
        self.assertEqual(len(body["support_tables"]["full_screen"]["categories"]), 25)
        self.assertTrue(body["support_tables"]["full_screen"]["all_primary_ge_20"])

        # Verify 32-image SCREEN-only extraction plan with 0 CONFIRM rows and deduplicated keys.
        ep = body["extraction_plan"]
        self.assertIsNotNone(ep)
        self.assertEqual(ep["split"], "SCREEN")
        self.assertEqual(ep["confirm_rows"], 0)
        self.assertEqual(ep["condition"], "image_caption")
        self.assertEqual(ep["depth_policy"], "fixed_midpoint")
        self.assertEqual(list(ep["poolings"]), ["last_caption", "caption_mean"])
        self.assertTrue(ep["same_forward_both_poolings"])
        self.assertEqual(ep["chunk_size"], E07_CHUNK_SIZE)

        seen_dedup_keys = set()
        total_plan_rows = 0
        for ch in ep["chunks"]:
            self.assertEqual(ch["split"], "SCREEN")
            self.assertLessEqual(ch["n_rows"], 32)
            self.assertEqual(ch["n_rows"], len(ch["rows"]))
            for row in ch["rows"]:
                self.assertEqual(row["split"], "SCREEN")
                self.assertNotIn(row["image_id"], confirm_set)
                key = (row["image_id"], row["language"], row["caption_text_sha256"])
                self.assertNotIn(key, seen_dedup_keys)
                seen_dedup_keys.add(key)
                total_plan_rows += 1
        self.assertEqual(total_plan_rows, body["unique_changed_views"]["total_unique_views"])
        self.assertEqual(total_plan_rows, ep["total_unique_views"])

        # Verify streamed progress events.
        self.assertGreaterEqual(len(events), 6)
        self.assertEqual(events[0]["event"], "e07_audit_start")
        self.assertEqual(events[-1]["event"], "e07_audit_complete")
        for ev in events:
            json.dumps(ev)  # Must be strictly JSON-serializable.

        # Verify explicit-ref resume returns identical ref without creating a new directory.
        audit_dirs_before = sorted((self.root / "phase2/audits/e07").iterdir())
        resume_events = []
        resumed_ref = audit_e07_captions(
            binding, self.producer, progress=resume_events.append, resume_ref=audit_ref
        )
        audit_dirs_after = sorted((self.root / "phase2/audits/e07").iterdir())
        self.assertEqual(plain(resumed_ref), plain(audit_ref))
        self.assertEqual(audit_dirs_before, audit_dirs_after)
        self.assertEqual(resume_events[-1]["event"], "e07_audit_resumed")

        # Tampered resume ref must be rejected.
        with self.assertRaises(ContractError):
            audit_e07_captions(
                binding,
                self.producer,
                resume_ref={"path": audit_ref["path"], "sha256": "0" * 64},
            )

    def test_draw0_parity_failure_blocks_immediately(self):
        def caption_factory(iid, pos, lang):
            return [f"cap-{lang}-{iid}-{j}" for j in range(3)]

        # Case 1: caption_id mismatch on one SCREEN row.
        def tamper_id(rec, pos):
            if pos == 5 and rec["language"] == "hi":
                rec = dict(rec, caption_id=rec["caption_id"] + ":tampered")
            return rec

        binding1, _, _ = _build_synthetic_stage1_dataset(
            self.root,
            self.base_protocol,
            screen_caption_factory=caption_factory,
            tamper_draw0=tamper_id,
            run_tag="tamper_id",
        )
        with self.assertRaisesRegex(ContractError, "Draw-0 parity mismatch"):
            audit_e07_captions(binding1, self.producer)
        self.assertFalse((self.root / "phase2/audits/e07").exists())

        # Case 2: caption_text mismatch on one SCREEN row.
        def tamper_text(rec, pos):
            if pos == 7 and rec["language"] == "bn":
                rec = dict(rec, caption_text=rec["caption_text"] + " altered")
            return rec

        binding2, _, _ = _build_synthetic_stage1_dataset(
            self.root,
            self.base_protocol,
            screen_caption_factory=caption_factory,
            tamper_draw0=tamper_text,
            run_tag="tamper_text",
        )
        with self.assertRaisesRegex(ContractError, "Draw-0 parity mismatch"):
            audit_e07_captions(binding2, self.producer)
        self.assertFalse((self.root / "phase2/audits/e07").exists())

    def test_unknown_mask_violation_rejected(self):
        # Imputing 6a20677eee04fd08 / plant.n.02 to 0 instead of None must fail immediately.
        def bad_targets(iid, split, screen_pos):
            return [1 if (k % 2 == 0) else 0 for k in range(25)]

        binding, _, _ = _build_synthetic_stage1_dataset(
            self.root,
            self.base_protocol,
            screen_caption_factory=lambda iid, pos, lang: [f"c-{lang}-{iid}-{j}" for j in range(3)],
            target_factory=bad_targets,
            run_tag="bad_mask",
        )
        with self.assertRaisesRegex(ContractError, "Unknown-cell mask violated"):
            audit_e07_captions(binding, self.producer)

    def test_confirm_leakage_in_captions_rejected(self):
        binding, _, _ = _build_synthetic_stage1_dataset(
            self.root,
            self.base_protocol,
            screen_caption_factory=lambda iid, pos, lang: [f"c-{lang}-{iid}-{j}" for j in range(3)],
            leak_confirm_in_captions=True,
            run_tag="confirm_leak",
        )
        with self.assertRaisesRegex(ContractError, "Caption split mismatch"):
            audit_e07_captions(binding, self.producer)

    def test_verdict_passed_restricted_single_caption_one_language(self):
        # en, hi, bn have 5 distinct captions (>250 changed); te has only 1 caption (0 changed -> unavailable_single_caption).
        def caption_factory(iid, pos, lang):
            if lang == "te":
                return [f"only-te-{iid}"]
            return [f"multi-{lang}-{iid}-{j}" for j in range(5)]

        binding, proto, _ = _build_synthetic_stage1_dataset(
            self.root, self.base_protocol, screen_caption_factory=caption_factory, run_tag="restricted_1lang"
        )
        audit_ref = audit_e07_captions(binding, self.producer)
        _, body, _ = load_e07_audit(self.root, proto, audit_ref)
        self.assertEqual(body["verdict"], "passed_restricted")
        self.assertTrue(body["dual_reporting_required"])
        self.assertEqual(body["language_coverage_status"]["te"]["status"], "unavailable_single_caption")
        self.assertEqual(body["unique_changed_views"]["per_language"]["te"], 0)
        self.assertIsNotNone(body["extraction_plan"])
        self.assertEqual(body["extraction_plan"]["per_language_counts"]["te"], 0)
        self.assertEqual(body["extraction_plan"]["confirm_rows"], 0)

    def test_verdict_passed_restricted_moderate_change_rate(self):
        # All 4 languages have ~80 changed images (between 60 and 99 -> moderate_change_rate).
        def caption_factory(iid, pos, lang):
            if pos < 95:
                return [f"m-{lang}-{iid}-{j}" for j in range(6)]
            return [f"single-{lang}-{iid}"]

        binding, proto, _ = _build_synthetic_stage1_dataset(
            self.root, self.base_protocol, screen_caption_factory=caption_factory, run_tag="restricted_mod"
        )
        audit_ref = audit_e07_captions(binding, self.producer)
        _, body, _ = load_e07_audit(self.root, proto, audit_ref)
        self.assertEqual(body["verdict"], "passed_restricted")
        self.assertTrue(body["dual_reporting_required"])
        self.assertIsNotNone(body["extraction_plan"])
        for lang in LANGUAGES:
            self.assertEqual(body["language_coverage_status"][lang]["status"], "moderate_change_rate")

    def test_verdict_blocked_single_caption_all_languages(self):
        # Every SCREEN image has only 1 caption -> 0 changed views -> blocked, extraction_plan=None.
        binding, proto, _ = _build_synthetic_stage1_dataset(
            self.root,
            self.base_protocol,
            screen_caption_factory=lambda iid, pos, lang: [f"single-{lang}-{iid}"],
            run_tag="blocked_single",
        )
        audit_ref = audit_e07_captions(binding, self.producer)
        _, body, _ = load_e07_audit(self.root, proto, audit_ref)
        self.assertEqual(body["verdict"], "blocked")
        self.assertIsNone(body["extraction_plan"])
        self.assertTrue(body["draw0_verified"])
        self.assertTrue(body["confirm_excluded"])
        self.assertEqual(body["unique_changed_views"]["total_unique_views"], 0)
        self.assertGreaterEqual(len(body["verdict_reasons"]), 1)

    def test_verdict_blocked_fewer_than_3_languages_at_quorum(self):
        # Only en and hi reach >= 60 changed views; bn and te have single captions -> blocked!
        def caption_factory(iid, pos, lang):
            if lang in ("en", "hi"):
                return [f"multi-{lang}-{iid}-{j}" for j in range(5)]
            return [f"single-{lang}-{iid}"]

        binding, proto, _ = _build_synthetic_stage1_dataset(
            self.root, self.base_protocol, screen_caption_factory=caption_factory, run_tag="blocked_2langs"
        )
        audit_ref = audit_e07_captions(binding, self.producer)
        _, body, _ = load_e07_audit(self.root, proto, audit_ref)
        self.assertEqual(body["verdict"], "blocked")
        self.assertIsNone(body["extraction_plan"])
        self.assertTrue(any("only 2/4 languages" in r for r in body["verdict_reasons"]))

    def test_verdict_blocked_full_screen_support_failure(self):
        # Break primary category 0 (vehicle.n.01) so SCREEN has only 5 positives (< 20).
        plant_idx = self.base_protocol["category_order"].index("plant.n.02")

        def low_support_targets(iid, split, screen_pos):
            rank = screen_pos if screen_pos is not None else int(iid, 16)
            vals = [1 if ((rank + k) % 2 == 0) else 0 for k in range(25)]
            if split == "SCREEN":
                vals[0] = 1 if screen_pos < 5 else 0
            if iid == "6a20677eee04fd08":
                vals[plant_idx] = None
            return vals

        binding, proto, _ = _build_synthetic_stage1_dataset(
            self.root,
            self.base_protocol,
            screen_caption_factory=lambda iid, pos, lang: [f"m-{lang}-{iid}-{j}" for j in range(5)],
            target_factory=low_support_targets,
            run_tag="blocked_support",
        )
        audit_ref = audit_e07_captions(binding, self.producer)
        _, body, _ = load_e07_audit(self.root, proto, audit_ref)
        self.assertEqual(body["verdict"], "blocked")
        self.assertIsNone(body["extraction_plan"])
        self.assertFalse(body["support_tables"]["full_screen"]["all_primary_ge_20"])

    def test_plan_wrapper_propagates_plan_identity(self):
        class DummyPlan:
            def __init__(self, binding, plan_identity):
                self.binding = binding
                self.plan_identity = plan_identity

        binding, proto, _ = _build_synthetic_stage1_dataset(
            self.root,
            self.base_protocol,
            screen_caption_factory=lambda iid, pos, lang: [f"m-{lang}-{iid}-{j}" for j in range(5)],
            run_tag="plan_id",
        )
        plan_id = "c" * 64
        plan = DummyPlan(binding, plan_id)
        audit_ref = audit_e07_captions(plan, self.producer)
        _, body, _ = load_e07_audit(self.root, proto, audit_ref, expected_plan_identity=plan_id)
        self.assertEqual(body["plan_identity"], plan_id)

        # Resume with matching plan succeeds; mismatched plan_identity fails.
        resumed = audit_e07_captions(plan, self.producer, resume_ref=audit_ref)
        self.assertEqual(plain(resumed), plain(audit_ref))
        with self.assertRaisesRegex(ContractError, "plan_identity mismatch"):
            audit_e07_captions(DummyPlan(binding, "d" * 64), self.producer, resume_ref=audit_ref)

    def test_notebook_structure(self):
        nb_path = REPO / "scripts/phase2/audit_e07.ipynb"
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertEqual(nb["nbformat"], 4)
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        self.assertEqual(len(code_cells), 3)
        setup_src = "".join(code_cells[0]["source"])
        settings_src = "".join(code_cells[1]["source"])
        exec_src = "".join(code_cells[2]["source"])
        self.assertIn("requirements-stage2-cpu.txt", setup_src)
        self.assertIn("OPENBLAS_NUM_THREADS", setup_src)
        self.assertIn("STAGE4_RELEASE_REF", settings_src)
        self.assertIn("RESUME_REF", settings_src)
        self.assertIn("src.phase2.e07_audit", exec_src)
        self.assertIn("'-u'", exec_src)
        self.assertNotIn("capture_output=True", exec_src)


if __name__ == "__main__":
    unittest.main()
