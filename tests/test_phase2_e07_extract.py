"""Unit and fixture tests for Phase 2 Stage 2 Branch C E07 L4 GPU extraction."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

try:
    import numpy as np
except ImportError:
    np = None

from src.common.contracts import (
    LANGUAGES,
    ContractError,
    canonical_bytes,
    fingerprint,
)
from src.common.storage import MARKER, file_sha256, publish
from src.models.hf import HFAdapter
from src.phase2.interfaces import (
    KIND_E07_AUDIT,
    KIND_E07_CHUNK,
    KIND_E07_RELEASE,
    KIND_PLAN,
    PHASE2_PARENT_DIRS,
    PHASE2_VERSION,
    REPO,
    SCHEMA_E07_AUDIT,
    SCHEMA_E07_FEATURES,
    SCHEMA_PLAN,
    Phase1Binding,
    frozen,
    init_phase2_dirs,
    plain,
    publish_phase2,
    storage_probe,
)
from src.stage2.evidence import bridge_hash

if np is not None:
    from src.phase2.extract_e07 import (
        preflight_e07_extraction,
        run_e07_extraction,
        verify_e07_release,
    )


def _json_bytes(obj):
    return canonical_bytes(obj) + b"\n"


def _jsonl_bytes(rows):
    return b"".join(canonical_bytes(r) + b"\n" for r in rows)


class FakeHFAdapter(HFAdapter):
    """Explicit HFAdapter fixture subclass for CPU-only E07 extraction tests."""

    def __init__(self, key, config, root, *, dim=8, flaky_keys=()):
        self.key = key
        self.config = dict(config)
        self.root = Path(root).resolve()
        self.dim = int(dim)
        self.depth = 18 if key == "qwen" else 17
        self.flaky_keys = set(flaky_keys)
        self.call_counts = {}
        self.forward_calls = []
        self.torch = None

    def describe(self):
        depths = [9, self.depth, 27, 36] if self.key == "qwen" else [9, self.depth, 26, 34]
        sites = [f"decoder:{d}:{p}" for d in depths for p in ("last_caption", "caption_mean")]
        return {
            "decoder_blocks": 36 if self.key == "qwen" else 34,
            "decoder_depths": depths,
            "midpoint": self.depth,
            "model_fingerprint": self.config["config_fingerprint"],
            "repeat_atol": self.config["repeat_atol"],
            "repeat_rtol": self.config["repeat_rtol"],
            "supported_sites": sites,
        }

    def describe_execution(self):
        last_key = f"decoder:{self.depth}:last_caption"
        mean_key = f"decoder:{self.depth}:caption_mean"
        site_specs = {
            last_key: {"module_path": f"model.language_model.layers.{self.depth - 1}", "pooling": "last_caption"},
            mean_key: {"module_path": f"model.language_model.layers.{self.depth - 1}", "pooling": "caption_mean"},
        }
        return {
            "execution_fingerprint": self.config["_execution_fingerprint"],
            "backend": {"device_type": "cuda", "device_name": "NVIDIA L4", "batch_size": 1, "eval": True,
                        "truncation": False, "padding": False},
            "source_sha256": {"hf.py": "0" * 64, "spans.py": "1" * 64},
            "site_specs": site_specs,
        }

    def _forward(self, row, sites, condition="image_caption", attention=False, purpose="preliminary_features"):
        assert not attention
        site_keys = [s.key for s in sites]
        self.forward_calls.append((row.row_key, site_keys, condition, purpose))
        count = self.call_counts.get(row.row_key, 0) + 1
        self.call_counts[row.row_key] = count

        seed_bytes = hashlib.sha256(f"{row.image_id}|{row.language}|{row.caption_text}".encode("utf-8")).digest()
        base = int.from_bytes(seed_bytes[:4], "little") % 1000
        vec_last = [float(base + i) * 0.01 for i in range(self.dim)]
        vec_mean = [float(base + 100 + i) * 0.01 for i in range(self.dim)]
        if row.row_key in self.flaky_keys and count > 1:
            vec_last[0] += 1.0

        values = {}
        for key in site_keys:
            if key.endswith(":last_caption"):
                values[key] = list(vec_last)
            elif key.endswith(":caption_mean"):
                values[key] = list(vec_mean)
            else:
                raise AssertionError(f"Unexpected site {key}")

        ids = [10, 11, 12, 13, 14]
        evidence = {
            "row_key": row.row_key,
            "condition": condition,
            "token_count": 5,
            "native_serialization": f"<image>Caption:\n{row.caption_text}",
            "caption_alignment": "fixture_exact_offsets",
            "processor_attention_mask": [1, 1, 1, 1, 1],
            "input_ids_sha256": hashlib.sha256(str(ids).encode()).hexdigest(),
            "image_positions": [0, 1],
            "caption_positions": [2, 3],
            "role_and_other_positions": [4],
            "last_caption": 3,
        }
        return values, evidence


def _build_fixture(root: Path, *, verdict="passed_full", n_chunks=2, confirm_rows=0, include_confirm_row=False):
    protocol = json.loads((REPO / "configs/protocol.json").read_text(encoding="utf-8"))
    proto = json.loads(json.dumps(protocol))
    proto["split_sizes"] = {"FIT": 2, "DEV-S": 1, "DEV-C": 1, "SCREEN": 4, "CONFIRM": 1}
    protocol_fp = fingerprint(proto)
    producer = {
        "code_commit": "4" * 40,
        "storage_scope": "local-development",
        "software_versions": {"python": "3.12.0"},
    }
    init_phase2_dirs(root)

    # Create pinned SCREEN images + one CONFIRM image (which must never be read).
    screen_ids = [f"{1000 + i:016x}" for i in range(4)]
    confirm_id = f"{9000:016x}"
    splits = {iid: "SCREEN" for iid in screen_ids}
    splits[confirm_id] = "CONFIRM"

    records = []
    for idx, iid in enumerate(screen_ids + [confirm_id]):
        split = splits[iid]
        rel_img = f"images/{split.lower()}/{iid}.jpg"
        img_file = root / rel_img
        img_file.parent.mkdir(parents=True, exist_ok=True)
        img_file.write_bytes(f"FAKE_IMAGE_BYTES_{iid}".encode("utf-8"))
        img_sha = file_sha256(img_file)
        for lang in LANGUAGES:
            records.append({
                "image_id": iid,
                "image_path": rel_img,
                "image_sha256": img_sha,
                "language": lang,
                "caption_id": f"{iid}_{lang}_0",
                "caption_text": f"draw0 {lang} caption for {iid}",
                "split": split,
            })

    targets = [
        {"image_id": iid, "split": splits[iid], "values": [0] * 25}
        for iid in screen_ids + [confirm_id]
    ]
    dataset_fp = fingerprint({"screen": screen_ids, "confirm": confirm_id})
    ds_payload = {
        "dataset.json": _json_bytes({"dataset_fingerprint": dataset_fp, "protocol_fingerprint": protocol_fp}),
        "records.jsonl": _jsonl_bytes(records),
        "targets.jsonl": _jsonl_bytes(targets),
        "splits.json": _json_bytes(splits),
    }
    for lang in LANGUAGES:
        ds_payload[f"captions/{lang}.jsonl"] = _jsonl_bytes([])
    ds_meta = {
        "kind": "stage1_dataset",
        "status": "passed",
        "run_id": "ds-run",
        "code_commit": "4" * 40,
        "storage_scope": "local-development",
        "software_versions": {"python": "3.12.0"},
        "protocol_fingerprint": protocol_fp,
        "dataset_fingerprint": dataset_fp,
    }
    ds_dir = publish(root, "shared/ds", ds_meta, ds_payload)
    dataset_ref = {"path": (ds_dir / MARKER).relative_to(root).as_posix(), "sha256": file_sha256(ds_dir / MARKER)}

    # Build Stage 2 readiness + model_plan for qwen (depth 18).
    models_cfg = json.loads((REPO / "configs/models.json").read_text(encoding="utf-8"))["models"]["qwen"]
    model_cfg = dict(models_cfg)
    model_cfg["config_fingerprint"] = fingerprint({k: v for k, v in model_cfg.items() if k != "config_fingerprint"})
    exec_fp = fingerprint({"model": "qwen", "fp": model_cfg["config_fingerprint"]})
    model_cfg["_execution_fingerprint"] = exec_fp
    last_key = "decoder:18:last_caption"
    mean_key = "decoder:18:caption_mean"
    site_specs = {
        last_key: {"module_path": "model.language_model.layers.17", "pooling": "last_caption"},
        mean_key: {"module_path": "model.language_model.layers.17", "pooling": "caption_mean"},
    }
    exec_id = {
        "execution_fingerprint": exec_fp,
        "backend": {"device_type": "cuda", "device_name": "NVIDIA L4", "batch_size": 1, "eval": True,
                    "truncation": False, "padding": False},
        "source_sha256": {"hf.py": "0" * 64, "spans.py": "1" * 64},
        "site_specs": site_specs,
    }
    dimensions = {last_key: 8, mean_key: 8}
    readiness_body = {
        "schema": "stage2.readiness.v1",
        "model": "qwen",
        "model_config": model_cfg,
        "execution_identity": exec_id,
        "dimensions": dimensions,
    }
    r_meta = {
        "kind": "stage2_model_readiness",
        "status": "passed",
        "run_id": "r-qwen",
        "code_commit": "4" * 40,
        "storage_scope": "local-development",
        "software_versions": {"python": "3.12.0"},
        "protocol_fingerprint": protocol_fp,
        "dataset_fingerprint": dataset_fp,
        "model_fingerprint": model_cfg["config_fingerprint"],
    }
    r_dir = publish(root, "cache/shared/stage2-plans/r-qwen", r_meta, {"readiness.json": _json_bytes(readiness_body)})
    r_ref = {"path": (r_dir / MARKER).relative_to(root).as_posix(), "sha256": file_sha256(r_dir / MARKER)}

    model_plan_body = {
        "schema": "stage2.plan.v1",
        "kind": "stage2_model_plan",
        "model": "qwen",
        "dataset_fingerprint": dataset_fp,
        "protocol_fingerprint": protocol_fp,
        "model_config": model_cfg,
        "execution_identity": exec_id,
        "dimensions": dimensions,
        "adapter": {"midpoint": 18, "decoder_blocks": 36},
        "bridge_sha256": bridge_hash(),
        "references": {"readiness": r_ref},
    }
    mp_meta = dict(r_meta, kind="stage2_model_plan", run_id="mp-qwen")
    mp_dir = publish(root, "cache/shared/stage2-plans/mp-qwen", mp_meta, {"plan.json": _json_bytes(model_plan_body)})
    mp_ref = {"path": (mp_dir / MARKER).relative_to(root).as_posix(), "sha256": file_sha256(mp_dir / MARKER)}

    # Build Phase 2 plan.
    probe_ref = storage_probe(root, proto, dataset_fp, producer)
    e29_blocks = {
        m: {c: {p: {l: {"block_id": "b", "prediction_ref": {"path": "x/_COMPLETE.json", "sha256": "0" * 64}}
                    for l in LANGUAGES} for p in ("last_caption", "caption_mean")}
            for c in ("image_caption", "caption_only")}
        for m in ("qwen", "gemma")
    }
    lineage_id = fingerprint({"dataset": dataset_fp, "stage4": "1" * 64})
    stage4_ref = {"path": "results/stage4/releases/r/_COMPLETE.json", "sha256": "2" * 64}
    plan_core = {
        "schema": SCHEMA_PLAN,
        "phase2_version": PHASE2_VERSION,
        "lineage_identity": lineage_id,
        "dataset_fingerprint": dataset_fp,
        "protocol_fingerprint": protocol_fp,
        "stage4_release": stage4_ref,
        "stage4_plan": {"path": "results/stage4/plan/_COMPLETE.json", "sha256": "3" * 64},
        "stage3_registry": {"path": "results/stage3/reg/_COMPLETE.json", "sha256": "4" * 64},
        "g0_gate": {"path": "results/stage3/g0/_COMPLETE.json", "sha256": "5" * 64},
        "g0_p_value": 0.5,
        "stage2_suite": {"path": "cache/shared/suite/_COMPLETE.json", "sha256": "6" * 64},
        "stage1_dataset": dataset_ref,
        "bootstrap": {"shape": [2000, 360], "sha256": "7" * 64},
        "midpoints": {"qwen": 18, "gemma": 17},
        "storage_probe": plain(probe_ref),
        "initialized_dirs": list(PHASE2_PARENT_DIRS),
        "e29_blocks": e29_blocks,
        "e11_fit_statistics": {},
        "gate_thresholds": {"e30": {}, "e11": {}},
        "stage5_decision_ranges": {},
        "e07_audit": None,
        "e17_auth": None,
        "confirm_excluded": True,
    }
    plan_id = fingerprint(plan_core)
    plan_body = dict(plan_core, plan_identity=plan_id)
    plan_ref = publish_phase2(root, proto, dataset_fp, "phase2/plans", KIND_PLAN, plan_body, producer)

    # Build E07 audit extraction_plan (only changed views; draw0/byte-identical are excluded).
    chunks = []
    if verdict != "blocked":
        for c_idx in range(n_chunks):
            lang = LANGUAGES[c_idx % len(LANGUAGES)]
            rows = []
            for r_i in range(2):
                iid = confirm_id if (include_confirm_row and c_idx == 0 and r_i == 1) else screen_ids[r_i]
                sp = "CONFIRM" if (include_confirm_row and c_idx == 0 and r_i == 1) else "SCREEN"
                txt = f"changed {lang} caption {c_idx}_{r_i} for {iid}"
                rows.append({
                    "image_id": iid,
                    "language": lang,
                    "split": sp,
                    "caption_id": f"{iid}_{lang}_draw1",
                    "caption_text": txt,
                    "caption_text_sha256": hashlib.sha256(txt.encode("utf-8")).hexdigest(),
                    "draw_keys": [1],
                    "aliased_caption_ids": [f"{iid}_{lang}_draw1"],
                })
            chunks.append({
                "chunk_id": f"chunk_{c_idx:04d}",
                "language": lang,
                "split": "SCREEN",
                "n_rows": len(rows),
                "rows": rows,
            })
        extraction_plan = {
            "schema": "phase2.e07_extraction_plan.v1",
            "split": "SCREEN",
            "confirm_rows": confirm_rows,
            "condition": "image_caption",
            "depth_policy": "fixed_midpoint",
            "poolings": ["last_caption", "caption_mean"],
            "same_forward_both_poolings": True,
            "chunk_size": 32,
            "total_unique_views": sum(c["n_rows"] for c in chunks),
            "total_chunks": len(chunks),
            "per_language_counts": {l: sum(c["n_rows"] for c in chunks if c["language"] == l) for l in LANGUAGES},
            "chunks": chunks,
        }
    else:
        extraction_plan = None

    audit_body = {
        "schema": SCHEMA_E07_AUDIT,
        "stage4_release": stage4_ref,
        "lineage_identity": lineage_id,
        "dataset_ref": dataset_ref,
        "dataset_fingerprint": dataset_fp,
        "plan_identity": plan_id,
        "draw0_verified": True,
        "confirm_excluded": True,
        "verdict": verdict,
        "verdict_reasons": ["fixture"],
        "screen_multiplicity": {},
        "fit_dev_s_multiplicity": {},
        "draw_buckets": {},
        "support_tables": {},
        "unique_changed_views": {"total_unique_views": 0 if extraction_plan is None else extraction_plan["total_unique_views"]},
        "extraction_plan": extraction_plan,
    }
    audit_ref = publish_phase2(root, proto, dataset_fp, "phase2/audits/e07", KIND_E07_AUDIT, audit_body, producer)

    binding = Phase1Binding(
        root=root,
        protocol=frozen(proto),
        stage4_release_ref=frozen(stage4_ref),
        stage4_release=frozen({}),
        stage4_plan=None,
        bootstrap_shape=(2000, 360),
        bootstrap_sha256="7" * 64,
        midpoints=frozen({"qwen": 18, "gemma": 17}),
        dataset_ref=frozen(dataset_ref),
        data_plan_ref=frozen({"path": "cache/shared/dp/_COMPLETE.json", "sha256": "8" * 64}),
        model_plan_refs=frozen({"qwen": mp_ref, "gemma": mp_ref}),
        suite_ref=frozen({"path": "cache/shared/suite/_COMPLETE.json", "sha256": "6" * 64}),
        stage3_registry_ref=frozen({"path": "results/stage3/reg/_COMPLETE.json", "sha256": "4" * 64}),
        g0_gate_ref=frozen({"path": "results/stage3/g0/_COMPLETE.json", "sha256": "5" * 64}),
        g0_p_value=0.5,
        dataset_fingerprint=dataset_fp,
        protocol_fingerprint=protocol_fp,
        lineage_identity=lineage_id,
    )
    return proto, producer, plan_ref, audit_ref, binding, model_cfg


@unittest.skipIf(np is None, "numpy required for E07 extraction tests")
class ExtractE07Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _run_with_patches(self, proto, producer, plan_ref, audit_ref, binding, adapter, *, resume_refs=None):
        events = []

        def fake_load_plan(r, p, ref):
            from src.phase2.interfaces import load_phase2_artifact
            from src.phase2.plan import Phase2Plan
            ref_obj, body, _ = load_phase2_artifact(r, p, ref, KIND_PLAN, expected_schema=SCHEMA_PLAN)
            return Phase2Plan(body, ref_obj, r, p, binding=binding)

        with mock.patch("src.phase2.extract_e07.load_phase2_plan", side_effect=fake_load_plan), \
             mock.patch("src.phase2.extract_e07.bind_phase1", return_value=binding):
            return run_e07_extraction(
                self.root,
                proto,
                plan_ref,
                audit_ref,
                "qwen",
                producer,
                adapter_factory=lambda m, cfg, r: adapter,
                resume_refs=resume_refs,
                progress=events.append,
            ), events

    def test_blocked_audit_refusal(self):
        proto, producer, plan_ref, audit_ref, binding, model_cfg = _build_fixture(self.root, verdict="blocked")
        adapter = FakeHFAdapter("qwen", model_cfg, self.root)
        with self.assertRaisesRegex(ContractError, "verdict is 'blocked'"):
            self._run_with_patches(proto, producer, plan_ref, audit_ref, binding, adapter)
        self.assertEqual(adapter.forward_calls, [])

    def test_confirm_rejection(self):
        proto, producer, plan_ref, audit_ref, binding, model_cfg = _build_fixture(
            self.root, verdict="passed_full", confirm_rows=1
        )
        adapter = FakeHFAdapter("qwen", model_cfg, self.root)
        with self.assertRaisesRegex(ContractError, "confirm_rows == 0"):
            self._run_with_patches(proto, producer, plan_ref, audit_ref, binding, adapter)
        self.assertEqual(adapter.forward_calls, [])

        tmp2 = tempfile.TemporaryDirectory()
        self.addCleanup(tmp2.cleanup)
        root2 = Path(tmp2.name)
        proto2, prod2, p_ref2, a_ref2, bind2, m_cfg2 = _build_fixture(
            root2, verdict="passed_full", confirm_rows=0, include_confirm_row=True
        )
        adapter2 = FakeHFAdapter("qwen", m_cfg2, root2)
        with mock.patch("src.phase2.extract_e07.load_phase2_plan", side_effect=lambda r, p, ref: (
                __import__("src.phase2.plan", fromlist=["Phase2Plan"]).Phase2Plan(
                    __import__("src.phase2.interfaces", fromlist=["load_phase2_artifact"]).load_phase2_artifact(
                        r, p, ref, KIND_PLAN, expected_schema=SCHEMA_PLAN
                    )[1],
                    ref, r, p, binding=bind2,
                ))), \
             mock.patch("src.phase2.extract_e07.bind_phase1", return_value=bind2):
            with self.assertRaisesRegex(ContractError, "CONFIRM is closed"):
                run_e07_extraction(root2, proto2, p_ref2, a_ref2, "qwen", prod2,
                                   adapter_factory=lambda m, cfg, r: adapter2)
        self.assertEqual(adapter2.forward_calls, [])

    def test_dual_pooling_same_forward_and_byte_identical_exclusion(self):
        proto, producer, plan_ref, audit_ref, binding, model_cfg = _build_fixture(
            self.root, verdict="passed_restricted", n_chunks=2
        )
        adapter = FakeHFAdapter("qwen", model_cfg, self.root)
        res, events = self._run_with_patches(proto, producer, plan_ref, audit_ref, binding, adapter)

        self.assertEqual(res["status"], "passed")
        self.assertEqual(res["total_unique_views"], 4)
        self.assertEqual(res["total_chunks"], 2)
        # 2 views per chunk + 2 boundary repeats per chunk = 4 forwards/chunk * 2 chunks = 8 forwards.
        self.assertEqual(len(adapter.forward_calls), 8)
        for row_key, site_keys, cond, purpose in adapter.forward_calls:
            self.assertEqual(site_keys, ["decoder:18:last_caption", "decoder:18:caption_mean"])
            self.assertEqual(cond, "image_caption")
            self.assertEqual(purpose, "preliminary_features")
            self.assertNotIn("draw0", row_key[2])

        event_names = [e["event"] for e in events]
        self.assertIn("e07_extract_start", event_names)
        self.assertEqual(event_names.count("e07_chunk_published"), 2)
        self.assertIn("e07_release_published", event_names)

        rel_ref, rel_body, _ = verify_e07_release(self.root, proto, res["release_ref"], expected_model="qwen")
        self.assertEqual(rel_body["e07_verdict"], "passed_restricted")
        self.assertTrue(rel_body["repeat_check_summary"]["all_passed"])

    def test_repeat_check_failure_detection(self):
        proto, producer, plan_ref, audit_ref, binding, model_cfg = _build_fixture(
            self.root, verdict="passed_full", n_chunks=1
        )
        # Perturb repeat forward on first view in chunk 0.
        first_row_key = (f"{1000:016x}", "en", f"{1000:016x}_en_draw1")
        adapter = FakeHFAdapter("qwen", model_cfg, self.root, flaky_keys={first_row_key})
        with self.assertRaisesRegex(ContractError, "Repeat determinism check failed"):
            self._run_with_patches(proto, producer, plan_ref, audit_ref, binding, adapter)

    def test_chunk_resume_and_tamper_rejection(self):
        proto, producer, plan_ref, audit_ref, binding, model_cfg = _build_fixture(
            self.root, verdict="passed_full", n_chunks=2
        )
        adapter1 = FakeHFAdapter("qwen", model_cfg, self.root)
        res1, _ = self._run_with_patches(proto, producer, plan_ref, audit_ref, binding, adapter1)
        chunk0_ref = res1["chunks"][0]["ref"]

        # Resume chunk 0: only chunk 1 (4 forwards) should run.
        adapter2 = FakeHFAdapter("qwen", model_cfg, self.root)
        res2, events2 = self._run_with_patches(
            proto, producer, plan_ref, audit_ref, binding, adapter2, resume_refs={0: chunk0_ref}
        )
        self.assertEqual(len(adapter2.forward_calls), 4)
        self.assertTrue(any(e["event"] == "e07_chunk_resumed" and e["chunk_index"] == 0 for e in events2))
        verify_e07_release(self.root, proto, res2["release_ref"], expected_model="qwen")

        # Passing chunk 0 as chunk 1 must be rejected.
        adapter3 = FakeHFAdapter("qwen", model_cfg, self.root)
        with self.assertRaisesRegex(ContractError, "chunk_index mismatch"):
            self._run_with_patches(
                proto, producer, plan_ref, audit_ref, binding, adapter3, resume_refs={1: chunk0_ref}
            )

    def test_release_manifest_integrity(self):
        proto, producer, plan_ref, audit_ref, binding, model_cfg = _build_fixture(
            self.root, verdict="passed_full", n_chunks=1
        )
        adapter = FakeHFAdapter("qwen", model_cfg, self.root)
        res, _ = self._run_with_patches(proto, producer, plan_ref, audit_ref, binding, adapter)
        # Tamper a chunk's .npy file on disk -> verify_e07_release must reject.
        chunk_dir = (self.root / res["chunks"][0]["ref"]["path"]).parent
        (chunk_dir / "last_caption.npy").write_bytes(b"corrupted")
        with self.assertRaises(ContractError):
            verify_e07_release(self.root, proto, res["release_ref"], expected_model="qwen")


if __name__ == "__main__":
    unittest.main()
