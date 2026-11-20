"""Unit, contract, CLI, and fixture tests for Phase 3 Stage B2 Worker 2 (`extract_e23`)."""

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

try:
    import numpy as np
except ImportError:
    np = None

HAS_NUMPY = np is not None

from src.common.contracts import LANGUAGES, ContractError, ModelInput, canonical_bytes, fingerprint
from src.common.storage import MARKER, file_sha256, publish
from src.models.hf import HFAdapter
from src.phase3.extract_e23 import (
    _build_authentic_model_input,
    _build_swapped_model_input,
    preflight_e23_extraction,
    redact_text,
    run_e23_extraction,
    safe_error,
    stage2_current_producer,
    verify_e23_release,
)
from src.phase3.interfaces import (
    CERTIFIED_STAGE_B1_GPU_PLAN,
    KIND_E23_CHUNK,
    KIND_E23_RELEASE,
    KIND_GPU_PLAN,
    KIND_PROBE,
    KIND_PROBE_PAYLOAD,
    PHASE3_GPU_DECISION_RANGES,
    PHASE3_GPU_VERSION,
    PHASE3_PARENT_DIRS,
    REPO,
    SCHEMA_E23_FEATURES,
    SCHEMA_GPU_PLAN,
    SCHEMA_PROBE,
    frozen,
    init_phase3_dirs,
    plain,
    publish_phase3,
)
from src.stage2.evidence import bridge_hash
from src.stage2.operations import current_producer as real_stage2_current_producer


def _json_bytes(obj):
    return canonical_bytes(obj) + b"\n"


def _jsonl_bytes(rows):
    return b"".join(canonical_bytes(r) + b"\n" for r in rows)


class _StdlibFlag:
    def __init__(self, c_contiguous=True):
        self.c_contiguous = c_contiguous


class _StdlibArray:
    """Minimal stdlib-only ndarray stand-in supporting `<f4` NPY v1.0 serialization."""

    def __init__(self, data, dtype="<f4"):
        if isinstance(data, _StdlibArray):
            rows = [list(r) for r in data._data]
        else:
            rows = [list(r) if isinstance(r, (list, tuple)) else [float(r)] for r in data]
        self._data = [[float(x) for x in r] for r in rows] if rows else []
        if data and not isinstance(data[0], (list, tuple, _StdlibArray)):
            self._data = [float(x) for x in data]
            self.shape = (len(self._data),)
        elif self._data:
            self.shape = (len(self._data), len(self._data[0]))
        else:
            self.shape = (0,)
        self.dtype = "<f4" if str(dtype) in ("<f4", "float32") else str(dtype)
        self.flags = _StdlibFlag(True)

    def _flat(self):
        if len(self.shape) == 1:
            return self._data
        out = []
        for row in self._data:
            out.extend(row)
        return out

    def __sub__(self, other):
        if len(self.shape) == 1:
            return _StdlibArray([a - b for a, b in zip(self._data, other._data)], dtype=self.dtype)
        return _StdlibArray(
            [[a - b for a, b in zip(r1, r2)] for r1, r2 in zip(self._data, other._data)],
            dtype=self.dtype,
        )

    def __eq__(self, other):
        if isinstance(other, str):
            return self.dtype == other
        return False


class _StdlibBool:
    def __init__(self, value):
        self._value = bool(value)

    def all(self):
        return self._value

    def __bool__(self):
        return self._value


def _make_fake_numpy():
    """Create a pure-stdlib fake `numpy` module for exercising extraction on Python without numpy."""
    import math

    mod = types.ModuleType("numpy")
    mod.ndarray = _StdlibArray
    mod.dtype = lambda s: "<f4" if str(s) in ("<f4", "float32") else str(s)

    def asarray(val, dtype="<f4"):
        if isinstance(val, _StdlibArray):
            return val
        return _StdlibArray(val, dtype=dtype)

    def ascontiguousarray(val, dtype="<f4"):
        return asarray(val, dtype=dtype)

    def stack(items, axis=0):
        assert axis == 0
        return _StdlibArray([item._data for item in items], dtype="<f4")

    def abs_fn(arr):
        if len(arr.shape) == 1:
            return _StdlibArray([abs(x) for x in arr._data], dtype=arr.dtype)
        return _StdlibArray([[abs(x) for x in r] for r in arr._data], dtype=arr.dtype)

    def max_fn(arr):
        flat = arr._flat()
        return max(flat) if flat else 0.0

    def isfinite(val):
        if isinstance(val, _StdlibArray):
            return _StdlibBool(all(math.isfinite(x) for x in val._flat()))
        return math.isfinite(float(val))

    def allclose(a, b, rtol=0.0, atol=0.0):
        fa = a._flat()
        fb = b._flat()
        return len(fa) == len(fb) and all(abs(x - y) <= atol + rtol * abs(y) for x, y in zip(fa, fb))

    def save(buf, arr, allow_pickle=False):
        assert not allow_pickle
        header = repr({
            "descr": "<f4",
            "fortran_order": False,
            "shape": arr.shape,
        }).encode("latin1")
        preamble_len = 10
        total = preamble_len + len(header) + 1
        pad = (16 - (total % 16)) % 16
        header_bytes = header + (b" " * pad) + b"\n"
        buf.write(b"\x93NUMPY\x01\x00")
        buf.write(struct.pack("<H", len(header_bytes)))
        buf.write(header_bytes)
        flat = arr._flat()
        if flat:
            buf.write(struct.pack(f"<{len(flat)}f", *flat))

    def load(path, allow_pickle=False):
        import ast

        assert not allow_pickle
        raw = Path(path).read_bytes()
        if not raw.startswith(b"\x93NUMPY"):
            raise ValueError("Not a valid NPY file")
        major, _minor = raw[6], raw[7]
        if major == 1:
            hlen = struct.unpack("<H", raw[8:10])[0]
            offset = 10 + hlen
            hdr_text = raw[10:offset].decode("latin1").strip()
        else:
            hlen = struct.unpack("<I", raw[8:12])[0]
            offset = 12 + hlen
            hdr_text = raw[12:offset].decode("latin1").strip()
        meta = ast.literal_eval(hdr_text)
        shape = meta["shape"]
        count = 1
        for dim in shape:
            count *= dim
        vals = list(struct.unpack(f"<{count}f", raw[offset : offset + 4 * count]))
        if len(shape) == 2:
            rows = [vals[i * shape[1] : (i + 1) * shape[1]] for i in range(shape[0])]
            return _StdlibArray(rows, dtype=meta["descr"])
        return _StdlibArray(vals, dtype=meta["descr"])

    mod.asarray = asarray
    mod.ascontiguousarray = ascontiguousarray
    mod.stack = stack
    mod.abs = abs_fn
    mod.max = max_fn
    mod.isfinite = isfinite
    mod.allclose = allclose
    mod.save = save
    mod.load = load
    return mod


class FakeHFAdapter(HFAdapter):
    """Explicit HFAdapter fixture subclass for CPU-only E23 extraction tests."""

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
            "backend": {
                "device_type": "cuda",
                "device_name": "NVIDIA L4",
                "batch_size": 1,
                "eval": True,
                "truncation": False,
                "padding": False,
            },
            "source_sha256": {"hf.py": "0" * 64, "spans.py": "1" * 64},
            "site_specs": site_specs,
        }

    def _forward(self, row, sites, condition="image_caption", attention=False, purpose="preliminary_features"):
        assert not attention
        site_keys = [s.key for s in sites]
        self.forward_calls.append({
            "image_id": row.image_id,
            "image_path": row.image_path,
            "image_sha256": row.image_sha256,
            "language": row.language,
            "caption_id": row.caption_id,
            "caption_text": row.caption_text,
            "split": row.split,
            "row_key": row.row_key,
            "site_keys": site_keys,
            "condition": condition,
            "purpose": purpose,
        })
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


def _publish_probe_without_subprocess(root: Path, protocol: dict, dataset_fp: str, producer: dict):
    payload_ref = publish_phase3(
        root,
        protocol,
        dataset_fp,
        "phase3/probes",
        KIND_PROBE_PAYLOAD,
        {"schema": SCHEMA_PROBE, "nonce": "probe-nonce", "remote_sync_guarantee": False},
        producer,
    )
    return publish_phase3(
        root,
        protocol,
        dataset_fp,
        "phase3/probes",
        KIND_PROBE,
        {
            "schema": SCHEMA_PROBE,
            "payload": plain(payload_ref),
            "independent_process_readback": True,
            "reader_pid": os.getpid() + 1,
            "writer_pid": os.getpid(),
            "remote_sync_guarantee": False,
        },
        producer,
        upstream=[plain(payload_ref)],
    )


def _build_e23_fixture(
    root: Path,
    *,
    model="qwen",
    n_chunks=2,
    confirm_excluded=True,
    e35_excluded=True,
    zero_model_scores_in_pairing=True,
    confirm_rows=0,
    include_confirm_in_chunk=False,
):
    protocol = json.loads((REPO / "configs/protocol.json").read_text(encoding="utf-8"))
    proto = json.loads(json.dumps(protocol))
    proto["split_sizes"] = {"FIT": 2, "DEV-S": 1, "DEV-C": 1, "SCREEN": 4, "CONFIRM": 1}
    protocol_fp = fingerprint(proto)
    producer = {
        "code_commit": "4" * 40,
        "storage_scope": "local-development",
        "software_versions": {"python": "3.12.0"},
    }
    init_phase3_dirs(root)

    screen_ids = [f"{1000 + i:016x}" for i in range(4)]
    confirm_id = f"{9000:016x}"
    img_meta = {}
    for iid, split in [(s, "SCREEN") for s in screen_ids] + [(confirm_id, "CONFIRM")]:
        rel_img = f"images/{split.lower()}/{iid}.jpg"
        img_file = root / rel_img
        img_file.parent.mkdir(parents=True, exist_ok=True)
        img_file.write_bytes(f"FAKE_IMAGE_BYTES_{iid}".encode("utf-8"))
        img_meta[iid] = (rel_img, file_sha256(img_file), split)

    dataset_fp = fingerprint({"screen": screen_ids, "confirm": confirm_id})
    probe_ref = _publish_probe_without_subprocess(root, proto, dataset_fp, producer)

    depth = 18 if model == "qwen" else 17
    models_cfg = json.loads((REPO / "configs/models.json").read_text(encoding="utf-8"))["models"][model]
    model_cfg = dict(models_cfg)
    model_cfg["config_fingerprint"] = fingerprint({k: v for k, v in model_cfg.items() if k != "config_fingerprint"})
    exec_fp = fingerprint({"model": model, "fp": model_cfg["config_fingerprint"]})
    model_cfg["_execution_fingerprint"] = exec_fp
    last_key = f"decoder:{depth}:last_caption"
    mean_key = f"decoder:{depth}:caption_mean"
    site_specs = {
        last_key: {"module_path": f"model.language_model.layers.{depth - 1}", "pooling": "last_caption"},
        mean_key: {"module_path": f"model.language_model.layers.{depth - 1}", "pooling": "caption_mean"},
    }
    exec_id = {
        "execution_fingerprint": exec_fp,
        "backend": {
            "device_type": "cuda",
            "device_name": "NVIDIA L4",
            "batch_size": 1,
            "eval": True,
            "truncation": False,
            "padding": False,
        },
        "source_sha256": {"hf.py": "0" * 64, "spans.py": "1" * 64},
        "site_specs": site_specs,
    }
    dimensions = {last_key: 8, mean_key: 8}
    readiness_body = {
        "schema": "stage2.readiness.v1",
        "model": model,
        "model_config": model_cfg,
        "execution_identity": exec_id,
        "dimensions": dimensions,
    }
    r_meta = {
        "kind": "stage2_model_readiness",
        "status": "passed",
        "run_id": f"r-{model}",
        "code_commit": "4" * 40,
        "storage_scope": "local-development",
        "software_versions": {"python": "3.12.0"},
        "protocol_fingerprint": protocol_fp,
        "dataset_fingerprint": dataset_fp,
        "model_fingerprint": model_cfg["config_fingerprint"],
        "seeds": {},
        "upstream_artifacts": [],
    }
    r_dir = publish(root, f"cache/shared/stage2-plans/r-{model}", r_meta, {"readiness.json": _json_bytes(readiness_body)})
    r_ref = {"path": (r_dir / MARKER).relative_to(root).as_posix(), "sha256": file_sha256(r_dir / MARKER)}

    model_plan_body = {
        "schema": "stage2.plan.v1",
        "kind": "stage2_model_plan",
        "model": model,
        "dataset_fingerprint": dataset_fp,
        "protocol_fingerprint": protocol_fp,
        "model_config": model_cfg,
        "execution_identity": exec_id,
        "dimensions": dimensions,
        "adapter": {"midpoint": depth, "decoder_blocks": 36 if model == "qwen" else 34},
        "bridge_sha256": bridge_hash(),
        "references": {"readiness": r_ref},
    }
    mp_meta = dict(r_meta, kind="stage2_model_plan", run_id=f"mp-{model}")
    mp_dir = publish(root, f"cache/shared/stage2-plans/mp-{model}", mp_meta, {"plan.json": _json_bytes(model_plan_body)})
    mp_ref = {"path": (mp_dir / MARKER).relative_to(root).as_posix(), "sha256": file_sha256(mp_dir / MARKER)}

    chunks = []
    pair_candidates = [
        (screen_ids[0], screen_ids[1]),
        (screen_ids[1], screen_ids[0]),
        (screen_ids[2], screen_ids[3]),
        (screen_ids[3], screen_ids[2]),
    ]
    for c_idx in range(n_chunks):
        shown_id, owner_id = pair_candidates[c_idx % len(pair_candidates)]
        if include_confirm_in_chunk and c_idx == 0:
            shown_id = confirm_id
        shown_path, shown_sha, shown_split = img_meta[shown_id]
        rows = []
        for lang in LANGUAGES[:2]:
            caption_txt = f"owner {owner_id} {lang} caption"
            row_entry = {
                "shown_image_id": shown_id,
                "shown_image_path": shown_path,
                "shown_image_sha256": shown_sha,
                "caption_owner_image_id": owner_id,
                "language": lang,
                "caption_id": f"{owner_id}_{lang}_0",
                "caption_draw_key": 0,
                "caption_text": caption_txt,
                "caption_text_sha256": hashlib.sha256(caption_txt.encode("utf-8")).hexdigest(),
            }
            if include_confirm_in_chunk and c_idx == 0:
                row_entry["split"] = shown_split
            rows.append(row_entry)
        chunks.append({
            "chunk_id": f"E23_SWAPPED_{c_idx:04d}",
            "split": "SCREEN",
            "n_pairs": 1,
            "n_views": len(rows),
            "rows": rows,
        })

    authentic_parity_sample = []
    for iid in (screen_ids[0], screen_ids[-1]):
        img_path, img_sha, _ = img_meta[iid]
        for lang in LANGUAGES:
            caption_txt = f"authentic {iid} {lang} caption"
            authentic_parity_sample.append({
                "image_id": iid,
                "image_path": img_path,
                "image_sha256": img_sha,
                "language": lang,
                "caption_id": f"{iid}_{lang}_0",
                "caption_draw_key": 0,
                "caption_text": caption_txt,
                "caption_text_sha256": hashlib.sha256(caption_txt.encode("utf-8")).hexdigest(),
            })

    e23_plan = {
        "schema": "phase3.e23_pair_plan.v1",
        "split": "SCREEN",
        "confirm_rows": confirm_rows,
        "seed_recipe": "G9Q6-E23-pair-v1",
        "zero_model_scores_in_pairing": zero_model_scores_in_pairing,
        "chunk_size_pairs": 32,
        "total_unique_directed_pairs": sum(c["n_pairs"] for c in chunks),
        "total_swapped_views": sum(c["n_views"] for c in chunks),
        "total_chunks": len(chunks),
        "per_category": {},
        "chunks": chunks,
        "authentic_parity_sample": authentic_parity_sample,
    }

    lineage_id = fingerprint({"dataset": dataset_fp, "stage4": "1" * 64})
    stage4_ref = {"path": "results/stage4/releases/r/_COMPLETE.json", "sha256": "2" * 64}
    core = plain({
        "schema": SCHEMA_GPU_PLAN,
        "phase3_gpu_version": PHASE3_GPU_VERSION,
        "lineage_identity": lineage_id,
        "dataset_fingerprint": dataset_fp,
        "protocol_fingerprint": protocol_fp,
        "phase1_stage4_release": stage4_ref,
        "phase3_cpu_plan": None,
        "bootstrap": {"shape": [2000, 360], "sha256": "7" * 64},
        "storage_probe": plain(probe_ref),
        "initialized_dirs": list(PHASE3_PARENT_DIRS),
        "models": {
            model: {
                "depths": [9, depth, 27, 36] if model == "qwen" else [9, depth, 26, 34],
                "midpoint_depth": depth,
                "model_plan_ref": mp_ref,
                "midpoint_groups": {},
                "source_systems": {},
            }
        },
        "e39_e31_plan": {"chunks": []},
        "e23_plan": e23_plan,
        "decision_ranges": plain(PHASE3_GPU_DECISION_RANGES),
        "e35_excluded": e35_excluded,
        "confirm_excluded": confirm_excluded,
    })
    plan_id = fingerprint(core)
    body = dict(core, plan_identity=plan_id)
    plan_ref = publish_phase3(
        root,
        proto,
        dataset_fp,
        "phase3/plans",
        KIND_GPU_PLAN,
        body,
        producer,
        upstream=[stage4_ref, plain(probe_ref)],
    )
    return proto, producer, plan_ref, model_cfg, screen_ids, confirm_id, img_meta


def _patch_job_plan_loader():
    def _loader(plan_manifest, root, protocol):
        from src.stage2.schema import artifact as s2_artifact

        path, meta = s2_artifact(plan_manifest, root, protocol)
        body = json.loads((path.parent / "plan.json").read_text(encoding="utf-8"))
        return frozen(body)

    return mock.patch("src.phase3.extract_e23.load_job_plan", side_effect=_loader)


class E23StdlibAndContractTests(unittest.TestCase):
    def test_lazy_imports_and_stage2_producer(self):
        if not HAS_NUMPY:
            self.assertNotIn("numpy", sys.modules)
        self.assertIs(stage2_current_producer, real_stage2_current_producer)

    def test_token_redaction(self):
        with mock.patch.dict(os.environ, {"HF_TOKEN": "custom_secret_token_value"}):
            msg = redact_text("Auth failed using hf_AbCdEf1234567890 and custom_secret_token_value")
            self.assertNotIn("hf_AbCdEf1234567890", msg)
            self.assertNotIn("custom_secret_token_value", msg)
            self.assertEqual(msg.count("[REDACTED]"), 2)
            exc_msg = safe_error(RuntimeError("token hf_xyz987 leaked"))
            self.assertEqual(exc_msg, "token [REDACTED] leaked")

    def test_notebook_contract(self):
        nb_path = REPO / "scripts/phase3/extract_e23.ipynb"
        self.assertTrue(nb_path.is_file(), "Missing scripts/phase3/extract_e23.ipynb")
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertEqual(nb.get("nbformat"), 4)
        self.assertEqual(len(nb.get("cells", [])), 4)
        setup_src = "".join(nb["cells"][1]["source"])
        cfg_src = "".join(nb["cells"][2]["source"])
        exec_src = "".join(nb["cells"][3]["source"])

        self.assertIn("CODE_REF = 'phase3'", setup_src)
        self.assertIn("requirements-l4.txt", setup_src)
        self.assertIn("'L4'", setup_src)
        self.assertIn("MODEL = 'qwen'", cfg_src)
        self.assertIn(CERTIFIED_STAGE_B1_GPU_PLAN["path"], cfg_src)
        self.assertIn(CERTIFIED_STAGE_B1_GPU_PLAN["sha256"], cfg_src)
        self.assertIn("RESUME_REFS = None", cfg_src)
        self.assertIn("HF_TOKEN", exec_src)
        self.assertIn("MODEL == 'gemma'", exec_src)
        self.assertIn("src.phase3.extract_e23", exec_src)
        self.assertIn("redact", exec_src)

    def test_cli_help_and_blocked_exit_stdlib(self):
        help_res = subprocess.run(
            [sys.executable, "-m", "src.phase3.extract_e23", "--help"],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        self.assertEqual(help_res.returncode, 0, help_res.stderr)
        self.assertIn("--plan", help_res.stdout)
        self.assertIn("--resume-refs", help_res.stdout)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            blocked_res = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.phase3.extract_e23",
                    "--model",
                    "qwen",
                    "--phase1-root",
                    str(root),
                    "--plan",
                    str(root / "hf_SecretToken123" / "_COMPLETE.json"),
                    "--preflight-only",
                ],
                cwd=REPO,
                capture_output=True,
                text=True,
            )
            self.assertEqual(blocked_res.returncode, 2)
            payload = json.loads(blocked_res.stdout.strip().splitlines()[-1])
            self.assertEqual(payload["status"], "blocked")
            self.assertNotIn("hf_SecretToken123", blocked_res.stdout)
            self.assertIn("[REDACTED]", blocked_res.stdout)

    def test_swapped_and_authentic_model_input_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shown_rel = "images/screen/shown.jpg"
            owner_rel = "images/screen/owner.jpg"
            (root / shown_rel).parent.mkdir(parents=True, exist_ok=True)
            (root / shown_rel).write_bytes(b"SHOWN_BYTES")
            (root / owner_rel).write_bytes(b"OWNER_BYTES")
            shown_sha = file_sha256(root / shown_rel)
            owner_sha = file_sha256(root / owner_rel)
            cap = "एक परीक्षा वाक्य"
            cap_sha = hashlib.sha256(cap.encode("utf-8")).hexdigest()

            swapped_row = {
                "shown_image_id": "0000000000000001",
                "shown_image_path": shown_rel,
                "shown_image_sha256": shown_sha,
                "caption_owner_image_id": "0000000000000002",
                "language": "hi",
                "caption_id": "0000000000000002_hi_0",
                "caption_draw_key": 0,
                "caption_text": cap,
                "caption_text_sha256": cap_sha,
            }
            mi = _build_swapped_model_input(root, swapped_row)
            self.assertEqual(mi.image_id, "0000000000000001")
            self.assertEqual(mi.image_path, shown_rel)
            self.assertEqual(mi.image_sha256, shown_sha)
            self.assertEqual(mi.caption_id, "0000000000000002_hi_0")
            self.assertEqual(mi.caption_text, cap)
            self.assertEqual(mi.language, "hi")
            self.assertEqual(mi.split, "SCREEN")

            # Equal shown and owner IDs must fail swapped guard
            with self.assertRaisesRegex(ContractError, "shown_image_id != caption_owner_image_id"):
                _build_swapped_model_input(root, dict(swapped_row, caption_owner_image_id="0000000000000001"))

            # CONFIRM split row must fail before image IO
            with self.assertRaisesRegex(ContractError, "CONFIRM is closed"):
                _build_swapped_model_input(root, dict(swapped_row, split="CONFIRM"))

            # Tampered image SHA must fail
            with self.assertRaisesRegex(ContractError, "SHA-256 mismatch"):
                _build_swapped_model_input(root, dict(swapped_row, shown_image_sha256="f" * 64))

            # Tampered caption text SHA must fail
            with self.assertRaisesRegex(ContractError, "Caption UTF-8 SHA-256 mismatch"):
                _build_swapped_model_input(root, dict(swapped_row, caption_text_sha256="f" * 64))

            auth_row = {
                "image_id": "0000000000000002",
                "image_path": owner_rel,
                "image_sha256": owner_sha,
                "language": "hi",
                "caption_id": "0000000000000002_hi_0",
                "caption_draw_key": 0,
                "caption_text": cap,
                "caption_text_sha256": cap_sha,
            }
            auth_mi = _build_authentic_model_input(root, auth_row)
            self.assertEqual(auth_mi.image_id, "0000000000000002")
            self.assertEqual(auth_mi.split, "SCREEN")
            with self.assertRaisesRegex(ContractError, "image_id == caption_owner_image_id"):
                _build_authentic_model_input(root, dict(auth_row, caption_owner_image_id="0000000000000001"))

    def test_preflight_plan_guards_and_bind_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proto, _, plan_ref, _, _, _, _ = _build_e23_fixture(root)
            with _patch_job_plan_loader(), mock.patch("src.phase3.gpu_plan.bind_phase1") as mock_bind:
                ctx = preflight_e23_extraction(root, proto, "qwen", plan_ref=plan_ref)
                mock_bind.assert_not_called()
                self.assertEqual(ctx["midpoint_depth"], 18)

        # Reject confirm_excluded=False, e35_excluded=False, zero_model_scores_in_pairing=False, confirm_rows=1, CONFIRM split row
        guard_cases = [
            ({"confirm_excluded": False}, "confirm_excluded=True"),
            ({"e35_excluded": False}, "e35_excluded=True"),
            ({"zero_model_scores_in_pairing": False}, "zero_model_scores_in_pairing=True"),
            ({"confirm_rows": 1}, "confirm_rows == 0"),
            ({"include_confirm_in_chunk": True}, "CONFIRM is closed"),
        ]
        for kwargs, pattern in guard_cases:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                proto, _, plan_ref, _, _, _, _ = _build_e23_fixture(root, **kwargs)
                with _patch_job_plan_loader():
                    with self.assertRaisesRegex(ContractError, pattern):
                        preflight_e23_extraction(root, proto, "qwen", plan_ref=plan_ref)

    def test_end_to_end_extraction_repeat_resume_and_tamper_with_stdlib_shim(self):
        fake_np = _make_fake_numpy() if not HAS_NUMPY else np
        with mock.patch.dict(sys.modules, {"numpy": fake_np}):
            self._run_full_extraction_suite()

    def _run_full_extraction_suite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proto, producer, plan_ref, model_cfg, screen_ids, _, _ = _build_e23_fixture(root, n_chunks=2)
            adapter = FakeHFAdapter("qwen", model_cfg, root)
            events = []

            with _patch_job_plan_loader():
                res = run_e23_extraction(
                    root,
                    proto,
                    "qwen",
                    producer,
                    plan_ref=plan_ref,
                    adapter_factory=lambda m, cfg, r: adapter,
                    progress=events.append,
                )

            self.assertEqual(res["status"], "passed")
            self.assertEqual(res["midpoint_depth"], 18)
            self.assertEqual(res["total_chunks"], 2)
            self.assertEqual(res["total_swapped_views"], 4)
            self.assertEqual(res["authentic_parity_count"], 8)
            self.assertTrue(res["e35_excluded"])
            self.assertTrue(res["confirm_excluded"])

            # 2 chunks * (2 views + 2 boundary repeats) = 8 + (8 parity + 2 boundary repeats) = 18 forwards
            self.assertEqual(len(adapter.forward_calls), 18)
            for call in adapter.forward_calls:
                self.assertEqual(call["site_keys"], ["decoder:18:last_caption", "decoder:18:caption_mean"])
                self.assertEqual(call["condition"], "image_caption")
                self.assertEqual(call["purpose"], "preliminary_features")
                self.assertEqual(call["split"], "SCREEN")

            # Swapped forwards must pass shown_image_id with caption_owner's caption
            first_call = adapter.forward_calls[0]
            self.assertEqual(first_call["image_id"], screen_ids[0])
            self.assertIn(screen_ids[1], first_call["caption_id"])

            event_names = [e["event"] for e in events]
            self.assertIn("e23_extract_start", event_names)
            self.assertEqual(event_names.count("e23_chunk_published"), 2)
            self.assertIn("e23_release_published", event_names)

            with _patch_job_plan_loader():
                rel_ref, rel_body, _ = verify_e23_release(
                    root, proto, res["release_ref"], expected_model="qwen"
                )
            self.assertEqual(rel_ref["path"], res["release_ref"]["path"])
            self.assertTrue(rel_body["repeat_check_summary"]["all_passed"])
            self.assertEqual(rel_body["authentic_parity_count"], 8)

            # Test --resume-refs skipping chunk 0
            chunk0_ref = res["chunks"][0]["ref"]
            adapter_res = FakeHFAdapter("qwen", model_cfg, root)
            events_res = []
            with _patch_job_plan_loader():
                res2 = run_e23_extraction(
                    root,
                    proto,
                    "qwen",
                    producer,
                    plan_ref=plan_ref,
                    adapter_factory=lambda m, cfg, r: adapter_res,
                    resume_refs={0: chunk0_ref},
                    progress=events_res.append,
                )
            # Chunk 0 skipped -> 4 (chunk 1) + 10 (authentic parity) = 14 forwards
            self.assertEqual(len(adapter_res.forward_calls), 14)
            self.assertTrue(any(e["event"] == "e23_chunk_resumed" and e["chunk_index"] == 0 for e in events_res))
            verify_e23_release(root, proto, res2["release_ref"], expected_model="qwen")

            # Passing chunk 0 as chunk 1 must be rejected
            adapter_bad_res = FakeHFAdapter("qwen", model_cfg, root)
            with _patch_job_plan_loader():
                with self.assertRaisesRegex(ContractError, "chunk_index mismatch"):
                    run_e23_extraction(
                        root,
                        proto,
                        "qwen",
                        producer,
                        plan_ref=plan_ref,
                        adapter_factory=lambda m, cfg, r: adapter_bad_res,
                        resume_refs={1: chunk0_ref},
                    )

            # Repeat failure detection
            flaky_key = (screen_ids[0], "en", f"{screen_ids[1]}_en_0")
            adapter_flaky = FakeHFAdapter("qwen", model_cfg, root, flaky_keys={flaky_key})
            with _patch_job_plan_loader():
                with self.assertRaisesRegex(ContractError, "Repeat determinism check failed"):
                    run_e23_extraction(
                        root,
                        proto,
                        "qwen",
                        producer,
                        plan_ref=plan_ref,
                        adapter_factory=lambda m, cfg, r: adapter_flaky,
                    )

            # Tamper detection on release array
            rel_dir = (root / res["release_ref"]["path"]).parent
            (rel_dir / "authentic_parity_last_caption.npy").write_bytes(b"corrupted")
            with self.assertRaises(ContractError):
                verify_e23_release(root, proto, res["release_ref"], expected_model="qwen")


@unittest.skipUnless(HAS_NUMPY, "numpy required for real numpy E23 tests")
class E23NumpyExtractionTests(unittest.TestCase):
    def test_real_numpy_extraction_resume_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proto, producer, plan_ref, model_cfg, screen_ids, _, _ = _build_e23_fixture(
                root, model="gemma", n_chunks=2
            )
            adapter = FakeHFAdapter("gemma", model_cfg, root)
            with _patch_job_plan_loader():
                res = run_e23_extraction(
                    root,
                    proto,
                    "gemma",
                    producer,
                    plan_ref=plan_ref,
                    adapter_factory=lambda m, cfg, r: adapter,
                    run_id="fixed_gemma_run",
                )
            self.assertEqual(res["midpoint_depth"], 17)
            verify_e23_release(root, proto, res["release_ref"], expected_model="gemma")

            chunk_dir = (root / res["chunks"][0]["ref"]["path"]).parent
            arr = np.load(chunk_dir / "last_caption.npy", allow_pickle=False)
            self.assertEqual(arr.shape, (2, 8))
            self.assertEqual(arr.dtype, np.dtype("<f4"))

            # Tamper chunk array on disk
            (chunk_dir / "caption_mean.npy").write_bytes(b"corrupted")
            with self.assertRaises(ContractError):
                verify_e23_release(root, proto, res["release_ref"], expected_model="gemma")


if __name__ == "__main__":
    unittest.main()
