"""Unit, contract, and synthetic-adapter tests for Phase 3 Stage B2 Worker 1 (extract_e39_e31)."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_SCIPY = importlib.util.find_spec("scipy") is not None

from src.common.contracts import (
    ContractError,
    FeatureSite,
    LANGUAGES,
    ModelInput,
    canonical_bytes,
    fingerprint,
)
from src.common.storage import MARKER, file_sha256, publish
from src.models.hf import HFAdapter
from src.phase3.gpu_plan import (
    CERTIFIED_STAGE_B1_GPU_PLAN,
    E31_SITES,
    E39_SITES,
    select_e39_token_indices,
)
from src.phase3.schema import (
    CONDITIONS,
    KIND_E39_E31_CHUNK,
    KIND_E39_E31_RELEASE,
    KIND_GPU_PLAN,
    PHASE3_GPU_VERSION,
    PHASE3_PARENT_DIRS,
    POOLINGS,
    REPO,
    SCHEMA_E39_E31_FEATURES,
    SCHEMA_GPU_PLAN,
    init_phase3_dirs,
    load_phase3_artifact,
    plain,
    publish_phase3,
    storage_probe,
)
import src.phase3.extract_e39_e31 as extract_e39_e31
from src.phase3.extract_e39_e31 import (
    expected_array_specs,
    preflight_e39_e31_extraction,
    recover_caption_token_texts,
    redact_text,
    run_e39_e31_extraction,
    safe_error,
    verify_e39_e31_release,
)
from src.stage2.evidence import bridge_hash
from src.stage2.operations import current_producer as stage2_current_producer

np = None


def _json_bytes(obj):
    return canonical_bytes(obj) + b"\n"


def _jsonl_bytes(rows):
    return b"".join(canonical_bytes(r) + b"\n" for r in rows)


class _WhitespaceOffsetTokenizer:
    """Minimal fast-tokenizer mock producing exact character offsets and special mask."""

    def __init__(self, image_token_id=151655, special_tokens=None):
        self.image_token_id = int(image_token_id)
        self.special_tokens = dict(special_tokens or {"<|im_start|>": 1, "<|im_end|>": 2, "<image>": image_token_id})
        self.all_special_ids = sorted(self.special_tokens.values())

    def _encode_with_offsets(self, text: str):
        ids = []
        offsets = []
        special_mask = []
        pos = 0
        n = len(text)
        while pos < n:
            if text[pos].isspace():
                pos += 1
                continue
            matched = None
            for tok_str, tok_id in self.special_tokens.items():
                if text.startswith(tok_str, pos):
                    matched = (tok_str, tok_id)
                    break
            if matched is not None:
                tok_str, tok_id = matched
                ids.append(tok_id)
                offsets.append((pos, pos + len(tok_str)))
                special_mask.append(1)
                pos += len(tok_str)
                continue
            end = pos
            while end < n and not text[end].isspace():
                if any(text.startswith(sp, end) for sp in self.special_tokens):
                    break
                end += 1
            word = text[pos:end]
            tok_id = 1000 + (int.from_bytes(hashlib.sha256(word.encode("utf-8")).digest()[:2], "little") % 50000)
            ids.append(tok_id)
            offsets.append((pos, end))
            special_mask.append(0)
            pos = end
        return {
            "input_ids": ids,
            "offset_mapping": offsets,
            "special_tokens_mask": special_mask,
        }

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True, return_special_tokens_mask=True):
        return self._encode_with_offsets(text)


if HAS_NUMPY:

    class _FakeTensor:
        """Pure-NumPy stand-in supporting the PyTorch slice/mean/cast operations in extract_e39_e31."""

        def __init__(self, arr):
            self._arr = np.asarray(arr, dtype=np.float32)

        @property
        def ndim(self):
            return self._arr.ndim

        @property
        def shape(self):
            return self._arr.shape

        def detach(self):
            return self

        def float(self):
            return _FakeTensor(self._arr.astype(np.float32, copy=False))

        def cpu(self):
            return self

        def numpy(self):
            return np.array(self._arr, dtype=np.float32, copy=True)

        def mean(self, axis=0):
            return _FakeTensor(self._arr.mean(axis=axis))

        def __getitem__(self, item):
            return _FakeTensor(self._arr[item])

    class _FakeInferenceMode:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeTorch:
        def inference_mode(self):
            return _FakeInferenceMode()

    class _FakeHookHandle:
        def __init__(self, hook_list, fn):
            self._hook_list = hook_list
            self._fn = fn
            self.removed = False

        def remove(self):
            if not self.removed:
                self.removed = True
                if self._fn in self._hook_list:
                    self._hook_list.remove(self._fn)

    class _FakeBlock:
        def __init__(self, block_idx):
            self.block_idx = block_idx
            self.hooks = []

        def register_forward_hook(self, fn):
            self.hooks.append(fn)
            return _FakeHookHandle(self.hooks, fn)

    class _FakeModel:
        def __init__(
            self,
            blocks,
            *,
            dim=8,
            flaky_hidden_keys=(),
            flaky_att_keys=(),
            raise_on_call=False,
        ):
            self.blocks = blocks
            self.dim = dim
            self.training = False
            self._param = SimpleNamespace(requires_grad=False)
            self.flaky_hidden_keys = set(flaky_hidden_keys)
            self.flaky_att_keys = set(flaky_att_keys)
            self.raise_on_call = raise_on_call
            self.call_counts = {}
            self.forward_calls = []

        def parameters(self):
            return [self._param]

        def __call__(self, *, meta=None, use_cache=None, output_attentions=False, return_dict=False, **kwargs):
            assert use_cache is False
            assert return_dict is True
            row_key = meta["row_key"]
            condition = meta["condition"]
            token_count = meta["token_count"]
            caption_pos = meta["caption_positions"]
            image_pos = meta["image_positions"]
            other_pos = meta["role_and_other_positions"]

            call_tag = (row_key, condition)
            count = self.call_counts.get(call_tag, 0) + 1
            self.call_counts[call_tag] = count
            self.forward_calls.append({
                "row_key": row_key,
                "condition": condition,
                "output_attentions": output_attentions,
                "token_count": token_count,
            })

            if self.raise_on_call:
                raise RuntimeError("synthetic model forward failure")

            base_seed = int.from_bytes(
                hashlib.sha256(f"{row_key}|{condition}".encode("utf-8")).digest()[:4],
                "little",
            ) % 100
            attentions = [] if output_attentions else None

            for b_idx, block in enumerate(self.blocks):
                depth_1b = b_idx + 1
                hidden = np.zeros((1, token_count, self.dim), dtype=np.float32)
                for tok_i in range(token_count):
                    for ch_i in range(self.dim):
                        hidden[0, tok_i, ch_i] = float(base_seed * 0.01 + depth_1b * 0.1 + tok_i * 0.02 + ch_i * 0.001)
                if call_tag in self.flaky_hidden_keys and count > 1:
                    hidden[0, caption_pos[0], 0] += 0.5

                h_tensor = _FakeTensor(hidden)
                for fn in list(block.hooks):
                    fn(block, (h_tensor,), h_tensor)

                if output_attentions:
                    n_heads = 2
                    att = np.zeros((1, n_heads, token_count, token_count), dtype=np.float32)
                    # Distribute weights per query token deterministically over image, caption, other.
                    # Make last caption query focus heavily on image, earlier caption queries differ.
                    for q in range(token_count):
                        if q == caption_pos[-1]:
                            w_img, w_cap, w_oth = 0.50, 0.30, 0.20
                        elif q in caption_pos:
                            w_img, w_cap, w_oth = 0.20, 0.50, 0.30
                        else:
                            w_img, w_cap, w_oth = 0.10, 0.40, 0.50
                        if call_tag in self.flaky_att_keys and count > 1 and q == caption_pos[-1]:
                            w_img, w_cap, w_oth = 0.10, 0.70, 0.20
                        for head_i in range(n_heads):
                            # Slight head variation that averages cleanly across the 2 heads
                            delta = 0.04 if head_i == 0 else -0.04
                            hi_img = w_img + delta
                            hi_cap = w_cap - delta
                            hi_oth = w_oth
                            for k in image_pos:
                                att[0, head_i, q, k] = hi_img / len(image_pos)
                            for k in caption_pos:
                                att[0, head_i, q, k] = hi_cap / len(caption_pos)
                            for k in other_pos:
                                att[0, head_i, q, k] = hi_oth / len(other_pos)
                    attentions.append(_FakeTensor(att))

            return SimpleNamespace(attentions=attentions)

    class FakeE39E31HFAdapter(HFAdapter):
        """Explicit HFAdapter fixture subclass for CPU-only E39+E31 combined hook tests."""

        def __init__(
            self,
            key,
            config,
            root,
            *,
            dim=8,
            flaky_hidden_keys=(),
            flaky_att_keys=(),
            raise_on_call=False,
            use_precomputed_tokens=False,
        ):
            self.key = key
            self.config = dict(config)
            self.root = Path(root).resolve()
            self.dim = int(dim)
            self.depths = [9, 18, 27, 36] if key == "qwen" else [9, 17, 26, 34]
            self.midpoint = 18 if key == "qwen" else 17
            self.n_blocks = 36 if key == "qwen" else 34
            self.use_precomputed_tokens = use_precomputed_tokens

            self.torch = _FakeTorch()
            self.blocks = [_FakeBlock(i) for i in range(self.n_blocks)]
            self.model = _FakeModel(
                self.blocks,
                dim=self.dim,
                flaky_hidden_keys=flaky_hidden_keys,
                flaky_att_keys=flaky_att_keys,
                raise_on_call=raise_on_call,
            )
            self.processor = SimpleNamespace(tokenizer=_WhitespaceOffsetTokenizer())

        def describe(self):
            sites = [f"decoder:{d}:{p}" for d in self.depths for p in POOLINGS]
            return {
                "decoder_blocks": self.n_blocks,
                "decoder_depths": list(self.depths),
                "midpoint": self.midpoint,
                "model_fingerprint": self.config["config_fingerprint"],
                "repeat_atol": self.config["repeat_atol"],
                "repeat_rtol": self.config["repeat_rtol"],
                "supported_sites": sites,
            }

        def describe_execution(self):
            site_specs = {
                f"decoder:{d}:{p}": {
                    "module_path": f"model.language_model.layers.{d - 1}",
                    "pooling": p,
                }
                for d in self.depths
                for p in POOLINGS
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

        def _prepare(self, row, condition="image_caption", purpose="preliminary_features"):
            assert purpose == "preliminary_features"
            tok = self.processor.tokenizer
            if condition == "image_caption":
                native = f"<|im_start|> <image> Caption:\n{row.caption_text} <|im_end|>"
                # Simulate Qwen image placeholder expansion (1 -> 2 tokens) in processed sequence
                aligned = tok(native)
                src_ids = aligned["input_ids"]
                img_idx = src_ids.index(tok.image_token_id)
                proc_ids = src_ids[:img_idx] + [tok.image_token_id, tok.image_token_id] + src_ids[img_idx + 1:]
                image_positions = [img_idx, img_idx + 1]
                shift = 1
                # Simulate Qwen decoded normalization artifact to ensure native_serialization is used for qwen
                decoded = native.replace("।", "\ufffd") if self.key == "qwen" else native
            elif condition == "caption_only":
                native = f"<|im_start|> Caption:\n{row.caption_text} <|im_end|>"
                aligned = tok(native)
                proc_ids = aligned["input_ids"]
                image_positions = []
                shift = 0
                decoded = native
            else:
                raise ContractError(f"Unsupported condition {condition!r}")

            from src.models.spans import caption_positions as span_caption_positions

            special = [t in tok.all_special_ids for t in aligned["input_ids"]]
            src_img = [i for i, t in enumerate(aligned["input_ids"]) if t == tok.image_token_id]
            src_cap = span_caption_positions(native, row.caption_text, aligned["offset_mapping"], special, src_img)
            caption_positions = [i + shift for i in src_cap]
            role_other = [i for i in range(len(proc_ids)) if i not in caption_positions and i not in image_positions]

            evidence = {
                "row_key": row.row_key,
                "condition": condition,
                "token_count": len(proc_ids),
                "native_serialization": native,
                "decoded_processor_sequence": decoded,
                "caption_alignment": "fixture_exact_offsets",
                "processor_attention_mask": [1] * len(proc_ids),
                "input_ids_sha256": hashlib.sha256(str(proc_ids).encode()).hexdigest(),
                "image_positions": image_positions,
                "caption_positions": caption_positions,
                "last_caption": caption_positions[-1],
                "role_and_other_positions": role_other,
            }
            if self.use_precomputed_tokens:
                evidence["caption_token_texts"] = [
                    native[a:b] for (a, b) in (aligned["offset_mapping"][i] for i in src_cap)
                ]
            inputs = {"meta": evidence}
            return inputs, evidence


def _build_phase3_gpu_fixture(
    root: Path,
    *,
    model="qwen",
    n_chunks=2,
    images_per_chunk=1,
    confirm_excluded=True,
    e35_excluded=True,
    confirm_rows=0,
    include_confirm_row=False,
    mismatch_depth_dim=False,
    bad_checkpoint=False,
):
    protocol = json.loads((REPO / "configs/protocol.json").read_text(encoding="utf-8"))
    proto = json.loads(json.dumps(protocol))
    n_screen = n_chunks * images_per_chunk
    proto["split_sizes"] = {"FIT": 2, "DEV-S": 1, "DEV-C": 1, "SCREEN": n_screen, "CONFIRM": 1}
    protocol_fp = fingerprint(proto)
    producer = {
        "code_commit": "4" * 40,
        "storage_scope": "local-development",
        "software_versions": {"python": "3.12.0"},
    }
    init_phase3_dirs(root)

    screen_ids = [f"{1000 + i:016x}" for i in range(n_screen)]
    confirm_id = f"{9000:016x}"
    splits = {iid: "SCREEN" for iid in screen_ids}
    splits[confirm_id] = "CONFIRM"

    records = []
    draw0_records = {}
    for iid in screen_ids + [confirm_id]:
        split = splits[iid]
        rel_img = f"images/{split.lower()}/{iid}.jpg"
        img_file = root / rel_img
        img_file.parent.mkdir(parents=True, exist_ok=True)
        img_file.write_bytes(f"FAKE_IMAGE_{iid}".encode("utf-8"))
        img_sha = file_sha256(img_file)
        for lang in LANGUAGES:
            if lang == "hi":
                cap_txt = f"एक कुत्ता {iid} ।"
            elif lang == "bn":
                cap_txt = f"একটি কুকুর {iid} ।"
            elif lang == "te":
                cap_txt = f"ఒక కుక్క {iid} ."
            else:
                cap_txt = f"A brown dog {iid} ."
            rec = {
                "image_id": iid,
                "image_path": rel_img,
                "image_sha256": img_sha,
                "language": lang,
                "caption_id": f"{iid}_{lang}_0",
                "caption_text": cap_txt,
                "split": split,
            }
            records.append(rec)
            if split == "SCREEN":
                draw0_records[(iid, lang)] = rec

    targets = [{"image_id": iid, "split": splits[iid], "values": [0] * 25} for iid in screen_ids + [confirm_id]]
    dataset_fp = fingerprint({"screen": screen_ids, "confirm": confirm_id})
    ds_payload = {
        "dataset.json": _json_bytes({"dataset_fingerprint": dataset_fp, "protocol_fingerprint": protocol_fp}),
        "records.jsonl": _jsonl_bytes(records),
        "targets.jsonl": _jsonl_bytes(targets),
        "splits.json": _json_bytes(splits),
    }
    ds_meta = {
        "kind": "stage1_dataset",
        "status": "passed",
        "run_id": "ds-run",
        "code_commit": "4" * 40,
        "storage_scope": "local-development",
        "software_versions": {"python": "3.12.0"},
        "protocol_fingerprint": protocol_fp,
        "dataset_fingerprint": dataset_fp,
        "model_fingerprint": None,
        "seeds": {},
        "upstream_artifacts": [],
    }
    ds_dir = publish(root, "shared/ds", ds_meta, ds_payload)
    dataset_ref = {"path": (ds_dir / MARKER).relative_to(root).as_posix(), "sha256": file_sha256(ds_dir / MARKER)}

    models_cfg = json.loads((REPO / "configs/models.json").read_text(encoding="utf-8"))["models"][model]
    model_cfg = dict(models_cfg)
    if bad_checkpoint:
        model_cfg["checkpoint_id"] = "wrong/checkpoint"
    model_cfg["config_fingerprint"] = fingerprint({k: v for k, v in model_cfg.items() if k != "config_fingerprint"})
    exec_fp = fingerprint({"model": model, "fp": model_cfg["config_fingerprint"]})
    model_cfg["_execution_fingerprint"] = exec_fp

    depths = [9, 18, 27, 36] if model == "qwen" else [9, 17, 26, 34]
    midpoint = 18 if model == "qwen" else 17
    n_blocks = 36 if model == "qwen" else 34

    site_specs = {
        f"decoder:{d}:{p}": {
            "module_path": f"model.language_model.layers.{d - 1}",
            "pooling": p,
        }
        for d in depths
        for p in POOLINGS
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
    dimensions = {f"decoder:{d}:{p}": 8 for d in depths for p in POOLINGS}
    if mismatch_depth_dim:
        dimensions[f"decoder:{depths[0]}:last_caption"] = 16

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
        "adapter": {"midpoint": midpoint, "decoder_blocks": n_blocks},
        "bridge_sha256": bridge_hash(),
        "references": {"readiness": r_ref},
    }
    mp_meta = dict(r_meta, kind="stage2_model_plan", run_id=f"mp-{model}")
    mp_dir = publish(root, f"cache/shared/stage2-plans/mp-{model}", mp_meta, {"plan.json": _json_bytes(model_plan_body)})
    mp_ref = {"path": (mp_dir / MARKER).relative_to(root).as_posix(), "sha256": file_sha256(mp_dir / MARKER)}

    probe_ref = storage_probe(root, proto, dataset_fp, producer)

    chunks = []
    total_views = 0
    for c_idx in range(n_chunks):
        batch_ids = screen_ids[c_idx * images_per_chunk : (c_idx + 1) * images_per_chunk]
        rows = []
        for iid in batch_ids:
            for lang in LANGUAGES:
                rec = draw0_records[(iid, lang)]
                row = {
                    "image_id": iid,
                    "image_path": rec["image_path"],
                    "image_sha256": rec["image_sha256"],
                    "language": lang,
                    "caption_id": rec["caption_id"],
                    "caption_draw_key": 0,
                    "caption_text": rec["caption_text"],
                    "caption_text_sha256": hashlib.sha256(rec["caption_text"].encode("utf-8")).hexdigest(),
                }
                if include_confirm_row and c_idx == 0 and lang == "te":
                    row["split"] = "CONFIRM"
                rows.append(row)
        total_views += len(rows)
        chunks.append({
            "chunk_id": f"SCREEN_{c_idx:04d}",
            "split": "SCREEN",
            "image_ids": batch_ids,
            "n_images": len(batch_ids),
            "n_views_per_condition": len(rows),
            "rows": rows,
        })

    e39_e31_plan = {
        "schema": "phase3.e39_e31_extraction_plan.v1",
        "split": "SCREEN",
        "confirm_rows": confirm_rows,
        "conditions": list(CONDITIONS),
        "e39_sites": list(E39_SITES),
        "e31_sites": list(E31_SITES),
        "chunk_size_images": 32,
        "total_chunks": len(chunks),
        "total_images": n_screen,
        "views_per_condition": total_views,
        "chunks": chunks,
    }

    gpu_plan_core = {
        "schema": SCHEMA_GPU_PLAN,
        "phase3_gpu_version": PHASE3_GPU_VERSION,
        "lineage_identity": fingerprint({"ds": dataset_fp}),
        "dataset_fingerprint": dataset_fp,
        "protocol_fingerprint": protocol_fp,
        "phase1_stage4_release": {"path": "results/stage4/r/_COMPLETE.json", "sha256": "1" * 64},
        "phase3_cpu_plan": None,
        "bootstrap": {"shape": [2000, n_screen], "sha256": "2" * 64},
        "storage_probe": plain(probe_ref),
        "initialized_dirs": list(PHASE3_PARENT_DIRS),
        "models": {
            "qwen": {
                "depths": [9, 18, 27, 36],
                "midpoint_depth": 18,
                "model_plan_ref": mp_ref,
                "midpoint_groups": {},
                "source_systems": {},
            },
            "gemma": {
                "depths": [9, 17, 26, 34],
                "midpoint_depth": 17,
                "model_plan_ref": mp_ref,
                "midpoint_groups": {},
                "source_systems": {},
            },
        },
        "e39_e31_plan": e39_e31_plan,
        "e23_plan": {"schema": "phase3.e23_pair_plan.v1", "chunks": [], "authentic_parity_sample": []},
        "decision_ranges": {},
        "e35_excluded": e35_excluded,
        "confirm_excluded": confirm_excluded,
    }
    gpu_plan_id = fingerprint(gpu_plan_core)
    gpu_plan_body = dict(gpu_plan_core, plan_identity=gpu_plan_id)
    gpu_plan_ref = publish_phase3(
        root,
        proto,
        dataset_fp,
        "phase3/plans",
        KIND_GPU_PLAN,
        gpu_plan_body,
        producer,
    )
    return proto, producer, gpu_plan_ref, model_cfg, model_plan_body, dataset_ref


class ExtractE39E31ContractAndStdlibTests(unittest.TestCase):
    """Stdlib-safe tests running cleanly without numpy, scipy, or torch."""

    def test_lazy_imports_producer_and_array_specs(self):
        if not HAS_NUMPY:
            self.assertNotIn("numpy", sys.modules)
        if not HAS_TORCH:
            self.assertNotIn("torch", sys.modules)
        if not HAS_SCIPY:
            self.assertNotIn("scipy", sys.modules)

        self.assertIs(extract_e39_e31.stage2_current_producer, stage2_current_producer)

        specs = expected_array_specs([9, 18, 27, 36], 8)
        self.assertEqual(len(specs), 76)
        self.assertEqual(specs["image_caption__d18__caption_mean"], 8)
        self.assertEqual(specs["caption_only__d36__last_non_punct_caption"], 8)
        self.assertEqual(specs["image_caption__d9__decoder_visual_mean"], 8)
        self.assertEqual(specs["image_caption__d27__attention_from_last_caption"], 3)
        self.assertEqual(specs["image_caption__d36__attention_from_caption_mean"], 3)

    def test_cli_help_and_blocked_exit_with_redaction(self):
        help_res = subprocess.run(
            [sys.executable, "-m", "src.phase3.extract_e39_e31", "--help"],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        self.assertEqual(help_res.returncode, 0, help_res.stderr)
        self.assertIn("--model", help_res.stdout)
        self.assertIn("--plan", help_res.stdout)

        with tempfile.TemporaryDirectory() as tmp:
            bad_proto = Path(tmp) / "missing_hf_SecretToken123456.json"
            env = dict(**__import__("os").environ, HF_TOKEN="custom_secret_value_99")
            blocked_res = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.phase3.extract_e39_e31",
                    "--model",
                    "qwen",
                    "--phase1-root",
                    tmp,
                    "--protocol",
                    str(bad_proto),
                ],
                cwd=REPO,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(blocked_res.returncode, 2)
            self.assertNotIn("hf_SecretToken123456", blocked_res.stdout)
            self.assertNotIn("custom_secret_value_99", blocked_res.stdout)
            payload = json.loads(blocked_res.stdout.strip().splitlines()[-1])
            self.assertEqual(payload["status"], "blocked")
            self.assertIn("[REDACTED]", payload["blockers"][0])

            with mock.patch.dict("os.environ", {"HF_TOKEN": "my_env_tok_xyz"}):
                self.assertEqual(redact_text("boom my_env_tok_xyz hf_AbCdEf123"), "boom [REDACTED] [REDACTED]")
                self.assertEqual(safe_error(ContractError("tok=my_env_tok_xyz")), "tok=[REDACTED]")

    def test_notebook_contract(self):
        nb_path = REPO / "scripts/phase3/extract_e39_e31.ipynb"
        self.assertTrue(nb_path.is_file(), "Missing scripts/phase3/extract_e39_e31.ipynb")
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertEqual(nb.get("nbformat"), 4)
        self.assertEqual(len(nb.get("cells", [])), 4)
        self.assertEqual(nb["cells"][0]["cell_type"], "markdown")

        for idx, cell in enumerate(nb["cells"][1:], start=1):
            self.assertEqual(cell["cell_type"], "code")
            self.assertEqual(cell["outputs"], [])
            self.assertIsNone(cell["execution_count"])
            compile("".join(cell["source"]), f"cell_{idx}", "exec")

        setup_src = "".join(nb["cells"][1]["source"])
        cfg_src = "".join(nb["cells"][2]["source"])
        exec_src = "".join(nb["cells"][3]["source"])

        self.assertIn("CODE_REF = 'phase3'", setup_src)
        self.assertIn("requirements-l4.txt", setup_src)
        self.assertIn("'L4' not in gpu_name", setup_src)
        self.assertIn("MODEL = 'qwen'", cfg_src)
        self.assertIn(CERTIFIED_STAGE_B1_GPU_PLAN["path"], cfg_src)
        self.assertIn(CERTIFIED_STAGE_B1_GPU_PLAN["sha256"], cfg_src)
        self.assertIn("src.phase3.extract_e39_e31", exec_src)
        self.assertIn("HF_TOKEN", exec_src)
        self.assertIn("[REDACTED]", exec_src)

    def test_recover_caption_token_texts_indic_unicode_and_mock_path(self):
        tok = _WhitespaceOffsetTokenizer()
        caption_hi = "एक कुत्ता दौड़ता है ।"
        native_qwen = f"<|im_start|> <image> Caption:\n{caption_hi} <|im_end|>"
        # Simulate Qwen decoded sequence replacing multi-byte danda with U+FFFD
        decoded_corrupt = native_qwen.replace("।", "\ufffd")
        adapter_qwen = SimpleNamespace(
            config={"adapter": "qwen", "image_token_ids": [tok.image_token_id]},
            processor=SimpleNamespace(tokenizer=tok),
        )
        row_hi = ModelInput(
            image_id="0000000000000001",
            image_path="images/screen/0000000000000001.jpg",
            image_sha256="a" * 64,
            language="hi",
            caption_id="cap_hi_0",
            caption_text=caption_hi,
            split="SCREEN",
        )
        # Image token expanded from 1 -> 2, so 5 caption tokens sit at indices 4..8
        raw_qwen = {
            "native_serialization": native_qwen,
            "decoded_processor_sequence": decoded_corrupt,
            "caption_positions": [4, 5, 6, 7, 8],
        }
        tokens_qwen = recover_caption_token_texts(adapter_qwen, row_hi, raw_qwen)
        self.assertEqual(tokens_qwen, ["एक", "कुत्ता", "दौड़ता", "है", "।"])
        self.assertNotIn("\ufffd", "".join(tokens_qwen))
        sel = select_e39_token_indices(tokens_qwen)
        self.assertEqual(sel["last_caption"], 4)
        self.assertEqual(sel["last_non_punct_caption"], 3)
        self.assertFalse(sel["last_non_punct_fallback_used"])

        # Gemma uses decoded_processor_sequence
        adapter_gemma = SimpleNamespace(
            config={"adapter": "gemma", "image_token_ids": [tok.image_token_id]},
            processor=SimpleNamespace(tokenizer=tok),
        )
        native_gemma = f"<|im_start|> Caption:\n{caption_hi} <|im_end|>"
        raw_gemma = {
            "native_serialization": native_gemma,
            "decoded_processor_sequence": native_gemma,
            "caption_positions": [2, 3, 4, 5, 6],
        }
        tokens_gemma = recover_caption_token_texts(adapter_gemma, row_hi, raw_gemma)
        self.assertEqual(tokens_gemma, ["एक", "कुत्ता", "दौड़ता", "है", "।"])

        # Precomputed mock path
        tokens_mock = recover_caption_token_texts(
            adapter_qwen,
            row_hi,
            {"caption_positions": [0, 1, 2], "caption_token_texts": ["एक", "कुत्ता", "।"]},
        )
        self.assertEqual(tokens_mock, ["एक", "कुत्ता", "।"])

        with self.assertRaisesRegex(ContractError, "length"):
            recover_caption_token_texts(
                adapter_qwen,
                row_hi,
                {"caption_positions": [0, 1], "caption_token_texts": ["एक"]},
            )

    def test_preflight_bind_false_and_safety_rejections(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proto, _, gpu_plan_ref, _, model_plan_body, _ = _build_phase3_gpu_fixture(root)

            with mock.patch("src.phase3.extract_e39_e31.load_job_plan", return_value=model_plan_body), \
                 mock.patch("src.phase3.gpu_plan.bind_phase1", side_effect=AssertionError("Must not call bind_phase1")):
                ctx = preflight_e39_e31_extraction(root, proto, "qwen", plan_ref=gpu_plan_ref)
                self.assertEqual(ctx["depths"], [9, 18, 27, 36])
                self.assertEqual(ctx["midpoint_depth"], 18)
                self.assertEqual(ctx["dimension"], 8)

            with self.assertRaisesRegex(ContractError, "Unsupported MODEL"):
                preflight_e39_e31_extraction(root, proto, "llama", plan_ref=gpu_plan_ref)

        for label, bad_kwargs in (
            ("confirm_excluded=False", {"confirm_excluded": False}),
            ("e35_excluded=False", {"e35_excluded": False}),
            ("confirm_rows=1", {"confirm_rows": 1}),
            ("confirm_row_split", {"include_confirm_row": True}),
            ("mismatch_depth_dim", {"mismatch_depth_dim": True}),
            ("bad_checkpoint", {"bad_checkpoint": True}),
        ):
            with self.subTest(case=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                proto, _, gpu_plan_ref, _, model_plan_body, _ = _build_phase3_gpu_fixture(root, **bad_kwargs)
                with mock.patch("src.phase3.extract_e39_e31.load_job_plan", return_value=model_plan_body):
                    with self.assertRaises(ContractError):
                        preflight_e39_e31_extraction(root, proto, "qwen", plan_ref=gpu_plan_ref)


@unittest.skipUnless(HAS_NUMPY, "numpy required for E39+E31 end-to-end extraction tests")
class ExtractE39E31EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        global np
        import numpy as np_mod

        np = np_mod

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _run_fixture(
        self,
        *,
        model="qwen",
        n_chunks=2,
        images_per_chunk=1,
        flaky_hidden_keys=(),
        flaky_att_keys=(),
        raise_on_call=False,
        use_precomputed_tokens=False,
        resume_refs=None,
        run_id=None,
    ):
        proto, producer, gpu_plan_ref, model_cfg, model_plan_body, _ = _build_phase3_gpu_fixture(
            self.root, model=model, n_chunks=n_chunks, images_per_chunk=images_per_chunk
        )
        adapter = FakeE39E31HFAdapter(
            model,
            model_cfg,
            self.root,
            dim=8,
            flaky_hidden_keys=flaky_hidden_keys,
            flaky_att_keys=flaky_att_keys,
            raise_on_call=raise_on_call,
            use_precomputed_tokens=use_precomputed_tokens,
        )
        events = []
        with mock.patch("src.phase3.extract_e39_e31.load_job_plan", return_value=model_plan_body):
            res = run_e39_e31_extraction(
                self.root,
                proto,
                model,
                producer,
                plan_ref=gpu_plan_ref,
                adapter_factory=lambda m, cfg, r: adapter,
                resume_refs=resume_refs,
                run_id=run_id,
                progress=events.append,
            )
        return proto, producer, gpu_plan_ref, model_plan_body, adapter, res, events

    def test_end_to_end_combined_forward_hooks_76_arrays_and_analytical_values(self):
        proto, _, gpu_plan_ref, _, adapter, res, events = self._run_fixture(
            model="qwen", n_chunks=2, images_per_chunk=1, run_id="test_batch"
        )
        self.assertEqual(res["status"], "passed")
        self.assertEqual(res["model"], "qwen")
        self.assertEqual(list(res["depths"]), [9, 18, 27, 36])
        self.assertEqual(res["total_images"], 2)
        self.assertEqual(res["views_per_condition"], 8)
        self.assertEqual(res["total_chunks"], 2)
        self.assertEqual(res["array_count_per_chunk"], 76)
        self.assertTrue(res["e35_excluded"])
        self.assertTrue(res["confirm_excluded"])

        # 4 views per chunk * 2 conditions + 2 repeat indices * 2 conditions = 12 forwards/chunk * 2 chunks = 24
        self.assertEqual(len(adapter.model.forward_calls), 24)
        for call in adapter.model.forward_calls:
            self.assertEqual(call["output_attentions"], call["condition"] == "image_caption")

        # All hooks must be cleaned up after forward passes
        self.assertTrue(all(len(b.hooks) == 0 for b in adapter.blocks))

        event_names = [e["event"] for e in events]
        self.assertIn("e39_e31_extract_start", event_names)
        self.assertEqual(event_names.count("e39_e31_chunk_published"), 2)
        self.assertIn("e39_e31_release_published", event_names)

        rel_ref, rel_body, _ = verify_e39_e31_release(self.root, proto, res["release_ref"], expected_model="qwen")
        self.assertEqual(rel_body["experiment"], "E39+E31")
        self.assertTrue(rel_body["repeat_check_summary"]["all_passed"])

        # Inspect chunk 0 arrays & rows.jsonl for exact analytical correctness
        chunk0_dir = (self.root / res["chunks"][0]["ref"]["path"]).parent
        chunk0_rows = [json.loads(line) for line in (chunk0_dir / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(chunk0_rows), 4)
        self.assertEqual([r["language"] for r in chunk0_rows], list(LANGUAGES))

        # Hindi row (index 1): caption "एक कुत्ता 00000000000003e8 ।" -> 4 tokens; last_non_punct_caption skips "।" (idx 3) -> idx 2
        hi_row = chunk0_rows[1]
        hi_sel_img = hi_row["e39_token_selection"]["image_caption"]
        self.assertEqual(hi_sel_img["token_count"], 4)
        self.assertEqual(hi_sel_img["last_caption"], 3)
        self.assertEqual(hi_sel_img["last_non_punct_caption"], 2)
        self.assertFalse(hi_sel_img["last_non_punct_fallback_used"])

        # Verify analytical vector values for Hindi image_caption at depth 18
        # In FakeE39E31HFAdapter, image_caption has 2 image tokens (0, 1), 1 start (2), 1 "Caption:" (3), caption at [4,5,6,7], end (8)
        row_key_hi = (hi_row["image_id"], "hi", hi_row["caption_id"])
        base_seed = int.from_bytes(
            hashlib.sha256(f"{row_key_hi}|image_caption".encode("utf-8")).digest()[:4],
            "little",
        ) % 100
        # last_non_punct_caption = token 2 inside caption -> processed pos = 6
        expected_last_non_punct = np.array(
            [base_seed * 0.01 + 18 * 0.1 + 6 * 0.02 + ch * 0.001 for ch in range(8)],
            dtype="<f4",
        )
        arr_lnp = np.load(chunk0_dir / "image_caption__d18__last_non_punct_caption.npy", allow_pickle=False)
        np.testing.assert_allclose(arr_lnp[1], expected_last_non_punct, rtol=0.0, atol=1e-6)

        # decoder_visual_mean averages image_positions [1, 2] -> mean pos = 1.5
        expected_vis_mean = np.array(
            [base_seed * 0.01 + 18 * 0.1 + 1.5 * 0.02 + ch * 0.001 for ch in range(8)],
            dtype="<f4",
        )
        arr_vis = np.load(chunk0_dir / "image_caption__d18__decoder_visual_mean.npy", allow_pickle=False)
        np.testing.assert_allclose(arr_vis[1], expected_vis_mean, rtol=0.0, atol=1e-6)

        # attention_from_last_caption has [w_img, w_cap, w_oth] = [0.50, 0.30, 0.20]
        arr_att_last = np.load(chunk0_dir / "image_caption__d18__attention_from_last_caption.npy", allow_pickle=False)
        np.testing.assert_allclose(arr_att_last[1], np.array([0.50, 0.30, 0.20], dtype="<f4"), rtol=0.0, atol=1e-6)

        # attention_from_caption_mean averages 1 last-caption query [0.5, 0.3, 0.2] and 3 earlier queries [0.2, 0.5, 0.3]
        expected_att_mean = np.array([0.275, 0.45, 0.275], dtype="<f4")
        arr_att_mean = np.load(chunk0_dir / "image_caption__d36__attention_from_caption_mean.npy", allow_pickle=False)
        np.testing.assert_allclose(arr_att_mean[1], expected_att_mean, rtol=0.0, atol=1e-6)

        # Token evidence files exist and match row count
        for ev_name in ("token_evidence_image_caption.jsonl", "token_evidence_caption_only.jsonl"):
            ev_lines = (chunk0_dir / ev_name).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(ev_lines), 4)

    def test_hook_cleanup_on_forward_exception(self):
        proto, producer, gpu_plan_ref, model_cfg, model_plan_body, _ = _build_phase3_gpu_fixture(
            self.root, model="qwen", n_chunks=1, images_per_chunk=1
        )
        adapter = FakeE39E31HFAdapter("qwen", model_cfg, self.root, raise_on_call=True)
        with mock.patch("src.phase3.extract_e39_e31.load_job_plan", return_value=model_plan_body):
            with self.assertRaisesRegex(RuntimeError, "synthetic model forward failure"):
                run_e39_e31_extraction(
                    self.root,
                    proto,
                    "qwen",
                    producer,
                    plan_ref=gpu_plan_ref,
                    adapter_factory=lambda m, cfg, r: adapter,
                    progress=[].append,
                )
        self.assertTrue(all(len(b.hooks) == 0 for b in adapter.blocks))

    def test_repeat_check_failure_hidden_and_attention(self):
        first_id = f"{1000:016x}"
        bad_key = ((first_id, "en", f"{first_id}_en_0"), "image_caption")

        with self.assertRaisesRegex(ContractError, "Repeat determinism check failed"):
            self._run_fixture(model="qwen", n_chunks=1, images_per_chunk=1, flaky_hidden_keys={bad_key})

        tmp2 = tempfile.TemporaryDirectory()
        self.addCleanup(tmp2.cleanup)
        root2 = Path(tmp2.name)
        proto2, prod2, plan_ref2, cfg2, mplan2, _ = _build_phase3_gpu_fixture(
            root2, model="qwen", n_chunks=1, images_per_chunk=1
        )
        adapter2 = FakeE39E31HFAdapter("qwen", cfg2, root2, flaky_att_keys={bad_key})
        with mock.patch("src.phase3.extract_e39_e31.load_job_plan", return_value=mplan2):
            with self.assertRaisesRegex(ContractError, "Repeat determinism check failed"):
                run_e39_e31_extraction(
                    root2,
                    proto2,
                    "qwen",
                    prod2,
                    plan_ref=plan_ref2,
                    adapter_factory=lambda m, cfg, r: adapter2,
                    progress=[].append,
                )

    def test_chunk_resume_and_tamper_rejection(self):
        proto, producer, gpu_plan_ref, model_cfg, model_plan_body, _ = _build_phase3_gpu_fixture(
            self.root, model="gemma", n_chunks=2, images_per_chunk=1
        )
        adapter1 = FakeE39E31HFAdapter("gemma", model_cfg, self.root, use_precomputed_tokens=True)
        events1 = []
        with mock.patch("src.phase3.extract_e39_e31.load_job_plan", return_value=model_plan_body):
            res1 = run_e39_e31_extraction(
                self.root,
                proto,
                "gemma",
                producer,
                plan_ref=gpu_plan_ref,
                adapter_factory=lambda m, cfg, r: adapter1,
                progress=events1.append,
            )
        self.assertEqual(len(adapter1.model.forward_calls), 24)
        chunk0_ref = res1["chunks"][0]["ref"]

        # Resume chunk 0 -> only chunk 1 (12 forwards) should execute.
        adapter2 = FakeE39E31HFAdapter("gemma", model_cfg, self.root, use_precomputed_tokens=True)
        events2 = []
        with mock.patch("src.phase3.extract_e39_e31.load_job_plan", return_value=model_plan_body):
            res2 = run_e39_e31_extraction(
                self.root,
                proto,
                "gemma",
                producer,
                plan_ref=gpu_plan_ref,
                adapter_factory=lambda m, cfg, r: adapter2,
                resume_refs={0: chunk0_ref},
                progress=events2.append,
            )
        self.assertEqual(len(adapter2.model.forward_calls), 12)
        self.assertTrue(any(e["event"] == "e39_e31_chunk_resumed" and e["chunk_index"] == 0 for e in events2))
        verify_e39_e31_release(self.root, proto, res2["release_ref"], expected_model="gemma")

        # Passing chunk 0 as chunk 1 must be rejected.
        adapter3 = FakeE39E31HFAdapter("gemma", model_cfg, self.root)
        with mock.patch("src.phase3.extract_e39_e31.load_job_plan", return_value=model_plan_body):
            with self.assertRaisesRegex(ContractError, "chunk_index mismatch"):
                run_e39_e31_extraction(
                    self.root,
                    proto,
                    "gemma",
                    producer,
                    plan_ref=gpu_plan_ref,
                    adapter_factory=lambda m, cfg, r: adapter3,
                    resume_refs={1: chunk0_ref},
                )

        # Corrupting an .npy array on disk must be rejected by verify_e39_e31_release.
        chunk1_dir = (self.root / res2["chunks"][1]["ref"]["path"]).parent
        (chunk1_dir / "image_caption__d17__caption_mean.npy").write_bytes(b"corrupted")
        with self.assertRaises(ContractError):
            verify_e39_e31_release(self.root, proto, res2["release_ref"], expected_model="gemma")


if __name__ == "__main__":
    unittest.main()
