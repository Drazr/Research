"""Phase 3 Stage B2 Worker 1: L4 GPU combined E39+E31 SCREEN extraction.

Extracts `SCREEN`-only draw-0 features for BOTH `image_caption` and `caption_only`
across all 4 fixed decoder depths (`qwen`: `[9, 18, 27, 36]`; `gemma`: `[9, 17, 26, 34]`)
from a single read-only forward pass per `(view, condition)`:
- All 8 `E39_SITES` sub-caption token slices & span means (both conditions).
- All 3 `E31_SITES` (`decoder_visual_mean`, `attention_from_last_caption`,
  `attention_from_caption_mean`) on the same `image_caption` forward pass.

`E35` (`G7`) is dropped (`e35_excluded=True`), and `CONFIRM` remains strictly
closed and rejected before any image IO or model loading.
"""

import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Optional

from src.common.contracts import (
    ContractError,
    FeatureSite,
    LANGUAGES,
    ModelInput,
    require_sha256,
)
from src.common.storage import file_sha256, new_run_id, phase1_root, within
from src.models.registry import create_adapter
from src.models.spans import caption_positions, validate_attention
from src.phase3.gpu_plan import (
    CERTIFIED_STAGE_B1_GPU_PLAN,
    E31_SITES,
    E39_SITES,
    load_phase3_gpu_plan,
    select_e39_token_indices,
)
from src.phase3.schema import (
    CONDITIONS,
    KIND_E39_E31_CHUNK,
    KIND_E39_E31_RELEASE,
    MODELS,
    POOLINGS,
    REPO,
    SCHEMA_E39_E31_FEATURES,
    check,
    contract,
    frozen,
    load_phase3_artifact,
    plain,
    publish_phase3,
    require_open_split,
)
from src.stage2.evidence import bridge_hash, validate_token_evidence
from src.stage2.operations import current_producer as stage2_current_producer
from src.stage2.plans import load_job_plan
from src.stage2.readiness import assert_runtime
from src.stage2.schema import artifact, json_bytes, jsonl, read_json

E39_E31_SPLIT = "SCREEN"
E39_E31_EXPECTED_DEPTHS = {
    "qwen": [9, 18, 27, 36],
    "gemma": [9, 17, 26, 34],
}
E39_E31_EXPECTED_MIDPOINTS = {
    "qwen": 18,
    "gemma": 17,
}
E39_SINGLE_TOKEN_SITES = (
    "first_caption",
    "middle_caption",
    "penultimate_caption",
    "last_non_punct_caption",
    "last_caption",
)
E39_SPAN_MEAN_SITES = {
    "first_half_mean": "first_half_span",
    "second_half_mean": "second_half_span",
    "caption_mean": "full_span",
}
E31_ATTENTION_SITES = (
    "attention_from_last_caption",
    "attention_from_caption_mean",
)
ATTENTION_MASS_RAW_TOL = 0.02
ATTENTION_MASS_NORM_TOL = 1e-4


def _require_numpy():
    try:
        import numpy as np

        return np
    except ImportError as exc:
        raise ContractError("numpy is required for E39+E31 array extraction and verification") from exc


def redact_text(text: str) -> str:
    """Redact Hugging Face tokens (`HF_TOKEN` env var or `hf_*`) from text."""
    message = str(text)
    token = os.environ.get("HF_TOKEN")
    if token:
        message = message.replace(token, "[REDACTED]")
    return re.sub(r"hf_[A-Za-z0-9]+", "[REDACTED]", message)


def safe_error(exc: BaseException) -> str:
    """Redact any Hugging Face token from exception messages."""
    return redact_text(str(exc))


def _peak_rss_bytes() -> int:
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(usage) * 1024 if sys.platform != "darwin" else int(usage)
    except Exception:
        return 0


def _emit(progress, event: Mapping[str, Any]) -> None:
    if progress is not None:
        progress(dict(event))


def _default_emit(event: Mapping[str, Any]) -> None:
    print(redact_text(json.dumps(plain(event), sort_keys=True)), flush=True)


def expected_array_specs(depths, dimension: int) -> dict:
    """Return mapping of all 76 E39+E31 array keys to trailing axis widths."""
    specs = {}
    for d in depths:
        for cond in CONDITIONS:
            for site in E39_SITES:
                specs[f"{cond}__d{d}__{site}"] = int(dimension)
        specs[f"image_caption__d{d}__decoder_visual_mean"] = int(dimension)
        for att_site in E31_ATTENTION_SITES:
            specs[f"image_caption__d{d}__{att_site}"] = 3
    return specs


def _check_caption_text_sha256(row: Mapping[str, Any], label: str) -> None:
    text = row.get("caption_text")
    check(isinstance(text, str) and bool(text.strip()), f"Empty caption_text in {label}")
    expected = row.get("caption_text_sha256")
    require_sha256(expected)
    actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
    check(actual == expected, f"Caption UTF-8 SHA-256 mismatch in {label}")


def _validate_e39_e31_plan(ep: Any) -> None:
    check(isinstance(ep, Mapping), "Missing e39_e31_plan in phase3_gpu_plan")
    check(
        ep.get("confirm_rows") == 0,
        f"e39_e31_plan must have confirm_rows == 0; got {ep.get('confirm_rows')!r}",
    )
    require_open_split(ep.get("split"))
    check(
        ep.get("split") == E39_E31_SPLIT,
        f"e39_e31_plan split must be {E39_E31_SPLIT!r}; got {ep.get('split')!r}",
    )
    check(
        list(ep.get("conditions", [])) == list(CONDITIONS),
        f"e39_e31_plan conditions must be {list(CONDITIONS)!r}",
    )
    check(
        list(ep.get("e39_sites", [])) == list(E39_SITES),
        f"e39_e31_plan e39_sites must match E39_SITES",
    )
    check(
        list(ep.get("e31_sites", [])) == list(E31_SITES),
        f"e39_e31_plan e31_sites must match E31_SITES",
    )

    chunks = ep.get("chunks")
    check(isinstance(chunks, (list, tuple)) and len(chunks) > 0, "e39_e31_plan chunks must be nonempty")
    check(len(chunks) == ep.get("total_chunks"), "e39_e31_plan total_chunks mismatch")

    total_views = 0
    for idx, chunk in enumerate(chunks):
        check(isinstance(chunk, Mapping), f"Chunk {idx} must be a mapping")
        require_open_split(chunk.get("split"))
        check(chunk.get("split") == E39_E31_SPLIT, f"Chunk {idx} split must be {E39_E31_SPLIT!r}")
        expected_cid = f"SCREEN_{idx:04d}"
        check(chunk.get("chunk_id") == expected_cid, f"Chunk {idx} chunk_id must be {expected_cid!r}")
        rows = chunk.get("rows")
        check(isinstance(rows, (list, tuple)) and len(rows) > 0, f"Chunk {idx} rows must be nonempty")
        check(
            chunk.get("n_views_per_condition") == len(rows),
            f"Chunk {idx} n_views_per_condition mismatch",
        )
        for r_idx, row in enumerate(rows):
            check(isinstance(row, Mapping), f"Chunk {idx} row {r_idx} must be a mapping")
            if "split" in row:
                require_open_split(row.get("split"))
                check(row.get("split") == E39_E31_SPLIT, f"Chunk {idx} row {r_idx} split must be SCREEN")
            check(row.get("language") in LANGUAGES, f"Chunk {idx} row {r_idx} invalid language")
            check(
                type(row.get("caption_draw_key", 0)) is int and int(row.get("caption_draw_key", 0)) == 0,
                f"Chunk {idx} row {r_idx} must use caption_draw_key == 0",
            )
            check(
                isinstance(row.get("image_id"), str) and bool(row.get("image_id")),
                f"Chunk {idx} row {r_idx} missing image_id",
            )
            check(
                isinstance(row.get("image_path"), str) and bool(row.get("image_path")),
                f"Chunk {idx} row {r_idx} missing image_path",
            )
            require_sha256(row.get("image_sha256"))
            _check_caption_text_sha256(row, f"chunk {idx} row {r_idx}")
        total_views += len(rows)

    check(total_views == ep.get("views_per_condition"), "e39_e31_plan views_per_condition mismatch")


def _load_model_plan_and_readiness(
    root: Path,
    protocol: Mapping[str, Any],
    plan,
    model: str,
    depths,
    midpoint_depth: int,
):
    model_spec = plan["models"][model]
    model_plan_ref = plain(model_spec["model_plan_ref"])
    model_plan_path = within(root, model_plan_ref["path"])
    check(file_sha256(model_plan_path) == model_plan_ref["sha256"], f"Stage 2 model_plan checksum mismatch for {model}")
    model_plan = load_job_plan(model_plan_path, root, protocol)
    check(model_plan["kind"] == "stage2_model_plan" and model_plan["model"] == model, "Invalid Stage 2 model_plan")
    check(model_plan["dataset_fingerprint"] == plan["dataset_fingerprint"], "Stage 2 model_plan dataset mismatch")
    check(model_plan["protocol_fingerprint"] == plan["protocol_fingerprint"], "Stage 2 model_plan protocol mismatch")
    check(int(model_plan["adapter"]["midpoint"]) == midpoint_depth, "Stage 2 model_plan midpoint depth mismatch")

    models_json_path = REPO / "configs/models.json"
    check(models_json_path.is_file(), "Missing configs/models.json")
    models_cfg = json.loads(models_json_path.read_text(encoding="utf-8"))["models"][model]
    check(
        model_plan["model_config"]["checkpoint_id"] == models_cfg["checkpoint_id"],
        f"Stage 2 model_plan checkpoint_id differs from configs/models.json for {model}",
    )
    check(
        model_plan["model_config"]["revision"] == models_cfg["revision"],
        f"Stage 2 model_plan revision differs from configs/models.json for {model}",
    )

    readiness_ref = model_plan["references"]["readiness"]
    rp, readiness_meta = artifact(within(root, readiness_ref["path"]), root, protocol, "stage2_model_readiness")
    check(file_sha256(rp) == readiness_ref["sha256"], "Stage 2 readiness manifest checksum mismatch")
    readiness_body = read_json(rp.parent / "readiness.json")
    check(
        readiness_meta["model_fingerprint"] == model_plan["model_config"]["config_fingerprint"],
        "Stage 2 readiness model_fingerprint mismatch",
    )
    check(
        readiness_body["model_config"]["config_fingerprint"] == model_plan["model_config"]["config_fingerprint"],
        "Stage 2 readiness config_fingerprint mismatch",
    )
    check(
        plain(readiness_body["execution_identity"]) == plain(model_plan["execution_identity"]),
        "Stage 2 readiness execution_identity mismatch",
    )
    check(
        plain(readiness_body["dimensions"]) == plain(model_plan["dimensions"]),
        "Stage 2 readiness dimensions mismatch",
    )

    dims_found = set()
    for d in depths:
        for pool in POOLINGS:
            site_key = FeatureSite("decoder", d, pool).key
            check(site_key in readiness_body["dimensions"], f"Readiness missing dimension for site: {site_key}")
            check(
                site_key in readiness_body["execution_identity"]["site_specs"],
                f"Readiness execution_identity missing site_spec: {site_key}",
            )
            dim_val = int(readiness_body["dimensions"][site_key])
            check(dim_val > 0, f"Non-positive dimension {dim_val} for {site_key}")
            dims_found.add(dim_val)
    check(len(dims_found) == 1, f"Mismatched decoder dimensions across depths/poolings: {sorted(dims_found)}")
    return model_plan, readiness_body, next(iter(dims_found))


@contract
def preflight_e39_e31_extraction(
    root,
    protocol,
    model: str,
    *,
    plan_ref=None,
):
    """Authenticate `phase3_gpu_plan` with `bind=False` and enforce pre-GPU E39+E31 gates."""
    require_open_split(E39_E31_SPLIT)
    check(model in MODELS, f"Unsupported MODEL selector {model!r}; expected one of {MODELS}")
    resolved_root = Path(root).resolve()

    effective_plan_ref = plain(plan_ref) if plan_ref is not None else plain(CERTIFIED_STAGE_B1_GPU_PLAN)
    plan = load_phase3_gpu_plan(resolved_root, protocol, effective_plan_ref, bind=False)
    check(plan["confirm_excluded"] is True, "phase3_gpu_plan must enforce confirm_excluded=True")
    check(plan["e35_excluded"] is True, "phase3_gpu_plan must enforce e35_excluded=True (E35 dropped)")

    e39_e31_plan = plan["e39_e31_plan"]
    _validate_e39_e31_plan(e39_e31_plan)

    check(model in plan["models"], f"phase3_gpu_plan missing model spec for {model}")
    model_spec = plan["models"][model]
    depths = [int(d) for d in model_spec["depths"]]
    check(
        depths == E39_E31_EXPECTED_DEPTHS[model],
        f"Expected depths {E39_E31_EXPECTED_DEPTHS[model]} for {model}; got {depths}",
    )
    midpoint_depth = int(model_spec["midpoint_depth"])
    check(
        midpoint_depth == E39_E31_EXPECTED_MIDPOINTS[model] and midpoint_depth in depths,
        f"Expected midpoint depth {E39_E31_EXPECTED_MIDPOINTS[model]} for {model}; got {midpoint_depth}",
    )

    model_plan, readiness_body, dimension = _load_model_plan_and_readiness(
        resolved_root, protocol, plan, model, depths, midpoint_depth
    )
    return {
        "root": resolved_root,
        "protocol": protocol,
        "plan": plan,
        "e39_e31_plan": e39_e31_plan,
        "model": model,
        "depths": depths,
        "midpoint_depth": midpoint_depth,
        "dimension": dimension,
        "model_plan": model_plan,
        "readiness": readiness_body,
    }


def _verify_adapter_contract(adapter, model_plan, readiness, model: str, depths, midpoint_depth: int):
    assert_runtime(adapter, model_plan)
    check(adapter.config["precision"] == "bfloat16", "E39+E31 extraction requires bfloat16 precision")
    check(adapter.config["attention_backend"] == "eager", "E39+E31 extraction requires eager attention_backend")

    desc = adapter.describe()
    exec_desc = adapter.describe_execution()
    check(
        desc["model_fingerprint"] == readiness["model_config"]["config_fingerprint"],
        "Adapter model_fingerprint differs from Stage 2 model_readiness",
    )
    check(
        int(desc["midpoint"]) == midpoint_depth,
        f"Adapter midpoint {desc['midpoint']} differs from expected depth {midpoint_depth}",
    )
    check(
        [int(d) for d in desc["decoder_depths"]] == list(depths),
        f"Adapter decoder_depths {desc['decoder_depths']} differ from expected depths {depths}",
    )
    check(
        all(1 <= int(d) <= len(adapter.blocks) for d in depths),
        f"Adapter blocks length {len(adapter.blocks)} incompatible with depths {depths}",
    )
    check(
        exec_desc["execution_fingerprint"] == readiness["execution_identity"]["execution_fingerprint"],
        "Adapter execution_fingerprint differs from Stage 2 model_readiness",
    )
    check(
        plain(exec_desc["site_specs"]) == plain(readiness["execution_identity"]["site_specs"]),
        "Adapter site_specs differ from Stage 2 model_readiness",
    )

    dims_found = set()
    for d in depths:
        for pool in POOLINGS:
            site = FeatureSite("decoder", d, pool)
            check(site.key in desc["supported_sites"], f"Adapter missing required site: {site.key}")
            check(site.key in readiness["dimensions"], f"Readiness missing dimension for site: {site.key}")
            dim_val = int(readiness["dimensions"][site.key])
            check(dim_val > 0, f"Invalid non-positive dimension {dim_val} for {site.key}")
            dims_found.add(dim_val)
    check(len(dims_found) == 1, f"Adapter decoder dimensions mismatch across depths/poolings: {sorted(dims_found)}")
    return next(iter(dims_found)), desc, exec_desc


def recover_caption_token_texts(adapter, model_input: ModelInput, raw_evidence: Mapping[str, Any]) -> list:
    """Recover exact Unicode substrings for every token in `raw_evidence['caption_positions']`."""
    caption_pos = list(raw_evidence.get("caption_positions", []))
    check(len(caption_pos) > 0, f"Empty caption_positions for {model_input.image_id}/{model_input.language}")

    precomputed = raw_evidence.get("caption_token_texts")
    if precomputed is not None:
        token_texts = [str(tok) for tok in precomputed]
        check(
            len(token_texts) == len(caption_pos),
            f"caption_token_texts length {len(token_texts)} != caption_positions length {len(caption_pos)}",
        )
        return token_texts

    adapter_kind = adapter.config.get("adapter")
    alignment_text = (
        raw_evidence["native_serialization"]
        if adapter_kind == "qwen"
        else raw_evidence["decoded_processor_sequence"]
    )
    tok = adapter.processor.tokenizer
    aligned = tok(
        alignment_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )
    all_special_ids = getattr(tok, "all_special_ids", None)
    if all_special_ids is not None:
        special_set = set(all_special_ids)
        special = [token in special_set for token in aligned["input_ids"]]
    else:
        special = [bool(m) for m in aligned["special_tokens_mask"]]

    image_token_ids = set(adapter.config.get("image_token_ids", ()))
    source_image_positions = [i for i, token in enumerate(aligned["input_ids"]) if token in image_token_ids]
    src_caption_pos = caption_positions(
        alignment_text,
        model_input.caption_text,
        aligned["offset_mapping"],
        special,
        source_image_positions,
    )
    offsets = aligned["offset_mapping"]
    token_texts = [
        alignment_text[int(offsets[idx][0]):int(offsets[idx][1])]
        for idx in src_caption_pos
    ]
    check(
        len(token_texts) == len(caption_pos),
        f"Aligned caption_token_texts length {len(token_texts)} != caption_positions length {len(caption_pos)}",
    )
    return token_texts


def _build_sanitized_evidence(
    raw_evidence: Mapping[str, Any],
    condition: str,
    model_input: ModelInput,
    max_input_tokens: int,
) -> dict:
    evidence = dict(raw_evidence)
    native_text = evidence.pop("native_serialization")
    evidence.pop("decoded_processor_sequence", None)
    evidence.pop("caption_token_texts", None)
    check("attention_by_block" not in evidence, "Raw evidence must not contain attention_by_block")
    evidence["native_serialization_sha256"] = hashlib.sha256(native_text.encode("utf-8")).hexdigest()
    validate_token_evidence(evidence, condition, model_input.row_key, max_input_tokens)
    return evidence


def _tensor_to_f4_1d(tensor, dimension: int, label: str):
    np_mod = _require_numpy()
    arr = np_mod.asarray(tensor.detach().float().cpu().numpy(), dtype="<f4")
    check(arr.shape == (dimension,) and bool(np_mod.isfinite(arr).all()), f"Invalid/nonfinite vector for {label}")
    return np_mod.ascontiguousarray(arr, dtype="<f4")


def _extract_view_condition(
    *,
    root: Path,
    adapter,
    row: Mapping[str, Any],
    condition: str,
    depths,
    dimension: int,
):
    np_mod = _require_numpy()
    row_split = row.get("split", E39_E31_SPLIT)
    require_open_split(row_split)
    check(row_split == E39_E31_SPLIT, f"Refused non-SCREEN row split {row_split!r}")

    if condition == "image_caption":
        img_file = within(root, row["image_path"])
        check(img_file.is_file(), f"Missing pinned SCREEN image: {row['image_path']}")
        check(file_sha256(img_file) == row["image_sha256"], f"Pinned SCREEN image SHA-256 mismatch: {row['image_path']}")
    _check_caption_text_sha256(row, f"{row['image_id']}/{row['language']}")

    check(
        not adapter.model.training and all(not p.requires_grad for p in adapter.model.parameters()),
        "Frozen evaluation mode required",
    )

    model_input = ModelInput(
        image_id=row["image_id"],
        image_path=row["image_path"],
        image_sha256=row["image_sha256"],
        language=row["language"],
        caption_id=row["caption_id"],
        caption_text=row["caption_text"],
        split=E39_E31_SPLIT,
    )
    inputs, raw_evidence = adapter._prepare(model_input, condition=condition, purpose="preliminary_features")
    caption_token_texts = recover_caption_token_texts(adapter, model_input, raw_evidence)
    token_selection = select_e39_token_indices(caption_token_texts)
    evidence = _build_sanitized_evidence(
        raw_evidence,
        condition,
        model_input,
        int(adapter.config["max_input_tokens"]),
    )

    token_count = int(evidence["token_count"])
    caption_pos = list(evidence["caption_positions"])
    image_pos = list(evidence["image_positions"])
    other_pos = list(evidence["role_and_other_positions"])
    check(len(caption_pos) == int(token_selection["token_count"]), "E39 token_selection count mismatch")

    captured = {}
    handles = []
    out = None

    def _make_hook(depth_val: int):
        def _collect(module, args, output):
            if depth_val in captured:
                raise ContractError(f"Hook invoked more than once for depth {depth_val}")
            captured[depth_val] = adapter._tensor(output).detach()

        return _collect

    vectors = {}
    try:
        for d in depths:
            block = adapter.blocks[d - 1]
            handles.append(block.register_forward_hook(_make_hook(d)))

        with adapter.torch.inference_mode():
            out = adapter.model(
                **inputs,
                use_cache=False,
                output_attentions=(condition == "image_caption"),
                return_dict=True,
            )

        attentions = None
        if condition == "image_caption":
            attentions = getattr(out, "attentions", None)
            check(
                attentions is not None and len(attentions) == len(adapter.blocks) and all(a is not None for a in attentions),
                "Backend did not expose every decoder attention layer",
            )

        for d in depths:
            check(d in captured, f"Missing hooked residual output for depth {d}")
            h = captured[d]
            check(
                h.ndim == 3 and tuple(h.shape) == (1, token_count, dimension),
                f"Decoder hook shape {tuple(h.shape)} != expected (1, {token_count}, {dimension}) at depth {d}",
            )

            cap_h = h[0, caption_pos]
            for site in E39_SINGLE_TOKEN_SITES:
                idx = int(token_selection[site])
                vectors[f"{condition}__d{d}__{site}"] = _tensor_to_f4_1d(
                    cap_h[idx], dimension, f"{condition}:d{d}:{site}"
                )
            for site, span_key in E39_SPAN_MEAN_SITES.items():
                s0, s1 = token_selection[span_key]
                check(0 <= int(s0) < int(s1) <= len(caption_pos), f"Invalid span {span_key}={token_selection[span_key]}")
                vectors[f"{condition}__d{d}__{site}"] = _tensor_to_f4_1d(
                    cap_h[int(s0):int(s1)].mean(0), dimension, f"{condition}:d{d}:{site}"
                )

            if condition == "image_caption":
                check(len(image_pos) > 0, "image_caption requires nonempty image_positions")
                vectors[f"image_caption__d{d}__decoder_visual_mean"] = _tensor_to_f4_1d(
                    h[0, image_pos].mean(0),
                    dimension,
                    f"image_caption:d{d}:decoder_visual_mean",
                )

                att = attentions[d - 1][0].detach().float().cpu().numpy()
                check(
                    att.ndim == 3 and att.shape[1] == token_count and att.shape[2] == token_count,
                    f"Attention tensor shape {att.shape} mismatch for token_count={token_count} at depth {d}",
                )
                validate_attention(att, caption_pos, image_pos)
                att64 = np_mod.asarray(att, dtype=np_mod.float64)
                w_last = att64[:, caption_pos[-1], :].mean(axis=0)
                w_mean = att64[:, caption_pos, :].mean(axis=(0, 1))
                for att_site, weights in (
                    ("attention_from_last_caption", w_last),
                    ("attention_from_caption_mean", w_mean),
                ):
                    raw_parts = np_mod.array(
                        [
                            float(weights[image_pos].sum()),
                            float(weights[caption_pos].sum()),
                            float(weights[other_pos].sum()),
                        ],
                        dtype=np_mod.float64,
                    )
                    raw_total = float(raw_parts.sum())
                    check(
                        bool(np_mod.isfinite(raw_total))
                        and raw_total > 0.0
                        and abs(raw_total - 1.0) <= ATTENTION_MASS_RAW_TOL,
                        f"Attention partition sum {raw_total} out of tolerance at depth {d} ({att_site})",
                    )
                    vec3 = np_mod.ascontiguousarray(raw_parts / raw_total, dtype="<f4")
                    check(
                        vec3.shape == (3,)
                        and bool(np_mod.isfinite(vec3).all())
                        and bool((vec3 >= 0.0).all())
                        and abs(float(vec3.sum()) - 1.0) <= ATTENTION_MASS_NORM_TOL,
                        f"Normalized attention vector failed contract at depth {d} ({att_site})",
                    )
                    vectors[f"image_caption__d{d}__{att_site}"] = vec3
    finally:
        for handle in handles:
            handle.remove()
        captured.clear()
        del out, inputs

    return vectors, evidence, token_selection


def _normalize_resume_refs(resume_refs, total_chunks: int) -> dict:
    if not resume_refs:
        return {}
    normalized = {}
    if isinstance(resume_refs, Mapping):
        for raw_idx, ref in resume_refs.items():
            idx = int(raw_idx)
            check(0 <= idx < total_chunks, f"Resume chunk_index out of range: {idx}")
            normalized[idx] = ref
    elif isinstance(resume_refs, (list, tuple)):
        for entry in resume_refs:
            check(
                isinstance(entry, Mapping) and "chunk_index" in entry and "ref" in entry,
                "List resume_refs entries must contain 'chunk_index' and 'ref'",
            )
            idx = int(entry["chunk_index"])
            check(0 <= idx < total_chunks, f"Resume chunk_index out of range: {idx}")
            normalized[idx] = entry["ref"]
    else:
        raise ContractError("resume_refs must be a mapping {chunk_index: ref} or list of {'chunk_index', 'ref'}")
    return normalized


def _expected_chunk_base_rows(chunk: Mapping[str, Any]) -> list:
    expected = []
    for row_index, row in enumerate(chunk["rows"]):
        expected.append({
            "row_index": row_index,
            "image_id": row["image_id"],
            "image_path": row["image_path"],
            "image_sha256": row["image_sha256"],
            "language": row["language"],
            "split": E39_E31_SPLIT,
            "caption_id": row["caption_id"],
            "caption_draw_key": int(row.get("caption_draw_key", 0)),
            "caption_text": row["caption_text"],
            "caption_text_sha256": row["caption_text_sha256"],
        })
    return expected


def _verify_npy_array(path: Path, expected_sha256: str, n_rows: int, width: int, *, is_attention: bool = False):
    np_mod = _require_numpy()
    check(path.is_file(), f"Missing array file: {path.name}")
    check(file_sha256(path) == expected_sha256, f"Array SHA-256 mismatch: {path.name}")
    arr = np_mod.load(path, allow_pickle=False)
    check(
        isinstance(arr, np_mod.ndarray)
        and arr.dtype == np_mod.dtype("<f4")
        and arr.shape == (n_rows, width)
        and arr.flags.c_contiguous
        and bool(np_mod.isfinite(arr).all()),
        f"Invalid array dtype/shape/finiteness in {path.name}",
    )
    if is_attention:
        check(
            bool((arr >= 0.0).all())
            and bool(np_mod.allclose(arr.sum(axis=1), 1.0, atol=ATTENTION_MASS_NORM_TOL, rtol=0.0)),
            f"Attention array row sums failed normalization check in {path.name}",
        )
    return arr


def _verify_token_selection_entry(entry: Any) -> None:
    check(isinstance(entry, Mapping), "e39_token_selection entry must be a mapping")
    n_tok = entry.get("token_count")
    check(type(n_tok) is int and n_tok >= 1, "Invalid e39_token_selection token_count")
    for site in E39_SINGLE_TOKEN_SITES:
        val = entry.get(site)
        check(type(val) is int and 0 <= val < n_tok, f"Invalid e39_token_selection index for {site}")
    check(
        type(entry.get("last_non_punct_fallback_used")) is bool,
        "e39_token_selection missing boolean last_non_punct_fallback_used",
    )
    for span_key in ("first_half_span", "second_half_span", "full_span"):
        span = entry.get(span_key)
        check(
            isinstance(span, list)
            and len(span) == 2
            and type(span[0]) is int
            and type(span[1]) is int
            and 0 <= span[0] < span[1] <= n_tok,
            f"Invalid e39_token_selection span {span_key}",
        )


def _verify_resumed_chunk(
    *,
    root: Path,
    protocol: Mapping[str, Any],
    plan,
    model: str,
    depths,
    midpoint_depth: int,
    dimension: int,
    model_fingerprint: str,
    execution_fingerprint: str,
    max_input_tokens: int,
    chunk_index: int,
    total_chunks: int,
    chunk: Mapping[str, Any],
    reference,
):
    ref, body, marker = load_phase3_artifact(
        root,
        protocol,
        reference,
        KIND_E39_E31_CHUNK,
        expected_schema=SCHEMA_E39_E31_FEATURES,
        expected_dataset=plan["dataset_fingerprint"],
        expected_plan_identity=plan["plan_identity"],
        expected_owner=model,
    )
    check(body.get("experiment") == "E39+E31", "Resumed chunk experiment mismatch")
    check(plain(body.get("plan")) == plain(plan.reference), "Resumed chunk plan mismatch")
    check(body.get("model") == model, "Resumed chunk model mismatch")
    check(list(body.get("depths", [])) == list(depths), "Resumed chunk depths mismatch")
    check(body.get("midpoint_depth") == midpoint_depth, "Resumed chunk midpoint_depth mismatch")
    check(list(body.get("conditions", [])) == list(CONDITIONS), "Resumed chunk conditions mismatch")
    check(list(body.get("e39_sites", [])) == list(E39_SITES), "Resumed chunk e39_sites mismatch")
    check(list(body.get("e31_sites", [])) == list(E31_SITES), "Resumed chunk e31_sites mismatch")
    check(body.get("split") == E39_E31_SPLIT, "Resumed chunk split mismatch")
    check(body.get("e35_excluded") is True, "Resumed chunk e35_excluded must be True")
    check(body.get("confirm_excluded") is True, "Resumed chunk confirm_excluded must be True")
    check(body.get("chunk_index") == chunk_index, "Resumed chunk_index mismatch")
    check(body.get("chunk_id") == chunk["chunk_id"], "Resumed chunk_id mismatch")
    check(body.get("total_chunks") == total_chunks, "Resumed chunk total_chunks mismatch")
    n_views = len(chunk["rows"])
    check(body.get("n_views_per_condition") == n_views, "Resumed chunk n_views_per_condition mismatch")
    check(body.get("dimension") == dimension, "Resumed chunk dimension mismatch")
    check(body.get("model_fingerprint") == model_fingerprint, "Resumed chunk model_fingerprint mismatch")
    check(body.get("execution_fingerprint") == execution_fingerprint, "Resumed chunk execution_fingerprint mismatch")

    rc = body.get("repeat_check")
    check(isinstance(rc, Mapping) and rc.get("passed") is True, "Resumed chunk repeat_check did not pass")

    folder = within(root, ref["path"]).parent
    check("rows.jsonl" in marker["files"], "Resumed chunk missing rows.jsonl")
    check("audits.json" in marker["files"], "Resumed chunk missing audits.json")
    check("token_evidence_image_caption.jsonl" in marker["files"], "Resumed chunk missing token_evidence_image_caption.jsonl")
    check("token_evidence_caption_only.jsonl" in marker["files"], "Resumed chunk missing token_evidence_caption_only.jsonl")

    rows_path = folder / "rows.jsonl"
    rows_sha = file_sha256(rows_path)
    check(marker["files"]["rows.jsonl"] == rows_sha, "Resumed chunk rows.jsonl SHA-256 mismatch")
    check(body.get("rows_sha256") == rows_sha, "Resumed chunk body rows_sha256 mismatch")

    actual_rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_base = _expected_chunk_base_rows(chunk)
    check(len(actual_rows) == len(expected_base) == n_views, "Resumed chunk rows.jsonl row count mismatch")
    for act, exp in zip(actual_rows, expected_base):
        for k, v in exp.items():
            check(act.get(k) == v, f"Resumed chunk row field {k} mismatch")
        sel = act.get("e39_token_selection")
        check(isinstance(sel, Mapping) and set(sel.keys()) == set(CONDITIONS), "Resumed chunk missing e39_token_selection")
        for cond in CONDITIONS:
            _verify_token_selection_entry(sel[cond])

    for cond in CONDITIONS:
        ev_fname = f"token_evidence_{cond}.jsonl"
        ev_path = folder / ev_fname
        check(file_sha256(ev_path) == marker["files"][ev_fname], f"Resumed chunk {ev_fname} SHA-256 mismatch")
        ev_rows = [json.loads(line) for line in ev_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        check(len(ev_rows) == n_views, f"Resumed chunk {ev_fname} line count mismatch")
        for r_idx, ev in enumerate(ev_rows):
            row = chunk["rows"][r_idx]
            row_key = (row["image_id"], row["language"], row["caption_id"])
            validate_token_evidence(ev, cond, row_key, max_input_tokens)

    check(file_sha256(folder / "audits.json") == marker["files"]["audits.json"], "Resumed chunk audits.json SHA-256 mismatch")

    specs = expected_array_specs(depths, dimension)
    for key, width in specs.items():
        fname = f"{key}.npy"
        check(fname in marker["files"], f"Resumed chunk missing {fname}")
        _verify_npy_array(
            folder / fname,
            marker["files"][fname],
            n_views,
            width,
            is_attention=("__attention_from_" in key),
        )

    return plain(ref), body, rows_sha


@contract
def run_e39_e31_extraction(
    root,
    protocol,
    model: str,
    producer: Mapping[str, Any],
    *,
    plan_ref=None,
    adapter_factory=None,
    resume_refs=None,
    run_id: Optional[str] = None,
    progress=None,
):
    """Execute Phase 3 Stage B2 Worker 1 (E39+E31) extraction for `qwen` or `gemma`."""
    emit = progress if progress is not None else _default_emit
    t_read_hash = time.monotonic()
    ctx = preflight_e39_e31_extraction(root, protocol, model, plan_ref=plan_ref)
    read_hash_s = time.monotonic() - t_read_hash

    resolved_root = ctx["root"]
    plan = ctx["plan"]
    ep = ctx["e39_e31_plan"]
    depths = ctx["depths"]
    midpoint_depth = ctx["midpoint_depth"]
    dimension = ctx["dimension"]
    model_plan = ctx["model_plan"]
    readiness = ctx["readiness"]
    chunks = ep["chunks"]
    total_chunks = len(chunks)

    model_fingerprint = model_plan["model_config"]["config_fingerprint"]
    execution_fingerprint = model_plan["execution_identity"]["execution_fingerprint"]
    max_input_tokens = int(model_plan["model_config"]["max_input_tokens"])
    bridge_sha = bridge_hash()
    atol = float(model_plan["model_config"].get("repeat_atol", 0.0))
    rtol = float(model_plan["model_config"].get("repeat_rtol", 0.0))
    att_atol = max(atol, 1e-5)

    specs = expected_array_specs(depths, dimension)
    site_specs = {
        FeatureSite("decoder", d, pool).key: plain(
            readiness["execution_identity"]["site_specs"][FeatureSite("decoder", d, pool).key]
        )
        for d in depths
        for pool in POOLINGS
    }

    emit({
        "event": "e39_e31_extract_start",
        "unit": f"{model}:preflight",
        "completed": 0,
        "total": total_chunks,
        "read_hash_s": round(read_hash_s, 6),
        "compute_s": 0.0,
        "write_s": 0.0,
        "peak_rss_bytes": _peak_rss_bytes(),
        "model": model,
        "depths": depths,
        "midpoint_depth": midpoint_depth,
        "views_per_condition": ep["views_per_condition"],
        "total_chunks": total_chunks,
        "n_arrays_per_chunk": len(specs),
        "e35_excluded": True,
        "confirm_excluded": True,
    })

    normalized_resume = _normalize_resume_refs(resume_refs, total_chunks)
    completed_chunks = {}
    t_res_start = time.monotonic()
    for chunk_index, ref in sorted(normalized_resume.items()):
        verified_ref, chunk_body, rows_sha = _verify_resumed_chunk(
            root=resolved_root,
            protocol=protocol,
            plan=plan,
            model=model,
            depths=depths,
            midpoint_depth=midpoint_depth,
            dimension=dimension,
            model_fingerprint=model_fingerprint,
            execution_fingerprint=execution_fingerprint,
            max_input_tokens=max_input_tokens,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            chunk=chunks[chunk_index],
            reference=ref,
        )
        completed_chunks[chunk_index] = {
            "chunk_index": chunk_index,
            "chunk_id": chunks[chunk_index]["chunk_id"],
            "n_images": chunks[chunk_index]["n_images"],
            "n_views_per_condition": len(chunks[chunk_index]["rows"]),
            "ref": verified_ref,
            "rows_sha256": rows_sha,
            "repeat_check": plain(chunk_body["repeat_check"]),
        }
    read_hash_s += time.monotonic() - t_res_start

    pending_indices = [idx for idx in range(total_chunks) if idx not in completed_chunks]
    adapter = None
    if pending_indices:
        if adapter_factory is None:
            adapter = create_adapter(model, plain(model_plan["model_config"]), resolved_root)
        else:
            adapter = adapter_factory(model, plain(model_plan["model_config"]), resolved_root)
        verified_dim, _, _ = _verify_adapter_contract(
            adapter, model_plan, readiness, model, depths, midpoint_depth
        )
        check(verified_dim == dimension, "Verified adapter dimension mismatch")

    np_mod = _require_numpy()
    batch_run_id = run_id or new_run_id()
    compute_s = 0.0
    write_s = 0.0

    for chunk_index in range(total_chunks):
        if chunk_index in completed_chunks:
            emit({
                "event": "e39_e31_chunk_resumed",
                "unit": f"{model}:{chunks[chunk_index]['chunk_id']}",
                "completed": chunk_index + 1,
                "total": total_chunks,
                "read_hash_s": round(read_hash_s, 6),
                "compute_s": round(compute_s, 6),
                "write_s": round(write_s, 6),
                "peak_rss_bytes": _peak_rss_bytes(),
                "chunk_index": chunk_index,
                "ref": completed_chunks[chunk_index]["ref"],
            })
            continue

        chunk = chunks[chunk_index]
        rows = chunk["rows"]
        n_views = len(rows)
        base_rows = _expected_chunk_base_rows(chunk)

        t_comp = time.monotonic()
        per_key_lists = {key: [] for key in specs}
        evidences_by_cond = {cond: [] for cond in CONDITIONS}
        row_selections = []

        for r_idx, row in enumerate(rows):
            sel_for_row = {}
            for cond in CONDITIONS:
                vecs, ev, tok_sel = _extract_view_condition(
                    root=resolved_root,
                    adapter=adapter,
                    row=row,
                    condition=cond,
                    depths=depths,
                    dimension=dimension,
                )
                for key, vec in vecs.items():
                    per_key_lists[key].append(vec)
                evidences_by_cond[cond].append(ev)
                sel_for_row[cond] = plain(tok_sel)
            row_selections.append(sel_for_row)

        # First/last row repeat determinism check for both conditions.
        repeat_indices = sorted({0, n_views - 1})
        repeat_details = {}
        max_diff_hidden = 0.0
        max_diff_attention = 0.0
        repeat_passed = True
        for ridx in repeat_indices:
            rrow = rows[ridx]
            cond_details = {}
            for cond in CONDITIONS:
                rep_vecs, _, rep_sel = _extract_view_condition(
                    root=resolved_root,
                    adapter=adapter,
                    row=rrow,
                    condition=cond,
                    depths=depths,
                    dimension=dimension,
                )
                sel_match = plain(rep_sel) == row_selections[ridx][cond]
                cond_hidden_diff = 0.0
                cond_att_diff = 0.0
                cond_ok = sel_match
                for key, rep_vec in rep_vecs.items():
                    first_vec = per_key_lists[key][ridx]
                    diff = float(np_mod.max(np_mod.abs(first_vec - rep_vec)))
                    is_att = "__attention_from_" in key
                    tol = att_atol if is_att else atol
                    ok_key = bool(np_mod.isfinite(diff) and np_mod.allclose(first_vec, rep_vec, rtol=rtol, atol=tol))
                    if is_att:
                        cond_att_diff = max(cond_att_diff, diff)
                    else:
                        cond_hidden_diff = max(cond_hidden_diff, diff)
                    if not ok_key:
                        cond_ok = False
                max_diff_hidden = max(max_diff_hidden, cond_hidden_diff)
                max_diff_attention = max(max_diff_attention, cond_att_diff)
                if not cond_ok:
                    repeat_passed = False
                cond_details[cond] = {
                    "token_selection_match": sel_match,
                    "max_abs_diff_hidden": cond_hidden_diff,
                    "max_abs_diff_attention": cond_att_diff,
                    "passed": cond_ok,
                }
            repeat_details[str(ridx)] = cond_details

        compute_s += time.monotonic() - t_comp
        check(
            repeat_passed,
            f"Repeat determinism check failed for {model} chunk {chunk_index}: "
            f"max_abs_diff_hidden={max_diff_hidden}, max_abs_diff_attention={max_diff_attention}, "
            f"atol={atol}, att_atol={att_atol}, rtol={rtol}",
        )

        stacked_arrays = {}
        for key, width in specs.items():
            arr = np_mod.ascontiguousarray(np_mod.stack(per_key_lists[key], axis=0), dtype="<f4")
            check(
                arr.shape == (n_views, width) and bool(np_mod.isfinite(arr).all()),
                f"Stacked array contract mismatch for {key}: {arr.shape}",
            )
            stacked_arrays[key] = arr

        chunk_rows = [
            dict(base_row, e39_token_selection=sel)
            for base_row, sel in zip(base_rows, row_selections)
        ]
        rows_bytes = jsonl(chunk_rows)
        rows_sha = hashlib.sha256(rows_bytes).hexdigest()
        repeat_check = {
            "first_index": 0,
            "last_index": n_views - 1,
            "max_abs_diff_hidden": max_diff_hidden,
            "max_abs_diff_attention": max_diff_attention,
            "passed": True,
        }
        audits_obj = {
            "repeat_check": repeat_check,
            "repeat_details": repeat_details,
            "atol": atol,
            "attention_atol": att_atol,
            "rtol": rtol,
            "evidence_bridge_sha256": bridge_sha,
            "n_arrays": len(stacked_arrays),
        }
        chunk_body = {
            "schema": SCHEMA_E39_E31_FEATURES,
            "experiment": "E39+E31",
            "plan": plain(plan.reference),
            "plan_identity": plan["plan_identity"],
            "model": model,
            "depths": list(depths),
            "midpoint_depth": midpoint_depth,
            "conditions": list(CONDITIONS),
            "e39_sites": list(E39_SITES),
            "e31_sites": list(E31_SITES),
            "split": E39_E31_SPLIT,
            "chunk_index": chunk_index,
            "chunk_id": chunk["chunk_id"],
            "total_chunks": total_chunks,
            "n_images": chunk["n_images"],
            "n_views_per_condition": n_views,
            "dimension": dimension,
            "array_count": len(stacked_arrays),
            "model_fingerprint": model_fingerprint,
            "execution_fingerprint": execution_fingerprint,
            "site_specs": site_specs,
            "evidence_bridge_sha256": bridge_sha,
            "repeat_check": repeat_check,
            "rows_sha256": rows_sha,
            "e35_excluded": True,
            "confirm_excluded": True,
        }

        t_w = time.monotonic()
        chunk_ref = publish_phase3(
            resolved_root,
            protocol,
            plan["dataset_fingerprint"],
            f"phase3/features/e39_e31/{model}/{batch_run_id}",
            KIND_E39_E31_CHUNK,
            chunk_body,
            producer,
            arrays=stacked_arrays,
            files={
                "rows.jsonl": rows_bytes,
                "token_evidence_image_caption.jsonl": jsonl(evidences_by_cond["image_caption"]),
                "token_evidence_caption_only.jsonl": jsonl(evidences_by_cond["caption_only"]),
                "audits.json": json_bytes(audits_obj),
            },
            upstream=[plain(plan.reference)],
            model_fingerprint=model_fingerprint,
            run_id=f"chunk_{chunk_index:04d}",
        )
        write_s += time.monotonic() - t_w

        completed_chunks[chunk_index] = {
            "chunk_index": chunk_index,
            "chunk_id": chunk["chunk_id"],
            "n_images": chunk["n_images"],
            "n_views_per_condition": n_views,
            "ref": plain(chunk_ref),
            "rows_sha256": rows_sha,
            "repeat_check": repeat_check,
        }
        emit({
            "event": "e39_e31_chunk_published",
            "unit": f"{model}:{chunk['chunk_id']}",
            "completed": chunk_index + 1,
            "total": total_chunks,
            "read_hash_s": round(read_hash_s, 6),
            "compute_s": round(compute_s, 6),
            "write_s": round(write_s, 6),
            "peak_rss_bytes": _peak_rss_bytes(),
            "chunk_index": chunk_index,
            "n_views_per_condition": n_views,
            "ref": plain(chunk_ref),
        })

    ordered_chunks = [completed_chunks[i] for i in range(total_chunks)]
    repeat_summary = {
        "chunks_checked": total_chunks,
        "all_passed": all(c["repeat_check"]["passed"] for c in ordered_chunks),
        "max_abs_diff_hidden": max(c["repeat_check"]["max_abs_diff_hidden"] for c in ordered_chunks),
        "max_abs_diff_attention": max(c["repeat_check"]["max_abs_diff_attention"] for c in ordered_chunks),
    }
    release_chunks = [
        {
            "chunk_index": c["chunk_index"],
            "chunk_id": c["chunk_id"],
            "n_images": c["n_images"],
            "n_views_per_condition": c["n_views_per_condition"],
            "ref": c["ref"],
            "rows_sha256": c["rows_sha256"],
        }
        for c in ordered_chunks
    ]
    release_body = {
        "schema": SCHEMA_E39_E31_FEATURES,
        "experiment": "E39+E31",
        "plan": plain(plan.reference),
        "plan_identity": plan["plan_identity"],
        "batch_run_id": batch_run_id,
        "model": model,
        "depths": list(depths),
        "midpoint_depth": midpoint_depth,
        "conditions": list(CONDITIONS),
        "e39_sites": list(E39_SITES),
        "e31_sites": list(E31_SITES),
        "split": E39_E31_SPLIT,
        "confirm_rows": 0,
        "total_images": ep["total_images"],
        "views_per_condition": ep["views_per_condition"],
        "total_chunks": total_chunks,
        "dimension": dimension,
        "array_count_per_chunk": len(specs),
        "model_fingerprint": model_fingerprint,
        "execution_fingerprint": execution_fingerprint,
        "site_specs": site_specs,
        "evidence_bridge_sha256": bridge_sha,
        "chunks": release_chunks,
        "repeat_check_summary": repeat_summary,
        "e35_excluded": True,
        "confirm_excluded": True,
    }

    batch_dir = resolved_root / f"phase3/features/e39_e31/{model}/{batch_run_id}"
    release_run_id = f"{batch_run_id}_release" if (run_id or batch_dir.exists()) else new_run_id()
    t_rel = time.monotonic()
    release_ref = publish_phase3(
        resolved_root,
        protocol,
        plan["dataset_fingerprint"],
        f"phase3/features/e39_e31/{model}",
        KIND_E39_E31_RELEASE,
        release_body,
        producer,
        upstream=[plain(plan.reference)] + [c["ref"] for c in release_chunks],
        model_fingerprint=model_fingerprint,
        run_id=release_run_id,
    )
    write_s += time.monotonic() - t_rel

    emit({
        "event": "e39_e31_release_published",
        "unit": f"{model}:release",
        "completed": total_chunks,
        "total": total_chunks,
        "read_hash_s": round(read_hash_s, 6),
        "compute_s": round(compute_s, 6),
        "write_s": round(write_s, 6),
        "peak_rss_bytes": _peak_rss_bytes(),
        "model": model,
        "release_ref": plain(release_ref),
    })

    return frozen({
        "status": "passed",
        "model": model,
        "depths": list(depths),
        "midpoint_depth": midpoint_depth,
        "total_images": ep["total_images"],
        "views_per_condition": ep["views_per_condition"],
        "total_chunks": total_chunks,
        "array_count_per_chunk": len(specs),
        "release_ref": plain(release_ref),
        "chunks": release_chunks,
        "e35_excluded": True,
        "confirm_excluded": True,
    })


@contract
def verify_e39_e31_release(
    root,
    protocol,
    release_ref,
    *,
    expected_plan_identity: Optional[str] = None,
    expected_model: Optional[str] = None,
):
    """Authenticate a published `phase3_e39_e31_feature_release` and all referenced chunks."""
    resolved_root = Path(root).resolve()
    ref, body, marker = load_phase3_artifact(
        resolved_root,
        protocol,
        release_ref,
        KIND_E39_E31_RELEASE,
        expected_schema=SCHEMA_E39_E31_FEATURES,
        expected_plan_identity=expected_plan_identity,
        expected_owner=expected_model,
    )
    check(body.get("experiment") == "E39+E31", "Release experiment mismatch")
    check(body.get("confirm_excluded") is True, "Release confirm_excluded must be True")
    check(body.get("e35_excluded") is True, "Release e35_excluded must be True")
    check(body.get("confirm_rows") == 0, "Release confirm_rows must be 0")
    check(body.get("split") == E39_E31_SPLIT, "Release split must be SCREEN")
    check(body.get("repeat_check_summary", {}).get("all_passed") is True, "Release repeat checks did not all pass")

    depths = [int(d) for d in body["depths"]]
    dim = int(body["dimension"])
    specs = expected_array_specs(depths, dim)
    check(int(body.get("array_count_per_chunk", len(specs))) == len(specs), "Release array_count_per_chunk mismatch")

    chunks = body.get("chunks", [])
    check(len(chunks) > 0 and len(chunks) == body["total_chunks"], "Release chunk count mismatch")
    total_views = 0
    total_images = 0
    for expected_idx, entry in enumerate(chunks):
        check(entry.get("chunk_index") == expected_idx, "Release chunk ordering mismatch")
        c_ref, c_body, c_marker = load_phase3_artifact(
            resolved_root,
            protocol,
            entry["ref"],
            KIND_E39_E31_CHUNK,
            expected_schema=SCHEMA_E39_E31_FEATURES,
            expected_dataset=marker["dataset_fingerprint"],
            expected_plan_identity=body["plan_identity"],
            expected_owner=body["model"],
        )
        check(c_body["chunk_index"] == expected_idx, "Chunk index mismatch")
        check(c_body["n_views_per_condition"] == entry["n_views_per_condition"], "Chunk n_views_per_condition mismatch")
        check(c_body["n_images"] == entry["n_images"], "Chunk n_images mismatch")
        check(c_body["rows_sha256"] == entry["rows_sha256"], "Chunk rows_sha256 mismatch")
        check(c_marker["files"]["rows.jsonl"] == entry["rows_sha256"], "Chunk marker rows.jsonl mismatch")
        check(c_body["repeat_check"]["passed"] is True, "Chunk repeat_check failed")
        check(c_body["e35_excluded"] is True and c_body["confirm_excluded"] is True, "Chunk safety flags mismatch")

        c_folder = within(resolved_root, c_ref["path"]).parent
        for fname in (
            "rows.jsonl",
            "token_evidence_image_caption.jsonl",
            "token_evidence_caption_only.jsonl",
            "audits.json",
        ):
            check(fname in c_marker["files"], f"Chunk missing {fname}")
            check(file_sha256(c_folder / fname) == c_marker["files"][fname], f"Chunk {fname} SHA-256 mismatch")

        n_views = int(entry["n_views_per_condition"])
        for key, width in specs.items():
            fname = f"{key}.npy"
            check(fname in c_marker["files"], f"Chunk missing {fname}")
            _verify_npy_array(
                c_folder / fname,
                c_marker["files"][fname],
                n_views,
                width,
                is_attention=("__attention_from_" in key),
            )
        total_views += n_views
        total_images += int(entry["n_images"])

    check(total_views == body["views_per_condition"], "Release views_per_condition sum mismatch")
    check(total_images == body["total_images"], "Release total_images sum mismatch")
    return ref, body, marker


def _parse_ref_arg(value):
    if value is None:
        return None
    text = value.strip()
    if text.startswith("{") or text.startswith("["):
        return json.loads(text)
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--phase1-root", default=None)
    parser.add_argument(
        "--plan",
        default=None,
        help="Optional explicit phase3_gpu_plan {'path', 'sha256'} JSON or path (defaults to CERTIFIED_STAGE_B1_GPU_PLAN)",
    )
    parser.add_argument("--protocol", type=Path, default=REPO / "configs/protocol.json")
    parser.add_argument("--resume-refs", default=None, help="Optional JSON mapping {chunk_index: {'path', 'sha256'}}")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--storage-scope",
        default="persistent-drive",
        choices=("persistent-drive", "local-development"),
    )
    args = parser.parse_args(argv)

    root = phase1_root(args.phase1_root)
    check(root.is_dir(), "PHASE1_ROOT must exist and be mounted before launch")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    plan_ref = _parse_ref_arg(args.plan) if args.plan else None

    if args.preflight_only:
        ctx = preflight_e39_e31_extraction(root, protocol, args.model, plan_ref=plan_ref)
        _default_emit({
            "event": "e39_e31_extract_preflight",
            "status": "passed",
            "model": args.model,
            "depths": ctx["depths"],
            "midpoint_depth": ctx["midpoint_depth"],
            "dimension": ctx["dimension"],
            "total_images": ctx["e39_e31_plan"]["total_images"],
            "views_per_condition": ctx["e39_e31_plan"]["views_per_condition"],
            "total_chunks": ctx["e39_e31_plan"]["total_chunks"],
            "weights_loaded": False,
            "e35_excluded": True,
            "confirm_excluded": True,
        })
        return 0

    producer = stage2_current_producer(scope=args.storage_scope)
    from importlib.metadata import version

    for pkg in ("torch", "torchvision", "transformers", "tokenizers", "safetensors", "huggingface-hub", "Pillow"):
        try:
            producer["software_versions"][pkg] = version(pkg)
        except Exception:
            pass

    resume_refs = _parse_ref_arg(args.resume_refs) if args.resume_refs else None
    result = run_e39_e31_extraction(
        root,
        protocol,
        args.model,
        producer,
        plan_ref=plan_ref,
        resume_refs=resume_refs,
        run_id=args.run_id,
        progress=_default_emit,
    )
    _default_emit(plain(result))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, OSError, ValueError, KeyError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        _default_emit({"status": "blocked", "blockers": [safe_error(exc)]})
        raise SystemExit(2)
