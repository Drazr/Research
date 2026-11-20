"""Phase 3 Stage B2 Worker 2: L4 GPU swapped-image E23 midpoint extraction.

Extracts `SCREEN`-only midpoint `image_caption` features (`last_caption` and
`caption_mean` on the same single forward pass via
`src.stage2.evidence.extract_with_evidence`) for all deduplicated directed
opposite-label swapped pairs enumerated in `plan["e23_plan"]["chunks"]`, plus
the 8 authentic anchor rows in `plan["e23_plan"]["authentic_parity_sample"]`.

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
    ExtractionRequest,
    FeatureSite,
    LANGUAGES,
    ModelInput,
    require_sha256,
)
from src.common.storage import file_sha256, new_run_id, phase1_root, within
from src.models.registry import create_adapter
from src.phase3.interfaces import (
    CERTIFIED_STAGE_B1_GPU_PLAN,
    E23_CHUNK_SIZE_PAIRS,
    KIND_E23_CHUNK,
    KIND_E23_RELEASE,
    MODELS,
    POOLINGS,
    REPO,
    SCHEMA_E23_FEATURES,
    check,
    contract,
    frozen,
    load_phase3_artifact,
    load_phase3_gpu_plan,
    plain,
    publish_phase3,
    require_open_split,
)
from src.stage2.evidence import bridge_hash, extract_with_evidence
from src.stage2.operations import current_producer as stage2_current_producer
from src.stage2.plans import load_job_plan
from src.stage2.readiness import assert_runtime
from src.stage2.schema import artifact, json_bytes, jsonl, read_json

E23_CONDITION = "image_caption"
E23_SPLIT = "SCREEN"
E23_EXPECTED_MIDPOINTS = {"qwen": 18, "gemma": 17}


def _require_numpy():
    try:
        import numpy as np

        return np
    except ImportError as exc:
        raise ContractError("numpy is required for E23 array extraction and verification") from exc


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


def _check_caption_text_sha256(row: Mapping[str, Any], label: str) -> None:
    text = row.get("caption_text")
    check(isinstance(text, str) and bool(text.strip()), f"Empty caption_text in {label}")
    expected = row.get("caption_text_sha256")
    require_sha256(expected)
    actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
    check(actual == expected, f"Caption UTF-8 SHA-256 mismatch in {label}")


def _validate_e23_plan(e23_plan: Any) -> None:
    check(isinstance(e23_plan, Mapping), "Missing e23_plan in phase3_gpu_plan")
    check(
        e23_plan.get("zero_model_scores_in_pairing") is True,
        "e23_plan must certify zero_model_scores_in_pairing=True",
    )
    check(
        e23_plan.get("confirm_rows") == 0,
        f"e23_plan must have confirm_rows == 0; got {e23_plan.get('confirm_rows')!r}",
    )
    require_open_split(e23_plan.get("split"))
    check(
        e23_plan.get("split") == E23_SPLIT,
        f"e23_plan split must be {E23_SPLIT!r}; got {e23_plan.get('split')!r}",
    )

    chunks = e23_plan.get("chunks")
    check(isinstance(chunks, (list, tuple)) and len(chunks) > 0, "e23_plan chunks must be nonempty")
    check(len(chunks) == e23_plan.get("total_chunks"), "e23_plan total_chunks mismatch")

    total_views = 0
    total_pairs = 0
    for idx, chunk in enumerate(chunks):
        check(isinstance(chunk, Mapping), f"Chunk {idx} must be a mapping")
        require_open_split(chunk.get("split"))
        check(chunk.get("split") == E23_SPLIT, f"Chunk {idx} split must be {E23_SPLIT!r}")
        expected_cid = f"E23_SWAPPED_{idx:04d}"
        check(chunk.get("chunk_id") == expected_cid, f"Chunk {idx} chunk_id must be {expected_cid!r}")
        n_pairs = chunk.get("n_pairs")
        check(
            type(n_pairs) is int and 0 < n_pairs <= E23_CHUNK_SIZE_PAIRS,
            f"Chunk {idx} invalid n_pairs={n_pairs!r}",
        )
        rows = chunk.get("rows")
        check(isinstance(rows, (list, tuple)) and len(rows) > 0, f"Chunk {idx} rows must be nonempty")
        check(chunk.get("n_views") == len(rows), f"Chunk {idx} n_views mismatch")
        for r_idx, row in enumerate(rows):
            check(isinstance(row, Mapping), f"Chunk {idx} row {r_idx} must be a mapping")
            if "split" in row:
                require_open_split(row.get("split"))
                check(row.get("split") == E23_SPLIT, f"Chunk {idx} row {r_idx} split must be {E23_SPLIT!r}")
            shown_id = row.get("shown_image_id")
            owner_id = row.get("caption_owner_image_id")
            check(
                isinstance(shown_id, str)
                and isinstance(owner_id, str)
                and shown_id != owner_id,
                f"Chunk {idx} row {r_idx} must have shown_image_id != caption_owner_image_id",
            )
            check(row.get("language") in LANGUAGES, f"Chunk {idx} row {r_idx} invalid language")
            require_sha256(row.get("shown_image_sha256"))
            _check_caption_text_sha256(row, f"chunk {idx} row {r_idx}")
        total_views += len(rows)
        total_pairs += n_pairs

    check(total_views == e23_plan.get("total_swapped_views"), "e23_plan total_swapped_views mismatch")
    check(total_pairs == e23_plan.get("total_unique_directed_pairs"), "e23_plan total_unique_directed_pairs mismatch")

    parity_rows = e23_plan.get("authentic_parity_sample")
    check(
        isinstance(parity_rows, (list, tuple)) and len(parity_rows) == 8,
        "e23_plan authentic_parity_sample must contain exactly 8 rows",
    )
    for p_idx, row in enumerate(parity_rows):
        check(isinstance(row, Mapping), f"authentic_parity_sample row {p_idx} must be a mapping")
        if "split" in row:
            require_open_split(row.get("split"))
            check(row.get("split") == E23_SPLIT, f"authentic_parity_sample row {p_idx} split must be {E23_SPLIT!r}")
        owner_id = row.get("caption_owner_image_id", row.get("image_id"))
        check(
            isinstance(row.get("image_id"), str) and row.get("image_id") == owner_id,
            f"authentic_parity_sample row {p_idx} must satisfy image_id == caption_owner_image_id",
        )
        check(row.get("language") in LANGUAGES, f"authentic_parity_sample row {p_idx} invalid language")
        require_sha256(row.get("image_sha256"))
        _check_caption_text_sha256(row, f"authentic_parity_sample row {p_idx}")


def _load_model_plan_and_readiness(
    root: Path,
    protocol: Mapping[str, Any],
    plan,
    model: str,
    depth: int,
):
    model_spec = plan["models"][model]
    model_plan_ref = plain(model_spec["model_plan_ref"])
    model_plan_path = within(root, model_plan_ref["path"])
    check(file_sha256(model_plan_path) == model_plan_ref["sha256"], f"Stage 2 model_plan checksum mismatch for {model}")
    model_plan = load_job_plan(model_plan_path, root, protocol)
    check(model_plan["kind"] == "stage2_model_plan" and model_plan["model"] == model, "Invalid Stage 2 model_plan")
    check(model_plan["dataset_fingerprint"] == plan["dataset_fingerprint"], "Stage 2 model_plan dataset mismatch")
    check(model_plan["protocol_fingerprint"] == plan["protocol_fingerprint"], "Stage 2 model_plan protocol mismatch")
    check(int(model_plan["adapter"]["midpoint"]) == depth, "Stage 2 model_plan midpoint depth mismatch")

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

    last_key = FeatureSite("decoder", depth, "last_caption").key
    mean_key = FeatureSite("decoder", depth, "caption_mean").key
    for site_key in (last_key, mean_key):
        check(site_key in readiness_body["dimensions"], f"Readiness missing dimension for site: {site_key}")
    dim_last = int(readiness_body["dimensions"][last_key])
    dim_mean = int(readiness_body["dimensions"][mean_key])
    check(dim_last == dim_mean and dim_last > 0, "Midpoint pooling dimensions must match and be positive")
    return model_plan, readiness_body


@contract
def preflight_e23_extraction(
    root,
    protocol,
    model: str,
    *,
    plan_ref=None,
):
    """Authenticate `phase3_gpu_plan` with `bind=False` and enforce pre-GPU E23 gates."""
    require_open_split(E23_SPLIT)
    check(model in MODELS, f"Unsupported MODEL selector {model!r}; expected one of {MODELS}")
    resolved_root = Path(root).resolve()

    effective_plan_ref = plain(plan_ref) if plan_ref is not None else plain(CERTIFIED_STAGE_B1_GPU_PLAN)
    plan = load_phase3_gpu_plan(resolved_root, protocol, effective_plan_ref, bind=False)
    check(plan["confirm_excluded"] is True, "phase3_gpu_plan must enforce confirm_excluded=True")
    check(plan["e35_excluded"] is True, "phase3_gpu_plan must enforce e35_excluded=True (E35 dropped)")

    e23_plan = plan["e23_plan"]
    _validate_e23_plan(e23_plan)

    check(model in plan["models"], f"phase3_gpu_plan missing model spec for {model}")
    depth = int(plan["models"][model]["midpoint_depth"])
    expected_depth = E23_EXPECTED_MIDPOINTS[model]
    check(depth == expected_depth, f"Expected midpoint depth {expected_depth} for {model}; got {depth}")

    model_plan, readiness_body = _load_model_plan_and_readiness(resolved_root, protocol, plan, model, depth)
    return {
        "root": resolved_root,
        "protocol": protocol,
        "plan": plan,
        "e23_plan": e23_plan,
        "model": model,
        "midpoint_depth": depth,
        "model_plan": model_plan,
        "readiness": readiness_body,
    }


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


def _expected_swapped_chunk_rows(chunk: Mapping[str, Any]) -> list:
    expected = []
    for row_index, row in enumerate(chunk["rows"]):
        expected.append({
            "row_index": row_index,
            "shown_image_id": row["shown_image_id"],
            "shown_image_path": row["shown_image_path"],
            "shown_image_sha256": row["shown_image_sha256"],
            "caption_owner_image_id": row["caption_owner_image_id"],
            "language": row["language"],
            "split": E23_SPLIT,
            "caption_id": row["caption_id"],
            "caption_draw_key": int(row.get("caption_draw_key", 0)),
            "caption_text": row["caption_text"],
            "caption_text_sha256": row["caption_text_sha256"],
        })
    return expected


def _expected_authentic_parity_rows(parity_rows) -> list:
    expected = []
    for row_index, row in enumerate(parity_rows):
        expected.append({
            "row_index": row_index,
            "image_id": row["image_id"],
            "image_path": row["image_path"],
            "image_sha256": row["image_sha256"],
            "language": row["language"],
            "split": E23_SPLIT,
            "caption_id": row["caption_id"],
            "caption_draw_key": int(row.get("caption_draw_key", 0)),
            "caption_text": row["caption_text"],
            "caption_text_sha256": row["caption_text_sha256"],
        })
    return expected


def _verify_npy_array(path: Path, expected_sha256: str, n_rows: int, dimension: int):
    np = _require_numpy()
    check(path.is_file(), f"Missing array file: {path.name}")
    check(file_sha256(path) == expected_sha256, f"Array SHA-256 mismatch: {path.name}")
    arr = np.load(path, allow_pickle=False)
    check(
        isinstance(arr, np.ndarray)
        and arr.dtype == np.dtype("<f4")
        and arr.shape == (n_rows, dimension)
        and arr.flags.c_contiguous
        and bool(np.isfinite(arr).all()),
        f"Invalid array dtype/shape/finiteness in {path.name}",
    )
    return arr


def _verify_resumed_chunk(
    *,
    root: Path,
    protocol: Mapping[str, Any],
    plan,
    model: str,
    depth: int,
    dimension: int,
    model_fingerprint: str,
    execution_fingerprint: str,
    chunk_index: int,
    total_chunks: int,
    chunk: Mapping[str, Any],
    reference,
):
    ref, body, marker = load_phase3_artifact(
        root,
        protocol,
        reference,
        KIND_E23_CHUNK,
        expected_schema=SCHEMA_E23_FEATURES,
        expected_dataset=plan["dataset_fingerprint"],
        expected_plan_identity=plan["plan_identity"],
        expected_owner=model,
    )
    check(body.get("experiment") == "E23", "Resumed chunk experiment mismatch")
    check(plain(body.get("plan")) == plain(plan.reference), "Resumed chunk plan mismatch")
    check(body.get("model") == model, "Resumed chunk model mismatch")
    check(body.get("midpoint_depth") == depth, "Resumed chunk midpoint_depth mismatch")
    check(body.get("condition") == E23_CONDITION, "Resumed chunk condition mismatch")
    check(list(body.get("poolings", [])) == list(POOLINGS), "Resumed chunk poolings mismatch")
    check(body.get("split") == E23_SPLIT, "Resumed chunk split mismatch")
    check(body.get("confirm_excluded") is True, "Resumed chunk confirm_excluded must be True")
    check(body.get("e35_excluded") is True, "Resumed chunk e35_excluded must be True")
    check(body.get("chunk_index") == chunk_index, "Resumed chunk_index mismatch")
    check(body.get("chunk_id") == chunk["chunk_id"], "Resumed chunk_id mismatch")
    check(body.get("total_chunks") == total_chunks, "Resumed chunk total_chunks mismatch")
    check(body.get("n_pairs") == chunk["n_pairs"], "Resumed chunk n_pairs mismatch")
    check(body.get("n_views") == len(chunk["rows"]), "Resumed chunk n_views mismatch")
    check(body.get("dimension") == dimension, "Resumed chunk dimension mismatch")
    check(body.get("model_fingerprint") == model_fingerprint, "Resumed chunk model_fingerprint mismatch")
    check(body.get("execution_fingerprint") == execution_fingerprint, "Resumed chunk execution_fingerprint mismatch")

    rc = body.get("repeat_check")
    check(isinstance(rc, Mapping) and rc.get("passed") is True, "Resumed chunk repeat_check did not pass")

    folder = within(root, ref["path"]).parent
    expected_rows = _expected_swapped_chunk_rows(chunk)
    expected_rows_sha = hashlib.sha256(jsonl(expected_rows)).hexdigest()

    check("rows.jsonl" in marker["files"], "Resumed chunk missing rows.jsonl")
    check(marker["files"]["rows.jsonl"] == expected_rows_sha, "Resumed chunk rows.jsonl SHA-256 mismatch")
    check(file_sha256(folder / "rows.jsonl") == expected_rows_sha, "Resumed chunk rows.jsonl on-disk mismatch")
    check(body.get("rows_sha256") == expected_rows_sha, "Resumed chunk body rows_sha256 mismatch")
    check("token_evidence.jsonl" in marker["files"], "Resumed chunk missing token_evidence.jsonl")
    check("audits.json" in marker["files"], "Resumed chunk missing audits.json")

    for name in ("last_caption.npy", "caption_mean.npy"):
        check(name in marker["files"], f"Resumed chunk missing {name}")
        _verify_npy_array(folder / name, marker["files"][name], len(expected_rows), dimension)

    return plain(ref), body, expected_rows_sha


def _verify_adapter_contract(adapter, model_plan, readiness, model: str, depth: int):
    assert_runtime(adapter, model_plan)
    check(adapter.config["precision"] == "bfloat16", "E23 extraction requires bfloat16 precision")
    check(adapter.config["attention_backend"] == "eager", "E23 extraction requires eager attention_backend")

    desc = adapter.describe()
    exec_desc = adapter.describe_execution()
    check(
        desc["model_fingerprint"] == readiness["model_config"]["config_fingerprint"],
        "Adapter model_fingerprint differs from Stage 2 model_readiness",
    )
    check(
        int(desc["midpoint"]) == depth,
        f"Adapter midpoint {desc['midpoint']} differs from expected depth {depth}",
    )
    check(
        exec_desc["execution_fingerprint"] == readiness["execution_identity"]["execution_fingerprint"],
        "Adapter execution_fingerprint differs from Stage 2 model_readiness",
    )
    check(
        plain(exec_desc["site_specs"]) == plain(readiness["execution_identity"]["site_specs"]),
        "Adapter site_specs differ from Stage 2 model_readiness",
    )

    last_site = FeatureSite("decoder", depth, "last_caption")
    mean_site = FeatureSite("decoder", depth, "caption_mean")
    for site in (last_site, mean_site):
        check(site.key in desc["supported_sites"], f"Adapter missing required midpoint site: {site.key}")
        check(site.key in readiness["dimensions"], f"Readiness missing dimension for site: {site.key}")
    dim_last = int(readiness["dimensions"][last_site.key])
    dim_mean = int(readiness["dimensions"][mean_site.key])
    check(dim_last == dim_mean and dim_last > 0, "Midpoint pooling dimensions must match and be positive")
    return last_site, mean_site, dim_last, desc, exec_desc


def _build_swapped_model_input(root: Path, row: Mapping[str, Any]) -> ModelInput:
    require_open_split(E23_SPLIT)
    if "split" in row:
        require_open_split(row["split"])
        check(row["split"] == E23_SPLIT, f"Swapped row split must be {E23_SPLIT!r}")
    check(
        row["shown_image_id"] != row["caption_owner_image_id"],
        "Swapped row must satisfy shown_image_id != caption_owner_image_id",
    )
    img_file = within(root, row["shown_image_path"])
    check(img_file.is_file(), f"Missing pinned SCREEN shown image: {row['shown_image_path']}")
    check(
        file_sha256(img_file) == row["shown_image_sha256"],
        f"Pinned SCREEN shown image SHA-256 mismatch: {row['shown_image_path']}",
    )
    _check_caption_text_sha256(row, f"swapped {row['shown_image_id']}/{row['language']}")
    return ModelInput(
        image_id=row["shown_image_id"],
        image_path=row["shown_image_path"],
        image_sha256=row["shown_image_sha256"],
        language=row["language"],
        caption_id=row["caption_id"],
        caption_text=row["caption_text"],
        split=E23_SPLIT,
    )


def _build_authentic_model_input(root: Path, row: Mapping[str, Any]) -> ModelInput:
    require_open_split(E23_SPLIT)
    if "split" in row:
        require_open_split(row["split"])
        check(row["split"] == E23_SPLIT, f"Authentic parity row split must be {E23_SPLIT!r}")
    owner_id = row.get("caption_owner_image_id", row["image_id"])
    check(row["image_id"] == owner_id, "Authentic parity row must satisfy image_id == caption_owner_image_id")
    img_file = within(root, row["image_path"])
    check(img_file.is_file(), f"Missing pinned SCREEN authentic image: {row['image_path']}")
    check(
        file_sha256(img_file) == row["image_sha256"],
        f"Pinned SCREEN authentic image SHA-256 mismatch: {row['image_path']}",
    )
    _check_caption_text_sha256(row, f"authentic {row['image_id']}/{row['language']}")
    return ModelInput(
        image_id=row["image_id"],
        image_path=row["image_path"],
        image_sha256=row["image_sha256"],
        language=row["language"],
        caption_id=row["caption_id"],
        caption_text=row["caption_text"],
        split=E23_SPLIT,
    )


def _extract_model_input(
    *,
    adapter,
    model_input: ModelInput,
    last_site: FeatureSite,
    mean_site: FeatureSite,
    dimension: int,
):
    np = _require_numpy()
    request = ExtractionRequest(
        inputs=[model_input],
        sites=[last_site, mean_site],
        condition=E23_CONDITION,
        purpose="preliminary_features",
    )
    result = extract_with_evidence(adapter, request)
    vec_last = np.asarray(result.features.vectors[last_site.key][0], dtype="<f4")
    vec_mean = np.asarray(result.features.vectors[mean_site.key][0], dtype="<f4")
    check(
        vec_last.shape == (dimension,) and bool(np.isfinite(vec_last).all()),
        f"Invalid/nonfinite last_caption vector for {model_input.image_id}/{model_input.language}",
    )
    check(
        vec_mean.shape == (dimension,) and bool(np.isfinite(vec_mean).all()),
        f"Invalid/nonfinite caption_mean vector for {model_input.image_id}/{model_input.language}",
    )
    return vec_last, vec_mean, dict(result.evidence[0])


def _run_repeat_check(
    *,
    rows,
    build_input_fn,
    root: Path,
    adapter,
    last_site: FeatureSite,
    mean_site: FeatureSite,
    dimension: int,
    last_vecs,
    mean_vecs,
    atol: float,
    rtol: float,
    label: str,
):
    np = _require_numpy()
    n_rows = len(rows)
    repeat_indices = sorted({0, n_rows - 1})
    repeat_details = {}
    max_diff_last = 0.0
    max_diff_mean = 0.0
    repeat_passed = True

    for ridx in repeat_indices:
        model_input = build_input_fn(root, rows[ridx])
        rep_last, rep_mean, _ = _extract_model_input(
            adapter=adapter,
            model_input=model_input,
            last_site=last_site,
            mean_site=mean_site,
            dimension=dimension,
        )
        diff_last = float(np.max(np.abs(last_vecs[ridx] - rep_last)))
        diff_mean = float(np.max(np.abs(mean_vecs[ridx] - rep_mean)))
        ok_last = bool(np.isfinite(diff_last) and np.allclose(last_vecs[ridx], rep_last, rtol=rtol, atol=atol))
        ok_mean = bool(np.isfinite(diff_mean) and np.allclose(mean_vecs[ridx], rep_mean, rtol=rtol, atol=atol))
        max_diff_last = max(max_diff_last, diff_last)
        max_diff_mean = max(max_diff_mean, diff_mean)
        if not (ok_last and ok_mean):
            repeat_passed = False
        repeat_details[str(ridx)] = {
            "max_abs_diff_last_caption": diff_last,
            "max_abs_diff_caption_mean": diff_mean,
            "passed": ok_last and ok_mean,
        }

    check(
        repeat_passed,
        f"Repeat determinism check failed for {label}: "
        f"max_abs_diff_last={max_diff_last}, max_abs_diff_mean={max_diff_mean}, atol={atol}, rtol={rtol}",
    )
    repeat_check = {
        "first_index": 0,
        "last_index": n_rows - 1,
        "repeat_indices": repeat_indices,
        "max_abs_diff_last_caption": max_diff_last,
        "max_abs_diff_caption_mean": max_diff_mean,
        "passed": True,
    }
    return repeat_check, repeat_details


@contract
def run_e23_extraction(
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
    """Execute Phase 3 Stage B2 Worker 2 E23 extraction for one model (`qwen` or `gemma`)."""
    np = _require_numpy()
    emit = progress if progress is not None else _default_emit
    t_read_hash = time.monotonic()
    ctx = preflight_e23_extraction(root, protocol, model, plan_ref=plan_ref)
    read_hash_s = time.monotonic() - t_read_hash

    resolved_root = ctx["root"]
    plan = ctx["plan"]
    e23_plan = ctx["e23_plan"]
    depth = ctx["midpoint_depth"]
    model_plan = ctx["model_plan"]
    readiness = ctx["readiness"]
    chunks = e23_plan["chunks"]
    total_chunks = len(chunks)
    parity_rows = e23_plan["authentic_parity_sample"]

    last_site = FeatureSite("decoder", depth, "last_caption")
    mean_site = FeatureSite("decoder", depth, "caption_mean")
    dimension = int(readiness["dimensions"][last_site.key])
    model_fingerprint = model_plan["model_config"]["config_fingerprint"]
    execution_fingerprint = model_plan["execution_identity"]["execution_fingerprint"]
    site_specs = {
        last_site.key: plain(readiness["execution_identity"]["site_specs"][last_site.key]),
        mean_site.key: plain(readiness["execution_identity"]["site_specs"][mean_site.key]),
    }
    bridge_sha = bridge_hash()
    atol = float(model_plan["model_config"].get("repeat_atol", 0.0))
    rtol = float(model_plan["model_config"].get("repeat_rtol", 0.0))

    emit({
        "event": "e23_extract_start",
        "unit": f"{model}:preflight",
        "completed": 0,
        "total": total_chunks,
        "read_hash_s": round(read_hash_s, 6),
        "compute_s": 0.0,
        "write_s": 0.0,
        "peak_rss_bytes": _peak_rss_bytes(),
        "model": model,
        "midpoint_depth": depth,
        "total_unique_directed_pairs": e23_plan["total_unique_directed_pairs"],
        "total_swapped_views": e23_plan["total_swapped_views"],
        "total_chunks": total_chunks,
        "authentic_parity_rows": len(parity_rows),
    })

    if not resume_refs:
        normalized_resume = {}
    else:
        normalized_resume = _normalize_resume_refs(resume_refs, total_chunks)
    completed_chunks = {}
    t_res_start = time.monotonic()
    for chunk_index, ref in sorted(normalized_resume.items()):
        verified_ref, chunk_body, rows_sha = _verify_resumed_chunk(
            root=resolved_root,
            protocol=protocol,
            plan=plan,
            model=model,
            depth=depth,
            dimension=dimension,
            model_fingerprint=model_fingerprint,
            execution_fingerprint=execution_fingerprint,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            chunk=chunks[chunk_index],
            reference=ref,
        )
        completed_chunks[chunk_index] = {
            "chunk_index": chunk_index,
            "chunk_id": chunks[chunk_index]["chunk_id"],
            "n_pairs": chunks[chunk_index]["n_pairs"],
            "n_views": len(chunks[chunk_index]["rows"]),
            "ref": verified_ref,
            "rows_sha256": rows_sha,
            "repeat_check": plain(chunk_body["repeat_check"]),
        }
    read_hash_s += time.monotonic() - t_res_start

    adapter = None
    batch_run_id = run_id or new_run_id()
    compute_s = 0.0
    write_s = 0.0

    # Ensure adapter is instantiated because authentic_parity_sample always runs (even if all chunks were resumed).
    if adapter_factory is None:
        adapter = create_adapter(model, plain(model_plan["model_config"]), resolved_root)
    else:
        adapter = adapter_factory(model, plain(model_plan["model_config"]), resolved_root)
    _, _, verified_dim, _, _ = _verify_adapter_contract(adapter, model_plan, readiness, model, depth)
    check(verified_dim == dimension, "Verified adapter dimension mismatch")

    for chunk_index in range(total_chunks):
        if chunk_index in completed_chunks:
            emit({
                "event": "e23_chunk_resumed",
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
        expected_rows = _expected_swapped_chunk_rows(chunk)

        t_comp = time.monotonic()
        last_vecs = []
        mean_vecs = []
        evidences = []
        for row in rows:
            model_input = _build_swapped_model_input(resolved_root, row)
            v_last, v_mean, ev = _extract_model_input(
                adapter=adapter,
                model_input=model_input,
                last_site=last_site,
                mean_site=mean_site,
                dimension=dimension,
            )
            last_vecs.append(v_last)
            mean_vecs.append(v_mean)
            evidences.append(ev)

        repeat_check, repeat_details = _run_repeat_check(
            rows=rows,
            build_input_fn=_build_swapped_model_input,
            root=resolved_root,
            adapter=adapter,
            last_site=last_site,
            mean_site=mean_site,
            dimension=dimension,
            last_vecs=last_vecs,
            mean_vecs=mean_vecs,
            atol=atol,
            rtol=rtol,
            label=f"{model} chunk {chunk_index}",
        )
        compute_s += time.monotonic() - t_comp

        last_arr = np.ascontiguousarray(np.stack(last_vecs, axis=0), dtype="<f4")
        mean_arr = np.ascontiguousarray(np.stack(mean_vecs, axis=0), dtype="<f4")
        check(
            last_arr.shape == (n_views, dimension) and mean_arr.shape == (n_views, dimension),
            "Chunk array shape mismatch",
        )

        rows_bytes = jsonl(expected_rows)
        rows_sha = hashlib.sha256(rows_bytes).hexdigest()
        audits_obj = {
            "repeat_check": repeat_check,
            "repeat_details": repeat_details,
            "atol": atol,
            "rtol": rtol,
            "evidence_bridge_sha256": bridge_sha,
        }
        chunk_body = {
            "schema": SCHEMA_E23_FEATURES,
            "experiment": "E23",
            "plan": plain(plan.reference),
            "plan_identity": plan["plan_identity"],
            "model": model,
            "midpoint_depth": depth,
            "condition": E23_CONDITION,
            "poolings": list(POOLINGS),
            "split": E23_SPLIT,
            "chunk_index": chunk_index,
            "chunk_id": chunk["chunk_id"],
            "total_chunks": total_chunks,
            "n_pairs": chunk["n_pairs"],
            "n_views": n_views,
            "dimension": dimension,
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
            f"phase3/features/e23/{model}/{batch_run_id}",
            KIND_E23_CHUNK,
            chunk_body,
            producer,
            arrays={"last_caption": last_arr, "caption_mean": mean_arr},
            files={
                "rows.jsonl": rows_bytes,
                "token_evidence.jsonl": jsonl(evidences),
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
            "n_pairs": chunk["n_pairs"],
            "n_views": n_views,
            "ref": plain(chunk_ref),
            "rows_sha256": rows_sha,
            "repeat_check": repeat_check,
        }
        emit({
            "event": "e23_chunk_published",
            "unit": f"{model}:{chunk['chunk_id']}",
            "completed": chunk_index + 1,
            "total": total_chunks,
            "read_hash_s": round(read_hash_s, 6),
            "compute_s": round(compute_s, 6),
            "write_s": round(write_s, 6),
            "peak_rss_bytes": _peak_rss_bytes(),
            "chunk_index": chunk_index,
            "n_views": n_views,
            "ref": plain(chunk_ref),
        })

    # Extract authentic parity sample (8 rows).
    t_parity_comp = time.monotonic()
    parity_last_vecs = []
    parity_mean_vecs = []
    parity_evidences = []
    for row in parity_rows:
        model_input = _build_authentic_model_input(resolved_root, row)
        v_last, v_mean, ev = _extract_model_input(
            adapter=adapter,
            model_input=model_input,
            last_site=last_site,
            mean_site=mean_site,
            dimension=dimension,
        )
        parity_last_vecs.append(v_last)
        parity_mean_vecs.append(v_mean)
        parity_evidences.append(ev)

    parity_repeat_check, parity_repeat_details = _run_repeat_check(
        rows=parity_rows,
        build_input_fn=_build_authentic_model_input,
        root=resolved_root,
        adapter=adapter,
        last_site=last_site,
        mean_site=mean_site,
        dimension=dimension,
        last_vecs=parity_last_vecs,
        mean_vecs=parity_mean_vecs,
        atol=atol,
        rtol=rtol,
        label=f"{model} authentic_parity_sample",
    )
    compute_s += time.monotonic() - t_parity_comp

    n_parity = len(parity_rows)
    parity_last_arr = np.ascontiguousarray(np.stack(parity_last_vecs, axis=0), dtype="<f4")
    parity_mean_arr = np.ascontiguousarray(np.stack(parity_mean_vecs, axis=0), dtype="<f4")
    check(
        parity_last_arr.shape == (n_parity, dimension) and parity_mean_arr.shape == (n_parity, dimension),
        "Authentic parity array shape mismatch",
    )

    expected_parity_rows = _expected_authentic_parity_rows(parity_rows)
    parity_rows_bytes = jsonl(expected_parity_rows)
    parity_rows_sha = hashlib.sha256(parity_rows_bytes).hexdigest()

    ordered_chunks = [completed_chunks[i] for i in range(total_chunks)]
    repeat_summary = {
        "chunks_checked": total_chunks,
        "authentic_parity_checked": True,
        "all_passed": (
            all(c["repeat_check"]["passed"] for c in ordered_chunks)
            and bool(parity_repeat_check["passed"])
        ),
        "max_abs_diff_last_caption": max(
            [c["repeat_check"]["max_abs_diff_last_caption"] for c in ordered_chunks]
            + [parity_repeat_check["max_abs_diff_last_caption"]]
        ),
        "max_abs_diff_caption_mean": max(
            [c["repeat_check"]["max_abs_diff_caption_mean"] for c in ordered_chunks]
            + [parity_repeat_check["max_abs_diff_caption_mean"]]
        ),
    }
    release_chunks = [
        {
            "chunk_index": c["chunk_index"],
            "chunk_id": c["chunk_id"],
            "n_pairs": c["n_pairs"],
            "n_views": c["n_views"],
            "ref": c["ref"],
            "rows_sha256": c["rows_sha256"],
        }
        for c in ordered_chunks
    ]
    release_audits_obj = {
        "authentic_parity_repeat_check": parity_repeat_check,
        "authentic_parity_repeat_details": parity_repeat_details,
        "repeat_check_summary": repeat_summary,
        "atol": atol,
        "rtol": rtol,
        "evidence_bridge_sha256": bridge_sha,
    }
    release_body = {
        "schema": SCHEMA_E23_FEATURES,
        "experiment": "E23",
        "plan": plain(plan.reference),
        "plan_identity": plan["plan_identity"],
        "model": model,
        "midpoint_depth": depth,
        "condition": E23_CONDITION,
        "poolings": list(POOLINGS),
        "split": E23_SPLIT,
        "zero_model_scores_in_pairing": True,
        "confirm_rows": 0,
        "total_unique_directed_pairs": e23_plan["total_unique_directed_pairs"],
        "total_swapped_views": e23_plan["total_swapped_views"],
        "total_chunks": total_chunks,
        "authentic_parity_count": n_parity,
        "authentic_parity_rows_sha256": parity_rows_sha,
        "authentic_parity_repeat_check": parity_repeat_check,
        "dimension": dimension,
        "model_fingerprint": model_fingerprint,
        "execution_fingerprint": execution_fingerprint,
        "site_specs": site_specs,
        "evidence_bridge_sha256": bridge_sha,
        "chunks": release_chunks,
        "repeat_check_summary": repeat_summary,
        "e35_excluded": True,
        "confirm_excluded": True,
    }

    release_run_id = f"{batch_run_id}_release"
    t_rel = time.monotonic()
    release_ref = publish_phase3(
        resolved_root,
        protocol,
        plan["dataset_fingerprint"],
        f"phase3/features/e23/{model}",
        KIND_E23_RELEASE,
        release_body,
        producer,
        arrays={
            "authentic_parity_last_caption": parity_last_arr,
            "authentic_parity_caption_mean": parity_mean_arr,
        },
        files={
            "authentic_parity_rows.jsonl": parity_rows_bytes,
            "authentic_parity_token_evidence.jsonl": jsonl(parity_evidences),
            "audits.json": json_bytes(release_audits_obj),
        },
        upstream=[plain(plan.reference)] + [c["ref"] for c in release_chunks],
        model_fingerprint=model_fingerprint,
        run_id=release_run_id,
    )
    write_s += time.monotonic() - t_rel

    emit({
        "event": "e23_release_published",
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
        "midpoint_depth": depth,
        "total_unique_directed_pairs": e23_plan["total_unique_directed_pairs"],
        "total_swapped_views": e23_plan["total_swapped_views"],
        "total_chunks": total_chunks,
        "authentic_parity_count": n_parity,
        "release_ref": plain(release_ref),
        "chunks": release_chunks,
        "e35_excluded": True,
        "confirm_excluded": True,
    })


@contract
def verify_e23_release(
    root,
    protocol,
    release_ref,
    *,
    expected_plan_identity: Optional[str] = None,
    expected_model: Optional[str] = None,
):
    """Authenticate a published `phase3_e23_feature_release` and all referenced chunks."""
    resolved_root = Path(root).resolve()
    ref, body, marker = load_phase3_artifact(
        resolved_root,
        protocol,
        release_ref,
        KIND_E23_RELEASE,
        expected_schema=SCHEMA_E23_FEATURES,
        expected_plan_identity=expected_plan_identity,
        expected_owner=expected_model,
    )
    check(body.get("experiment") == "E23", "Release experiment mismatch")
    check(body.get("confirm_excluded") is True, "Release confirm_excluded must be True")
    check(body.get("e35_excluded") is True, "Release e35_excluded must be True")
    check(body.get("confirm_rows") == 0, "Release confirm_rows must be 0")
    check(body.get("zero_model_scores_in_pairing") is True, "Release zero_model_scores_in_pairing must be True")
    check(body.get("split") == E23_SPLIT, "Release split must be SCREEN")
    check(body.get("repeat_check_summary", {}).get("all_passed") is True, "Release repeat checks did not all pass")
    check(
        body.get("authentic_parity_repeat_check", {}).get("passed") is True,
        "Release authentic parity repeat check did not pass",
    )

    rel_folder = within(resolved_root, ref["path"]).parent
    n_parity = int(body["authentic_parity_count"])
    dim = int(body["dimension"])
    check(n_parity == 8, f"Expected 8 authentic parity rows; got {n_parity}")
    check("authentic_parity_rows.jsonl" in marker["files"], "Release missing authentic_parity_rows.jsonl")
    check(
        marker["files"]["authentic_parity_rows.jsonl"] == body["authentic_parity_rows_sha256"],
        "Release authentic_parity_rows.jsonl SHA-256 mismatch",
    )
    check(
        file_sha256(rel_folder / "authentic_parity_rows.jsonl") == body["authentic_parity_rows_sha256"],
        "Release authentic_parity_rows.jsonl on-disk mismatch",
    )
    check("authentic_parity_token_evidence.jsonl" in marker["files"], "Release missing authentic_parity_token_evidence.jsonl")
    for name in ("authentic_parity_last_caption.npy", "authentic_parity_caption_mean.npy"):
        check(name in marker["files"], f"Release missing {name}")
        _verify_npy_array(rel_folder / name, marker["files"][name], n_parity, dim)

    chunks = body.get("chunks", [])
    check(len(chunks) > 0 and len(chunks) == body["total_chunks"], "Release chunk count mismatch")
    total_views = 0
    total_pairs = 0
    for expected_idx, entry in enumerate(chunks):
        check(entry.get("chunk_index") == expected_idx, "Release chunk ordering mismatch")
        c_ref, c_body, c_marker = load_phase3_artifact(
            resolved_root,
            protocol,
            entry["ref"],
            KIND_E23_CHUNK,
            expected_schema=SCHEMA_E23_FEATURES,
            expected_dataset=marker["dataset_fingerprint"],
            expected_plan_identity=body["plan_identity"],
            expected_owner=body["model"],
        )
        check(c_body["chunk_index"] == expected_idx, "Chunk index mismatch")
        check(c_body["n_views"] == entry["n_views"], "Chunk n_views mismatch")
        check(c_body["n_pairs"] == entry["n_pairs"], "Chunk n_pairs mismatch")
        check(c_body["rows_sha256"] == entry["rows_sha256"], "Chunk rows_sha256 mismatch")
        check(c_marker["files"]["rows.jsonl"] == entry["rows_sha256"], "Chunk marker rows.jsonl mismatch")
        check(c_body["repeat_check"]["passed"] is True, "Chunk repeat_check failed")
        check(c_body.get("e35_excluded") is True, "Chunk e35_excluded must be True")
        check(c_body.get("confirm_excluded") is True, "Chunk confirm_excluded must be True")
        c_folder = within(resolved_root, c_ref["path"]).parent
        for name in ("last_caption.npy", "caption_mean.npy"):
            check(name in c_marker["files"], f"Chunk missing {name}")
            _verify_npy_array(c_folder / name, c_marker["files"][name], int(entry["n_views"]), dim)
        total_views += int(entry["n_views"])
        total_pairs += int(entry["n_pairs"])

    check(total_views == body["total_swapped_views"], "Release total_swapped_views sum mismatch")
    check(total_pairs == body["total_unique_directed_pairs"], "Release total_unique_directed_pairs sum mismatch")
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
        ctx = preflight_e23_extraction(root, protocol, args.model, plan_ref=plan_ref)
        _default_emit({
            "event": "e23_extract_preflight",
            "status": "passed",
            "model": args.model,
            "midpoint_depth": ctx["midpoint_depth"],
            "total_unique_directed_pairs": ctx["e23_plan"]["total_unique_directed_pairs"],
            "total_swapped_views": ctx["e23_plan"]["total_swapped_views"],
            "total_chunks": ctx["e23_plan"]["total_chunks"],
            "authentic_parity_count": len(ctx["e23_plan"]["authentic_parity_sample"]),
            "weights_loaded": False,
            "e35_excluded": True,
            "confirm_excluded": True,
        })
        return 0

    producer = stage2_current_producer(scope=args.storage_scope, run_id=args.run_id)
    from importlib.metadata import version

    for pkg in ("torch", "torchvision", "transformers", "tokenizers", "safetensors", "huggingface-hub", "Pillow"):
        try:
            producer["software_versions"][pkg] = version(pkg)
        except Exception:
            pass

    resume_refs = _parse_ref_arg(args.resume_refs) if args.resume_refs else None
    result = run_e23_extraction(
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
