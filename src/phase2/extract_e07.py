"""Phase 2 Stage 2 Branch C: L4 GPU audited unique changed-view E07 midpoint extraction.

Extracts ONLY the deduplicated unique genuinely-changed SCREEN `image_caption`
views enumerated in `phase2_e07_audit["extraction_plan"]["chunks"]` at the
model's fixed midpoint depth (`qwen` depth 18, `gemma` depth 17), capturing BOTH
`last_caption` and `caption_mean` on the same single forward via
`src.stage2.evidence.extract_with_evidence`.

CONFIRM remains strictly closed and rejected before any IO or model loading.
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
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np

from src.common.contracts import (
    ContractError,
    ExtractionRequest,
    FeatureSite,
    LANGUAGES,
    ModelInput,
    require_sha256,
)
from src.common.storage import file_sha256, new_run_id, phase1_root, verify, within
from src.models.registry import create_adapter
from src.phase2.e07_audit import load_e07_audit
from src.phase2.interfaces import (
    KIND_E07_CHUNK,
    KIND_E07_RELEASE,
    MODELS,
    POOLINGS,
    REPO,
    SCHEMA_E07_FEATURES,
    check,
    contract,
    frozen,
    load_phase2_artifact,
    normalize_ref,
    plain,
    publish_phase2,
    require_open_split,
    stage1_caption_paths,
)
from src.phase2.plan import load_phase2_plan
from src.stage2.evidence import bridge_hash, extract_with_evidence
from src.stage2.operations import current_producer as stage2_current_producer
from src.stage2.plans import load_job_plan
from src.stage2.readiness import assert_runtime
from src.stage2.schema import artifact, json_bytes, jsonl, read_json

E07_ALLOWED_VERDICTS = ("passed_full", "passed_restricted")
E07_CONDITION = "image_caption"
E07_SPLIT = "SCREEN"


def safe_error(exc: BaseException) -> str:
    """Redact any Hugging Face token from exception messages."""
    message = str(exc)
    token = os.environ.get("HF_TOKEN")
    if token:
        message = message.replace(token, "[REDACTED]")
    return re.sub(r"hf_[A-Za-z0-9]+", "[REDACTED]", message)


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
    print(json.dumps(dict(event), sort_keys=True), flush=True)


def _load_screen_images(root: Path, binding) -> dict:
    """Load verified SCREEN-only (image_path, image_sha256) from Stage 1 records/splits."""
    caption_paths = stage1_caption_paths(binding)
    files = caption_paths["files"]

    splits_entry = files["splits.json"]
    splits_path = within(root, splits_entry["path"])
    check(file_sha256(splits_path) == splits_entry["sha256"], "Stage 1 splits.json checksum mismatch")
    splits_doc = json.loads(splits_path.read_text(encoding="utf-8"))
    if E07_SPLIT in splits_doc and isinstance(splits_doc[E07_SPLIT], list):
        screen_id_set = set(splits_doc[E07_SPLIT])
    else:
        screen_id_set = {iid for iid, sp in splits_doc.items() if sp == E07_SPLIT}

    records_entry = files["records.jsonl"]
    records_path = within(root, records_entry["path"])
    check(file_sha256(records_path) == records_entry["sha256"], "Stage 1 records.jsonl checksum mismatch")

    screen_images = {}
    with records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            split = rec.get("split")
            if split == "CONFIRM":
                continue
            if split != E07_SPLIT:
                continue
            iid = rec["image_id"]
            check(iid in screen_id_set, f"Split mismatch for SCREEN image {iid}")
            pair = (rec["image_path"], rec["image_sha256"])
            if iid in screen_images:
                check(screen_images[iid] == pair, f"Inconsistent image metadata for {iid}")
            else:
                screen_images[iid] = pair
    return screen_images


def _validate_extraction_plan(ep: Any, screen_images: Mapping[str, tuple]) -> None:
    check(isinstance(ep, Mapping), "Missing extraction_plan in phase2_e07_audit")
    check(ep.get("split") == E07_SPLIT, f"E07 extraction_plan must be SCREEN-only; got {ep.get('split')!r}")
    check(ep.get("confirm_rows") == 0, f"E07 extraction_plan must have confirm_rows == 0; got {ep.get('confirm_rows')!r}")
    check(ep.get("condition") == E07_CONDITION, f"E07 extraction_plan condition must be {E07_CONDITION!r}")
    check(list(ep.get("poolings", [])) == list(POOLINGS), f"E07 extraction_plan poolings must be {list(POOLINGS)!r}")
    check(ep.get("same_forward_both_poolings") is True, "E07 extraction_plan must require same_forward_both_poolings=True")

    chunks = ep.get("chunks")
    check(isinstance(chunks, (list, tuple)), "E07 extraction_plan chunks must be a list")
    check(len(chunks) == ep.get("total_chunks"), "E07 extraction_plan total_chunks mismatch")
    check(len(chunks) > 0, "E07 extraction_plan must contain at least one chunk")

    total_rows = 0
    for idx, chunk in enumerate(chunks):
        require_open_split(chunk.get("split"))
        check(chunk.get("split") == E07_SPLIT, f"Chunk {idx} split must be SCREEN")
        check(chunk.get("language") in LANGUAGES, f"Chunk {idx} has invalid language: {chunk.get('language')!r}")
        rows = chunk.get("rows")
        check(isinstance(rows, (list, tuple)) and 0 < len(rows) <= 32, f"Chunk {idx} invalid row count")
        check(chunk.get("n_rows") == len(rows), f"Chunk {idx} n_rows mismatch")
        for row in rows:
            require_open_split(row.get("split"))
            check(row.get("split") == E07_SPLIT, f"Row in chunk {idx} is not SCREEN")
            check(row.get("language") == chunk["language"], f"Row language mismatch in chunk {idx}")
            text_sha = hashlib.sha256(row["caption_text"].encode("utf-8")).hexdigest()
            check(text_sha == row.get("caption_text_sha256"), f"Row caption_text_sha256 mismatch in chunk {idx}")
            check(row["image_id"] in screen_images, f"Row image_id {row['image_id']!r} not in verified SCREEN images")
        total_rows += len(rows)

    check(total_rows == ep.get("total_unique_views"), "E07 extraction_plan total_unique_views mismatch")


def _load_model_plan_and_readiness(root: Path, protocol: Mapping[str, Any], binding, model: str, depth: int):
    model_plan_ref = binding.model_plan_refs[model]
    model_plan_path = within(root, model_plan_ref["path"])
    check(file_sha256(model_plan_path) == model_plan_ref["sha256"], f"Stage 2 model_plan checksum mismatch for {model}")
    model_plan = load_job_plan(model_plan_path, root, protocol)
    check(model_plan["kind"] == "stage2_model_plan" and model_plan["model"] == model, "Invalid Stage 2 model_plan")
    check(model_plan["dataset_fingerprint"] == binding.dataset_fingerprint, "Stage 2 model_plan dataset mismatch")
    check(model_plan["protocol_fingerprint"] == binding.protocol_fingerprint, "Stage 2 model_plan protocol mismatch")
    check(int(model_plan["adapter"]["midpoint"]) == depth, "Stage 2 model_plan midpoint depth mismatch")

    models_json_path = REPO / "configs/models.json"
    if models_json_path.is_file():
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
    return model_plan, readiness_body


@contract
def preflight_e07_extraction(
    root,
    protocol,
    plan_ref,
    e07_audit_ref,
    model: str,
):
    """Authenticate Stage 1 plan & E07 audit and enforce pre-GPU gates before loading weights."""
    check(model in MODELS, f"Unsupported MODEL selector {model!r}; expected one of {MODELS}")
    resolved_root = Path(root).resolve()

    plan = load_phase2_plan(resolved_root, protocol, plan_ref)
    check(plan["confirm_excluded"] is True, "phase2_plan must exclude CONFIRM")

    audit_ref, audit_body, _ = load_e07_audit(
        resolved_root,
        protocol,
        e07_audit_ref,
        expected_dataset=plan["dataset_fingerprint"],
        expected_lineage_identity=plan["lineage_identity"],
    )
    if audit_body.get("plan_identity") is not None:
        check(audit_body["plan_identity"] == plan["plan_identity"], "E07 audit plan_identity mismatch")
    if plan["e07_audit"] is not None:
        check(plain(audit_ref) == plain(plan["e07_audit"]), "E07 audit ref differs from phase2_plan.e07_audit")

    check(audit_body.get("draw0_verified") is True, "E07 audit draw0_verified must be True")
    check(audit_body.get("confirm_excluded") is True, "E07 audit confirm_excluded must be True")
    verdict = audit_body.get("verdict")
    check(
        verdict in E07_ALLOWED_VERDICTS,
        f"E07 extraction refused: audit verdict is {verdict!r} (must be in {E07_ALLOWED_VERDICTS})",
    )

    suite_ref = plain(plan["stage2_suite"])
    suite_path = within(resolved_root, suite_ref["path"])
    check(file_sha256(suite_path) == suite_ref["sha256"], "Stage 2 suite checksum mismatch")
    sp, _ = artifact(suite_path, resolved_root, protocol, "stage2_suite_index")
    suite_body = read_json(sp.parent / "report.json")
    binding = SimpleNamespace(
        root=resolved_root,
        dataset_ref=plain(plan["stage1_dataset"]),
        dataset_fingerprint=str(plan["dataset_fingerprint"]),
        protocol_fingerprint=str(plan["protocol_fingerprint"]),
        model_plan_refs=plain(suite_body["model_plans"]),
    )
    depth = int(plan["midpoints"][model])

    screen_images = _load_screen_images(resolved_root, binding)
    _validate_extraction_plan(audit_body.get("extraction_plan"), screen_images)

    model_plan, readiness_body = _load_model_plan_and_readiness(resolved_root, protocol, binding, model, depth)
    return {
        "root": resolved_root,
        "protocol": protocol,
        "plan": plan,
        "audit_ref": audit_ref,
        "audit_body": audit_body,
        "binding": binding,
        "model": model,
        "midpoint_depth": depth,
        "screen_images": screen_images,
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
            check(isinstance(entry, Mapping) and "chunk_index" in entry and "ref" in entry,
                  "List resume_refs entries must contain 'chunk_index' and 'ref'")
            idx = int(entry["chunk_index"])
            check(0 <= idx < total_chunks, f"Resume chunk_index out of range: {idx}")
            normalized[idx] = entry["ref"]
    else:
        raise ContractError("resume_refs must be a mapping {chunk_index: ref} or list of {'chunk_index', 'ref'}")
    return normalized


def _expected_chunk_rows(chunk: Mapping[str, Any], screen_images: Mapping[str, tuple]) -> list:
    expected = []
    for row_index, row in enumerate(chunk["rows"]):
        image_path, image_sha256 = screen_images[row["image_id"]]
        expected.append({
            "row_index": row_index,
            "image_id": row["image_id"],
            "language": row["language"],
            "split": E07_SPLIT,
            "caption_id": row["caption_id"],
            "caption_text": row["caption_text"],
            "caption_text_sha256": row["caption_text_sha256"],
            "draw_keys": list(row["draw_keys"]),
            "aliased_caption_ids": list(row["aliased_caption_ids"]),
            "image_path": image_path,
            "image_sha256": image_sha256,
        })
    return expected


def _verify_npy_array(path: Path, expected_sha256: str, n_views: int, dimension: int) -> np.ndarray:
    check(path.is_file(), f"Missing chunk array file: {path.name}")
    check(file_sha256(path) == expected_sha256, f"Chunk array SHA-256 mismatch: {path.name}")
    arr = np.load(path, allow_pickle=False)
    check(
        isinstance(arr, np.ndarray)
        and arr.dtype == np.dtype("<f4")
        and arr.shape == (n_views, dimension)
        and arr.flags.c_contiguous
        and np.isfinite(arr).all(),
        f"Invalid chunk array dtype/shape/finiteness in {path.name}",
    )
    return arr


def _verify_resumed_chunk(
    *,
    root: Path,
    protocol: Mapping[str, Any],
    plan,
    audit_ref,
    model: str,
    depth: int,
    dimension: int,
    model_fingerprint: str,
    execution_fingerprint: str,
    chunk_index: int,
    total_chunks: int,
    chunk: Mapping[str, Any],
    screen_images: Mapping[str, tuple],
    reference,
):
    ref, body, marker = load_phase2_artifact(
        root,
        protocol,
        reference,
        KIND_E07_CHUNK,
        expected_schema=SCHEMA_E07_FEATURES,
        expected_dataset=plan["dataset_fingerprint"],
        expected_plan_identity=plan["plan_identity"],
        expected_owner=model,
    )
    check(body.get("experiment") == "E07", "Resumed chunk experiment mismatch")
    check(plain(body.get("e07_audit_ref")) == plain(audit_ref), "Resumed chunk e07_audit_ref mismatch")
    check(body.get("model") == model, "Resumed chunk model mismatch")
    check(body.get("midpoint_depth") == depth, "Resumed chunk midpoint_depth mismatch")
    check(body.get("condition") == E07_CONDITION, "Resumed chunk condition mismatch")
    check(list(body.get("poolings", [])) == list(POOLINGS), "Resumed chunk poolings mismatch")
    check(body.get("split") == E07_SPLIT, "Resumed chunk split mismatch")
    check(body.get("confirm_excluded") is True, "Resumed chunk confirm_excluded must be True")
    check(body.get("chunk_index") == chunk_index, "Resumed chunk_index mismatch")
    check(body.get("total_chunks") == total_chunks, "Resumed chunk total_chunks mismatch")
    check(body.get("n_views") == len(chunk["rows"]), "Resumed chunk n_views mismatch")
    check(body.get("dimension") == dimension, "Resumed chunk dimension mismatch")
    check(body.get("model_fingerprint") == model_fingerprint, "Resumed chunk model_fingerprint mismatch")
    check(body.get("execution_fingerprint") == execution_fingerprint, "Resumed chunk execution_fingerprint mismatch")

    rc = body.get("repeat_check")
    check(isinstance(rc, Mapping) and rc.get("passed") is True, "Resumed chunk repeat_check did not pass")

    folder = within(root, ref["path"]).parent
    expected_rows = _expected_chunk_rows(chunk, screen_images)
    expected_rows_bytes = jsonl(expected_rows)
    expected_rows_sha = hashlib.sha256(expected_rows_bytes).hexdigest()

    check("rows.jsonl" in marker["files"], "Resumed chunk missing rows.jsonl")
    check(marker["files"]["rows.jsonl"] == expected_rows_sha, "Resumed chunk rows.jsonl SHA-256 mismatch")
    check(file_sha256(folder / "rows.jsonl") == expected_rows_sha, "Resumed chunk rows.jsonl on-disk mismatch")
    check(body.get("rows_sha256") == expected_rows_sha, "Resumed chunk body rows_sha256 mismatch")

    for name in ("last_caption.npy", "caption_mean.npy"):
        check(name in marker["files"], f"Resumed chunk missing {name}")
        _verify_npy_array(folder / name, marker["files"][name], len(expected_rows), dimension)

    return plain(ref), body, expected_rows_sha


def _verify_adapter_contract(adapter, model_plan, readiness, model: str, depth: int):
    assert_runtime(adapter, model_plan)
    check(adapter.config["precision"] == "bfloat16", "E07 extraction requires bfloat16 precision")
    check(adapter.config["attention_backend"] == "eager", "E07 extraction requires eager attention_backend")

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


def _extract_single_view(
    *,
    root: Path,
    adapter,
    row: Mapping[str, Any],
    image_path: str,
    image_sha256: str,
    last_site: FeatureSite,
    mean_site: FeatureSite,
    dimension: int,
):
    img_file = within(root, image_path)
    check(img_file.is_file(), f"Missing pinned SCREEN image: {image_path}")
    check(file_sha256(img_file) == image_sha256, f"Pinned SCREEN image SHA-256 mismatch: {image_path}")
    check(
        hashlib.sha256(row["caption_text"].encode("utf-8")).hexdigest() == row["caption_text_sha256"],
        f"Caption UTF-8 SHA-256 mismatch for {row['image_id']}/{row['language']}",
    )

    model_input = ModelInput(
        image_id=row["image_id"],
        image_path=image_path,
        image_sha256=image_sha256,
        language=row["language"],
        caption_id=row["caption_id"],
        caption_text=row["caption_text"],
        split=E07_SPLIT,
    )
    request = ExtractionRequest(
        inputs=[model_input],
        sites=[last_site, mean_site],
        condition=E07_CONDITION,
        purpose="preliminary_features",
    )
    result = extract_with_evidence(adapter, request)
    vec_last = np.asarray(result.features.vectors[last_site.key][0], dtype="<f4")
    vec_mean = np.asarray(result.features.vectors[mean_site.key][0], dtype="<f4")
    check(
        vec_last.shape == (dimension,) and np.isfinite(vec_last).all(),
        f"Invalid/nonfinite last_caption vector for {row['image_id']}/{row['language']}",
    )
    check(
        vec_mean.shape == (dimension,) and np.isfinite(vec_mean).all(),
        f"Invalid/nonfinite caption_mean vector for {row['image_id']}/{row['language']}",
    )
    return vec_last, vec_mean, dict(result.evidence[0])


@contract
def run_e07_extraction(
    root,
    protocol,
    plan_ref,
    e07_audit_ref,
    model: str,
    producer: Mapping[str, Any],
    *,
    adapter_factory=None,
    resume_refs=None,
    run_id: Optional[str] = None,
    progress=None,
):
    """Execute Phase 2 Stage 2 Branch C E07 extraction for one model (`qwen` or `gemma`)."""
    emit = progress if progress is not None else _default_emit
    t_read_hash = time.monotonic()
    ctx = preflight_e07_extraction(root, protocol, plan_ref, e07_audit_ref, model)
    read_hash_s = time.monotonic() - t_read_hash

    resolved_root = ctx["root"]
    plan = ctx["plan"]
    audit_ref = ctx["audit_ref"]
    audit_body = ctx["audit_body"]
    depth = ctx["midpoint_depth"]
    screen_images = ctx["screen_images"]
    model_plan = ctx["model_plan"]
    readiness = ctx["readiness"]
    ep = audit_body["extraction_plan"]
    chunks = ep["chunks"]
    total_chunks = len(chunks)

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
        "event": "e07_extract_start",
        "unit": f"{model}:preflight",
        "completed": 0,
        "total": total_chunks,
        "read_hash_s": round(read_hash_s, 6),
        "compute_s": 0.0,
        "write_s": 0.0,
        "peak_rss_bytes": _peak_rss_bytes(),
        "model": model,
        "midpoint_depth": depth,
        "total_unique_views": ep["total_unique_views"],
        "total_chunks": total_chunks,
    })

    normalized_resume = _normalize_resume_refs(resume_refs, total_chunks)
    completed_chunks = {}
    t_res_start = time.monotonic()
    for chunk_index, ref in sorted(normalized_resume.items()):
        verified_ref, chunk_body, rows_sha = _verify_resumed_chunk(
            root=resolved_root,
            protocol=protocol,
            plan=plan,
            audit_ref=audit_ref,
            model=model,
            depth=depth,
            dimension=dimension,
            model_fingerprint=model_fingerprint,
            execution_fingerprint=execution_fingerprint,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            chunk=chunks[chunk_index],
            screen_images=screen_images,
            reference=ref,
        )
        completed_chunks[chunk_index] = {
            "chunk_index": chunk_index,
            "chunk_id": chunks[chunk_index]["chunk_id"],
            "language": chunks[chunk_index]["language"],
            "n_views": len(chunks[chunk_index]["rows"]),
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
        _, _, verified_dim, _, _ = _verify_adapter_contract(adapter, model_plan, readiness, model, depth)
        check(verified_dim == dimension, "Verified adapter dimension mismatch")

    batch_run_id = run_id or new_run_id()
    compute_s = 0.0
    write_s = 0.0

    for chunk_index in range(total_chunks):
        if chunk_index in completed_chunks:
            emit({
                "event": "e07_chunk_resumed",
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
        expected_rows = _expected_chunk_rows(chunk, screen_images)

        t_comp = time.monotonic()
        last_vecs = []
        mean_vecs = []
        evidences = []
        for row in rows:
            img_path, img_sha = screen_images[row["image_id"]]
            v_last, v_mean, ev = _extract_single_view(
                root=resolved_root,
                adapter=adapter,
                row=row,
                image_path=img_path,
                image_sha256=img_sha,
                last_site=last_site,
                mean_site=mean_site,
                dimension=dimension,
            )
            last_vecs.append(v_last)
            mean_vecs.append(v_mean)
            evidences.append(ev)

        # First/last repeat determinism check through extract_with_evidence.
        repeat_indices = sorted({0, n_views - 1})
        repeat_details = {}
        max_diff_last = 0.0
        max_diff_mean = 0.0
        repeat_passed = True
        for ridx in repeat_indices:
            rrow = rows[ridx]
            img_path, img_sha = screen_images[rrow["image_id"]]
            rep_last, rep_mean, _ = _extract_single_view(
                root=resolved_root,
                adapter=adapter,
                row=rrow,
                image_path=img_path,
                image_sha256=img_sha,
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

        compute_s += time.monotonic() - t_comp
        check(
            repeat_passed,
            f"Repeat determinism check failed for {model} chunk {chunk_index}: "
            f"max_abs_diff_last={max_diff_last}, max_abs_diff_mean={max_diff_mean}, atol={atol}, rtol={rtol}",
        )

        last_arr = np.ascontiguousarray(np.stack(last_vecs, axis=0), dtype="<f4")
        mean_arr = np.ascontiguousarray(np.stack(mean_vecs, axis=0), dtype="<f4")
        check(last_arr.shape == (n_views, dimension) and mean_arr.shape == (n_views, dimension),
              "Chunk array shape mismatch")

        repeat_check = {
            "first_index": 0,
            "last_index": n_views - 1,
            "max_abs_diff_last_caption": max_diff_last,
            "max_abs_diff_caption_mean": max_diff_mean,
            "passed": True,
        }
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
            "schema": SCHEMA_E07_FEATURES,
            "experiment": "E07",
            "plan": plain(plan.reference),
            "plan_identity": plan["plan_identity"],
            "e07_audit_ref": plain(audit_ref),
            "model": model,
            "midpoint_depth": depth,
            "condition": E07_CONDITION,
            "poolings": list(POOLINGS),
            "split": E07_SPLIT,
            "chunk_index": chunk_index,
            "chunk_id": chunk["chunk_id"],
            "language": chunk["language"],
            "total_chunks": total_chunks,
            "n_views": n_views,
            "dimension": dimension,
            "model_fingerprint": model_fingerprint,
            "execution_fingerprint": execution_fingerprint,
            "site_specs": site_specs,
            "evidence_bridge_sha256": bridge_sha,
            "repeat_check": repeat_check,
            "rows_sha256": rows_sha,
            "confirm_excluded": True,
        }

        t_w = time.monotonic()
        chunk_ref = publish_phase2(
            resolved_root,
            protocol,
            plan["dataset_fingerprint"],
            f"phase2/features/e07/{model}/{batch_run_id}",
            KIND_E07_CHUNK,
            chunk_body,
            producer,
            arrays={"last_caption": last_arr, "caption_mean": mean_arr},
            files={
                "rows.jsonl": rows_bytes,
                "token_evidence.jsonl": jsonl(evidences),
                "audits.json": json_bytes(audits_obj),
            },
            upstream=[plain(plan.reference), plain(audit_ref)],
            model_fingerprint=model_fingerprint,
            run_id=f"chunk_{chunk_index:04d}",
        )
        write_s += time.monotonic() - t_w

        completed_chunks[chunk_index] = {
            "chunk_index": chunk_index,
            "chunk_id": chunk["chunk_id"],
            "language": chunk["language"],
            "n_views": n_views,
            "ref": plain(chunk_ref),
            "rows_sha256": rows_sha,
            "repeat_check": repeat_check,
        }
        emit({
            "event": "e07_chunk_published",
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

    ordered_chunks = [completed_chunks[i] for i in range(total_chunks)]
    repeat_summary = {
        "chunks_checked": total_chunks,
        "all_passed": all(c["repeat_check"]["passed"] for c in ordered_chunks),
        "max_abs_diff_last_caption": max(c["repeat_check"]["max_abs_diff_last_caption"] for c in ordered_chunks),
        "max_abs_diff_caption_mean": max(c["repeat_check"]["max_abs_diff_caption_mean"] for c in ordered_chunks),
    }
    release_chunks = [
        {
            "chunk_index": c["chunk_index"],
            "chunk_id": c["chunk_id"],
            "language": c["language"],
            "n_views": c["n_views"],
            "ref": c["ref"],
            "rows_sha256": c["rows_sha256"],
        }
        for c in ordered_chunks
    ]
    release_body = {
        "schema": SCHEMA_E07_FEATURES,
        "experiment": "E07",
        "plan": plain(plan.reference),
        "plan_identity": plan["plan_identity"],
        "e07_audit_ref": plain(audit_ref),
        "e07_verdict": audit_body["verdict"],
        "model": model,
        "midpoint_depth": depth,
        "condition": E07_CONDITION,
        "poolings": list(POOLINGS),
        "split": E07_SPLIT,
        "total_unique_views": ep["total_unique_views"],
        "per_language_counts": plain(ep["per_language_counts"]),
        "dimension": dimension,
        "model_fingerprint": model_fingerprint,
        "execution_fingerprint": execution_fingerprint,
        "chunks": release_chunks,
        "repeat_check_summary": repeat_summary,
        "confirm_excluded": True,
    }

    t_rel = time.monotonic()
    release_ref = publish_phase2(
        resolved_root,
        protocol,
        plan["dataset_fingerprint"],
        f"phase2/features/e07/{model}/releases",
        KIND_E07_RELEASE,
        release_body,
        producer,
        upstream=[plain(plan.reference), plain(audit_ref)] + [c["ref"] for c in release_chunks],
        model_fingerprint=model_fingerprint,
        run_id=batch_run_id,
    )
    write_s += time.monotonic() - t_rel

    emit({
        "event": "e07_release_published",
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
        "e07_verdict": audit_body["verdict"],
        "total_unique_views": ep["total_unique_views"],
        "total_chunks": total_chunks,
        "release_ref": plain(release_ref),
        "chunks": release_chunks,
        "confirm_excluded": True,
    })


@contract
def verify_e07_release(
    root,
    protocol,
    release_ref,
    *,
    expected_plan_identity: Optional[str] = None,
    expected_model: Optional[str] = None,
):
    """Authenticate a published `phase2_e07_release` and all its referenced chunk manifests."""
    resolved_root = Path(root).resolve()
    ref, body, marker = load_phase2_artifact(
        resolved_root,
        protocol,
        release_ref,
        KIND_E07_RELEASE,
        expected_schema=SCHEMA_E07_FEATURES,
        expected_plan_identity=expected_plan_identity,
        expected_owner=expected_model,
    )
    check(body.get("experiment") == "E07", "Release experiment mismatch")
    check(body.get("confirm_excluded") is True, "Release confirm_excluded must be True")
    check(body.get("split") == E07_SPLIT, "Release split must be SCREEN")
    check(body.get("e07_verdict") in E07_ALLOWED_VERDICTS, "Release e07_verdict invalid")
    check(body.get("repeat_check_summary", {}).get("all_passed") is True, "Release repeat checks did not all pass")

    chunks = body.get("chunks", [])
    check(len(chunks) > 0, "Release must list at least one chunk")
    total_views = 0
    for expected_idx, entry in enumerate(chunks):
        check(entry.get("chunk_index") == expected_idx, "Release chunk ordering mismatch")
        c_ref, c_body, c_marker = load_phase2_artifact(
            resolved_root,
            protocol,
            entry["ref"],
            KIND_E07_CHUNK,
            expected_schema=SCHEMA_E07_FEATURES,
            expected_dataset=marker["dataset_fingerprint"],
            expected_plan_identity=body["plan_identity"],
            expected_owner=body["model"],
        )
        check(c_body["chunk_index"] == expected_idx, "Chunk index mismatch")
        check(c_body["n_views"] == entry["n_views"], "Chunk n_views mismatch")
        check(c_body["rows_sha256"] == entry["rows_sha256"], "Chunk rows_sha256 mismatch")
        check(c_marker["files"]["rows.jsonl"] == entry["rows_sha256"], "Chunk marker rows.jsonl mismatch")
        check(c_body["repeat_check"]["passed"] is True, "Chunk repeat_check failed")
        total_views += int(entry["n_views"])

    check(total_views == body["total_unique_views"], "Release total_unique_views sum mismatch")
    return ref, body, marker


def _parse_ref_arg(value):
    if value is None:
        return None
    text = value.strip()
    if text.startswith("{") or text.startswith("["):
        parsed = json.loads(text)
        return parsed
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--phase1-root", required=True)
    parser.add_argument("--plan", required=True, help="Explicit phase2_plan {'path', 'sha256'} JSON or path")
    parser.add_argument("--e07-audit", required=True, help="Explicit phase2_e07_audit {'path', 'sha256'} JSON or path")
    parser.add_argument("--protocol", type=Path, default=REPO / "configs/protocol.json")
    parser.add_argument("--resume-refs", help="Optional JSON mapping {chunk_index: {'path', 'sha256'}}")
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
    plan_ref = _parse_ref_arg(args.plan)
    audit_ref = _parse_ref_arg(args.e07_audit)

    if args.preflight_only:
        ctx = preflight_e07_extraction(root, protocol, plan_ref, audit_ref, args.model)
        _default_emit({
            "event": "e07_extract_preflight",
            "status": "passed",
            "model": args.model,
            "midpoint_depth": ctx["midpoint_depth"],
            "e07_verdict": ctx["audit_body"]["verdict"],
            "total_unique_views": ctx["audit_body"]["extraction_plan"]["total_unique_views"],
            "total_chunks": ctx["audit_body"]["extraction_plan"]["total_chunks"],
            "weights_loaded": False,
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
    result = run_e07_extraction(
        root,
        protocol,
        plan_ref,
        audit_ref,
        args.model,
        producer,
        resume_refs=resume_refs,
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
