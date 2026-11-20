"""Phase 3 wire schemas, namespace guards, storage probe, and explicit-ref resume.

All Phase 3 artifacts live under `PHASE1_ROOT/phase3/` using the immutable
`stage1.v1` completion-marker wrapper (`_COMPLETE.json` written last). Phase 1
and Phase 2 directories remain strictly read-only.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from src.common.contracts import ContractError, canonical_bytes, fingerprint, require_sha256
from src.common.storage import MARKER, file_sha256, new_run_id, phase1_root, publish, verify, within
from src.stage2.schema import REPO, check, contract, frozen, json_bytes, plain

PHASE3_CPU_VERSION = "phase3-cpu-stageA1.v1"
PHASE3_GPU_VERSION = "phase3-gpu-stageB1.v1"
PHASE3_NAMESPACE = "phase3"

MODELS = ("qwen", "gemma")
POOLINGS = ("last_caption", "caption_mean")
CONDITIONS = ("image_caption", "caption_only")
ALLOWED_SPLITS = ("FIT", "DEV-S", "DEV-C", "SCREEN")
CLOSED_SPLITS = ("CONFIRM",)

KIND_PROBE_PAYLOAD = "phase3_storage_probe_payload"
KIND_PROBE = "phase3_storage_probe"
KIND_CPU_PLAN = "phase3_cpu_plan"
KIND_GPU_PLAN = "phase3_gpu_plan"
KIND_E39_E31_CHUNK = "phase3_e39_e31_feature_chunk"
KIND_E39_E31_RELEASE = "phase3_e39_e31_feature_release"
KIND_E23_CHUNK = "phase3_e23_feature_chunk"
KIND_E23_RELEASE = "phase3_e23_feature_release"
KIND_PREDICTION = "phase3_prediction_block"
KIND_CALIBRATORS = "phase3_e41_calibrators"
KIND_MAPS = "phase3_e12_maps"
KIND_ANALYSIS = "phase3_analysis_block"
KIND_RELEASE = "phase3_release"

SCHEMA_PROBE = "phase3.storage_probe.v1"
SCHEMA_CPU_PLAN = "phase3.cpu_plan.v1"
SCHEMA_GPU_PLAN = "phase3.gpu_plan.v1"
SCHEMA_E39_E31_FEATURES = "phase3.e39_e31_features.v1"
SCHEMA_E23_FEATURES = "phase3.e23_features.v1"
SCHEMA_PREDICTION = "phase3.prediction.v1"
SCHEMA_CALIBRATORS = "phase3.calibrators.v1"
SCHEMA_MAPS = "phase3.maps.v1"
SCHEMA_ANALYSIS = "phase3.analysis.v1"
SCHEMA_RELEASE = "phase3.release.v1"

PHASE3_PARENT_DIRS = (
    "phase3/probes",
    "phase3/plans",
    "phase3/releases",
    "phase3/predictions/e40_crossed/qwen",
    "phase3/predictions/e40_crossed/gemma",
    "phase3/results/e40/qwen",
    "phase3/results/e40/gemma",
    "phase3/predictions/e41_cov/qwen",
    "phase3/predictions/e41_cov/gemma",
    "phase3/calibrators/e41/qwen",
    "phase3/calibrators/e41/gemma",
    "phase3/results/e41/qwen",
    "phase3/results/e41/gemma",
    "phase3/results/e44/qwen",
    "phase3/results/e44/gemma",
    "phase3/maps/e12/qwen",
    "phase3/maps/e12/gemma",
    "phase3/predictions/e12/qwen",
    "phase3/predictions/e12/gemma",
    "phase3/results/e12/qwen",
    "phase3/results/e12/gemma",
    # Track B GPU parent dirs pre-initialized once to avoid Drive mkdir races.
    "phase3/features/e39_e31/qwen",
    "phase3/features/e39_e31/gemma",
    "phase3/results/e39_e31/qwen",
    "phase3/results/e39_e31/gemma",
    "phase3/features/e23/qwen",
    "phase3/features/e23/gemma",
    "phase3/results/e23/qwen",
    "phase3/results/e23/gemma",
    "phase3/g7_e35/outputs/qwen",
    "phase3/g7_e35/outputs/gemma",
    "phase3/g7_e35/results/qwen",
    "phase3/g7_e35/results/gemma",
)


def require_open_split(split):
    """Reject CONFIRM and unknown splits before any IO or computation."""
    check(split not in CLOSED_SPLITS, "CONFIRM is closed to preliminary Phase 3 processing")
    check(split in ALLOWED_SPLITS, f"Unknown or unauthorized Phase 3 split: {split!r}")
    return split


def require_phase3_namespace(namespace):
    """Ensure every Phase 3 write stays strictly inside PHASE1_ROOT/phase3/."""
    check(isinstance(namespace, str) and bool(namespace), "Nonempty relative Phase 3 namespace required")
    check(
        "\\" not in namespace and ":" not in namespace and not namespace.startswith("/"),
        "Phase 3 namespace must be a portable relative POSIX path",
    )
    parts = [p for p in namespace.split("/") if p]
    check(
        parts and parts[0] == PHASE3_NAMESPACE and ".." not in parts and "." not in parts,
        f"Phase 3 writes must stay under '{PHASE3_NAMESPACE}/'; got {namespace!r}",
    )
    normalized = "/".join(parts)
    check(normalized == namespace, f"Noncanonical Phase 3 namespace: {namespace!r}")
    return normalized


def normalize_ref(root, value):
    """Return canonical {'path', 'sha256'} relative to root for an explicit marker."""
    root = Path(root).resolve()
    if isinstance(value, MappingRefTypes):
        ref = plain(value)
        check(set(ref.keys()) == {"path", "sha256"}, f"Invalid completion reference keys: {sorted(ref.keys())}")
        rel = Path(ref["path"]).as_posix()
        check(not rel.startswith("/") and ".." not in rel.split("/"), f"Nonportable ref path: {rel!r}")
        require_sha256(ref["sha256"])
        return frozen({"path": rel, "sha256": ref["sha256"]})
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        resolved = within(root, path.relative_to(root).as_posix() if path.resolve().is_relative_to(root) else str(path))
        check(resolved.name == MARKER, f"Completion reference must point to {MARKER}: {resolved}")
        return frozen({"path": resolved.relative_to(root).as_posix(), "sha256": file_sha256(resolved)})
    raise ContractError(f"Unsupported Phase 3 completion reference: {value!r}")


from collections.abc import Mapping  # noqa: E402

MappingRefTypes = Mapping


def init_phase3_dirs(root):
    """Pre-create all PHASE3_PARENT_DIRS once before concurrent workers launch."""
    root = Path(root).resolve()
    created = []
    for rel in PHASE3_PARENT_DIRS:
        require_phase3_namespace(rel)
        target = within(root, rel)
        target.mkdir(parents=True, exist_ok=True)
        created.append(rel)
    return tuple(created)


@contract
def publish_phase3(
    root,
    protocol,
    dataset_fingerprint,
    namespace,
    kind,
    body,
    producer,
    *,
    arrays=None,
    files=None,
    upstream=(),
    model_fingerprint=None,
    seeds=None,
    run_id=None,
):
    """Publish one immutable Phase 3 artifact under PHASE1_ROOT/phase3/."""
    root = Path(root).resolve()
    ns = require_phase3_namespace(namespace)
    check(isinstance(body, Mapping) and "schema" in body, "Phase 3 artifact body must be a mapping with 'schema'")
    run = run_id or new_run_id()
    check("/" not in run and "\\" not in run and ".." not in run and bool(run), f"Invalid run_id: {run!r}")

    payload = {"body.json": json_bytes(plain(body))}
    if arrays:
        import numpy as np

        for name, arr in arrays.items():
            fname = name if name.endswith(".npy") else f"{name}.npy"
            check("/" not in fname and "\\" not in fname, f"Invalid array filename: {fname!r}")
            buf = __import__("io").BytesIO()
            np.save(buf, np.asarray(arr), allow_pickle=False)
            payload[fname] = buf.getvalue()
    if files:
        for fname, raw in files.items():
            check("/" not in fname and "\\" not in fname, f"Invalid payload filename: {fname!r}")
            payload[fname] = raw if isinstance(raw, (bytes, bytearray)) else str(raw).encode("utf-8")

    proto_fp = fingerprint(plain(protocol))
    meta = {
        "schema": "stage1.v1",
        "kind": kind,
        "status": "passed",
        "protocol_fingerprint": proto_fp,
        "dataset_fingerprint": dataset_fingerprint,
        "code_commit": producer["code_commit"],
        "storage_scope": producer["storage_scope"],
        "software_versions": plain(producer["software_versions"]),
        "run_id": run,
        "model_fingerprint": model_fingerprint,
        "seeds": plain(seeds) if seeds is not None else {},
        "upstream_artifacts": [plain(u) for u in upstream],
        "phase3_version": PHASE3_CPU_VERSION,
    }

    folder = publish(root, f"{ns}/{run}", meta, payload)
    marker_path = folder / MARKER
    verify(folder, expected={
        "kind": kind,
        "status": "passed",
        "protocol_fingerprint": proto_fp,
        "dataset_fingerprint": dataset_fingerprint,
    })
    return frozen({"path": marker_path.relative_to(root).as_posix(), "sha256": file_sha256(marker_path)})


@contract
def load_phase3_artifact(
    root,
    protocol,
    reference,
    expected_kind,
    *,
    expected_schema=None,
    expected_dataset=None,
    expected_plan_identity=None,
    expected_owner=None,
):
    """Authenticate a Phase 3 artifact on disk and return (ref, frozen body, frozen marker)."""
    root = Path(root).resolve()
    ref = normalize_ref(root, reference)
    require_phase3_namespace(Path(ref["path"]).parent.as_posix())
    marker_path = within(root, ref["path"])
    check(file_sha256(marker_path) == ref["sha256"], "Phase 3 completion manifest checksum mismatch")
    expected_meta = {
        "kind": expected_kind,
        "status": "passed",
        "protocol_fingerprint": fingerprint(plain(protocol)),
    }
    if expected_dataset is not None:
        expected_meta["dataset_fingerprint"] = expected_dataset
    marker = verify(marker_path.parent, expected=expected_meta)
    check("body.json" in marker["files"], "Phase 3 artifact missing body.json")
    body = json.loads((marker_path.parent / "body.json").read_text(encoding="utf-8"))
    if expected_schema is not None:
        check(
            body.get("schema") == expected_schema,
            f"Phase 3 schema mismatch: expected {expected_schema!r}, got {body.get('schema')!r}",
        )
    if expected_plan_identity is not None:
        check(body.get("plan_identity") == expected_plan_identity, "Phase 3 plan_identity mismatch")
    if expected_owner is not None:
        owner_val = body.get("owner", body.get("model"))
        check(owner_val == expected_owner, f"Foreign Phase 3 owner: expected {expected_owner!r}, got {owner_val!r}")
    return frozen(ref), frozen(body), frozen(marker)


@contract
def verify_resume_ref(
    root,
    protocol,
    reference,
    expected_kind,
    *,
    expected_schema=None,
    expected_dataset=None,
    expected_plan_identity=None,
    expected_owner=None,
    expected_inputs=None,
):
    """Verify an explicit resume reference and reject incompatible inputs/lineage."""
    ref, body, marker = load_phase3_artifact(
        root,
        protocol,
        reference,
        expected_kind,
        expected_schema=expected_schema,
        expected_dataset=expected_dataset,
        expected_plan_identity=expected_plan_identity,
        expected_owner=expected_owner,
    )
    if expected_inputs is not None:
        check(
            plain(body.get("inputs")) == plain(expected_inputs),
            "Resume artifact inputs do not match current run inputs",
        )
    return ref, body, marker


@contract
def storage_probe(root, protocol, dataset_fingerprint, producer):
    """Publish a probe under phase3/probes/, verify in a fresh subprocess, return receipt ref."""
    root = Path(root).resolve()
    payload_ref = publish_phase3(
        root,
        protocol,
        dataset_fingerprint,
        "phase3/probes",
        KIND_PROBE_PAYLOAD,
        {"schema": SCHEMA_PROBE, "nonce": new_run_id(), "remote_sync_guarantee": False},
        producer,
    )
    import tempfile

    default_proto_path = REPO / "configs/protocol.json"
    default_proto = json.loads(default_proto_path.read_text(encoding="utf-8"))
    temp_proto = None
    try:
        proto_arg = default_proto_path
        if fingerprint(protocol) != fingerprint(default_proto):
            tmp = tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False)
            tmp.write(json.dumps(plain(protocol), sort_keys=True) + "\n")
            tmp.close()
            temp_proto = Path(tmp.name)
            proto_arg = temp_proto
        cmd = [
            sys.executable,
            "-m",
            "src.phase3.schema",
            "probe-read",
            "--phase1-root",
            str(root),
            "--manifest",
            str(root / payload_ref["path"]),
            "--protocol",
            str(proto_arg),
        ]
        child = subprocess.run(cmd, cwd=REPO, check=True, capture_output=True, text=True)
    finally:
        if temp_proto is not None and temp_proto.exists():
            temp_proto.unlink()
    observed = json.loads(child.stdout)
    check(
        type(observed.get("pid")) is int
        and observed["pid"] != os.getpid()
        and observed.get("payload") == plain(payload_ref),
        "Independent Phase 3 storage probe readback mismatch",
    )
    return publish_phase3(
        root,
        protocol,
        dataset_fingerprint,
        "phase3/probes",
        KIND_PROBE,
        {
            "schema": SCHEMA_PROBE,
            "payload": plain(payload_ref),
            "independent_process_readback": True,
            "reader_pid": observed["pid"],
            "writer_pid": os.getpid(),
            "remote_sync_guarantee": False,
        },
        producer,
        upstream=[plain(payload_ref)],
    )


@contract
def verify_storage_probe(root, protocol, probe_ref, dataset_fingerprint):
    """Authenticate an existing two-process Phase 3 storage probe receipt."""
    ref, body, _ = load_phase3_artifact(
        root,
        protocol,
        probe_ref,
        KIND_PROBE,
        expected_schema=SCHEMA_PROBE,
        expected_dataset=dataset_fingerprint,
    )
    check(
        body["independent_process_readback"] is True
        and type(body["reader_pid"]) is int
        and type(body["writer_pid"]) is int
        and body["reader_pid"] != body["writer_pid"],
        "Invalid two-process storage probe receipt",
    )
    load_phase3_artifact(
        root,
        protocol,
        body["payload"],
        KIND_PROBE_PAYLOAD,
        expected_schema=SCHEMA_PROBE,
        expected_dataset=dataset_fingerprint,
    )
    return ref, body


def _cli(argv=None):
    parser = argparse.ArgumentParser(description="Phase 3 independent storage probe reader.")
    parser.add_argument("command", choices=["probe-read"])
    parser.add_argument("--phase1-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--protocol", type=Path, default=REPO / "configs/protocol.json")
    args = parser.parse_args(argv)
    root = phase1_root(args.phase1_root)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    ref, _, _ = load_phase3_artifact(root, protocol, args.manifest, KIND_PROBE_PAYLOAD, expected_schema=SCHEMA_PROBE)
    print(json.dumps({"pid": os.getpid(), "payload": plain(ref)}), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_cli())
    except (ContractError, OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "blocked", "blockers": [str(exc)]}), flush=True)
        raise SystemExit(2)
