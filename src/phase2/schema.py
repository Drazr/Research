"""Phase 2 wire schemas, namespace guards, storage probe, and explicit-ref resume.

All Phase 2 artifacts live under `PHASE1_ROOT/phase2/` using the immutable
`stage1.v1` completion-marker wrapper (`_COMPLETE.json` written last). Phase 1
directories remain strictly read-only.
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

PHASE2_VERSION = "phase2-stage1.v1"
PHASE2_NAMESPACE = "phase2"

MODELS = ("qwen", "gemma")
POOLINGS = ("last_caption", "caption_mean")
CONDITIONS = ("image_caption", "caption_only")
ALLOWED_SPLITS = ("FIT", "DEV-S", "DEV-C", "SCREEN")
CLOSED_SPLITS = ("CONFIRM",)

KIND_PROBE_PAYLOAD = "phase2_storage_probe_payload"
KIND_PROBE = "phase2_storage_probe"
KIND_PLAN = "phase2_plan"
KIND_E07_AUDIT = "phase2_e07_audit"
KIND_E17_AUTH = "phase2_e17_auth"
KIND_E29_GATE = "phase2_e29_gate"
KIND_E07_CHUNK = "phase2_e07_chunk"
KIND_E07_RELEASE = "phase2_e07_release"
KIND_PREDICTION = "phase2_prediction_block"
KIND_CALIBRATORS = "phase2_e17_calibrators"
KIND_ANALYSIS = "phase2_analysis_block"
KIND_RELEASE = "phase2_release"

SCHEMA_PROBE = "phase2.storage_probe.v1"
SCHEMA_PLAN = "phase2.plan.v1"
SCHEMA_E07_AUDIT = "phase2.e07_audit.v1"
SCHEMA_E17_AUTH = "phase2.e17_auth.v1"
SCHEMA_E29_GATE = "phase2.e29_gate.v1"
SCHEMA_E07_FEATURES = "phase2.e07_features.v1"
SCHEMA_PREDICTION = "phase2.prediction.v1"
SCHEMA_CALIBRATORS = "phase2.calibrators.v1"
SCHEMA_ANALYSIS = "phase2.analysis.v1"
SCHEMA_RELEASE = "phase2.release.v1"

STATUSES = ("ok", "descriptive_low_support", "undefined", "unavailable", "blocked")
SUPERVISION_LABELS = (
    "within_context_source_fit",
    "source_only_transfer",
    "source_dev_c_calibrated",
    "target_dev_c_supervised_calibration",
    "target_fit_input_adapted",
)
E07_VERDICTS = ("passed_full", "passed_restricted", "blocked")
GATE_DECISIONS = ("OPEN", "PARK", "UNRESOLVED")

PHASE2_PARENT_DIRS = (
    "phase2/probes",
    "phase2/plans",
    "phase2/audits/e07",
    "phase2/features/e07/qwen",
    "phase2/features/e07/qwen/releases",
    "phase2/features/e07/gemma",
    "phase2/features/e07/gemma/releases",
    "phase2/predictions/e17_dev_c/qwen",
    "phase2/predictions/e17_dev_c/gemma",
    "phase2/predictions/e30_crossed/qwen",
    "phase2/predictions/e30_crossed/gemma",
    "phase2/predictions/e11_adapted/qwen",
    "phase2/predictions/e11_adapted/gemma",
    "phase2/predictions/e07_screen/qwen",
    "phase2/predictions/e07_screen/gemma",
    "phase2/calibrators/e17/qwen",
    "phase2/calibrators/e17/gemma",
    "phase2/results/e29/qwen",
    "phase2/results/e29/gemma",
    "phase2/results/e17/qwen",
    "phase2/results/e17/gemma",
    "phase2/results/e30/qwen",
    "phase2/results/e30/gemma",
    "phase2/results/e11/qwen",
    "phase2/results/e11/gemma",
    "phase2/results/e07/qwen",
    "phase2/results/e07/gemma",
    "phase2/results/e29_gates",
    "phase2/releases",
)


def require_open_split(split):
    """Reject CONFIRM and unknown splits before any IO or computation."""
    check(split not in CLOSED_SPLITS, "CONFIRM is closed to preliminary Phase 2 processing")
    check(split in ALLOWED_SPLITS, f"Unknown or unauthorized Phase 2 split: {split!r}")
    return split


def require_phase2_namespace(namespace):
    """Ensure every Phase 2 write stays strictly inside PHASE1_ROOT/phase2/."""
    check(isinstance(namespace, str) and bool(namespace), "Nonempty relative Phase 2 namespace required")
    check("\\" not in namespace and ":" not in namespace and not namespace.startswith("/"),
          "Phase 2 namespace must be a portable relative POSIX path")
    parts = [p for p in namespace.split("/") if p]
    check(parts and parts[0] == PHASE2_NAMESPACE and ".." not in parts and "." not in parts,
          f"Phase 2 writes must stay under '{PHASE2_NAMESPACE}/'; got {namespace!r}")
    normalized = "/".join(parts)
    check(normalized == namespace, f"Noncanonical Phase 2 namespace: {namespace!r}")
    return normalized


def normalize_ref(root, value):
    """Return canonical {'path', 'sha256'} relative to root for an explicit marker."""
    root = Path(root).resolve()
    if isinstance(value, dict) or hasattr(value, "keys"):
        ref = plain(value)
        check(set(ref) == {"path", "sha256"}, "Explicit {'path', 'sha256'} reference required")
        require_sha256(ref["sha256"])
        target = within(root, ref["path"])
        check(target.name == MARKER, "Reference must point to _COMPLETE.json")
        return ref
    p = Path(value)
    p = p.resolve() if p.is_absolute() else within(root, p.as_posix())
    try:
        rel = p.relative_to(root).as_posix()
    except ValueError as exc:
        raise ContractError("Artifact outside PHASE1_ROOT") from exc
    check(p.name == MARKER, "Reference must point to _COMPLETE.json")
    check(p.is_file(), f"Missing completion manifest: {rel}")
    return {"path": rel, "sha256": file_sha256(p)}


def _validate_producer(producer):
    for key in ("code_commit", "storage_scope", "software_versions"):
        check(key in producer, f"Producer metadata missing: {key}")
    check(isinstance(producer["code_commit"], str) and len(producer["code_commit"]) == 40,
          "Producer requires 40-character code_commit")
    check(producer["storage_scope"] in ("local-development", "persistent-drive"),
          "Unknown storage_scope")
    check(isinstance(producer["software_versions"], dict) and bool(producer["software_versions"]),
          "Actual producer software_versions required")


def init_phase2_dirs(root):
    """Pre-create all shared Phase 2 parent directories once before parallel writers."""
    root = Path(root).resolve()
    created = []
    for rel in PHASE2_PARENT_DIRS:
        require_phase2_namespace(rel)
        target = within(root, rel)
        target.mkdir(parents=True, exist_ok=True)
        created.append(rel)
    return tuple(created)


@contract
def publish_phase2(
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
    """Publish an immutable Phase 2 bundle under PHASE1_ROOT/phase2/ and verify readback."""
    root = Path(root).resolve()
    namespace = require_phase2_namespace(namespace)
    require_sha256(dataset_fingerprint)
    _validate_producer(producer)
    check(isinstance(kind, str) and kind.startswith("phase2_"), f"Invalid Phase 2 artifact kind: {kind!r}")
    run = run_id or new_run_id()

    payload = {"body.json": json_bytes(body)}
    if files:
        for name, content in files.items():
            check(name != "body.json", "Reserved payload name: body.json")
            payload[name] = content
    if arrays:
        from src.stage3.artifacts import array_bytes
        for name, arr in arrays.items():
            filename = name if name.endswith(".npy") else f"{name}.npy"
            check(filename not in payload, f"Duplicate payload filename: {filename}")
            payload[filename] = array_bytes(arr)

    meta = {
        "kind": kind,
        "status": "passed",
        "run_id": run,
        "code_commit": producer["code_commit"],
        "storage_scope": producer["storage_scope"],
        "software_versions": plain(producer["software_versions"]),
        "protocol_fingerprint": fingerprint(protocol),
        "dataset_fingerprint": dataset_fingerprint,
        "model_fingerprint": model_fingerprint,
        "seeds": plain(seeds) if seeds is not None else {},
        "upstream_artifacts": plain(list(upstream)),
        "phase2_version": PHASE2_VERSION,
    }
    folder = publish(root, f"{namespace}/{run}", meta, payload)
    marker_path = folder / MARKER
    verify(folder, expected={
        "kind": kind,
        "status": "passed",
        "protocol_fingerprint": fingerprint(protocol),
        "dataset_fingerprint": dataset_fingerprint,
    })
    return frozen({"path": marker_path.relative_to(root).as_posix(), "sha256": file_sha256(marker_path)})


@contract
def load_phase2_artifact(
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
    """Authenticate a Phase 2 artifact on disk and return (ref, frozen body, frozen marker)."""
    root = Path(root).resolve()
    ref = normalize_ref(root, reference)
    require_phase2_namespace(Path(ref["path"]).parent.as_posix())
    marker_path = within(root, ref["path"])
    check(file_sha256(marker_path) == ref["sha256"], "Phase 2 completion manifest checksum mismatch")
    expected_meta = {
        "kind": expected_kind,
        "status": "passed",
        "protocol_fingerprint": fingerprint(protocol),
    }
    if expected_dataset is not None:
        expected_meta["dataset_fingerprint"] = expected_dataset
    marker = verify(marker_path.parent, expected=expected_meta)
    check("body.json" in marker["files"], "Phase 2 artifact missing body.json")
    body = json.loads((marker_path.parent / "body.json").read_text(encoding="utf-8"))
    if expected_schema is not None:
        check(body.get("schema") == expected_schema,
              f"Phase 2 schema mismatch: expected {expected_schema!r}, got {body.get('schema')!r}")
    if expected_plan_identity is not None:
        check(body.get("plan_identity") == expected_plan_identity, "Phase 2 plan_identity mismatch")
    if expected_owner is not None:
        owner_val = body.get("owner", body.get("model"))
        check(owner_val == expected_owner, f"Foreign Phase 2 owner: expected {expected_owner!r}, got {owner_val!r}")
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
    ref, body, marker = load_phase2_artifact(
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
        check(plain(body.get("inputs")) == plain(expected_inputs),
              "Resume artifact inputs do not match current run inputs")
    return ref, body, marker


@contract
def storage_probe(root, protocol, dataset_fingerprint, producer):
    """Publish a probe under phase2/probes/, verify in a fresh subprocess, return receipt ref."""
    root = Path(root).resolve()
    payload_ref = publish_phase2(
        root,
        protocol,
        dataset_fingerprint,
        "phase2/probes",
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
            "src.phase2.schema",
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
        "Independent Phase 2 storage probe readback mismatch",
    )
    return publish_phase2(
        root,
        protocol,
        dataset_fingerprint,
        "phase2/probes",
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
    """Authenticate an existing two-process Phase 2 storage probe receipt."""
    ref, body, _ = load_phase2_artifact(
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
    load_phase2_artifact(
        root,
        protocol,
        body["payload"],
        KIND_PROBE_PAYLOAD,
        expected_schema=SCHEMA_PROBE,
        expected_dataset=dataset_fingerprint,
    )
    return ref, body


def _cli(argv=None):
    parser = argparse.ArgumentParser(description="Phase 2 independent storage probe reader.")
    parser.add_argument("command", choices=["probe-read"])
    parser.add_argument("--phase1-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--protocol", type=Path, default=REPO / "configs/protocol.json")
    args = parser.parse_args(argv)
    root = phase1_root(args.phase1_root)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    ref, _, _ = load_phase2_artifact(root, protocol, args.manifest, KIND_PROBE_PAYLOAD, expected_schema=SCHEMA_PROBE)
    print(json.dumps({"pid": os.getpid(), "payload": plain(ref)}), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_cli())
    except (ContractError, OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "blocked", "blockers": [str(exc)]}), flush=True)
        raise SystemExit(2)
