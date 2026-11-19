"""Phase 2 Stage 1 plan preparation, authentication, and Stage 1 bundle verification.

Implements `phase2_plan` (`phase2.plan.v1`) without importing sibling worker
modules (`e07_audit.py`, `e17_auth.py`); optional E07 and E17 artifacts are
authenticated strictly through the frozen schemas in `src.phase2.interfaces`.
"""

from collections.abc import Mapping
from pathlib import Path
import time

from src.common.contracts import LANGUAGES, fingerprint, require_sha256
from src.phase2.interfaces import (
    CONDITIONS,
    E07_VERDICTS,
    KIND_E07_AUDIT,
    KIND_E17_AUTH,
    KIND_PLAN,
    MODELS,
    PHASE2_PARENT_DIRS,
    PHASE2_VERSION,
    POOLINGS,
    SCHEMA_E07_AUDIT,
    SCHEMA_E17_AUTH,
    SCHEMA_PLAN,
    bind_phase1,
    check,
    contract,
    e11_fit_statistics_refs,
    e29_screen_blocks,
    frozen,
    init_phase2_dirs,
    load_phase2_artifact,
    plain,
    publish_phase2,
    storage_probe,
    verify_storage_probe,
)
from .gates import E29_GATE_THRESHOLDS, STAGE5_DECISION_RANGES

E17_DEV_C_PERMISSION = "phase2.e17_dev_c_transfer"

REQUIRED_PLAN_KEYS = frozenset({
    "schema",
    "phase2_version",
    "plan_identity",
    "lineage_identity",
    "dataset_fingerprint",
    "protocol_fingerprint",
    "stage4_release",
    "stage4_plan",
    "stage3_registry",
    "g0_gate",
    "g0_p_value",
    "stage2_suite",
    "stage1_dataset",
    "bootstrap",
    "midpoints",
    "storage_probe",
    "initialized_dirs",
    "e29_blocks",
    "e11_fit_statistics",
    "gate_thresholds",
    "stage5_decision_ranges",
    "e07_audit",
    "e17_auth",
    "confirm_excluded",
})

REQUIRED_E07_AUDIT_KEYS = frozenset({
    "schema",
    "stage4_release",
    "lineage_identity",
    "dataset_ref",
    "dataset_fingerprint",
    "draw0_verified",
    "confirm_excluded",
    "verdict",
    "verdict_reasons",
    "screen_multiplicity",
    "fit_dev_s_multiplicity",
    "draw_buckets",
    "support_tables",
    "unique_changed_views",
    "extraction_plan",
})

REQUIRED_E17_AUTH_KEYS = frozenset({
    "schema",
    "stage4_release",
    "lineage_identity",
    "dataset_fingerprint",
    "permission",
    "confirm_excluded",
    "models",
    "calibrator_recipe",
})


class Phase2Plan(Mapping):
    """Immutable authenticated Phase 2 Stage 1 plan."""

    def __init__(self, body, reference, root, protocol, binding=None):
        self._body = frozen(body)
        self.reference = frozen(reference)
        self.root = Path(root).resolve()
        self.protocol = frozen(protocol)
        self.binding = binding

    def __getitem__(self, key):
        return self._body[key]

    def __iter__(self):
        return iter(self._body)

    def __len__(self):
        return len(self._body)


def _emit(progress, event):
    if progress is not None:
        progress(event)


def _validate_e29_blocks_structure(e29_blocks):
    check(isinstance(e29_blocks, Mapping) and set(e29_blocks.keys()) == set(MODELS),
          "phase2_plan e29_blocks must cover all models")
    for model in MODELS:
        by_cond = e29_blocks[model]
        check(isinstance(by_cond, Mapping) and set(by_cond.keys()) == set(CONDITIONS),
              f"phase2_plan e29_blocks[{model}] must cover {CONDITIONS}")
        count = 0
        for condition in CONDITIONS:
            by_pool = by_cond[condition]
            check(isinstance(by_pool, Mapping) and set(by_pool.keys()) == set(POOLINGS),
                  f"phase2_plan e29_blocks[{model}][{condition}] must cover {POOLINGS}")
            for pooling in POOLINGS:
                by_lang = by_pool[pooling]
                check(isinstance(by_lang, Mapping) and set(by_lang.keys()) == set(LANGUAGES),
                      f"phase2_plan e29_blocks[{model}][{condition}][{pooling}] must cover {LANGUAGES}")
                count += len(by_lang)
        check(count == 16, f"Expected 16 E29 midpoint SCREEN blocks for {model}, got {count}")


def _verify_e07_audit_ref(
    root,
    protocol,
    reference,
    *,
    expected_dataset,
    expected_stage4_release,
    expected_lineage_identity,
):
    """Verify a published `phase2_e07_audit` purely via the frozen wire schema."""
    ref, body, marker = load_phase2_artifact(
        root,
        protocol,
        reference,
        KIND_E07_AUDIT,
        expected_schema=SCHEMA_E07_AUDIT,
        expected_dataset=expected_dataset,
    )
    missing = REQUIRED_E07_AUDIT_KEYS - set(body.keys())
    check(not missing, f"phase2_e07_audit missing required keys: {sorted(missing)}")
    check(body["dataset_fingerprint"] == expected_dataset, "E07 audit dataset_fingerprint mismatch")
    check(plain(body["stage4_release"]) == plain(expected_stage4_release),
          "E07 audit stage4_release mismatch")
    check(body["lineage_identity"] == expected_lineage_identity,
          "E07 audit lineage_identity mismatch")
    check(body["draw0_verified"] is True, "E07 audit draw0_verified must be True")
    check(body["confirm_excluded"] is True, "E07 audit confirm_excluded must be True")
    check(body["verdict"] in E07_VERDICTS, f"Invalid E07 audit verdict: {body['verdict']!r}")
    check(isinstance(body["verdict_reasons"], (list, tuple)), "E07 audit verdict_reasons must be a sequence")
    if body["verdict"] == "blocked":
        check(body["extraction_plan"] is None, "Blocked E07 audit must not carry an extraction_plan")
    else:
        check(body["extraction_plan"] is not None, "Passed E07 audit must include extraction_plan")
    return ref, body, marker


def _verify_e17_auth_ref(
    root,
    protocol,
    reference,
    *,
    expected_dataset,
    expected_stage4_release,
    expected_lineage_identity,
):
    """Verify a published `phase2_e17_auth` purely via the frozen wire schema."""
    ref, body, marker = load_phase2_artifact(
        root,
        protocol,
        reference,
        KIND_E17_AUTH,
        expected_schema=SCHEMA_E17_AUTH,
        expected_dataset=expected_dataset,
    )
    missing = REQUIRED_E17_AUTH_KEYS - set(body.keys())
    check(not missing, f"phase2_e17_auth missing required keys: {sorted(missing)}")
    check(body["dataset_fingerprint"] == expected_dataset, "E17 auth dataset_fingerprint mismatch")
    check(plain(body["stage4_release"]) == plain(expected_stage4_release),
          "E17 auth stage4_release mismatch")
    check(body["lineage_identity"] == expected_lineage_identity,
          "E17 auth lineage_identity mismatch")
    check(body["permission"] == E17_DEV_C_PERMISSION,
          f"Unexpected E17 permission: {body['permission']!r}")
    check(body["confirm_excluded"] is True, "E17 auth confirm_excluded must be True")
    models_section = body["models"]
    check(isinstance(models_section, Mapping) and set(models_section.keys()) == set(MODELS),
          "E17 auth models section must cover qwen and gemma")
    for model in MODELS:
        m = models_section[model]
        for key in (
            "midpoint_depth",
            "screen_blocks",
            "dev_c_diagonals",
            "off_diagonal_blocks",
            "source_systems",
            "diagonal_reproduction_check",
        ):
            check(key in m, f"E17 auth models[{model}] missing: {key}")
        diag_check = m["diagonal_reproduction_check"]
        check(
            diag_check.get("passed") is True
            and float(diag_check.get("max_abs_diff", -1.0)) == 0.0
            and float(diag_check.get("atol", -1.0)) == 0.0
            and float(diag_check.get("rtol", -1.0)) == 0.0,
            f"E17 auth diagonal reproduction check failed for {model}",
        )
    return ref, body, marker


def _plan_identity(body_without_identity):
    return fingerprint(plain(body_without_identity))


@contract
def prepare_phase2(
    root,
    protocol,
    stage4_release_ref,
    producer,
    *,
    e07_audit_ref=None,
    e17_auth_ref=None,
    progress=None,
):
    """Bind Phase 1, run storage probe, pre-init dirs, and publish immutable `phase2_plan`."""
    resolved_root = Path(root).resolve()
    total_steps = 5

    t0 = time.monotonic()
    binding = bind_phase1(resolved_root, protocol, stage4_release_ref)
    _emit(progress, {
        "event": "phase2_prepare_step",
        "unit": "bind_phase1",
        "completed": 1,
        "total": total_steps,
        "read_hash_s": round(time.monotonic() - t0, 4),
        "lineage_identity": binding.lineage_identity,
    })

    t1 = time.monotonic()
    initialized_dirs = init_phase2_dirs(resolved_root)
    _emit(progress, {
        "event": "phase2_prepare_step",
        "unit": "init_phase2_dirs",
        "completed": 2,
        "total": total_steps,
        "write_s": round(time.monotonic() - t1, 4),
        "dir_count": len(initialized_dirs),
    })

    t2 = time.monotonic()
    probe_ref = storage_probe(resolved_root, protocol, binding.dataset_fingerprint, producer)
    verify_storage_probe(resolved_root, protocol, probe_ref, binding.dataset_fingerprint)
    _emit(progress, {
        "event": "phase2_prepare_step",
        "unit": "storage_probe",
        "completed": 3,
        "total": total_steps,
        "write_s": round(time.monotonic() - t2, 4),
        "probe_ref": plain(probe_ref),
    })

    t3 = time.monotonic()
    e29_blocks = {model: plain(e29_screen_blocks(binding, model)) for model in MODELS}
    _validate_e29_blocks_structure(e29_blocks)
    e11_stats = {model: plain(e11_fit_statistics_refs(binding, model)) for model in MODELS}

    verified_e07_ref = None
    if e07_audit_ref is not None:
        verified_e07_ref, _, _ = _verify_e07_audit_ref(
            resolved_root,
            protocol,
            e07_audit_ref,
            expected_dataset=binding.dataset_fingerprint,
            expected_stage4_release=binding.stage4_release_ref,
            expected_lineage_identity=binding.lineage_identity,
        )

    verified_e17_ref = None
    if e17_auth_ref is not None:
        verified_e17_ref, _, _ = _verify_e17_auth_ref(
            resolved_root,
            protocol,
            e17_auth_ref,
            expected_dataset=binding.dataset_fingerprint,
            expected_stage4_release=binding.stage4_release_ref,
            expected_lineage_identity=binding.lineage_identity,
        )
    _emit(progress, {
        "event": "phase2_prepare_step",
        "unit": "bind_inputs_and_optional_refs",
        "completed": 4,
        "total": total_steps,
        "compute_s": round(time.monotonic() - t3, 4),
        "e07_bound": verified_e07_ref is not None,
        "e17_bound": verified_e17_ref is not None,
    })

    core_body = {
        "schema": SCHEMA_PLAN,
        "phase2_version": PHASE2_VERSION,
        "lineage_identity": binding.lineage_identity,
        "dataset_fingerprint": binding.dataset_fingerprint,
        "protocol_fingerprint": binding.protocol_fingerprint,
        "stage4_release": plain(binding.stage4_release_ref),
        "stage4_plan": plain(binding.stage4_plan.reference),
        "stage3_registry": plain(binding.stage3_registry_ref),
        "g0_gate": plain(binding.g0_gate_ref),
        "g0_p_value": float(binding.g0_p_value),
        "stage2_suite": plain(binding.suite_ref),
        "stage1_dataset": plain(binding.dataset_ref),
        "bootstrap": {
            "shape": [int(x) for x in binding.bootstrap_shape],
            "sha256": binding.bootstrap_sha256,
        },
        "midpoints": plain(binding.midpoints),
        "storage_probe": plain(probe_ref),
        "initialized_dirs": list(initialized_dirs),
        "e29_blocks": e29_blocks,
        "e11_fit_statistics": e11_stats,
        "gate_thresholds": plain(E29_GATE_THRESHOLDS),
        "stage5_decision_ranges": plain(STAGE5_DECISION_RANGES),
        "e07_audit": plain(verified_e07_ref) if verified_e07_ref is not None else None,
        "e17_auth": plain(verified_e17_ref) if verified_e17_ref is not None else None,
        "confirm_excluded": True,
    }
    plan_id = _plan_identity(core_body)
    body = dict(core_body, plan_identity=plan_id)

    upstream = [plain(binding.stage4_release_ref), plain(probe_ref)]
    if verified_e07_ref is not None:
        upstream.append(plain(verified_e07_ref))
    if verified_e17_ref is not None:
        upstream.append(plain(verified_e17_ref))

    t4 = time.monotonic()
    plan_ref = publish_phase2(
        resolved_root,
        protocol,
        binding.dataset_fingerprint,
        "phase2/plans",
        KIND_PLAN,
        body,
        producer,
        upstream=upstream,
    )
    loaded = load_phase2_plan(resolved_root, protocol, plan_ref)
    check(plain(loaded) == plain(body), "Fresh phase2_plan readback content mismatch")
    _emit(progress, {
        "event": "phase2_prepare_step",
        "unit": "publish_and_readback_plan",
        "completed": 5,
        "total": total_steps,
        "write_s": round(time.monotonic() - t4, 4),
        "plan_ref": plain(plan_ref),
        "plan_identity": plan_id,
    })
    return Phase2Plan(body, plan_ref, resolved_root, protocol, binding=binding)


@contract
def load_phase2_plan(root, protocol, manifest):
    """Authenticate an immutable `phase2_plan` artifact on disk."""
    resolved_root = Path(root).resolve()
    ref, body, marker = load_phase2_artifact(
        resolved_root,
        protocol,
        manifest,
        KIND_PLAN,
        expected_schema=SCHEMA_PLAN,
    )
    missing = REQUIRED_PLAN_KEYS - set(body.keys())
    check(not missing, f"phase2_plan missing required keys: {sorted(missing)}")
    check(body["phase2_version"] == PHASE2_VERSION,
          f"Unexpected phase2_version: {body['phase2_version']!r}")
    check(body["confirm_excluded"] is True, "phase2_plan must enforce confirm_excluded=True")
    check(body["protocol_fingerprint"] == fingerprint(protocol),
          "phase2_plan protocol_fingerprint mismatch")
    check(body["dataset_fingerprint"] == marker["dataset_fingerprint"],
          "phase2_plan dataset_fingerprint mismatch with marker")
    require_sha256(body["lineage_identity"])
    require_sha256(body["plan_identity"])
    require_sha256(body["bootstrap"]["sha256"])
    check(
        isinstance(body["bootstrap"]["shape"], (list, tuple))
        and len(body["bootstrap"]["shape"]) == 2
        and all(type(x) is int and x > 0 for x in body["bootstrap"]["shape"]),
        "Invalid bootstrap shape in phase2_plan",
    )

    body_plain = plain(body)
    core = {k: v for k, v in body_plain.items() if k != "plan_identity"}
    check(_plan_identity(core) == body["plan_identity"], "phase2_plan plan_identity tampered or inconsistent")

    check(body_plain["gate_thresholds"] == plain(E29_GATE_THRESHOLDS),
          "phase2_plan gate_thresholds do not match frozen E29_GATE_THRESHOLDS")
    check(body_plain["stage5_decision_ranges"] == plain(STAGE5_DECISION_RANGES),
          "phase2_plan stage5_decision_ranges do not match frozen STAGE5_DECISION_RANGES")
    check(tuple(body["initialized_dirs"]) == PHASE2_PARENT_DIRS,
          "phase2_plan initialized_dirs mismatch")
    _validate_e29_blocks_structure(body["e29_blocks"])

    verify_storage_probe(resolved_root, protocol, body["storage_probe"], body["dataset_fingerprint"])

    if body["e07_audit"] is not None:
        _verify_e07_audit_ref(
            resolved_root,
            protocol,
            body["e07_audit"],
            expected_dataset=body["dataset_fingerprint"],
            expected_stage4_release=body["stage4_release"],
            expected_lineage_identity=body["lineage_identity"],
        )
    if body["e17_auth"] is not None:
        _verify_e17_auth_ref(
            resolved_root,
            protocol,
            body["e17_auth"],
            expected_dataset=body["dataset_fingerprint"],
            expected_stage4_release=body["stage4_release"],
            expected_lineage_identity=body["lineage_identity"],
        )

    return Phase2Plan(body, ref, resolved_root, protocol)


@contract
def verify_stage1_bundle(root, protocol, plan_ref, e07_audit_ref, e17_auth_ref):
    """Verify that `phase2_plan`, `phase2_e07_audit`, and `phase2_e17_auth` share identical Phase 1 lineage."""
    resolved_root = Path(root).resolve()
    plan = load_phase2_plan(resolved_root, protocol, plan_ref)

    e07_ref, e07_body, _ = _verify_e07_audit_ref(
        resolved_root,
        protocol,
        e07_audit_ref,
        expected_dataset=plan["dataset_fingerprint"],
        expected_stage4_release=plan["stage4_release"],
        expected_lineage_identity=plan["lineage_identity"],
    )
    e17_ref, e17_body, _ = _verify_e17_auth_ref(
        resolved_root,
        protocol,
        e17_auth_ref,
        expected_dataset=plan["dataset_fingerprint"],
        expected_stage4_release=plan["stage4_release"],
        expected_lineage_identity=plan["lineage_identity"],
    )

    if plan["e07_audit"] is not None:
        check(plain(e07_ref) == plain(plan["e07_audit"]),
              "Supplied e07_audit reference differs from reference pinned in phase2_plan")
    if plan["e17_auth"] is not None:
        check(plain(e17_ref) == plain(plan["e17_auth"]),
              "Supplied e17_auth reference differs from reference pinned in phase2_plan")

    return frozen({
        "status": "passed",
        "stage1_complete": True,
        "phase2_version": plan["phase2_version"],
        "lineage_identity": plan["lineage_identity"],
        "plan_identity": plan["plan_identity"],
        "dataset_fingerprint": plan["dataset_fingerprint"],
        "protocol_fingerprint": plan["protocol_fingerprint"],
        "stage4_release": plain(plan["stage4_release"]),
        "plan": plain(plan.reference),
        "e07_audit": plain(e07_ref),
        "e07_verdict": e07_body["verdict"],
        "e17_auth": plain(e17_ref),
        "e17_permission": e17_body["permission"],
        "e17_diagonal_checks_passed": True,
        "confirm_excluded": True,
        "parallel_stage2_ready": {
            "E29": True,
            "E17": True,
            "E07_GPU": e07_body["verdict"] != "blocked",
        },
    })
