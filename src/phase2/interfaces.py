"""Frozen Phase 2 Stage 1 public API for parallel worker branches.

Chats 2 (E07 audit), 3 (E17 DEV-C preflight), and 4 (E29 gates + Stage 1 plan)
import shared symbols exclusively from this module.
"""

from src.stage2.schema import REPO, check, contract, frozen, plain
from .inputs import (
    PHASE1_STAGE4_RELEASE,
    PHASE1_STAGE5_SYNTHESIS,
    Phase1Binding,
    bind_phase1,
    e11_fit_statistics_refs,
    e17_inputs,
    e29_screen_blocks,
    load_frozen_source_head,
    midpoint_depth,
    midpoint_groups,
    read_verified_midpoint_features,
    stage1_caption_paths,
)
from .schema import (
    ALLOWED_SPLITS,
    CLOSED_SPLITS,
    CONDITIONS,
    E07_VERDICTS,
    GATE_DECISIONS,
    KIND_ANALYSIS,
    KIND_CALIBRATORS,
    KIND_E07_AUDIT,
    KIND_E07_CHUNK,
    KIND_E07_RELEASE,
    KIND_E17_AUTH,
    KIND_E29_GATE,
    KIND_PLAN,
    KIND_PREDICTION,
    KIND_PROBE,
    KIND_PROBE_PAYLOAD,
    KIND_RELEASE,
    MODELS,
    PHASE2_NAMESPACE,
    PHASE2_PARENT_DIRS,
    PHASE2_VERSION,
    POOLINGS,
    SCHEMA_ANALYSIS,
    SCHEMA_CALIBRATORS,
    SCHEMA_E07_AUDIT,
    SCHEMA_E07_FEATURES,
    SCHEMA_E17_AUTH,
    SCHEMA_E29_GATE,
    SCHEMA_PLAN,
    SCHEMA_PREDICTION,
    SCHEMA_PROBE,
    SCHEMA_RELEASE,
    STATUSES,
    SUPERVISION_LABELS,
    init_phase2_dirs,
    load_phase2_artifact,
    normalize_ref,
    publish_phase2,
    require_open_split,
    require_phase2_namespace,
    storage_probe,
    verify_resume_ref,
    verify_storage_probe,
)


def current_producer(scope="local-development"):
    """Delegate to Stage 3's pinned CPU runtime/git producer metadata helper."""
    from src.stage3.operations import current_producer as _stage3_producer

    return _stage3_producer(scope=scope)
