"""Read-only Phase 1 lineage binder and midpoint input resolvers for Phase 2.

Authenticates the accepted Phase 1 Stage 4 release and traverses its verified
Stage 3 registry, unflagged G0 gate, Stage 2 suite, and Stage 1 dataset
metadata without opening raw images, mutating Phase 1 artifacts, or rerunning
Phase 1 validators. CONFIRM is rejected before any IO.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Tuple

from src.common.contracts import LANGUAGES, ContractError, fingerprint
from src.common.storage import file_sha256, verify, within
from src.stage2.schema import check, contract, frozen, plain
from .schema import CONDITIONS, MODELS, POOLINGS, require_open_split

PHASE1_STAGE4_RELEASE = frozen({
    "path": "results/stage4/releases/20261119T064951Z-07e6f916a67f46fdb88c1812763534bc/_COMPLETE.json",
    "sha256": "eb5cad38deecdd34e57f88f54a8a4fd8c47d7686fc3d159c240cde657d27c94c",
})

PHASE1_STAGE5_SYNTHESIS = frozen({
    "path": "results/stage5/synthesis/20261119T094836Z-a93c8ec225f843f1994337b9b107e3d1/_COMPLETE.json",
    "sha256": "a9a59d0c48ff8d5e6c01d89f40b9a8455bbf610f2ee89b246311628fcb2d3540",
})


@dataclass(frozen=True)
class Phase1Binding:
    """Verified read-only Phase 1 provenance graph for Phase 2 workers."""

    root: Path
    protocol: Mapping[str, Any]
    stage4_release_ref: Mapping[str, str]
    stage4_release: Mapping[str, Any]
    stage4_plan: Any
    bootstrap_shape: Tuple[int, int]
    bootstrap_sha256: str
    midpoints: Mapping[str, int]
    dataset_ref: Mapping[str, str]
    data_plan_ref: Mapping[str, str]
    model_plan_refs: Mapping[str, Mapping[str, str]]
    suite_ref: Mapping[str, str]
    stage3_registry_ref: Mapping[str, str]
    g0_gate_ref: Mapping[str, str]
    g0_p_value: float
    dataset_fingerprint: str
    protocol_fingerprint: str
    lineage_identity: str


@contract
def bind_phase1(root, protocol, stage4_release_ref=None):
    """Open and authenticate the accepted Phase 1 Stage 4 release and lineage."""
    from src.stage4.plan import DEPTHS, load_bootstrap
    from src.stage5.interfaces import open_release

    resolved_root = Path(root).resolve()
    ref_arg = plain(PHASE1_STAGE4_RELEASE) if stage4_release_ref is None else plain(stage4_release_ref)
    run = open_release(resolved_root, protocol, ref_arg)
    plan = run.plan
    release = run.release
    fit = plan.fit
    store = plan.store

    # Verify frozen bootstrap matrix and compute deterministic content hash.
    draws = load_bootstrap(plan)
    bootstrap_sha256 = hashlib.sha256(draws.tobytes()).hexdigest()

    # Verify G0 gate metadata directly from the accepted registry binding.
    g0_body = store.json(release["g0_gate"], "body.json", "stage3_g0_gate")
    check(
        g0_body["flagged"] is False
        and g0_body["structural_checks"] == "passed"
        and 0.01 < float(g0_body["p"]) <= 1.0,
        "Accepted Phase 1 G0 gate must be unflagged and passed",
    )

    # Confirm CONFIRM is absent from plan targets and SCREEN IDs.
    check(
        all(target["split"] != "CONFIRM" for target in fit["targets"].values()),
        "CONFIRM targets must not enter preliminary binding",
    )
    check(
        all(fit["targets"][image_id]["split"] == "SCREEN" for image_id in plan["screen_ids"]),
        "SCREEN ID population mismatch",
    )

    # Resolve Stage 1 dataset manifest from verified Stage 2 data plan (no image IO).
    data_plan_ref = plain(fit["data_plan"])
    data_plan = store.json(data_plan_ref, "plan.json", "stage2_data_plan")
    dataset_ref = store.ref(data_plan["references"]["dataset"])
    dataset_marker_path = within(resolved_root, dataset_ref["path"])
    check(file_sha256(dataset_marker_path) == dataset_ref["sha256"], "Stage 1 dataset marker mismatch")
    store.marker(dataset_ref, "stage1_dataset")

    midpoints = {model: int(DEPTHS[model][1]) for model in MODELS}
    model_plan_refs = {model: plain(fit["model_plans"][model]) for model in MODELS}
    protocol_fp = fingerprint(protocol)
    dataset_fp = str(plan["dataset_fingerprint"])

    lineage_payload = {
        "stage4_release": plain(run.release_ref),
        "stage4_plan": plain(plan.reference),
        "stage3_registry": plain(release["stage3_registry"]),
        "stage3_plan": plain(fit.reference),
        "g0_gate": plain(release["g0_gate"]),
        "stage2_suite": plain(fit["suite"]),
        "stage2_data_plan": data_plan_ref,
        "stage2_model_plans": model_plan_refs,
        "stage1_dataset": dataset_ref,
        "dataset_fingerprint": dataset_fp,
        "protocol_fingerprint": protocol_fp,
        "bootstrap_sha256": bootstrap_sha256,
        "midpoints": midpoints,
    }

    return Phase1Binding(
        root=resolved_root,
        protocol=frozen(protocol),
        stage4_release_ref=run.release_ref,
        stage4_release=release,
        stage4_plan=plan,
        bootstrap_shape=tuple(int(x) for x in draws.shape),
        bootstrap_sha256=bootstrap_sha256,
        midpoints=frozen(midpoints),
        dataset_ref=frozen(dataset_ref),
        data_plan_ref=frozen(data_plan_ref),
        model_plan_refs=frozen(model_plan_refs),
        suite_ref=frozen(plain(fit["suite"])),
        stage3_registry_ref=frozen(plain(release["stage3_registry"])),
        g0_gate_ref=frozen(plain(release["g0_gate"])),
        g0_p_value=float(g0_body["p"]),
        dataset_fingerprint=dataset_fp,
        protocol_fingerprint=protocol_fp,
        lineage_identity=fingerprint(lineage_payload),
    )


def midpoint_depth(binding, model):
    check(model in MODELS, f"Unknown model selector: {model!r}")
    return int(binding.midpoints[model])


@contract
def midpoint_groups(binding, model, *, condition="image_caption", pooling="last_caption"):
    """Return {language: group_id} for the four language groups at the fixed midpoint."""
    check(model in MODELS, f"Unknown model selector: {model!r}")
    check(condition in CONDITIONS, f"Unknown decoder condition: {condition!r}")
    check(pooling in POOLINGS, f"Unknown caption pooling: {pooling!r}")
    site = f"decoder:{midpoint_depth(binding, model)}:{pooling}"
    groups = {}
    for gid, group in binding.stage4_plan.fit["groups"].items():
        if (
            group["source"] == "vlm"
            and group["model"] == model
            and group["condition"] == condition
            and group["site"] == site
        ):
            lang = group["language"]
            check(lang in LANGUAGES and lang not in groups, f"Duplicate or invalid midpoint group: {lang!r}")
            groups[lang] = gid
    check(set(groups) == set(LANGUAGES), f"Incomplete four-language midpoint groups for {model}/{condition}/{site}")
    return frozen({lang: groups[lang] for lang in LANGUAGES})


def _block_lookup(binding, model, condition, pooling, split):
    require_open_split(split)
    gids = midpoint_groups(binding, model, condition=condition, pooling=pooling)
    gid_to_lang = {gid: lang for lang, gid in gids.items()}
    plan = binding.stage4_plan
    predictions = binding.stage4_release["predictions"]
    found = {}
    for bid, block in plan["blocks"].items():
        if block["owner"] != model or block["split"] != split:
            continue
        target_gid = block["target_group"]
        if target_gid not in gid_to_lang:
            continue
        target_lang = gid_to_lang[target_gid]
        check(target_lang not in found, f"Duplicate Stage 4 block for {model}/{condition}/{pooling}/{split}/{target_lang}")
        routes = {}
        for route in block["routes"]:
            src_gid = route["source_group"]
            src_lang = plan.fit["groups"][src_gid]["language"]
            check(src_lang in LANGUAGES and src_lang not in routes, "Invalid route source language")
            routes[src_lang] = route["route_id"]
        found[target_lang] = {
            "block_id": bid,
            "target_group": target_gid,
            "prediction_ref": plain(predictions[bid]),
            "routes": routes,
        }
    check(set(found) == set(LANGUAGES), f"Missing Stage 4 {split} blocks for {model}/{condition}/{pooling}")
    return found


@contract
def e29_screen_blocks(binding, model):
    """Resolve the 16 accepted Stage 4 midpoint SCREEN blocks per model for E29."""
    check(model in MODELS, f"Unknown model selector: {model!r}")
    result = {}
    for condition in CONDITIONS:
        result[condition] = {}
        for pooling in POOLINGS:
            blocks = _block_lookup(binding, model, condition, pooling, "SCREEN")
            for target_lang, entry in blocks.items():
                check(set(entry["routes"]) == set(LANGUAGES),
                      f"Incomplete 4-source routes in E29 block {model}/{condition}/{pooling}/{target_lang}")
            result[condition][pooling] = blocks
    return frozen(result)


def _source_system_record(binding, gid):
    plan = binding.stage4_plan
    fit = plan.fit
    store = plan.store
    ordinary = plan["ordinary"][gid]
    candidate_body = store.json(ordinary["candidate"], "body.json", "stage3_candidate")
    check(
        candidate_body["group_id"] == gid and candidate_body["assignment_id"] == "true",
        "Ordinary candidate group/assignment mismatch",
    )
    return {
        "group_id": gid,
        "selection_ref": plain(ordinary["selection"]),
        "candidate_ref": plain(ordinary["candidate"]),
        "C": float(ordinary["C"]),
        "statistics_ref": plain(candidate_body["statistics"]),
        "dimension": int(fit["groups"][gid]["dimension"]),
        "site": fit["groups"][gid]["site"],
        "model_plan_ref": plain(fit["groups"][gid]["model_plan"]),
    }


@contract
def e17_inputs(binding, model):
    """Resolve all reused Phase 1 inputs and the 12 new off-diagonal DEV-C targets for E17."""
    check(model in MODELS, f"Unknown model selector: {model!r}")
    gids = midpoint_groups(binding, model, condition="image_caption", pooling="last_caption")
    screen_blocks = _block_lookup(binding, model, "image_caption", "last_caption", "SCREEN")
    dev_c_blocks = _block_lookup(binding, model, "image_caption", "last_caption", "DEV-C")

    dev_c_diagonals = {}
    dev_c_feature_releases = {}
    source_systems = {}
    off_diagonal_targets = {}
    fit_groups = binding.stage4_plan.fit["groups"]

    for lang in LANGUAGES:
        gid = gids[lang]
        diag_block = dev_c_blocks[lang]
        check(set(diag_block["routes"]) == {lang}, f"Expected diagonal-only Stage 4 DEV-C block for {model}/{lang}")
        dev_c_diagonals[lang] = {
            "block_id": diag_block["block_id"],
            "prediction_ref": diag_block["prediction_ref"],
            "route_id": diag_block["routes"][lang],
            "group_id": gid,
        }
        group = fit_groups[gid]
        dev_c_feature_releases[lang] = {
            "group_id": gid,
            "release_ref": plain(group["releases"]["DEV-C"]),
            "dimension": int(group["dimension"]),
            "site": group["site"],
            "model_plan_ref": plain(group["model_plan"]),
        }
        source_systems[lang] = _source_system_record(binding, gid)
        off_diagonal_targets[lang] = tuple(src for src in LANGUAGES if src != lang)

    return frozen({
        "model": model,
        "midpoint_depth": midpoint_depth(binding, model),
        "condition": "image_caption",
        "pooling": "last_caption",
        "screen_blocks": screen_blocks,
        "dev_c_diagonals": dev_c_diagonals,
        "dev_c_feature_releases": dev_c_feature_releases,
        "source_systems": source_systems,
        "off_diagonal_targets": off_diagonal_targets,
    })


@contract
def e11_fit_statistics_refs(binding, model):
    """Resolve reused Stage 3 FIT statistics and SCREEN caption_mean inputs for conditional E11."""
    check(model in MODELS, f"Unknown model selector: {model!r}")
    gids = midpoint_groups(binding, model, condition="image_caption", pooling="caption_mean")
    screen_blocks = _block_lookup(binding, model, "image_caption", "caption_mean", "SCREEN")
    fit_groups = binding.stage4_plan.fit["groups"]
    out = {}
    for lang in LANGUAGES:
        gid = gids[lang]
        sys_rec = _source_system_record(binding, gid)
        out[lang] = {
            **sys_rec,
            "screen_block_id": screen_blocks[lang]["block_id"],
            "screen_prediction_ref": screen_blocks[lang]["prediction_ref"],
            "screen_routes": screen_blocks[lang]["routes"],
            "screen_feature_release_ref": plain(fit_groups[gid]["releases"]["SCREEN"]),
        }
    return frozen(out)


@contract
def stage1_caption_paths(binding):
    """Verify the Stage 1 dataset marker and return read-only caption/metadata file paths (no image IO)."""
    marker_path = within(binding.root, binding.dataset_ref["path"])
    check(file_sha256(marker_path) == binding.dataset_ref["sha256"], "Stage 1 dataset manifest changed")
    folder = marker_path.parent
    marker = verify(folder, expected={
        "kind": "stage1_dataset",
        "status": "passed",
        "protocol_fingerprint": binding.protocol_fingerprint,
        "dataset_fingerprint": binding.dataset_fingerprint,
    })
    required = ["dataset.json", "records.jsonl", "targets.jsonl", "splits.json"]
    required += [f"captions/{lang}.jsonl" for lang in LANGUAGES]
    for rel in required:
        check(rel in marker["files"], f"Stage 1 dataset missing required file: {rel}")
    return frozen({
        "dataset_dir": folder.relative_to(binding.root).as_posix(),
        "dataset_ref": plain(binding.dataset_ref),
        "files": {rel: {"path": (folder / rel).relative_to(binding.root).as_posix(),
                        "sha256": marker["files"][rel]} for rel in required},
    })


@contract
def read_verified_midpoint_features(binding, model, condition, pooling, target_lang, split):
    """Read one verified Stage 2 numeric feature matrix at midpoint; CONFIRM is rejected."""
    from src.stage3.data import _read_verified_group

    require_open_split(split)
    check(target_lang in LANGUAGES, f"Unknown language: {target_lang!r}")
    gid = midpoint_groups(binding, model, condition=condition, pooling=pooling)[target_lang]
    return _read_verified_group(binding.stage4_plan.fit, binding.stage4_plan.store, gid, split)


@contract
def load_frozen_source_head(binding, model, condition, pooling, source_lang):
    """Load frozen Stage 3 ordinary (coef, intercept, estimable, prevalence, stats) without FIT arrays."""
    from src.stage4.predictions import _selected_parameters

    check(source_lang in LANGUAGES, f"Unknown source language: {source_lang!r}")
    gid = midpoint_groups(binding, model, condition=condition, pooling=pooling)[source_lang]
    return _selected_parameters(binding.stage4_plan, gid)
