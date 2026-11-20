"""Phase 3 GPU Track Stage B1: pre-GPU CPU prep, E39+E31 & E23 specs, and plan publisher.

Implements Stage B1 (`phase3-gpu-stageB1.v1`, with G7 `E35` explicitly dropped so Track B
stays strictly focused on `E39+E31` and `E23` within `G9-Q6`):
- Authenticates Phase 1 Stage 4 release (`bind_phase1`), optional/certified Stage A1
  `phase3_cpu_plan`, and Stage 1 caption/target/split metadata with zero raw image IO.
- Builds deterministic `E23` disjoint opposite-label `SCREEN` image pairs (label/metadata-only,
  zero model scores accessed) and deduplicated directed swapped-image extraction chunks.
- Freezes `E39+E31` 8-slice sub-caption token extraction + decoder visual-token + attention
  mass spec and 12-chunk `SCREEN`-only extraction manifest (`confirm_rows = 0`).
- Publishes immutable `phase3_gpu_plan` under `phase3/plans/`.
"""

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import time
import unicodedata

from src.common.contracts import LANGUAGES, ContractError, ModelInput, TargetRecord, fingerprint
from src.common.storage import file_sha256, phase1_root, within
from src.phase2.interfaces import (
    PHASE1_STAGE4_RELEASE,
    bind_phase1,
    current_producer,
    midpoint_depth,
    midpoint_groups,
    stage1_caption_paths,
)
from .cpu_plan import load_phase3_cpu_plan
from .schema import (
    CONDITIONS,
    KIND_GPU_PLAN,
    MODELS,
    PHASE3_GPU_VERSION,
    POOLINGS,
    REPO,
    SCHEMA_GPU_PLAN,
    check,
    contract,
    frozen,
    init_phase3_dirs,
    load_phase3_artifact,
    plain,
    publish_phase3,
    require_open_split,
    storage_probe,
    verify_storage_probe,
)

CERTIFIED_STAGE_A1_CPU_PLAN = frozen({
    "path": "phase3/plans/20261120T020039Z-0d82ded3b7c3497e999d70a33f35c3a8/_COMPLETE.json",
    "sha256": "93cbf480e9d8c13401aaa98315444f3a4d2b3b57ec769b458a799dc269c40439",
})

CERTIFIED_STAGE_B1_GPU_PLAN = frozen({
    "path": "phase3/plans/20261120T060305Z-11b534d878f84475a50b561efbd06650/_COMPLETE.json",
    "sha256": "3603f36631ec8165445613e08caf2a3b74bd7626f84f79fdde59fc34f0088b1c",
})

E39_SITES = (
    "first_caption",
    "middle_caption",
    "penultimate_caption",
    "last_non_punct_caption",
    "first_half_mean",
    "second_half_mean",
    "last_caption",
    "caption_mean",
)
E31_SITES = (
    "decoder_visual_mean",
    "attention_from_last_caption",
    "attention_from_caption_mean",
)

E39_E31_CHUNK_SIZE_IMAGES = 32
E23_CHUNK_SIZE_PAIRS = 32
E23_PAIR_SEED_PREFIX = "G9Q6-E23-pair-v1"

PHASE3_GPU_DECISION_RANGES = frozen({
    "E39": {
        "terminal_token_recovery_fraction_min": 0.50,
        "auroc_gain_min": 0.015,
        "negligible_ci_within": (-0.005, 0.005),
        "checks": {"stage2_draw0_parity_tol": 1e-4},
    },
    "E31": {
        "decoder_visual_auroc_min": 0.90,
        "attention_gap_corr_min": 0.30,
        "checks": {"attention_mass_sum_tol": 1e-4},
    },
    "E23": {
        "signed_shift_toward_positive_min": 0.25,
        "primary_min_pairs": 20,
        "negligible_shift_ci_upper_lt": 0.05,
        "checks": {
            "zero_model_scores_in_pairing": True,
            "authentic_anchor_parity_tol": 1e-4,
        },
    },
})


class Phase3GpuPlan(Mapping):
    """Read-only authenticated Phase 3 GPU plan."""

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


def _has_non_punct_content(token_text):
    for ch in str(token_text):
        cat = unicodedata.category(ch)
        if cat and cat[0] not in ("P", "Z", "C"):
            return True
    return False


def select_e39_token_indices(token_texts):
    """Deterministically map L >= 1 caption tokens to all 8 E39 sub-caption sites."""
    tokens = tuple(token_texts)
    n_tok = len(tokens)
    check(n_tok >= 1, "E39 requires at least 1 caption token")

    first_idx = 0
    middle_idx = n_tok // 2
    penult_idx = max(0, n_tok - 2)
    last_idx = n_tok - 1

    last_non_punct_idx = None
    for idx in range(n_tok - 1, -1, -1):
        if _has_non_punct_content(tokens[idx]):
            last_non_punct_idx = idx
            break
    fallback_used = last_non_punct_idx is None
    if fallback_used:
        last_non_punct_idx = last_idx

    first_half_end = max(1, n_tok // 2)
    second_half_start = n_tok // 2

    return {
        "token_count": n_tok,
        "first_caption": first_idx,
        "middle_caption": middle_idx,
        "penultimate_caption": penult_idx,
        "last_non_punct_caption": last_non_punct_idx,
        "last_non_punct_fallback_used": fallback_used,
        "last_caption": last_idx,
        "first_half_span": [0, first_half_end],
        "second_half_span": [second_half_start, n_tok],
        "full_span": [0, n_tok],
    }


def _text_sha256(text):
    check(isinstance(text, str) and bool(text.strip()), "Nonempty caption text required")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_target_values(rec):
    if isinstance(rec, TargetRecord):
        return rec.values
    if isinstance(rec, Mapping):
        return tuple(rec["values"])
    return tuple(getattr(rec, "values"))


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_e23_pair_plan(
    screen_ids,
    screen_targets,
    draw0_records,
    category_order,
    primary_categories,
    min_screen_support,
):
    """Build deterministic E23 disjoint SCREEN opposite-label pairs & swapped chunks.

    Uses strictly SCREEN labels and metadata (zero model scores accessed).
    """
    ordered_screen = sorted(screen_ids)
    check(len(ordered_screen) > 0, "E23 requires nonempty SCREEN image IDs")
    cats = list(category_order)
    prim_set = set(primary_categories)

    if "6a20677eee04fd08" in screen_targets and "plant.n.02" in cats:
        plant_idx = cats.index("plant.n.02")
        check(
            _extract_target_values(screen_targets["6a20677eee04fd08"])[plant_idx] is None,
            "Invalid unknown cell mask on 6a20677eee04fd08 / plant.n.02",
        )

    per_category = {}
    directed_pairs = set()

    for idx, cat in enumerate(cats):
        pos_ids = []
        neg_ids = []
        unk_count = 0
        for iid in ordered_screen:
            val = _extract_target_values(screen_targets[iid])[idx]
            if val == 1:
                pos_ids.append(iid)
            elif val == 0:
                neg_ids.append(iid)
            elif val is None:
                unk_count += 1
            else:
                raise ContractError(f"Unexpected SCREEN target value {val!r} for {iid}/{cat}")

        pos_perm = sorted(
            pos_ids,
            key=lambda iid: (
                hashlib.sha256(f"{E23_PAIR_SEED_PREFIX}|{cat}|pos|{iid}".encode("utf-8")).hexdigest(),
                iid,
            ),
        )
        neg_perm = sorted(
            neg_ids,
            key=lambda iid: (
                hashlib.sha256(f"{E23_PAIR_SEED_PREFIX}|{cat}|neg|{iid}".encode("utf-8")).hexdigest(),
                iid,
            ),
        )
        k_pairs = min(len(pos_perm), len(neg_perm))
        pairs = [
            {
                "pair_index": i,
                "positive_image_id": pos_perm[i],
                "negative_image_id": neg_perm[i],
            }
            for i in range(k_pairs)
        ]
        for pair in pairs:
            p_id = pair["positive_image_id"]
            n_id = pair["negative_image_id"]
            directed_pairs.add((n_id, p_id))
            directed_pairs.add((p_id, n_id))

        status = (
            "ok"
            if k_pairs >= int(min_screen_support)
            else ("descriptive_low_support" if k_pairs > 0 else "undefined")
        )
        per_category[cat] = {
            "category": cat,
            "is_primary": cat in prim_set,
            "n_positive": len(pos_perm),
            "n_negative": len(neg_perm),
            "n_unknown": unk_count,
            "n_pairs": k_pairs,
            "support_status": status,
            "pairs": pairs,
            "unpaired_positive_ids": pos_perm[k_pairs:],
            "unpaired_negative_ids": neg_perm[k_pairs:],
        }

    unique_directed = sorted(directed_pairs)
    chunks = []
    total_views = 0
    for c_idx, offset in enumerate(range(0, len(unique_directed), E23_CHUNK_SIZE_PAIRS)):
        batch = unique_directed[offset : offset + E23_CHUNK_SIZE_PAIRS]
        rows = []
        for shown_id, owner_id in batch:
            for lang in LANGUAGES:
                shown_rec = draw0_records[(shown_id, lang)]
                owner_rec = draw0_records[(owner_id, lang)]
                rows.append({
                    "shown_image_id": shown_id,
                    "shown_image_path": shown_rec.image_path,
                    "shown_image_sha256": shown_rec.image_sha256,
                    "caption_owner_image_id": owner_id,
                    "language": lang,
                    "caption_id": owner_rec.caption_id,
                    "caption_draw_key": 0,
                    "caption_text": owner_rec.caption_text,
                    "caption_text_sha256": _text_sha256(owner_rec.caption_text),
                })
        total_views += len(rows)
        chunks.append({
            "chunk_id": f"E23_SWAPPED_{c_idx:04d}",
            "split": "SCREEN",
            "n_pairs": len(batch),
            "n_views": len(rows),
            "rows": rows,
        })

    parity_ids = [ordered_screen[0], ordered_screen[-1]] if len(ordered_screen) >= 2 else list(ordered_screen)
    authentic_parity_sample = []
    for iid in parity_ids:
        for lang in LANGUAGES:
            rec = draw0_records[(iid, lang)]
            authentic_parity_sample.append({
                "image_id": iid,
                "image_path": rec.image_path,
                "image_sha256": rec.image_sha256,
                "language": lang,
                "caption_id": rec.caption_id,
                "caption_draw_key": 0,
                "caption_text": rec.caption_text,
                "caption_text_sha256": _text_sha256(rec.caption_text),
            })

    return {
        "schema": "phase3.e23_pair_plan.v1",
        "split": "SCREEN",
        "confirm_rows": 0,
        "seed_recipe": E23_PAIR_SEED_PREFIX,
        "zero_model_scores_in_pairing": True,
        "chunk_size_pairs": E23_CHUNK_SIZE_PAIRS,
        "total_unique_directed_pairs": len(unique_directed),
        "total_swapped_views": total_views,
        "total_chunks": len(chunks),
        "per_category": per_category,
        "chunks": chunks,
        "authentic_parity_sample": authentic_parity_sample,
    }


def build_e39_e31_extraction_plan(screen_ids, draw0_records):
    """Build deterministic SCREEN-only 32-image chunk plan for combined E39+E31 GPU forward."""
    ordered_screen = sorted(screen_ids)
    chunks = []
    total_views_per_condition = 0
    for c_idx, offset in enumerate(range(0, len(ordered_screen), E39_E31_CHUNK_SIZE_IMAGES)):
        batch_ids = ordered_screen[offset : offset + E39_E31_CHUNK_SIZE_IMAGES]
        rows = []
        for iid in batch_ids:
            for lang in LANGUAGES:
                rec = draw0_records[(iid, lang)]
                rows.append({
                    "image_id": iid,
                    "image_path": rec.image_path,
                    "image_sha256": rec.image_sha256,
                    "language": lang,
                    "caption_id": rec.caption_id,
                    "caption_draw_key": 0,
                    "caption_text": rec.caption_text,
                    "caption_text_sha256": _text_sha256(rec.caption_text),
                })
        total_views_per_condition += len(rows)
        chunks.append({
            "chunk_id": f"SCREEN_{c_idx:04d}",
            "split": "SCREEN",
            "image_ids": batch_ids,
            "n_images": len(batch_ids),
            "n_views_per_condition": len(rows),
            "rows": rows,
        })

    return {
        "schema": "phase3.e39_e31_extraction_plan.v1",
        "split": "SCREEN",
        "confirm_rows": 0,
        "conditions": list(CONDITIONS),
        "e39_sites": list(E39_SITES),
        "e31_sites": list(E31_SITES),
        "chunk_size_images": E39_E31_CHUNK_SIZE_IMAGES,
        "total_chunks": len(chunks),
        "total_images": len(ordered_screen),
        "views_per_condition": total_views_per_condition,
        "chunks": chunks,
    }


def _load_screen_metadata(binding, protocol):
    caption_meta = stage1_caption_paths(binding)
    files_meta = caption_meta["files"]
    dataset_doc = json.loads(within(binding.root, files_meta["dataset.json"]["path"]).read_text(encoding="utf-8"))
    splits_doc = json.loads(within(binding.root, files_meta["splits.json"]["path"]).read_text(encoding="utf-8"))
    target_rows = _read_jsonl(within(binding.root, files_meta["targets.jsonl"]["path"]))
    record_rows = _read_jsonl(within(binding.root, files_meta["records.jsonl"]["path"]))

    category_order = list(protocol["category_order"])
    check(list(dataset_doc.get("category_order", [])) == category_order, "Dataset category_order mismatch")

    screen_ids_raw = splits_doc.get("SCREEN", [])
    check(len(screen_ids_raw) == int(protocol["split_sizes"]["SCREEN"]), "SCREEN split size mismatch")
    screen_id_set = set(screen_ids_raw)
    confirm_id_set = set(splits_doc.get("CONFIRM", []))
    check(screen_id_set.isdisjoint(confirm_id_set), "CONFIRM leaked into SCREEN splits")

    screen_targets = {}
    for row in target_rows:
        iid = row["image_id"]
        sp = row["split"]
        if sp == "CONFIRM":
            continue
        require_open_split(sp)
        if sp == "SCREEN":
            check(iid in screen_id_set, f"Unexpected SCREEN target ID {iid}")
            screen_targets[iid] = TargetRecord(iid, sp, tuple(row["values"]))
    check(set(screen_targets) == screen_id_set, "Incomplete SCREEN targets in targets.jsonl")

    draw0_records = {}
    for row in record_rows:
        iid = row["image_id"]
        sp = row["split"]
        if sp == "CONFIRM":
            continue
        require_open_split(sp)
        if sp != "SCREEN":
            continue
        check(iid in screen_id_set, f"Unexpected SCREEN record ID {iid}")
        lang = row["language"]
        check(lang in LANGUAGES, f"Invalid language {lang!r}")
        key = (iid, lang)
        check(key not in draw0_records, f"Duplicate SCREEN draw-0 record for {key}")
        draw0_records[key] = ModelInput(
            image_id=iid,
            image_path=row["image_path"],
            image_sha256=row["image_sha256"],
            language=lang,
            caption_id=row["caption_id"],
            caption_text=row["caption_text"],
            split=sp,
        )
    check(len(draw0_records) == len(screen_id_set) * len(LANGUAGES), "Incomplete SCREEN draw-0 records")
    return sorted(screen_id_set), screen_targets, draw0_records


def _model_specs(binding, cpu_plan_body):
    from src.stage4.plan import DEPTHS

    specs = {}
    for model in MODELS:
        e40_inv = (cpu_plan_body or {}).get("e40_inventory", {}).get(model)
        mid = int(midpoint_depth(binding, model))
        specs[model] = {
            "depths": list(DEPTHS[model]),
            "midpoint_depth": mid,
            "model_plan_ref": plain(binding.model_plan_refs[model]),
            "midpoint_groups": {
                cond: {
                    pool: plain(midpoint_groups(binding, model, condition=cond, pooling=pool))
                    for pool in POOLINGS
                }
                for cond in CONDITIONS
            },
            "source_systems": (
                plain(e40_inv["source_systems"]) if isinstance(e40_inv, Mapping) else {}
            ),
        }
    return specs


def _emit(progress, event):
    if progress is not None:
        progress(event)


@contract
def prepare_phase3_gpu(
    root,
    protocol,
    producer,
    *,
    cpu_plan_ref=None,
    stage4_release_ref=None,
    run_id=None,
    progress=None,
):
    """Build Stage B1 deterministic E39+E31 and E23 plans and publish `phase3_gpu_plan`."""
    resolved_root = Path(root).resolve()
    t0 = time.monotonic()
    _emit(progress, {"event": "phase3_gpu_prepare_start", "unit": "bind", "completed": 0, "total": 4})

    binding = bind_phase1(resolved_root, protocol, stage4_release_ref or plain(PHASE1_STAGE4_RELEASE))
    init_dirs = init_phase3_dirs(resolved_root)

    upstream = [plain(binding.stage4_release_ref)]
    resolved_cpu_ref = cpu_plan_ref
    if resolved_cpu_ref is None and (resolved_root / CERTIFIED_STAGE_A1_CPU_PLAN["path"]).is_file():
        resolved_cpu_ref = plain(CERTIFIED_STAGE_A1_CPU_PLAN)

    cpu_plan_body = None
    cpu_plan_ref_out = None
    if resolved_cpu_ref is not None:
        cpu_plan = load_phase3_cpu_plan(resolved_root, protocol, resolved_cpu_ref, bind=False)
        check(cpu_plan["dataset_fingerprint"] == binding.dataset_fingerprint, "CPU plan dataset mismatch")
        check(cpu_plan["confirm_excluded"] is True, "CPU plan must exclude CONFIRM")
        probe_ref = plain(cpu_plan["storage_probe"])
        verify_storage_probe(resolved_root, protocol, probe_ref, binding.dataset_fingerprint)
        cpu_plan_ref_out = plain(cpu_plan.reference)
        cpu_plan_body = plain(cpu_plan._body)
        upstream.append(cpu_plan_ref_out)
    else:
        probe_ref = plain(storage_probe(resolved_root, protocol, binding.dataset_fingerprint, producer))
        verify_storage_probe(resolved_root, protocol, probe_ref, binding.dataset_fingerprint)
    upstream.append(probe_ref)
    _emit(progress, {"event": "phase3_gpu_probe_ok", "unit": "storage_probe", "completed": 1, "total": 4})

    screen_ids, screen_targets, draw0_records = _load_screen_metadata(binding, protocol)
    e39_e31_plan = build_e39_e31_extraction_plan(screen_ids, draw0_records)
    e23_plan = build_e23_pair_plan(
        screen_ids,
        screen_targets,
        draw0_records,
        protocol["category_order"],
        protocol["primary_categories"],
        int(protocol["minimum_positive_and_negative"]["SCREEN"]),
    )
    _emit(progress, {
        "event": "phase3_gpu_plans_built",
        "unit": "pair_and_token_plans",
        "completed": 2,
        "total": 4,
        "e39_e31_chunks": e39_e31_plan["total_chunks"],
        "e23_unique_directed_pairs": e23_plan["total_unique_directed_pairs"],
        "e23_chunks": e23_plan["total_chunks"],
    })

    core = plain({
        "schema": SCHEMA_GPU_PLAN,
        "phase3_gpu_version": PHASE3_GPU_VERSION,
        "lineage_identity": binding.lineage_identity,
        "dataset_fingerprint": binding.dataset_fingerprint,
        "protocol_fingerprint": binding.protocol_fingerprint,
        "phase1_stage4_release": plain(binding.stage4_release_ref),
        "phase3_cpu_plan": cpu_plan_ref_out,
        "bootstrap": {"shape": list(binding.bootstrap_shape), "sha256": binding.bootstrap_sha256},
        "storage_probe": probe_ref,
        "initialized_dirs": list(init_dirs),
        "models": _model_specs(binding, cpu_plan_body),
        "e39_e31_plan": e39_e31_plan,
        "e23_plan": e23_plan,
        "decision_ranges": plain(PHASE3_GPU_DECISION_RANGES),
        "e35_excluded": True,
        "confirm_excluded": True,
    })
    plan_id = fingerprint(core)
    body = dict(core, plan_identity=plan_id)

    plan_ref = publish_phase3(
        resolved_root,
        protocol,
        binding.dataset_fingerprint,
        "phase3/plans",
        KIND_GPU_PLAN,
        body,
        producer,
        upstream=upstream,
        run_id=run_id,
    )
    _emit(progress, {
        "event": "phase3_gpu_plan_published",
        "status": "passed",
        "unit": "publish",
        "completed": 4,
        "total": 4,
        "elapsed_s": round(time.monotonic() - t0, 3),
        "plan_ref": plain(plan_ref),
        "plan_identity": plan_id,
    })
    return Phase3GpuPlan(body, plan_ref, resolved_root, protocol, binding=binding)


@contract
def load_phase3_gpu_plan(root, protocol, reference, *, bind=True):
    """Authenticate `phase3_gpu_plan` on disk and optionally bind Phase 1."""
    resolved_root = Path(root).resolve()
    ref, body, _ = load_phase3_artifact(
        resolved_root, protocol, reference, KIND_GPU_PLAN, expected_schema=SCHEMA_GPU_PLAN
    )
    check(body.get("confirm_excluded") is True, "phase3_gpu_plan must enforce confirm_excluded=True")
    check(body.get("e35_excluded") is True, "phase3_gpu_plan must record e35_excluded=True")
    core = {k: v for k, v in plain(body).items() if k != "plan_identity"}
    check(fingerprint(core) == body["plan_identity"], "phase3_gpu_plan plan_identity mismatch")
    verify_storage_probe(resolved_root, protocol, body["storage_probe"], body["dataset_fingerprint"])
    binding = bind_phase1(resolved_root, protocol, plain(body["phase1_stage4_release"])) if bind else None
    return Phase3GpuPlan(body, ref, resolved_root, protocol, binding=binding)


def _default_emit(event):
    print(json.dumps(plain(event), sort_keys=True), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Phase 3 GPU Track Stage B1 preparation.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prep = sub.add_parser("prepare")
    p_prep.add_argument("--phase1-root", default=None)
    p_prep.add_argument("--protocol", default=str(REPO / "configs/protocol.json"))
    p_prep.add_argument("--cpu-plan", default=None)
    p_prep.add_argument("--storage-scope", default="persistent-drive")
    p_prep.add_argument("--run-id", default=None)

    p_ver = sub.add_parser("verify")
    p_ver.add_argument("--phase1-root", default=None)
    p_ver.add_argument("--protocol", default=str(REPO / "configs/protocol.json"))
    p_ver.add_argument("--plan", required=True)

    args = parser.parse_args(argv)
    root = phase1_root(args.phase1_root)
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))

    if args.command == "verify":
        ref = json.loads(args.plan) if args.plan.strip().startswith("{") else args.plan
        plan = load_phase3_gpu_plan(root, protocol, ref, bind=True)
        _default_emit({"event": "phase3_gpu_plan_verified", "status": "passed", "plan_ref": plain(plan.reference), "plan_identity": plan["plan_identity"]})
        return 0

    cpu_ref = None
    if args.cpu_plan:
        cpu_ref = json.loads(args.cpu_plan) if args.cpu_plan.strip().startswith("{") else args.cpu_plan
    producer = current_producer(scope=args.storage_scope)
    plan = prepare_phase3_gpu(root, protocol, producer, cpu_plan_ref=cpu_ref, run_id=args.run_id, progress=_default_emit)
    _default_emit({"event": "complete", "status": "passed", "plan_ref": plain(plan.reference), "plan_identity": plan["plan_identity"]})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        _default_emit({"status": "blocked", "blockers": [str(exc)]})
        raise SystemExit(2)
