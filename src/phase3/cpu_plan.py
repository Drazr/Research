"""Phase 3 CPU Track Stage A1: bindings, inventories, preflight checks, and plan publisher.

Implements phase3.md §A1 (`phase3-cpu-stageA1.v1`):
- Authenticates Phase 1 Stage 4 release (`bind_phase1`) and certified Phase 2 artifacts
  (`phase2_plan`, `phase2_e17_auth`, E29, E30, E11, E17) without mutating Phase 1/2.
- Resolves and verifies per-worker inventories for E40, E41, E44, and E12.
- Runs storage_probe under `phase3/probes/`, pre-initializes all `PHASE3_PARENT_DIRS`,
  and publishes the immutable `phase3_cpu_plan` artifact.
"""

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import sys
import time

from src.common.contracts import ContractError, LANGUAGES, fingerprint
from src.common.storage import phase1_root
from src.phase2.e17_auth import E17_DEV_C_PERMISSION, load_e17_auth
from src.phase2.gates import load_e29_gate
from src.phase2.interfaces import (
    KIND_ANALYSIS as P2_KIND_ANALYSIS,
    PHASE1_STAGE4_RELEASE,
    SCHEMA_ANALYSIS as P2_SCHEMA_ANALYSIS,
    bind_phase1,
    current_producer,
    e11_fit_statistics_refs,
    e17_inputs,
    load_frozen_source_head,
    load_phase2_artifact,
    midpoint_depth,
    midpoint_groups,
    read_verified_midpoint_features,
)
from src.phase2.plan import load_phase2_plan
from .schema import (
    CONDITIONS,
    KIND_CPU_PLAN,
    MODELS,
    PHASE3_CPU_VERSION,
    PHASE3_PARENT_DIRS,
    POOLINGS,
    REPO,
    SCHEMA_CPU_PLAN,
    check,
    contract,
    frozen,
    init_phase3_dirs,
    load_phase3_artifact,
    plain,
    publish_phase3,
    storage_probe,
    verify_storage_probe,
)

# Certified Phase 2 completion references (phase2.md §2.9 & §4.7).
CERTIFIED_PHASE2_FOR_PHASE3_CPU = frozen({
    "phase2_plan": {
        "path": "phase2/plans/20261119T163039Z-52eb4650cec3436bb4bf989afe3cc303/_COMPLETE.json",
        "sha256": "c2d8e153802ebfefd1caddcb7b0cfc9e681bd1b978a422e4475c3187effbd62d",
    },
    "phase2_e17_auth": {
        "path": "phase2/plans/20261119T162232Z-710a83aae9064c1aa26086b34b17bf45/_COMPLETE.json",
        "sha256": "129cba3ef774dc4c69ecfeabc8547e2aa2186567ec1ee77fb047fc39ba96ddff",
    },
    "models": {
        "qwen": {
            "e29_analysis": {
                "path": "phase2/results/e29/qwen/20261119T173548Z-73ad41f4722e42fa931817b6f2b28681/_COMPLETE.json",
                "sha256": "77a8dc8ff50f8886cb706e09e7540952a66c6d87fa7f1278a830f196016c9926",
            },
            "e29_gate": {
                "path": "phase2/results/e29_gates/20261119T173548Z-73ad41f4722e42fa931817b6f2b28681-gate/_COMPLETE.json",
                "sha256": "3975fedbd35e63b2e21f5f94b196782f5642ec9e35ec99bb84d7b547c40341ef",
            },
            "e30_analysis": {
                "path": "phase2/results/e30/qwen/20261119T230416Z-8c601fa863604e3395624e37382fdedb/_COMPLETE.json",
                "sha256": "0dbe25f3fe6659b6e18419153c354b97ee4b124d8b29bd5096e7652c7a0b1f9f",
            },
            "e11_analysis": {
                "path": "phase2/results/e11/qwen/20261119T230112Z-af0550ca6a0c494e8a4f9d1728af213e/_COMPLETE.json",
                "sha256": "552d0f4f59bf0760ff7ee8da62b64197550d9087071becd2a260080f203f4c1b",
            },
            "e17_analysis": {
                "path": "phase2/results/e17/qwen/20261119T174402Z-317626df13c944b188686ba32d9b92cb/_COMPLETE.json",
                "sha256": "fcb7c5b1e31f1dfdb43572e077879fdacba18cc8384f77b647d195c04227d771",
            },
        },
        "gemma": {
            "e29_analysis": {
                "path": "phase2/results/e29/gemma/20261119T173446Z-35a50762bbf445439faff5951c00eb80/_COMPLETE.json",
                "sha256": "ae3b5183ce126c7e2f6545384949798a4831756da460294585abf980fa12ec16",
            },
            "e29_gate": {
                "path": "phase2/results/e29_gates/20261119T173446Z-35a50762bbf445439faff5951c00eb80-gate/_COMPLETE.json",
                "sha256": "7434bee02f3875af67287a520e0731fd1fa8d066c189159a85fc13b0e65fe512",
            },
            "e30_analysis": {
                "path": "phase2/results/e30/gemma/20261119T225928Z-682c54d91984488aacffba40ab0f6b2c/_COMPLETE.json",
                "sha256": "0de28dc4d6d4592e4312c3b64c86c8810267d1a4d10f9f36bb319a32b5382c6a",
            },
            "e11_analysis": {
                "path": "phase2/results/e11/gemma/20261119T230113Z-af983c1752eb4861a0c9390d53c096e5/_COMPLETE.json",
                "sha256": "09fa67ee3d4aed49726e57403077f47aeca47bab6fe4a67559ce2def0173dbbc",
            },
            "e17_analysis": {
                "path": "phase2/results/e17/gemma/20261119T220802Z-167de08008b74942813cdf81ca061e43/_COMPLETE.json",
                "sha256": "ab3d81143f917c2133d9beb888f64b38330995bcb4f325b7c33ed629e5b05c50",
            },
        },
    },
})

PHASE3_CPU_DECISION_RANGES = frozen({
    "E40": {
        "caption_only_replication_ratio_min": 0.50,
        "final_depth_interaction_absent_lt": 0.015,
        "negligible_ci_within": (-0.015, 0.015),
        "checks": {"matched_en_self_check_tol": 0.0, "midpoint_parity_tol": 1e-12},
    },
    "E41": {
        "label_free_nll_gap_to_target_max": 0.015,
        "cov_auroc_gap_reduction_min": 0.010,
        "cov_or_fraction_of_g_res_min": 0.25,
        "cov_negligible_ci_upper_lt": 0.005,
        "pca_subspace_ranks": (64, 256),
        "checks": {"mean_shift_ranking_tol": 1e-12, "e17_ranking_tol": 1e-12, "positive_temperature": True},
    },
    "E44": {
        "nominal_coverages": (0.50, 0.75, 0.90),
        "primary_nominal": 0.75,
        "coverage_drift_max": 0.030,
        "retained_risk_reduction_min": 0.020,
        "positive_retention_min": 0.50,
        "negligible_risk_ci_upper_lt": 0.005,
        "checks": {"min_outcome_support": 20, "phase1_e19_repro_tol": 1e-12},
    },
    "E12": {
        "pca_rank": 64,
        "ridge_alphas": (0.01, 0.1, 1.0, 10.0, 100.0),
        "c_grid": (0.01, 0.1, 1.0, 10.0),
        "auroc_gap_reduction_min": 0.010,
        "or_fraction_of_g_res_min": 0.33,
        "constraint_diff_min": 0.010,
        "negligible_ci_upper_lt": 0.005,
        "checks": {"zero_target_semantic_labels": True, "paired_fit_images": 2160, "paired_dev_s_images": 180},
    },
})


class Phase3CpuPlan(Mapping):
    """Read-only authenticated Phase 3 CPU plan."""

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


def _depth_groups(binding, model, depth, condition, pooling):
    site = f"decoder:{int(depth)}:{pooling}"
    groups = {}
    for gid, group in binding.stage4_plan.fit["groups"].items():
        if (
            group["source"] == "vlm"
            and group["model"] == model
            and group["condition"] == condition
            and group["site"] == site
        ):
            lang = group["language"]
            check(lang in LANGUAGES and lang not in groups, f"Duplicate/invalid group {gid}")
            groups[lang] = gid
    check(set(groups) == set(LANGUAGES), f"Incomplete 4-language groups for {model}/{condition}/{site}")
    return {lang: groups[lang] for lang in LANGUAGES}


def _block_lookup_at_depth(binding, model, depth, condition, pooling, split):
    gids = _depth_groups(binding, model, depth, condition, pooling)
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
        t_lang = gid_to_lang[target_gid]
        check(t_lang not in found, f"Duplicate Stage 4 block {model}/{depth}/{condition}/{pooling}/{split}/{t_lang}")
        routes = {}
        for route in block["routes"]:
            src_lang = plan.fit["groups"][route["source_group"]]["language"]
            routes[src_lang] = route["route_id"]
        found[t_lang] = {
            "block_id": bid,
            "target_group": target_gid,
            "prediction_ref": plain(predictions[bid]),
            "routes": routes,
        }
    check(set(found) == set(LANGUAGES), f"Missing Stage 4 {split} blocks for {model}/{depth}/{condition}/{pooling}")
    return found


def _build_e40_inventory(binding, model, e30_body):
    from src.stage4.plan import DEPTHS

    depths = [int(d) for d in DEPTHS[model]]
    fit_groups = binding.stage4_plan.fit["groups"]
    matched_blocks = {}
    feature_releases = {}
    source_systems = {}
    provenance_checks = {}

    for depth in depths:
        dkey = str(depth)
        matched_blocks[dkey] = {}
        feature_releases[dkey] = {}
        source_systems[dkey] = {}
        provenance_checks[dkey] = {}
        for cond in CONDITIONS:
            matched_blocks[dkey][cond] = {}
            feature_releases[dkey][cond] = {}
            source_systems[dkey][cond] = {}
            dims = set()
            model_plans = set()
            for pool in POOLINGS:
                blocks = _block_lookup_at_depth(binding, model, depth, cond, pool, "SCREEN")
                matched_blocks[dkey][cond][pool] = blocks
                gids = _depth_groups(binding, model, depth, cond, pool)
                feature_releases[dkey][cond][pool] = {
                    lang: {
                        "group_id": gids[lang],
                        "release_ref": plain(fit_groups[gids[lang]]["releases"]["SCREEN"]),
                        "dimension": int(fit_groups[gids[lang]]["dimension"]),
                        "site": fit_groups[gids[lang]]["site"],
                    }
                    for lang in LANGUAGES
                }
                source_systems[dkey][cond][pool] = {
                    lang: {
                        "group_id": gids[lang],
                        "selection_ref": plain(binding.stage4_plan["ordinary"][gids[lang]]["selection"]),
                        "candidate_ref": plain(binding.stage4_plan["ordinary"][gids[lang]]["candidate"]),
                        "C": float(binding.stage4_plan["ordinary"][gids[lang]]["C"]),
                    }
                    for lang in LANGUAGES
                }
                for lang in LANGUAGES:
                    dims.add(int(fit_groups[gids[lang]]["dimension"]))
                    model_plans.add(json.dumps(plain(fit_groups[gids[lang]]["model_plan"]), sort_keys=True))
            check(len(dims) == 1, f"Dimension mismatch across poolings for {model}/{dkey}/{cond}: {dims}")
            check(len(model_plans) == 1, f"model_plan mismatch across poolings for {model}/{dkey}/{cond}")
            provenance_checks[dkey][cond] = {"dimension": next(iter(dims)), "hook_depth": depth, "aligned": True}

    return {
        "depths": depths,
        "reused_midpoint_crossed_refs": plain(e30_body["crossed_block_refs"]),
        "matched_blocks": matched_blocks,
        "screen_feature_releases": feature_releases,
        "source_systems": source_systems,
        "provenance_checks": provenance_checks,
    }


def _build_e41_inventory(binding, model, e11_body, e17_body):
    from src.stage4.plan import DEPTHS

    depths = [int(midpoint_depth(binding, model)), int(DEPTHS[model][-1])]
    fit_groups = binding.stage4_plan.fit["groups"]
    by_depth = {}
    for depth in depths:
        dkey = str(depth)
        by_depth[dkey] = {}
        for pool in POOLINGS:
            gids = _depth_groups(binding, model, depth, "image_caption", pool)
            screen_blocks = _block_lookup_at_depth(binding, model, depth, "image_caption", pool, "SCREEN")
            dev_c_blocks = _block_lookup_at_depth(binding, model, depth, "image_caption", pool, "DEV-C")
            by_depth[dkey][pool] = {
                "group_ids": gids,
                "screen_blocks": screen_blocks,
                "dev_c_diagonals": dev_c_blocks,
                "feature_releases": {
                    lang: {
                        "FIT": plain(fit_groups[gids[lang]]["releases"]["FIT"]),
                        "DEV-C": plain(fit_groups[gids[lang]]["releases"]["DEV-C"]),
                        "SCREEN": plain(fit_groups[gids[lang]]["releases"]["SCREEN"]),
                    }
                    for lang in LANGUAGES
                },
            }
    return {
        "permission": "phase3.e41_cov_and_calibration",
        "depths": depths,
        "reused_e17_midpoint_dev_c_blocks": plain(e17_body["dev_c_block_refs"]),
        "reused_e17_midpoint_calibrators_ref": plain(e17_body["calibrators_ref"]),
        "reused_e11_midpoint_adapted_refs": plain(e11_body["adapted_block_refs"]),
        "by_depth": by_depth,
    }


def _build_e44_inventory(binding, model, e11_body, e17_body):
    mid_inputs = e17_inputs(binding, model)
    e11_stats = e11_fit_statistics_refs(binding, model)
    return {
        "permission": "phase3.e44_selective_cutoffs",
        "midpoint_depth": int(midpoint_depth(binding, model)),
        "last_caption_dev_c_diagonals": plain(mid_inputs["dev_c_diagonals"]),
        "last_caption_screen_blocks": plain(mid_inputs["screen_blocks"]),
        "reused_e17_off_diag_dev_c_blocks": plain(e17_body["dev_c_block_refs"]),
        "reused_e17_calibrators_ref": plain(e17_body["calibrators_ref"]),
        "caption_mean_fit_stats": plain(e11_stats),
        "caption_mean_dev_c_releases": {
            lang: plain(binding.stage4_plan.fit["groups"][e11_stats[lang]["group_id"]]["releases"]["DEV-C"])
            for lang in LANGUAGES
        },
        "reused_e11_screen_adapted_refs": plain(e11_body["adapted_block_refs"]),
    }


def _build_e12_inventory(binding, model):
    fit_groups = binding.stage4_plan.fit["groups"]
    gids = midpoint_groups(binding, model, condition="image_caption", pooling="caption_mean")
    screen_blocks = _block_lookup_at_depth(binding, model, midpoint_depth(binding, model), "image_caption", "caption_mean", "SCREEN")
    return {
        "midpoint_depth": int(midpoint_depth(binding, model)),
        "condition": "image_caption",
        "pooling": "caption_mean",
        "pca_rank": 64,
        "group_ids": gids,
        "screen_blocks": screen_blocks,
        "feature_releases": {
            lang: {
                "FIT": plain(fit_groups[gids[lang]]["releases"]["FIT"]),
                "DEV-S": plain(fit_groups[gids[lang]]["releases"]["DEV-S"]),
                "SCREEN": plain(fit_groups[gids[lang]]["releases"]["SCREEN"]),
            }
            for lang in LANGUAGES
        },
    }


def _run_preflight_checks(binding, model):
    """Verify paired FIT/DEV-S IDs across languages and exact en->en diagonal parity."""
    import numpy as np
    from src.stage3.engine import transform
    from src.stage3.readouts import expected_rows
    from src.stage4.predictions import load_prediction_block

    fit = binding.stage4_plan.fit
    store = binding.stage4_plan.store
    gids_mean = midpoint_groups(binding, model, condition="image_caption", pooling="caption_mean")

    fit_ids_ref = None
    dev_s_ids_ref = None
    for lang in LANGUAGES:
        gid = gids_mean[lang]
        f_rows = [r["image_id"] for r in expected_rows(fit, gid, "FIT", store)]
        d_rows = [r["image_id"] for r in expected_rows(fit, gid, "DEV-S", store)]
        check(len(f_rows) == 2160, f"Expected 2160 FIT rows for {model}/{lang}; got {len(f_rows)}")
        check(len(d_rows) == 180, f"Expected 180 DEV-S rows for {model}/{lang}; got {len(d_rows)}")
        if fit_ids_ref is None:
            fit_ids_ref = f_rows
            dev_s_ids_ref = d_rows
        else:
            check(f_rows == fit_ids_ref, f"FIT row IDs differ between languages for {model}/{lang}")
            check(d_rows == dev_s_ids_ref, f"DEV-S row IDs differ between languages for {model}/{lang}")

    # In-memory en -> en caption_mean SCREEN diagonal reproduction check (atol=0, rtol=0)
    rows_en, X_en = read_verified_midpoint_features(
        binding, model, "image_caption", "caption_mean", "en", "SCREEN"
    )
    coef, intercept, estimable, _, stats = load_frozen_source_head(
        binding, model, "image_caption", "caption_mean", "en"
    )
    logits = np.asarray(transform(X_en, stats) @ coef.T) + intercept
    check(np.isfinite(logits).all(), f"Nonfinite preflight logits for {model}")
    scr_blocks = _block_lookup_at_depth(binding, model, midpoint_depth(binding, model), "image_caption", "caption_mean", "SCREEN")
    s4_pred = load_prediction_block(
        binding.stage4_plan, scr_blocks["en"]["prediction_ref"], expected_block=scr_blocks["en"]["block_id"]
    )
    en_idx = [r["route_id"] for r in binding.stage4_plan["blocks"][scr_blocks["en"]["block_id"]]["routes"]].index(
        scr_blocks["en"]["routes"]["en"]
    )
    diff = float(np.max(np.abs(logits - s4_pred.logits[en_idx])))
    check(diff == 0.0, f"Preflight en->en caption_mean SCREEN diagonal mismatch for {model}: {diff}")

    return {
        "paired_fit_images": len(fit_ids_ref),
        "paired_dev_s_images": len(dev_s_ids_ref),
        "en_caption_mean_screen_atol0_diff": diff,
        "passed": True,
    }


def _emit(progress, event):
    if progress is not None:
        progress(event)


@contract
def prepare_phase3_cpu(
    root,
    protocol,
    producer,
    *,
    phase2_refs=None,
    stage4_release_ref=None,
    run_id=None,
    progress=None,
):
    """Bind Phase 1 & Phase 2, run storage_probe, pre-init phase3/, and publish `phase3_cpu_plan`."""
    resolved_root = Path(root).resolve()
    from src.stage4.plan import DEPTHS
    refs = plain(CERTIFIED_PHASE2_FOR_PHASE3_CPU) if phase2_refs is None else plain(phase2_refs)
    t0 = time.monotonic()
    _emit(progress, {"event": "phase3_cpu_prepare_start", "unit": "bind", "completed": 0, "total": 4})

    binding = bind_phase1(resolved_root, protocol, stage4_release_ref or plain(PHASE1_STAGE4_RELEASE))
    p2_plan = load_phase2_plan(resolved_root, protocol, refs["phase2_plan"])
    check(p2_plan["confirm_excluded"] is True, "phase2_plan must exclude CONFIRM")
    check(p2_plan["dataset_fingerprint"] == binding.dataset_fingerprint, "Dataset fingerprint mismatch")
    check(p2_plan["lineage_identity"] == binding.lineage_identity, "Lineage mismatch")

    p2_auth_ref, p2_auth_body, _ = load_e17_auth(resolved_root, protocol, refs["phase2_e17_auth"])
    check(p2_auth_body["permission"] == E17_DEV_C_PERMISSION, "Invalid Phase 2 E17 permission")

    init_dirs = init_phase3_dirs(resolved_root)
    probe_ref = storage_probe(resolved_root, protocol, binding.dataset_fingerprint, producer)
    verify_storage_probe(resolved_root, protocol, probe_ref, binding.dataset_fingerprint)
    _emit(progress, {"event": "phase3_cpu_probe_ok", "unit": "storage_probe", "completed": 1, "total": 4})

    verified_p2_models = {}
    e40_inv, e41_inv, e44_inv, e12_inv, preflight = {}, {}, {}, {}, {}
    upstream = [plain(binding.stage4_release_ref), plain(p2_plan.reference), plain(p2_auth_ref), plain(probe_ref)]

    for model in MODELS:
        mr = refs["models"][model]
        gate_ref, gate_body, _ = load_e29_gate(resolved_root, protocol, mr["e29_gate"], plan=p2_plan, expected_model=model)
        e29_ref, e29_body, _ = load_phase2_artifact(
            resolved_root, protocol, mr["e29_analysis"], P2_KIND_ANALYSIS,
            expected_schema=P2_SCHEMA_ANALYSIS, expected_dataset=binding.dataset_fingerprint,
            expected_plan_identity=p2_plan["plan_identity"], expected_owner=model,
        )
        e30_ref, e30_body, _ = load_phase2_artifact(
            resolved_root, protocol, mr["e30_analysis"], P2_KIND_ANALYSIS,
            expected_schema=P2_SCHEMA_ANALYSIS, expected_dataset=binding.dataset_fingerprint,
            expected_plan_identity=p2_plan["plan_identity"], expected_owner=model,
        )
        e11_ref, e11_body, _ = load_phase2_artifact(
            resolved_root, protocol, mr["e11_analysis"], P2_KIND_ANALYSIS,
            expected_schema=P2_SCHEMA_ANALYSIS, expected_dataset=binding.dataset_fingerprint,
            expected_plan_identity=p2_plan["plan_identity"], expected_owner=model,
        )
        e17_ref, e17_body, _ = load_phase2_artifact(
            resolved_root, protocol, mr["e17_analysis"], P2_KIND_ANALYSIS,
            expected_schema=P2_SCHEMA_ANALYSIS, expected_dataset=binding.dataset_fingerprint,
            expected_plan_identity=p2_plan["plan_identity"], expected_owner=model,
        )
        verified_p2_models[model] = {
            "e29_gate": plain(gate_ref),
            "e29_analysis": plain(e29_ref),
            "e30_analysis": plain(e30_ref),
            "e11_analysis": plain(e11_ref),
            "e17_analysis": plain(e17_ref),
        }
        upstream.extend([plain(gate_ref), plain(e29_ref), plain(e30_ref), plain(e11_ref), plain(e17_ref)])

        e40_inv[model] = _build_e40_inventory(binding, model, e30_body)
        e41_inv[model] = _build_e41_inventory(binding, model, e11_body, e17_body)
        e44_inv[model] = _build_e44_inventory(binding, model, e11_body, e17_body)
        e12_inv[model] = _build_e12_inventory(binding, model)
        preflight[model] = _run_preflight_checks(binding, model)

    _emit(progress, {"event": "phase3_cpu_preflight_ok", "unit": "preflight", "completed": 2, "total": 4})

    core = {
        "schema": SCHEMA_CPU_PLAN,
        "phase3_version": PHASE3_CPU_VERSION,
        "lineage_identity": binding.lineage_identity,
        "dataset_fingerprint": binding.dataset_fingerprint,
        "protocol_fingerprint": binding.protocol_fingerprint,
        "phase1_stage4_release": plain(binding.stage4_release_ref),
        "phase2_refs": {
            "phase2_plan": plain(p2_plan.reference),
            "phase2_plan_identity": p2_plan["plan_identity"],
            "phase2_e17_auth": plain(p2_auth_ref),
            "models": verified_p2_models,
        },
        "bootstrap": {"shape": list(binding.bootstrap_shape), "sha256": binding.bootstrap_sha256},
        "depths": {m: list(DEPTHS[m]) for m in MODELS},
        "midpoints": plain(binding.midpoints),
        "storage_probe": plain(probe_ref),
        "initialized_dirs": list(init_dirs),
        "e40_inventory": e40_inv,
        "e41_inventory": e41_inv,
        "e44_inventory": e44_inv,
        "e12_inventory": e12_inv,
        "preflight_checks": preflight,
        "decision_ranges": plain(PHASE3_CPU_DECISION_RANGES),
        "confirm_excluded": True,
    }
    plan_id = fingerprint(core)
    body = dict(core, plan_identity=plan_id)

    plan_ref = publish_phase3(
        resolved_root,
        protocol,
        binding.dataset_fingerprint,
        "phase3/plans",
        KIND_CPU_PLAN,
        body,
        producer,
        upstream=upstream,
        run_id=run_id,
    )
    _emit(progress, {
        "event": "phase3_cpu_plan_published",
        "status": "passed",
        "unit": "publish",
        "completed": 4,
        "total": 4,
        "elapsed_s": round(time.monotonic() - t0, 3),
        "plan_ref": plain(plan_ref),
        "plan_identity": plan_id,
        "preflight_checks": preflight,
    })
    return Phase3CpuPlan(body, plan_ref, resolved_root, protocol, binding=binding)


@contract
def load_phase3_cpu_plan(root, protocol, reference, *, bind=True):
    """Authenticate `phase3_cpu_plan` on disk and optionally bind Phase 1."""
    resolved_root = Path(root).resolve()
    ref, body, _ = load_phase3_artifact(
        resolved_root, protocol, reference, KIND_CPU_PLAN, expected_schema=SCHEMA_CPU_PLAN
    )
    check(body.get("confirm_excluded") is True, "phase3_cpu_plan must enforce confirm_excluded=True")
    core = {k: v for k, v in plain(body).items() if k != "plan_identity"}
    check(fingerprint(core) == body["plan_identity"], "phase3_cpu_plan plan_identity mismatch")
    verify_storage_probe(resolved_root, protocol, body["storage_probe"], body["dataset_fingerprint"])
    binding = bind_phase1(resolved_root, protocol, plain(body["phase1_stage4_release"])) if bind else None
    return Phase3CpuPlan(body, ref, resolved_root, protocol, binding=binding)


def _default_emit(event):
    print(json.dumps(event, sort_keys=True), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Phase 3 CPU Track Stage A1 preparation.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prep = sub.add_parser("prepare")
    p_prep.add_argument("--phase1-root", default=None)
    p_prep.add_argument("--protocol", default=str(REPO / "configs/protocol.json"))
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
        plan = load_phase3_cpu_plan(root, protocol, ref, bind=True)
        _default_emit({"event": "phase3_cpu_plan_verified", "status": "passed", "plan_ref": plain(plan.reference), "plan_identity": plan["plan_identity"]})
        return 0

    producer = current_producer(scope=args.storage_scope)
    plan = prepare_phase3_cpu(root, protocol, producer, run_id=args.run_id, progress=_default_emit)
    _default_emit({"event": "complete", "status": "passed", "plan_ref": plain(plan.reference), "plan_identity": plan["plan_identity"]})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        _default_emit({"status": "blocked", "blockers": [str(exc)]})
        raise SystemExit(2)
