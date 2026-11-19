"""Phase 2 Stage 2 CPU worker for E29: paired pooling gap contrast and gate publication.

Implements phase2.md §1.3-1.5 and §2.1-2.3:
- Consumes the 16 bound Stage 4 SCREEN midpoint blocks per model (`qwen` or `gemma`)
  and the frozen Stage 4 `(2000, 360)` bootstrap draw matrix via `src.stage4.metrics`.
- Computes within-draw paired `G(last_caption) - G(caption_mean)` and residual
  `G(caption_mean)` across the 14 fixed primary categories and equally weighted
  12 off-diagonal routes (plus per-route rows, 4 receiving diagonals, 25-category
  tables, AP/Brier/NLL/0.5-threshold companions, and `caption_only` diagnostic).
- Evaluates and publishes `phase2_e29_gate` via `src.phase2.gates`, then publishes
  `phase2_analysis_block` (embedding compact reusable `draw0_reference` for Stage 4
  `score_e07`) with explicit-ref resume and streamed JSON progress.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from src.common.contracts import LANGUAGES, ContractError, fingerprint
from src.common.storage import new_run_id, phase1_root
from src.phase2.gates import evaluate_e29_gates, load_e29_gate, publish_e29_gate
from src.phase2.interfaces import (
    CONDITIONS,
    KIND_ANALYSIS,
    MODELS,
    POOLINGS,
    REPO,
    SCHEMA_ANALYSIS,
    bind_phase1,
    check,
    contract,
    e29_screen_blocks,
    midpoint_depth,
    normalize_ref,
    plain,
    publish_phase2,
    require_open_split,
    verify_resume_ref,
)
from src.phase2.plan import load_phase2_plan
from src.stage4.metrics import (
    METRICS,
    binary_metrics,
    bootstrap_binary_metrics,
    bootstrap_counts,
    bootstrap_interval,
    fixed_macro,
    paired_image_alignment,
    probabilities,
    support_status,
    targets_for_prediction,
)
from src.stage4.plan import load_bootstrap
from src.stage4.predictions import load_prediction_block

LOSS_METRICS = frozenset({"brier", "nll"})
SUPERVISION_DIAGONAL = "within_context_source_fit"
SUPERVISION_OFF_DIAGONAL = "source_only_transfer"


def _peak_rss_bytes():
    try:
        import resource
        import sys

        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB; macOS reports bytes.
        return rss if sys.platform == "darwin" else rss * 1024
    except Exception:
        return None


def _emit_progress(progress, event, *, unit, completed, total, read_hash_s=0.0, compute_s=0.0, write_s=0.0, **extra):
    if progress is None:
        return
    payload = {
        "event": event,
        "unit": unit,
        "completed": int(completed),
        "total": int(total),
        "read_hash_s": round(float(read_hash_s), 4),
        "compute_s": round(float(compute_s), 4),
        "write_s": round(float(write_s), 4),
        "peak_rss_bytes": _peak_rss_bytes(),
    }
    payload.update(extra)
    progress(payload)


def _route_key(source_lang, target_lang):
    return f"{source_lang}->{target_lang}"


def _off_diagonal_pairs():
    return [(s, t) for s in LANGUAGES for t in LANGUAGES if s != t]


def _gap_value(metric, diag_val, route_val):
    if diag_val is None or route_val is None:
        return None
    if metric in LOSS_METRICS:
        return float(route_val - diag_val)
    return float(diag_val - route_val)


def _gap_draws(metric, diag_draws, route_draws):
    if metric in LOSS_METRICS:
        return route_draws - diag_draws
    return diag_draws - route_draws


def _diag_gain_value(metric, last_val, mean_val):
    if last_val is None or mean_val is None:
        return None
    if metric in LOSS_METRICS:
        return float(last_val - mean_val)
    return float(mean_val - last_val)


def _diag_gain_draws(metric, last_draws, mean_draws):
    if metric in LOSS_METRICS:
        return last_draws - mean_draws
    return mean_draws - last_draws


def _interval_record(draw_values):
    interval = bootstrap_interval(draw_values)
    if interval["valid_draws"] == 0:
        interval["reason"] = "no_valid_bootstrap_draws"
    return interval


def _category_status_and_reason(value, pos_n, neg_n, minimum, unestimable):
    status = support_status(value, pos_n, neg_n, minimum)
    if status == "ok":
        return status, None
    if status == "descriptive_low_support":
        return status, f"screen_support_below_minimum_{minimum}"
    if unestimable:
        return status, "unestimable_head"
    return status, "undefined_metric_or_zero_denominator"


def _summarize_macro_and_categories(
    *,
    categories,
    primary_indices,
    cat_values,
    cat_draws,
    cat_supports,
    cat_unestimable,
    minimum,
):
    """Build 25-category table and 14-primary macro summary from aligned category arrays."""
    per_category = {}
    for col, cat in enumerate(categories):
        known_n, pos_n, neg_n = cat_supports[col]
        val = cat_values[col]
        status, reason = _category_status_and_reason(
            val, pos_n, neg_n, minimum, bool(cat_unestimable[col])
        )
        per_category[cat] = {
            "value": None if val is None else float(val),
            "interval": _interval_record(cat_draws[col]),
            "status": status,
            "reason": reason,
            "support": {"known": int(known_n), "positive": int(pos_n), "negative": int(neg_n)},
        }

    primary_vals = [cat_values[idx] for idx in primary_indices]
    defined_count = sum(v is not None for v in primary_vals)
    ok_count = sum(
        cat_values[idx] is not None and min(cat_supports[idx][1], cat_supports[idx][2]) >= minimum
        for idx in primary_indices
    )
    if defined_count < len(primary_indices):
        macro_val = None
        macro_status = "undefined"
        macro_reason = "unresolved_primary_macro"
    else:
        macro_val = float(fixed_macro([0.0 if v is None else v for v in cat_values], primary_indices))
        if ok_count == len(primary_indices):
            macro_status = "ok"
            macro_reason = None
        else:
            macro_status = "descriptive_low_support"
            macro_reason = f"primary_category_support_below_minimum_{minimum}"

    stacked = np.stack(cat_draws, axis=0)
    macro_draws = fixed_macro(stacked, primary_indices, axis=0)
    primary_macro = {
        "value": macro_val,
        "interval": _interval_record(macro_draws),
        "status": macro_status,
        "reason": macro_reason,
        "support": {
            "categories": len(primary_indices),
            "defined_categories": defined_count,
            "ok_categories": ok_count,
        },
    }
    return primary_macro, macro_draws, per_category


def _evaluate_arm(pred, aligned, y, known, counts, categories, total_draws):
    """Evaluate one route's 25-category point estimates, bootstrap draws, and support."""
    n_cats = len(categories)
    probs = probabilities(pred.logits[aligned])
    estimable = np.asarray(pred.estimable, dtype=bool)
    check(estimable.shape == (n_cats,), f"Unexpected estimable shape: {estimable.shape}")
    nan_draws = np.full(total_draws, np.nan, dtype="f8")

    points = {m: [] for m in METRICS}
    draws = {m: [] for m in METRICS}
    supports = []
    for col in range(n_cats):
        y_col = y[:, col]
        known_col = known[:, col]
        known_n = int(known_col.sum())
        pos_n = int(y_col[known_col].sum())
        neg_n = known_n - pos_n
        supports.append((known_n, pos_n, neg_n))
        if estimable[col]:
            p_col = probs[:, col]
            bm = binary_metrics(y_col, p_col, known_col)
            bd = bootstrap_binary_metrics(y_col, p_col, known_col, counts)
            for m in METRICS:
                points[m].append(bm["metrics"][m])
                draws[m].append(bd[m])
        else:
            for m in METRICS:
                points[m].append(None)
                draws[m].append(nan_draws)
    return {
        "points": points,
        "draws": draws,
        "supports": supports,
        "unestimable": [not bool(estimable[col]) for col in range(n_cats)],
    }


def _evaluate_condition(
    *,
    condition,
    blocks_by_pooling,
    aligned_by_key,
    y,
    known,
    counts,
    categories,
    primary_indices,
    total_draws,
    minimum,
):
    """Compute all E29 paired contrasts for one condition (`image_caption` or `caption_only`)."""
    n_cats = len(categories)
    arms = {}
    for pooling in POOLINGS:
        for target_lang in LANGUAGES:
            block_routes = blocks_by_pooling[pooling][target_lang]
            for route_id, pred in block_routes.items():
                src_lang = pred.route["source_language"] if "source_language" in pred.route else None
                if src_lang is None:
                    # Stage 4 route dicts carry route_id and group ids; resolve via route_id or block
                    raise ContractError(f"Missing source language metadata on route {route_id}")
                key = (pooling, src_lang, target_lang)
                arms[key] = _evaluate_arm(
                    pred,
                    aligned_by_key[(condition, pooling, src_lang, target_lang)],
                    y,
                    known,
                    counts,
                    categories,
                    total_draws,
                )

    routes_all = {pooling: {} for pooling in POOLINGS}
    diagonals = {}
    per_route = {}
    twelve_route_means = {}

    for metric in METRICS:
        # 1. All 16 routes per pooling
        for pooling in POOLINGS:
            for s in LANGUAGES:
                for t in LANGUAGES:
                    rkey = _route_key(s, t)
                    arm = arms[(pooling, s, t)]
                    macro_summary, _, cat_table = _summarize_macro_and_categories(
                        categories=categories,
                        primary_indices=primary_indices,
                        cat_values=arm["points"][metric],
                        cat_draws=arm["draws"][metric],
                        cat_supports=arm["supports"],
                        cat_unestimable=arm["unestimable"],
                        minimum=minimum,
                    )
                    entry = routes_all[pooling].setdefault(
                        rkey,
                        {
                            "route": rkey,
                            "source_language": s,
                            "target_language": t,
                            "supervision_label": SUPERVISION_DIAGONAL if s == t else SUPERVISION_OFF_DIAGONAL,
                            "metrics": {},
                        },
                    )
                    entry["metrics"][metric] = {
                        "primary_macro": macro_summary,
                        "categories": cat_table,
                    }

        # 2. 4 receiving diagonals (t -> t) + delta_d = caption_mean improvement over last_caption
        for t in LANGUAGES:
            last_arm = arms[("last_caption", t, t)]
            mean_arm = arms[("caption_mean", t, t)]
            gain_vals = [
                _diag_gain_value(metric, last_arm["points"][metric][c], mean_arm["points"][metric][c])
                for c in range(n_cats)
            ]
            gain_draws = [
                _diag_gain_draws(metric, last_arm["draws"][metric][c], mean_arm["draws"][metric][c])
                for c in range(n_cats)
            ]
            gain_unest = [
                last_arm["unestimable"][c] or mean_arm["unestimable"][c] for c in range(n_cats)
            ]
            gain_macro, _, gain_cats = _summarize_macro_and_categories(
                categories=categories,
                primary_indices=primary_indices,
                cat_values=gain_vals,
                cat_draws=gain_draws,
                cat_supports=mean_arm["supports"],
                cat_unestimable=gain_unest,
                minimum=minimum,
            )
            diag_entry = diagonals.setdefault(
                t,
                {
                    "route": _route_key(t, t),
                    "language": t,
                    "supervision_label": SUPERVISION_DIAGONAL,
                    "metrics": {},
                },
            )
            diag_entry["metrics"][metric] = {
                "last_caption": routes_all["last_caption"][_route_key(t, t)]["metrics"][metric],
                "caption_mean": routes_all["caption_mean"][_route_key(t, t)]["metrics"][metric],
                "delta_d": {
                    "primary_macro": gain_macro,
                    "categories": gain_cats,
                },
            }

        # 3. 12 off-diagonal routes (s != t): G_last, G_res (G_mean), delta_g = G_last - G_mean
        off_pairs = _off_diagonal_pairs()
        route_g_last_macro_draws = []
        route_g_res_macro_draws = []
        route_delta_g_macro_draws = []
        route_g_last_cat_draws = [[] for _ in range(n_cats)]
        route_g_res_cat_draws = [[] for _ in range(n_cats)]
        route_delta_g_cat_draws = [[] for _ in range(n_cats)]
        route_g_last_cat_vals = [[] for _ in range(n_cats)]
        route_g_res_cat_vals = [[] for _ in range(n_cats)]
        route_delta_g_cat_vals = [[] for _ in range(n_cats)]
        route_g_last_macro_vals = []
        route_g_res_macro_vals = []
        route_delta_g_macro_vals = []

        for s, t in off_pairs:
            rkey = _route_key(s, t)
            route_last = arms[("last_caption", s, t)]
            diag_last = arms[("last_caption", t, t)]
            route_mean = arms[("caption_mean", s, t)]
            diag_mean = arms[("caption_mean", t, t)]

            g_last_vals = [
                _gap_value(metric, diag_last["points"][metric][c], route_last["points"][metric][c])
                for c in range(n_cats)
            ]
            g_last_draws = [
                _gap_draws(metric, diag_last["draws"][metric][c], route_last["draws"][metric][c])
                for c in range(n_cats)
            ]
            g_last_unest = [
                diag_last["unestimable"][c] or route_last["unestimable"][c] for c in range(n_cats)
            ]
            g_last_macro, g_last_mdraws, g_last_cats = _summarize_macro_and_categories(
                categories=categories,
                primary_indices=primary_indices,
                cat_values=g_last_vals,
                cat_draws=g_last_draws,
                cat_supports=route_last["supports"],
                cat_unestimable=g_last_unest,
                minimum=minimum,
            )

            g_res_vals = [
                _gap_value(metric, diag_mean["points"][metric][c], route_mean["points"][metric][c])
                for c in range(n_cats)
            ]
            g_res_draws = [
                _gap_draws(metric, diag_mean["draws"][metric][c], route_mean["draws"][metric][c])
                for c in range(n_cats)
            ]
            g_res_unest = [
                diag_mean["unestimable"][c] or route_mean["unestimable"][c] for c in range(n_cats)
            ]
            g_res_macro, g_res_mdraws, g_res_cats = _summarize_macro_and_categories(
                categories=categories,
                primary_indices=primary_indices,
                cat_values=g_res_vals,
                cat_draws=g_res_draws,
                cat_supports=route_mean["supports"],
                cat_unestimable=g_res_unest,
                minimum=minimum,
            )

            delta_g_vals = [
                None if (g_last_vals[c] is None or g_res_vals[c] is None) else float(g_last_vals[c] - g_res_vals[c])
                for c in range(n_cats)
            ]
            delta_g_draws = [g_last_draws[c] - g_res_draws[c] for c in range(n_cats)]
            delta_g_unest = [g_last_unest[c] or g_res_unest[c] for c in range(n_cats)]
            delta_g_macro, delta_g_mdraws, delta_g_cats = _summarize_macro_and_categories(
                categories=categories,
                primary_indices=primary_indices,
                cat_values=delta_g_vals,
                cat_draws=delta_g_draws,
                cat_supports=route_mean["supports"],
                cat_unestimable=delta_g_unest,
                minimum=minimum,
            )

            route_entry = per_route.setdefault(
                rkey,
                {
                    "route": rkey,
                    "source_language": s,
                    "target_language": t,
                    "supervision_label": SUPERVISION_OFF_DIAGONAL,
                    "metrics": {},
                },
            )
            route_entry["metrics"][metric] = {
                "g_last": {"primary_macro": g_last_macro, "categories": g_last_cats},
                "g_res": {"primary_macro": g_res_macro, "categories": g_res_cats},
                "delta_g": {"primary_macro": delta_g_macro, "categories": delta_g_cats},
            }

            route_g_last_macro_vals.append(g_last_macro["value"])
            route_g_res_macro_vals.append(g_res_macro["value"])
            route_delta_g_macro_vals.append(delta_g_macro["value"])
            route_g_last_macro_draws.append(g_last_mdraws)
            route_g_res_macro_draws.append(g_res_mdraws)
            route_delta_g_macro_draws.append(delta_g_mdraws)
            for c in range(n_cats):
                route_g_last_cat_vals[c].append(g_last_vals[c])
                route_g_res_cat_vals[c].append(g_res_vals[c])
                route_delta_g_cat_vals[c].append(delta_g_vals[c])
                route_g_last_cat_draws[c].append(g_last_draws[c])
                route_g_res_cat_draws[c].append(g_res_draws[c])
                route_delta_g_cat_draws[c].append(delta_g_draws[c])

        # 4. 12-route equal-weight means (computed within every draw b)
        def _mean_summary(macro_vals, macro_draws_list, cat_vals_by_c, cat_draws_by_c):
            m_val = None if any(v is None for v in macro_vals) else float(sum(macro_vals) / len(macro_vals))
            m_draws = np.mean(np.stack(macro_draws_list, axis=0), axis=0)
            cat_table = {}
            for c, cat in enumerate(categories):
                c_vals = cat_vals_by_c[c]
                c_val = None if any(v is None for v in c_vals) else float(sum(c_vals) / len(c_vals))
                c_draws = np.mean(np.stack(cat_draws_by_c[c], axis=0), axis=0)
                cat_table[cat] = {
                    "value": c_val,
                    "interval": _interval_record(c_draws),
                }
            return {
                "primary_macro": {
                    "value": m_val,
                    "interval": _interval_record(m_draws),
                },
                "categories": cat_table,
            }

        twelve_route_means[metric] = {
            "g_last_mean": _mean_summary(
                route_g_last_macro_vals,
                route_g_last_macro_draws,
                route_g_last_cat_vals,
                route_g_last_cat_draws,
            ),
            "g_res_mean": _mean_summary(
                route_g_res_macro_vals,
                route_g_res_macro_draws,
                route_g_res_cat_vals,
                route_g_res_cat_draws,
            ),
            "delta_g_mean": _mean_summary(
                route_delta_g_macro_vals,
                route_delta_g_macro_draws,
                route_delta_g_cat_vals,
                route_delta_g_cat_draws,
            ),
        }

    return {
        "condition": condition,
        "twelve_route_means": twelve_route_means,
        "per_route": per_route,
        "diagonals": diagonals,
        "routes_all": routes_all,
    }


def _build_draw0_reference(
    *,
    model,
    depth,
    screen_ids,
    target_mask_fingerprint,
    categories,
    primary_categories,
    image_caption_blocks_meta,
    image_caption_results,
):
    """Build compact reusable draw-0 midpoint reference for Stage 4 `score_e07`."""
    routes_summary = {pooling: {} for pooling in POOLINGS}
    for pooling in POOLINGS:
        for rkey, rdata in image_caption_results["routes_all"][pooling].items():
            routes_summary[pooling][rkey] = {
                m: {
                    "primary_macro": rdata["metrics"][m]["primary_macro"]["value"],
                    "categories": {
                        cat: rdata["metrics"][m]["categories"][cat]["value"]
                        for cat in categories
                    },
                }
                for m in METRICS
            }

    per_route_summary = {}
    for rkey, rdata in image_caption_results["per_route"].items():
        per_route_summary[rkey] = {
            m: {
                "g_last": rdata["metrics"][m]["g_last"]["primary_macro"]["value"],
                "g_res": rdata["metrics"][m]["g_res"]["primary_macro"]["value"],
                "delta_g": rdata["metrics"][m]["delta_g"]["primary_macro"]["value"],
            }
            for m in METRICS
        }

    diagonals_summary = {
        t: {
            m: {
                "last_caption": image_caption_results["diagonals"][t]["metrics"][m]["last_caption"]["primary_macro"]["value"],
                "caption_mean": image_caption_results["diagonals"][t]["metrics"][m]["caption_mean"]["primary_macro"]["value"],
                "delta_d": image_caption_results["diagonals"][t]["metrics"][m]["delta_d"]["primary_macro"]["value"],
            }
            for m in METRICS
        }
        for t in LANGUAGES
    }

    twelve_route_summary = {
        m: {
            "g_last_mean": image_caption_results["twelve_route_means"][m]["g_last_mean"]["primary_macro"]["value"],
            "g_res_mean": image_caption_results["twelve_route_means"][m]["g_res_mean"]["primary_macro"]["value"],
            "delta_g_mean": image_caption_results["twelve_route_means"][m]["delta_g_mean"]["primary_macro"]["value"],
        }
        for m in METRICS
    }

    return {
        "model": model,
        "midpoint_depth": int(depth),
        "condition": "image_caption",
        "split": "SCREEN",
        "n_images": len(screen_ids),
        "categories": list(categories),
        "primary_categories": list(primary_categories),
        "screen_ids_sha256": fingerprint(list(screen_ids)),
        "target_mask_fingerprint": target_mask_fingerprint,
        "blocks": plain(image_caption_blocks_meta),
        "point_estimates": {
            "routes": routes_summary,
            "per_route": per_route_summary,
            "diagonals": diagonals_summary,
            "twelve_route_mean": twelve_route_summary,
        },
    }


def _gate_inputs_from_results(image_caption_results):
    """Extract primary AUROC estimands required by `evaluate_e29_gates`."""
    diagonals = {
        t: image_caption_results["diagonals"][t]["metrics"]["auroc"]["caption_mean"]["primary_macro"]["value"]
        for t in LANGUAGES
    }
    means_auroc = image_caption_results["twelve_route_means"]["auroc"]
    dg_macro = means_auroc["delta_g_mean"]["primary_macro"]
    gres_macro = means_auroc["g_res_mean"]["primary_macro"]

    dg_ci = (
        None
        if (dg_macro["interval"]["lower"] is None or dg_macro["interval"]["upper"] is None)
        else [dg_macro["interval"]["lower"], dg_macro["interval"]["upper"]]
    )
    gres_ci = (
        None
        if (gres_macro["interval"]["lower"] is None or gres_macro["interval"]["upper"] is None)
        else [gres_macro["interval"]["lower"], gres_macro["interval"]["upper"]]
    )

    route_residuals = {}
    for rkey, rdata in image_caption_results["per_route"].items():
        r_gres = rdata["metrics"]["auroc"]["g_res"]["primary_macro"]
        r_ci = (
            None
            if (r_gres["interval"]["lower"] is None or r_gres["interval"]["upper"] is None)
            else [r_gres["interval"]["lower"], r_gres["interval"]["upper"]]
        )
        route_residuals[rkey] = {
            "point": r_gres["value"],
            "ci": r_ci,
        }

    return {
        "diagonals": diagonals,
        "valid_draws": int(dg_macro["interval"]["valid_draws"]),
        "delta_g_mean": dg_macro["value"],
        "delta_g_ci": dg_ci,
        "g_res_mean": gres_macro["value"],
        "g_res_ci": gres_ci,
        "route_residuals": route_residuals,
    }


@contract
def run_e29(
    root,
    protocol,
    plan_ref,
    model,
    producer,
    *,
    resume_analysis_ref=None,
    resume_gate_ref=None,
    run_id=None,
    progress=None,
):
    """Execute E29 CPU paired pooling gap contrast and gate publication for one model."""
    check(model in MODELS, f"Unknown model selector: {model!r}")
    resolved_root = Path(root).resolve()
    require_open_split("SCREEN")

    t0 = time.monotonic()
    plan = load_phase2_plan(resolved_root, protocol, plan_ref)
    check(plan["confirm_excluded"] is True, "phase2_plan must enforce confirm_excluded=True")
    binding = bind_phase1(resolved_root, protocol, plain(plan["stage4_release"]))
    check(binding.lineage_identity == plan["lineage_identity"], "Phase 1 lineage_identity mismatch with phase2_plan")
    check(binding.dataset_fingerprint == plan["dataset_fingerprint"], "Dataset fingerprint mismatch with phase2_plan")
    check(binding.bootstrap_sha256 == plan["bootstrap"]["sha256"], "Bootstrap SHA-256 mismatch with phase2_plan")
    check(list(binding.bootstrap_shape) == list(plan["bootstrap"]["shape"]), "Bootstrap shape mismatch with phase2_plan")

    blocks_meta = e29_screen_blocks(binding, model)
    check(plain(blocks_meta) == plain(plan["e29_blocks"][model]), "Resolved E29 blocks differ from phase2_plan")
    depth = midpoint_depth(binding, model)
    read_hash_s = time.monotonic() - t0
    _emit_progress(
        progress,
        "e29_plan_bound",
        unit="init",
        completed=0,
        total=3,
        read_hash_s=read_hash_s,
        model=model,
        midpoint_depth=depth,
    )

    if resume_analysis_ref is not None:
        t_res = time.monotonic()
        res_ref, res_body, _ = verify_resume_ref(
            resolved_root,
            protocol,
            resume_analysis_ref,
            KIND_ANALYSIS,
            expected_schema=SCHEMA_ANALYSIS,
            expected_dataset=plan["dataset_fingerprint"],
            expected_plan_identity=plan["plan_identity"],
            expected_owner=model,
        )
        check(res_body.get("experiment") == "E29", "Resume analysis artifact is not E29")
        check(res_body.get("confirm_excluded") is True, "Resume analysis artifact missing confirm_excluded=True")
        check(int(res_body.get("midpoint_depth", -1)) == depth, "Resume analysis midpoint_depth mismatch")
        check(plain(res_body.get("bootstrap")) == plain(plan["bootstrap"]), "Resume analysis bootstrap mismatch")
        check(plain(res_body.get("input_blocks")) == plain(blocks_meta), "Resume analysis input_blocks mismatch")
        gate_ref, gate_body, _ = load_e29_gate(
            resolved_root,
            protocol,
            res_body["gate_ref"],
            plan=plan,
            expected_model=model,
        )
        if resume_gate_ref is not None:
            explicit_gate_ref = normalize_ref(resolved_root, resume_gate_ref)
            check(plain(gate_ref) == plain(explicit_gate_ref), "Resume gate ref differs from analysis gate_ref")
        _emit_progress(
            progress,
            "e29_resumed",
            unit="resume",
            completed=3,
            total=3,
            read_hash_s=time.monotonic() - t_res,
            model=model,
            analysis_ref=plain(res_ref),
            gate_ref=plain(gate_ref),
        )
        return res_ref, res_body, gate_ref, gate_body

    # Validate explicit resume_gate_ref early if supplied alone.
    verified_resume_gate = None
    if resume_gate_ref is not None:
        verified_resume_gate = load_e29_gate(
            resolved_root,
            protocol,
            resume_gate_ref,
            plan=plan,
            expected_model=model,
        )

    # Load frozen bootstrap and all 16 Stage 4 SCREEN midpoint blocks.
    t_load = time.monotonic()
    draws = load_bootstrap(binding.stage4_plan)
    screen_ids = tuple(binding.stage4_plan["screen_ids"])
    counts = bootstrap_counts(draws, len(screen_ids))
    total_draws = int(draws.shape[0])

    loaded_blocks = {}
    all_preds = []
    pred_keys = []
    for condition in CONDITIONS:
        loaded_blocks[condition] = {}
        for pooling in POOLINGS:
            loaded_blocks[condition][pooling] = {}
            for target_lang in LANGUAGES:
                entry = blocks_meta[condition][pooling][target_lang]
                block_preds = load_prediction_block(
                    binding.stage4_plan,
                    entry["prediction_ref"],
                    expected_block=entry["block_id"],
                )
                check(set(block_preds.keys()) == set(entry["routes"].values()),
                      f"Route mismatch in Stage 4 block {entry['block_id']}")
                # Attach source_language onto Prediction.route copy via dict lookup
                by_route_id = {}
                for src_lang, rid in entry["routes"].items():
                    pred = block_preds[rid]
                    by_route_id[rid] = (src_lang, pred)
                    all_preds.append(pred)
                    pred_keys.append((condition, pooling, src_lang, target_lang))
                loaded_blocks[condition][pooling][target_lang] = {
                    rid: _PredictionWithSource(pred, src_lang)
                    for rid, (src_lang, pred) in by_route_id.items()
                }

    ids, aligned_tuple = paired_image_alignment(binding.stage4_plan, *all_preds)
    check(ids == screen_ids, "Aligned SCREEN IDs mismatch plan screen_ids")
    aligned_by_key = {key: aligned for key, aligned in zip(pred_keys, aligned_tuple)}

    y_raw, known_raw = targets_for_prediction(binding.stage4_plan, all_preds[0])
    aligned_0 = aligned_tuple[0]
    y = y_raw[aligned_0]
    known = known_raw[aligned_0]
    target_mask_fp = fingerprint({
        "screen_ids": list(ids),
        "y_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
        "known_sha256": hashlib.sha256(known.tobytes()).hexdigest(),
    })

    categories = tuple(binding.stage4_plan["categories"])
    primary_categories = tuple(protocol["primary_categories"])
    cat_index = {cat: idx for idx, cat in enumerate(categories)}
    primary_indices = [cat_index[cat] for cat in primary_categories]
    read_blocks_s = time.monotonic() - t_load
    _emit_progress(
        progress,
        "e29_blocks_verified",
        unit="load_blocks",
        completed=1,
        total=3,
        read_hash_s=read_blocks_s,
        model=model,
        blocks=16,
        routes=len(all_preds),
    )

    # Compute paired contrasts per condition.
    t_comp = time.monotonic()
    condition_results = {}
    for condition in CONDITIONS:
        condition_results[condition] = _evaluate_condition(
            condition=condition,
            blocks_by_pooling=loaded_blocks[condition],
            aligned_by_key=aligned_by_key,
            y=y,
            known=known,
            counts=counts,
            categories=categories,
            primary_indices=primary_indices,
            total_draws=total_draws,
            minimum=int(protocol["minimum_positive_and_negative"]["SCREEN"]),
        )
    compute_s = time.monotonic() - t_comp
    _emit_progress(
        progress,
        "e29_contrasts_computed",
        unit="compute_contrasts",
        completed=2,
        total=3,
        compute_s=compute_s,
        model=model,
    )

    # Evaluate gates on primary image_caption AUROC estimands.
    gate_inputs = _gate_inputs_from_results(condition_results["image_caption"])
    evaluation = evaluate_e29_gates(model, **gate_inputs)

    draw0_reference = _build_draw0_reference(
        model=model,
        depth=depth,
        screen_ids=ids,
        target_mask_fingerprint=target_mask_fp,
        categories=categories,
        primary_categories=primary_categories,
        image_caption_blocks_meta=blocks_meta["image_caption"],
        image_caption_results=condition_results["image_caption"],
    )

    analysis_run_id = run_id or new_run_id()
    gate_run_id = f"{analysis_run_id}-gate"

    core_analysis_body = {
        "schema": SCHEMA_ANALYSIS,
        "experiment": "E29",
        "plan": plain(plan.reference),
        "plan_identity": plan["plan_identity"],
        "model": model,
        "midpoint_depth": depth,
        "bootstrap": plain(plan["bootstrap"]),
        "confirm_excluded": True,
        "input_blocks": plain(blocks_meta),
        "image_caption": condition_results["image_caption"],
        "caption_only": condition_results["caption_only"],
        "draw0_reference": draw0_reference,
        "gate_decisions": plain(evaluation["decisions"]),
        "supervision_labels": {
            "diagonal": SUPERVISION_DIAGONAL,
            "off_diagonal": SUPERVISION_OFF_DIAGONAL,
        },
    }

    t_write = time.monotonic()
    if verified_resume_gate is not None:
        gate_ref, gate_body, _ = verified_resume_gate
        check(plain(gate_body["observed"]) == plain(evaluation["observed"]),
              "Resume gate observed values do not match current E29 evaluation")
        check(plain(gate_body["decisions"]) == plain(evaluation["decisions"]),
              "Resume gate decisions do not match current E29 evaluation")
    else:
        expected_analysis_ref = {
            "path": f"phase2/results/e29/{model}/{analysis_run_id}/_COMPLETE.json",
            "sha256": fingerprint(core_analysis_body),
        }
        gate_ref = publish_e29_gate(
            resolved_root,
            protocol,
            plan,
            model,
            expected_analysis_ref,
            evaluation,
            producer,
            run_id=gate_run_id,
        )
        _, gate_body, _ = load_e29_gate(
            resolved_root,
            protocol,
            gate_ref,
            plan=plan,
            expected_model=model,
        )

    analysis_body = dict(core_analysis_body, gate_ref=plain(gate_ref))
    upstream_refs = [plain(plan.reference), plain(gate_ref)]
    for condition in CONDITIONS:
        for pooling in POOLINGS:
            for lang in LANGUAGES:
                upstream_refs.append(plain(blocks_meta[condition][pooling][lang]["prediction_ref"]))

    analysis_ref = publish_phase2(
        resolved_root,
        protocol,
        plan["dataset_fingerprint"],
        f"phase2/results/e29/{model}",
        KIND_ANALYSIS,
        analysis_body,
        producer,
        upstream=upstream_refs,
        run_id=analysis_run_id,
    )
    _, verified_analysis_body, _ = verify_resume_ref(
        resolved_root,
        protocol,
        analysis_ref,
        KIND_ANALYSIS,
        expected_schema=SCHEMA_ANALYSIS,
        expected_dataset=plan["dataset_fingerprint"],
        expected_plan_identity=plan["plan_identity"],
        expected_owner=model,
    )
    write_s = time.monotonic() - t_write

    _emit_progress(
        progress,
        "e29_published",
        unit="publish",
        completed=3,
        total=3,
        write_s=write_s,
        model=model,
        analysis_ref=plain(analysis_ref),
        gate_ref=plain(gate_ref),
        decisions=plain(evaluation["decisions"]),
        valid_draws=gate_inputs["valid_draws"],
    )
    return analysis_ref, verified_analysis_body, gate_ref, gate_body


class _PredictionWithSource:
    """Lightweight wrapper attaching `source_language` onto a Stage 4 Prediction."""

    def __init__(self, pred, source_language):
        self.reference = pred.reference
        self.route = dict(pred.route, source_language=source_language)
        self.split = pred.split
        self.rows = pred.rows
        self.logits = pred.logits
        self.estimable = pred.estimable
        self.prevalence = pred.prevalence


def _parse_ref_arg(value):
    if value is None:
        return None
    text = value.strip()
    if text.startswith("{"):
        parsed = json.loads(text)
        check(isinstance(parsed, dict), "JSON reference argument must decode to an object")
        return parsed
    return text


def _emit_cli(event):
    print(json.dumps(event), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-root", required=True)
    parser.add_argument("--protocol", type=Path, default=REPO / "configs/protocol.json")
    parser.add_argument("--plan", required=True, help="Explicit phase2_plan {'path', 'sha256'} JSON or path")
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument(
        "--storage-scope",
        default="persistent-drive",
        choices=("persistent-drive", "local-development"),
    )
    parser.add_argument("--resume-analysis-ref", help="Optional explicit E29 analysis {'path', 'sha256'} or path")
    parser.add_argument("--resume-gate-ref", help="Optional explicit E29 gate {'path', 'sha256'} or path")
    parser.add_argument("--run-id", help="Optional explicit run_id")
    parser.add_argument("--output-ref", type=Path, help="Optional JSON file path for emitted references")
    args = parser.parse_args(argv)

    root = phase1_root(args.phase1_root)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    from src.phase2.interfaces import current_producer

    producer = current_producer(args.storage_scope)
    _emit_cli({
        "event": "e29_start",
        "model": args.model,
        "code_commit": producer["code_commit"],
        "storage_scope": producer["storage_scope"],
    })
    analysis_ref, analysis_body, gate_ref, gate_body = run_e29(
        root,
        protocol,
        _parse_ref_arg(args.plan),
        args.model,
        producer,
        resume_analysis_ref=_parse_ref_arg(args.resume_analysis_ref),
        resume_gate_ref=_parse_ref_arg(args.resume_gate_ref),
        run_id=args.run_id,
        progress=_emit_cli,
    )
    summary = {
        "event": "e29_complete",
        "status": "passed",
        "model": args.model,
        "analysis_ref": plain(analysis_ref),
        "gate_ref": plain(gate_ref),
        "decisions": plain(gate_body["decisions"]),
        "valid_draws": int(gate_body["observed"]["valid_draws"]),
    }
    if args.output_ref is not None:
        args.output_ref.parent.mkdir(parents=True, exist_ok=True)
        args.output_ref.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _emit_cli(summary)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        _emit_cli({"status": "blocked", "blockers": [str(exc)], "stage2_e29_complete": False})
        raise SystemExit(2)
