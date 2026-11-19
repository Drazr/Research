"""CPU analysis worker for Stage 4 E01-E04 using canonical prediction releases."""

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from src.common.contracts import ContractError
from src.common.storage import phase1_root
from src.stage2.schema import REPO, check, contract, plain
from src.stage4.interfaces import (
    MACRO,
    METRICS,
    analysis_units,
    binary_metrics,
    bootstrap_binary_metrics,
    bootstrap_counts,
    bootstrap_interval,
    current_producer,
    fixed_macro,
    load_analysis_block,
    load_bootstrap,
    load_stage4_plan,
    paired_image_alignment,
    prediction_index,
    probabilities,
    publish_analysis_block,
    required_rows,
    support_status,
    targets_for_prediction,
)
from src.stage4.predictions import PREDICTION_KIND

E01_E04_EXPERIMENTS = ("E01", "E02", "E03", "E04")
E01_E04_OWNERS = ("e03", "qwen", "gemma")
LOSS_METRICS = frozenset({"brier", "nll"})
PAIRED_KINDS = frozenset({
    "transfer_gap",
    "transfer_gap_change",
    "input_delta",
    "visual_delta",
    "pool_delta",
    "neural_minus_shallow",
})
BASELINE_KINDS = frozenset({"fit_prior", "always_absent"})
SINGLE_PROBE_KINDS = frozenset({"absolute", "caption_only", "caption_mean", "image_only"})
E04_UNAVAILABLE_ARMS = (
    {
        "experiment": "E04",
        "arm": "post_caption",
        "status": "unavailable",
        "reason": "The accepted Stage 2 plan did not expose this decoder site.",
    },
    {
        "experiment": "E04",
        "arm": "decoder_visual_pool",
        "status": "unavailable",
        "reason": "The accepted Stage 2 plan did not expose decoder visual-token pooling.",
    },
)


def emit(event):
    print(json.dumps(event), flush=True)


def _experiment_rank(unit_id, unit):
    exp_idx = E01_E04_EXPERIMENTS.index(unit["experiment"])
    owner_idx = E01_E04_OWNERS.index(unit["owner"])
    part = unit["part"]
    part_key = (0, int(part), "") if part.isdigit() else (1, 0, part)
    return (owner_idx, exp_idx, part_key, unit_id)


def select_e01_e04_units(plan, *, owner=None, experiments=E01_E04_EXPERIMENTS, unit_ids=None):
    """Return deterministically ordered E01-E04 unit IDs for the requested scope."""
    check(owner is None or owner in (*E01_E04_OWNERS, "all"),
          f"Unknown E01-E04 owner: {owner}")
    selected_exps = tuple(experiments)
    check(bool(selected_exps) and all(exp in E01_E04_EXPERIMENTS for exp in selected_exps),
          "Experiments must be a non-empty subset of E01, E02, E03, E04")
    units = analysis_units(plan)
    eligible = {
        uid: unit for uid, unit in units.items()
        if unit["experiment"] in selected_exps
        and (owner in (None, "all") or unit["owner"] == owner)
    }
    if unit_ids is not None:
        requested = tuple(unit_ids)
        check(bool(requested), "At least one unit ID required when --unit is specified")
        for uid in requested:
            check(uid in eligible, f"Unit {uid} is not in the selected E01-E04 scope")
        return tuple(sorted(set(requested), key=lambda uid: _experiment_rank(uid, eligible[uid])))
    return tuple(sorted(eligible, key=lambda uid: _experiment_rank(uid, eligible[uid])))


def _metric_denominator(metric, known_n, pos_n, neg_n, pred_pos_n):
    if metric == "auroc":
        return pos_n * neg_n
    if metric in ("ap", "sensitivity"):
        return pos_n
    if metric == "specificity":
        return neg_n
    if metric == "f1":
        return pred_pos_n + pos_n
    return known_n


def _eval_arm(plan, blocks, counts, block_id, route_id, view, cache):
    """Evaluate one aligned SCREEN arm once per unit and cache point + bootstrap results."""
    key = (block_id, route_id, view)
    cached = cache.get(key)
    if cached is not None:
        return cached
    pred = blocks[block_id][route_id]
    check(pred.split == "SCREEN", "E01-E04 scoring requires SCREEN predictions")
    ids, (aligned,) = paired_image_alignment(plan, pred, contexts=(view,))
    y_raw, known_raw = targets_for_prediction(plan, pred)
    y = y_raw[aligned]
    known = known_raw[aligned]
    probs = probabilities(pred.logits[aligned])
    estimable = pred.estimable
    prevalence = pred.prevalence
    n_cats = len(plan["categories"])
    total_draws = counts.shape[0]
    nan_draws = np.full(total_draws, np.nan, dtype="f8")

    points = []
    draws = []
    supports = []
    for col in range(n_cats):
        y_col = y[:, col]
        known_col = known[:, col]
        known_n = int(known_col.sum())
        pos_n = int(y_col[known_col].sum())
        neg_n = known_n - pos_n
        if estimable[col]:
            p_col = probs[:, col]
            bm = binary_metrics(y_col, p_col, known_col)
            pred_pos_n = int((p_col[known_col] >= 0.5).sum()) if known_n else 0
            points.append(bm["metrics"])
            draws.append(bootstrap_binary_metrics(y_col, p_col, known_col, counts))
            supports.append((known_n, pos_n, neg_n, pred_pos_n))
        else:
            points.append(dict.fromkeys(METRICS))
            draws.append({metric: nan_draws for metric in METRICS})
            supports.append((known_n, pos_n, neg_n, None))

    result = {
        "pred": pred,
        "ids": ids,
        "y": y,
        "known": known,
        "estimable": estimable,
        "prevalence": prevalence,
        "points": points,
        "draws": draws,
        "supports": supports,
    }
    cache[key] = result
    return result


def _eval_baseline(y, known, prevalence, kind, categories):
    """Evaluate fixed FIT-prior or always-absent reference on SCREEN targets."""
    n_rows = y.shape[0]
    points = []
    supports = []
    for col in range(len(categories)):
        y_col = y[:, col]
        known_col = known[:, col]
        known_n = int(known_col.sum())
        pos_n = int(y_col[known_col].sum())
        neg_n = known_n - pos_n
        p_val = float(prevalence[col]) if kind == "fit_prior" else 0.0
        p_col = np.full(n_rows, p_val, dtype="f8")
        bm = binary_metrics(y_col, p_col, known_col)
        pred_pos_n = int((p_col[known_col] >= 0.5).sum()) if known_n else 0
        points.append(bm["metrics"])
        supports.append((known_n, pos_n, neg_n, pred_pos_n))
    return points, supports


def _transfer_gap_value(metric, diag_val, base_val):
    if diag_val is None or base_val is None:
        return None
    if metric in LOSS_METRICS:
        return float(base_val - diag_val)
    return float(diag_val - base_val)


def _transfer_gap_draws(metric, diag_draws, base_draws):
    if metric in LOSS_METRICS:
        return base_draws - diag_draws
    return diag_draws - base_draws


def _category_status_and_reason(value, pos_n, neg_n, minimum, unestimable):
    status = support_status(value, pos_n, neg_n, minimum)
    if status == "ok":
        return status, None
    if status == "descriptive_low_support":
        return status, f"screen_support_below_minimum_{minimum}"
    if unestimable:
        return status, "unestimable_head"
    return status, "undefined_metric_or_zero_denominator"


def _macro_row(cid, metric, cat_values, cat_supports, cat_draws, primary_indices, minimum, include_interval):
    primary_vals = [cat_values[idx] for idx in primary_indices]
    defined_count = sum(v is not None for v in primary_vals)
    ok_count = sum(
        cat_values[idx] is not None and min(cat_supports[idx][1], cat_supports[idx][2]) >= minimum
        for idx in primary_indices
    )
    support = {
        "categories": len(primary_indices),
        "defined_categories": defined_count,
        "ok_categories": ok_count,
    }
    if defined_count < len(primary_indices):
        value = None
        status = "undefined"
        reason = "unresolved_primary_macro"
    else:
        value = float(fixed_macro([0.0 if v is None else v for v in cat_values], primary_indices))
        if ok_count == len(primary_indices):
            status = "ok"
            reason = None
        else:
            status = "descriptive_low_support"
            reason = f"primary_category_support_below_minimum_{minimum}"
    row = {
        "comparison_id": cid,
        "category": MACRO,
        "statistic": metric,
        "value": value,
        "status": status,
        "reason": reason,
        "support": support,
    }
    if include_interval:
        stacked = np.stack(cat_draws, axis=0)
        macro_draws = fixed_macro(stacked, primary_indices, axis=0)
        row["interval"] = bootstrap_interval(macro_draws)
    return row


def _verify_aligned_arms(plan, arms, contexts):
    """Cross-check paired image alignment and shared target/mask agreement across arms."""
    if len(arms) <= 1:
        return
    ids, _ = paired_image_alignment(plan, *(arm["pred"] for arm in arms), contexts=contexts)
    ref_y, ref_known = arms[0]["y"], arms[0]["known"]
    for arm in arms[1:]:
        check(arm["ids"] == ids, "Paired arm image ordering mismatch")
        check(np.array_equal(arm["y"], ref_y) and np.array_equal(arm["known"], ref_known),
              "Paired arm target or known-cell mask mismatch")


def compute_e01_e04_unit(plan, unit_id, blocks, counts):
    """Compute all required tidy rows for one E01-E04 analysis unit."""
    units = analysis_units(plan)
    check(unit_id in units, f"Unknown Stage 4 analysis unit: {unit_id}")
    unit = units[unit_id]
    check(unit["experiment"] in E01_E04_EXPERIMENTS,
          f"Unit {unit_id} does not belong to E01-E04")

    categories = tuple(plan["categories"])
    primary_indices = tuple(categories.index(cat) for cat in plan["primary"])
    minimum = int(plan.protocol["minimum_positive_and_negative"]["SCREEN"])
    arm_cache = {}
    rows = []

    for cid, comp in unit["comparisons"].items():
        kind = comp["kind"]
        statistics = tuple(comp["statistics"])
        inputs = comp["inputs"]

        if kind in BASELINE_KINDS:
            check(len(inputs) == 1, f"{kind} expects 1 input")
            view = comp.get("view")
            arm = _eval_arm(plan, blocks, counts, inputs[0]["block_id"], inputs[0]["route_id"], view, arm_cache)
            points, supports = _eval_baseline(arm["y"], arm["known"], arm["prevalence"], kind, categories)
            for metric in statistics:
                cat_vals = []
                for col, category in enumerate(categories):
                    known_n, pos_n, neg_n, pred_pos_n = supports[col]
                    val = points[col][metric]
                    cat_vals.append(val)
                    status, reason = _category_status_and_reason(val, pos_n, neg_n, minimum, False)
                    rows.append({
                        "comparison_id": cid,
                        "category": category,
                        "statistic": metric,
                        "value": val,
                        "status": status,
                        "reason": reason,
                        "support": {
                            "known": known_n,
                            "positive": pos_n,
                            "negative": neg_n,
                            "denominator": _metric_denominator(metric, known_n, pos_n, neg_n, pred_pos_n),
                        },
                    })
                rows.append(_macro_row(cid, metric, cat_vals, supports, None, primary_indices, minimum, False))
            continue

        if kind in SINGLE_PROBE_KINDS:
            check(len(inputs) == 1, f"{kind} expects 1 input")
            view = comp.get("view")
            arm = _eval_arm(plan, blocks, counts, inputs[0]["block_id"], inputs[0]["route_id"], view, arm_cache)
            for metric in statistics:
                cat_vals = []
                cat_draws = []
                for col, category in enumerate(categories):
                    known_n, pos_n, neg_n, pred_pos_n = arm["supports"][col]
                    unest = not bool(arm["estimable"][col])
                    val = arm["points"][col][metric]
                    draw_vec = arm["draws"][col][metric]
                    cat_vals.append(val)
                    cat_draws.append(draw_vec)
                    status, reason = _category_status_and_reason(val, pos_n, neg_n, minimum, unest)
                    support = {"known": known_n, "positive": pos_n, "negative": neg_n}
                    if pred_pos_n is not None:
                        support["denominator"] = _metric_denominator(metric, known_n, pos_n, neg_n, pred_pos_n)
                    rows.append({
                        "comparison_id": cid,
                        "category": category,
                        "statistic": metric,
                        "value": val,
                        "status": status,
                        "reason": reason,
                        "support": support,
                        "interval": bootstrap_interval(draw_vec),
                    })
                rows.append(_macro_row(cid, metric, cat_vals, arm["supports"], cat_draws,
                                       primary_indices, minimum, True))
            continue

        if kind == "pooled_context_mean":
            check(len(inputs) == 1, "pooled_context_mean expects 1 input")
            views = tuple(comp["views"])
            check(bool(views), "pooled_context_mean requires views")
            arms = [
                _eval_arm(plan, blocks, counts, inputs[0]["block_id"], inputs[0]["route_id"], v, arm_cache)
                for v in views
            ]
            _verify_aligned_arms(plan, arms, views)
            for metric in statistics:
                cat_vals = []
                cat_draws = []
                for col, category in enumerate(categories):
                    known_n, pos_n, neg_n, _ = arms[0]["supports"][col]
                    unest = any(not bool(arm["estimable"][col]) for arm in arms)
                    view_vals = [arm["points"][col][metric] for arm in arms]
                    val = None if any(v is None for v in view_vals) else float(sum(view_vals) / len(views))
                    draw_vec = np.mean(np.stack([arm["draws"][col][metric] for arm in arms], axis=0), axis=0)
                    cat_vals.append(val)
                    cat_draws.append(draw_vec)
                    status, reason = _category_status_and_reason(val, pos_n, neg_n, minimum, unest)
                    rows.append({
                        "comparison_id": cid,
                        "category": category,
                        "statistic": metric,
                        "value": val,
                        "status": status,
                        "reason": reason,
                        "support": {
                            "known": known_n,
                            "positive": pos_n,
                            "negative": neg_n,
                            "views": len(views),
                        },
                        "interval": bootstrap_interval(draw_vec),
                    })
                rows.append(_macro_row(cid, metric, cat_vals, arms[0]["supports"], cat_draws,
                                       primary_indices, minimum, True))
            continue

        if kind in PAIRED_KINDS:
            if kind == "neural_minus_shallow":
                check(len(inputs) == 2, "neural_minus_shallow expects 2 inputs")
                contexts = (None, comp.get("view"))
            elif kind == "transfer_gap_change":
                check(len(inputs) == 4, "transfer_gap_change expects 4 inputs")
                contexts = (None, None, None, None)
            else:
                check(len(inputs) == 2, f"{kind} expects 2 inputs")
                contexts = (None, None)

            arms = [
                _eval_arm(plan, blocks, counts, inp["block_id"], inp["route_id"], ctx, arm_cache)
                for inp, ctx in zip(inputs, contexts, strict=True)
            ]
            _verify_aligned_arms(plan, arms, contexts)

            for metric in statistics:
                cat_vals = []
                cat_draws = []
                for col, category in enumerate(categories):
                    known_n, pos_n, neg_n, _ = arms[0]["supports"][col]
                    unest = any(not bool(arm["estimable"][col]) for arm in arms)
                    if kind == "transfer_gap":
                        val = _transfer_gap_value(metric, arms[0]["points"][col][metric],
                                                  arms[1]["points"][col][metric])
                        draw_vec = _transfer_gap_draws(metric, arms[0]["draws"][col][metric],
                                                       arms[1]["draws"][col][metric])
                    elif kind == "transfer_gap_change":
                        gap_comb = _transfer_gap_value(metric, arms[0]["points"][col][metric],
                                                       arms[1]["points"][col][metric])
                        gap_text = _transfer_gap_value(metric, arms[2]["points"][col][metric],
                                                       arms[3]["points"][col][metric])
                        val = None if (gap_comb is None or gap_text is None) else float(gap_comb - gap_text)
                        draw_comb = _transfer_gap_draws(metric, arms[0]["draws"][col][metric],
                                                        arms[1]["draws"][col][metric])
                        draw_text = _transfer_gap_draws(metric, arms[2]["draws"][col][metric],
                                                        arms[3]["draws"][col][metric])
                        draw_vec = draw_comb - draw_text
                    else:
                        a_val = arms[0]["points"][col][metric]
                        b_val = arms[1]["points"][col][metric]
                        val = None if (a_val is None or b_val is None) else float(a_val - b_val)
                        draw_vec = arms[0]["draws"][col][metric] - arms[1]["draws"][col][metric]
                    cat_vals.append(val)
                    cat_draws.append(draw_vec)
                    status, reason = _category_status_and_reason(val, pos_n, neg_n, minimum, unest)
                    rows.append({
                        "comparison_id": cid,
                        "category": category,
                        "statistic": metric,
                        "value": val,
                        "status": status,
                        "reason": reason,
                        "support": {
                            "known": known_n,
                            "positive": pos_n,
                            "negative": neg_n,
                        },
                        "interval": bootstrap_interval(draw_vec),
                    })
                rows.append(_macro_row(cid, metric, cat_vals, arms[0]["supports"], cat_draws,
                                       primary_indices, minimum, True))
            continue

        raise ContractError(f"Unsupported E01-E04 comparison kind: {kind}")

    check(required_rows(plan, unit) <= {(r["comparison_id"], r["category"], r["statistic"]) for r in rows},
          f"Incomplete rows produced for {unit_id}")
    return rows


def _filter_prediction_refs(plan, references, required_blocks):
    """Select only the required prediction block refs while checking for duplicate block IDs."""
    required = set(required_blocks)
    by_block = {}
    selected = []
    for reference in references:
        ref = plan.store.ref(reference)
        block_id = plan.store.json(ref, "body.json", PREDICTION_KIND)["block_id"]
        check(block_id not in by_block, "Duplicate prediction block")
        by_block[block_id] = ref
        if block_id in required:
            selected.append(ref)
    check(required <= set(by_block), "Missing required prediction blocks")
    return selected, {bid: by_block[bid] for bid in sorted(required)}


def _index_resume_analyses(plan, units, resume_refs):
    """Verify prior analysis references and map unit_id -> (ref, body)."""
    indexed = {}
    for reference in resume_refs:
        ref = plan.store.ref(reference)
        body = load_analysis_block(plan, ref, units=units)
        uid = body["unit_id"]
        check(uid not in indexed, f"Duplicate resume analysis reference for {uid}")
        indexed[uid] = (ref, body)
    return indexed


@contract
def run_e01_e04(plan, prediction_refs, producer, *, owner=None, experiments=E01_E04_EXPERIMENTS,
                unit_ids=None, resume_refs=(), progress=None):
    """Run or resume E01-E04 analysis units, checkpointing each completed unit."""
    progress = progress or (lambda event: None)
    run_wall_start = time.perf_counter()
    run_cpu_start = time.process_time()

    units = analysis_units(plan)
    selected_units = select_e01_e04_units(plan, owner=owner, experiments=experiments, unit_ids=unit_ids)
    needed_blocks = sorted({bid for uid in selected_units for bid in units[uid]["required_blocks"]})

    progress({
        "event": "e01_e04_start",
        "owner": owner or "all",
        "experiments": list(experiments),
        "units": list(selected_units),
        "required_prediction_blocks": len(needed_blocks),
    })

    load_start = time.perf_counter()
    filtered_refs, block_refs = _filter_prediction_refs(plan, prediction_refs, needed_blocks)
    blocks = prediction_index(plan, filtered_refs, required_blocks=needed_blocks)
    resume_map = _index_resume_analyses(plan, units, resume_refs)
    load_seconds = time.perf_counter() - load_start

    counts = None
    completed = {}
    reused_units = []
    computed_units = []
    total_statuses = Counter()

    for idx, uid in enumerate(selected_units, start=1):
        unit = units[uid]
        unit_preds = {bid: block_refs[bid] for bid in unit["required_blocks"]}
        if uid in resume_map:
            prev_ref, prev_body = resume_map[uid]
            if plain(prev_body["predictions"]) == plain(unit_preds):
                unit_statuses = Counter(row["status"] for row in prev_body["rows"])
                total_statuses.update(unit_statuses)
                completed[uid] = prev_ref
                reused_units.append(uid)
                progress({
                    "event": "analysis_unit_complete",
                    "index": idx,
                    "total": len(selected_units),
                    "unit_id": uid,
                    "reused": True,
                    "reference": prev_ref,
                    "rows": len(prev_body["rows"]),
                    "status_counts": dict(unit_statuses),
                })
                continue
            progress({
                "event": "analysis_unit_resume_skipped",
                "unit_id": uid,
                "reason": "prediction_inputs_mismatch",
                "skipped_reference": prev_ref,
            })

        if counts is None:
            draws = load_bootstrap(plan)
            counts = bootstrap_counts(draws, len(plan["screen_ids"]))

        unit_wall_start = time.perf_counter()
        unit_cpu_start = time.process_time()
        rows = compute_e01_e04_unit(plan, uid, blocks, counts)
        ref = publish_analysis_block(plan, uid, unit_preds, rows, producer)
        unit_wall = time.perf_counter() - unit_wall_start
        unit_cpu = time.process_time() - unit_cpu_start
        unit_statuses = Counter(row["status"] for row in rows)
        total_statuses.update(unit_statuses)
        completed[uid] = ref
        computed_units.append(uid)
        progress({
            "event": "analysis_unit_complete",
            "index": idx,
            "total": len(selected_units),
            "unit_id": uid,
            "reused": False,
            "reference": ref,
            "rows": len(rows),
            "status_counts": dict(unit_statuses),
            "wall_seconds": unit_wall,
            "cpu_seconds": unit_cpu,
        })

    summary = {
        "event": "e01_e04_complete",
        "status": "passed",
        "owner": owner or "all",
        "experiments": list(experiments),
        "plan": plain(plan.reference),
        "plan_identity": plan["identity"],
        "units": completed,
        "analyses": [completed[uid] for uid in selected_units],
        "reused_units": reused_units,
        "computed_units": computed_units,
        "status_counts": dict(total_statuses),
        "optional_coverage": list(E04_UNAVAILABLE_ARMS),
        "timing": {
            "load_seconds": load_seconds,
            "wall_seconds": time.perf_counter() - run_wall_start,
            "cpu_seconds": time.process_time() - run_cpu_start,
        },
    }
    progress(summary)
    return summary


def _parse_ref_arg(raw):
    text = raw.strip()
    if text.startswith("{"):
        return json.loads(text)
    return text


def _load_refs_file(path, preferred_key):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if preferred_key in payload and isinstance(payload[preferred_key], list):
            return payload[preferred_key]
        if "units" in payload and isinstance(payload["units"], dict):
            return list(payload["units"].values())
        return list(payload.values())
    raise ContractError(f"Unsupported reference collection format in {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--owner", required=True, choices=(*E01_E04_OWNERS, "all"))
    parser.add_argument("--predictions", help="JSON file of canonical prediction block references.")
    parser.add_argument("--prediction-ref", action="append", default=[],
                        help="Explicit prediction block completion path or JSON reference.")
    parser.add_argument("--experiments", nargs="+", choices=E01_E04_EXPERIMENTS,
                        default=list(E01_E04_EXPERIMENTS))
    parser.add_argument("--unit", action="append", dest="units",
                        help="Optional specific E01-E04 unit ID(s) to execute.")
    parser.add_argument("--resume-analyses", help="Optional JSON file of completed analysis references.")
    parser.add_argument("--resume-ref", action="append", default=[],
                        help="Optional explicit completed analysis reference to verify and reuse.")
    parser.add_argument("--output-index", help="Optional JSON path to write the completed analysis index.")
    parser.add_argument("--storage-scope", default="persistent-drive",
                        choices=("persistent-drive", "local-development"))
    args = parser.parse_args(argv)

    root = phase1_root(args.phase1_root)
    protocol = json.loads((REPO / "configs/protocol.json").read_text(encoding="utf-8"))
    producer = current_producer(args.storage_scope)
    plan = load_stage4_plan(root, protocol, _parse_ref_arg(args.manifest))

    pred_refs = []
    if args.predictions:
        pred_refs.extend(_load_refs_file(args.predictions, "predictions"))
    pred_refs.extend(_parse_ref_arg(raw) for raw in args.prediction_ref)
    check(bool(pred_refs), "Explicit canonical prediction references are required")

    resume_refs = []
    if args.resume_analyses:
        resume_refs.extend(_load_refs_file(args.resume_analyses, "analyses"))
    resume_refs.extend(_parse_ref_arg(raw) for raw in args.resume_ref)

    summary = run_e01_e04(
        plan,
        pred_refs,
        producer,
        owner=args.owner,
        experiments=tuple(args.experiments),
        unit_ids=tuple(args.units) if args.units else None,
        resume_refs=tuple(resume_refs),
        progress=emit,
    )
    if args.output_index:
        out_path = Path(args.output_index)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    try:
        main()
    except (ContractError, OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        emit({"status": "blocked", "blockers": [str(exc)]})
        raise SystemExit(2)
