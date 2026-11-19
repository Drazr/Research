"""CPU worker for Stage 4 E25 paired-error and label-stratified association analysis."""

import argparse
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np

from src.common.contracts import ContractError, LANGUAGES, fingerprint
from src.common.storage import phase1_root
from src.stage2.schema import REPO, check, contract, plain
from .interfaces import (
    DEPTHS, MACRO, PERMUTATION_DRAWS, analysis_units, association_orders,
    bootstrap_counts, bootstrap_interval, current_producer, e25_memberships,
    fixed_macro, load_analysis_block, load_bootstrap, load_stage4_plan,
    paired_image_alignment, prediction_index, probabilities, publish_analysis_block,
    support_status, targets_for_prediction,
)

E25_PARTS = (*LANGUAGES, "diagonal")


def _peak_memory_bytes():
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(usage * 1024)
    except ImportError:
        _, peak = tracemalloc.get_traced_memory()
        return int(peak)


def emit(event):
    print(json.dumps(event), flush=True)


def e25_units(plan, model, *, parts=None):
    """Return ordered E25 analysis units owned by one model."""
    check(model in DEPTHS, f"Unknown Stage 4 model: {model}")
    units = analysis_units(plan)
    selected_parts = E25_PARTS if not parts else tuple(parts)
    check(all(part in E25_PARTS for part in selected_parts), "Unknown E25 unit part")
    check(len(set(selected_parts)) == len(selected_parts), "Duplicate E25 unit part")
    ordered = []
    for part in selected_parts:
        uid = f"E25/{model}/{part}"
        check(uid in units, f"Missing frozen E25 unit: {uid}")
        ordered.append((uid, units[uid]))
    return ordered


def _decisions_and_estimable(pred_a, pred_b, idx_a, idx_b, baseline):
    """Return boolean 0.5 decisions (N x C) and per-category estimability (C,)."""
    n_rows = len(idx_a)
    n_cats = pred_a.logits.shape[1]
    if baseline == "probe":
        estimable = np.asarray(pred_a.estimable, dtype=bool) & np.asarray(pred_b.estimable, dtype=bool)
        d_a = np.zeros((n_rows, n_cats), dtype=bool)
        d_b = np.zeros((n_rows, n_cats), dtype=bool)
        if estimable.any():
            d_a[:, estimable] = probabilities(pred_a.logits[idx_a][:, estimable]) >= 0.5
            d_b[:, estimable] = probabilities(pred_b.logits[idx_b][:, estimable]) >= 0.5
        return d_a, d_b, estimable
    if baseline == "fit_prior":
        d_a = np.broadcast_to(np.asarray(pred_a.prevalence, dtype="f8") >= 0.5, (n_rows, n_cats)).copy()
        d_b = np.broadcast_to(np.asarray(pred_b.prevalence, dtype="f8") >= 0.5, (n_rows, n_cats)).copy()
        return d_a, d_b, np.ones(n_cats, dtype=bool)
    if baseline == "always_absent":
        return (np.zeros((n_rows, n_cats), dtype=bool),
                np.zeros((n_rows, n_cats), dtype=bool),
                np.ones(n_cats, dtype=bool))
    check(False, f"Unknown E25 baseline: {baseline}")


def _safe_ratio(numer, denom):
    numer_arr, denom_arr = np.broadcast_arrays(np.asarray(numer, dtype="f8"), np.asarray(denom, dtype="f8"))
    return np.divide(numer_arr, denom_arr, out=np.full(numer_arr.shape, np.nan, dtype="f8"), where=denom_arr > 0)


def _stratum_status(minimum, screen_pos, screen_neg, *events):
    if min(screen_pos, screen_neg) < minimum:
        return "descriptive_low_support", f"screen_class_support_below_{minimum}"
    if any(min(hits, total - hits) < minimum for hits, total in events):
        return "descriptive_low_support", f"event_outcome_support_below_{minimum}"
    return "ok", None


def _stratum_orders(category, label, ids, mask, memberships, orders_cache):
    key = (category, int(label))
    if key not in orders_cache:
        stratum_ids = [image_id for image_id, keep in zip(ids, mask) if keep]
        check(fingerprint(stratum_ids) == memberships[category][str(int(label))],
              f"E25 stratum membership mismatch for {category} label={label}")
        orders_cache[key] = association_orders(category, int(label), stratum_ids)
    return orders_cache[key]


def _make_row(cid, category, metric, value, status, reason, support, interval=None):
    row = {
        "comparison_id": cid,
        "category": category,
        "statistic": metric,
        "value": None if value is None else float(value),
        "status": status,
        "reason": reason,
        "support": support,
    }
    if interval is not None:
        row["interval"] = interval
    return row


def _eval_unstratified(cid, categories, primary_indices, y, known, d_a, d_b, estimable, counts, min_support):
    """Evaluate label-free E25 metrics for one comparison across all categories and primary_macro."""
    metrics = ("disagreement", "common_wrong", "error_a", "error_b")
    n_cats = len(categories)
    n_draws = counts.shape[0]
    point_vals = {m: np.full(n_cats, np.nan, dtype="f8") for m in metrics}
    boot_vals = {m: np.full((n_cats, n_draws), np.nan, dtype="f8") for m in metrics}
    cat_statuses = {m: [] for m in metrics}
    rows = []

    for col, category in enumerate(categories):
        mask = known[:, col]
        y_col = y[:, col]
        pos = int(y_col[mask].sum())
        n = int(mask.sum())
        neg = n - pos
        base_support = {"known": n, "positive": pos, "negative": neg}

        if not estimable[col]:
            for metric in metrics:
                cat_statuses[metric].append("undefined")
                rows.append(_make_row(cid, category, metric, None, "undefined",
                                      "unestimable_head", dict(base_support, denominator=0)))
            continue

        if n == 0:
            for metric in metrics:
                cat_statuses[metric].append("undefined")
                rows.append(_make_row(cid, category, metric, None, "undefined",
                                      "zero_denominator", dict(base_support, denominator=0)))
            continue

        da_m = d_a[mask, col]
        db_m = d_b[mask, col]
        y_m = y_col[mask]
        ea_m = da_m != y_m
        eb_m = db_m != y_m
        dis_m = da_m != db_m
        cw_m = ea_m & eb_m

        k_dis = int(dis_m.sum())
        k_cw = int(cw_m.sum())
        k_a = int(ea_m.sum())
        k_b = int(eb_m.sum())

        w = counts[:, mask]
        n_b = w.sum(axis=1)
        indicators = np.column_stack((dis_m, cw_m, ea_m, eb_m)).astype(np.int64)
        totals_b = w @ indicators

        counts_by_metric = {
            "disagreement": (k_dis, totals_b[:, 0]),
            "common_wrong": (k_cw, totals_b[:, 1]),
            "error_a": (k_a, totals_b[:, 2]),
            "error_b": (k_b, totals_b[:, 3]),
        }

        for metric in metrics:
            num, num_b = counts_by_metric[metric]
            val = float(num / n)
            status = support_status(val, pos, neg, min_support)
            reason = None if status == "ok" else f"screen_class_support_below_{min_support}"
            draws = _safe_ratio(num_b, n_b)
            point_vals[metric][col] = val
            boot_vals[metric][col] = draws
            cat_statuses[metric].append(status)
            support = dict(base_support, denominator=n, numerator=num,
                           disagreements=k_dis, common_wrong=k_cw, errors_a=k_a, errors_b=k_b)
            rows.append(_make_row(cid, category, metric, val, status, reason, support,
                                  interval=bootstrap_interval(draws)))

    for metric in metrics:
        p_statuses = [cat_statuses[metric][i] for i in primary_indices]
        ok_count = p_statuses.count("ok")
        low_count = p_statuses.count("descriptive_low_support")
        undef_count = p_statuses.count("undefined")
        macro_support = {
            "categories": len(primary_indices),
            "defined_categories": ok_count + low_count,
            "ok_categories": ok_count,
            "low_support_categories": low_count,
            "undefined_categories": undef_count,
        }
        if undef_count > 0:
            rows.append(_make_row(cid, MACRO, metric, None, "undefined",
                                  "unresolved_primary_category", macro_support))
        else:
            macro_val = float(fixed_macro(point_vals[metric], primary_indices))
            macro_draws = fixed_macro(boot_vals[metric], primary_indices, axis=0)
            status = "ok" if low_count == 0 else "descriptive_low_support"
            reason = None if status == "ok" else "low_support_primary_category"
            rows.append(_make_row(cid, MACRO, metric, macro_val, status, reason, macro_support,
                                  interval=bootstrap_interval(macro_draws)))
    return rows


def _eval_stratified(cid, label, ids, categories, primary_indices, y, known, d_a, d_b,
                     estimable, counts, min_support, memberships, orders_cache):
    """Evaluate label-stratified E25 metrics, conditionals, and permutation null."""
    boot_metrics = ("error_a", "error_b", "common_wrong", "independence", "excess",
                    "error_b_given_a", "error_a_given_b")
    all_metrics = (*boot_metrics, "null_lower", "null_upper")
    n_cats = len(categories)
    n_draws = counts.shape[0]
    point_vals = {m: np.full(n_cats, np.nan, dtype="f8") for m in boot_metrics}
    boot_vals = {m: np.full((n_cats, n_draws), np.nan, dtype="f8") for m in boot_metrics}
    null_overlaps = np.full((n_cats, PERMUTATION_DRAWS), np.nan, dtype="f8")
    cat_statuses = {m: [] for m in all_metrics}
    rows = []

    target_bool = bool(label)
    for col, category in enumerate(categories):
        mask = known[:, col] & (y[:, col] == target_bool)
        n = int(mask.sum())
        pos = n if label == 1 else 0
        neg = n if label == 0 else 0
        base_support = {"known": n, "positive": pos, "negative": neg, "stratum": n}

        if not estimable[col]:
            for metric in all_metrics:
                cat_statuses[metric].append("undefined")
                rows.append(_make_row(cid, category, metric, None, "undefined",
                                      "unestimable_head", dict(base_support, denominator=0)))
            continue

        if n == 0:
            for metric in all_metrics:
                cat_statuses[metric].append("undefined")
                rows.append(_make_row(cid, category, metric, None, "undefined",
                                      "empty_stratum", dict(base_support, denominator=0)))
            continue

        ea_m = d_a[mask, col] != target_bool
        eb_m = d_b[mask, col] != target_bool
        cw_m = ea_m & eb_m

        k_a = int(ea_m.sum())
        k_b = int(eb_m.sum())
        k_ab = int(cw_m.sum())

        p_a = float(k_a / n)
        p_b = float(k_b / n)
        p_ab = float(k_ab / n)
        indep = float(p_a * p_b)
        excess = float(p_ab - indep)

        w = counts[:, mask]
        n_b = w.sum(axis=1)
        indicators = np.column_stack((ea_m, eb_m, cw_m)).astype(np.int64)
        totals_b = w @ indicators
        ka_b = totals_b[:, 0]
        kb_b = totals_b[:, 1]
        kab_b = totals_b[:, 2]

        ea_draws = _safe_ratio(ka_b, n_b)
        eb_draws = _safe_ratio(kb_b, n_b)
        cw_draws = _safe_ratio(kab_b, n_b)
        indep_draws = ea_draws * eb_draws
        excess_draws = cw_draws - indep_draws
        b_given_a_draws = _safe_ratio(kab_b, ka_b)
        a_given_b_draws = _safe_ratio(kab_b, kb_b)

        screen_pos = int((known[:, col] & y[:, col]).sum())
        screen_neg = int(known[:, col].sum()) - screen_pos
        marginal_events = ((k_a, n), (k_b, n))
        joint_event = ((k_ab, n),)
        common_counts = {"errors_a": k_a, "errors_b": k_b, "common_wrong": k_ab}

        stratum_specs = (
            ("error_a", p_a, ea_draws, k_a, marginal_events[:1]),
            ("error_b", p_b, eb_draws, k_b, marginal_events[1:]),
            ("common_wrong", p_ab, cw_draws, k_ab, joint_event),
            ("independence", indep, indep_draws, None, marginal_events),
            ("excess", excess, excess_draws, None, marginal_events + joint_event),
        )
        for metric, val, draws, numerator, events in stratum_specs:
            support = dict(base_support, denominator=n, **common_counts)
            if numerator is not None:
                support["numerator"] = numerator
            status, reason = _stratum_status(min_support, screen_pos, screen_neg, *events)
            point_vals[metric][col] = val
            boot_vals[metric][col] = draws
            cat_statuses[metric].append(status)
            rows.append(_make_row(cid, category, metric, val, status, reason, support,
                                  interval=bootstrap_interval(draws)))

        cond_specs = (
            ("error_b_given_a", k_a, b_given_a_draws),
            ("error_a_given_b", k_b, a_given_b_draws),
        )
        for metric, denom, draws in cond_specs:
            support = dict(base_support, denominator=denom, numerator=k_ab, **common_counts)
            if denom == 0:
                cat_statuses[metric].append("undefined")
                rows.append(_make_row(cid, category, metric, None, "undefined",
                                      "zero_conditional_denominator", support))
            else:
                val = float(k_ab / denom)
                status, reason = _stratum_status(min_support, screen_pos, screen_neg, (k_ab, denom))
                point_vals[metric][col] = val
                boot_vals[metric][col] = draws
                cat_statuses[metric].append(status)
                rows.append(_make_row(cid, category, metric, val, status, reason, support,
                                      interval=bootstrap_interval(draws)))

        orders = _stratum_orders(category, label, ids, mask, memberships, orders_cache)
        if k_a == 0 or k_b == 0:
            null_vec = np.zeros(PERMUTATION_DRAWS, dtype="f8")
        else:
            ea_idx = np.flatnonzero(ea_m)
            null_vec = eb_m[orders[:, ea_idx]].sum(axis=1, dtype=np.int64) / float(n)
        null_overlaps[col] = null_vec
        q_low, q_high = np.quantile(null_vec, [0.025, 0.975], method="linear")
        null_support = dict(base_support, denominator=n, permutations=PERMUTATION_DRAWS, **common_counts)
        null_status, null_reason = _stratum_status(min_support, screen_pos, screen_neg, *marginal_events)
        for metric, val in (("null_lower", float(q_low)), ("null_upper", float(q_high))):
            cat_statuses[metric].append(null_status)
            rows.append(_make_row(cid, category, metric, val, null_status, null_reason, null_support))

    for metric in boot_metrics:
        p_statuses = [cat_statuses[metric][i] for i in primary_indices]
        ok_count = p_statuses.count("ok")
        low_count = p_statuses.count("descriptive_low_support")
        undef_count = p_statuses.count("undefined")
        macro_support = {
            "categories": len(primary_indices),
            "defined_categories": ok_count + low_count,
            "ok_categories": ok_count,
            "low_support_categories": low_count,
            "undefined_categories": undef_count,
        }
        if undef_count > 0:
            rows.append(_make_row(cid, MACRO, metric, None, "undefined",
                                  "unresolved_primary_category", macro_support))
        else:
            macro_val = float(fixed_macro(point_vals[metric], primary_indices))
            macro_draws = fixed_macro(boot_vals[metric], primary_indices, axis=0)
            status = "ok" if low_count == 0 else "descriptive_low_support"
            reason = None if status == "ok" else "low_support_primary_category"
            rows.append(_make_row(cid, MACRO, metric, macro_val, status, reason, macro_support,
                                  interval=bootstrap_interval(macro_draws)))

    p_null_statuses = [cat_statuses["null_lower"][i] for i in primary_indices]
    ok_count = p_null_statuses.count("ok")
    low_count = p_null_statuses.count("descriptive_low_support")
    undef_count = p_null_statuses.count("undefined")
    macro_null_support = {
        "categories": len(primary_indices),
        "defined_categories": ok_count + low_count,
        "ok_categories": ok_count,
        "low_support_categories": low_count,
        "undefined_categories": undef_count,
        "permutations": PERMUTATION_DRAWS,
    }
    if undef_count > 0:
        for metric in ("null_lower", "null_upper"):
            rows.append(_make_row(cid, MACRO, metric, None, "undefined",
                                  "unresolved_primary_category", macro_null_support))
    else:
        macro_null_vec = fixed_macro(null_overlaps, primary_indices, axis=0)
        q_low, q_high = np.quantile(macro_null_vec, [0.025, 0.975], method="linear")
        status = "ok" if low_count == 0 else "descriptive_low_support"
        reason = None if status == "ok" else "low_support_primary_category"
        rows.append(_make_row(cid, MACRO, "null_lower", float(q_low), status, reason, macro_null_support))
        rows.append(_make_row(cid, MACRO, "null_upper", float(q_high), status, reason, macro_null_support))

    return rows


@contract
def compute_e25_unit(plan, unit_id, unit, prediction_blocks, *,
                     counts=None, memberships=None, orders_cache=None):
    """Compute all tidy E25 rows for one frozen analysis unit."""
    check(unit["experiment"] == "E25", f"Not an E25 analysis unit: {unit_id}")
    categories = tuple(plan["categories"])
    primary_indices = tuple(categories.index(cat) for cat in plan["primary"])
    min_support = int(plan.protocol["minimum_positive_and_negative"]["SCREEN"])
    if counts is None:
        counts = bootstrap_counts(load_bootstrap(plan), len(plan["screen_ids"]))
    if memberships is None:
        memberships = e25_memberships(plan)
    if orders_cache is None:
        orders_cache = {}

    alignment_cache = {}
    targets_cache = {}
    rows = []

    for cid, comp in unit["comparisons"].items():
        check(comp["kind"] == "error_pair" and len(comp["inputs"]) == 2,
              f"Unexpected E25 comparison descriptor: {cid}")
        in_a, in_b = comp["inputs"]
        pair_key = (in_a["block_id"], in_a["route_id"], in_b["block_id"], in_b["route_id"])
        pred_a = prediction_blocks[in_a["block_id"]][in_a["route_id"]]
        pred_b = prediction_blocks[in_b["block_id"]][in_b["route_id"]]

        if pair_key not in alignment_cache:
            ids, (idx_a, idx_b) = paired_image_alignment(plan, pred_a, pred_b)
            route_a_key = (in_a["block_id"], in_a["route_id"])
            route_b_key = (in_b["block_id"], in_b["route_id"])
            if route_a_key not in targets_cache:
                targets_cache[route_a_key] = targets_for_prediction(plan, pred_a)
            if route_b_key not in targets_cache:
                targets_cache[route_b_key] = targets_for_prediction(plan, pred_b)
            y_a, known_a = targets_cache[route_a_key]
            y_b, known_b = targets_cache[route_b_key]
            y_aligned = y_a[idx_a]
            known_aligned = known_a[idx_a]
            check(np.array_equal(y_aligned, y_b[idx_b])
                  and np.array_equal(known_aligned, known_b[idx_b]),
                  "Paired E25 arms disagree on image targets or known-cell masks")
            alignment_cache[pair_key] = (ids, idx_a, idx_b, y_aligned, known_aligned)

        ids, idx_a, idx_b, y_aligned, known_aligned = alignment_cache[pair_key]
        d_a, d_b, estimable = _decisions_and_estimable(pred_a, pred_b, idx_a, idx_b, comp["baseline"])
        label = comp["label"]
        if label is None:
            rows.extend(_eval_unstratified(
                cid, categories, primary_indices, y_aligned, known_aligned,
                d_a, d_b, estimable, counts, min_support,
            ))
        else:
            check(label in (0, 1), f"Invalid E25 label stratum: {label}")
            rows.extend(_eval_stratified(
                cid, label, ids, categories, primary_indices, y_aligned, known_aligned,
                d_a, d_b, estimable, counts, min_support, memberships, orders_cache,
            ))

    return rows


@contract
def run_e25(plan, model, prediction_refs, producer, *,
            resume_refs=(), parts=None, progress=None):
    """Execute and checkpoint E25 units for one model with explicit resume verification."""
    progress = progress or (lambda event: None)
    all_units = analysis_units(plan)
    selected = e25_units(plan, model, parts=parts)
    required_blocks = sorted({bid for _, unit in selected for bid in unit["required_blocks"]})

    t_load = time.perf_counter()
    blocks = prediction_index(plan, prediction_refs, required_blocks=required_blocks)
    block_refs = {bid: plain(next(iter(routes.values())).reference) for bid, routes in blocks.items()}
    load_seconds = time.perf_counter() - t_load
    progress({
        "event": "e25_predictions_verified",
        "model": model,
        "blocks": len(required_blocks),
        "wall_seconds": load_seconds,
        "peak_memory_bytes": _peak_memory_bytes(),
    })

    resumed = {}
    for ref in resume_refs:
        canonical = plan.store.ref(ref)
        body = load_analysis_block(plan, canonical, units=all_units)
        uid = body["unit_id"]
        unit = all_units[uid]
        check(unit["experiment"] == "E25" and unit["owner"] == model,
              f"Foreign resumed analysis unit for E25/{model}: {uid}")
        check(uid not in resumed, f"Duplicate resumed E25 unit: {uid}")
        expected_inputs = {bid: block_refs[bid] for bid in unit["required_blocks"]}
        check(plain(body["predictions"]) == expected_inputs,
              f"Resumed E25 unit {uid} does not match current prediction references")
        resumed[uid] = canonical
        progress({"event": "e25_unit_reused", "model": model, "unit_id": uid, "reference": canonical})

    memberships = e25_memberships(plan)
    counts = bootstrap_counts(load_bootstrap(plan), len(plan["screen_ids"]))
    orders_cache = {}
    completed = {}
    reused_ids = []
    published_ids = []
    timings = {"load_predictions_seconds": load_seconds, "units": {}}

    for uid, unit in selected:
        if uid in resumed:
            completed[uid] = resumed[uid]
            reused_ids.append(uid)
            continue

        progress({
            "event": "e25_unit_start",
            "model": model,
            "unit_id": uid,
            "part": unit["part"],
            "comparisons": len(unit["comparisons"]),
        })
        t0 = time.perf_counter()
        cpu0 = time.process_time()
        rows = compute_e25_unit(plan, uid, unit, blocks,
                                counts=counts, memberships=memberships, orders_cache=orders_cache)
        t1 = time.perf_counter()
        unit_pred_refs = {bid: block_refs[bid] for bid in unit["required_blocks"]}
        ref = publish_analysis_block(plan, uid, unit_pred_refs, rows, producer, memberships=memberships)
        t2 = time.perf_counter()
        load_analysis_block(plan, ref, units=all_units)
        t3 = time.perf_counter()
        cpu1 = time.process_time()

        unit_timing = {
            "compute_seconds": t1 - t0,
            "publish_seconds": t2 - t1,
            "readback_seconds": t3 - t2,
            "wall_seconds": t3 - t0,
            "cpu_seconds": cpu1 - cpu0,
            "peak_memory_bytes": _peak_memory_bytes(),
        }
        timings["units"][uid] = unit_timing
        completed[uid] = ref
        published_ids.append(uid)
        progress({
            "event": "e25_unit_complete",
            "model": model,
            "unit_id": uid,
            "part": unit["part"],
            "rows": len(rows),
            "reference": ref,
            **unit_timing,
        })

    return {
        "status": "passed",
        "experiment": "E25",
        "model": model,
        "plan": plain(plan.reference),
        "units": completed,
        "analyses": [completed[uid] for uid, _ in selected],
        "reused": reused_ids,
        "published": published_ids,
        "timings": timings,
    }


def _parse_ref_arg(value):
    text = value.strip()
    if text.startswith("{"):
        return json.loads(text)
    return text


def _load_ref_list(path_str, key):
    data = json.loads(Path(path_str).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and key in data and isinstance(data[key], list):
        return data[key]
    check(False, f"Expected a JSON list or mapping containing '{key}' in {path_str}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-root", required=True)
    parser.add_argument("--manifest", required=True, help="Explicit Stage 4 plan _COMPLETE.json path or JSON ref.")
    parser.add_argument("--model", required=True, choices=tuple(DEPTHS))
    parser.add_argument("--predictions-file", help="JSON file with list of prediction refs or {'predictions': [...]}.")
    parser.add_argument("--prediction-ref", action="append", default=[],
                        help="Explicit prediction block _COMPLETE.json path or JSON ref (repeatable).")
    parser.add_argument("--resume-file", help="Optional JSON file with previously completed E25 analysis refs.")
    parser.add_argument("--resume-ref", action="append", default=[],
                        help="Optional previously completed E25 analysis ref (repeatable).")
    parser.add_argument("--part", action="append", choices=E25_PARTS, default=[],
                        help="Optional subset of E25 parts (default: all five parts for MODEL).")
    parser.add_argument("--output-manifest", help="Optional path to write the completed E25 summary JSON.")
    parser.add_argument("--storage-scope", default="persistent-drive",
                        choices=("persistent-drive", "local-development"))
    args = parser.parse_args()

    tracemalloc.start()
    root = phase1_root(args.phase1_root)
    protocol = json.loads((REPO / "configs/protocol.json").read_text(encoding="utf-8"))
    producer = current_producer(args.storage_scope)

    prediction_refs = []
    if args.predictions_file:
        prediction_refs.extend(_load_ref_list(args.predictions_file, "predictions"))
    prediction_refs.extend(_parse_ref_arg(item) for item in args.prediction_ref)
    check(bool(prediction_refs), "Explicit prediction references are required for E25")

    resume_refs = []
    if args.resume_file:
        resume_refs.extend(_load_ref_list(args.resume_file, "analyses"))
    resume_refs.extend(_parse_ref_arg(item) for item in args.resume_ref)

    emit({"event": "e25_worker_start", "model": args.model, "code_commit": producer["code_commit"]})
    plan = load_stage4_plan(root, protocol, _parse_ref_arg(args.manifest))
    summary = run_e25(
        plan,
        args.model,
        prediction_refs,
        producer,
        resume_refs=resume_refs,
        parts=args.part or None,
        progress=emit,
    )
    if args.output_manifest:
        out_path = Path(args.output_manifest)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    emit({"event": "e25_worker_complete", **summary})


if __name__ == "__main__":
    try:
        main()
    except (ContractError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        emit({"status": "blocked", "blockers": [str(exc)], "experiment": "E25"})
        raise SystemExit(2)
