"""E19 CPU analysis worker: source DEV-C confidence cutoffs and selective SCREEN evaluation."""

import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from src.common.contracts import ContractError, LANGUAGES
from src.common.storage import phase1_root
from src.stage2.schema import REPO, check, plain
from src.stage4.interfaces import (
    DEPTHS, MACRO, analysis_units, binary_metrics, bootstrap_binary_metrics, bootstrap_counts,
    bootstrap_interval, confidence_cutoff, current_producer, fixed_macro, load_analysis_block,
    load_bootstrap, load_stage4_plan, paired_image_alignment, prediction_index, probabilities,
    publish_analysis_block, support_status, targets_for_prediction,
)
from src.stage4.predictions import PREDICTION_KIND

E19_BASELINES = ("probe", "fit_prior", "always_absent")
E19_NOMINALS = (0.5, 0.75, 0.9)
E19_SELECTIVE_METRICS = (
    "target_coverage", "retained_risk", "base_risk", "score_blind_risk",
    "positive_retention", "negative_retention", "retained_sensitivity", "retained_specificity",
    "overall_retained_positive_correct", "overall_retained_negative_correct",
)


def emit(event, **fields):
    print(json.dumps({"event": event, **plain(fields)}, allow_nan=False), flush=True)


def peak_memory():
    """Return process high-water RSS in bytes on Linux/macOS, or None on Windows."""
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)
    except ImportError:
        return None


def e19_units(plan, model, sources=None):
    """Return ordered E19 unit IDs owned by the selected model."""
    check(model in DEPTHS, "MODEL must be 'qwen' or 'gemma'")
    selected = LANGUAGES if sources is None else tuple(sources)
    check(bool(selected) and all(lang in LANGUAGES for lang in selected)
          and len(set(selected)) == len(selected), "Invalid E19 source language selection")
    units = analysis_units(plan)
    ordered = [f"E19/{model}/{lang}" for lang in selected]
    for unit_id in ordered:
        check(unit_id in units, f"Missing E19 unit in Stage 4 plan: {unit_id}")
    return tuple(ordered)


def compute_e19_cutoffs(plan, dev_pred):
    """Freeze per-source/category DEV-C confidence cutoffs at nominal 50%, 75% and 90% coverage."""
    check(dev_pred.split == "DEV-C", "E19 cutoffs must be computed from DEV-C only")
    probs = probabilities(dev_pred.logits)
    records = []
    lookup = {}
    for baseline in E19_BASELINES:
        for idx, category in enumerate(plan["categories"]):
            if baseline == "probe":
                scores = probs[:, idx] if dev_pred.estimable[idx] else probs[:0, idx]
            elif baseline == "fit_prior":
                scores = np.full(len(dev_pred.rows), float(dev_pred.prevalence[idx]), dtype="f8")
            else:
                scores = np.full(len(dev_pred.rows), 0.0, dtype="f8")
            for nominal in E19_NOMINALS:
                result = confidence_cutoff(scores, nominal)
                record = {
                    "baseline": baseline, "category": category, "nominal": nominal,
                    "eligible": int(result["eligible"]), "retained": int(result["retained"]),
                    "cutoff": result["cutoff"],
                }
                records.append(record)
                lookup[(baseline, category, nominal)] = record
    return records, lookup


def _safe_ratio(numerator, denominator):
    numer, denom = np.broadcast_arrays(numerator, denominator)
    return np.divide(numer, denom, out=np.full(numer.shape, np.nan, dtype="f8"), where=denom > 0)


def _scalar_ratio(numerator, denominator):
    return None if denominator <= 0 else float(numerator / denominator)


def _selective_indicators(y, p, known, cutoff):
    """Build exact integer indicator columns for one category under a fixed DEV-C cutoff."""
    d = p >= 0.5
    conf = np.maximum(p, 1.0 - p)
    ret = conf >= cutoff
    err = d != y
    k = known
    return np.column_stack((
        k,                  # 0: N (known)
        k & y,              # 1: N_pos
        k & ~y,             # 2: N_neg
        k & ret,            # 3: R (retained)
        k & ret & y,        # 4: R_pos
        k & ret & ~y,       # 5: R_neg
        k & err,            # 6: E (base errors)
        k & ret & err,      # 7: E_ret (retained errors)
        k & ret & y & d,    # 8: TP_ret (retained true positives)
        k & ret & ~y & ~d,  # 9: TN_ret (retained true negatives)
    )).astype(np.int64, copy=False)


def _selective_ratios(sums):
    """Compute all 10 selective SCREEN metrics from indicator sums of shape (..., 10)."""
    n, n_pos, n_neg, r, r_pos, r_neg, e, e_ret, tp_ret, tn_ret = (sums[..., i] for i in range(10))
    base_risk = _safe_ratio(e, n)
    return {
        "target_coverage": _safe_ratio(r, n),
        "retained_risk": _safe_ratio(e_ret, r),
        "base_risk": base_risk,
        "score_blind_risk": np.where(r > 0, base_risk, np.nan),
        "positive_retention": _safe_ratio(r_pos, n_pos),
        "negative_retention": _safe_ratio(r_neg, n_neg),
        "retained_sensitivity": _safe_ratio(tp_ret, r_pos),
        "retained_specificity": _safe_ratio(tn_ret, r_neg),
        "overall_retained_positive_correct": _safe_ratio(tp_ret, n_pos),
        "overall_retained_negative_correct": _safe_ratio(tn_ret, n_neg),
    }


def _category_status(metric, value, n_pos, n_neg, point_counts, screen_min, subgroup_min):
    """Apply frozen split support and subgroup/correctness support rules."""
    if value is None:
        return "undefined", f"Zero denominator for {metric}."
    if min(n_pos, n_neg) < screen_min:
        return "descriptive_low_support", "Below SCREEN minimum positive/negative support."
    if metric == "retained_risk":
        r_err, r_cor = point_counts["retained_errors"], point_counts["retained_correct"]
        if min(r_err, r_cor) < subgroup_min:
            return "descriptive_low_support", "Below minimum retained correctness outcome support."
    elif metric in ("base_risk", "score_blind_risk"):
        err, cor = point_counts["errors"], point_counts["correct"]
        if min(err, cor) < subgroup_min:
            return "descriptive_low_support", "Below minimum correctness outcome support."
    elif metric in ("retained_sensitivity", "retained_specificity",
                    "overall_retained_positive_correct", "overall_retained_negative_correct"):
        outcome = "positive" if metric in ("retained_sensitivity", "overall_retained_positive_correct") else "negative"
        successes = point_counts[f"retained_true_{outcome}"]
        if min(successes, point_counts["denominator"] - successes) < subgroup_min:
            return "descriptive_low_support", f"Below minimum {outcome} subgroup outcome support."
    return "ok", None


def _metric_support(metric, counts_row):
    """Build machine-readable support dictionary with known/positive/negative and metric denominators."""
    n, n_pos, n_neg, r, r_pos, r_neg, e, e_ret, tp_ret, tn_ret = (int(v) for v in counts_row)
    base = {"known": n, "positive": n_pos, "negative": n_neg}
    if metric == "target_coverage":
        return {**base, "retained": r, "denominator": n}
    if metric == "retained_risk":
        return {**base, "retained": r, "retained_errors": e_ret, "retained_correct": r - e_ret, "denominator": r}
    if metric == "base_risk":
        return {**base, "errors": e, "correct": n - e, "denominator": n}
    if metric == "score_blind_risk":
        return {**base, "retained": r, "errors": e, "correct": n - e, "denominator": n if r > 0 else 0}
    if metric == "positive_retention":
        return {**base, "retained_positive": r_pos, "denominator": n_pos}
    if metric == "negative_retention":
        return {**base, "retained_negative": r_neg, "denominator": n_neg}
    if metric == "retained_sensitivity":
        return {**base, "retained_positive": r_pos, "retained_true_positive": tp_ret, "denominator": r_pos}
    if metric == "retained_specificity":
        return {**base, "retained_negative": r_neg, "retained_true_negative": tn_ret, "denominator": r_neg}
    if metric == "overall_retained_positive_correct":
        return {**base, "retained_positive": r_pos, "retained_true_positive": tp_ret, "denominator": n_pos}
    if metric == "overall_retained_negative_correct":
        return {**base, "retained_negative": r_neg, "retained_true_negative": tn_ret, "denominator": n_neg}
    raise ContractError(f"Unknown E19 selective metric: {metric}")


def evaluate_e19_unit(plan, unit_id, predictions, counts):
    """Evaluate one E19 source unit across all 4 SCREEN targets, 3 coverages and 3 baselines."""
    units = analysis_units(plan)
    unit = units[unit_id]
    check(unit["experiment"] == "E19", f"Expected an E19 analysis unit, got {unit_id}")
    categories = tuple(plan["categories"])
    primary_indices = [categories.index(cat) for cat in plan["primary"]]
    screen_min = int(plan.protocol["minimum_positive_and_negative"]["SCREEN"])
    devc_min = int(plan.protocol["minimum_positive_and_negative"]["DEV-C"])
    subgroup_min = screen_min
    total_draws = int(counts.shape[0])

    first_comp = next(iter(unit["comparisons"].values()))
    dev_input = first_comp["inputs"][0]
    dev_pred = predictions[dev_input["block_id"]][dev_input["route_id"]]
    check(dev_pred.split == "DEV-C", "E19 cutoff source prediction must be DEV-C")
    dev_y, dev_known = targets_for_prediction(plan, dev_pred)
    dev_support = [
        {
            "known": int(dev_known[:, idx].sum()),
            "positive": int((dev_known[:, idx] & dev_y[:, idx]).sum()),
            "negative": int((dev_known[:, idx] & ~dev_y[:, idx]).sum()),
        }
        for idx in range(len(categories))
    ]
    cutoffs, cutoff_lookup = compute_e19_cutoffs(plan, dev_pred)

    # Cache per-target aligned arrays and underlying probe AUROC/AP (independent of nominal coverage).
    screen_cache = {}
    rows = []

    for cid, comp in unit["comparisons"].items():
        check(comp["inputs"][0] == dev_input, "Inconsistent DEV-C input within E19 unit")
        screen_input = comp["inputs"][1]
        screen_key = (screen_input["block_id"], screen_input["route_id"])
        if screen_key not in screen_cache:
            pred = predictions[screen_input["block_id"]][screen_input["route_id"]]
            check(pred.split == "SCREEN", "E19 evaluation target must be SCREEN")
            _, (aligned,) = paired_image_alignment(plan, pred)
            y_all, known_all = targets_for_prediction(plan, pred)
            y_aligned = y_all[aligned]
            known_aligned = known_all[aligned]
            probs_aligned = probabilities(pred.logits)[aligned]
            probe_disc = {}
            for idx in range(len(categories)):
                if pred.estimable[idx]:
                    bm = binary_metrics(y_aligned[:, idx], probs_aligned[:, idx], known_aligned[:, idx])
                    boot_bm = bootstrap_binary_metrics(
                        y_aligned[:, idx], probs_aligned[:, idx], known_aligned[:, idx], counts
                    )
                    probe_disc[idx] = {
                        "auroc": (bm["metrics"]["auroc"], boot_bm["auroc"]),
                        "ap": (bm["metrics"]["ap"], boot_bm["ap"]),
                        "known": bm["known"], "positive": bm["positive"], "negative": bm["negative"],
                    }
                else:
                    kn = int(known_aligned[:, idx].sum())
                    pos = int((known_aligned[:, idx] & y_aligned[:, idx]).sum())
                    probe_disc[idx] = {
                        "auroc": (None, np.full(total_draws, np.nan)),
                        "ap": (None, np.full(total_draws, np.nan)),
                        "known": kn, "positive": pos, "negative": kn - pos,
                    }
            screen_cache[screen_key] = (pred, y_aligned, known_aligned, probs_aligned, probe_disc)

        pred, y_aligned, known_aligned, probs_aligned, probe_disc = screen_cache[screen_key]
        nominal = comp["nominal"]
        baseline = comp["baseline"]
        stats = tuple(comp["statistics"])

        cat_values = {stat: np.full(len(categories), np.nan, dtype="f8") for stat in stats}
        cat_statuses = {stat: [] for stat in stats}
        cat_draws = {stat: np.full((len(categories), total_draws), np.nan, dtype="f8")
                     for stat in stats if stat != "source_coverage"}

        for idx, category in enumerate(categories):
            c_rec = cutoff_lookup[(baseline, category, nominal)]
            cutoff = c_rec["cutoff"]
            d_sup = dev_support[idx]

            # 1. source_coverage (DEV-C, no SCREEN bootstrap)
            src_val = _scalar_ratio(c_rec["retained"], c_rec["eligible"])
            src_status = support_status(src_val, d_sup["positive"], d_sup["negative"], devc_min)
            src_reason = (None if src_status == "ok" else
                          "Unestimable source head." if src_val is None else
                          "Below DEV-C minimum positive/negative support.")
            rows.append({
                "comparison_id": cid, "category": category, "statistic": "source_coverage",
                "value": src_val, "status": src_status, "reason": src_reason,
                "support": {**d_sup, "eligible": c_rec["eligible"], "retained": c_rec["retained"]},
            })
            if src_val is not None:
                cat_values["source_coverage"][idx] = src_val
            cat_statuses["source_coverage"].append(src_status)

            # 2. Selective SCREEN metrics
            y_col = y_aligned[:, idx]
            k_col = known_aligned[:, idx]
            kn = int(k_col.sum())
            pos = int((k_col & y_col).sum())
            neg = kn - pos
            unestimable = (baseline == "probe" and (not pred.estimable[idx] or cutoff is None))

            if unestimable:
                empty_draws = np.full(total_draws, np.nan, dtype="f8")
                empty_interval = bootstrap_interval(empty_draws)
                base_sup = {"known": kn, "positive": pos, "negative": neg, "denominator": 0}
                for metric in E19_SELECTIVE_METRICS:
                    rows.append({
                        "comparison_id": cid, "category": category, "statistic": metric,
                        "value": None, "status": "undefined", "reason": "Unestimable source head.",
                        "support": dict(base_sup), "interval": dict(empty_interval),
                    })
                    cat_statuses[metric].append("undefined")
            else:
                if baseline == "probe":
                    p_col = probs_aligned[:, idx]
                elif baseline == "fit_prior":
                    p_col = np.full(len(y_col), float(pred.prevalence[idx]), dtype="f8")
                else:
                    p_col = np.zeros(len(y_col), dtype="f8")

                indicators = _selective_indicators(y_col, p_col, k_col, cutoff)
                point_sums = indicators.sum(axis=0)
                boot_sums = counts @ indicators
                point_ratios = _selective_ratios(point_sums)
                boot_ratios = _selective_ratios(boot_sums)

                for metric in E19_SELECTIVE_METRICS:
                    raw_val = point_ratios[metric]
                    val = float(raw_val) if np.isfinite(raw_val) else None
                    sup = _metric_support(metric, point_sums)
                    status, reason = _category_status(metric, val, pos, neg, sup, screen_min, subgroup_min)
                    draws_arr = boot_ratios[metric]
                    interval = bootstrap_interval(draws_arr)
                    rows.append({
                        "comparison_id": cid, "category": category, "statistic": metric,
                        "value": val, "status": status, "reason": reason,
                        "support": sup, "interval": interval,
                    })
                    if val is not None:
                        cat_values[metric][idx] = val
                    cat_draws[metric][idx] = draws_arr
                    cat_statuses[metric].append(status)

            # 3. Underlying SCREEN AUROC and AP (probe only)
            if baseline == "probe":
                disc = probe_disc[idx]
                for metric in ("auroc", "ap"):
                    val, draws_arr = disc[metric]
                    if unestimable:
                        status, reason = "undefined", "Unestimable source head."
                    else:
                        status = support_status(val, pos, neg, screen_min)
                        reason = (None if status == "ok" else
                                  f"Undefined {metric} due to single-class SCREEN support." if val is None else
                                  "Below SCREEN minimum positive/negative support.")
                    rows.append({
                        "comparison_id": cid, "category": category, "statistic": metric,
                        "value": val, "status": status, "reason": reason,
                        "support": {"known": kn, "positive": pos, "negative": neg, "denominator": kn},
                        "interval": bootstrap_interval(draws_arr),
                    })
                    if val is not None:
                        cat_values[metric][idx] = val
                    cat_draws[metric][idx] = draws_arr
                    cat_statuses[metric].append(status)

        # 4. Fixed primary_macro summary across the 14 frozen primary categories
        for stat in stats:
            prim_vals = cat_values[stat][primary_indices]
            prim_statuses = [cat_statuses[stat][i] for i in primary_indices]
            defined_count = int(np.isfinite(prim_vals).sum())
            ok_count = sum(1 for s in prim_statuses if s == "ok")
            macro_raw = float(fixed_macro(cat_values[stat], primary_indices))
            if np.isfinite(macro_raw):
                macro_val = macro_raw
                if ok_count == len(primary_indices):
                    macro_status, macro_reason = "ok", None
                else:
                    macro_status = "descriptive_low_support"
                    macro_reason = "One or more primary categories have low subgroup or correctness support."
            else:
                macro_val = None
                macro_status = "undefined"
                macro_reason = "Unresolved primary macro: one or more primary categories are undefined."
            row = {
                "comparison_id": cid, "category": MACRO, "statistic": stat,
                "value": macro_val, "status": macro_status, "reason": macro_reason,
                "support": {
                    "categories": len(primary_indices),
                    "defined_categories": defined_count,
                    "ok_categories": ok_count,
                },
            }
            if stat != "source_coverage":
                macro_draws = fixed_macro(cat_draws[stat], primary_indices, axis=0)
                row["interval"] = bootstrap_interval(macro_draws)
            rows.append(row)

    return rows, cutoffs


def _index_prediction_refs(plan, prediction_refs, needed_blocks):
    """Index explicit prediction references by block ID and verify only needed E19 blocks."""
    ref_by_block = {}
    for reference in prediction_refs:
        ref = plan.store.ref(reference)
        block_id = plan.store.json(ref, "body.json", PREDICTION_KIND)["block_id"]
        check(block_id not in ref_by_block, "Duplicate prediction block reference")
        ref_by_block[block_id] = ref
    missing = sorted(set(needed_blocks) - set(ref_by_block))
    check(not missing, f"Missing required E19 prediction blocks: {missing}")
    selected_refs = [ref_by_block[bid] for bid in sorted(needed_blocks)]
    predictions = prediction_index(plan, selected_refs, required_blocks=needed_blocks)
    return ref_by_block, predictions


def run_e19(plan, model, prediction_refs, producer, *, sources=None, resume_refs=(), progress=emit):
    """Run E19 source-confidence analysis for one model, reusing verified completed units."""
    t_start = time.monotonic()
    unit_ids = e19_units(plan, model, sources=sources)
    units = analysis_units(plan)
    needed_blocks = sorted({bid for uid in unit_ids for bid in units[uid]["required_blocks"]})
    ref_by_block, predictions = _index_prediction_refs(plan, prediction_refs, needed_blocks)

    resumed = {}
    for ref in resume_refs:
        loaded = load_analysis_block(plan, ref, units=units)
        uid = loaded["unit_id"]
        check(uid in unit_ids, f"Resume reference is not an owned E19 unit for {model}: {uid}")
        check(uid not in resumed, f"Duplicate resume reference for {uid}")
        expected_inputs = {bid: ref_by_block[bid] for bid in units[uid]["required_blocks"]}
        check(plain(loaded["predictions"]) == expected_inputs,
              f"Resume reference for {uid} does not match current prediction inputs")
        resumed[uid] = plan.store.ref(ref)
        progress("e19_unit_reused", unit_id=uid, model=model, reference=resumed[uid])

    published = {}
    pending_units = [uid for uid in unit_ids if uid not in resumed]
    if pending_units:
        t_boot = time.monotonic()
        draws = load_bootstrap(plan)
        counts = bootstrap_counts(draws, len(plan["screen_ids"]))
        progress("e19_bootstrap_ready", model=model, draws=int(draws.shape[0]),
                 images=int(draws.shape[1]), elapsed_seconds=time.monotonic() - t_boot)
        for uid in pending_units:
            t_unit = time.monotonic()
            progress("e19_unit_start", unit_id=uid, model=model)
            rows, cutoffs = evaluate_e19_unit(plan, uid, predictions, counts)
            unit_pred_refs = {bid: ref_by_block[bid] for bid in units[uid]["required_blocks"]}
            ref = publish_analysis_block(plan, uid, unit_pred_refs, rows, producer, cutoffs=cutoffs)
            load_analysis_block(plan, ref, units=units)
            published[uid] = ref
            progress("e19_unit_complete", unit_id=uid, model=model, reference=ref,
                     rows=len(rows), cutoffs=len(cutoffs),
                     elapsed_seconds=time.monotonic() - t_unit, peak_rss_bytes=peak_memory())

    all_refs = {uid: resumed[uid] if uid in resumed else published[uid] for uid in unit_ids}
    summary = {
        "status": "passed", "experiment": "E19", "model": model, "plan": plain(plan.reference),
        "units": all_refs, "analyses": list(all_refs.values()), "reused": [uid for uid in unit_ids if uid in resumed],
        "published": [uid for uid in unit_ids if uid in published],
        "elapsed_seconds": time.monotonic() - t_start, "peak_rss_bytes": peak_memory(),
    }
    progress("e19_complete", **summary)
    return summary


def _load_refs_file(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "predictions" in data:
        data = data["predictions"]
    if isinstance(data, dict):
        data = list(data.values())
    check(isinstance(data, list) and all(isinstance(item, (str, dict)) for item in data),
          "Prediction references JSON must be a list of explicit completion references")
    return data


def _parse_ref_arg(raw):
    raw = raw.strip()
    return json.loads(raw) if raw.startswith("{") else raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-root", required=True)
    parser.add_argument("--manifest", required=True, help="Prepared Stage 4 plan manifest path or JSON ref")
    parser.add_argument("--model", required=True, choices=tuple(DEPTHS))
    parser.add_argument("--predictions", required=True, help="JSON file listing explicit prediction refs")
    parser.add_argument("--sources", default=None, help="Optional comma-separated source subset (en,hi,bn,te)")
    parser.add_argument("--resume", action="append", default=[], help="Explicit completed E19 analysis ref")
    parser.add_argument("--resume-file", default=None, help="Optional JSON file listing completed E19 refs")
    parser.add_argument("--output-index", help="Optional path for the completed E19 handoff JSON")
    parser.add_argument("--storage-scope", default="persistent-drive",
                        choices=("persistent-drive", "local-development"))
    args = parser.parse_args()
    root = phase1_root(args.phase1_root)
    protocol = json.loads((REPO / "configs/protocol.json").read_text(encoding="utf-8"))
    producer = current_producer(args.storage_scope)
    emit("stage4_e19_start", model=args.model, code_commit=producer["code_commit"])
    plan = load_stage4_plan(root, protocol, _parse_ref_arg(args.manifest))
    pred_refs = _load_refs_file(args.predictions)
    resume_refs = [_parse_ref_arg(item) for item in args.resume]
    if args.resume_file is not None:
        raw_resume = json.loads(Path(args.resume_file).read_text(encoding="utf-8"))
        if isinstance(raw_resume, dict) and "analyses" in raw_resume:
            raw_resume = raw_resume["analyses"]
        if isinstance(raw_resume, dict):
            raw_resume = list(raw_resume.values())
        check(isinstance(raw_resume, list), "Resume file must contain a list of references")
        resume_refs.extend(raw_resume)
    sources = None if args.sources is None else [s.strip() for s in args.sources.split(",") if s.strip()]
    summary = run_e19(plan, args.model, pred_refs, producer, sources=sources, resume_refs=resume_refs, progress=emit)
    if args.output_index:
        output = Path(args.output_index)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except (ContractError, OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        emit("stage4_e19_blocked", status="blocked", blockers=[str(exc)])
        raise SystemExit(2)
