"""Complete Stage 4 result assembly with independent Drive readback."""

import json
import subprocess
import sys
from collections import Counter

import numpy as np

from src.stage2.schema import REPO, check, contract, plain
from src.stage3.artifacts import Store
from .analysis import ANALYSIS_KIND, MACRO, analysis_units, load_analysis_block
from .metrics import confidence_cutoff, probabilities
from .plan import _publish, load_bootstrap
from .predictions import PREDICTION_KIND, load_prediction_block

PENDING = "stage4_release_readback"
RELEASE = "stage4_release"
OPTIONAL_COVERAGE = [
    {"experiment": "E04", "arm": "post_caption", "status": "unavailable",
     "reason": "The accepted Stage 2 plan did not expose this decoder site."},
    {"experiment": "E04", "arm": "decoder_visual_pool", "status": "unavailable",
     "reason": "The accepted Stage 2 plan did not expose decoder visual-token pooling."},
]


def _validate_screen_support(plan):
    by_split = {}
    for split in ("SCREEN", "DEV-C"):
        targets = [t["values"] for t in plan.fit["targets"].values() if t["split"] == split]
        columns = zip(*targets, strict=True)
        totals = {category: (col.count(1), col.count(0)) for category, col in
                  zip(plan["categories"], columns, strict=True)}
        minimum = plan.protocol["minimum_positive_and_negative"][split]
        check(all(min(totals[c]) >= minimum for c in plan["primary"]),
              "Unexpected missing fixed primary support")
        by_split[split] = totals
    return by_split


def _validate_analysis_support(unit, rows, support):
    for row in rows:
        category = row["category"]
        if category == MACRO:
            continue
        comp = unit["comparisons"][row["comparison_id"]]
        split = "DEV-C" if comp["kind"] == "selective" and row["statistic"] == "source_coverage" else "SCREEN"
        pos, neg = support[split][category]
        if comp.get("label") == 0:
            pos = 0
        if comp.get("label") == 1:
            neg = 0
        counts = row["support"]
        check((counts["positive"], counts["negative"]) == (pos, neg),
              "Analysis row changed its declared paired cohort")


def _validate_e19(plan, unit, entry, prediction_refs):
    if unit["experiment"] != "E19":
        return
    one = next(iter(unit["comparisons"].values()))["inputs"][0]
    pred = load_prediction_block(plan, prediction_refs[one["block_id"]],
                                 expected_block=one["block_id"])[one["route_id"]]
    check(pred.split == "DEV-C", "Cutoff source must be DEV-C")
    p = probabilities(pred.logits)
    category_to_index = {category: idx for idx, category in enumerate(plan["categories"])}
    for row in entry["cutoffs"]:
        idx = category_to_index[row["category"]]
        baseline = row["baseline"]
        if baseline == "probe":
            scores = p[:, idx] if pred.estimable[idx] else p[:0, idx]
        else:
            prior = pred.prevalence[idx] if baseline == "fit_prior" else 0
            scores = np.full(len(pred.rows), prior)
        expected = confidence_cutoff(scores, row["nominal"])
        check(all(row[key] == value for key, value in expected.items()), "Wrong source-only E19 cutoff")


@contract
def assemble_release(plan, prediction_refs, analysis_refs, *, progress=None):
    """Check exact prediction and analysis coverage and return the release body."""
    progress = progress or (lambda event: None)
    load_bootstrap(plan)
    support = _validate_screen_support(plan)
    units = analysis_units(plan)
    predictions, analyses, statuses = {}, {}, Counter()
    for ref in prediction_refs:
        predict = load_prediction_block(plan, ref)
        block_id = plan.store.json(ref, "body.json", PREDICTION_KIND)["block_id"]
        check(block_id not in predictions, "Duplicate prediction ownership")
        predictions[block_id] = plan.store.ref(ref)
        block = plan["blocks"][block_id]
        check(set(predict) == {r["route_id"] for r in block["routes"]}, "Prediction route coverage mismatch")
        if len(predictions) % 10 == 0:
            progress({"event": "stage4_verified_predictions", "count": len(predictions)})
    check(set(predictions) == set(plan["blocks"]), "Incomplete canonical prediction inventory")

    for ref in analysis_refs:
        block = load_analysis_block(plan, ref, units=units)
        uid = block["unit_id"]
        check(uid not in analyses, "Duplicate analysis ownership")
        check(plain(block["predictions"]) == {bid: predictions[bid] for bid in units[uid]["required_blocks"]},
              "Analysis references a different prediction release")
        _validate_analysis_support(units[uid], block["rows"], support)
        _validate_e19(plan, units[uid], block, predictions)
        statuses.update(row["status"] for row in block["rows"])
        analyses[uid] = plan.store.ref(ref)
        if len(analyses) % 5 == 0:
            progress({"event": "stage4_verified_analyses", "count": len(analyses)})
    check(set(analyses) == set(units), "Incomplete analysis inventory")
    return {
        "schema": "stage4.release.v1", "plan": plain(plan.reference), "plan_identity": plan["identity"],
        "stage3_registry": plain(plan["registry"]), "g0_gate": plain(plan["g0_gate"]),
        "dataset_fingerprint": plan["dataset_fingerprint"], "predictions": predictions, "analyses": analyses,
        "optional_coverage": OPTIONAL_COVERAGE, "status_counts": dict(statuses),
        "route_counts": {"SCREEN": 529, "DEV-C": 8}, "analysis_units": len(units), "stage4_complete": False,
    }


@contract
def independent_readback(root, protocol, reference, *, progress=None):
    store = Store(root, protocol)
    body = store.json(reference, "body.json", PENDING)
    check(body["schema"] == "stage4.release.v1" and body["stage4_complete"] is False,
          "Wrong Stage 4 readback inventory")
    store.dataset = body["dataset_fingerprint"]
    store.marker(reference, PENDING)
    store.complete(body["plan"], "stage4_plan")
    for key, kind in (("predictions", PREDICTION_KIND), ("analyses", ANALYSIS_KIND)):
        for number, ref in enumerate(body[key].values(), start=1):
            store.complete(ref, kind)
            if progress is not None and (number % 10 == 0 or number == len(body[key])):
                progress({"event": "stage4_readback_progress", "kind": key,
                          "count": number, "total": len(body[key])})
    return store.ref(reference)


@contract
def validate_stage4(plan, prediction_refs, analysis_refs, producer, *, progress=None):
    body = assemble_release(plan, prediction_refs, analysis_refs, progress=progress)
    upstream = [plan.reference, *body["predictions"].values(), *body["analyses"].values()]
    pending = _publish(plan.store, plan["dataset_fingerprint"], "results/stage4/readback", PENDING,
                       body, {}, upstream, producer)
    if progress is not None:
        progress({"event": "stage4_readback_start", "reference": pending})
    cmd = [sys.executable, "-u", "-m", "src.stage4.operations", "readback",
           "--phase1-root", str(plan.root), "--readback-ref", json.dumps(pending)]
    subprocess.run(cmd, cwd=REPO, check=True)
    body.update(stage4_complete=True, independent_readback=pending)
    return _publish(plan.store, plan["dataset_fingerprint"], "results/stage4/releases", RELEASE,
                    body, {}, [pending, plan.reference], producer)


@contract
def load_stage4_release(plan, reference):
    """Verify the accepted summary and readback binding for Stage 5 consumption."""
    ref = plan.store.ref(reference)
    accepted = plan.store.json(ref, "body.json", RELEASE)
    expected = {key: value for key, value in accepted.items() if key != "independent_readback"}
    expected["stage4_complete"] = False
    pending = plan.store.json(accepted["independent_readback"], "body.json", PENDING)
    check(accepted["schema"] == "stage4.release.v1" and accepted["stage4_complete"] is True
          and accepted["plan"] == plain(plan.reference) and accepted["plan_identity"] == plan["identity"]
          and accepted["stage3_registry"] == plain(plan["registry"])
          and accepted["g0_gate"] == plain(plan["g0_gate"])
          and set(accepted["predictions"]) == set(plan["blocks"])
          and set(accepted["analyses"]) == set(analysis_units(plan))
          and pending == expected, "Stage 4 release is incomplete or incompatible")
    return accepted
