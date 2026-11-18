"""Frozen comparison work and checkpoint tables for the analysis workers."""

from itertools import combinations
from math import ceil, isfinite

from src.common.contracts import LANGUAGES, fingerprint
from src.stage2.schema import check, contract, frozen, plain
from .metrics import METRICS
from .plan import DEPTHS, STATUSES, _publish
from .predictions import PREDICTION_KIND, load_prediction_block

ANALYSIS_KIND = "stage4_analysis_block"
MACRO = "primary_macro"
E19_METRICS = (
    "source_coverage", "target_coverage", "retained_risk", "base_risk", "score_blind_risk",
    "positive_retention", "negative_retention", "retained_sensitivity", "retained_specificity",
    "overall_retained_positive_correct", "overall_retained_negative_correct", "auroc", "ap",
)
E25_METRICS = ("disagreement", "common_wrong", "error_a", "error_b")
E25_STRATIFIED = ("error_a", "error_b", "common_wrong", "independence", "excess",
                  "error_b_given_a", "error_a_given_b", "null_lower", "null_upper")


def analysis_units(plan):
    """Enumerate immutable unit descriptors, each with explicit required block IDs."""
    routes = {}
    for bid, block in plan["blocks"].items():
        target = plan.fit["groups"][block["target_group"]]
        for route in block["routes"]:
            source = plan.fit["groups"][route["source_group"]]
            key = (block["split"], block["owner"],
                   target.get("condition", target.get("recipe", {}).get("family")),
                   target.get("site"), source["language"], target["language"])
            routes[key] = {"block_id": bid, "route_id": route["route_id"]}
    units = {}

    def add(exp, owner, part, kind, inputs, statistics, **fields):
        unit_id = f"{exp}/{owner}/{part}"
        item = {"kind": kind, "inputs": inputs, "statistics": list(statistics), **fields}
        cid = fingerprint([unit_id, item])
        unit = units.setdefault(unit_id, {"experiment": exp, "owner": owner, "part": part,
                                         "comparisons": {}, "required_blocks": set()})
        check(cid not in unit["comparisons"], "Duplicate declared comparison")
        unit["comparisons"][cid] = item
        unit["required_blocks"].update(ref["block_id"] for ref in inputs)

    for model, depths in DEPTHS.items():
        def dec(depth, pool, condition, s, t, split="SCREEN"):
            return routes[(split, model, condition, f"decoder:{depth}:{pool}", s, t)]

        for depth in depths:
            for s in LANGUAGES:
                for t in LANGUAGES:
                    base = dec(depth, "last_caption", "image_caption", s, t)
                    text = dec(depth, "last_caption", "caption_only", s, t)
                    diag = dec(depth, "last_caption", "image_caption", t, t)
                    text_diag = dec(depth, "last_caption", "caption_only", t, t)
                    add("E01", model, str(depth), "absolute", [base], METRICS)
                    for baseline in ("fit_prior", "always_absent"):
                        add("E01", model, str(depth), baseline, [base], METRICS)
                    if s != t:
                        add("E01", model, str(depth), "transfer_gap", [diag, base], METRICS)
                        add("E02", model, str(depth), "transfer_gap_change",
                            [diag, base, text_diag, text], METRICS)
                    add("E02", model, str(depth), "caption_only", [text], METRICS)
                    add("E02", model, str(depth), "input_delta", [base, text], METRICS)
                    for condition, last in (("image_caption", base), ("caption_only", text)):
                        mean = dec(depth, "caption_mean", condition, s, t)
                        add("E04", model, str(depth), "caption_mean", [mean], METRICS)
                        add("E04", model, str(depth), "pool_delta", [mean, last], METRICS)
        for stage in ("encoder", "projector"):
            visual = routes[("SCREEN", model, "image_only", f"{stage}:None:visual_mean", None, None)]
            add("E02", model, "visual", "image_only", [visual], METRICS)
            for t in LANGUAGES:
                for condition in ("image_caption", "caption_only"):
                    neural = dec(depths[1], "last_caption", condition, t, t)
                    add("E02", model, "visual", "visual_delta", [neural, visual], METRICS)
        for s in LANGUAGES:
            dev = dec(depths[1], "last_caption", "image_caption", s, s, "DEV-C")
            for t in LANGUAGES:
                screen = dec(depths[1], "last_caption", "image_caption", s, t)
                for nominal in (.5, .75, .9):
                    for baseline in ("probe", "fit_prior", "always_absent"):
                        stats = E19_METRICS if baseline == "probe" else E19_METRICS[:-2]
                        add("E19", model, s, "selective", [dev, screen], stats,
                            nominal=nominal, baseline=baseline)
        for s in (*LANGUAGES, "diagonal"):
            for a, b in combinations(LANGUAGES, 2):
                ra = dec(depths[1], "last_caption", "image_caption", a if s == "diagonal" else s, a)
                rb = dec(depths[1], "last_caption", "image_caption", b if s == "diagonal" else s, b)
                for baseline in ("probe", "fit_prior", "always_absent"):
                    add("E25", model, s, "error_pair", [ra, rb], E25_METRICS,
                        label=None, baseline=baseline)
                    for label in (0, 1):
                        add("E25", model, s, "error_pair", [ra, rb], E25_STRATIFIED,
                            label=label, baseline=baseline)
    pooled = routes[("SCREEN", "e03", "pooled_char_tfidf", None, None, None)]
    for t in LANGUAGES:
        add("E03", "e03", "shared", "absolute", [pooled], METRICS, view=t)
        text_routes = [routes[("SCREEN", "e03", family, None, t, t)]
                       for family in ("word_tfidf", "char_tfidf", "length")]
        for shallow in text_routes:
            add("E03", "e03", "shared", "absolute", [shallow], METRICS, view=t)
        for model, depths in DEPTHS.items():
            neural = routes[("SCREEN", model, "caption_only",
                             f"decoder:{depths[1]}:last_caption", t, t)]
            for shallow in (*text_routes, pooled):
                add("E03", model, "comparison", "neural_minus_shallow", [neural, shallow], METRICS, view=t)
    add("E03", "e03", "shared", "pooled_context_mean", [pooled], METRICS, views=list(LANGUAGES))
    for unit in units.values():
        unit["required_blocks"] = sorted(unit["required_blocks"])
    return frozen(units)


def required_rows(plan, unit):
    return {(cid, category, metric)
            for cid, comp in unit["comparisons"].items()
            for category in (*plan["categories"], MACRO)
            for metric in comp["statistics"]}


def _validate_rows(plan, unit, rows):
    check(isinstance(rows, (tuple, list)) and rows, "Empty analysis results")
    seen = set()
    for row in rows:
        cid, category, metric = row["comparison_id"], row["category"], row["statistic"]
        key = (cid, category, metric)
        check(cid in unit["comparisons"] and category in (*plan["categories"], MACRO)
              and isinstance(metric, str) and metric and key not in seen, "Invalid or duplicate analysis row")
        seen.add(key)
        status, value, reason = row["status"], row["value"], row["reason"]
        check(status in STATUSES and status not in ("unavailable", "blocked"),
              "Scheduled analysis row unavailable or blocked")
        check((value is None and status == "undefined")
              or (type(value) in (int, float) and isfinite(value) and status != "undefined"),
              "Inconsistent analysis value and status")
        check(status == "ok" or isinstance(reason, str) and bool(reason.strip()), "Missing status reason")
        support = row["support"]
        check(isinstance(support, dict) and support and all(type(n) is int and n >= 0 for n in support.values()),
              "Missing or invalid support counts")
        if category != MACRO:
            check(support["known"] == support["positive"] + support["negative"], "Inconsistent support counts")
        interval = row.get("interval")
        if interval is not None:
            valid, total = interval["valid_draws"], interval["total_draws"]
            low, high = interval["lower"], interval["upper"]
            check(type(valid) is int and 0 <= valid <= total == plan["bootstrap"]["draws"]
                  and ((valid == 0 and low is None and high is None)
                       or (valid > 0 and isfinite(low) and isfinite(high) and low <= high)),
                  "Invalid paired-bootstrap interval")
        comp = unit["comparisons"][cid]
        needs_interval = comp["kind"] in ("transfer_gap", "transfer_gap_change", "input_delta",
                                         "visual_delta", "pool_delta", "neural_minus_shallow",
                                         "selective", "error_pair") and metric not in (
                                             "source_coverage", "null_lower", "null_upper")
        if needs_interval and status == "ok":
            check(interval is not None, "Missing interval on an estimable planned comparison")
    check(required_rows(plan, unit) <= seen, "Incomplete required analysis rows")


def _analysis_inputs(plan, unit, predictions):
    check(set(predictions) == set(unit["required_blocks"]), "Wrong analysis prediction input coverage")
    return {bid: plan.store.ref(predictions[bid]) for bid in sorted(predictions)}


def _validate_cutoffs(plan, unit, cutoffs):
    if unit["experiment"] != "E19":
        check(not cutoffs, "Cutoffs belong to E19")
        return
    expected = {(baseline, category, nominal)
                for baseline in ("probe", "fit_prior", "always_absent")
                for category in plan["categories"] for nominal in (.5, .75, .9)}
    keys = set()
    for row in cutoffs:
        key = (row["baseline"], row["category"], row["nominal"])
        check(key in expected and key not in keys, "Unexpected E19 cutoff key")
        keys.add(key)
        n, retained, cutoff = row["eligible"], row["retained"], row["cutoff"]
        check(type(n) is int and type(retained) is int and 0 <= retained <= n
              and ((n == 0 and cutoff is None) or (n > 0 and .5 <= cutoff <= 1
                                                   and retained >= ceil(row["nominal"] * n))),
              "Invalid E19 cutoff or tie support")
    check(keys == expected, "Missing E19 cutoff records")


def e25_memberships(plan):
    """Hash the sorted known-label image memberships used by E25."""
    return {category: {str(label): fingerprint([
        image_id for image_id in plan["screen_ids"]
        if plan.fit["targets"][image_id]["values"][idx] == label
    ]) for label in (0, 1)} for idx, category in enumerate(plan["categories"])}


def _validate_memberships(plan, unit, memberships):
    expected = e25_memberships(plan) if unit["experiment"] == "E25" else {}
    check(plain(memberships) == expected, "Wrong E25 association membership identities")


@contract
def publish_analysis_block(plan, unit_id, prediction_refs, rows, producer, *, cutoffs=(), memberships=None):
    """Validate a completed worker unit, then publish tidy rows and its exact inputs."""
    unit = analysis_units(plan)[unit_id]
    inputs = _analysis_inputs(plan, unit, prediction_refs)
    for bid, ref in inputs.items():
        load_prediction_block(plan, ref, expected_block=bid)
    _validate_rows(plan, unit, rows)
    _validate_cutoffs(plan, unit, cutoffs)
    memberships = {} if memberships is None else memberships
    _validate_memberships(plan, unit, memberships)
    body = {
        "schema": "stage4.analysis.v1", "plan": plain(plan.reference), "plan_identity": plan["identity"],
        "unit_id": unit_id, "predictions": inputs, "rows": rows, "cutoffs": cutoffs,
        "association_memberships": memberships,
    }
    return _publish(plan.store, plan["dataset_fingerprint"],
                    f"results/stage4/analysis/{unit['experiment']}/{unit['owner']}",
                    ANALYSIS_KIND, body, {}, [plan.reference, *inputs.values()], producer)


@contract
def load_analysis_block(plan, reference, *, units=None):
    units = analysis_units(plan) if units is None else units
    body = plan.store.json(reference, "body.json", ANALYSIS_KIND)
    check(body["schema"] == "stage4.analysis.v1" and body["plan"] == plain(plan.reference)
          and body["plan_identity"] == plan["identity"], "Analysis lineage mismatch")
    unit = units[body["unit_id"]]
    _analysis_inputs(plan, unit, body["predictions"])
    _validate_rows(plan, unit, body["rows"])
    _validate_cutoffs(plan, unit, body["cutoffs"])
    _validate_memberships(plan, unit, body["association_memberships"])
    return frozen(body)
