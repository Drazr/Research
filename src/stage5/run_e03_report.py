"""CPU reporting worker for the shared Stage 5 E03 shallow-reference bundle."""

import argparse
from collections import Counter
import json
from pathlib import Path

# Stage 3 controls native library initialization order in the pinned runtime.
from src.stage5.interfaces import REPORT_KIND, current_producer, open_release, publish_owner_report
from src.common.contracts import ContractError, LANGUAGES
from src.common.storage import phase1_root
from src.stage2.schema import REPO, check, contract, plain
from src.stage4.interfaces import MACRO, METRICS, required_rows

OWNER = "e03"
SHARED_UNIT_ID = "E03/e03/shared"
LANGUAGE_SPECIFIC_FAMILIES = ("word_tfidf", "char_tfidf", "length")
POOLED_FAMILY = "pooled_char_tfidf"
EXPECTED_FAMILIES = (*LANGUAGE_SPECIFIC_FAMILIES, POOLED_FAMILY)
EXPECTED_ROUTE_COUNT = len(LANGUAGES) * len(LANGUAGE_SPECIFIC_FAMILIES) + 1
EXPECTED_COMPARISON_COUNT = len(LANGUAGES) * len(EXPECTED_FAMILIES) + 1
SCOPE_ORDER = ("language_specific", "pooled_per_context", "pooled_context_macro")
RECIPE_ORDER = {family: idx for idx, family in enumerate(EXPECTED_FAMILIES)}
LANGUAGE_ORDER = {lang: idx for idx, lang in enumerate(LANGUAGES)}
METRIC_ORDER = {metric: idx for idx, metric in enumerate(METRICS)}


def emit(event):
    print(json.dumps(event), flush=True)


def _read_text(path):
    return Path(path).read_text(encoding="utf-8-sig")


def _parse_reference_arg(raw, name):
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ContractError(f"Invalid JSON for {name}: {exc}") from exc
    check(isinstance(parsed, dict) and set(parsed) == {"path", "sha256"}
          and all(isinstance(parsed[k], str) and parsed[k].strip() for k in ("path", "sha256")),
          f"{name} must be a JSON object with non-empty path and sha256")
    return parsed


def _classify_comparison(comp, route):
    kind = comp["kind"]
    family = route["condition"]
    if kind == "absolute":
        view = comp.get("view")
        check(view in LANGUAGES, "E03 absolute comparison requires a scheduled language view")
        if family == POOLED_FAMILY:
            check(route["source_context"] is None and route["target_context"] is None,
                  "Pooled-character route must have pooled source and target contexts")
            return "pooled_per_context", (family, view)
        check(family in LANGUAGE_SPECIFIC_FAMILIES, "Unexpected E03 shallow recipe family")
        check(route["source_context"] == view and route["target_context"] == view,
              "Language-specific E03 route context mismatch")
        return "language_specific", (family, view)
    if kind == "pooled_context_mean":
        check(family == POOLED_FAMILY and route["source_context"] is None and route["target_context"] is None,
              "Pooled context macro must use the pooled-character route")
        check(list(comp.get("views") or ()) == list(LANGUAGES),
              "Pooled context macro must equally weight all four scheduled languages")
        return "pooled_context_macro", (family, tuple(LANGUAGES))
    check(False, f"Unexpected comparison kind in shared E03 report: {kind}")


@contract
def verify_e03_report(report, run=None):
    """Verify exact shared E03 coverage, recipes, routes, rows and provenance."""
    check(isinstance(report, dict), "Shared E03 report body must be a mapping")
    check(report.get("schema") == "stage5.owner_report.v1", "Wrong Stage 5 owner report schema")
    check(report.get("owner") == OWNER, "Report owner is not shared E03")
    check(report.get("optional_coverage") == {}, "Shared E03 report must not carry model optional coverage")

    tables = report.get("tables")
    check(isinstance(tables, dict) and set(tables) == {SHARED_UNIT_ID},
          "Shared E03 report must contain only the accepted E03/e03/shared unit")
    table = tables[SHARED_UNIT_ID]
    check(table.get("cutoffs") == [], "Shared E03 unit must not carry E19 cutoffs")
    check(table.get("association_memberships") == {}, "Shared E03 unit must not carry E25 memberships")

    routes = report.get("routes")
    check(isinstance(routes, dict) and len(routes) == EXPECTED_ROUTE_COUNT,
          "Shared E03 report route count mismatch")
    expected_route_specs = {
        *((family, lang, lang) for lang in LANGUAGES for family in LANGUAGE_SPECIFIC_FAMILIES),
        (POOLED_FAMILY, None, None),
    }
    observed_route_specs = set()
    route_blocks = set()
    for route_id, route in routes.items():
        check(isinstance(route_id, str) and route_id, "Invalid E03 route identifier")
        check(route.get("owner") == OWNER and route.get("split") == "SCREEN",
              "Shared E03 route must belong to e03 on SCREEN")
        check(route.get("site") is None and route.get("depth_role") is None,
              "Shared E03 route must not carry neural decoder site or depth role")
        check(isinstance(route.get("fit_lineage"), dict) and bool(route["fit_lineage"]),
              "Shared E03 route is missing accepted Stage 3 fit lineage")
        spec = (route.get("condition"), route.get("source_context"), route.get("target_context"))
        check(spec in expected_route_specs and spec not in observed_route_specs,
              "Unexpected or duplicate shared E03 route recipe/context")
        observed_route_specs.add(spec)
        route_blocks.add(route["block"])
    check(observed_route_specs == expected_route_specs, "Incomplete shared E03 route coverage")

    predictions = table.get("predictions")
    check(isinstance(predictions, dict) and set(predictions) == route_blocks
          and len(predictions) == EXPECTED_ROUTE_COUNT,
          "Shared E03 prediction reference coverage mismatch")

    comparisons = table.get("comparisons")
    check(isinstance(comparisons, dict) and len(comparisons) == EXPECTED_COMPARISON_COUNT,
          "Shared E03 comparison count mismatch")
    expected_design = {
        *(("language_specific", (family, lang)) for lang in LANGUAGES for family in LANGUAGE_SPECIFIC_FAMILIES),
        *(("pooled_per_context", (POOLED_FAMILY, lang)) for lang in LANGUAGES),
        ("pooled_context_macro", (POOLED_FAMILY, tuple(LANGUAGES))),
    }
    observed_design = set()
    for cid, comp in comparisons.items():
        check(comp.get("kind") != "neural_minus_shallow",
              "Neural-minus-shallow comparisons belong to model workers, not shared E03")
        inputs = comp.get("inputs")
        check(isinstance(inputs, list) and len(inputs) == 1, "Shared E03 comparison must take one route input")
        inp = inputs[0]
        check(inp["route_id"] in routes and routes[inp["route_id"]]["block"] == inp["block_id"]
              and inp["block_id"] in predictions,
              "Shared E03 comparison input does not match verified routes/predictions")
        check(list(comp.get("statistics") or ()) == list(METRICS),
              "Shared E03 comparison statistics mismatch")
        design_key = _classify_comparison(comp, routes[inp["route_id"]])
        check(design_key not in observed_design, "Duplicate shared E03 comparison design")
        observed_design.add(design_key)
    check(observed_design == expected_design, "Incomplete shared E03 comparison design coverage")

    categories = list(report.get("categories") or ())
    primary = list(report.get("primary") or ())
    check(bool(categories) and bool(primary) and set(primary) <= set(categories),
          "Invalid categories or primary cohort in shared E03 report")
    expected_keys = {
        (cid, category, metric)
        for cid, comp in comparisons.items()
        for category in (*categories, MACRO)
        for metric in comp["statistics"]
    }
    if run is not None:
        check(expected_keys == required_rows(run.plan, run.units[SHARED_UNIT_ID]),
              "Shared E03 required row inventory differs from Stage 4 plan")

    rows = table.get("rows")
    check(isinstance(rows, list) and len(rows) == len(expected_keys), "Shared E03 row count mismatch")
    seen_keys = set()
    status_counts = Counter()
    interval_rows = 0
    for row in rows:
        key = (row["comparison_id"], row["category"], row["statistic"])
        check(key in expected_keys and key not in seen_keys, "Unexpected or duplicate shared E03 row")
        seen_keys.add(key)
        check("value" in row and "status" in row and "support" in row, "Shared E03 row missing required fields")
        status_counts[row["status"]] += 1
        if row.get("interval") is not None:
            interval_rows += 1
    check(seen_keys == expected_keys, "Incomplete shared E03 row coverage")
    return {
        "unit_id": SHARED_UNIT_ID,
        "routes": len(routes),
        "predictions": len(predictions),
        "comparisons": len(comparisons),
        "rows": len(rows),
        "interval_rows": interval_rows,
        "status_counts": dict(sorted(status_counts.items())),
    }


@contract
def build_shallow_reference_rows(report):
    """Project the verified canonical E03 report into sorted display rows in memory."""
    verify_e03_report(report)
    table = report["tables"][SHARED_UNIT_ID]
    routes = report["routes"]
    primary_set = set(report["primary"])
    category_order = {cat: idx for idx, cat in enumerate((*report["categories"], MACRO))}

    display_rows = []
    for row in table["rows"]:
        cid = row["comparison_id"]
        comp = table["comparisons"][cid]
        inp = comp["inputs"][0]
        route = routes[inp["route_id"]]
        scope, _ = _classify_comparison(comp, route)
        views = [comp["view"]] if comp["kind"] == "absolute" else list(comp["views"])
        context = comp["view"] if comp["kind"] == "absolute" else "context_macro"
        display_rows.append({
            "unit_id": SHARED_UNIT_ID,
            "comparison_id": cid,
            "kind": comp["kind"],
            "scope": scope,
            "recipe": route["condition"],
            "context": context,
            "views": views,
            "source_context": route["source_context"],
            "target_context": route["target_context"],
            "split": route["split"],
            "route_id": inp["route_id"],
            "block_id": inp["block_id"],
            "category": row["category"],
            "in_primary_cohort": True if row["category"] == MACRO else row["category"] in primary_set,
            "statistic": row["statistic"],
            "value": row["value"],
            "status": row["status"],
            "reason": row.get("reason"),
            "support": plain(row["support"]),
            "interval": plain(row.get("interval")),
            "analysis_reference": plain(table["reference"]),
            "prediction_reference": plain(table["predictions"][inp["block_id"]]),
            "fit_lineage": plain(route["fit_lineage"]),
        })

    display_rows.sort(key=lambda item: (
        SCOPE_ORDER.index(item["scope"]),
        RECIPE_ORDER[item["recipe"]],
        LANGUAGE_ORDER.get(item["context"], len(LANGUAGES)),
        category_order[item["category"]],
        METRIC_ORDER[item["statistic"]],
    ))
    return display_rows


@contract
def format_e03_summary(report, *, run=None):
    """Build a compact in-memory coverage and primary-macro summary from the canonical report."""
    coverage = verify_e03_report(report, run=run)
    table = report["tables"][SHARED_UNIT_ID]
    routes = report["routes"]
    comp_meta = {}
    for cid, comp in table["comparisons"].items():
        route = routes[comp["inputs"][0]["route_id"]]
        scope, _ = _classify_comparison(comp, route)
        comp_meta[cid] = (scope, route["condition"])

    scope_counts = Counter()
    recipe_counts = Counter()
    primary_macro_rows = 0
    for row in table["rows"]:
        scope, recipe = comp_meta[row["comparison_id"]]
        scope_counts[scope] += 1
        recipe_counts[recipe] += 1
        if row["category"] == MACRO:
            primary_macro_rows += 1

    return {
        "owner": OWNER,
        "schema": report["schema"],
        "release": plain(report["release"]),
        "analysis_reference": plain(table["reference"]),
        "coverage": coverage,
        "rows_by_scope": dict(sorted(scope_counts.items())),
        "rows_by_recipe": dict(sorted(recipe_counts.items())),
        "primary_macro_rows": primary_macro_rows,
        "screen_images": len(report["screen_ids"]),
        "categories": len(report["categories"]),
        "primary_cohort": len(report["primary"]),
    }


@contract
def run_e03_report(root, protocol, release_ref, producer, *, resume=None, progress=None):
    """Open the accepted Stage 4 release, publish or resume shared E03, and verify coverage."""
    run = open_release(root, protocol, release_ref)
    ref = publish_owner_report(run, OWNER, producer, resume=resume, progress=progress)
    body = run.plan.store.json(ref, "body.json", REPORT_KIND)
    summary = format_e03_summary(body, run=run)
    if progress:
        progress({
            "event": "stage5_e03_report_verified",
            "owner": OWNER,
            "unit": SHARED_UNIT_ID,
            "comparisons": summary["coverage"]["comparisons"],
            "rows": summary["coverage"]["rows"],
            "routes": summary["coverage"]["routes"],
        })
    return {
        "reference": plain(ref),
        "manifest": str(run.plan.root / ref["path"]),
        "summary": summary,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-root", required=True)
    parser.add_argument("--release", required=True, help="JSON object with path and sha256")
    parser.add_argument("--resume", default=None, help="Optional prior E03 report JSON object with path and sha256")
    parser.add_argument("--storage-scope", choices=("persistent-drive", "local-development"),
                        default="persistent-drive")
    args = parser.parse_args(argv)

    release_ref = _parse_reference_arg(args.release, "--release")
    resume_ref = _parse_reference_arg(args.resume, "--resume") if args.resume is not None else None
    producer = current_producer(args.storage_scope)
    emit({
        "event": "stage5_e03_report_start",
        "owner": OWNER,
        "code_commit": producer["code_commit"],
        "resume": resume_ref is not None,
    })
    protocol = json.loads(_read_text(REPO / "configs/protocol.json"))
    result = run_e03_report(
        phase1_root(args.phase1_root),
        protocol,
        release_ref,
        producer,
        resume=resume_ref,
        progress=emit,
    )
    emit({
        "status": "passed",
        "owner": OWNER,
        "reference": result["reference"],
        "manifest": result["manifest"],
        "summary": result["summary"],
        "stage5_complete": False,
    })
    return result


if __name__ == "__main__":
    try:
        main()
    except (ContractError, OSError, ValueError, TypeError, KeyError) as exc:
        emit({"status": "blocked", "owner": OWNER, "blockers": [str(exc)], "stage5_complete": False})
        raise SystemExit(2) from exc
