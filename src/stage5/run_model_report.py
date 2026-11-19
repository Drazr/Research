"""Stage 5 CPU per-model reporting worker for Qwen and Gemma."""

import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from collections import Counter
import json
from pathlib import Path

# Stage 3 controls native library initialization order in the pinned runtime.
from src.stage5.interfaces import REPORT_KIND, current_producer, open_release, publish_owner_report
from src.common.contracts import ContractError, LANGUAGES
from src.common.storage import phase1_root
from src.stage2.schema import REPO, check, contract, plain
from src.stage4.interfaces import DEPTHS

MODELS = ("qwen", "gemma")
FAMILIES = ("E01", "E02", "E03", "E04", "E19", "E25")
E03_SHALLOW_CONDITIONS = ("word_tfidf", "char_tfidf", "length", "pooled_char_tfidf")
EXPECTED_UNIT_COUNTS = {"E01": 4, "E02": 5, "E03": 1, "E04": 4, "E19": 4, "E25": 5}


def emit(event):
    """Emit one JSON progress/result event with immediate flushing."""
    print(json.dumps(plain(event), allow_nan=False), flush=True)


def read_text(path):
    return Path(path).read_text(encoding="utf-8-sig")


def parse_reference_arg(raw):
    """Parse a JSON reference object passed directly or as a JSON file path."""
    if isinstance(raw, dict):
        return raw
    text = str(raw).strip()
    if text.startswith("{"):
        parsed = json.loads(text)
    else:
        parsed = json.loads(read_text(text))
    check(isinstance(parsed, dict) and set(parsed) == {"path", "sha256"},
          "Reference must be a JSON object with exact keys 'path' and 'sha256'")
    return parsed


def expected_model_units(model):
    """Return the exact 23 scheduled Stage 4 analysis unit IDs owned by one model."""
    check(model in MODELS, "MODEL must be 'qwen' or 'gemma'")
    depths = DEPTHS[model]
    units = []
    for depth in depths:
        units.append(f"E01/{model}/{depth}")
    for depth in depths:
        units.append(f"E02/{model}/{depth}")
    units.append(f"E02/{model}/visual")
    units.append(f"E03/{model}/comparison")
    for depth in depths:
        units.append(f"E04/{model}/{depth}")
    for source in LANGUAGES:
        units.append(f"E19/{model}/{source}")
    for source in (*LANGUAGES, "diagonal"):
        units.append(f"E25/{model}/{source}")
    return tuple(sorted(units))


@contract
def validate_model_report_body(body, model):
    """Verify completeness, lineage, and route/comparison matching rules on one model report body."""
    check(model in MODELS, "MODEL must be 'qwen' or 'gemma'")
    check(isinstance(body, dict) and body.get("schema") == "stage5.owner_report.v1",
          "Expected stage5.owner_report.v1 report body")
    check(body.get("owner") == model, f"Report owner mismatch: expected {model}")
    for key in ("release", "plan", "registry", "g0_gate", "suite",
                "dataset_fingerprint", "categories", "primary", "screen_ids",
                "optional_coverage", "tables", "routes"):
        check(key in body, f"Missing report field: {key}")
    check(set(body["primary"]).issubset(set(body["categories"])),
          "Primary cohort must be a subset of categories")

    tables = body["tables"]
    routes = body["routes"]
    expected_uids = set(expected_model_units(model))
    check(set(tables) == expected_uids,
          f"Incomplete or unexpected unit inventory for {model}: expected {len(expected_uids)} units")

    expected_depths = DEPTHS[model]
    midpoint_site = f"decoder:{expected_depths[1]}:last_caption"
    referenced_predictions = set()

    for uid, table in sorted(tables.items()):
        exp, owner, part = uid.split("/")
        check(owner == model, f"Foreign unit owner in report: {uid}")
        check(exp in FAMILIES, f"Unknown experiment family: {exp}")
        for field in ("reference", "comparisons", "predictions", "rows", "cutoffs", "association_memberships"):
            check(field in table, f"Missing table field '{field}' in {uid}")
        check(bool(table["comparisons"]) and bool(table["rows"]), f"Empty comparisons or rows in {uid}")
        referenced_predictions.update(table["predictions"])

        # Verify every comparison's route descriptors and family-specific matching rules.
        for cid, comp in table["comparisons"].items():
            inputs = comp["inputs"]
            check(isinstance(inputs, list) and bool(inputs), f"Missing comparison inputs in {uid}:{cid}")
            resolved = []
            for item in inputs:
                bid, rid = item["block_id"], item["route_id"]
                check(bid in table["predictions"], f"Unlisted prediction block {bid} in {uid}")
                check(rid in routes, f"Unresolved route {rid} in {uid}")
                route = routes[rid]
                check(route["block"] == bid, f"Route block mismatch for {rid} in {uid}")
                check(isinstance(route.get("fit_lineage"), dict) and bool(route["fit_lineage"]),
                      f"Missing fit lineage on route {rid}")
                if route["depth_role"] is not None:
                    check(0 <= route["depth_role"] < len(expected_depths),
                          f"Invalid depth_role on route {rid}")
                resolved.append(route)

            if exp == "E03":
                check(part == "comparison" and comp["kind"] == "neural_minus_shallow",
                      f"Unexpected E03 model comparison in {uid}")
                check(len(resolved) == 2, f"E03 comparison requires neural and shallow routes in {uid}")
                neural, shallow = resolved
                view = comp.get("view")
                check(view in LANGUAGES, f"Invalid E03 view in {uid}")
                check(neural["owner"] == model and neural["split"] == "SCREEN"
                      and neural["condition"] == "caption_only" and neural["site"] == midpoint_site
                      and neural["source_context"] == view and neural["target_context"] == view,
                      f"E03 neural route mismatch in {uid}")
                check(shallow["owner"] == "e03" and shallow["split"] == "SCREEN"
                      and shallow["condition"] in E03_SHALLOW_CONDITIONS,
                      f"E03 shallow route mismatch in {uid}")
                if shallow["condition"] == "pooled_char_tfidf":
                    check(shallow["source_context"] is None and shallow["target_context"] is None,
                          f"Pooled E03 route must have null language context in {uid}")
                else:
                    check(shallow["source_context"] == view and shallow["target_context"] == view,
                          f"Per-language E03 route mismatch in {uid}")
            elif exp == "E19":
                check(comp["kind"] == "selective" and len(resolved) == 2,
                      f"E19 comparison requires DEV-C and SCREEN routes in {uid}")
                dev_route, screen_route = resolved
                check(dev_route["split"] == "DEV-C" and screen_route["split"] == "SCREEN",
                      f"E19 split pairing mismatch in {uid}")
                check(dev_route["source_context"] == part and dev_route["target_context"] == part
                      and screen_route["source_context"] == part,
                      f"E19 source-context matching violation in {uid}")
            elif exp == "E25":
                check(comp["kind"] == "error_pair" and len(resolved) == 2,
                      f"E25 comparison requires paired SCREEN routes in {uid}")
                ra, rb = resolved
                check(ra["split"] == "SCREEN" and rb["split"] == "SCREEN"
                      and ra["site"] == midpoint_site and rb["site"] == midpoint_site
                      and ra["target_context"] != rb["target_context"],
                      f"E25 route pairing violation in {uid}")
                if part == "diagonal":
                    check(ra["source_context"] == ra["target_context"]
                          and rb["source_context"] == rb["target_context"],
                          f"E25 diagonal pair mismatch in {uid}")
                else:
                    check(ra["source_context"] == part and rb["source_context"] == part,
                          f"E25 fixed-source pair mismatch in {uid}")
            else:
                check(all(r["owner"] == model and r["split"] == "SCREEN" for r in resolved),
                      f"{exp} routes must belong to {model} SCREEN in {uid}")

        # Verify cutoff lineage and E25 membership presence rules.
        if exp == "E19":
            check(isinstance(table["cutoffs"], list), f"E19 cutoffs must be a list in {uid}")
        else:
            check(not table["cutoffs"], f"Unexpected cutoffs on non-E19 unit {uid}")
        if exp == "E25":
            check(isinstance(table["association_memberships"], dict),
                  f"E25 association_memberships must be a dict in {uid}")
        else:
            check(not table["association_memberships"],
                  f"Unexpected association_memberships on non-E25 unit {uid}")

        # Verify every row points to a declared comparison and carries support/status metadata.
        seen_rows = set()
        for row in table["rows"]:
            cid = row["comparison_id"]
            key = (cid, row["category"], row["statistic"])
            check(cid in table["comparisons"], f"Row comparison_id not declared in {uid}")
            check(key not in seen_rows, f"Duplicate report row in {uid}")
            seen_rows.add(key)
            check("status" in row and "support" in row and "value" in row,
                  f"Row missing value/status/support fields in {uid}")

    check(referenced_predictions == {route["block"] for route in routes.values()},
          "Route metadata blocks do not match table prediction references")
    return True


def _route_descriptor(routes, item):
    route = routes[item["route_id"]]
    return {
        "block_id": item["block_id"],
        "route_id": item["route_id"],
        "owner": route["owner"],
        "split": route["split"],
        "condition": route["condition"],
        "site": route["site"],
        "depth_role": route["depth_role"],
        "source_context": route["source_context"],
        "target_context": route["target_context"],
        "source_group": route["source_group"],
        "target_group": route["target_group"],
        "fit_lineage": route["fit_lineage"],
    }


def assemble_family_views(body):
    """Assemble in-memory display views for all six families from a verified owner report body."""
    model = body["owner"]
    validate_model_report_body(body, model)
    routes = body["routes"]
    unavailable_by_family = {fam: [] for fam in FAMILIES}
    for item in body.get("optional_coverage", ()):
        exp = item.get("experiment")
        if exp in unavailable_by_family:
            unavailable_by_family[exp].append(dict(item))

    families = {}
    for family in FAMILIES:
        unit_views = {}
        comparison_kinds = Counter()
        row_count = 0
        status_counts = Counter()
        for uid, table in sorted(body["tables"].items()):
            exp, _, part = uid.split("/")
            if exp != family:
                continue
            comparisons = {}
            for cid, comp in table["comparisons"].items():
                comparison_kinds[comp["kind"]] += 1
                comparisons[cid] = {
                    **{k: v for k, v in comp.items() if k != "inputs"},
                    "inputs": [_route_descriptor(routes, item) for item in comp["inputs"]],
                }
            for row in table["rows"]:
                row_count += 1
                status_counts[row["status"]] += 1
            unit_views[uid] = {
                "part": part,
                "reference": table["reference"],
                "predictions": table["predictions"],
                "comparisons": comparisons,
                "rows": table["rows"],
                "cutoffs": table["cutoffs"],
                "association_memberships": table["association_memberships"],
            }
        families[family] = {
            "units": unit_views,
            "unit_count": len(unit_views),
            "comparison_kinds": dict(sorted(comparison_kinds.items())),
            "row_count": row_count,
            "status_counts": dict(sorted(status_counts.items())),
            "unavailable_arms": unavailable_by_family[family],
        }
    return {
        "schema": body["schema"],
        "model": model,
        "depths": list(DEPTHS[model]),
        "release": body["release"],
        "plan": body["plan"],
        "registry": body["registry"],
        "g0_gate": body["g0_gate"],
        "suite": body["suite"],
        "dataset_fingerprint": body["dataset_fingerprint"],
        "categories": body["categories"],
        "primary": body["primary"],
        "screen_count": len(body["screen_ids"]),
        "families": families,
    }


def summarize_model_report(body):
    """Build a compact JSON-serializable summary directly from a verified report body."""
    model = body["owner"]
    validate_model_report_body(body, model)
    unavailable_by_family = {fam: [] for fam in FAMILIES}
    for item in body.get("optional_coverage", ()):
        exp = item.get("experiment")
        if exp in unavailable_by_family:
            unavailable_by_family[exp].append(dict(item))

    buckets = {
        fam: {"units": 0, "comparisons": 0, "comparison_kinds": Counter(),
              "rows": 0, "status_counts": Counter(), "cutoffs": 0, "membership_units": 0}
        for fam in FAMILIES
    }
    total_units = 0
    total_comparisons = 0
    total_rows = 0
    total_cutoffs = 0
    for uid, table in body["tables"].items():
        fam = buckets[uid.split("/")[0]]
        comp_count = len(table["comparisons"])
        row_count = len(table["rows"])
        cutoff_count = len(table["cutoffs"])
        fam["units"] += 1
        fam["comparisons"] += comp_count
        fam["rows"] += row_count
        fam["cutoffs"] += cutoff_count
        if table["association_memberships"]:
            fam["membership_units"] += 1
        for comp in table["comparisons"].values():
            fam["comparison_kinds"][comp["kind"]] += 1
        for row in table["rows"]:
            fam["status_counts"][row["status"]] += 1
        total_units += 1
        total_comparisons += comp_count
        total_rows += row_count
        total_cutoffs += cutoff_count

    family_summary = {
        family: {
            "units": buckets[family]["units"],
            "comparisons": buckets[family]["comparisons"],
            "comparison_kinds": dict(sorted(buckets[family]["comparison_kinds"].items())),
            "rows": buckets[family]["rows"],
            "status_counts": dict(sorted(buckets[family]["status_counts"].items())),
            "cutoffs": buckets[family]["cutoffs"],
            "membership_units": buckets[family]["membership_units"],
            "unavailable_arms": unavailable_by_family[family],
        }
        for family in FAMILIES
    }
    return {
        "schema": body["schema"],
        "model": model,
        "depths": list(DEPTHS[model]),
        "unit_count": total_units,
        "comparison_count": total_comparisons,
        "row_count": total_rows,
        "cutoff_count": total_cutoffs,
        "route_count": len(body["routes"]),
        "category_count": len(body["categories"]),
        "primary_count": len(body["primary"]),
        "screen_count": len(body["screen_ids"]),
        "unavailable_arms": list(body.get("optional_coverage", ())),
        "families": family_summary,
    }


def format_model_report_summary(summary):
    """Render a compact human-readable report overview from a model summary dict."""
    lines = [
        f"Stage 5 model report ({summary['schema']}): model={summary['model']} depths={summary['depths']}",
        (f"Coverage: {summary['unit_count']} units, {summary['comparison_count']} comparisons, "
         f"{summary['row_count']} rows, {summary['route_count']} routes, "
         f"{summary['primary_count']}/{summary['category_count']} primary categories, "
         f"{summary['screen_count']} SCREEN images"),
    ]
    for family in FAMILIES:
        fam = summary["families"][family]
        extra = []
        if fam["cutoffs"]:
            extra.append(f"cutoffs={fam['cutoffs']}")
        if fam["unavailable_arms"]:
            arms = ",".join(a["arm"] for a in fam["unavailable_arms"])
            extra.append(f"unavailable=[{arms}]")
        suffix = f" ({', '.join(extra)})" if extra else ""
        lines.append(
            f"  {family}: units={fam['units']} comparisons={fam['comparisons']} "
            f"rows={fam['rows']} statuses={fam['status_counts']}{suffix}"
        )
    return "\n".join(lines)


@contract
def run_model_report(root, protocol, release_reference, model, producer, *, resume=None, progress=None):
    """Open the accepted Stage 4 release, publish or reuse one model report, and return reference + summary."""
    check(model in MODELS, "MODEL must be 'qwen' or 'gemma'")
    run = open_release(root, protocol, release_reference)
    ref = publish_owner_report(run, model, producer, resume=resume, progress=progress)
    body = run.plan.store.json(ref, "body.json", REPORT_KIND)
    summary = summarize_model_report(body)
    return {"reference": ref, "resumed": resume is not None, "summary": summary}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-root", required=True)
    parser.add_argument("--release", required=True,
                        help="Accepted Stage 4 release reference JSON object or JSON file path")
    parser.add_argument("--model", required=True, choices=MODELS,
                        help="Single model selector: qwen or gemma")
    parser.add_argument("--storage-scope", choices=("persistent-drive", "local-development"),
                        default="persistent-drive")
    parser.add_argument("--resume", default=None,
                        help="Optional previous compatible report reference JSON object or file path")
    args = parser.parse_args(argv)

    root = phase1_root(args.phase1_root)
    protocol = json.loads(read_text(REPO / "configs/protocol.json"))
    release_ref = parse_reference_arg(args.release)
    resume_ref = None if args.resume is None else parse_reference_arg(args.resume)
    producer = current_producer(args.storage_scope)

    emit({
        "event": "stage5_model_report_start",
        "model": args.model,
        "code_commit": producer["code_commit"],
        "storage_scope": args.storage_scope,
        "resuming": resume_ref is not None,
    })
    result = run_model_report(root, protocol, release_ref, args.model, producer,
                              resume=resume_ref, progress=emit)
    emit({
        "status": "passed",
        "model": args.model,
        "resumed": result["resumed"],
        "reference": result["reference"],
        "manifest": str(root / result["reference"]["path"]),
        "summary": result["summary"],
        "stage5_complete": False,
    })


if __name__ == "__main__":
    try:
        main()
    except (ContractError, OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        emit({"status": "blocked", "blockers": [str(exc)], "stage5_complete": False})
        raise SystemExit(2) from exc
