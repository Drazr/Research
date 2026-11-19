"""Verify report coverage, align model designs, and publish the reviewed findings."""

from src.common.contracts import fingerprint
from src.stage2.schema import check, contract, plain

from .interfaces import OWNERS, REPORT_KIND, _write, build_owner_report, load_owner_report

SYNTHESIS_KIND = "stage5_synthesis"
FINAL_KIND = "stage5_release"
DISPOSITIONS = ("retain", "revise", "park within tested scope", "insufficient evidence")


def _comparison_key(report, unit, comp):
    inputs = []
    for item in comp["inputs"]:
        route = report["routes"][item["route_id"]]
        site = route["site"]
        if route["depth_role"] is not None:
            stage, _, pool = site.split(":")
            site = f"{stage}:role{route['depth_role']}:{pool}"
        inputs.append([route["split"], route["condition"], site,
                       route["source_context"], route["target_context"]])
    settings = {k: value for k, value in comp.items() if k != "inputs"}
    # Comparison metadata includes the E19 policy and E25 label stratum and baseline.
    return fingerprint([unit.split("/")[0], settings, inputs])


def _comparison_index(report):
    index = {}
    for uid, table in report["tables"].items():
        for cid, comp in table["comparisons"].items():
            key = _comparison_key(report, uid, comp)
            check(key not in index, "Ambiguous cross-model comparison design")
            index[key] = {"unit": uid, "comparison_id": cid}
    return index


@contract
def publish_synthesis(run, reports, producer, *, progress=None):
    """Check all worker rows against the release and publish paired comparison pointers."""
    check(set(reports) == set(OWNERS), "Require both model reports and shared E03")
    refs = {owner: run.plan.store.ref(reports[owner]) for owner in OWNERS}
    indexes = {}
    for owner in OWNERS:
        expected = build_owner_report(run, owner, progress=progress)
        actual = load_owner_report(run, owner, refs[owner], expected)
        if owner != "e03":
            indexes[owner] = _comparison_index(actual)
    left, right = indexes["qwen"], indexes["gemma"]
    check(left.keys() == right.keys(), "Unmatched scheduled cross-model comparisons")
    body = {
        "schema": "stage5.synthesis.v1", "release": run.release_ref, "reports": refs,
        "paired_views": [{"qwen": left[key], "gemma": right[key]} for key in sorted(left)],
        "stage5_complete": False,
    }
    return _write(run, "synthesis", SYNTHESIS_KIND, body, [run.release_ref, *refs.values()], producer)


def _find_evidence(reports, evidence):
    check(set(evidence) == {"unit", "comparison_id", "category", "statistic"}, "Wrong evidence fields")
    uid = evidence["unit"]
    _, owner, _ = uid.split("/")
    table = reports[owner]["tables"][uid]
    for row in table["rows"]:
        if all(row[key] == evidence[key] for key in ("comparison_id", "category", "statistic")):
            return {"analysis": table["reference"], "unit": uid, "row": row}
    check(False, "Decision evidence does not resolve to a report row")


def _review_decisions(reports, reviewed):
    check(set(reviewed) == {"summary", "decisions"}, "Expected summary and decisions")
    check(isinstance(reviewed["summary"], str) and reviewed["summary"].strip(), "Missing synthesis summary")
    check(isinstance(reviewed["decisions"], list), "Decisions must be a list")
    entries = []
    seen = set()
    for entry in reviewed["decisions"]:
        check(set(entry) == {"id", "question", "scope", "finding", "disposition", "alternatives",
                             "next_uncertainty", "decision_change", "next_test", "evidence"},
              "Wrong decision fields")
        for field in ("id", "scope", "finding", "alternatives", "next_uncertainty", "decision_change"):
            check(isinstance(entry[field], str) and entry[field].strip(), f"Missing decision {field}")
        check(entry["id"] not in seen, "Duplicate decision ID")
        seen.add(entry["id"])
        check(entry["question"] in ("G9", "Q6", "G9-Q6"), "Decision outside the scheduled questions")
        check(entry["disposition"] in DISPOSITIONS, "Unknown decision disposition")
        test = entry["next_test"]
        if entry["disposition"] in ("retain", "revise"):
            check(isinstance(test, dict) and set(test) == {"test", "dependencies"}
                  and all(isinstance(value, str) and value.strip() for value in test.values()),
                  "Retained/revised direction needs one immediate test and its dependencies")
        else:
            check(test is None, "Parked or unresolved directions have no scheduled follow-up test")
        evidence = entry["evidence"]
        check(isinstance(evidence, list) and evidence, "Decision requires exact evidence")
        entries.append(dict(entry, evidence_rows=[_find_evidence(reports, ref) for ref in evidence]))
    return {"summary": reviewed["summary"], "decisions": entries}


@contract
def publish_final_report(run, synthesis, reviewed, producer):
    """Finalize a reviewed synthesis; verification cannot decide research claims."""
    store = run.plan.store
    synthesis_ref, _, marker = store.marker(synthesis, SYNTHESIS_KIND)
    body = store.json(synthesis_ref, "body.json", SYNTHESIS_KIND)
    check(body["schema"] == "stage5.synthesis.v1" and body["stage5_complete"] is False
          and body["release"] == plain(run.release_ref) and set(body["reports"]) == set(OWNERS),
          "Incompatible synthesis")
    check(marker["upstream_artifacts"] == [body["release"], *(body["reports"][o] for o in OWNERS)],
          "Wrong synthesis provenance")
    reports = {}
    for owner in OWNERS:
        report = store.json(body["reports"][owner], "body.json", REPORT_KIND)
        check(report["schema"] == "stage5.owner_report.v1" and report["owner"] == owner
              and report["release"] == body["release"], "Incompatible reviewed report")
        reports[owner] = report
    review = _review_decisions(reports, reviewed)
    production = producer["storage_scope"] == "persistent-drive"
    if production:
        for ref, kind in [(run.release_ref, "stage4_release"), (synthesis_ref, SYNTHESIS_KIND),
                          *((ref, REPORT_KIND) for ref in body["reports"].values())]:
            check(store.marker(ref, kind)[2]["storage_scope"] == "persistent-drive",
                  "Production reporting requires persistent Drive evidence")
    final = {
        "schema": "stage5.release.v1", "release": run.release_ref, "synthesis": synthesis_ref,
        "reports": body["reports"], "review": review, "stage5_complete": production,
    }
    namespace = "releases" if production else "fixture-reports"
    return _write(run, namespace, FINAL_KIND, final, [run.release_ref, synthesis_ref], producer)
