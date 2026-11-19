"""Frozen Stage 5 report worker interface."""

from dataclasses import dataclass

from src.common.contracts import canonical_bytes, fingerprint
from src.common.storage import new_run_id, publish
from src.stage2.schema import check, contract, frozen, plain
from src.stage3.artifacts import Store
from src.stage3.operations import current_producer
from src.stage4.interfaces import DEPTHS, analysis_units, load_analysis_block, load_stage4_plan, load_stage4_release

OWNERS = ("qwen", "gemma", "e03")
REPORT_KIND = "stage5_owner_report"


@dataclass(frozen=True)
class ReportRun:
    plan: object
    release_ref: object
    release: object
    units: object


@contract
def open_release(root, protocol, reference):
    """Open an explicitly hash-pinned accepted release and its Stage 4 plan."""
    check(isinstance(reference, dict), "Supply the Stage 4 release path and SHA-256")
    store = Store(root, protocol)
    ref = store.ref(reference)
    body = store.json(ref, "body.json", "stage4_release")
    plan = load_stage4_plan(root, protocol, body["plan"])
    release = load_stage4_release(plan, ref)
    return ReportRun(plan, frozen(ref), frozen(release), analysis_units(plan))


def _write(run, namespace, kind, body, upstream, producer):
    plan = run.plan
    run_id = new_run_id()
    meta = {
        "kind": kind, "status": "passed", "run_id": run_id,
        "code_commit": producer["code_commit"], "storage_scope": producer["storage_scope"],
        "software_versions": producer["software_versions"],
        "dataset_fingerprint": plan["dataset_fingerprint"],
        "protocol_fingerprint": fingerprint(plan.protocol), "model_fingerprint": None,
        "seeds": {}, "upstream_artifacts": plain(upstream),
    }
    folder = publish(plan.root, f"results/stage5/{namespace}/{run_id}", meta,
                     {"body.json": canonical_bytes(plain(body)) + b"\n"})
    reader = Store(plan.root, plan.protocol)
    reader.dataset = plan["dataset_fingerprint"]
    ref = reader.ref(folder / "_COMPLETE.json")
    check(reader.json(ref, "body.json", kind) == plain(body), "Stage 5 readback mismatch")
    return ref


@contract
def build_owner_report(run, owner, *, progress=None):
    """Assemble the exact accepted unit rows, route metadata and lineage for one owner."""
    check(owner in OWNERS, "Unknown report owner")
    plan = run.plan
    tables = {}
    for uid, unit in sorted(run.units.items()):
        if unit["owner"] != owner:
            continue
        reference = run.release["analyses"][uid]
        block = load_analysis_block(plan, reference, units=run.units)
        refs = {bid: run.release["predictions"][bid] for bid in unit["required_blocks"]}
        check(block["unit_id"] == uid and block["predictions"] == refs, "Wrong accepted analysis lineage")
        tables[uid] = {"reference": reference, "comparisons": unit["comparisons"],
                       "predictions": refs, "rows": block["rows"], "cutoffs": block["cutoffs"],
                       "association_memberships": block["association_memberships"]}
        if progress:
            progress({"event": "report_unit_read", "owner": owner, "unit": uid, "rows": len(block["rows"])})

    # Resolve comparison routes without opening logits or feature arrays.
    routes = {}
    for bid in sorted({bid for table in tables.values() for bid in table["predictions"]}):
        block = plan["blocks"][bid]
        target = plan.fit["groups"][block["target_group"]]
        for route in block["routes"]:
            source_id = route["source_group"]
            source = plan.fit["groups"][source_id]
            site = target.get("site")
            depth_role = None
            if site and site.startswith("decoder:"):
                depth_role = DEPTHS[target["model"]].index(int(site.split(":")[1]))
            routes[route["route_id"]] = {
                "block": bid, "split": block["split"], "source_context": source["language"],
                "target_context": target["language"], "owner": block["owner"],
                "condition": target.get("condition", target.get("recipe", {}).get("family")),
                "site": site, "depth_role": depth_role, "target_group": block["target_group"],
                "source_group": source_id, "fit_lineage": plan["ordinary"][source_id],
            }
    return plain({
        "schema": "stage5.owner_report.v1", "owner": owner, "release": run.release_ref,
        "plan": plan.reference, "registry": run.release["stage3_registry"], "g0_gate": run.release["g0_gate"],
        "suite": plan["suite"], "dataset_fingerprint": plan["dataset_fingerprint"],
        "categories": plan["categories"], "primary": plan["primary"], "screen_ids": plan["screen_ids"],
        "optional_coverage": {} if owner == "e03" else run.release["optional_coverage"],
        "tables": tables, "routes": routes,
    })


@contract
def load_owner_report(run, owner, reference, expected):
    """Verify a worker artifact against an exact accepted-input projection."""
    observed = run.plan.store.json(reference, "body.json", REPORT_KIND)
    check(expected["owner"] == owner and observed == expected, "Incompatible or incomplete owner report")
    return observed


@contract
def publish_owner_report(run, owner, producer, *, resume=None, progress=None):
    """Publish or reuse the one complete, checked report for an owner."""
    body = build_owner_report(run, owner, progress=progress)
    if resume is not None:
        load_owner_report(run, owner, resume, body)
        marker = run.plan.store.marker(resume, REPORT_KIND)[2]
        check(marker["code_commit"] == producer["code_commit"], "Report resume code commit differs")
        return run.plan.store.ref(resume)
    return _write(run, f"reports/{owner}", REPORT_KIND, body,
                  [run.release_ref, *(t["reference"] for t in body["tables"].values())], producer)
