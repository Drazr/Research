"""Accepted Stage 3 inputs and immutable Stage 4 work inventory."""

from collections.abc import Mapping

import numpy as np

from src.common.contracts import LANGUAGES, canonical_bytes, fingerprint
from src.common.storage import new_run_id, publish
from src.stage2.schema import check, contract, frozen, plain
from src.stage3.artifacts import Store, array_bytes
from src.stage3.data import CS, load_fit_plan
from src.stage3.readouts import _body

SCHEMA = "stage4.plan.v1"
DEPTHS = {"qwen": (9, 18, 27, 36), "gemma": (9, 17, 26, 34)}
POOLS = ("last_caption", "caption_mean")
CONDITIONS = ("image_caption", "caption_only")
BOOTSTRAP_SEED = 2026111804
BOOTSTRAP_DRAWS = 2000
PERMUTATION_DRAWS = 1000
STATUSES = ("ok", "descriptive_low_support", "undefined", "unavailable", "blocked")


class ScorePlan(Mapping):
    def __init__(self, body, reference, fit, store):
        self._body = frozen(body)
        self.reference = frozen(reference)
        self.fit = fit
        self.root = fit.root
        self.store = store
        self.protocol = fit.protocol

    def __getitem__(self, key):
        return self._body[key]

    def __iter__(self):
        return iter(self._body)

    def __len__(self):
        return len(self._body)


def _unique_references(store, refs, count):
    check(len(refs) == count, "Wrong accepted registry inventory size")
    items = [store.ref(ref) for ref in refs]
    check(len({ref["path"] for ref in items}) == count, "Duplicate registry reference")
    return items


def _accepted_inputs(root, protocol, registry_ref):
    store = Store(root, protocol)
    registry_ref = store.ref(registry_ref)
    registry = store.json(registry_ref, "body.json", "stage3_system_registry")
    check(registry["stage3_complete"] is True and registry["ordinary_systems"] == 145,
          "Require a complete Stage 3 system registry")
    fit = load_fit_plan(registry["plan"], root, protocol)
    pending, _ = _body(fit, registry["independent_readback"], "registry_readback", store)
    expected_pending = {k: v for k, v in registry.items() if k != "independent_readback"}
    expected_pending.update(stage3_complete=False, schema="stage3.registry_readback.v1")
    check(pending == expected_pending, "Stage 3 readback inventory mismatch")
    check(registry["schema"] == "stage3.system_registry.v1"
          and registry["fit_identity"] == fit["fit_identity"]
          and registry["plan"] == plain(fit.reference)
          and registry["suite"] == plain(fit["suite"])
          and registry["stage2_groups"] == plain(fit["groups"]),
          "Stage 3 registry is not bound to its accepted upstream inputs")
    store.marker(registry_ref, "stage3_system_registry")
    gate, _ = _body(fit, registry["g0_gate"], "g0_gate", store)
    check(gate["plan"] == plain(fit.reference)
          and gate["flagged"] is False and .01 < gate["p"] <= 1
          and gate["structural_checks"] == "passed"
          and set(gate["groups"]) == set(fit["groups"]),
          "Unresolved or incomplete G0 gate")
    groups = sorted(fit["groups"])
    check(len(groups) == 145, "Wrong frozen Stage 3 group count")
    selections = _unique_references(store, registry["selections"], 580)
    candidates = _unique_references(store, registry["candidates"], 2320)
    ordinary = {}
    for index, gid in enumerate(groups):
        # Stage 3 writes lexicographically sorted (group, assignment[, C]).
        # Resolve the ordinary slot, then verify its actual semantic identity.
        selection_ref = selections[4 * index + 3]
        block = candidates[16 * index + 12:16 * index + 16]
        selection, _ = _body(fit, selection_ref, "selection", store)
        c = selection["selected_C"]
        check(selection["group_id"] == gid and selection["assignment_id"] == "true"
              and c in CS and selection["candidates"] == block
              and selection["selected"] == block[CS.index(c)],
              "Wrong ordinary selection or candidate lineage")
        ordinary[gid] = {"selection": selection_ref, "candidate": selection["selected"], "C": c}
    return fit, store, registry_ref, registry, ordinary


def _inventory(fit):
    languages = tuple(fit.protocol["languages"])
    check(languages == LANGUAGES, "Wrong language inventory")
    vlm = {}
    text = {}
    for gid, group in fit["groups"].items():
        if group["source"] == "vlm":
            key = (group["model"], group["condition"], group["site"], group["language"])
            check(key not in vlm, "Duplicate VLM feature group")
            vlm[key] = gid
        else:
            check(group["source"] == "e03", "Unknown feature source")
            key = (group["recipe"]["family"], group["language"])
            check(key not in text, "Duplicate E03 feature group")
            text[key] = gid
    expected_vlm = {(model, condition, f"decoder:{depth}:{pool}", language)
                    for model, depths in DEPTHS.items() for depth in depths
                    for condition in CONDITIONS for pool in POOLS for language in languages}
    expected_vlm |= {(model, "image_only", f"{stage}:None:visual_mean", None)
                     for model in DEPTHS for stage in ("encoder", "projector")}
    expected_text = {(family, language) for family in ("word_tfidf", "char_tfidf", "length")
                     for language in languages} | {("pooled_char_tfidf", None)}
    check(set(vlm) == expected_vlm and set(text) == expected_text, "Wrong frozen model or site inventory")
    blocks = {}

    def add(target, split, sources, owner, priority):
        block_id = fingerprint([SCHEMA, target, split])
        blocks[block_id] = {
            "target_group": target, "split": split, "owner": owner, "priority": priority,
            "routes": [{"source_group": gid, "route_id": fingerprint([SCHEMA, gid, target, split])}
                       for gid in sources],
        }

    for (model, condition, site, language), gid in sorted(vlm.items()):
        if condition == "image_only":
            add(gid, "SCREEN", [gid], model, 1)
        else:
            midpoint = condition == "image_caption" and site == f"decoder:{DEPTHS[model][1]}:last_caption"
            sources = [vlm[(model, condition, site, source)] for source in languages]
            add(gid, "SCREEN", sources, model, 0 if midpoint else 1)
            if midpoint:
                add(gid, "DEV-C", [gid], model, 0)
    for gid in text.values():
        add(gid, "SCREEN", [gid], "e03", 1)
    check(sum(len(b["routes"]) for b in blocks.values() if b["split"] == "SCREEN") == 529
          and sum(len(b["routes"]) for b in blocks.values() if b["split"] == "DEV-C") == 8,
          "Wrong Stage 4 route counts")
    return blocks


def _body_from_registry(root, protocol, registry_ref):
    fit, store, registry_ref, registry, ordinary = _accepted_inputs(root, protocol, registry_ref)
    ids = sorted(i for i, target in fit["targets"].items() if target["split"] == "SCREEN")
    check(len(ids) == protocol["split_sizes"]["SCREEN"], "Wrong SCREEN population")
    body = {
        "schema": SCHEMA, "registry": registry_ref, "fit_plan": plain(fit.reference),
        "suite": plain(fit["suite"]), "g0_gate": registry["g0_gate"],
        "dataset_fingerprint": fit["dataset_fingerprint"], "fit_identity": fit["fit_identity"],
        "categories": plain(fit["categories"]), "primary": plain(fit["primary"]),
        "ordinary": ordinary, "blocks": _inventory(fit), "screen_ids": ids,
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "generator": "PCG64",
                      "numpy": np.__version__, "quantile": "linear"},
        "e25_permutations": {"draws": PERMUTATION_DRAWS, "seed_recipe": "G9Q6-E25-association-v1"},
    }
    return dict(body, identity=fingerprint(body)), fit, store


def _publish(store, dataset, namespace, kind, body, arrays, upstream, producer):
    run = new_run_id()
    meta = {
        "kind": kind, "status": "passed", "run_id": run, "code_commit": producer["code_commit"],
        "storage_scope": producer["storage_scope"], "software_versions": producer["software_versions"],
        "dataset_fingerprint": dataset, "protocol_fingerprint": fingerprint(store.protocol),
        "model_fingerprint": None, "seeds": {"bootstrap": BOOTSTRAP_SEED},
        "upstream_artifacts": plain(upstream),
    }
    files = {"body.json": canonical_bytes(plain(body)) + b"\n"}
    files.update({name + ".npy": array_bytes(value) for name, value in arrays.items()})
    return Store(store.root, store.protocol).ref(publish(store.root, namespace + "/" + run, meta, files)
                                                 / "_COMPLETE.json")


def _draws(size):
    return np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(
        0, size, size=(BOOTSTRAP_DRAWS, size), dtype=np.int64)


@contract
def prepare_stage4(root, protocol, registry_ref, producer):
    """Authorize Stage 4 and publish its work inventory and shared image draws."""
    body, fit, store = _body_from_registry(root, protocol, registry_ref)
    # The coordinator creates common parents before separate Drive runtimes write.
    from .analysis import analysis_units
    preview = ScorePlan(body, {}, fit, store)
    for owner in (*DEPTHS, "e03"):
        (store.root / "predictions" / "stage4" / owner).mkdir(parents=True, exist_ok=True)
    for unit in analysis_units(preview).values():
        (store.root / "results" / "stage4" / "analysis" / unit["experiment"] /
         unit["owner"]).mkdir(parents=True, exist_ok=True)
    ref = _publish(store, body["dataset_fingerprint"], "cache/shared/stage4/plans", "stage4_plan",
                   body, {"bootstrap": _draws(len(body["screen_ids"]))},
                   [body["registry"], body["fit_plan"]], producer)
    return ScorePlan(body, ref, fit, store)


@contract
def load_stage4_plan(root, protocol, manifest):
    opening = Store(root, protocol)
    ref = opening.ref(manifest)
    body = opening.json(ref, "body.json", "stage4_plan")
    expected, fit, store = _body_from_registry(root, protocol, body["registry"])
    check(body == expected, "Unrecognized Stage 4 work inventory")
    store.marker(ref, "stage4_plan")
    return ScorePlan(body, ref, fit, store)


def load_bootstrap(plan):
    """Return the one frozen matrix, in plan.screen_ids order."""
    size = len(plan["screen_ids"])
    result = plan.store.array(plan.reference, "bootstrap.npy", "<i8", (BOOTSTRAP_DRAWS, size))
    check(np.array_equal(result, _draws(size)), "Stage 4 bootstrap recipe mismatch")
    return result


def scoring_blocks(plan, owner):
    check(owner in (*DEPTHS, "e03"), "Unknown Stage 4 scoring owner")
    return tuple(sorted((key for key, b in plan["blocks"].items() if b["owner"] == owner),
                        key=lambda key: (plan["blocks"][key]["priority"], key)))
