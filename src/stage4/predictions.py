"""Authorized numeric scoring, canonical prediction blocks and readers."""

from dataclasses import dataclass

import numpy as np

from src.common.contracts import fingerprint
from src.stage2.schema import check, contract, frozen, plain
from src.stage3.data import _read_verified_group
from src.stage3.engine import transform
from src.stage3.readouts import _body, expected_rows, load_statistics, weights
from .plan import _publish

PREDICTION_KIND = "stage4_prediction_block"


@dataclass(frozen=True)
class Prediction:
    reference: dict
    route: dict
    split: str
    rows: tuple
    logits: np.ndarray
    estimable: np.ndarray
    prevalence: np.ndarray


def _owner_block(plan, owner, block_id):
    block = plan["blocks"].get(block_id)
    check(block is not None and block["owner"] == owner, "Foreign scoring block")
    check(block["split"] in ("DEV-C", "SCREEN"), "Closed Stage 4 scoring split")
    return block


def _selected_parameters(plan, gid):
    fit, store = plan.fit, plan.store
    selected = plan["ordinary"][gid]
    ref = selected["candidate"]
    body, _ = _body(fit, ref, "candidate", store)
    check(body["group_id"] == gid and body["assignment_id"] == "true"
          and body["C"] == selected["C"], "Wrong frozen candidate")
    rows = expected_rows(fit, gid, "FIT", store)
    check(body["fit_rows"] == fingerprint(rows), "Wrong source FIT row identity")
    group = fit["groups"][gid]
    dimension = group["dimension"]
    category_count = len(plan["categories"])
    valid = store.array(ref, "estimable.npy", "|b1", (category_count,))
    coef = store.array(ref, "coefficients.npy", "<f8", (category_count, dimension))
    intercept = store.array(ref, "intercepts.npy", "<f8", (category_count,))
    check(len(body["records"]) == category_count, "Wrong fit status coverage")
    for idx, record in enumerate(body["records"]):
        if valid[idx]:
            check(record["status"] == "fitted" and record["attempts"]
                  and record["attempts"][-1]["converged"], "Invalid fitted head")
        else:
            check(record["status"] == "unestimable" and not coef[idx].any()
                  and intercept[idx] == 0, "Invalid unestimable head padding")
    stats = load_statistics(fit, body["statistics"], gid, rows, store)
    row_values = [fit["targets"][row["image_id"]]["values"] for row in rows]
    row_weights = weights(rows, group)
    known = np.array([[v is not None for v in row] for row in row_values], dtype=bool)
    y = np.array([[v if v is not None else 0 for v in row] for row in row_values], dtype="f8")
    den = row_weights @ known
    prior = np.divide(row_weights @ y, den, out=np.zeros(category_count), where=den > 0)
    return coef, intercept, valid, prior, stats


def _provenance(plan, block):
    target = plan.fit["groups"][block["target_group"]]
    split = block["split"]
    return {
        "target_release": plain(target["releases"][split]),
        "sources": [plain(plan["ordinary"][r["source_group"]]) for r in block["routes"]],
    }


@contract
def score_and_publish_block(plan, owner, block_id, producer):
    """Read one authorized target matrix and score all planned heads against it."""
    block = _owner_block(plan, owner, block_id)
    gid, split = block["target_group"], block["split"]
    rows, matrix = _read_verified_group(plan.fit, plan.store, gid, split)
    logits, valid, priors, scalers = [], [], [], []
    for route in block["routes"]:
        coef, intercept, estimable, prevalence, stats = _selected_parameters(plan, route["source_group"])
        result = np.asarray(transform(matrix, stats) @ coef.T) + intercept
        check(np.isfinite(result).all(), "Nonfinite Stage 4 logits")
        logits.append(result)
        valid.append(estimable)
        priors.append(prevalence)
        scalers.append(plain(stats["reference"]))
    provenance = _provenance(plan, block)
    body = {
        "schema": "stage4.prediction.v1", "plan": plain(plan.reference),
        "plan_identity": plan["identity"], "block_id": block_id,
        "block": plain(block), "rows": plain(rows), "provenance": provenance,
        "scalers": scalers,
    }
    arrays = {"logits": np.asarray(logits, dtype="<f8"), "estimable": np.asarray(valid, dtype=bool),
              "prevalence": np.asarray(priors, dtype="<f8")}
    refs = [plan.reference, provenance["target_release"]]
    refs += [r["selection"] for r in provenance["sources"]]
    refs += [r["candidate"] for r in provenance["sources"]] + scalers
    return _publish(plan.store, plan["dataset_fingerprint"], "predictions/stage4/" + owner,
                    PREDICTION_KIND, body, arrays, refs, producer)


@contract
def load_prediction_block(plan, reference, *, expected_block=None):
    """Authenticate a canonical block and return route-level read-only views."""
    store = plan.store
    ref = store.ref(reference)
    body = store.json(ref, "body.json", PREDICTION_KIND)
    block_id = body["block_id"]
    check(expected_block is None or expected_block == block_id, "Wrong prediction block")
    block = plan["blocks"].get(block_id)
    check(block is not None and body["block"] == plain(block)
          and body["schema"] == "stage4.prediction.v1" and body["plan"] == plain(plan.reference)
          and body["plan_identity"] == plan["identity"] and body["provenance"] == _provenance(plan, block),
          "Prediction lineage mismatch")
    gid, split = block["target_group"], block["split"]
    rows = expected_rows(plan.fit, gid, split, store)
    check(body["rows"] == rows, "Prediction row order, mask join, or view mismatch")
    expected_scalers = []
    expected_valid = []
    for route in block["routes"]:
        source = route["source_group"]
        selected = plan["ordinary"][source]
        candidate, _ = _body(plan.fit, selected["candidate"], "candidate", store)
        check(candidate["group_id"] == source and candidate["assignment_id"] == "true"
              and candidate["C"] == selected["C"], "Prediction has wrong ordinary candidate")
        expected_scalers.append(candidate["statistics"])
        expected_valid.append([r["status"] == "fitted" for r in candidate["records"]])
    check(body["scalers"] == expected_scalers, "Prediction scaler lineage mismatch")
    count, cats = len(block["routes"]), len(plan["categories"])
    logits = store.array(ref, "logits.npy", "<f8", (count, len(rows), cats))
    valid = store.array(ref, "estimable.npy", "|b1", (count, cats))
    prevalence = store.array(ref, "prevalence.npy", "<f8", (count, cats))
    check(np.array_equal(valid, expected_valid) and np.all((0 <= prevalence) & (prevalence <= 1))
          and not np.any(logits * (~valid[:, None, :])), "Invalid prediction estimability or prevalence")
    rows = frozen(rows)
    return {r["route_id"]: Prediction(frozen(ref), r, split, rows, logits[i], valid[i], prevalence[i])
            for i, r in enumerate(block["routes"])}


@contract
def prediction_index(plan, references, *, required_blocks=()):
    """Read explicit block references once per consumer and verify requested coverage."""
    blocks = {}
    for reference in references:
        ref = plan.store.ref(reference)
        block_id = plan.store.json(ref, "body.json", PREDICTION_KIND)["block_id"]
        check(block_id not in blocks, "Duplicate prediction block")
        blocks[block_id] = load_prediction_block(plan, ref, expected_block=block_id)
    check(set(required_blocks) <= set(blocks), "Missing required prediction blocks")
    return blocks
