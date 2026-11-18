"""Shared scoring, pairing, support, and deterministic resampling helpers."""

import hashlib
from math import ceil

import numpy as np
from scipy.special import expit

from src.stage2.schema import check
from .plan import PERMUTATION_DRAWS

METRICS = ("auroc", "ap", "brier", "nll", "sensitivity", "specificity", "f1", "balanced_accuracy")


def probabilities(logits):
    logits = np.asarray(logits, dtype="f8")
    check(np.isfinite(logits).all(), "Nonfinite logits")
    return expit(logits)


def confidence_cutoff(p, nominal):
    """Source-only E19 cutoff, using the specified rank and retaining ties."""
    p = np.asarray(p, dtype="f8")
    check(p.ndim == 1 and nominal in (.5, .75, .9)
          and np.all((0 <= p) & (p <= 1)), "Invalid E19 source probabilities")
    if not len(p):
        return {"eligible": 0, "retained": 0, "cutoff": None}
    confidence = np.maximum(p, 1 - p)
    index = len(p) - ceil(nominal * len(p))
    cutoff = float(np.partition(confidence, index)[index])
    return {"eligible": len(p), "retained": int((confidence >= cutoff).sum()), "cutoff": cutoff}


def targets_for_prediction(plan, prediction):
    values = []
    for row in prediction.rows:
        target = plan.fit["targets"][row["image_id"]]
        check(target["split"] == row["split"] == prediction.split, "Cross-split target join")
        values.append(target["values"])
    known = np.array([[v is not None for v in row] for row in values], dtype=bool)
    y = np.array([[0 if v is None else v for v in row] for row in values], dtype=bool)
    return y, known


def paired_image_alignment(plan, *predictions, contexts=None):
    """Return sorted SCREEN IDs and per-arm row indices, including pooled-view selection."""
    check(bool(predictions), "At least one arm required")
    contexts = (None,) * len(predictions) if contexts is None else contexts
    check(len(contexts) == len(predictions), "Specify one view selector per arm")
    ids = tuple(plan["screen_ids"])
    aligned = []
    for pred, language in zip(predictions, contexts):
        check(pred.split == "SCREEN", "Pairing requires SCREEN predictions")
        positions = {}
        for i, row in enumerate(pred.rows):
            if language is not None and row["language"] != language:
                continue
            image_id = row["image_id"]
            check(image_id not in positions, "Select exactly one view per paired image")
            positions[image_id] = i
        check(set(positions) == set(ids), "Pairing population mismatch")
        aligned.append(np.array([positions[image_id] for image_id in ids], dtype=np.intp))
    return ids, tuple(aligned)


def _safe_ratio(numerator, denominator):
    numer, denom = np.broadcast_arrays(numerator, denominator)
    return np.divide(numer, denom, out=np.full(numer.shape, np.nan), where=denom > 0)


def _weighted_metrics(y, p, mass):
    """Score image multiplicities in batches, using one tied rank ordering."""
    desc = np.argsort(-p, kind="stable")
    y, p, mass = y[desc], p[desc], mass[:, desc]
    starts = np.r_[0, np.flatnonzero(np.diff(p)) + 1]
    pos_mass, neg_mass = mass * y, mass * ~y
    pos, neg = pos_mass.sum(axis=1), neg_mass.sum(axis=1)
    pos_ties = np.add.reduceat(pos_mass, starts, axis=1)
    neg_ties = np.add.reduceat(neg_mass, starts, axis=1)
    cumulative_pos = np.cumsum(pos_ties, axis=1)
    cumulative_neg = np.cumsum(neg_ties, axis=1)
    auc = _safe_ratio(np.sum(pos_ties * (neg[:, None] - cumulative_neg + .5 * neg_ties), axis=1), pos * neg)
    precision = np.nan_to_num(_safe_ratio(cumulative_pos, cumulative_pos + cumulative_neg), nan=0)
    ap = _safe_ratio(np.sum(precision * pos_ties, axis=1), pos)
    clipped = np.clip(p, 1e-15, 1 - 1e-15)
    nll = -(y * np.log(clipped) + ~y * np.log1p(-clipped))
    positive = p >= .5
    tp = pos_mass[:, positive].sum(axis=1)
    fp = neg_mass[:, positive].sum(axis=1)
    sensitivity, specificity = _safe_ratio(tp, pos), _safe_ratio(neg - fp, neg)
    return {
        "auroc": auc, "ap": ap, "brier": _safe_ratio(mass @ ((p - y) ** 2), pos + neg),
        "nll": _safe_ratio(mass @ nll, pos + neg), "sensitivity": sensitivity, "specificity": specificity,
        "f1": _safe_ratio(2 * tp, tp + fp + pos), "balanced_accuracy": (sensitivity + specificity) / 2,
    }


def _binary_inputs(y, p, known):
    y, p, known = np.asarray(y), np.asarray(p, dtype="f8"), np.asarray(known)
    check(y.ndim == 1 and y.shape == p.shape == known.shape and known.dtype == bool
          and np.all((y[known] == 0) | (y[known] == 1))
          and np.all((0 <= p[known]) & (p[known] <= 1)), "Invalid binary observations")
    return y.astype(bool), p, known


def binary_metrics(y, p, known):
    """Return point statistics and support. Mathematically undefined values are None."""
    y, p, known = _binary_inputs(y, p, known)
    den = int(known.sum())
    positive = int(y[known].sum())
    stats = dict.fromkeys(METRICS)
    if den:
        stats = {key: float(value[0]) if np.isfinite(value[0]) else None
                 for key, value in _weighted_metrics(y[known], p[known], np.ones((1, den))).items()}
    return {"metrics": stats, "known": den, "positive": positive, "negative": den - positive}


def bootstrap_counts(draws, size):
    """Convert shared image draw indices into one reusable multiplicity matrix."""
    d = np.asarray(draws)
    check(d.ndim == 2 and d.shape[1] == size and size > 0 and np.issubdtype(d.dtype, np.integer)
          and np.all((0 <= d) & (d < size)), "Invalid image bootstrap draws")
    counts = np.zeros(d.shape, dtype=np.int64)
    np.add.at(counts, (np.arange(d.shape[0])[:, None], d), 1)
    counts.flags.writeable = False
    return counts


def bootstrap_binary_metrics(y, p, known, counts):
    """Return per-draw float arrays. NaN denotes an undefined draw."""
    y, p, known = _binary_inputs(y, p, known)
    counts = np.asarray(counts)
    check(counts.ndim == 2 and counts.shape[1] == len(y)
          and np.issubdtype(counts.dtype, np.integer) and np.all(counts >= 0), "Invalid multiplicities")
    if not known.any():
        return {key: np.full(counts.shape[0], np.nan) for key in METRICS}
    return _weighted_metrics(y[known], p[known], counts[:, known])


def bootstrap_interval(draw_values):
    values = np.asarray(draw_values, dtype="f8")
    check(values.ndim == 1 and not np.isinf(values).any(), "Invalid bootstrap summary input")
    finite = values[np.isfinite(values)]
    lower, upper = np.quantile(finite, [.025, .975], method="linear") if finite.size else (None, None)
    return {"lower": None if lower is None else float(lower), "upper": None if upper is None else float(upper),
            "valid_draws": int(finite.size), "total_draws": int(values.size)}


def fixed_macro(values, primary_indices, *, axis=-1):
    """Equal-weight fixed cohort. A missing term propagates through the mean."""
    values = np.asarray(values, dtype="f8")
    check(len(primary_indices) > 0 and len(set(primary_indices)) == len(primary_indices), "Invalid macro cohort")
    return np.mean(np.take(values, primary_indices, axis=axis), axis=axis)


def support_status(value, positive, negative, minimum):
    if value is None:
        return "undefined"
    return "ok" if min(positive, negative) >= minimum else "descriptive_low_support"


def association_orders(category_id, label, image_ids):
    """Return frozen E25 permutation indices for one sorted, unique image stratum."""
    ids = tuple(image_ids)
    check(ids == tuple(sorted(set(ids))) and label in (0, 1), "Invalid E25 stratum")
    key = f"G9Q6-E25-association-v1|{category_id}|{int(label)}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    rng = np.random.Generator(np.random.PCG64(seed))
    return np.stack([rng.permutation(len(ids)) for _ in range(PERMUTATION_DRAWS)])
