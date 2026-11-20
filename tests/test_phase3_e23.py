"""Unit, contract, CLI, notebook, and end-to-end fixture tests for Phase 3 Stage B3 Worker 2 (`run_e23`)."""

import ast
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

try:
    import numpy as np
    import scipy  # noqa: F401

    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

from src.common.contracts import ContractError, LANGUAGES, ModelInput, TargetRecord, canonical_bytes, fingerprint
from src.common.storage import file_sha256
from src.phase2.interfaces import Phase1Binding
from src.phase3.extract_e23 import _expected_authentic_parity_rows, _expected_swapped_chunk_rows
from src.phase3.interfaces import (
    CERTIFIED_STAGE_B1_GPU_PLAN,
    CERTIFIED_STAGE_B2_RELEASES,
    KIND_ANALYSIS,
    KIND_E23_CHUNK,
    KIND_E23_RELEASE,
    KIND_GPU_PLAN,
    KIND_PREDICTION,
    KIND_PROBE,
    KIND_PROBE_PAYLOAD,
    MODELS,
    PHASE3_GPU_DECISION_RANGES,
    PHASE3_GPU_VERSION,
    PHASE3_PARENT_DIRS,
    POOLINGS,
    REPO,
    SCHEMA_ANALYSIS,
    SCHEMA_E23_FEATURES,
    SCHEMA_GPU_PLAN,
    SCHEMA_PREDICTION,
    SCHEMA_PROBE,
    build_e23_pair_plan,
    frozen,
    init_phase3_dirs,
    load_phase3_artifact,
    plain,
    publish_phase3,
    require_open_split,
)
from src.phase3.run_e23 import (
    AUTHENTIC_PARITY_TOL,
    E23_CONDITION,
    E23_EXPERIMENT,
    E23_LINEAR_SWAP_METRICS,
    E23_MACRO_COHORTS,
    E23_PREDICTIONS_NAMESPACE,
    E23_RATIO_SWAP_METRICS,
    E23_RESULTS_NAMESPACE,
    E23_SPLIT,
    E23_SWAP_METRICS,
    SUPERVISION_DIAGONAL,
    SUPERVISION_OFF_DIAGONAL,
    _parse_ref_arg,
    _require_numpy,
    build_e23_primary_cohort_audit,
    compute_e23_category_point_and_draws,
    compute_e23_grounding_fractions,
    compute_e23_pair_shifts,
    e23_category_support_status,
    evaluate_e23_decision_summary,
    run_e23,
)
from typing import Any

from src.stage2.schema import check, json_bytes, jsonl


def _load_protocol():
    return json.loads((REPO / "configs/protocol.json").read_text(encoding="utf-8"))


def _fake_producer():
    return {
        "code_commit": "0" * 40,
        "storage_scope": "local-development",
        "software_versions": {"python": "3.12.0", "numpy": "2.1.3"},
    }


def _make_interval_entry(val, lo, hi, valid_draws=2000, total_draws=2000, status="ok"):
    return {
        "value": val,
        "interval": {
            "lower": lo,
            "upper": hi,
            "valid_draws": int(valid_draws),
            "total_draws": int(total_draws),
        },
        "status": status,
        "reason": None if status == "ok" else "low_support",
    }


# ---------------------------------------------------------------------------
# Pure-stdlib NPY + array + Stage 3/Stage 4 shim for Windows Python 3.9 without numpy
# ---------------------------------------------------------------------------


def _canonical_dtype(dt):
    s = str(dt)
    if s in ("<f4", "float32", "f4"):
        return "<f4"
    if s in ("<f8", "float64", "f8", "<class 'float'>", "float"):
        return "<f8"
    if s in ("|b1", "bool", "<class 'bool'>"):
        return "|b1"
    if s in ("<i8", "int64", "i8", "intp", "<class 'int'>", "int"):
        return "<i8"
    return s


class _DtypeObj:
    def __init__(self, dt):
        self.str = _canonical_dtype(dt)
        if self.str in ("<f4", "<f8"):
            self.kind = "f"
        elif self.str == "|b1":
            self.kind = "b"
        else:
            self.kind = "i"

    def __eq__(self, other):
        if isinstance(other, _DtypeObj):
            return self.str == other.str
        return self.str == _canonical_dtype(other)

    def __repr__(self):
        return f"dtype({self.str!r})"


class _FlagsObj:
    def __init__(self, c_contiguous=True, writeable=True):
        self.c_contiguous = c_contiguous
        self.writeable = writeable


def _infer_shape_and_flat(data, kind):
    def _cast(v):
        if kind == "b":
            return bool(v)
        if kind == "i":
            return int(v)
        return float(v)

    if isinstance(data, _ShimArray):
        return data.shape, [_cast(v) for v in data._flat]

    if not isinstance(data, (list, tuple)):
        return (), [_cast(data)]

    def _walk(node):
        if isinstance(node, _ShimArray):
            node = node.tolist()
        if not isinstance(node, (list, tuple)):
            return (), [_cast(node)]
        if len(node) == 0:
            return (0,), []
        sub_shapes = []
        flat = []
        for item in node:
            sh, fl = _walk(item)
            sub_shapes.append(sh)
            flat.extend(fl)
        first_sh = sub_shapes[0]
        if any(sh != first_sh for sh in sub_shapes):
            raise ValueError("Ragged array in _ShimArray")
        return (len(node),) + first_sh, flat

    return _walk(data)


def _strides_for_shape(shape):
    if not shape:
        return ()
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * max(1, shape[i + 1])
    return tuple(strides)


def _unflatten_nested(flat, shape):
    if len(shape) == 0:
        return flat[0]
    if len(shape) == 1:
        return list(flat[: shape[0]])
    step = 1
    for d in shape[1:]:
        step *= d
    return [_unflatten_nested(flat[i * step : (i + 1) * step], shape[1:]) for i in range(shape[0])]


def _broadcast_shapes(s1, s2):
    n = max(len(s1), len(s2))
    p1 = (1,) * (n - len(s1)) + s1
    p2 = (1,) * (n - len(s2)) + s2
    out = []
    for d1, d2 in zip(p1, p2):
        if d1 == d2:
            out.append(d1)
        elif d1 == 1:
            out.append(d2)
        elif d2 == 1:
            out.append(d1)
        else:
            raise ValueError(f"Cannot broadcast shapes {s1} and {s2}")
    return tuple(out), p1, p2


def _iter_coords(shape):
    if not shape:
        yield ()
        return
    if any(d == 0 for d in shape):
        return
    ranges = [range(d) for d in shape]
    import itertools

    for coord in itertools.product(*ranges):
        yield coord


def _coord_to_flat_idx(coord, padded_shape, orig_shape):
    if not orig_shape:
        return 0
    strides = _strides_for_shape( padded_shape )
    # compute actual strides for padded_shape where size==1 has stride 0
    real_strides = [1] * len(padded_shape)
    for i in range(len(padded_shape) - 2, -1, -1):
        real_strides[i] = real_strides[i + 1] * max(1, padded_shape[i + 1])
    idx = 0
    for c, d, st in zip(coord, padded_shape, real_strides):
        eff_c = 0 if d == 1 else c
        idx += eff_c * st
    return idx


class _ShimArray:
    """N-dimensional pure-Python array supporting NPY serialization, broadcasting, slicing, and matmul."""

    def __init__(self, data, dtype="<f8", *, _shape=None, _flat=None):
        self.dtype = _DtypeObj(dtype)
        if _shape is not None and _flat is not None:
            self.shape = tuple(int(x) for x in _shape)
            if self.dtype.kind == "b":
                self._flat = [bool(x) for x in _flat]
            elif self.dtype.kind == "i":
                self._flat = [int(x) for x in _flat]
            else:
                self._flat = [float(x) for x in _flat]
        else:
            sh, fl = _infer_shape_and_flat(data, self.dtype.kind)
            self.shape = sh
            self._flat = fl
        self.ndim = len(self.shape)
        self.size = len(self._flat)
        self.flags = _FlagsObj(True, True)

    def copy(self):
        return _ShimArray(None, dtype=self.dtype.str, _shape=self.shape, _flat=list(self._flat))

    def astype(self, dt, copy=True):
        return _ShimArray(None, dtype=dt, _shape=self.shape, _flat=list(self._flat))

    def tolist(self):
        return _unflatten_nested(self._flat, self.shape)

    def tobytes(self):
        if not self._flat:
            return b""
        if self.dtype.str == "<f4":
            return struct.pack(f"<{len(self._flat)}f", *self._flat)
        if self.dtype.str == "<f8":
            return struct.pack(f"<{len(self._flat)}d", *self._flat)
        if self.dtype.str == "<i8":
            return struct.pack(f"<{len(self._flat)}q", *self._flat)
        if self.dtype.str == "|b1":
            return struct.pack(f"<{len(self._flat)}?", *self._flat)
        raise ValueError(f"Unsupported dtype for tobytes: {self.dtype.str}")

    def all(self):
        return all(bool(x) for x in self._flat)

    def any(self):
        return any(bool(x) for x in self._flat)

    def sum(self, axis=None):
        return _shim_sum(self, axis=axis)

    @property
    def T(self):
        if self.ndim != 2:
            raise ValueError("Transpose only supported for 2D arrays in shim")
        r, c = self.shape
        out = []
        for j in range(c):
            for i in range(r):
                out.append(self._flat[i * c + j])
        return _ShimArray(None, dtype=self.dtype.str, _shape=(c, r), _flat=out)

    def __len__(self):
        return self.shape[0] if self.shape else 0

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __invert__(self):
        return _ShimArray(None, dtype="|b1", _shape=self.shape, _flat=[not bool(x) for x in self._flat])

    def __neg__(self):
        return _ShimArray(None, dtype=self.dtype.str, _shape=self.shape, _flat=[-x for x in self._flat])

    def _binop(self, other, op, out_dtype=None):
        other_arr = other if isinstance(other, _ShimArray) else _ShimArray(other, dtype=self.dtype.str)
        out_shape, p1, p2 = _broadcast_shapes(self.shape, other_arr.shape)
        if out_dtype is None:
            if self.dtype.str == "<f8" or other_arr.dtype.str == "<f8":
                out_dtype = "<f8"
            elif self.dtype.str == "<f4" or other_arr.dtype.str == "<f4":
                out_dtype = "<f4"
            elif self.dtype.str == "<i8" or other_arr.dtype.str == "<i8":
                out_dtype = "<i8"
            else:
                out_dtype = "|b1"
        flat = []
        for coord in _iter_coords(out_shape):
            i1 = _coord_to_flat_idx(coord, p1, self.shape)
            i2 = _coord_to_flat_idx(coord, p2, other_arr.shape)
            flat.append(op(self._flat[i1], other_arr._flat[i2]))
        if not out_shape:
            return flat[0]
        return _ShimArray(None, dtype=out_dtype, _shape=out_shape, _flat=flat)

    def __add__(self, other):
        return self._binop(other, lambda a, b: a + b)

    def __radd__(self, other):
        return self._binop(other, lambda a, b: b + a)

    def __sub__(self, other):
        return self._binop(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return self._binop(other, lambda a, b: b - a)

    def __mul__(self, other):
        return self._binop(
            other,
            lambda a, b: (float(a) * float(b))
            if (self.dtype.kind == "f" or (isinstance(other, _ShimArray) and other.dtype.kind == "f") or isinstance(other, float))
            else (int(a) * int(b)),
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._binop(other, lambda a, b: float(a) / float(b) if b != 0 else float("nan"), out_dtype="<f8")

    def __gt__(self, other):
        return self._binop(other, lambda a, b: a > b, out_dtype="|b1")

    def __ge__(self, other):
        return self._binop(other, lambda a, b: a >= b, out_dtype="|b1")

    def __lt__(self, other):
        return self._binop(other, lambda a, b: a < b, out_dtype="|b1")

    def __le__(self, other):
        return self._binop(other, lambda a, b: a <= b, out_dtype="|b1")

    def __and__(self, other):
        return self._binop(other, lambda a, b: bool(a) and bool(b), out_dtype="|b1")

    def __or__(self, other):
        return self._binop(other, lambda a, b: bool(a) or bool(b), out_dtype="|b1")

    def __matmul__(self, other):
        other_arr = other if isinstance(other, _ShimArray) else _ShimArray(other, dtype="<f8")
        if self.ndim == 2 and other_arr.ndim == 2:
            m, k1 = self.shape
            k2, n = other_arr.shape
            if k1 != k2:
                raise ValueError(f"Matmul shape mismatch: {self.shape} @ {other_arr.shape}")
            out = []
            for i in range(m):
                row_off = i * k1
                for j in range(n):
                    s = 0.0
                    for k in range(k1):
                        s += float(self._flat[row_off + k]) * float(other_arr._flat[k * n + j])
                    out.append(s)
            return _ShimArray(None, dtype="<f8", _shape=(m, n), _flat=out)
        if self.ndim == 2 and other_arr.ndim == 1:
            m, k1 = self.shape
            (k2,) = other_arr.shape
            if k1 != k2:
                raise ValueError(f"Matmul shape mismatch: {self.shape} @ {other_arr.shape}")
            out = []
            for i in range(m):
                row_off = i * k1
                s = 0.0
                for k in range(k1):
                    s += float(self._flat[row_off + k]) * float(other_arr._flat[k])
                out.append(s)
            return _ShimArray(None, dtype="<f8", _shape=(m,), _flat=out)
        raise ValueError(f"Unsupported matmul dims: {self.shape} @ {other_arr.shape}")

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        # Handle None (np.newaxis) and integer/slice/list/boolean indexing
        expanded_shape = []
        axis_specs = []
        src_axis = 0
        for k in key:
            if k is None:
                axis_specs.append(("newaxis", [0]))
                expanded_shape.append(1)
            else:
                d = self.shape[src_axis]
                src_axis += 1
                if isinstance(k, int):
                    idx = k if k >= 0 else d + k
                    if not (0 <= idx < d):
                        raise IndexError(f"Index {k} out of bounds for axis with size {d}")
                    axis_specs.append(("scalar", [idx]))
                elif isinstance(k, slice):
                    idxs = list(range(*k.indices(d)))
                    axis_specs.append(("list", idxs))
                    expanded_shape.append(len(idxs))
                elif isinstance(k, _ShimArray) and k.dtype.kind == "b":
                    if k.ndim != 1 or len(k) != d:
                        raise IndexError("Boolean index shape mismatch")
                    idxs = [i for i, flag in enumerate(k._flat) if flag]
                    axis_specs.append(("list", idxs))
                    expanded_shape.append(len(idxs))
                elif isinstance(k, (_ShimArray, list, tuple)):
                    raw_list = k._flat if isinstance(k, _ShimArray) else list(k)
                    idxs = [int(x) if int(x) >= 0 else d + int(x) for x in raw_list]
                    axis_specs.append(("list", idxs))
                    expanded_shape.append(len(idxs))
                else:
                    raise TypeError(f"Unsupported index type: {type(k)}")
        while src_axis < self.ndim:
            d = self.shape[src_axis]
            src_axis += 1
            idxs = list(range(d))
            axis_specs.append(("list", idxs))
            expanded_shape.append(d)

        src_strides = _strides_for_shape(self.shape)
        import itertools

        idx_lists = [spec[1] for spec in axis_specs]
        out_flat = []
        for combo in itertools.product(*idx_lists):
            src_coords = [c for c, (kind, _) in zip(combo, axis_specs) if kind != "newaxis"]
            flat_pos = sum(c * st for c, st in zip(src_coords, src_strides))
            out_flat.append(self._flat[flat_pos])

        if not expanded_shape:
            return out_flat[0]
        return _ShimArray(None, dtype=self.dtype.str, _shape=tuple(expanded_shape), _flat=out_flat)


def _shim_sum(arr, axis=None):
    a = arr if isinstance(arr, _ShimArray) else _ShimArray(arr)
    if axis is None:
        return sum(a._flat)
    if axis < 0:
        axis += a.ndim
    out_shape = a.shape[:axis] + a.shape[axis + 1 :]
    out_dtype = "<f8" if a.dtype.kind == "f" else "<i8"
    if not out_shape:
        return sum(a._flat)
    src_strides = _strides_for_shape(a.shape)
    out_flat = []
    for coord in _iter_coords(out_shape):
        s = 0.0 if out_dtype == "<f8" else 0
        for k in range(a.shape[axis]):
            full_coord = coord[:axis] + (k,) + coord[axis:]
            pos = sum(c * st for c, st in zip(full_coord, src_strides))
            s += a._flat[pos]
        out_flat.append(s)
    return _ShimArray(None, dtype=out_dtype, _shape=out_shape, _flat=out_flat)


def _shim_mean(arr, axis=None):
    a = arr if isinstance(arr, _ShimArray) else _ShimArray(arr, dtype="<f8")
    if axis is None:
        return float(sum(float(x) for x in a._flat) / len(a._flat))
    if axis < 0:
        axis += a.ndim
    denom = float(a.shape[axis])
    s = _shim_sum(a.astype("<f8"), axis=axis)
    if isinstance(s, _ShimArray):
        return s / denom
    return float(s) / denom


def _make_stdlib_scientific_modules():
    """Build stdlib stand-in modules for `numpy`, `src.stage3.engine`, `src.stage4.*` when numpy is absent."""
    np_mod = types.ModuleType("numpy")
    np_mod.ndarray = _ShimArray
    np_mod.dtype = _DtypeObj
    np_mod.float64 = "<f8"
    np_mod.float32 = "<f4"
    np_mod.int64 = "<i8"
    np_mod.intp = "<i8"
    np_mod.integer = int
    np_mod.nan = float("nan")

    def asarray(val, dtype=None):
        if isinstance(val, _ShimArray):
            return val if dtype is None or val.dtype == _DtypeObj(dtype) else val.astype(dtype)
        if dtype is None:
            # Infer dtype
            dt = "<f8"
            if isinstance(val, (list, tuple)) and val:
                first = val[0]
                while isinstance(first, (list, tuple)) and first:
                    first = first[0]
                if isinstance(first, bool):
                    dt = "|b1"
                elif isinstance(first, int):
                    dt = "<i8"
            return _ShimArray(val, dtype=dt)
        return _ShimArray(val, dtype=dtype)

    def array(val, dtype=None):
        return asarray(val, dtype=dtype).copy()

    def ascontiguousarray(val, dtype=None):
        return asarray(val, dtype=dtype)

    def full(shape, fill_value, dtype="<f8"):
        sh = (shape,) if isinstance(shape, int) else tuple(shape)
        count = 1
        for d in sh:
            count *= d
        return _ShimArray(None, dtype=dtype, _shape=sh, _flat=[fill_value] * count)

    def zeros(shape, dtype="<f8"):
        return full(shape, False if _canonical_dtype(dtype) == "|b1" else 0, dtype=dtype)

    def ones(shape, dtype="<f8"):
        return full(shape, True if _canonical_dtype(dtype) == "|b1" else 1, dtype=dtype)

    def empty(shape, dtype="<f8"):
        return zeros(shape, dtype=dtype)

    def arange(n, dtype="<i8"):
        return _ShimArray(list(range(int(n))), dtype=dtype)

    def stack(items, axis=0):
        arrs = [asarray(x) for x in items]
        if axis != 0:
            raise ValueError("Shim stack only supports axis=0")
        sh = (len(arrs),) + arrs[0].shape
        flat = []
        for a in arrs:
            flat.extend(a._flat)
        return _ShimArray(None, dtype=arrs[0].dtype.str, _shape=sh, _flat=flat)

    def concatenate(items, axis=0):
        arrs = [asarray(x) for x in items]
        if axis != 0:
            raise ValueError("Shim concatenate only supports axis=0")
        total_0 = sum(a.shape[0] for a in arrs)
        sh = (total_0,) + arrs[0].shape[1:]
        flat = []
        for a in arrs:
            flat.extend(a._flat)
        return _ShimArray(None, dtype=arrs[0].dtype.str, _shape=sh, _flat=flat)

    def take(arr, indices, axis=0):
        a = asarray(arr)
        if axis < 0:
            axis += a.ndim
        key = [slice(None)] * a.ndim
        key[axis] = list(indices)
        return a[tuple(key)]

    def abs_fn(val):
        a = asarray(val)
        return _ShimArray(None, dtype=a.dtype.str, _shape=a.shape, _flat=[abs(x) for x in a._flat])

    def max_fn(val):
        a = asarray(val)
        return max(a._flat) if a._flat else 0.0

    def isfinite(val):
        if isinstance(val, (int, float)):
            return math.isfinite(float(val))
        a = asarray(val)
        return _ShimArray(None, dtype="|b1", _shape=a.shape, _flat=[math.isfinite(float(x)) for x in a._flat])

    def isinf(val):
        a = asarray(val)
        return _ShimArray(None, dtype="|b1", _shape=a.shape, _flat=[math.isinf(float(x)) for x in a._flat])

    def any_fn(val):
        a = asarray(val)
        return any(bool(x) for x in a._flat)

    def all_fn(val):
        a = asarray(val)
        return all(bool(x) for x in a._flat)

    def array_equal(a, b):
        aa = asarray(a)
        bb = asarray(b)
        return aa.shape == bb.shape and aa._flat == bb._flat

    def allclose(a, b, rtol=1e-5, atol=1e-8):
        aa = asarray(a)
        bb = asarray(b)
        return aa.shape == bb.shape and all(
            abs(float(x) - float(y)) <= atol + rtol * abs(float(y)) for x, y in zip(aa._flat, bb._flat)
        )

    def divide(num, den, out=None, where=True):
        n_arr = asarray(num, dtype="<f8")
        d_arr = asarray(den, dtype="<f8")
        w_arr = asarray(where, dtype="|b1") if not isinstance(where, bool) else full(n_arr.shape, where, dtype="|b1")
        out_arr = out.copy() if out is not None else full(n_arr.shape, float("nan"), dtype="<f8")
        for i in range(out_arr.size):
            if w_arr._flat[i] and d_arr._flat[i] != 0.0:
                out_arr._flat[i] = n_arr._flat[i] / d_arr._flat[i]
        return out_arr

    def where_fn(cond, x, y):
        c_arr = asarray(cond, dtype="|b1")
        x_arr = x if isinstance(x, _ShimArray) else full(c_arr.shape, x, dtype="<f8")
        y_arr = y if isinstance(y, _ShimArray) else full(c_arr.shape, y, dtype="<f8")
        flat = [x_arr._flat[i] if c_arr._flat[i] else y_arr._flat[i] for i in range(c_arr.size)]
        return _ShimArray(None, dtype="<f8", _shape=c_arr.shape, _flat=flat)

    def quantile(arr, qs, method="linear"):
        a = asarray(arr, dtype="<f8")
        vals = sorted(float(x) for x in a._flat)
        n = len(vals)
        out = []
        for q in qs:
            pos = q * (n - 1)
            lo = int(math.floor(pos))
            hi = int(math.ceil(pos))
            frac = pos - lo
            out.append(vals[lo] + frac * (vals[hi] - vals[lo]))
        return tuple(out)

    def save(buf, arr, allow_pickle=False):
        a = asarray(arr)
        header = repr({
            "descr": a.dtype.str,
            "fortran_order": False,
            "shape": a.shape,
        }).encode("latin1")
        preamble_len = 10
        total = preamble_len + len(header) + 1
        pad = (16 - (total % 16)) % 16
        header_bytes = header + (b" " * pad) + b"\n"
        if hasattr(buf, "write"):
            target = buf
            close_after = False
        else:
            target = open(buf, "wb")
            close_after = True
        try:
            target.write(b"\x93NUMPY\x01\x00")
            target.write(struct.pack("<H", len(header_bytes)))
            target.write(header_bytes)
            target.write(a.tobytes())
        finally:
            if close_after:
                target.close()

    def load(path, allow_pickle=False):
        raw = Path(path).read_bytes()
        if not raw.startswith(b"\x93NUMPY"):
            raise ValueError("Not a valid NPY file")
        major = raw[6]
        if major == 1:
            hlen = struct.unpack("<H", raw[8:10])[0]
            offset = 10 + hlen
            hdr_text = raw[10:offset].decode("latin1").strip()
        else:
            hlen = struct.unpack("<I", raw[8:12])[0]
            offset = 12 + hlen
            hdr_text = raw[12:offset].decode("latin1").strip()
        meta = ast.literal_eval(hdr_text)
        shape = tuple(meta["shape"])
        descr = _canonical_dtype(meta["descr"])
        count = 1
        for d in shape:
            count *= d
        payload = raw[offset:]
        if descr == "<f4":
            vals = list(struct.unpack(f"<{count}f", payload[: 4 * count]))
        elif descr == "<f8":
            vals = list(struct.unpack(f"<{count}d", payload[: 8 * count]))
        elif descr == "<i8":
            vals = list(struct.unpack(f"<{count}q", payload[: 8 * count]))
        elif descr == "|b1":
            vals = list(struct.unpack(f"<{count}?", payload[:count]))
        else:
            raise ValueError(f"Unsupported NPY descr in shim: {descr}")
        return _ShimArray(None, dtype=descr, _shape=shape, _flat=vals)

    np_mod.asarray = asarray
    np_mod.array = array
    np_mod.ascontiguousarray = ascontiguousarray
    np_mod.full = full
    np_mod.zeros = zeros
    np_mod.ones = ones
    np_mod.empty = empty
    np_mod.arange = arange
    np_mod.stack = stack
    np_mod.concatenate = concatenate
    np_mod.take = take
    np_mod.abs = abs_fn
    np_mod.max = max_fn
    np_mod.sum = _shim_sum
    np_mod.mean = _shim_mean
    np_mod.isfinite = isfinite
    np_mod.isinf = isinf
    np_mod.any = any_fn
    np_mod.all = all_fn
    np_mod.array_equal = array_equal
    np_mod.allclose = allclose
    np_mod.divide = divide
    np_mod.where = where_fn
    np_mod.quantile = quantile
    np_mod.save = save
    np_mod.load = load

    # Stage 3 engine shim
    s3_engine = types.ModuleType("src.stage3.engine")

    def transform(x, stats):
        x_arr = asarray(x, dtype="<f8")
        check(x_arr.shape[1] == len(stats["scale"]), "Feature dimension mismatch")
        check(stats["center"], "Dense center policy mismatch")
        return (x_arr - asarray(stats["mean"], dtype="<f8")) / asarray(stats["scale"], dtype="<f8")

    s3_engine.transform = transform

    # Stage 4 metrics shim
    s4_metrics = types.ModuleType("src.stage4.metrics")

    def probabilities(logits):
        z = asarray(logits, dtype="<f8")
        check(isfinite(z).all(), "Nonfinite logits")
        flat = [1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, float(v))))) for v in z._flat]
        return _ShimArray(None, dtype="<f8", _shape=z.shape, _flat=flat)

    def bootstrap_counts(draws, size):
        d = asarray(draws, dtype="<i8")
        b_rows, n_cols = d.shape
        check(n_cols == size and size > 0, "Invalid image bootstrap draws")
        flat = [0] * (b_rows * size)
        for b in range(b_rows):
            row_off = b * size
            d_off = b * n_cols
            for j in range(n_cols):
                idx = int(d._flat[d_off + j])
                check(0 <= idx < size, "Bootstrap draw index out of range")
                flat[row_off + idx] += 1
        return _ShimArray(None, dtype="<i8", _shape=(b_rows, size), _flat=flat)

    def bootstrap_interval(draw_values):
        v = asarray(draw_values, dtype="<f8")
        check(v.ndim == 1 and not isinf(v).any(), "Invalid bootstrap summary input")
        finite = [float(x) for x in v._flat if math.isfinite(float(x))]
        if finite:
            lo, hi = quantile(_ShimArray(finite, dtype="<f8"), [0.025, 0.975], method="linear")
        else:
            lo, hi = None, None
        return {
            "lower": None if lo is None else float(lo),
            "upper": None if hi is None else float(hi),
            "valid_draws": len(finite),
            "total_draws": v.size,
        }

    def fixed_macro(values, primary_indices, *, axis=-1):
        v = asarray(values, dtype="<f8")
        check(len(primary_indices) > 0 and len(set(primary_indices)) == len(primary_indices), "Invalid macro cohort")
        return _shim_mean(take(v, primary_indices, axis=axis), axis=axis)

    def paired_image_alignment(plan, *predictions, contexts=None):
        check(bool(predictions), "At least one arm required")
        contexts = (None,) * len(predictions) if contexts is None else contexts
        ids = tuple(plan["screen_ids"])
        aligned = []
        for pred, language in zip(predictions, contexts):
            check(pred.split == "SCREEN", "Pairing requires SCREEN predictions")
            positions = {}
            for i, row in enumerate(pred.rows):
                if language is not None and row["language"] != language:
                    continue
                iid = row["image_id"]
                check(iid not in positions, "Select exactly one view per paired image")
                positions[iid] = i
            check(set(positions) == set(ids), "Pairing population mismatch")
            aligned.append(_ShimArray([positions[iid] for iid in ids], dtype="<i8"))
        return ids, tuple(aligned)

    s4_metrics.probabilities = probabilities
    s4_metrics.bootstrap_counts = bootstrap_counts
    s4_metrics.bootstrap_interval = bootstrap_interval
    s4_metrics.fixed_macro = fixed_macro
    s4_metrics.paired_image_alignment = paired_image_alignment

    # Stage 4 predictions & plan shims
    @dataclass(frozen=True)
    class _ShimPrediction:
        reference: dict
        route: dict
        split: str
        rows: tuple
        logits: Any
        estimable: Any
        prevalence: Any

    s4_predictions = types.ModuleType("src.stage4.predictions")
    s4_predictions.Prediction = _ShimPrediction
    s4_predictions.load_prediction_block = lambda plan, ref, expected_block=None: {}

    s4_plan = types.ModuleType("src.stage4.plan")
    s4_plan.DEPTHS = {"qwen": (9, 18, 27, 36), "gemma": (9, 17, 26, 34)}
    s4_plan.load_bootstrap = lambda plan: None

    pkg3 = types.ModuleType("src.stage3")
    pkg3.__path__ = []
    pkg3.engine = s3_engine
    pkg4 = types.ModuleType("src.stage4")
    pkg4.__path__ = []
    pkg4.metrics = s4_metrics
    pkg4.predictions = s4_predictions
    pkg4.plan = s4_plan
    import src as _src_pkg

    _src_pkg.stage3 = pkg3
    _src_pkg.stage4 = pkg4

    return {
        "src.stage3": pkg3,
        "src.stage4": pkg4,
        "numpy": np_mod,
        "src.stage3.engine": s3_engine,
        "src.stage4.metrics": s4_metrics,
        "src.stage4.predictions": s4_predictions,
        "src.stage4.plan": s4_plan,
    }


def _publish_probe_without_subprocess(root: Path, protocol: dict, dataset_fp: str, producer: dict):
    payload_ref = publish_phase3(
        root,
        protocol,
        dataset_fp,
        "phase3/probes",
        KIND_PROBE_PAYLOAD,
        {"schema": SCHEMA_PROBE, "nonce": "e23-probe-nonce", "remote_sync_guarantee": False},
        producer,
    )
    return publish_phase3(
        root,
        protocol,
        dataset_fp,
        "phase3/probes",
        KIND_PROBE,
        {
            "schema": SCHEMA_PROBE,
            "payload": plain(payload_ref),
            "independent_process_readback": True,
            "reader_pid": os.getpid() + 1,
            "writer_pid": os.getpid(),
            "remote_sync_guarantee": False,
        },
        producer,
        upstream=[plain(payload_ref)],
    )


def _build_synthetic_e23_environment(
    root: Path,
    protocol: dict,
    *,
    model: str = "qwen",
    n_images: int = 44,
    n_draws: int = 40,
    dim: int = 6,
    parity_feature_noise: float = 1e-6,
    confirm_excluded: bool = True,
    e35_excluded: bool = True,
    zero_model_scores_in_pairing: bool = True,
    confirm_rows: int = 0,
):
    """Build a complete synthetic Stage B1 + Stage B2 E23 + Stage 2/3/4 fixture on disk."""
    np_active = _require_numpy()
    from src.stage3.engine import transform

    try:
        from src.stage4.predictions import Prediction
    except Exception:
        Prediction = sys.modules["src.stage4.predictions"].Prediction

    init_phase3_dirs(root)
    producer = _fake_producer()
    categories = list(protocol["category_order"])
    primary_categories = list(protocol["primary_categories"])
    n_cats = len(categories)
    plant_idx = categories.index("plant.n.02")
    depth = 18 if model == "qwen" else 17

    # Screen IDs including preserved unknown cell 6a20677eee04fd08 at index 0
    raw_ids = [f"{2000 + i:016x}" for i in range(n_images)]
    raw_ids[0] = "6a20677eee04fd08"
    screen_ids = tuple(sorted(raw_ids))

    py_rng = random.Random(20261120)
    draws_nested = [[py_rng.randrange(0, n_images) for _ in range(n_images)] for _ in range(n_draws)]
    draws = np_active.asarray(draws_nested, dtype="<i8")
    bootstrap_sha256 = hashlib.sha256(draws.tobytes()).hexdigest()

    # Build targets:
    # - First 10 categories: balanced 22 pos / 22 neg -> K_c = 22 >= 20 ("ok"), except plant.n.02 has 1 unknown -> K_c = 21 >= 20 ("ok")
    # - Next 10 categories: 6 pos / 38 neg -> K_c = 6 < 20 ("descriptive_low_support")
    # - Final 5 categories: 0 pos / 44 neg -> K_c = 0 ("undefined")
    targets_dict = {}
    target_records = {}
    draw0_records = {}
    for i, iid in enumerate(screen_ids):
        vals = []
        for c_idx in range(n_cats):
            if c_idx < 10:
                v = 1 if (i % 2 == 0) else 0
            elif c_idx < 20:
                v = 1 if i < 6 else 0
            else:
                v = 0
            if iid == "6a20677eee04fd08" and c_idx == plant_idx:
                v = None
            vals.append(v)
        targets_dict[iid] = {"split": "SCREEN", "values": vals}
        target_records[iid] = TargetRecord(iid, "SCREEN", tuple(vals))
        for lang in LANGUAGES:
            draw0_records[(iid, lang)] = ModelInput(
                image_id=iid,
                image_path=f"images/screen/{iid}.jpg",
                image_sha256=fingerprint(["img", iid]),
                language=lang,
                caption_id=f"{iid}_{lang}_0",
                caption_text=f"authentic caption {iid} {lang}",
                split="SCREEN",
            )

    e23_plan = build_e23_pair_plan(
        screen_ids=screen_ids,
        screen_targets=target_records,
        draw0_records=draw0_records,
        category_order=categories,
        primary_categories=primary_categories,
        min_screen_support=int(protocol["minimum_positive_and_negative"]["SCREEN"]),
    )
    e23_plan["zero_model_scores_in_pairing"] = zero_model_scores_in_pairing
    e23_plan["confirm_rows"] = confirm_rows

    # Synthetic Stage 3 frozen heads and scalers for both poolings × 4 source languages
    frozen_heads = {p: {} for p in POOLINGS}
    ordinary = {}
    fit_groups = {}
    for p_idx, pooling in enumerate(POOLINGS):
        for l_idx, lang in enumerate(LANGUAGES):
            gid = f"grp-{model}-image_caption-{pooling}-{lang}"
            coef_nested = [
                [
                    (0.85 if d == (c_idx % dim) else (-0.25 if d == ((c_idx + 1) % dim) else 0.05))
                    * (1.1 if pooling == "caption_mean" else 0.9)
                    for d in range(dim)
                ]
                for c_idx in range(n_cats)
            ]
            # Make category 24 unestimable on 'te' to exercise non-estimable zeroing & handling
            est_list = [not (c_idx == n_cats - 1 and lang == "te") for c_idx in range(n_cats)]
            for c_idx, is_est in enumerate(est_list):
                if not is_est:
                    coef_nested[c_idx] = [0.0] * dim
            intercept_nested = [
                0.0 if not est_list[c_idx] else (0.02 * ((c_idx % 5) - 2))
                for c_idx in range(n_cats)
            ]
            coef = np_active.asarray(coef_nested, dtype="<f8")
            intercept = np_active.asarray(intercept_nested, dtype="<f8")
            estimable = np_active.asarray(est_list, dtype="|b1")
            prevalence = np_active.asarray([0.5] * n_cats, dtype="<f8")
            stats = {
                "mean": np_active.asarray([0.05 * d for d in range(dim)], dtype="<f8"),
                "variance": np_active.asarray([1.0] * dim, dtype="<f8"),
                "scale": np_active.asarray([1.0] * dim, dtype="<f8"),
                "center": True,
                "reference": {
                    "path": f"checkpoints/stage3/{gid}_stats/_COMPLETE.json",
                    "sha256": fingerprint(["stats", gid]),
                },
            }
            frozen_heads[pooling][lang] = (coef, intercept, estimable, prevalence, stats)
            ordinary[gid] = {
                "selection": {
                    "path": f"results/stage3/{gid}_sel/_COMPLETE.json",
                    "sha256": fingerprint(["sel", gid]),
                },
                "candidate": {
                    "path": f"results/stage3/{gid}_cand/_COMPLETE.json",
                    "sha256": fingerprint(["cand", gid]),
                },
                "C": 1.0,
            }
            fit_groups[gid] = {
                "source": "vlm",
                "model": model,
                "condition": E23_CONDITION,
                "site": f"decoder:{depth}:{pooling}",
                "language": lang,
                "dimension": dim,
            }

    # Build deterministic image signal vector + caption signal vector per image/language
    img_vectors = {}
    cap_vectors = {}
    for i, iid in enumerate(screen_ids):
        vals = targets_dict[iid]["values"]
        base_v = [0.0] * dim
        for c_idx, v in enumerate(vals):
            sign = 1.0 if v == 1 else (-1.0 if v == 0 else 0.0)
            base_v[c_idx % dim] += 0.35 * sign
        img_vectors[iid] = base_v
        for l_idx, lang in enumerate(LANGUAGES):
            cap_vectors[(iid, lang)] = [base_v[d] + 0.03 * (l_idx + 1) * ((d % 3) - 1) for d in range(dim)]

    def _feature_vec_for_view(shown_iid: str, owner_iid: str, lang: str, pooling: str):
        # Mixture of shown image representation (visual weight alpha) and caption owner representation (1 - alpha)
        alpha = 0.65 if pooling == "caption_mean" else 0.45
        iv = img_vectors[shown_iid]
        cv = cap_vectors[(owner_iid, lang)]
        return [float(alpha * iv[d] + (1.0 - alpha) * cv[d]) for d in range(dim)]

    # Stage 2 authentic draw-0 SCREEN features and Stage 4 SCREEN prediction blocks
    stage2_features = {p: {} for p in POOLINGS}
    blocks_meta = {E23_CONDITION: {p: {} for p in POOLINGS}}
    stage4_pred_map = {}

    for pooling in POOLINGS:
        for t_lang in LANGUAGES:
            rows = tuple(
                {
                    "image_id": iid,
                    "split": "SCREEN",
                    "language": t_lang,
                    "caption_id": f"{iid}_{t_lang}_0",
                    "row_index": idx,
                }
                for idx, iid in enumerate(screen_ids)
            )
            mat_nested = [_feature_vec_for_view(iid, iid, t_lang, pooling) for iid in screen_ids]
            mat = np_active.ascontiguousarray(np_active.asarray(mat_nested, dtype="<f4"), dtype="<f4")
            stage2_features[pooling][t_lang] = (rows, mat)

            block_id = f"blk-{model}-{E23_CONDITION}-{pooling}-{t_lang}"
            pred_ref = {
                "path": f"predictions/stage4/{model}/{pooling}_{t_lang}/_COMPLETE.json",
                "sha256": fingerprint(["s4-pred", model, pooling, t_lang]),
            }
            routes_map = {s_lang: f"route:{model}:{pooling}:{s_lang}->{t_lang}" for s_lang in LANGUAGES}
            blocks_meta[E23_CONDITION][pooling][t_lang] = {
                "block_id": block_id,
                "target_group": f"grp-{model}-image_caption-{pooling}-{t_lang}",
                "prediction_ref": pred_ref,
                "routes": routes_map,
            }
            b_preds = {}
            for s_lang in LANGUAGES:
                rid = routes_map[s_lang]
                coef, intercept, est, prev, stats = frozen_heads[pooling][s_lang]
                z = np_active.asarray(transform(mat, stats) @ coef.T, dtype="<f8") + intercept
                b_preds[rid] = Prediction(
                    reference=pred_ref,
                    route={"route_id": rid, "source_language": s_lang, "target_language": t_lang},
                    split="SCREEN",
                    rows=rows,
                    logits=z,
                    estimable=est,
                    prevalence=prev,
                )
            stage4_pred_map[block_id] = b_preds

    class _PlanStub(dict):
        pass

    fit_dict = {"targets": targets_dict, "groups": fit_groups}
    s4_plan = _PlanStub({
        "screen_ids": list(screen_ids),
        "categories": list(categories),
        "ordinary": ordinary,
        "fit": fit_dict,
    })
    s4_plan.reference = {"path": "cache/shared/stage4/plans/p1/_COMPLETE.json", "sha256": "1" * 64}
    s4_plan.fit = fit_dict
    s4_plan.store = None

    dataset_fp = fingerprint(["e23-stage-b3-fixture", model, n_images])
    protocol_fp = fingerprint(protocol)
    stage4_rel_ref = {
        "path": "results/stage4/releases/r1/_COMPLETE.json",
        "sha256": fingerprint(["s4-rel", dataset_fp]),
    }
    lineage_id = fingerprint(["e23-lineage", dataset_fp, bootstrap_sha256])

    binding = Phase1Binding(
        root=root,
        protocol=frozen(protocol),
        stage4_release_ref=frozen(stage4_rel_ref),
        stage4_release=frozen({"predictions": {}}),
        stage4_plan=s4_plan,
        bootstrap_shape=(n_draws, n_images),
        bootstrap_sha256=bootstrap_sha256,
        midpoints=frozen({"qwen": 18, "gemma": 17}),
        dataset_ref=frozen({"path": "shared/ds/_COMPLETE.json", "sha256": dataset_fp}),
        data_plan_ref=frozen({"path": "shared/dp/_COMPLETE.json", "sha256": "a" * 64}),
        model_plan_refs=frozen({
            "qwen": {"path": "shared/mp_q/_COMPLETE.json", "sha256": "b" * 64},
            "gemma": {"path": "shared/mp_g/_COMPLETE.json", "sha256": "c" * 64},
        }),
        suite_ref=frozen({"path": "results/suite/_COMPLETE.json", "sha256": "d" * 64}),
        stage3_registry_ref=frozen({"path": "results/s3/_COMPLETE.json", "sha256": "e" * 64}),
        g0_gate_ref=frozen({"path": "results/g0/_COMPLETE.json", "sha256": "f" * 64}),
        g0_p_value=0.362,
        dataset_fingerprint=dataset_fp,
        protocol_fingerprint=protocol_fp,
        lineage_identity=lineage_id,
    )

    # Publish Stage B1 phase3_gpu_plan
    probe_ref = _publish_probe_without_subprocess(root, protocol, dataset_fp, producer)
    core = plain({
        "schema": SCHEMA_GPU_PLAN,
        "phase3_gpu_version": PHASE3_GPU_VERSION,
        "lineage_identity": lineage_id,
        "dataset_fingerprint": dataset_fp,
        "protocol_fingerprint": protocol_fp,
        "phase1_stage4_release": stage4_rel_ref,
        "phase3_cpu_plan": None,
        "bootstrap": {"shape": [n_draws, n_images], "sha256": bootstrap_sha256},
        "storage_probe": plain(probe_ref),
        "initialized_dirs": list(PHASE3_PARENT_DIRS),
        "models": {
            model: {
                "depths": [9, depth, 27, 36] if model == "qwen" else [9, depth, 26, 34],
                "midpoint_depth": depth,
                "model_plan_ref": plain(binding.model_plan_refs[model]),
                "midpoint_groups": {},
                "source_systems": {},
            }
        },
        "e39_e31_plan": {"chunks": []},
        "e23_plan": e23_plan,
        "decision_ranges": plain(PHASE3_GPU_DECISION_RANGES),
        "e35_excluded": e35_excluded,
        "confirm_excluded": confirm_excluded,
    })
    plan_id = fingerprint(core)
    plan_body = dict(core, plan_identity=plan_id)
    plan_ref = publish_phase3(
        root,
        protocol,
        dataset_fp,
        "phase3/plans",
        KIND_GPU_PLAN,
        plan_body,
        producer,
        upstream=[stage4_rel_ref, plain(probe_ref)],
    )

    # Publish Stage B2 E23 swapped feature chunks and release
    model_fp = fingerprint(["model-fp", model])
    exec_fp = fingerprint(["exec-fp", model])
    rel_chunks = []
    total_chunks = len(e23_plan["chunks"])

    for c_idx, chunk in enumerate(e23_plan["chunks"]):
        expected_rows = _expected_swapped_chunk_rows(chunk)
        n_views = len(expected_rows)
        last_nested = [
            _feature_vec_for_view(r["shown_image_id"], r["caption_owner_image_id"], r["language"], "last_caption")
            for r in expected_rows
        ]
        mean_nested = [
            _feature_vec_for_view(r["shown_image_id"], r["caption_owner_image_id"], r["language"], "caption_mean")
            for r in expected_rows
        ]
        last_arr = np_active.ascontiguousarray(np_active.asarray(last_nested, dtype="<f4"), dtype="<f4")
        mean_arr = np_active.ascontiguousarray(np_active.asarray(mean_nested, dtype="<f4"), dtype="<f4")

        rows_bytes = jsonl(expected_rows)
        rows_sha = hashlib.sha256(rows_bytes).hexdigest()
        rc = {
            "first_index": 0,
            "last_index": n_views - 1,
            "repeat_indices": sorted({0, n_views - 1}),
            "max_abs_diff_last_caption": 0.0,
            "max_abs_diff_caption_mean": 0.0,
            "passed": True,
        }
        chunk_body = {
            "schema": SCHEMA_E23_FEATURES,
            "experiment": "E23",
            "plan": plain(plan_ref),
            "plan_identity": plan_id,
            "model": model,
            "midpoint_depth": depth,
            "condition": E23_CONDITION,
            "poolings": list(POOLINGS),
            "split": E23_SPLIT,
            "chunk_index": c_idx,
            "chunk_id": chunk["chunk_id"],
            "total_chunks": total_chunks,
            "n_pairs": chunk["n_pairs"],
            "n_views": n_views,
            "dimension": dim,
            "model_fingerprint": model_fp,
            "execution_fingerprint": exec_fp,
            "site_specs": {},
            "evidence_bridge_sha256": "b" * 64,
            "repeat_check": rc,
            "rows_sha256": rows_sha,
            "e35_excluded": True,
            "confirm_excluded": True,
        }
        c_ref = publish_phase3(
            root,
            protocol,
            dataset_fp,
            f"phase3/features/e23/{model}/batch1",
            KIND_E23_CHUNK,
            chunk_body,
            producer,
            arrays={"last_caption": last_arr, "caption_mean": mean_arr},
            files={
                "rows.jsonl": rows_bytes,
                "token_evidence.jsonl": b"",
                "audits.json": json_bytes({"repeat_check": rc}),
            },
            upstream=[plain(plan_ref)],
            model_fingerprint=model_fp,
            run_id=f"chunk_{c_idx:04d}",
        )
        rel_chunks.append({
            "chunk_index": c_idx,
            "chunk_id": chunk["chunk_id"],
            "n_pairs": chunk["n_pairs"],
            "n_views": n_views,
            "ref": plain(c_ref),
            "rows_sha256": rows_sha,
        })

    expected_parity_rows = _expected_authentic_parity_rows(e23_plan["authentic_parity_sample"])
    parity_rows_bytes = jsonl(expected_parity_rows)
    parity_rows_sha = hashlib.sha256(parity_rows_bytes).hexdigest()

    parity_last_nested = [
        [
            v + float(parity_feature_noise)
            for v in _feature_vec_for_view(r["image_id"], r["image_id"], r["language"], "last_caption")
        ]
        for r in expected_parity_rows
    ]
    parity_mean_nested = [
        [
            v + float(parity_feature_noise)
            for v in _feature_vec_for_view(r["image_id"], r["image_id"], r["language"], "caption_mean")
        ]
        for r in expected_parity_rows
    ]
    parity_last_arr = np_active.ascontiguousarray(np_active.asarray(parity_last_nested, dtype="<f4"), dtype="<f4")
    parity_mean_arr = np_active.ascontiguousarray(np_active.asarray(parity_mean_nested, dtype="<f4"), dtype="<f4")

    parity_rc = {
        "first_index": 0,
        "last_index": len(expected_parity_rows) - 1,
        "repeat_indices": [0, len(expected_parity_rows) - 1],
        "max_abs_diff_last_caption": 0.0,
        "max_abs_diff_caption_mean": 0.0,
        "passed": True,
    }
    repeat_summary = {
        "chunks_checked": total_chunks,
        "authentic_parity_checked": True,
        "all_passed": True,
        "max_abs_diff_last_caption": 0.0,
        "max_abs_diff_caption_mean": 0.0,
    }
    release_body = {
        "schema": SCHEMA_E23_FEATURES,
        "experiment": "E23",
        "plan": plain(plan_ref),
        "plan_identity": plan_id,
        "model": model,
        "midpoint_depth": depth,
        "condition": E23_CONDITION,
        "poolings": list(POOLINGS),
        "split": E23_SPLIT,
        "zero_model_scores_in_pairing": True,
        "confirm_rows": 0,
        "total_unique_directed_pairs": e23_plan["total_unique_directed_pairs"],
        "total_swapped_views": e23_plan["total_swapped_views"],
        "total_chunks": total_chunks,
        "authentic_parity_count": len(expected_parity_rows),
        "authentic_parity_rows_sha256": parity_rows_sha,
        "authentic_parity_repeat_check": parity_rc,
        "dimension": dim,
        "model_fingerprint": model_fp,
        "execution_fingerprint": exec_fp,
        "site_specs": {},
        "evidence_bridge_sha256": "b" * 64,
        "chunks": rel_chunks,
        "repeat_check_summary": repeat_summary,
        "e35_excluded": True,
        "confirm_excluded": True,
    }
    release_ref = publish_phase3(
        root,
        protocol,
        dataset_fp,
        f"phase3/features/e23/{model}",
        KIND_E23_RELEASE,
        release_body,
        producer,
        arrays={
            "authentic_parity_last_caption": parity_last_arr,
            "authentic_parity_caption_mean": parity_mean_arr,
        },
        files={
            "authentic_parity_rows.jsonl": parity_rows_bytes,
            "authentic_parity_token_evidence.jsonl": b"",
            "audits.json": json_bytes({"repeat_check_summary": repeat_summary}),
        },
        upstream=[plain(plan_ref)] + [c["ref"] for c in rel_chunks],
        model_fingerprint=model_fp,
        run_id="batch1_release",
    )

    return {
        "binding": binding,
        "plan_ref": plan_ref,
        "plan_id": plan_id,
        "release_ref": release_ref,
        "draws": draws,
        "blocks_meta": blocks_meta,
        "stage4_pred_map": stage4_pred_map,
        "frozen_heads": frozen_heads,
        "stage2_features": stage2_features,
        "screen_ids": screen_ids,
        "producer": producer,
        "e23_plan": e23_plan,
    }


def _patch_e23_runtime(fx):
    def _fake_read_features(binding, m, cond, pool, t_lang, split):
        require_open_split(split)
        return fx["stage2_features"][pool][t_lang]

    def _fake_load_head(binding, m, cond, pool, s_lang):
        return fx["frozen_heads"][pool][s_lang]

    def _fake_midpoint_groups(binding, m, *, condition="image_caption", pooling="last_caption"):
        return frozen({lang: f"grp-{m}-{condition}-{pooling}-{lang}" for lang in LANGUAGES})

    return (
        mock.patch("src.phase3.gpu_plan.bind_phase1", return_value=fx["binding"]),
        mock.patch("src.phase3.run_e23.e29_screen_blocks", return_value=frozen(fx["blocks_meta"])),
        mock.patch("src.phase3.run_e23.midpoint_groups", side_effect=_fake_midpoint_groups),
        mock.patch("src.phase3.run_e23.load_frozen_source_head", side_effect=_fake_load_head),
        mock.patch("src.phase3.run_e23.read_verified_midpoint_features", side_effect=_fake_read_features),
        mock.patch("src.stage4.plan.load_bootstrap", return_value=fx["draws"]),
        mock.patch(
            "src.stage4.predictions.load_prediction_block",
            side_effect=lambda p, ref, expected_block=None: fx["stage4_pred_map"][expected_block],
        ),
    )


class Phase3E23StdlibTests(unittest.TestCase):
    """Stdlib-only contract, CLI, notebook, pairing-formula, and decision-rule tests."""

    def test_lazy_imports_and_constants(self):
        if not HAS_NUMPY:
            self.assertNotIn("numpy", sys.modules)
            self.assertNotIn("scipy", sys.modules)
        self.assertEqual(E23_EXPERIMENT, "E23")
        self.assertEqual(E23_CONDITION, "image_caption")
        self.assertEqual(E23_SPLIT, "SCREEN")
        self.assertEqual(E23_PREDICTIONS_NAMESPACE, "phase3/predictions/e23")
        self.assertEqual(E23_RESULTS_NAMESPACE, "phase3/results/e23")
        self.assertEqual(AUTHENTIC_PARITY_TOL, 1e-4)
        self.assertEqual(len(E23_LINEAR_SWAP_METRICS), 5)
        self.assertEqual(len(E23_RATIO_SWAP_METRICS), 2)
        self.assertEqual(len(E23_SWAP_METRICS), 7)
        self.assertEqual(
            E23_MACRO_COHORTS,
            ("supported_primary_macro", "fixed_primary_macro", "all_supported_macro"),
        )
        self.assertEqual(SUPERVISION_DIAGONAL, "within_context_source_fit")
        self.assertEqual(SUPERVISION_OFF_DIAGONAL, "source_only_transfer")
        self.assertEqual(set(CERTIFIED_STAGE_B2_RELEASES["e23"].keys()), set(MODELS))
        with self.assertRaises(ContractError):
            require_open_split("CONFIRM")

    def test_cli_help_and_blocked_exit_stdlib(self):
        help_res = subprocess.run(
            [sys.executable, "-m", "src.phase3.run_e23", "--help"],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        self.assertEqual(help_res.returncode, 0, help_res.stderr)
        for flag in (
            "--model",
            "--phase1-root",
            "--plan",
            "--release",
            "--protocol",
            "--resume-predictions",
            "--resume-analysis",
            "--run-id",
            "--storage-scope",
        ):
            self.assertIn(flag, help_res.stdout)

        with tempfile.TemporaryDirectory() as tmp:
            bad_res = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.phase3.run_e23",
                    "--model",
                    "qwen",
                    "--phase1-root",
                    tmp,
                    "--plan",
                    "phase3/plans/nonexistent/_COMPLETE.json",
                    "--storage-scope",
                    "local-development",
                ],
                cwd=REPO,
                capture_output=True,
                text=True,
            )
            self.assertEqual(bad_res.returncode, 2)
            lines = [json.loads(line) for line in bad_res.stdout.strip().splitlines() if line.strip()]
            self.assertEqual(lines[-1]["status"], "blocked")
            self.assertFalse(lines[-1]["phase3_e23_complete"])

    def test_notebook_contract(self):
        nb_path = REPO / "scripts/phase3/run_e23.ipynb"
        self.assertTrue(nb_path.is_file(), "Missing scripts/phase3/run_e23.ipynb")
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertEqual(nb.get("nbformat"), 4)
        self.assertEqual(len(nb.get("cells", [])), 4)
        self.assertEqual(nb["cells"][0]["cell_type"], "markdown")
        for idx in (1, 2, 3):
            self.assertEqual(nb["cells"][idx]["cell_type"], "code")

        setup_src = "".join(nb["cells"][1]["source"])
        cfg_src = "".join(nb["cells"][2]["source"])
        exec_src = "".join(nb["cells"][3]["source"])

        self.assertIn("CODE_REF = 'phase3'", setup_src)
        self.assertIn("OPENBLAS_NUM_THREADS", setup_src)
        self.assertIn("requirements-stage2-cpu.txt", setup_src)
        self.assertIn("MODEL = 'qwen'", cfg_src)
        self.assertIn(CERTIFIED_STAGE_B1_GPU_PLAN["path"], cfg_src)
        self.assertIn(CERTIFIED_STAGE_B1_GPU_PLAN["sha256"], cfg_src)
        self.assertIn("RELEASE_REF = None", cfg_src)
        self.assertIn("RESUME_PREDICTIONS_REF = None", cfg_src)
        self.assertIn("RESUME_ANALYSIS_REF = None", cfg_src)
        self.assertIn("src.phase3.run_e23", exec_src)
        self.assertIn("e23_complete", exec_src)

    def test_pair_shift_and_grounding_fraction_formulas_stdlib(self):
        # 1. Pure visual grounding: P depends only on shown image (pos=0.90, neg=0.10)
        vis = compute_e23_pair_shifts(
            p_auth_pos=0.90,
            p_auth_neg=0.10,
            p_swap_pos_shown_neg_cap=0.90,
            p_swap_neg_shown_pos_cap=0.10,
        )
        self.assertAlmostEqual(vis["delta_auth"], 0.80)
        self.assertAlmostEqual(vis["shift_neg_caption_to_pos_image"], 0.80)
        self.assertAlmostEqual(vis["shift_pos_caption_to_neg_image"], 0.80)
        self.assertAlmostEqual(vis["signed_drop_on_neg_image"], -0.80)
        self.assertAlmostEqual(vis["signed_shift_toward_shown_image"], 0.80)
        vgf_vis, cdf_vis = compute_e23_grounding_fractions(
            vis["signed_shift_toward_shown_image"], vis["delta_auth"]
        )
        self.assertAlmostEqual(vgf_vis, 1.0)
        self.assertAlmostEqual(cdf_vis, 0.0)

        # 2. Pure caption dominance: P depends only on caption owner (pos_cap=0.90, neg_cap=0.10)
        cap = compute_e23_pair_shifts(
            p_auth_pos=0.90,
            p_auth_neg=0.10,
            p_swap_pos_shown_neg_cap=0.10,
            p_swap_neg_shown_pos_cap=0.90,
        )
        self.assertAlmostEqual(cap["delta_auth"], 0.80)
        self.assertAlmostEqual(cap["shift_neg_caption_to_pos_image"], 0.0)
        self.assertAlmostEqual(cap["shift_pos_caption_to_neg_image"], 0.0)
        self.assertAlmostEqual(cap["signed_drop_on_neg_image"], 0.0)
        self.assertAlmostEqual(cap["signed_shift_toward_shown_image"], 0.0)
        vgf_cap, cdf_cap = compute_e23_grounding_fractions(
            cap["signed_shift_toward_shown_image"], cap["delta_auth"]
        )
        self.assertAlmostEqual(vgf_cap, 0.0)
        self.assertAlmostEqual(cdf_cap, 1.0)

        # 3. Partial visual grounding: P_swap(pos, neg)=0.50, P_swap(neg, pos)=0.30
        part = compute_e23_pair_shifts(
            p_auth_pos=0.90,
            p_auth_neg=0.10,
            p_swap_pos_shown_neg_cap=0.50,
            p_swap_neg_shown_pos_cap=0.30,
        )
        self.assertAlmostEqual(part["delta_auth"], 0.80)
        self.assertAlmostEqual(part["shift_neg_caption_to_pos_image"], 0.40)
        self.assertAlmostEqual(part["shift_pos_caption_to_neg_image"], 0.60)
        self.assertAlmostEqual(part["signed_drop_on_neg_image"], -0.60)
        self.assertAlmostEqual(part["signed_shift_toward_shown_image"], 0.50)
        vgf_part, cdf_part = compute_e23_grounding_fractions(
            part["signed_shift_toward_shown_image"], part["delta_auth"]
        )
        self.assertAlmostEqual(vgf_part, 0.625)
        self.assertAlmostEqual(cdf_part, 0.375)

        # 4. Near-zero delta_auth denominator returns (None, None)
        self.assertEqual(compute_e23_grounding_fractions(0.1, 1e-13), (None, None))
        self.assertEqual(compute_e23_grounding_fractions(None, 0.5), (None, None))

        # 5. Category support status rules
        self.assertEqual(e23_category_support_status(0.4, 25, 20), ("ok", None))
        self.assertEqual(
            e23_category_support_status(0.4, 12, 20),
            ("descriptive_low_support", "pair_support_below_minimum_20"),
        )
        self.assertEqual(
            e23_category_support_status(None, 0, 20),
            ("undefined", "zero_opposite_label_pairs"),
        )
        self.assertEqual(
            e23_category_support_status(None, 25, 20, unestimable=True),
            ("undefined", "unestimable_head"),
        )
        self.assertEqual(
            e23_category_support_status(None, 25, 20, unestimable=False),
            ("undefined", "near_zero_delta_auth_denominator"),
        )

    def test_primary_cohort_audit_and_unknown_cell_exclusion(self):
        protocol = _load_protocol()
        cats = list(protocol["category_order"])
        prim = list(protocol["primary_categories"])
        plant_idx = cats.index("plant.n.02")

        screen_ids = [f"{i:016x}" for i in range(60)]
        screen_ids[0] = "6a20677eee04fd08"
        screen_targets = {}
        draw0_records = {}
        for i, iid in enumerate(screen_ids):
            vals = []
            for c_idx, cat in enumerate(cats):
                if c_idx == 0:
                    v = 1 if i < 10 else 0  # K_c = 10 -> descriptive_low_support
                else:
                    v = 1 if (i % 2 == 0) else 0  # K_c = 30 (or 29 on plant.n.02) -> ok
                if iid == "6a20677eee04fd08" and c_idx == plant_idx:
                    v = None
                vals.append(v)
            screen_targets[iid] = TargetRecord(iid, "SCREEN", tuple(vals))
            for lang in LANGUAGES:
                draw0_records[(iid, lang)] = ModelInput(
                    image_id=iid,
                    image_path=f"images/{iid}.jpg",
                    image_sha256="a" * 64,
                    language=lang,
                    caption_id=f"{iid}_{lang}_0",
                    caption_text=f"cap {iid} {lang}",
                    split="SCREEN",
                )

        pair_plan = build_e23_pair_plan(screen_ids, screen_targets, draw0_records, cats, prim, 20)
        audit = build_e23_primary_cohort_audit(pair_plan["per_category"], cats, prim, 20)
        self.assertFalse(audit["all_primary_meet_min_pairs"])
        self.assertEqual(audit["min_primary_pairs"], 10)
        self.assertIn(cats[0], audit["descriptive_low_support_primary_categories"])
        self.assertIn("plant.n.02", audit["supported_primary_categories"])
        self.assertEqual(audit["per_category_n_pairs"]["plant.n.02"], 29)

    def test_decision_summary_evaluation_rules(self):
        ranges = plain(PHASE3_GPU_DECISION_RANGES["E23"])
        audit_stub = {
            "all_primary_meet_min_pairs": False,
            "min_primary_pairs": 12,
            "n_supported_primary_categories": 10,
            "supported_primary_categories": ["dog.n.01"],
            "descriptive_low_support_primary_categories": ["cat.n.01"],
        }

        def _make_pooling_block(ss_val, ss_lo, ss_hi, vgf_val=0.60):
            m_table = {}
            for m in E23_SWAP_METRICS:
                val = vgf_val if m == "visual_grounding_fraction" else (1.0 - vgf_val if m == "caption_dominance_fraction" else ss_val)
                entry = _make_interval_entry(val, ss_lo, ss_hi)
                m_table[m] = {
                    "primary_macro": entry,
                    "supported_primary_macro": entry,
                    "fixed_primary_macro": entry,
                    "all_supported_macro": entry,
                }
            rg = {"route_count": 12, "routes": [], "metrics": m_table}
            return {
                "twelve_off_diagonal_mean": rg,
                "four_diagonal_mean": rg,
                "sixteen_all_route_mean": rg,
            }

        def _make_contrast_block(diff_val, diff_lo, diff_hi):
            m_table = {
                m: {"supported_primary_macro": _make_interval_entry(diff_val, diff_lo, diff_hi)}
                for m in E23_SWAP_METRICS
            }
            rg = {"route_count": 12, "routes": [], "metrics": m_table}
            return {
                "twelve_off_diagonal_mean": rg,
                "four_diagonal_mean": rg,
                "sixteen_all_route_mean": rg,
            }

        # Case 1: Consequential visual grounding (>= 0.25, CI > 0) + pooling contrast significant
        p_conseq = {
            "last_caption": _make_pooling_block(0.28, 0.22, 0.34, vgf_val=0.55),
            "caption_mean": _make_pooling_block(0.36, 0.30, 0.42, vgf_val=0.70),
        }
        c_sig = _make_contrast_block(0.08, 0.03, 0.13)
        d1 = evaluate_e23_decision_summary(
            poolings_summary=p_conseq,
            pooling_contrast=c_sig,
            primary_cohort_audit=audit_stub,
            decision_ranges=ranges,
            zero_model_scores_in_pairing=True,
            authentic_anchor_parity_passed=True,
            max_authentic_parity_diff=1e-6,
            en_matched_self_check_passed=True,
        )
        self.assertTrue(d1["sanity_checks"]["all_passed"])
        self.assertTrue(d1["any_consequential_visual_grounding"])
        self.assertFalse(d1["all_negligible_visual_shift"])
        self.assertEqual(d1["outcome"], "visual_grounding_consequential")
        self.assertTrue(
            d1["caption_mean_minus_last_caption"]["twelve_off_diagonal_mean"]["pooling_changes_signed_shift"]
        )

        # Case 2: Negligible visual shift (CI upper < 0.05)
        p_neg = {
            "last_caption": _make_pooling_block(0.01, -0.01, 0.03, vgf_val=0.02),
            "caption_mean": _make_pooling_block(0.02, -0.01, 0.04, vgf_val=0.03),
        }
        c_null = _make_contrast_block(0.01, -0.01, 0.03)
        d2 = evaluate_e23_decision_summary(
            poolings_summary=p_neg,
            pooling_contrast=c_null,
            primary_cohort_audit=audit_stub,
            decision_ranges=ranges,
            zero_model_scores_in_pairing=True,
            authentic_anchor_parity_passed=True,
            max_authentic_parity_diff=1e-6,
            en_matched_self_check_passed=True,
        )
        self.assertFalse(d2["any_consequential_visual_grounding"])
        self.assertTrue(d2["all_negligible_visual_shift"])
        self.assertEqual(d2["outcome"], "caption_dominated_negligible_visual_shift")
        self.assertFalse(
            d2["caption_mean_minus_last_caption"]["twelve_off_diagonal_mean"]["pooling_changes_signed_shift"]
        )


class Phase3E23EndToEndTests(unittest.TestCase):
    """End-to-end synthetic fixture tests (runs with real numpy or stdlib scientific shim)."""

    def test_end_to_end_scoring_bootstrap_resume_and_tamper(self):
        if HAS_NUMPY:
            self._run_end_to_end_suite()
        else:
            shim_mods = _make_stdlib_scientific_modules()
            with mock.patch.dict(sys.modules, shim_mods):
                self._run_end_to_end_suite()

    def _run_end_to_end_suite(self):
        protocol = _load_protocol()
        np_active = _require_numpy()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fx = _build_synthetic_e23_environment(root, protocol, model="qwen")
            events = []

            p1, p2, p3, p4, p5, p6, p7 = _patch_e23_runtime(fx)
            with p1, p2, p3, p4, p5, p6, p7:
                res = run_e23(
                    root,
                    protocol,
                    "qwen",
                    fx["producer"],
                    plan_ref=fx["plan_ref"],
                    release_ref=fx["release_ref"],
                    run_id="e23_run_1",
                    progress=events.append,
                )

            self.assertEqual(res["status"], "passed")
            self.assertEqual(res["model"], "qwen")
            self.assertEqual(res["midpoint_depth"], 18)
            self.assertFalse(res["resumed_predictions"])
            self.assertFalse(res["resumed_analysis"])
            self.assertTrue(res["e35_excluded"])
            self.assertTrue(res["confirm_excluded"])
            self.assertTrue(res["authentic_anchor_parity"]["passed"])
            self.assertLessEqual(res["authentic_anchor_parity"]["max_feature_abs_diff"], AUTHENTIC_PARITY_TOL)
            self.assertLessEqual(res["authentic_anchor_parity"]["max_logit_abs_diff"], AUTHENTIC_PARITY_TOL)
            self.assertTrue(res["decision_summary"]["sanity_checks"]["all_passed"])

            event_names = [e["event"] for e in events]
            self.assertIn("e23_parity_and_chunks_verified", event_names)
            self.assertIn("e23_predictions_published", event_names)
            self.assertIn("e23_bootstrap_evaluated", event_names)
            self.assertIn("e23_analysis_published", event_names)

            # Verify published prediction artifact on disk
            pred_ref, pred_body, pred_marker = load_phase3_artifact(
                root,
                protocol,
                res["predictions_ref"],
                KIND_PREDICTION,
                expected_schema=SCHEMA_PREDICTION,
                expected_plan_identity=fx["plan_id"],
                expected_owner="qwen",
            )
            self.assertEqual(pred_body["experiment"], "E23")
            self.assertTrue(pred_body["confirm_excluded"])
            self.assertTrue(pred_body["e35_excluded"])
            pred_folder = (root / pred_ref["path"]).parent
            for pool in POOLINGS:
                logits_arr = np_active.load(pred_folder / f"{pool}_logits.npy", allow_pickle=False)
                est_arr = np_active.load(pred_folder / f"{pool}_estimable.npy", allow_pickle=False)
                self.assertEqual(logits_arr.shape, (4, fx["e23_plan"]["total_swapped_views"], 25))
                self.assertEqual(est_arr.shape, (4, 25))
                # Category 24 on source language 'te' (index 3) is unestimable -> logits must be 0.0
                self.assertFalse(bool(est_arr[3, 24]))
                self.assertFalse(bool(np_active.any(logits_arr * (~est_arr[:, None, :]))))

            # Verify published analysis artifact on disk
            ana_ref, ana_body, _ = load_phase3_artifact(
                root,
                protocol,
                res["analysis_ref"],
                KIND_ANALYSIS,
                expected_schema=SCHEMA_ANALYSIS,
                expected_plan_identity=fx["plan_id"],
                expected_owner="qwen",
            )
            self.assertEqual(ana_body["experiment"], "E23")
            self.assertTrue(ana_body["confirm_excluded"])
            self.assertTrue(ana_body["e35_excluded"])
            self.assertEqual(ana_body["confirm_rows"], 0)
            self.assertTrue(ana_body["zero_model_scores_in_pairing"])

            # Check primary cohort audit & category support statuses
            pca = ana_body["primary_cohort_audit"]
            self.assertGreater(pca["n_supported_primary_categories"], 0)
            self.assertGreater(pca["n_descriptive_low_support_primary_categories"], 0)
            self.assertLessEqual(pca["per_category_n_pairs"]["plant.n.02"], 6)
            self.assertNotIn("plant.n.02", pca["supported_all_categories"])

            # Check poolings_summary & caption_mean_minus_last_caption structure
            for pool in POOLINGS:
                p_sum = ana_body["poolings_summary"][pool]
                self.assertEqual(p_sum["twelve_off_diagonal_mean"]["route_count"], 12)
                self.assertEqual(p_sum["four_diagonal_mean"]["route_count"], 4)
                self.assertEqual(p_sum["sixteen_all_route_mean"]["route_count"], 16)
                self.assertEqual(len(p_sum["per_route"]), 16)
                off_ss = p_sum["twelve_off_diagonal_mean"]["metrics"]["signed_shift_toward_shown_image"]
                self.assertEqual(off_ss["supported_primary_macro"]["status"], "ok")
                self.assertIsNotNone(off_ss["supported_primary_macro"]["value"])
                self.assertGreater(off_ss["supported_primary_macro"]["value"], 0.0)
                # Unestimable category 24 on te->en must be marked undefined with reason unestimable_head
                last_cat = protocol["category_order"][-1]
                te_en_cat = p_sum["per_route"]["te->en"]["metrics"]["signed_shift_toward_shown_image"]["categories"][last_cat]
                self.assertEqual(te_en_cat["status"], "undefined")
                self.assertEqual(te_en_cat["reason"], "unestimable_head")

            contrast = ana_body["caption_mean_minus_last_caption"]
            c_ss = contrast["twelve_off_diagonal_mean"]["metrics"]["signed_shift_toward_shown_image"]["supported_primary_macro"]
            mean_ss = ana_body["poolings_summary"]["caption_mean"]["twelve_off_diagonal_mean"]["metrics"]["signed_shift_toward_shown_image"]["supported_primary_macro"]["value"]
            last_ss = ana_body["poolings_summary"]["last_caption"]["twelve_off_diagonal_mean"]["metrics"]["signed_shift_toward_shown_image"]["supported_primary_macro"]["value"]
            self.assertAlmostEqual(c_ss["value"], mean_ss - last_ss, places=10)

            # Test --resume-predictions
            events_rp = []
            with p1, p2, p3, p4, p5, p6, p7:
                res_rp = run_e23(
                    root,
                    protocol,
                    "qwen",
                    fx["producer"],
                    plan_ref=fx["plan_ref"],
                    release_ref=fx["release_ref"],
                    resume_predictions=res["predictions_ref"],
                    run_id="e23_run_2",
                    progress=events_rp.append,
                )
            self.assertTrue(res_rp["resumed_predictions"])
            self.assertFalse(res_rp["resumed_analysis"])
            self.assertEqual(res_rp["predictions_ref"], res["predictions_ref"])
            self.assertTrue(any(e["event"] == "e23_predictions_resumed" for e in events_rp))

            # Test --resume-analysis
            events_ra = []
            with p1, p2, p3, p4, p5, p6, p7:
                res_ra = run_e23(
                    root,
                    protocol,
                    "qwen",
                    fx["producer"],
                    plan_ref=fx["plan_ref"],
                    release_ref=fx["release_ref"],
                    resume_analysis=res["analysis_ref"],
                    run_id="e23_run_3",
                    progress=events_ra.append,
                )
            self.assertTrue(res_ra["resumed_analysis"])
            self.assertEqual(res_ra["analysis_ref"], res["analysis_ref"])
            self.assertTrue(any(e["event"] == "e23_analysis_resumed" for e in events_ra))

            # Test default plan_ref=None and release_ref=None resolving to CERTIFIED constants
            with (
                p1,
                p2,
                p3,
                p4,
                p5,
                p6,
                p7,
                mock.patch("src.phase3.run_e23.CERTIFIED_STAGE_B1_GPU_PLAN", fx["plan_ref"]),
                mock.patch(
                    "src.phase3.run_e23.CERTIFIED_STAGE_B2_RELEASES",
                    {"e23": {"qwen": fx["release_ref"], "gemma": fx["release_ref"]}},
                ),
            ):
                res_def = run_e23(
                    root,
                    protocol,
                    "qwen",
                    fx["producer"],
                    plan_ref=None,
                    release_ref=None,
                    resume_analysis=res["analysis_ref"],
                )
            self.assertEqual(res_def["analysis_ref"], res["analysis_ref"])

            # Foreign owner rejection on --resume-analysis
            with p1, p2, p3, p4, p5, p6, p7:
                with self.assertRaises(ContractError):
                    run_e23(
                        root,
                        protocol,
                        "gemma",
                        fx["producer"],
                        plan_ref=fx["plan_ref"],
                        release_ref=fx["release_ref"],
                        resume_analysis=res["analysis_ref"],
                    )

            # Tamper detection on resumed predictions array
            (pred_folder / "last_caption_logits.npy").write_bytes(b"corrupted_bytes")
            with p1, p2, p3, p4, p5, p6, p7:
                with self.assertRaises(ContractError):
                    run_e23(
                        root,
                        protocol,
                        "qwen",
                        fx["producer"],
                        plan_ref=fx["plan_ref"],
                        release_ref=fx["release_ref"],
                        resume_predictions=res["predictions_ref"],
                    )

        # Test authentic anchor parity failure when feature diff > 1e-4
        with tempfile.TemporaryDirectory() as tmp_bad_parity:
            root_bp = Path(tmp_bad_parity)
            fx_bp = _build_synthetic_e23_environment(
                root_bp, protocol, model="qwen", parity_feature_noise=5e-4
            )
            p1, p2, p3, p4, p5, p6, p7 = _patch_e23_runtime(fx_bp)
            with p1, p2, p3, p4, p5, p6, p7:
                with self.assertRaisesRegex(ContractError, "Authentic anchor feature parity exceeded tolerance"):
                    run_e23(
                        root_bp,
                        protocol,
                        "qwen",
                        fx_bp["producer"],
                        plan_ref=fx_bp["plan_ref"],
                        release_ref=fx_bp["release_ref"],
                    )

        # Test en->en matched logit self-check failure when Stage 4 en->en logits are perturbed
        with tempfile.TemporaryDirectory() as tmp_bad_en:
            root_be = Path(tmp_bad_en)
            fx_be = _build_synthetic_e23_environment(root_be, protocol, model="qwen")
            en_blk_id = fx_be["blocks_meta"][E23_CONDITION]["last_caption"]["en"]["block_id"]
            en_route_id = fx_be["blocks_meta"][E23_CONDITION]["last_caption"]["en"]["routes"]["en"]
            orig_pred = fx_be["stage4_pred_map"][en_blk_id][en_route_id]
            fx_be["stage4_pred_map"][en_blk_id][en_route_id] = type(orig_pred)(
                reference=orig_pred.reference,
                route=orig_pred.route,
                split=orig_pred.split,
                rows=orig_pred.rows,
                logits=orig_pred.logits + 1e-6,
                estimable=orig_pred.estimable,
                prevalence=orig_pred.prevalence,
            )
            p1, p2, p3, p4, p5, p6, p7 = _patch_e23_runtime(fx_be)
            with p1, p2, p3, p4, p5, p6, p7:
                with self.assertRaisesRegex(ContractError, "en->en matched self-check failed"):
                    run_e23(
                        root_be,
                        protocol,
                        "qwen",
                        fx_be["producer"],
                        plan_ref=fx_be["plan_ref"],
                        release_ref=fx_be["release_ref"],
                    )


@unittest.skipUnless(HAS_NUMPY, "numpy/scipy required for real numpy E23 tests")
class Phase3E23RealNumpyTests(unittest.TestCase):
    def test_real_numpy_end_to_end_gemma(self):
        protocol = _load_protocol()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fx = _build_synthetic_e23_environment(root, protocol, model="gemma")
            p1, p2, p3, p4, p5, p6, p7 = _patch_e23_runtime(fx)
            with p1, p2, p3, p4, p5, p6, p7:
                res = run_e23(
                    root,
                    protocol,
                    "gemma",
                    fx["producer"],
                    plan_ref=fx["plan_ref"],
                    release_ref=fx["release_ref"],
                )
            self.assertEqual(res["status"], "passed")
            self.assertEqual(res["midpoint_depth"], 17)


if __name__ == "__main__":
    unittest.main()
