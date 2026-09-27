"""RABIT-KV2 Stage3C component profiler (diagnostic only; OFF by default).

Enabled only by ``VLLM_RABIT2_STAGE3C_COMPONENT_PROFILE=1``. When disabled,
``rabit2_stage3c_profile_scope`` returns a shared no-op context manager and
nothing else in this module runs: no patching, no CUDA events, no
synchronization, no allocation, no records.

When enabled, each non-initial chunk (q_len > 1) runs inside a profiling scope
that temporarily replaces the call-time attribute lookups the Stage3C path
already performs (``Rabit2CausalChunkPlan.__init__`` / ``apply_step``, the
Stage4D3.4 tail helpers and every Triton kernel launch ``K[grid](...)``) with
timing wrappers, and restores the originals when the chunk ends. The wrapped
callables are the unmodified originals, called with unmodified arguments; no
frozen source is edited. A profiled run is NEVER a latency result.

Two separate timing domains (never combined):

HOST (``time.perf_counter``), a tree of windows. For every window:
    inclusive_ms       -- time inside the wrapped call;
    overhead_ms        -- profiler bookkeeping around the call (event records,
                          stack handling);
    children_window_ms -- sum of the full windows of instrumented callees;
    exclusive_ms       -- inclusive_ms - children_window_ms.
  The root ``chunk`` spans the scope (``wall_ms``); its exclusive time is the
  un-instrumented remainder. By construction
      wall_ms = sum(exclusive_ms of all windows incl. root) + sum(overhead_ms).

GPU (CUDA events on the current stream). One event pair around every GPU
leaf call (``chunk_plan`` and every Triton kernel launch) plus one pair around
the whole scope (``span_ms``). No synchronization inside the scope: all
elapsed times are read after a single ``torch.cuda.synchronize()`` at scope
exit. GPU leaves are never nested (``gpu_nested`` counts violations). If the
device is idle when a leaf's start event is reached, that leaf's time includes
the part of its host launch path after the start event.

One log line per chunk call (i.e. per layer):
    RABIT2_STAGE3C_COMPONENT_PROFILE={json}
"""

from __future__ import annotations

import contextlib
import functools
import json
import os
import sys
import time

import torch

import vllm.v1.attention.ops.rabit_kv2 as _r
import vllm.v1.attention.ops.rabit_kv2_stage3c_tile32 as _t32
from vllm.logger import init_logger

logger = init_logger(__name__)

COMPONENT_PROFILE_ENV = "VLLM_RABIT2_STAGE3C_COMPONENT_PROFILE"
LOG_TAG = "RABIT2_STAGE3C_COMPONENT_PROFILE"
SCHEMA = "rabit2_stage3c_component_profile/v2"
ROOT = "chunk"
TRITON_ATTN_MODULE = "vllm.v1.attention.backends.triton_attn"

# (attribute, window key). Owners are resolved at scope entry.
_RABIT_FUNCS = (
    ("_rabit2_stage4b1_exactmeta_emit_tail_partial", "tail"),
    ("_rabit2_stage4d3_4_fast_prep", "tail_prep"),
    ("rabit2_online_decode_attention_triton", "reference_attention_call"),
)
_RABIT_KERNELS = (
    ("_rabit2_stage4b3_gqa4_closed_page_partial_kernel", "closed_page"),
    ("_rabit2_stage4d3_4_k_stats_codes_kernel", "tail_prep.k_stats_codes"),
    ("_rabit2_stage4d3_4_pack_k3_kernel", "tail_prep.k3_pack"),
    ("_rabit2_stage4d3_4_meta64_kernel", "tail_prep.meta8g64"),
    ("_rabit2_stage4b1_exactmeta_tail_partial_kernel", "tail_partial.open_recent"),
    ("_rabit2_tail_partial_kernel", "tail_partial.recent_only"),
    ("_rabit2_reduce_partials_kernel", "reduce"),
)
_TILE32_KERNELS = (
    ("_rabit2_tile32_closed_page_partial_kernel", "closed_page"),
    ("_rabit2_tile32_reduce_partials_kernel", "reduce"),
)
_CHUNK_PLAN_METHODS = (
    ("__init__", "chunk_plan"),
    ("apply_step", "apply_step"),
)
# Windows that also get a CUDA event pair (mutually exclusive in time).
GPU_KEYS = frozenset({"chunk_plan", *(k for _, k in _RABIT_KERNELS), *(k for _, k in _TILE32_KERNELS)})

_NULL_SCOPE = contextlib.nullcontext()
_EVENT_POOL: list = []


def rabit2_stage3c_component_profiling() -> bool:
    return os.environ.get(COMPONENT_PROFILE_ENV, "0") == "1"


def rabit2_stage3c_profile_scope(q_len: int, context_len: int):
    """Profiling scope for one Stage3C chunk; a no-op unless enabled."""
    if q_len <= 1 or not rabit2_stage3c_component_profiling():
        return _NULL_SCOPE
    return _ChunkProfile(int(q_len), int(context_len))


def _event():
    return _EVENT_POOL.pop() if _EVENT_POOL else torch.cuda.Event(enable_timing=True)


class _KernelProxy:
    """Times ``kernel[grid](*args, **kwargs)``; forwards everything else."""

    def __init__(self, prof: "_ChunkProfile", key: str, kernel) -> None:
        self._prof, self._key, self._kernel = prof, key, kernel

    def __getitem__(self, grid):
        launch = self._kernel[grid]
        return lambda *a, **k: self._prof.window(self._key, launch, a, k)

    def __getattr__(self, name):
        return getattr(self._kernel, name)


class _ChunkProfile:
    def __init__(self, q_len: int, context_len: int) -> None:
        self.q_len, self.context_len = q_len, context_len
        self.impl = _t32.rabit2_stage3c_impl()
        self.host: dict[str, dict] = {}
        self.stack: list[str] = []
        self.child_windows: list[float] = []
        self.pending: list[tuple[str, object, object]] = []
        self.patches: list[tuple[object, str, object, object]] = []
        self.gpu_open = False
        self.gpu_nested = 0

    # -- patching ---------------------------------------------------------
    def _patch(self, owner, name: str, replacement) -> None:
        original = getattr(owner, name)
        setattr(owner, name, replacement)
        self.patches.append((owner, name, original, replacement))

    def _wrap(self, key: str, fn):
        @functools.wraps(fn)
        def wrapper(*a, **k):
            return self.window(key, fn, a, k)

        return wrapper

    def _install(self) -> None:
        for name, key in _CHUNK_PLAN_METHODS:
            fn = _r.Rabit2CausalChunkPlan.__dict__[name]
            self._patch(_r.Rabit2CausalChunkPlan, name, self._wrap(key, fn))
        for name, key in _RABIT_FUNCS:
            self._patch(_r, name, self._wrap(key, getattr(_r, name)))
        for owner, table in ((_r, _RABIT_KERNELS), (_t32, _TILE32_KERNELS)):
            for name, key in table:
                self._patch(owner, name, _KernelProxy(self, key, getattr(owner, name)))
        ta = sys.modules.get(TRITON_ATTN_MODULE)
        if ta is not None and hasattr(ta, "rabit2_online_decode_attention_triton"):
            fn = ta.rabit2_online_decode_attention_triton
            self._patch(ta, "rabit2_online_decode_attention_triton", self._wrap("reference_attention_call", fn))

    def _restore(self) -> None:
        while self.patches:
            owner, name, original, replacement = self.patches.pop()
            if getattr(owner, name) is not replacement:
                raise RuntimeError(f"RABIT-2 Stage3C profiler: {name} changed while patched")
            setattr(owner, name, original)

    # -- timing -----------------------------------------------------------
    def window(self, key: str, fn, a, k):
        t_enter = time.perf_counter()
        gpu = key in GPU_KEYS and not self.gpu_open
        if key in GPU_KEYS and self.gpu_open:
            self.gpu_nested += 1
        self.stack.append(key)
        self.child_windows.append(0.0)
        if gpu:
            self.gpu_open = True
            e0, e1 = _event(), _event()
            e0.record()
        h0 = time.perf_counter()
        try:
            out = fn(*a, **k)
        finally:
            h1 = time.perf_counter()
            children = self.child_windows.pop()
            self.stack.pop()
            if gpu:
                e1.record()
                self.pending.append((key, e0, e1))
                self.gpu_open = False
        t_leave = time.perf_counter()
        n = self.host.setdefault(key, {"calls": 0, "inclusive_ms": 0.0, "overhead_ms": 0.0,
                                       "children_window_ms": 0.0})
        n["calls"] += 1
        n["inclusive_ms"] += (h1 - h0) * 1000.0
        n["overhead_ms"] += ((t_leave - t_enter) - (h1 - h0)) * 1000.0
        n["children_window_ms"] += children * 1000.0
        self.child_windows[-1] += t_leave - t_enter
        return out

    # -- scope ------------------------------------------------------------
    def __enter__(self):
        self._install()
        self.e_start, self.e_end = _event(), _event()
        self.child_windows = [0.0]
        self.e_start.record()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        t1 = time.perf_counter()
        try:
            self.e_end.record()
            torch.cuda.synchronize()
        finally:
            self._restore()
        if exc_type is None:
            self.result = self._collect((t1 - self.t0) * 1000.0)
            logger.info("%s=%s", LOG_TAG, json.dumps(self.result, sort_keys=True, separators=(",", ":")))
        _EVENT_POOL.extend([self.e_start, self.e_end, *(e for _, p, q in self.pending for e in (p, q))])
        self.pending = []
        return False

    def _collect(self, wall_ms: float) -> dict:
        gpu: dict[str, dict] = {}
        for key, e0, e1 in self.pending:
            g = gpu.setdefault(key, {"calls": 0, "gpu_ms": 0.0})
            g["calls"] += 1
            g["gpu_ms"] += e0.elapsed_time(e1)
        host = {}
        for key, n in self.host.items():
            host[key] = {**n, "exclusive_ms": n["inclusive_ms"] - n["children_window_ms"]}
        root_children = self.child_windows[0] * 1000.0
        host[ROOT] = {"calls": 1, "inclusive_ms": wall_ms, "overhead_ms": 0.0, "children_window_ms": root_children,
                      "exclusive_ms": wall_ms - root_children}

        def rnd(d):
            return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in d.items()}

        return {"schema": SCHEMA, "impl": self.impl, "q_len": self.q_len, "context_len": self.context_len,
                "gpu_nested": self.gpu_nested,
                "host": {"wall_ms": round(wall_ms, 6), "windows": {k: rnd(v) for k, v in host.items()}},
                "gpu": {"span_ms": round(self.e_start.elapsed_time(self.e_end), 6),
                        "leaves": {k: rnd(v) for k, v in gpu.items()}}}

    def record(self) -> dict:
        return self.result
