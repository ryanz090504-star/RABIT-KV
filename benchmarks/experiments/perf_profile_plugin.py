"""
RABIT-KV performance-risk PROFILER (DIAGNOSTIC ONLY; attribution, never latency). Loaded into the vLLM engine process
as a general plugin (entry-point group vllm.general_plugins) by perf_profile_worker.py -- it attaches from OUTSIDE:
no file of the vendored vLLM tree is edited (rabit_kv2.py, triton_attn.py and the Stage3C modules stay byte-identical
to the accepted serving evidence). When the plugin directory is not on sys.path / PYTHONPATH nothing of this module
exists in the process (the unprofiled reference legs).

It replaces call-time lookups with timing wrappers that call the unmodified originals with unmodified arguments:
    step        Worker.execute_model / Worker.sample_tokens            (one scheduler step)
    attention   TritonAttentionImpl.forward, .do_kv_cache_update       (per layer; BF16 and RABIT)
    rabit       TritonAttentionImpl._forward_rabit_kv2                 (per layer: the per-request Python loop)
    branches    rabit2_bulk_append_exact, context_attention_fwd        (initial prefill: cache build, dense attention)
                rabit2_stage3c_forward_shared_decode                   (later / chunked prefill)
                Rabit2SingleSequenceRuntime.append                     (decode: cache update / aging)
                rabit2_online_decode_attention_triton                  (decode: packed-page attention)
    leaves      Rabit2CausalChunkPlan.__init__ / apply_step, rabit2_shared_decode_closed_pages, the tail helpers and
                every Triton kernel launch K[grid](...) of rabit_kv2.py / rabit_kv2_stage3c_shared_decode.py /
                rabit_kv2_stage3c_tile32.py

Two domains, never combined (same contract as rabit_kv2_stage3c_profile.py):
  HOST  time.perf_counter window tree: inclusive, bookkeeping overhead, children windows, exclusive.
  GPU   CUDA event pairs: one pair around every window; elapsed times are read after ONE torch.cuda.synchronize() at
        the end of a scheduler step (or earlier when many events are pending). An event pair measures the span of the
        GPU timeline between the two record points: the device work enqueued in the window PLUS any time the device
        waits for the host to enqueue it. Windows nest (step > attention > rabit > branch > leaf), so shares are
        computed within one level, never by adding levels.
Totals are accumulated per PHASE (the worker writes the phase name to a file; it is read once per step) and one
    RABIT_PERF_PROFILE={json}
line is logged when a phase ends. Mode `torch_profiler` records ONE torch.profiler trace of the measured phase instead
(cross-check only; no CUDA-event windows are installed in that mode).
"""

from __future__ import annotations

import functools
import json
import os
import re
import sys
import time

MODE_ENV = "RABIT_PERF_PROFILE"  # "cuda_events" | "torch_profiler"
PHASE_FILE_ENV = "RABIT_PERF_PHASE_FILE"
LOG_TAG, INSTALL_TAG, TORCH_TAG = "RABIT_PERF_PROFILE", "RABIT_PERF_INSTALL", "RABIT_PERF_TORCH_PROFILE"
SCHEMA = "rabit_perf_profile/v1"
MAX_PENDING_EVENTS = 60000
LEVEL = {"step.execute_model": 0, "step.sample_tokens": 0, "attention.forward": 1, "attention.bf16_cache_update": 1,
         "rabit.forward": 2, "initial_prefill.bulk_append": 3, "initial_prefill.dense_attention": 3,
         "chunked_prefill.shared_decode": 3, "decode.append_aging": 3, "decode.attention": 3}
BRANCHES = tuple(k for k, v in LEVEL.items() if v == 3)
KERNEL_LAUNCH_RE = re.compile(r"\b(_rabit2\w*_kernel)\[")

_state = None


def _log(tag: str, payload: dict) -> None:
    print(f"{tag}={json.dumps(payload, sort_keys=True, separators=(',', ':'))}", flush=True)


# -- shape metadata (integers summed per window key; never changes a call) ----------------------------------------
def _meta_step(a, k):
    n = list((a[1] if len(a) > 1 else k["scheduler_output"]).num_scheduled_tokens.values())
    kind = "empty" if not n else "decode_only" if max(n) == 1 else "prefill_only" if min(n) > 1 else "mixed"
    return {"tokens": sum(n), "seqs": len(n)}, f"stepkind.{kind}"


def _meta_attention(idx):
    def meta(a, k):
        md = a[idx] if len(a) > idx else k.get("attn_metadata")
        if md is None:
            return {}, None
        return {"tokens": int(md.num_actual_tokens), "seqs": int(md.query_start_loc.shape[0]) - 1}, None

    return meta


def _meta_seq(a, k):
    return {"tokens": int(a[1].shape[0]), "context_tokens": int(a[0].total_tokens)}, None


def _meta_dense(a, k):
    return {"tokens": int((k["q"] if "q" in k else a[0]).shape[0])}, None


class Profiler:
    """Window tree + CUDA events, accumulated per phase. `torch` may be None in unit tests (host domain only)."""

    def __init__(self, torch=None, phase_file: str | None = None) -> None:
        self.torch = torch
        self.gpu = None  # decided at the first window (inside the engine process; never initializes CUDA)
        self.phase_file, self.phase_mtime, self.phase = phase_file, None, "setup"
        self.acc = self._empty()
        self.stack: list[str] = []
        self.child_windows: list[float] = [0.0]
        self.pending: list[tuple] = []
        self.pool: list = []
        self.depth = 0

    @staticmethod
    def _empty() -> dict:
        return {"host": {}, "gpu": {}, "steps": 0, "syncs": 0, "meta_errors": 0, "t_first": None, "t_last": None}

    def _event(self):
        return self.pool.pop() if self.pool else self.torch.cuda.Event(enable_timing=True)

    def _branch(self):
        for name in reversed(self.stack):
            if name in BRANCHES:
                return name
        return None

    # -- phase ---------------------------------------------------------------------------------------------
    def poll_phase(self) -> None:
        """Called at the start of a scheduler step (outside any window)."""
        if not self.phase_file:
            return
        try:
            m = os.stat(self.phase_file).st_mtime_ns
        except OSError:
            return
        if m == self.phase_mtime:
            return
        self.phase_mtime = m
        with open(self.phase_file, encoding="utf-8") as fh:
            new = fh.read().strip() or self.phase
        if new != self.phase:
            self.dump()
            self.phase, self.acc = new, self._empty()

    def report(self) -> dict:
        a = self.acc
        host = {}
        for key, v in a["host"].items():
            row = {x: (round(y, 6) if isinstance(y, float) else y) for x, y in v.items()}
            row["exclusive_ms"] = round(v["inclusive_ms"] - v["children_window_ms"], 6)
            host[key] = row
        return {"schema": SCHEMA, "phase": self.phase, "pid": os.getpid(), "steps": a["steps"], "syncs": a["syncs"],
                "meta_errors": a["meta_errors"], "gpu_domain": bool(self.gpu),
                "wall_first_to_last_step_s": (a["t_last"] - a["t_first"]) if a["t_first"] is not None else None,
                "host": host,
                "gpu": {key: {"calls": v["calls"], "gpu_ms": round(v["gpu_ms"], 6)} for key, v in a["gpu"].items()},
                "levels": LEVEL}

    def dump(self) -> None:
        self.resolve()
        _log(LOG_TAG, self.report())

    def resolve(self) -> None:
        if not self.pending:
            return
        self.torch.cuda.synchronize()
        self.acc["syncs"] += 1
        g = self.acc["gpu"]
        for keys, e0, e1 in self.pending:
            ms = e0.elapsed_time(e1)
            for key in keys:
                n = g.setdefault(key, {"calls": 0, "gpu_ms": 0.0})
                n["calls"] += 1
                n["gpu_ms"] += ms
            self.pool.append(e0)
            self.pool.append(e1)
        self.pending = []

    # -- timing --------------------------------------------------------------------------------------------
    def window(self, key: str, fn, a, k, meta=None):
        if self.depth == 0 and key.startswith("step."):
            self.poll_phase()
        t_enter = time.perf_counter()
        if self.gpu is None:
            self.gpu = bool(self.torch is not None and self.torch.cuda.is_initialized())
        sums, alias = {}, None
        if meta is not None:
            try:
                sums, alias = meta(a, k)
            except Exception:  # noqa: BLE001  (metadata only; never changes the call)
                self.acc["meta_errors"] += 1
        if key.startswith("kernel.") or key.startswith("leaf."):
            key = f"{self._branch() or 'no_branch'}/{key}"
        keys = (key, alias) if alias else (key,)
        self.stack.append(key)
        self.child_windows.append(0.0)
        self.depth += 1
        gpu = self.gpu
        if gpu:
            e0, e1 = self._event(), self._event()
            e0.record()
        result = None
        h0 = time.perf_counter()
        try:
            result = fn(*a, **k)
            return result
        finally:
            h1 = time.perf_counter()
            if gpu:
                e1.record()
                self.pending.append((keys, e0, e1))
            children = self.child_windows.pop()
            self.stack.pop()
            self.depth -= 1
            if gpu and (len(self.pending) >= MAX_PENDING_EVENTS or (self.depth == 0 and key.startswith("step."))):
                self.resolve()
            t_leave = time.perf_counter()
            acc = self.acc
            for name in keys:
                n = acc["host"].setdefault(name, {"calls": 0, "inclusive_ms": 0.0, "overhead_ms": 0.0,
                                                  "children_window_ms": 0.0})
                n["calls"] += 1
                n["inclusive_ms"] += (h1 - h0) * 1000.0
                n["overhead_ms"] += ((t_leave - t_enter) - (h1 - h0)) * 1000.0
                n["children_window_ms"] += children * 1000.0
                for s, val in sums.items():
                    n[s] = n.get(s, 0) + val
                if result is True:
                    n["returned_true"] = n.get("returned_true", 0) + 1
                elif result is False:
                    n["returned_false"] = n.get("returned_false", 0) + 1
            self.child_windows[-1] += t_leave - t_enter
            if self.depth == 0:
                self.child_windows[0] = 0.0
                if key == "step.execute_model":
                    acc["steps"] += 1
                if acc["t_first"] is None:
                    acc["t_first"] = t_enter
                acc["t_last"] = t_leave

    def wrap(self, key: str, fn, meta=None):
        @functools.wraps(fn)
        def wrapper(*a, **k):
            return self.window(key, fn, a, k, meta)

        wrapper.__rabit_perf_wrapped__ = True
        return wrapper


class KernelProxy:
    """Times ``kernel[grid](*args, **kwargs)``; forwards everything else to the unmodified Triton kernel."""

    def __init__(self, prof: Profiler, key: str, kernel) -> None:
        self._prof, self._key, self._kernel = prof, key, kernel

    def __getitem__(self, grid):
        launch = self._kernel[grid]
        return lambda *a, **k: self._prof.window(self._key, launch, a, k)

    def __getattr__(self, name):
        return getattr(self._kernel, name)


def _patch(prof: Profiler, owner, name: str, key: str, patched: list, meta=None) -> None:
    fn = owner.__dict__[name]
    if getattr(fn, "__rabit_perf_wrapped__", False):
        return
    setattr(owner, name, prof.wrap(key, fn, meta))
    patched.append(f"{owner.__name__}.{name}->{key}")


def _install_cuda_events(prof: Profiler) -> list:
    import vllm.v1.attention.backends.triton_attn as ta
    import vllm.v1.attention.ops.rabit_kv2 as r
    import vllm.v1.attention.ops.rabit_kv2_stage3c_shared_decode as sd
    import vllm.v1.attention.ops.rabit_kv2_stage3c_tile32 as t32
    import vllm.v1.worker.gpu_worker as gw

    patched: list = []
    _patch(prof, gw.Worker, "execute_model", "step.execute_model", patched, _meta_step)
    if "sample_tokens" in gw.Worker.__dict__:
        _patch(prof, gw.Worker, "sample_tokens", "step.sample_tokens", patched)
    impl = ta.TritonAttentionImpl
    _patch(prof, impl, "forward", "attention.forward", patched, _meta_attention(6))
    _patch(prof, impl, "do_kv_cache_update", "attention.bf16_cache_update", patched)
    _patch(prof, impl, "_forward_rabit_kv2", "rabit.forward", patched, _meta_attention(5))
    # the branch callables are looked up at call time in the triton_attn module namespace / on the runtime class
    _patch(prof, ta, "rabit2_bulk_append_exact", "initial_prefill.bulk_append", patched, _meta_seq)
    _patch(prof, ta, "context_attention_fwd", "initial_prefill.dense_attention", patched, _meta_dense)
    _patch(prof, ta, "rabit2_stage3c_forward_shared_decode", "chunked_prefill.shared_decode", patched, _meta_seq)
    _patch(prof, ta, "rabit2_online_decode_attention_triton", "decode.attention", patched)
    _patch(prof, r.Rabit2SingleSequenceRuntime, "append", "decode.append_aging", patched)
    _patch(prof, r.Rabit2CausalChunkPlan, "__init__", "leaf.chunk_plan_init", patched)
    _patch(prof, r.Rabit2CausalChunkPlan, "apply_step", "leaf.chunk_plan_apply_step", patched)
    _patch(prof, sd, "rabit2_shared_decode_closed_pages", "leaf.shared_decode_closed_pages", patched)
    for name, key in (("_rabit2_stage4b1_exactmeta_emit_tail_partial", "leaf.tail_emit"),
                      ("_rabit2_stage4d3_4_fast_prep", "leaf.tail_prep")):
        if name in r.__dict__:
            _patch(prof, r, name, key, patched)
    # every Triton kernel that is LAUNCHED (K[grid](...)) from these modules, in every module namespace that holds it
    launched: set = set()
    for mod in (r, sd, t32):
        with open(mod.__file__, encoding="utf-8") as fh:
            launched |= set(KERNEL_LAUNCH_RE.findall(fh.read()))
    for name in sorted(launched):
        for owner in (r, sd, t32):
            obj = owner.__dict__.get(name)
            if obj is not None and not isinstance(obj, KernelProxy) and hasattr(obj, "__getitem__"):
                setattr(owner, name, KernelProxy(prof, f"kernel.{name}", obj))
                patched.append(f"{owner.__name__}.{name}->kernel")
    return sorted(set(patched))


class _TorchProfilerPhase:
    """One torch.profiler trace over the `measured` phase (cross-check only)."""

    def __init__(self, torch, phase_file: str) -> None:
        self.torch, self.phase_file, self.mtime, self.phase, self.prof = torch, phase_file, None, "setup", None
        self.t0 = None

    def step(self, fn, a, k):
        try:
            m = os.stat(self.phase_file).st_mtime_ns
            if m != self.mtime:
                self.mtime = m
                with open(self.phase_file, encoding="utf-8") as fh:
                    new = fh.read().strip() or self.phase
                if new != self.phase:
                    self._switch(new)
        except OSError:
            pass
        return fn(*a, **k)

    def _switch(self, new: str) -> None:
        tp = self.torch.profiler
        if self.phase == "measured" and self.prof is not None:
            try:
                self.torch.cuda.synchronize()
                traced_s = time.perf_counter() - self.t0
                self.prof.__exit__(None, None, None)
                rows = []
                for e in self.prof.key_averages():
                    dev = getattr(e, "self_device_time_total", None)
                    if dev is None:
                        dev = getattr(e, "self_cuda_time_total", 0.0)
                    rows.append({"key": e.key, "count": int(e.count), "self_cpu_ms": e.self_cpu_time_total / 1000.0,
                                 "self_device_ms": dev / 1000.0})
                _log(TORCH_TAG, {"ok": True, "rows": len(rows), "traced_wall_s": traced_s,
                                 "total_self_device_ms": sum(x["self_device_ms"] for x in rows),
                                 "total_self_cpu_ms": sum(x["self_cpu_ms"] for x in rows),
                                 "top_by_self_device_ms": sorted(rows, key=lambda x: -x["self_device_ms"])[:60],
                                 "top_by_self_cpu_ms": sorted(rows, key=lambda x: -x["self_cpu_ms"])[:60]})
            except Exception as exc:  # noqa: BLE001  (cross-check only; never fails the leg)
                _log(TORCH_TAG, {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:1500]})
            self.prof = None
        if new == "measured":
            try:
                self.prof = tp.profile(activities=[tp.ProfilerActivity.CPU, tp.ProfilerActivity.CUDA])
                self.prof.__enter__()
                self.t0 = time.perf_counter()
            except Exception as exc:  # noqa: BLE001
                self.prof = None
                _log(TORCH_TAG, {"ok": False, "error": f"start: {type(exc).__name__}: {exc}"[:1500]})
        self.phase = new


def install() -> None:
    """vLLM general-plugin entry point; idempotent; a no-op unless RABIT_PERF_PROFILE is set."""
    global _state
    mode = os.environ.get(MODE_ENV, "")
    if _state is not None or mode not in ("cuda_events", "torch_profiler"):
        return
    try:
        import torch

        phase_file = os.environ[PHASE_FILE_ENV]
        if mode == "cuda_events":
            prof = Profiler(torch, phase_file)
            patched = _install_cuda_events(prof)
            _state = prof
        else:
            import vllm.v1.worker.gpu_worker as gw

            tprof = _TorchProfilerPhase(torch, phase_file)
            original = gw.Worker.__dict__["execute_model"]

            @functools.wraps(original)
            def execute_model(*a, **k):
                return tprof.step(original, a, k)

            gw.Worker.execute_model = execute_model
            patched, _state = ["vllm.v1.worker.gpu_worker.Worker.execute_model->torch_profiler_phase"], tprof
        _log(INSTALL_TAG, {"ok": True, "mode": mode, "pid": os.getpid(), "patched": patched,
                           "argv0": os.path.basename(sys.argv[0]) if sys.argv else None})
    except Exception as exc:  # noqa: BLE001  (reported; the runner invalidates the leg when no ok line is seen)
        _log(INSTALL_TAG, {"ok": False, "mode": mode, "pid": os.getpid(),
                           "error": f"{type(exc).__name__}: {exc}"[:1500]})
