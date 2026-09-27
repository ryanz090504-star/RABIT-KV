"""
Parser and attribution for Stage3C COMPONENT PROFILE records (pure Python; no
torch). Diagnostic only: nothing here is a latency result.

Input: engine log lines
    ... RABIT2_STAGE3C_COMPONENT_PROFILE={json}
emitted by vllm/v1/attention/ops/rabit_kv2_stage3c_profile.py, one per Stage3C
chunk call (= one per layer per request).

Accounting contract (HOST and GPU are separate domains and are NEVER combined):

GPU (CUDA events):
  total = total_stage3c_gpu_span_ms (scope event pair)
  mutually exclusive leaves (event pairs, never nested):
    chunk_plan_encode, closed_page, k_stats_codes, k3_pack, meta8g64,
    tail_partial (open_recent + recent_only), reduce
  unattributed_gpu = span - sum(leaves)   (device idle + un-evented work)
  share = component / span

HOST (perf_counter), exclusive times of a window tree (no double counting):
  total = total_stage3c_host_wall_ms
  mutually exclusive components:
    chunk_plan_host, apply_step_host,
    launch_host.{closed_page, tail_prep, tail_partial, reduce},
    wrapper_exclusive_host.{reference_attention_call, tail, tail_prep},
    profiler_overhead_host
  unattributed_host = wall - sum(components)   (= root exclusive: loop, output
                                                copy issue, tensor views, ...)
  share = component / wall

parse_record()  strict schema / arithmetic validation; raises ProfileError.
aggregate()     sums one request's records (all layers).
attribute()     the two domain tables above with absolute times and shares.
classify()      pre-registered per-domain rule outputs (for review only).
"""

from __future__ import annotations

import json
import math
import re

TAG = "RABIT2_STAGE3C_COMPONENT_PROFILE"
SCHEMA = "rabit2_stage3c_component_profile/v2"
LINE = re.compile(re.escape(TAG) + r"=(\{.*\})\s*$")
IMPLS = ("reference", "tile32")
TOL = 1e-4  # ms per record; records are rounded to 1e-6 ms

KERNELS = ("closed_page", "tail_prep.k_stats_codes", "tail_prep.k3_pack", "tail_prep.meta8g64",
           "tail_partial.open_recent", "tail_partial.recent_only", "reduce")
HOST_LEAVES = ("chunk_plan", "apply_step", *KERNELS)
HOST_PARENTS = ("reference_attention_call", "tail", "tail_prep")
ROOT = "chunk"
HOST_KNOWN = (ROOT, *HOST_PARENTS, *HOST_LEAVES)
GPU_LEAVES = ("chunk_plan", *KERNELS)
HOST_FIELDS = ("calls", "inclusive_ms", "overhead_ms", "children_window_ms", "exclusive_ms")
REQUIRED_HOST = {
    "reference": (ROOT, "chunk_plan", "apply_step", "reference_attention_call", "closed_page", "tail", "tail_prep",
                  "tail_prep.k_stats_codes", "tail_prep.k3_pack", "tail_prep.meta8g64", "tail_partial.open_recent",
                  "reduce"),
    "tile32": (ROOT, "chunk_plan", "apply_step", "closed_page", "tail", "tail_prep", "tail_prep.k_stats_codes",
               "tail_prep.k3_pack", "tail_prep.meta8g64", "tail_partial.open_recent", "reduce"),
}
FORBIDDEN_HOST = {"reference": (), "tile32": ("reference_attention_call",)}

# Mutually exclusive component definitions used in the sums (the ONLY ones).
GPU_COMPONENTS = {
    "chunk_plan_encode_gpu": ("chunk_plan",),
    "closed_page_gpu": ("closed_page",),
    "k_stats_codes_gpu": ("tail_prep.k_stats_codes",),
    "k3_pack_gpu": ("tail_prep.k3_pack",),
    "meta8g64_gpu": ("tail_prep.meta8g64",),
    "tail_partial_gpu": ("tail_partial.open_recent", "tail_partial.recent_only"),
    "reduce_gpu": ("reduce",),
}
GPU_GROUPS = {"tail_prep_gpu": ("k_stats_codes_gpu", "k3_pack_gpu", "meta8g64_gpu")}  # display only
HOST_COMPONENTS = {  # (window, field)
    "chunk_plan_host": (("chunk_plan", "exclusive_ms"),),
    "apply_step_host": (("apply_step", "exclusive_ms"),),
    "launch_host.closed_page": (("closed_page", "exclusive_ms"),),
    "launch_host.tail_prep": tuple((k, "exclusive_ms") for k in ("tail_prep.k_stats_codes", "tail_prep.k3_pack",
                                                                  "tail_prep.meta8g64")),
    "launch_host.tail_partial": (("tail_partial.open_recent", "exclusive_ms"),
                                 ("tail_partial.recent_only", "exclusive_ms")),
    "launch_host.reduce": (("reduce", "exclusive_ms"),),
    "wrapper_exclusive_host.reference_attention_call": (("reference_attention_call", "exclusive_ms"),),
    "wrapper_exclusive_host.tail": (("tail", "exclusive_ms"),),
    "wrapper_exclusive_host.tail_prep": (("tail_prep", "exclusive_ms"),),
    "profiler_overhead_host": tuple((k, "overhead_ms") for k in HOST_PARENTS + HOST_LEAVES),
}
HOST_GROUPS = {"orchestration_launch_host": tuple(k for k in HOST_COMPONENTS if k.startswith(("launch_host.",
                                                                                                "wrapper_exclusive")))}


def parent_of(key: str, impl: str) -> str | None:
    if key == ROOT:
        return None
    if key.startswith("tail_prep."):
        return "tail_prep"
    if key == "tail_prep" or key.startswith("tail_partial."):
        return "tail"
    if key in ("closed_page", "reduce", "tail"):
        return "reference_attention_call" if impl == "reference" else ROOT
    return ROOT  # chunk_plan, apply_step, reference_attention_call


class ProfileError(ValueError):
    pass


def _num(x, name):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ProfileError(f"{name}: not a finite number ({x!r})")
    return float(x)


def _nonneg(x, name):
    if _num(x, name) < 0:
        raise ProfileError(f"{name}: negative ({x!r})")
    return float(x)


def _int(x, name, minimum=0):
    if isinstance(x, bool) or not isinstance(x, int) or x < minimum:
        raise ProfileError(f"{name}: expected int >= {minimum} ({x!r})")
    return x


def find_records(lines) -> list[str]:
    """JSON payloads of every profile line (anything after the tag must be one JSON object)."""
    out = []
    for ln in lines:
        if TAG in ln:
            m = LINE.search(ln.rstrip("\r\n"))
            if not m:
                raise ProfileError(f"malformed profile line: {ln[:200]!r}")
            out.append(m.group(1))
    return out


def parse_record(payload: str) -> dict:
    try:
        r = json.loads(payload)
    except json.JSONDecodeError as e:
        raise ProfileError(f"profile payload is not JSON: {e}") from e
    if not isinstance(r, dict) or r.get("schema") != SCHEMA:
        raise ProfileError(f"unexpected schema {r.get('schema') if isinstance(r, dict) else r!r}")
    impl = r.get("impl")
    if impl not in IMPLS:
        raise ProfileError(f"unknown impl {impl!r}")
    _int(r.get("q_len"), "q_len", 2)
    _int(r.get("context_len"), "context_len", 1)
    if _int(r.get("gpu_nested"), "gpu_nested") != 0:
        raise ProfileError("nested GPU event windows (would double count)")
    host, gpu = r.get("host"), r.get("gpu")
    if not isinstance(host, dict) or not isinstance(gpu, dict):
        raise ProfileError("host / gpu domain missing")
    wall = _nonneg(host.get("wall_ms"), "host.wall_ms")
    span = _nonneg(gpu.get("span_ms"), "gpu.span_ms")
    win, leaves = host.get("windows"), gpu.get("leaves")
    if not isinstance(win, dict) or not isinstance(leaves, dict):
        raise ProfileError("host.windows / gpu.leaves missing")

    # ---- HOST domain
    unknown = sorted(set(win) - set(HOST_KNOWN))
    if unknown:
        raise ProfileError(f"unknown host window(s) {unknown}")
    missing = [k for k in REQUIRED_HOST[impl] if k not in win]
    if missing:
        raise ProfileError(f"{impl}: missing required host window(s) {missing}")
    bad = [k for k in FORBIDDEN_HOST[impl] if k in win]
    if bad:
        raise ProfileError(f"{impl}: host window(s) not allowed for this impl {bad}")
    for key, w in win.items():
        if not isinstance(w, dict) or any(f not in w for f in HOST_FIELDS):
            raise ProfileError(f"host.{key}: missing field(s)")
        _int(w["calls"], f"host.{key}.calls", 1)
        for f in ("inclusive_ms", "overhead_ms", "children_window_ms"):
            _nonneg(w[f], f"host.{key}.{f}")
        exc = _num(w["exclusive_ms"], f"host.{key}.exclusive_ms")
        if abs(exc - (w["inclusive_ms"] - w["children_window_ms"])) > TOL:
            raise ProfileError(f"host.{key}: exclusive_ms != inclusive_ms - children_window_ms")
        if exc < -TOL:
            raise ProfileError(f"host.{key}: negative exclusive time {exc} (nested timers inconsistent)")
        if key in HOST_LEAVES and w["children_window_ms"] > TOL:
            raise ProfileError(f"host.{key}: leaf window with instrumented children")
        p = parent_of(key, impl)
        if p is not None and p not in win:
            raise ProfileError(f"host.{key}: parent window {p} missing")
    root = win[ROOT]
    if root["calls"] != 1 or abs(root["inclusive_ms"] - wall) > TOL or root["overhead_ms"] != 0:
        raise ProfileError("host root window must be one call equal to wall_ms with no overhead")
    n = len(win)
    for p in (ROOT, *HOST_PARENTS):
        if p in win:
            kids = sum(w["inclusive_ms"] + w["overhead_ms"] for k, w in win.items() if parent_of(k, impl) == p)
            if abs(kids - win[p]["children_window_ms"]) > TOL * n:
                raise ProfileError(f"host.{p}: children_window_ms != sum of child windows")
    total = sum(w["exclusive_ms"] + w["overhead_ms"] for w in win.values())
    if abs(total - wall) > TOL * n:
        raise ProfileError("host: exclusive + overhead does not sum to wall_ms")

    # ---- GPU domain
    unknown = sorted(set(leaves) - set(GPU_LEAVES))
    if unknown:
        raise ProfileError(f"unknown gpu leaf(s) {unknown}")
    for key in GPU_LEAVES:
        if (key in leaves) != (key in win):
            raise ProfileError(f"gpu.{key}: event pair missing or without host window")
    for key, g in leaves.items():
        if not isinstance(g, dict) or "calls" not in g or "gpu_ms" not in g:
            raise ProfileError(f"gpu.{key}: missing field(s)")
        _int(g["calls"], f"gpu.{key}.calls", 1)
        _nonneg(g["gpu_ms"], f"gpu.{key}.gpu_ms")
        if g["calls"] != win[key]["calls"]:
            raise ProfileError(f"gpu.{key}: {g['calls']} event pairs for {win[key]['calls']} calls")
    if sum(g["gpu_ms"] for g in leaves.values()) > span + TOL * max(1, len(leaves)):
        raise ProfileError("gpu: sum of exclusive leaves exceeds the Stage3C span (overlap / double count)")
    return r


def aggregate(records: list[dict]) -> dict:
    """Sum one request's records (one per layer)."""
    if not records:
        raise ProfileError("no profile records")
    impls = {r["impl"] for r in records}
    qs = {(r["q_len"], r["context_len"]) for r in records}
    if len(impls) != 1 or len(qs) != 1:
        raise ProfileError(f"mixed impl / q_len within one request: {impls} {qs}")
    impl, (q_len, ctx) = impls.pop(), qs.pop()
    win: dict[str, dict] = {}
    leaves: dict[str, dict] = {}
    for r in records:
        for k, w in r["host"]["windows"].items():
            a = win.setdefault(k, {f: 0 for f in HOST_FIELDS})
            for f in HOST_FIELDS:
                a[f] += w[f]
        for k, g in r["gpu"]["leaves"].items():
            a = leaves.setdefault(k, {"calls": 0, "gpu_ms": 0.0})
            a["calls"] += g["calls"]
            a["gpu_ms"] += g["gpu_ms"]
    return {"impl": impl, "q_len": q_len, "context_len": ctx, "records": len(records),
            "host_wall_ms": sum(r["host"]["wall_ms"] for r in records),
            "gpu_span_ms": sum(r["gpu"]["span_ms"] for r in records),
            "host_windows": win, "gpu_leaves": leaves}


def attribute(agg: dict, num_layers: int) -> dict:
    win, leaves = agg["host_windows"], agg["gpu_leaves"]
    span, wall = agg["gpu_span_ms"], agg["host_wall_ms"]
    gpu = {name: sum(leaves.get(k, {}).get("gpu_ms", 0.0) for k in keys) for name, keys in GPU_COMPONENTS.items()}
    host = {name: sum(win.get(k, {}).get(f, 0.0) for k, f in parts) for name, parts in HOST_COMPONENTS.items()}
    un_gpu = span - sum(gpu.values())
    un_host = wall - sum(host.values())
    return {
        "impl": agg["impl"], "q_len": agg["q_len"], "context_len": agg["context_len"], "records": agg["records"],
        "records_expected": num_layers,
        "GPU": {
            "timing": "CUDA events; one synchronize per scope; never combined with HOST",
            "total_stage3c_gpu_span_ms": span,
            "components_ms": gpu,
            "groups_ms_display_only": {g: sum(gpu[c] for c in cs) for g, cs in GPU_GROUPS.items()},
            "unattributed_gpu_ms": un_gpu,
            "share_of_gpu_span": {k: v / span for k, v in gpu.items()} | {"unattributed_gpu": un_gpu / span},
            "sum_components": list(GPU_COMPONENTS),
        },
        "HOST": {
            "timing": "perf_counter; exclusive window times; never combined with GPU",
            "total_stage3c_host_wall_ms": wall,
            "components_ms": host,
            "groups_ms_display_only": {g: sum(host[c] for c in cs) for g, cs in HOST_GROUPS.items()},
            "unattributed_host_ms": un_host,
            "share_of_host_wall": {k: v / wall for k, v in host.items()} | {"unattributed_host": un_host / wall},
            "sum_components": list(HOST_COMPONENTS),
        },
        "call_counts": {k: v["calls"] for k, v in sorted(win.items())},
        "combined_boundaries": COMBINED,
        "diagnostic_only": True,
    }


# Which boundaries remain combined or unattributed (stated, not guessed).
COMBINED = {
    "chunk_plan_encode_gpu": "one event pair around Rabit2CausalChunkPlan.__init__ (V2 quantization, page encode, "
                             "cat/stack/index_copy_); its internal split is not measured.",
    "k_stats_codes_gpu": "K statistics and K codes are one fused kernel; not separable.",
    "meta8g64_gpu": "four META8g64 launches (k_min, k_scale, v_min, v_scale) summed.",
    "unattributed_gpu": "device idle inside the span plus un-evented device work (output copy_, contiguous copies "
                        "in wrappers); not separable.",
    "gpu_leaf_launch_gap": "if the device is idle when a leaf's start event is reached, the leaf time includes the "
                           "part of its host launch path after the start event.",
    "wrapper_exclusive_host": "per-query wrapper code (buffer allocation, contiguous(), layout arithmetic, "
                              "Stage4D3.4 workspace lookup) timed as window-exclusive time, not per statement.",
    "unattributed_host": "root exclusive: Stage3C loop overhead, output copy_ issue, tensor views, tile scheduling "
                         "and per-tile allocations (tile32); not separable without editing the timed code.",
}

# Pre-registered per-domain rules (fixed before any profile exists). Shares are within ONE domain.
DOMINANT = 0.50
LARGE = 0.15
ACTIONS = {
    "tail_prep_gpu": "batch/fuse tail preparation first",
    "tail_partial+reduce_gpu": "build multi-query tail/reduce kernels",
    "closed_page_gpu": "closed-page work still dominates: revisit closed-page batching",
    "chunk_plan_encode_gpu": "ChunkPlan encode dominates: vectorize the chunk-plan writer",
    "python_state_exposure_host": "vectorize/remove per-query Python state exposure",
    "per_query_launch_host": "proceed to a fuller batched chunk-attention path",
    "chunk_plan_host": "ChunkPlan construction dominates: vectorize the chunk-plan writer",
    "several_per_query": "proceed to a fuller batched chunk-attention path",
    "unattributed": "unattributed residual dominates: instrument further before optimizing",
    "none": "no pre-registered rule fires; review the full breakdown",
}


def classify(att: dict) -> dict:
    g, h = att["GPU"]["share_of_gpu_span"], att["HOST"]["share_of_host_wall"]
    gg = {"tail_prep_gpu": g["k_stats_codes_gpu"] + g["k3_pack_gpu"] + g["meta8g64_gpu"],
          "tail_partial+reduce_gpu": g["tail_partial_gpu"] + g["reduce_gpu"],
          "closed_page_gpu": g["closed_page_gpu"], "chunk_plan_encode_gpu": g["chunk_plan_encode_gpu"],
          "unattributed": g["unattributed_gpu"]}
    hh = {"python_state_exposure_host": h["apply_step_host"] + h["wrapper_exclusive_host.reference_attention_call"]
          + h["wrapper_exclusive_host.tail"] + h["wrapper_exclusive_host.tail_prep"],
          "per_query_launch_host": sum(v for k, v in h.items() if k.startswith("launch_host.")),
          "chunk_plan_host": h["chunk_plan_host"], "unattributed": h["unattributed_host"]}

    def rule(shares):
        top = max(shares, key=shares.get)
        return (top if shares[top] >= DOMINANT else "none"), top, shares[top]

    gr, gtop, gshare = rule(gg)
    hr, htop, hshare = rule(hh)
    per_query = {"tail_prep_gpu": gg["tail_prep_gpu"], "tail_partial_gpu": g["tail_partial_gpu"],
                 "reduce_gpu": g["reduce_gpu"], "python_state_exposure_host": hh["python_state_exposure_host"],
                 "per_query_launch_host": hh["per_query_launch_host"]}
    large = sorted(k for k, v in per_query.items() if v >= LARGE)
    return {
        "gpu_rule": gr, "gpu_action": ACTIONS[gr], "gpu_largest": gtop, "gpu_largest_share": gshare,
        "host_rule": hr, "host_action": ACTIONS[hr], "host_largest": htop, "host_largest_share": hshare,
        "several_per_query_large": len(large) >= 3, "per_query_components_ge_large_share": large,
        "several_per_query_action": ACTIONS["several_per_query"] if len(large) >= 3 else None,
        "thresholds": {"dominant_share": DOMINANT, "large_share": LARGE},
        "note": "per-domain shares only (HOST share of host wall; GPU share of GPU span); pre-registered rule output; "
                "requires review before any optimization",
    }
