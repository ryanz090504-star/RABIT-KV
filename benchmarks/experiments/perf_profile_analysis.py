"""
RABIT-KV performance-risk PROFILING DIAGNOSTIC -- pure-Python log parsing and attribution (no torch, no vLLM).
DIAGNOSTIC attribution only. Two domains are reported side by side and never added together:
  gpu_span_ms   CUDA event-pair span of a window on the device timeline (device work enqueued in the window plus
                any time the device waits for the host inside it);
  host_ms       perf_counter time of the window on the engine thread (inclusive; `exclusive` removes child windows).
Windows nest: step > attention > rabit.forward > branch > {leaf, kernel}. Shares are always relative to the summed
scheduler-step windows of the same domain.
"""

from __future__ import annotations

import json
import re

LEG_RE = re.compile(r"^\[leg:([A-Za-z0-9_]+)\] (.*)$")
BRANCHES = ("initial_prefill.bulk_append", "initial_prefill.dense_attention", "chunked_prefill.shared_decode",
            "decode.append_aging", "decode.attention")
STEP_KEYS = ("step.execute_model", "step.sample_tokens")


def split_legs(text: str) -> tuple[dict, list[str]]:
    legs: dict = {}
    top: list[str] = []
    for raw in text.splitlines():
        m = LEG_RE.match(raw)
        if m:
            legs.setdefault(m.group(1), []).append(m.group(2))
        else:
            top.append(raw)
    return legs, top


def tagged(lines: list[str], tag: str) -> list:
    out, prefix = [], tag + "="
    for line in lines:
        i = line.find(prefix)
        if i < 0 or (i > 0 and (line[i - 1].isalnum() or line[i - 1] == "_")):
            continue
        try:
            out.append(json.loads(line[i + len(prefix):]))
        except json.JSONDecodeError:
            out.append({"__unparsed__": line[:300]})
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def request_stats(rows: list[dict]) -> dict:
    ttft, tpot = [], []
    for r in rows:
        if r.get("first_token_latency") is not None:
            ttft.append(r["first_token_latency"] * 1000.0)
        if r.get("first_token_ts") is not None and r.get("last_token_ts") is not None and r.get("output_tokens", 0) > 1:
            tpot.append((r["last_token_ts"] - r["first_token_ts"]) / (r["output_tokens"] - 1) * 1000.0)
    sched_to_first = [(r["first_token_ts"] - r["scheduled_ts"]) * 1000.0 for r in rows
                      if r.get("first_token_ts") is not None and r.get("scheduled_ts") is not None]
    return {"requests": len(rows), "mean_ttft_ms": _mean(ttft), "mean_tpot_ms": _mean(tpot),
            "mean_scheduled_to_first_token_ms": _mean(sched_to_first),
            "output_tokens": sorted({r.get("output_tokens") for r in rows}),
            "prompt_tokens": sorted({r.get("prompt_tokens") for r in rows}),
            "finish_reasons": sorted({str(r.get("finish_reason")) for r in rows})}


def breakdown(profile: dict) -> dict:
    """Attribution of one phase profile (see module docstring)."""
    host, gpu = profile["host"], profile["gpu"]

    def h(key, field="inclusive_ms"):
        return host.get(key, {}).get(field, 0.0)

    def g(key):
        return gpu.get(key, {}).get("gpu_ms", 0.0)

    def calls(key):
        return host.get(key, {}).get("calls", 0)

    step_h = sum(h(k) for k in STEP_KEYS)
    step_g = sum(g(k) for k in STEP_KEYS)

    def row(name, hv, gv, n=None, extra=None):
        d = {"component": name, "host_ms": hv, "gpu_span_ms": gv,
             "host_share": hv / step_h if step_h else None, "gpu_span_share": gv / step_g if step_g else None}
        if n is not None:
            d["calls"] = n
        if extra:
            d.update(extra)
        return d

    att_h, att_g = h("attention.forward"), g("attention.forward")
    cu_h, cu_g = h("attention.bf16_cache_update"), g("attention.bf16_cache_update")
    rab_h, rab_g = h("rabit.forward"), g("rabit.forward")
    br_h, br_g = sum(h(b) for b in BRANCHES), sum(g(b) for b in BRANCHES)
    rabit = bool(calls("rabit.forward"))
    level1 = [row("attention.forward (all attention-layer work)", att_h, att_g, calls("attention.forward")),
              row("attention.bf16_cache_update (BF16 KV write; no-op for RABIT)", cu_h, cu_g,
                  calls("attention.bf16_cache_update")),
              row("step.sample_tokens", h("step.sample_tokens"), g("step.sample_tokens"), calls("step.sample_tokens")),
              row("OTHER MODEL TIME (execute_model minus attention windows)",
                  h("step.execute_model") - att_h - cu_h, g("step.execute_model") - att_g - cu_g)]
    components = []
    if rabit:
        for b in BRANCHES:
            extra = {x: host[b][x] for x in ("tokens", "context_tokens", "returned_true", "returned_false")
                     if b in host and x in host[b]}
            components.append(row(b, h(b), g(b), calls(b), extra))
        components.append(row("HOST/CONTROL: per-request loop in _forward_rabit_kv2 (rabit.forward minus branches)",
                              rab_h - br_h, rab_g - br_g, calls("rabit.forward")))
        components.append(row("attention.forward wrapper outside rabit.forward", att_h - rab_h, att_g - rab_g))
    sub = {}
    for b in BRANCHES:
        if not calls(b):
            continue
        kern = sorted(((k.split("/", 1)[1], host[k]) for k in host if k.startswith(b + "/kernel.")),
                      key=lambda kv: -gpu.get(b + "/" + kv[0], {}).get("gpu_ms", 0.0))
        leaf = sorted(((k.split("/", 1)[1], host[k]) for k in host if k.startswith(b + "/leaf.")),
                      key=lambda kv: -kv[1]["inclusive_ms"])
        ksum_h = sum(v["inclusive_ms"] for _, v in kern)
        ksum_g = sum(g(b + "/" + n) for n, _ in kern)
        kcalls = sum(v["calls"] for _, v in kern)
        sub[b] = {
            "branch": row(b, h(b), g(b), calls(b)),
            "kernels": [row(n, v["inclusive_ms"], g(b + "/" + n), v["calls"]) for n, v in kern],
            "kernel_launches": kcalls,
            "kernel_windows_total": row("sum of kernel-launch windows", ksum_h, ksum_g, kcalls),
            "outside_kernel_windows": row("branch minus kernel-launch windows (Python / torch ops between launches)",
                                          h(b) - ksum_h, g(b) - ksum_g),
            "host_ms_per_kernel_launch": ksum_h / kcalls if kcalls else None,
            "leaves_inclusive_of_their_kernels": [
                row(n, v["inclusive_ms"], g(b + "/" + n), v["calls"], {"host_exclusive_ms": v["exclusive_ms"]})
                for n, v in leaf],
        }
    kinds = {}
    for k, v in host.items():
        if k.startswith("stepkind."):
            kinds[k.split(".", 1)[1]] = {
                "steps": v["calls"], "host_ms": v["inclusive_ms"], "gpu_span_ms": g(k), "tokens": v.get("tokens"),
                "sequence_slots": v.get("seqs"), "host_ms_per_step": v["inclusive_ms"] / v["calls"],
                "gpu_span_ms_per_step": g(k) / v["calls"],
                "mean_sequences_per_step": v.get("seqs", 0) / v["calls"],
                "host_share": v["inclusive_ms"] / step_h if step_h else None,
                "gpu_span_share": g(k) / step_g if step_g else None}
    all_kernel_calls = sum(v["calls"] for k, v in host.items() if "/kernel." in k)
    all_kernel_host = sum(v["inclusive_ms"] for k, v in host.items() if "/kernel." in k)
    all_kernel_gpu = sum(v["gpu_ms"] for k, v in gpu.items() if "/kernel." in k)
    excl = sorted(((k, v["exclusive_ms"], v["calls"]) for k, v in host.items() if not k.startswith("stepkind.")),
                  key=lambda t: -t[1])
    overhead = sum(v["overhead_ms"] for k, v in host.items() if not k.startswith("stepkind."))
    return {
        "steps": profile["steps"], "syncs": profile["syncs"], "meta_errors": profile.get("meta_errors"),
        "wall_first_to_last_step_s": profile.get("wall_first_to_last_step_s"),
        "step_windows_total": {"host_ms": step_h, "gpu_span_ms": step_g},
        "level1": level1, "rabit_components": components, "branch_detail": sub, "step_kinds": kinds,
        "kernel_launch_totals": {
            "launches": all_kernel_calls, "host_ms_in_launch_windows": all_kernel_host,
            "gpu_span_ms_in_launch_windows": all_kernel_gpu,
            "host_ms_per_launch": all_kernel_host / all_kernel_calls if all_kernel_calls else None,
            "launches_per_step": all_kernel_calls / profile["steps"] if profile["steps"] else None,
            "host_share": all_kernel_host / step_h if step_h else None,
            "gpu_span_share": all_kernel_gpu / step_g if step_g else None},
        "host_exclusive_top": [{"window": k, "host_exclusive_ms": e, "calls": n,
                                "host_share": e / step_h if step_h else None} for k, e, n in excl[:25]],
        "profiler_bookkeeping_overhead_ms": overhead,
        "windows": sum(v["calls"] for k, v in host.items() if not k.startswith("stepkind.")),
    }


def leg_report(lines: list[str]) -> dict:
    leg = (tagged(lines, "PERF_LEG") or [None])[0]
    summary = (tagged(lines, "PERF_MEASURED_SUMMARY") or [None])[0]
    reqs = tagged(lines, "PERF_REQUEST")
    profiles = {p.get("phase"): p for p in tagged(lines, "RABIT_PERF_PROFILE")}
    installs = tagged(lines, "RABIT_PERF_INSTALL")
    torch_rows = tagged(lines, "RABIT_PERF_TORCH_PROFILE")
    util = (tagged(lines, "PERF_GPU_UTILIZATION") or [None])[0]
    out = {
        "leg": leg, "complete": any(line.strip() == "PERF_WORKER_COMPLETE" for line in lines),
        "failure": tagged(lines, "PERF_REQUEST_FAILURE"),
        "effective_engine_config": (tagged(lines, "PERF_EFFECTIVE_ENGINE_CONFIG") or [None])[0],
        "stage_impl": (tagged(lines, "PERF_STAGE_IMPL") or [None])[0],
        "workload": (tagged(lines, "PERF_WORKLOAD") or [None])[0],
        "plugin_module_loaded": (tagged(lines, "PERF_PLUGIN_MODULE_LOADED") or [None])[0],
        "installs": installs, "measured_summary": summary, "measured_calls": tagged(lines, "PERF_MEASURED_CALL"),
        "request_stats": request_stats(reqs),
        "output_token_ids_sha256": [r.get("output_token_ids_sha256") for r in reqs],
        "gpu_utilization": ({k: v for k, v in util.items() if k != "utilization_gpu_pct"} if util else None),
        "profile_phases": sorted(p for p in profiles if p),
        "torch_profiler": torch_rows[0] if torch_rows else None,
        "multi_sequence_logged": any("RABIT2_STAGE3C_MULTI_SEQUENCE_ACTIVE" in line for line in lines),
        "chunked_prefill_logged": [line[line.find("RABIT2_STAGE3C_CHUNKED_PREFILL_ACTIVE"):][:120] for line in lines
                                   if "RABIT2_STAGE3C_CHUNKED_PREFILL_ACTIVE" in line][:1],
    }
    if summary and summary.get("wall_s"):
        out["requests_per_s"] = summary["requests"] / summary["wall_s"]
    for phase in ("measured", "warmup", "conditioning"):
        if phase in profiles:
            out[f"breakdown_{phase}"] = breakdown(profiles[phase])
    if "measured" in profiles:
        out["measured_profile_raw"] = profiles["measured"]
    return out


def overhead(unprofiled: dict, profiled: dict) -> dict:
    """Instrumentation overhead of a profiled leg against its unprofiled reference (same case, same dtype)."""
    u, p = unprofiled.get("measured_summary") or {}, profiled.get("measured_summary") or {}
    out = {"unprofiled_wall_s": u.get("wall_s"), "profiled_wall_s": p.get("wall_s"),
           "unprofiled_requests": u.get("requests"), "profiled_requests": p.get("requests")}
    if u.get("wall_s") and p.get("wall_s") and u.get("requests") == p.get("requests"):
        out["wall_ratio_profiled_over_unprofiled"] = p["wall_s"] / u["wall_s"]
        out["overhead_pct"] = (p["wall_s"] / u["wall_s"] - 1.0) * 100.0
    elif u.get("wall_s") and p.get("wall_s"):
        ur, pr = u["requests"] / u["wall_s"], p["requests"] / p["wall_s"]
        out["note"] = "different request counts; compared by requests per second"
        out["wall_ratio_profiled_over_unprofiled"] = ur / pr
        out["overhead_pct"] = (ur / pr - 1.0) * 100.0
    a, b = unprofiled.get("output_token_ids_sha256") or [], profiled.get("output_token_ids_sha256") or []
    n = min(len(a), len(b))
    out["output_hash_agreement"] = {"compared": n, "equal": sum(1 for i in range(n) if a[i] == b[i])}
    return out
