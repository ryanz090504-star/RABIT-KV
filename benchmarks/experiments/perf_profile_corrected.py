"""
RABIT-KV performance-risk profiling diagnostic -- POST-RUN ADDENDUM (pure Python; reads the archived legs.json /
summary.json of attempt 1 and writes corrected_attribution.json next to them; nothing archived is rewritten).

Why: the CUDA-event profiler perturbed the RABIT legs far more than intended (profiled / unprofiled wall 3.7x, 6.3x,
8.8x for cases 1, 2, 3). Every window recorded its own bookkeeping time (`overhead_ms`, dominated by two
torch.cuda.Event.record() calls per window), but a parent's INCLUSIVE time contains the bookkeeping of all windows
below it, so the inclusive shares in summary.json over-weight the windows with the most children.

Correction used here: the host EXCLUSIVE time of a window (inclusive minus the full child windows, i.e. minus the
children AND their bookkeeping) contains no profiler bookkeeping of other windows. A flat profile of exclusive times
therefore sums to the overhead-corrected step time, and a subtree's corrected time is the sum of the exclusive times
of the windows in it. The residual against the unprofiled wall is reported per case; it is NOT removed.
The GPU-span domain cannot be corrected the same way (while the host is bookkeeping, the device idles inside the
span); it is used only for the two kernels whose launches are few and long, where span ~= device time.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "mlsys2027" / "diagnostics" / "perf_risk_profile" / "attempt_1"
BRANCHES = ("initial_prefill.bulk_append", "initial_prefill.dense_attention", "chunked_prefill.shared_decode",
            "decode.append_aging", "decode.attention")
TAIL_PATH = ("leaf.tail_emit", "leaf.tail_prep", "kernel._rabit2_stage4d3_4_k_stats_codes_kernel",
             "kernel._rabit2_stage4d3_4_pack_k3_kernel", "kernel._rabit2_stage4d3_4_meta64_kernel",
             "kernel._rabit2_stage4b1_exactmeta_tail_partial_kernel", "kernel._rabit2_tail_partial_kernel")
CLOSED_PAGE = ("leaf.shared_decode_closed_pages", "kernel._rabit2_shared_decode_closed_page_partial_kernel",
               "kernel._rabit2_stage4b3_gqa4_closed_page_partial_kernel")
REDUCE = ("kernel._rabit2_tile32_reduce_partials_kernel", "kernel._rabit2_reduce_partials_kernel")
LONG_KERNELS = ("chunked_prefill.shared_decode/kernel._rabit2_shared_decode_closed_page_partial_kernel",)


def corrected(profile: dict, unprofiled_wall_s: float, profiled_wall_s: float, bf16_wall_s: float) -> dict:
    host, gpu = profile["host"], profile["gpu"]
    keys = [k for k in host if not k.startswith("stepkind.")]
    ex = {k: host[k]["exclusive_ms"] for k in keys}
    total = sum(ex.values())
    overhead = sum(host[k]["overhead_ms"] for k in keys)

    def part(name, ks, calls=None):
        ms = sum(ex.get(k, 0.0) for k in ks)
        d = {"component": name, "corrected_host_ms": ms, "share_of_corrected_step_time": ms / total}
        if calls is not None:
            d["calls"] = calls
        return d

    def n(k):
        return host.get(k, {}).get("calls", 0)

    groups = [part("OTHER MODEL TIME (execute_model / sample_tokens / attention wrapper, own time)",
                   ["step.execute_model", "step.sample_tokens", "attention.forward", "attention.bf16_cache_update"]),
              part("HOST/CONTROL: per-request Python loop in _forward_rabit_kv2 (own time)", ["rabit.forward"],
                   n("rabit.forward"))]
    detail = {}
    for b in BRANCHES:
        sub = [k for k in keys if k == b or k.startswith(b + "/")]
        groups.append(part(f"{b} (whole subtree)", sub, n(b)))
        if not n(b):
            continue
        rest = [k for k in sub if k != b and k.split("/", 1)[1] not in TAIL_PATH + CLOSED_PAGE + REDUCE]
        detail[b] = [
            part("branch own Python / torch ops", [b], n(b)),
            part("per-token open-tail path: tail_emit + tail_prep + their 6-7 kernel launches",
                 [f"{b}/{x}" for x in TAIL_PATH], n(f"{b}/leaf.tail_emit")),
            part("closed-page (packed-page) attention launches", [f"{b}/{x}" for x in CLOSED_PAGE]),
            part("reduce-partials launches", [f"{b}/{x}" for x in REDUCE]),
        ] + [part(k.split("/", 1)[1], [k], n(k)) for k in sorted(rest, key=lambda k: -ex[k])]
    tail_all = [k for k in keys if "/" in k and k.split("/", 1)[1] in TAIL_PATH]
    kern = [k for k in keys if "/kernel." in k]
    launches = sum(n(k) for k in kern)
    decode_iter = n("decode.attention")
    chunk_iter = n("chunked_prefill.shared_decode/leaf.tail_emit")
    return {
        "profiled_wall_s": profiled_wall_s, "unprofiled_wall_s": unprofiled_wall_s, "bf16_unprofiled_wall_s": bf16_wall_s,
        "instrumentation_wall_ratio": profiled_wall_s / unprofiled_wall_s,
        "recorded_bookkeeping_s": overhead / 1000.0, "corrected_step_time_s": total / 1000.0,
        "residual_corrected_over_unprofiled": total / 1000.0 / unprofiled_wall_s,
        "groups": groups, "branch_detail": detail,
        "cross_branch": [
            part("per-token open-tail path (all branches)", tail_all),
            part("all Triton kernel-launch windows (host time of the launches)", kern, launches),
            part("decode path (append_aging + decode.attention subtrees)",
                 [k for k in keys if k.split("/")[0] in ("decode.append_aging", "decode.attention")]),
        ],
        "serial_iterations": {
            "decode (layer x request x token)": decode_iter, "chunked prefill (layer x token)": chunk_iter,
            "kernel_launches": launches,
            "kernel_launches_per_decode_iteration": (sum(n(k) for k in kern if k.split("/")[0] in
                                                         ("decode.append_aging", "decode.attention")) / decode_iter)
            if decode_iter else None,
            "kernel_launches_per_chunk_iteration": (sum(n(k) for k in kern if k.startswith("chunked_prefill.")) /
                                                    chunk_iter) if chunk_iter else None,
            "unprofiled_wall_ms_per_serial_iteration": unprofiled_wall_s * 1000.0 / (decode_iter + chunk_iter)
            if decode_iter + chunk_iter else None},
        "approx_device_time_of_long_kernels": [
            {"kernel": k, "launches": gpu[k]["calls"], "gpu_span_s": gpu[k]["gpu_ms"] / 1000.0,
             "ms_per_launch": gpu[k]["gpu_ms"] / gpu[k]["calls"],
             "fraction_of_unprofiled_wall": gpu[k]["gpu_ms"] / 1000.0 / unprofiled_wall_s}
            for k in LONG_KERNELS if k in gpu],
        "chunk_shape": {x: host["chunked_prefill.shared_decode"].get(x) for x in ("calls", "tokens", "context_tokens")}
        if "chunked_prefill.shared_decode" in host else None,
    }


def main() -> int:
    legs = json.loads((OUT_DIR / "legs.json").read_text(encoding="utf-8"))
    out = {"title": "Overhead-corrected host attribution (addendum to summary.json; DIAGNOSTIC evidence only)",
           "method": __doc__.strip().split("\n\n", 1)[1], "cases": {}}
    for case in ("c1", "c2", "c3"):
        w = {x: legs[f"{case}_{x}"]["measured_summary"]["wall_s"] for x in ("rabit_off", "rabit_events", "bf16_off")}
        out["cases"][case] = corrected(legs[f"{case}_rabit_events"]["measured_profile_raw"], w["rabit_off"],
                                       w["rabit_events"], w["bf16_off"])
    t = legs["c1_rabit_torch"]["torch_profiler"]
    dev = {r["key"]: r for r in t["top_by_self_device_ms"]}
    rabit_dev = sum(r["self_device_ms"] for k, r in dev.items() if k.startswith("_rabit2"))
    attn_cpu = next(r for r in t["top_by_self_cpu_ms"] if r["key"] == "vllm::unified_attention_with_output")
    out["torch_profiler_cross_check_c1"] = {
        "traced_wall_s": t["traced_wall_s"], "total_self_device_s": t["total_self_device_ms"] / 1000.0,
        "device_busy_fraction_upper_bound": t["total_self_device_ms"] / 1000.0 / t["traced_wall_s"],
        "rabit_triton_kernels_self_device_s_in_top60": rabit_dev / 1000.0,
        "rabit_triton_kernels_fraction_of_traced_wall": rabit_dev / 1000.0 / t["traced_wall_s"],
        "attention_op_self_cpu_s": attn_cpu["self_cpu_ms"] / 1000.0,
        "attention_op_self_cpu_fraction_of_total_self_cpu": attn_cpu["self_cpu_ms"] / t["total_self_cpu_ms"],
        "attention_op_self_cpu_fraction_of_traced_wall": attn_cpu["self_cpu_ms"] / 1000.0 / t["traced_wall_s"],
        "note": "vllm::unified_attention_with_output is the custom op that encloses TritonAttentionImpl.forward; its "
                "SELF cpu time is the Python / Triton-launch time of the RABIT attention path not inside a torch op."}
    (OUT_DIR / "corrected_attribution.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    for case, c in out["cases"].items():
        print(f"== {case}: profiled {c['profiled_wall_s']:.1f}s unprofiled {c['unprofiled_wall_s']:.1f}s ratio "
              f"{c['instrumentation_wall_ratio']:.2f} bookkeeping {c['recorded_bookkeeping_s']:.1f}s corrected "
              f"{c['corrected_step_time_s']:.1f}s residual x{c['residual_corrected_over_unprofiled']:.2f}")
        for g in c["groups"] + c["cross_branch"]:
            print(f"   {g['component'][:86]:86s} {g['corrected_host_ms'] / 1000:8.1f}s {100 * g['share_of_corrected_step_time']:5.1f}%")
        for b, rows in c["branch_detail"].items():
            for g in rows:
                if g["share_of_corrected_step_time"] >= 0.005:
                    print(f"     [{b}] {g['component'][:70]:70s} {g['corrected_host_ms'] / 1000:8.1f}s {100 * g['share_of_corrected_step_time']:5.1f}%")
        print("   ", json.dumps(c["serial_iterations"]), json.dumps(c["approx_device_time_of_long_kernels"]), c["chunk_shape"])
    print(json.dumps(out["torch_profiler_cross_check_c1"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
