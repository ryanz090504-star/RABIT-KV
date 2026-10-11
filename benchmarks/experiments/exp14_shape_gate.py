"""
MLSys 2027 Experiment 14 -- RABIT-KV Model-B SHAPE correctness gate (runs INSIDE the Modal container, in its own fresh
process, AFTER the unchanged RABIT gate (exp3_correctness_gate.py) and BEFORE any measured leg).

Why: the accepted RABIT gate exercises the Llama-3.1-8B shape only (8 KV heads x head_dim 128, GQA ratio 4). Model B
(Qwen2.5-7B-Instruct: 28 query heads / 4 KV heads, head_dim 128, GQA ratio 7) does NOT satisfy the frozen fast-path
predicates, so serving dispatches to the source's own exact fallbacks:
  * decode attention: rabit2_online_decode_attention_triton (= the Stage4B3 GQA4 entry) falls back to
    _rabit2_online_decode_attention_triton_stage4b2_exact when (q_heads // kv_heads) % 4 != 0;
  * one-token append: Rabit2SingleSequenceRuntime.append (= _rabit2_final_fast_decode_append) falls back to
    _rabit2_final_old_append unless num_kv_heads == 8 and head_size == 128;
  * the compiled (1, 8, 128) V2 quantizer (_quantize_v2_primary_ref) falls back to
    _quantize_v2_primary_ref_stage4b1_exact (the exact reference quantizer).
This gate proves those paths at Model B's shape BEFORE any measurement, with the Llama shape as a POSITIVE CONTROL
(same checks, same tolerances; it must also pass, which shows the checks themselves are valid).

Per geometry, it replays the EXACT serving call sequence of the latency workload (initial prefill of 2048 tokens via
rabit2_bulk_append_exact, then 31 decode steps of runtime.append + rabit2_online_decode_attention_triton -- the calls
made by TritonAttentionImpl._forward_rabit_kv2), plus short-prefix boundary replays, and requires:
  1. layout: rabit2_page_layout page bytes == the expected value; _rabit2_stage3b1_validate_layout accepts it;
     Rabit2OnlineStateRef constructs (K head_size % 64 and META8g64 page alignment);
  2. state: after every step, total_tokens and closed_pages equal those of Rabit2OnlineStateRef fed the same tokens,
     and every closed physical page is BYTE-IDENTICAL (torch.equal) to the reference page;
  3. attention (fixed dtype-aware numerical conformance checks -- POST-FAILURE CORRECTNESS-CRITERION AMENDMENT,
     replacing the inherited fixed 5e-3 absolute threshold): after every step, with
       r = the UNROUNDED FP32 kernel-semantics attention output computed from the INDEPENDENT Rabit2OnlineStateRef
           state (closed pages dequantized in FP32; open-group K / V rounded to BF16 as the runtime does; residual
           tokens exact BF16; PyTorch FP32 GQA attention _gqa_ref), and
       y = the runtime BF16 decode output converted to FP32,
     both must hold:
       C1 (normwise):  max|y - r| <= u * max|r|, u = 2^-8 (BF16 unit roundoff); if max|r| == 0, y must be exactly 0;
       C2 (relative):  |y_i - r_i| / |r_i| <= u for every element with |r_i| >= 0.01 * max|r| (the pre-existing 1 %
                       floor of the numerical diagnostic).
     These are fixed dtype-aware numerical conformance checks (runtime output is stored in BF16; the independent
     reference represents the same intended numerical semantics in FP32; u is the BF16 unit-roundoff scale; C1 handles
     near-zero values; the floor makes relative error meaningful away from zero). They are NOT a rigorous worst-case
     floating-point theorem, NOT a universal error bound and NOT a proof of arbitrary-input Triton attention
     correctness. Recorded but NOT gating: the old 5e-3 error vs the FP32 references of Attempt 1 (MAX_ABS_TOL), the
     BF16-rounded-reference error, the BF16 ULP histogram / max ULP, and the same C1 / C2 against the runtime-state
     materialization (secondary consistency diagnostic);
  4. dispatch: call counters on the three fallback functions match the frozen predicates on the 2048-token replay
     (Model B: every decode call uses the exact decode fallback, every decode-step append uses the exact append
     fallback and invokes the exact V2 quantizer fallback at least once; control: none of them during decode steps
     -- bulk prefill uses the exact batched quantizer at BOTH shapes and is therefore excluded from that count).
The exact checks (1, 2, 4) are unchanged and strict; a numerical-conformance pass never overrides an exact-check
failure (checkpoint_failures). Emits EXP14_SHAPE_GATE_SUMMARY=<json>; exit 0 only if every check passes. Imports vllm-kvquant; never modifies it
(the counters wrap module attributes in THIS process only and are restored).
"""

from __future__ import annotations

import json
import math
import sys
import traceback

GEOMETRIES = {
    "model_b_qwen2_5_7b": {"q_heads": 28, "kv_heads": 4, "head_dim": 128, "expected_page_bytes": 12416,
                           "expect_gqa4": False, "expect_fast_append": False},
    "control_llama3_1_8b": {"q_heads": 32, "kv_heads": 8, "head_dim": 128, "expected_page_bytes": 24832,
                            "expect_gqa4": True, "expect_fast_append": True},
}
MAIN_PREFILL = 2048  # latency-workload prompt length (single initial prefill chunk: 2048 < max_num_batched_tokens)
MAIN_DECODE_STEPS = 31  # 32 output tokens -> 31 decode steps after the prefill-produced first token
MAIN_SEEDS = (140001, 140002)
BOUNDARY_PREFILLS = (1, 4, 5, 35, 36, 37, 68)  # page-closure / residual boundaries (R4, 32-token groups)
BOUNDARY_DECODE_STEPS = 8
BOUNDARY_SEED = 140100
MAX_ABS_TOL = 5e-3  # inherited Attempt-1 threshold: DIAGNOSTIC ONLY since the criterion amendment (never gates)
U_BF16 = 2.0 ** -8  # BF16 unit roundoff (round-to-nearest, 8 significand bits)
RELATIVE_FLOOR = 0.01  # C2 applies where |r_i| >= RELATIVE_FLOOR * max|r| (pre-existing diagnostic floor)


def emit(tag: str, payload) -> None:
    print(f"{tag}={json.dumps(payload, sort_keys=True, default=str)}", flush=True)


def _materialize(runtime, cache, block_row, decode_rabit2_page_ref, torch):
    """Verbatim statements of tests/quantization/test_rabit_kv2_stage3c.py::_materialize (verified by AST)."""
    k_parts = []
    v_parts = []
    cache2d = cache.reshape(cache.shape[0], -1)
    for page_idx in range(runtime.closed_pages):
        physical = int(block_row[page_idx].item())
        k, v = decode_rabit2_page_ref(
            cache2d[physical], layout=runtime.layout, dtype=torch.float32
        )
        k_parts.append(k)
        v_parts.append(v)
    tail_k, tail_v = runtime.tail_materialize(torch.float32)
    if tail_k.numel():
        k_parts.append(tail_k)
        v_parts.append(tail_v)
    return torch.cat(k_parts, dim=0), torch.cat(v_parts, dim=0)


def _gqa_ref(q, k, v, softmax_scale, torch):
    """Verbatim logic of tests/quantization/test_rabit_kv2_stage3c.py::_gqa_ref (q [QH,D], k/v [T,KVH,D])."""
    rep = q.shape[0] // k.shape[1]
    k_rep = k.repeat_interleave(rep, dim=1)
    v_rep = v.repeat_interleave(rep, dim=1)
    scores = torch.einsum("hd,thd->ht", q.float(), k_rep.float())
    probs = torch.softmax(scores * softmax_scale, dim=-1)
    return torch.einsum("ht,thd->hd", probs, v_rep.float())


def conformance_decision(max_abs_r: float, max_abs_err: float, max_rel_above_floor: float,
                         y_all_zero: bool) -> dict:
    """C1 / C2 decision from the per-checkpoint maxima (shared by conformance() and the offline posterior test)."""
    if max_abs_r == 0.0:
        ok = bool(y_all_zero)
        return {"max_abs_r": 0.0, "max_abs_err": max_abs_err, "max_rel_above_floor": 0.0,
                "C1_ratio": 0.0 if ok else math.inf, "C2_ratio": 0.0, "C1_pass": ok, "C2_pass": ok, "passed": ok}
    c1_pass = max_abs_err <= U_BF16 * max_abs_r
    c2_pass = max_rel_above_floor <= U_BF16
    return {"max_abs_r": max_abs_r, "max_abs_err": max_abs_err, "max_rel_above_floor": max_rel_above_floor,
            "C1_ratio": max_abs_err / (U_BF16 * max_abs_r), "C2_ratio": max_rel_above_floor / U_BF16,
            "C1_pass": c1_pass, "C2_pass": c2_pass, "passed": c1_pass and c2_pass}


def conformance(y, r) -> dict:
    """Fixed dtype-aware numerical conformance checks C1 / C2 on flat sequences of floats.
    y: runtime BF16 output values (as FP32); r: UNROUNDED FP32 kernel-semantics reference values."""
    y, r = [float(v) for v in y], [float(v) for v in r]
    if len(y) != len(r) or not y:
        return {"passed": False, "C1_pass": False, "C2_pass": False, "error": "empty or mismatched output shapes"}
    if not all(math.isfinite(v) for v in y) or not all(math.isfinite(v) for v in r):
        return {"passed": False, "C1_pass": False, "C2_pass": False, "error": "non-finite value"}
    max_abs_r = max(abs(v) for v in r)
    max_abs_err = max(abs(a - b) for a, b in zip(y, r))
    floor = RELATIVE_FLOOR * max_abs_r
    rel = max((abs(a - b) / abs(b) for a, b in zip(y, r) if max_abs_r > 0.0 and abs(b) >= floor), default=0.0)
    return conformance_decision(max_abs_r, max_abs_err, rel, all(a == 0.0 for a in y))


def checkpoint_failures(t: int, exact_failures: list, conf: dict) -> list:
    """Every exact-invariant failure fails the checkpoint regardless of C1 / C2; C1 / C2 failure also fails it."""
    out = [f"T={t}: {m}" for m in exact_failures]
    if not conf.get("passed"):
        out.append(f"T={t}: attention conformance failed C1={conf.get('C1_ratio')} C2={conf.get('C2_ratio')} "
                   f"{conf.get('error', '')}".rstrip())
    return out


def kernel_semantics_state(r, pages, layout, open_kv, recent_kv, dev, torch):
    """[T, KVH, D] FP32 K / V with the runtime's numerical semantics: closed pages dequantized in FP32, open-group K / V
    rounded to BF16, residual tokens exact BF16."""
    k_parts, v_parts = [], []
    for page in pages:
        k, v = r.decode_rabit2_page_ref(page, layout=layout, dtype=torch.float32)
        k_parts.append(k.float())
        v_parts.append(v.float())
    ok, ov = open_kv
    if ok.numel():
        k_parts.append(ok.float())
        v_parts.append(ov.float())
    rk, rv = recent_kv
    if rk is not None and rk.numel():
        k_parts.append(rk.float())
        v_parts.append(rv.float())
    return torch.cat(k_parts).to(dev), torch.cat(v_parts).to(dev)


def _ord16(x, torch):
    b = x.contiguous().view(torch.int16).to(torch.int32)
    return torch.where(b >= 0, b, -(b & 0x7FFF))


class _Counter:
    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    def __call__(self, *a, **kw):
        self.calls += 1
        return self.fn(*a, **kw)


def replay(r, torch, geom: dict, prefill: int, steps: int, seed: int, quant=None) -> dict:
    """One serving-sequence replay: bulk prefill, then `steps` decode steps; checks after the prefill and every step."""
    qh, h, d = geom["q_heads"], geom["kv_heads"], geom["head_dim"]
    dev, dt = torch.device("cuda"), torch.bfloat16
    torch.manual_seed(seed)
    total = prefill + steps
    k_all = torch.randn((total, h, d), dtype=dt, device=dev)
    v_all = torch.randn_like(k_all)
    rt = r.Rabit2SingleSequenceRuntime(h, d, d)
    ref = r.Rabit2OnlineStateRef(num_kv_heads=h, head_size_k=d, head_size_v=d)
    pages = total // 32 + 16
    cache = torch.zeros((pages, 1, 1, 1, rt.layout.page_bytes), dtype=torch.uint8, device=dev)
    bt = torch.arange(pages, dtype=torch.int32, device=dev)
    scale = d ** -0.5
    out = {"prefill": prefill, "steps": steps, "seed": seed, "checkpoints": 0,
           "max_C1_ratio": 0.0, "max_C2_ratio": 0.0,
           "diagnostics": {"old_max_abs_vs_runtime_state_fp32": 0.0, "old_max_abs_vs_reference_state_fp32": 0.0,
                           "old_5e-3_would_fail": 0, "max_abs_vs_bf16_rounded_ref": 0.0, "max_ulp_vs_bf16_rounded_ref": 0,
                           "runtime_state_max_C1_ratio": 0.0, "runtime_state_max_C2_ratio": 0.0,
                           "runtime_state_conformance_failures": 0},
           "failures": [], "decode_step_quant_fallback_calls": []}

    def check(t: int) -> None:
        exact = []  # strict exact invariants (unchanged)
        if rt.total_tokens != t or ref.total_tokens != t:
            exact.append(f"total_tokens runtime={rt.total_tokens} ref={ref.total_tokens}")
        if rt.closed_pages != len(ref.pages):
            exact.append(f"closed_pages runtime={rt.closed_pages} ref={len(ref.pages)}")
        c2d = cache.reshape(cache.shape[0], -1)
        for p in range(min(rt.closed_pages, len(ref.pages))):
            if not torch.equal(c2d[int(bt[p].item())], ref.pages[p].reshape(-1).to(dev)):
                exact.append(f"closed page {p} bytes differ from Rabit2OnlineStateRef")
                break
        q = torch.randn((1, qh, d), dtype=dt, device=dev)
        got16 = r.rabit2_online_decode_attention_triton(q, cache, bt, rt, softmax_scale=scale)[0]
        got = got16.float()
        # PRIMARY reference: independent Rabit2OnlineStateRef state, kernel semantics, unrounded FP32 output
        k_s, v_s = kernel_semantics_state(r, ref.pages, ref.layout, ref._decode_open(dtype=torch.bfloat16),
                                          (ref.recent_k, ref.recent_v), dev, torch)
        ref_out = _gqa_ref(q[0], k_s, v_s, scale, torch)
        conf = conformance(got.flatten().tolist(), ref_out.flatten().tolist())
        out["max_C1_ratio"] = max(out["max_C1_ratio"], conf.get("C1_ratio", math.inf))
        out["max_C2_ratio"] = max(out["max_C2_ratio"], conf.get("C2_ratio", math.inf))
        # DIAGNOSTICS (never gate): Attempt-1 FP32 references / 5e-3, BF16-rounded reference + ULP, runtime-state C1/C2
        dg = out["diagnostics"]
        k_rt, v_rt = _materialize(rt, cache, bt, r.decode_rabit2_page_ref, torch)
        k_ref, v_ref = ref.materialize(dtype=torch.float32)
        e_rt = float((got - _gqa_ref(q[0], k_rt, v_rt, scale, torch)).abs().max().item())
        e_ref = float((got - _gqa_ref(q[0], k_ref.to(dev), v_ref.to(dev), scale, torch)).abs().max().item())
        dg["old_max_abs_vs_runtime_state_fp32"] = max(dg["old_max_abs_vs_runtime_state_fp32"], e_rt)
        dg["old_max_abs_vs_reference_state_fp32"] = max(dg["old_max_abs_vs_reference_state_fp32"], e_ref)
        dg["old_5e-3_would_fail"] += int(not (e_rt < MAX_ABS_TOL and e_ref < MAX_ABS_TOL))
        ref16 = ref_out.to(torch.bfloat16)
        dg["max_abs_vs_bf16_rounded_ref"] = max(dg["max_abs_vs_bf16_rounded_ref"],
                                                float((got - ref16.float()).abs().max().item()))
        dg["max_ulp_vs_bf16_rounded_ref"] = max(dg["max_ulp_vs_bf16_rounded_ref"],
                                                int((_ord16(got16, torch) - _ord16(ref16, torch)).abs().max().item()))
        pages_rt = [c2d[int(bt[p].item())] for p in range(rt.closed_pages)]
        k_r2, v_r2 = kernel_semantics_state(r, pages_rt, rt.layout, rt.tail_materialize(torch.bfloat16), (None, None),
                                            dev, torch)  # tail_materialize already includes the residual tokens
        conf_rt = conformance(got.flatten().tolist(), _gqa_ref(q[0], k_r2, v_r2, scale, torch).flatten().tolist())
        dg["runtime_state_max_C1_ratio"] = max(dg["runtime_state_max_C1_ratio"], conf_rt.get("C1_ratio", math.inf))
        dg["runtime_state_max_C2_ratio"] = max(dg["runtime_state_max_C2_ratio"], conf_rt.get("C2_ratio", math.inf))
        dg["runtime_state_conformance_failures"] += int(not conf_rt.get("passed"))
        out["failures"].extend(checkpoint_failures(t, exact, conf))
        out["checkpoints"] += 1

    r.rabit2_bulk_append_exact(rt, k_all[:prefill], v_all[:prefill], cache, bt)  # initial-prefill path
    ref.append(k_all[:prefill], v_all[:prefill])
    torch.cuda.synchronize()
    check(prefill)
    for i in range(prefill, total):  # decode path: runtime.append + online decode attention
        q0 = quant.calls if quant is not None else 0
        rt.append(k_all[i:i + 1], v_all[i:i + 1], cache, bt)
        if quant is not None:
            out["decode_step_quant_fallback_calls"].append(quant.calls - q0)
        ref.append(k_all[i:i + 1], v_all[i:i + 1])
        torch.cuda.synchronize()
        check(i + 1)
    return out


def run_geometry(r, torch, name: str, geom: dict) -> dict:
    from vllm.v1.kv_cache_interface import rabit2_page_layout

    h, d, qh = geom["kv_heads"], geom["head_dim"], geom["q_heads"]
    res: dict = {"geometry": {k: geom[k] for k in ("q_heads", "kv_heads", "head_dim")}, "checks": {}}
    layout = rabit2_page_layout(block_size=32, num_kv_heads=h, head_size_k=d, head_size_v=d)
    res["page_bytes"] = layout.page_bytes
    res["checks"]["page_bytes_expected"] = layout.page_bytes == geom["expected_page_bytes"]
    r._rabit2_stage3b1_validate_layout(layout)  # raises on a violated alignment invariant
    r.Rabit2OnlineStateRef(num_kv_heads=h, head_size_k=d, head_size_v=d)  # raises on a violated alignment invariant
    res["checks"]["alignment_invariants"] = True
    res["predicates"] = {"gqa4_engaged": qh % h == 0 and (qh // h) % 4 == 0,
                         "fast_append_engaged": h == 8 and d == 128}
    res["checks"]["predicates_as_expected"] = (res["predicates"]["gqa4_engaged"] == geom["expect_gqa4"] and
                                               res["predicates"]["fast_append_engaged"] == geom["expect_fast_append"])

    dec = _Counter(r._rabit2_online_decode_attention_triton_stage4b2_exact)
    app = _Counter(r._rabit2_final_old_append)
    qnt = _Counter(r._quantize_v2_primary_ref_stage4b1_exact)
    r._rabit2_online_decode_attention_triton_stage4b2_exact, r._rabit2_final_old_append = dec, app
    r._quantize_v2_primary_ref_stage4b1_exact = qnt
    try:
        main = [replay(r, torch, geom, MAIN_PREFILL, MAIN_DECODE_STEPS, s, qnt) for s in MAIN_SEEDS]
        step_q = [c for x in main for c in x["decode_step_quant_fallback_calls"]]
        main_counts = {"decode_calls": len(MAIN_SEEDS) * (MAIN_DECODE_STEPS + 1),
                       "append_calls": len(MAIN_SEEDS) * MAIN_DECODE_STEPS,
                       "decode_fallback_calls": dec.calls, "append_fallback_calls": app.calls,
                       "decode_step_quant_fallback_calls_total": sum(step_q),
                       "decode_steps_with_quant_fallback": sum(1 for c in step_q if c >= 1)}
        boundary = [replay(r, torch, geom, p, BOUNDARY_DECODE_STEPS, BOUNDARY_SEED + p) for p in BOUNDARY_PREFILLS]
    finally:
        r._rabit2_online_decode_attention_triton_stage4b2_exact, r._rabit2_final_old_append = dec.fn, app.fn
        r._quantize_v2_primary_ref_stage4b1_exact = qnt.fn
    res["main_workload_replays"], res["boundary_replays"], res["main_dispatch_counts"] = main, boundary, main_counts
    if geom["expect_gqa4"]:
        res["checks"]["dispatch_matches_predicates"] = (main_counts["decode_fallback_calls"] == 0 and
                                                        main_counts["append_fallback_calls"] == 0 and
                                                        main_counts["decode_step_quant_fallback_calls_total"] == 0)
    else:
        res["checks"]["dispatch_matches_predicates"] = (
            main_counts["decode_fallback_calls"] == main_counts["decode_calls"] and
            main_counts["append_fallback_calls"] == main_counts["append_calls"] and
            main_counts["decode_steps_with_quant_fallback"] == main_counts["append_calls"])
    res["checks"]["main_workload_state_and_attention"] = all(not x["failures"] for x in main)
    res["checks"]["boundary_state_and_attention"] = all(not x["failures"] for x in boundary)
    res["max_C1_ratio"] = max(x["max_C1_ratio"] for x in main + boundary)
    res["max_C2_ratio"] = max(x["max_C2_ratio"] for x in main + boundary)
    res["diagnostic_old_max_abs_fp32"] = max(max(x["diagnostics"]["old_max_abs_vs_runtime_state_fp32"],
                                                 x["diagnostics"]["old_max_abs_vs_reference_state_fp32"])
                                             for x in main + boundary)
    res["diagnostic_old_5e-3_would_fail"] = sum(x["diagnostics"]["old_5e-3_would_fail"] for x in main + boundary)
    res["passed"] = all(res["checks"].values())
    return res


def main() -> int:
    summary: dict = {"criterion": {"kind": "fixed dtype-aware numerical conformance checks",
                                   "C1": "max|y - r| <= u * max|r| (y == 0 if max|r| == 0)",
                                   "C2": "|y_i - r_i| / |r_i| <= u where |r_i| >= floor * max|r|",
                                   "u": U_BF16, "floor": RELATIVE_FLOOR,
                                   "r": "unrounded FP32 kernel-semantics output from the independent "
                                        "Rabit2OnlineStateRef state"},
                     "diagnostic_old_max_abs_tol_non_gating": MAX_ABS_TOL, "geometries": {}}
    try:
        import torch
        import vllm.v1.attention.ops.rabit_kv2 as r

        summary["cuda_available"] = bool(torch.cuda.is_available())
        if not summary["cuda_available"]:
            raise RuntimeError("CUDA is required")
        for name, geom in GEOMETRIES.items():
            summary["geometries"][name] = run_geometry(r, torch, name, geom)
            g = summary["geometries"][name]
            print(f"{name}: passed={g['passed']} checks={g['checks']} max_C1_ratio={g['max_C1_ratio']:.4f} "
                  f"max_C2_ratio={g['max_C2_ratio']:.4f} diag_old_max_abs={g['diagnostic_old_max_abs_fp32']:.3e} "
                  f"dispatch={g['main_dispatch_counts']}", flush=True)
        summary["passed"] = all(g["passed"] for g in summary["geometries"].values())
    except Exception as exc:  # noqa: BLE001  (any error is a gate failure)
        traceback.print_exc()
        summary["passed"] = False
        summary["error"] = f"{type(exc).__name__}: {exc}"
    emit("EXP14_SHAPE_GATE_SUMMARY", summary)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
