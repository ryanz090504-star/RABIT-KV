"""
MLSys 2027 Experiment 14 -- NON-EVIDENCE numerical diagnosis of the Model-B shape gate's attention criterion (runs
INSIDE a Modal H100 container; no model, no engine, no download). Written after probe Attempt 1 failed only the
short-prefix attention tolerance (5e-3 absolute) at BOTH geometries.

It replays EXACTLY the shape-gate tensors (same seeds, same RNG consumption order, same serving call sequence as
exp14_shape_gate.replay) and, at every checkpoint, compares the runtime decode output against three references
computed from BOTH the runtime state and the independent Rabit2OnlineStateRef state:
  R_fp32    -- every dequantized value in FP32 (the Attempt-1 gate reference);
  R_sem     -- the kernel's documented dtype semantics: closed pages dequantized in FP32; open-group (incomplete
               32-token group) K / V dequantized then ROUNDED TO BF16 (rabit_kv2.py stage4d3_4 tail kernel:
               `.to(tl.bfloat16).to(tl.float32)`); BF16 residual tokens exact;
  R_bf16all -- every dequantized value rounded to BF16.
For each: output dtype, max |reference|, max abs error vs the FP32-accumulated reference output, max relative error
(elements with |ref| >= 1e-2 * max|ref|), and after rounding the reference OUTPUT to the runtime output dtype: max
abs error, exact-equality count and BF16 ULP distance histogram. Also the error in units of the BF16 ULP at the
reference value. Recomputes the Attempt-1 gate numbers as a determinism cross-check.
It changes no cache state or quantization. Emits EXP14_NUMDIAG_SUMMARY=<json>.
"""

from __future__ import annotations

import hashlib
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp14_shape_gate as sg  # noqa: E402  (frozen gate: geometries, seeds, reference helpers)

RABIT_KV2 = Path("/root/vllm-kvquant/vllm/v1/attention/ops/rabit_kv2.py")
EXPECTED_RABIT_SHA256_LF = "7e628c94eebb9fe689bf416ea61f748c0f909a82d0f229c463edd1a0df92e6ae"
REL_FLOOR = 1e-2  # relative error only where |ref| >= REL_FLOOR * max|ref|


def ord16(x, torch):
    """Monotone integer ordering of BF16 values (ULP distance = |ord16(a) - ord16(b)|)."""
    b = x.contiguous().view(torch.int16).to(torch.int32)
    return torch.where(b >= 0, b, -(b & 0x7FFF))


def ulp_bf16(v, torch):
    """BF16 ULP at |v| (normal range): 2**(floor(log2|v|) - 7)."""
    _, e = torch.frexp(v.abs().clamp_min(2.0 ** -126))
    return torch.ldexp(torch.ones_like(v), (e - 8).to(torch.int32))


def compare(got, ref32, torch) -> dict:
    g = got.float()
    mx = float(ref32.abs().max().item())
    err = (g - ref32).abs()
    mask = ref32.abs() >= REL_FLOOR * mx
    rel = float((err[mask] / ref32.abs()[mask]).max().item()) if bool(mask.any()) else None
    ref_r = ref32.to(got.dtype)
    err_r = (g - ref_r.float()).abs()
    ulps = (ord16(got, torch) - ord16(ref_r, torch)).abs()
    return {"ref_max_abs": mx, "max_abs_err_vs_fp32_ref": float(err.max().item()), "max_rel_err": rel,
            "max_err_in_bf16_ulp_at_ref": float((err / ulp_bf16(ref32, torch)).max().item()),
            "max_abs_err_vs_dtype_rounded_ref": float(err_r.max().item()),
            "exact_equal_elements": int((ulps == 0).sum().item()), "elements": int(ulps.numel()),
            "max_ulp": int(ulps.max().item()),
            "ulp_hist": {str(k): int((ulps == k).sum().item()) for k in (0, 1, 2)} | {"3+": int((ulps >= 3).sum().item())}}


def runtime_state(r, rt, cache, bt, torch, open_dtype, page_dtype):
    """Runtime state: closed pages via decode_rabit2_page_ref(page_dtype); tail via tail_materialize(open_dtype)."""
    k_parts, v_parts = [], []
    c2d = cache.reshape(cache.shape[0], -1)
    for p in range(rt.closed_pages):
        k, v = r.decode_rabit2_page_ref(c2d[int(bt[p].item())], layout=rt.layout, dtype=page_dtype)
        k_parts.append(k.float())
        v_parts.append(v.float())
    tk, tv = rt.tail_materialize(open_dtype)
    if tk.numel():
        k_parts.append(tk.float())
        v_parts.append(tv.float())
    return torch.cat(k_parts), torch.cat(v_parts)


def reference_state(r, ref, dev, torch, open_dtype, page_dtype):
    """Independent Rabit2OnlineStateRef state with the same per-region dtype semantics."""
    k_parts, v_parts = [], []
    for page in ref.pages:
        k, v = r.decode_rabit2_page_ref(page, layout=ref.layout, dtype=page_dtype)
        k_parts.append(k.float())
        v_parts.append(v.float())
    ok, ov = ref._decode_open(dtype=open_dtype)
    if ok.numel():
        k_parts.append(ok.float())
        v_parts.append(ov.float())
    if ref.recent_k is not None:
        k_parts.append(ref.recent_k.float())
        v_parts.append(ref.recent_v.float())
    return torch.cat(k_parts).to(dev), torch.cat(v_parts).to(dev)


SEMANTICS = {"R_fp32": ("float32", "float32"), "R_sem": ("bfloat16", "float32"), "R_bf16all": ("bfloat16", "bfloat16")}


def replay(r, torch, geom: dict, prefill: int, steps: int, seed: int) -> list[dict]:
    """Identical tensors / RNG order to exp14_shape_gate.replay; per-checkpoint numerical records."""
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
    rows = []

    def check(t: int) -> None:
        c2d = cache.reshape(cache.shape[0], -1)
        bytes_ok = rt.closed_pages == len(ref.pages) and all(
            torch.equal(c2d[int(bt[p].item())], ref.pages[p].reshape(-1).to(dev)) for p in range(rt.closed_pages))
        q = torch.randn((1, qh, d), dtype=dt, device=dev)
        got = r.rabit2_online_decode_attention_triton(q, cache, bt, rt, softmax_scale=scale)[0]
        # Attempt-1 gate numbers (determinism cross-check)
        k_rt, v_rt = sg._materialize(rt, cache, bt, r.decode_rabit2_page_ref, torch)
        k_ref, v_ref = ref.materialize(dtype=torch.float32)
        gate_rt = float((got.float() - sg._gqa_ref(q[0], k_rt, v_rt, scale, torch)).abs().max().item())
        gate_ref = float((got.float() - sg._gqa_ref(q[0], k_ref.to(dev), v_ref.to(dev), scale, torch)).abs().max().item())
        row = {"T": t, "closed_pages": rt.closed_pages, "open_tokens": 0 if rt.open_k is None else int(rt.open_k.shape[0]),
               "runtime_output_dtype": str(got.dtype), "bytes_identical": bool(bytes_ok),
               "gate_attempt1_max_abs_runtime_state": gate_rt, "gate_attempt1_max_abs_reference_state": gate_ref,
               "gate_attempt1_would_fail": not (gate_rt < sg.MAX_ABS_TOL and gate_ref < sg.MAX_ABS_TOL)}
        for name, (od, pd) in SEMANTICS.items():
            odt, pdt = getattr(torch, od), getattr(torch, pd)
            ks, vs = runtime_state(r, rt, cache, bt, torch, odt, pdt)
            kr, vr = reference_state(r, ref, dev, torch, odt, pdt)
            ref_rt = sg._gqa_ref(q[0], ks, vs, scale, torch)
            ref_ref = sg._gqa_ref(q[0], kr, vr, scale, torch)
            row[name] = {"reference_output_dtype": str(ref_rt.dtype),
                         "runtime_state": compare(got, ref_rt, torch), "reference_state": compare(got, ref_ref, torch),
                         "states_identical": bool(torch.equal(ks, kr) and torch.equal(vs, vr))}
        rows.append(row)

    r.rabit2_bulk_append_exact(rt, k_all[:prefill], v_all[:prefill], cache, bt)
    ref.append(k_all[:prefill], v_all[:prefill])
    torch.cuda.synchronize()
    check(prefill)
    for i in range(prefill, total):
        rt.append(k_all[i:i + 1], v_all[i:i + 1], cache, bt)
        ref.append(k_all[i:i + 1], v_all[i:i + 1])
        torch.cuda.synchronize()
        check(i + 1)
    return rows


def summarize(rows: list[dict]) -> dict:
    out = {"checkpoints": len(rows), "bytes_identical_all": all(x["bytes_identical"] for x in rows),
           "gate_attempt1_failures": sum(x["gate_attempt1_would_fail"] for x in rows),
           "max_ref_abs": max(x["R_fp32"]["runtime_state"]["ref_max_abs"] for x in rows)}
    for name in SEMANTICS:
        for side in ("runtime_state", "reference_state"):
            vals = [x[name][side] for x in rows]
            out[f"{name}.{side}"] = {
                "max_abs_err_vs_fp32_ref": max(v["max_abs_err_vs_fp32_ref"] for v in vals),
                "max_rel_err": max((v["max_rel_err"] or 0.0) for v in vals),
                "max_err_in_bf16_ulp_at_ref": max(v["max_err_in_bf16_ulp_at_ref"] for v in vals),
                "max_abs_err_vs_dtype_rounded_ref": max(v["max_abs_err_vs_dtype_rounded_ref"] for v in vals),
                "max_ulp": max(v["max_ulp"] for v in vals),
                "ulp_hist": {k: sum(v["ulp_hist"][k] for v in vals) for k in ("0", "1", "2", "3+")},
                "all_exact": all(v["exact_equal_elements"] == v["elements"] for v in vals)}
        out[f"{name}.states_identical_all"] = all(x[name]["states_identical"] for x in rows)
    return out


def main() -> int:
    summary: dict = {"non_evidence": True, "rel_floor": REL_FLOOR, "geometries": {}}
    try:
        import torch
        import vllm.v1.attention.ops.rabit_kv2 as r

        sha = hashlib.sha256(RABIT_KV2.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        summary["rabit_kv2_sha256_lf"] = sha
        if sha != EXPECTED_RABIT_SHA256_LF:
            raise RuntimeError(f"rabit_kv2.py is not the frozen source: {sha}")
        summary["gpu"] = torch.cuda.get_device_name(0)
        for gname, geom in sg.GEOMETRIES.items():
            reps = {}
            for s in sg.MAIN_SEEDS:
                reps[f"main_P{sg.MAIN_PREFILL}_seed{s}"] = replay(r, torch, geom, sg.MAIN_PREFILL, sg.MAIN_DECODE_STEPS, s)
            for p in sg.BOUNDARY_PREFILLS:
                reps[f"boundary_P{p}"] = replay(r, torch, geom, p, sg.BOUNDARY_DECODE_STEPS, sg.BOUNDARY_SEED + p)
            summary["geometries"][gname] = {"replays": {k: {"summary": summarize(v), "checkpoints": v}
                                                        for k, v in reps.items()},
                                            "overall": summarize([x for v in reps.values() for x in v])}
            print(f"{gname}: {json.dumps(summary['geometries'][gname]['overall'], sort_keys=True)}", flush=True)
        summary["completed"] = True
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        summary["completed"] = False
        summary["error"] = f"{type(exc).__name__}: {exc}"
    print("EXP14_NUMDIAG_SUMMARY=" + json.dumps(summary, sort_keys=True), flush=True)
    return 0 if summary["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
