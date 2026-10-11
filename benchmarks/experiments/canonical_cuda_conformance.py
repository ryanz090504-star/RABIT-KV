"""
canonical-quality-v2 GPU SEMANTIC-CONFORMANCE DIAGNOSTIC -- comparison logic (torch). DESCRIPTIVE ONLY: no scoring, no
PPL, no generation, no tolerance, no threshold. It observes the frozen canonical implementation
(canonical_rabit_quality.py, unchanged) and the frozen independent oracle (the *_ref functions of
vllm-kvquant/vllm/v1/attention/ops/kvquant_k3.py, extracted by AST exactly as in the accepted CPU parity, 8fa9a9c).

Paths compared from IDENTICAL raw BF16 K / V bytes [T, H, D]:
    A  CPU canonical      cq.canonical_state(raw.cpu())
    B  CUDA canonical     cq.canonical_state(raw on the device)
    C  CUDA oracle        quantize_rabit2_kv_ref / dequantize_rabit2_kv_ref on the device   (never calls cq)
    D  CPU oracle         the same oracle on CPU (context: extends the accepted CPU parity to these bytes)
Primary comparison: B vs C (same device), every canonical field, bit-exact.
Descriptive comparison: A vs B, field by field, plus a TRACE of the intermediate values (primary min / max / scale,
META8 group min / max / scale ...) recomputed with the same expressions as cq.k3 / cq.v2 / cq.meta8_roundtrip; the trace
is self-checked against the frozen functions on each device before it is used.
"""

from __future__ import annotations

import hashlib

import torch
import torch.nn.functional as F

import canonical_rabit_quality as cq

K_CHAIN = ["k.primary_min", "k.primary_max", "k.primary_scale", "k.codes",
           "k.min_meta.group_min_fp32", "k.min_meta.group_max_fp32", "k.min_meta.group_scale_fp32",
           "k.min_meta.codes", "k.min_meta.min_bf16", "k.min_meta.scale_bf16", "k.min_meta.decoded",
           "k.scale_meta.group_min_fp32", "k.scale_meta.group_max_fp32", "k.scale_meta.group_scale_fp32",
           "k.scale_meta.codes", "k.scale_meta.min_bf16", "k.scale_meta.scale_bf16", "k.scale_meta.decoded",
           "k.decoded"]
V_CHAIN = [f.replace("k.", "v.", 1) for f in K_CHAIN]


# ------------------------------------------------------------------------------------------------ statistics
def _bits(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.bfloat16:
        return x.contiguous().view(torch.int16)
    if x.dtype == torch.float32:
        return x.contiguous().view(torch.int32)
    return x


def tensor_sha256(x: torch.Tensor) -> str:
    return hashlib.sha256(_bits(x.detach().cpu()).numpy().tobytes()).hexdigest()


def stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Exact comparison statistics of two tensors (moved to CPU; values are not altered). No tolerance."""
    a, b = a.detach().cpu(), b.detach().cpu()
    out = {"dtype": [str(a.dtype), str(b.dtype)], "shape": [list(a.shape), list(b.shape)], "total": int(a.numel())}
    if a.shape != b.shape or a.dtype != b.dtype:
        out.update(equal=False, bitwise_equal=False, comparable=False)
        return out
    floating = a.is_floating_point()
    neq = (a != b)
    if floating:
        neq = neq & ~(torch.isnan(a) & torch.isnan(b))
    differing = int(neq.sum())
    out.update(equal=bool(torch.equal(a, b)), bitwise_equal=bool(torch.equal(_bits(a), _bits(b))), comparable=True,
               differing=differing)
    if not differing:
        return out
    flat = int(neq.reshape(-1).nonzero()[0])
    idx = [int(i) for i in torch.unravel_index(torch.tensor(flat), a.shape)] if a.ndim else []
    av, bv = a.reshape(-1)[flat], b.reshape(-1)[flat]
    out["first_mismatch"] = {"index": idx, "a": float(av) if floating else int(av), "b": float(bv) if floating else int(bv)}
    if floating:
        d = (a.double() - b.double()).abs()
        out["max_abs_error"] = float(d.max())
        nz = a != 0
        out["max_rel_error"] = float((d[nz] / a.double().abs()[nz]).max()) if bool(nz.any()) else None
        out["rel_error_denominator"] = "abs(a), elements with a != 0"
    else:
        out["max_abs_code_difference"] = int((a.to(torch.int64) - b.to(torch.int64)).abs().max())
    return out


def scalar(a, b) -> dict:
    return {"a": a, "b": b, "equal": a == b}


# ------------------------------------------------------------------------------------------------ field extraction
def canonical_fields(raw_k: torch.Tensor, raw_v: torch.Tensor) -> dict:
    """Every canonical field of cq.canonical_state on the device of the inputs (frozen code only)."""
    st = cq.canonical_state(raw_k, raw_v)
    n, old = int(st["n"]), int(st["old_count"])
    f = {"scalars": {"membership.old_count": old, "membership.recent_count": int(st["recent_count"]),
                     "membership.closed_pages": int(st["closed_pages"]), "membership.open_count": int(st["open_count"]),
                     "membership.residual_first_index": old, "membership.residual_last_index": n - 1},
         "tensors": {"decoded_k": st["decoded_k"], "decoded_v": st["decoded_v"],
                     "residual_k_bf16": st["decoded_k"][old:].to(torch.bfloat16),
                     "residual_v_bf16": st["decoded_v"][old:].to(torch.bfloat16),
                     "hf_layout_k_bf16": st["decoded_k"].permute(1, 0, 2).unsqueeze(0).to(torch.bfloat16),
                     "hf_layout_v_bf16": st["decoded_v"].permute(1, 0, 2).unsqueeze(0).to(torch.bfloat16)}}
    if old:
        for side in ("k", "v"):
            f["tensors"][f"{side}.codes"] = st[side]["codes"]
            for m in ("min_meta", "scale_meta"):
                meta = st[side][m]
                f["tensors"][f"{side}.{m}.codes"] = meta["codes"]
                f["tensors"][f"{side}.{m}.min_bf16"] = meta["min"]
                f["tensors"][f"{side}.{m}.scale_bf16"] = meta["scale"]
                f["scalars"][f"{side}.{m}.pad"] = int(meta["pad"])
                f["scalars"][f"{side}.{m}.primary_shape"] = list(meta["decoded"].shape)
        f["scalars"]["k.pad_seq"] = int(st["k"]["pad_seq"])
    return f


def oracle_fields(o: dict, raw_k: torch.Tensor, raw_v: torch.Tensor) -> dict:
    """The same fields from the frozen independent oracle (never calls canonical_rabit_quality)."""
    n, d = int(raw_k.shape[0]), int(raw_k.shape[2])
    os_ = o["quantize_rabit2_kv_ref"](raw_k, raw_v)
    old = int(os_["old_count"])
    dk, dv = o["dequantize_rabit2_kv_ref"](os_, dtype=torch.float32)
    f = {"scalars": {"membership.old_count": old, "membership.recent_count": n - old,
                     "membership.closed_pages": old // int(o["RABIT2_GROUP_SIZE"]),
                     "membership.open_count": old % int(o["RABIT2_GROUP_SIZE"]),
                     "membership.residual_first_index": old, "membership.residual_last_index": n - 1},
         "tensors": {"decoded_k": dk, "decoded_v": dv, "residual_k_bf16": os_["recent_k"],
                     "residual_v_bf16": os_["recent_v"],
                     "hf_layout_k_bf16": dk.permute(1, 0, 2).unsqueeze(0).to(torch.bfloat16),
                     "hf_layout_v_bf16": dv.permute(1, 0, 2).unsqueeze(0).to(torch.bfloat16)},
         "packed": {}}
    if old:
        ok, ov = os_["old_k"], os_["old_v"]
        f["tensors"]["k.codes"] = o["unpack_int3_values"](ok["packed"], d)
        f["tensors"]["v.codes"] = o["unpack_int2_values"](ov["packed"], ov["padded_dim"])
        f["packed"] = {"k.packed": ok["packed"], "v.packed": ov["packed"]}
        for side, q in (("k", ok), ("v", ov)):
            for m, key in (("min_meta", "min"), ("scale_meta", "scale")):
                f["tensors"][f"{side}.{m}.codes"] = q[key]["codes"]
                f["tensors"][f"{side}.{m}.min_bf16"] = q[key]["min"]
                f["tensors"][f"{side}.{m}.scale_bf16"] = q[key]["scale"]
                f["scalars"][f"{side}.{m}.pad"] = int(q[key]["pad"])
                f["scalars"][f"{side}.{m}.primary_shape"] = list(q[key]["orig_shape"])
        f["scalars"]["k.pad_seq"] = int(ok["pad_seq"])
    return f


def compare_fields(a: dict, b: dict, o: dict | None = None) -> dict:
    """Field-by-field statistics of two field sets. With `o`, the canonical codes of `a` are also packed with the
    oracle's packers and compared with the oracle's packed bytes in `b`."""
    out = {name: scalar(v, b["scalars"].get(name)) for name, v in a["scalars"].items()}
    out["field_names_identical"] = {"a": sorted(a["tensors"]), "b": sorted(b["tensors"]),
                                    "equal": sorted(a["tensors"]) == sorted(b["tensors"])
                                    and sorted(a["scalars"]) == sorted(b["scalars"])}
    for name, t in a["tensors"].items():
        out[name] = stats(t, b["tensors"][name]) if name in b["tensors"] else {"equal": False, "missing_in_b": True}
    if o is not None and b.get("packed"):
        out["k.packed"] = stats(o["pack_int3_values"](a["tensors"]["k.codes"]), b["packed"]["k.packed"])
        out["v.packed"] = stats(o["pack_int2_values"](a["tensors"]["v.codes"]), b["packed"]["v.packed"])
    return out


def all_equal(cmp: dict) -> bool:
    return all(v.get("equal") is True and v.get("bitwise_equal", True) is True for v in cmp.values())


def unequal_fields(cmp: dict) -> list:
    return sorted(k for k, v in cmp.items() if not (v.get("equal") is True and v.get("bitwise_equal", True) is True))


# ------------------------------------------------------------------------------------------------ trace (intermediates)
def _meta_trace(primary: torch.Tensor, prefix: str, out: dict) -> torch.Tensor:
    """The expressions of cq.meta8_roundtrip, exposing the intermediate FP32 group values."""
    data = primary.detach().float().contiguous()
    shape = tuple(data.shape)
    flat = data.reshape(-1)
    pad = (-flat.numel()) % cq.META_GROUP
    if pad:
        flat = torch.cat([flat, flat[-1:].expand(pad)], dim=0)
    g = flat.reshape(-1, cq.META_GROUP)
    mn = g.amin(dim=-1, keepdim=True)
    mx = g.amax(dim=-1, keepdim=True)
    sc = (mx - mn) / 255.0
    sc = torch.where(sc.abs() < 1e-12, torch.ones_like(sc), sc)
    codes = torch.round((g - mn) / sc).clamp(0, 255).to(torch.uint8)
    mn16, sc16 = mn.to(torch.bfloat16), sc.to(torch.bfloat16)
    dec = (codes.float() * sc16.float() + mn16.float()).reshape(-1)
    if pad:
        dec = dec[:-pad]
    dec = dec.reshape(shape)
    out.update({f"{prefix}.group_min_fp32": mn, f"{prefix}.group_max_fp32": mx, f"{prefix}.group_scale_fp32": sc,
                f"{prefix}.codes": codes, f"{prefix}.min_bf16": mn16, f"{prefix}.scale_bf16": sc16,
                f"{prefix}.decoded": dec})
    return dec


def trace(raw_k: torch.Tensor, raw_v: torch.Tensor) -> dict:
    """Intermediate values of the canonical quantization of the OLD region, with the expressions of cq.k3 / cq.v2."""
    out: dict = {}
    t, h, d = raw_k.shape
    x = raw_k.detach().float().permute(1, 0, 2).unsqueeze(0)
    pad = (-t) % cq.GROUP
    if pad:
        x = F.pad(x, (0, 0, 0, pad))
    tp = t + pad
    g = x.reshape(1, h, tp // cq.GROUP, cq.GROUP, d)
    mn = g.amin(dim=3, keepdim=True)
    mx = g.amax(dim=3, keepdim=True)
    sc = (mx - mn) / cq.K_LEVELS
    sc = torch.where(sc.abs() < 1e-8, torch.ones_like(sc), sc)
    codes = torch.round((g - mn) / sc).clamp(0, 7).to(torch.uint8)
    out.update({"k.primary_min": mn, "k.primary_max": mx, "k.primary_scale": sc, "k.codes": codes})
    dmn = _meta_trace(mn, "k.min_meta", out)
    dsc = _meta_trace(sc, "k.scale_meta", out)
    vals = codes.float() * dsc + dmn
    out["k.decoded"] = vals.reshape(1, h, tp, d)[:, :, :t, :].squeeze(0).permute(1, 0, 2).contiguous()

    x = raw_v.detach().float()
    pad = (-d) % cq.GROUP
    if pad:
        x = F.pad(x, (0, pad))
    dp = d + pad
    g = x.reshape(t, h, dp // cq.GROUP, cq.GROUP)
    mn = g.amin(dim=-1, keepdim=True)
    mx = g.amax(dim=-1, keepdim=True)
    sc = (mx - mn) / cq.V_LEVELS
    sc = torch.where(sc.abs() < 1e-8, torch.ones_like(sc), sc)
    codes = torch.round((g - mn) / sc).clamp(0, 3).to(torch.uint8)
    out.update({"v.primary_min": mn, "v.primary_max": mx, "v.primary_scale": sc, "v.codes": codes})
    dmn = _meta_trace(mn, "v.min_meta", out)
    dsc = _meta_trace(sc, "v.scale_meta", out)
    out["v.decoded"] = (codes.float() * dsc + dmn).reshape(t, h, dp)[..., :d].contiguous()
    return out


def trace_matches_frozen_code(tr: dict, raw_k: torch.Tensor, raw_v: torch.Tensor) -> bool:
    """The trace reproduces cq.k3 / cq.v2 on the same device, bit for bit (otherwise it is not used)."""
    t, h, d = raw_k.shape
    k, v = cq.k3(raw_k), cq.v2(raw_v)
    tp = tr["k.codes"].shape[2] * cq.GROUP
    checks = [(tr["k.codes"].reshape(1, h, tp, d).squeeze(0).permute(1, 0, 2), k["codes"]),
              (tr["k.decoded"], k["decoded"]), (tr["v.codes"].reshape(t, h, -1), v["codes"]),
              (tr["v.decoded"], v["decoded"])]
    for side, q in (("k", k), ("v", v)):
        for m in ("min_meta", "scale_meta"):
            checks += [(tr[f"{side}.{m}.codes"], q[m]["codes"]), (tr[f"{side}.{m}.min_bf16"], q[m]["min"]),
                       (tr[f"{side}.{m}.scale_bf16"], q[m]["scale"]), (tr[f"{side}.{m}.decoded"], q[m]["decoded"])]
    return all(a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b) for a, b in checks)


def earliest(cmp: dict, chain: list):
    for name in chain:
        if cmp[name].get("differing", 0) or cmp[name].get("comparable") is False:
            return name
    return None


def code_vs_parameter_differences(ta: dict, tb: dict, side: str) -> dict:
    """Do code differences sit only where the PRIMARY FP32 min / scale they were computed from differ?"""
    ca, cb = ta[f"{side}.codes"].cpu(), tb[f"{side}.codes"].cpu()
    pdiff = (ta[f"{side}.primary_min"].cpu() != tb[f"{side}.primary_min"].cpu()) | (
        ta[f"{side}.primary_scale"].cpu() != tb[f"{side}.primary_scale"].cpu())
    cdiff = ca != cb
    return {"code_differences": int(cdiff.sum()),
            "code_differences_where_primary_min_or_scale_differs": int((cdiff & pdiff.expand_as(cdiff)).sum()),
            "code_differences_where_primary_min_and_scale_are_identical": int((cdiff & ~pdiff.expand_as(cdiff)).sum()),
            "primary_parameter_elements_differing": int(pdiff.sum()), "primary_parameter_elements": int(pdiff.numel())}


def division_probe(ta: dict, tb: dict) -> dict:
    """DESCRIPTIVE probe of the scale expressions `(max - min) / c`: on CPU, from the CPU max / min (only where they
    are bit-identical to the device ones), how many device scale values equal true division and how many equal
    multiplication by the FP32 reciprocal of c. Counts only; no conclusion is drawn here."""
    out = {}
    for name, mx, mn, sc, c, eps in (("k.primary_scale", "k.primary_max", "k.primary_min", "k.primary_scale", cq.K_LEVELS, 1e-8),
                                     ("v.primary_scale", "v.primary_max", "v.primary_min", "v.primary_scale", cq.V_LEVELS, 1e-8)):
        a_mx, a_mn, b_sc = ta[mx].cpu(), ta[mn].cpu(), tb[sc].cpu()
        same_inputs = bool(torch.equal(a_mx, tb[mx].cpu()) and torch.equal(a_mn, tb[mn].cpu()))
        diff = a_mx - a_mn
        div = diff / c
        mul = diff * torch.tensor(1.0 / c, dtype=torch.float32)
        keep = div.abs() >= eps  # elements not replaced by 1 in the frozen expression
        out[name] = {"divisor": c, "max_and_min_bit_identical_on_both_devices": same_inputs, "elements": int(keep.sum()),
                     "device_scale_equals_cpu_true_division": int((b_sc[keep] == div[keep]).sum()),
                     "device_scale_equals_cpu_multiply_by_fp32_reciprocal": int((b_sc[keep] == mul[keep]).sum()),
                     "cpu_scale_equals_cpu_true_division": int((ta[sc].cpu()[keep] == div[keep]).sum())}
    return out


# ------------------------------------------------------------------------------------------------ one raw K / V pair
def layer_report(o: dict, raw_k: torch.Tensor, raw_v: torch.Tensor) -> dict:
    """raw_k / raw_v: BF16 [T, H, D] on the device under test. Returns the full descriptive report."""
    cpu_k, cpu_v = raw_k.cpu(), raw_v.cpu()
    old = int(raw_k.shape[0]) - min(int(raw_k.shape[0]), cq.RESIDUAL)
    rep = {"raw": {"dtype": str(raw_k.dtype), "shape": list(raw_k.shape), "device": str(raw_k.device),
                   "k_sha256": tensor_sha256(cpu_k), "v_sha256": tensor_sha256(cpu_v),
                   "host_copy_bitwise_identical": bool(torch.equal(_bits(cpu_k), _bits(raw_k).cpu())
                                                       and torch.equal(_bits(cpu_v), _bits(raw_v).cpu()))}}
    a, b = canonical_fields(cpu_k, cpu_v), canonical_fields(raw_k, raw_v)
    c, d = oracle_fields(o, raw_k, raw_v), oracle_fields(o, cpu_k, cpu_v)
    rep["cuda_canonical_vs_cuda_oracle"] = compare_fields(b, c, o)
    rep["cpu_canonical_vs_cpu_oracle"] = compare_fields(a, d, o)
    rep["cpu_canonical_vs_cuda_canonical"] = compare_fields(a, b)
    rep["cpu_oracle_vs_cuda_oracle"] = compare_fields(d, c)
    rep["residual_equals_raw"] = {
        "cuda_canonical": bool(torch.equal(b["tensors"]["residual_k_bf16"], raw_k[old:])
                               and torch.equal(b["tensors"]["residual_v_bf16"], raw_v[old:])),
        "cuda_oracle": bool(torch.equal(c["tensors"]["residual_k_bf16"], raw_k[old:])
                            and torch.equal(c["tensors"]["residual_v_bf16"], raw_v[old:]))}
    if old:
        ta, tb = trace(cpu_k[:old], cpu_v[:old]), trace(raw_k[:old], raw_v[:old])
        ok = trace_matches_frozen_code(ta, cpu_k[:old], cpu_v[:old]) and trace_matches_frozen_code(tb, raw_k[:old], raw_v[:old])
        rep["trace_reproduces_frozen_code_on_both_devices"] = ok
        tcmp = {name: stats(ta[name], tb[name]) for name in K_CHAIN + V_CHAIN}
        rep["trace_cpu_vs_cuda"] = tcmp
        rep["earliest_divergence"] = {"k": earliest(tcmp, K_CHAIN), "v": earliest(tcmp, V_CHAIN),
                                      "order": "dependency order of the frozen expressions: primary min, max, scale, "
                                               "codes, META8 of min, META8 of scale, decoded"}
        rep["codes_vs_primary_parameters"] = {s: code_vs_parameter_differences(ta, tb, s) for s in ("k", "v")}
        rep["division_probe"] = division_probe(ta, tb)
    rep["summary"] = {
        "cuda_canonical_equals_cuda_oracle": all_equal(rep["cuda_canonical_vs_cuda_oracle"]),
        "cuda_canonical_vs_cuda_oracle_unequal_fields": unequal_fields(rep["cuda_canonical_vs_cuda_oracle"]),
        "cpu_canonical_equals_cpu_oracle": all_equal(rep["cpu_canonical_vs_cpu_oracle"]),
        "cpu_canonical_equals_cuda_canonical": all_equal(rep["cpu_canonical_vs_cuda_canonical"]),
        "cpu_vs_cuda_canonical_unequal_fields": unequal_fields(rep["cpu_canonical_vs_cuda_canonical"]),
        "cpu_oracle_equals_cuda_oracle": all_equal(rep["cpu_oracle_vs_cuda_oracle"])}
    return rep


# ------------------------------------------------------------------------------------------------ synthetic geometries
SYNTHETIC_LENGTHS = (1, 4, 5, 31, 32, 33, 35, 36, 37, 63, 64, 67, 68, 69, 75)


def synthetic_report(o: dict, parity_suite, device: torch.device) -> dict:
    """Both geometries, the accepted suite's distributions / seeds, boundary lengths; canonical vs oracle ON `device`:
    every field (compare_fields), the accepted T1 comparison, and the accepted T2 token-by-token aging."""
    out = {"lengths": list(SYNTHETIC_LENGTHS), "device": str(device), "geometries": {}}
    for gname, (h, d) in parity_suite.GEOMETRIES.items():
        g = {"kv_heads": h, "head_dim": d, "cases": 0, "field_failures": [], "t1_failures": [], "aging_cases": 0,
             "aging_failures": []}
        for di, dist in enumerate(parity_suite.DISTRIBUTIONS):
            raw_k, raw_v = parity_suite.make_raw(dist, max(SYNTHETIC_LENGTHS), h, d, seed=1400 + 10 * di + h)
            raw_k, raw_v = raw_k.to(device), raw_v.to(device)
            for n in SYNTHETIC_LENGTHS:
                g["cases"] += 1
                cmp = compare_fields(canonical_fields(raw_k[:n], raw_v[:n]), oracle_fields(o, raw_k[:n], raw_v[:n]), o)
                if not all_equal(cmp):
                    g["field_failures"].append({"distribution": dist, "n": n, "fields": unequal_fields(cmp),
                                                "detail": {k: cmp[k] for k in unequal_fields(cmp)}})
                t1 = parity_suite.t1_full_state(o, raw_k[:n], raw_v[:n])
                if t1:
                    g["t1_failures"].append({"distribution": dist, "n": n, "fields": t1})
            for p in parity_suite.PREFILLS:
                g["aging_cases"] += 1
                t2 = parity_suite.t2_aging(o, raw_k, raw_v, p)
                if t2:
                    g["aging_failures"].append({"distribution": dist, "prefill": p, "failures": t2})
        g["passed"] = not (g["field_failures"] or g["t1_failures"] or g["aging_failures"])
        out["geometries"][gname] = g
    out["passed"] = all(g["passed"] for g in out["geometries"].values())
    return out
