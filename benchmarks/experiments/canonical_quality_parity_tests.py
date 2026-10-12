"""
canonical-quality-v2 PARITY TESTS (CPU; torch required -- run in the CPU-only container of
canonical_quality_parity_modal.py; no GPU, no model). Correctness testing only, not a quality experiment.

ORACLE: the pure-torch *_ref functions and RABIT2_* constants of the FROZEN
vllm-kvquant/vllm/v1/attention/ops/kvquant_k3.py, extracted by AST (the module itself imports vLLM / Triton) -- the
same quantize_rabit2_kv_ref / dequantize_rabit2_kv_ref that Rabit2OnlineStateRef equals
(tests/quantization/test_rabit_kv2_physical.py) and that the physical pages reproduce byte-exactly.
OLD HARNESS (negative control): the quantizer closures of the frozen benchmarks/quality/hotpotqa.py, extracted by AST.

Checks, for Qwen (H_kv 4) and Llama (H_kv 8), D = 128, four value distributions and N in LENGTHS:
  T1 full-state parity: residual / open / closed membership, K3 codes, K min / scale META8 codes + BF16 min / scale,
     V2 codes, V min / scale META8, decoded K and V -- all torch.equal (bit-exact);
  T2 sequential aging: CanonicalLayerState grown token by token from several prefill lengths; decoded state equal to the
     oracle state of the same first N raw tokens at EVERY N (detects cumulative requantization);
  T3 HF integration: CanonicalRabitCache.from_prefill + one-token update(): returned HF-layout BF16 K / V equal the
     oracle decoded state cast to BF16, and get_seq_length() == N;
  T4 negative control: the OLD harness reproduces K but NOT canonical V (V META8g64 grouping order).
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import canonical_rabit_quality as cq  # noqa: E402

LENGTHS = (1, 4, 5, 31, 32, 33, 35, 36, 37, 67, 68, 69, 75)
GEOMETRIES = {"qwen2_5_7b": (4, 128), "llama3_1_8b": (8, 128)}
DISTRIBUTIONS = ("normal", "asymmetric", "outliers", "near_zero")
PREFILLS = (0, 1, 4, 5, 33, 36, 68)
ORACLE_FUNCS = {"_kvquant_k3_packed_dim", "pack_int3_values", "unpack_int3_values", "_packed_dim", "pack_lowbit_values",
                "unpack_lowbit_values", "pack_int2_values", "unpack_int2_values", "encode_metadata_uint8_group_ref",
                "decode_metadata_uint8_group_ref", "quantize_k3_sequence_affine_ref", "dequantize_k3_sequence_affine_ref",
                "quantize_v2_group_affine_ref", "dequantize_v2_group_affine_ref", "quantize_rabit2_kv_ref",
                "dequantize_rabit2_kv_ref"}
OLD_FUNCS = {"encode_metadata", "decode_metadata", "q_group_sym", "q_group_affine", "q_seq_affine", "q_tensor",
             "q_with_residual", "dequantize_state"}
OLD_RABIT2 = {"k_bits": 3, "v_bits": 2, "k_style": "seq_affine", "v_style": "group_affine", "k_group": 32,
              "v_group": 32, "residual": 4, "metadata_mode": "int8", "metadata_group_size": 64}


def load_oracle(path: Path) -> tuple[dict, dict]:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in ORACLE_FUNCS) or (
        isinstance(n, ast.Assign) and all(isinstance(t, ast.Name) and t.id.startswith(("RABIT2_", "KVQUANT_K3_")) for t in n.targets))]
    missing = ORACLE_FUNCS - {n.name for n in keep if isinstance(n, ast.FunctionDef)}
    if missing:
        raise RuntimeError(f"oracle functions not found: {missing}")
    ns: dict = {"torch": torch, "Any": Any, "__name__": "kvquant_k3_oracle"}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), ns)  # noqa: S102
    seg = "\n".join(ast.get_source_segment(src, n) for n in keep)
    return ns, {"file_sha256_lf": hashlib.sha256(src.replace("\r\n", "\n").encode()).hexdigest(),
                "extracted_sha256": hashlib.sha256(seg.encode()).hexdigest(), "functions": sorted(ORACLE_FUNCS)}


def load_old_harness(path: Path) -> dict:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in OLD_FUNCS]
    ns: dict = {"torch": torch, "F": F, "dtype": torch.bfloat16}
    exec(compile(ast.Module(body=fns, type_ignores=[]), str(path), "exec"), ns)  # noqa: S102
    return ns


def make_raw(dist: str, n: int, h: int, d: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    k = torch.randn((n, h, d), generator=g)
    v = torch.randn((n, h, d), generator=g)
    if dist == "asymmetric":  # per-channel offsets and scales (K-bias-like), skewed V
        k = k * (0.2 + 3 * torch.rand((1, h, d), generator=g)) + 25 * torch.randn((1, h, d), generator=g)
        v = v.abs() ** 1.5 * (0.5 + torch.rand((1, h, d), generator=g)) - 0.3
    elif dist == "outliers":  # a few huge channels / tokens
        k[:, :, :: 37] *= 150.0
        v[:: 11] *= 60.0
    elif dist == "near_zero":
        k = k * 1e-3
        v = v * 1e-4
        k[:, :, :7] = 0.0
        v[::3, :, 5:40] = 0.0
    return k.to(torch.bfloat16), v.to(torch.bfloat16)


def eq(a, b) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)


def meta_eq(mine: dict, orc: dict) -> bool:
    return eq(mine["codes"], orc["codes"]) and eq(mine["min"], orc["min"]) and eq(mine["scale"], orc["scale"]) \
        and mine["pad"] == orc["pad"]


def t1_full_state(o, raw_k, raw_v) -> list:
    fails = []
    n = raw_k.shape[0]
    st, os_ = cq.canonical_state(raw_k, raw_v), o["quantize_rabit2_kv_ref"](raw_k, raw_v)
    if (st["old_count"], st["recent_count"]) != (os_["old_count"], n - os_["old_count"]):
        fails.append("membership")
    if st["old_count"]:
        ok, ov = os_["old_k"], os_["old_v"]
        d = raw_k.shape[2]
        if not eq(st["k"]["codes"], o["unpack_int3_values"](ok["packed"], d)):
            fails.append("K3 codes")
        if not (meta_eq(st["k"]["min_meta"], ok["min"]) and meta_eq(st["k"]["scale_meta"], ok["scale"])):
            fails.append("K metadata")
        if not eq(st["v"]["codes"], o["unpack_int2_values"](ov["packed"], ov["padded_dim"])):
            fails.append("V2 codes")
        if not (meta_eq(st["v"]["min_meta"], ov["min"]) and meta_eq(st["v"]["scale_meta"], ov["scale"])):
            fails.append("V metadata")
        if st["closed_pages"] != os_["old_count"] // 32 or st["open_count"] != os_["old_count"] % 32:
            fails.append("page/open membership")
    dk, dv = o["dequantize_rabit2_kv_ref"](os_, dtype=torch.float32)
    if not eq(st["decoded_k"], dk):
        fails.append("decoded K")
    if not eq(st["decoded_v"], dv):
        fails.append("decoded V")
    return fails


def t2_aging(o, raw_k, raw_v, prefill: int) -> list:
    fails = []
    s = cq.CanonicalLayerState()
    start = max(prefill, 1)
    s.append(raw_k[:start], raw_v[:start])
    for n in range(start, raw_k.shape[0] + 1):
        if n > start:
            s.append(raw_k[n - 1:n], raw_v[n - 1:n])
        if s.n != n:
            fails.append(f"N={n}: length {s.n}")
            break
        dk, dv = s.decoded()
        ok, ov = o["dequantize_rabit2_kv_ref"](o["quantize_rabit2_kv_ref"](raw_k[:n], raw_v[:n]), dtype=torch.float32)
        if not (eq(dk, ok) and eq(dv, ov)):
            fails.append(f"N={n}: decoded state differs from oracle")
            break
    return fails


def t3_hf(o, raw_k, raw_v, prefill: int) -> list:
    from transformers.cache_utils import DynamicCache  # noqa: PLC0415
    Cache = cq.make_canonical_cache_class()
    layers = 2
    pre = DynamicCache()
    for li in range(layers):
        pre.update(raw_k[:prefill].permute(1, 0, 2).unsqueeze(0), raw_v[:prefill].permute(1, 0, 2).unsqueeze(0), li)
    c = Cache.from_prefill(pre)
    fails = []
    for n in range(prefill, raw_k.shape[0] + 1):
        if n > prefill:
            for li in range(layers):
                rk, rv = c.update(raw_k[n - 1:n].permute(1, 0, 2).unsqueeze(0),
                                  raw_v[n - 1:n].permute(1, 0, 2).unsqueeze(0), li)
        ok, ov = o["dequantize_rabit2_kv_ref"](o["quantize_rabit2_kv_ref"](raw_k[:n], raw_v[:n]), dtype=torch.float32)
        want_k = ok.permute(1, 0, 2).unsqueeze(0).to(torch.bfloat16)
        want_v = ov.permute(1, 0, 2).unsqueeze(0).to(torch.bfloat16)
        if c.get_seq_length() != n or not all(eq(c.key_cache[li], want_k) and eq(c.value_cache[li], want_v)
                                              for li in range(layers)):
            fails.append(f"N={n}: HF cache state differs")
            break
    try:
        c.update(raw_k[:2].permute(1, 0, 2).unsqueeze(0), raw_v[:2].permute(1, 0, 2).unsqueeze(0), 0)
        fails.append("multi-token update accepted")
    except ValueError:
        pass
    return fails


def t4_old_harness(o, oldh, raw_k, raw_v) -> dict:
    """Old harness on HF layout [1, H, N, D]; compare its dequantized K / V with the canonical oracle (BF16)."""
    n = raw_k.shape[0]
    hk, hv = raw_k.permute(1, 0, 2).unsqueeze(0), raw_v.permute(1, 0, 2).unsqueeze(0)
    qk = oldh["dequantize_state"](oldh["q_with_residual"](hk, 3, "k", OLD_RABIT2))
    qv = oldh["dequantize_state"](oldh["q_with_residual"](hv, 2, "v", OLD_RABIT2))
    ok, ov = o["dequantize_rabit2_kv_ref"](o["quantize_rabit2_kv_ref"](raw_k, raw_v), dtype=torch.float32)
    wk = ok.permute(1, 0, 2).unsqueeze(0).to(torch.bfloat16)
    wv = ov.permute(1, 0, 2).unsqueeze(0).to(torch.bfloat16)
    return {"n": n, "old_k_equals_canonical": eq(qk, wk), "old_v_equals_canonical": eq(qv, wv)}


def run(repo_root: Path) -> dict:
    o, oracle_meta = load_oracle(repo_root / "vllm-kvquant/vllm/v1/attention/ops/kvquant_k3.py")
    oldh = load_old_harness(repo_root / "benchmarks/quality/hotpotqa.py")
    torch.set_num_threads(max(1, torch.get_num_threads()))
    res: dict = {"oracle": oracle_meta, "lengths": list(LENGTHS), "prefills": list(PREFILLS), "geometries": {}}
    for gname, (h, d) in GEOMETRIES.items():
        g: dict = {"T1_full_state": {}, "T2_sequential_aging": {}, "T3_hf_cache": {}, "T4_old_harness": {}}
        for di, dist in enumerate(DISTRIBUTIONS):
            raw_k, raw_v = make_raw(dist, max(LENGTHS), h, d, seed=1400 + 10 * di + h)
            g["T1_full_state"][dist] = {str(n): t1_full_state(o, raw_k[:n], raw_v[:n]) for n in LENGTHS}
            g["T2_sequential_aging"][dist] = {str(p): t2_aging(o, raw_k, raw_v, p) for p in PREFILLS}
            if dist in ("normal", "asymmetric"):
                g["T3_hf_cache"][dist] = {str(p): t3_hf(o, raw_k, raw_v, max(p, 1)) for p in (1, 5, 36)}
            g["T4_old_harness"][dist] = [t4_old_harness(o, oldh, raw_k[:n], raw_v[:n]) for n in LENGTHS]
        flat = [f for sect in ("T1_full_state", "T2_sequential_aging", "T3_hf_cache") for dd in g[sect].values()
                for f in dd.values() if f]
        t4 = [x for dd in g["T4_old_harness"].values() for x in dd]
        g["summary"] = {
            "parity_failures": sum(len(f) for f in flat),
            "T4_old_k_matches_everywhere": all(x["old_k_equals_canonical"] for x in t4),
            "T4_old_v_mismatch_count": sum(not x["old_v_equals_canonical"] for x in t4),
            "T4_old_v_mismatch_min_n": min((x["n"] for x in t4 if not x["old_v_equals_canonical"]), default=None),
            "T4_cases": len(t4)}
        res["geometries"][gname] = g
    res["passed"] = all(G["summary"]["parity_failures"] == 0 for G in res["geometries"].values())
    res["negative_control_passed"] = all(G["summary"]["T4_old_v_mismatch_count"] > 0
                                         for G in res["geometries"].values())
    res["torch_version"] = torch.__version__
    res["python_version"] = sys.version.split()[0]
    return res


if __name__ == "__main__":
    r = run(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[1])
    print(json.dumps({k: v for k, v in r.items() if k != "geometries"} | {
        g: G["summary"] for g, G in r["geometries"].items()}, indent=1))
    raise SystemExit(0 if r["passed"] and r["negative_control_passed"] else 1)
