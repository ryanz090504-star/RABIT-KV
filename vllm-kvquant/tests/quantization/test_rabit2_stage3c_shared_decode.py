# SPDX-License-Identifier: Apache-2.0
"""shared_decode vs reference (and tile32) Stage3C non-initial chunked prefill.

The reference is the frozen per-token path (Rabit2CausalChunkPlan.apply_step +
rabit2_online_decode_attention_triton per query) and remains the oracle.
tile32 and shared_decode (every QUERY_BLOCK) must reproduce, with EXACT
equality (torch.equal; never allclose): persistent cache bytes, logical
runtime state, per-query attention output, and the following decode step.
A kernel-side decode counter (compiled only in the COUNT_DECODES variant)
proves each compressed page is decoded once per query block, not per query.

Named test_rabit2_* (not test_rabit_kv2*) so the frozen correctness gate's
historical pytest selection is unchanged.
"""

from __future__ import annotations

import ast
import difflib
import inspect
import json
import math

import pytest
import torch

H, D, QH = 8, 128, 32
SCALE = D ** -0.5
QGROUPS = H * ((QH // H) // 4)
Q_LENS = [2, 31, 32, 33, 64, 512, 1024]
PREFIXES = [64, 65, 79, 95]  # prefix % 32 == 0, 1, 15, 31
QUERY_BLOCKS = [4, 8, 16, 32]

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
_REF_CACHE: dict = {}


def _mods():
    import vllm.v1.attention.ops.rabit_kv2 as r
    import vllm.v1.attention.ops.rabit_kv2_stage3c_shared_decode as sd
    import vllm.v1.attention.ops.rabit_kv2_stage3c_tile32 as t32

    return r, t32, sd


def _alloc(r, tokens):
    rt = r.Rabit2SingleSequenceRuntime(H, D, D)
    pages = (tokens + 31) // 32 + 16
    cache = torch.zeros((pages, 1, 1, 1, rt.layout.page_bytes), dtype=torch.uint8, device="cuda")
    bt = torch.arange(pages, dtype=torch.int32, device="cuda")
    return rt, cache, bt


def _state(rt):
    out = {"closed_pages": int(rt.closed_pages), "total_tokens": int(rt.total_tokens)}
    for name in ("open_k", "open_v_packed", "open_v_min", "open_v_scale", "recent_k", "recent_v"):
        x = getattr(rt, name)
        out[name] = None if x is None else x.clone()
    return out


def _state_equal(a, b):
    if a["closed_pages"] != b["closed_pages"] or a["total_tokens"] != b["total_tokens"]:
        return False
    for name, xa in a.items():
        xb = b[name]
        if isinstance(xa, torch.Tensor) or isinstance(xb, torch.Tensor):
            if xa is None or xb is None or xa.dtype != xb.dtype or not torch.equal(xa, xb):
                return False
    return True


def _mismatch(a, b):
    """First-failure report (never used to relax the check)."""
    d = (a.float() - b.float()).abs()
    rel = d / b.float().abs().clamp_min(1e-12)
    idx = (a != b).nonzero()
    first = idx[0].tolist() if len(idx) else None
    return {"first_index": first, "first_values": None if first is None else
            [float(a[tuple(first)]), float(b[tuple(first)])],
            "n_mismatch": int(len(idx)), "max_abs": float(d.max()), "max_rel": float(rel.max()),
            "mean_abs": float(d.mean())}


def _inputs(prefix, q_len, seed):
    torch.manual_seed(seed)
    total = prefix + q_len + 1
    k = torch.randn((total, H, D), dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    q = torch.randn((q_len + 1, QH, D), dtype=torch.bfloat16, device="cuda")
    return k, v, q


def _run(impl, prefix, q_len, seed, query_block=None, counter=None):
    """One chunk with the given implementation, then the next decode step. Returns everything compared."""
    r, t32, sd = _mods()
    k, v, q = _inputs(prefix, q_len, seed)
    rt, cache, bt = _alloc(r, prefix + q_len + 1)
    r.rabit2_bulk_append_exact(rt, k[:prefix], v[:prefix], cache, bt)
    kc, vc, qc = k[prefix:prefix + q_len], v[prefix:prefix + q_len], q[:q_len]
    out = torch.empty((q_len, QH, D), dtype=torch.bfloat16, device="cuda")
    if impl == "reference":
        plan = r.Rabit2CausalChunkPlan(rt, kc, vc, cache, bt)
        for i in range(q_len):
            plan.apply_step(i)
            out[i].copy_(r.rabit2_online_decode_attention_triton(qc[i:i + 1], cache, bt, rt, softmax_scale=SCALE)[0])
    elif impl == "tile32":
        assert t32.rabit2_stage3c_forward_tile32(rt, qc, kc, vc, cache, bt, out, SCALE) is True
    else:
        assert sd.rabit2_stage3c_forward_shared_decode(rt, qc, kc, vc, cache, bt, out, SCALE,
                                                       query_block=query_block, decode_counter=counter) is True
    torch.cuda.synchronize()
    chunk = {"cache": cache.clone(), "state": _state(rt), "out": out}
    pos = prefix + q_len
    rt.append(k[pos:pos + 1], v[pos:pos + 1], cache, bt)
    nxt = r.rabit2_online_decode_attention_triton(q[-1:], cache, bt, rt, softmax_scale=SCALE)
    torch.cuda.synchronize()
    return chunk, {"cache": cache, "state": _state(rt), "out": nxt}


def _reference(prefix, q_len, seed):
    key = (prefix, q_len, seed)
    if key not in _REF_CACHE:
        _REF_CACHE.clear()  # keep at most one (possibly 16352-token) oracle resident
        _REF_CACHE[key] = _run("reference", prefix, q_len, seed)
    return _REF_CACHE[key]


def _assert_exact(name, got, ref, shape):
    for phase in (0, 1):
        g, o = got[phase], ref[phase]
        tag = f"{name} {shape} {'chunk' if phase == 0 else 'next decode'}"
        cache_ok, state_ok = torch.equal(g["cache"], o["cache"]), _state_equal(g["state"], o["state"])
        assert cache_ok, f"{tag}: persistent cache bytes differ"
        assert state_ok, f"{tag}: runtime state differs"
        assert torch.equal(g["out"], o["out"]), (
            f"{tag}: attention output differs (cache exact={cache_ok}, state exact={state_ok})",
            _mismatch(g["out"], o["out"]))


def _seed(prefix, q_len):
    return 97000 + prefix * 7 + q_len


@cuda
@pytest.mark.parametrize("query_block", QUERY_BLOCKS)
@pytest.mark.parametrize("prefix", PREFIXES)
@pytest.mark.parametrize("q_len", Q_LENS)
def test_shared_decode_matches_reference_exactly(q_len, prefix, query_block):
    ref = _reference(prefix, q_len, _seed(prefix, q_len))
    _assert_exact(f"shared_decode[qb={query_block}]",
                  _run("shared_decode", prefix, q_len, _seed(prefix, q_len), query_block), ref,
                  (prefix, q_len))
    if query_block == QUERY_BLOCKS[0]:
        _assert_exact("tile32", _run("tile32", prefix, q_len, _seed(prefix, q_len)), ref, (prefix, q_len))


@cuda
@pytest.mark.parametrize("query_block", QUERY_BLOCKS)
@pytest.mark.parametrize("prefix", [4, 5, 31, 32, 36])
@pytest.mark.parametrize("q_len", [2, 33, 64])
def test_shared_decode_near_residual_and_first_page(q_len, prefix, query_block):
    # R4 residual window, open page not yet closed, first and later page closures.
    ref = _reference(prefix, q_len, _seed(prefix, q_len))
    _assert_exact(f"shared_decode[qb={query_block}]",
                  _run("shared_decode", prefix, q_len, _seed(prefix, q_len), query_block), ref,
                  (prefix, q_len))


@cuda
@pytest.mark.parametrize("query_block", QUERY_BLOCKS)
def test_shared_decode_32k_model_limit_chunk(query_block):
    # The Experiment 5 failure shape: first chunk 16384, second chunk q_len 16352.
    ref = _reference(16384, 16352, 98000)
    _assert_exact(f"shared_decode[qb={query_block}]",
                  _run("shared_decode", 16384, 16352, 98000, query_block), ref, (16384, 16352))
    if query_block == QUERY_BLOCKS[0]:
        _assert_exact("tile32", _run("tile32", 16384, 16352, 98000), ref, (16384, 16352))


def _expected_decodes(prefix, q_len, query_block):
    r, t32, _ = _mods()
    rt, cache, bt = _alloc(r, prefix + q_len)
    k = torch.zeros((prefix + q_len, H, D), dtype=torch.bfloat16, device="cuda")
    r.rabit2_bulk_append_exact(rt, k[:prefix], k[:prefix], cache, bt)
    plan = r.Rabit2CausalChunkPlan(rt, k[prefix:], k[prefix:], cache, bt)
    tiles = t32.rabit2_tile32_tiles([t32.rabit2_chunk_closed_pages_after(plan, i) for i in range(q_len)])
    shared = sum(c * QGROUPS * math.ceil((b - a) / query_block) for a, b, c in tiles)
    per_query = sum(c * QGROUPS * (b - a) for a, b, c in tiles)  # tile32 / reference: one decode per query
    return shared, per_query, len(tiles)


@cuda
@pytest.mark.parametrize("query_block", QUERY_BLOCKS)
@pytest.mark.parametrize("prefix,q_len", [(79, 33), (95, 1024), (16384, 512)])
def test_shared_decode_decodes_once_per_query_block(prefix, q_len, query_block):
    """Kernel-side atomic count of closed-page decodes (COUNT_DECODES variant); outputs still exact."""
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    got = _run("shared_decode", prefix, q_len, _seed(prefix, q_len), query_block, counter)
    shared, per_query, tiles = _expected_decodes(prefix, q_len, query_block)
    measured = int(counter.item())
    print("SHARED_DECODE_COUNTS=" + json.dumps({
        "prefix": prefix, "q_len": q_len, "query_block": query_block, "tiles": tiles,
        "measured_shared_decode_page_decodes": measured, "expected_shared_decode": shared,
        "per_query_page_decodes_tile32_reference": per_query,
        "reuse_factor": per_query / measured if measured else None}, sort_keys=True))
    assert measured == shared > 0
    assert per_query >= shared and (q_len < 2 or query_block == 1 or per_query > shared)
    _assert_exact(f"shared_decode[qb={query_block},count]", got,
                  _reference(prefix, q_len, _seed(prefix, q_len)), (prefix, q_len))


@cuda
def test_shared_decode_falls_back_for_non_gqa4_without_touching_state():
    r, _, sd = _mods()
    rt, cache, bt = _alloc(r, 128)
    k = torch.randn((64, H, D), dtype=torch.bfloat16, device="cuda")
    r.rabit2_bulk_append_exact(rt, k, k, cache, bt)
    before, cache_before = _state(rt), cache.clone()
    q = torch.randn((8, H, D), dtype=torch.bfloat16, device="cuda")  # ratio 1, not GQA4
    out = torch.empty_like(q)
    assert sd.rabit2_stage3c_forward_shared_decode(rt, q, k[:8], k[:8], cache, bt, out, SCALE) is False
    assert _state_equal(before, _state(rt)) and torch.equal(cache, cache_before)


def test_selector_three_way_default_reference(monkeypatch):
    _, t32, sd = _mods()
    monkeypatch.delenv(sd.STAGE3C_IMPL_ENV, raising=False)
    assert sd.rabit2_stage3c_impl() == "reference"
    for impl in ("reference", "tile32", "shared_decode"):
        monkeypatch.setenv(sd.STAGE3C_IMPL_ENV, impl)
        assert sd.rabit2_stage3c_impl() == impl
    monkeypatch.setenv(sd.STAGE3C_IMPL_ENV, "fast")
    with pytest.raises(ValueError):
        sd.rabit2_stage3c_impl()
    # The frozen tile32 selector is untouched (it still knows only reference / tile32).
    monkeypatch.setenv(sd.STAGE3C_IMPL_ENV, "tile32")
    assert t32.rabit2_stage3c_impl() == "tile32" and t32.STAGE3C_IMPLS == ("reference", "tile32")


def test_query_block_selector(monkeypatch):
    _, _, sd = _mods()
    monkeypatch.delenv(sd.QUERY_BLOCK_ENV, raising=False)
    assert sd.rabit2_shared_decode_query_block() == sd.DEFAULT_QUERY_BLOCK == 8
    for qb in (4, 8, 16, 32):
        monkeypatch.setenv(sd.QUERY_BLOCK_ENV, str(qb))
        assert sd.rabit2_shared_decode_query_block() == qb
    for bad in ("12", "64", "0", "x"):
        monkeypatch.setenv(sd.QUERY_BLOCK_ENV, bad)
        with pytest.raises(ValueError):
            sd.rabit2_shared_decode_query_block()


def test_decode_and_per_query_body_are_verbatim_reference():
    r, _, sd = _mods()
    ref = inspect.getsource(r._rabit2_stage4b3_gqa4_closed_page_partial_kernel.fn).splitlines()
    ref = [ln.strip() for ln in ref[next(i for i, ln in enumerate(ref) if "logical_page = tl.program_id(0)" in ln):]]
    src = inspect.getsource(sd._rabit2_shared_decode_closed_page_partial_kernel.fn).splitlines()
    cut = lambda a, b: [ln.strip() for ln in src[next(i for i, x in enumerate(src) if a in x) + 1:  # noqa: E731
                                                  next(i for i, x in enumerate(src) if b in x)]]
    mine = cut("BEGIN VERBATIM REFERENCE DECODE", "END VERBATIM REFERENCE DECODE") + \
        cut("BEGIN VERBATIM REFERENCE PER-QUERY BODY", "END VERBATIM REFERENCE PER-QUERY BODY")
    diff = [ln for ln in difflib.unified_diff([x for x in ref if x], [x for x in mine if x], lineterm="", n=0)
            if not ln.startswith(("---", "+++", "@@"))]
    # Only the query-row / partial pointer bases and the partial-block masks may differ.
    assert sorted(diff) == sorted([
        "-# Reuse decoded K/V for four Q heads.",
        "-q_ptr + qh * HEAD_SIZE + d,", "+q_row + qh * HEAD_SIZE + d,",
        "-mask=dmask,", "+mask=dmask & valid,", "-mask=dmask,", "+mask=dmask & valid,",
        "-tl.store(partial_m_ptr + seg, m)", "+tl.store(pm + seg, m, mask=valid)",
        "-tl.store(partial_l_ptr + seg, l)", "+tl.store(pl + seg, l, mask=valid)",
        "-partial_acc_ptr + seg * HEAD_SIZE + d,", "+pacc + seg * HEAD_SIZE + d,",
    ]), diff
    # The decode happens before (outside) the per-query loop: exactly one decode per program.
    tree = ast.parse(inspect.getsource(sd._rabit2_shared_decode_closed_page_partial_kernel.fn).lstrip())
    fn = tree.body[0]
    loop = next(n for n in fn.body if isinstance(n, ast.For) and ast.unparse(n.iter) == "range(QUERY_BLOCK)")
    in_loop = {id(n) for n in ast.walk(loop)}
    decode_loads = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and ast.unparse(n.func) == "tl.load"
                    and "page" in ast.unparse(n.args[0])]
    assert decode_loads and not any(id(n) in in_loop for n in decode_loads)
