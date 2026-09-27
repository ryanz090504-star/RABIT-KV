# SPDX-License-Identifier: Apache-2.0
"""tile32 vs reference Stage3C (non-initial chunked prefill) equivalence.

The reference is the frozen per-token path (Rabit2CausalChunkPlan.apply_step +
rabit2_online_decode_attention_triton per query). tile32 must reproduce, with
EXACT equality: persistent cache bytes, logical runtime state, per-query
attention output, and the following decode step.

Named test_rabit2_* (not test_rabit_kv2*) so the frozen correctness gate's
historical pytest selection is unchanged.
"""

from __future__ import annotations

import pytest
import torch

H, D, QH = 8, 128, 32
SCALE = D ** -0.5
Q_LENS = [2, 31, 32, 33, 64, 1024]
# prefix % 32 == 0, 1, 15, 31 (page-aligned and unaligned chunk starts)
PREFIXES = [64, 65, 79, 95]

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


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


def _assert_same_state(a, b):
    assert a["closed_pages"] == b["closed_pages"] and a["total_tokens"] == b["total_tokens"]
    for name, xa in a.items():
        xb = b[name]
        if isinstance(xa, torch.Tensor) or isinstance(xb, torch.Tensor):
            assert xa is not None and xb is not None, name
            assert xa.dtype == xb.dtype and torch.equal(xa, xb), name


def _diff(a, b):
    d = (a.float() - b.float()).abs()
    rel = d / b.float().abs().clamp_min(1e-12)
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()), "max_rel": float(rel.max())}


def _run_pair(prefix, q_len, seed):
    import vllm.v1.attention.ops.rabit_kv2 as r
    import vllm.v1.attention.ops.rabit_kv2_stage3c_tile32 as t32

    torch.manual_seed(seed)
    total = prefix + q_len + 1
    k = torch.randn((total, H, D), dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    q = torch.randn((q_len + 1, QH, D), dtype=torch.bfloat16, device="cuda")

    ref, cref, btref = _alloc(r, total)
    til, ctil, bttil = _alloc(r, total)
    # Context exactly as the engine builds it: initial chunk via bulk append.
    r.rabit2_bulk_append_exact(ref, k[:prefix], v[:prefix], cref, btref)
    r.rabit2_bulk_append_exact(til, k[:prefix], v[:prefix], ctil, bttil)

    kc, vc, qc = k[prefix:prefix + q_len], v[prefix:prefix + q_len], q[:q_len]
    out_ref = torch.empty((q_len, QH, D), dtype=torch.bfloat16, device="cuda")
    plan = r.Rabit2CausalChunkPlan(ref, kc, vc, cref, btref)
    for i in range(q_len):
        plan.apply_step(i)
        out_ref[i].copy_(r.rabit2_online_decode_attention_triton(qc[i:i + 1], cref, btref, ref,
                                                                 softmax_scale=SCALE)[0])

    out_til = torch.empty_like(out_ref)
    assert t32.rabit2_stage3c_forward_tile32(til, qc, kc, vc, ctil, bttil, out_til, SCALE) is True
    torch.cuda.synchronize()
    return r, ref, cref, btref, til, ctil, bttil, out_ref, out_til, k, v, q, prefix + q_len


def _check(prefix, q_len, seed):
    r, ref, cref, btref, til, ctil, bttil, out_ref, out_til, k, v, q, pos = _run_pair(prefix, q_len, seed)
    assert torch.equal(cref, ctil), "persistent cache bytes differ"
    _assert_same_state(_state(ref), _state(til))
    assert torch.equal(out_ref, out_til), ("attention output differs", _diff(out_til, out_ref))
    # Next decode step on both (exact one-token append path, unchanged).
    ref.append(k[pos:pos + 1], v[pos:pos + 1], cref, btref)
    til.append(k[pos:pos + 1], v[pos:pos + 1], ctil, bttil)
    a = r.rabit2_online_decode_attention_triton(q[-1:], cref, btref, ref, softmax_scale=SCALE)
    b = r.rabit2_online_decode_attention_triton(q[-1:], ctil, bttil, til, softmax_scale=SCALE)
    torch.cuda.synchronize()
    assert torch.equal(cref, ctil), "cache bytes differ after next decode"
    _assert_same_state(_state(ref), _state(til))
    assert torch.equal(a, b), ("next decode output differs", _diff(b, a))


@cuda
@pytest.mark.parametrize("prefix", PREFIXES)
@pytest.mark.parametrize("q_len", Q_LENS)
def test_tile32_matches_reference_exactly(prefix, q_len):
    _check(prefix, q_len, 91000 + prefix * 7 + q_len)


@cuda
@pytest.mark.parametrize("prefix", [4, 5, 31, 32, 36])
@pytest.mark.parametrize("q_len", [2, 33, 64])
def test_tile32_matches_reference_near_residual_and_first_page(prefix, q_len):
    # Small contexts: R4 residual window, open page not yet closed, first closures.
    _check(prefix, q_len, 92000 + prefix * 7 + q_len)


@cuda
@pytest.mark.slow_test
def test_tile32_matches_reference_32k_model_limit_chunk():
    # The Experiment 5 failure shape: first chunk 16384, second chunk q_len 16352.
    _check(16384, 16352, 93000)


@cuda
def test_tile32_falls_back_for_non_gqa4_without_touching_state():
    import vllm.v1.attention.ops.rabit_kv2 as r
    import vllm.v1.attention.ops.rabit_kv2_stage3c_tile32 as t32

    rt, cache, bt = _alloc(r, 128)
    k = torch.randn((64, H, D), dtype=torch.bfloat16, device="cuda")
    r.rabit2_bulk_append_exact(rt, k, k, cache, bt)
    before, cache_before = _state(rt), cache.clone()
    q = torch.randn((8, H, D), dtype=torch.bfloat16, device="cuda")  # ratio 1, not GQA4
    out = torch.empty_like(q)
    assert t32.rabit2_stage3c_forward_tile32(rt, q, k[:8], k[:8], cache, bt, out, SCALE) is False
    _assert_same_state(before, _state(rt))
    assert torch.equal(cache, cache_before)


def test_selector_defaults_to_reference_and_rejects_unknown(monkeypatch):
    import vllm.v1.attention.ops.rabit_kv2_stage3c_tile32 as t32

    monkeypatch.delenv(t32.STAGE3C_IMPL_ENV, raising=False)
    assert t32.rabit2_stage3c_impl() == "reference"
    monkeypatch.setenv(t32.STAGE3C_IMPL_ENV, "reference")
    assert t32.rabit2_stage3c_impl() == "reference"
    monkeypatch.setenv(t32.STAGE3C_IMPL_ENV, "tile32")
    assert t32.rabit2_stage3c_impl() == "tile32"
    monkeypatch.setenv(t32.STAGE3C_IMPL_ENV, "fast")
    with pytest.raises(ValueError):
        t32.rabit2_stage3c_impl()


def test_tiles_cover_steps_with_constant_closed_count():
    import vllm.v1.attention.ops.rabit_kv2_stage3c_tile32 as t32

    for seq in ([5] * 3, [5] * 40, [1, 1, 2, 2, 2, 3], [0] * 33 + [1] * 32 + [2]):
        tiles = t32.rabit2_tile32_tiles(seq)
        assert [i for a, b, _ in tiles for i in range(a, b)] == list(range(len(seq)))
        assert all(b - a <= t32.TILE_MAX_QUERIES and all(seq[i] == c for i in range(a, b)) for a, b, c in tiles)
