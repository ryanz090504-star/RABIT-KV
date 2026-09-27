# SPDX-License-Identifier: Apache-2.0
"""Stage3C component profiler: OFF by default and pure; pure observation when ON.

With VLLM_RABIT2_STAGE3C_COMPONENT_PROFILE unset the scope is a shared no-op:
no patching, no CUDA events, no synchronize, no records. With it set, reference
and tile32 Stage3C must produce EXACTLY the same cache bytes, runtime state and
attention output as without profiling, every patched attribute must be restored
(after normal exit and after an exception), and the record must satisfy the
accounting contract (exclusive HOST windows sum to the wall; GPU leaves are
event pairs, one per call, never exceeding the span).

Named test_rabit2_* (not test_rabit_kv2*) so the frozen correctness gate's
historical pytest selection is unchanged.
"""

from __future__ import annotations

import pytest
import torch

H, D, QH = 8, 128, 32
SCALE = D ** -0.5
CASES = [(64, 2), (79, 33), (95, 64), (16384, 32)]
TOL = 1e-3

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _mods():
    import vllm.v1.attention.ops.rabit_kv2 as r
    import vllm.v1.attention.ops.rabit_kv2_stage3c_profile as prof
    import vllm.v1.attention.ops.rabit_kv2_stage3c_tile32 as t32

    return r, t32, prof


def _patchable_snapshot():
    r, t32, prof = _mods()
    snap = {("plan", n): r.Rabit2CausalChunkPlan.__dict__[n] for n, _ in prof._CHUNK_PLAN_METHODS}
    for owner, table in ((r, prof._RABIT_FUNCS), (r, prof._RABIT_KERNELS), (t32, prof._TILE32_KERNELS)):
        for name, _ in table:
            snap[(owner.__name__, name)] = getattr(owner, name)
    return snap


def _assert_restored(before):
    after = _patchable_snapshot()
    assert before.keys() == after.keys()
    for k in before:
        assert after[k] is before[k], k


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


def _setup(prefix, q_len, seed):
    r, _, _ = _mods()
    torch.manual_seed(seed)
    k = torch.randn((prefix + q_len, H, D), dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    q = torch.randn((q_len, QH, D), dtype=torch.bfloat16, device="cuda")
    rt, cache, bt = _alloc(r, prefix + q_len)
    r.rabit2_bulk_append_exact(rt, k[:prefix], v[:prefix], cache, bt)
    torch.cuda.synchronize()
    return rt, cache, bt, q, k[prefix:], v[prefix:]


def _run(impl, rt, cache, bt, q, kc, vc):
    """One Stage3C chunk exactly as TritonAttentionImpl runs it (reference loop or tile32)."""
    r, t32, _ = _mods()
    q_len = q.shape[0]
    out = torch.empty((q_len, QH, D), dtype=torch.bfloat16, device="cuda")
    if impl == "tile32":
        assert t32.rabit2_stage3c_forward_tile32(rt, q, kc, vc, cache, bt, out, SCALE) is True
    else:
        plan = r.Rabit2CausalChunkPlan(rt, kc, vc, cache, bt)
        for i in range(q_len):
            plan.apply_step(i)
            out[i : i + 1].copy_(r.rabit2_online_decode_attention_triton(q[i : i + 1], cache, bt, rt,
                                                                          softmax_scale=SCALE))
    return out


def _chunk(impl, prefix, q_len, seed, profiled):
    _, _, prof = _mods()
    rt, cache, bt, q, kc, vc = _setup(prefix, q_len, seed)
    scope = prof.rabit2_stage3c_profile_scope(q_len, prefix)
    assert (scope is prof._NULL_SCOPE) is (not profiled)
    with scope as p:
        out = _run(impl, rt, cache, bt, q, kc, vc)
    torch.cuda.synchronize()
    return rt, cache, out, (p.record() if profiled else None)


def _expected_tiles(prefix, q_len):
    r, t32, _ = _mods()
    rt, cache, bt = _alloc(r, prefix + q_len)
    k = torch.zeros((prefix + q_len, H, D), dtype=torch.bfloat16, device="cuda")
    r.rabit2_bulk_append_exact(rt, k[:prefix], k[:prefix], cache, bt)
    plan = r.Rabit2CausalChunkPlan(rt, k[prefix:], k[prefix:], cache, bt)
    return t32.rabit2_tile32_tiles([t32.rabit2_chunk_closed_pages_after(plan, i) for i in range(q_len)])


def test_profile_flag_defaults_off(monkeypatch):
    _, _, prof = _mods()
    monkeypatch.delenv(prof.COMPONENT_PROFILE_ENV, raising=False)
    assert prof.rabit2_stage3c_component_profiling() is False
    assert prof.rabit2_stage3c_profile_scope(512, 16384) is prof._NULL_SCOPE
    monkeypatch.setenv(prof.COMPONENT_PROFILE_ENV, "0")
    assert prof.rabit2_stage3c_profile_scope(512, 16384) is prof._NULL_SCOPE
    monkeypatch.setenv(prof.COMPONENT_PROFILE_ENV, "1")
    assert prof.rabit2_stage3c_component_profiling() is True
    # Decode (q_len == 1) is never profiled.
    assert prof.rabit2_stage3c_profile_scope(1, 16384) is prof._NULL_SCOPE


@cuda
@pytest.mark.parametrize("impl", ["reference", "tile32"])
def test_profile_off_is_pure(monkeypatch, impl):
    """OFF: no patching, no CUDA events, no synchronize, no records, during a real chunk."""
    _, t32, prof = _mods()
    monkeypatch.setenv(t32.STAGE3C_IMPL_ENV, impl)
    monkeypatch.delenv(prof.COMPONENT_PROFILE_ENV, raising=False)
    rt, cache, bt, q, kc, vc = _setup(64, 33, 95000)
    before, pool = _patchable_snapshot(), len(prof._EVENT_POOL)
    counts = {"event": 0, "sync": 0, "log": 0}
    real_event, real_sync = torch.cuda.Event, torch.cuda.synchronize

    def ev(*a, **k):
        counts["event"] += 1
        return real_event(*a, **k)

    def sy(*a, **k):
        counts["sync"] += 1
        return real_sync(*a, **k)

    monkeypatch.setattr(torch.cuda, "Event", ev)
    monkeypatch.setattr(torch.cuda, "synchronize", sy)
    monkeypatch.setattr(prof.logger, "info", lambda *a, **k: counts.__setitem__("log", counts["log"] + 1))
    with prof.rabit2_stage3c_profile_scope(33, 64):
        _assert_restored(before)
        _run(impl, rt, cache, bt, q, kc, vc)
    monkeypatch.setattr(torch.cuda, "synchronize", real_sync)
    torch.cuda.synchronize()
    assert counts == {"event": 0, "sync": 0, "log": 0}, counts
    assert len(prof._EVENT_POOL) == pool
    _assert_restored(before)


@cuda
@pytest.mark.parametrize("impl", ["reference", "tile32"])
@pytest.mark.parametrize("prefix,q_len", CASES)
def test_profile_on_is_exact_and_accounts(monkeypatch, impl, prefix, q_len):
    _, t32, prof = _mods()
    monkeypatch.setenv(t32.STAGE3C_IMPL_ENV, impl)
    before = _patchable_snapshot()
    seed = 94000 + prefix + q_len
    monkeypatch.delenv(prof.COMPONENT_PROFILE_ENV, raising=False)
    rt0, c0, o0, _ = _chunk(impl, prefix, q_len, seed, profiled=False)
    monkeypatch.setenv(prof.COMPONENT_PROFILE_ENV, "1")
    rt1, c1, o1, rec = _chunk(impl, prefix, q_len, seed, profiled=True)
    _assert_restored(before)
    assert torch.equal(c0, c1), "cache bytes differ with profiling ON"
    _assert_same_state(_state(rt0), _state(rt1))
    assert torch.equal(o0, o1), "attention output differs with profiling ON"

    assert rec["schema"] == prof.SCHEMA and rec["impl"] == impl and rec["q_len"] == q_len
    assert rec["gpu_nested"] == 0
    win, leaves = rec["host"]["windows"], rec["gpu"]["leaves"]
    assert win["chunk"]["calls"] == 1 and win["chunk_plan"]["calls"] == 1
    assert win["apply_step"]["calls"] == q_len and win["tail"]["calls"] == q_len
    tails = sum(win.get(k, {}).get("calls", 0) for k in ("tail_partial.open_recent", "tail_partial.recent_only"))
    assert tails == q_len
    assert win.get("tail_prep", {}).get("calls", 0) == win.get("tail_partial.open_recent", {}).get("calls", 0)
    if impl == "reference":
        assert win["reference_attention_call"]["calls"] == q_len
        assert win["reduce"]["calls"] == q_len and win["closed_page"]["calls"] == q_len
    else:
        tiles = _expected_tiles(prefix, q_len)
        assert "reference_attention_call" not in win
        assert win["reduce"]["calls"] == len(tiles)
        assert win["closed_page"]["calls"] == sum(1 for _, _, c in tiles if c)
    # HOST: exclusive = inclusive - children; nothing negative; exclusive + overhead sums to the wall.
    for key, w in win.items():
        assert abs(w["exclusive_ms"] - (w["inclusive_ms"] - w["children_window_ms"])) <= TOL, key
        assert w["exclusive_ms"] >= -TOL, key
    total = sum(w["exclusive_ms"] + w["overhead_ms"] for w in win.values())
    assert abs(total - rec["host"]["wall_ms"]) <= TOL * len(win)
    # GPU: one event pair per call of every GPU leaf; leaves never exceed the span.
    assert set(leaves) == {k for k in win if k in prof.GPU_KEYS}
    for key, g in leaves.items():
        assert g["calls"] == win[key]["calls"] and g["gpu_ms"] >= 0.0, key
    assert sum(g["gpu_ms"] for g in leaves.values()) <= rec["gpu"]["span_ms"] + TOL


@cuda
def test_profile_restores_on_exception(monkeypatch):
    r, _, prof = _mods()
    monkeypatch.setenv(prof.COMPONENT_PROFILE_ENV, "1")
    before = _patchable_snapshot()
    with pytest.raises(RuntimeError, match="boom"):
        with prof.rabit2_stage3c_profile_scope(8, 64):
            assert r.Rabit2CausalChunkPlan.__dict__["apply_step"] is not before[("plan", "apply_step")]
            raise RuntimeError("boom")
    _assert_restored(before)
    # An exception raised inside a wrapped component also restores everything.
    rt, cache, bt, q, kc, vc = _setup(64, 8, 96000)
    with pytest.raises(RuntimeError, match="sequentially"):
        with prof.rabit2_stage3c_profile_scope(8, 64):
            plan = r.Rabit2CausalChunkPlan(rt, kc, vc, cache, bt)
            plan.apply_step(1)  # out of order -> frozen apply_step raises
    _assert_restored(before)
