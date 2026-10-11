"""
MLSys 2027 Experiment 3 -- RABIT-KV physical correctness gate (runs INSIDE the
Modal container, in its own fresh process, BEFORE any performance leg).

`regression()` below is copied VERBATIM (decorator removed) from the canonical
embedded runner in benchmarks/performance/benchmark_deployment.py (RUNNER_Z).
run_experiment3_deployment.py verifies AST equality with the canonical source
before every run, so no new correctness criteria are introduced. It runs:
  * dispatch/marker preflight;
  * Stage4D3.4 prep exactness (62 cases) and full-attention exactness;
  * final fast decode-append state/byte exactness (202 steps);
  * pytest on tests/quantization/test_kvquant_k3.py and test_rabit_kv2*.py
    (excluding test_rabit_kv2_stage4b1.py), raising if pytest fails.

The wrapper prints the exact pytest command (built with the same rule as
regression()), the frozen-source SHA, and a machine-readable result line.
This file never modifies vllm-kvquant; it only imports it.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

RABIT_KV2 = Path("/root/vllm-kvquant/vllm/v1/attention/ops/rabit_kv2.py")


def emit(tag: str, payload) -> None:
    print(f"{tag}={json.dumps(payload, sort_keys=True, default=str)}", flush=True)


# ---- BEGIN VERBATIM COPY OF CANONICAL regression() --------------------------
def regression():
    import torch
    import vllm.v1.attention.ops.rabit_kv2 as r
    from vllm.v1.attention.ops.kvquant_k3 import unpack_int3_values

    print("=" * 116)
    print("RABIT-2 FINAL targeted decode-append regression")
    print("=" * 116)

    for attr in (
        "_RABIT2_STAGE4D2_VECTORIZED_WRITER",
        "_RABIT2_STAGE4D2_2_WRITER_ONLY_LOCKED",
        "_RABIT2_STAGE4D3_4_TRITON_TAILPREP",
    ):
        if not getattr(r, attr, False):
            raise RuntimeError(f"required marker missing: {attr}")
    if r._rabit2_stage4b1_exactmeta_emit_tail_partial is not r._rabit2_stage4d3_4_emit_tail_partial:
        raise RuntimeError("Stage4D3.4 tail-prep dispatch is not active")
    if not getattr(r, "_RABIT2_FINAL_FAST_DECODE_APPEND", False):
        raise RuntimeError("final fast decode-append marker missing")
    if not hasattr(r.tl, "div_rn"):
        raise RuntimeError("tl.div_rn unavailable")
    print("Dispatch preflight: PASSED")

    H, QH, D = 8, 32, 128
    device = torch.device("cuda")
    dtype = torch.bfloat16

    def same_meta(ref, got, groups):
        return (
            torch.equal(ref["codes"], got["codes"][:groups])
            and torch.equal(ref["min"], got["min"][:groups])
            and torch.equal(ref["scale"], got["scale"][:groups])
        )

    # Strict open-tail prep exactness for every legal open length, two seeds.
    cases = 0
    for t in range(1, 32):
        for seed in (71000 + t, 72000 + t):
            torch.manual_seed(seed)
            rt = r.Rabit2SingleSequenceRuntime(H, D, D)
            cache = torch.zeros(
                (8, 1, 1, 1, rt.layout.page_bytes),
                dtype=torch.uint8,
                device=device,
            )
            bt = torch.arange(8, dtype=torch.int32, device=device)
            k = torch.randn((t + 4, H, D), dtype=dtype, device=device)
            v = torch.randn_like(k)
            r.rabit2_bulk_append_exact(rt, k, v, cache, bt)

            if rt.open_k is None or int(rt.open_k.shape[0]) != t:
                raise RuntimeError(f"bad synthetic open length for t={t}")

            q = torch.randn((QH, D), dtype=dtype, device=device)
            ws, vg1, vg2 = r._rabit2_stage4d3_4_fast_prep(rt, q)
            torch.cuda.synchronize()

            refk = r.quantize_k3_sequence_affine_ref(rt.open_k)
            ref_codes = unpack_int3_values(refk["packed"], D)
            if not torch.equal(ref_codes, ws["k_codes"]):
                raise RuntimeError(f"K codes mismatch at t={t} seed={seed}")
            if not torch.equal(refk["packed"], ws["k_packed"]):
                diff = int((refk["packed"] != ws["k_packed"]).sum().item())
                raise RuntimeError(
                    f"K packed-byte mismatch at t={t} seed={seed} count={diff}"
                )

            kg = (H * D + 63) // 64
            if not same_meta(refk["min"], ws["kmin_meta"], kg):
                raise RuntimeError(f"K-min metadata mismatch at t={t} seed={seed}")
            if not same_meta(refk["scale"], ws["kscale_meta"], kg):
                raise RuntimeError(f"K-scale metadata mismatch at t={t} seed={seed}")

            ref_vm = r.encode_metadata_uint8_group_ref(rt.open_v_min)
            ref_vs = r.encode_metadata_uint8_group_ref(rt.open_v_scale)
            if not same_meta(ref_vm, ws["vmin_meta"], vg1):
                raise RuntimeError(f"V-min metadata mismatch at t={t} seed={seed}")
            if not same_meta(ref_vs, ws["vscale_meta"], vg2):
                raise RuntimeError(f"V-scale metadata mismatch at t={t} seed={seed}")
            cases += 1

    print(f"Stage4D3.4 prep exactness: PASSED ({cases}/62)")

    # Full attention exactness against the preserved Stage4B1 helper.
    ctx = 2048
    torch.manual_seed(73000)
    rt = r.Rabit2SingleSequenceRuntime(H, D, D)
    pages = 96
    cache = torch.zeros(
        (pages, 1, 1, 1, rt.layout.page_bytes),
        dtype=torch.uint8,
        device=device,
    )
    bt = torch.arange(pages, dtype=torch.int32, device=device)
    k = torch.randn((ctx, H, D), dtype=dtype, device=device)
    v = torch.randn_like(k)
    r.rabit2_bulk_append_exact(rt, k, v, cache, bt)
    query = torch.randn((1, QH, D), dtype=dtype, device=device)

    new_emit = r._rabit2_stage4d3_4_emit_tail_partial
    old_emit = r._rabit2_stage4d3_4_old_emit_tail_partial

    r._rabit2_stage4b1_exactmeta_emit_tail_partial = old_emit
    old_out = r.rabit2_online_decode_attention_triton(
        query, cache, bt, rt, softmax_scale=D ** -0.5
    ).clone()
    torch.cuda.synchronize()

    r._rabit2_stage4b1_exactmeta_emit_tail_partial = new_emit
    new_out = r.rabit2_online_decode_attention_triton(
        query, cache, bt, rt, softmax_scale=D ** -0.5
    ).clone()
    torch.cuda.synchronize()

    if not torch.equal(old_out, new_out):
        diff = float((old_out.float() - new_out.float()).abs().max().item())
        raise RuntimeError(f"full attention exactness failed max_abs={diff}")
    print("Stage4D3.4 full attention exactness: PASSED (torch.equal)")

    # Report scratch footprint. Shared per CUDA stream, not per request/layer.
    ws = r._rabit2_stage4d3_4_workspace(rt, query[0])
    scratch = 0
    for key, value in ws.items():
        if isinstance(value, torch.Tensor):
            scratch += value.numel() * value.element_size()
        elif isinstance(value, dict):
            for x in value.values():
                if isinstance(x, torch.Tensor):
                    scratch += x.numel() * x.element_size()
    print(f"Stage4D3.4 scratch_bytes_per_stream={scratch}")


    # ------------------------------------------------------------------
    # Final targeted fix gate: one-token append must be state/byte exact.
    # ------------------------------------------------------------------
    def _same_optional(name, a, b):
        if a is None or b is None:
            if a is not None or b is not None:
                raise RuntimeError(f"{name}: None mismatch")
            return
        if not torch.equal(a, b):
            max_abs = (
                float((a.float() - b.float()).abs().max().item())
                if a.numel() else 0.0
            )
            raise RuntimeError(
                f"{name}: tensor mismatch shape={tuple(a.shape)} "
                f"max_abs={max_abs}"
            )

    exact_steps = 0
    for seed in (88011, 88029):
        torch.manual_seed(seed)
        ref_rt = r.Rabit2SingleSequenceRuntime(H, D, D)
        new_rt = r.Rabit2SingleSequenceRuntime(H, D, D)
        cache_ref = torch.zeros(
            (12, 1, 1, 1, ref_rt.layout.page_bytes),
            dtype=torch.uint8, device=device
        )
        cache_new = torch.zeros_like(cache_ref)
        bt2 = torch.arange(12, dtype=torch.int32, device=device)

        for step in range(101):
            kk = torch.randn((1, H, D), dtype=dtype, device=device)
            vv = torch.randn_like(kk)

            r._rabit2_final_old_append(ref_rt, kk, vv, cache_ref, bt2)
            new_rt.append(kk, vv, cache_new, bt2)
            torch.cuda.synchronize()

            if ref_rt.closed_pages != new_rt.closed_pages:
                raise RuntimeError(
                    f"closed_pages mismatch seed={seed} step={step}"
                )
            if ref_rt.total_tokens != new_rt.total_tokens:
                raise RuntimeError(
                    f"total_tokens mismatch seed={seed} step={step}"
                )
            _same_optional("open_k", ref_rt.open_k, new_rt.open_k)
            _same_optional(
                "open_v_packed", ref_rt.open_v_packed, new_rt.open_v_packed
            )
            _same_optional("open_v_min", ref_rt.open_v_min, new_rt.open_v_min)
            _same_optional(
                "open_v_scale", ref_rt.open_v_scale, new_rt.open_v_scale
            )
            _same_optional("recent_k", ref_rt.recent_k, new_rt.recent_k)
            _same_optional("recent_v", ref_rt.recent_v, new_rt.recent_v)

            if not torch.equal(cache_ref, cache_new):
                diff = int((cache_ref != cache_new).sum().item())
                raise RuntimeError(
                    f"physical cache byte mismatch seed={seed} "
                    f"step={step} count={diff}"
                )

            if step in (4, 7, 31, 35, 63, 67, 95, 100):
                qq = torch.randn((1, QH, D), dtype=dtype, device=device)
                ref_o = r.rabit2_online_decode_attention_triton(
                    qq, cache_ref, bt2, ref_rt, softmax_scale=D ** -0.5
                ).clone()
                new_o = r.rabit2_online_decode_attention_triton(
                    qq, cache_new, bt2, new_rt, softmax_scale=D ** -0.5
                ).clone()
                torch.cuda.synchronize()
                if not torch.equal(ref_o, new_o):
                    diff = float(
                        (ref_o.float() - new_o.float()).abs().max().item()
                    )
                    raise RuntimeError(
                        f"append attention mismatch seed={seed} "
                        f"step={step} max_abs={diff}"
                    )
            exact_steps += 1

    print(
        f"FINAL fast decode append exactness: PASSED "
        f"({exact_steps} state/byte steps + attention checkpoints)"
    )

    test_dir = Path("/root/vllm-kvquant/tests/quantization")
    tests = [test_dir / "test_kvquant_k3.py"]
    tests.extend(
        p for p in sorted(test_dir.glob("test_rabit_kv2*.py"))
        if p.name != "test_rabit_kv2_stage4b1.py"
    )
    p = subprocess.run(
        [
            sys.executable, "-m", "pytest", "-q", "--tb=short",
            f"--confcutdir={test_dir}",
            *[str(x) for x in tests],
        ],
        text=True, capture_output=True,
    )
    print(p.stdout)
    if p.stderr:
        print(p.stderr)
    print(f"pytest exit={p.returncode}")
    if p.returncode != 0:
        raise RuntimeError("Stage4D3.4 regression tests failed")

    print("RABIT-2 FINAL TARGETED REGRESSION PASSED")

# ---- END VERBATIM COPY OF CANONICAL regression() ----------------------------


def pytest_command() -> list[str]:
    """Same test selection rule as regression() above."""
    test_dir = Path("/root/vllm-kvquant/tests/quantization")
    tests = [test_dir / "test_kvquant_k3.py"]
    tests.extend(
        p for p in sorted(test_dir.glob("test_rabit_kv2*.py"))
        if p.name != "test_rabit_kv2_stage4b1.py"
    )
    return [
        sys.executable, "-m", "pytest", "-q", "--tb=short",
        f"--confcutdir={test_dir}",
        *[str(x) for x in tests],
    ]


def main() -> int:
    sha = hashlib.sha256(RABIT_KV2.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    emit("EXP3_GATE_BEGIN", {"pytest_command": pytest_command(), "rabit_kv2_sha256_lf": sha})
    try:
        regression()
    except Exception as exc:  # noqa: BLE001
        emit("EXP3_GATE_RESULT", {"passed": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1
    emit("EXP3_GATE_RESULT", {"passed": True})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
