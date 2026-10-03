"""
Offline tests of the RABIT-KV performance-risk profiling diagnostic harness (no GPU, no torch, no vLLM import).
Run: python benchmarks/mlsys2027/test_perf_profile.py
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import re
import sys
import tempfile
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import perf_profile_analysis as pa  # noqa: E402
import perf_profile_plugin as pp  # noqa: E402

VK = ROOT / "vllm-kvquant" / "vllm" / "v1"
SRC = {"ta": VK / "attention" / "backends" / "triton_attn.py", "r": VK / "attention" / "ops" / "rabit_kv2.py",
       "sd": VK / "attention" / "ops" / "rabit_kv2_stage3c_shared_decode.py",
       "t32": VK / "attention" / "ops" / "rabit_kv2_stage3c_tile32.py", "gw": VK / "worker" / "gpu_worker.py"}


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def module_assign(path: Path, name: str):
    for node in ast.parse(read(path)).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node.value
    raise AssertionError(f"{name} not assigned in {path.name}")


class FakeEvent:
    clock = 0.0
    created = 0

    def __init__(self, enable_timing=False):
        FakeEvent.created += 1
        self.t = None

    def record(self):
        FakeEvent.clock += 1.0
        self.t = FakeEvent.clock

    def elapsed_time(self, other):
        return other.t - self.t


class FakeCuda:
    Event = FakeEvent
    syncs = 0

    @staticmethod
    def is_initialized():
        return True

    @classmethod
    def synchronize(cls):
        cls.syncs += 1


class FakeTorch:
    cuda = FakeCuda


class Sched:
    def __init__(self, n):
        self.num_scheduled_tokens = n


def test_host_window_tree():
    p = pp.Profiler()

    def kernel():
        time.sleep(0.002)

    def branch():
        p.window("kernel._rabit2_x_kernel", kernel, (), {})
        p.window("leaf.tail_emit", lambda: p.window("kernel._rabit2_y_kernel", kernel, (), {}), (), {})
        time.sleep(0.002)
        return True

    def step(_self, sched):
        return p.window("decode.attention", branch, (), {})

    assert p.window("step.execute_model", step, (None, Sched({"a": 1, "b": 1})), {}, pp._meta_step) is True
    rep = p.report()
    h = rep["host"]
    assert rep["steps"] == 1 and rep["gpu_domain"] is False and rep["gpu"] == {}
    assert set(h) == {"step.execute_model", "stepkind.decode_only", "decode.attention",
                      "decode.attention/kernel._rabit2_x_kernel", "decode.attention/kernel._rabit2_y_kernel",
                      "decode.attention/leaf.tail_emit"}, sorted(h)
    assert h["step.execute_model"]["tokens"] == 2 and h["step.execute_model"]["seqs"] == 2
    assert h["decode.attention"]["returned_true"] == 1
    assert h["step.execute_model"]["inclusive_ms"] >= h["decode.attention"]["inclusive_ms"] >= 6.0
    b = h["decode.attention"]
    assert 1.5 <= b["exclusive_ms"] < b["inclusive_ms"] - 3.5, b  # the two kernel sleeps are children
    assert h["decode.attention/leaf.tail_emit"]["exclusive_ms"] < 1.0
    assert p.depth == 0 and p.stack == [] and p.child_windows == [0.0]


def test_step_kinds():
    for n, kind in (({"a": 1}, "decode_only"), ({"a": 2048, "b": 9}, "prefill_only"), ({"a": 1, "b": 77}, "mixed"),
                    ({}, "empty")):
        sums, alias = pp._meta_step((None, Sched(n)), {})
        assert alias == f"stepkind.{kind}" and sums == {"tokens": sum(n.values()), "seqs": len(n)}


def test_gpu_domain_events_resolved_once_and_pooled():
    FakeEvent.clock, FakeEvent.created, FakeCuda.syncs = 0.0, 0, 0
    p = pp.Profiler(FakeTorch)

    def step(_self, sched):
        for _ in range(3):
            p.window("kernel._rabit2_x_kernel", lambda: None, (), {})

    for _ in range(2):
        p.window("step.execute_model", step, (None, Sched({"a": 1})), {}, pp._meta_step)
    rep = p.report()
    assert rep["gpu_domain"] is True and rep["syncs"] == 2 and FakeCuda.syncs == 2
    assert rep["gpu"]["no_branch/kernel._rabit2_x_kernel"] == {"calls": 6, "gpu_ms": 6.0}
    assert rep["gpu"]["step.execute_model"]["calls"] == 2 and rep["gpu"]["stepkind.decode_only"]["calls"] == 2
    assert rep["gpu"]["step.execute_model"]["gpu_ms"] == rep["gpu"]["stepkind.decode_only"]["gpu_ms"] == 14.0
    assert FakeEvent.created == 8 and len(p.pool) == 8 and p.pending == []  # second step reused the pool


def test_wrapper_is_transparent():
    p = pp.Profiler()
    seen = {}

    def fn(a, b, *, c):
        seen["args"] = (a, b, c)
        return ("out", a)

    w = p.wrap("rabit.forward", fn)
    x = object()
    assert w(x, 2, c=3) == ("out", x) and seen["args"] == (x, 2, 3) and w.__name__ == "fn"

    def boom():
        raise ValueError("original error")

    try:
        p.wrap("decode.attention", boom)()
    except ValueError as exc:
        assert str(exc) == "original error"
    else:
        raise AssertionError("exception swallowed")
    assert p.depth == 0 and p.stack == []
    bad = p.wrap("attention.forward", lambda: 5, lambda a, k: 1 / 0)  # metadata errors never change the call
    assert bad() == 5 and p.acc["meta_errors"] == 1


def test_kernel_proxy_forwards():
    p = pp.Profiler()

    class K:
        attr = "kept"

        def __getitem__(self, grid):
            return lambda *a, **k: ("launched", grid, a, k)

    proxy = pp.KernelProxy(p, "kernel._rabit2_k_kernel", K())
    assert proxy[(4, 2)](1, x=2) == ("launched", (4, 2), (1,), {"x": 2}) and proxy.attr == "kept"
    assert p.report()["host"]["no_branch/kernel._rabit2_k_kernel"]["calls"] == 1


def test_phase_file_switch_dumps_previous_phase():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "phase"
        f.write_text("warmup", encoding="utf-8")
        p = pp.Profiler(None, str(f))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            p.window("step.execute_model", lambda *_: None, (None, Sched({"a": 1})), {}, pp._meta_step)
            time.sleep(0.02)
            f.write_text("measured", encoding="utf-8")
            p.window("step.execute_model", lambda *_: None, (None, Sched({"a": 5})), {}, pp._meta_step)
        lines = [json.loads(x.split("=", 1)[1]) for x in buf.getvalue().splitlines() if x.startswith(pp.LOG_TAG + "=")]
        assert [x["phase"] for x in lines] == ["setup", "warmup"], [x["phase"] for x in lines]
        assert lines[0]["steps"] == 0 and lines[1]["steps"] == 1 and p.phase == "measured" and p.acc["steps"] == 1


def test_install_is_noop_without_env():
    import os

    assert pp.MODE_ENV not in os.environ
    pp.install()
    assert pp._state is None


def test_plugin_dir_is_a_discoverable_entry_point():
    import importlib.metadata as md

    import perf_profile_worker as w

    with tempfile.TemporaryDirectory() as d:
        info = w.build_plugin_dir(Path(d) / "plugin")
        sys.path.insert(0, info["dir"])
        try:
            eps = [e for e in md.entry_points(group="vllm.general_plugins") if e.name == "rabit_perf_profile"]
            assert len(eps) == 1 and eps[0].value == "perf_profile_plugin:install", eps
        finally:
            sys.path.remove(info["dir"])
        assert read(Path(info["dir"]) / "perf_profile_plugin.py") == read(HERE / "perf_profile_plugin.py")


def test_every_patched_name_exists_in_the_serving_source():
    ta, r, sd, gw = read(SRC["ta"]), read(SRC["r"]), read(SRC["sd"]), read(SRC["gw"])
    for name in ("forward", "do_kv_cache_update", "_forward_rabit_kv2"):
        assert re.search(rf"^    def {name}\(", ta, re.M), name
    assert re.search(r"^class TritonAttentionImpl\(", ta, re.M)
    imported = set(re.findall(r"\b(rabit2_\w+|context_attention_fwd|Rabit2\w+)\b", ta.split("logger = ")[0]))
    for name in ("rabit2_bulk_append_exact", "context_attention_fwd", "rabit2_stage3c_forward_shared_decode",
                 "rabit2_online_decode_attention_triton"):
        assert name in imported, name
        assert re.search(rf"^\s+{name}\($", ta, re.M) or re.search(rf"\b{name}\(", ta), name
    assert "context_attention_fwd(\n                    q=q_seq," in ta  # dense attention is called with q= keyword
    for name in ("execute_model", "sample_tokens"):
        assert re.search(rf"^    def {name}\(", gw, re.M), name
    assert re.search(r"^class Worker\(", gw, re.M)
    assert re.search(r"^class Rabit2CausalChunkPlan\b", r, re.M) and re.search(r"^class Rabit2SingleSequenceRuntime\b", r, re.M)
    assert re.search(r"^    def apply_step\(", r, re.M)
    assert re.search(r"^(    def append\(|Rabit2SingleSequenceRuntime\.append = )", r, re.M)
    assert re.search(r"^_rabit2_stage4b1_exactmeta_emit_tail_partial = ", r, re.M)
    assert re.search(r"^def _rabit2_stage4d3_4_fast_prep\(", r, re.M)
    assert re.search(r"^def rabit2_shared_decode_closed_pages\(", sd, re.M)
    assert "_r._rabit2_stage4b1_exactmeta_emit_tail_partial" in sd and "_r.Rabit2CausalChunkPlan(" in sd
    launched = set()
    for key in ("r", "sd", "t32"):
        launched |= set(pp.KERNEL_LAUNCH_RE.findall(read(SRC[key])))
    assert {"_rabit2_shared_decode_closed_page_partial_kernel", "_rabit2_tile32_reduce_partials_kernel",
            "_rabit2_final_v2_age_kernel", "_rabit2_stage4b3_gqa4_closed_page_partial_kernel"} <= launched
    for name in launched:  # a launched kernel is never called as a helper from inside another Triton kernel
        for key in ("r", "sd", "t32"):
            assert not re.search(rf"(?<!def ){name}\(", read(SRC[key])), name


def test_vllm_tree_is_not_touched_by_the_profiler():
    for path in (ROOT / "vllm-kvquant" / "vllm").rglob("*.py"):
        s = path.read_text(encoding="utf-8", errors="replace")
        assert "perf_profile" not in s and "RABIT_PERF_" not in s, path


def test_engine_kwargs_and_image_identical_to_accepted_harness():
    ref = ast.dump(module_assign(HERE / "exp6_worker.py", "BASE_ENGINE_KWARGS"))
    assert ast.dump(module_assign(HERE / "perf_profile_worker.py", "BASE_ENGINE_KWARGS")) == ref
    assert ast.dump(module_assign(HERE / "exp5_attempt2_engine_worker.py", "BASE_ENGINE_KWARGS")) == ref
    assert ast.dump(module_assign(HERE / "perf_profile_modal.py", "image")) == \
        ast.dump(module_assign(HERE / "exp6_modal.py", "image"))
    for name in ("MODEL", "BASE_COMMIT"):
        assert ast.dump(module_assign(HERE / "perf_profile_modal.py", name)) == \
            ast.dump(module_assign(HERE / "exp6_modal.py", name))
    w = read(HERE / "perf_profile_worker.py")
    assert "SamplingParams(temperature=0.0, max_tokens=OUTPUT_TOKENS, ignore_eos=True)" in w
    assert 'prompt = [bos] + [filler] * (case["prompt_tokens"] - 1)' in w
    assert 'tok.encode(" the", add_special_tokens=False)[-1]' in w
    assert 'gpu="H100!:1"' in read(HERE / "perf_profile_modal.py")


def test_fixed_cases_and_legs():
    import perf_profile_worker as w
    import run_perf_profile_diagnostic as run

    assert w.CASES["c1"] == {"kind": "concurrency", "prompt_tokens": 2048, "concurrency": 8}
    assert w.CASES["c2"] == {"kind": "concurrency", "prompt_tokens": 8192, "concurrency": 32}
    assert w.CASES["c3"] == {"kind": "single", "prompt_tokens": 32736, "context_point": 32768}
    assert (w.STAGE3C_IMPL, w.QUERY_BLOCK, w.OUTPUT_TOKENS) == ("shared_decode", 32, 32)
    labels = [leg[0] for leg in run.LEGS]
    assert len(labels) == len(set(labels)) == 13
    for case in ("c1", "c2", "c3"):
        modes = {(d, m) for _, c, d, m, *_ in run.LEGS if c == case}
        assert {("rabit_kv2", "off"), ("rabit_kv2", "cuda_events"), ("bfloat16", "off")} <= modes
        by = {(d, m): (cond, meas) for _, c, d, m, cond, meas, _ in run.LEGS if c == case}
        assert by[("rabit_kv2", "off")] == by[("rabit_kv2", "cuda_events")] == by[("bfloat16", "off")]  # matched
    assert sum(1 for leg in run.LEGS if leg[3] == "torch_profiler") == 1  # exactly ONE torch.profiler run
    assert sum(leg[6] for leg in run.LEGS) + 600 < 21600
    assert re.fullmatch(r"([a-z0-9_]+=c[123]:(rabit_kv2|bfloat16):(off|cuda_events|torch_profiler):\d+:\d+:\d+,?)+",
                        run.legs_arg())
    assert run.IDENTITY == {n: run.sha256_lf(ROOT / p) for n, p in run.IDENTITY_PATHS.items()}


def test_breakdown_shares():
    def n(calls, inc, child=0.0, **extra):
        return {"calls": calls, "inclusive_ms": inc, "overhead_ms": 0.1, "children_window_ms": child,
                "exclusive_ms": inc - child, **extra}

    host = {"step.execute_model": n(10, 1000.0, 900.0, tokens=80, seqs=80), "step.sample_tokens": n(10, 50.0),
            "stepkind.decode_only": n(10, 1000.0, 900.0, tokens=80, seqs=80),
            "attention.forward": n(320, 800.0, 780.0), "attention.bf16_cache_update": n(320, 10.0),
            "rabit.forward": n(320, 780.0, 600.0), "decode.append_aging": n(2560, 200.0, 120.0),
            "decode.attention": n(2560, 400.0, 250.0),
            "decode.attention/kernel._rabit2_a_kernel": n(2560, 250.0),
            "decode.append_aging/kernel._rabit2_b_kernel": n(5120, 120.0),
            "decode.attention/leaf.tail_prep": n(2560, 40.0)}
    gpu = {k: {"calls": v["calls"], "gpu_ms": v["inclusive_ms"] * 0.5} for k, v in host.items()}
    b = pa.breakdown({"host": host, "gpu": gpu, "steps": 10, "syncs": 10, "meta_errors": 0})
    assert b["step_windows_total"] == {"host_ms": 1050.0, "gpu_span_ms": 525.0}
    comp = {c["component"].split(" ")[0].rstrip(":"): c for c in b["rabit_components"]}
    assert abs(comp["decode.attention"]["host_share"] - 400.0 / 1050.0) < 1e-12
    assert abs(comp["HOST/CONTROL"]["host_ms"] - 180.0) < 1e-9
    other = [x for x in b["level1"] if x["component"].startswith("OTHER MODEL TIME")][0]
    assert abs(other["host_ms"] - 190.0) < 1e-9 and abs(other["gpu_span_ms"] - 95.0) < 1e-9
    d = b["branch_detail"]["decode.attention"]
    assert d["kernel_launches"] == 2560 and abs(d["outside_kernel_windows"]["host_ms"] - 150.0) < 1e-9
    assert b["kernel_launch_totals"]["launches"] == 7680 and b["kernel_launch_totals"]["launches_per_step"] == 768.0
    assert b["step_kinds"]["decode_only"]["mean_sequences_per_step"] == 8.0
    assert "stepkind.decode_only" not in [x["window"] for x in b["host_exclusive_top"]]


def test_log_parsing():
    text = "\n".join([
        'PERFRUN_SOURCE_IDENTITY={"match": true}',
        '[leg:c1_rabit_events] PERF_LEG={"label": "c1_rabit_events"}',
        '[leg:c1_rabit_events] (EngineCore pid=7) RABIT_PERF_INSTALL={"ok":true}',
        '[leg:c1_rabit_events] PERF_REQUEST={"i": 0, "prompt_tokens": 2048, "output_tokens": 32, "finish_reason": '
        '"length", "first_token_ts": 2.0, "last_token_ts": 5.1, "scheduled_ts": 1.0, "first_token_latency": 1.5, '
        '"output_token_ids_sha256": "aa"}',
        '[leg:c1_rabit_events] PERF_MEASURED_SUMMARY={"wall_s": 4.0, "calls": 1, "requests": 1}',
        "[leg:c1_rabit_events] PERF_WORKER_COMPLETE",
    ])
    legs, top = pa.split_legs(text)
    assert list(legs) == ["c1_rabit_events"] and pa.tagged(top, "PERFRUN_SOURCE_IDENTITY") == [{"match": True}]
    r = pa.leg_report(legs["c1_rabit_events"])
    assert r["complete"] and r["installs"] == [{"ok": True}] and r["requests_per_s"] == 0.25
    assert r["request_stats"]["mean_ttft_ms"] == 1500.0 and abs(r["request_stats"]["mean_tpot_ms"] - 100.0) < 1e-9
    assert pa.tagged(["XPERF_LEG={}"], "PERF_LEG") == []
    o = pa.overhead({"measured_summary": {"wall_s": 10.0, "requests": 4}, "output_token_ids_sha256": ["a", "b"]},
                    {"measured_summary": {"wall_s": 12.5, "requests": 4}, "output_token_ids_sha256": ["a", "c"]})
    assert abs(o["overhead_pct"] - 25.0) < 1e-9 and o["output_hash_agreement"] == {"compared": 2, "equal": 1}


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
