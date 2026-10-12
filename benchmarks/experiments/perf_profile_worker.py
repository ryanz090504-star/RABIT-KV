"""
RABIT-KV performance-risk PROFILING DIAGNOSTIC -- one leg (runs INSIDE the Modal container; one fresh engine process
per leg). DIAGNOSTIC attribution only: NOT a latency benchmark, NOT paper performance evidence, never a replacement
for the accepted Experiment 5 / Experiment 6 numbers.

Three fixed cases (Llama-3.1-8B only; no quality scoring):
  c1  2048-token prompts, concurrency 8    (accepted Experiment 6 shape; SHORTENED request counts)
  c2  8192-token prompts, concurrency 32   (accepted Experiment 6 shape; SHORTENED request counts)
  c3  32,736-token prompt, single request  (accepted Experiment 5 B32768 shape; 1 warmup + SHORTENED reps)
c1 / c2 reuse exp6_worker.py unchanged in everything that defines the workload: BASE_ENGINE_KWARGS with
max_num_seqs = C, the frozen exp6_workload prompt sets (a PREFIX of the shadow-conditioning set, the 2 warmup prompts,
a PREFIX of the measured set), 32 greedy output tokens (temperature 0, ignore_eos), ONE closed-loop llm.generate call
per phase, VLLM_RABIT2_STAGE3C_IMPL=shared_decode / VLLM_RABIT2_SHARED_DECODE_QUERY_BLOCK=32 for rabit_kv2.
c3 reuses exp5_attempt2_engine_worker.py: BASE_ENGINE_KWARGS unchanged (max_num_seqs 32), prompt = BOS + filler x
(32736 - 1), 32 greedy output tokens, one request per llm.generate.

Profiling modes (--mode):
  off             nothing is installed: the plugin directory is not created and not on sys.path / PYTHONPATH;
  cuda_events     perf_profile_plugin.py is loaded by vLLM's own general-plugin mechanism (a temporary
                  directory holding a copy of the plugin and an entry-point dist-info, put on sys.path and
                  PYTHONPATH before vLLM is imported); CUDA event pairs + perf_counter windows;
  torch_profiler  same loading; ONE torch.profiler trace of the measured phase (cross-check only).
The in-tree opt-in Stage3C profilers (VLLM_RABIT2_STAGE3C_PROFILE / _COMPONENT_PROFILE) stay OFF in every leg.
The phase (conditioning | warmup | measured | flush) is written to a file read by the plugin once per scheduler step;
a 1-token `flush` request after the measured phase makes the plugin log the measured-phase totals.
A 2 Hz nvidia-smi utilization sampler runs during the measured phase of EVERY leg (identical for both dtypes).
Machine lines use the PERF_ prefix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp6_workload as wl  # noqa: E402

# Identical to exp6_worker / exp5_attempt2_engine_worker BASE_ENGINE_KWARGS (verified by AST in the tests).
BASE_ENGINE_KWARGS = {
    "dtype": "bfloat16",
    "block_size": 32,
    "max_model_len": 32768,
    "max_num_batched_tokens": 16384,
    "max_num_seqs": 32,
    "enable_prefix_caching": False,
    "enable_chunked_prefill": True,
    "gpu_memory_utilization": 0.82,
    "enforce_eager": True,
    "trust_remote_code": True,
    "disable_log_stats": False,
    "attention_config": {"backend": "TRITON_ATTN"},
}
ALLOWED_KV_CACHE_DTYPES = ("bfloat16", "rabit_kv2")
OUTPUT_TOKENS = 32
CASES = {"c1": {"kind": "concurrency", "prompt_tokens": 2048, "concurrency": 8},
         "c2": {"kind": "concurrency", "prompt_tokens": 8192, "concurrency": 32},
         "c3": {"kind": "single", "prompt_tokens": 32736, "context_point": 32768}}
MODES = ("off", "cuda_events", "torch_profiler")
STAGE3C_ENV = ("VLLM_RABIT2_STAGE3C_IMPL", "VLLM_RABIT2_SHARED_DECODE_QUERY_BLOCK")
STAGE3C_IMPL, QUERY_BLOCK = "shared_decode", 32
INTREE_PROFILE_ENV = ("VLLM_RABIT2_STAGE3C_PROFILE", "VLLM_RABIT2_STAGE3C_COMPONENT_PROFILE")
MODE_ENV, PHASE_FILE_ENV = "RABIT_PERF_PROFILE", "RABIT_PERF_PHASE_FILE"
PLUGIN_SOURCE = Path(__file__).resolve().parent / "perf_profile_plugin.py"
PLUGIN_DIST = "rabit_perf_profile-0.0.dist-info"
REQUEST_FAILURE_EXIT = 75
UTIL_SAMPLE_MS = 500


def emit(tag: str, payload) -> None:
    print(f"{tag}={json.dumps(payload, sort_keys=True, default=str)}", flush=True)


def gpu_memory_used_mib() -> list[int]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         text=True, capture_output=True, check=True).stdout
    return [int(x.strip()) for x in out.strip().splitlines()]


def build_plugin_dir(target: Path) -> dict:
    """A directory that makes perf_profile_plugin.install a discoverable vllm.general_plugins entry point."""
    target.mkdir(parents=True)
    shutil.copyfile(PLUGIN_SOURCE, target / "perf_profile_plugin.py")
    dist = target / PLUGIN_DIST
    dist.mkdir()
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: rabit-perf-profile\nVersion: 0.0\n", encoding="utf-8")
    (dist / "entry_points.txt").write_text(
        "[vllm.general_plugins]\nrabit_perf_profile = perf_profile_plugin:install\n", encoding="utf-8")
    return {"dir": str(target), "plugin_sha256": hashlib.sha256(PLUGIN_SOURCE.read_bytes()).hexdigest()}


class UtilSampler:
    """nvidia-smi utilization.gpu / memory.used every UTIL_SAMPLE_MS during the measured phase (separate process)."""

    def __init__(self) -> None:
        self.rows: list[tuple[int, int]] = []
        self.proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits",
             "-lms", str(UTIL_SAMPLE_MS)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self) -> None:
        for line in self.proc.stdout:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and all(p.isdigit() for p in parts):
                self.rows.append((int(parts[0]), int(parts[1])))

    def stop(self) -> dict:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.thread.join(timeout=5)
        u = [r[0] for r in self.rows]
        return {"interval_ms": UTIL_SAMPLE_MS, "samples": len(u), "utilization_gpu_pct": u,
                "mean_utilization_gpu_pct": (sum(u) / len(u)) if u else None,
                "max_memory_used_mib": max((r[1] for r in self.rows), default=None)}


def request_row(i: int, o) -> dict:
    m = o.metrics
    ids = list(o.outputs[0].token_ids) if o.outputs else []
    return {"i": i, "prompt_tokens": len(o.prompt_token_ids),
            "prompt_token_ids_sha256": wl.ids_sha256(o.prompt_token_ids),
            "output_tokens": len(ids), "output_token_ids_sha256": wl.ids_sha256(ids),
            "finish_reason": o.outputs[0].finish_reason if o.outputs else None,
            "queued_ts": getattr(m, "queued_ts", None), "scheduled_ts": getattr(m, "scheduled_ts", None),
            "first_token_ts": getattr(m, "first_token_ts", None), "last_token_ts": getattr(m, "last_token_ts", None),
            "first_token_latency": getattr(m, "first_token_latency", None)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=sorted(CASES))
    ap.add_argument("--kv-cache-dtype", required=True, choices=ALLOWED_KV_CACHE_DTYPES)
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--conditioning", type=int, required=True, help="c1/c2: shadow-conditioning prefix length")
    ap.add_argument("--measured", type=int, required=True, help="c1/c2: measured prefix length; c3: measured reps")
    args = ap.parse_args()
    case = CASES[args.case]
    if any(os.environ.get(v, "0") != "0" for v in INTREE_PROFILE_ENV):
        raise SystemExit("the in-tree Stage3C profilers must be OFF")
    if args.kv_cache_dtype == "rabit_kv2":
        os.environ[STAGE3C_ENV[0]] = STAGE3C_IMPL
        os.environ[STAGE3C_ENV[1]] = str(QUERY_BLOCK)
    else:
        for v in STAGE3C_ENV:
            os.environ.pop(v, None)

    phase_file = None
    if args.mode == "off":
        if os.environ.get(MODE_ENV) or os.environ.get(PHASE_FILE_ENV):
            raise SystemExit("profiling environment must be unset for an unprofiled leg")
        plugin = {"dir": None, "plugin_sha256": None}
    else:
        tmp = Path(tempfile.mkdtemp(prefix="rabit_perf_"))
        plugin = build_plugin_dir(tmp / "plugin")
        phase_file = tmp / "phase"
        phase_file.write_text("setup", encoding="utf-8")
        os.environ[MODE_ENV] = args.mode
        os.environ[PHASE_FILE_ENV] = str(phase_file)
        os.environ["PYTHONPATH"] = plugin["dir"] + os.pathsep + os.environ.get("PYTHONPATH", "")
        sys.path.insert(0, plugin["dir"])
    emit("PERF_LEG", {"label": args.label, "case": args.case, **case, "kv_cache_dtype": args.kv_cache_dtype,
                      "mode": args.mode, "conditioning": args.conditioning, "measured": args.measured,
                      "plugin": plugin, "output_tokens": OUTPUT_TOKENS})

    def set_phase(name: str) -> None:
        if phase_file is not None:
            t = phase_file.with_suffix(".tmp")
            t.write_text(name, encoding="utf-8")
            os.replace(t, phase_file)
        print(f"PERF_PHASE {name}", flush=True)

    from vllm import LLM, SamplingParams
    from vllm.utils.torch_utils import get_kv_cache_torch_dtype
    from vllm.v1.kv_cache_interface import get_kv_quant_mode

    emit("PERF_PLUGIN_MODULE_LOADED", {"perf_profile_plugin_in_sys_modules": "perf_profile_plugin" in sys.modules})
    if args.kv_cache_dtype == "rabit_kv2":
        import vllm.v1.attention.ops.rabit_kv2_stage3c_shared_decode as sd

        s3 = {"applicable": True, "effective_impl": sd.rabit2_stage3c_impl(),
              "effective_query_block": sd.rabit2_shared_decode_query_block(),
              "env": {v: os.environ.get(v) for v in STAGE3C_ENV},
              "intree_profiling_env": {v: os.environ.get(v) for v in INTREE_PROFILE_ENV}}
        emit("PERF_STAGE_IMPL", s3)
        if (s3["effective_impl"], s3["effective_query_block"]) != (STAGE3C_IMPL, QUERY_BLOCK):
            raise RuntimeError(f"Stage3C selector does not report the requested configuration: {s3}")
    else:
        emit("PERF_STAGE_IMPL", {"applicable": False, "env": {v: os.environ.get(v) for v in STAGE3C_ENV}})

    kwargs = {"model": args.model_dir, **BASE_ENGINE_KWARGS, "kv_cache_dtype": args.kv_cache_dtype}
    if case["kind"] == "concurrency":
        kwargs["max_num_seqs"] = case["concurrency"]
    emit("PERF_REQUESTED_ENGINE_KWARGS", kwargs)
    llm = LLM(**kwargs)
    cfg = llm.llm_engine.vllm_config
    mc, cc, sc = cfg.model_config, cfg.cache_config, cfg.scheduler_config
    emit("PERF_EFFECTIVE_ENGINE_CONFIG", {
        "model_dtype": str(mc.dtype), "max_model_len": mc.max_model_len, "enforce_eager": mc.enforce_eager,
        "block_size": cc.block_size, "gpu_memory_utilization": cc.gpu_memory_utilization,
        "enable_prefix_caching": cc.enable_prefix_caching, "max_num_batched_tokens": sc.max_num_batched_tokens,
        "max_num_seqs": sc.max_num_seqs, "enable_chunked_prefill": sc.enable_chunked_prefill,
        "attention_backend": str(cfg.attention_config.backend),
        "engine_cache_dtype": cc.cache_dtype, "kv_quant_mode": get_kv_quant_mode(cc.cache_dtype).name,
        "resolved_kv_torch_dtype": str(get_kv_cache_torch_dtype(cc.cache_dtype, mc.dtype)),
        "num_gpu_blocks": cc.num_gpu_blocks, "capacity_tokens": cc.num_gpu_blocks * cc.block_size})
    emit("PERF_GPU_MEMORY", {"phase": "after_engine_init", "memory_used_mib": gpu_memory_used_mib()})
    sp = SamplingParams(temperature=0.0, max_tokens=OUTPUT_TOKENS, ignore_eos=True)

    if case["kind"] == "concurrency":
        length = case["prompt_tokens"]
        if not (1 <= args.conditioning <= wl.SHADOW_CONDITIONING_REQUESTS and 1 <= args.measured <= wl.MEASURED_REQUESTS):
            raise SystemExit("prefix lengths out of range")
        shadow = wl.shadow_conditioning_prompts(length)[:args.conditioning]
        warm = wl.warmup_prompts(length)
        measured = [wl.measured_prompts(length)[:args.measured]]  # ONE closed-loop llm.generate call
        flush_ids = warm[0][:4]
    else:
        tok = llm.get_tokenizer()
        bos = tok.bos_token_id
        filler = tok.encode(" the", add_special_tokens=False)[-1]
        prompt = [bos] + [filler] * (case["prompt_tokens"] - 1)
        shadow, warm = [], [prompt]
        measured = [[prompt] for _ in range(args.measured)]  # one request per llm.generate (Experiment 5)
        flush_ids = prompt[:4]
    emit("PERF_WORKLOAD", {"prompt_tokens": case["prompt_tokens"], "output_tokens": OUTPUT_TOKENS,
                           "temperature": 0.0, "ignore_eos": True, "conditioning_requests": len(shadow),
                           "warmup_requests": len(warm), "measured_calls": len(measured),
                           "measured_requests": sum(len(c) for c in measured),
                           "measured_prompt_set": wl.set_digest([p for c in measured for p in c])["ordered_set_sha256"],
                           "measured_first_prompt_sha256": wl.ids_sha256(measured[0][0])})

    phase = "conditioning"
    try:
        if shadow:
            set_phase("conditioning")
            llm.generate([{"prompt_token_ids": p} for p in shadow], sp, use_tqdm=False)
        phase = "warmup"
        set_phase("warmup")
        t0 = time.perf_counter()
        wouts = llm.generate([{"prompt_token_ids": p} for p in warm], sp, use_tqdm=False)
        emit("PERF_WARMUP", {"wall_s": time.perf_counter() - t0, "requests": [request_row(i, o) for i, o in enumerate(wouts)]})
        phase = "measured"
        set_phase("measured")
        sampler = UtilSampler()
        calls = []
        t_all = time.perf_counter()
        for batch in measured:
            t0 = time.perf_counter()
            outs = llm.generate([{"prompt_token_ids": p} for p in batch], sp, use_tqdm=False)
            calls.append((time.perf_counter() - t0, outs))
        wall_all = time.perf_counter() - t_all
        util = sampler.stop()
        phase = "flush"
        set_phase("flush")
        llm.generate([{"prompt_token_ids": flush_ids}], SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True),
                     use_tqdm=False)
    except Exception as exc:  # noqa: BLE001  (reported, never retried)
        emit("PERF_REQUEST_FAILURE", {"phase": phase, "error": f"{type(exc).__name__}: {exc}"[:2000]})
        return REQUEST_FAILURE_EXIT
    for c, (wall_s, outs) in enumerate(calls):
        for i, o in enumerate(outs):
            emit("PERF_REQUEST", {"call": c, **request_row(i, o)})
        emit("PERF_MEASURED_CALL", {"call": c, "wall_s": wall_s, "requests": len(outs)})
    emit("PERF_MEASURED_SUMMARY", {"wall_s": wall_all, "calls": len(calls),
                                   "requests": sum(len(o) for _, o in calls)})
    emit("PERF_GPU_UTILIZATION", util)
    emit("PERF_GPU_MEMORY", {"phase": "after_measured", "memory_used_mib": gpu_memory_used_mib()})
    print("PERF_WORKER_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
