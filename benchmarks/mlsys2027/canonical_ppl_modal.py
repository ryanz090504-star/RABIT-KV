"""
canonical-quality-v2 continuation PPL -- Modal app (strict ONE H100 80GB; fail-closed hardware guard before anything
else). Launched only by benchmarks/mlsys2027/run_canonical_ppl.py.

One invocation = one model (llama3_1_8b | qwen2_5_7b), N = 32 WikiText-2 windows, context 1024, continuation 128,
arms bf16_batched (legacy BF16 control) / bf16 (stepwise) / rabit (canonical RABIT, stepwise, aging during decode).

Self-contained: no sibling-module import at module level; repository paths are resolved ONLY in local context
(modal.is_local()); remotely the needed files are at /repo/benchmarks/mlsys2027/. Only these four files are shipped --
canonical_rabit_quality.py is the sole RABIT implementation in the container; no legacy quality script
(benchmarks/quality/*), no kvquant_k3 and no vLLM source is present.

Order inside the container (every step aborts the run on failure, before any model output is scored):
  hardware guard -> shipped-file hashes -> runtime torch / CUDA environment == the environment of the accepted CUDA
  conformance diagnostics -> model snapshot at the pinned revision + file-by-file manifest verification
  -> WikiText SHA-256 -> token-pool SHA-256 -> model load + geometry -> warm-up -> arms in order (bf16_batched, bf16,
  rabit; windows 1..32).
Post-failure amendment (d7ba819): the former 'GPU canonical state == CPU canonical state, bit-exact' gate was removed
(cross-device bit-exactness is not a semantic requirement; no tolerance replaces it). CUDA semantics are validated
separately against the frozen independent oracle (ec80638 Llama, fd8d275 Qwen); the oracle is NOT shipped here.
The result crosses the RPC boundary as ONE strict-JSON str (parity Attempt 3 transport rule).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import modal

REMOTE_DIR = "/repo/benchmarks/mlsys2027"
FILES = ["canonical_rabit_quality.py", "canonical_ppl_core.py", "canonical_ppl_identity.py", "exp14_model_snapshot.py"]
CANONICAL_RABIT_QUALITY_SHA256_LF = "195cb4896c4708cc6ecc450930733962ea3e08e21440fc9406dedbe1bb4c11d5"  # c360697
RESULT_PATH_ENV = "CANONICAL_PPL_RESULT_PATH"
EXPECTED_SHA_ENV = "CANONICAL_PPL_EXPECTED_FILE_SHA256_LF"
FORBIDDEN_MODULES = ("continuation_ppl", "multilingual_ppl", "niah", "passage_retrieval", "hotpotqa", "qasper",
                     "run_suite", "kvquant_k3", "vllm", "canonical_quality_parity_tests", "canonical_cuda_conformance")
# the runtime of the accepted CUDA conformance diagnostics (ec80638, fd8d275); a different runtime is not validated
VALIDATED_RUNTIME = {"python_minor": "3.11", "torch": "2.11.0+cu130", "torch_cuda": "13.0",
                     "cuda_device": "NVIDIA H100 80GB HBM3", "transformers": "4.48.2"}
JSON_NATIVE = (dict, list, str, int, float, bool, type(None))
RESULT_KEYS = ("model_key", "model", "hardware", "environment", "runtime_environment", "files", "dataset", "geometry",
               "policy", "logical_kv", "arms", "aggregates", "legacy_unreachable", "timing")

app = modal.App("rabit-kv-canonical-quality-v2-ppl")
model_cache = modal.Volume.from_name("modelscope-llama31-cache", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.11.0", "transformers==4.48.2", "accelerate", "requests", "sentencepiece", "modelscope")
if modal.is_local():  # local repository layout is consulted ONLY when building the app locally
    from pathlib import Path as _Path

    _here = _Path(__file__).resolve().parent
    for _name in FILES:
        image = image.add_local_file(str(_here / _name), f"{REMOTE_DIR}/{_name}", copy=True)


def _gpu_query() -> list:
    fields = "name,memory.total,memory.used,driver_version"
    out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return [dict(zip(fields.split(","), [x.strip() for x in line.split(",")])) for line in out.splitlines() if line.strip()]


def _emit(tag: str, payload) -> None:
    print(f"{tag}={json.dumps(payload, sort_keys=True, default=str)}", flush=True)


def assert_json_native(obj, path: str = "$") -> None:
    """EXACT built-in JSON types only (str / int subclasses such as TorchVersion are rejected); no NaN."""
    t = type(obj)
    if t not in JSON_NATIVE:
        raise TypeError(f"non-JSON-native value at {path}: {t.__module__}.{t.__qualname__}")
    if t is dict:
        for k, v in obj.items():
            if type(k) is not str:
                raise TypeError(f"non-str key at {path}: {type(k).__qualname__}")
            assert_json_native(v, f"{path}.{k}")
    elif t is list:
        for i, v in enumerate(obj):
            assert_json_native(v, f"{path}[{i}]")
    elif t is float and (obj != obj or obj in (float("inf"), float("-inf"))):
        raise TypeError(f"non-finite float at {path}")


def validate_payload(payload) -> dict:
    """Local-side checks on the RPC payload; returns the parsed result."""
    if type(payload) is not str:
        raise TypeError(f"payload is {type(payload).__qualname__}, expected str")
    res = json.loads(payload)
    missing = [k for k in RESULT_KEYS if k not in res]
    if missing:
        raise ValueError(f"schema: missing {missing}")
    assert_json_native(res)
    return res


@app.function(image=image, gpu="H100!:1", timeout=3 * 3600, volumes={"/model_cache": model_cache})
def run_ppl(model_key: str, expected_file_sha256_lf: dict) -> str:
    _gpus = _gpu_query()
    _hw_ok = (len(_gpus) == 1 and "H100" in _gpus[0].get("name", "")
              and 79 * 1024 <= int(float(_gpus[0].get("memory.total", 0))) <= 82 * 1024)
    _emit("CANONICAL_PPL_HARDWARE_CHECK", {"gpus": _gpus, "passed": _hw_ok, "required": "exactly 1 x NVIDIA H100 80GB"})
    if not _hw_ok:
        _emit("CANONICAL_PPL_HARDWARE_MISMATCH", {"gpus": _gpus})
        raise RuntimeError(f"hardware mismatch: {_gpus}; no model loaded, nothing scored")

    import gc
    import importlib.metadata as md
    import time
    from pathlib import Path

    t_start = time.time()
    # ---- shipped files: exactly the four expected files, hashes equal to the launcher's committed tree
    repo_files = sorted(str(p.relative_to("/repo")) for p in Path("/repo").rglob("*") if p.is_file()
                        and "__pycache__" not in p.parts)
    file_sha = {n: hashlib.sha256((Path(REMOTE_DIR) / n).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for n in FILES}
    files_ok = (repo_files == sorted(f"benchmarks/mlsys2027/{n}" for n in FILES) and file_sha == expected_file_sha256_lf
                and file_sha["canonical_rabit_quality.py"] == CANONICAL_RABIT_QUALITY_SHA256_LF)
    _emit("CANONICAL_PPL_FILES", {"repo_files": repo_files, "sha256_lf": file_sha, "passed": files_ok})
    if not files_ok:
        raise RuntimeError("shipped files differ from the committed harness; no model loaded")

    sys.path.insert(0, REMOTE_DIR)
    import canonical_ppl_identity as ident

    ident.check_constants()
    if not ident.hardware_ok(_gpus):
        raise RuntimeError("hardware predicate disagreement")
    m = ident.MODELS[model_key]

    import requests
    import torch
    from modelscope import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import canonical_ppl_core as core
    import canonical_rabit_quality as crq

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA device is required")
    device, dtype = torch.device("cuda"), torch.bfloat16
    environment = {"python": str(sys.version.split()[0]), "torch": str(torch.__version__),
                   "torch_cuda": str(torch.version.cuda), "cuda_device": str(torch.cuda.get_device_name(0)),
                   "packages": {p: str(md.version(p)) for p in ("transformers", "accelerate", "modelscope", "requests",
                                                                "sentencepiece", "tokenizers", "safetensors")}}
    _emit("CANONICAL_PPL_ENVIRONMENT", environment)
    runtime = {"python_minor": ".".join(environment["python"].split(".")[:2]), "torch": environment["torch"],
               "torch_cuda": environment["torch_cuda"], "cuda_device": environment["cuda_device"],
               "transformers": environment["packages"]["transformers"]}
    runtime_ok = runtime == VALIDATED_RUNTIME
    _emit("CANONICAL_PPL_RUNTIME", {"runtime": runtime, "validated": VALIDATED_RUNTIME, "passed": runtime_ok})
    if not runtime_ok:
        raise RuntimeError(f"runtime {runtime} is not the runtime of the accepted CUDA conformance diagnostics; "
                           "no model loaded")

    # ---- model identity: pinned immutable revision, verified file by file BEFORE loading
    os.environ["MODELSCOPE_CACHE"] = "/model_cache"
    model_dir = snapshot_download(m["model_id"], revision=m["revision"], cache_dir="/model_cache")
    try:
        model_cache.commit()
    except Exception:  # noqa: BLE001
        pass
    verification = ident.verify_dir(model_dir, model_key)
    _emit("CANONICAL_PPL_MODEL", verification)
    if not verification["passed"]:
        raise RuntimeError(f"model snapshot differs from the frozen manifest: {verification['mismatches']} "
                           f"extra={verification['extra_files']}; no model loaded")

    # ---- dataset: pinned raw text and pinned token pool (frozen legacy windows)
    response = requests.get(ident.WIKITEXT_URL, timeout=30)
    response.raise_for_status()
    wikitext_sha = hashlib.sha256(response.content).hexdigest()
    if wikitext_sha != ident.WIKITEXT_SHA256:
        raise RuntimeError(f"WikiText-2 test text differs from the pinned SHA-256: {wikitext_sha}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    samples, context_tokens, eval_tokens = ident.SAMPLES, ident.CONTEXT_TOKENS, ident.EVAL_TOKENS
    pool = core.build_token_pool(core.wikitext_lines(response.content.decode("utf-8")), tokenizer, samples,
                                 context_tokens, eval_tokens, ident.LINE_BLOCK)
    pool_sha = core.pool_sha256(pool)
    dataset = {"url": ident.WIKITEXT_URL, "wikitext_sha256": wikitext_sha, "token_pool_sha256": pool_sha,
               "pool_tokens": len(pool), "samples": samples, "context_tokens": context_tokens,
               "eval_tokens": eval_tokens, "add_special_tokens": False}
    _emit("CANONICAL_PPL_DATASET", dataset)
    if pool_sha != m["token_pool_sha256"]:
        raise RuntimeError("token pool differs from the pinned frozen windows; no model loaded")
    windows = core.split_windows(pool, samples, context_tokens, eval_tokens, device)

    # ---- model (same loading / seed / TF32 settings as the legacy protocol)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype, device_map=None, low_cpu_mem_usage=True,
                                                 trust_remote_code=True, attn_implementation="sdpa").to(device)
    model.eval()
    cfg = model.config
    geometry = {"layers": int(cfg.num_hidden_layers), "kv_heads": int(cfg.num_key_value_heads),
                "attention_heads": int(cfg.num_attention_heads),
                "head_dim": int(cfg.hidden_size // cfg.num_attention_heads), "architecture": type(model).__name__}
    if (geometry["layers"], geometry["kv_heads"], geometry["head_dim"]) != (m["layers"], m["kv_heads"], m["head_dim"]):
        raise RuntimeError(f"unexpected model geometry: {geometry}")
    if crq.POLICY != {"k_bits": 3, "v_bits": 2, "group_size": 32, "residual_tokens": 4, "metadata_bits": 8,
                      "metadata_group_size": 64}:
        raise RuntimeError("canonical policy is not K3 / V2 / G32 / R4 / META8g64")

    with torch.inference_mode():  # untimed warm-up (legacy protocol)
        _ = model(input_ids=windows[0][0][:, :32], use_cache=True)
    torch.cuda.synchronize()

    n_end = context_tokens + eval_tokens - 1
    bf16_mb = lambda n: geometry["layers"] * 2 * n * geometry["kv_heads"] * geometry["head_dim"] * 2 / 2**20  # noqa: E731
    rabit_mb = lambda n: crq.logical_bytes(n, geometry["layers"], geometry["kv_heads"], geometry["head_dim"])["total"] / 2**20  # noqa: E731
    logical_kv = {"note": "logical storage of the canonical representation; NOT allocator capacity",
                  "at_prefill_tokens": context_tokens, "at_final_tokens": n_end,
                  "bf16_mb": [bf16_mb(context_tokens), bf16_mb(n_end)],
                  "rabit_mb": [rabit_mb(context_tokens), rabit_mb(n_end)]}

    arms, aggregates, seconds = {}, {}, {}
    for arm in core.ARMS:
        print(f"Running {arm}...", flush=True)
        rows, t0 = [], time.time()
        for i, (context_ids, continuation_ids) in enumerate(windows, start=1):
            gc.collect()
            torch.cuda.empty_cache()
            row = core.score(model, context_ids, continuation_ids, arm)
            row["window"] = i
            rows.append(row)
            print(f"  sample {i}/{samples}: PPL={row['ppl']:.4f}", flush=True)
        torch.cuda.synchronize()
        arms[arm], aggregates[arm], seconds[arm] = rows, core.aggregate(rows), time.time() - t0
        print(flush=True)

    loaded = sorted(n for n in sys.modules if n.split(".")[0] in FORBIDDEN_MODULES)
    legacy_unreachable = {"forbidden_modules_loaded": loaded, "repo_files": repo_files,
                          "rabit_implementation": str(crq.__file__), "passed": not loaded}
    if loaded:
        raise RuntimeError(f"legacy modules were imported: {loaded}")

    res = {"kind": "canonical-quality-v2 continuation PPL (logical quality; NOT physical serving evidence)",
           "model_key": model_key, "model": verification, "hardware": {"gpus": _gpus, "passed": _hw_ok},
           "environment": environment,
           "runtime_environment": {"runtime": runtime, "validated": dict(VALIDATED_RUNTIME), "passed": runtime_ok},
           "files": {"sha256_lf": file_sha, "passed": files_ok}, "dataset": dataset,
           "geometry": geometry, "policy": dict(crq.POLICY), "logical_kv": logical_kv,
           "arm_order": list(core.ARMS), "arms": arms, "aggregates": aggregates,
           "legacy_unreachable": legacy_unreachable,
           "timing": {"arm_seconds": seconds, "total_seconds": time.time() - t_start,
                      "note": "experiment planning only; not deployment latency"}}
    assert_json_native(res)
    payload = json.dumps(res, ensure_ascii=False, allow_nan=False)
    _emit("CANONICAL_PPL_REMOTE", {"completed": True, "model_key": model_key,
                                   "ppl": {a: aggregates[a]["ppl"] for a in core.ARMS}})
    return payload


@app.local_entrypoint()
def main(model_key: str):
    expected = json.loads(os.environ[EXPECTED_SHA_ENV])
    payload = run_ppl.remote(model_key, expected)
    res = validate_payload(payload)
    data = payload.encode("utf-8")
    path = os.environ[RESULT_PATH_ENV]
    with open(path, "wb") as fh:
        fh.write(data)
    with open(path, "rb") as fh:
        back = fh.read()
    if back != data or json.loads(back.decode("utf-8")) != res:
        raise SystemExit("local result file does not round-trip")
    print("CANONICAL_PPL_LOCAL=" + json.dumps({"payload_type": "str", "bytes": len(data),
                                               "sha256": hashlib.sha256(data).hexdigest(),
                                               "model_key": res["model_key"]}), flush=True)
