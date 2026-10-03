"""
canonical-quality-v2 GPU SEMANTIC-CONFORMANCE DIAGNOSTIC -- Modal app (strict ONE H100 80GB; fail-closed hardware guard
first). Launched only by benchmarks/mlsys2027/run_canonical_cuda_conformance.py.

NO scoring, NO PPL, NO generation, NO quality benchmark: for Llama-3.1-8B (pinned revision, verified manifest) it runs
ONLY the 1024-token BF16 prefill of the frozen canonical-PPL window 1 to obtain the raw K / V of the 32 layers (the
logits are discarded unread), then compares, from identical bytes, CPU canonical / CUDA canonical / CUDA frozen oracle
field by field (canonical_cuda_conformance.py). It also runs the synthetic Llama / Qwen geometry cases on CUDA and
re-evaluates the Attempt-1 gate (core.prefill_state_parity, unchanged) for the record.

Shipped files: the four files of the PPL harness (unchanged), the comparison module, the accepted parity suite (for
its oracle loader, distributions and T1 / T2 comparisons) and the frozen oracle source kvquant_k3.py.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import modal

REMOTE_REPO = "/repo"
FILES = ["benchmarks/mlsys2027/canonical_rabit_quality.py", "benchmarks/mlsys2027/canonical_ppl_core.py",
         "benchmarks/mlsys2027/canonical_ppl_identity.py", "benchmarks/mlsys2027/exp14_model_snapshot.py",
         "benchmarks/mlsys2027/canonical_cuda_conformance.py", "benchmarks/mlsys2027/canonical_quality_parity_tests.py",
         "vllm-kvquant/vllm/v1/attention/ops/kvquant_k3.py"]
ORACLE = "vllm-kvquant/vllm/v1/attention/ops/kvquant_k3.py"
CANONICAL_RABIT_QUALITY_SHA256_LF = "195cb4896c4708cc6ecc450930733962ea3e08e21440fc9406dedbe1bb4c11d5"  # c360697
ORACLE_FILE_SHA256_LF = "c2a48e97d9ca71fe55eb400938c3b3e665422b1534c4fb8fbe6ef02f2e60cc0f"  # accepted parity 8fa9a9c
ORACLE_EXTRACTED_SHA256 = "d58f58a5431bb5eeb2eb24bf3a86b6ea3160c83b9c4ecc1585dd4cad95450f3c"  # accepted parity 8fa9a9c
MODEL_KEY = "llama3_1_8b"
RESULT_PATH_ENV = "CANONICAL_CUDA_CONFORMANCE_RESULT_PATH"
EXPECTED_SHA_ENV = "CANONICAL_CUDA_CONFORMANCE_EXPECTED_FILE_SHA256_LF"
JSON_NATIVE = (dict, list, str, int, float, bool, type(None))
RESULT_KEYS = ("hardware", "environment", "files", "oracle", "model", "dataset", "window", "geometry", "layers",
               "attempt1_gate_reevaluated", "synthetic", "no_scoring", "timing")

app = modal.App("rabit-kv-canonical-quality-v2-cuda-conformance")
model_cache = modal.Volume.from_name("modelscope-llama31-cache", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.11.0", "transformers==4.48.2", "accelerate", "requests", "sentencepiece", "modelscope")
if modal.is_local():  # local repository layout is consulted ONLY when building the app locally
    from pathlib import Path as _Path

    _root = _Path(__file__).resolve().parents[2]
    for _rel in FILES:
        image = image.add_local_file(str(_root / _rel), f"{REMOTE_REPO}/{_rel}", copy=True)


def _gpu_query() -> list:
    fields = "name,memory.total,memory.used,driver_version"
    out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return [dict(zip(fields.split(","), [x.strip() for x in line.split(",")])) for line in out.splitlines() if line.strip()]


def _emit(tag: str, payload) -> None:
    print(f"{tag}={json.dumps(payload, sort_keys=True, default=str)}", flush=True)


def assert_json_native(obj, path: str = "$") -> None:
    """EXACT built-in JSON types only; no NaN / inf."""
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
    if type(payload) is not str:
        raise TypeError(f"payload is {type(payload).__qualname__}, expected str")
    res = json.loads(payload)
    missing = [k for k in RESULT_KEYS if k not in res]
    if missing:
        raise ValueError(f"schema: missing {missing}")
    assert_json_native(res)
    return res


@app.function(image=image, gpu="H100!:1", timeout=3600, volumes={"/model_cache": model_cache})
def conformance(expected_file_sha256_lf: dict) -> str:
    _gpus = _gpu_query()
    _hw_ok = (len(_gpus) == 1 and "H100" in _gpus[0].get("name", "")
              and 79 * 1024 <= int(float(_gpus[0].get("memory.total", 0))) <= 82 * 1024)
    _emit("CUDA_CONFORMANCE_HARDWARE_CHECK", {"gpus": _gpus, "passed": _hw_ok, "required": "exactly 1 x NVIDIA H100 80GB"})
    if not _hw_ok:
        raise RuntimeError(f"hardware mismatch: {_gpus}; nothing loaded")

    import importlib.metadata as md
    import struct
    import time
    from pathlib import Path

    t_start = time.time()
    repo_files = sorted(str(p.relative_to(REMOTE_REPO)).replace(os.sep, "/") for p in Path(REMOTE_REPO).rglob("*")
                        if p.is_file() and "__pycache__" not in p.parts)
    file_sha = {n: hashlib.sha256((Path(REMOTE_REPO) / n).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for n in FILES}
    files_ok = (repo_files == sorted(FILES) and file_sha == expected_file_sha256_lf
                and file_sha["benchmarks/mlsys2027/canonical_rabit_quality.py"] == CANONICAL_RABIT_QUALITY_SHA256_LF
                and file_sha[ORACLE] == ORACLE_FILE_SHA256_LF)
    _emit("CUDA_CONFORMANCE_FILES", {"repo_files": repo_files, "sha256_lf": file_sha, "passed": files_ok})
    if not files_ok:
        raise RuntimeError("shipped files differ from the committed / frozen files; nothing loaded")

    sys.path.insert(0, f"{REMOTE_REPO}/benchmarks/mlsys2027")
    import canonical_ppl_identity as ident

    ident.check_constants()
    m = ident.MODELS[MODEL_KEY]

    import requests
    import torch
    from modelscope import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import canonical_cuda_conformance as conf
    import canonical_ppl_core as core
    import canonical_quality_parity_tests as suite
    import canonical_rabit_quality as crq

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA device is required")
    device, dtype = torch.device("cuda"), torch.bfloat16
    environment = {"python": str(sys.version.split()[0]), "torch": str(torch.__version__),
                   "torch_cuda": str(torch.version.cuda), "cudnn": str(torch.backends.cudnn.version()),
                   "cuda_device": str(torch.cuda.get_device_name(0)),
                   "packages": {p: str(md.version(p)) for p in ("transformers", "accelerate", "modelscope", "requests",
                                                                "sentencepiece", "tokenizers", "safetensors")}}
    _emit("CUDA_CONFORMANCE_ENVIRONMENT", environment)

    # ---- frozen independent oracle (AST extraction, exactly as the accepted CPU parity)
    oracle, oracle_meta = suite.load_oracle(Path(REMOTE_REPO) / ORACLE)
    oracle_meta["equals_accepted_parity_oracle"] = (oracle_meta["file_sha256_lf"] == ORACLE_FILE_SHA256_LF
                                                    and oracle_meta["extracted_sha256"] == ORACLE_EXTRACTED_SHA256)
    oracle_meta["oracle_namespace_references_canonical_module"] = any(
        getattr(v, "__name__", "") == "canonical_rabit_quality" for v in oracle.values())
    _emit("CUDA_CONFORMANCE_ORACLE", oracle_meta)
    if not oracle_meta["equals_accepted_parity_oracle"] or oracle_meta["oracle_namespace_references_canonical_module"]:
        raise RuntimeError("oracle differs from the accepted parity oracle; nothing loaded")

    # ---- model identity: pinned immutable revision, verified file by file BEFORE loading
    os.environ["MODELSCOPE_CACHE"] = "/model_cache"
    model_dir = snapshot_download(m["model_id"], revision=m["revision"], cache_dir="/model_cache")
    verification = ident.verify_dir(model_dir, MODEL_KEY)
    _emit("CUDA_CONFORMANCE_MODEL", verification)
    if not verification["passed"]:
        raise RuntimeError("model snapshot differs from the frozen manifest; no model loaded")

    # ---- the frozen canonical-PPL window 1 context
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
    if pool_sha != m["token_pool_sha256"]:
        raise RuntimeError("token pool differs from the pinned frozen windows; no model loaded")
    context_ids = core.split_windows(pool, samples, context_tokens, eval_tokens, device)[0][0]
    ids = [int(x) for x in context_ids[0].tolist()]
    dataset = {"url": ident.WIKITEXT_URL, "wikitext_sha256": wikitext_sha, "token_pool_sha256": pool_sha}
    window = {"window": 1, "context_tokens": len(ids), "bos_token_id": tokenizer.bos_token_id,
              "contains_bos": tokenizer.bos_token_id in ids,
              "context_ids_sha256_int64_le": hashlib.sha256(struct.pack(f"<{len(ids)}q", *ids)).hexdigest(),
              "equals_first_1024_pool_tokens": ids == [int(x) for x in pool[:context_tokens]]}
    _emit("CUDA_CONFORMANCE_WINDOW", {**dataset, **window})

    # ---- model (the loading / seed / TF32 settings of the PPL harness, unchanged)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype, device_map=None, low_cpu_mem_usage=True,
                                                 trust_remote_code=True, attn_implementation="sdpa").to(device)
    model.eval()
    cfg = model.config
    geometry = {"layers": int(cfg.num_hidden_layers), "kv_heads": int(cfg.num_key_value_heads),
                "head_dim": int(cfg.hidden_size // cfg.num_attention_heads), "architecture": type(model).__name__}
    if (geometry["layers"], geometry["kv_heads"], geometry["head_dim"]) != (m["layers"], m["kv_heads"], m["head_dim"]):
        raise RuntimeError(f"unexpected model geometry: {geometry}")

    # ---- ONE 1024-token BF16 prefill: raw K / V only (the logits are dropped unread; nothing is scored)
    with torch.inference_mode():
        prefill = model(input_ids=context_ids, use_cache=True)
        legacy = prefill.past_key_values.to_legacy_cache()
        raw = [(k[0].permute(1, 0, 2).contiguous(), v[0].permute(1, 0, 2).contiguous()) for k, v in legacy]
        cache = core.canonical_cache_from_prefill(prefill.past_key_values)  # the object the PPL harness would use
        del prefill
        layers = []
        for li, (rk, rv) in enumerate(raw):
            rep = conf.layer_report(oracle, rk, rv)
            # the HF-layout BF16 K / V actually held by CanonicalRabitCache vs the oracle state on the same device
            ofields = conf.oracle_fields(oracle, rk, rv)
            rep["canonical_cache_vs_cuda_oracle"] = {
                "hf_layout_k_bf16": conf.stats(cache.key_cache[li], ofields["tensors"]["hf_layout_k_bf16"]),
                "hf_layout_v_bf16": conf.stats(cache.value_cache[li], ofields["tensors"]["hf_layout_v_bf16"])}
            rep["accepted_t1_comparison_on_cuda"] = suite.t1_full_state(oracle, rk, rv)
            rep["summary"]["canonical_cache_equals_cuda_oracle"] = conf.all_equal(rep["canonical_cache_vs_cuda_oracle"])
            rep["layer"] = li
            layers.append(rep)
            _emit("CUDA_CONFORMANCE_LAYER", {"layer": li, **rep["summary"],
                                             "earliest_divergence": rep.get("earliest_divergence")})
        del cache, raw, legacy
        torch.cuda.empty_cache()
        # the Attempt-1 gate, re-evaluated with the unchanged function (its own prefill; no scoring)
        gate = core.prefill_state_parity(model, context_ids)
        _emit("CUDA_CONFORMANCE_ATTEMPT1_GATE", gate)
        synthetic = conf.synthetic_report(oracle, suite, device)
    _emit("CUDA_CONFORMANCE_SYNTHETIC", {g: {k: v for k, v in G.items() if not k.endswith("failures")} | {
        "failures": len(G["field_failures"]) + len(G["t1_failures"]) + len(G["aging_failures"])}
        for g, G in synthetic["geometries"].items()})

    res = {"kind": "canonical-quality-v2 GPU semantic-conformance diagnostic (descriptive; NO scoring / PPL / generation)",
           "hardware": {"gpus": _gpus, "passed": _hw_ok}, "environment": environment,
           "files": {"sha256_lf": file_sha, "passed": files_ok}, "oracle": oracle_meta, "model": verification,
           "dataset": dataset, "window": window, "geometry": geometry, "policy": dict(crq.POLICY), "layers": layers,
           "attempt1_gate_reevaluated": gate, "synthetic": synthetic,
           "no_scoring": {"prefill_forwards": 2, "continuation_tokens_scored": 0, "logits_read": False,
                          "generation": False, "note": "one prefill for the raw K / V and one inside the unchanged "
                                                       "core.prefill_state_parity; no loss, no PPL"},
           "timing": {"total_seconds": time.time() - t_start}}
    assert_json_native(res)
    payload = json.dumps(res, ensure_ascii=False, allow_nan=False)
    _emit("CUDA_CONFORMANCE_REMOTE", {"completed": True, "layers": len(layers),
                                      "layers_cuda_canonical_equals_cuda_oracle": sum(
                                          r["summary"]["cuda_canonical_equals_cuda_oracle"] for r in layers),
                                      "synthetic_passed": synthetic["passed"]})
    return payload


@app.local_entrypoint()
def main():
    expected = json.loads(os.environ[EXPECTED_SHA_ENV])
    payload = conformance.remote(expected)
    res = validate_payload(payload)
    data = payload.encode("utf-8")
    path = os.environ[RESULT_PATH_ENV]
    with open(path, "wb") as fh:
        fh.write(data)
    with open(path, "rb") as fh:
        back = fh.read()
    if back != data or json.loads(back.decode("utf-8")) != res:
        raise SystemExit("local result file does not round-trip")
    print("CUDA_CONFORMANCE_LOCAL=" + json.dumps({"payload_type": "str", "bytes": len(data),
                                                  "sha256": hashlib.sha256(data).hexdigest()}), flush=True)
