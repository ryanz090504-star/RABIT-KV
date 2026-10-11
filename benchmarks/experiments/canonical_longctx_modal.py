"""
canonical-quality-v2 LONG-CONTEXT suite -- Modal app (strict ONE H100 80GB; fail-closed hardware guard before anything
else). Launched only by benchmarks/mlsys2027/run_canonical_longctx.py.

ONE invocation = the whole registered suite for Llama-3.1-8B, in fixed order NIAH (57) -> Passage Retrieval (200) ->
HotpotQA (100); per example BF16 then canonical RABIT on the identical prompt token ids.

Self-contained: no sibling-module import at module level; only the six shipped files are in the container --
canonical_rabit_quality.py is the sole RABIT implementation; no legacy quality script (benchmarks/quality/*), no
kvquant_k3 oracle and no vLLM source is present.

Order inside the container (every identity step aborts the run on failure, before any model output exists):
  hardware guard -> shipped-file hashes -> runtime == the validated conformance runtime -> model snapshot at the pinned
  revision + manifest -> dataset files (pinned SHA-256 / size) -> WikiText SHA-256 -> ALL prompts built and every
  prompt-set SHA-256 checked -> model load + geometry -> tasks in order.
NO RESULT-DEPENDENT CONTROL FLOW: after the identity checks the container runs every example of every task
unconditionally. While the tasks run it prints ONLY progress lines (task, position, arm) -- no prediction, no score,
no aggregate. Predictions and scores leave the container only in the returned payload (and, best-effort, as raw rows
in a crash-recovery volume that nothing reads during the run).
The result crosses the RPC boundary as ONE strict-JSON str (parity Attempt 3 transport rule).

SMOKE TEST (smoke_test=True; orchestration only, NOT a quality run): the same container performs every identity step
above, writes and commits one line to the crash-recovery volume, and RETURNS BEFORE the model is loaded -- no forward
pass, no generation, no BF16 / RABIT pair, no score; the task lists of its payload are empty.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import modal

REMOTE_DIR = "/repo/benchmarks/mlsys2027"
FILES = ["canonical_rabit_quality.py", "canonical_longctx_core.py", "canonical_longctx_tasks.py",
         "canonical_longctx_pins.py", "canonical_ppl_identity.py", "exp14_model_snapshot.py"]
CANONICAL_RABIT_QUALITY_SHA256_LF = "195cb4896c4708cc6ecc450930733962ea3e08e21440fc9406dedbe1bb4c11d5"  # c360697
MODEL_KEY = "llama3_1_8b"
TASK_ORDER = ("niah", "passage_retrieval", "hotpotqa")
RESULT_PATH_ENV = "CANONICAL_LONGCTX_RESULT_PATH"
EXPECTED_SHA_ENV = "CANONICAL_LONGCTX_EXPECTED_FILE_SHA256_LF"
FORBIDDEN_MODULES = ("continuation_ppl", "multilingual_ppl", "niah", "passage_retrieval", "hotpotqa", "qasper",
                     "run_suite", "kvquant_k3", "vllm", "canonical_quality_parity_tests", "canonical_cuda_conformance")
VALIDATED_RUNTIME = {"python_minor": "3.11", "torch": "2.11.0+cu130", "torch_cuda": "13.0",
                     "cuda_device": "NVIDIA H100 80GB HBM3", "transformers": "4.48.2"}
SUITE_TIMEOUT_S = 12 * 3600
JSON_NATIVE = (dict, list, str, int, float, bool, type(None))
RESULT_KEYS = ("smoke_test", "hardware", "environment", "runtime_environment", "files", "model", "datasets", "prompt_sets", "geometry",
               "policy", "task_order", "tasks", "legacy_unreachable", "artifact_persistence", "timing")

app = modal.App("rabit-kv-canonical-quality-v2-longctx")
model_cache = modal.Volume.from_name("modelscope-llama31-cache", create_if_missing=True)
artifacts = modal.Volume.from_name("rabit-kv-canonical-longctx-artifacts", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.11.0", "transformers==4.48.2", "accelerate", "requests", "sentencepiece", "modelscope", "datasets>=3.0.0",
    "pyarrow")
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


def _progress(task: str, position: int, total: int, arm: str) -> None:
    """The ONLY output while the tasks run: where the suite is. Never a prediction, a score or an aggregate."""
    print(f"CANONICAL_LONGCTX_PROGRESS task={task} unit={position}/{total} arm={arm}", flush=True)


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


@app.function(image=image, gpu="H100!:1", timeout=SUITE_TIMEOUT_S,
              volumes={"/model_cache": model_cache, "/artifacts": artifacts})
def run_suite(attempt: int, expected_file_sha256_lf: dict, smoke_test: bool = False) -> str:
    _gpus = _gpu_query()
    _hw_ok = (len(_gpus) == 1 and "H100" in _gpus[0].get("name", "")
              and 79 * 1024 <= int(float(_gpus[0].get("memory.total", 0))) <= 82 * 1024)
    _emit("CANONICAL_LONGCTX_HARDWARE_CHECK", {"gpus": _gpus, "passed": _hw_ok, "required": "exactly 1 x NVIDIA H100 80GB"})
    if not _hw_ok:
        raise RuntimeError(f"hardware mismatch: {_gpus}; no model loaded, nothing generated")

    import gc
    import importlib.metadata as md
    import time
    from pathlib import Path

    t_start = time.time()
    repo_files = sorted(str(p.relative_to("/repo")) for p in Path("/repo").rglob("*") if p.is_file()
                        and "__pycache__" not in p.parts)
    file_sha = {n: hashlib.sha256((Path(REMOTE_DIR) / n).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for n in FILES}
    files_ok = (repo_files == sorted(f"benchmarks/mlsys2027/{n}" for n in FILES) and file_sha == expected_file_sha256_lf
                and file_sha["canonical_rabit_quality.py"] == CANONICAL_RABIT_QUALITY_SHA256_LF)
    _emit("CANONICAL_LONGCTX_FILES", {"repo_files": repo_files, "sha256_lf": file_sha, "passed": files_ok})
    if not files_ok:
        raise RuntimeError("shipped files differ from the committed harness; no model loaded")

    sys.path.insert(0, REMOTE_DIR)
    import canonical_longctx_pins as pins
    import canonical_longctx_tasks as tasks
    import canonical_ppl_identity as ident

    ident.check_constants()
    if not ident.hardware_ok(_gpus):
        raise RuntimeError("hardware predicate disagreement")
    m = ident.MODELS[MODEL_KEY]

    import requests
    import torch
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from modelscope import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import canonical_longctx_core as core
    import canonical_rabit_quality as crq

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA device is required")
    device, dtype = torch.device("cuda"), torch.bfloat16
    environment = {"python": str(sys.version.split()[0]), "torch": str(torch.__version__),
                   "torch_cuda": str(torch.version.cuda), "cuda_device": str(torch.cuda.get_device_name(0)),
                   "packages": {p: str(md.version(p)) for p in ("transformers", "accelerate", "modelscope", "requests",
                                                                "sentencepiece", "tokenizers", "safetensors", "datasets",
                                                                "pyarrow", "huggingface_hub")}}
    _emit("CANONICAL_LONGCTX_ENVIRONMENT", environment)
    runtime = {"python_minor": ".".join(environment["python"].split(".")[:2]), "torch": environment["torch"],
               "torch_cuda": environment["torch_cuda"], "cuda_device": environment["cuda_device"],
               "transformers": environment["packages"]["transformers"]}
    runtime_ok = runtime == VALIDATED_RUNTIME
    _emit("CANONICAL_LONGCTX_RUNTIME", {"runtime": runtime, "validated": VALIDATED_RUNTIME, "passed": runtime_ok})
    if not runtime_ok:
        raise RuntimeError(f"runtime {runtime} is not the runtime of the accepted CUDA conformance diagnostics; "
                           "no model loaded")

    # ---- model identity: pinned immutable revision, verified file by file BEFORE loading
    os.environ["MODELSCOPE_CACHE"] = "/model_cache"
    model_dir = snapshot_download(m["model_id"], revision=m["revision"], cache_dir="/model_cache")
    verification = ident.verify_dir(model_dir, MODEL_KEY)
    _emit("CANONICAL_LONGCTX_MODEL", verification)
    if not verification["passed"]:
        raise RuntimeError("model snapshot differs from the frozen manifest; no model loaded")

    # ---- datasets: pinned LongBench parquet files and the pinned WikiText-2 filler
    datasets_info, data = {}, {}
    for task, d in tasks.DATASETS.items():
        path = hf_hub_download(repo_id=tasks.LONGBENCH_REPO, repo_type="dataset", revision=d["revision"],
                               filename=d["filename"])
        raw = Path(path).read_bytes()
        data[task] = load_dataset("parquet", data_files={"test": path}, split="test")
        datasets_info[task] = {"repo": tasks.LONGBENCH_REPO, "revision": d["revision"], "filename": d["filename"],
                               "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "rows": len(data[task])}
        if (datasets_info[task]["sha256"], len(raw), len(data[task])) != (d["sha256"], d["bytes"], d["rows"]):
            raise RuntimeError(f"{task}: dataset file differs from the frozen identity: {datasets_info[task]}")
    response = requests.get(tasks.WIKITEXT_URL, timeout=30)
    response.raise_for_status()
    datasets_info["wikitext_sha256"] = hashlib.sha256(response.content).hexdigest()
    if datasets_info["wikitext_sha256"] != tasks.WIKITEXT_SHA256:
        raise RuntimeError("WikiText-2 test text differs from the pinned SHA-256")
    _emit("CANONICAL_LONGCTX_DATASETS", datasets_info)

    # ---- every prompt of the suite, built and identity-checked BEFORE the model is loaded
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    parts = tasks.niah_parts(tokenizer, response.content.decode("utf-8"))
    units = {"niah": [{"key": [int(c), round(float(d), 2)], "ids": tasks.niah_prompt_ids(parts, c, d),
                       "original_tokens": int(c), "answers": [tasks.NIAH_SECRET]} for c, d in tasks.niah_cases()]}
    for task in ("passage_retrieval", "hotpotqa"):
        units[task] = []
        for index in tasks.select_indices(task, list(data[task]["length"])):
            example = data[task][index]
            ids, original = tasks.longbench_prompt_ids(task, example, tokenizer)
            units[task].append({"key": int(index), "ids": ids, "original_tokens": int(original),
                                "answers": tasks.normalize_answers(example["answers"])})
    prompt_sets = {}
    for task in TASK_ORDER:
        digest = hashlib.sha256("\n".join(
            f"{json.dumps(u['key'])}\t{len(u['ids'])}\t{tasks.ids_sha256(u['ids'])}" for u in units[task]).encode()).hexdigest()
        prompt_sets[task] = {"units": len(units[task]), "prompt_set_sha256": digest,
                             "passed": {"units": len(units[task]), "prompt_set_sha256": digest} == pins.PROMPT_SETS[task]}
    _emit("CANONICAL_LONGCTX_PROMPT_SETS", prompt_sets)
    if not all(p["passed"] for p in prompt_sets.values()):
        raise RuntimeError("a prompt set differs from the pinned identity; no model loaded")

    if smoke_test:  # plumbing only: stop BEFORE the model is loaded; nothing is forwarded, generated or scored
        smoke_path, smoke = f"/artifacts/smoke_test_{attempt}/smoke.jsonl", {"line_written": False, "committed": False}
        os.makedirs(os.path.dirname(smoke_path), exist_ok=True)
        with open(smoke_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"smoke_test": attempt, "prompt_sets": prompt_sets}) + "\n")
        smoke["line_written"] = True
        artifacts.commit()
        smoke["committed"] = True
        loaded = sorted(n for n in sys.modules if n.split(".")[0] in FORBIDDEN_MODULES)
        res = {"kind": "canonical-quality-v2 long-context suite -- PLUMBING SMOKE TEST (no model load, no inference, no "
                       "score; NOT a quality result)",
               "smoke_test": True, "attempt": attempt, "model_key": MODEL_KEY, "model": verification,
               "hardware": {"gpus": _gpus, "passed": _hw_ok}, "environment": environment,
               "runtime_environment": {"runtime": runtime, "validated": dict(VALIDATED_RUNTIME), "passed": runtime_ok},
               "files": {"sha256_lf": file_sha, "passed": files_ok}, "datasets": datasets_info, "prompt_sets": prompt_sets,
               "geometry": None, "policy": dict(crq.POLICY), "task_order": list(TASK_ORDER),
               "tasks": {task: [] for task in TASK_ORDER},
               "tokenizer": {"class": type(tokenizer).__name__, "eos_token": str(tokenizer.eos_token),
                             "eos_token_id": tokenizer.eos_token_id},
               "imports": {"generation_core_arms": list(core.ARMS), "datasets": str(md.version("datasets")),
                           "pyarrow": str(md.version("pyarrow")), "huggingface_hub": str(md.version("huggingface_hub"))},
               "model_loaded": False, "forward_passes": 0, "units_prepared": {t: len(units[t]) for t in TASK_ORDER},
               "legacy_unreachable": {"forbidden_modules_loaded": loaded, "repo_files": repo_files, "passed": not loaded},
               "artifact_persistence": {"path": smoke_path, **smoke},
               "timing": {"total_seconds": time.time() - t_start}}
        assert_json_native(res)
        _emit("CANONICAL_LONGCTX_SMOKE", {"completed": True, "model_loaded": False, "forward_passes": 0})
        return json.dumps(res, ensure_ascii=False, allow_nan=False)

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
                "head_dim": int(cfg.hidden_size // cfg.num_attention_heads), "architecture": type(model).__name__}
    if (geometry["layers"], geometry["kv_heads"], geometry["head_dim"]) != (m["layers"], m["kv_heads"], m["head_dim"]):
        raise RuntimeError(f"unexpected model geometry: {geometry}")
    if crq.POLICY != {"k_bits": 3, "v_bits": 2, "group_size": 32, "residual_tokens": 4, "metadata_bits": 8,
                      "metadata_group_size": 64}:
        raise RuntimeError("canonical policy is not K3 / V2 / G32 / R4 / META8g64")
    eos_id = tokenizer.eos_token_id
    max_new = {"niah": tasks.NIAH_MAX_NEW_TOKENS, "passage_retrieval": tasks.LONGBENCH_MAX_NEW_TOKENS,
               "hotpotqa": tasks.LONGBENCH_MAX_NEW_TOKENS}

    def as_tensor(ids):
        return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    def run_arm(task, unit, arm):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gen = core.generate(model, as_tensor(unit["ids"]), arm, max_new[task], eos_id)
        torch.cuda.synchronize()
        gen["seconds"] = time.perf_counter() - t0
        gen["prediction"] = tokenizer.decode(gen["generated_ids"], skip_special_tokens=True).strip()
        gen.update(tasks.niah_score(gen["prediction"]) if task == "niah" else tasks.score(task, gen["prediction"], unit["answers"]))
        return gen

    # ---- crash-recovery rows (best effort; never read during the run; a failure here never changes the run)
    persist = {"path": f"/artifacts/attempt_{attempt}/rows.jsonl", "rows_written": 0, "errors": 0}

    def persist_row(task, row):
        try:
            os.makedirs(os.path.dirname(persist["path"]), exist_ok=True)
            with open(persist["path"], "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"task": task, **row}, ensure_ascii=False) + "\n")
            persist["rows_written"] += 1
            if persist["rows_written"] % 10 == 0:
                artifacts.commit()
        except Exception:  # noqa: BLE001
            persist["errors"] += 1

    results, seconds = {}, {}
    for task in TASK_ORDER:  # fixed order; every unit of every task runs unconditionally
        t_task = time.time()
        if task == "niah":  # legacy warm-up: one unreported BF16 case at the smallest context, first depth
            core.generate(model, as_tensor(tasks.niah_prompt_ids(parts, min(tasks.NIAH_CONTEXTS), tasks.NIAH_DEPTHS[0])),
                          "bf16", max_new[task], eos_id)
        else:  # legacy warm-up: one 2-token forward
            with torch.inference_mode():
                model(input_ids=torch.tensor([[tokenizer.bos_token_id or 1, tokenizer.eos_token_id or 2]],
                                             dtype=torch.long, device=device), use_cache=True)
        torch.cuda.synchronize()
        rows = []
        for position, unit in enumerate(units[task]):
            row = {"index": position, "key": unit["key"], "prompt_tokens": len(unit["ids"]),
                   "original_tokens": unit["original_tokens"], "prompt_ids_sha256": tasks.ids_sha256(unit["ids"]),
                   "answers": unit["answers"], "arms": {}}
            for arm in core.ARMS:
                _progress(task, position + 1, len(units[task]), arm)
                row["arms"][arm] = run_arm(task, unit, arm)
            rows.append(row)
            persist_row(task, row)
        results[task], seconds[task] = rows, time.time() - t_task
    try:
        artifacts.commit()
    except Exception:  # noqa: BLE001
        persist["errors"] += 1

    loaded = sorted(n for n in sys.modules if n.split(".")[0] in FORBIDDEN_MODULES)
    legacy_unreachable = {"forbidden_modules_loaded": loaded, "repo_files": repo_files,
                          "rabit_implementation": str(crq.__file__), "passed": not loaded}
    if loaded:
        raise RuntimeError(f"legacy modules were imported: {loaded}")

    res = {"kind": "canonical-quality-v2 long-context suite (logical quality; NOT physical serving evidence)",
           "smoke_test": False, "attempt": attempt, "model_key": MODEL_KEY, "model": verification,
           "hardware": {"gpus": _gpus, "passed": _hw_ok},
           "environment": environment,
           "runtime_environment": {"runtime": runtime, "validated": dict(VALIDATED_RUNTIME), "passed": runtime_ok},
           "files": {"sha256_lf": file_sha, "passed": files_ok}, "datasets": datasets_info, "prompt_sets": prompt_sets,
           "geometry": geometry, "policy": dict(crq.POLICY), "eos_token_id": eos_id, "max_new_tokens": max_new,
           "task_order": list(TASK_ORDER), "arm_order": list(core.ARMS), "tasks": results,
           "legacy_unreachable": legacy_unreachable, "artifact_persistence": persist,
           "timing": {"task_seconds": seconds, "total_seconds": time.time() - t_start,
                      "note": "experiment planning only; not deployment latency"}}
    assert_json_native(res)
    payload = json.dumps(res, ensure_ascii=False, allow_nan=False)
    _emit("CANONICAL_LONGCTX_REMOTE", {"completed": True, "units": {t: len(results[t]) for t in TASK_ORDER}})
    return payload


@app.local_entrypoint()
def main(attempt: int, smoke_test: bool = False):
    expected = json.loads(os.environ[EXPECTED_SHA_ENV])
    payload = run_suite.remote(attempt, expected, smoke_test)
    res = validate_payload(payload)
    data = payload.encode("utf-8")
    path = os.environ[RESULT_PATH_ENV]
    with open(path, "wb") as fh:
        fh.write(data)
    with open(path, "rb") as fh:
        back = fh.read()
    if back != data or json.loads(back.decode("utf-8")) != res:
        raise SystemExit("local result file does not round-trip")
    print("CANONICAL_LONGCTX_LOCAL=" + json.dumps({"payload_type": "str", "bytes": len(data),
                                                   "sha256": hashlib.sha256(data).hexdigest()}), flush=True)
