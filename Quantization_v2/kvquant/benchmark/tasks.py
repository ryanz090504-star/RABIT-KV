"""Task quality benchmark: LongBench, Needle In A Haystack, RULER.

Evaluates quantized KV cache on downstream task accuracy, aligned
with the TurboQuant evaluation methodology.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import statistics
import time
from typing import Any


@dataclass
class TaskResult:
    """Result from one task quality evaluation."""

    dataset: str
    task: str
    metric: str
    policy_name: str
    nbits: int | None

    baseline_score: float
    quantized_score: float
    accuracy_recovery: float
    kv_cache_dtype: str | None = None
    kv_source: str = "transformers_reference_generate"
    hardware: str | None = None
    vllm_commit: str | None = None
    quant_kernel: str | None = "transformers_reference_roundtrip"
    memory_breakdown: dict[str, Any] = field(default_factory=dict)
    num_examples: int | None = None
    context_len: int | None = None
    evidence_label: str = "quality_task_not_deploy_speedup"
    metadata: dict[str, Any] = field(default_factory=dict)


def run_needle_retrieval(
    model_name: str,
    policy,
    *,
    needle_dataset: str,
    max_examples: int = 50,
    context_lens: list[int] | None = None,
    max_new_tokens: int = 20,
    dtype: str = "float16",
    dataset_name: str | None = None,
) -> TaskResult:
    """Run exact-match retrieval/generation task quality with Transformers.

    The implementation is intentionally a reference path: it quantizes the
    stored HuggingFace KV cache between decode steps, then dequantizes it back
    into the model. It measures quality, not serving latency.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from kvquant.policies import policy_signature

    if not torch.cuda.is_available():
        raise RuntimeError("Task quality benchmark requires CUDA")

    items = _load_task_items(needle_dataset)
    if not items:
        raise ValueError(f"no task examples found in {needle_dataset}")
    eval_items = _expand_context_lens(items[:max_examples], context_lens)

    device = torch.device("cuda")
    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=None,
    ).to(device)
    model.eval()

    rows = []
    for item in eval_items:
        rows.append(
            _eval_one_needle(
                model=model,
                tokenizer=tokenizer,
                policy=policy,
                item=item,
                ctx_len=int(item.get("_context_len") or 0),
                max_new_tokens=max_new_tokens,
                dtype=dtype,
            )
        )

    baseline_score = statistics.mean(1.0 if row["baseline_correct"] else 0.0 for row in rows)
    quantized_score = statistics.mean(1.0 if row["quantized_correct"] else 0.0 for row in rows)
    accuracy_recovery = quantized_score / baseline_score if baseline_score > 0 else (1.0 if quantized_score == 0 else 0.0)
    nbits = getattr(policy, "nbits", getattr(policy, "low_bits", None))

    return TaskResult(
        dataset=dataset_name or _infer_dataset_name(needle_dataset),
        task="exact_match_generation",
        metric="exact_match",
        policy_name=policy.name,
        nbits=nbits,
        baseline_score=baseline_score,
        quantized_score=quantized_score,
        accuracy_recovery=accuracy_recovery,
        kv_cache_dtype=_kv_cache_dtype(nbits),
        hardware=torch.cuda.get_device_name(0),
        num_examples=len(rows),
        context_len=max((row.get("context_len") or 0) for row in rows) if rows else None,
        metadata={
            "model": model_name,
            "dataset_path": needle_dataset,
            "max_new_tokens": max_new_tokens,
            "dtype": dtype,
            "examples": rows,
            "policy_signature": policy_signature(policy),
            "reference_path": "transformers_cache_quant_dequant_generate",
        },
    )


def _load_needle_items(path: str) -> list[dict]:
    return _load_task_items(path)


def _load_task_items(path: str) -> list[dict]:
    import json
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _eval_one_needle(
    *,
    model,
    tokenizer,
    policy,
    item: dict,
    ctx_len: int,
    max_new_tokens: int,
    dtype: str,
) -> dict:
    """Evaluate one needle item. Returns {baseline_correct, quantized_correct}."""
    prompt, answer, context_len = _build_prompt(item, tokenizer, ctx_len)
    t0 = time.perf_counter()
    baseline = _greedy_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        target_dtype=getattr(__import__("torch"), dtype),
        policy=None,
    )
    t1 = time.perf_counter()
    quantized = _greedy_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        target_dtype=getattr(__import__("torch"), dtype),
        policy=None if _is_full_precision_policy(policy) else policy,
    )
    t2 = time.perf_counter()
    return {
        "id": item.get("id") or item.get("task_id") or item.get("name"),
        "answer": answer,
        "baseline_output": baseline,
        "quantized_output": quantized,
        "baseline_correct": _exact_match(baseline, answer),
        "quantized_correct": _exact_match(quantized, answer),
        "context_len": context_len,
        "baseline_wall_ms": (t1 - t0) * 1000,
        "quantized_wall_ms": (t2 - t1) * 1000,
    }


def _greedy_generate(
    *,
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    target_dtype,
    policy,
) -> str:
    import torch
    from kvquant.benchmark.quality import _dequant_cache, _quant_cache

    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"].to(device)
    generated: list[int] = []
    stored_cache = None
    cache_in = None

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, use_cache=True)
        if policy is None:
            cache_in = outputs.past_key_values
        else:
            stored_cache = _quant_cache(outputs.past_key_values, policy)
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

        for _ in range(max_new_tokens):
            token_id = int(next_token.item())
            if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                break
            generated.append(token_id)
            if policy is None:
                outputs = model(input_ids=next_token, past_key_values=cache_in, use_cache=True)
                cache_in = outputs.past_key_values
            else:
                cache_arg = _dequant_cache(stored_cache, target_dtype)
                outputs = model(input_ids=next_token, past_key_values=cache_arg, use_cache=True)
                stored_cache = _quant_cache(outputs.past_key_values, policy)
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _build_prompt(item: dict, tokenizer, ctx_len: int) -> tuple[str, str, int]:
    context = _first_present(item, "context", "haystack", "input", "passage", default="")
    question = _first_present(item, "question", "query", "instruction", "prompt", default="")
    answer = _answer_text(item)
    if not context and question:
        prompt = question
    else:
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    if ctx_len and ctx_len > 0:
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(ids) > ctx_len:
            ids = ids[-ctx_len:]
            prompt = tokenizer.decode(ids, skip_special_tokens=True)
    actual_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    return prompt, answer, actual_len


def _answer_text(item: dict) -> str:
    value = _first_present(item, "answer", "answers", "needle", "target", "expected", default="")
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value)


def _first_present(item: dict, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in item and item[key] not in (None, ""):
            return item[key]
    return default


def _exact_match(output: str, answer: str) -> bool:
    out = _normalize_answer(output)
    ans = _normalize_answer(answer)
    return bool(ans) and ans in out


def _normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _expand_context_lens(items: list[dict], context_lens: list[int] | None) -> list[dict]:
    if not context_lens:
        return [dict(item) for item in items]
    expanded = []
    for item in items:
        for ctx_len in context_lens:
            row = dict(item)
            row["_context_len"] = int(ctx_len)
            expanded.append(row)
    return expanded


def _infer_dataset_name(path: str) -> str:
    from pathlib import Path
    stem = Path(path).stem.lower()
    if "longbench" in stem:
        return "longbench"
    if "ruler" in stem:
        return "ruler"
    return "needle"


def _kv_cache_dtype(nbits: int | None) -> str:
    if nbits is None or nbits >= 16:
        return "fp16"
    return f"kvquant_k{nbits}"


def _is_full_precision_policy(policy: object) -> bool:
    if getattr(policy, "name", "") == "no_quant":
        return True
    nbits = getattr(policy, "nbits", getattr(policy, "low_bits", None))
    return nbits is None or int(nbits) >= 16
