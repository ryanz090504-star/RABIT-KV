"""
RABIT-KV Fast LongBench HotpotQA Evaluator
====================================================

Quality-only benchmark using the official LongBench-E HotpotQA task.

This script:
- Loads one or more official LongBench samples.
- Uses the official prompt format and retrieval score.
- Prefills the long prompt once.
- Quantizes/dequantizes the prefix KV cache once for RABIT-KV methods.
- Generates a short answer and reports retrieval accuracy.

It is NOT a deployment latency or real packed-memory benchmark.
"""

import os
import time

import modal

app = modal.App("rabit-kv-longbench-hotpotqa")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "transformers==4.48.2",
    "accelerate",
    "sentencepiece",
    "modelscope",
    "datasets>=3.0.0",
    "pyarrow",
)

model_cache = modal.Volume.from_name(
    "modelscope-llama31-cache",
    create_if_missing=True,
)


@app.function(
    gpu="H100",
    image=image,
    timeout=21600,
    volumes={"/model_cache": model_cache},
)
def run_longbench(
    model_name: str = "LLM-Research/Meta-Llama-3.1-8B-Instruct",
    sample_start: int = 0,
    samples: int = 1,
    length_bucket: str = "8k+",
    max_input_tokens: int = 16384,
    max_new_tokens: int = 32,
    methods: str = "bf16,rabit4,rabit2",
):
    import gc
    import json
    import re

    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from modelscope import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache

    dtype = torch.bfloat16
    device = torch.device("cuda")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")
    if samples < 1:
        raise ValueError("samples must be at least 1.")
    if sample_start < 0:
        raise ValueError("sample_start must be non-negative.")
    if length_bucket not in {"all", "0-4k", "4-8k", "8k+"}:
        raise ValueError(
            "length_bucket must be one of: all, 0-4k, 4-8k, 8k+."
        )
    if max_input_tokens < 1024:
        raise ValueError("max_input_tokens must be at least 1024.")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive.")

    requested = [
        item.strip().lower()
        for item in methods.split(",")
        if item.strip()
    ]
    allowed = {"bf16", "rabit4", "rabit3", "rabit2"}
    invalid = [item for item in requested if item not in allowed]
    if invalid:
        raise ValueError(
            f"Unsupported methods: {invalid}. "
            "Use bf16,rabit4,rabit3,rabit2."
        )
    if "bf16" not in requested:
        requested.insert(0, "bf16")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {model_name}")
    print("Dataset: LongBench-E hotpotqa_e")
    print(f"Length bucket: {length_bucket}")
    print(f"Samples: {samples}, starting at filtered index {sample_start}")
    print(f"Maximum input length: {max_input_tokens} tokens")
    print(f"Methods: {requested}")
    print("Metric: official LongBench QA token-level F1")
    print("Mode: quality-only prefix-cache quantization")
    print()

    os.environ["MODELSCOPE_CACHE"] = "/model_cache"
    local_model_dir = snapshot_download(
        model_name,
        cache_dir="/model_cache",
    )
    try:
        model_cache.commit()
    except Exception:
        pass

    tokenizer = AutoTokenizer.from_pretrained(
        local_model_dir,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        local_model_dir,
        torch_dtype=dtype,
        device_map=None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()

    # Newer versions of `datasets` no longer execute dataset scripts.
    # Download the official LongBench-E Qasper Parquet file directly.
    parquet_path = hf_hub_download(
        repo_id="zai-org/LongBench",
        repo_type="dataset",
        revision="92b6c5fbfb0c97b91e92d9ef79802f95ce74b05e",
        filename="hotpotqa_e/test-00000-of-00001.parquet",
    )
    dataset = load_dataset(
        "parquet",
        data_files={"test": parquet_path},
        split="test",
    )

    def in_bucket(example):
        length = int(example["length"])
        if length_bucket == "all":
            return True
        if length_bucket == "0-4k":
            return length < 4000
        if length_bucket == "4-8k":
            return 4000 <= length < 8000
        return length >= 8000

    filtered_indices = [
        index
        for index, example in enumerate(dataset)
        if in_bucket(example)
    ]

    stop = sample_start + samples
    if stop > len(filtered_indices):
        raise ValueError(
            f"Requested filtered samples [{sample_start}, {stop}) but "
            f"bucket {length_bucket!r} has only {len(filtered_indices)} examples."
        )

    prompt_template = (
        "Answer the question based on the given passages. "
        "Only give me the answer and do not output any other words.\n\n"
        "The following are given passages.\n"
        "{context}\n\n"
        "Answer the question based on the given passages. "
        "Only give me the answer and do not output any other words.\n\n"
        "Question: {input}\n"
        "Answer:"
    )

    def normalize_answers(value):
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except Exception:
                pass
            return [value]
        return [str(value)]

    def normalize_answer(value):
        import string

        def remove_articles(text):
            return re.sub(r"\\b(a|an|the)\\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punctuation(text):
            punctuation = set(string.punctuation)
            return "".join(
                character
                for character in text
                if character not in punctuation
            )

        return white_space_fix(
            remove_articles(
                remove_punctuation(value.lower())
            )
        )

    def qa_f1_score(prediction, ground_truth):
        from collections import Counter

        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()

        if not prediction_tokens or not ground_truth_tokens:
            return float(prediction_tokens == ground_truth_tokens)

        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        same = sum(common.values())
        if same == 0:
            return 0.0

        precision = same / len(prediction_tokens)
        recall = same / len(ground_truth_tokens)
        return 2 * precision * recall / (precision + recall)

    def build_prompt_ids(example):
        user_prompt = prompt_template.format(
            context=example["context"],
            input=example["input"],
        )

        if hasattr(tokenizer, "apply_chat_template"):
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            rendered = user_prompt

        token_ids = tokenizer(
            rendered,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids[0]

        original_tokens = int(token_ids.numel())
        if original_tokens > max_input_tokens:
            first_count = max_input_tokens // 2
            last_count = max_input_tokens - first_count
            token_ids = torch.cat(
                [token_ids[:first_count], token_ids[-last_count:]],
                dim=0,
            )

        return (
            token_ids.unsqueeze(0).to(device),
            original_tokens,
            int(token_ids.numel()),
        )

    def cache_to_tuple(cache):
        if isinstance(cache, tuple):
            return cache
        if isinstance(cache, list):
            return tuple(cache)
        if hasattr(cache, "to_legacy_cache"):
            return cache.to_legacy_cache()
        if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
            return tuple(
                (k, v)
                for k, v in zip(cache.key_cache, cache.value_cache)
            )
        raise RuntimeError(f"Unsupported cache type: {type(cache)}")

    def tuple_to_dynamic_cache(kv_tuple):
        cache = DynamicCache()
        for layer_idx, (k, v) in enumerate(kv_tuple):
            cache.update(k, v, layer_idx)
        return cache

    def config_for_method(method):
        if method == "rabit4":
            return {
                "k_bits": 4,
                "v_bits": 4,
                "k_style": "group_sym",
                "v_style": "group_sym",
                "k_group": 64,
                "v_group": 64,
                "residual": 0,
            }
        if method == "rabit3":
            return {
                "k_bits": 3,
                "v_bits": 3,
                "k_style": "group_sym",
                "v_style": "group_sym",
                "k_group": 32,
                "v_group": 32,
                "residual": 4,
            }
        if method == "rabit2":
            return {
                "k_bits": 3,
                "v_bits": 2,
                "k_style": "seq_affine",
                "v_style": "group_affine",
                "k_group": 32,
                "v_group": 32,
                "residual": 4,
            }
        raise ValueError(f"No RABIT-KV configuration for {method}.")

    # Fake-quant path: preserve quantization math but skip physical bit packing.
    def q_group_sym(tensor, bits, group_size):
        x = tensor.detach().float()
        orig_shape = tuple(x.shape)
        pad = (-orig_shape[-1]) % group_size
        if pad:
            x = F.pad(x, (0, pad))
        grouped = x.reshape(*x.shape[:-1], -1, group_size)
        levels = 2 ** bits
        absmax = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        codes = torch.round(
            ((grouped / absmax) + 1.0) * 0.5 * (levels - 1)
        ).clamp(0, levels - 1)
        return {
            "type": "int",
            "style": "group_sym",
            "bits": bits,
            "codes": codes.to(torch.uint8),
            "orig_shape": orig_shape,
            "pad": pad,
            "absmax": absmax.to(dtype),
        }

    def q_group_affine(tensor, bits, group_size):
        x = tensor.detach().float()
        orig_shape = tuple(x.shape)
        pad = (-orig_shape[-1]) % group_size
        if pad:
            x = F.pad(x, (0, pad))
        grouped = x.reshape(*x.shape[:-1], -1, group_size)
        q_min = grouped.amin(dim=-1, keepdim=True)
        q_max = grouped.amax(dim=-1, keepdim=True)
        levels = 2 ** bits
        scale = (q_max - q_min) / (levels - 1)
        scale = torch.where(
            scale.abs() < 1e-8,
            torch.ones_like(scale),
            scale,
        )
        codes = torch.round(
            (grouped - q_min) / scale
        ).clamp(0, levels - 1)
        return {
            "type": "int",
            "style": "group_affine",
            "bits": bits,
            "codes": codes.to(torch.uint8),
            "orig_shape": orig_shape,
            "pad": pad,
            "min": q_min.to(dtype),
            "scale": scale.to(dtype),
        }

    def q_seq_affine(tensor, bits, seq_group):
        if tensor.dim() < 4:
            return q_group_affine(tensor, bits, seq_group)

        x = tensor.detach().float()
        orig_shape = tuple(x.shape)
        pad_seq = (-orig_shape[-2]) % seq_group
        if pad_seq:
            x = F.pad(x, (0, 0, 0, pad_seq))

        batch, heads, seq_len, head_dim = x.shape
        grouped = x.reshape(
            batch,
            heads,
            seq_len // seq_group,
            seq_group,
            head_dim,
        )
        q_min = grouped.amin(dim=3, keepdim=True)
        q_max = grouped.amax(dim=3, keepdim=True)
        levels = 2 ** bits
        scale = (q_max - q_min) / (levels - 1)
        scale = torch.where(
            scale.abs() < 1e-8,
            torch.ones_like(scale),
            scale,
        )
        codes = torch.round(
            (grouped - q_min) / scale
        ).clamp(0, levels - 1)
        return {
            "type": "int",
            "style": "seq_affine",
            "bits": bits,
            "codes": codes.to(torch.uint8),
            "orig_shape": orig_shape,
            "pad_seq": pad_seq,
            "seq_group": seq_group,
            "min": q_min.to(dtype),
            "scale": scale.to(dtype),
        }

    def q_tensor(tensor, bits, side, config):
        style = config[f"{side}_style"]
        group_size = config[f"{side}_group"]

        if style == "group_sym":
            return q_group_sym(tensor, bits, group_size)
        if style == "group_affine":
            return q_group_affine(tensor, bits, group_size)
        if style == "seq_affine":
            return q_seq_affine(tensor, bits, group_size)
        raise ValueError(f"Unsupported quantization style: {style}")

    def q_with_residual(tensor, bits, side, config):
        residual = min(
            int(config.get("residual", 0)),
            int(tensor.shape[-2]),
        )
        if residual <= 0:
            return q_tensor(tensor, bits, side, config)
        if tensor.shape[-2] <= residual:
            return {
                "type": "bf16",
                "data": tensor.detach().to(dtype),
            }

        return {
            "type": "split",
            "old": q_tensor(
                tensor[..., :-residual, :],
                bits,
                side,
                config,
            ),
            "recent": {
                "type": "bf16",
                "data": tensor[..., -residual:, :].detach().to(dtype),
            },
        }

    def dequantize_state(state):
        if state["type"] == "bf16":
            return state["data"].to(dtype)
        if state["type"] == "split":
            return torch.cat(
                [
                    dequantize_state(state["old"]),
                    dequantize_state(state["recent"]),
                ],
                dim=-2,
            )

        codes = state["codes"].float()

        if state["style"] == "group_sym":
            levels = 2 ** state["bits"]
            x = (
                (codes / (levels - 1)) * 2.0 - 1.0
            ) * state["absmax"].float()
            x = x.reshape(*state["orig_shape"][:-1], -1)
            if state["pad"]:
                x = x[..., : state["orig_shape"][-1]]
            return x.to(dtype)

        if state["style"] == "group_affine":
            x = (
                codes * state["scale"].float()
                + state["min"].float()
            )
            x = x.reshape(*state["orig_shape"][:-1], -1)
            if state["pad"]:
                x = x[..., : state["orig_shape"][-1]]
            return x.to(dtype)

        if state["style"] == "seq_affine":
            batch, heads, original_seq, head_dim = state["orig_shape"]
            padded_seq = original_seq + int(state["pad_seq"])
            seq_group = int(state["seq_group"])
            x = (
                codes * state["scale"].float()
                + state["min"].float()
            )
            x = x.reshape(
                batch,
                heads,
                padded_seq // seq_group,
                seq_group,
                head_dim,
            )
            x = x.reshape(batch, heads, padded_seq, head_dim)
            if state["pad_seq"]:
                x = x[..., :original_seq, :]
            return x.to(dtype)

        raise ValueError(f"Unsupported quantized state: {state['style']}")

    def quantize_then_dequantize_cache(cache, config):
        rebuilt_layers = []
        for key, value in cache_to_tuple(cache):
            quantized_key = q_with_residual(
                key,
                config["k_bits"],
                "k",
                config,
            )
            quantized_value = q_with_residual(
                value,
                config["v_bits"],
                "v",
                config,
            )
            rebuilt_layers.append(
                (
                    dequantize_state(quantized_key),
                    dequantize_state(quantized_value),
                )
            )
        return tuple_to_dynamic_cache(tuple(rebuilt_layers))


    def generate_answer(method, prompt_ids):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        prefix_ids = prompt_ids[:, :-1]
        final_prompt_token = prompt_ids[:, -1:]

        start = time.perf_counter()
        with torch.inference_mode():
            prefill_output = model(
                input_ids=prefix_ids,
                use_cache=True,
            )
            prefix_cache = prefill_output.past_key_values

            if method == "bf16":
                cache = prefix_cache
            else:
                cache = quantize_then_dequantize_cache(
                    prefix_cache,
                    config_for_method(method),
                )

            step_output = model(
                input_ids=final_prompt_token,
                past_key_values=cache,
                use_cache=True,
            )
            cache = step_output.past_key_values
            next_token = torch.argmax(
                step_output.logits[:, -1, :],
                dim=-1,
                keepdim=True,
            )

            generated = []
            eos_id = tokenizer.eos_token_id

            for _ in range(max_new_tokens):
                token_id = int(next_token.item())
                if eos_id is not None and token_id == eos_id:
                    break
                generated.append(token_id)

                step_output = model(
                    input_ids=next_token,
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = step_output.past_key_values
                next_token = torch.argmax(
                    step_output.logits[:, -1, :],
                    dim=-1,
                    keepdim=True,
                )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        answer = tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()

        del prefill_output, prefix_cache, cache, step_output, next_token
        gc.collect()
        torch.cuda.empty_cache()

        return answer, elapsed

    # Short warm-up so the first reported method does not absorb initialization.
    warmup_ids = torch.tensor(
        [[tokenizer.bos_token_id or 1, tokenizer.eos_token_id or 2]],
        dtype=torch.long,
        device=device,
    )
    with torch.inference_mode():
        _ = model(input_ids=warmup_ids, use_cache=True)
    torch.cuda.synchronize()

    totals = {method: 0.0 for method in requested}
    rows = []

    selected_indices = filtered_indices[sample_start:stop]

    for local_index, dataset_index in enumerate(
        selected_indices,
        start=1,
    ):
        example = dataset[dataset_index]
        answers = normalize_answers(example["answers"])
        prompt_ids, original_tokens, used_tokens = build_prompt_ids(example)

        print(
            f"Sample {local_index}/{samples} "
            f"(dataset index {dataset_index}): "
            f"{original_tokens} original tokens, "
            f"{used_tokens} used tokens"
        )
        print(f"Ground truth: {answers}")

        for method in requested:
            prediction, elapsed = generate_answer(method, prompt_ids)
            score = max(
                qa_f1_score(prediction, answer)
                for answer in answers
            )
            totals[method] += score
            rows.append(
                {
                    "sample": dataset_index,
                    "method": method,
                    "prediction": prediction,
                    "score": score,
                    "seconds": elapsed,
                    "used_tokens": used_tokens,
                }
            )
            print(
                f"  {method:<8} score={score:.3f} "
                f"answer={prediction!r}"
            )
        print()

    print("=" * 98)
    print("LongBench-E HotpotQA Results")
    print("=" * 98)
    print(
        f"{'Method':<14}"
        f"{'F1 %':<14}"
        f"{'F1 total':<20}"
        f"{'Samples':<10}"
    )
    print("-" * 98)
    for method in requested:
        average = totals[method] / samples
        print(
            f"{method:<14}"
            f"{average * 100:<14.1f}"
            f"{totals[method]:<20.2f}"
            f"{samples:<10}"
        )

    print()
    print(
        "Important: the reported run time is only for experiment planning. "
        "It is not deployment latency."
    )

    return {
        "settings": {
            "model": model_name,
            "dataset": "hotpotqa_e",
            "sample_start": sample_start,
            "samples": samples,
            "length_bucket": length_bucket,
            "max_input_tokens": max_input_tokens,
            "max_new_tokens": max_new_tokens,
            "methods": requested,
        },
        "rows": rows,
        "f1": {
            method: totals[method] / samples
            for method in requested
        },
    }


@app.local_entrypoint()
def main(
    model_name: str = "LLM-Research/Meta-Llama-3.1-8B-Instruct",
    sample_start: int = 0,
    samples: int = 1,
    length_bucket: str = "8k+",
    max_input_tokens: int = 16384,
    max_new_tokens: int = 32,
    methods: str = "bf16,rabit4,rabit2",
):
    run_longbench.remote(
        model_name=model_name,
        sample_start=sample_start,
        samples=samples,
        length_bucket=length_bucket,
        max_input_tokens=max_input_tokens,
        max_new_tokens=max_new_tokens,
        methods=methods,
    )
