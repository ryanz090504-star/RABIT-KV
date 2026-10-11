"""
RABIT-KV Updated LongBench Qasper Evaluator
====================================================

Quality-only benchmark using the official LongBench-E Qasper task.

This script:
- Loads one or more official LongBench-E Qasper samples.
- Evaluates BF16 plus the latest 8/4/3/2-bit RABIT-KV operating points.
- Applies grouped UINT8 metadata quantization selected by the PPL-memory search.
- Quantizes/dequantizes the long prefix KV cache once before answer generation.
- Reports token-level F1 and logical prefix-KV memory for every method.

It is NOT a deployment latency or real packed-memory benchmark.
"""

import os
import time

import modal

app = modal.App("rabit-kv-longbench-qasper-latest")

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
    methods: str = "bf16,rabit8,rabit4,rabit3,rabit2",
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
    allowed = {"bf16", "rabit8", "rabit4", "rabit3", "rabit2"}
    invalid = [item for item in requested if item not in allowed]
    if invalid:
        raise ValueError(
            f"Unsupported methods: {invalid}. "
            "Use bf16,rabit8,rabit4,rabit3,rabit2."
        )
    if "bf16" not in requested:
        requested.insert(0, "bf16")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {model_name}")
    print("Dataset: LongBench-E qasper_e")
    print(f"Length bucket: {length_bucket}")
    print(f"Samples: {samples}, starting at filtered index {sample_start}")
    print(f"Maximum input length: {max_input_tokens} tokens")
    print(f"Methods: {requested}")
    print("Metric: official LongBench QA token-level F1")
    print("Mode: quality-only prefix-cache quantization with latest all-bit policies")
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
        revision="52edb9d18ea01d49ec7580fefcfe6b9d97a0fa96",
        filename="qasper_e/test-00000-of-00001.parquet",
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
        "You are given a scientific article and a question. "
        "Answer the question as concisely as you can, using a single phrase "
        "or sentence if possible. If the question cannot be answered based "
        "on the information in the article, write \"unanswerable\". "
        "If the question is a yes/no question, answer \"yes\", \"no\", "
        "or \"unanswerable\". Do not provide any explanation.\n\n"
        "Article: {context}\n\n"
        "Answer the question based on the above article as concisely as you "
        "can, using a single phrase or sentence if possible. If the question "
        "cannot be answered based on the information in the article, write "
        "\"unanswerable\". If the question is a yes/no question, answer "
        "\"yes\", \"no\", or \"unanswerable\". Do not provide any "
        "explanation.\n\n"
        "Question: {input}\n\n"
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

    def encode_metadata(tensor, config):
        """Store metadata as BF16 or grouped UINT8, matching the PPL search."""
        mode = str(config.get("metadata_mode", "bf16")).lower()
        data = tensor.detach().float().contiguous()
        if mode == "bf16":
            return {
                "meta_type": "bf16",
                "data": data.to(dtype).contiguous(),
            }
        if mode not in {"int8", "uint8"}:
            raise ValueError(f"Unsupported metadata_mode: {mode}")

        group_size = max(8, int(config.get("metadata_group_size", 64)))
        orig_shape = tuple(data.shape)
        flat = data.reshape(-1)
        pad = (-flat.numel()) % group_size
        if pad:
            flat = torch.cat([flat, flat[-1:].expand(pad)], dim=0)
        grouped = flat.reshape(-1, group_size)
        meta_min = grouped.amin(dim=-1, keepdim=True)
        meta_max = grouped.amax(dim=-1, keepdim=True)
        meta_scale = (meta_max - meta_min) / 255.0
        meta_scale = torch.where(
            meta_scale.abs() < 1e-12,
            torch.ones_like(meta_scale),
            meta_scale,
        )
        codes = torch.round(
            (grouped - meta_min) / meta_scale
        ).clamp(0, 255).to(torch.uint8)
        return {
            "meta_type": "uint8_group",
            "codes": codes.contiguous(),
            "min": meta_min.to(dtype).contiguous(),
            "scale": meta_scale.to(dtype).contiguous(),
            "orig_shape": orig_shape,
            "pad": int(pad),
            "group_size": int(group_size),
        }

    def decode_metadata(meta):
        if torch.is_tensor(meta):
            return meta.float()
        meta_type = meta.get("meta_type", "bf16")
        if meta_type == "bf16":
            return meta["data"].float()
        if meta_type == "uint8_group":
            values = (
                meta["codes"].float() * meta["scale"].float()
                + meta["min"].float()
            )
            flat = values.reshape(-1)
            if meta.get("pad", 0):
                flat = flat[:-int(meta["pad"])]
            return flat.reshape(meta["orig_shape"])
        raise ValueError(f"Unsupported metadata representation: {meta_type}")

    def tensor_bytes(tensor):
        return int(tensor.numel() * tensor.element_size())

    def metadata_bytes(meta):
        if torch.is_tensor(meta):
            return tensor_bytes(meta)
        meta_type = meta.get("meta_type", "bf16")
        if meta_type == "bf16":
            return tensor_bytes(meta["data"])
        if meta_type == "uint8_group":
            return (
                tensor_bytes(meta["codes"])
                + tensor_bytes(meta["min"])
                + tensor_bytes(meta["scale"])
            )
        raise ValueError(f"Unsupported metadata representation: {meta_type}")

    def config_for_method(method):
        # Winners from the all-bit PPL-memory search.
        if method == "rabit8":
            return {
                "name": "8b META8g256 G128 R0",
                "k_bits": 8,
                "v_bits": 8,
                "k_style": "group_sym",
                "v_style": "group_sym",
                "k_group": 128,
                "v_group": 128,
                "residual": 0,
                "metadata_mode": "int8",
                "metadata_group_size": 256,
            }
        if method == "rabit4":
            return {
                "name": "4b META8g64 SYM G128 R0",
                "k_bits": 4,
                "v_bits": 4,
                "k_style": "group_sym",
                "v_style": "group_sym",
                "k_group": 128,
                "v_group": 128,
                "residual": 0,
                "metadata_mode": "int8",
                "metadata_group_size": 64,
            }
        if method == "rabit3":
            return {
                "name": "3b META8g64 SYM G32 R2",
                "k_bits": 3,
                "v_bits": 3,
                "k_style": "group_sym",
                "v_style": "group_sym",
                "k_group": 32,
                "v_group": 32,
                "residual": 2,
                "metadata_mode": "int8",
                "metadata_group_size": 64,
            }
        if method == "rabit2":
            return {
                "name": "2b META8g64 K3V2 G32 R4",
                "k_bits": 3,
                "v_bits": 2,
                "k_style": "seq_affine",
                "v_style": "group_affine",
                "k_group": 32,
                "v_group": 32,
                "residual": 4,
                "metadata_mode": "int8",
                "metadata_group_size": 64,
            }
        raise ValueError(f"No RABIT-KV configuration for {method}.")

    # Fake-quant path: preserve quantization math but skip physical bit packing.
    def q_group_sym(tensor, bits, group_size, config):
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
            "absmax": encode_metadata(absmax, config),
        }

    def q_group_affine(tensor, bits, group_size, config):
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
            "min": encode_metadata(q_min, config),
            "scale": encode_metadata(scale, config),
        }

    def q_seq_affine(tensor, bits, seq_group, config):
        if tensor.dim() < 4:
            return q_group_affine(tensor, bits, seq_group, config)

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
            "min": encode_metadata(q_min, config),
            "scale": encode_metadata(scale, config),
        }

    def q_tensor(tensor, bits, side, config):
        style = config[f"{side}_style"]
        group_size = config[f"{side}_group"]

        if style == "group_sym":
            return q_group_sym(tensor, bits, group_size, config)
        if style == "group_affine":
            return q_group_affine(tensor, bits, group_size, config)
        if style == "seq_affine":
            return q_seq_affine(tensor, bits, group_size, config)
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
            ) * decode_metadata(state["absmax"])
            x = x.reshape(*state["orig_shape"][:-1], -1)
            if state["pad"]:
                x = x[..., : state["orig_shape"][-1]]
            return x.to(dtype)

        if state["style"] == "group_affine":
            x = (
                codes * decode_metadata(state["scale"])
                + decode_metadata(state["min"])
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
                codes * decode_metadata(state["scale"])
                + decode_metadata(state["min"])
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

    def stored_state_logical_bytes(state):
        if state["type"] == "bf16":
            return tensor_bytes(state["data"])
        if state["type"] == "split":
            return (
                stored_state_logical_bytes(state["old"])
                + stored_state_logical_bytes(state["recent"])
            )
        payload = (int(state["codes"].numel()) * int(state["bits"]) + 7) // 8
        if state["style"] == "group_sym":
            return payload + metadata_bytes(state["absmax"])
        return (
            payload
            + metadata_bytes(state["min"])
            + metadata_bytes(state["scale"])
        )

    def bf16_cache_bytes(cache):
        return sum(
            tensor_bytes(key) + tensor_bytes(value)
            for key, value in cache_to_tuple(cache)
        )

    def quantize_then_dequantize_cache(cache, config):
        rebuilt_layers = []
        logical_bytes = 0
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
            logical_bytes += stored_state_logical_bytes(quantized_key)
            logical_bytes += stored_state_logical_bytes(quantized_value)
            rebuilt_layers.append(
                (
                    dequantize_state(quantized_key),
                    dequantize_state(quantized_value),
                )
            )
        return (
            tuple_to_dynamic_cache(tuple(rebuilt_layers)),
            logical_bytes / (1024 ** 2),
        )


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
                logical_kv_mb = bf16_cache_bytes(prefix_cache) / (1024 ** 2)
            else:
                cache, logical_kv_mb = quantize_then_dequantize_cache(
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

        return answer, elapsed, logical_kv_mb

    print("Configurations:")
    for method in requested:
        if method == "bf16":
            print("  bf16: BF16 baseline")
        else:
            print(f"  {method}: {config_for_method(method)['name']}")
    print()

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
    memory_totals = {method: 0.0 for method in requested}
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
            prediction, elapsed, logical_kv_mb = generate_answer(method, prompt_ids)
            score = max(
                qa_f1_score(prediction, answer)
                for answer in answers
            )
            totals[method] += score
            memory_totals[method] += logical_kv_mb
            rows.append(
                {
                    "sample": dataset_index,
                    "method": method,
                    "prediction": prediction,
                    "score": score,
                    "seconds": elapsed,
                    "logical_prefix_kv_mb": logical_kv_mb,
                    "used_tokens": used_tokens,
                }
            )
            print(
                f"  {method:<8} score={score:.3f} "
                f"KV={logical_kv_mb:.3f} MB "
                f"answer={prediction!r}"
            )
        print()

    print("=" * 98)
    print("LongBench-E Qasper Results")
    print("=" * 98)
    print(
        f"{'Method':<14}"
        f"{'F1 %':<14}"
        f"{'F1 total':<14}"
        f"{'Avg KV MB':<14}"
        f"{'Samples':<10}"
    )
    print("-" * 98)
    for method in requested:
        average = totals[method] / samples
        print(
            f"{method:<14}"
            f"{average * 100:<14.1f}"
            f"{totals[method]:<14.2f}"
            f"{memory_totals[method] / samples:<14.3f}"
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
            "dataset": "qasper_e",
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
        "average_logical_prefix_kv_mb": {
            method: memory_totals[method] / samples
            for method in requested
        },
        "configurations": {
            method: ({"name": "BF16 baseline"} if method == "bf16" else config_for_method(method))
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
    methods: str = "bf16,rabit8,rabit4,rabit3,rabit2",
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
