"""
RABIT-KV Long-Context Continuation Perplexity Evaluator
========================================================

Quality-only benchmark for the latest RABIT-KV operating points.

For each independent WikiText-2 sample, this script:
1. Prefills a long context once.
2. Quantizes/dequantizes the prefix KV cache once.
3. Scores a fixed continuation with teacher forcing.
4. Reports aggregate continuation PPL and logical prefix-KV memory.

The quantization policies match the selected all-bit PPL-memory search:
- 8-bit: META8g256, symmetric G128, R0
- 4-bit: META8g64, symmetric G128, R0
- 3-bit: META8g64, symmetric G32, R2
- 2-bit target: META8g64, K3/V2 affine G32, R4

This is NOT deployment latency and does NOT measure physically allocated GPU
memory. Logical memory counts packed payload bits, metadata, and residual BF16
storage.
"""

import math
import os
import statistics
import time

import modal

app = modal.App("rabit-kv-continuation-ppl-latest")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "transformers==4.48.2",
    "accelerate",
    "requests",
    "sentencepiece",
    "modelscope",
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
def run_quality(
    model_name: str = "LLM-Research/Meta-Llama-3.1-8B-Instruct",
    context_tokens: int = 1024,
    eval_tokens: int = 128,
    samples: int = 8,
    methods: str = "bf16,rabit8,rabit4,rabit3,rabit2",
):
    import gc

    import requests
    import torch
    import torch.nn.functional as F
    from modelscope import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache

    dtype = torch.bfloat16
    device = torch.device("cuda")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")
    if context_tokens < 16:
        raise ValueError("context_tokens must be at least 16.")
    if eval_tokens < 2:
        raise ValueError("eval_tokens must be at least 2.")
    if samples < 1:
        raise ValueError("samples must be at least 1.")

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
    print("Dataset: WikiText-2 test")
    print(f"Context tokens per sample: {context_tokens}")
    print(f"Continuation tokens per sample: {eval_tokens}")
    print(f"Independent samples: {samples}")
    print(f"Total scored continuation tokens per method: {samples * eval_tokens}")
    print(f"Methods: {requested}")
    print("Metric: teacher-forced continuation perplexity")
    print("Mode: quality-only prefix-cache quantization with latest all-bit policies")
    print("Memory: logical packed prefix-KV storage, including metadata and residuals")
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

    # Match the earlier RABIT-KV WikiText-2 preprocessing.
    url = (
        "https://raw.githubusercontent.com/pytorch/examples/main/"
        "word_language_model/data/wikitext-2/test.txt"
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    lines = [
        line.strip()
        for line in response.text.splitlines()
        if line.strip()
    ]

    needed = samples * (context_tokens + eval_tokens)
    token_ids = []
    for start in range(0, len(lines), 64):
        text = "\n".join(lines[start : start + 64])
        token_ids.extend(
            tokenizer(
                text,
                add_special_tokens=False,
            )["input_ids"]
        )
        if len(token_ids) >= needed:
            break
    if len(token_ids) < needed:
        raise RuntimeError(
            f"Need {needed} WikiText tokens, found {len(token_ids)}."
        )

    pool = torch.tensor(
        token_ids[:needed],
        dtype=torch.long,
    )
    sample_tensors = []
    span = context_tokens + eval_tokens
    for sample_index in range(samples):
        sequence = pool[
            sample_index * span : (sample_index + 1) * span
        ].unsqueeze(0).to(device)
        sample_tensors.append(
            (
                sequence[:, :context_tokens],
                sequence[:, context_tokens:],
            )
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
                (key, value)
                for key, value in zip(cache.key_cache, cache.value_cache)
            )
        raise RuntimeError(f"Unsupported cache type: {type(cache)}")

    def tuple_to_dynamic_cache(kv_tuple):
        cache = DynamicCache()
        for layer_index, (key, value) in enumerate(kv_tuple):
            cache.update(key, value, layer_index)
        return cache

    def tensor_bytes(tensor):
        return int(tensor.numel() * tensor.element_size())

    def encode_metadata(tensor, config):
        mode = str(config.get("metadata_mode", "bf16")).lower()
        data = tensor.detach().float().contiguous()
        if mode == "bf16":
            return {
                "meta_type": "bf16",
                "data": data.to(dtype).contiguous(),
            }
        if mode not in {"int8", "uint8"}:
            raise ValueError(f"Unsupported metadata_mode: {mode}")

        group_size = max(
            8,
            int(config.get("metadata_group_size", 64)),
        )
        original_shape = tuple(data.shape)
        flat = data.reshape(-1)
        pad = (-flat.numel()) % group_size
        if pad:
            flat = torch.cat(
                [flat, flat[-1:].expand(pad)],
                dim=0,
            )
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
            "orig_shape": original_shape,
            "pad": int(pad),
            "group_size": int(group_size),
        }

    def decode_metadata(metadata):
        if torch.is_tensor(metadata):
            return metadata.float()
        metadata_type = metadata.get("meta_type", "bf16")
        if metadata_type == "bf16":
            return metadata["data"].float()
        if metadata_type == "uint8_group":
            values = (
                metadata["codes"].float()
                * metadata["scale"].float()
                + metadata["min"].float()
            )
            flat = values.reshape(-1)
            if metadata.get("pad", 0):
                flat = flat[:-int(metadata["pad"])]
            return flat.reshape(metadata["orig_shape"])
        raise ValueError(
            f"Unsupported metadata representation: {metadata_type}"
        )

    def metadata_bytes(metadata):
        if torch.is_tensor(metadata):
            return tensor_bytes(metadata)
        metadata_type = metadata.get("meta_type", "bf16")
        if metadata_type == "bf16":
            return tensor_bytes(metadata["data"])
        if metadata_type == "uint8_group":
            return (
                tensor_bytes(metadata["codes"])
                + tensor_bytes(metadata["min"])
                + tensor_bytes(metadata["scale"])
            )
        raise ValueError(
            f"Unsupported metadata representation: {metadata_type}"
        )

    def config_for_method(method):
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

    def q_group_sym(tensor, bits, group_size, config):
        values = tensor.detach().float()
        original_shape = tuple(values.shape)
        pad = (-original_shape[-1]) % group_size
        if pad:
            values = F.pad(values, (0, pad))
        grouped = values.reshape(
            *values.shape[:-1],
            -1,
            group_size,
        )
        levels = 2 ** bits
        absmax = grouped.abs().amax(
            dim=-1,
            keepdim=True,
        ).clamp(min=1e-8)
        codes = torch.round(
            ((grouped / absmax) + 1.0)
            * 0.5
            * (levels - 1)
        ).clamp(0, levels - 1)
        return {
            "type": "int",
            "style": "group_sym",
            "bits": bits,
            "codes": codes.to(torch.uint8),
            "orig_shape": original_shape,
            "pad": pad,
            "absmax": encode_metadata(absmax, config),
        }

    def q_group_affine(tensor, bits, group_size, config):
        values = tensor.detach().float()
        original_shape = tuple(values.shape)
        pad = (-original_shape[-1]) % group_size
        if pad:
            values = F.pad(values, (0, pad))
        grouped = values.reshape(
            *values.shape[:-1],
            -1,
            group_size,
        )
        quant_min = grouped.amin(dim=-1, keepdim=True)
        quant_max = grouped.amax(dim=-1, keepdim=True)
        levels = 2 ** bits
        scale = (quant_max - quant_min) / (levels - 1)
        scale = torch.where(
            scale.abs() < 1e-8,
            torch.ones_like(scale),
            scale,
        )
        codes = torch.round(
            (grouped - quant_min) / scale
        ).clamp(0, levels - 1)
        return {
            "type": "int",
            "style": "group_affine",
            "bits": bits,
            "codes": codes.to(torch.uint8),
            "orig_shape": original_shape,
            "pad": pad,
            "min": encode_metadata(quant_min, config),
            "scale": encode_metadata(scale, config),
        }

    def q_seq_affine(tensor, bits, sequence_group, config):
        if tensor.dim() < 4:
            return q_group_affine(
                tensor,
                bits,
                sequence_group,
                config,
            )

        values = tensor.detach().float()
        original_shape = tuple(values.shape)
        pad_sequence = (-original_shape[-2]) % sequence_group
        if pad_sequence:
            values = F.pad(
                values,
                (0, 0, 0, pad_sequence),
            )

        batch, heads, sequence_length, head_dimension = values.shape
        grouped = values.reshape(
            batch,
            heads,
            sequence_length // sequence_group,
            sequence_group,
            head_dimension,
        )
        quant_min = grouped.amin(dim=3, keepdim=True)
        quant_max = grouped.amax(dim=3, keepdim=True)
        levels = 2 ** bits
        scale = (quant_max - quant_min) / (levels - 1)
        scale = torch.where(
            scale.abs() < 1e-8,
            torch.ones_like(scale),
            scale,
        )
        codes = torch.round(
            (grouped - quant_min) / scale
        ).clamp(0, levels - 1)
        return {
            "type": "int",
            "style": "seq_affine",
            "bits": bits,
            "codes": codes.to(torch.uint8),
            "orig_shape": original_shape,
            "pad_seq": pad_sequence,
            "seq_group": sequence_group,
            "min": encode_metadata(quant_min, config),
            "scale": encode_metadata(scale, config),
        }

    def q_tensor(tensor, bits, side, config):
        style = config[f"{side}_style"]
        group_size = config[f"{side}_group"]
        if style == "group_sym":
            return q_group_sym(
                tensor,
                bits,
                group_size,
                config,
            )
        if style == "group_affine":
            return q_group_affine(
                tensor,
                bits,
                group_size,
                config,
            )
        if style == "seq_affine":
            return q_seq_affine(
                tensor,
                bits,
                group_size,
                config,
            )
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
            values = (
                (codes / (levels - 1)) * 2.0 - 1.0
            ) * decode_metadata(state["absmax"])
            values = values.reshape(
                *state["orig_shape"][:-1],
                -1,
            )
            if state["pad"]:
                values = values[..., : state["orig_shape"][-1]]
            return values.to(dtype)

        if state["style"] == "group_affine":
            values = (
                codes * decode_metadata(state["scale"])
                + decode_metadata(state["min"])
            )
            values = values.reshape(
                *state["orig_shape"][:-1],
                -1,
            )
            if state["pad"]:
                values = values[..., : state["orig_shape"][-1]]
            return values.to(dtype)

        if state["style"] == "seq_affine":
            batch, heads, original_sequence, head_dimension = state[
                "orig_shape"
            ]
            padded_sequence = original_sequence + int(state["pad_seq"])
            sequence_group = int(state["seq_group"])
            values = (
                codes * decode_metadata(state["scale"])
                + decode_metadata(state["min"])
            )
            values = values.reshape(
                batch,
                heads,
                padded_sequence // sequence_group,
                sequence_group,
                head_dimension,
            )
            values = values.reshape(
                batch,
                heads,
                padded_sequence,
                head_dimension,
            )
            if state["pad_seq"]:
                values = values[..., :original_sequence, :]
            return values.to(dtype)

        raise ValueError(
            f"Unsupported quantized state: {state['style']}"
        )

    def stored_state_logical_bytes(state):
        if state["type"] == "bf16":
            return tensor_bytes(state["data"])
        if state["type"] == "split":
            return (
                stored_state_logical_bytes(state["old"])
                + stored_state_logical_bytes(state["recent"])
            )
        payload = (
            int(state["codes"].numel()) * int(state["bits"]) + 7
        ) // 8
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

    def evaluate_one(context_ids, continuation_ids, method):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.inference_mode():
            prefill = model(
                input_ids=context_ids,
                use_cache=True,
            )
            first_logit = prefill.logits[:, -1:, :]
            prefix_cache = prefill.past_key_values

            if method == "bf16":
                cache_for_evaluation = prefix_cache
                logical_kv_mb = (
                    bf16_cache_bytes(prefix_cache) / (1024 ** 2)
                )
            else:
                (
                    cache_for_evaluation,
                    logical_kv_mb,
                ) = quantize_then_dequantize_cache(
                    prefix_cache,
                    config_for_method(method),
                )

            continuation_output = model(
                input_ids=continuation_ids[:, :-1],
                past_key_values=cache_for_evaluation,
                use_cache=False,
            )
            logits = torch.cat(
                [first_logit, continuation_output.logits],
                dim=1,
            )
            loss_sum = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                continuation_ids.reshape(-1),
                reduction="sum",
            )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        token_count = int(continuation_ids.numel())
        loss_sum_value = float(loss_sum.item())
        mean_loss = loss_sum_value / token_count

        del (
            prefill,
            prefix_cache,
            cache_for_evaluation,
            continuation_output,
            logits,
            loss_sum,
        )
        gc.collect()
        torch.cuda.empty_cache()

        return {
            "loss_sum": loss_sum_value,
            "tokens": token_count,
            "mean_loss": mean_loss,
            "ppl": math.exp(min(mean_loss, 50.0)),
            "logical_kv_mb": logical_kv_mb,
            "seconds": elapsed,
        }

    print("Configurations:")
    for method in requested:
        if method == "bf16":
            print("  bf16: BF16 baseline")
        else:
            print(f"  {method}: {config_for_method(method)['name']}")
    print()

    # Untimed warm-up.
    warmup_ids = sample_tensors[0][0][:, :32]
    with torch.inference_mode():
        _ = model(input_ids=warmup_ids, use_cache=True)
    torch.cuda.synchronize()

    all_results = []
    for method in requested:
        print(f"Running {method}...")
        method_rows = []
        for sample_number, (context_ids, continuation_ids) in enumerate(
            sample_tensors,
            start=1,
        ):
            row = evaluate_one(
                context_ids,
                continuation_ids,
                method,
            )
            method_rows.append(row)
            print(
                f"  sample {sample_number}/{samples}: "
                f"PPL={row['ppl']:.4f}, "
                f"KV={row['logical_kv_mb']:.3f} MB, "
                f"quality-run time={row['seconds']:.2f}s"
            )

        total_loss_sum = sum(row["loss_sum"] for row in method_rows)
        total_tokens = sum(row["tokens"] for row in method_rows)
        aggregate_loss = total_loss_sum / total_tokens
        aggregate_ppl = math.exp(min(aggregate_loss, 50.0))
        median_sample_ppl = statistics.median(
            row["ppl"] for row in method_rows
        )
        average_kv_mb = statistics.mean(
            row["logical_kv_mb"] for row in method_rows
        )
        median_seconds = statistics.median(
            row["seconds"] for row in method_rows
        )
        all_results.append(
            {
                "method": method,
                "ppl": aggregate_ppl,
                "loss": aggregate_loss,
                "median_sample_ppl": median_sample_ppl,
                "tokens": total_tokens,
                "avg_logical_kv_mb": average_kv_mb,
                "seconds": median_seconds,
            }
        )
        print()

    baseline = next(
        result
        for result in all_results
        if result["method"] == "bf16"
    )

    print("=" * 116)
    print("RABIT-KV LONG-CONTEXT CONTINUATION PERPLEXITY")
    print("=" * 116)
    print(
        f"{'Method':<12}"
        f"{'PPL':<12}"
        f"{'PPL d%':<12}"
        f"{'Median sample PPL':<20}"
        f"{'Avg KV MB':<14}"
        f"{'Compression':<14}"
        f"{'Tokens':<10}"
    )
    print("-" * 116)
    for result in all_results:
        ppl_delta = (
            (result["ppl"] / baseline["ppl"]) - 1.0
        ) * 100.0
        compression = (
            baseline["avg_logical_kv_mb"]
            / result["avg_logical_kv_mb"]
        )
        print(
            f"{result['method']:<12}"
            f"{result['ppl']:<12.4f}"
            f"{ppl_delta:<12.2f}"
            f"{result['median_sample_ppl']:<20.4f}"
            f"{result['avg_logical_kv_mb']:<14.3f}"
            f"{compression:<14.3f}"
            f"{result['tokens']:<10}"
        )

    print()
    print(
        "Important: quality-run time is only for experiment planning. "
        "It is not deployment latency."
    )

    return {
        "settings": {
            "model": model_name,
            "context_tokens": context_tokens,
            "eval_tokens": eval_tokens,
            "samples": samples,
            "methods": requested,
            "total_scored_tokens_per_method": samples * eval_tokens,
        },
        "results": all_results,
    }


@app.local_entrypoint()
def main(
    model_name: str = "LLM-Research/Meta-Llama-3.1-8B-Instruct",
    context_tokens: int = 1024,
    eval_tokens: int = 128,
    samples: int = 8,
    methods: str = "bf16,rabit8,rabit4,rabit3,rabit2",
):
    run_quality.remote(
        model_name=model_name,
        context_tokens=context_tokens,
        eval_tokens=eval_tokens,
        samples=samples,
        methods=methods,
    )
