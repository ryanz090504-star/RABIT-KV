"""
RABIT-KV Multilingual Long-Context Continuation Perplexity Evaluator
====================================================================

Diagnostic quality-only benchmark for Chinese and Spanish using fixed,
revision-pinned Wikimedia Wikipedia subsets.

For each language and independent article sample, this script:
1. Selects a deterministic long article span.
2. Prefills a long context once.
3. Quantizes/dequantizes the prefix KV cache once.
4. Scores a fixed continuation with teacher forcing.
5. Reports continuation PPL and logical prefix-KV memory.

The comparison is intentionally focused on the final Pareto points:
- BF16 baseline
- 8-bit: META8g256, symmetric G128, R0
- 2-bit target: META8g64, K3/V2 affine G32, R4

This is a multilingual diagnostic, not a deployment-latency benchmark. It does
not measure physically allocated GPU memory. Logical memory counts packed
payload bits, metadata, and residual BF16 storage.
"""

import math
import os
import statistics
import time

import modal

app = modal.App("rabit-kv-multilingual-continuation-ppl")

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
def run_quality(
    model_name: str = "LLM-Research/Meta-Llama-3.1-8B-Instruct",
    languages: str = "zh,es",
    context_tokens: int = 1024,
    eval_tokens: int = 128,
    samples: int = 8,
    methods: str = "bf16,rabit8,rabit2",
    dataset_revision: str = "cf584d1dc131caa92a5cb910f41a8b7591b12732",
    shuffle_seed: int = 20260804,
    shuffle_buffer: int = 1000,
):
    import gc

    import hashlib

    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
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
    if shuffle_buffer < 32:
        raise ValueError("shuffle_buffer must be at least 32.")

    language_specs = {
        "zh": {"name": "Chinese", "config": "20231101.zh"},
        "es": {"name": "Spanish", "config": "20231101.es"},
    }
    requested_languages = [
        item.strip().lower()
        for item in languages.split(",")
        if item.strip()
    ]
    if not requested_languages:
        raise ValueError("At least one language must be requested.")
    invalid_languages = [
        item for item in requested_languages if item not in language_specs
    ]
    if invalid_languages:
        raise ValueError(
            f"Unsupported languages: {invalid_languages}. Use zh,es."
        )

    requested = [
        item.strip().lower()
        for item in methods.split(",")
        if item.strip()
    ]
    allowed = {"bf16", "rabit8", "rabit2"}
    invalid = [item for item in requested if item not in allowed]
    if invalid:
        raise ValueError(
            f"Unsupported methods: {invalid}. "
            "Use bf16,rabit8,rabit2."
        )
    if "bf16" not in requested:
        requested.insert(0, "bf16")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {model_name}")
    print("Dataset: Wikimedia Wikipedia 20231101 (revision pinned)")
    print(f"Dataset revision: {dataset_revision}")
    print(f"Languages: {requested_languages}")
    print(f"Context tokens per sample: {context_tokens}")
    print(f"Continuation tokens per sample: {eval_tokens}")
    print(f"Independent article samples per language: {samples}")
    print(f"Scored tokens per language and method: {samples * eval_tokens}")
    print(f"Total scored tokens per method: {len(requested_languages) * samples * eval_tokens}")
    print(f"Methods: {requested}")
    print("Metric: teacher-forced continuation perplexity")
    print("Mode: multilingual quality-only prefix-cache quantization diagnostic")
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

    span = context_tokens + eval_tokens

    def stable_offset(article_id, max_offset, language_code):
        if max_offset <= 0:
            return 0
        digest = hashlib.sha256(
            f"{language_code}:{article_id}:{shuffle_seed}".encode("utf-8")
        ).digest()
        return int.from_bytes(digest[:8], "big") % (max_offset + 1)

    def build_language_samples(language_code):
        spec = language_specs[language_code]
        stream = load_dataset(
            "wikimedia/wikipedia",
            spec["config"],
            split="train",
            streaming=True,
            revision=dataset_revision,
        )
        # Fixed revision + seed + buffer gives deterministic article selection.
        stream = stream.shuffle(
            seed=shuffle_seed + sum(ord(ch) for ch in language_code),
            buffer_size=shuffle_buffer,
        )

        selected = []
        scanned = 0
        for article in stream:
            scanned += 1
            text = str(article.get("text", "")).strip()
            if not text:
                continue
            token_ids = tokenizer(
                text,
                add_special_tokens=False,
            )["input_ids"]
            if len(token_ids) < span:
                continue

            article_id = str(article.get("id", scanned))
            offset = stable_offset(
                article_id,
                len(token_ids) - span,
                language_code,
            )
            sequence = torch.tensor(
                token_ids[offset : offset + span],
                dtype=torch.long,
            ).unsqueeze(0).to(device)
            selected.append(
                {
                    "context": sequence[:, :context_tokens],
                    "continuation": sequence[:, context_tokens:],
                    "article_id": article_id,
                    "title": str(article.get("title", "")),
                    "article_tokens": len(token_ids),
                    "offset": offset,
                }
            )
            if len(selected) >= samples:
                break

        if len(selected) < samples:
            raise RuntimeError(
                f"Only found {len(selected)} sufficiently long {spec['name']} "
                f"articles after scanning {scanned}; need {samples}."
            )

        print(
            f"Selected {len(selected)} {spec['name']} articles "
            f"after scanning {scanned}:"
        )
        for index, row in enumerate(selected, start=1):
            title = row["title"].replace("\n", " ")[:70]
            print(
                f"  {index:02d}. id={row['article_id']} "
                f"tokens={row['article_tokens']} offset={row['offset']} "
                f"title={title!r}"
            )
        print()
        return selected

    language_samples = {
        language_code: build_language_samples(language_code)
        for language_code in requested_languages
    }

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
    first_language = requested_languages[0]
    warmup_ids = language_samples[first_language][0]["context"][:, :32]
    with torch.inference_mode():
        _ = model(input_ids=warmup_ids, use_cache=True)
    torch.cuda.synchronize()

    results_by_language = {}
    for language_code in requested_languages:
        spec = language_specs[language_code]
        print("=" * 124)
        print(f"LANGUAGE: {spec['name']} ({language_code})")
        print("=" * 124)
        language_results = []

        for method in requested:
            print(f"Running {language_code}/{method}...")
            method_rows = []
            for sample_number, sample in enumerate(
                language_samples[language_code],
                start=1,
            ):
                row = evaluate_one(
                    sample["context"],
                    sample["continuation"],
                    method,
                )
                method_rows.append(row)
                title = sample["title"].replace("\n", " ")[:42]
                print(
                    f"  sample {sample_number}/{samples}: "
                    f"PPL={row['ppl']:.4f}, "
                    f"KV={row['logical_kv_mb']:.3f} MB, "
                    f"title={title!r}, "
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
            language_results.append(
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
            result for result in language_results
            if result["method"] == "bf16"
        )
        for result in language_results:
            result["ppl_delta_pct"] = (
                (result["ppl"] / baseline["ppl"]) - 1.0
            ) * 100.0
            result["compression"] = (
                baseline["avg_logical_kv_mb"]
                / result["avg_logical_kv_mb"]
            )

        results_by_language[language_code] = language_results

        print("-" * 124)
        print(f"RABIT-KV MULTILINGUAL CONTINUATION PPL — {spec['name'].upper()}")
        print("-" * 124)
        print(
            f"{'Method':<12}"
            f"{'PPL':<14}"
            f"{'PPL d%':<14}"
            f"{'Median sample PPL':<22}"
            f"{'Avg KV MB':<16}"
            f"{'Compression':<16}"
            f"{'Tokens':<10}"
        )
        print("-" * 124)
        for result in language_results:
            print(
                f"{result['method']:<12}"
                f"{result['ppl']:<14.4f}"
                f"{result['ppl_delta_pct']:<14.2f}"
                f"{result['median_sample_ppl']:<22.4f}"
                f"{result['avg_logical_kv_mb']:<16.3f}"
                f"{result['compression']:<16.3f}"
                f"{result['tokens']:<10}"
            )
        print()

    print("=" * 118)
    print("MULTILINGUAL RELATIVE-QUALITY SUMMARY")
    print("=" * 118)
    language_headers = "".join(
        f"{language_specs[code]['name'] + ' d%':<18}"
        for code in requested_languages
    )
    print(
        f"{'Method':<12}"
        f"{language_headers}"
        f"{'Mean d%':<14}"
        f"{'Worst d%':<14}"
        f"{'Mean comp':<14}"
    )
    print("-" * 118)

    summary_rows = []
    for method in requested:
        method_language_rows = [
            next(
                row for row in results_by_language[code]
                if row["method"] == method
            )
            for code in requested_languages
        ]
        deltas = [row["ppl_delta_pct"] for row in method_language_rows]
        compressions = [row["compression"] for row in method_language_rows]
        summary_row = {
            "method": method,
            "mean_delta_pct": statistics.mean(deltas),
            "worst_delta_pct": max(deltas),
            "mean_compression": statistics.mean(compressions),
            "language_deltas": {
                code: row["ppl_delta_pct"]
                for code, row in zip(
                    requested_languages,
                    method_language_rows,
                )
            },
        }
        summary_rows.append(summary_row)
        delta_columns = "".join(
            f"{summary_row['language_deltas'][code]:<18.2f}"
            for code in requested_languages
        )
        print(
            f"{method:<12}"
            f"{delta_columns}"
            f"{summary_row['mean_delta_pct']:<14.2f}"
            f"{summary_row['worst_delta_pct']:<14.2f}"
            f"{summary_row['mean_compression']:<14.3f}"
        )

    print()
    print(
        "Important: this is a deterministic multilingual diagnostic using "
        "Wikipedia article spans. Quality-run time is not deployment latency."
    )

    return {
        "settings": {
            "model": model_name,
            "dataset": "wikimedia/wikipedia",
            "dataset_revision": dataset_revision,
            "languages": requested_languages,
            "context_tokens": context_tokens,
            "eval_tokens": eval_tokens,
            "samples_per_language": samples,
            "methods": requested,
            "shuffle_seed": shuffle_seed,
            "shuffle_buffer": shuffle_buffer,
            "scored_tokens_per_language_method": samples * eval_tokens,
        },
        "results_by_language": results_by_language,
        "summary": summary_rows,
    }

@app.local_entrypoint()
def main(
    model_name: str = "LLM-Research/Meta-Llama-3.1-8B-Instruct",
    languages: str = "zh,es",
    context_tokens: int = 1024,
    eval_tokens: int = 128,
    samples: int = 8,
    methods: str = "bf16,rabit8,rabit2",
    dataset_revision: str = "cf584d1dc131caa92a5cb910f41a8b7591b12732",
    shuffle_seed: int = 20260804,
    shuffle_buffer: int = 1000,
):
    run_quality.remote(
        model_name=model_name,
        languages=languages,
        context_tokens=context_tokens,
        eval_tokens=eval_tokens,
        samples=samples,
        methods=methods,
        dataset_revision=dataset_revision,
        shuffle_seed=shuffle_seed,
        shuffle_buffer=shuffle_buffer,
    )
