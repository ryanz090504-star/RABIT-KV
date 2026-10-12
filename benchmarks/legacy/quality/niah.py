"""
RABIT-KV Latest Needle-in-a-Haystack Quality Evaluator
=======================================================

Quality-only long-context benchmark:
- Builds deterministic synthetic prompts at 4K, 8K, and 16K by default.
- Inserts one unique secret code at one or more controlled depths.
- Prefills each prompt once.
- Quantizes/dequantizes the prefix KV cache once using the latest all-bit policies.
- Greedily generates a short answer.
- Reports exact-code retrieval and logical compressed prefix-KV memory.

This script is NOT a deployment-latency benchmark and does NOT measure physical
packed GPU allocation. It measures quality and logical packed KV storage only.
"""

import os
import time

import modal

app = modal.App("rabit-kv-niah-latest")

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
def run_niah(
    model_name: str = "LLM-Research/Meta-Llama-3.1-8B-Instruct",
    context_lengths: str = "4096,8192,16384",
    needle_depths: str = "0.5",
    max_new_tokens: int = 16,
    methods: str = "bf16,rabit8,rabit4,rabit3,rabit2",
):
    import gc
    import re

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
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive.")

    parsed_contexts = [
        int(item.strip())
        for item in str(context_lengths).split(",")
        if item.strip()
    ]
    if not parsed_contexts:
        raise ValueError("context_lengths cannot be empty.")
    if any(length < 512 for length in parsed_contexts):
        raise ValueError("Every context length must be at least 512 tokens.")
    parsed_contexts = list(dict.fromkeys(parsed_contexts))

    parsed_depths = [
        float(item.strip())
        for item in str(needle_depths).split(",")
        if item.strip()
    ]
    if not parsed_depths:
        raise ValueError("needle_depths cannot be empty.")
    if any(not 0.0 <= depth <= 1.0 for depth in parsed_depths):
        raise ValueError("Every needle depth must be between 0 and 1.")
    parsed_depths = list(dict.fromkeys(parsed_depths))

    requested = [
        item.strip().lower()
        for item in str(methods).split(",")
        if item.strip()
    ]
    allowed = {"bf16", "rabit8", "rabit4", "rabit3", "rabit2"}
    invalid = [item for item in requested if item not in allowed]
    if invalid:
        raise ValueError(
            f"Unsupported methods: {invalid}. "
            "Use bf16,rabit8,rabit4,rabit3,rabit2."
        )
    # Keep BF16 first so every compression ratio uses the matching baseline.
    requested = [item for item in requested if item != "bf16"]
    requested.insert(0, "bf16")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {model_name}")
    print(f"Context lengths: {parsed_contexts}")
    print(f"Needle depths: {parsed_depths}")
    print(f"Methods: {requested}")
    print("Metric: exact retrieval of the hidden code")
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

    secret_code = "RABIT-7291"
    instruction = (
        "Read the following document carefully. It contains one hidden secret code. "
        "After the document, answer the question using only the exact secret code.\n\n"
        "DOCUMENT START\n"
    )
    needle = (
        "\nIMPORTANT RECORD: The hidden secret code is RABIT-7291. "
        "Remember this exact code.\n"
    )
    question = (
        "\nDOCUMENT END\n\n"
        "Question: What is the hidden secret code?\n"
        "Answer with only the exact code:"
    )

    instruction_ids = tokenizer(
        instruction,
        add_special_tokens=False,
    )["input_ids"]
    needle_ids = tokenizer(
        needle,
        add_special_tokens=False,
    )["input_ids"]
    question_ids = tokenizer(
        question,
        add_special_tokens=False,
    )["input_ids"]

    # WikiText-2 is used only as natural-language distractor text.
    wt2_url = (
        "https://raw.githubusercontent.com/pytorch/examples/main/"
        "word_language_model/data/wikitext-2/test.txt"
    )
    response = requests.get(wt2_url, timeout=30)
    response.raise_for_status()
    filler_text = "\n".join(
        line.strip()
        for line in response.text.splitlines()
        if line.strip()
    )
    base_filler_ids = tokenizer(
        filler_text,
        add_special_tokens=False,
    )["input_ids"]
    if not base_filler_ids:
        raise RuntimeError("WikiText-2 filler tokenization produced no tokens.")

    def build_prompt(context_tokens, needle_depth):
        fixed_tokens = (
            len(instruction_ids)
            + len(needle_ids)
            + len(question_ids)
        )
        filler_needed = int(context_tokens) - fixed_tokens
        if filler_needed < 128:
            raise ValueError(
                f"context_tokens={context_tokens} is too small for the prompt template."
            )

        repeats = (filler_needed + len(base_filler_ids) - 1) // len(base_filler_ids)
        filler_ids = (base_filler_ids * repeats)[:filler_needed]

        before_count = int(round(filler_needed * float(needle_depth)))
        before_count = max(0, min(before_count, filler_needed))

        prompt_ids_list = (
            instruction_ids
            + filler_ids[:before_count]
            + needle_ids
            + filler_ids[before_count:]
            + question_ids
        )
        if len(prompt_ids_list) != int(context_tokens):
            raise RuntimeError(
                f"Prompt length mismatch: got {len(prompt_ids_list)}, "
                f"expected {context_tokens}."
            )

        return torch.tensor(
            prompt_ids_list,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)

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
        for layer_idx, (key, value) in enumerate(kv_tuple):
            cache.update(key, value, layer_idx)
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

        # Keep the final prompt token outside prefill so the first generated token
        # is computed using the quantized prefix cache.
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
        match = re.search(r"RABIT-\d{4}", answer.upper())
        extracted_code = match.group(0) if match else ""
        correct = extracted_code == secret_code

        del prefill_output, prefix_cache, cache, step_output, next_token
        gc.collect()
        torch.cuda.empty_cache()

        return {
            "method": method,
            "answer": answer,
            "extracted_code": extracted_code,
            "correct": bool(correct),
            "logical_kv_mb": float(logical_kv_mb),
            "seconds": float(elapsed),
        }

    print("Configurations:")
    print("  bf16: BF16 baseline")
    for method in requested:
        if method != "bf16":
            print(f"  {method}: {config_for_method(method)['name']}")
    print()

    # One small BF16 warm-up prevents CUDA initialization from contaminating
    # the first reported case. Its time is not reported.
    warmup_context = min(parsed_contexts)
    warmup_prompt = build_prompt(warmup_context, parsed_depths[0])
    print(f"Running one BF16 warm-up at {warmup_context} tokens...")
    _ = generate_answer("bf16", warmup_prompt)
    del warmup_prompt
    print()

    results = []
    total_cases = len(parsed_contexts) * len(parsed_depths)
    case_index = 0

    for context_tokens in parsed_contexts:
        for needle_depth in parsed_depths:
            case_index += 1
            prompt_ids = build_prompt(context_tokens, needle_depth)
            print(
                f"Case {case_index}/{total_cases}: "
                f"context={context_tokens}, depth={needle_depth:.0%}"
            )

            bf16_mb = None
            for method in requested:
                result = generate_answer(method, prompt_ids)
                result["context_tokens"] = int(context_tokens)
                result["needle_depth"] = float(needle_depth)
                if method == "bf16":
                    bf16_mb = result["logical_kv_mb"]
                result["compression"] = (
                    bf16_mb / result["logical_kv_mb"]
                    if bf16_mb is not None and result["logical_kv_mb"] > 0
                    else 1.0
                )
                results.append(result)

                print(
                    f"  {method:<8} "
                    f"{'PASS' if result['correct'] else 'FAIL':<5} "
                    f"KV={result['logical_kv_mb']:.3f} MB "
                    f"Comp={result['compression']:.3f}x "
                    f"answer={result['answer']!r}"
                )
            print()
            del prompt_ids
            gc.collect()
            torch.cuda.empty_cache()

    print("=" * 132)
    print("RABIT-KV NEEDLE-IN-A-HAYSTACK RESULTS")
    print("=" * 132)
    print(
        f"{'Context':<10}"
        f"{'Depth':<10}"
        f"{'Method':<10}"
        f"{'Result':<10}"
        f"{'KV MB':<14}"
        f"{'Compression':<14}"
        f"{'Extracted code':<20}"
        f"{'Answer':<36}"
    )
    print("-" * 132)
    for result in results:
        clipped_answer = result["answer"].replace("\n", " ")[:34]
        print(
            f"{result['context_tokens']:<10}"
            f"{result['needle_depth']:<10.2f}"
            f"{result['method']:<10}"
            f"{'PASS' if result['correct'] else 'FAIL':<10}"
            f"{result['logical_kv_mb']:<14.3f}"
            f"{result['compression']:<14.3f}"
            f"{result['extracted_code']:<20}"
            f"{clipped_answer:<36}"
        )

    print()
    print("AGGREGATE ACCURACY")
    print("-" * 86)
    print(
        f"{'Method':<12}"
        f"{'Passed':<12}"
        f"{'Total':<12}"
        f"{'Accuracy':<14}"
        f"{'Avg KV MB':<16}"
        f"{'Avg Compression':<18}"
    )
    print("-" * 86)

    aggregate = {}
    for method in requested:
        rows = [row for row in results if row["method"] == method]
        passed = sum(int(row["correct"]) for row in rows)
        total = len(rows)
        accuracy = passed / total if total else 0.0
        avg_kv_mb = sum(row["logical_kv_mb"] for row in rows) / total
        avg_compression = sum(row["compression"] for row in rows) / total
        aggregate[method] = {
            "passed": passed,
            "total": total,
            "accuracy": accuracy,
            "avg_kv_mb": avg_kv_mb,
            "avg_compression": avg_compression,
        }
        print(
            f"{method:<12}"
            f"{passed:<12}"
            f"{total:<12}"
            f"{accuracy * 100:<14.1f}"
            f"{avg_kv_mb:<16.3f}"
            f"{avg_compression:<18.3f}"
        )

    print()
    print(
        "Important: reported run time is for experiment planning only. "
        "It is not deployment latency."
    )

    return {
        "settings": {
            "model": model_name,
            "context_lengths": parsed_contexts,
            "needle_depths": parsed_depths,
            "max_new_tokens": max_new_tokens,
            "methods": requested,
            "secret_code": secret_code,
            "mode": "quality-only prefix-cache quantization",
        },
        "configurations": {
            method: (
                {"name": "BF16 baseline"}
                if method == "bf16"
                else config_for_method(method)
            )
            for method in requested
        },
        "results": results,
        "aggregate": aggregate,
    }


@app.local_entrypoint()
def main(
    model_name: str = "LLM-Research/Meta-Llama-3.1-8B-Instruct",
    context_lengths: str = "4096,8192,16384",
    needle_depths: str = "0.5",
    max_new_tokens: int = 16,
    methods: str = "bf16,rabit8,rabit4,rabit3,rabit2",
):
    run_niah.remote(
        model_name=model_name,
        context_lengths=context_lengths,
        needle_depths=needle_depths,
        max_new_tokens=max_new_tokens,
        methods=methods,
    )
