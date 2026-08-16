"""
RABIT-KV memory-quality search for Llama-3.1-8B.

Goal:
    Minimize effective KV-cache storage while keeping perplexity close to BF16.

The script performs one complete staged run:
    1. Builds a fixed BF16 reference on the full evaluation windows.
    2. Profiles K-layer sensitivity from the same model and data.
    3. Screens a broad set of K3/V2 candidates on a shorter fixed subset.
    4. Confirms only the strongest candidates on the full 4-window protocol.
    5. Selects the lowest-memory candidate satisfying the final PPL cap.
    6. Saves JSON and CSV result files locally.

Candidate families include:
    - residual-window sweep R0-R4
    - K/V group-size sweeps (32/64/128)
    - grouped INT8 metadata compression
    - layer-adaptive K2/K3 allocation
    - layer-selective residual protection
    - combined adaptive policies

Important:
    This remains a quality and logical-memory benchmark. Quantized KV is
    dequantized before attention. It does not measure production latency or
    physical packed vLLM memory.
"""

import modal

app = modal.App("rabit-kv-memory-quality-search")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "transformers==4.48.2",
    "accelerate",
    "requests",
    "sentencepiece",
    "modelscope",
)

model_cache = modal.Volume.from_name("modelscope-llama31-cache", create_if_missing=True)

@app.function(gpu="H100", image=image, timeout=86400, volumes={"/model_cache": model_cache})
def run_benchmark(model_name: str = "LLM-Research/Meta-Llama-3.1-8B-Instruct", mode: str = "final", max_ppl_increase_pct: float = 3.0, screening_ppl_increase_pct: float = 6.0, screen_tokens: int = 128, finalists: int = 7, bit_modes: str = "16,2"):
    import gc
    import math
    import statistics
    import time
    import os

    import requests
    import torch
    import torch.nn.functional as F
    from modelscope import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache

    # =============================================================
    # Config
    # =============================================================
    MODEL_NAME = str(model_name)
    MODEL_DTYPE = torch.bfloat16
    MAX_TOKENS = 256

    # quick: profile/evaluate on 2 windows for fast debugging.
    # final: profile on 2 windows, evaluate selected greedy path on 4 windows.
    # paper: profile on 4 windows, evaluate selected greedy path on 8 windows.
    mode = mode.lower().strip()
    max_ppl_increase_pct = float(max_ppl_increase_pct)
    screening_ppl_increase_pct = float(screening_ppl_increase_pct)
    screen_tokens = int(screen_tokens)
    finalists = int(finalists)
    BIT_MODES = [int(x.strip()) for x in str(bit_modes).split(",") if x.strip()]

    if mode not in {"quick", "final", "paper"}:
        raise ValueError("mode must be 'quick', 'final', or 'paper'")
    if max_ppl_increase_pct < 0 or screening_ppl_increase_pct < 0:
        raise ValueError("PPL caps must be non-negative")
    if screen_tokens < 32 or screen_tokens > MAX_TOKENS:
        raise ValueError("screen_tokens must be between 32 and MAX_TOKENS")
    if finalists < 3:
        raise ValueError("finalists must be at least 3")
    if not BIT_MODES:
        raise ValueError("bit_modes must contain at least one bit value")
    for b in BIT_MODES:
        if b not in {16, 8, 4, 3, 2, 1}:
            raise ValueError("bit_modes can only contain 16, 8, 4, 3, 2, or 1")

    MODE_CFG = {
        "quick": {"num_windows": 2, "screen_windows": 1},
        "final": {"num_windows": 4, "screen_windows": 1},
        "paper": {"num_windows": 8, "screen_windows": 2},
    }
    NUM_WINDOWS = MODE_CFG[mode]["num_windows"]
    SCREEN_WINDOWS = MODE_CFG[mode]["screen_windows"]
    TARGET_BITS = [2]

    # The ladder is intentionally ordered so each row adds one idea.
    # This lets you see which improvement actually helps and what memory tradeoff it creates.
    # Main Phase 4 rerun after the first final-mode result:
    # clipping made PPL worse, so this version removes clipping from the main ladder.
    # The ladder tests residual windows and K/V grouping without the bad clipping step.
    METHODS = [
        {
            "id": "p3_uniform",
            "name": "P3 uniform",
            "style": "uniform",
            "k_group": None,
            "v_group": None,
            "clip_by_bits": {},
            "residual_tokens": 0,
            "per_token_scale": False,
            "mixed_sensitive_layers": False,
            "only_2bit": False,
            "note": "global min/max baseline",
        },
        {
            "id": "s1_group",
            "name": "S1 group",
            "style": "group",
            "k_group": 32,
            "v_group": 32,
            "clip_by_bits": {},
            "residual_tokens": 0,
            "per_token_scale": False,
            "mixed_sensitive_layers": False,
            "only_2bit": False,
            "note": "per-group min/max, no clipping",
        },
        {
            "id": "s2_resid32",
            "name": "S2 +resid32",
            "style": "group",
            "k_group": 32,
            "v_group": 32,
            "clip_by_bits": {},
            "residual_tokens": 32,
            "per_token_scale": False,
            "mixed_sensitive_layers": False,
            "only_2bit": False,
            "note": "newest 32 tokens fp16, no clipping",
        },
        {
            "id": "s3_resid64",
            "name": "S3 +resid64",
            "style": "group",
            "k_group": 32,
            "v_group": 32,
            "clip_by_bits": {},
            "residual_tokens": 64,
            "per_token_scale": False,
            "mixed_sensitive_layers": False,
            "only_2bit": False,
            "note": "newest 64 tokens fp16, no clipping",
        },
        {
            "id": "s4_token_scale",
            "name": "S4 +token",
            "style": "group",
            "k_group": 32,
            "v_group": 32,
            "clip_by_bits": {},
            "residual_tokens": 64,
            "per_token_scale": True,
            "mixed_sensitive_layers": False,
            "only_2bit": False,
            "note": "resid64 + per-token scale, no clipping",
        },
        {
            "id": "s5_asym_kv",
            "name": "S5 asym K/V",
            "style": "group",
            "k_group": 16,
            "v_group": 64,
            "clip_by_bits": {},
            "residual_tokens": 64,
            "per_token_scale": True,
            "mixed_sensitive_layers": False,
            "only_2bit": False,
            "note": "K group16, V group64, resid64, no clipping",
        },
        {
            "id": "s6_asym_resid32",
            "name": "S6 asym r32",
            "style": "group",
            "k_group": 16,
            "v_group": 64,
            "clip_by_bits": {},
            "residual_tokens": 32,
            "per_token_scale": True,
            "mixed_sensitive_layers": False,
            "only_2bit": True,
            "note": "2-bit compression tune: asym K/V + resid32",
        },
        {
            "id": "s7_mixed",
            "name": "S7 mixed",
            "style": "group",
            "k_group": 16,
            "v_group": 64,
            "clip_by_bits": {},
            "residual_tokens": 64,
            "per_token_scale": True,
            "mixed_sensitive_layers": True,
            "only_2bit": True,
            "note": "2-bit except first/last layers use 4-bit, no clipping",
        },
    ]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"Mode: {mode}")
    print(f"Model: {MODEL_NAME}")
    print(f"Model dtype: {MODEL_DTYPE}")
    print(f"Tokens/window: {MAX_TOKENS} ({MAX_TOKENS - 1} predicted steps)")
    print(f"Full evaluation windows: {NUM_WINDOWS}")
    print(f"Screening windows: {SCREEN_WINDOWS}")
    print(f"Screening tokens/window: {screen_tokens}")
    print(f"Final PPL cap: {max_ppl_increase_pct:.2f}% above BF16")
    print(f"Screening PPL cap: {screening_ppl_increase_pct:.2f}% above screening BF16")
    print(f"Finalists: {finalists}")
    print(f"Bit modes: {BIT_MODES}")
    print("Objective: lowest effective KV memory subject to the final PPL cap.")
    print("Memory = logical packed KV storage including metadata and residual tokens.")
    print("This script does not report production latency.")
    print()

    # =============================================================
    # Data + model
    # =============================================================
    url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt"
    print("Loading WikiText-2 test data...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    lines = [line.strip() for line in response.text.splitlines() if line.strip()]

    print("Loading Llama-3.1 model from ModelScope cache...")
    os.environ["MODELSCOPE_CACHE"] = "/model_cache"
    local_model_dir = snapshot_download(MODEL_NAME, cache_dir="/model_cache")
    try:
        model_cache.commit()
    except Exception:
        pass

    tokenizer = AutoTokenizer.from_pretrained(local_model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        local_model_dir,
        torch_dtype=MODEL_DTYPE,
        device_map=None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()
    print("Model loaded.")
    print()

    def build_token_pool(needed_tokens):
        ids = []
        batch_size = 64
        for start in range(0, len(lines), batch_size):
            text = "\n".join(lines[start : start + batch_size])
            out = tokenizer(text, add_special_tokens=False)
            ids.extend(out["input_ids"])
            if len(ids) >= needed_tokens:
                break
        if len(ids) < needed_tokens:
            raise RuntimeError(f"Need {needed_tokens} tokens, but only got {len(ids)} tokens.")
        return torch.tensor(ids[:needed_tokens], dtype=torch.long)

    needed = NUM_WINDOWS * MAX_TOKENS
    pool_ids = build_token_pool(needed)
    windows = [
        pool_ids[i * MAX_TOKENS : (i + 1) * MAX_TOKENS].unsqueeze(0).to(device)
        for i in range(NUM_WINDOWS)
    ]

    # =============================================================
    # Cache helpers
    # =============================================================
    def cache_to_tuple(cache):
        if cache is None:
            return None
        if isinstance(cache, tuple):
            return cache
        if isinstance(cache, list):
            return tuple(cache)
        if hasattr(cache, "to_legacy_cache"):
            return cache.to_legacy_cache()
        if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
            return tuple((k, v) for k, v in zip(cache.key_cache, cache.value_cache))
        raise RuntimeError(f"Unsupported cache type: {type(cache)}")

    def tuple_to_dynamic_cache(kv_tuple):
        cache = DynamicCache()
        for layer_idx, (k, v) in enumerate(kv_tuple):
            cache.update(k, v, layer_idx)
        return cache

    def tensor_bytes(tensor):
        return tensor.numel() * tensor.element_size()

    def fp_cache_kb(cache):
        kv_tuple = cache_to_tuple(cache)
        return sum(tensor_bytes(t) for layer in kv_tuple for t in layer) / 1024

    # =============================================================
    # Bit packing
    # =============================================================
    def pack_bits(codes, bits):
        if bits == 8:
            return codes.to(torch.uint8).contiguous(), tuple(codes.shape), codes.numel()

        flat = codes.flatten().to(torch.int64)
        shape = tuple(codes.shape)
        numel = flat.numel()
        nbytes = (numel * bits + 7) // 8

        offsets = torch.arange(numel, device=flat.device, dtype=torch.int64) * bits
        byte_idx = offsets // 8
        bit_offset = offsets % 8
        shifted = flat << bit_offset

        packed_i = torch.zeros(nbytes + 1, device=flat.device, dtype=torch.int64)
        packed_i.scatter_add_(0, byte_idx, shifted & 255)

        high = shifted >> 8
        mask = high > 0
        if bool(mask.any()):
            packed_i.scatter_add_(0, byte_idx[mask] + 1, high[mask] & 255)

        return packed_i[:nbytes].to(torch.uint8), shape, numel

    def unpack_bits(packed, bits, shape, numel):
        if bits == 8:
            return packed.view(shape).to(torch.int64)

        p = packed.to(torch.int64)
        p = torch.cat([p, torch.zeros(1, device=p.device, dtype=torch.int64)])
        offsets = torch.arange(numel, device=p.device, dtype=torch.int64) * bits
        byte_idx = offsets // 8
        bit_offset = offsets % 8
        two_bytes = p[byte_idx] | (p[byte_idx + 1] << 8)
        codes = (two_bytes >> bit_offset) & ((1 << bits) - 1)
        return codes.view(shape)

    # =============================================================
    # Quantizers
    # =============================================================
    def quantile_from_sorted(sorted_x, q):
        if q <= 0.0:
            return sorted_x[..., 0:1]
        if q >= 1.0:
            return sorted_x[..., -1:]
        n = sorted_x.shape[-1]
        pos = q * (n - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        frac = pos - lo
        if lo == hi:
            return sorted_x[..., lo : lo + 1]
        return sorted_x[..., lo : lo + 1] * (1.0 - frac) + sorted_x[..., hi : hi + 1] * frac

    def encode_metadata(tensor, method):
        """Store quantization metadata as BF16 or grouped UINT8.

        Grouped UINT8 stores one BF16 min and scale per metadata group. This
        reduces metadata from two bytes per value to about one byte per value
        while keeping the representation easy to reproduce in a packed kernel.
        """
        mode = str(method.get("metadata_mode", "bf16")).lower()
        data = tensor.detach().float().contiguous()
        if mode == "bf16":
            return {
                "meta_type": "bf16",
                "data": data.to(MODEL_DTYPE).contiguous(),
            }
        if mode not in {"int8", "uint8"}:
            raise ValueError(f"Unsupported metadata_mode: {mode}")

        group_size = max(8, int(method.get("metadata_group_size", 256)))
        orig_shape = tuple(data.shape)
        flat = data.reshape(-1)
        pad = (-flat.numel()) % group_size
        if pad:
            fill = flat[-1:].expand(pad)
            flat = torch.cat([flat, fill], dim=0)
        grouped = flat.reshape(-1, group_size)
        meta_min = grouped.amin(dim=-1, keepdim=True)
        meta_max = grouped.amax(dim=-1, keepdim=True)
        meta_scale = (meta_max - meta_min) / 255.0
        meta_scale = torch.where(
            meta_scale.abs() < 1e-12,
            torch.ones_like(meta_scale),
            meta_scale,
        )
        codes = torch.round((grouped - meta_min) / meta_scale).clamp(0, 255).to(torch.uint8)
        return {
            "meta_type": "uint8_group",
            "codes": codes.contiguous(),
            "min": meta_min.to(MODEL_DTYPE).contiguous(),
            "scale": meta_scale.to(MODEL_DTYPE).contiguous(),
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
            values = meta["codes"].float() * meta["scale"].float() + meta["min"].float()
            flat = values.reshape(-1)
            if meta.get("pad", 0):
                flat = flat[:-int(meta["pad"])]
            return flat.reshape(meta["orig_shape"])
        raise ValueError(f"Unsupported metadata representation: {meta_type}")

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

    def quantize_tensor_uniform_minmax(tensor, bits):
        if bits == 16:
            return {"type": "fp16", "data": tensor.detach().to(MODEL_DTYPE).contiguous()}

        x = tensor.detach().float()
        x_min = x.amin()
        x_max = x.amax()
        levels = 2 ** bits
        scale = (x_max - x_min) / (levels - 1)
        if float(scale.abs().item()) < 1e-8:
            scale = torch.ones((), device=x.device, dtype=torch.float32)

        codes = torch.round((x - x_min) / scale).clamp(0, levels - 1).to(torch.uint8)
        packed, shape, numel = pack_bits(codes, bits)
        return {
            "type": "int",
            "style": "uniform",
            "bits": bits,
            "packed": packed,
            "shape": shape,
            "numel": numel,
            "orig_shape": tuple(tensor.shape),
            "min": x_min.to(MODEL_DTYPE).reshape(1).contiguous(),
            "scale": scale.to(MODEL_DTYPE).reshape(1).contiguous(),
        }

    def quantize_tensor_group(tensor, bits, method, side):
        if bits == 16:
            return {"type": "fp16", "data": tensor.detach().to(MODEL_DTYPE).contiguous()}

        x = tensor.detach().float()
        orig_shape = tuple(x.shape)
        group_size = int(method["k_group"] if side == "k" else method["v_group"])

        token_absmax = None
        if method.get("per_token_scale", False):
            token_absmax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
            x = x / token_absmax

        pad = (-orig_shape[-1]) % group_size
        if pad:
            x = F.pad(x, (0, pad))

        grouped = x.reshape(*x.shape[:-1], -1, group_size)
        clip_pct = float(method.get("clip_by_bits", {}).get(bits, 0.0))

        if clip_pct > 0.0:
            sorted_grouped, _ = grouped.sort(dim=-1)
            q_min = quantile_from_sorted(sorted_grouped, clip_pct)
            q_max = quantile_from_sorted(sorted_grouped, 1.0 - clip_pct)
            grouped = grouped.clamp(q_min, q_max)
        else:
            q_min = grouped.amin(dim=-1, keepdim=True)
            q_max = grouped.amax(dim=-1, keepdim=True)

        levels = 2 ** bits

        # v13 metadata-reduced option:
        # symmetric absmax quantization stores one fp16 absmax per group instead of min + scale.
        # This directly targets the overhead gap between our actual compression and the ideal bit compression.
        if bool(method.get("symmetric_absmax", False)):
            q_abs = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
            codes = torch.round(((grouped / q_abs) + 1.0) * 0.5 * (levels - 1)).clamp(0, levels - 1).to(torch.uint8)
            packed, shape, numel = pack_bits(codes, bits)

            q = {
                "type": "int",
                "style": "group_sym",
                "bits": bits,
                "packed": packed,
                "shape": shape,
                "numel": numel,
                "orig_shape": orig_shape,
                "pad": pad,
                "absmax": encode_metadata(q_abs, method),
                "group_size": group_size,
                "clip_pct": clip_pct,
                "per_token_scale": bool(method.get("per_token_scale", False)),
            }
            if token_absmax is not None:
                q["token_absmax"] = encode_metadata(token_absmax, method)
            return q

        scale = (q_max - q_min) / (levels - 1)
        scale = torch.where(scale.abs() < 1e-8, torch.ones_like(scale), scale)

        codes = torch.round((grouped - q_min) / scale).clamp(0, levels - 1).to(torch.uint8)
        packed, shape, numel = pack_bits(codes, bits)

        q = {
            "type": "int",
            "style": "group",
            "bits": bits,
            "packed": packed,
            "shape": shape,
            "numel": numel,
            "orig_shape": orig_shape,
            "pad": pad,
            "min": encode_metadata(q_min, method),
            "scale": encode_metadata(scale, method),
            "group_size": group_size,
            "clip_pct": clip_pct,
            "per_token_scale": bool(method.get("per_token_scale", False)),
        }
        if token_absmax is not None:
            q["token_absmax"] = encode_metadata(token_absmax, method)
        return q


    def quantize_tensor_k_seq_group_sym(tensor, bits, method, side):
        # v14 hetero K/V path:
        # K is quantized over the sequence axis for each channel, closer to KIVI-style K handling.
        # Tensor shape is [batch, kv_heads, seq, head_dim].
        if bits == 16:
            return {"type": "fp16", "data": tensor.detach().to(MODEL_DTYPE).contiguous()}
        if tensor.dim() < 4:
            return quantize_tensor_group(tensor, bits, method, side)

        x = tensor.detach().float()
        orig_shape = tuple(x.shape)
        seq_group = int(method.get("k_seq_group", method.get("k_group", 32)))
        seq_group = max(1, seq_group)
        pad_seq = (-orig_shape[-2]) % seq_group
        if pad_seq:
            x = F.pad(x, (0, 0, 0, pad_seq))

        bsz, nheads, seq_len, head_dim = x.shape
        grouped = x.reshape(bsz, nheads, seq_len // seq_group, seq_group, head_dim)
        levels = 2 ** bits
        q_abs = grouped.abs().amax(dim=3, keepdim=True).clamp(min=1e-8)
        codes = torch.round(((grouped / q_abs) + 1.0) * 0.5 * (levels - 1)).clamp(0, levels - 1).to(torch.uint8)
        packed, shape, numel = pack_bits(codes, bits)
        return {
            "type": "int",
            "style": "seq_group_sym",
            "bits": bits,
            "packed": packed,
            "shape": shape,
            "numel": numel,
            "orig_shape": orig_shape,
            "pad_seq": pad_seq,
            "absmax": encode_metadata(q_abs, method),
            "seq_group": seq_group,
        }

    def quantize_tensor_k_seq_group_affine(tensor, bits, method, side):
        # v16 KIVI-style K path:
        # K is grouped along sequence for each channel/head-dim coordinate.
        # Unlike the v14 symmetric version, this stores min+scale per K channel group.
        # That costs more metadata, but can be much better at 2-bit when K is not zero-centered.
        if bits == 16:
            return {"type": "fp16", "data": tensor.detach().to(MODEL_DTYPE).contiguous()}
        if tensor.dim() < 4:
            return quantize_tensor_group(tensor, bits, method, side)

        x = tensor.detach().float()
        orig_shape = tuple(x.shape)
        seq_group = int(method.get("k_seq_group", method.get("k_group", 32)))
        seq_group = max(1, seq_group)

        pad_seq = (-orig_shape[-2]) % seq_group
        if pad_seq:
            x = F.pad(x, (0, 0, 0, pad_seq))

        bsz, nheads, seq_len, head_dim = x.shape
        grouped = x.reshape(bsz, nheads, seq_len // seq_group, seq_group, head_dim)
        levels = 2 ** bits

        q_min = grouped.amin(dim=3, keepdim=True)
        q_max = grouped.amax(dim=3, keepdim=True)
        scale = (q_max - q_min) / (levels - 1)
        scale = torch.where(scale.abs() < 1e-8, torch.ones_like(scale), scale)

        codes = torch.round((grouped - q_min) / scale).clamp(0, levels - 1).to(torch.uint8)
        packed, shape, numel = pack_bits(codes, bits)

        return {
            "type": "int",
            "style": "seq_group_affine",
            "bits": bits,
            "packed": packed,
            "shape": shape,
            "numel": numel,
            "orig_shape": orig_shape,
            "pad_seq": pad_seq,
            "min": encode_metadata(q_min, method),
            "scale": encode_metadata(scale, method),
            "seq_group": seq_group,
        }

    def _codebook_centers_and_codes_lastdim(x, bits):
        # Data-driven non-uniform scalar quantization per group.
        # For 2-bit this creates 4 centroids from equal-count sorted bins.
        # Codes remain bit-packed; metadata is the fp16 centroid table per group.
        levels = 2 ** int(bits)
        sorted_x, _ = x.sort(dim=-1)
        n = int(sorted_x.shape[-1])
        centers = []
        for i in range(levels):
            lo = int(round(i * n / levels))
            hi = int(round((i + 1) * n / levels))
            hi = max(hi, lo + 1)
            hi = min(hi, n)
            lo = min(lo, hi - 1)
            centers.append(sorted_x[..., lo:hi].mean(dim=-1))
        centers = torch.stack(centers, dim=-1).contiguous()
        thresholds = 0.5 * (centers[..., :-1] + centers[..., 1:])
        codes = (x.unsqueeze(-1) > thresholds.unsqueeze(-2)).sum(dim=-1).to(torch.uint8)
        return centers, codes

    def quantize_tensor_group_codebook(tensor, bits, method, side):
        # V17 NUQ/codebook path over the last dimension.
        # This is intended mainly for 2-bit, where uniform scalar levels waste precision.
        if bits == 16:
            return {"type": "fp16", "data": tensor.detach().to(MODEL_DTYPE).contiguous()}

        x = tensor.detach().float()
        orig_shape = tuple(x.shape)
        group_size = int(method["k_group"] if side == "k" else method["v_group"])
        group_size = max(1, group_size)

        pad = (-orig_shape[-1]) % group_size
        if pad:
            x = F.pad(x, (0, pad))

        grouped = x.reshape(*x.shape[:-1], -1, group_size)
        centers, codes = _codebook_centers_and_codes_lastdim(grouped, bits)
        packed, shape, numel = pack_bits(codes, bits)

        return {
            "type": "int",
            "style": "group_codebook",
            "bits": int(bits),
            "packed": packed,
            "shape": shape,
            "numel": numel,
            "orig_shape": orig_shape,
            "pad": int(pad),
            "codebook": encode_metadata(centers, method),
            "group_size": int(group_size),
        }

    def quantize_tensor_k_seq_group_codebook(tensor, bits, method, side):
        # V17 NUQ/codebook K path: K is grouped along sequence for each channel.
        # This follows the KIVI-style observation that K is more naturally quantized per-channel.
        if bits == 16:
            return {"type": "fp16", "data": tensor.detach().to(MODEL_DTYPE).contiguous()}
        if tensor.dim() < 4:
            return quantize_tensor_group_codebook(tensor, bits, method, side)

        x = tensor.detach().float()
        orig_shape = tuple(x.shape)
        seq_group = int(method.get("k_seq_group", method.get("k_group", 128)))
        seq_group = max(1, seq_group)
        pad_seq = (-orig_shape[-2]) % seq_group
        if pad_seq:
            x = F.pad(x, (0, 0, 0, pad_seq))

        bsz, nheads, seq_len, head_dim = x.shape
        grouped = x.reshape(bsz, nheads, seq_len // seq_group, seq_group, head_dim)
        grouped_t = grouped.transpose(3, 4).contiguous()  # [B,H,seq_groups,head_dim,seq_group]
        centers, codes = _codebook_centers_and_codes_lastdim(grouped_t, bits)
        packed, shape, numel = pack_bits(codes, bits)

        return {
            "type": "int",
            "style": "seq_group_codebook",
            "bits": int(bits),
            "packed": packed,
            "shape": shape,
            "numel": numel,
            "orig_shape": orig_shape,
            "pad_seq": int(pad_seq),
            "codebook": encode_metadata(centers, method),
            "seq_group": int(seq_group),
        }

    def quantize_tensor(tensor, bits, method, side):
        if method["style"] == "uniform":
            return quantize_tensor_uniform_minmax(tensor, bits)
        if method["style"] == "group":
            if side == "k" and method.get("k_quant_axis", "dim") == "seq_codebook":
                return quantize_tensor_k_seq_group_codebook(tensor, bits, method, side)
            if method.get("codebook_quant", False) and side in set(str(method.get("codebook_sides", "kv"))):
                return quantize_tensor_group_codebook(tensor, bits, method, side)
            if side == "k" and method.get("k_quant_axis", "dim") == "seq":
                return quantize_tensor_k_seq_group_sym(tensor, bits, method, side)
            if side == "k" and method.get("k_quant_axis", "dim") == "seq_affine":
                return quantize_tensor_k_seq_group_affine(tensor, bits, method, side)
            return quantize_tensor_group(tensor, bits, method, side)
        raise ValueError(f"Unknown quantizer style: {method['style']}")

    def dequantize_tensor(q):
        if q["type"] == "fp16":
            return q["data"].to(MODEL_DTYPE)

        codes = unpack_bits(q["packed"], q["bits"], q["shape"], q["numel"]).float()

        if q.get("style") in {"group_codebook", "seq_group_codebook"}:
            cb = decode_metadata(q["codebook"])
            x = torch.gather(cb.unsqueeze(-2).expand(*codes.shape, cb.shape[-1]), -1, codes.long().unsqueeze(-1)).squeeze(-1)
        elif q.get("style") in {"group_sym", "seq_group_sym"}:
            levels = 2 ** int(q["bits"])
            x = ((codes / (levels - 1)) * 2.0 - 1.0) * decode_metadata(q["absmax"])
        else:
            x = codes * decode_metadata(q["scale"]) + decode_metadata(q["min"])

        if q.get("style") in {"group", "group_sym", "group_codebook"}:
            x = x.reshape(*q["orig_shape"][:-1], -1)
            if q.get("pad", 0):
                x = x[..., : q["orig_shape"][-1]]
            if q.get("per_token_scale", False):
                x = x * decode_metadata(q["token_absmax"])
        elif q.get("style") in {"seq_group_sym", "seq_group_affine"}:
            bsz, nheads, orig_seq, head_dim = q["orig_shape"]
            seq_group = int(q["seq_group"])
            padded_seq = orig_seq + int(q.get("pad_seq", 0))
            x = x.reshape(bsz, nheads, padded_seq // seq_group, seq_group, head_dim)
            x = x.reshape(bsz, nheads, padded_seq, head_dim)
            if q.get("pad_seq", 0):
                x = x[..., :orig_seq, :]
        elif q.get("style") == "seq_group_codebook":
            bsz, nheads, orig_seq, head_dim = q["orig_shape"]
            seq_group = int(q["seq_group"])
            padded_seq = orig_seq + int(q.get("pad_seq", 0))
            x = x.reshape(bsz, nheads, padded_seq // seq_group, head_dim, seq_group)
            x = x.transpose(3, 4).contiguous().reshape(bsz, nheads, padded_seq, head_dim)
            if q.get("pad_seq", 0):
                x = x[..., :orig_seq, :]
        else:
            x = x.reshape(q["orig_shape"])

        return x.to(MODEL_DTYPE)

    def stored_tensor_bytes(q):
        if q["type"] == "fp16":
            return tensor_bytes(q["data"])
        if q["type"] == "split":
            return stored_tensor_bytes(q["old"]) + stored_tensor_bytes(q["recent"])
        if q["type"] == "sink_split":
            return stored_tensor_bytes(q["sink"]) + stored_tensor_bytes(q["middle"]) + stored_tensor_bytes(q["recent"])
        if q["type"] == "heads":
            return sum(stored_tensor_bytes(hq) for hq in q["heads"])
        if q["type"] == "dim_blocks":
            return sum(stored_tensor_bytes(piece["q"]) for piece in q["pieces"])
        if q["type"] == "token_exceptions":
            # v14 logical storage: exception tokens are represented as a sparse patch.
            # Do not double-charge their low-bit base payload bytes.
            base_bytes = stored_tensor_bytes(q["base"])
            saved = int(q.get("saved_base_payload_bytes", 0))
            return max(0, base_bytes - saved) + tensor_bytes(q["indices"]) + tensor_bytes(q["values"])

        if "codebook" in q:
            total = tensor_bytes(q["packed"]) + metadata_bytes(q["codebook"])
        elif "absmax" in q:
            total = tensor_bytes(q["packed"]) + metadata_bytes(q["absmax"])
        else:
            total = (
                tensor_bytes(q["packed"])
                + metadata_bytes(q["min"])
                + metadata_bytes(q["scale"])
            )
        if "token_absmax" in q:
            total += metadata_bytes(q["token_absmax"])
        return total

    def choose_layer_bits(target_bits, method, layer_idx, num_layers):
        if target_bits == 16:
            return 16

        # RABIT-KV Lite rule:
        # most layers stay at the target bit, but selected layers can be upgraded
        # according to a learned/profiler-produced layer_bit_map.
        layer_bit_map = method.get("layer_bit_map", None)
        if layer_bit_map is not None:
            if layer_idx in layer_bit_map:
                return int(layer_bit_map[layer_idx])
            str_key = str(layer_idx)
            if str_key in layer_bit_map:
                return int(layer_bit_map[str_key])

        # Older importance-aware rule kept for comparison/backward compatibility.
        protected_layers = method.get("protected_layers", None)
        protected_bits = int(method.get("protected_bits", 4))
        if protected_layers is not None and target_bits < protected_bits:
            if layer_idx in set(protected_layers):
                return protected_bits

        # Older fixed rule kept for comparison/backward compatibility.
        if method.get("mixed_sensitive_layers", False) and target_bits == 2:
            sensitive = {0, 1, num_layers - 2, num_layers - 1}
            if layer_idx in sensitive:
                return 4
        return target_bits

    def get_head_bit_map_for_layer(method, layer_idx):
        head_bit_map = method.get("head_bit_map", None)
        if not head_bit_map:
            return {}

        if layer_idx in head_bit_map:
            return head_bit_map[layer_idx]
        str_layer = str(layer_idx)
        if str_layer in head_bit_map:
            return head_bit_map[str_layer]
        return {}

    def quantize_tensor_k_rope_blocks(tensor, default_bits, method, side, layer_idx):
        # v13: RoPE-aware K-block protection.
        #
        # Motivation:
        # - Whole-layer or whole-head protection wastes memory.
        # - v13 protects only the most energetic RoPE-style blocks inside K.
        #
        # Practical version:
        # - Split head_dim into small contiguous RoPE-like blocks.
        # - Estimate block importance from K energy on the current cache.
        # - Quantize top blocks with a higher bit; leave the rest at target bit.
        #
        # This is a fake-quant benchmark implementation. It measures the quality/memory
        # tradeoff of the idea; it is not a production packed kernel.
        if tensor.dim() < 4:
            return quantize_tensor(tensor, default_bits, method, side)

        block_size = int(method.get("k_rope_block_size", 16))
        top_blocks = int(method.get("k_rope_top_blocks", 1))
        top_bits = int(method.get("k_rope_top_bits", max(default_bits, 3)))
        if block_size <= 0 or top_blocks <= 0 or top_bits <= default_bits:
            return quantize_tensor(tensor, default_bits, method, side)

        head_dim = int(tensor.shape[-1])
        if head_dim <= block_size:
            return quantize_tensor(tensor, max(default_bits, top_bits), method, side)

        n_blocks = (head_dim + block_size - 1) // block_size
        top_blocks = min(top_blocks, n_blocks)

        # Compute block importance. This approximates RoPE-block energy:
        # high-energy K blocks are more dangerous to quantize too aggressively.
        x = tensor.detach().float()
        scores = []
        for b in range(n_blocks):
            start = b * block_size
            end = min((b + 1) * block_size, head_dim)
            block = x[..., start:end]
            score = block.square().mean()
            scores.append(score)

        scores_t = torch.stack(scores)
        selected = set(int(i) for i in torch.topk(scores_t, k=top_blocks).indices.detach().cpu().tolist())

        pieces = []
        for b in range(n_blocks):
            start = b * block_size
            end = min((b + 1) * block_size, head_dim)
            bit = int(top_bits) if b in selected else int(default_bits)

            # Use block-sized groups to avoid wasting packed codes on padding.
            local_method = dict(method)
            local_method["k_group"] = max(1, end - start)
            local_method["head_bit_map"] = {}
            local_method["k_rope_blocks"] = False

            piece = tensor[..., start:end]
            pieces.append(
                {
                    "start": int(start),
                    "end": int(end),
                    "bits": int(bit),
                    "q": quantize_tensor(piece, bit, local_method, side),
                }
            )

        return {
            "type": "dim_blocks",
            "orig_shape": tuple(tensor.shape),
            "block_size": int(block_size),
            "top_blocks": int(top_blocks),
            "top_bits": int(top_bits),
            "selected_blocks": sorted(selected),
            "pieces": pieces,
        }

    def quantize_tensor_token_exceptions(tensor, default_bits, method, side, layer_idx):
        # v13: outlier-token correction.
        #
        # Motivation:
        # - Low-bit KV failure is often caused by a few abnormal tokens.
        # - Instead of protecting a whole residual window/head/layer, keep only the
        #   highest-norm old tokens in higher precision and overwrite them after dequant.
        #
        # Memory note:
        # - This simple implementation quantizes the full tensor AND stores fp16 copies
        #   of exception tokens. That is conservative for memory. A production version
        #   would skip packing exception tokens into the base quantized tensor.
        if tensor.dim() < 4:
            return quantize_tensor(tensor, default_bits, method, side)

        count = int(method.get("v_outlier_tokens", 0))
        if count <= 0:
            return quantize_tensor_headwise(tensor, default_bits, {**method, "v_outlier_tokens": 0}, side, layer_idx)

        seq_len = int(tensor.shape[-2])
        if seq_len <= 1:
            return quantize_tensor_headwise(tensor, default_bits, {**method, "v_outlier_tokens": 0}, side, layer_idx)

        count = min(count, seq_len)

        # Score token abnormality by max magnitude across batch/head/channel.
        # This is intentionally simple and stable.
        score = tensor.detach().float().abs().amax(dim=(0, 1, 3))
        indices = torch.topk(score, k=count).indices.sort().values

        base_method = dict(method)
        base_method["v_outlier_tokens"] = 0
        base_q = quantize_tensor_headwise(tensor, default_bits, base_method, side, layer_idx)

        values = tensor.index_select(-2, indices.to(tensor.device)).detach().to(MODEL_DTYPE).contiguous()
        saved_base_payload_bytes = int(math.ceil(values.numel() * int(default_bits) / 8.0))
        return {
            "type": "token_exceptions",
            "base": base_q,
            "indices": indices.to(torch.int16).contiguous(),
            "values": values,
            "saved_base_payload_bytes": saved_base_payload_bytes,
        }

    def quantize_tensor_headwise(tensor, default_bits, method, side, layer_idx):
        # Tensor shape for KV cache is normally [batch, kv_heads, seq, head_dim].
        # v13 K path: protect important RoPE-like K blocks instead of whole heads.
        if side == "k" and bool(method.get("k_rope_blocks", False)):
            return quantize_tensor_k_rope_blocks(tensor, default_bits, method, side, layer_idx)

        # v13 V path: keep a tiny number of abnormal V tokens as exceptions.
        if side == "v" and int(method.get("v_outlier_tokens", 0)) > 0:
            return quantize_tensor_token_exceptions(tensor, default_bits, method, side, layer_idx)

        # We only split along the head axis when a head_bit_map exists for this layer.
        head_bits = get_head_bit_map_for_layer(method, layer_idx)
        if not head_bits:
            return quantize_tensor(tensor, default_bits, method, side)

        if tensor.dim() < 4:
            return quantize_tensor(tensor, default_bits, method, side)

        num_heads = int(tensor.shape[1])
        q_heads = []
        for head_idx in range(num_heads):
            bit = default_bits
            if head_idx in head_bits:
                bit = max(default_bits, int(head_bits[head_idx]))
            elif str(head_idx) in head_bits:
                bit = max(default_bits, int(head_bits[str(head_idx)]))

            head_tensor = tensor[:, head_idx : head_idx + 1, ...]
            q_heads.append(quantize_tensor(head_tensor, bit, method, side))

        return {
            "type": "heads",
            "head_dim": 1,
            "heads": q_heads,
        }

    def quantize_tensor_with_residual(tensor, bits, method, side, layer_idx=None):
        if bits == 16:
            return {"type": "fp16", "data": tensor.detach().to(MODEL_DTYPE).contiguous()}

        residual_tokens = int(method.get("residual_tokens", 0))
        sink_tokens = int(method.get("sink_tokens", 0))
        sink_bits = method.get("sink_bits", 16)
        sink_bits = 16 if sink_bits is None else int(sink_bits)

        # v15 sink-token strategy:
        # A few earliest tokens can be protected separately from the recent residual.
        # This targets attention-sink behavior without forcing a large residual window.
        # It is especially useful for 2-bit, where lowering residual overhead matters.
        if side == "k":
            sink_tokens = int(method.get("k_sink_tokens", sink_tokens))
            sink_bits = int(method.get("k_sink_bits", sink_bits))
        elif side == "v":
            sink_tokens = int(method.get("v_sink_tokens", sink_tokens))
            sink_bits = int(method.get("v_sink_bits", sink_bits))

        # v13/v14 selective-correction strategy:
        # residual_tokens is no longer forced to be the same for every layer.
        # A method can define layer_residual_map so only sensitive layers keep a larger fp16
        # recent-window correction, while less sensitive layers use smaller budgets.
        layer_residual_map = method.get("layer_residual_map", None)
        if layer_residual_map is not None and layer_idx is not None:
            if layer_idx in layer_residual_map:
                residual_tokens = int(layer_residual_map[layer_idx])
            elif str(layer_idx) in layer_residual_map:
                residual_tokens = int(layer_residual_map[str(layer_idx)])

        seq_len = tensor.shape[-2]
        sink_tokens = max(0, min(int(sink_tokens), int(seq_len)))
        residual_tokens = max(0, min(int(residual_tokens), int(seq_len) - sink_tokens))

        # Three-way split: sink prefix + low-bit middle + fp16 recent suffix.
        if sink_tokens > 0 and seq_len > sink_tokens + residual_tokens:
            sink = tensor[..., :sink_tokens, :]
            middle = tensor[..., sink_tokens : seq_len - residual_tokens, :]
            recent = tensor[..., seq_len - residual_tokens :, :] if residual_tokens > 0 else None

            if sink_bits >= 16:
                sink_q = {"type": "fp16", "data": sink.detach().to(MODEL_DTYPE).contiguous()}
            else:
                sink_q = quantize_tensor_headwise(sink, max(bits, sink_bits), method, side, layer_idx)

            q = {
                "type": "sink_split",
                "sink": sink_q,
                "middle": quantize_tensor_headwise(middle, bits, method, side, layer_idx),
                "recent": {"type": "fp16", "data": recent.detach().to(MODEL_DTYPE).contiguous()}
                          if residual_tokens > 0 else {"type": "fp16", "data": tensor[..., 0:0, :].detach().to(MODEL_DTYPE).contiguous()},
            }
            return q

        if residual_tokens > 0:
            if seq_len <= residual_tokens:
                return {"type": "fp16", "data": tensor.detach().to(MODEL_DTYPE).contiguous()}

            old = tensor[..., : seq_len - residual_tokens, :]
            recent = tensor[..., seq_len - residual_tokens :, :]
            return {
                "type": "split",
                "old": quantize_tensor_headwise(old, bits, method, side, layer_idx),
                "recent": {"type": "fp16", "data": recent.detach().to(MODEL_DTYPE).contiguous()},
            }

        return quantize_tensor_headwise(tensor, bits, method, side, layer_idx)

    def dequantize_state(q):
        if q["type"] == "split":
            old = dequantize_state(q["old"])
            recent = dequantize_state(q["recent"])
            return torch.cat([old, recent], dim=-2)

        if q["type"] == "sink_split":
            sink = dequantize_state(q["sink"])
            middle = dequantize_state(q["middle"])
            recent = dequantize_state(q["recent"])
            return torch.cat([sink, middle, recent], dim=-2)

        if q["type"] == "heads":
            return torch.cat([dequantize_state(hq) for hq in q["heads"]], dim=q.get("head_dim", 1))

        if q["type"] == "dim_blocks":
            return torch.cat([dequantize_state(piece["q"]) for piece in q["pieces"]], dim=-1)

        if q["type"] == "token_exceptions":
            x = dequantize_state(q["base"])
            idx = q["indices"].to(device=x.device, dtype=torch.long)
            vals = q["values"].to(device=x.device, dtype=x.dtype)
            x.index_copy_(-2, idx, vals)
            return x

        return dequantize_tensor(q)

    def _mapped_bits(method, side, layer_idx, default_bits):
        mapping = method.get(f"{side}_layer_bit_map", None)
        if mapping:
            if layer_idx in mapping:
                return int(mapping[layer_idx])
            if str(layer_idx) in mapping:
                return int(mapping[str(layer_idx)])
        return int(default_bits)

    def quantize_cache(cache, target_bits, method):
        kv_tuple = cache_to_tuple(cache)
        num_layers = len(kv_tuple)
        output = []
        for layer_idx, (k, v) in enumerate(kv_tuple):
            layer_bits = choose_layer_bits(target_bits, method, layer_idx, num_layers)
            k_default = int(method.get("k_target_bits", layer_bits))
            v_default = int(method.get("v_target_bits", layer_bits))
            k_bits = _mapped_bits(method, "k", layer_idx, k_default)
            v_bits = _mapped_bits(method, "v", layer_idx, v_default)
            output.append(
                (
                    quantize_tensor_with_residual(k, k_bits, method, "k", layer_idx),
                    quantize_tensor_with_residual(v, v_bits, method, "v", layer_idx),
                )
            )
        return tuple(output)

    def dequantize_cache_to_dynamic(stored_cache):
        layers = []
        for qk, qv in stored_cache:
            layers.append((dequantize_state(qk), dequantize_state(qv)))
        return tuple_to_dynamic_cache(tuple(layers))

    def stored_cache_kb(stored_cache):
        return sum(stored_tensor_bytes(q) for layer in stored_cache for q in layer) / 1024

    def stored_tensor_breakdown(q):
        out = {"payload": 0, "metadata": 0, "residual": 0, "auxiliary": 0}

        def add(other):
            for key in out:
                out[key] += int(other.get(key, 0))

        qtype = q["type"]
        if qtype == "fp16":
            out["residual"] += tensor_bytes(q["data"])
            return out
        if qtype in {"split", "sink_split"}:
            for key in ("old", "sink", "middle", "recent"):
                if key in q:
                    add(stored_tensor_breakdown(q[key]))
            return out
        if qtype == "heads":
            for item in q["heads"]:
                add(stored_tensor_breakdown(item))
            return out
        if qtype == "dim_blocks":
            for piece in q["pieces"]:
                add(stored_tensor_breakdown(piece["q"]))
            return out
        if qtype == "token_exceptions":
            add(stored_tensor_breakdown(q["base"]))
            out["payload"] = max(0, out["payload"] - int(q.get("saved_base_payload_bytes", 0)))
            out["auxiliary"] += tensor_bytes(q["indices"]) + tensor_bytes(q["values"])
            return out

        out["payload"] += tensor_bytes(q["packed"])
        if "codebook" in q:
            out["metadata"] += metadata_bytes(q["codebook"])
        elif "absmax" in q:
            out["metadata"] += metadata_bytes(q["absmax"])
        else:
            out["metadata"] += metadata_bytes(q["min"]) + metadata_bytes(q["scale"])
        if "token_absmax" in q:
            out["metadata"] += metadata_bytes(q["token_absmax"])
        return out

    def stored_cache_breakdown(stored_cache):
        total = {"payload": 0, "metadata": 0, "residual": 0, "auxiliary": 0}
        for layer in stored_cache:
            for q in layer:
                part = stored_tensor_breakdown(q)
                for key in total:
                    total[key] += int(part[key])
        return {key + "_kb": value / 1024 for key, value in total.items()}

    # =============================================================
    # Benchmark
    # =============================================================
    def run_one_trial(target_bits, input_ids, method):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        steps = input_ids.shape[1] - 1
        total_loss = 0.0
        total_tokens = 0
        kv_kb_now = 0.0
        kv_breakdown = {
            "payload_kb": 0.0,
            "metadata_kb": 0.0,
            "residual_kb": 0.0,
            "auxiliary_kb": 0.0,
        }

        cache_in = None
        stored_cache = None

        start = time.perf_counter()

        with torch.inference_mode():
            for i in range(steps):
                current = input_ids[:, i : i + 1]
                target = input_ids[:, i + 1 : i + 2]

                if target_bits == 16:
                    outputs = model(input_ids=current, past_key_values=cache_in, use_cache=True)
                    cache_in = outputs.past_key_values
                    kv_kb_now = fp_cache_kb(cache_in)
                else:
                    cache_arg = None if stored_cache is None else dequantize_cache_to_dynamic(stored_cache)
                    outputs = model(input_ids=current, past_key_values=cache_arg, use_cache=True)
                    stored_cache = quantize_cache(outputs.past_key_values, target_bits, method)
                    kv_kb_now = stored_cache_kb(stored_cache)
                    kv_breakdown = stored_cache_breakdown(stored_cache)
                    del cache_arg

                logits = outputs.logits[:, -1, :]
                loss = F.cross_entropy(logits.float(), target.reshape(-1))
                total_loss += float(loss.item())
                total_tokens += 1

                del outputs, logits, loss

        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000
        avg_loss = total_loss / total_tokens

        del cache_in, stored_cache
        gc.collect()
        torch.cuda.empty_cache()

        return {
            "loss": avg_loss,
            "ppl": math.exp(min(avg_loss, 50)),
            "pure_kv_kb": kv_kb_now,
            "pure_kv_mb": kv_kb_now / 1024,
            "payload_kb": kv_breakdown["payload_kb"],
            "metadata_kb": kv_breakdown["metadata_kb"],
            "residual_kb": kv_breakdown["residual_kb"],
            "auxiliary_kb": kv_breakdown["auxiliary_kb"],
            "ms": elapsed_ms,
        }

    def summarize_trials(target_bits, method, trials, fp16_kv_kb, fp16_loss):
        losses = [r["loss"] for r in trials]
        median_loss = statistics.median(losses)
        pure_kv_kb = statistics.median(r["pure_kv_kb"] for r in trials)
        ppl = math.exp(min(median_loss, 50))
        return {
            "bits": target_bits,
            "method_id": method["id"],
            "method": method["name"],
            "loss": median_loss,
            "loss_stdev": statistics.stdev(losses) if len(losses) > 1 else 0.0,
            "ppl": ppl,
            "ppl_delta": ppl - math.exp(min(fp16_loss, 50)),
            "pure_kv_kb": pure_kv_kb,
            "pure_kv_mb": pure_kv_kb / 1024,
            "compression": fp16_kv_kb / pure_kv_kb if pure_kv_kb > 0 else float("inf"),
            "payload_kb": statistics.median(r.get("payload_kb", 0.0) for r in trials),
            "metadata_kb": statistics.median(r.get("metadata_kb", 0.0) for r in trials),
            "residual_kb": statistics.median(r.get("residual_kb", 0.0) for r in trials),
            "auxiliary_kb": statistics.median(r.get("auxiliary_kb", 0.0) for r in trials),
            "ms": statistics.median(r["ms"] for r in trials),
            "note": method["note"],
        }

    fp16_method = METHODS[0]
    phase4_methods = [m for m in METHODS if m["id"] != "p3_uniform"]
    p3_method = METHODS[0]

    # =============================================================
    # Adaptive-overhead RABIT-KV bit sweep
    # =============================================================
    method_by_id = {m["id"]: m for m in METHODS}

    def clone_method(base_id, new_id, new_name, note, **updates):
        base = dict(method_by_id[base_id])
        base.update({"id": new_id, "name": new_name, "note": note})
        base.update(updates)
        return base

    def make_profile_quant_method():
        return clone_method(
            "s1_group",
            "llama31_profile_sym_g32",
            "Llama-3.1 profile SYM G32",
            "target-model KV profiling method",
            k_group=32,
            v_group=32,
            residual_tokens=0,
            layer_bit_map={},
            head_bit_map={},
            protected_layers=None,
            protected_bits=4,
            symmetric_absmax=True,
            per_token_scale=False,
        )

    def normalized_quant_error_for_head(tensor, bits, method, side):
        q = quantize_tensor(tensor, bits, method, side)
        dq = dequantize_tensor(q).float()
        x = tensor.detach().float()
        mse = (dq - x).square().mean()
        energy = x.square().mean().clamp(min=1e-12)
        return float((mse / energy).detach().cpu().item())

    def profile_llama31_head_map(profile_input_windows, target_bits=2, top_heads=24, top_4bit_heads=8, max_layers=8):
        # Target-model profiling:
        # Build the layer/head protection map directly from Llama-3.1 KV tensors.
        # v14 changes the selection from raw sensitivity to benefit-per-byte with a layer-spread cap.
        if top_heads <= 0:
            return {}, []

        profile_method = make_profile_quant_method()
        scores = {}

        with torch.inference_mode():
            for input_ids in profile_input_windows:
                outputs = model(input_ids=input_ids, use_cache=True)
                kv_tuple = cache_to_tuple(outputs.past_key_values)

                for layer_idx, (k, v) in enumerate(kv_tuple):
                    if k.dim() < 4 or v.dim() < 4:
                        continue

                    num_heads = int(k.shape[1])
                    for head_idx in range(num_heads):
                        kh = k[:, head_idx : head_idx + 1, ...]
                        vh = v[:, head_idx : head_idx + 1, ...]

                        higher_bits = min(4, int(target_bits) + 1)
                        k_err_low = normalized_quant_error_for_head(kh, target_bits, profile_method, "k")
                        v_err_low = normalized_quant_error_for_head(vh, target_bits, profile_method, "v")
                        k_err_high = normalized_quant_error_for_head(kh, higher_bits, profile_method, "k")
                        v_err_high = normalized_quant_error_for_head(vh, higher_bits, profile_method, "v")

                        benefit = 1.5 * max(0.0, k_err_low - k_err_high) + max(0.0, v_err_low - v_err_high)

                        # Approximate extra payload cost of upgrading this head by one bit.
                        head_elems = kh.numel() + vh.numel()
                        extra_bytes = max(1.0, head_elems * max(1, higher_bits - int(target_bits)) / 8.0)
                        benefit_per_byte = benefit / extra_bytes
                        key = (int(layer_idx), int(head_idx))
                        scores[key] = scores.get(key, 0.0) + benefit_per_byte

                del outputs, kv_tuple

        by_layer = {}
        for (layer, head), score in scores.items():
            by_layer.setdefault(layer, []).append({"layer": layer, "head": head, "score": score})

        # Layer-spread cap: avoid protecting 24 heads across 15 layers, because later residual
        # policies pay partly at layer granularity. Prefer layers with multiple useful heads.
        layer_rank = []
        for layer, items in by_layer.items():
            items_sorted = sorted(items, key=lambda x: x["score"], reverse=True)
            keep = items_sorted[: max(1, int(math.ceil(top_heads / max(1, max_layers))))]
            layer_score = sum(x["score"] for x in keep) / math.sqrt(len(keep))
            layer_rank.append({"layer": layer, "layer_score": layer_score, "items": items_sorted})
        layer_rank.sort(key=lambda x: x["layer_score"], reverse=True)
        active_layers = set(x["layer"] for x in layer_rank[: max(1, int(max_layers))])

        ranked = []
        for layer in active_layers:
            ranked.extend(by_layer[layer])
        ranked = sorted(ranked, key=lambda x: x["score"], reverse=True)

        selected = ranked[: int(top_heads)]
        head_map = {}
        for rank, item in enumerate(selected):
            layer = int(item["layer"])
            head = int(item["head"])
            bit = 4 if rank < int(top_4bit_heads) else 3
            head_map.setdefault(layer, {})[head] = bit

        return head_map, selected

    # Llama-3.1 final setting:
    # RABIT-v13 is evaluated here as a Llama-3.1 improvement family:
    # group-wise symmetric quantization, residual windows, RoPE-aware K-block protection,
    # V outlier-token correction, and a target-model profiled layer/head map.
    pruned_head_map = {}
    profiled_head_rank = []

    def make_global_r0():
        return clone_method(
            "p3_uniform",
            "rabit_global_r0",
            "global R0",
            "global min/max, no residual, no head protection",
            residual_tokens=0,
            head_bit_map={},
            layer_bit_map={},
        )

    def make_asym(k_group=32, v_group=64, residual=0, pruned=False):
        name = f"K{k_group}V{v_group} R{residual}"
        note = f"K group{k_group}, V group{v_group}, newest {residual} fp16"
        if residual == 0:
            note = f"K group{k_group}, V group{v_group}, no residual"
        if pruned:
            name += " pruned"
            note += "; pruned important heads protected only when target bit is lower"

        return clone_method(
            "s1_group",
            f"rabit_k{k_group}_v{v_group}_r{residual}" + ("_pruned" if pruned else ""),
            name,
            note,
            k_group=int(k_group),
            v_group=int(v_group),
            residual_tokens=int(residual),
            layer_bit_map={},
            head_bit_map=pruned_head_map if pruned else {},
            protected_layers=None,
            protected_bits=4,
        )

    def make_group(group_size=32, residual=0, pruned=False):
        return make_asym(group_size, group_size, residual=residual, pruned=pruned)

    def make_asym_sym(k_group=32, v_group=64, residual=0, pruned=False):
        name = f"SYM K{k_group}V{v_group} R{residual}"
        note = f"symmetric absmax; K group{k_group}, V group{v_group}, newest {residual} fp16"
        if residual == 0:
            note = f"symmetric absmax; K group{k_group}, V group{v_group}, no residual"
        if pruned:
            name += " pruned"
            note += "; pruned important heads protected only when target bit is lower"

        return clone_method(
            "s1_group",
            f"rabit_sym_k{k_group}_v{v_group}_r{residual}" + ("_pruned" if pruned else ""),
            name,
            note,
            k_group=int(k_group),
            v_group=int(v_group),
            residual_tokens=int(residual),
            layer_bit_map={},
            head_bit_map=pruned_head_map if pruned else {},
            protected_layers=None,
            protected_bits=4,
            symmetric_absmax=True,
        )

    def make_group_sym(group_size=32, residual=0, pruned=False):
        return make_asym_sym(group_size, group_size, residual=residual, pruned=pruned)

    def make_hetero_kseq_vtoken(k_seq_group=32, v_group=32, residual=0, pruned=False):
        # v14: K uses sequence-axis/channel-wise grouping; V keeps token-wise dim grouping.
        name = f"HET Kseq{k_seq_group} Vtok{v_group} R{residual}"
        note = (
            f"hetero K/V: K seq-group {k_seq_group} per channel, "
            f"V token-dim group {v_group}, newest {residual} fp16"
        )
        if residual == 0:
            note = f"hetero K/V: K seq-group {k_seq_group} per channel, V token-dim group {v_group}, no residual"
        if pruned:
            name += " profiled"
            note += "; target-model profiled heads protected when target bit is lower"
        return clone_method(
            "s1_group",
            f"rabit_v14_hetero_kseq{k_seq_group}_vtok{v_group}_r{residual}" + ("_profiled" if pruned else ""),
            name,
            note,
            k_group=int(k_seq_group),
            v_group=int(v_group),
            k_seq_group=int(k_seq_group),
            k_quant_axis="seq",
            v_quant_axis="token",
            residual_tokens=int(residual),
            layer_bit_map={},
            head_bit_map=pruned_head_map if pruned else {},
            protected_layers=None,
            protected_bits=4,
            symmetric_absmax=True,
        )

    def make_v16_kivi_affine(
        k_bits=2,
        v_bits=2,
        k_seq_group=32,
        v_group=32,
        residual=4,
        sink_tokens=0,
        v_outlier_tokens=0,
        v_affine=True,
        pruned=False,
        label="K affine sequence, V token",
    ):
        # v16: KIVI-style K per-channel affine quantization.
        # K uses sequence-axis affine min/scale groups. V uses per-token dim grouping.
        # v_affine=True makes V use affine min/scale instead of symmetric absmax.
        name = f"V16-KIVI K{k_bits}V{v_bits} Kseq{k_seq_group} Vtok{v_group} R{residual}"
        if sink_tokens > 0:
            name += f" S{sink_tokens}"
        if v_outlier_tokens > 0:
            name += f" X{v_outlier_tokens}"
        note = (
            f"v16 {label}: K seq-affine {k_bits}b group {k_seq_group}; "
            f"V token {'affine' if v_affine else 'symmetric'} {v_bits}b group {v_group}; "
            f"recent R{residual}; sink {sink_tokens}; V outlier tokens {v_outlier_tokens}"
        )
        if pruned:
            name += " profiled"
            note += "; profiled heads protected when target bit is lower"

        return clone_method(
            "s1_group",
            f"rabit_v16_kivi_k{k_bits}_v{v_bits}_kseq{k_seq_group}_vtok{v_group}_r{residual}_s{sink_tokens}_x{v_outlier_tokens}" + ("_profiled" if pruned else ""),
            name,
            note,
            k_group=int(k_seq_group),
            v_group=int(v_group),
            k_seq_group=int(k_seq_group),
            k_quant_axis="seq_affine",
            v_quant_axis="token",
            residual_tokens=int(residual),
            sink_tokens=int(sink_tokens),
            sink_bits=16,
            v_outlier_tokens=int(v_outlier_tokens),
            k_target_bits=int(k_bits),
            v_target_bits=int(v_bits),
            layer_bit_map={},
            head_bit_map=pruned_head_map if pruned else {},
            protected_layers=None,
            protected_bits=max(int(k_bits), int(v_bits), 3),
            symmetric_absmax=not bool(v_affine),
        )


    def make_v18_k3v2_meta(
        k_axis="seq_affine",
        v_affine=True,
        k_seq_group=32,
        v_group=32,
        residual=4,
        sink_tokens=0,
        v_outlier_tokens=0,
        pruned=False,
        label="metadata sweep",
    ):
        # V18 main idea:
        # The best previous point was K3V2, but memory was inflated by metadata.
        # This candidate family keeps the K3V2 allocation and sweeps metadata cost:
        #   k_axis="seq_affine": K stores min+scale per sequence group.
        #   k_axis="seq":        K stores one symmetric absmax per sequence group.
        #   v_affine=True:       V stores min+scale per group.
        #   v_affine=False:      V stores one symmetric absmax per group.
        name = f"V18-META K3V2 K{str(k_axis).replace('seq_', '')}{k_seq_group} V{'aff' if v_affine else 'sym'}{v_group} R{residual}"
        if sink_tokens > 0:
            name += f" S{sink_tokens}"
        if v_outlier_tokens > 0:
            name += f" X{v_outlier_tokens}"
        if pruned:
            name += " profiled"

        note = (
            f"v18 {label}: K=3b using {k_axis} group {k_seq_group}; "
            f"V=2b using {'affine min+scale' if v_affine else 'symmetric absmax'} group {v_group}; "
            f"recent R{residual}; sink {sink_tokens}; V outlier tokens {v_outlier_tokens}"
        )

        return clone_method(
            "s1_group",
            f"rabit_v18_meta_k3v2_{k_axis}_k{k_seq_group}_v{'aff' if v_affine else 'sym'}{v_group}_r{residual}_s{sink_tokens}_x{v_outlier_tokens}" + ("_profiled" if pruned else ""),
            name,
            note,
            k_group=int(k_seq_group),
            v_group=int(v_group),
            k_seq_group=int(k_seq_group),
            k_quant_axis=str(k_axis),
            v_quant_axis="token",
            residual_tokens=int(residual),
            sink_tokens=int(sink_tokens),
            sink_bits=16,
            v_outlier_tokens=int(v_outlier_tokens),
            k_target_bits=3,
            v_target_bits=2,
            layer_bit_map={},
            head_bit_map=pruned_head_map if pruned else {},
            protected_layers=None,
            protected_bits=3,
            symmetric_absmax=not bool(v_affine),
        )

    def make_v17_nuq_codebook(
        k_bits=2,
        v_bits=2,
        k_seq_group=128,
        v_group=128,
        residual=0,
        sink_tokens=0,
        pruned=False,
        k_axis="seq_codebook",
        codebook_sides="kv",
        label="codebook NUQ",
    ):
        # v17: non-uniform/codebook quantization.
        # Each group stores a small learned-from-data centroid table instead of uniform min/max levels.
        # Codes are still packed to the requested bit width; for 2-bit, each group has 4 centroids.
        name = f"V17-NUQ K{k_bits}V{v_bits} Kseq{k_seq_group} Vg{v_group} R{residual}"
        if sink_tokens > 0:
            name += f" S{sink_tokens}"
        if pruned:
            name += " profiled"
        note = (
            f"v17 {label}: K axis={k_axis}, K={k_bits}b group {k_seq_group}; "
            f"V={v_bits}b group {v_group}; residual R{residual}; sink {sink_tokens}; "
            f"codebook sides={codebook_sides}"
        )
        if pruned:
            note += "; profiled heads protected when target bit is lower"

        return clone_method(
            "s1_group",
            f"rabit_v17_nuq_k{k_bits}_v{v_bits}_kseq{k_seq_group}_vg{v_group}_r{residual}_s{sink_tokens}_{codebook_sides}" + ("_profiled" if pruned else ""),
            name,
            note,
            k_group=int(k_seq_group),
            v_group=int(v_group),
            k_seq_group=int(k_seq_group),
            k_quant_axis=str(k_axis),
            v_quant_axis="token",
            residual_tokens=int(residual),
            sink_tokens=int(sink_tokens),
            sink_bits=16,
            k_target_bits=int(k_bits),
            v_target_bits=int(v_bits),
            layer_bit_map={},
            head_bit_map=pruned_head_map if pruned else {},
            protected_layers=None,
            protected_bits=max(int(k_bits), int(v_bits), 3),
            codebook_quant=True,
            codebook_sides=str(codebook_sides),
            symmetric_absmax=False,
        )

    def make_selective_residual(
        k_group=32,
        v_group=64,
        protected_residual=16,
        other_residual=0,
        pruned=True,
        symmetric=False,
    ):
        # v13: selective residual/correction budget.
        # Instead of giving every layer the same fp16 recent window, selected layers can
        # receive a larger residual correction while other layers use a smaller budget.
        protected_layers = sorted(pruned_head_map.keys())
        layer_residual_map = {int(layer): int(protected_residual) for layer in protected_layers}

        prefix = "SEL SYM" if symmetric else "SEL"
        name = f"{prefix} K{k_group}V{v_group} P{protected_residual}O{other_residual}"
        note = (
            f"selective residual: protected layers R{protected_residual}, "
            f"other layers R{other_residual}; K group{k_group}, V group{v_group}"
        )
        if symmetric:
            note = "symmetric absmax; " + note
        if pruned:
            name += " pruned"
            note += "; pruned important heads protected only when target bit is lower"

        return clone_method(
            "s1_group",
            f"rabit_sel_{'sym_' if symmetric else ''}k{k_group}_v{v_group}_p{protected_residual}_o{other_residual}" + ("_pruned" if pruned else ""),
            name,
            note,
            k_group=int(k_group),
            v_group=int(v_group),
            residual_tokens=int(other_residual),
            layer_residual_map=layer_residual_map,
            layer_bit_map={},
            head_bit_map=pruned_head_map if pruned else {},
            protected_layers=None,
            protected_bits=4,
            symmetric_absmax=bool(symmetric),
        )

    def make_selective_group(group_size=32, protected_residual=16, other_residual=0, pruned=True, symmetric=False):
        return make_selective_residual(
            group_size,
            group_size,
            protected_residual=protected_residual,
            other_residual=other_residual,
            pruned=pruned,
            symmetric=symmetric,
        )

    def make_tiered_selective_residual(
        k_group=32,
        v_group=32,
        high_residual=16,
        medium_residual=12,
        other_residual=8,
        pruned=True,
        symmetric=False,
    ):
        # v13: tiered selective residual budget.
        # Different sensitivity tiers can receive different fp16 recent-window budgets:
        #   high-sensitive layers get H residual
        #   medium-sensitive layers get M residual
        #   other layers get O residual.
        high_layers = []
        medium_layers = []
        for layer, heads in pruned_head_map.items():
            num_heads = len(heads)
            num_4bit = sum(1 for b in heads.values() if int(b) >= 4)
            score = float(num_heads) + 0.5 * float(num_4bit)
            if score >= 4.0:
                high_layers.append(int(layer))
            else:
                medium_layers.append(int(layer))

        layer_residual_map = {}
        for layer in high_layers:
            layer_residual_map[int(layer)] = int(high_residual)
        for layer in medium_layers:
            layer_residual_map[int(layer)] = int(medium_residual)

        prefix = "TIER SYM" if symmetric else "TIER"
        name = f"{prefix} K{k_group}V{v_group} H{high_residual}M{medium_residual}O{other_residual}"
        note = (
            f"tiered selective residual: high layers R{high_residual}, "
            f"medium layers R{medium_residual}, other layers R{other_residual}; "
            f"high={high_layers}, medium={medium_layers}; K group{k_group}, V group{v_group}"
        )
        if symmetric:
            note = "symmetric absmax; " + note
        if pruned:
            name += " pruned"
            note += "; pruned important heads protected only when target bit is lower"

        return clone_method(
            "s1_group",
            f"rabit_tier_{'sym_' if symmetric else ''}k{k_group}_v{v_group}_h{high_residual}_m{medium_residual}_o{other_residual}" + ("_pruned" if pruned else ""),
            name,
            note,
            k_group=int(k_group),
            v_group=int(v_group),
            residual_tokens=int(other_residual),
            layer_residual_map=layer_residual_map,
            layer_bit_map={},
            head_bit_map=pruned_head_map if pruned else {},
            protected_layers=None,
            protected_bits=4,
            symmetric_absmax=bool(symmetric),
        )

    def make_tiered_group(group_size=32, high_residual=16, medium_residual=12, other_residual=8, pruned=True, symmetric=False):
        return make_tiered_selective_residual(
            group_size,
            group_size,
            high_residual=high_residual,
            medium_residual=medium_residual,
            other_residual=other_residual,
            pruned=pruned,
            symmetric=symmetric,
        )

    def split_high_medium_layers():
        high_layers = []
        medium_layers = []
        for layer, heads in pruned_head_map.items():
            num_heads = len(heads)
            num_4bit = sum(1 for b in heads.values() if int(b) >= 4)
            score = float(num_heads) + 0.5 * float(num_4bit)
            if score >= 4.0:
                high_layers.append(int(layer))
            else:
                medium_layers.append(int(layer))
        return sorted(high_layers), sorted(medium_layers)

    def capped_head_map(policy="all3"):
        # v13: head-protection compression refinement.
        # v13 tests softer protection:
        #   all3       = every protected head capped at 3-bit
        #   high4med3  = high-sensitive layers keep original protection, medium layers capped at 3-bit
        #   high3med2  = high-sensitive layers capped at 3-bit, medium layers removed back to target bit
        high_layers, medium_layers = split_high_medium_layers()
        high_set = set(high_layers)
        medium_set = set(medium_layers)

        out = {}
        for layer, heads in pruned_head_map.items():
            layer_i = int(layer)
            new_heads = {}
            for h, b in heads.items():
                b_i = int(b)
                if policy == "all3":
                    new_b = min(b_i, 3)
                elif policy == "high4med3":
                    new_b = b_i if layer_i in high_set else min(b_i, 3)
                elif policy == "high3med2":
                    if layer_i in high_set:
                        new_b = min(b_i, 3)
                    else:
                        new_b = 2
                else:
                    new_b = b_i

                if new_b > 2:
                    new_heads[int(h)] = int(new_b)
            if new_heads:
                out[layer_i] = new_heads
        return out

    def make_tiered_compressed_residual(
        k_group=32,
        v_group=32,
        high_residual=14,
        medium_residual=10,
        other_residual=8,
        head_policy="all3",
        symmetric=False,
    ):
        # v13: combine tiered residual budgets with softer/capped head protection.
        # This directly targets the weird lower-bit behavior:
        #   3-bit compressed more than 2-bit because 2-bit used too much correction overhead.
        #
        # The goal is to make 2-bit more compressed while keeping PPL in a reasonable band.
        high_layers, medium_layers = split_high_medium_layers()
        layer_residual_map = {}
        for layer in high_layers:
            layer_residual_map[int(layer)] = int(high_residual)
        for layer in medium_layers:
            layer_residual_map[int(layer)] = int(medium_residual)

        head_map = capped_head_map(head_policy)

        prefix = "TIER-CAP SYM" if symmetric else "TIER-CAP"
        name = f"{prefix} K{k_group}V{v_group} H{high_residual}M{medium_residual}O{other_residual} {head_policy}"
        note = (
            f"tiered residual + capped head protection ({head_policy}): "
            f"high layers R{high_residual}, medium layers R{medium_residual}, other layers R{other_residual}; "
            f"high={high_layers}, medium={medium_layers}; K group{k_group}, V group{v_group}"
        )
        if symmetric:
            note = "symmetric absmax; " + note

        return clone_method(
            "s1_group",
            f"rabit_tiercap_{head_policy}_{'sym_' if symmetric else ''}k{k_group}_v{v_group}_h{high_residual}_m{medium_residual}_o{other_residual}",
            name,
            note,
            k_group=int(k_group),
            v_group=int(v_group),
            residual_tokens=int(other_residual),
            layer_residual_map=layer_residual_map,
            layer_bit_map={},
            head_bit_map=head_map,
            protected_layers=None,
            protected_bits=3,
            symmetric_absmax=bool(symmetric),
        )

    def make_tiered_compressed_group(group_size=32, high_residual=14, medium_residual=10, other_residual=8, head_policy="all3", symmetric=False):
        return make_tiered_compressed_residual(
            group_size,
            group_size,
            high_residual=high_residual,
            medium_residual=medium_residual,
            other_residual=other_residual,
            head_policy=head_policy,
            symmetric=symmetric,
        )

    def make_v13_structured(
        k_group=32,
        v_group=32,
        high_residual=12,
        medium_residual=8,
        other_residual=6,
        head_policy="all3",
        symmetric=True,
        k_rope=True,
        k_block_size=16,
        k_top_blocks=1,
        k_top_bits=3,
        v_outlier_tokens=1,
        label="full",
    ):
        # v13: structured protection.
        # K gets RoPE-aware block protection. V gets a tiny outlier-token exception list.
        # Tiered residual remains as a safety net, but can be smaller than v13.
        high_layers, medium_layers = split_high_medium_layers()
        layer_residual_map = {}
        for layer in high_layers:
            layer_residual_map[int(layer)] = int(high_residual)
        for layer in medium_layers:
            layer_residual_map[int(layer)] = int(medium_residual)

        head_map = capped_head_map(head_policy)

        prefix = "V13"
        if k_rope and v_outlier_tokens > 0:
            family = "FULL"
        elif k_rope:
            family = "KBLK"
        elif v_outlier_tokens > 0:
            family = "TOK"
        else:
            family = "TIER"

        sym_txt = " SYM" if symmetric else ""
        name = (
            f"{prefix}-{family}{sym_txt} G{k_group} "
            f"B{k_block_size}T{k_top_blocks}b{k_top_bits} "
            f"H{high_residual}M{medium_residual}O{other_residual} X{v_outlier_tokens}"
        )
        note = (
            f"v13 {label}: RoPE-aware K-blocks={bool(k_rope)} "
            f"(block={k_block_size}, top={k_top_blocks}, bit={k_top_bits}); "
            f"V outlier tokens={v_outlier_tokens}; "
            f"tiered residual high R{high_residual}, medium R{medium_residual}, other R{other_residual}; "
            f"head policy={head_policy}; high={high_layers}, medium={medium_layers}"
        )
        if symmetric:
            note = "symmetric absmax; " + note

        return clone_method(
            "s1_group",
            f"rabit_v13_{family.lower()}_{'sym_' if symmetric else ''}g{k_group}_b{k_block_size}_t{k_top_blocks}_b{k_top_bits}_h{high_residual}_m{medium_residual}_o{other_residual}_x{v_outlier_tokens}_{head_policy}",
            name,
            note,
            k_group=int(k_group),
            v_group=int(v_group),
            residual_tokens=int(other_residual),
            layer_residual_map=layer_residual_map,
            layer_bit_map={},
            head_bit_map=head_map,
            protected_layers=None,
            protected_bits=3,
            symmetric_absmax=bool(symmetric),
            k_rope_blocks=bool(k_rope),
            k_rope_block_size=int(k_block_size),
            k_rope_top_blocks=int(k_top_blocks),
            k_rope_top_bits=int(k_top_bits),
            v_outlier_tokens=int(v_outlier_tokens),
        )

    def make_v14_hetero_structured(k_seq_group=32, **kwargs):
        # v14 = v13 structured protection + K sequence-axis quantization.
        method = make_v13_structured(**kwargs)
        method = dict(method)
        method["id"] = "rabit_v14_hetero_" + method["id"]
        method["name"] = "V14-HET " + method["name"]
        method["note"] = "v14 hetero K sequence-axis quantization; " + method.get("note", "")
        method["k_quant_axis"] = "seq"
        method["k_seq_group"] = int(k_seq_group)
        return method

    def make_v15_sink_mixedkv(
        k_bits=3,
        v_bits=2,
        k_group=32,
        v_group=32,
        residual=2,
        sink_tokens=8,
        sink_bits=16,
        symmetric=True,
        pruned=False,
        label="sink mixed K/V",
    ):
        # v15: low-cost attention-sink protection + mixed K/V bit allocation.
        # This attacks the actual 2-bit problem: previous "2-bit" rows needed so much
        # correction that they became pseudo-3-bit in memory.
        name = f"V15-SINK MIX K{k_bits}V{v_bits} G{k_group}/{v_group} S{sink_tokens} R{residual}"
        note = (
            f"v15 {label}: K={k_bits}b, V={v_bits}b, sink first {sink_tokens} tokens at {sink_bits}b, "
            f"recent R{residual}; K group{k_group}, V group{v_group}"
        )
        if symmetric:
            note = "symmetric absmax; " + note
        if pruned:
            name += " pruned"
            note += "; profiled heads protected when target bit is lower"

        return clone_method(
            "s1_group",
            f"rabit_v15_sink_mix_k{k_bits}_v{v_bits}_g{k_group}_{v_group}_s{sink_tokens}_r{residual}" + ("_pruned" if pruned else ""),
            name,
            note,
            k_group=int(k_group),
            v_group=int(v_group),
            residual_tokens=int(residual),
            sink_tokens=int(sink_tokens),
            sink_bits=int(sink_bits),
            k_target_bits=int(k_bits),
            v_target_bits=int(v_bits),
            layer_bit_map={},
            head_bit_map=pruned_head_map if pruned else {},
            protected_layers=None,
            protected_bits=max(int(k_bits), int(v_bits), 3),
            symmetric_absmax=bool(symmetric),
        )

    def make_v15_sink_v13_full(
        residual=2,
        sink_tokens=8,
        sink_bits=16,
        high_residual=8,
        medium_residual=6,
        other_residual=2,
        head_policy="all3",
        k_top_blocks=1,
        k_top_bits=3,
        v_outlier_tokens=1,
    ):
        # v15: v13 FULL structure but with smaller residual overhead and explicit sink protection.
        method = make_v13_structured(
            32, 32,
            high_residual, medium_residual, other_residual,
            head_policy,
            True,
            True,
            16,
            k_top_blocks,
            k_top_bits,
            v_outlier_tokens,
            "v15 sink + lower residual full",
        )
        method = dict(method)
        method["id"] = "rabit_v15_sink_" + method["id"]
        method["name"] = f"V15-SINK {method['name']} S{sink_tokens}"
        method["note"] = f"v15 sink first {sink_tokens} tokens at {sink_bits}b; " + method.get("note", "")
        method["sink_tokens"] = int(sink_tokens)
        method["sink_bits"] = int(sink_bits)
        method["residual_tokens"] = int(residual)
        return method

    # =============================================================
    # RABIT-KV: the unified configuration space
    # =============================================================
    #
    # RABIT-KV is ONE method. It is not a collection of versions.
    #
    # The method exposes a single configuration space with 7 axes. Every
    # historical "version" (v13..v18) is simply a point in this space:
    #
    #   A. k_axis        how K is grouped        {channel, seq, seq_affine, seq_codebook}
    #   B. v_axis        how V is grouped        {token, token_affine, token_codebook}
    #   C. group sizes   metadata granularity    k_group / v_group in {32, 64, 128}
    #   D. k_bits/v_bits per-side bit allocation (K and V need not be equal)
    #   E. residual      fp16 recent window      R in {0, 2, 4, 8, 16}
    #   F. sink          fp16 attention sink     S in {0, 4, 8}
    #   G. protection    profiled head/block/token protection (on/off)
    #
    # Given a target bit budget, RABIT-KV enumerates the region of this space
    # that is feasible at that budget and evaluates it. The axes are the method;
    # the enumerated points are the search.
    #
    # The mapping from the old version names to these axes (for the ablation
    # table in the paper) is recorded in COMPONENT_ORIGIN below.

    COMPONENT_ORIGIN = {
        "group_sym":      "symmetric metadata-reduced scaling",
        "kv_asym":        "separate K and V grouping",
        "k_seq":          "K grouped along sequence axis (not channel)",
        "k_affine":       "affine (zero-point) K scaling",
        "codebook":       "non-uniform codebook quantization",
        "residual":       "fp16 recent-token window",
        "sink":           "fp16 attention-sink tokens",
        "mixed_kv_bits":  "asymmetric K/V bit allocation",
        "profiled":       "target-model head/layer sensitivity protection",
        "k_block":        "RoPE-aware K block protection",
        "v_outlier":      "sparse V outlier-token correction",
    }

    def rabit_config(
        target_bits,
        k_bits=None,
        v_bits=None,
        k_axis="channel",          # channel | seq | seq_affine | seq_codebook
        v_axis="token",            # token | token_affine | token_codebook
        k_group=32,
        v_group=32,
        residual=0,
        sink=0,
        symmetric=True,
        protect=False,             # profiled head protection
        k_block=False,             # RoPE-aware K block protection
        v_outlier=0,               # sparse V outlier tokens
        label="",
    ):
        """Instantiate ONE RABIT-KV configuration.

        This is the single constructor for the method. It dispatches to the
        underlying quantization kernels, but the *method identity* is this
        function: RABIT-KV(target_bits, axes...).
        """
        kb = int(k_bits if k_bits is not None else target_bits)
        vb = int(v_bits if v_bits is not None else target_bits)

        # --- Route to the kernel family implied by the axes ---------------
        if k_axis == "seq_codebook" or v_axis == "token_codebook":
            m = make_v17_nuq_codebook(
                kb, vb, k_group, v_group,
                residual=residual, pruned=protect,
                k_axis="seq_codebook", codebook_sides="kv",
                label=label,
            )
        elif k_axis == "seq_affine":
            if sink > 0:
                m = make_v16_kivi_affine(
                    kb, vb, k_group, v_group,
                    residual=residual, sink_tokens=sink,
                    v_affine=(v_axis == "token_affine"),
                    v_outlier_tokens=v_outlier, label=label,
                )
            else:
                m = make_v18_k3v2_meta(
                    "seq_affine", (v_axis == "token_affine"),
                    k_group, v_group, residual=residual, label=label,
                )
        elif k_axis == "seq":
            if k_block or v_outlier:
                m = make_v14_hetero_structured(
                    k_seq_group=k_group, k_group=k_group, v_group=v_group,
                    high_residual=max(residual, 8), medium_residual=max(residual, 6),
                    other_residual=residual,
                    head_policy="high4med3" if protect else "all3",
                    symmetric=symmetric, k_rope=k_block, k_block_size=16,
                    k_top_blocks=1 if k_block else 0, k_top_bits=min(kb + 1, 4),
                    v_outlier_tokens=v_outlier, label=label,
                )
            elif kb != vb or v_axis == "token_affine":
                m = make_v18_k3v2_meta(
                    "seq", (v_axis == "token_affine"),
                    k_group, v_group, residual=residual, label=label,
                )
            else:
                m = make_hetero_kseq_vtoken(k_group, v_group, residual, pruned=protect)
        elif sink > 0:
            m = make_v15_sink_mixedkv(
                kb, vb, k_group, v_group,
                residual=residual, sink_tokens=sink, sink_bits=16,
                pruned=protect, label=label,
            )
        elif k_block or v_outlier:
            m = make_v13_structured(
                k_group, v_group,
                max(residual, 10), max(residual, 8), residual,
                "high4med3" if protect else "all3",
                symmetric, k_block, 16,
                1 if k_block else 0, min(kb + 1, 4),
                v_outlier, label,
            )
        elif k_group != v_group:
            m = (make_asym_sym if symmetric else make_asym)(k_group, v_group, residual, pruned=protect)
        else:
            m = (make_group_sym if symmetric else make_group)(k_group, residual, pruned=protect)

        m = dict(m)
        m["rabit_axes"] = {
            "target_bits": int(target_bits),
            "k_bits": kb, "v_bits": vb,
            "k_axis": k_axis, "v_axis": v_axis,
            "k_group": k_group, "v_group": v_group,
            "residual": residual, "sink": sink,
            "symmetric": symmetric, "protect": protect,
            "k_block": k_block, "v_outlier": v_outlier,
        }
        if label:
            m["rabit_label"] = str(label)
        return m

    def config_signature(m):
        """Compact, method-agnostic name for a RABIT-KV configuration.

        This is what goes in the paper's 'Selected configuration' column.
        It describes axes, never a version number.
        """
        if m.get("rabit_label"):
            return str(m["rabit_label"])
        a = m.get("rabit_axes", {})
        if not a:
            return "fp16"
        parts = []
        kb, vb = a["k_bits"], a["v_bits"]
        parts.append(f"K{kb}V{vb}" if kb != vb else f"{kb}b")

        kmap = {"channel": "Kch", "seq": "Kseq", "seq_affine": "Kaff", "seq_codebook": "Knuq"}
        vmap = {"token": "Vtok", "token_affine": "Vaff", "token_codebook": "Vnuq"}
        parts.append(f"{kmap[a['k_axis']]}{a['k_group']}")
        parts.append(f"{vmap[a['v_axis']]}{a['v_group']}")

        if a["symmetric"] and a["k_axis"] == "channel":
            parts.append("SYM")
        if a["residual"]:
            parts.append(f"R{a['residual']}")
        if a["sink"]:
            parts.append(f"S{a['sink']}")
        if a["protect"]:
            parts.append("P")
        if a["k_block"]:
            parts.append("KB")
        if a["v_outlier"]:
            parts.append(f"X{a['v_outlier']}")
        return " ".join(parts)

    # -------------------------------------------------------------
    # Step 3 of the pipeline: generate the feasible configuration
    # region for a given target bit budget.
    #
    # This is ONE function for ALL bit budgets. The budget controls
    # *which region of the axis space is worth searching*, because the
    # binding constraint changes with the budget:
    #
    #   high bits (8) -> quantization error is tiny; metadata overhead
    #                    dominates. Search coarse groups, no residual.
    #   mid bits (4,3)-> error and metadata are comparable. Search
    #                    grouping + small residual + protection.
    #   low bits (2)  -> error dominates and collapses the model.
    #                    Search axis changes (seq/affine), asymmetric
    #                    K/V bits, sinks, and residual stabilization.
    # -------------------------------------------------------------
    def profile_k_layer_sensitivity(profile_input_windows):
        """Rank layers by the normalized-error benefit of K3 over K2.

        The ranking is used only to construct adaptive candidates. Final PPL is
        still measured by the normal sequential cache benchmark.
        """
        method = make_v18_k3v2_meta(
            k_axis="seq_affine",
            v_affine=True,
            k_seq_group=32,
            v_group=32,
            residual=0,
            label="layer sensitivity profile",
        )
        method = dict(method)
        method["metadata_mode"] = "bf16"
        scores = {}
        counts = {}

        with torch.inference_mode():
            for input_ids in profile_input_windows:
                outputs = model(input_ids=input_ids, use_cache=True)
                kv_tuple = cache_to_tuple(outputs.past_key_values)
                for layer_idx, (k, _v) in enumerate(kv_tuple):
                    q2 = quantize_tensor(k, 2, method, "k")
                    q3 = quantize_tensor(k, 3, method, "k")
                    x = k.detach().float()
                    energy = x.square().mean().clamp(min=1e-12)
                    err2 = (dequantize_tensor(q2).float() - x).square().mean() / energy
                    err3 = (dequantize_tensor(q3).float() - x).square().mean() / energy
                    benefit = float(torch.clamp(err2 - err3, min=0).detach().cpu().item())
                    scores[layer_idx] = scores.get(layer_idx, 0.0) + benefit
                    counts[layer_idx] = counts.get(layer_idx, 0) + 1
                    del q2, q3, x, err2, err3
                del outputs, kv_tuple

        ranked = [
            {
                "layer": int(layer),
                "score": float(scores[layer] / max(1, counts[layer])),
            }
            for layer in scores
        ]
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked

    sensitive_layer_rank = []

    def _top_layers(fraction):
        layers = [int(item["layer"]) for item in sensitive_layer_rank]
        if not layers:
            return []
        count = max(1, min(len(layers), int(math.ceil(len(layers) * float(fraction)))))
        return layers[:count]

    def make_search_candidate(
        name,
        k_group=32,
        v_group=32,
        residual=4,
        metadata_mode="bf16",
        metadata_group_size=256,
        k3_fraction=1.0,
        residual_fraction=None,
        residual_low=0,
        note="",
    ):
        method = make_v18_k3v2_meta(
            k_axis="seq_affine",
            v_affine=True,
            k_seq_group=int(k_group),
            v_group=int(v_group),
            residual=int(residual),
            sink_tokens=0,
            v_outlier_tokens=0,
            pruned=False,
            label=name,
        )
        method = dict(method)
        method["id"] = "search_" + "".join(
            c.lower() if c.isalnum() else "_" for c in name
        ).strip("_")
        method["name"] = name
        method["metadata_mode"] = str(metadata_mode)
        method["metadata_group_size"] = int(metadata_group_size)
        method["note"] = note or (
            f"K3/V2 affine; Kseq{k_group}; Vgroup{v_group}; R{residual}; "
            f"metadata={metadata_mode}g{metadata_group_size}"
        )

        if float(k3_fraction) < 1.0:
            protected = _top_layers(k3_fraction)
            method["k_target_bits"] = 2
            method["k_layer_bit_map"] = {int(layer): 3 for layer in protected}
            method["note"] += (
                f"; K3 on top {len(protected)}/{len(sensitive_layer_rank)} layers, "
                "K2 elsewhere"
            )

        if residual_fraction is not None:
            protected = _top_layers(residual_fraction)
            method["residual_tokens"] = int(residual_low)
            method["layer_residual_map"] = {
                int(layer): int(residual) for layer in protected
            }
            method["note"] += (
                f"; R{residual} on top {len(protected)}/{len(sensitive_layer_rank)} "
                f"layers, R{residual_low} elsewhere"
            )

        return method

    def candidates_for_bit(bits):
        if bits != 2:
            raise ValueError("Memory-quality search currently supports bit_modes='16,2'.")

        candidates = []
        seen = set()

        def add(name, **kwargs):
            if name in seen:
                return
            seen.add(name)
            candidates.append((name, make_search_candidate(name, **kwargs)))

        # Current best checkpoint and isolated residual sweep.
        add("BASE G32 R4", k_group=32, v_group=32, residual=4)
        for r in range(4):
            add(f"RES G32 R{r}", k_group=32, v_group=32, residual=r)

        # Metadata-overhead search through K/V group geometry.
        for kg, vg in [
            (32, 64), (32, 128),
            (64, 32), (64, 64), (64, 128),
            (128, 32), (128, 64), (128, 128),
        ]:
            add(f"GRP K{kg} V{vg} R3", k_group=kg, v_group=vg, residual=3)

        for kg, vg in [(64, 64), (64, 128), (128, 64), (128, 128)]:
            add(f"GRP K{kg} V{vg} R4", k_group=kg, v_group=vg, residual=4)

        # Quantized metadata: two precision/overhead points.
        add(
            "META8g64 G32 R4",
            k_group=32, v_group=32, residual=4,
            metadata_mode="int8", metadata_group_size=64,
        )
        add(
            "META8g256 G32 R4",
            k_group=32, v_group=32, residual=4,
            metadata_mode="int8", metadata_group_size=256,
        )
        add(
            "META8g64 G64 R4",
            k_group=64, v_group=64, residual=4,
            metadata_mode="int8", metadata_group_size=64,
        )
        add(
            "META8g256 G64 R4",
            k_group=64, v_group=64, residual=4,
            metadata_mode="int8", metadata_group_size=256,
        )

        # K3 only where profiling predicts the extra K bit is most valuable.
        for fraction in (0.75, 0.50, 0.25):
            pct = int(round(fraction * 100))
            add(
                f"ADAPT K3-{pct} G32 R4",
                k_group=32, v_group=32, residual=4,
                k3_fraction=fraction,
            )
        for fraction in (0.75, 0.50):
            pct = int(round(fraction * 100))
            add(
                f"ADAPT-M8 K3-{pct} R4",
                k_group=32, v_group=32, residual=4,
                metadata_mode="int8", metadata_group_size=256,
                k3_fraction=fraction,
            )

        # Keep the BF16 residual only in the most sensitive layers.
        for fraction, low in [(0.75, 1), (0.50, 1), (0.25, 1), (0.50, 0)]:
            pct = int(round(fraction * 100))
            add(
                f"SELRES R4-{pct} R{low}",
                k_group=32, v_group=32, residual=4,
                residual_fraction=fraction,
                residual_low=low,
            )

        # Combined candidates target payload, metadata, and residual overhead together.
        add(
            "COMBO K3-75 R4-50/R1",
            k_group=32, v_group=32, residual=4,
            k3_fraction=0.75, residual_fraction=0.50, residual_low=1,
        )
        add(
            "COMBO K3-50 R4-50/R1",
            k_group=32, v_group=32, residual=4,
            k3_fraction=0.50, residual_fraction=0.50, residual_low=1,
        )
        add(
            "COMBO-M8 K3-75 R4-50/R1",
            k_group=32, v_group=32, residual=4,
            metadata_mode="int8", metadata_group_size=256,
            k3_fraction=0.75, residual_fraction=0.50, residual_low=1,
        )

        return candidates

    # =============================================================
    # Staged search: short screening, then full confirmation
    # =============================================================
    if set(BIT_MODES) != {16, 2}:
        raise ValueError("Use bit_modes='16,2' for this memory-quality search.")

    full_windows = windows
    screening_windows = [
        window[:, :screen_tokens].contiguous()
        for window in windows[:SCREEN_WINDOWS]
    ]

    def baseline_row(eval_windows, stage):
        trials = [run_one_trial(16, window, method_by_id["p3_uniform"]) for window in eval_windows]
        loss = statistics.median(row["loss"] for row in trials)
        kv_kb = statistics.median(row["pure_kv_kb"] for row in trials)
        ppl = math.exp(min(loss, 50))
        return {
            "stage": stage,
            "bits": 16,
            "candidate": "BF16 baseline",
            "ppl": ppl,
            "ppl_delta_pct": 0.0,
            "pure_kv_kb": kv_kb,
            "pure_kv_mb": kv_kb / 1024,
            "compression": 1.0,
            "effective_kv_bits": 16.0,
            "payload_kb": 0.0,
            "metadata_kb": 0.0,
            "residual_kb": kv_kb,
            "auxiliary_kb": 0.0,
            "ms": statistics.median(row["ms"] for row in trials),
            "note": "uncompressed reference",
        }, loss, kv_kb

    print("Running BF16 screening reference...")
    screen_baseline, screen_fp16_loss, screen_fp16_kv_kb = baseline_row(
        screening_windows, "screen"
    )
    print(
        f"Screen BF16: PPL {screen_baseline['ppl']:.4f}, "
        f"KV {screen_baseline['pure_kv_mb']:.3f} MB"
    )

    print("Running BF16 full reference...")
    full_baseline, full_fp16_loss, full_fp16_kv_kb = baseline_row(
        full_windows, "confirm"
    )
    print(
        f"Full BF16: PPL {full_baseline['ppl']:.4f}, "
        f"KV {full_baseline['pure_kv_mb']:.3f} MB"
    )
    print()

    print("Profiling layer sensitivity for adaptive K2/K3 candidates...")
    sensitive_layer_rank = profile_k_layer_sensitivity(screening_windows)
    print("Most K-sensitive layers:")
    print(
        ", ".join(
            f"L{item['layer']}={item['score']:.3e}"
            for item in sensitive_layer_rank[:12]
        )
    )
    print()

    candidate_specs = candidates_for_bit(2)
    candidate_methods = {name: method for name, method in candidate_specs}
    print(f"Screening {len(candidate_specs)} memory-quality candidates...")

    def evaluate_candidate(name, method, eval_windows, base_loss, base_kv_kb, stage):
        trials = [run_one_trial(2, window, method) for window in eval_windows]
        row = summarize_trials(2, method, trials, base_kv_kb, base_loss)
        row["stage"] = stage
        row["candidate"] = name
        row["config"] = name
        row["ppl_delta_pct"] = 100.0 * (
            row["ppl"] / math.exp(min(base_loss, 50)) - 1.0
        )
        row["effective_kv_bits"] = (
            16.0 / row["compression"] if row["compression"] > 0 else float("inf")
        )
        row["payload_mb"] = row["payload_kb"] / 1024
        row["metadata_mb"] = row["metadata_kb"] / 1024
        row["residual_mb"] = row["residual_kb"] / 1024
        row["auxiliary_mb"] = row["auxiliary_kb"] / 1024
        return row

    screen_rows = []
    for index, (name, method) in enumerate(candidate_specs, start=1):
        print(f"[{index:02d}/{len(candidate_specs):02d}] Screening {name}...")
        row = evaluate_candidate(
            name,
            method,
            screening_windows,
            screen_fp16_loss,
            screen_fp16_kv_kb,
            "screen",
        )
        row["screen_pass"] = row["ppl_delta_pct"] <= screening_ppl_increase_pct
        screen_rows.append(row)
        print(
            f"    PPL {row['ppl']:.3f} ({row['ppl_delta_pct']:+.2f}%), "
            f"KV {row['pure_kv_mb']:.3f} MB, "
            f"EffBits {row['effective_kv_bits']:.3f}, "
            f"meta {row['metadata_mb']:.3f} MB, "
            f"resid {row['residual_mb']:.3f} MB"
        )

    def pareto_frontier(rows):
        frontier = []
        for row in rows:
            dominated = False
            for other in rows:
                if other is row:
                    continue
                if (
                    other["ppl"] <= row["ppl"]
                    and other["pure_kv_mb"] <= row["pure_kv_mb"]
                    and (
                        other["ppl"] < row["ppl"]
                        or other["pure_kv_mb"] < row["pure_kv_mb"]
                    )
                ):
                    dominated = True
                    break
            if not dominated:
                frontier.append(row)
        return sorted(frontier, key=lambda row: (row["pure_kv_mb"], row["ppl"]))

    screen_pareto = pareto_frontier(screen_rows)
    screen_pass = [row for row in screen_pareto if row["screen_pass"]]

    finalist_names = []

    def add_finalist(name):
        if name in candidate_methods and name not in finalist_names:
            finalist_names.append(name)

    # Always confirm the current checkpoint.
    add_finalist("BASE G32 R4")

    # Primary selection: lowest memory on the screening Pareto frontier under cap.
    for row in sorted(screen_pass, key=lambda item: (item["pure_kv_mb"], item["ppl"])):
        add_finalist(row["candidate"])
        if len(finalist_names) >= max(1, finalists - 2):
            break

    # Safety finalists: best screening PPL and absolute lowest memory.
    add_finalist(min(screen_rows, key=lambda item: item["ppl"])["candidate"])
    add_finalist(min(screen_rows, key=lambda item: item["pure_kv_mb"])["candidate"])

    # Fill remaining slots from the Pareto frontier, then all rows.
    for row in screen_pareto + sorted(screen_rows, key=lambda item: (item["pure_kv_mb"], item["ppl"])):
        add_finalist(row["candidate"])
        if len(finalist_names) >= finalists:
            break

    finalist_names = finalist_names[:finalists]
    print()
    print("Full-confirmation finalists:")
    for name in finalist_names:
        print(f"  - {name}")
    print()

    confirmed_rows = []
    for index, name in enumerate(finalist_names, start=1):
        print(f"[{index:02d}/{len(finalist_names):02d}] Confirming {name} on full protocol...")
        row = evaluate_candidate(
            name,
            candidate_methods[name],
            full_windows,
            full_fp16_loss,
            full_fp16_kv_kb,
            "confirm",
        )
        row["final_pass"] = row["ppl_delta_pct"] <= max_ppl_increase_pct
        confirmed_rows.append(row)
        print(
            f"    PPL {row['ppl']:.3f} ({row['ppl_delta_pct']:+.2f}%), "
            f"KV {row['pure_kv_mb']:.3f} MB, "
            f"Compression {row['compression']:.3f}x, "
            f"EffBits {row['effective_kv_bits']:.3f}"
        )

    final_eligible = [row for row in confirmed_rows if row["final_pass"]]
    if final_eligible:
        selected = min(
            final_eligible,
            key=lambda item: (item["pure_kv_mb"], item["ppl"]),
        )
        selection_status = "accepted_under_final_cap"
    else:
        selected = min(
            confirmed_rows,
            key=lambda item: (max(0.0, item["ppl_delta_pct"]), item["pure_kv_mb"]),
        )
        selection_status = "no_candidate_met_final_cap"

    selected = dict(selected)
    selected["selection_status"] = selection_status

    def method_summary(method):
        return {
            "k_bits_default": int(method.get("k_target_bits", 3)),
            "v_bits_default": int(method.get("v_target_bits", 2)),
            "k_group": int(method.get("k_group", 32)),
            "v_group": int(method.get("v_group", 32)),
            "k_quant_axis": str(method.get("k_quant_axis", "seq_affine")),
            "residual_tokens": int(method.get("residual_tokens", 0)),
            "metadata_mode": str(method.get("metadata_mode", "bf16")),
            "metadata_group_size": int(method.get("metadata_group_size", 256)),
            "k_layer_bit_map": {
                str(key): int(value)
                for key, value in method.get("k_layer_bit_map", {}).items()
            },
            "layer_residual_map": {
                str(key): int(value)
                for key, value in method.get("layer_residual_map", {}).items()
            },
            "note": str(method.get("note", "")),
        }

    return {
        "mode": mode,
        "model": MODEL_NAME,
        "max_tokens": MAX_TOKENS,
        "screen_tokens": screen_tokens,
        "screen_windows": SCREEN_WINDOWS,
        "full_windows": NUM_WINDOWS,
        "screening_ppl_cap_pct": screening_ppl_increase_pct,
        "final_ppl_cap_pct": max_ppl_increase_pct,
        "screen_baseline": screen_baseline,
        "full_baseline": full_baseline,
        "sensitive_layer_rank": sensitive_layer_rank,
        "screen_rows": sorted(
            screen_rows,
            key=lambda item: (item["pure_kv_mb"], item["ppl"]),
        ),
        "screen_pareto": screen_pareto,
        "finalist_names": finalist_names,
        "confirmed_rows": sorted(
            confirmed_rows,
            key=lambda item: (not item["final_pass"], item["pure_kv_mb"], item["ppl"]),
        ),
        "selected": selected,
        "selected_config": method_summary(candidate_methods[selected["candidate"]]),
        "candidate_configs": {
            name: method_summary(method)
            for name, method in candidate_specs
        },
    }

def print_results_locally(result, output_dir="rabit_kv_search_results"):
    import csv
    import json
    import os

    os.makedirs(output_dir, exist_ok=True)

    baseline = result["full_baseline"]
    rows = result["confirmed_rows"]
    selected = result["selected"]

    print()
    print("RABIT-KV MEMORY-QUALITY SEARCH - FULL CONFIRMATION")
    print("=" * 150)
    print(
        f"{'Configuration':<34}"
        f"{'PPL':<10}"
        f"{'PPL d%':<10}"
        f"{'KV MB':<10}"
        f"{'Comp':<9}"
        f"{'EffBits':<10}"
        f"{'Payload':<10}"
        f"{'Metadata':<10}"
        f"{'Residual':<10}"
        f"{'Pass':<7}"
    )
    print("-" * 150)
    print(
        f"{'BF16 baseline':<34}"
        f"{baseline['ppl']:<10.3f}"
        f"{0.0:<10.2f}"
        f"{baseline['pure_kv_mb']:<10.3f}"
        f"{1.0:<9.3f}"
        f"{16.0:<10.3f}"
        f"{'-':<10}{'-':<10}{baseline['pure_kv_mb']:<10.3f}"
        f"{'YES':<7}"
    )
    for row in rows:
        print(
            f"{row['candidate']:<34}"
            f"{row['ppl']:<10.3f}"
            f"{row['ppl_delta_pct']:<10.2f}"
            f"{row['pure_kv_mb']:<10.3f}"
            f"{row['compression']:<9.3f}"
            f"{row['effective_kv_bits']:<10.3f}"
            f"{row['payload_mb']:<10.3f}"
            f"{row['metadata_mb']:<10.3f}"
            f"{row['residual_mb']:<10.3f}"
            f"{('YES' if row['final_pass'] else 'NO'):<7}"
        )
    print("-" * 150)
    print()
    print("SELECTED CANDIDATE")
    print(f"  Name:              {selected['candidate']}")
    print(f"  Status:            {selected['selection_status']}")
    print(f"  PPL:               {selected['ppl']:.4f}")
    print(f"  PPL increase:      {selected['ppl_delta_pct']:+.3f}%")
    print(f"  KV memory:         {selected['pure_kv_mb']:.4f} MB")
    print(f"  Compression:       {selected['compression']:.4f}x")
    print(f"  Effective bits:    {selected['effective_kv_bits']:.4f}")
    print(f"  Payload memory:    {selected['payload_mb']:.4f} MB")
    print(f"  Metadata memory:   {selected['metadata_mb']:.4f} MB")
    print(f"  Residual memory:   {selected['residual_mb']:.4f} MB")
    print()
    print("Selected configuration:")
    for key, value in result["selected_config"].items():
        print(f"  {key}: {value}")

    json_path = os.path.join(output_dir, "RABIT_KV_memory_quality_search.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    csv_path = os.path.join(output_dir, "RABIT_KV_confirmed_candidates.csv")
    fields = [
        "candidate", "ppl", "ppl_delta_pct", "pure_kv_mb", "compression",
        "effective_kv_bits", "payload_mb", "metadata_mb", "residual_mb",
        "auxiliary_mb", "final_pass", "ms", "note",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    screen_csv_path = os.path.join(output_dir, "RABIT_KV_screening_candidates.csv")
    screen_fields = [
        "candidate", "ppl", "ppl_delta_pct", "pure_kv_mb", "compression",
        "effective_kv_bits", "payload_mb", "metadata_mb", "residual_mb",
        "auxiliary_mb", "screen_pass", "ms", "note",
    ]
    with open(screen_csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=screen_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["screen_rows"])

    print()
    print(f"Saved JSON: {json_path}")
    print(f"Saved confirmed CSV: {csv_path}")
    print(f"Saved screening CSV: {screen_csv_path}")


@app.local_entrypoint()
def main(
    model_name: str = "LLM-Research/Meta-Llama-3.1-8B-Instruct",
    mode: str = "final",
    max_ppl_increase_pct: float = 3.0,
    screening_ppl_increase_pct: float = 6.0,
    screen_tokens: int = 128,
    finalists: int = 7,
    bit_modes: str = "16,2",
    output_dir: str = "rabit_kv_search_results",
):
    result = run_benchmark.remote(
        model_name=model_name,
        mode=mode,
        max_ppl_increase_pct=max_ppl_increase_pct,
        screening_ppl_increase_pct=screening_ppl_increase_pct,
        screen_tokens=screen_tokens,
        finalists=finalists,
        bit_modes=bit_modes,
    )
    print_results_locally(result, output_dir=output_dir)
