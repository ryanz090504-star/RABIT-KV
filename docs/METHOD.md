# RABIT-KV Method

## Overview

RABIT-KV is a target-bit-aware KV-cache quantization method designed to reduce KV memory during LLM inference while preserving model quality and supporting a physically compressed deployment path.

The final operating point is:

**K3 / V2 / G32 / R4 / META8g64**

## Quantization policy

### Keys: K3

Keys use **3-bit sequence-axis affine quantization** with group size 32.

The key path keeps more precision than the value path because aggressively reducing key precision caused larger quality degradation during development.

### Values: V2

Values use **2-bit affine quantization** with group size 32 over the last dimension.

### Residual window: R4

The newest **four K/V tokens remain in BF16**. Older entries are aged into the packed low-bit cache.

The residual window protects the most recent attention context while keeping the dominant long-context storage compressed.

### Metadata: META8g64

Primary quantization metadata is stored as grouped **UINT8** values with metadata group size 64, together with higher-precision secondary parameters required by the physical page codec.

Metadata is included in reported logical storage rather than treated as free overhead.

## Physical representation

The deployment KV-cache dtype is:

```text
rabit_kv2
```

The authoritative implementation is:

```text
vllm-kvquant/vllm/v1/attention/ops/rabit_kv2.py
```

The implementation stores packed K/V payloads and metadata in physical vLLM KV pages. It is therefore distinct from fake quantization approaches that quantize and immediately dequantize back into a full-precision cache.

The final decode path includes packed-page updates, a four-token BF16 residual, low-bit tail preparation, INT3 key packing, grouped metadata handling, V2 ageing, and compressed attention execution through custom Triton kernels.

## What “2-bit target” means

RABIT-KV should not be described as literal two-bit total storage.

The final representation contains:

- 3-bit keys;
- 2-bit values;
- grouped metadata;
- a four-token BF16 residual window.

“2-bit target” refers to the selected operating point centered on 2-bit values.

## Metrics

RABIT-KV reports four different quantities that should not be conflated:

1. **Quality** — continuation perplexity and downstream benchmark scores.
2. **Logical KV compression** — packed prefix-KV payload + metadata + residual accounting used by the quality harness.
3. **Physical allocator capacity** — the number of KV tokens that the vLLM allocator can hold under a fixed GPU configuration.
4. **Deployment latency** — real-engine TTFT, TPOT, and wall-clock measurements.

## Final operating results

On Llama 3.1 8B Instruct:

- WikiText-2 continuation PPL: **8.5020 → 8.6317 (+1.53%)**
- logical KV compression: approximately **5.18–5.27×**
- physical KV capacity: **393,024 → 2,074,592 tokens (5.2785×)**
- NIAH: **15/15 → 15/15**
- Passage Retrieval: **100% → 100%**
- Qasper 8K+: **35.9 → 35.6 F1**
- HotpotQA 8K+: **60.6 → 55.2 F1**
- RABIT-KV median TPOT: **21.61 ms/token**

The HotpotQA result is the clearest quality regression and is retained as part of the reported result set.
