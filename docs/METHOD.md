# RABIT-KV method

RABIT-KV prioritizes physical KV capacity while exposing serving cost and model-dependent quality. One frozen operating point is evaluated: **K3 / V2 / G32 / R4 / META8g64**. It is not a universally suitable or individually optimal set of choices; the development ablations remain legacy-evaluator evidence.

## Representation and lifecycle

- **K3:** 3-bit affine keys grouped along the sequence axis, one minimum/scale pair per head and channel over 32 tokens. Incomplete groups are zero-padded for quantization statistics.
- **V2:** 2-bit affine values grouped over 32 channels per token and head.
- **R4:** up to four newest tokens remain exact BF16. When appending would exceed four residual tokens, the oldest residual token ages into the open group.
- **META8g64:** each primary minimum/scale array is encoded in groups of 64 with UINT8 codes and BF16 secondary minimum/scale parameters. Decoding uses the metadata-decoded parameters. Value metadata follows canonical `(token, head, channel-group)` ordering.

An open group holds fewer than 32 aged tokens. Its key statistics and codes change as it grows. Once it reaches 32 tokens it closes into an immutable physical page. For a sequence of at least four tokens, the state consists of `floor((n - 4) / 32)` closed pages, `(n - 4) mod 32` open-group tokens, and four residual tokens.

The [physical implementation](../vllm-kvquant/vllm/v1/attention/ops/rabit_kv2.py) integrates `rabit_kv2` pages into vLLM's allocator. Closed pages contain packed K/V payloads and metadata. Open-group and residual state are outside the paged pool and total at most 35 tokens per sequence. Initial serving prefill uses exact dense attention before encoding the state; later chunks attend to packed history. Decode appends and ages a token before attending to the state at its position. Current paths run per request and perform aging/open-tail work per token.

## Storage and capacity accounting

“2-bit target” names the operating point, not total storage. Equal-sized K/V payloads average **2.5 bits per element**, derived from `(3 + 2) / 2`, before metadata and auxiliary state.

For Llama geometry, one layer's closed page occupies **24,832 bytes per 32 tokens**, or **776 bytes per token-layer**, versus **4,096 bytes** in BF16. Derived closed-page storage is **3.03125 bits per K/V element**, from `776 * 8 / 2048`. These figures include page metadata but exclude residual/open-group state and allocator rounding. Source: [matched summary, theoretical byte accounting](../results/experiments/matched_serving_baselines/exp13/matched_capacity_latency_summary.json).

The headline is instead **observed allocator capacity**, blocks times block size: **393,024 BF16 versus 2,074,592 RABIT tokens (5.2785×)**. This does not measure total device memory or predict throughput. The matched serving table in the [README](../README.md) shows the cost beside capacity: **30.10 versus 24.19 ms/token TPOT** and **144.65 versus 59.99 ms TTFT**. The tested TurboQuant system has different required backends and BF16 boundary layers; comparison is between deployed systems, not quantizer kernels.

## Canonical quality versus serving

The [canonical evaluator](../benchmarks/experiments/canonical_rabit_quality.py) reproduces stored-cache representation and online aging through a Hugging Face cache object. It does not reproduce serving kernels or scheduling.

Continuation PPL begins with 1,024 exact BF16 context tokens. The first scored token uses the prefill logit, identical between arms; the remaining 127 tokens are teacher-forced one at a time against the evolving cache. Across 32 windows, each arm scores 4,096 tokens. PPL is `exp(total NLL / total scored tokens)`.

Long-context generation prefills all but the last prompt token in BF16, converts that prefix to canonical state, then steps the last prompt token and generated tokens through the aging cache. Every generated token is predicted from canonical state. The evaluator never executes a later serving prefill chunk, so its quality is unmeasured. Same-device independent-oracle conformance validates stored state and aging; it is not formal verification or end-to-end serving/evaluator output parity. CPU and CUDA arithmetic can produce different codes; no cross-device tolerance is defined.

At this policy, Llama continuation PPL is **7.515 → 7.633 (+1.56%)**, while Qwen is **6.787 → 112.39 (+1555.9%)**. Llama retains NIAH **57/57** and Passage Retrieval **100%** on 200 examples; HotpotQA is **59.30 → 58.27 F1**, with a paired confidence interval spanning zero, which does not establish equivalence. [Canonical evidence](../results/experiments/canonical_quality_v2/canonical_quality_final_record.json) supplies full precision, confidence intervals, and acceptance status.

Qwen serves with the same representation through untuned fallback kernels, but shows severe logical quality loss. Its cause is unknown. No canonical Qwen long-context, Qasper, multilingual, or ablation measurements exist. Legacy evaluator results cannot support canonical quality claims or isolate causes of the legacy-to-canonical differences. See [reproducibility and provenance](REPRODUCIBILITY.md).
