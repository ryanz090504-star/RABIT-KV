# RABIT-KV

Ryan Zhang's original RABIT-KV code is licensed under [Apache-2.0](LICENSE).
See [licensing scope and third-party exceptions](docs/LICENSING.md) for covered
code, preserved third-party terms, and excluded paper assets and research data.

RABIT-KV is a **target-bit-aware KV-cache quantization and physically packed serving system** for LLM inference. Its frozen **K3 / V2 / G32 / R4 / META8g64** operating point jointly accounts for asymmetric precision, metadata, residual state, online aging, and allocator capacity.

The **2-bit target** names the operating point centered on 2-bit values: keys use 3 bits, values use 2 bits, metadata adds storage, and the newest four tokens remain BF16. It is not literal two-bit total storage. Capacity and quality below refer to this single frozen configuration.

**Capacity first:** on the evaluated H100 configuration, RABIT-KV provides **5.28×** the paged KV capacity of BF16, **2.64×** that of FP8, and **1.72×** that of the tested TurboQuant configuration. This comes with higher latency and lower throughput in the evaluated workloads. Quality is strongly model-dependent.

## Matched serving results

Llama-3.1-8B-Instruct, one NVIDIA H100 80GB HBM3, eager execution, single request, 2,048 prompt tokens and 32 output tokens. Latencies are medians of the two leg medians; each leg has 30 measured requests. These are descriptive statistics without confidence intervals.

| Cache | Observed allocator capacity (tokens) | TPOT (ms/token) | TTFT (ms) |
|---|---:|---:|---:|
| BF16 | 393,024 | 24.19 | 59.99 |
| Native per-tensor FP8 e4m3 | 785,568 | 25.75 | 63.43 |
| Tested TurboQuant K3/V4 | 1,203,200 | 27.79 | 55.03 |
| RABIT-KV | 2,074,592 | 30.10 | 144.65 |

RABIT-KV TPOT is **24.42% higher than BF16** and **8.33% higher than tested TurboQuant**, computed from unrounded values. TurboQuant uses `turboquant_k3v4_nc`: 28 quantized layers and four BF16 boundary layers. Its own backend plus FlashAttention v2 on those boundary layers differs from the Triton backend used by the other methods. This is a matched, method-native **system comparison**, not an isolated quantizer-kernel comparison, quality comparison, or throughput claim.

Source: [matched serving summary](results/experiments/matched_serving_baselines/exp13/matched_capacity_latency_summary.json). All methods use the same patched vLLM snapshot, including the recorded upstream TurboQuant cache-dtype dispatch fix; details are in [reproducibility](docs/REPRODUCIBILITY.md).

Capacity is the live engine's allocated blocks times block size. It measures the paged pool and excludes per-sequence residual/open-group state outside that pool; it is neither total-device-memory compression nor a throughput multiplier.

## Canonical quality

These results use a validated logical Hugging Face evaluator of stored-cache representation and online aging, separately from physical serving. They do not validate serving quality end to end.

| Model / task | BF16 | Canonical RABIT | Outcome |
|---|---:|---:|---|
| Llama-3.1-8B WikiText-2 continuation PPL | 7.515 | 7.633 | +1.56%; paired 95% CI [+1.12%, +2.05%] |
| Qwen2.5-7B WikiText-2 continuation PPL | 6.787 | 112.39 | +1555.9%; paired 95% CI [+1320%, +1828%] |
| Llama NIAH | 57/57 | 57/57 | No score loss on this deterministic grid |
| Llama Passage Retrieval | 100.0% | 100.0% | 200 examples; saturated scores |
| Llama HotpotQA, primary scorer | 59.30 F1 | 58.27 F1 | −1.03 points; paired 95% CI [−4.24, +2.20] |
| Llama HotpotQA, official secondary scorer | 59.85 F1 | 58.71 F1 | −1.14 points; paired 95% CI [−4.30, +2.03] |

Continuation PPL uses 32 paired windows, 1,024 context tokens and 128 scored tokens per window. HotpotQA uses 100 examples from the pinned 8k+ subset. Its primary scorer retains historical article handling; the secondary scorer uses the official LongBench definition on the same predictions. A confidence interval spanning zero does **not** establish equivalence. Sources: [accepted canonical record](results/experiments/canonical_quality_v2/canonical_quality_final_record.json), [PPL attempt](results/experiments/canonical_quality_v2/continuation_ppl/two_model_attempt_2/summary.json), [long-context attempt](results/experiments/canonical_quality_v2/long_context/attempt_1/summary.json).

Qwen's severe quality failure occurs at the same frozen operating point; its cause is undiagnosed. Separately, Qwen serving increases paged capacity from **902,656 to 4,764,640 tokens**, while TPOT rises from **26.96 to 48.70 ms/token (+80.7%)** through untuned fallback paths. Capacity transfer does not establish quality transfer. Source: [Qwen serving](results/experiments/cross_model/serving/capacity_latency_summary.json).

## Scaling and limitations

- At 8,192-token prompts, RABIT sustained offered concurrency 64 while BF16 reached only 47 overlapping in-flight requests. RABIT throughput was lower at every tested concurrency. See [concurrency report](results/experiments/concurrency_scaling/L8192/shadow_conditioned/combined_report.json).
- At 32,736 prompt tokens plus 32 output tokens, median TTFT was **114.38 s** for RABIT versus **4.435 s** for BF16. This is the maximum tested model-limit point, not a demonstrated maximum feasible context. See [context scaling](results/experiments/context_scaling/attempt_2/context_scaling_summary.json).
- Current execution serializes per-request/per-token work. Profiling is diagnostic and materially perturbs timing; it does not replace accepted unprofiled results. Long-context prefill also has substantial closed-page device work. These costs have not been shown intrinsic to the representation.
- Evaluation covers one GPU type and one frozen operating point. Eager latency comparisons need not transfer to CUDA graphs. There are no canonical ablations, Qwen long-context results, Qasper results, or multilingual results. Retrieval saturation cannot resolve small effects.
- Logical evaluation uses exact dense initial prefill and does not measure quality of later serving prefill chunks over packed history. CPU/CUDA states are not bit-identical; validation uses a same-device independent oracle.

## Code, paper, and evidence

- [Method and cache lifecycle](docs/METHOD.md)
- [Evidence, environment, and public reproduction instructions](docs/REPRODUCIBILITY.md)
- [Public reproduction adapter](benchmarks/public/reproduce.py)
- [Public machine-readable summary](results/summary.json)
- [Evidence directory index](results/README.md)
- [Scientific source navigation](benchmarks/README.md)
- [Historical documents](docs/history/README.md)
- [Complete historical-to-current path manifest](docs/PATH_MIGRATION.json)
- [Physical cache implementation](vllm-kvquant/vllm/v1/attention/ops/rabit_kv2.py)
- [Canonical logical evaluator](benchmarks/experiments/canonical_rabit_quality.py)
- [Preserved paper source](paper/main.tex), [supplement](paper/appendix.tex), [frozen paper PDF](paper/frozen/main.pdf), [frozen supplement PDF](paper/frozen/appendix.pdf)

Historical quality evidence in `results/legacy/quality/` and the earlier experiment directories is **legacy logical-evaluator evidence**, not canonical quality. The earlier evaluator used different value-metadata ordering and did not age generated tokens. Historical deployment sessions remain physical evidence, but their absolute latencies must not be pooled with the matched session above. The old public summary is preserved at `research-backup-2026-10-11:results/summary.json`; the new summary records its provenance instead of relabeling it canonical.

The public adapter supports read-only planning and explicit reproduction into fresh external output directories. It preserves the frozen workers, protocols, statistics, and validity gates and never commits or pushes. See the reproduction guide for the clean-source and environment prerequisites. Frozen scientific files, evidence, historical documents and paper assets retain their original bytes. The public directory mapping is applied; embedded historical paths remain unchanged and resolve through the path manifest or the pinned preserved source checkout. The migration record and schema-consumer audit are maintained in [public release history](docs/PUBLIC_RELEASE_HISTORY.md).
