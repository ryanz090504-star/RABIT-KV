# Component ablation: metadata precision (A001)

What does the second-level 8-bit quantization of the group parameters (META8g64) cost? This runs in the logical
evaluator, with FlashAttention fixed. The arms cross exact (FP32) group parameters with padding exclusion.

| Arm (PPL, 32 test windows) | Qwen2.5-7B | Llama-3.1-8B |
|---|---:|---:|
| BF16 | 6.791 | 7.514 |
| Frozen policy | 113.93 | 7.639 |
| Exact metadata | 83.88 | 7.641 |
| Padding excluded | 7.841 | 7.619 |
| Padding excluded + exact metadata | 6.895 | 7.623 |

Share of the excess mean token NLL over BF16 that exact metadata removes (95% interval):

| | With padding excluded | Under the frozen policy |
|---|---:|---:|
| Qwen2.5-7B | 0.89 [0.84, 0.95] | 0.11 [0.09, 0.13] |
| Llama-3.1-8B | -0.04 [-0.26, 0.13] | -0.01 [-0.23, 0.14] |

- **Qwen2.5-7B:** with padding excluded, exact metadata removes most of the remaining excess. Under the frozen policy it removes only a small share. The two components interact, so the shares are conditional and do not add up.
- **Llama-3.1-8B:** the ablation detects no effect of META8g64. That is not a statement that it has no cost.
- **Not a measured system:** the exact-metadata arms store FP32 group parameters. That layout would occupy 1,152 bytes per token per layer, a pool ratio of 3.56× over BF16, not the 5.28× of the measured operating point. Neither change is implemented in the serving engine, and no capacity, latency, throughput or serving-quality result follows from this ablation.
- **Audit note:** recomputing the analysis locally reproduced the in-run values except for the last bits of 10 percentage fields (`pod-vs-local-ulp-fields.json`). No decision changes.

**Files.**

- `freeze-record.json`, `audit.json`, `analysis/` (numbers and table), `pod-vs-local-ulp-fields.json`.
- **`artifacts/`:** per-token scores, aggregates and validity gates.
- **`code/`:** the scoring worker, the analysis, the independent audit and the table script.
