# Serving-engine perplexity (N32, attempt 002)

The packed RABIT-KV cache inside the vLLM serving engine, scored against BF16 serving in the same engine. This is a
**physical** quality measurement of the serving path, separate from the logical evaluator used for the canonical
quality results.

- **Model:** Llama-3.1-8B-Instruct, revision `359efdbb8af05b788a4ad4185215c6b8caa9052c`, one NVIDIA H100.
- **Task:** WikiText-2 continuation; 32 windows; 1,024 context tokens and 128 scored tokens per window.
- **Cells:** BF16 single-chunk prefill; RABIT-KV single-chunk; RABIT-KV forced two-chunk prefill (512 + 512); BF16 two-chunk.
- **Protocol:** the protocol was frozen locally before the run (`frozen/preregistration-v2.json`). Windows 5–32 are the primary set, 1–4 the pilot, all 32 secondary.

| Set | BF16 | RABIT-KV single | RABIT-KV two-chunk | Single vs BF16 | Two-chunk vs BF16 |
|---|---:|---:|---:|---:|---:|
| Primary, windows 5–32 (97.5% intervals) | 7.493 | 7.600 | 7.640 | +1.42% [+0.86, +2.00] | +1.96% [+1.07, +2.93] |
| Secondary, all 32 (95% intervals) | 7.521 | 7.630 | 7.677 | +1.44% [+0.99, +1.91] | +2.07% [+1.36, +2.83] |
| Pilot, windows 1–4 | 7.718 | 7.843 | 7.940 | +1.61% | +2.87% |

Paired window bootstrap, 100,000 resamples, seed 20261020. The two primary co-endpoints use 97.5% intervals
(Bonferroni); the rest use 95%. The BF16 single and two-chunk cells are bitwise identical.

**What this does and does not show.**

- It is an end-to-end difference between two serving paths, not a quantization-only effect. For example, the dense initial-prefill kernels also differ between the two paths.
- It covers one model and one task. Requests run one at a time with instrumentation, so the run carries no latency information.
- An interrupted earlier attempt had already scored BF16 single windows 1–32 and RABIT-KV single windows 1–23 plus part of window 24. The primary windows are therefore not untouched, and none of the windows form a held-out set.
- The primary schedule interaction is +0.53% [-0.08, +1.19] (95%). The interval includes zero, so no effect of the prefill schedule is claimed.
- No equivalence is claimed.

**Files.**

- **Report and results:**
  - `REPORT.md`: the run report, including validity checks and costs;
  - `frozen-analysis.json`: the frozen analysis output;
  - `paired-window-differences.csv`, `paired-token-differences.csv`: every paired difference, no exclusions;
  - `summary.json`, `validity-audit.json`, `registration.json`.
- **`frozen/`:** the protocol, inputs and analysis script as frozen before the run.
- **`code/`:** the in-engine scoring worker and the checks it ran.
