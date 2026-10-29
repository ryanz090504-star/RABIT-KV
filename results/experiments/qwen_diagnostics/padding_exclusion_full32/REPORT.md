# Attempt 008: full-32 paired padding-exclusion experiment

**Post-hoc fixed-Flash experiment. Canonical automatic-backend records and attempt 006 remain unchanged and separate.**

This report uses every frozen window for each pinned model, with no outcome-based exclusions. It extends attempt 007 from four to 32 windows. Windows 1–4 are reported separately from 5–32; the latter are not a wholly unseen holdout because earlier diagnostics inspected Qwen window 5 and the frozen/full-32 results.

## Results

PPL below is independently reconstructed from all individual token losses using float64 summation. Frozen-style FP32 row-reduction aggregates are also retained in the audit, with rounding differences checked.

| Model / stratum | BF16 | Frozen policy | Exact K | Padding excluded |
|---|---:|---:|---:|---:|
| qwen2_5_7b / all | 6.790927985 | 113.932320131 | 6.863993492 | 7.841209676 |
| qwen2_5_7b / previous_first4 | 4.908664315 | 93.279252459 | 5.024652170 | 5.502981861 |
| qwen2_5_7b / remaining28 | 7.113233992 | 117.234593565 | 7.176783465 | 8.248070242 |
| llama3_1_8b / all | 7.513652587 | 7.639205798 | 7.611670594 | 7.619292863 |
| llama3_1_8b / previous_first4 | 7.698928002 | 7.845344071 | 7.853829860 | 7.858816817 |
| llama3_1_8b / remaining28 | 7.487551174 | 7.610202984 | 7.577691376 | 7.585676362 |

Each contrast uses paired window mean NLL; PPL effect is `100*(exp(mean paired NLL difference)-1)`. Confidence intervals are percentile paired-window bootstrap, 20,000 resamples, seed 20261022. These quantify variability across these frozen windows, not uncertainty from model-training seeds or a new population. Intervals are unadjusted exploratory intervals; overlapping strata are not independent replications.

| Model / stratum | Padding vs BF16 % [95% CI] | Padding vs frozen % [95% CI] | Improved / worsened vs frozen |
|---|---:|---:|---:|
| qwen2_5_7b / all | 15.465952 [12.118392, 19.055734] | -93.117660 [-94.092016, -91.987986] | 32 / 0 |
| qwen2_5_7b / previous_first4 | 12.107521 [4.744622, 18.719535] | -94.100530 [-96.348477, -89.426303] | 4 / 0 |
| qwen2_5_7b / remaining28 | 15.953872 [12.374165, 19.929023] | -92.964474 [-93.965237, -91.813430] | 28 / 0 |
| llama3_1_8b / all | 1.405978 [0.925645, 1.912744] | -0.260668 [-0.560783, 0.040105] | 21 / 11 |
| llama3_1_8b / previous_first4 | 2.076767 [0.812564, 3.356823] | 0.171729 [-0.779107, 1.131678] | 2 / 2 |
| llama3_1_8b / remaining28 | 1.310511 [0.812411, 1.853635] | -0.322286 [-0.629394, -0.014406] | 19 / 9 |

Qwen: padding exclusion removes 94.900544% of frozen-policy excess NLL (paired bootstrap 95% [93.806685%, 95.953628%]). Residual degradation against BF16 and the exact-K comparison must be considered when deciding whether quality is restored; no equivalence tolerance was prespecified. A large reduction in catastrophic loss is not proof of BF16 equivalence.

The per-window CSV and audit JSON retain every paired NLL effect and exact-K contrast, including strata-specific intervals. No window was removed or rerun because of its loss.

Qwen quality is **not fully restored**: the padding policy remains 15.47% above BF16 over all 32 windows, and 15.95% above BF16 in windows 5–32. Exact K is much closer to BF16, indicating substantial remaining K-policy loss. The intervention improves every Qwen window relative to frozen policy.

Llama shows **no demonstrated full-32 regression relative to frozen policy**: the point estimate improves 0.261%, but the 95% interval permits up to 0.040% worsening. The first-four estimate worsens 0.172%; windows 5–32 improve 0.322%. This does not establish equivalence or safety across other inputs. Against BF16, Llama still has a 1.406% gap (95% interval 0.926–1.913%).

## Validity and independent audit

- qwen2_5_7b: 131 rows and 16768 individual losses audited; max absolute discrepancy between token summation and FP32 row reduction 6.32493756711e-05. First-four bitwise repeats versus 007: {'bf16': True, 'rabit': True, 'k_exact': True, 'range_real': True}. Oracle checkpoints: {'rabit': 252, 'range_real': 252}.
- llama3_1_8b: 131 rows and 16768 individual losses audited; max absolute discrepancy between token summation and FP32 row reduction 3.61116981367e-05. First-four bitwise repeats versus 007: {'bf16': True, 'rabit': True, 'k_exact': True, 'range_real': True}. Oracle checkpoints: {'rabit': 288, 'range_real': 288}.

All rows have 128 finite nonnegative NLLs, correct target IDs, hashed 1024-token contexts, target positions 1024–1151 and predictor positions 1023–1150. Every first target matches BF16 prefill. All row and aggregate reductions were recomputed independently. Model/tokenizer file verification and complete token-pool hashes match the frozen records. Source hashes match the preflight-staged files.

Pass-through equals BF16 bitwise; the independent frozen oracle equals frozen policy bitwise; the independent padding oracle equals the new policy bitwise, for all 128 losses of window 1 in each model. Those gates run before windows 2–32 of each model. Nine cache checkpoints across every layer pass for both policies; oracle model replay covers all 127 decode steps. No validity failure or scientific retry occurred.

Before GPU execution, 56 independent NumPy/packed quantization cases passed. The actual staged variant functions passed synthetic full cache-aging checks at every length 1024–1152 for both geometries, HF cache checks over 1024–1062, and exact-K / expected-V checks. Complete groups remain unchanged. The prior evidence manifest (265 files) was verified.

Every model attention call checks that Flash SDPA is enabled and math, memory-efficient and cuDNN SDPA are disabled; unsupported Flash execution would fail. Counts independently verify 128 SDPA calls and 127 cache updates per layer per row, and 159 K/V quantizer invocations per layer for cache arms. Frozen/new policies invoke META8 twice per K and twice per V call; exact K retains V metadata only; pass-through invokes none. The dispatch functions are checked at entry and reset after every row. No silent fallback is allowed.

## Interpretation and remaining uncertainty

The earlier causal trace remains the mechanism evidence: first decode, layer 0 quantized cache read diverges before SDPA, with equal projections/RoPE and exact residual entries. Artificial zeros distort the unfinished K group extrema. The new policy repeats the final real token only in temporary range padding and trims the result, preserving complete groups, frozen V, R4, and META8. Its independent packed oracle is modified separately. Attempt 008 extends the quality evidence; it does not introduce new layer-localization claims.

This remains post-hoc, fixed-backend, fixed-input evidence. It does not validate the physical serving kernel, throughput, memory cost, longer contexts, other models, or an unseen corpus. Residual K quantization, V quantization, metadata and propagated activation effects are not additive causal components. Model-specific learned offsets/biases remain an unresolved explanation for the sensitivity.

## Execution and provenance

App `ap-89XqSGLS50czOGJh2NoWWj`; call `fc-01M5HWYS9HDVWFQPEJSDYYT12S`; volume `vo-jsAxCi6OzKvHZ0Rxr1gWcA`. Detached execution, one H100, max one container, retries=0, 1800-second function cap and 1740-second subprocess cap. Logs and intermediate scores were periodically committed to the persistent diagnostic volume. Runner elapsed: 1734.271943 seconds.

Before launching, all prior calls and 006/007 local logs, volume contents and output IDs were inspected. No active or completed job was duplicated. A CPU mounted-module preflight checked the staged script hash before the single GPU call. Output directories and IDs were fresh, and prior attempts were not reused.

Hardware: NVIDIA H100 80GB HBM3, 81559, 610.57.04. Torch 2.11.0+cu130, CUDA 13.0, transformers 4.48.2; image `im-HuxOhBSvQGDiM2LySuwUTJ`. Current driver differs from the historical canonical run, as documented in the prior report.

Qwen: `Qwen/Qwen2.5-7B-Instruct@16c174980d8a1492910551634b4969e69cdc2444`; Llama: `LLM-Research/Meta-Llama-3.1-8B-Instruct@359efdbb8af05b788a4ad4185215c6b8caa9052c`. Model cache was read-only and offline; all pinned snapshot files were checked before load. The automatic-backend canonical scores and attempt 006 are not merged into this fixed-backend comparison.

## Charges, cleanup and preservation

Provider snapshot 2026-10-22T21:55:54.921047+00:00: attempt 008 metered app charges **$2.13956289**; reported cumulative charges across attempts **$3.99533111**. These are reported resource charges, not a final invoice; recent usage may lag. Missing billing rows for failed attempts are unknown, not zero. Retained result-volume and model-cache storage charges are not covered by this app-resource total; physical storage billing is uncertain.

All eight diagnostic apps are verified STOPPED. An explicit stop request confirmed that the detached app had already stopped after completion; evidence volumes and the isolated workspace remain retained. Prior attempts, canonical records, manuscripts, preregistration and research/release refs are preserved. No commit, push, publication or submission was performed.

## Evidence

- [Independent complete audit](gpu-attempt-008/independent-audit.json)
- [Per-window paired effects](gpu-attempt-008/paired-window-effects.csv)
- [Registration and declared analysis](stage008/registration.json)
- [Raw outputs and per-token rows](gpu-attempt-008/artifacts/)
- [Offline staged-policy/cache checks](offline/008-preflight.json)
- [Independent NumPy/packed checks](offline/008-real-range-reference.json)
- [Provider billing](provider-20261022T215554Z/billing.json)
- [Prior report: attempts 001–007 and separate canonical scores](REPORT.md)
