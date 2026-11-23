# RABIT-KV research update: KV-cache quantization on Qwen and Llama

*Ryan Zhang, November 2026.*

**What this is.** A research update that follows the [October 2026 paper](../../paper/v2/main.pdf). It reports serving-engine experiments on Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct, made after the paper.

**What it does not change:**
- the paper;
- the v0.1.0 release;
- any earlier result.

The README sections on the released operating point remain the reference for the released system.

**Evidence:** [`results/experiments/kv_cache_2026_11/`](../../results/experiments/kv_cache_2026_11/). Every number below is recomputed from two public tables by one script (`scripts/recompute.py`). The script checks 257 values against the frozen analysis outputs, and all agree.

## 1. Summary

Three cache configurations are compared. They are kept separate throughout.

| | Configuration | What changes | Where it exists |
|---|---|---|---|
| **A** | **Released operating point** | K3/V2/G32/R4/META8g64. The open key group is quantized as a 32-token group padded with zeros | the released serving engine and v0.1.0 |
| **B** | **Padding fix** | As A, but the open key group's range is taken from its real tokens only. Stored pages, metadata and capacity are unchanged | research patch to the serving engine; **not in this repository** |
| **C** | **Padding fix + BF16 K minima ("K-min")** | As B, but each closed page stores its per-channel key minima in BF16 in **every layer**, instead of META8g64. Key steps and value metadata stay META8g64 | research patch; **not in this repository** |

The findings fall into two groups.

**Exploratory results** (privately frozen in advance, not confirmatory):

| Study | Data | Result |
|---|---|---|
| H01 | 32 exploratory Qwen windows | **A → B:** Qwen serving perplexity falls by about 92–93 %. B remains +10.9 % / +12.2 % above BF16 |
| C3 | the same 32 windows | **B → C:** a further −8.7 % / −9.3 %. These windows were also used for debugging, so this result can only motivate a confirmation |
| H1 | BF16 replay on C4's windows | **Mechanism:** switching to BF16 minima reduces Qwen's attention-score error strongly in layer 0 and the last layer. For Llama the reduction is negligible. **Within a model, the size of the reduction does not predict which windows gain** |

**Confirmatory result** (C4, on fresh windows):

| Claim | One-chunk | Two-chunk | Judgement |
|---|---|---|---|
| **S1:** for Qwen, C beats B | −4.28 % | −4.20 % | **holds** |
| **S2:** Qwen C within +3 % of BF16 | | | **not supported** |
| **S3:** Llama C no more than +0.5 % worse than B | | | **not shown** |

C4 as a whole **did not pass**, because its pre-specified definition required both S1 and S3.

**Costs:**
- C reduces paged KV capacity by **3.72 %** (Llama: 5.28× → 5.08× BF16).
- No latency or throughput was measured for B or C.

## 2. Studies and data

All studies use:
- WikiText-2 continuation perplexity, measured in the vLLM-based RABIT-KV serving engine;
- one NVIDIA H100 80GB HBM3, one request at a time;
- windows of 1,024 context tokens and 128 teacher-forced scored tokens;
- full-vocabulary FP32 NLL with an FP64 cross-check and an engine-logprob check;
- two prefill schedules: one chunk (budget 16,384), and two chunks (512 + 512).

**The window is the unit of analysis.** Each window's score is the mean of its 128 token NLLs. Intervals are paired over windows: a 20,000-resample percentile bootstrap and a t interval.

| Study | Status | Windows | Cells |
|---|---|---|---|
| **H01** | exploratory, estimation only; protocol frozen privately before scoring | 32 WikiText-2 *validation* windows (Qwen), held out from earlier scoring only | A, B, BF16 × 2 schedules |
| **C3 a2** | exploratory; decision rule frozen privately; the H01 windows were by then used for debugging | the same 32 windows | B, C, BF16 × 2 schedules |
| **C4** | **confirmatory**: protocol, hypotheses, error control and analysis code frozen privately *before* the windows were built | 32 fresh validation windows per model (R01 for Qwen, R01-L for Llama), at least two unscored windows away from every H01 window | B (twice), C, BF16 × 2 schedules; one attempt per model |
| **H1** | exploratory mechanism diagnostic (§5); registered privately before the run | C4's windows, both models | BF16 replay; no cache was run |

**Caveats on the data:**
- **Not a public preregistration.** "Frozen" means fixed and hashed locally before the data existed or were scored. No public preregistration was made.
- **Not a new population.** C4's windows come from the same split as H01's: fresh windows, but they may share Wikipedia articles with neighbouring H01 windows.
- **Limited scope.** One corpus, one model per family, one GPU type.

## 3. Exploratory results on Qwen (H01, C3)

**Perplexity, 32 exploratory windows:**

| Schedule | BF16 | A: released | B: padding fix | C: padding fix + K-min |
|---|---:|---:|---:|---:|
| One chunk | 7.784 | 120.86 | 8.632 | 7.880 |
| Two chunks | 7.784 | 112.02 | 8.736 | 7.925 |

**Paired contrasts** (bootstrap 95 % interval; windows higher / lower):

| Contrast | One chunk | Two chunks |
|---|---|---|
| A − BF16 (context) | +1452.8 % [+1182.5, +1790.1]; 32/0 | +1339.2 % [+1135.9, +1571.9]; 32/0 |
| **B − A** (H01) | **−92.86 %** [−94.13, −91.30]; 0/32 | **−92.20 %** [−93.30, −90.90]; 0/32 |
| B − BF16 (H01) | +10.90 % [+7.97, +14.06]; 28/4 | +12.23 % [+9.64, +14.98]; 32/0 |
| **C − B** (C3) | **−8.72 %** [−11.23, −6.26]; 4/28 | **−9.29 %** [−11.59, −7.01]; 3/29 |
| C − BF16 (C3) | +1.23 % [+0.74, +1.72]; 25/7 | +1.81 % [+0.98, +2.63]; 25/7 |

**How to read this:**
- **The padding fix is a serving result.** B reproduces in the serving engine what the paper's logical-evaluator diagnostics predicted: the zero-padded open key group causes most of Qwen's loss.
- **B is not BF16 quality.** Both of its intervals exclude zero, and the gap is spread across windows.
- **C3 is not evidence of the size of the K-min effect.** It was measured on windows already used for debugging, and its size (−8.7 % / −9.3 %) was not reproduced on fresh windows (§4).

## 4. Confirmatory result (C4)

**Hypotheses, each a one-sided test at α = 0.025, both schedules required:**

| Hypothesis | Claim | Contrast | Passes if the upper limit is … |
|---|---|---|---|
| Q1 | S1 | Qwen C − B | < 0 |
| Q2 (tested only after Q1) | S2 | Qwen C − BF16 | ≤ ln 1.03 |
| L | S3 | Llama C − B | ≤ ln 1.005 |

**Error control:**
- each decision bound is the upper limit of a two-sided 95 % interval, and **both** the bootstrap and the t bound must meet the criterion;
- S1 and S3 together keep the family-wise error ≤ 0.05 (Bonferroni);
- S2 is a fixed-sequence step after Q1.

**"C4 passed" required S1 and S3.** No window was excluded (n = 32 everywhere). Baseline B ran twice, in two different code trees, and was bit-identical in all 64 window × schedule pairs per model.

**Perplexity, 32 windows per model:**

| Model | Schedule | BF16 | B: padding fix | C: + K-min | C vs B | C vs BF16 |
|---|---|---:|---:|---:|---:|---:|
| Qwen | one chunk | 7.879 | 8.386 | 8.026 | −4.28 % | +1.87 % |
| Qwen | two chunks | 7.879 | 8.408 | 8.054 | −4.20 % | +2.23 % |
| Llama | one chunk | 7.013 | 7.137 | 7.138 | +0.02 % | +1.79 % |
| Llama | two chunks | 7.013 | 7.122 | 7.151 | +0.40 % | +1.97 % |

**Tests** (bootstrap 95 % interval; t interval; windows higher / lower):

| Hypothesis | One chunk | Two chunks | Result |
|---|---|---|---|
| **Q1** Qwen C − B | −4.28 % [−6.63, −2.25]; t [−6.59, −1.93]; 8/24 | −4.20 % [−6.30, −2.40]; t [−6.23, −2.13]; 5/27 | **pass**: **S1 holds** |
| **Q2** Qwen C − BF16 | +1.87 % [+1.16, +2.60]; t [+1.12, +2.63] | +2.23 % [+1.38, +3.13]; t [+1.31, **+3.16**] | **not met**: **S2 not supported** |
| **L** Llama C − B | +0.02 % [−0.28, +0.32]; t [−0.30, +0.34]; 14/18 | +0.40 % [−0.01, **+0.85**]; t [−0.06, **+0.86**]; 16/16 | **not met**: **S3 not shown** |

What the two failures mean:
- **S3 "not shown" does not mean Llama was shown to get worse.** Llama's two-chunk interval includes zero; it is the pre-set +0.5 % margin that is not met.
- **S2 fails on its upper limit only.** The two-chunk point estimate is +2.23 %, but the upper limit exceeds +3 %.

**Descriptive, not tested:**
- **Qwen, B vs BF16:** B is +6.43 % (one chunk) and +6.71 % (two chunks) above BF16 on these windows.
- **Qwen, how much of that gap C removes:** 0.70 (one chunk) and 0.66 (two chunks).
- **Llama:** C removes none of its +1.8 % / +1.6 % gap to BF16 (shares −0.01 and −0.26).

**What may be said.** On these fresh Qwen windows, serving with BF16 K minima has lower perplexity than the padding-fixed quantized path: −4.3 % and −4.2 %, confirmatory at one-sided α = 0.025.

**What may not be said:**
- that C is within 3 % of BF16, or equivalent to, or lossless relative to BF16;
- that it does not worsen Llama;
- that C4 passed;
- anything about latency or throughput.

## 5. Exploratory mechanism diagnostic (H1)

**Question.** Does switching the stored K minima from META8g64 to BF16 reduce the error of the attention scores? In which layers? And do the windows with the larger reduction gain more in C4?

**Method:**
- **Replay.** A BF16 forward pass through Hugging Face transformers, over each model's own 32 C4 windows (1,151 tokens each). It captures the post-RoPE queries and keys of every layer. Capture leaves the logits bit-identical.
- **Two arms per window and layer.** The same pages, K3 codes and stored steps are used in both. Only the minimum's storage changes: META8g64, or BF16.
- **Error.** For the 127 decode queries, on every page already closed at that query, the metric is e = q·(k̂ − k)/√128 in FP64.
- **dE** = RMS error with META8 minima − RMS error with BF16 minima, in logit units.

**Window summaries:**
- **S_all**, the mean of dE over all layers (primary);
- **S_0**, layer 0 (reported separately);
- **S_rest**, the mean over layers ≥ 1.

**dE over the 32 windows** (median [interquartile range]):

| Summary | Qwen (28 layers) | Llama (32 layers) |
|---|---|---|
| S_0, layer 0 | 0.933 [0.868, 0.949] | 0.00090 [0.00081, 0.00096] |
| S_rest, layers ≥ 1 | 0.124 [0.116, 0.134] | 0.00132 [0.00128, 0.00134] |
| **S_all**, all layers | **0.152** [0.144, 0.162]<br>bootstrap 95 % of the median [0.148, 0.160] | **0.00130** [0.00127, 0.00132]<br>bootstrap 95 % of the median [0.00128, 0.00132] |

**Where Qwen's reduction sits:**
- **Layer 0:** the median score error falls from 1.43 to 0.50 (−65 %).
- **Layer 27, the last:** it falls from 4.36 to 1.10 (−75 %).
- **Layers 1, 3, 13 and 19:** small reductions, about 2–6 %.
- **The other layers:** under about 1 %.

Layer shares, with the denominator stated (median over windows of the per-window ratio):

| Model | Last layer, of the sum over all layers | Layer 0, of the sum over all layers | Last layer, of the sum over layers ≥ 1 (S_rest) |
|---|---:|---:|---:|
| Qwen | 77.0 % | 21.3 % | 97.9 % |
| Llama | 4.1 % | 2.2 % | 4.2 % |

**Llama:** every layer's reduction is between about 0.3 and 1 %, and no layer stands out.

**Association with the C4 gain.** The gain is G = NLL(B) − NLL(C), so positive means C is better. Spearman ρ over the 32 windows, with a bootstrap 95 % interval over windows:

| Summary | Qwen one chunk | Qwen two chunks | Llama one chunk | Llama two chunks |
|---|---|---|---|---|
| **S_all** (primary: Qwen one chunk) | **−0.09** [−0.45, +0.30] | +0.12 [−0.24, +0.46] | −0.22 [−0.55, +0.14] | −0.08 [−0.43, +0.29] |
| S_0 | +0.21 [−0.13, +0.53] | +0.33 [−0.02, +0.62] | +0.25 [−0.10, +0.55] | −0.32 [−0.585, −0.001] |
| S_rest | −0.14 [−0.50, +0.26] | +0.05 [−0.32, +0.43] | −0.23 [−0.56, +0.12] | −0.08 [−0.42, +0.30] |
| F_all (descriptive) | −0.01 [−0.42, +0.39] | −0.29 [−0.66, +0.13] | −0.11 [−0.43, +0.22] | −0.06 [−0.42, +0.31] |

- **16 registered intervals, 15 include zero.** The exception is Llama two-chunk S_0, which is negative and barely excludes zero. With 16 intervals, about one such result is expected by chance.
- Under the pre-registered reading rules, a weak or null ρ weakens only the reading that *this metric predicts window gains*. It does not show that the mechanism is absent.

**H1 in plain terms:**
1. **Which layers get closer to BF16?** For Qwen, mainly the first and the last layer, where the score error falls by two thirds to three quarters. For Llama, every layer moves only slightly.
2. **Do the windows with the larger reduction gain more?** **No clear evidence.**

These two answers differ, and both are reported:
- **Across models,** the larger error reduction goes with the larger gain: Qwen has both, Llama neither.
- **Within a model,** the reduction is nearly the same in every window, while the gains vary widely, so it cannot rank the windows.

**Limitation.** H1 measures the first-order, pre-softmax score error of closed pages, layer by layer, for fixed BF16 inputs. It does not measure:
- how errors propagate through the network;
- the open group or the four recent tokens;
- values, the softmax or the outputs;
- the two-chunk prefill path.

**Reproducibility.** The Pod recomputed one window × layer on CPU and matched the GPU result. Locally, 16 window × layer pairs from saved tensors were recomputed, both with the registered code and with an independent re-implementation:
- every error value agrees;
- one fraction differs by one unit in the last floating-point place (the same count, 412/35,840), so the registered bitwise check formally failed on that pair.

## 6. Costs

**Page size, C vs A or B:**
- the page grows by **3.87 %** (Qwen 12,416 → 12,896 bytes; Llama 24,832 → 25,792);
- measured same-budget token capacity falls by **3.72 %** (Qwen 4,764,960 → 4,587,616 tokens; Llama 2,074,752 → 1,997,536);
- for Llama the capacity over BF16 goes from **5.28× to 5.08×**.

**B leaves stored pages unchanged.**

**Not measured:** latency and throughput for B or C. The engine was instrumented in every run reported here.

## 7. Status of the code and what is published

**The serving-engine changes for B and C are research patches. They are not part of this repository or of the v0.1.0 release.**
- The released engine still runs configuration A, as described in the README.
- Quality measured with B or C must not be combined with A's 5.28× capacity, and C's capacity is the 5.08× above.

**Published here** (`results/experiments/kv_cache_2026_11/`):
- **Tables:**
  - `serving_window_nll.csv`: one row per study, model, window and cell (896 rows), with each window's token hash;
  - `h1_layer_metrics.csv`: one row per model, window and layer (1,920 rows).
- **Frozen outputs:** the analysis results (`frozen/`) and the analysis and metric code (`frozen_code/`), byte for byte (two H1 output files are LF-normalized; see the folder README).
- **Windows:** C4's window specifications and window builder (`windows/`), so the windows can be rebuilt from the public WikiText-2 validation file.
- **`scripts/recompute.py`:** recomputes every number above and checks it against `frozen/`.
- **`provenance.json` and `MANIFEST.sha256`:** per-cell digests of the raw score files, and the hashes of the published files.

**Not published:**
- raw per-token score files;
- Pod logs and archives;
- spend records;
- model tensors;
- credentials.

So the statistics can be re-derived from this repository, but the GPU runs cannot be repeated from it alone.
