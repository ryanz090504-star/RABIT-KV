# RABIT-KV — MLSys 2027 Experiment Plan

This plan is derived entirely from the repository audit (see conversation history / audit
summary). It specifies every experiment needed to move RABIT-KV from its current single
operating-point evidence set to an MLSys-quality evaluation, without touching any frozen
result.

**Nothing in this document has been executed.** This is a plan only.

## Ground rules (apply to every experiment below)

1. **Canonical results are immutable.** `results/quality/`, `results/performance/`, and
   `results/summary.json` are never overwritten, appended to, or used as a write target by
   any experiment in this plan.
2. **All new evidence goes under `results/mlsys2027/`**, in the per-experiment subpaths
   specified below. Each subpath is new; none collide with existing files.
3. **Fake/logical vs physical evaluation must never be conflated.** Experiments run through
   `benchmarks/quality/*.py` measure quality under one-shot HF-side quantize/dequantize of
   the prefix KV cache and report *logical* packed-byte memory. They are quality-only and
   are not deployment evidence. Experiments run through `benchmarks/performance/*.py` (or
   `vllm bench *`) run the real vLLM engine and report *physical* allocator capacity and
   real decode latency. Every table/figure produced by this plan must carry an explicit
   "logical (fake-quant)" or "physical (real engine)" label.
4. **Matched BF16/RABIT-KV runs must be byte-for-byte identical in engine configuration
   except `kv_cache_dtype`.** Any experiment that produces a BF16-vs-RABIT-KV comparison
   must state the full fixed engine config and confirm only the dtype flag differs. Where
   an experiment sweeps a parameter (context length, concurrency/`max_num_seqs`), that
   parameter must be swept identically and simultaneously for every dtype under comparison
   at each point — `kv_cache_dtype` remains the only difference *within* each matched pair,
   not across the sweep as a whole.
5. **No speedup or capacity-gain claim is made until it is measured under this plan.**
   The current repository contains zero committed BF16 latency evidence and no raw log for
   the previously-cited BF16 capacity number (393,024 tokens) — every number in the paper's
   systems section must trace to a `results/mlsys2027/` file produced under this plan.
6. **The HotpotQA regression (60.6 → 55.2 F1, canonical, `results/quality/hotpotqa.log`)
   stays visible everywhere it is currently reported** and is additionally the subject of
   dedicated ablation (P0-D) and error-analysis (P1-15) work — it is not to be diluted or
   removed from any summary table produced under this plan.
7. Every experiment below must, on completion, write a `manifest.json`-style provenance
   record (git commit, script SHA, engine config, method list) into its own output
   directory, following the pattern already used by `benchmarks/quality/run_suite.py`.
8. **No approximate/fake-quant baseline is presented as equivalent to a real-engine
   measurement.** This applies generally (rule 3) and specifically to FP8 (Experiment 4)
   and TurboQuant (Experiment 13): physical capacity/latency claims for these methods must
   come from the real engine, never from a hand-written approximation.

---

## P0-A — Quality frontier

### Experiment 1 — BF16 / 8 / 4 / 3 / 2 quality-compression frontier

- **Research question:** How does downstream quality (continuation PPL, NIAH retrieval,
  passage retrieval, HotpotQA F1, Qasper F1) degrade as a function of *logical* KV
  compression across the bit frontier (8-bit, 4-bit, 3-bit, and the final 2-bit-target
  K3/V2 policy), relative to BF16?
- **Why MLSys reviewers would care:** A single reported operating point (K3V2) cannot
  establish that it is a good point on the quality-compression tradeoff. Reviewers expect
  a frontier plot showing where the chosen method sits relative to more/less aggressive
  variants of itself before comparing to external baselines.
- **Hypothesis:** Quality degrades monotonically (or near-monotonically) as bit-width
  decreases; K3/V2 (rabit2) sits at a knee of the curve; HotpotQA is the most
  compression-sensitive benchmark across the whole frontier, not only at the final point.
- **Control / baseline:** `bf16`.
- **Treatment:** `rabit8` (8b, META8g256, G128, R0), `rabit4` (4b, META8g64, G128, R0),
  `rabit3` (3b, META8g64, G32, R2), `rabit2` (final: K3/V2, META8g64, G32, R4).
  These four configs already exist verbatim in each script's `config_for_method`.
- **Fixed variables:** model, dataset, context/eval token counts, sample counts, seed —
  identical to the canonical committed runs for each benchmark.
- **Existing script(s) to reuse:**
  `benchmarks/quality/continuation_ppl.py`, `multilingual_ppl.py` (handled separately
  in Experiment 2; its canonical run covered bf16/rabit2 only), `niah.py`, `passage_retrieval.py`, `hotpotqa.py`, `qasper.py`.
- **Exact code changes required:** **None.** Pass
  `--methods bf16,rabit8,rabit4,rabit3,rabit2` instead of the canonical `bf16,rabit2`.
- **Model:** `LLM-Research/Meta-Llama-3.1-8B-Instruct`.
- **Dataset/workload:** WikiText-2 test (continuation_ppl), synthetic NIAH prompts at
  4K/8K/16K × 5 depths, LongBench `passage_retrieval_en` (10 samples), LongBench-E
  `hotpotqa` 8k+ (20 samples), LongBench-E `qasper` 8k+ (24 samples) — all identical to the
  canonical suite.
- **GPU:** NVIDIA H100 80GB HBM3, via Modal.
- **Metrics:** PPL and PPL delta % (continuation), exact-match accuracy (NIAH, passage
  retrieval), F1 and F1 delta (HotpotQA, Qasper), logical avg KV MB, logical compression ×.
- **Repetitions/samples:** identical to canonical per benchmark (8 / 15 / 10 / 20 / 24) —
  one run per method; statistical repetition is deferred to P1-12.
- **Raw output path:**
  `results/mlsys2027/quality_frontier/{continuation_ppl,niah,passage_retrieval,hotpotqa,qasper}.log`
  (+ one `manifest.json`).
- **Paper figure/table:** Figure — "Quality vs. logical compression frontier" (one panel or
  line per benchmark, x = logical compression ×, y = quality delta, points at 8/4/3/2-bit).
  Table — per-bit-width, per-benchmark deltas including the canonical 2-bit row for
  cross-reference.
- **Completion criterion:** all 5 scripts complete for all 5 methods with exit code 0; each
  script's own printed aggregate table is captured in the log; frontier table cross-checked
  against the canonical `bf16`/`rabit2` numbers (must match `results/quality/*.log` for
  those two methods, confirming no drift).
- **Estimated engineering difficulty:** Low (no code changes; CLI-argument change only).
- **Estimated GPU cost:** Low. ~2–3× the per-script cost of the canonical bf16/rabit2 run
  (3 additional methods vs. 1 additional method previously); each script's canonical run
  completed in well under an hour end-to-end on H100 including model download — expect on
  the order of a few H100-hours total across all 5 scripts.

### Experiment 2 — Extend multilingual support to 4-bit and 3-bit

- **Research question:** Does the English quality-compression frontier trend
  (Experiment 1) hold for Chinese and Spanish continuation PPL at the 4-bit and 3-bit
  points, not just 8-bit and the final 2-bit target?
- **Why MLSys reviewers would care:** Multilingual generalization is currently only shown
  at the final 2-bit target: the canonical multilingual run
  (`results/quality/multilingual_ppl.log`, `run_suite.py` `--methods bf16,rabit2`) contains
  **only `bf16` and `rabit2`**. There is **no canonical `rabit8` multilingual result**. A
  reviewer will ask whether the omission of 8/4/3-bit multilingual data is because it was
  unfavorable or simply unrun. Closing this gap removes an easy rejection point.
- **Hypothesis:** the shape of the multilingual degradation curve mirrors English.
  At the final 2-bit target, Spanish shows a slightly larger relative PPL delta than Chinese
  (+2.21% vs +2.11%). Experiment 2 tests whether language-specific sensitivity diverges at
  the intermediate operating points.
- **Control / baseline:** `bf16`.
- **Treatment:** `rabit8`, `rabit4`, `rabit3`, `rabit2`.
- **Canonical regression baselines:** `bf16` and `rabit2` are the **only** methods with a
  canonical multilingual result, and therefore the only ones regression-checked. `rabit8`,
  `rabit4` and `rabit3` are **all new MLSys 2027 evidence**; no claim is made that `rabit8`
  reproduces a canonical multilingual result.
- **Fixed variables:** languages `zh,es`; pinned dataset revision
  `cf584d1dc131caa92a5cb910f41a8b7591b12732`; shuffle seed `20260804`, shuffle buffer 1000;
  context/eval tokens 1024/128; samples/language = 8.
- **Existing script to reuse:** `benchmarks/quality/multilingual_ppl.py`.
- **Exact code changes required:** `config_for_method` in `multilingual_ppl.py` already
  contains `rabit4` and `rabit3` entries identical to `continuation_ppl.py` (3b: K3/V3
  group_sym, G32, R2, META8g64; 4b: K4/V4 group_sym, G128, R0, META8g64); the whole
  preset/quantization/accounting/evaluation section is byte-identical between the two
  scripts, and the Experiment 2 runner re-verifies this at preflight. The only change is to
  extend the `allowed` set from `{"bf16","rabit8","rabit2"}` to
  `{"bf16","rabit8","rabit4","rabit3","rabit2"}` (plus its error message). The default
  `methods` string stays `bf16,rabit8,rabit2` so a default invocation behaves exactly as
  before; the runner passes `--methods bf16,rabit8,rabit4,rabit3,rabit2` explicitly. No
  change to `bf16`/`rabit8`/`rabit2` logic.
- **Runner:** `benchmarks/mlsys2027/run_experiment2_multilingual_frontier.py`.
- **Model:** `LLM-Research/Meta-Llama-3.1-8B-Instruct`.
- **Dataset/workload:** pinned Chinese/Spanish Wikipedia revision (as canonical).
- **GPU:** H100 80GB, Modal.
- **Metrics:** PPL, PPL delta % per language, mean/worst relative delta across languages.
- **Repetitions/samples:** 8 samples/language (matches canonical).
- **Raw output path:** `results/mlsys2027/multilingual_frontier/multilingual_ppl.log`
  (with `manifest.json` and `regression_check.json` alongside).
- **Paper figure/table:** extends Experiment 1's frontier table/figure with zh/es rows at
  all 4 bit-widths.
- **Completion criterion:** script runs to completion for all 5 methods across both
  languages; **`bf16` and `rabit2`** reproduce the canonical
  `results/quality/multilingual_ppl.log` values for both languages within explicit
  tolerances (PPL 0.5% relative, avg logical KV MB 0.1% relative — the same tolerances as
  Experiment 1). `rabit8`/`rabit4`/`rabit3` have no canonical baseline and are recorded as
  new evidence, not regression-checked.
- **Estimated engineering difficulty:** Low–medium (small, isolated code addition plus a
  regression check against canonical numbers).
- **Estimated GPU cost:** Low, comparable to Experiment 1's per-script cost.

---

## P0-B — Matched deployment baselines

### Experiment 3 — BF16 vs RABIT-KV matched latency and physical capacity

- **Research question:** What is RABIT-KV's real decode latency (TPOT/TTFT/wall) and
  physical KV-allocator capacity relative to a BF16 baseline, under an engine configuration
  identical in every respect except `kv_cache_dtype`?
- **Why MLSys reviewers would care:** This is the paper's central systems claim and its
  largest current evidence gap. The repository has zero committed BF16 latency runs and no
  raw log for the previously-asserted BF16 capacity figure (393,024 tokens appears only in
  derived JSON, never in a run log). Without this experiment, no capacity-gain or latency
  claim is defensible.
- **Hypothesis:** capacity gain is large (the fake-quant/logical accounting already
  suggests ~5×) and reproducible under a real matched run; latency direction is *not*
  assumed — RABIT-KV's online packing/aging/dequant path may add per-token overhead in
  eager mode that a naive reader would not expect from a "compression" method. Report
  whatever is measured.
- **Control:** explicit `kv_cache_dtype="bfloat16"` (native BF16 KV cache). Not `auto`:
  in vLLM, `auto` resolves to the model dtype (BF16 here) but can be silently overridden
  by a checkpoint `kv_cache_scheme`/`quantization_config`; an explicit value cannot. The
  run records the actually resolved KV dtype (engine `cache_dtype` + resolved torch dtype,
  worker KV-cache spec) and cross-checks it physically (implied bytes/token must match
  2-byte BF16 KV within 1%).
- **Treatment:** `kv_cache_dtype=rabit_kv2`.
- **Variables that must remain fixed:** eager execution (`enforce_eager=True`), Triton
  attention backend, CUDA graphs disabled, `torch.compile` disabled,
  `gpu_memory_utilization=0.82`, `block_size=32`, `max_model_len=32768`,
  `max_num_batched_tokens=16384`, `max_num_seqs=32`, prefix caching disabled, chunked
  prefill enabled, context tokens = 2048, output tokens = 32, same model checkpoint, same
  GPU/Modal image, same vLLM commit (`f329ce4...`).
- **Canonical script (NOT modified):** `benchmarks/performance/benchmark_deployment.py`.
  Its measurement code is a zlib+base64 blob (`RUNNER_Z`, `kv_cache_dtype="rabit_kv2"`
  hardcoded), its `install()` can append a patch to `rabit_kv2.py`, and it writes into the
  protected `results/performance/`. It stays untouched as the provenance of the canonical
  result.
- **New Experiment 3 files instead:** `benchmarks/mlsys2027/`
  `run_experiment3_deployment.py` (local runner), `exp3_deployment_modal.py` (Modal app),
  `exp3_engine_worker.py` (one engine per leg, parameterized only by `kv_cache_dtype`),
  `exp3_correctness_gate.py` (canonical `regression()` verbatim). Before every run the
  runner proves by AST against the decoded canonical `RUNNER_Z` that the worker's
  `LLM(...)` arguments are identical except `kv_cache_dtype`, the Modal image expression is
  identical, the gate's `regression()` is identical, and the TTFT/TPOT/wall and prompt
  definitions are shared. The `vllm-kvquant` snapshot is `git archive` of the committed tree.
- **Protocol:** one Modal container / one physical H100 / one image / one model snapshot.
  (1) Record the idle GPU baseline. (2) Correctness gate in a fresh process (dispatch
  preflight, Stage4D3.4 prep + attention exactness, fast decode-append exactness, pytest on
  `test_kvquant_k3.py` + `test_rabit_kv2*.py` excluding `stage4b1`); must pass before any
  measurement and is not timed. (3) Counterbalanced **ABBA** legs, A = `bfloat16`,
  B = `rabit_kv2`: A1, B1, B2, A2, each a fresh worker/engine process with **5 full-shape
  warmups** (2048 in / 32 out, excluded from statistics; replaces the canonical 2 × 8-token
  warmup, after which rep 0 was still a 50 ms outlier) and **15 measured reps** → **30
  measured reps per dtype**.
- **GPU clean state:** before every leg, no GPU compute process and `memory.used` within
  256 MiB of the idle baseline (polled at most 60 s while a previous process releases
  memory; a wait, never a rerun). A stale allocation hard-fails the experiment rather than
  shrinking the next dtype's allocator capacity.
- **Matched-config enforcement:** requested kwargs, effective engine config, workload
  (including a hash of the prompt token IDs), resolved KV dtype and worker KV spec are
  flattened per leg and compared across all four legs. Only the kv_cache_dtype-induced
  allowlist (`requested.kv_cache_dtype`, `kv_dtype.requested_kv_cache_dtype`,
  `kv_dtype.engine_cache_dtype`, `kv_dtype.resolved_kv_torch_dtype`,
  `kv_cache_representation.kv_spec`) may differ, and only between dtypes, never between
  the two legs of one dtype. Any other difference hard-fails.
- **Model:** `LLM-Research/Meta-Llama-3.1-8B-Instruct`.
- **Dataset/workload:** synthetic single-request decode microbenchmark, 2048-token prefill,
  32 generated tokens (identical to canonical methodology).
- **GPU:** NVIDIA H100 80GB HBM3.
- **Metrics:** TPOT median/p50/p90, TTFT median, wall-time median, allocator capacity
  (tokens), capacity ratio ×, latency delta % (RABIT-KV vs BF16, signed — do not report as
  "speedup" unless negative/positive is confirmed).
- **Repetitions/samples:** ABBA, 15 measured reps per leg = 30 per dtype (vs. canonical
  5), plus 5 excluded full-shape warmups per leg. All raw samples, per-leg statistics and
  pooled 30-sample statistics are reported; headline TPOT median/p90, TTFT median and wall
  median come from the pooled samples. Capacity (`num_gpu_blocks × block_size`, physical
  allocator) is measured in both legs of each dtype and the two measurements must agree
  exactly, else hard fail.
- **BF16 capacity:** the historical 393,024-token BF16 figure is a derived value with no
  run log in the repository; it is NOT reused. The new BF16 capacity is directly measured
  by this experiment.
- **Raw output path:** `results/mlsys2027/deployment/`: `modal_session.log`,
  `correctness_gate.log`, `bf16_deployment.log`, `rabit_kv2_deployment.log`,
  `manifest.json`, `matched_config_diff.json`, `integrity_check.json`,
  `matched_capacity_latency_summary.json`.
- **Paper figure/table:** Table — "Matched BF16 vs. RABIT-KV deployment" (capacity tokens,
  ratio, TPOT/TTFT/wall absolute + delta %). This is the headline systems table for the
  paper.
- **Completion criterion:** correctness gate passes; all four ABBA legs complete; every
  integrity check passes (matched config per the allowlist rule, GPU clean before every
  leg, duplicate capacities identical, resolved KV dtypes as expected, 5 warmups + 15 reps
  per leg with 2048/32 tokens, no Triton JIT during measurement); summary JSON contains the
  capacity ratio and signed (rabit_kv2 − bf16) latency deltas from pooled samples; the plan
  document and any paper draft state the measured direction of latency change explicitly
  (no assumed speedup).
- **Estimated engineering difficulty:** Medium (careful, programmatically verified
  equivalence to the canonical engine configuration).
- **Estimated GPU cost:** Low. One correctness gate (~1–2 min) plus four engine boot-ups
  (~90 s each, per canonical log) and 4 × 20 short requests; well under an H100-hour. No
  automatic retries; any rerun is an explicit decision.

### Experiment 4 — FP8 baseline: matched physical capacity and latency (real engine only)

- **Research question:** How does RABIT-KV compare to the fork's native, real-engine FP8
  KV-cache dtype in physical allocator capacity and real decode latency, with BF16, FP8 and
  RABIT-KV all measured fresh in the **same** matched session?
- **Why MLSys reviewers would care:** FP8 KV cache is already a supported, well-understood
  baseline in vLLM itself. A paper claiming novel KV compression that never compares
  against the framework's own FP8 option invites an immediate "why not just use FP8"
  question — but that comparison must be a real measurement, not an approximation.
- **Hypothesis:** FP8 stores 1 byte per KV element vs. BF16's 2, so a capacity gain over
  BF16 is expected; RABIT-KV's gain over FP8 is expected to be smaller than its gain over
  BF16. Neither ratio is hard-coded or assumed — both are measured. Latency direction for
  FP8 vs. BF16 vs. RABIT-KV is not assumed either; report whatever is measured.
- **Explicit rule (per audit correction):** Do **not** create a hand-written FP8 fake-quant
  quality baseline and present it as equivalent to vLLM's native FP8 path. This experiment
  produces **physical capacity and latency only**. See "FP8 quality — status" below.
- **No reuse of Experiment 3 samples:** Experiment 3's run #1 and independent replication
  showed substantial cross-session variation in absolute latency, so BF16 and RABIT-KV are
  **freshly re-measured in this Experiment 4 matched session**, alongside FP8. No Experiment 3 latency or capacity sample is
  read, reused or pooled; Experiment 3 results are not cross-referenced for any Experiment 4
  number.
- **Native FP8 audit (vllm-kvquant source, read-only):**
  - Requested dtype: `kv_cache_dtype="fp8_e4m3"`. `fp8` is an exact alias on CUDA (same
    `STR_DTYPE_TO_TORCH_DTYPE` entry, same `FP8_PER_TENSOR` quant mode, both accepted by the
    query-quant assertion); `fp8_e4m3` is chosen because it names the format explicitly.
    `fp8_e5m2` is excluded (on CUDA the FP8 query-quant path asserts `fp8`/`fp8_e4m3`);
    `fp8_inc` (Gaudi), `fp8_ds_mla` (MLA) and `fp8_per_token_head` (a different, scaled
    per-token-head mode) are not native per-tensor FP8 for this model.
  - Resolution: engine `cache_dtype = "fp8_e4m3"`; storage `torch.uint8`; the Triton
    backend views it as `current_platform.fp8_dtype() = torch.float8_e4m3fn`;
    `get_kv_quant_mode → FP8_PER_TENSOR`.
  - KV scales: the Llama-3.1-8B-Instruct checkpoint has no `quantization_config`, so no
    `BaseKVCacheMethod` is attached and `set_default_quant_scales()` leaves k/v/q scales at
    1.0; `calculate_kv_scales` stays `False` (deprecated option, not set).
  - Checkpoint override: `resolve_kv_cache_dtype_string` and the attention layer only
    rewrite `auto`; an explicit `fp8_e4m3` cannot be overridden by the checkpoint.
  - Query: on CUDA, for `fp8`/`fp8_e4m3` KV caches the attention layer also quantizes the
    query to FP8 (static per-tensor, scale 1.0) before the Triton kernel; this is part of the
    native FP8 path being measured.
  - Hardware: Triton FP8 KV requires SM89+ (checked in `TritonAttentionImpl` and
    `triton_reshape_and_cache_flash`); H100 is SM90.
  - Every leg records `calculate_kv_scales`, `kv_cache_dtype_skip_layers` and the checkpoint
    `quantization_config`; each must be `False` / `[]` / `None` on all six legs.
- **Baselines and treatment:** A = `bfloat16` (explicit, not `auto`), B = `fp8_e4m3`,
  C = `rabit_kv2` — all three freshly re-measured in this Experiment 4 matched session.
  The FP8 baseline is **native FP8 E4M3 on this checkpoint, using the engine's default
  scale behavior (scale 1.0), with native query FP8 conversion as part of the measured
  backend path.** It is a physical capacity/latency baseline only; no FP8
  quality-equivalence claim is made.
- **Protocol (mirrored A-B-C-C-B-A):** one Modal container / one physical H100 / one image /
  one model snapshot. (1) Idle GPU baseline. (2) The frozen RABIT-KV correctness gate
  (`exp3_correctness_gate.py`, unchanged) runs once and must pass before any measurement.
  (3) Six legs in the order **A1, B1, C1, C2, B2, A2**, each a fresh worker/engine process
  with **5 full-shape warmups** (excluded) and **15 measured reps** → **30 measured samples
  per dtype** (BF16, FP8, RABIT-KV).
- **Variables that must remain fixed:** identical to Experiment 3 — eager execution, Triton
  attention backend, CUDA graphs disabled, `torch.compile` disabled,
  `gpu_memory_utilization=0.82`, `block_size=32`, `max_model_len=32768`,
  `max_num_batched_tokens=16384`, `max_num_seqs=32`, prefix caching disabled, chunked
  prefill enabled, model dtype `bfloat16`, context tokens = 2048, output tokens = 32, greedy
  decoding, identical prompt token IDs on every leg. Only `kv_cache_dtype` differs.
- **Code (new Experiment 4 files; Experiment 3 files are not modified):**
  `benchmarks/mlsys2027/run_experiment4_fp8_baseline.py` (local runner),
  `exp4_deployment_modal.py` (Modal app), `exp4_engine_worker.py` (one engine per leg). The
  frozen Experiment 3 worker cannot run FP8 (its `--kv-cache-dtype` choices are
  `bfloat16`/`rabit_kv2`), so a derived worker is used. Before any run the runner proves by
  AST that the Experiment 4 worker's engine kwargs equal the canonical runner and the
  Experiment 3 worker, that its workload constants, prompt construction and timed region are
  identical to Experiment 3's, that the Modal image equals the canonical image, that the
  Modal clean-state and watchdog helpers equal Experiment 3's, and that the gate's
  `regression()` equals the canonical one. The gate and watchdog are the unchanged,
  committed Experiment 3 files.
- **Safety / robustness (as Experiment 3):** idle GPU baseline; before every leg no compute
  process and `memory.used` within 256 MiB of baseline; process-group watchdog (gate 600 s,
  each leg 900 s; whole group killed on timeout; a timeout aborts the run); no automatic
  retries; protected-path post-check on every terminal path; every integrity check is
  `passed` / `failed` / `not_run` / `not_evaluated`, and a summary exists only if all pass.
  Protected: canonical results, `vllm-kvquant`, Experiment 1 and 2 outputs, all Experiment 3
  evidence (run #1, `failed_attempt_1`, `replication_1`, `replication_comparison.json`) and
  Experiment 1–3 code.
- **Matched-config enforcement:** requested kwargs, effective engine config, workload
  (including a hash of the prompt token IDs) and resolved KV dtype are flattened per leg and
  compared across all six legs. Only the dtype-induced allowlist
  (`requested.kv_cache_dtype`, `kv_dtype.requested_kv_cache_dtype`,
  `kv_dtype.engine_cache_dtype`, `kv_dtype.resolved_kv_torch_dtype`,
  `kv_dtype.kv_quant_mode`, `kv_dtype.fp8_storage_view_dtype`) may differ, and only between
  dtypes, never between the two legs of one dtype. Any other difference hard-fails.
- **Capacity:** per leg: requested/resolved KV dtype, `num_gpu_blocks`, `block_size`,
  physical capacity tokens, logged available KV memory and implied bytes/token. Duplicate
  capacities must be identical (A1 = A2, B1 = B2, C1 = C2), else hard fail. Implied
  bytes/token is cross-checked against 2-byte (BF16) and 1-byte (FP8) KV elements within 1%
  (an element-size check, not a capacity-ratio expectation); RABIT-KV bytes/token is
  reported only. Reported ratios: FP8/BF16, RABIT/BF16, RABIT/FP8.
- **Latency:** raw samples kept for every leg; per leg TPOT median/p90, TTFT median, wall
  median; pooled within each dtype only (30 samples each). Pairwise signed deltas
  FP8 − BF16, RABIT − BF16, RABIT − FP8 for TPOT median, TPOT p90, TTFT median and wall
  median, worded slower/faster only according to the measured sign.
- **Order effects:** A1 vs A2, B1 vs B2, C1 vs C2 — median drift %, raw sample ranges and
  range overlap for TPOT, TTFT and wall. Drift is reported, never hidden.
- **FP8 functional check:** FP8 must initialize and complete the measured workload (2048
  prompt / 32 output tokens on every request, SHA-256 of generated token IDs recorded per
  request). Within-dtype determinism of those hashes is reported as functional evidence
  only; it is not a quality metric and hashes are not compared across dtypes.
- **FP8 quality — status:** Not attempted in P0. The existing quality scripts
  (`benchmarks/quality/*.py`) are HF-side fake-quant only, with no real-engine code path;
  producing a genuine FP8 quality number requires a real-engine quality harness (prefill +
  generate through the actual vLLM engine with the FP8 KV cache, scored the same way as
  the canonical benchmarks), which does not exist today and is nontrivial new
  infrastructure. Until that harness is built, FP8 quality is reported in the paper as
  **"not evaluated — requires real-engine harness,"** not approximated with a hand-written
  fake-quant stand-in. If pursued, this becomes new P1 work, and may share infrastructure
  with Experiment 16 (full TurboQuant quality integration).
- **Model:** `LLM-Research/Meta-Llama-3.1-8B-Instruct`.
- **Dataset/workload:** same synthetic single-request decode microbenchmark as
  Experiment 3 (2048-token prefill, 32 generated tokens).
- **GPU:** NVIDIA H100 80GB HBM3.
- **Raw output path:** `results/mlsys2027/fp8_baseline/`: `modal_session.log`,
  `correctness_gate.log`, `bf16_deployment.log`, `fp8_e4m3_deployment.log`,
  `rabit_kv2_deployment.log`, `manifest.json`, `matched_config_diff.json`,
  `integrity_check.json`, `matched_capacity_latency_summary.json`.
- **Paper figure/table:** a three-row matched deployment table (BF16, FP8, RABIT-KV) from this
  single session: capacity tokens and pairwise ratios, TPOT/TTFT/wall absolute + signed
  deltas. The quality column for the FP8 row is explicitly marked "not evaluated," never
  left blank or silently omitted.
- **Completion criterion:** correctness gate passes; all six legs complete; every integrity
  check passes (matched config per the allowlist rule, GPU clean before every leg, duplicate
  capacities identical, resolved KV dtypes as expected, 5 warmups + 15 reps per leg with
  2048/32 tokens, no Triton JIT during measurement); the summary contains the three capacity
  ratios, the pairwise signed latency deltas and the order effects from this session only;
  no FP8 quality number appears anywhere in P0 output.
- **Estimated engineering difficulty:** Medium (verified reuse of the Experiment 3 machinery;
  FP8's functional status under this eager/Triton configuration is confirmed only by the
  run itself).
- **Estimated GPU cost:** Low–moderate. One correctness gate plus six engine boot-ups and
  6 × 20 short requests in one container; roughly 1.5× Experiment 3. No automatic retries;
  any rerun is an explicit decision.

---

## P0-C — Systems scaling

### Experiment 5 — Context-length scaling

- **Research question:** How do real decode latency (TPOT/TTFT/wall), actual KV cache
  memory/pool usage, and the maximum feasible context length scale with prefill context
  length for BF16 vs. RABIT-KV?
- **Important scoping note (per audit correction):** the vLLM allocator's reported
  *physical capacity* (the token count established at engine startup, per Experiment 3) is
  a property of the fixed engine configuration and GPU memory budget — the engine does not
  reallocate the KV cache pool per request or per context length. This experiment therefore
  does **not** re-measure "capacity vs. context length" as if capacity varies with
  workload; capacity is measured once per dtype as a single matched point measurement
  (Experiment 3) and is not re-derived here. What genuinely varies with context length is
  per-request latency and how much of the fixed pool a given request's context actually
  occupies at runtime — that is what this experiment measures.
- **Why MLSys reviewers would care:** The paper's only current deployment evidence is a
  single 2048-token context point, despite `max_model_len=32768` being part of the
  configuration. A systems paper claiming long-context benefit needs a latency curve and a
  maximum-feasible-context result, not a single point — and needs to state the capacity
  metric correctly rather than implying it changes with workload.
- **Hypothesis:** RABIT-KV's per-token decode overhead relative to BF16 (from Experiment 3)
  may grow, shrink, or stay flat as context grows — must be measured, not assumed.
  Separately, RABIT-KV's smaller per-token memory footprint is expected to let it reach
  materially longer maximum feasible context under the fixed `gpu_memory_utilization=0.82`
  budget than BF16 before OOM — this is the correct way to express a "long-context benefit"
  claim from this sweep (maximum feasible context length), not a re-measurement of
  allocator capacity per point.
- **Control:** `bf16` at each context length.
- **Treatment:** `rabit_kv2` at each context length.
- **Variables that must remain fixed:** all engine args from Experiment 3 except context
  length; output tokens fixed at 32; ≥10 reps per (dtype, context-length) cell.
- **Existing script to reuse:** `benchmarks/performance/benchmark_deployment.py`
  (dtype-parameterized per Experiment 3).
- **Exact code changes required:** add a `--context-tokens` CLI parameter to the embedded
  runner (currently hardcoded to 2048); add a thin outer sweep driver (new script,
  e.g. `benchmarks/performance/sweep_context_length.py`) that repeatedly invokes the
  parameterized deployment benchmark across the context grid, and at each point records
  TPOT/TTFT/wall plus the actual KV blocks/tokens consumed by the live request (from the
  engine's runtime state/logs) — not a re-query of allocator capacity, which the driver
  does not touch. No changes to the RABIT-KV implementation itself.
- **Model:** `LLM-Research/Meta-Llama-3.1-8B-Instruct`.
- **Dataset/workload:** synthetic prefill at context lengths `{512, 2048, 4096, 8192,
  16384, 32768}`, 32 generated tokens per point (methodology matches canonical).
- **GPU:** NVIDIA H100 80GB HBM3.
- **Metrics:** TPOT/TTFT/wall vs. context length (both dtypes); actual KV memory/blocks
  consumed by the live request at each context point (distinct from, and clearly labeled
  separately from, the Experiment 3 allocator-capacity measurement); maximum feasible
  context length reached by each dtype under the fixed memory budget before OOM/rejection;
  a live-usage compression ratio at matched context length (actual bytes/blocks used,
  RABIT-KV vs. BF16) — labeled explicitly as distinct from Experiment 3's allocator-
  capacity ratio.
- **Repetitions/samples:** ≥10 reps per (dtype, context-length) cell (12 cells total).
- **Raw output path:**
  `results/mlsys2027/context_scaling/{bf16,rabit_kv2}_ctx{N}.log` + `summary.json`.
- **Paper figure/table:** Figure — "Latency vs. context length" (two lines: BF16,
  RABIT-KV); Table — "Maximum feasible context length, BF16 vs. RABIT-KV"; Figure — "Actual
  KV memory usage vs. context length" (live usage, not allocator capacity).
- **Completion criterion:** full sweep completes for both dtypes across all six context
  points without OOM or error where feasible; summary table complete; each dtype's maximum
  feasible context length under `gpu_memory_utilization=0.82` is identified and reported —
  if BF16 cannot reach a given context length, that is reported explicitly as a BF16
  ceiling (a finding, not an assumed magnitude), not silently skipped; allocator capacity
  itself is not re-reported per point, only cross-referenced from Experiment 3.
- **Estimated engineering difficulty:** Medium (parameterization plus sweep orchestration;
  must handle possible BF16 OOM at the largest context points gracefully and report it as a
  finding).
- **Estimated GPU cost:** Medium–high. 6 context points × 2 dtypes × ≥10 reps, plus 12
  separate engine boot-ups; largest context points (16K/32K) will dominate wall-clock cost.

### Experiment 6 — Concurrency / throughput scaling

- **Research question:** How does serving throughput scale with concurrent request count
  for BF16 vs. RABIT-KV, and does RABIT-KV's capacity advantage translate into materially
  higher sustained concurrency before throughput degrades or requests fail?
- **Why MLSys reviewers would care:** Single-stream decode latency does not demonstrate the
  practical payoff of a KV-compression method — the systems argument for KV compression is
  almost always "more concurrent requests in the same HBM," which this repository has never
  measured.
- **Hypothesis:** RABIT-KV sustains substantially higher maximum concurrency (roughly
  proportional to the capacity ratio measured in Experiment 3) before throughput
  degradation or request rejection, at some added per-request latency cost consistent with
  Experiments 3 and 5.
- **Control:** `bf16` across a concurrency sweep.
- **Treatment:** `rabit_kv2` across the same concurrency sweep.
- **Explicit workload definition (per audit correction — fixed before any run):**
  - **Concurrency grid:** `{1, 4, 8, 16, 32, 64}` (extendable upward for RABIT-KV only if
    BF16 fails first — see `max_num_seqs` rule below).
  - **Request count per point:** a fixed total of 256 requests issued per concurrency
    level, enough to keep the target concurrency saturated for a stable measurement window;
    identical count at every point and for both dtypes.
  - **Prompt length:** fixed per sweep point, reusing 2048 and 8192 tokens from
    Experiment 5 as two separate concurrency sweeps (i.e., this experiment runs twice, once
    per prompt length, each with its own full concurrency grid).
  - **Output length:** fixed at 32 generated tokens per request, matching Experiment 3's
    canonical protocol. (A longer output length, e.g. 128, is a documented option for a
    follow-up but must not be varied within a single sweep.)
  - **Warmup protocol:** 2 warmup requests issued and discarded before measurement begins
    at each concurrency point (matching the canonical deployment benchmark's warmup count),
    applied identically to both dtypes.
  - **Arrival/workload pattern:** closed-loop, fixed concurrency — exactly
    `max_num_seqs`-bounded concurrent requests kept in flight, with the next request issued
    immediately as a slot frees (matching `vllm bench throughput`'s default offline-batch
    behavior). This is explicitly **not** an open-loop/Poisson arrival process: the research
    question here is maximum sustained concurrency under saturation, not tail latency under
    bursty arrivals. An open-loop/SLA-style latency study is a distinct, separately-scoped
    experiment, not part of this plan.
  - **`max_num_seqs` matching rule:** at every concurrency point, BF16 and RABIT-KV use the
    identical `max_num_seqs` value and all other applicable engine settings. When target
    concurrency exceeds the canonical `max_num_seqs=32`, `max_num_seqs` is raised for both
    dtypes together to the same new value at that point — never raised for one dtype and
    left unchanged for the other. This extends Ground Rule 4/8 across the full sweep, not
    just within a single point.
- **Variables that must remain fixed:** model, `gpu_memory_utilization`, `block_size`, and
  other engine args from Experiment 3 not explicitly swept above.
- **Existing script to reuse:** the standard `vllm bench throughput` CLI (the repository's
  own `benchmarks/benchmark_throughput.py` is a deprecated stub pointing to this CLI) as
  the primary path, since it is upstream-maintained and accepts `--kv-cache-dtype`
  directly; `vllm-kvquant/benchmarks/benchmark_long_document_qa_throughput.py` (generic,
  already accepts `kv_cache_dtype`, `--num-documents`, `--repeat-count`,
  `--enable-prefix-caching`) as a documented fallback if `vllm bench throughput` proves
  incompatible with `rabit_kv2` wiring.
- **Exact code changes required:** none expected if `vllm bench throughput` accepts
  `kv_cache_dtype=rabit_kv2` cleanly — **this must be smoke-tested first, correctness
  before performance**, since this is the first exercise of `rabit_kv2` under concurrent,
  multi-request decode/aging in this repository (all existing regression tests and the
  canonical deployment benchmark are effectively single-stream). If the CLI path is
  incompatible, adapt via CLI arguments to `benchmark_long_document_qa_throughput.py`
  only — no source changes to `rabit_kv2.py` are anticipated or permitted under this
  experiment.
- **Model:** `LLM-Research/Meta-Llama-3.1-8B-Instruct`.
- **Dataset/workload:** synthetic prompt set matching
  `benchmark_long_document_qa_throughput.py`'s document sampling, or the standard workload
  used by `vllm bench throughput`'s defaults — whichever path is used, keep it identical
  across BF16 and RABIT-KV runs.
- **GPU:** NVIDIA H100 80GB HBM3.
- **Metrics:** aggregate throughput (tokens/sec), per-request latency at each concurrency
  level, maximum sustained concurrency before failure/eviction, GPU memory utilization.
- **Repetitions/samples:** ≥3 full-sweep trials per (dtype, prompt-length) combination for
  stability, each trial issuing the fixed 256-request workload defined above per
  concurrency point.
- **Raw output path:**
  `results/mlsys2027/concurrency_scaling/{bf16,rabit_kv2}_conc{N}.log` + `summary.json`.
- **Paper figure/table:** Figure — "Throughput vs. concurrency"; Table — "Maximum sustained
  concurrency, BF16 vs. RABIT-KV".
- **Completion criterion:** a correctness smoke test (small concurrency, short generation,
  output sanity-checked against single-stream generations) passes before the perf sweep is
  trusted; the sweep completes for both dtypes across the concurrency grid without crashing
  the harness; each dtype's maximum sustained concurrency point is identified and reported.
- **Estimated engineering difficulty:** High. This is the first concurrent/multi-request
  exercise of `rabit_kv2` in this repository — no existing evidence (109 regression tests
  are single-request/state-exactness focused) covers this regime, so correctness issues are
  a real possibility and must be resolved before any throughput number is reported.
- **Estimated GPU cost:** Highest of the P0-C experiments — 2 context points × 6+
  concurrency levels × 2 dtypes × ≥3 reps, each potentially running many seconds of
  sustained generation per point.

---

## P0-D — Scientific ablations

All five ablations below share the same shape: hold four of {K, V, G, R, META} fixed at
the canonical `rabit2` values and vary the fifth, using the *quality-only, logical*
fake-quant harness (not the physical engine, which hardcodes K3/V2/G32 in its page layout
and Triton kernels and cannot be ablated without new kernel work — see the audit's
"Technical blockers" section). Every ablation must report both the quality metric *and*
the logical KV MB / compression ×, since several of these knobs change compression as well
as quality.

### Experiment 7 — K-bit ablation

- **Research question:** Holding V=2, G=32, R=4, META=8g64 fixed, how does quality change
  as key bit-width varies (K2, K3 [control], K4)?
- **Why MLSys reviewers would care:** Establishes causal attribution for the asymmetric
  K3/V2 design — is it justified, or would a different split do as well? Core to defending
  the final operating point as a considered choice.
- **Hypothesis:** key precision matters more than value precision for retrieval-sensitive
  tasks (consistent with `docs/METHOD.md`'s existing claim that aggressive key compression
  caused larger degradation during development); K2 should show disproportionately large
  HotpotQA/NIAH degradation relative to a similarly-sized V-bit change (Experiment 8).
- **Control:** `rabit2` (K3/V2/G32/R4/META8g64, the canonical committed point).
- **Treatment:** K2/V2/G32/R4/META8g64 and K4/V2/G32/R4/META8g64.
- **Variables that must remain fixed:** V bits = 2, G = 32, R = 4, metadata
  mode/group = uint8/64, model, dataset, sample counts identical to canonical.
- **Existing scripts to reuse:** `benchmarks/quality/{continuation_ppl,niah,
  passage_retrieval,hotpotqa,qasper}.py`.
- **Exact code changes required:** add new `config_for_method` entries (e.g. `rabit2_k2`,
  `rabit2_k4`) in each of the five scripts that copy the existing `rabit2` config and vary
  only `k_bits`; extend the `allowed` method sets and default `methods` strings accordingly.
  No change to the `rabit2` control config itself.
- **Model:** `LLM-Research/Meta-Llama-3.1-8B-Instruct`.
- **Dataset/workload:** same per-benchmark datasets as Experiment 1.
- **GPU:** NVIDIA H100 80GB HBM3.
- **Metrics:** PPL delta %, F1 delta, NIAH/passage-retrieval accuracy, logical KV MB
  (K-bit changes logical size — must be recomputed and reported, not assumed constant).
- **Repetitions/samples:** identical sample counts to canonical, per benchmark.
- **Raw output path:**
  `results/mlsys2027/ablations/k_bit/{continuation_ppl,niah,passage_retrieval,hotpotqa,qasper}.log`.
- **Paper figure/table:** Figure — "K-bit ablation" (quality metric vs. K-bit at fixed
  V=2), one panel per benchmark or a combined table; HotpotQA row highlighted given its
  canonical sensitivity.
- **Completion criterion:** K2/K3(control)/K4 all measured across the 5 benchmarks with
  V/G/R/META held fixed; K3 row reproduces the canonical `rabit2` numbers exactly.
- **Estimated engineering difficulty:** Low–medium (isolated config addition, mirrors the
  existing pattern in each script).
- **Estimated GPU cost:** Low–medium, roughly 2× the per-benchmark cost of one Experiment 1
  method (two new K variants across 5 scripts).

### Experiment 8 — V-bit ablation

- **Research question:** Holding K=3, G=32, R=4, META=8g64 fixed, how does quality change
  as value bit-width varies across a pre-registered V1/V2/V3 sweep?
- **Why MLSys reviewers would care:** Completes the K/V asymmetry justification started in
  Experiment 7 — together they directly test whether K3/V2 is the right split.
- **Pre-registration (per audit correction — fixed before any run or result is seen):** the
  full sweep is **V1, V2 (control), V3**, fixed and documented here before execution. V1 is
  included by default because it is technically supported by the harness's
  `q_group_sym`/`q_group_affine`/`q_seq_affine` quantizers (`levels = 2**bits`, so `bits=1`
  is mechanically valid — 2 quantization levels). The **only** permitted reason to drop or
  substitute V1 is a technical/correctness failure identified *before* any quality metric
  is examined — e.g., the run crashes, produces NaN/Inf logits, or fails a basic sanity
  check such as all-codes-identical degeneracy that makes the run itself invalid. This is a
  pre-registered escape hatch, not a discretionary one: quality being poor is not, by
  itself, grounds to drop or substitute V1. If V1 runs successfully and completes all five
  benchmarks, its quality results — however poor — are reported in full. Only if V1 fails
  the correctness gate is the sweep extended to include V4 as a substitute, and that
  substitution is itself reported explicitly (what failed, and why V4 was chosen in its
  place).
- **Hypothesis:** value precision affects quality less than key precision at fixed total
  budget (motivating the asymmetric K3/V2 choice) but more than metadata or residual-window
  choices (Experiments 10, 11). V1's actual viability is an open question this
  pre-registered sweep answers directly — it is not assumed numerically degenerate in
  advance.
- **Control:** `rabit2` (K3/V2/G32/R4/META8g64).
- **Treatment:** K3/V1/G32/R4/META8g64 and K3/V3/G32/R4/META8g64 — the full pre-registered
  V1/V2/V3 sweep (see Pre-registration above).
- **Variables that must remain fixed:** K bits = 3, G = 32, R = 4, metadata mode/group,
  model, dataset, sample counts.
- **Existing scripts to reuse:** same 5 quality scripts as Experiment 7.
- **Exact code changes required:** same pattern as Experiment 7 — new `config_for_method`
  entries varying only `v_bits`; a lightweight correctness check (NaN/Inf/degeneracy) run
  immediately after quantization and before scoring, applied uniformly to all three
  variants (not just V1), so the correctness gate is not applied selectively.
- **Model / dataset / GPU:** identical to Experiment 7.
- **Metrics:** same as Experiment 7. If V1 fails the pre-registered correctness gate and is
  substituted with V4, both the failure mode and the substitution are reported as findings.
- **Repetitions/samples:** identical sample counts to canonical, per benchmark.
- **Raw output path:** `results/mlsys2027/ablations/v_bit/`.
- **Paper figure/table:** Figure — "V-bit ablation" (paired with Experiment 7's figure to
  form a K-vs-V sensitivity comparison); V1's result — collapse or otherwise — is plotted,
  not omitted.
- **Completion criterion:** V1/V2(control)/V3 variants measured across the 5 benchmarks
  with K/G/R/META fixed, per the pre-registered sweep; V2 row reproduces canonical
  `rabit2` numbers exactly; any V1→V4 substitution is documented with its triggering
  failure mode.
- **Estimated engineering difficulty:** Low–medium.
- **Estimated GPU cost:** Low–medium, comparable to Experiment 7.

### Experiment 9 — Group-size ablation

- **Research question:** Holding K=3, V=2, R=4, META=8g64 fixed, how does quantization
  group size (G) affect quality and logical compression?
- **Why MLSys reviewers would care:** G is a knob that trades finer-grained scale/min
  accuracy against metadata overhead; reviewers will want to know whether G=32 is near-
  optimal or an arbitrary choice.
- **Hypothesis:** smaller G improves quality (finer per-group scale/min) at the cost of
  more metadata overhead and thus lower compression; there is a knee where metadata
  overhead starts to dominate total logical size.
- **Control:** `rabit2` (G32).
- **Treatment:** G16 and G64 at fixed bit-width (a bits-fixed sweep, distinct from the
  `rabit3`/`rabit4`/`rabit8` presets in Experiment 1, which change G *and* bits together).
- **Variables that must remain fixed:** K=3, V=2, R=4, metadata group=64, model, dataset,
  sample counts.
- **Existing scripts to reuse:** same 5 quality scripts.
- **Exact code changes required:** new `config_for_method` entries varying only
  `k_group`/`v_group`.
- **Model / dataset / GPU:** identical to Experiment 7.
- **Metrics:** quality deltas **and** logical KV MB / compression × (G directly changes
  metadata volume — must report both axes, not quality alone).
- **Repetitions/samples:** identical sample counts to canonical, per benchmark.
- **Raw output path:** `results/mlsys2027/ablations/group_size/`.
- **Paper figure/table:** Figure — "Group size vs. quality vs. compression" (dual-axis or
  paired subplots).
- **Completion criterion:** G16/G32(control)/G64 measured across 5 benchmarks; G32 row
  reproduces canonical numbers.
- **Estimated engineering difficulty:** Low–medium.
- **Estimated GPU cost:** Low–medium, comparable to Experiment 7.

### Experiment 10 — Residual-window ablation

- **Research question:** Holding K=3, V=2, G=32, META=8g64 fixed, how many recent tokens
  need to remain in BF16 (R) to preserve quality?
- **Why MLSys reviewers would care:** R is the cheapest lever to recover quality (it only
  affects a small, fixed number of recent tokens regardless of context length) — reviewers
  will want to know if R=4 is well-justified or conservative/wasteful.
- **Hypothesis:** quality is expected to be sensitive to R at low values (R=0, R=1), with
  diminishing returns as R increases. **Open question, not an assumption (per audit
  correction):** whether this sensitivity is concentrated in HotpotQA/Qasper specifically,
  and whether it relates to the canonical HotpotQA regression, is exactly what this
  ablation tests — no causal link between R and the HotpotQA regression is asserted before
  this data exists. R is the cheapest ablation in this set to run regardless of outcome.
- **Control:** `rabit2` (R4).
- **Treatment:** R0, R2, R8.
- **Variables that must remain fixed:** K=3, V=2, G=32, META=8g64, model, dataset, sample
  counts.
- **Existing scripts to reuse:** same 5 quality scripts.
- **Exact code changes required:** new `config_for_method` entries varying only
  `residual`.
- **Model / dataset / GPU:** identical to Experiment 7.
- **Metrics:** quality deltas plus logical KV MB (R has a small effect on aggregate MB
  since only a few tokens are affected — report the actual magnitude rather than assuming
  it is negligible).
- **Repetitions/samples:** identical sample counts to canonical, per benchmark.
- **Raw output path:** `results/mlsys2027/ablations/residual_window/`.
- **Paper figure/table:** Figure — "Residual window vs. quality", with the HotpotQA line
  reported alongside the others — not singled out as causally linked to R until the data
  in this experiment (and, where relevant, Experiment 15) supports that interpretation.
- **Completion criterion:** R0/R2/R4(control)/R8 measured across 5 benchmarks; R4 row
  reproduces canonical numbers.
- **Estimated engineering difficulty:** Low.
- **Estimated GPU cost:** Low, comparable to Experiment 7 (4 variants, but each is cheap).

### Experiment 11 — Metadata ablation

- **Research question:** Holding K=3, V=2, G=32, R=4 fixed, does grouped UINT8 metadata
  (META8g64) cost quality relative to full-precision (BF16) metadata, and how does
  metadata group size affect the tradeoff?
- **Why MLSys reviewers would care:** `docs/METHOD.md` already asserts metadata is
  "included in reported logical storage rather than treated as free overhead," but no
  ablation currently quantifies the quality cost of compressing it — this closes that gap
  cheaply, since the harness already supports the axis.
- **Hypothesis:** UINT8 grouped metadata vs. BF16 metadata shows negligible quality
  difference (confirming the design choice), with sensitivity increasing only at very
  coarse metadata group sizes.
- **Control:** `rabit2` (uint8 metadata, group=64).
- **Treatment:** BF16 metadata (`metadata_mode="bf16"`, already a supported code path in
  each script's `encode_metadata`/`decode_metadata` helpers), and uint8 metadata at
  group=32 and group=128.
- **Variables that must remain fixed:** K=3, V=2, G=32, R=4, model, dataset, sample counts.
- **Existing scripts to reuse:** same 5 quality scripts — `encode_metadata` already
  branches on `metadata_mode` (`bf16` vs. `int8`/`uint8`) and `metadata_group_size`, so this
  ablation needs **no new quantization logic**, only new `config_for_method` entries.
- **Exact code changes required:** new `config_for_method` entries selecting
  `metadata_mode="bf16"` or varying `metadata_group_size` (32/128) while leaving K/V/G/R
  untouched.
- **Model / dataset / GPU:** identical to Experiment 7.
- **Metrics:** quality deltas plus logical KV MB (BF16 metadata substantially increases
  logical size — must be reported, this is expected to be the largest compression cost of
  any variant in this ablation set).
- **Repetitions/samples:** identical sample counts to canonical, per benchmark.
- **Raw output path:** `results/mlsys2027/ablations/metadata/`.
- **Paper figure/table:** Table — "Metadata scheme vs. quality vs. compression".
- **Completion criterion:** BF16-metadata and uint8 g32/g64(control)/g128 variants measured
  across 5 benchmarks; g64 row reproduces canonical numbers.
- **Estimated engineering difficulty:** Low (the parameterization already exists in the
  harness; this is the cheapest ablation to implement).
- **Estimated GPU cost:** Low, comparable to Experiment 7.

---

## P0.5 — External baseline (required, reduced scope)

Per audit correction: an external algorithmic baseline (not just BF16/FP8) is required for
P0, not deferred to P1 — but only in reduced, physical-only form. Full quality integration
remains optional (Experiment 16, P1) given its high engineering cost.

### Experiment 13 — External baseline: matched physical comparison (BF16 / FP8 / TurboQuant / RABIT-KV)

- **Research question:** How do BF16, native FP8, one representative TurboQuant operating
  point, and RABIT-KV compare in physical allocator capacity and real decode latency, under
  the most-matched-possible engine configuration?
- **Why MLSys reviewers would care:** a second real, already-integrated quantization method
  (TurboQuant — Hadamard rotation + Lloyd-Max quantization, credited to DRIVE/EDEN/HIGGS in
  `vllm/model_executor/layers/quantization/turboquant/config.py`, with 45 existing tests in
  `tests/quantization/test_turboquant.py`) run in the *same engine* controls for
  infrastructure confounds far better than a cross-paper comparison, and its presence in
  this fork — currently unused and unmentioned in any RABIT-KV document — makes omitting it
  a visible gap once a reviewer inspects the codebase.
- **Hypothesis:** none. No method is assumed to win on any axis; this experiment reports
  whatever is measured for all four methods side by side.
- **Baseline set:** `bf16`, `fp8` (reused from Experiment 4), `rabit_kv2` (reused from
  Experiment 3).
- **Additional baseline:** one representative TurboQuant preset — recommend
  `turboquant_k3v4_nc` as the closest bit-budget analog to RABIT-KV's K3-weighted key
  precision, run through the real vLLM engine (TurboQuant is already a physical backend,
  not a fake-quant harness).
- **Variables that must remain fixed:** as much of the Experiment 3 matched config as is
  compatible with TurboQuant's backend requirements. **Risk to flag explicitly:**
  TurboQuant requires `TurboQuantAttentionBackend`
  (`vllm/v1/attention/backends/turboquant_attn.py`) rather than RABIT-KV/BF16/FP8's Triton
  backend — engine config compatibility must be verified, and any unavoidable difference
  (e.g., attention backend) must be documented per method rather than silently assumed
  identical.
- **Existing script to reuse:** `benchmarks/performance/benchmark_deployment.py` pattern,
  extended to select an attention backend in addition to `kv_cache_dtype`.
- **Exact code changes required:** extend the deployment benchmark's engine-launch
  parameterization (already extended for `kv_cache_dtype` in Experiments 3/4) to also
  select the attention backend, since TurboQuant needs `TurboQuantAttentionBackend`. Before
  trusting any number, re-run `tests/quantization/test_turboquant.py`'s 45 tests against
  this exact model/shape as a correctness gate — this repository has not validated
  TurboQuant against Llama-3.1-8B-Instruct's specific head_dim/GQA configuration.
- **Explicit scope limit (per audit correction):** this experiment produces **capacity and
  latency only**, matching the FP8 restriction in Experiment 4. No TurboQuant quality
  numbers are produced here — that is Experiment 16 (P1, optional).
- **Model:** `LLM-Research/Meta-Llama-3.1-8B-Instruct`.
- **Dataset/workload:** same synthetic single-request decode microbenchmark as
  Experiment 3.
- **GPU:** NVIDIA H100 80GB HBM3.
- **Metrics:** capacity tokens/ratio and TPOT/TTFT/wall median for all four methods,
  reported side by side with signed deltas vs. BF16 for each non-BF16 method; no method
  declared a "winner" in this document — that judgment is left to the measured numbers.
- **Repetitions/samples:** capacity once per method; ≥20 decode reps for the new TurboQuant
  leg (BF16/FP8/RABIT-KV reused from Experiments 3/4, not re-run).
- **Raw output path:**
  `results/mlsys2027/external_baseline/turboquant_<preset>_deployment.log` +
  `results/mlsys2027/external_baseline/summary.json` (cross-references Experiments 3/4
  rather than duplicating their numbers).
- **Paper figure/table:** Table — "Matched physical comparison: BF16 / FP8 / TurboQuant /
  RABIT-KV" — the paper's primary external-baseline table.
- **Completion criterion:** correctness gate (`test_turboquant.py`) passes on this
  model/shape; all four methods measured for capacity and latency under the
  most-matched-possible engine config, with any unavoidable per-method configuration
  differences documented explicitly.
- **Estimated engineering difficulty:** High (backend-selection code change plus
  correctness re-validation on an unfamiliar codepath).
- **Estimated GPU cost:** Medium (one additional method beyond Experiments 3/4's existing
  BF16/FP8/RABIT-KV legs).

---

## P1 — Stretch goals (pursue after P0 and P0.5 are complete and time permits)

### Experiment 12 — Larger sample sizes with paired per-example confidence intervals

- **Research question:** Are the canonical quality deltas (in particular HotpotQA's
  60.6 → 55.2 F1) statistically robust, or within noise given the current small sample
  counts (8/10/15/20/24)?
- **Why MLSys reviewers would care:** reviewers routinely challenge point estimates
  without error bars, especially an 8-sample PPL claim and a 20-sample F1 claim.
- **Methodology (per audit correction — split by benchmark type, not uniform reseeding):**
  1. **Deterministic QA/retrieval benchmarks** (HotpotQA, Qasper, NIAH, passage retrieval):
     these are deterministic given a fixed prompt set and greedy decoding — re-running with
     a different RNG seed on the same prompts produces no new information, since the model
     call itself is not stochastic. The correct way to reduce uncertainty is (i) **increase
     the number of independent examples** drawn from the underlying LongBench-E/NIAH pools
     (larger N, not repeated seeds on the same N), and (ii) compute **paired per-example
     bootstrap 95% confidence intervals** on the BF16-vs-RABIT-KV delta — paired because
     both methods are scored on the identical example set, which substantially reduces
     variance relative to unpaired CIs.
  2. **Continuation PPL** (WikiText-2, multilingual): increase the number of independent
     text windows (samples) and/or scored continuation tokens per window, and report
     bootstrap or block-bootstrap uncertainty on the PPL delta (bootstrapping over
     independent windows, or a block-bootstrap over per-token log-likelihoods) — not
     multiple seeds of the same fixed windows, since the windows are the actual unit of
     independence here.
- **Hypothesis:** with paired per-example CIs and larger N, the core directional findings
  (small PPL degradation; larger HotpotQA regression) hold, and the HotpotQA delta's CI
  excludes zero.
- **Control / treatment:** `bf16` vs. `rabit2` (and optionally the frontier methods from
  Experiment 1) at larger N.
- **Variables that must remain fixed:** methodology identical to canonical (same decoding,
  same prompt pool) except N.
- **Existing scripts to reuse:** the same 5 quality scripts, invoked with larger
  `--samples` (or a larger slice of the underlying LongBench-E bucket where available).
- **Exact code changes required:** (1) increase `--samples`/`--eval-tokens` per script's
  existing CLI where the underlying dataset pool supports it; (2) add per-example logging
  to each script if not already present (needed for pairing; HotpotQA's logging is shared
  with Experiment 15 and should be built once); (3) add a new, small paired-bootstrap
  analysis script (e.g. `benchmarks/quality/paired_bootstrap_ci.py`) that reads the
  per-example BF16/RABIT-KV outputs and computes the paired bootstrap CI on the delta. This
  does not modify any of the five canonical scripts' quantization or generation logic.
- **Model / GPU:** identical to canonical.
- **Metrics:** paired bootstrap 95% CI on each headline delta (PPL %, F1 points,
  accuracy %), reported per benchmark.
- **Repetitions/samples:** as large as the underlying dataset pool and GPU budget allow —
  for LongBench-E buckets, use the full available 8k+ bucket rather than the canonical
  slice where it is larger; for WikiText-2/multilingual, increase samples and/or
  eval-tokens by a documented factor (e.g., 4×) rather than re-seeding the same window
  count.
- **Raw output path:** `results/mlsys2027/variance/`.
- **Paper figure/table:** Table with paired bootstrap CIs for all headline numbers; error
  bars added to the Experiment 1 frontier figure.
- **Completion criterion:** paired bootstrap CIs computed for every headline delta
  (including HotpotQA) using per-example pairing for the QA/retrieval benchmarks and
  window/token-count-based uncertainty for PPL; overlap or non-overlap with zero explicitly
  reported.
- **Estimated engineering difficulty:** Low–medium (mostly additional compute plus a small,
  new per-example logging + bootstrap analysis script).
- **Estimated GPU cost:** the highest of the P1 items — proportional to the sample-count
  increase chosen (budget-dependent; recommend prioritizing HotpotQA and continuation PPL
  first, since those carry the paper's two most load-bearing numbers).

> **Note:** the TurboQuant comparison originally planned here has been split — a required,
> reduced, physical-only comparison now runs in **P0.5, Experiment 13** (see below). Full
> real-engine TurboQuant *quality* integration remains optional and is now **Experiment 16**
> at the end of this section.

### Experiment 14 — One additional compatible model

- **Research question:** Does RABIT-KV's quality/compression/latency profile generalize to
  a second model with a different head_dim / GQA configuration?
- **Why MLSys reviewers would care:** single-model evaluation is a standard, easy
  rejection point for ML-systems papers; even one additional model substantially
  strengthens the generality claim.
- **Hypothesis:** core compression ratio and latency behavior transfer as long as the
  second model's head_dim/num_kv_heads satisfy the same alignment invariants already
  assumed in the physical layout code (multiples of 32/64, per comments in
  `kv_cache_interface.py`/`kvquant_k3.py`); the quality degradation pattern (HotpotQA most
  sensitive) likely recurs, but its magnitude may differ.
- **Control:** `bf16` on Model B.
- **Treatment:** `rabit_kv2` on Model B.
- **Variables that must remain fixed:** identical benchmark methodology and configs to the
  canonical suite; only the model checkpoint changes.
- **Existing scripts to reuse:** the same quality (Experiment 1) and deployment
  (Experiment 3) scripts, with `model_name` swapped.
- **Exact code changes required:** before any perf/quality number is trusted, validate
  `rabit2_page_layout`'s alignment assertions (`vllm/v1/kv_cache_interface.py`) against
  Model B's head_dim/num_kv_heads, and re-run the parametrized correctness suite
  (`tests/quantization/test_rabit_kv2_*.py`, `test_kvquant_k3.py`) extended to cover Model
  B's shape. If alignment holds, no source changes to `rabit_kv2.py` are anticipated; if it
  does not, this experiment blocks on physical-layout engineering work outside this
  experiment's scope and should be re-scoped or dropped for this cycle.
- **Model:** to be selected via a quick compatibility check (head_dim/num_kv_heads
  divisibility by 32/64) against candidates such as another Llama-3.1 size or a GQA model
  with a compatible configuration — final choice deferred, not decided in this plan.
- **GPU:** NVIDIA H100 80GB HBM3.
- **Dataset/workload/metrics:** the full canonical quality + deployment suite, applied to
  Model B.
- **Repetitions/samples:** matching Experiments 1 and 3 protocols.
- **Raw output path:** `results/mlsys2027/second_model/`.
- **Paper figure/table:** adds a "Model B" row/column to the headline quality and
  deployment tables.
- **Completion criterion:** alignment/correctness validation passes on Model B's shape;
  the full quality + deployment suite then completes for Model B.
- **Estimated engineering difficulty:** Medium–high (correctness validation gate before any
  performance claim; actual perf/quality re-run is otherwise low-difficulty reuse).
- **Estimated GPU cost:** roughly equal to a combined Experiment 1 + Experiment 3 run for
  one additional model.

### Experiment 15 — HotpotQA error analysis

- **Research question:** What specifically characterizes the canonical 60.6 → 55.2 F1
  regression on HotpotQA — which questions, hop types, or answer-span positions degrade —
  and does the pattern correlate with any of the Experiment 7–11 (K/V/G/R/META) ablations?
- **Why MLSys reviewers would care:** an explained, understood limitation reads as far more
  credible than an unexplained regression number, and directly strengthens the paper's
  honesty about its own weakest result.
- **Hypothesis — open, not assumed (per audit correction):** candidate explanations
  include (a) supporting facts falling outside the residual window, (b) reduced key
  precision degrading attention to non-recent supporting facts, (c) reduced value
  precision, (d) group-size quantization error, or (e) a factor unrelated to any single
  ablated knob (e.g., a property of the questions themselves). This experiment does not
  presuppose which explanation, if any, is correct.
- **Explicit sequencing (per audit correction):** this experiment runs **after**
  Experiments 7–11 (the K/V/G/R/META ablations), not before or in isolation, so that any
  correlation between per-example failure patterns and a specific ablated knob can be
  checked against real ablation data rather than asserted from a single bf16-vs-rabit2
  comparison. **A causal interpretation is written into the paper only if the per-example
  error analysis and the corresponding ablation(s) jointly support it.** If no ablation
  correlates with the failure pattern, the result is reported as an open, unexplained
  regression — an honest and acceptable outcome, not a gap to paper over.
- **Control / treatment:** the same `bf16` vs. `rabit2` HotpotQA generations as canonical,
  but analyzed per-example instead of only in aggregate.
- **Variables that must remain fixed:** the same 20-sample, 8k+ LongBench-E HotpotQA bucket
  as canonical (optionally expand sample count in coordination with Experiment 12, but the
  core 20-sample analysis must be reported first so it is directly comparable to the
  canonical number).
- **Existing script to reuse:** `benchmarks/quality/hotpotqa.py`.
- **Exact code changes required:** additive logging only — dump per-example question, gold
  answer, both predictions (BF16/RABIT-KV), per-example F1, and approximate
  supporting-fact token offsets relative to the residual window, to a structured JSON file.
  No change to the quantization logic itself.
- **Model:** `LLM-Research/Meta-Llama-3.1-8B-Instruct`.
- **Dataset:** the same LongBench-E HotpotQA 8k+ bucket (20 samples canonical; optionally
  more via Experiment 12).
- **GPU:** NVIDIA H100 80GB HBM3.
- **Metrics:** per-example F1, qualitative failure categorization, and an explicit
  per-ablation correlation check against each of Experiments 7–11's results.
- **Repetitions/samples:** one pass (generation is deterministic given the fixed config);
  cross-referenced against the Experiment 7–11 ablation runs on the same (or overlapping)
  examples where possible.
- **Raw output path:** `results/mlsys2027/hotpotqa_error_analysis/`.
- **Paper figure/table:** Table of representative failure cases plus a categorization
  breakdown; a correlation table against Experiments 7–11, with any null results reported
  as such rather than omitted.
- **Completion criterion:** per-example dump produced; failure categorization written;
  correlation (or explicit lack thereof) with each of the Experiment 7–11 ablations is
  checked and reported; any causal claim appearing in the paper is gated on this joint
  evidence, not asserted from this experiment alone.
- **Estimated engineering difficulty:** Low–medium (logging addition plus semi-manual
  analysis).
- **Estimated GPU cost:** Low (single generation pass; can be combined with Experiment 1's
  HotpotQA run to avoid a duplicate GPU cost).

### Experiment 16 — Full TurboQuant quality integration (optional, P1 stretch)

- **Research question:** How does RABIT-KV compare to TurboQuant on *quality* (PPL/F1/
  accuracy), not just the physical capacity/latency comparison already covered by the
  required Experiment 13 (P0.5)?
- **Why MLSys reviewers would care:** a full quality comparison against a second real
  quantization method would strengthen the paper's positioning beyond the physical-only
  P0.5 result, but is not required to defend the core claims — Experiment 13 already
  provides a real, matched systems comparison.
- **Hypothesis:** none assumed; deferred pending infrastructure.
- **Status:** optional, pursued only if a real-engine quality harness is built — this is
  substantial new infrastructure (the existing quality scripts are HF-side fake-quant only
  and have no real-engine code path), and may share work with Experiment 4's deferred FP8
  real-engine quality evaluation. Do **not** approximate this with a hand-written fake-quant
  stand-in for TurboQuant's Hadamard-rotation + Lloyd-Max scheme — the same rule that
  applies to FP8 quality (Experiment 4) applies here.
- **Control:** `bf16`.
- **Treatment:** the TurboQuant presets not yet covered by Experiment 13 (e.g.
  `turboquant_k8v4`, `turboquant_4bit_nc`, `turboquant_3bit_nc`), run through the real vLLM
  engine.
- **Variables that must remain fixed:** same dataset/sample counts as Experiment 1, same
  engine-compatibility caveats as Experiment 13.
- **Existing scripts to reuse:** none directly — see code changes.
- **Exact code changes required:** a new real-engine quality benchmark (prefill + generate
  through vLLM with the TurboQuant backend, scored the same way as the canonical
  benchmarks) — this is the largest single engineering item in this entire plan.
- **Model:** `LLM-Research/Meta-Llama-3.1-8B-Instruct`.
- **Dataset/workload/metrics:** same as Experiment 1, applied per remaining TurboQuant
  preset.
- **GPU:** NVIDIA H100 80GB HBM3.
- **Repetitions/samples:** matching Experiment 1's protocol.
- **Raw output path:** `results/mlsys2027/turboquant_quality/`.
- **Paper figure/table:** adds remaining TurboQuant points to the Experiment 1 frontier
  figure, if pursued.
- **Completion criterion:** at least one additional TurboQuant preset's quality measured
  through the real engine, correctness-gated per Experiment 13.
- **Estimated engineering difficulty:** High — new real-engine quality infrastructure,
  the single highest-cost item in this plan.
- **Estimated GPU cost:** Medium–high, comparable to Experiment 1 per additional preset,
  plus infrastructure-development overhead not counted in GPU time.

---

## P2 — Submission compliance (paper-readiness audit)

Not a GPU-experiment phase — a checklist-driven audit required before submission. Cheap to
maintain continuously from Week 1 and expensive to retrofit at the last minute, so
compliance habits (no identifying content in new docs/scripts, provenance manifests per
Ground Rule 7) start immediately; the full audit pass below is finalized in the last days
before the deadline.

- **Double-blind anonymization audit.** *Checks:* the manuscript, appendix, and any
  supplementary code/artifact bundle contain no author names, institution names, grant
  numbers, or identifying acknowledgments; self-citations are phrased in third person per
  MLSys/anonymity guidelines. *Why it matters:* anonymization violations are a common
  desk-reject reason, independent of scientific merit. *Completion criterion:* a full
  read-through of the compiled PDF and any submitted artifact against the venue's
  anonymity policy, with every finding fixed before submission.
- **No identifying GitHub URL in the anonymized manuscript.** *Checks:* no link to this
  repository (or any fork/mirror that would reveal authorship via commit history, issue
  tracker, or personal GitHub username) appears anywhere in the PDF — including footnotes,
  code listings, and supplementary material. *Why it matters:* a working repository URL is
  one of the most common accidental de-anonymization vectors. *Completion criterion:* if a
  code release is referenced, it uses an anonymized hosting mechanism per venue policy
  (e.g., an anonymized code-sharing service or a redacted archive) rather than this
  repository's real URL; a text search of the final PDF for the repository name/URL/author
  GitHub handles returns no matches.
- **OpenReview profile readiness.** *Checks:* all authors have complete, correctly
  affiliated OpenReview profiles, and conflict-of-interest/reviewer-exclusion fields are
  set correctly, before the submission deadline. *Why it matters:* profile mismatches are a
  common last-minute submission blocker unrelated to the paper's content. *Completion
  criterion:* every author's profile verified complete and correct at least 48 hours before
  the deadline.
- **10-page main-paper limit check.** *Checks:* the compiled PDF's main body (excluding
  references and any clearly-labeled appendix) meets MLSys 2027's page limit under the
  required template, font, and margins; figures/tables are not used to smuggle content past
  the limit in ways that violate formatting rules. *Why it matters:* a formatting violation
  is grounds for administrative rejection regardless of content. *Completion criterion:*
  final compiled PDF checked against the official template and page count immediately
  before submission.
- **Reproducibility/provenance audit.** *Checks:* every number in the paper traces to a
  specific file under `results/mlsys2027/` (or the existing canonical `results/` for
  numbers carried over from the audit), each such file carries the provenance manifest
  required by Ground Rule 7, and `docs/REPRODUCIBILITY.md` (or an MLSys-specific
  reproducibility appendix) accurately describes how to regenerate each figure/table. *Why
  it matters:* MLSys reviewers and the artifact-evaluation process specifically check this.
  *Completion criterion:* a full claim-by-claim pass confirming every reported number has a
  traceable source file.
- **Claim-to-result traceability audit.** *Checks:* build an explicit claim → evidence-file
  → figure/table mapping for every quantitative claim in the paper's abstract, introduction,
  and results sections; verify no claim (e.g., "RABIT-KV achieves N× speedup") is stated
  without a corresponding measured result from this plan. Specifically re-verify that the
  HotpotQA regression (60.6 → 55.2 F1), the matched BF16 comparison (Experiment 3), and any
  FP8/TurboQuant comparison (Experiments 4/13) are all still visible and accurately stated
  — not softened, hedged away, or dropped — in the final manuscript. *Why it matters:* this
  is the direct enforcement mechanism for Ground Rules 5 and 6. *Completion criterion:* a
  written claim → evidence mapping exists and is checked against the final manuscript text,
  with zero unsupported claims remaining.

**Output:** `results/mlsys2027/submission_compliance/checklist.md`, populated
incrementally as each item is audited (not created now — this is a plan only).

---

## Proposed execution order — 5 weeks to the MLSys 2027 deadline

| Week | Focus | Experiments | Rationale |
|---|---|---|---|
| **1** | Quality frontier + deployment script prep | 1, 2; begin code changes for 3 (dtype parameterization); begin light P2 compliance habits (no identifying content in new docs/scripts) | Cheapest, lowest-risk experiments first (no code changes for 1); unblocks everything in P0-B/P0-C by writing the shared `--kv-cache-dtype` parameterization early; compliance habits are free to start now and expensive to retrofit later |
| **2** | Matched deployment baselines | 3, 4 (physical-only: capacity + latency, no fake-quant FP8 quality) | Closes the paper's single biggest evidence gap (matched BF16) plus a real-engine FP8 baseline, before investing in scaling studies built on top of the same harness |
| **3** | Systems scaling + external-baseline groundwork | 5, 6; begin Experiment 13 (TurboQuant backend-selection code + correctness gate) | Reuses the Experiment 3 harness; concurrency (6) is flagged high-difficulty/high-risk (first concurrent exercise of `rabit_kv2`) — start it early in the week to leave slack for correctness debugging; Experiment 13 is flagged high-difficulty and benefits from an early start in parallel |
| **4** | Scientific ablations + finish external baseline | 7, 8, 9, 10, 11 (ablations, quality-only, can run largely in parallel); finish Experiment 13 (P0.5) | Ablations are cheap/parallelizable once the config-entry pattern from Experiment 7 is established; run 10 (residual) and 7 (K-bit) with no presumed link to the HotpotQA regression — that correlation is only established later, in Experiment 15; land Experiment 13 this week so the external-baseline table is ready with margin |
| **5** | Selective P1 + submission compliance + writing buffer | 12 (variance on headline numbers, paired bootstrap CIs), 15 (HotpotQA error analysis, run only after 7–11 are in hand) as must-do; 14 (second model) and 16 (full TurboQuant quality) only if ahead of schedule; full P2 compliance audit pass | 12 and 15 directly defend the paper's two most exposed claims (statistical robustness, HotpotQA characterization); 14 and 16 are the highest-engineering-cost stretch items and are cut first under time pressure; P2 must be completed regardless of how much of P1 is reached |

**Cut order if the schedule slips:** drop Experiment 16 (full TurboQuant quality
integration) first, then Experiment 14 (second model) — both are high engineering cost
relative to their marginal paper impact given a complete P0 + P0.5. Do **not** cut
Experiment 13 (P0.5 external baseline, reduced/physical-only) — it is required, not
optional. Do not cut Experiment 15 or the headline portion of Experiment 12 — both are
low-cost and directly protect the paper's most reviewer-exposed claims. Do not cut anything
in P0 (Experiments 1–11) — they are the minimum bar for an MLSys-quality submission per the
audit's own findings. Do not cut P2 under any circumstance — an anonymization or
page-limit failure is a desk-reject risk independent of experimental completeness.
