# RUNPOD-QWEN-H001R: pre-specified replication on the WikiText-2 validation split (prepared, NOT launched)

**Status:** prepared on 2026-10-23. Only offline gates have run: no GPU, no provider call, no image push, no commit, and the registration has not been published.

**Freeze (private, not a public preregistration):** `freeze/H001R-freeze-record.json` is a private, timestamped record that pins the full SHA-256 of every file in this folder that the run depends on, plus the local image ID. Its timestamp comes from this machine's clock and is self-attested; no third party has witnessed it. The launch gate `private_freeze_record_verified` (`launch/freeze_gate.py`) accepts only that exact record and re-checks every hash in it. This run may be described as "pre-specified and privately frozen before scoring", but not as "publicly preregistered", unless the author later publishes the record and states when.

**No scoring before freeze:** `freeze/unscored-verification.json` was produced by `freeze/verify_unscored.py`. It found:
- **Dry runs:** both real-data dry runs (attempt 008 and RUNPOD-QWEN-002) used only canonical test windows.
- **Scored output:** none of the 64 validation-window contexts occurs in any scored output, log or archive in the worktree.
- **Launch:** no launch, quote, release or push record exists, and no container ran from the H001/H001R images other than the `--rm` offline gates.
- **Transcripts:** no 12-word validation-only passage appears in any available agent transcript (positive control detected).

**Registration:** `image/stage/registration.json`, SHA-256 `d27086605ba8ee2b6ec543b9805209520d2ac16b84615259d36f716280ed22be`. It is written before any validation-split window is scored; the worker reads it, and the audit applies it.

**Relation to H001:** H001R relabels the earlier H001 preparation, which was never published, pushed or launched. Its files are kept byte for byte in `superseded-H001/`. The inputs, windows, arms, controls, anchors, thresholds and analysis are identical; only labels and claims changed (`H001R-CHANGES.md`). The reason is the validation-split audit (`validation-split-audit/AUDIT.md`): no recorded project use was found, but prior use cannot be ruled out.

## Question

Do two post-hoc findings replicate on the WikiText-2 validation split, whose windows are disjoint from every window that produced them?
1. **Padding exclusion:** for Qwen2.5-7B, excluding the zero padding of the open key group from its quantization range removes most of the frozen policy's excess loss.
2. **Closed-page localization:** the key error that remains after that is located in the completed (closed) pages, not in the open group.

**This is a replication, not held-out or confirmatory validation.** The inputs are not claimed to be unseen: the audit found no recorded use, but an unrecoverable 2026-05-03 GPT-2 session downloaded all WikiText-2 splits, and older transcripts, ChatGPT conversations, deleted scripts and remote provider logs cannot be inspected. Results are reported as **"replication criteria met"** or **"replication criteria not met"**, never as "confirmed".

**Scope:**
- Logical evaluator only, with attention fixed to the FlashAttention backend, as in the diagnostics being tested.
- The run does not test the serving engine or canonical (automatic-backend) quality, and its exact-key arms are diagnostic controls, not policies.

## Inputs

- **Data:** the WikiText-2 **validation** split, `valid.txt` from pytorch/examples at commit `d5678bc8…` (1,121,681 bytes, SHA-256 `f0737ed3…`). The test split from the same commit is byte-identical to the canonical `wikitext.txt`.
- **Windows:** the same construction as the canonical protocol, giving 32 consecutive, non-overlapping windows (1,024 context + 128 scored tokens). All 32 are used; none are excluded.
- **Token pools (unchanged from H001):**
  - Qwen: `32cc9e1c…`
  - Llama: `7b1140d7…`
  - Both are computed with the pinned tokenizer files, which first reproduce both canonical test pools exactly, and re-verified on the GPU before scoring.
- **Disjointness and recorded-use checks (offline gates):**
  - No commit in any ref refers to the validation split, and no `load_dataset` call for WikiText validation appears in the history.
  - None of 8 sampled validation-split passages appears in any branch or in the diagnostics folder.
  - No 32-token sequence of either pool occurs anywhere in the full test-split token stream (296,959 tokens for Llama, 305,907 for Qwen), and no line of 40+ characters is shared with the test split.
  - These checks show disjointness from the scored windows and no recorded use. They cannot exclude unrecorded prior use; the registration's `input_status` says so (`prior_exposure_excludable: false`) and pins `AUDIT.md` by SHA-256 `cc04e78d…`.

## Arms, controls, anchors

- **Six scientific arms:** bf16, rabit (frozen policy), range_real (padding excluded), k_exact, open_exact, closed_exact, each on all 32 windows for both models. Their definitions are unchanged from attempt 010, with the frozen policy added.
- **Window-1 controls, each required to match bitwise:**
  - pass-through equals bf16;
  - each oracle replay (frozen, range, k, open, closed) equals its arm;
  - same-device oracle checkpoints at 9 decode steps in every layer, for every quantized arm.
- **Anchors (environment fingerprint, never analysed):** canonical test window 1 for bf16, range_real and k_exact must repeat attempt 008 bit for bit.
- **Worker:** the audited stage010 scorer with 10 lines changed and 17 added (`gpu_diag-vs-stage010.diff`). Every frozen source is byte-identical.

## Pre-specified replication criteria (Qwen2.5-7B; `tools/prereg_analysis.py`)

- **Intervals:** paired-window bootstrap, seed 20261024, 20,000 resamples. Decisions use two-sided 97.5% intervals (Bonferroni over H1 and H2); 95% intervals are also reported.

| Hypothesis | Replication criteria met only if all hold | Post-hoc estimate (scored test windows) |
|---|---|---|
| H1 padding exclusion | range_real vs rabit: upper bound < 0. Fraction of the frozen excess NLL removed: lower bound ≥ 0.80 | 0.949 [0.938, 0.960], 32 windows |
| H2 closed-page localization | k_exact vs range_real: upper bound < 0. closed_exact vs range_real: upper bound < 0. Closed fraction of the gap: lower bound ≥ 0.75. Open fraction: upper bound ≤ 0.25 | 1.016 [0.913, 1.073] and −0.017 [−0.097, 0.074], 8 windows |

- **Secondary, Llama-3.1-8B, not a primary replication criterion:** range_real vs rabit 95% upper bound ≤ +0.5% PPL (no material harm). All other contrasts are descriptive. Llama gap fractions are not interpreted if their denominator crosses zero.
- **Thresholds:** set below the post-hoc estimates. The test asks whether the qualitative finding replicates, not whether its magnitude does.
- **Failure rule:** if any control or anchor fails, no replication decision is made from the run. A rerun repeats this registration byte for byte, and every attempt is reported.

## Offline gates run

| Gate set | Result |
|---|---|
| `tools/offline_gates.py` → `offline-gates.json` (local CPU) | **33 of 33 pass** (the 29 H001 gates, renamed where they said "held-out", plus 4 new relabel gates) |
| Image `runpod-qwen-h001r:local` (`sha256:e0eedf6e…`), built from the same hash-locked wheels → `container-offline-gates.log` | **pass**: Python 3.11.12, torch 2.11.0+cu130, transformers 4.48.2. Stage hashes match. 2,000 aging states and 1,270 HF updates agree with the independent oracle, including the frozen arm |

The 4 new gates check that:
- the registration's science (data, windows, arms, controls, anchors, models, analysis settings and thresholds) equals the superseded H001 registration;
- the image differs from H001 only in label strings, in exactly 7 files;
- the analysis returns the same numbers and decisions as H001's on 5 cases (two seen-data dry runs and three synthetic), with only the key names differing;
- the registration makes no held-out, unseen-input or confirmatory claim outside its explicit negations.

## Cost estimate (unchanged from H001)

| Item | Basis | Seconds |
|---|---|---|
| Offline gates, model re-hash, GPU preflight | RUNPOD-QWEN-002: 49 s gates; Llama re-hash and re-read 89 s; models already on the volume | ≈ 300 |
| Science | 402 rows; 4.7–6.6 s per row, from 002 (90 rows in 425 s) and 008 (262 rows in 1,727 s) | 1,900–2,650 |
| Audit, archive, transfer | 002: 216 s for 85 MB | ≈ 250 |
| **Expected Pod time** | | **≈ 2,450–3,200 s** |

- **Expected cost at $3.99/h:** about **$2.70–$3.55**.
- **Hard worst case:** $6.98. That is the watchdog cap of 6,000 s plus the controller margin of 300 s.
- **Running total:** RunPod spend so far is ≈ $6.97, so it would reach about $9.7–$10.5, or $13.95 in the worst case. Volume storage is extra, at $0.015/h.
- **Science cap:** 3,600 s. The gate requires a balance of at least the worst case plus $5.

## Launch sequence (each step needs explicit authorization; none has been run)

```sh
# 0. Private freeze check (offline; already passes; must still pass at launch):
python -B diagnostics/qwen-logical/holdout-001/launch/freeze_gate.py
# 1. Push the locally gated image (outward; ~2.9 GB compressed):
docker tag runpod-qwen-h001r:local docker.io/ryanzxas/runpod-qwen-001:h001r
docker push docker.io/ryanzxas/runpod-qwen-001:h001r
python -B diagnostics/qwen-logical/holdout-001/launch/verify_image.py docker.io/ryanzxas/runpod-qwen-001:h001r
# 2. Fresh gates, then the single GPU launch (within 10 minutes of the quote):
cd diagnostics/qwen-logical/holdout-001/launch
python -B quote_gpu.py
python -B controller.py launch gpu -H001R
# 3. After transfer: re-run the audit locally against the fetched results
python -B ../tools/audit_holdout.py results/<pod-id>/extracted/RUNPOD-QWEN-H001R/run-gpu ../image
```

`quote_gpu.py` refuses to write `release.json` unless `private_freeze_record_verified` passes. That gate replaces the former `preregistration_publicly_recorded_by_author` gate, which required a public record that does not exist. `release.json` carries `freeze_record_sha256` and `public_preregistration: false`.

**What the launch does:**
- **Controller:** identical to the rp-001 controller except for its docstring, the 6,000 s cap and the result folder.
- **Pod:** creates exactly one Pod (`runpod-qwen-h001r-gpu`, H100 SXM, CA-MTL-1) on the existing volume `u3t6e85stf`, with no new volume.
- **Stop:** the Pod self-stops after a verified acknowledgement, the watchdog stops it at the cap, and the controller stops it in `finally`. It is never deleted.
