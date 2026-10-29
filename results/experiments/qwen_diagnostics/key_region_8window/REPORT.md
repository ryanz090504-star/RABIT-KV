# RUNPOD-QWEN-002: frozen five-arm K-group diagnostic, completed

This is a post-hoc diagnostic run with the **fixed FLASH_ATTENTION backend**. It is not a canonical automatic-backend score, and it is not a deployable policy. The frozen worker, inputs, windows (1, 5, 9, 13, 17, 21, 25, 29), arms, scoring, bootstrap (seed 20261022, 20,000 paired-window resamples) and analysis are byte-identical to `stage010`.

## Execution and provenance

- **Pod:** `rgq6afrneuel82` (`runpod-qwen-002-gpu`), H100 SXM in CA-MTL-1 at $3.99/h. One create request, no retry.
- **Image:** `docker.io/ryanzxas/runpod-qwen-001@sha256:ec63d6daf8a0832b954270183e3d29427855ba743dda98648f8299cfbaccd0e0`.
- **Hardware and software (preflight record):** NVIDIA H100 80GB HBM3 with driver 595.91.07; torch 2.11.0+cu130 on CUDA 13.0, Python 3.11.12.
- **Preflight checks:** 1,024 GPU oracle-state checks passed. Explicit fp32 attention RMSE was 1.16e-4.
- **Volume:** `u3t6e85stf`, mounted over `nfs4`; the volume ID was confirmed inside the Pod.
- **Model resume:**
  - Llama: all 18 files re-hashed in full and kept.
  - Qwen: 7 verified files kept and the 2,575,302,656-byte partial shard **quarantined** to `/workspace/models-quarantine/RUNPOD-QWEN-002/`. The 8 missing files (15.24 GB) were downloaded.
  - Both snapshots then passed a full re-read from the volume, before preflight.
- **Timeline (Pod clock, seconds):**

| t (s) | Step |
|---|---|
| 49 | Offline gates passed |
| 2,140 | Both snapshots verified |
| 2,180 | GPU preflight passed |
| 2,608 | Science complete (427 s, within the 840 s cap); in-Pod audit passed |

## Audit of every per-token result

- **Coverage:** for each model, 45 rows: 5 arms × 8 windows, plus 5 window-1 controls. That is 5,760 per-token losses per model.
- **Per-row checks:** every row is finite and non-negative, and the target IDs and context hashes match the frozen token pool. That pool is byte-identical to attempt 008's.
- **Controls:** all 5 bitwise controls pass (pass-through, plus one oracle replay for each K arm). All same-device oracle checkpoints pass: 252 for Qwen, 288 for Llama.
- **Match to attempt 008:** bf16, range_real and k_exact are **bitwise identical per token to Modal attempt 008**, on a different provider and driver.
- **Re-audit:** the frozen `audit.py`, re-run locally, reproduces the in-Pod `audit.json` exactly.
- **Independent check:** a separately written check (`independent_check_002.py`) reproduces all 7 paired effects and both gap fractions to within 1e-9.

## Paired effects (PPL change, paired-window bootstrap 95% CI, windows better/worse)

| Contrast | Qwen 2.5 7B | Llama 3.1 8B |
|---|---|---|
| range_real vs bf16 | +11.91% [+6.01, +18.14], 0/8 | +1.08% [+0.29, +1.99], 2/6 |
| k_exact vs range_real | −9.56% [−14.12, −4.81], 8/0 | +0.20% [−0.11, +0.50], 3/5 |
| **closed_exact vs range_real** | **−9.70% [−14.62, −4.61], 7/1** | +0.38% [−0.30, +1.11], 2/6 |
| open_exact vs range_real | +0.17% [−0.58, +1.02], 4/4 | +0.26% [−0.25, +0.79], 4/4 |
| closed_exact vs bf16 | +1.05% [+0.55, +1.51], 1/7 | +1.46% [+0.70, +2.24], 1/7 |
| open_exact vs bf16 | +12.10% [+5.85, +18.92], 1/7 | +1.34% [+0.43, +2.23], 1/7 |
| k_exact vs bf16 | +1.21% [+0.21, +2.04], 1/7 | +1.28% [+0.49, +2.05], 2/6 |

PPL over the 8 windows:

| Arm | Qwen | Llama |
|---|---|---|
| bf16 | 8.003 | 7.399 |
| range_real | 8.956 | 7.479 |
| k_exact | 8.100 | 7.494 |
| open_exact | 8.971 | 7.498 |
| closed_exact | 8.087 | 7.507 |

Fraction of the range_real-versus-exact-K gap removed:

| Model | closed_exact | open_exact |
|---|---|---|
| Qwen | **1.016** [0.913, 1.073] | −0.017 [−0.097, 0.074] |
| Llama | 1.94 [−6.32, 9.87] | 1.31 [−6.87, 8.75] |

For Llama the denominator crosses zero, so those fractions cannot be interpreted, as pre-registered.

**What this supports:**
- **Qwen:** the K error remaining after padding exclusion is located in the **completed (closed) K groups**. Making only those exact removes essentially the entire range-versus-exact-K gap, while making only the unfinished group exact removes none of it. Because these are interventions whose effects propagate through later activations, the fractions are descriptive, not an additive decomposition.
- **Llama:** there is no detectable K-group effect; its quantization gap is small and not attributable to K at this resolution.
- **Scope:** all windows were seen in earlier attempts, so this is not a holdout. These fixed-backend results stay separate from the automatic-backend canonical scores.

## Cost, stop and cleanup

- **Self-stop:** the Pod stopped itself 6 s after the verified acknowledgement ("Exited by user" 19:44:35Z), before the controller sent any stop. Its in-Pod key was refused by REST (403) and accepted by GraphQL.
- **Cost:** balance $20.7893 → $17.6532 = **$3.14** for 2,824 s, matching $3.99/h.
- **Running total:** RUNPOD-QWEN-001 ≈ $3.83 (settled) plus RUNPOD-QWEN-002 $3.14 = about **$6.97**, plus volume storage at $0.015/h.
- **Pods:** `rgq6afrneuel82` and `zus4ccj0zwq38t` are both EXITED at $0/h. Neither was deleted (deletion was not authorized).
- **Volume contents:** both complete, verified, read-only model snapshots; the quarantined partial shard (2.58 GB); and run folders `RUNPOD-QWEN-001/` and `RUNPOD-QWEN-002/`.
- **Local evidence:**
  - `runpod-qwen-001-results/rgq6afrneuel82/`: archive SHA-256 `ee82ba78…`, 63 of 63 manifest files verified.
  - `launch-state-gpu-002/`
  - `independent-check-002.json`
  - `quote-gpu-002-20261023T185753Z.json`

## Preservation

- Attempts 008/009/010 manifests (392/436/489 files) all match.
- The frozen stage is unchanged.
- The RUNPOD-QWEN-001, cpu3c and cpu5c records are kept.
- Git HEAD is `8648b77` with no tracked changes. No commit, push, or manuscript or ref edits.
