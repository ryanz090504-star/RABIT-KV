# RUNPOD-QWEN-H001R run record (2026-10-24)

**Design:** a pre-specified replication on the WikiText-2 validation split. It was privately frozen before scoring (freeze record `ca3b5f4b…`). It is not held-out, not confirmatory, and not publicly preregistered.

## Launch

| Step | Result |
|---|---|
| Pre-launch, local | Freeze gate passed. Offline gates 33/33. Registration `d2708660…` and image `sha256:e0eedf6e…` unchanged |
| Provider check, 03:35Z | Balance $17.42. Volume `u3t6e85stf` (150 GB, CA-MTL-1). Only the 2 known Pods, both EXITED. H100 SXM available (stock Low) at $3.99/h |
| Image | Pushed to the private `docker.io/ryanzxas/runpod-qwen-001:h001r` (anonymous access refused). Digest `sha256:e0eedf6e0f1d6215b13adb20eed0f1c0a0b59a63960d7e4a1f887f626b1448da`. Independently verified three ways: recomputed index hash, by-digest fetch and tag resolution, all equal to the local image ID |
| Quote gates, 03:36:37Z | 15/15 passed, including `private_freeze_record_verified`. Worst case $6.98 |
| Pod | Exactly one create request at 03:36:47Z, with no retry. `b3l0x0iilue1ic` (`runpod-qwen-h001r-gpu`): H100 80GB HBM3, CA-MTL-1, cap 6,000 s |
| In the Pod | Offline gates passed at 55 s. Model snapshots hash-verified at 183 s. GPU preflight passed at 232 s. Science finished at 2,566 s. Audit passed at 2,567 s |
| Transfer | 108,068,731 bytes. Archive SHA-256 `954d8262…` verified. 67/67 manifest files verified. Acknowledgement sent |
| Stop | **The Pod stopped itself** at 04:21:46Z, after the acknowledgement and before any controller stop (`self-stop.json`). It was EXITED at 2,726 s; total wall time 2,732 s. The Pod was not deleted, and the earlier Pods are intact |
| Cost | Balance $17.4198 → $14.5233, so **$2.90 was debited**. The wall-clock upper bound is $3.03 (2,732 s at $3.99/h); volume storage at $0.015/h is extra. RunPod running total ≈ $9.87 |

## Independent local audit (`results/b3l0x0iilue1ic/independent-audit/`)

**Every check passes.** The audit did not rely on any pool or hash the Pod reported:
- Both validation token pools were rebuilt locally; they equal the registration and the Pod's pools.
- All 198 rows and 25,344 per-token NLL values per model are on registered validation windows: context hashes and targets match the local pools. No row is on a test window.
- Every value is finite and non-negative, and the loss sums agree.
- Window-1 controls are bitwise equal, and the test-window anchors are bitwise equal to attempt 008.
- The run used the frozen registration and stage files.
- The frozen audit, re-run locally on a copy, is valid, and the analysis recomputed from the per-token NLLs equals it exactly.
- The retrieved files are unmodified.

**One reported discrepancy:**
- The Pod's own `audit.json` analysis differs from the local one in 162 compared floats, by at most 3.6e-15 absolute and 2.9e-16 relative. That is last-bit floating-point noise between Linux and Windows, both on numpy 2.4.6.
- All decisions are identical. The smallest decision margin is 0.124, about 10^13 times the difference.
- The byte-equality check `pod_audit_equals_local_audit` is therefore reported as failed-at-ULP, not passed.

## Pre-specified outcomes (decision intervals: 97.5%, paired-window bootstrap, seed 20261024)

**Qwen2.5-7B, H1 padding exclusion: replication criteria met.**
- range_real vs rabit: −91.43% PPL, interval [−92.88, −89.72]. range_real is better in 32 of 32 windows.
- Fraction of the frozen excess removed: 0.941, interval [0.924, 0.956], against a threshold of ≥ 0.80. The post-hoc estimate was 0.949.

**Qwen2.5-7B, H2 closed-page localization: replication criteria met.**
- k_exact vs range_real: −13.21%, interval [−17.01, −9.62].
- closed_exact vs range_real: −12.76%, interval [−16.67, −9.11].
- Closed fraction: 0.963, interval [0.901, 1.001], against ≥ 0.75.
- Open fraction: 0.031, interval [−0.021, 0.093], against ≤ 0.25.

**Llama-3.1-8B, S1 (secondary): met.**
- range_real vs rabit: −0.116%, with a 95% upper bound of +0.150%, against ≤ +0.5%.
- The Llama gap fractions are not interpreted, because their denominators cross zero.

## Scope

This is a logical evaluator with a fixed FlashAttention backend. It does not test serving or the canonical automatic backend. The exact-K arms are diagnostic controls. Prior exposure to the validation split cannot be excluded (see `validation-split-audit/AUDIT.md`).
