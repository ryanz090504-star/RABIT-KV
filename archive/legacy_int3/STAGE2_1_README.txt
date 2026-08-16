RABIT-KV Stage 2.1 correction

Stage 2 page bytes were correct, but R4 alone is not enough for an exact online
implementation of K3 sequence-axis G32. The current incomplete 32-token K group
must retain original BF16 K values to avoid cumulative re-quantization drift.

This patch adds exact per-layer/per-sequence online staging accounting:
- up to 31 original BF16 K values for the open sequence group
- newest R4 K values
- newest R4 V values

For Llama-3.1-8B this is 79,872 bytes per layer per active sequence. Page bytes
remain 24,832 and page-only compression remains 5.278x. Effective compression
including staging approaches 5.246x at 16K.
