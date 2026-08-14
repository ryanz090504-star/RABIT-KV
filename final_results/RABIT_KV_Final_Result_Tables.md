# RABIT-KV Final Consolidated Results

## Experimental scope

- Model: Llama 3.1 8B Instruct
- Production GPU: NVIDIA H100 80GB HBM3
- The research tables report the frozen RABIT-KV quality policy.
- The production table reports the current INT3 `kvquant_k3` deployment proof-of-concept.
- The current INT3 kernel is not yet an exact deployment of the final K3/V2 affine + R4 policy.

## Table 1. Final operating points

| Method | Policy | PPL | PPL Δ | Logical KV MB | Compression | Effective bits |
| --- | --- | --- | --- | --- | --- | --- |
| BF16 | Uncompressed baseline | 11.23 | 0.0% | 31.875 | 1.00× | 16.00 |
| RABIT-8 | SYM K32V64 R0 | 11.22 | 0.0% | 16.685 | 1.91× | — |
| RABIT-4 | SYM G64 R0 | 11.45 | +2.0% | 8.467 | 3.76× | — |
| RABIT-3 | SYM G32 R4 | 11.68 | +4.0% | 7.363 | 4.33× | — |
| RABIT-2 target | K3/V2; K sequence-affine; V token-affine; R4 | 11.51 | +2.6% | 7.441 | 4.28× | 3.74 |

**Main quality result:** the RABIT-KV 2-bit target operating point reaches
11.51 PPL, +2.6% relative to BF16,
with 4.28× logical KV compression.

## Table 2. Long-context quality

| Method | Continuation PPL | Qasper F1 | Qasper N | Hotpot F1 | Hotpot N | NIAH | Passage retrieval |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BF16 | 3.285 | 35.5 | 24 | 60.6 | 20 | Pass / Pass / Pass | 100% |
| RABIT-4 | 3.350 | 31.8 | 24 | 59.4 | 20 | Pass / Pass / Pass | 100% |
| RABIT-3 | — | 36.7 | 14 | 60.9 | 20 | — | — |
| RABIT-2 target | 3.397 | 34.7 | 24 | 55.3 | 20 | Pass / Pass / Pass | 100% |

Notes:

- Qasper uses the complete 24-example 8K+ bucket for BF16, RABIT-4, and RABIT-2.
- RABIT-3 Qasper was measured only on the final 14 examples and scored
  36.7 F1; it is not directly comparable to the 24-example rows.
- HotpotQA uses 20 examples from the 8K+ bucket.
- The continuation-PPL test uses 1,024 context tokens, 128 continuation tokens, and one sample.

## Table 3. Two-bit component ablation

| Configuration | PPL | PPL Δ | KV MB | Compression | Effective bits | Component |
| --- | --- | --- | --- | --- | --- | --- |
| BF16 baseline | 11.23 | 0.0% | 31.875 | 1.00× | 16.00 | Uncompressed reference |
| A Uniform2 G32 R0 | 175.74 | +1465.5% | 4.980 | 6.40× | 2.50 | Ordinary uniform symmetric 2-bit |
| B Uniform2 G32 R4 | 62.87 | +460.1% | 5.402 | 5.90× | 2.71 | Add four-token FP16 residual |
| C K2/V2 affine R4 | 12.42 | +10.6% | 6.441 | 4.95× | 3.23 | Use sequence/token affine axes |
| D K2/V3 affine R4 | 12.04 | +7.3% | 7.422 | 4.29× | 3.73 | Protect V instead of K |
| E K3/V2 affine R0 | 12.06 | +7.4% | 6.988 | 4.56× | 3.51 | Mixed K3/V2 without residual |
| F FINAL K3/V2 affine R4 | 11.51 | +2.6% | 7.441 | 4.28× | 3.74 | Final K3/V2 + affine + R4 |

### Component effects

| Comparison | PPL change | KV MB change |
| --- | --- | --- |
| Residual on uniform 2-bit | -112.87 | +0.422 |
| Affine-axis structure at K2/V2 | -50.45 | +1.039 |
| Protect K with 3 bits | -0.91 | +1.000 |
| Residual on final K3/V2 | -0.55 | +0.453 |
| K3/V2 vs reversed K2/V3 | -0.53 | +0.019 |

The final policy reduces PPL by 93.5% relative to ordinary
uniform 2-bit quantization, from 175.74 to 11.51.

## Table 4. Controlled production deployment

Both modes use the Triton attention backend, eager execution, no CUDA graphs,
and no `torch.compile`.

### Practical KV-cache capacity

| Format | Capacity | Relative capacity |
| --- | ---: | ---: |
| BF16 Triton | 315,200 tokens | 1.000× |
| INT3 `kvquant_k3` Triton | 1,551,776 tokens | 4.923× |

### TTFT and TPOT

| Context | BF16 TTFT ms | INT3 TTFT ms | BF16 TPOT ms | INT3 TPOT ms | INT3/BF16 TTFT | INT3/BF16 TPOT |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | 19.915 | 52.813 | 13.527 | 41.011 | 2.652× | 3.032× |
| 2048 | 66.794 | 182.177 | 13.467 | 40.800 | 2.727× | 3.030× |
| 4096 | 161.130 | 542.244 | 13.751 | 40.979 | 3.365× | 2.980× |

TTFT is approximated by a one-output-token request. TPOT is calculated as:

`(latency for 65 output tokens - latency for 1 output token) / 64`

**Deployment result:** the current INT3 kernel increases practical KV-cache
capacity by 4.923×, but TPOT is approximately 3× higher than
the controlled BF16 Triton path.

## Final defensible claims

1. The RABIT-KV 2-bit target operating point reaches 11.51 PPL, only 2.6%
   above BF16, with 4.28× logical KV compression.
2. The final policy reduces PPL by 93.5% relative to
   ordinary uniform 2-bit quantization.
3. RABIT-3 maintains BF16-level HotpotQA performance on the current 20-example
   8K+ evaluation: 60.9 versus
   60.6 F1.
4. RABIT-2 remains close to BF16 on the full Qasper 8K+ bucket:
   34.7 versus
   35.5 F1.
5. The current INT3 vLLM implementation increases practical KV-cache capacity
   by 4.923×.
6. The current INT3 Triton kernel has approximately 3× higher TPOT than the
   controlled BF16 Triton path.

## Required limitations

- “2-bit target” is a target operating point, not literal 2-bit physical
  storage. The final policy uses about 3.74 effective bits after metadata and
  residual storage.
- The current production kernel is INT3 and is not yet an exact implementation
  of K3/V2 affine + R4.
- Small F1 differences should not be described as statistically significant.
- Current LongBench conclusions must retain their sample counts.
