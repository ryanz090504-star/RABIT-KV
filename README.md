# RABIT-KV

RABIT-KV is a target-bit-aware KV-cache quantization project for LLM inference,
combining a low-bit K/V policy with a physically packed vLLM cache and custom
Triton attention work.

## Milestone 1 鈥?Frozen Core + Exact Physical Deployment

Frozen policy: **K3 / V2 / G32 / R4 / META8g64**

| Result | Milestone 1 |
|---|---:|
| BF16 PPL | 11.2258 |
| RABIT-KV PPL | **11.4774** |
| PPL change | **+2.24%** |
| Logical KV compression | **4.89x** |
| BF16 physical KV capacity | 393,024 tokens |
| RABIT physical KV capacity | **2,074,592 tokens** |
| Physical capacity gain | **5.28x** |
| Locked regression suite | **105 passed** |

Stable deployment checkpoint: **Stage4D2.2 writer-only locked**.

RABIT-KV does **not** yet claim an end-to-end latency win over BF16.
Stage4D3.4 is an exact experimental prototype: 13.27x faster tail preparation
and 6.04x faster full-attention microbenchmark, but it has not yet been
production-integrated.

## Layout

```text
RABIT-KV/
鈹溾攢鈹€ README.md
鈹溾攢鈹€ Quantization_v2/
鈹溾攢鈹€ vllm-kvquant/
鈹溾攢鈹€ scripts/quality/
鈹溾攢鈹€ benchmarks/deployment/
鈹?  鈹溾攢鈹€ stage4c/
鈹?  鈹溾攢鈹€ stage4d2_locked/
鈹?  鈹斺攢鈹€ experimental/stage4d3_4/
鈹溾攢鈹€ results/
鈹?  鈹溾攢鈹€ milestone1/
鈹?  鈹溾攢鈹€ quality/
鈹?  鈹溾攢鈹€ capacity_latency/
鈹?  鈹斺攢鈹€ experiments/
鈹溾攢鈹€ docs/
鈹斺攢鈹€ archive/
```

`vllm-kvquant/` is the authoritative integrated deployment tree. Root debug
copies from the original research workspace are preserved in
`archive/root_snapshots/`.

## Finish line

Milestone 1 is frozen **before** D3.4 production integration.

Next:
1. integrate D3.4 into the real Llama 3.1 8B engine;
2. measure real TPOT;
3. allow at most two additional focused performance cycles;
4. freeze performance;
5. run one final unified quality suite without tuning from it;
6. package v1.0 / README / paper.

See `results/milestone1/RESULTS.md` and `docs/`.
