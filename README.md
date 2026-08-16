# RABIT-KV

**Target-bit-aware KV-cache quantization for LLM inference with a physically packed vLLM cache and custom Triton kernels.**

> **5.28× physical KV capacity · 21.61 ms/token median TPOT · +1.53% WikiText-2 continuation PPL**

RABIT-KV compresses the KV cache used during LLM inference while preserving long-context retrieval and most downstream quality. The implementation is integrated into a custom vLLM fork and uses a physically packed cache rather than fake quantization.

## Method

The final operating point is:

**K3 / V2 / G32 / R4 / META8g64**

- **K3** — 3-bit sequence-axis affine quantization for keys
- **V2** — 2-bit affine quantization for values
- **G32** — quantization group size 32
- **R4** — newest four K/V tokens retained in BF16
- **META8g64** — grouped UINT8 metadata with metadata group size 64
- **`rabit_kv2`** — physically packed vLLM KV-cache dtype

The “2-bit target” label refers to the operating point, not literal two-bit total storage. Keys use 3 bits, values use 2 bits, and the representation also includes metadata and a four-token BF16 residual window.

The authoritative implementation is:

```text
vllm-kvquant/vllm/v1/attention/ops/rabit_kv2.py
```

## Results

Final measurements use **Llama 3.1 8B Instruct** on an **NVIDIA H100 80GB HBM3** unless otherwise noted.

### Quality

| Benchmark | BF16 | RABIT-KV | Change / outcome |
|---|---:|---:|---:|
| WikiText-2 continuation PPL | 8.5020 | **8.6317** | **+1.53%** |
| Chinese continuation PPL | 10.1176 | **10.3308** | **+2.11%** |
| Spanish continuation PPL | 5.0035 | **5.1139** | **+2.21%** |
| NIAH, 4K/8K/16K × 5 depths | 15/15 | **15/15** | **100% retained** |
| Passage Retrieval | 100.0% | **100.0%** | **no loss** |
| Qasper 8K+ | 35.9 F1 | **35.6 F1** | **−0.3 pt** |
| HotpotQA 8K+ | 60.6 F1 | **55.2 F1** | **−5.4 pts** |

WikiText-2 continuation evaluation used 1,024 context tokens, 128 scored continuation tokens, and 8 samples. Its logical KV storage fell from **128.000 MB** to **24.710 MB**, corresponding to **5.18× logical compression**.

Across the long-context quality suite, measured logical KV compression is approximately **5.18–5.27×**.

### Physical capacity

Under the controlled H100 configuration:

| Cache | Allocator capacity |
|---|---:|
| BF16 | 393,024 tokens |
| RABIT-KV | **2,074,592 tokens** |

**Physical capacity gain: 5.2785×**

This is a real allocator-capacity measurement, not a logical byte estimate.

### Deployment latency

Final real-engine RABIT-KV decode measurement:

| Metric | Result |
|---|---:|
| Median TPOT | **21.61 ms/token** |
| Median TTFT | **121.14 ms** |
| Median wall time | **790.78 ms** |
| Context | 2,048 tokens |
| Generated tokens | 32 |

The final deployment path passed **105 regression tests** plus targeted state/byte and attention exactness checks.

`21.61 ms/token` is reported as the measured RABIT-KV deployment TPOT. It is **not** presented as a matched BF16 speedup claim because the final frozen deployment benchmark did not include a directly matched BF16 TPOT measurement.

## Repository structure

```text
RABIT-KV/
├── README.md
├── .gitignore
├── vllm-kvquant/                    # integrated RABIT-KV vLLM implementation
├── benchmarks/
│   ├── quality/
│   │   ├── continuation_ppl.py
│   │   ├── multilingual_ppl.py
│   │   ├── niah.py
│   │   ├── passage_retrieval.py
│   │   ├── hotpotqa.py
│   │   ├── qasper.py
│   │   └── run_suite.py
│   └── performance/
│       └── benchmark_deployment.py
├── results/
│   ├── summary.json
│   ├── quality/
│   │   ├── continuation_ppl.log
│   │   ├── multilingual_ppl.log
│   │   ├── niah.log
│   │   ├── passage_retrieval.log
│   │   ├── hotpotqa.log
│   │   └── qasper.log
│   └── performance/
│       ├── capacity.json
│       ├── latency.json
│       └── deployment.log
└── docs/
    ├── METHOD.md
    └── REPRODUCIBILITY.md
```

## Benchmarks

Quality benchmarks live under:

```text
benchmarks/quality/
```

They cover continuation perplexity, multilingual continuation, Needle-in-a-Haystack, passage retrieval, HotpotQA, and Qasper.

The integrated deployment benchmark is:

```text
benchmarks/performance/benchmark_deployment.py
```

It validates the final compressed-cache path and measures the deployment configuration used for the reported performance results.

## Result sources

The compact source of truth is:

```text
results/summary.json
```

Raw quality evidence:

```text
results/quality/
```

Deployment evidence and machine-readable performance summaries:

```text
results/performance/
```

Canonical committed results should be treated as immutable evidence. Reproduced runs should be written to separate output paths rather than overwriting the committed files.

## Reproduction

See:

```text
docs/REPRODUCIBILITY.md
```

for the recorded environment, controlled engine configuration, and benchmark entry points.

## Limitations

- Evaluation is centered on **Llama 3.1 8B Instruct** and **NVIDIA H100 80GB HBM3**.
- HotpotQA is the clearest quality regression in the final suite: **60.6 → 55.2 F1**.
- Broader model families, GPUs, concurrency regimes, and serving workloads remain untested.
- Quality-suite memory values are logical packed-prefix KV measurements; the **5.2785×** capacity result is a separate physical allocator measurement.
- “2-bit target” is an operating-point label, not literal two-bit total storage.
