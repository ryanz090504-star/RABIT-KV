# Reproducibility

## Recorded environment

The final deployment measurements were recorded with:

- **Model:** `LLM-Research/Meta-Llama-3.1-8B-Instruct`
- **GPU:** NVIDIA H100 80GB HBM3
- **CUDA:** 13.0 / 13.0.2
- **PyTorch:** 2.11.0+cu130 in the final deployment path
- **vLLM:** custom `0.10.0+kvquant` fork
- **vLLM base commit:** `f329ce405b12623fb8b1cf1830f12e5a712523be`

The authoritative RABIT-KV implementation is:

```text
vllm-kvquant/vllm/v1/attention/ops/rabit_kv2.py
```

## Controlled deployment configuration

The physical-capacity and deployment measurements use:

- eager execution
- Triton attention backend
- CUDA graphs disabled
- `torch.compile` disabled
- `gpu_memory_utilization = 0.82`
- block size 32
- max model length 32768
- max batched tokens 16384
- max sequences 32
- prefix caching disabled
- chunked prefill enabled

## Canonical committed results

Compact summary:

```text
results/summary.json
```

Raw quality evidence:

```text
results/quality/
```

Performance summaries and raw deployment log:

```text
results/performance/
├── capacity.json
├── latency.json
└── deployment.log
```

The committed result files are the canonical evidence and should not be overwritten during reproduction.

## Quality suite

Entry point:

```text
benchmarks/quality/run_suite.py
```

Individual benchmark scripts:

```text
benchmarks/quality/continuation_ppl.py
benchmarks/quality/multilingual_ppl.py
benchmarks/quality/niah.py
benchmarks/quality/passage_retrieval.py
benchmarks/quality/hotpotqa.py
benchmarks/quality/qasper.py
```

The quality scripts use Modal to request an H100 environment and run the recorded Llama 3.1 8B configuration.

From the repository root:

```powershell
python .\benchmarks\quality\run_suite.py
```

For a clean public reproduction workflow, reproduced quality outputs should be written under:

```text
results/reproduced/quality/
```

rather than replacing `results/quality/`.

## Deployment benchmark

Entry point:

```text
benchmarks/performance/benchmark_deployment.py
```

From the repository root:

```powershell
python .\benchmarks\performance\benchmark_deployment.py
```

The benchmark validates the integrated `rabit_kv2` source and records the deployment run separately from the committed canonical result.

Reproduced deployment output should be written to:

```text
results/performance/deployment_reproduced.log
```

## Reported physical capacity

```text
BF16      393,024 tokens
RABIT-KV  2,074,592 tokens
ratio     5.2785×
```

## Reported deployment latency

Configuration:

```text
context tokens = 2048
output tokens  = 32
```

Final RABIT-KV medians:

```text
TPOT = 21.6095 ms/token
TTFT = 121.1379 ms
wall = 790.7785 ms
```

These latency values are RABIT-KV deployment measurements. They are not presented as a matched BF16 speedup result.

## Quality evaluation

The committed quality suite compares BF16 against the final RABIT-KV operating point only.

Key results:

```text
WikiText-2 continuation PPL: 8.5020 -> 8.6317 (+1.53%)
Chinese continuation PPL:   10.1176 -> 10.3308 (+2.11%)
Spanish continuation PPL:    5.0035 ->  5.1139 (+2.21%)
NIAH:                        15/15   -> 15/15
Passage Retrieval:           100%    -> 100%
Qasper 8K+:                  35.9    -> 35.6 F1
HotpotQA 8K+:                60.6    -> 55.2 F1
```

WikiText-2 should be described specifically as **continuation perplexity**, not generic full-dataset perplexity.

## Integrity checks

The final deployment path passed:

- **105 regression tests**
- targeted decode-append state/byte exactness checks
- full-attention exactness checks

The quality harness also verifies the expected SHA-256 of the authoritative `rabit_kv2.py` source before running.
