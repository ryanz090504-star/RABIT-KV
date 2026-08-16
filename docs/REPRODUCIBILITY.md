# Reproducibility

## Primary recorded deployment environment

- Llama 3.1 8B Instruct
- NVIDIA H100 80GB HBM3
- CUDA 13.0 / 13.0.2
- PyTorch 2.11.0+cu130 in D3.4
- custom vLLM 0.10.0+kvquant
- vLLM base commit `f329ce405b12623fb8b1cf1830f12e5a712523be`

## Controlled configuration

- eager execution
- Triton attention backend
- CUDA graphs disabled
- torch.compile disabled
- `gpu_memory_utilization=0.82`
- block size 32
- max model length 32768
- max batched tokens 16384
- max sequences 32
- prefix caching disabled
- chunked prefill enabled

## Milestone evidence

Stage4C V3 controlled capacity:

`results/capacity_latency/rabit2_stage4c_v3_results.json`

Stage4D2.2 regression:

`results/capacity_latency/rabit2_stage4d2_2_lock_writer_only.log`

Stage4D3.4 prototype:

`results/experiments/stage4d3_4/rabit2_stage4d3_4_triton_fused_tailprep_prototype.log`

Expected physical capacity:

- BF16: 393,024 tokens
- RABIT: 2,074,592 tokens
- ratio: 5.2785x

Latest META8 quality:

- BF16 PPL 11.225819
- RABIT PPL 11.477385
- +2.241%
- 4.892491x logical compression

The raw META8 quality logfile was absent from the initial GitHub snapshot.
The final post-performance unified quality rerun will replace this consolidated
quality evidence with a fully archived raw run.