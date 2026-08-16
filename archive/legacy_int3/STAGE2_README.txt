RABIT-2 Stage 2: exact physical page allocation and layout

Adds a new vLLM cache dtype: rabit_kv2
- K payload: packed 3-bit
- V payload: packed 2-bit
- K sequence group: 32 tokens
- V dimension group: 32 values
- Primary metadata: UINT8
- Second-level metadata: BF16 min/scale per 64 metadata values
- R4 residual: separate per-sequence BF16 ring (not reserved in every page)

Stage 2 establishes allocator and layout correctness only.
Stage 3 will add cache-write, residual-ring, and fused attention kernels.
Do not report Stage-2 runtime as deployment latency.

Apply from the kvquant_pkg directory:
  Expand-Archive .\RABIT2_vLLM_stage2_allocator.zip -DestinationPath . -Force

Run:
  python.exe -m modal run .\modal_rabit2_stage2_allocator_test.py

Use --block-size 32 for rabit_kv2.
