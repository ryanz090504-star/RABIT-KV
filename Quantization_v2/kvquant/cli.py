"""Unified CLI for KV cache quantization experiments.

kvq quality    --policy doc_naive --nbits 4  → PPL
kvq latency    --policy doc_naive --nbits 4  → vLLM/deploy
kvq tasks      --dataset needl                → task quality
kvq memory     --policy doc_naive --nbits 4  → storage
kvq report     --input results/*.jsonl        → Markdown
kvq validate   --input results/*.jsonl        → evidence audit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kvquant.policies import build_policy, list_policies
from kvquant.results import write_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="KV cache quantization benchmark")
    subs = parser.add_subparsers(dest="cmd", required=True)

    subs.add_parser("list-policies", help="List available quantization policies")

    # ── quality ──
    q = subs.add_parser("quality", help="Run PPL benchmark")
    q.add_argument("--model", default="TinyLlama/TinyLlama_v1.1")
    q.add_argument("--policy", default="document_naive")
    q.add_argument("--nbits", type=int, default=4)
    q.add_argument("--text-file", required=True)
    q.add_argument("--max-tokens", type=int, default=256)
    q.add_argument("--num-windows", type=int, default=4)
    q.add_argument("--batch-size", type=int, default=1)
    q.add_argument("--warmup-steps", type=int, default=3)
    q.add_argument("--dtype", default="float16")
    q.add_argument("--error-breakdown", action="store_true",
                   help="Compute per-layer quantization error (MSE, cosine, SNR)")
    q.add_argument("--output")

    # ── latency ──
    l = subs.add_parser("latency", help="Run kernel latency diagnostics or future vLLM deploy benchmark")
    l.add_argument("--model", default="TinyLlama/TinyLlama_v1.1")
    l.add_argument("--policy", default="document_naive")
    l.add_argument("--nbits", type=int, default=4)
    l.add_argument("--backend", choices=["triton", "vllm"], default="triton")
    l.add_argument("--iters", type=int, default=100)
    l.add_argument("--warmup-iters", type=int, default=10)
    l.add_argument("--snapshot-dir")
    l.add_argument("--input-len", type=int, default=1024)
    l.add_argument("--output-len", type=int, default=128)
    l.add_argument("--num-prompts", type=int, default=100)
    l.add_argument("--request-rate", type=float)
    l.add_argument("--max-concurrency", type=int)
    l.add_argument("--vllm-root", help="Path to editable vLLM fork with kvquant_k3 support")
    l.add_argument("--kv-cache-dtype", help="Expected vLLM KV cache dtype, e.g. kvquant_k3")
    l.add_argument("--output")

    # ── tasks ──
    t = subs.add_parser("tasks", help="Run task quality benchmark")
    t.add_argument("--model", default="TinyLlama/TinyLlama_v1.1")
    t.add_argument("--policy", default="document_naive")
    t.add_argument("--nbits", type=int, default=4)
    t.add_argument("--dataset", choices=["needle", "longbench", "ruler"], default="needle")
    t.add_argument("--dataset-path")
    t.add_argument("--max-examples", type=int, default=50)
    t.add_argument("--max-new-tokens", type=int, default=20)
    t.add_argument("--context-lens", help="Comma-separated context token lengths, e.g. 4096,8192")
    t.add_argument("--dtype", default="float16")
    t.add_argument("--output")

    # ── memory ──
    m = subs.add_parser("memory", help="Estimate KV cache memory")
    m.add_argument("--model", default="TinyLlama/TinyLlama_v1.1")
    m.add_argument("--policy", default="document_naive")
    m.add_argument("--nbits", type=int, default=4)
    m.add_argument("--layers", type=int, default=22)
    m.add_argument("--kv-heads", type=int, default=4)
    m.add_argument("--head-dim", type=int, default=64)
    m.add_argument("--batch-size", type=int, default=8)
    m.add_argument("--seq-len", type=int, default=4096)
    m.add_argument("--memory-budget-gb", type=float)
    m.add_argument("--output")

    # ── report ──
    r = subs.add_parser("report", help="Generate Markdown report")
    r.add_argument("--input", nargs="+", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--title", default="KV Cache 量化实验报告")

    # ── validate ──
    v = subs.add_parser("validate", help="Audit evidence labels")
    v.add_argument("--input", nargs="+", required=True)
    v.add_argument("--config")

    # ── vLLM fork readiness ──
    vc = subs.add_parser("vllm-check", help="Check whether a vLLM fork exposes kvquant_k3 integration surfaces")
    vc.add_argument("--vllm-root", required=True)

    args = parser.parse_args(argv)

    if args.cmd == "list-policies":
        for name in list_policies():
            print(name)
        return 0

    policy = None
    if hasattr(args, "policy") and args.policy:
        policy = _build_policy_from_args(args.policy, getattr(args, "nbits", 4))

    if args.cmd == "quality":
        from kvquant.benchmark.quality import run_quality_benchmark
        result = run_quality_benchmark(
            model_name=args.model,
            policy=policy,
            text_file=args.text_file,
            max_tokens=args.max_tokens,
            num_windows=args.num_windows,
            batch_size=args.batch_size,
            warmup_steps=args.warmup_steps,
            dtype=args.dtype,
            error_breakdown=args.error_breakdown,
        )
        row = _dataclass_to_dict(result)
        if args.output:
            write_jsonl(args.output, row)
        print(json.dumps(row, indent=2, sort_keys=True, default=str))
        return 0

    if args.cmd == "latency":
        if args.backend == "triton":
            from kvquant.benchmark.latency import run_triton_kernel_benchmark
            result = run_triton_kernel_benchmark(
                model_name=args.model,
                policy_name=args.policy,
                nbits=args.nbits,
                iters=args.iters,
                warmup_iters=args.warmup_iters,
                snapshot_dir=args.snapshot_dir,
                kv_cache_dtype=args.kv_cache_dtype,
            )
        elif args.backend == "vllm":
            from kvquant.benchmark.latency import run_vllm_benchmark
            result = run_vllm_benchmark(
                model_name=args.model,
                policy_name=args.policy,
                nbits=args.nbits,
                input_len=args.input_len,
                output_len=args.output_len,
                num_prompts=args.num_prompts,
                request_rate=args.request_rate,
                max_concurrency=args.max_concurrency,
                vllm_root=args.vllm_root,
                kv_cache_dtype=args.kv_cache_dtype,
            )
        else:
            parser.error(f"unknown latency backend: {args.backend}")
        row = _dataclass_to_dict(result)
        if args.output:
            write_jsonl(args.output, row)
        print(json.dumps(row, indent=2, sort_keys=True, default=str))
        return 0

    if args.cmd == "tasks":
        from kvquant.benchmark.tasks import run_needle_retrieval
        result = run_needle_retrieval(
            model_name=args.model,
            policy=policy,
            needle_dataset=args.dataset_path or "data/needle_retrieval.sample.jsonl",
            max_examples=args.max_examples,
            context_lens=_parse_int_list(args.context_lens),
            max_new_tokens=args.max_new_tokens,
            dtype=args.dtype,
            dataset_name=args.dataset,
        )
        row = _dataclass_to_dict(result)
        if args.output:
            write_jsonl(args.output, row)
        print(json.dumps(row, indent=2, sort_keys=True, default=str))
        return 0

    if args.cmd == "memory":
        from kvquant.benchmark.memory import estimate_memory
        result = estimate_memory(
            model=args.model,
            policy=policy,
            layers=args.layers,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            memory_budget_gb=args.memory_budget_gb,
        )
        row = _dataclass_to_dict(result)
        if args.output:
            write_jsonl(args.output, row)
        print(json.dumps(row, indent=2, sort_keys=True, default=str))
        return 0

    if args.cmd == "report":
        from kvquant.report import generate_report
        generate_report(inputs=args.input, output=args.output, title=args.title)
        print(args.output)
        return 0

    if args.cmd == "validate":
        from kvquant.validation import validate_result_files
        result = validate_result_files(inputs=args.input)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("status") == "pass" else 1

    if args.cmd == "vllm-check":
        from kvquant.vllm_plugin.fork_check import check_vllm_fork
        result = check_vllm_fork(args.vllm_root)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("status") == "pass" else 1

    parser.error(f"unknown command: {args.cmd}")
    return 2


def _dataclass_to_dict(obj: object) -> dict:
    """Convert a dataclass instance to a JSON-safe dict."""
    from dataclasses import asdict, is_dataclass
    if is_dataclass(obj):
        return asdict(obj)
    return dict(obj)


def _build_policy_from_args(name: str, nbits: int):
    try:
        return build_policy(name, nbits=nbits)
    except TypeError as exc:
        message = str(exc)
        if "unexpected keyword argument 'nbits'" not in message:
            raise
        if name == "attention_mixed":
            return build_policy(name, low_bits=nbits)
        return build_policy(name)


def _parse_int_list(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
