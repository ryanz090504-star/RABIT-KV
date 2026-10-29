"""One fresh engine cell; runs only inside the explicitly launched container."""
import argparse
import importlib.metadata
import json
import math
import os
import sys
from pathlib import Path
from core import *
from n32_validate import validate
from manifest import inputs


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', type=int, required=True)
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--run-dir', type=Path, required=True)
    args = ap.parse_args()
    cache, budget = CELLS[args.cell]
    dest = args.run_dir / f'cell-{args.cell}'
    dest.mkdir(exist_ok=False)
    os.environ['PILOT_CELL_DIR'] = str(dest)
    for key in ('VLLM_RABIT2_STAGE3C_IMPL', 'VLLM_RABIT2_STAGE3C_COMPONENT_PROFILE'):
        assert key not in os.environ
    import torch
    from vllm import LLM, SamplingParams
    from vllm.config.compilation import CompilationMode, CUDAGraphMode
    import vllm.v1.attention.ops.rabit_kv2 as r
    assert torch.cuda.device_count() == 1
    assert torch.__version__ == '2.11.0+cu130' and torch.version.cuda == '13.0'
    assert r._RABIT2_FINAL_FAST_DECODE_APPEND
    assert r.Rabit2SingleSequenceRuntime.append is r._rabit2_final_fast_decode_append
    assert r.rabit2_online_decode_attention_triton is r.rabit2_online_decode_attention_triton_stage4b3_gqa4
    assert r._rabit2_stage4b1_exactmeta_emit_tail_partial is r._rabit2_stage4d3_4_emit_tail_partial
    inp = inputs()
    validate_inputs(inp)
    kwargs = dict(BASE, model=args.model_dir, tokenizer=args.model_dir,
                  revision=REVISION, tokenizer_revision=REVISION,
                  kv_cache_dtype=cache, max_num_batched_tokens=budget,
                  logprobs_mode='raw_logprobs')
    assert kwargs == json.loads(Path('/pilot/frozen/preregistration-v2.json').read_text())['cells'][args.cell]
    write(dest / 'requested.json', kwargs)
    llm = LLM(**kwargs)
    cfg = llm.llm_engine.vllm_config
    assert cfg.use_v2_model_runner, 'V1 fallback forbidden'
    assert not cfg.model_config.logits_processors
    assert cfg.cache_config.cache_dtype == cache
    assert cfg.cache_config.block_size == 32
    assert not cfg.cache_config.enable_prefix_caching
    assert cfg.scheduler_config.enable_chunked_prefill
    assert cfg.scheduler_config.max_num_batched_tokens == budget
    assert cfg.model_config.enforce_eager and cfg.model_config.dtype == torch.bfloat16
    assert cfg.compilation_config.mode == CompilationMode.NONE
    assert cfg.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
    assert cfg.parallel_config.tensor_parallel_size == 1
    write(dest / 'effective.json', {'cache': cache, 'budget': budget,
          'config': str(cfg), 'runtime': {p: importlib.metadata.version(p)
           for p in ('torch', 'transformers', 'vllm', 'triton', 'tokenizers')},
          'torch_cuda': torch.version.cuda, 'd1_source_hashes': json.loads(Path('/root/vllm-kvquant/d1-instrumentation.json').read_text()), 'source_hashes': json.loads(Path('/root/vllm-kvquant/pilot-instrumentation.json').read_text())})
    write(dest / 'packages.json', sorted((d.metadata.get('Name','unknown'),d.version)
                                        for d in importlib.metadata.distributions()))
    generated = []
    for w in inp['windows']:
        sp = SamplingParams(temperature=0, max_tokens=128, ignore_eos=True, logprobs=1,
                            extra_args={'pilot_kind':'score','pilot_window':w['window']})
        out = llm.generate([{'prompt_token_ids': w['context']}], sp, use_tqdm=False)[0]
        sample = out.outputs[0]
        assert list(sample.token_ids) == w['targets']
        assert out.num_cached_tokens == 0
        assert len(sample.logprobs) == 128
        raw = [-float(d[t].logprob) for d, t in zip(sample.logprobs, w['targets'])]
        assert all(math.isfinite(n) and n >= 0 for n in raw)
        current = [s for s in rows(dest / 'token_scores.jsonl') if s['window'] == w['window']]
        assert len(current) == 128
        assert max(abs(a - b['nll']) for a, b in zip(raw, current)) < 2e-5
        generated.append({'window': w['window'], 'request_id': out.request_id,
                          'token_ids': list(sample.token_ids), 'engine_raw_nll': raw})
        write(dest / f'window-{w["window"]}.json', generated[-1])
        validate(current, rows(dest/f'path-window-{w["window"]:02d}.jsonl'), w, cache, budget, rows(dest/f'oracle-window-{w["window"]:02d}.jsonl'))
        print(json.dumps(dict(progress_window=w['window'],valid=True)),flush=True)
    for i,prompt in enumerate(inp['free_prompts']):
        out=llm.generate([{'prompt_token_ids':prompt['ids']}], SamplingParams(temperature=0,max_tokens=32,ignore_eos=True,extra_args={'pilot_kind':'free','pilot_window':i+1}),use_tqdm=False)[0]
        write(dest/f'free-{i}.json',dict(prompt=prompt,kind='free',output_ids=list(out.outputs[0].token_ids),text=out.outputs[0].text))
    scored = rows(dest / 'token_scores.jsonl')
    ppl=validate_rows(scored,inp['windows'])
    write(dest / 'result.json', {'valid': True, 'scored_tokens': 4096, 'ppl': ppl,
                                'cache': cache, 'prefill_budget': budget, 'latency_claim': False})


if __name__ == '__main__':
    main()
