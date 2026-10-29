"""Fail-closed textual instrumentation of hash-bound, isolated sources only."""
import ast
from pathlib import Path
from core import sha, write


def once(s, old, new):
    if s.count(old) != 1:
        raise ValueError('source anchor is not unique: ' + old[:80])
    return s.replace(old, new)


def add_import(s, name):
    future = 'from __future__ import annotations\n'
    if future in s:
        return once(s, future, future + f'import {name}\n')
    return f'import {name}\n' + s


def instrument(root):
    root = Path(root)
    if (root / 'pilot-instrumentation.json').exists():
        raise ValueError('snapshot already instrumented')
    records = {}
    def edit(rel, transform):
        p = root / rel
        before = sha(p)
        original = p.read_text(encoding='utf-8')
        changed = transform(original)
        compile(changed, rel, 'exec')
        p.write_text(changed, encoding='utf-8', newline='\n')
        records[rel] = {'original_sha256': before, 'instrumented_sha256': sha(p)}

    def backend(s):
        s = add_import(s, 'pilot_trace')
        # Insert before the branch, so BF16 and packed schedules both get traced.
        return once(s, '        if self._is_rabit_kv2:\n            return self._forward_rabit_kv2(',
                    '        pilot_trace.attention(self, layer, attn_metadata, kv_cache)\n'
                    '        if self._is_rabit_kv2:\n            return self._forward_rabit_kv2(')

    def runner(s):
        # Publish identical CPU scheduler metadata for the BF16 control as well.
        s = add_import(s,'pilot_trace')
        s = once(s,'            req_id = new_req_data.req_id\n',
                 '            req_id = new_req_data.req_id\n'
                 '            pilot_trace.register_request(req_id, new_req_data.sampling_params)\n')
        return once(s, '            if self.cache_config.cache_dtype == "rabit_kv2":\n',
                    '            if self.cache_config.cache_dtype in ("rabit_kv2", "bfloat16"):\n')

    def sampler(s):
        s = add_import(s, 'pilot_sample')
        return once(s, '        if return_logprobs:\n            if self.logprobs_mode',
                    '        sampled = pilot_sample.override_sample(logits, input_batch, sampled)\n'
                    '        if return_logprobs:\n            if self.logprobs_mode')

    def ops(s):
        s = add_import(s, 'pilot_trace')
        s = once(s, '    kv_cache.reshape(kv_cache.shape[0], -1).index_copy_(\n'
                    '        0, physical_ids, pages_tensor\n    )\n    return end',
                 '    _pilot_before = pilot_trace.before_write(kv_cache, physical_ids)\n'
                 '    kv_cache.reshape(kv_cache.shape[0], -1).index_copy_(\n'
                 '        0, physical_ids, pages_tensor\n    )\n'
                 '    pilot_trace.written(kv_cache, physical_ids, pages_tensor, _pilot_before)\n    return end')
        s = once(s, '        cache2d.index_copy_(0, physical_id, page.reshape(1, -1))\n',
                 '        _pilot_before = pilot_trace.before_write(kv_cache, physical_id)\n'
                 '        cache2d.index_copy_(0, physical_id, page.reshape(1, -1))\n'
                 '        pilot_trace.written(kv_cache, physical_id, page.reshape(1, -1), _pilot_before)\n')
        # Later-chunk plans prewrite their closed pages before exposing each token.
        # Observe both bulk writers too; reads must remain tied to actual writes.
        s = once(s, '            cache2d.index_copy_(0, physical_ids, pages_tensor)\n',
                 '            _pilot_before = pilot_trace.before_write(kv_cache, physical_ids)\n'
                 '            cache2d.index_copy_(0, physical_ids, pages_tensor)\n'
                 '            pilot_trace.written(kv_cache, physical_ids, pages_tensor, _pilot_before)\n')
        s = once(s, '            kv_cache.reshape(kv_cache.shape[0], -1).index_copy_(\n'
                    '                0, physical_ids, pages_tensor\n            )\n',
                 '            _pilot_before = pilot_trace.before_write(kv_cache, physical_ids)\n'
                 '            kv_cache.reshape(kv_cache.shape[0], -1).index_copy_(\n'
                 '                0, physical_ids, pages_tensor\n            )\n'
                 '            pilot_trace.written(kv_cache, physical_ids, pages_tensor, _pilot_before)\n')
        # Hook the real closed-page kernel launch, not just an attention wrapper.
        tree = ast.parse(s)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Expr) and
                 isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Subscript) and
                 isinstance(n.value.func.value, ast.Name) and
                 n.value.func.value.id == '_rabit2_stage4b3_gqa4_closed_page_partial_kernel']
        if len(calls) != 1:
            raise ValueError('packed read kernel anchor changed')
        lines = s.splitlines(keepends=True)
        lines.insert(calls[0].end_lineno, '        pilot_trace.read(kv_cache, block_table_row, runtime)\n')
        s = ''.join(lines)
        # Startup/warmup can use fallback before CURRENT exists; scored Llama may not.
        s = s.replace('        return _rabit2_final_old_append(\n',
                      '        pilot_trace.forbidden("old_append")\n        return _rabit2_final_old_append(\n')
        s = once(s, '        return _rabit2_online_decode_attention_triton_stage4b2_exact(\n',
                 '        pilot_trace.forbidden("non_gqa4_attention")\n'
                 '        return _rabit2_online_decode_attention_triton_stage4b2_exact(\n')
        s = once(s, '    open_slot = int(self._rabit2_final_open_len)\n',
                 '    pilot_trace.event("fast_append")\n    open_slot = int(self._rabit2_final_open_len)\n')
        s = once(s, '        self.init_closed = int(runtime.closed_pages)\n',
                 '        pilot_trace.event("chunk_plan")\n        self.init_closed = int(runtime.closed_pages)\n')
        s = once(s, '    return removed\n\n\ndef rabit2_total_live_sidecars',
                 '    pilot_trace.finished(finished, rabit2_total_live_sidecars())\n'
                 '    return removed\n\n\ndef rabit2_total_live_sidecars')
        return s

    edit('vllm/v1/attention/backends/triton_attn.py', backend)
    edit('vllm/v1/worker/gpu/model_runner.py', runner)
    edit('vllm/v1/worker/gpu/sample/sampler.py', sampler)
    edit('vllm/v1/attention/ops/rabit_kv2.py', ops)
    write(root / 'pilot-instrumentation.json', records)
    return records
