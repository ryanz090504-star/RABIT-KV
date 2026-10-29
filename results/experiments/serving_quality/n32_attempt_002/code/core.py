"""Pilot protocol and validation. Stdlib only; importing never contacts Modal."""
import hashlib
import json
import math
import struct
from pathlib import Path

SOURCE_COMMIT = '45fd03d8c10f2c14f96dcaedfcad25c4f56fe746'
REVISION = '359efdbb8af05b788a4ad4185215c6b8caa9052c'
MODEL_ID = 'LLM-Research/Meta-Llama-3.1-8B-Instruct'
DATA_SHA = 'd790b833ef8cf03a90db7bf1271b7520b83c45ce07ba3c1a9699df81e239eca0'
POOL_SHA = '88dadfe6521c338ca71b1edfe5e85f08b8e2466e0884f8003b5b34a7ee3c849f'
CELLS = [('bfloat16', 16384), ('rabit_kv2', 16384),
         ('rabit_kv2', 512), ('bfloat16', 512)]
BASE = dict(dtype='bfloat16', block_size=32, max_model_len=32768,
            max_num_seqs=32, enable_prefix_caching=False,
            enable_chunked_prefill=True, gpu_memory_utilization=0.82,
            enforce_eager=True, trust_remote_code=False, seed=0,
            attention_config={'backend': 'TRITON_ATTN'},
            compilation_config={'mode': 0, 'cudagraph_mode': 'NONE'})


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')


def digest_ids(ids):
    return hashlib.sha256(struct.pack(f'<{len(ids)}q', *ids)).hexdigest()


def token_pool(text, encode):
    lines = [s.strip() for s in text.splitlines() if s.strip()]
    pool = []
    for i in range(0, len(lines), 64):
        pool.extend(encode('\n'.join(lines[i:i + 64])))
        if len(pool) >= 32 * 1152:
            break
    if len(pool) < 32 * 1152:
        raise ValueError('insufficient dataset tokens')
    return pool[:32 * 1152]


def validate_inputs(inp):
    assert inp['model_revision'] == inp['tokenizer_revision'] == REVISION
    assert inp['model_id'] == MODEL_ID
    assert inp['dataset_sha256'] == DATA_SHA
    assert inp['full_n32_pool_sha256'] == POOL_SHA
    assert len(inp['windows']) == 32
    for i, w in enumerate(inp['windows']):
        assert w['window'] == i + 1 and w['stream_offset'] == i * 1152
        assert len(w['context']) == 1024 and len(w['targets']) == 128
        assert w['token_sha256'] == digest_ids(w['context'] + w['targets'])
        assert all(type(t) is int and 0 <= t < 128256 for t in w['context'] + w['targets'])
    assert inp['scoring'] == {'target_indices': [0, 127], 'absolute_positions': [1024, 1151],
                             'first_logit_position': 1023, 'decode_inputs': 127,
                             'bos': False, 'chat_template': False}


def validate_rows(rows, windows):
    assert len(rows) == 128 * len(windows)
    for w in windows:
        rr = [r for r in rows if r['window'] == w['window']]
        assert [r['target_index'] for r in rr] == list(range(128))
        assert [r['target_id'] for r in rr] == w['targets']
        assert [r['absolute_position'] for r in rr] == list(range(1024, 1152))
        assert all(math.isfinite(r['nll']) and r['nll'] >= 0 for r in rr)
    return math.exp(math.fsum(r['nll'] for r in rows) / (128 * len(windows)))


def validate_trace(events, cache, budget):
    """Require every real window/layer, not warmup aggregate counters."""
    assert len(events) > 0
    for window in range(1, 5):
        if cache == 'rabit_kv2':
            assert any(e['event'] == 'cleanup' and e['window'] == window and e['live_sidecars'] == 0
                       for e in events)
        for layer in range(32):
            ee = [e for e in events if e.get('window') == window and e.get('layer') == layer]
            sched = [e for e in ee if e['event'] == 'attention']
            expected = [(0, 1024)] if budget == 16384 else [(0, 512), (512, 512)]
            assert [(e['context'], e['query']) for e in sched if e['query'] > 1] == expected
            assert len([e for e in sched if e['query'] == 1]) == 127
            assert all(e['backend'] == 'TRITON_ATTN' for e in sched)
            if cache == 'rabit_kv2':
                assert all(e['packed'] for e in sched)
                assert any(e['event'] == 'write' and e['pages'] > 0 and e['changed'] for e in ee)
                assert any(e['event'] == 'read' and e['pages'] > 0 and e['page_bytes'] == 24832 for e in ee)
                writes = [e for e in ee if e['event'] == 'write']
                reads = [e for e in ee if e['event'] == 'read']
                assert all(e['write_readback_equal'] for e in writes)
                written_ids = {i for e in writes for i in e['physical_ids']}
                assert all(set(e['physical_ids']) <= written_ids for e in reads)
                assert len(reads) == (127 if budget == 16384 else 639)
                for e in reads:
                    assert e['pages'] == (e['total_tokens'] - 4) // 32
                    assert len(e['physical_ids']) == e['pages']
                assert any(e['event'] == 'fast_append' for e in ee)
                if budget == 512:
                    assert any(e['event'] == 'chunk_plan' for e in ee)
    assert not any(e['event'] == 'forbidden_fallback' for e in events)


class RequestMap:
    """Atomic batch remapping, including SWAP and destination replacement."""
    def __init__(self):
        self.rows = {}

    def update(self, update):
        if update is None:
            return
        for i in update.removed:
            self.rows.pop(i, None)
        for i, params, prompt, outputs in update.added:
            args = params.extra_args or {}
            if 'pilot_targets' in args:
                self.rows[i] = (args, outputs)
            else:
                self.rows.pop(i, None)
        for src, dst, direction in update.moved:
            a, b = self.rows.pop(src, None), self.rows.pop(dst, None)
            if a is not None:
                self.rows[dst] = a
            if getattr(direction, 'name', '') == 'SWAP' and b is not None:
                self.rows[src] = b
