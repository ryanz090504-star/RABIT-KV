"""Per-window fail-closed N32 runtime gates retaining D1 numerical bounds."""
import math

def validate(rows,events,w,cache,budget,oracle):
    wid=w['window'];packed=cache=='rabit_kv2'
    assert len(rows)==128 and [r['window'] for r in rows]==[wid]*128
    assert all(r['kind']=='score' for r in rows)
    assert [r['target_index'] for r in rows]==list(range(128))
    assert [r['target_id'] for r in rows]==w['targets']
    assert [r['logit_position'] for r in rows]==list(range(1023,1151))
    assert [r['absolute_position'] for r in rows]==list(range(1024,1152))
    assert all(math.isfinite(r['nll']) and r['nll']>=0 for r in rows)
    assert len({r['request'] for r in rows})==1
    assert events and all(e['window']==wid and e['kind']=='score' for e in events)
    assert len({e['request'] for e in events})==1
    assert not any(e['event']=='forbidden_fallback' for e in events)
    scheduled=([(0,1024)] if budget==16384 else [(0,512),(512,512)])+[(n,1) for n in range(1024,1151)]
    expectedL=([513,516,544,548,768,1024] if budget==512 else [])+[1025,1028,1055,1056,1060,1088,1151]
    for layer in range(32):
        ee=[e for e in events if e.get('layer')==layer]
        attn=[e for e in ee if e['event']=='attention']
        assert [(e['context'],e['query']) for e in attn]==scheduled
        assert all(e['packed']==packed and e['backend']=='TRITON_ATTN' for e in attn)
        if packed:
            assert all(e['cache_dtype']=='torch.uint8' and e['cache_shape'][-1]==24832 for e in attn)
            seen=set();reads=[]
            for e in ee:
                if e['event']=='write':
                    assert e['write_readback_equal'] and e['page_bytes']==24832
                    assert e['pages']==len(e['physical_ids']);seen.update(e['physical_ids'])
                if e['event']=='read':
                    assert set(e['physical_ids'])<=seen
                    assert e['pages']==len(e['physical_ids'])==(e['total_tokens']-4)//32 and e['page_bytes']==24832
                    reads.append(e['total_tokens'])
            assert reads==list(range(1025 if budget==16384 else 513,1152))
            assert any(e['event']=='fast_append' for e in ee)
            if budget==512:assert any(e['event']=='chunk_plan' for e in ee)
        name=f'model.layers.{layer}.self_attn.attn'
        oo=[e for e in oracle if e.get('layer')==name]
        rope=[e for e in oo if e['event']=='rope']
        assert [(e['context'],e['query']) for e in rope]==scheduled
        assert all(e['valid'] and e['cache_error']==0 for e in rope)
        args=[e for e in oo if e['event']=='attention_arguments']
        assert [(e['context'],e['query']) for e in args]==scheduled
        assert all(e['causal'] and e['query_heads']==32 and e['kv_heads']==8 and abs(e['softmax_scale']-128**-.5)<1e-12 for e in args)
        for kind in ['actual_dense','matched_context_dense','matched_unified_dense']:
            these=[e for e in oo if e['event']==kind]
            assert [e['query_index'] for e in these]==[0,3,31,1023 if budget==16384 else 511]
            assert all(e['valid'] and e['max_abs']<.025 for e in these)
        checks=[e for e in oo if e['event']==('packed_dequantized_dense' if packed else 'bf16_decode_dense')]
        assert [e['total_tokens'] for e in checks]==expectedL
        assert all(e['valid'] and e['max_abs']<.025 for e in checks)
    if packed:assert any(e['event']=='cleanup' and e['live_sidecars']==0 for e in events)
    ii=[e for e in oracle if e['event']=='model_inputs']
    assert [(e['context'],e['query']) for e in ii]==scheduled
    sequence=w['context']+w['targets'][:-1]
    for e in ii:
        c,n=e['context'],e['query'];assert e['ids']==sequence[c:c+n] and e['positions']==list(range(c,c+n))
    ll=[e for e in oracle if e['event']=='independent_logit_score']
    assert [e['target_index'] for e in ll]==list(range(128))
    assert [e['target_id'] for e in ll]==w['targets']
    assert all(e['error']<2e-5 and abs(e['nll']-r['nll'])<1e-12 for e,r in zip(ll,rows))
    assert all(e.get('window')==wid for e in oracle)
    return math.exp(math.fsum(r['nll'] for r in rows)/128)
