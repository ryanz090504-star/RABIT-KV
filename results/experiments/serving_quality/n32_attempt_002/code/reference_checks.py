"""CPU-only frozen Python state/causality checks. Does NOT execute Triton/model."""
import ast
import copy
import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any
import torch

HERE=Path(__file__).resolve().parent


def load_reference():
    ns=dict(torch=torch,Any=Any,dataclass=dataclasses.dataclass,field=dataclasses.field,
            cdiv=lambda a,b:(a+b-1)//b)
    records={}
    files=[('vllm/v1/attention/ops/kvquant_k3.py',lambda n:isinstance(n,ast.FunctionDef) and n.lineno<604),
           ('vllm/v1/kv_cache_interface.py',lambda n:getattr(n,'name','') in ['rabit2_packed_dim','rabit2_encoded_metadata_bytes','Rabit2PageLayout','rabit2_page_layout']),
           ('vllm/v1/attention/ops/rabit_kv2.py',lambda n:(isinstance(n,ast.FunctionDef) and (n.lineno<550 or n.name in ['_rabit2_stage3b1_validate_layout','_rabit2_stage4b4_exact_v2_batch','_rabit2_stage4b4_encode_page_from_primary'])) or getattr(n,'name','') in ['Rabit2SingleSequenceRuntime','Rabit2CausalChunkPlan'])]
    for rel,predicate in files:
        raw=(HERE/'source'/rel).read_bytes();tree=ast.parse(raw.decode())
        nodes=[]
        for n in tree.body:
            if predicate(n):nodes.append(n)
            elif isinstance(n,ast.Assign):
                if all(isinstance(t,ast.Name) and t.id.startswith(('RABIT2_','KVQUANT_')) for t in n.targets):nodes.append(n)
        module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])
        exec(compile(ast.fix_missing_locations(module),rel,'exec'),ns)
        records[rel]=dict(sha256=hashlib.sha256(raw).hexdigest(),selected_nodes=[getattr(n,'name','constant') for n in nodes])
    return ns,records


def allocate(r,total):
    rt=r['Rabit2SingleSequenceRuntime'](8,128,128)
    pages=(total+31)//32+4
    cache=torch.zeros((pages,rt.layout.page_bytes),dtype=torch.uint8)
    table=torch.randperm(pages).to(torch.int32)
    return rt,cache,table


def materialize(r,rt,cache,table):
    pairs=[r['decode_rabit2_page_ref'](cache[int(table[i])],layout=rt.layout) for i in range(rt.closed_pages)]
    pairs.append(rt.tail_materialize(torch.float32))
    return tuple(torch.cat([p[j] for p in pairs]) for j in [0,1])


def dense(q,k,v):
    # q=[1,32,128], contiguous groups of four queries share one KV head.
    k=k.repeat_interleave(4,dim=1).transpose(0,1).double()
    v=v.repeat_interleave(4,dim=1).transpose(0,1).double()
    qq=q.transpose(0,1).double()
    scores=(qq@k.transpose(-1,-2))/math.sqrt(128)
    return (scores.softmax(-1)@v).transpose(0,1).float()


def main():
    torch.set_num_threads(2);torch.manual_seed(20261020)
    r,provenance=load_reference();rows=[];allsteps=0
    for prefix,chunk in [(0,8),(3,8),(4,8),(31,8),(35,16),(60,64),(64,64),(96,64),(127,64),(512,512),(1024,127)]:
        total=prefix+chunk;k=torch.randn(total,8,128).bfloat16();v=torch.randn_like(k)
        ref,cr,bt=allocate(r,total);fast,cf,_=allocate(r,total)
        if prefix:
            ref.append(k[:prefix],v[:prefix],cr,bt);fast.append(k[:prefix],v[:prefix],cf,bt)
        plan=r['Rabit2CausalChunkPlan'](fast,k[prefix:],v[prefix:],cf,bt)
        checks={0,chunk-1}|{j for j in range(chunk) if (prefix+j+1-4)%32 in [0,1,31]}
        max_manual_sdpa=0.;max_quantized_raw=0.;direct_count=0
        for j in range(chunk):
            ref.append(k[prefix+j:prefix+j+1],v[prefix+j:prefix+j+1],cr,bt);plan.apply_step(j)
            n=prefix+j+1;aged=max(n-4,0)
            assert fast.total_tokens==n and fast.closed_pages==aged//32
            assert (0 if fast.open_k is None else len(fast.open_k))==aged%32
            assert torch.equal(fast.recent_k,k[max(0,n-4):n])
            assert torch.equal(fast.recent_v,v[max(0,n-4):n])
            for name in ['open_k','open_v_packed','open_v_min','open_v_scale','recent_k','recent_v']:
                a,b=getattr(ref,name),getattr(fast,name)
                assert (a is None and b is None) or (a is not None and b is not None and torch.equal(a,b)),(prefix,j,name)
            ids=bt[:fast.closed_pages].long();assert torch.equal(cr[ids],cf[ids]);allsteps+=1
            if j in checks:
                mk,mv=materialize(r,fast,cf,bt)
                q=torch.randn(1,32,128).bfloat16()
                got=dense(q,mk,mv)
                sdpa=torch.nn.functional.scaled_dot_product_attention(q.transpose(0,1).float(),mk.repeat_interleave(4,dim=1).transpose(0,1),mv.repeat_interleave(4,dim=1).transpose(0,1),is_causal=False).transpose(0,1)
                err=float((got-sdpa).abs().max());assert err<5e-6;max_manual_sdpa=max(max_manual_sdpa,err)
                # Independent whole-prefix quantization bypasses chunk-plan/update logic.
                state=r['quantize_rabit2_kv_ref'](k[:n],v[:n])
                dk,dv=r['dequantize_rabit2_kv_ref'](state,dtype=torch.float32)
                assert torch.equal(mk,dk) and torch.equal(mv,dv),(prefix,j,'whole-prefix mismatch')
                direct_count+=1
                max_quantized_raw=max(max_quantized_raw,float((got-dense(q,k[:n],v[:n])).abs().max()))
        rows.append(dict(prefix=prefix,chunk=chunk,state_steps=chunk,direct_dequantized_checks=direct_count,
                         dense_fp64_vs_sdpa_max_abs=max_manual_sdpa,quantized_vs_raw_attention_max_abs=max_quantized_raw))
    # Mutate all future K/V at selected pilot-sized chunk cuts. Visible state and
    # dequantized attention must be unchanged, despite eagerly prewritten pages.
    poisoned=[];prefix=512;chunk=512
    k=torch.randn(1024,8,128).bfloat16();v=torch.randn_like(k);q=torch.randn(1,32,128).bfloat16()
    for cut in [0,3,31,35,127,255,510]:
        outputs=[]
        for poison in [False,True]:
            kk=k.clone();vv=v.clone()
            if poison:kk[prefix+cut+1:]=1000;vv[prefix+cut+1:]=-1000
            rt,cache,bt=allocate(r,1024);rt.append(kk[:prefix],vv[:prefix],cache,bt)
            plan=r['Rabit2CausalChunkPlan'](rt,kk[prefix:],vv[prefix:],cache,bt)
            for step in range(cut+1):plan.apply_step(step)
            mk,mv=materialize(r,rt,cache,bt);outputs.append((mk,mv,dense(q,mk,mv)))
        assert all(torch.equal(a,b) for a,b in zip(*outputs)),cut;poisoned.append(cut)
    report=dict(valid=True,torch=torch.__version__,device='CPU',state_steps=allsteps,cases=rows,
                future_poison_cuts=poisoned,provenance=provenance,
                limits='Extracted frozen Python state/codec only; no final CUDA append/tail kernels, RoPE, actual model QKV, or GPU backend executed. Shared codec limits mathematical independence.')
    with (HERE/'cpu-reference-results.json').open('x') as f:json.dump(report,f,indent=2)
    print('CPU reference checks passed:',allsteps,'state steps')


if __name__=='__main__':main()
