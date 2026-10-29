import os,sys,json,hashlib,time,math,gc,subprocess,importlib,contextlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
import transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers.cache_utils import DynamicCache
import canonical_rabit_quality as cq
import canonical_ppl_core as core
import canonical_ppl_identity as ident
import canonical_quality_parity_tests as parity
HERE=Path(__file__).parent; OUT=Path(sys.argv[1]);OUT.mkdir(parents=True,exist_ok=True)
reg=json.loads((HERE/'registration.json').read_text())
def write(name,x):
 (OUT/name).write_text(json.dumps(x,indent=2,allow_nan=False))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(x,y):
 a=x.float();b=y.float();d=a-b
 return {'equal':torch.equal(x,y),'max_abs':d.abs().max().item(),'rmse':d.square().mean().sqrt().item(),'reference_rms':b.square().mean().sqrt().item(),'shape':list(x.shape)}
start=time.time();torch.set_num_threads(4);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
hw=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader,nounits'],text=True).strip()
assert hw.count('\n')==0 and 'H100' in hw and 79*1024<=int(hw.split(',')[1])<=82*1024
assert str(torch.__version__)=='2.11.0+cu130' and transformers.__version__=='4.48.2'
write('provenance.json',{'hardware':hw,'torch':str(torch.__version__),'cuda':torch.version.cuda,'transformers':transformers.__version__,'python':sys.version,'files':{p.name:sha(p) for p in HERE.iterdir() if p.is_file()},'registration':reg})
text=(HERE/'wikitext.txt').read_text();assert sha(HERE/'wikitext.txt')==ident.WIKITEXT_SHA256
held=(HERE/'wikitext2-valid.txt').read_text();assert sha(HERE/'wikitext2-valid.txt')==reg['replication_data']['sha256']
o,om=parity.load_oracle(HERE/'kvquant_k3.py');write('oracle-source.json',om)
K,V,M=cq.k3,cq.v2,cq.meta8_roundtrip

# POST-HOC variant: real-token extrema in incomplete K groups. Frozen source remains unchanged.
import ast
oracle_original=o['quantize_k3_sequence_affine_ref']
_tree=ast.parse((HERE/'kvquant_k3.py').read_text())
_fn=next(n for n in _tree.body if isinstance(n,ast.FunctionDef) and n.name=='quantize_k3_sequence_affine_ref')
_src=ast.get_source_segment((HERE/'kvquant_k3.py').read_text(),_fn)
_old='x = torch.nn.functional.pad(x, (0, 0, 0, pad_seq))'
_new='x = torch.cat([x, x[:, :, -1:, :].expand(-1, -1, pad_seq, -1)], dim=2)'
assert _src.count(_old)==1
exec(compile(_src.replace(_old,_new),'<independent-packed-real-range-oracle>','exec'),o)
oracle_real=o['quantize_k3_sequence_affine_ref'];o['quantize_k3_sequence_affine_ref']=oracle_original
def real_range_k(x):
 n=x.shape[0];pad=(-n)%32
 padded=torch.cat([x,x[-1:].expand(pad,-1,-1)]) if pad else x
 result=K(padded);result['decoded']=result['decoded'][:n];return result

ACTIVE_ARM='bf16'
BASE={'oracle':'rabit','oracle_range':'range_real','oracle_k':'k_exact','oracle_open':'open_exact','oracle_closed':'closed_exact'}
def open_k(x):
 return {'decoded':x.float()} if x.shape[0]%32 else real_range_k(x)
def closed_k(x):
 return real_range_k(x) if x.shape[0]%32 else {'decoded':x.float()}
def reference_state(rk,rv,arm):
 # Independent full-state packed oracle, followed by position-based exact overwrite.
 base=BASE.get(arm,arm);n=rk.shape[0];old=max(n-4,0);closed=(old//32)*32
 s=o['quantize_rabit2_kv_ref'](rk,rv);dk,dv=o['dequantize_rabit2_kv_ref'](s,dtype=torch.float32)
 dk=dk.clone()
 if base=='k_exact':dk[:old]=rk[:old].float()
 elif base=='open_exact':dk[closed:old]=rk[closed:old].float()
 elif base=='closed_exact':dk[:closed]=rk[:closed].float()
 return dk,dv
@contextlib.contextmanager
def variant(arm):
 global ACTIVE_ARM
 assert arm in reg['arms'];ACTIVE_ARM=arm;base=BASE.get(arm,arm)
 try:
  o['quantize_k3_sequence_affine_ref']=oracle_original if base=='rabit' else oracle_real
  selected_k={'bf16':K,'rabit':K,'passthrough':lambda x:{'decoded':x.float()},'range_real':real_range_k,'k_exact':lambda x:{'decoded':x.float()},'open_exact':open_k,'closed_exact':closed_k}[base]
  selected_v=(lambda x:{'decoded':x.float()}) if base=='passthrough' else V
  assert cq.meta8_roundtrip is M
  counts={'k':0,'v':0,'meta':0,'k_complete':0,'k_incomplete':0}
  def ck(x):
   counts['k']+=1;counts['k_complete' if x.shape[0]%32==0 else 'k_incomplete']+=1
   return selected_k(x)
  def cv(x):counts['v']+=1;return selected_v(x)
  def cm(x):counts['meta']+=1;return M(x)
  cq.k3,cq.v2,cq.meta8_roundtrip=ck,cv,cm
  yield counts
 finally:
  cq.k3,cq.v2,cq.meta8_roundtrip=K,V,M;o['quantize_k3_sequence_affine_ref']=oracle_original
class OracleCache(DynamicCache):
 @classmethod
 def from_prefill(cls,p):
  obj=cls();obj.raw=[(k.clone(),v.clone()) for k,v in p.to_legacy_cache()];obj.key_cache=[k for k,v in obj.raw];obj.value_cache=[v for k,v in obj.raw];obj._seen_tokens=obj.raw[0][0].shape[-2];return obj
 def update(self,k,v,i,cache_kwargs=None):
  rk,rv=self.raw[i];rk=torch.cat([rk,k],2);rv=torch.cat([rv,v],2);self.raw[i]=(rk,rv)
  dk,dv=reference_state(rk[0].permute(1,0,2),rv[0].permute(1,0,2),ACTIVE_ARM)
  self.key_cache[i]=dk.permute(1,0,2)[None].to(k.dtype);self.value_cache[i]=dv.permute(1,0,2)[None].to(v.dtype)
  if i==0:self._seen_tokens+=1
  return self.key_cache[i],self.value_cache[i]
def score(model,ctx,cont,arm,trace=False):
 saved={};checks=[];step=[0];handles=[];attention_index=[0];sdpa_calls=[0];cache_updates=[0]
 def capture(name):
  def fn(mod,args,kwargs,out):
   if step[0]==1:
    z=out[0] if isinstance(out,tuple) else out
    if isinstance(z,torch.Tensor):saved[name]=z.detach().cpu()
  return fn
 if trace:
  for i,l in enumerate(model.model.layers):
   for n,mod in [('input_norm',l.input_layernorm),('q_proj',l.self_attn.q_proj),('k_proj',l.self_attn.k_proj),('v_proj',l.self_attn.v_proj),('attention_output',l.self_attn),('layer_output',l)]:
    handles.append(mod.register_forward_hook(capture(f'{i:02d}/{n}'),with_kwargs=True))
 orig_sdpa=F.scaled_dot_product_attention
 def sdpa(q,k,v,*args,**kwargs):
  assert torch.backends.cuda.flash_sdp_enabled() and not torch.backends.cuda.math_sdp_enabled() and not torch.backends.cuda.mem_efficient_sdp_enabled() and not torch.backends.cuda.cudnn_sdp_enabled()
  sdpa_calls[0]+=1
  out=orig_sdpa(q,k,v,*args,**kwargs)
  if trace and step[0]==1:
   i=attention_index[0];attention_index[0]+=1
   for name,z in [('q_rope',q),('k_repeated',k),('v_repeated',v),('sdpa_output',out)]:saved[f'{i:02d}/{name}']=z.detach().cpu()
   mask=kwargs.get('attn_mask',args[0] if args else None)
   if mask is not None:saved[f'{i:02d}/mask']=mask.detach().cpu()
   # independent explicit FP32 attention, with observed mask and scale
   logits=q.float()@k.float().transpose(-2,-1)/math.sqrt(q.shape[-1])
   if mask is not None:logits=logits+mask
   ref=torch.softmax(logits,-1)@v.float()
   checks.append({'layer':i,'operation':'sdpa vs explicit fp32','is_causal':kwargs.get('is_causal',False),'mask_none':mask is None,**stats(out,ref)})
  return out
 F.scaled_dot_product_attention=sdpa
 try:
  with variant(arm) as dispatch,torch.inference_mode(),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
   pre=model(input_ids=ctx,use_cache=True);logits=[pre.logits[:,-1:,:]]
   cache=pre.past_key_values if arm=='bf16' else OracleCache.from_prefill(pre.past_key_values) if arm in BASE else core.canonical_cache_from_prefill(pre.past_key_values)
   raw=[(k.clone(),v.clone()) for k,v in pre.past_key_values.to_legacy_cache()] if trace and arm in ['rabit','range_real','k_exact','open_exact','closed_exact'] else None
   orig_update=cache.update
   def update(k,v,i,cache_kwargs=None):
    cache_updates[0]+=1
    assert k.shape[-2]==1 and v.shape[-2]==1
    assert cache_kwargs['cache_position'].tolist()==[1023+step[0]]
    dk,dv=orig_update(k,v,i,cache_kwargs)
    if trace and step[0]==1:
     saved[f'{i:02d}/k_before_cache']=k.detach().cpu();saved[f'{i:02d}/v_before_cache']=v.detach().cpu()
     if cache_kwargs:
      for n in ['cache_position','cos','sin']:
       if n in cache_kwargs:saved[f'{i:02d}/{n}']=cache_kwargs[n].detach().cpu()
    if raw is not None:
     rk,rv=raw[i];rk=torch.cat([rk,k],2);rv=torch.cat([rv,v],2);raw[i]=(rk,rv)
     if step[0] in reg['checkpoint_steps']:
      ok,ov=reference_state(rk[0].permute(1,0,2),rv[0].permute(1,0,2),arm)
      eq=torch.equal(dk,ok.permute(1,0,2)[None].to(dk.dtype)) and torch.equal(dv,ov.permute(1,0,2)[None].to(dv.dtype))
      checks.append({'layer':i,'step':step[0],'operation':'same-device oracle cache','equal':eq,'n':dk.shape[2]});assert eq
    return dk,dv
   cache.update=update
   for t in range(cont.shape[1]-1):
    step[0]=t+1;out=model(input_ids=cont[:,t:t+1],past_key_values=cache,use_cache=True);logits.append(out.logits)
    assert out.past_key_values is cache and cache.get_seq_length()==1024+t+1
   row=core._row(torch.cat(logits,1),cont)
   layers=model.config.num_hidden_layers
   assert cache_updates[0]==127*layers and sdpa_calls[0]==128*layers
   if arm=='bf16' or arm in BASE:assert not any(dispatch.values())
   else:
    assert dispatch['k']>0 and dispatch['v']>0
    assert (dispatch['meta']==0) if arm=='passthrough' else (dispatch['meta']>0)
   assert len(row['token_nll'])==128 and all(math.isfinite(v) for v in row['token_nll'])
   assert abs(math.fsum(row['token_nll'])-row['loss_sum'])<1e-3
   row['audit']={'arm':arm,'dispatch':dispatch.copy(),'backend':'FLASH_ATTENTION only; math/memory-efficient/cuDNN disabled at every SDPA call','sdpa_calls':sdpa_calls[0],'cache_updates':cache_updates[0],'cache_class':type(cache).__name__,'target_ids':cont[0].tolist(),'target_positions':list(range(1024,1152)),'predictor_positions':list(range(1023,1151)),'context_sha256':hashlib.sha256(ctx.cpu().numpy().astype('<i8').tobytes()).hexdigest()}
  return row,saved,checks
 finally:
  F.scaled_dot_product_attention=orig_sdpa
  for h in handles:h.remove()
for key,windows in reg['windows'].items():
 spec=ident.MODELS[key];path=Path('/models/models')/spec['model_id'].replace('/','--')/'snapshots'/spec['revision']
 ver=ident.verify_dir(path,key);write(key+'-identity.json',ver);assert ver['passed']
 tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
 anchor_pool=core.build_token_pool(core.wikitext_lines(text),tok,32,1024,128);assert core.pool_sha256(anchor_pool)==spec['token_pool_sha256']
 pool=core.build_token_pool(core.wikitext_lines(held),tok,32,1024,128);assert core.pool_sha256(pool)==reg['replication_data']['token_pool_sha256'][key]
 write(key+'-tokens.json',{'pool':pool,'sha256':core.pool_sha256(pool),'tokenizer_files':{n:sha(path/n) for n,_,_ in spec['files'] if 'token' in n or n in ['vocab.json','merges.txt']}})
 model=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval()
 with torch.inference_mode():model(input_ids=torch.tensor([pool[:32]],device='cuda'))
 assert model.config._attn_implementation=='sdpa'
 write(key+'-config.json',model.config.to_dict())
 result={a:[] for a in reg['arms']};traces={}
 def run_row(arm,w):
  seq=torch.tensor([pool[(w-1)*1152:w*1152]],device='cuda');ctx,cont=seq[:,:1024],seq[:,1024:]
  row,tr,checks=score(model,ctx,cont,arm,trace=w==1 and arm in ['bf16','rabit','passthrough','range_real','k_exact','open_exact','closed_exact'])
  row['window']=w
  if arm!='bf16':assert row['token_nll'][0]==next(z for z in result['bf16'] if z['window']==w)['token_nll'][0]
  result[arm].append(row)
  if tr:
   traces[arm]=tr;torch.save({n:t for n,t in tr.items() if n.startswith('00/')},OUT/(key+'-'+arm+'-trace-layer0.pt'));write(key+'-'+arm+'-checks.json',checks)
  write(key+'-scores.json',result);print(key,arm,w,row['ppl'],flush=True)
 # Complete all validity controls on window 1 BEFORE the remaining scientific windows.
 for arm in reg['arms']:
  run_row(arm,1)
  ref={'passthrough':'bf16',**BASE}.get(arm)
  if ref:assert result[arm][0]['token_nll']==result[ref][0]['token_nll']
 write(key+'-validity-gates.json',{'passed':True,'window':1,'controls':['passthrough','oracle_range','oracle_k','oracle_open','oracle_closed'],'no_fallback':True})
 comp={a:{n:stats(t,traces['bf16'][n]) for n,t in traces[a].items() if n in traces['bf16']} for a in ['passthrough','rabit','range_real','k_exact','open_exact','closed_exact']}
 write(key+'-trace-comparison.json',comp)
 # Environment anchors: canonical test window 1, compared bitwise with attempt 008 by the audit; never analysed.
 anchors={}
 for arm in reg['anchor_arms']:
  seq=torch.tensor([anchor_pool[:1152]],device='cuda');row,_,_=score(model,seq[:,:1024],seq[:,1024:],arm);row['window']='test-1';anchors[arm]=row
 write(key+'-anchors.json',anchors)
 for w in windows[1:]:
  for arm in reg['scientific_arms']:run_row(arm,w)
 write(key+'-aggregates.json',{a:{'ppl':math.exp(math.fsum(r['loss_sum'] for r in rr)/(128*len(rr))),'tokens':128*len(rr)} for a,rr in result.items()})
 del model,traces;gc.collect();torch.cuda.empty_cache()
write('complete.json',{'completed':True,'elapsed_seconds':time.time()-start,'attempt':reg['attempt'],'post_hoc':True})
