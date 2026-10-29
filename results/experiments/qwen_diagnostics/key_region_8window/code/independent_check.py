"""Independent (non-audit.py) check of RUNPOD-QWEN-002 per-token results. argv: run-gpu dir, out json."""
import json,math,hashlib,struct,sys
from pathlib import Path
import numpy as np
run=Path(sys.argv[1]);a=run/'artifacts';h=Path(__file__).parent
reg=json.loads((h/'stage/registration.json').read_text());st=json.loads((h/'stage-sha256.json').read_text())
J=lambda n:json.loads((a/n).read_text())
prov=J('provenance.json');gpu=json.loads((run/'gpu-preflight/gpu.json').read_text())
out={'provenance':{'staged_files_match_frozen':prov['files']==st,'registration_equals_frozen':prov['registration']==reg,'hardware':gpu['hardware'],'torch':gpu['torch'],'cuda':gpu['cuda'],'python':gpu['python'],'gpu_oracle_state_checks':gpu['gpu_oracle_state_checks'],'explicit_fp32_attention_rmse':gpu['explicit_fp32_attention_rmse']},'models':{}}
W=reg['windows']['qwen2_5_7b'];assert W==reg['windows']['llama3_1_8b']==[1,5,9,13,17,21,25,29]
for m in ['qwen2_5_7b','llama3_1_8b']:
 rows=J(m+'-scores.json');tok=J(m+'-tokens.json');pool=tok['pool'];base=json.loads((h/'baseline008'/(m+'-tokens.json')).read_text())
 ident=J(m+'-identity.json');vg=J(m+'-validity-gates.json');agg=J(m+'-aggregates.json')
 r={'identity_passed':ident['passed'],'validity_gates_passed':vg['passed'],'token_pool_equals_008':tok==base,'pool_sha256_ok':hashlib.sha256(struct.pack('<%dq'%len(pool),*pool)).hexdigest()==tok['sha256']}
 n_tok=0;bad=[];per={}
 for arm,rr in rows.items():
  want=W if arm in reg['scientific_arms'] else [1]
  if [x['window'] for x in rr]!=want:bad.append((arm,'windows'))
  for x in rr:
   v=np.array(x['token_nll'],dtype=np.float64);n_tok+=len(v)
   if len(v)!=128 or not np.all(np.isfinite(v)) or np.any(v<0):bad.append((arm,x['window'],'nll'))
   off=(x['window']-1)*1152
   if x['audit']['target_ids']!=pool[off+1024:off+1152]:bad.append((arm,x['window'],'targets'))
   if abs(math.fsum(x['token_nll'])-x['loss_sum'])>1e-3:bad.append((arm,x['window'],'sum'))
  per[arm]=np.array([np.mean(x['token_nll']) for x in rr])
  tot=math.exp(math.fsum(x['loss_sum'] for x in rr)/(128*len(rr)))
  if abs(tot-agg[arm]['ppl'])>1e-9*tot:bad.append((arm,'aggregate'))
 ctl={c:rows[c][0]['token_nll']==rows[ref][0]['token_nll'] for c,ref in {'passthrough':'bf16','oracle_range':'range_real','oracle_k':'k_exact','oracle_open':'open_exact','oracle_closed':'closed_exact'}.items()}
 rng=np.random.default_rng(reg['analysis']['bootstrap_seed']);ix=rng.integers(0,8,size=(reg['analysis']['bootstrap_replicates'],8))
 eff={}
 for x,y in [('open_exact','range_real'),('closed_exact','range_real'),('k_exact','range_real'),('range_real','bf16'),('open_exact','bf16'),('closed_exact','bf16'),('k_exact','bf16')]:
  d=per[x]-per[y];lo,hi=np.quantile(d[ix].mean(1),[.025,.975])
  eff[x+'_vs_'+y]={'mean_delta_nll':float(d.mean()),'ppl_rel_pct':100*math.expm1(d.mean()),'ci95_pct':[100*math.expm1(lo),100*math.expm1(hi)],'improved':int((d<0).sum()),'worsened':int((d>0).sum()),'per_window_delta_nll':d.round(6).tolist()}
 gap=per['range_real']-per['k_exact']
 r.update({'rows':sum(map(len,rows.values())),'token_losses':n_tok,'problems':bad,'bitwise_controls':ctl,'ppl_8window':{k:float(math.exp(per[k].mean())) for k in reg['scientific_arms']},
  'effects':eff,'gap_fraction_removed':{k:float((per['range_real']-per[k]).mean()/gap.mean()) for k in ['open_exact','closed_exact']},'gap_mean_nll':float(gap.mean())})
 out['models'][m]=r
Path(sys.argv[2]).write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=1)[:200])
