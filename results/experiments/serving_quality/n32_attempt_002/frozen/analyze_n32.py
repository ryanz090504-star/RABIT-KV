"""Preregistered paired-window analysis; no cloud access or scoring implementation."""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from review_scores import quantile

HERE=Path(__file__).resolve().parent


def interval(values,quantiles,seed,blocks=1,reps=100000):
    assert len(values)%blocks==0
    means=[math.fsum(values[j:j+blocks])/blocks for j in range(0,len(values),blocks)]
    rng=random.Random(seed)
    samples=[math.fsum(means[rng.randrange(len(means))] for _ in means)/len(means) for _ in range(reps)]
    return [100*math.expm1(quantile(samples,q)) for q in quantiles]


def analyze(root):
    for name,digest in json.loads((HERE/'freeze-sha256.json').read_text()).items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest,name
    inp=json.loads((HERE/'inputs-n32.json').read_text());arms=[]
    audit=json.loads((root/'validity-audit.json').read_text())
    assert audit['valid'] and audit['full_four_cell_gates'] and audit['scored_windows']==32
    assert audit['input_sha256']==hashlib.sha256((HERE/'inputs-n32.json').read_bytes()).hexdigest()
    for i in range(4):
        d=root/f'cell-{i}'
        result=json.loads((d/'result.json').read_text());assert result['valid']
        rows=[json.loads(s) for s in (d/'token_scores.jsonl').read_text().splitlines()]
        m={(r['window'],r['target_index']):r for r in rows};assert len(m)==len(rows)==4096
        nll=[]
        for w in inp['windows']:
            scores=[]
            for j,t in enumerate(w['targets']):
                r=m[w['window'],j];assert r['target_id']==t and r['absolute_position']==1024+j and r['logit_position']==1023+j
                assert math.isfinite(r['nll']) and r['nll']>=0;scores.append(r['nll'])
            engine=json.loads((d/f'window-{w["window"]}.json').read_text())
            assert engine['token_ids']==w['targets'] and len(engine['engine_raw_nll'])==128
            assert max(abs(a-b) for a,b in zip(scores,engine['engine_raw_nll']))<2e-5
            nll.append(scores)
        assert abs(math.exp(math.fsum(x for w in nll for x in w)/4096)-result['ppl'])<1e-10
        arms.append(nll)
    results={}
    for name,indices in [('inspected4',range(4)),('previously_uninspected_physical28',range(4,32)),('all32',range(32))]:
        ix=list(indices);res={'windows':[i+1 for i in ix],'tokens_per_cell':128*len(ix),
            'ppl_by_cell':[math.exp(math.fsum(x for i in ix for x in arm[i])/(128*len(ix))) for arm in arms]}
        for endpoint,a,b in [('single',1,0),('later',2,3),('packed_schedule',2,1),('bf16_schedule',3,0)]:
            differences=[math.fsum(x-y for x,y in zip(arms[a][i],arms[b][i]))/128 for i in ix]
            q=[.0125,.9875] if name=='previously_uninspected_physical28' and endpoint in ['single','later'] else [.025,.975]
            res[endpoint]=dict(paired_window_delta_nll=differences,mean_delta_nll=math.fsum(differences)/len(ix),
                relative_ppl_pct=100*math.expm1(math.fsum(differences)/len(ix)),
                interval_quantiles=q,window_bootstrap_pct=interval(differences,q,20261020))
            if len(ix)>4:res[endpoint]['four_window_block_bootstrap_pct']=interval(differences,q,20261020,blocks=4)
        # Difference in differences, paired within each window.
        interaction=[math.fsum((arms[2][i][j]-arms[3][i][j])-(arms[1][i][j]-arms[0][i][j]) for j in range(128))/128 for i in ix]
        res['schedule_interaction']=dict(mean_delta_nll=math.fsum(interaction)/len(ix),relative_ratio_pct=100*math.expm1(math.fsum(interaction)/len(ix)),window_bootstrap_pct=interval(interaction,[.025,.975],20261020))
        results[name]=res
    return results


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--scores-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    data=analyze(a.scores_root)
    with a.output.open('x') as f:json.dump(data,f,indent=2,allow_nan=False)
