"""H1 analysis (registered in REGISTRATION-H1.md, section 5). Local, after the Pod; exploratory; changes no C4 judgement.
Unit = window (32 per model). Layers and tokens are never treated as independent observations.
  per window w, from the Pod's metrics (dE = E_meta8 - E_bf16, attention-score RMS error reduction, logit units):
    S_all(w)  = mean over all layers of dE(w, layer)      [primary summary]
    S_0(w)    = dE(w, 0)                                   [layer 0, reported separately]
    S_rest(w) = mean over layers >= 1 of dE(w, layer)
    F_all(w)  = mean over layers of the fraction of K minima moved > 0.5 K3 step by META8 (descriptive)
  C4 gain G(w) = mean NLL(off) - mean NLL(K-min) of that window (positive = K-min better), from the verified raw C4 scores;
    one-chunk = primary (the one-chunk prefill stores exactly the BF16 context K that the replay reproduces), two-chunk = secondary
  association per model: Spearman rho(S, G) with a percentile bootstrap 95 % interval over windows (20,000 resamples of the 32
    windows, random.Random(20261118)); Pearson r descriptive. Expected direction under H1: rho > 0.
  model contrast (descriptive): per-layer median dE over windows, per model; median S_all with bootstrap interval.
  provenance: per-window HF BF16 NLL vs C4 bf16_single NLL (same windows): median and max |difference|, Pearson r.
Reading rules (fixed in advance): a weak or null rho weakens only the reading that this metric PREDICTS window gains; it does not by
itself show that the mechanism is absent (|rho| < 0.3 is not a refutation). This is a BF16 fixed-input replay: it measures the
first-order score error of closed pages, not how errors propagate through the network, and not the open group or V.
argv: <capture out dir> <post-c4-explore dir> <out json> [--synthetic <json with gains>]  (the last form is for test T4 only)"""
import json, math, random, sys
from pathlib import Path
import numpy as np

D, PCE, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
SYN = sys.argv[5] if len(sys.argv) > 5 and sys.argv[4] == '--synthetic' else None
B, SEED = 20000, 20261118


def rank(x):
    x = np.asarray(x, float); o = np.argsort(x, kind='mergesort'); r = np.empty(len(x)); r[o] = np.arange(len(x))
    for v in np.unique(x):                      # average ranks for ties
        idx = np.where(x == v)[0]
        if len(idx) > 1:
            r[idx] = r[idx].mean()
    return r


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float('nan')
    return float(np.corrcoef(ra, rb)[0, 1])


def boot_rho(a, b, seed):
    a, b = np.asarray(a, float), np.asarray(b, float); n = len(a); rng = random.Random(seed); vals = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        v = spearman(a[idx], b[idx])
        if not math.isnan(v):
            vals.append(v)
    vals.sort()
    if len(vals) < B // 2:                      # constant summary (e.g. no minimum ever moved > 0.5 step): no interval
        return [float('nan'), float('nan')], len(vals)
    return [vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]], len(vals)


def boot_median(x, seed):
    x = np.asarray(x, float); n = len(x); rng = random.Random(seed)
    v = sorted(float(np.median(x[[rng.randrange(n) for _ in range(n)]])) for _ in range(B))
    return [v[int(0.025 * B)], v[int(0.975 * B) - 1]]


if SYN:
    syn = json.loads(Path(SYN).read_text())
    gains = syn['gains']; bf16_nll = syn.get('bf16_nll', {})
else:
    sys.path.insert(0, str(PCE)); import load_c4 as L
    ver = L.verify()
    assert all(not v['score_file_mismatches'] and v['archives_equal_transfer_record'] for k, v in ver.items() if k in L.PODS)
    gains, bf16_nll = {}, {}
    for model in ('qwen', 'llama'):
        S = L.load(model)
        for st, (_, off, kmin, bf16) in L.SETTINGS.items():
            gains[f'{model}/{st}'] = {w: float(np.mean(S[off][w]) - np.mean(S[kmin][w])) for w in S[off]}
        bf16_nll[model] = {w: float(np.mean(S['bf16_single'][w])) for w in S['bf16_single']}

res = {'note': 'H1 exploratory analysis; BF16 fixed-input replay; C4 judgements unchanged', 'models': {}}
for mi, model in enumerate(('qwen', 'llama')):
    mp = D / f'metrics-{model}.jsonl'
    if not mp.exists():
        res['models'][model] = 'no metrics'; continue
    rows = [json.loads(x) for x in mp.read_text().splitlines() if x.strip()]
    wins = sorted({r['window'] for r in rows}); layers = sorted({r['layer'] for r in rows})
    dE = {(r['window'], r['layer']): r['dE'] for r in rows}
    fr = {(r['window'], r['layer']): r['min_ratio_frac_gt_half'] for r in rows}
    assert all((w, l) in dE for w in wins for l in layers), 'incomplete window x layer grid'
    S_all = {w: float(np.mean([dE[(w, l)] for l in layers])) for w in wins}
    S_0 = {w: dE[(w, 0)] for w in wins}
    S_rest = {w: float(np.mean([dE[(w, l)] for l in layers if l >= 1])) for w in wins}
    F_all = {w: float(np.mean([fr[(w, l)] for l in layers])) for w in wins}
    per_layer = {l: {'median_dE': float(np.median([dE[(w, l)] for w in wins])), 'q25_dE': float(np.percentile([dE[(w, l)] for w in wins], 25)),
                     'q75_dE': float(np.percentile([dE[(w, l)] for w in wins], 75)),
                     'median_rel': float(np.median([r['rel'] for r in rows if r['layer'] == l])),
                     'median_frac_gt_half': float(np.median([fr[(w, l)] for w in wins]))} for l in layers}
    m = {'windows': len(wins), 'layers': len(layers), 'per_layer': per_layer,
         'S_all_median': float(np.median(list(S_all.values()))), 'S_all_median_boot95': boot_median(list(S_all.values()), SEED + 100 + mi),
         'S_0_median': float(np.median(list(S_0.values()))), 'S_rest_median': float(np.median(list(S_rest.values()))),
         'layer0_share_of_layer_sum_median': float(np.median([dE[(w, 0)] / sum(dE[(w, l)] for l in layers) if sum(dE[(w, l)] for l in layers) else float('nan') for w in wins])),
         'F_all_median': float(np.median(list(F_all.values()))), 'association': {}}
    for si, st in enumerate(('single', 'two')):
        G = gains[f'{model}/{st}']
        common = [w for w in wins if w in G]
        for ki, (lab, Sx) in enumerate((('S_all', S_all), ('S_0', S_0), ('S_rest', S_rest), ('F_all', F_all))):
            a = [Sx[w] for w in common]; b = [G[w] for w in common]
            ci, nb = boot_rho(a, b, SEED + 10 * mi + 3 * si + ki)
            m['association'][f'{st}/{lab}'] = {'n_windows': len(common), 'spearman': spearman(a, b), 'spearman_boot95': ci, 'bootstrap_valid': nb,
                                               'pearson': float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 else float('nan'),
                                               'role': 'primary' if (st, lab) == ('single', 'S_all') else ('layer 0, reported separately' if lab == 'S_0' else 'secondary')}
    nllf = D / f'nll-{model}.json'
    if nllf.exists() and model in bf16_nll:
        hf = json.loads(nllf.read_text())
        c = [w for w in hf if w in bf16_nll[model]]
        diff = [hf[w]['mean'] - bf16_nll[model][w] for w in c]
        m['provenance_hf_vs_c4_bf16_single'] = {'windows': len(c), 'median_abs_diff': float(np.median(np.abs(diff))), 'max_abs_diff': float(np.max(np.abs(diff))),
                                                'mean_diff': float(np.mean(diff)), 'pearson': float(np.corrcoef([hf[w]['mean'] for w in c], [bf16_nll[model][w] for w in c])[0, 1]),
                                                'flag_window_mismatch': float(np.median(np.abs(diff))) > 0.02}
    res['models'][model] = m
OUT.write_text(json.dumps(res, indent=1) + '\n')
for model, m in res['models'].items():
    if isinstance(m, dict):
        a = m['association']['single/S_all']
        print(model, 'S_all median', round(m['S_all_median'], 5), 'S_0 median', round(m['S_0_median'], 5), 'rho(S_all, gain one-chunk)',
              round(a['spearman'], 3), [round(x, 3) for x in a['spearman_boot95']])
print('H1_ANALYSIS_DONE')
