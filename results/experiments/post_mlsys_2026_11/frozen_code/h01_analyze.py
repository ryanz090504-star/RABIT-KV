"""QSP-H01 pre-specified paired analysis (PROTOCOL.md section 5). Stdlib only. Recomputes every window mean NLL from the 128 rows.
Unit: window. Per prefill setting, the analysis set is the windows valid in all three cells of that setting (BF16, off,
on); no window is dropped for its loss. For each contrast the window differences d_w = NLL_x - NLL_y are summarised by
  mean (primary estimate), 95% paired percentile bootstrap CI over windows (20,000 resamples, seeded), and a t interval
  (secondary); PPL change = 100 (exp(.) - 1) applied to the estimate and to each CI endpoint.
Primary contrasts: on - off and on - BF16, in each setting. Context: off - BF16. Secondary: on_two - on_single.
argv: <cells dir> <frozen windows json> <out json>"""
import json, math, random, sys
from pathlib import Path

RUN, WIN, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
SETTINGS = {'single': ('bf16_single', 'off_single', 'on_single'), 'two_chunk': ('bf16_two', 'off_two', 'on_two')}
RESAMPLES, CONF, SEED = 20000, 0.95, 20261110
# two-sided 97.5% t quantiles, df 1..40 (beyond 40 the normal 1.96 is used; Student t values to 3 decimals)
T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110,
        2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042, 2.040, 2.037, 2.035, 2.032,
        2.030, 2.028, 2.026, 2.024, 2.023, 2.021]
order = [w['id'] for w in json.loads(WIN.read_text())['windows']]


def load(cell):
    p = RUN / cell / 'result.json'
    res = json.loads(p.read_text()) if p.exists() else {'windows': []}
    out = {}
    for r in res['windows']:
        if r.get('status') == 'scored' and r.get('valid'):
            rows = [json.loads(s) for s in (RUN / cell / f"scores-{r['window']}.jsonl").read_text().splitlines() if s.strip()]
            assert len(rows) == 128 and [x['nll'] for x in rows] == r['nll']
            m = math.fsum(x['nll'] for x in rows) / 128
            assert m == r['mean_nll']
            out[r['window']] = m
    status = {r['window']: r.get('status') if r.get('valid') or r.get('status') != 'scored' else 'invalid' for r in res['windows']}
    return out, {w: status.get(w, 'missing') for w in order}


def pct(xs, q):
    s = sorted(xs)
    h = (len(s) - 1) * q
    lo = math.floor(h)
    return s[lo] + (h - lo) * (s[min(lo + 1, len(s) - 1)] - s[lo])


def summarise(d, seed):
    n = len(d)
    if n < 2:
        return {'n': n}
    mean = math.fsum(d) / n
    sd = math.sqrt(math.fsum((x - mean) ** 2 for x in d) / (n - 1))
    rng = random.Random(seed)
    boots = [math.fsum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(RESAMPLES)]
    a = (1 - CONF) / 2
    blo, bhi = pct(boots, a), pct(boots, 1 - a)
    t = T975[n - 2] if n - 1 <= len(T975) else 1.96
    tlo, thi = mean - t * sd / math.sqrt(n), mean + t * sd / math.sqrt(n)
    ppl = lambda x: 100 * math.expm1(x)
    return {'n': n, 'mean_nll_diff': mean, 'sd': sd, 'bootstrap_ci': [blo, bhi], 't_ci': [tlo, thi],
            'ppl_change_pct': ppl(mean), 'ppl_change_bootstrap_ci_pct': [ppl(blo), ppl(bhi)], 'ppl_change_t_ci_pct': [ppl(tlo), ppl(thi)],
            'windows_higher': sum(x > 0 for x in d), 'windows_lower': sum(x < 0 for x in d), 'windows_equal': sum(x == 0 for x in d),
            'bootstrap': {'resamples': RESAMPLES, 'confidence': CONF, 'seed': seed, 'method': 'paired percentile over windows'}}


cells = {c: load(c) for s in SETTINGS.values() for c in s}
summary = {'job': 'QSP-H01', 'unit': 'window (1024 context + 128 targets)', 'frozen_order': order,
           'cell_status': {c: v[1] for c, v in cells.items()}, 'settings': {}}
seed = SEED
for name, (b, f, o) in SETTINGS.items():
    A = [w for w in order if all(w in cells[c][0] for c in (b, f, o))]
    m = {c: [cells[c][0][w] for w in A] for c in (b, f, o)}
    s = {'analysis_windows': A, 'n': len(A),
         'excluded': {w: {c: cells[c][1][w] for c in (b, f, o)} for w in order if w not in A},
         'mean_nll': {c: (math.fsum(m[c]) / len(A) if A else None) for c in (b, f, o)}}
    s['ppl'] = {c: (math.exp(v) if v is not None else None) for c, v in s['mean_nll'].items()}
    for label, x, y in (('on_vs_off', o, f), ('on_vs_bf16', o, b), ('off_vs_bf16', f, b)):
        s[label] = summarise([p - q for p, q in zip(m[x], m[y])], seed)
        seed += 1
    summary['settings'][name] = s
B = [w for w in order if w in cells['on_single'][0] and w in cells['on_two'][0]]
summary['secondary_on_two_vs_on_single'] = {'windows': B, **summarise([cells['on_two'][0][w] - cells['on_single'][0][w] for w in B], seed)}
summary['bf16_single_equals_two_bitwise'] = {w: cells['bf16_single'][0].get(w) == cells['bf16_two'][0].get(w)
                                             for w in order if w in cells['bf16_single'][0] and w in cells['bf16_two'][0]}
OUT.write_text(json.dumps(summary, indent=2))
print(json.dumps({k: {kk: (vv if not isinstance(vv, dict) else {a: vv[a] for a in ('n', 'mean_nll_diff', 'bootstrap_ci', 'ppl_change_pct',
                  'ppl_change_bootstrap_ci_pct') if a in vv}) for kk, vv in v.items() if kk not in ('analysis_windows', 'excluded')}
                  for k, v in summary['settings'].items()}, indent=1))
