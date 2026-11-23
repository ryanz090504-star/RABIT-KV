"""QSP-R01 C3 pre-specified paired analysis (REGISTRATION-C3.md section 6). Stdlib only. Exploratory windows (H01) only.
Statistics are QSP-H01's (h01_analyze.py): unit = window (mean NLL over its 128 targets, recomputed from the rows); the analysis
set of a setting is the windows valid in all three cells of that setting; no window is dropped for its loss; for each contrast
the window differences d_w are summarised by the mean, a 95% paired percentile bootstrap CI over windows (20,000 resamples,
seed 20261112 + contrast index), a t interval, and PPL change = 100 (exp(.) - 1) on the estimate and each CI endpoint.
Contrasts, per setting (single, two_chunk): kmin - on (primary), kmin - BF16, on - BF16.
Descriptive, per setting: share of the on - BF16 gap removed, (mean(on) - mean(kmin)) / (mean(on) - mean(BF16)), with a
percentile bootstrap CI from the same paired resampling (seed 20261112 + 100 + setting index).
Cross-setting (windows valid in on_single, on_two, kmin_single, kmin_two): on_two - on_single, kmin_two - kmin_single, and
the change in the two-chunk excess (kmin_two - kmin_single) - (on_two - on_single).
Engine page allocation (from the per-window observations): per packed cell, one cache shape over all windows; the page
dimension equals the cell's page size; per setting, the K-min cell's KV bytes per layer (blocks x page bytes) within 0.1 % of
the padding-fix cell's, and fewer blocks; the measured block ratio on / kmin is reported against 12,896 / 12,416.
Decision (fixed before scoring, section 7): 'C4_candidate' only if, in BOTH settings, n >= 16 and the bootstrap CI of
kmin - on lies entirely below 0, and the allocation check passes; 'inconclusive_few_windows' if a setting has n < 16;
otherwise 'stop_before_C4'.
argv: <cells dir> <frozen windows json> <out json>"""
import json, math, random, sys
from pathlib import Path

RUN, WIN, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
SETTINGS = {'single': ('bf16_single', 'on_single', 'kmin_single'), 'two_chunk': ('bf16_two', 'on_two', 'kmin_two')}
PAGE = {'on_single': 12416, 'on_two': 12416, 'kmin_single': 12896, 'kmin_two': 12896}
RESAMPLES, CONF, SEED = 20000, 0.95, 20261112
MIN_N, ALLOC_TOL = 16, 1e-3
T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110,
        2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042, 2.040, 2.037, 2.035, 2.032,
        2.030, 2.028, 2.026, 2.024, 2.023, 2.021]
order = [w['id'] for w in json.loads(WIN.read_text())['windows']]


def load(cell):
    p = RUN / cell / 'result.json'
    res = json.loads(p.read_text()) if p.exists() else {'windows': []}
    out, shapes = {}, set()
    for r in res['windows']:
        if r.get('status') == 'scored' and r.get('valid'):
            rows = [json.loads(s) for s in (RUN / cell / f"scores-{r['window']}.jsonl").read_text().splitlines() if s.strip()]
            assert len(rows) == 128 and [x['nll'] for x in rows] == r['nll']
            m = math.fsum(x['nll'] for x in rows) / 128
            assert m == r['mean_nll']
            out[r['window']] = m
        for sh in ((r.get('checks') or {}).get('observed') or {}).get('cache_shapes') or []:
            shapes.add(tuple(sh))
    status = {r['window']: r.get('status') if r.get('valid') or r.get('status') != 'scored' else 'invalid' for r in res['windows']}
    return out, {w: status.get(w, 'missing') for w in order}, sorted(shapes)


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


def share(on, km, bf, seed):
    """(mean on - mean kmin) / (mean on - mean BF16), with a paired percentile bootstrap CI (resamples with a zero gap skipped)."""
    n = len(on)
    if n < 2:
        return {'n': n}
    f = lambda idx: (lambda g: (math.fsum(on[i] - km[i] for i in idx) / g) if g != 0 else None)(math.fsum(on[i] - bf[i] for i in idx))
    est = f(range(n))
    rng = random.Random(seed)
    boots = [b for b in (f([rng.randrange(n) for _ in range(n)]) for _ in range(RESAMPLES)) if b is not None]
    a = (1 - CONF) / 2
    return {'n': n, 'share': est, 'bootstrap_ci': [pct(boots, a), pct(boots, 1 - a)] if boots else None,
            'resamples_used': len(boots), 'seed': seed, 'descriptive_only': True}


cells = {c: load(c) for s in SETTINGS.values() for c in s}
summary = {'job': 'QSP-R01-C3', 'data': 'H01 windows (exploratory); confirmation set R01 not built or scored',
           'unit': 'window (1024 context + 128 targets)', 'frozen_order': order,
           'cell_status': {c: v[1] for c, v in cells.items()}, 'settings': {}}
seed = SEED
for si, (name, (b, o, k)) in enumerate(SETTINGS.items()):
    A = [w for w in order if all(w in cells[c][0] for c in (b, o, k))]
    m = {c: [cells[c][0][w] for w in A] for c in (b, o, k)}
    s = {'analysis_windows': A, 'n': len(A),
         'excluded': {w: {c: cells[c][1][w] for c in (b, o, k)} for w in order if w not in A},
         'mean_nll': {c: (math.fsum(m[c]) / len(A) if A else None) for c in (b, o, k)}}
    s['ppl'] = {c: (math.exp(v) if v is not None else None) for c, v in s['mean_nll'].items()}
    for label, x, y in (('kmin_vs_on', k, o), ('kmin_vs_bf16', k, b), ('on_vs_bf16', o, b)):
        s[label] = summarise([p - q for p, q in zip(m[x], m[y])], seed)
        seed += 1
    s['share_of_on_bf16_gap_removed'] = share(m[o], m[k], m[b], SEED + 100 + si)
    summary['settings'][name] = s
X = [w for w in order if all(w in cells[c][0] for c in ('on_single', 'on_two', 'kmin_single', 'kmin_two'))]
g = lambda c, w: cells[c][0][w]
summary['cross_setting'] = {
    'windows': X,
    'on_two_vs_on_single': summarise([g('on_two', w) - g('on_single', w) for w in X], seed),
    'kmin_two_vs_kmin_single': summarise([g('kmin_two', w) - g('kmin_single', w) for w in X], seed + 1),
    'two_chunk_excess_change': summarise([(g('kmin_two', w) - g('kmin_single', w)) - (g('on_two', w) - g('on_single', w)) for w in X], seed + 2)}
summary['bf16_single_equals_two_bitwise'] = {w: cells['bf16_single'][0].get(w) == cells['bf16_two'][0].get(w)
                                             for w in order if w in cells['bf16_single'][0] and w in cells['bf16_two'][0]}
# engine page allocation
alloc = {'cells': {}, 'checks': {}}
for c in list(PAGE) + ['bf16_single', 'bf16_two']:
    sh = cells[c][2]
    rec = {'cache_shapes': [list(x) for x in sh]}
    if len(sh) == 1:
        x = sh[0]
        per = math.prod(x) * (1 if c in PAGE else 2)
        rec.update(blocks=x[0], page_dim=x[-1] if c in PAGE else None, kv_bytes_per_layer=per, token_capacity=x[0] * 32)
    alloc['cells'][c] = rec
for c, pb in PAGE.items():
    alloc['checks'][f'{c}_one_shape_page_{pb}'] = len(cells[c][2]) == 1 and cells[c][2][0][-1] == pb
for name, (b, o, k) in SETTINGS.items():
    co, ck = alloc['cells'][o], alloc['cells'][k]
    if 'blocks' in co and 'blocks' in ck:
        rel = ck['kv_bytes_per_layer'] / co['kv_bytes_per_layer'] - 1
        alloc[f'{name}_kv_bytes_rel_kmin_vs_on'] = rel
        alloc[f'{name}_block_ratio_on_over_kmin'] = co['blocks'] / ck['blocks']
        alloc['checks'][f'{name}_kv_bytes_within_0.1pct'] = abs(rel) <= ALLOC_TOL
        alloc['checks'][f'{name}_kmin_fewer_blocks'] = ck['blocks'] < co['blocks']
    else:
        alloc['checks'][f'{name}_kv_bytes_within_0.1pct'] = False
        alloc['checks'][f'{name}_kmin_fewer_blocks'] = False
alloc['expected_block_ratio'] = 12896 / 12416
alloc['passed'] = all(alloc['checks'].values())
summary['engine_page_allocation'] = alloc
# decision
ns = {n: summary['settings'][n]['n'] for n in SETTINGS}
below = {n: (summary['settings'][n]['kmin_vs_on'].get('bootstrap_ci') or [0, 0])[1] < 0 for n in SETTINGS}
if any(v < MIN_N for v in ns.values()):
    decision = 'inconclusive_few_windows'
elif all(below.values()) and alloc['passed']:
    decision = 'C4_candidate'
else:
    decision = 'stop_before_C4'
summary['decision'] = {'value': decision, 'n': ns, 'kmin_vs_on_ci_below_zero': below, 'allocation_passed': alloc['passed'],
                       'rule': 'C4_candidate iff both settings n >= 16 and kmin-on bootstrap CI upper < 0 and allocation passed'}
OUT.write_text(json.dumps(summary, indent=2))
print(json.dumps({'decision': summary['decision'], 'allocation': alloc['checks'],
                  **{k: {kk: (vv if not isinstance(vv, dict) else {a: vv[a] for a in ('n', 'mean_nll_diff', 'bootstrap_ci', 'ppl_change_pct',
                         'ppl_change_bootstrap_ci_pct', 'share') if a in vv}) for kk, vv in v.items() if kk not in ('analysis_windows', 'excluded')}
                     for k, v in summary['settings'].items()}}, indent=1))
