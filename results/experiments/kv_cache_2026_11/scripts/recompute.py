"""Recompute every number of the RABIT-KV research update on KV-cache quantization (docs/research/KV_CACHE_UPDATE_QWEN_LLAMA.md) from the two public
tables, and compare with the frozen analysis outputs in ../frozen/. No GPU, no model, no network.

  tables/serving_window_nll.csv  one row per (study, model, window, cell): the window's mean NLL over its 128 scored targets
  tables/h1_layer_metrics.csv    one row per (model, window, layer): the H1 attention-score error metric

Serving studies (statistics re-implemented from the frozen code in ../frozen_code/, same seeds and window order):
  h01    Qwen, 32 exploratory windows: released cache with zero padding (off) vs padding fix (on) vs BF16   (h01_analyze.py)
  c3_a2  Qwen, the same 32 exploratory windows: padding fix vs padding fix + BF16 K minima vs BF16       (c3_analyze.py)
  c4     Qwen R01 and Llama R01-L, 32 fresh windows each: confirmatory test of BF16 K minima            (c4_analyze.py)
H1 (exploratory, BF16 replay): per-window layer summaries of dE and their Spearman association with the C4 gains
  (analyze_h1.py), and the layer shares with explicit denominators.

The serving part uses only the Python standard library. The H1 part needs numpy (the registered analysis used numpy).
Usage:  python recompute.py [--out recomputed.json]      exit status 0 iff every comparison with the frozen outputs passes."""
import csv, json, math, random, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
FROZEN = HERE / 'frozen'
OUT = Path(sys.argv[sys.argv.index('--out') + 1]) if '--out' in sys.argv else None
TOL = 1e-12
T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110,
        2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042, 2.040, 2.037, 2.035, 2.032,
        2.030, 2.028, 2.026, 2.024, 2.023, 2.021]                         # h01 / c3: tabulated t quantiles
checks = []


def check(label, got, want, tol=TOL):
    if isinstance(want, list) and want and isinstance(want[0], str):
        ok = got == want
    elif isinstance(want, list):
        ok = len(got) == len(want) and all(abs(a - b) <= tol * max(1.0, abs(b)) for a, b in zip(got, want))
    elif isinstance(want, (bool, str)) or want is None:
        ok = got == want
    else:
        ok = abs(got - want) <= tol * max(1.0, abs(want))
    checks.append({'check': label, 'ok': ok, **({} if ok else {'got': got, 'want': want})})
    return ok


# ---------------- statistics (as in the frozen code) ----------------
def pct(xs, q):
    s = sorted(xs); h = (len(s) - 1) * q; lo = math.floor(h)
    return s[lo] + (h - lo) * (s[min(lo + 1, len(s) - 1)] - s[lo])


def _betacf(a, b, x):
    tiny = 1e-300; qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap; d = 1 / (d if abs(d) > tiny else tiny); h = d
    for m in range(1, 500):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)), -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1 + aa * d; d = 1 / (d if abs(d) > tiny else tiny)
            c = 1 + aa / c; c = c if abs(c) > tiny else tiny; h *= d * c
        if abs(d * c - 1) < 1e-16:
            break
    return h


def _ibeta(a, b, x):
    if x <= 0 or x >= 1:
        return 0.0 if x <= 0 else 1.0
    lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    return math.exp(lb) * _betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - math.exp(lb) * _betacf(b, a, 1 - x) / b


def t_quantile(q, df):                                                  # c4: exact Student t quantile
    t_cdf = lambda t: (lambda p: 1 - p if t > 0 else p)(0.5 * _ibeta(df / 2, 0.5, df / (df + t * t)))
    lo, hi = 0.0, 200.0
    for _ in range(200):
        mid = (lo + hi) / 2; lo, hi = (mid, hi) if t_cdf(mid) < q else (lo, mid)
    return (lo + hi) / 2


def summarise(d, seed, exact_t):
    n = len(d); mean = math.fsum(d) / n
    sd = math.sqrt(math.fsum((x - mean) ** 2 for x in d) / (n - 1))
    rng = random.Random(seed)
    boots = [math.fsum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(20000)]
    blo, bhi = pct(boots, 0.025), pct(boots, 0.975)
    tq = t_quantile(0.975, n - 1) if exact_t else (T975[n - 2] if n - 1 <= len(T975) else 1.96)
    tlo, thi = mean - tq * sd / math.sqrt(n), mean + tq * sd / math.sqrt(n)
    ppl = lambda x: 100 * math.expm1(x)
    return {'n': n, 'mean_nll_diff': mean, 'bootstrap_ci': [blo, bhi], 't_ci': [tlo, thi], 'ppl_change_pct': ppl(mean),
            'ppl_change_bootstrap_ci_pct': [ppl(blo), ppl(bhi)], 'ppl_change_t_ci_pct': [ppl(tlo), ppl(thi)],
            'windows_lower': sum(x < 0 for x in d), 'windows_higher': sum(x > 0 for x in d)}


def compare(label, s, f):
    for k in ('mean_nll_diff', 'bootstrap_ci', 't_ci'):
        check(f'{label}.{k}', s[k], f[k])
    check(f'{label}.windows_higher', s['windows_higher'], f['windows_higher'])


# ---------------- load the serving table ----------------
T = {}
for r in csv.DictReader((HERE / 'tables' / 'serving_window_nll.csv').open()):
    T.setdefault((r['study'], r['model']), {}).setdefault(r['cell'], {})[r['window_id']] = (int(r['order_index']), float(r['mean_nll']))
order = lambda key: [w for w, _ in sorted(((w, v[0]) for w, v in next(iter(T[key].values())).items()), key=lambda t: t[1])]
val = lambda key, cell, ws: [T[key][cell][w][1] for w in ws]
res = {}

# h01: off (released, zero padding) / on (padding fix) / bf16
fz = json.loads((FROZEN / 'h01-summary.json').read_text()); key = ('h01', 'qwen'); ws = order(key); check('h01.order', ws, fz['frozen_order']); seed = 20261110
res['h01'] = {}
for name, (b, f, o) in {'single': ('bf16_single', 'off_single', 'on_single'), 'two_chunk': ('bf16_two', 'off_two', 'on_two')}.items():
    m = {c: val(key, c, ws) for c in (b, f, o)}; res['h01'][name] = {'ppl': {c: math.exp(math.fsum(m[c]) / 32) for c in m}}
    for c in m:
        check(f'h01.{name}.ppl.{c}', res['h01'][name]['ppl'][c], fz['settings'][name]['ppl'][c])
    for lab, x, y in (('on_vs_off', o, f), ('on_vs_bf16', o, b), ('off_vs_bf16', f, b)):
        s = summarise([p - q for p, q in zip(m[x], m[y])], seed, False); seed += 1
        compare(f'h01.{name}.{lab}', s, fz['settings'][name][lab]); res['h01'][name][lab] = s

# c3_a2: on (padding fix) / kmin (padding fix + BF16 K minima) / bf16; exploratory, same windows as h01
fz = json.loads((FROZEN / 'c3-a2-summary.json').read_text()); key = ('c3_a2', 'qwen'); ws = order(key); check('c3.order', ws, fz['frozen_order']); seed = 20261112
res['c3_a2'] = {}
for si, (name, (b, o, k)) in enumerate({'single': ('bf16_single', 'on_single', 'kmin_single'), 'two_chunk': ('bf16_two', 'on_two', 'kmin_two')}.items()):
    m = {c: val(key, c, ws) for c in (b, o, k)}; res['c3_a2'][name] = {'ppl': {c: math.exp(math.fsum(m[c]) / 32) for c in m}}
    for lab, x, y in (('kmin_vs_on', k, o), ('kmin_vs_bf16', k, b), ('on_vs_bf16', o, b)):
        s = summarise([p - q for p, q in zip(m[x], m[y])], seed, False); seed += 1
        compare(f'c3.{name}.{lab}', s, fz['settings'][name][lab]); res['c3_a2'][name][lab] = s
    gap = (math.fsum(m[o]) - math.fsum(m[k])) / (math.fsum(m[o]) - math.fsum(m[b]))
    check(f'c3.{name}.share_point', gap, fz['settings'][name]['share_of_on_bf16_gap_removed']['share'], 1e-9); res['c3_a2'][name]['share_of_gap_removed'] = gap

# c4: confirmatory; hypotheses Q1, Q2 (Qwen) and L (Llama); exact t; both settings must pass
HYP = {'qwen': {'Q1': ('kmin', 'off', 0.0, 'lt'), 'Q2': ('kmin', 'bf16', math.log(1.03), 'le')}, 'llama': {'L': ('kmin', 'off', math.log(1.005), 'le')}}
HI = {'Q1': 0, 'Q2': 1, 'L': 2}
ST = {'single': {'base': 'base_on_single', 'off': 'off_single', 'kmin': 'kmin_single', 'bf16': 'bf16_single'},
      'two_chunk': {'base': 'base_on_two', 'off': 'off_two', 'kmin': 'kmin_two', 'bf16': 'bf16_two'}}
res['c4'] = {}
for model in ('qwen', 'llama'):
    fz = json.loads((FROZEN / f'c4-summary-{model}.json').read_text()); key = ('c4', model); ws = order(key); check(f'c4.{model}.order', ws, fz['frozen_order'])
    r = res['c4'][model] = {'hypotheses': {}, 'descriptive': {}}
    for h, (x, y, thr, op) in HYP[model].items():
        per = {}
        for si, (sname, st) in enumerate(ST.items()):
            d = [a - b for a, b in zip(val(key, st[x], ws), val(key, st[y], ws))]
            s = summarise(d, 20261113 + 10 * HI[h] + si, True)
            meets = (lambda u: u < thr) if op == 'lt' else (lambda u: u <= thr)
            s['pass'] = len(d) >= 24 and meets(s['bootstrap_ci'][1]) and meets(s['t_ci'][1])     # no window was excluded in C4 (n = 32)
            f = fz['hypotheses'][h]['settings'][sname]
            compare(f'c4.{model}.{h}.{sname}', s, f['complete_case']); check(f'c4.{model}.{h}.{sname}.pass', s['pass'], f['pass'])
            per[sname] = s
        r['hypotheses'][h] = {'settings': per, 'pass': all(v['pass'] for v in per.values())}
        check(f'c4.{model}.{h}.pass', r['hypotheses'][h]['pass'], fz['hypotheses'][h]['pass'])
    seed = 20261113 + 500
    for sname, st in ST.items():
        for lab, (x, y) in {'off_vs_bf16': ('off', 'bf16'), 'kmin_vs_bf16': ('kmin', 'bf16'), 'kmin_vs_off': ('kmin', 'off')}.items():
            s = summarise([a - b for a, b in zip(val(key, st[x], ws), val(key, st[y], ws))], seed, True); seed += 1
            compare(f'c4.{model}.desc.{sname}.{lab}', s, fz['descriptive'][sname][lab]); r['descriptive'].setdefault(sname, {})[lab] = s
        mo, mk, mb = (math.fsum(val(key, st[c], ws)) / 32 for c in ('off', 'kmin', 'bf16'))
        r['descriptive'][sname]['ppl'] = {'off': math.exp(mo), 'kmin': math.exp(mk), 'bf16': math.exp(mb)}
        r['descriptive'][sname]['share_of_off_bf16_gap_removed'] = (mo - mk) / (mo - mb)
        check(f'c4.{model}.desc.{sname}.share', r['descriptive'][sname]['share_of_off_bf16_gap_removed'], fz['descriptive'][sname]['share_of_off_bf16_gap_removed'], 1e-9)
        same = sum(T[key][st['base']][w][1] == T[key][st['off']][w][1] for w in ws)
        check(f'c4.{model}.{sname}.base_on_equals_off_window_means', same, 32); r['descriptive'][sname]['base_on_equals_off_windows'] = same
q, l = res['c4']['qwen']['hypotheses'], res['c4']['llama']['hypotheses']
res['c4']['decision'] = {'S1_qwen_kmin_better_than_padding_fix': q['Q1']['pass'], 'S2_qwen_kmin_within_3pct_of_bf16': q['Q1']['pass'] and q['Q2']['pass'],
                         'S3_llama_no_regression_beyond_0_5pct': l['L']['pass'],
                         'label': 'C4_passed' if q['Q1']['pass'] and l['L']['pass'] else ('C4_not_passed_llama_regression_not_excluded' if q['Q1']['pass'] else 'C4_not_passed_qwen_superiority_not_shown')}
check('c4.decision_label', res['c4']['decision']['label'], 'C4_not_passed_llama_regression_not_excluded')

# ---------------- H1 (needs numpy) ----------------
try:
    import numpy as np
except ImportError:
    np = None
    checks.append({'check': 'h1.skipped_numpy_missing', 'ok': False})
if np is not None:
    def rank(x):
        x = np.asarray(x, float); o = np.argsort(x, kind='mergesort'); r_ = np.empty(len(x)); r_[o] = np.arange(len(x))
        for v in np.unique(x):
            idx = np.where(x == v)[0]
            if len(idx) > 1:
                r_[idx] = r_[idx].mean()
        return r_

    def spearman(a, b):
        ra, rb = rank(a), rank(b)
        return float('nan') if ra.std() == 0 or rb.std() == 0 else float(np.corrcoef(ra, rb)[0, 1])

    def boot_rho(a, b, seed, B=20000):
        a, b = np.asarray(a, float), np.asarray(b, float); n = len(a); rng = random.Random(seed); v = []
        for _ in range(B):
            idx = [rng.randrange(n) for _ in range(n)]; s = spearman(a[idx], b[idx])
            if not math.isnan(s):
                v.append(s)
        v.sort(); return [v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]]

    def boot_median(x, seed, B=20000):
        x = np.asarray(x, float); n = len(x); rng = random.Random(seed)
        v = sorted(float(np.median(x[[rng.randrange(n) for _ in range(n)]])) for _ in range(B)); return [v[int(0.025 * B)], v[int(0.975 * B) - 1]]

    fa = json.loads((FROZEN / 'h1-analysis.json').read_text()); fsh = json.loads((FROZEN / 'h1-layer-shares.json').read_text())
    rows = list(csv.DictReader((HERE / 'tables' / 'h1_layer_metrics.csv').open()))
    res['h1'] = {}
    for mi, model in enumerate(('qwen', 'llama')):
        R = [r for r in rows if r['model'] == model]
        wins = sorted({r['window_id'] for r in R}); layers = sorted({int(r['layer']) for r in R}); last = layers[-1]
        dE = {(r['window_id'], int(r['layer'])): float(r['dE']) for r in R}; fr = {(r['window_id'], int(r['layer'])): float(r['min_ratio_frac_gt_half']) for r in R}
        check(f'h1.{model}.grid', len(dE), 32 * len(layers))
        S = {'S_all': {w: float(np.mean([dE[(w, l)] for l in layers])) for w in wins}, 'S_0': {w: dE[(w, 0)] for w in wins},
             'S_rest': {w: float(np.mean([dE[(w, l)] for l in layers if l >= 1])) for w in wins}, 'F_all': {w: float(np.mean([fr[(w, l)] for l in layers])) for w in wins}}
        f = fa['models'][model]
        m = {'S_all_median': float(np.median(list(S['S_all'].values()))), 'S_0_median': float(np.median(list(S['S_0'].values()))),
             'S_rest_median': float(np.median(list(S['S_rest'].values()))), 'S_all_median_boot95': boot_median(list(S['S_all'].values()), 20261118 + 100 + mi),
             'quartiles': {k: [float(np.percentile(list(v.values()), q_)) for q_ in (25, 50, 75)] for k, v in S.items()},
             'per_layer_median_dE': {l: float(np.median([dE[(w, l)] for w in wins])) for l in layers},
             'per_layer_median_rel': {l: float(np.median([float(r['rel']) for r in R if int(r['layer']) == l])) for l in layers},
             'per_layer_median_E': {l: [float(np.median([float(r['E_meta8']) for r in R if int(r['layer']) == l])), float(np.median([float(r['E_bf16']) for r in R if int(r['layer']) == l]))] for l in (0, 1, last)},
             'association': {}}
        for k in ('S_all_median', 'S_0_median', 'S_rest_median', 'S_all_median_boot95'):
            check(f'h1.{model}.{k}', m[k], f[k])
        for l in layers:
            check(f'h1.{model}.layer{l}.median_dE', m['per_layer_median_dE'][l], f['per_layer'][str(l)]['median_dE'])
        for si, st in enumerate(('single', 'two')):
            cs = ST['single' if st == 'single' else 'two_chunk']
            G = {w: T[('c4', model)][cs['off']][w][1] - T[('c4', model)][cs['kmin']][w][1] for w in wins}   # positive = K-min lower NLL
            for ki, lab in enumerate(('S_all', 'S_0', 'S_rest', 'F_all')):
                a = [S[lab][w] for w in wins]; b = [G[w] for w in wins]
                rho = spearman(a, b); ci = boot_rho(a, b, 20261118 + 10 * mi + 3 * si + ki)
                fz_ = f['association'][f'{st}/{lab}']
                check(f'h1.{model}.{st}/{lab}.spearman', rho, fz_['spearman'], 1e-9); check(f'h1.{model}.{st}/{lab}.boot95', ci, fz_['spearman_boot95'], 1e-9)
                m['association'][f'{st}/{lab}'] = {'spearman': rho, 'boot95': ci, 'pearson': float(np.corrcoef(a, b)[0, 1])}
            m.setdefault('gain_off_minus_kmin', {})[st] = {'median': float(np.median(list(G.values()))), 'n_positive': sum(v > 0 for v in G.values())}
        tot = {w: sum(dE[(w, l)] for l in layers) for w in wins}; rest = {w: sum(dE[(w, l)] for l in layers if l >= 1) for w in wins}
        m['shares'] = {'last_of_all_layer_sum': float(np.median([dE[(w, last)] / tot[w] for w in wins])), 'layer0_of_all_layer_sum': float(np.median([dE[(w, 0)] / tot[w] for w in wins])),
                       'last_of_S_rest_sum': float(np.median([dE[(w, last)] / rest[w] for w in wins]))}
        fs = fsh['models'][model]
        check(f'h1.{model}.share_last_all', m['shares']['last_of_all_layer_sum'], fs['share_of_all_layer_sum']['last_layer']['median'], 1e-9)
        check(f'h1.{model}.share_last_rest', m['shares']['last_of_S_rest_sum'], fs['share_of_S_rest_sum']['last_layer']['median'], 1e-9)
        res['h1'][model] = m
    ivs = [x['boot95'] for m in res['h1'].values() for x in m['association'].values()]
    res['h1']['intervals'] = {'registered': len(ivs), 'include_zero': sum(lo <= 0 <= hi for lo, hi in ivs)}
    check('h1.interval_count', [res['h1']['intervals']['registered'], res['h1']['intervals']['include_zero']], [16, 15])

res['checks'] = {'total': len(checks), 'failed': [c for c in checks if not c['ok']]}
if OUT:
    OUT.write_text(json.dumps(res, indent=1, default=str) + '\n')
print('checks', len(checks), 'failed', len(res['checks']['failed']))
for c in res['checks']['failed'][:20]:
    print('FAIL', c)
sys.exit(0 if not res['checks']['failed'] else 1)
