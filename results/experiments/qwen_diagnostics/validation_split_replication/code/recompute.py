"""H001R post-run reanalysis (written 2026-10-24 after the run; NOT part of the frozen record; changes no gate or threshold).

1. Recomputes every reported effect from the preserved per-token rows with an implementation separate from
   prereg_analysis.py: pure Python (math.fsum means, own percentile, own decision rules). numpy is used only to
   draw the registered bootstrap index matrix (np.random.default_rng(seed).integers), which the frozen analysis
   defines; nothing else is imported from the frozen tools.
2. Lists every field where the Pod's audit.json differs from the local re-run of the frozen audit, with ULP distances.
3. Tests the numerical cause: each differing field's input is recomputed and passed through a correctly rounded
   (50-digit Decimal) expm1, to see which platform's libm result is the correctly rounded one.
Retrieved files are opened read-only; output goes to argv[1] (must not exist).
argv: <output dir>"""
import hashlib, json, math, struct, sys
from decimal import Decimal, getcontext
from pathlib import Path
import numpy as np

getcontext().prec = 50
H = Path(__file__).resolve().parents[1]
RUN = H / 'results/b3l0x0iilue1ic/extracted/RUNPOD-QWEN-H001R/run-gpu'
LOCAL = H / 'results/b3l0x0iilue1ic/independent-audit/run-gpu-copy/audit.json'
out = Path(sys.argv[1]).resolve(); out.mkdir(parents=True, exist_ok=False)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputs = [RUN / 'audit.json', LOCAL, H / 'image/stage/registration.json'] + [RUN / 'artifacts' / f'{m}-scores.json' for m in ('qwen2_5_7b', 'llama3_1_8b')]
before = {str(p.relative_to(H).as_posix()): sha(p) for p in inputs}
reg = json.loads((H / 'image/stage/registration.json').read_text())
an = reg['analysis']; SEED, B = an['bootstrap_seed'], an['bootstrap_replicates']; ARMS = reg['scientific_arms']
POD, LOC = json.loads((RUN / 'audit.json').read_text()), json.loads(LOCAL.read_text())


def pctl(xs, level):  # two-sided percentile interval, linear interpolation between order statistics
    s = sorted(xs); n = len(s); out_ = []
    for q in ((1 - level) / 2, 1 - (1 - level) / 2):
        h = (n - 1) * q; lo = math.floor(h); f = h - lo
        out_.append(s[lo] if f == 0 or lo + 1 >= n else s[lo] + f * (s[lo + 1] - s[lo]))
    return out_


def expm1_cr(x):  # correctly rounded expm1 via 50-digit Decimal
    return float(Decimal(x).exp() - 1)


pct = lambda v: 100 * math.expm1(v)
res = {'note': 'post-run reanalysis; not frozen; no threshold, gate or archived file changed', 'inputs_sha256': before, 'models': {}}
for m in ('qwen2_5_7b', 'llama3_1_8b'):
    rows = json.loads((RUN / 'artifacts' / f'{m}-scores.json').read_text())
    w = {a: [math.fsum(r['token_nll']) / 128 for r in rows[a]] for a in ARMS}
    n = 32; assert all(len(v) == n and [r['window'] for r in rows[a]] == list(range(1, 33)) for a, v in w.items())
    ix = np.random.default_rng(SEED).integers(0, n, size=(B, n)).tolist()
    boot = lambda d: [math.fsum(d[i] for i in row) / n for row in ix]
    con, frac, raw = {}, {}, {}
    for a, b in [('range_real', 'rabit'), ('range_real', 'bf16'), ('rabit', 'bf16'), ('k_exact', 'range_real'), ('closed_exact', 'range_real'),
                 ('open_exact', 'range_real'), ('k_exact', 'bf16'), ('closed_exact', 'bf16'), ('open_exact', 'bf16')]:
        d = [x - y for x, y in zip(w[a], w[b])]; bd = boot(d); md = math.fsum(d) / n
        q975, q95 = pctl(bd, 0.975), pctl(bd, 0.95)
        raw[f'{a}_vs_{b}'] = {'mean_delta_nll': md, 'q975': q975, 'q95': q95}
        con[f'{a}_vs_{b}'] = {'mean_delta_nll': md, 'ppl_change_pct': pct(md), 'ci975_pct': [pct(v) for v in q975], 'ci95_pct': [pct(v) for v in q95],
                              'windows_better': sum(x < 0 for x in d), 'windows_worse': sum(x > 0 for x in d)}
    for name, (na, nb, da, db) in {'padding_exclusion_of_frozen_excess': ('rabit', 'range_real', 'rabit', 'bf16'),
                                   'closed_exact_of_range_vs_exactK_gap': ('range_real', 'closed_exact', 'range_real', 'k_exact'),
                                   'open_exact_of_range_vs_exactK_gap': ('range_real', 'open_exact', 'range_real', 'k_exact')}.items():
        num = [x - y for x, y in zip(w[na], w[nb])]; den = [x - y for x, y in zip(w[da], w[db])]
        bn, bd = boot(num), boot(den); r = [p / q for p, q in zip(bn, bd) if abs(q) > 1e-12]
        frac[name] = {'point': math.fsum(num) / math.fsum(den), 'ci975': pctl(r, 0.975), 'ci95': pctl(r, 0.95),
                      'denominator_ci975': pctl(bd, 0.975), 'denominator_crosses_zero': min(bd) <= 0 <= max(bd)}
    if m == 'qwen2_5_7b':
        c, f = con, frac
        h1 = {'range_real_lower_than_rabit': c['range_real_vs_rabit']['ci975_pct'][1] < 0,
              'padding_fraction_lower_bound_ge_0.80': f['padding_exclusion_of_frozen_excess']['ci975'][0] >= 0.80}
        h2 = {'exactK_gap_exists': c['k_exact_vs_range_real']['ci975_pct'][1] < 0,
              'closed_exact_lower_than_range_real': c['closed_exact_vs_range_real']['ci975_pct'][1] < 0,
              'closed_fraction_lower_bound_ge_0.75': f['closed_exact_of_range_vs_exactK_gap']['ci975'][0] >= 0.75,
              'open_fraction_upper_bound_le_0.25': f['open_exact_of_range_vs_exactK_gap']['ci975'][1] <= 0.25}
        dec = {'H1_padding_exclusion': {'conditions': h1, 'criteria_met': all(h1.values())},
               'H2_closed_page_localization': {'conditions': h2, 'criteria_met': all(h2.values())}}
        margins = {'H1 range_real_vs_rabit ci975 upper (pct) < 0': -c['range_real_vs_rabit']['ci975_pct'][1],
                   'H1 padding fraction ci975 lower - 0.80': f['padding_exclusion_of_frozen_excess']['ci975'][0] - 0.80,
                   'H2 k_exact_vs_range_real ci975 upper (pct) < 0': -c['k_exact_vs_range_real']['ci975_pct'][1],
                   'H2 closed_exact_vs_range_real ci975 upper (pct) < 0': -c['closed_exact_vs_range_real']['ci975_pct'][1],
                   'H2 closed fraction ci975 lower - 0.75': f['closed_exact_of_range_vs_exactK_gap']['ci975'][0] - 0.75,
                   'H2 0.25 - open fraction ci975 upper': 0.25 - f['open_exact_of_range_vs_exactK_gap']['ci975'][1]}
    else:
        s1 = {'range_real_vs_rabit_ci95_upper_le_+0.5pct': con['range_real_vs_rabit']['ci95_pct'][1] <= 0.5}
        dec = {'S1_padding_exclusion_noninferior': {'conditions': s1, 'met': all(s1.values()), 'primary_replication_criterion': False}}
        margins = {'S1 0.5 - range_real_vs_rabit ci95 upper (pct)': 0.5 - con['range_real_vs_rabit']['ci95_pct'][1]}
    mine = {'contrasts': con, 'fractions': frac, 'decisions': dec}
    # Compare with both reports: max |difference| over every numeric leaf, and decision equality.
    cmp = {}
    for label, rep in (('pod', POD), ('local', LOC)):
        ra = rep['models'][m]['analysis']; worst = [0.0, 0.0, '']
        for grp in ('contrasts', 'fractions'):
            for k, v in mine[grp].items():
                for f_, x in v.items():
                    y = ra[grp][k][f_]
                    for xx, yy in (zip(x, y) if isinstance(x, list) else [(x, y)]):
                        if isinstance(xx, bool) or isinstance(xx, int):
                            assert xx == yy, (m, label, k, f_)
                        else:
                            ad = abs(xx - yy); rd = ad / max(abs(yy), 1e-300)
                            if ad > worst[0]: worst = [ad, rd, f'{grp}/{k}/{f_}']
        cmp[label] = {'max_abs_diff': worst[0], 'rel_at_max': worst[1], 'field_at_max': worst[2],
                      'decisions_equal': ra['decisions'] == dec, 'valid_for_replication_decisions': rep['models'][m]['run_valid_for_replication_decisions']}
    res['models'][m] = {'recomputed': mine, 'decision_margins': margins, 'min_margin': min(margins.values()), 'vs_reports': cmp, '_raw': raw}

# 2. Exact field-level differences between the Pod's and the local frozen-audit output.
ulp = lambda a, b: abs(struct.unpack('<q', struct.pack('<d', a))[0] - struct.unpack('<q', struct.pack('<d', b))[0])
diffs, n_leaves = [], [0]


def walk(a, b, p):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), p
        for k in a: walk(a[k], b[k], f'{p}/{k}')
    elif isinstance(a, list):
        assert len(a) == len(b), p
        for i, (x, y) in enumerate(zip(a, b)): walk(x, y, f'{p}[{i}]')
    else:
        n_leaves[0] += 1
        if not (a == b and type(a) == type(b)):
            diffs.append({'path': p, 'pod': a, 'local': b, 'abs': abs(a - b), 'ulp': ulp(a, b)})


walk(POD, LOC, '')
# 3. Cause: every differing field is 100*expm1(x) of an x that both reports share; which side is correctly rounded?
for d in diffs:
    parts = d['path'].split('/'); m, k, fld = parts[2], parts[5], parts[6]
    r = res['models'][m]['_raw'][k]
    if fld == 'ppl_change_pct':
        x = r['mean_delta_nll']; x_pod = POD['models'][m]['analysis']['contrasts'][k]['mean_delta_nll']; x_loc = LOC['models'][m]['analysis']['contrasts'][k]['mean_delta_nll']
    else:
        i = int(fld.split('[')[1][0]); x = r['q975' if fld.startswith('ci975') else 'q95'][i]; x_pod = x_loc = None
    cr = 100 * expm1_cr(x)
    d.update({'input_x_recomputed': x, 'input_x_equal_in_both_reports': None if x_pod is None else (x_pod == x_loc == x),
              'local_equals_100*math.expm1(x)_here': d['local'] == pct(x), '100*expm1_correctly_rounded(x)': cr,
              'pod_equals_cr': d['pod'] == cr, 'local_equals_cr': d['local'] == cr})
summary = {'leaves_compared': n_leaves[0], 'leaves_differing': len(diffs),
           'differing_field_kinds': sorted({d['path'].split('/')[-1].split('[')[0] for d in diffs}),
           'max_abs': max(d['abs'] for d in diffs), 'max_ulp': max(d['ulp'] for d in diffs),
           'all_differences_reproduced_from_shared_input_x': all(d['local_equals_100*math.expm1(x)_here'] for d in diffs),
           'pod_matches_correctly_rounded': sum(d['pod_equals_cr'] for d in diffs), 'local_matches_correctly_rounded': sum(d['local_equals_cr'] for d in diffs),
           'non_pct_fields_differing': [d['path'] for d in diffs if not d['path'].split('/')[-1].startswith(('ppl_change_pct', 'ci975_pct', 'ci95_pct'))]}
res['audit_json_differences'] = {'summary': summary, 'fields': diffs}
for v in res['models'].values(): v.pop('_raw')
after = {str(p.relative_to(H).as_posix()): sha(p) for p in inputs}
res['inputs_unmodified'] = before == after
(out / 'reanalysis.json').write_text(json.dumps(res, indent=2, allow_nan=False), newline='\n')
print(json.dumps({'summary': summary, 'inputs_unmodified': res['inputs_unmodified'],
                  'models': {m: {'vs_reports': v['vs_reports'], 'min_margin': v['min_margin'], 'decisions': v['recomputed']['decisions']} for m, v in res['models'].items()}}, indent=1))
