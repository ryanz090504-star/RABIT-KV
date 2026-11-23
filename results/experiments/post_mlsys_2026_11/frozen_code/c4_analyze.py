"""QSP-R01 C4 pre-specified analysis (stdlib only; fixed before R01 is built). One model per call; the joint decision is
c4_decide(). Runs only on a 'complete' attempt (c4_complete.py); its output is sealed with the scores.

Unit: the window; window score = mean of its 128 FP32 NLL rows (recomputed from scores-<w>.jsonl and required to equal
result.json). A window enters a contrast iff it is valid in BOTH cells of that contrast; nothing is dropped for its loss.

Error control (ALPHA_ONE_SIDED, fixed): every hypothesis is a one-sided test at alpha = 0.025 in each setting. Its decision
bound is the UPPER limit of a two-sided 95 % interval: the 97.5th percentile of the paired percentile bootstrap (the 2.5th is
reported as the lower limit) and mean + t_{0.975, n-1} * sd / sqrt(n). Each hypothesis needs BOTH settings (intersection-union:
size <= 0.025, no adjustment). Q1 and L are separate claims, each at 0.025, so reporting S1 and S3 independently has a
family-wise one-sided error <= 0.025 + 0.025 = 0.05 (Bonferroni). Q2 is tested only after Q1 passes, at the same 0.025
(fixed sequence), so {Q1, Q2} together stay <= 0.025 and {S1, S2, S3} <= 0.05. The bootstrap level is nominal (approximate);
requiring the bootstrap AND the t bound (and the sensitivity bound below) can only lower the rejection rate.

Statistics for a contrast d_w = x_w - y_w (n windows): mean; paired percentile bootstrap over windows (20,000 resamples,
random.Random(seed), linear-interpolated percentiles, as H01 / C3); t interval with the exact Student-t quantile (regularised
incomplete beta, no table); PPL ratio exp(.) and change 100 (exp(.) - 1) on the estimate and both bounds.

Hypotheses:
  Qwen  Q1  kmin - off  : upper < 0               in BOTH settings  (S1: better than the current quantized version)
  Qwen  Q2  kmin - bf16 : upper <= ln(1.03)       in BOTH settings  (S2: within +3 % PPL of BF16; tested only if Q1 passes)
  Llama L   kmin - off  : upper <= ln(1.005)      in BOTH settings  (S3: no regression beyond +0.5 % PPL vs the current version)
Estimates are ALWAYS the complete-case ones: windows valid in both cells of the contrast. No missing, unrun or invalid score is
ever replaced by a value, reported as data, or pooled. A setting passes only if ALL of the following hold (no tie-break):
  (a) n >= 24;  (b) the complete-case bootstrap upper bound meets the criterion;  (c) the complete-case t upper bound meets it;
  (d, e) worst-case SENSITIVITY BOUND (a pass-blocking stress test, not data): the bootstrap and t upper bounds recomputed with
      each excluded window (invalid / error in either cell) represented by the least favourable observed difference max(d)
      must also meet the criterion. It can only remove a pass, never create one or change an estimate.
A window 'not_run' in either cell makes the hypothesis 'not_analysable_not_run' (an attempt with a not_run window is
incomplete under c4_complete.py K3 and is never analysed for a decision; this is a second guard).
Seeds: 20261113 + 10 * hypothesis index + setting index; the sensitivity bound adds 1,000.
Descriptive only (no decision): off - bf16 per setting, kmin - bf16 (Llama), the share of the off - bf16 gap removed
(off - kmin) / (off - bf16) (Qwen), cross-setting contrasts, window counts lower / higher, the base_on vs off bitwise identity,
and engine page allocation (measured cache shapes / blocks; capacity arithmetic only).
argv: <cells dir> <windows json> <model> <out json>"""
import json, math, random, sys
from pathlib import Path

SETTINGS = {'single': {'base': 'base_on_single', 'off': 'off_single', 'kmin': 'kmin_single', 'bf16': 'bf16_single'},
            'two_chunk': {'base': 'base_on_two', 'off': 'off_two', 'kmin': 'kmin_two', 'bf16': 'bf16_two'}}
HYP = {'qwen': {'Q1': ('kmin', 'off', 0.0, 'lt'), 'Q2': ('kmin', 'bf16', math.log(1.03), 'le')},
       'llama': {'L': ('kmin', 'off', math.log(1.005), 'le')}}
HYP_INDEX = {'Q1': 0, 'Q2': 1, 'L': 2}
ALPHA_ONE_SIDED = 0.025                      # per hypothesis and setting (Q1, Q2, L)
RESAMPLES, CONF, SEED, MIN_N, N_WINDOWS = 20000, 1 - 2 * ALPHA_ONE_SIDED, 20261113, 24, 32
assert abs(CONF - 0.95) < 1e-12
ERROR_CONTROL = {'alpha_one_sided_per_hypothesis': {'Q1': ALPHA_ONE_SIDED, 'Q2': ALPHA_ONE_SIDED, 'L': ALPHA_ONE_SIDED},
                 'decision_bound': 'upper limit of the two-sided 95 % interval: bootstrap 97.5th percentile; t quantile 0.975',
                 'settings': 'intersection-union (both must pass), no adjustment',
                 'family': 'S1 (Q1) and S3 (L) at 0.025 each: family-wise <= 0.05 (Bonferroni); Q2 fixed sequence after Q1 at 0.025'}
PAGE = {'qwen': (12416, 12896), 'llama': (24832, 25792)}


# ---------------- Student t quantile (exact; no table) ----------------
def _betacf(a, b, x):
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 500):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)), -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1 + aa * d
            d = 1 / (d if abs(d) > tiny else tiny)
            c = 1 + aa / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1) < 1e-16:
            break
    return h


def _ibeta(a, b, x):
    if x <= 0 or x >= 1:
        return 0.0 if x <= 0 else 1.0
    lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lb) * _betacf(a, b, x) / a
    return 1 - math.exp(lb) * _betacf(b, a, 1 - x) / b


def t_cdf(t, df):
    p = 0.5 * _ibeta(df / 2, 0.5, df / (df + t * t))
    return 1 - p if t > 0 else p


def t_quantile(q, df):
    lo, hi = 0.0, 200.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if t_cdf(mid, df) < q else (lo, mid)
    return (lo + hi) / 2


# ---------------- statistics ----------------
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
    tq = t_quantile(1 - a, n - 1)
    tlo, thi = mean - tq * sd / math.sqrt(n), mean + tq * sd / math.sqrt(n)
    ppl = lambda x: 100 * math.expm1(x)
    return {'n': n, 'mean_nll_diff': mean, 'sd': sd, 'bootstrap_ci': [blo, bhi], 't_quantile': tq, 't_ci': [tlo, thi],
            'ppl_ratio': math.exp(mean), 'ppl_change_pct': ppl(mean), 'ppl_change_bootstrap_ci_pct': [ppl(blo), ppl(bhi)],
            'ppl_change_t_ci_pct': [ppl(tlo), ppl(thi)], 'windows_lower': sum(x < 0 for x in d), 'windows_higher': sum(x > 0 for x in d),
            'windows_equal': sum(x == 0 for x in d), 'bootstrap': {'resamples': RESAMPLES, 'seed': seed, 'method': 'paired percentile over windows'}}


def meets(bound, thr, op):
    return bound < thr if op == 'lt' else bound <= thr


def load(cells_dir, cell):
    p = cells_dir / cell / 'result.json'
    res = json.loads(p.read_text())
    out, shapes = {}, set()
    status = {r['window']: ('valid' if r.get('status') == 'scored' and r.get('valid') else ('invalid' if r.get('status') == 'scored' else r.get('status')))
              for r in res['windows']}
    for r in res['windows']:
        if r.get('status') == 'scored' and r.get('valid'):
            rows = [json.loads(s) for s in (cells_dir / cell / f"scores-{r['window']}.jsonl").read_text().splitlines() if s.strip()]
            assert len(rows) == 128 and [x['nll'] for x in rows] == r['nll'], ('rows differ from result.json', cell, r['window'])
            m = math.fsum(x['nll'] for x in rows) / 128
            assert m == r['mean_nll']
            out[r['window']] = {'mean': m, 'nll': r['nll']}
        for sh in ((r.get('checks') or {}).get('observed') or {}).get('cache_shapes') or []:
            shapes.add(tuple(sh))
    return out, sorted(shapes), status


def test_setting(name, si, A, B, cx, cy, thr, op, order, stx=None, sty=None):
    common = [w for w in order if w in A and w in B]
    excluded = [w for w in order if w not in common]
    reasons = {w: {cx: (stx or {}).get(w, 'missing'), cy: (sty or {}).get(w, 'missing')} for w in excluded}
    not_run = [w for w, r in reasons.items() if 'not_run' in r.values() or 'missing' in r.values()]
    d = [A[w]['mean'] - B[w]['mean'] for w in common]
    seed = SEED + 10 * HYP_INDEX[name] + si
    s = summarise(d, seed)
    rec = {'contrast': f'{cx} - {cy}', 'criterion': f"upper {'<' if op == 'lt' else '<='} {thr!r}",
           'alpha_one_sided': ALPHA_ONE_SIDED, 'interval': 'two-sided 95 %; decision on the upper limit',
           'n': len(common), 'excluded': excluded, 'excluded_reasons': reasons, 'estimate': 'complete case', 'complete_case': s}
    if len(d) >= 2:
        sens = summarise(d + [max(d)] * len(excluded), seed + 1000)
        rec['worst_case_sensitivity_bound'] = {
            'excluded_windows': len(excluded), 'least_favourable_observed_difference': max(d) if excluded else None,
            'bootstrap_upper': sens['bootstrap_ci'][1], 't_upper': sens['t_ci'][1],
            'note': 'pass-blocking stress test only; no excluded score is replaced, reported or used as an estimate'}
        cond = {'n_ge_24': len(common) >= MIN_N, 'no_not_run_window': not not_run,
                'bootstrap_upper': meets(s['bootstrap_ci'][1], thr, op), 't_upper': meets(s['t_ci'][1], thr, op),
                'sensitivity_bootstrap_upper': meets(sens['bootstrap_ci'][1], thr, op), 'sensitivity_t_upper': meets(sens['t_ci'][1], thr, op)}
    else:
        cond = {'n_ge_24': False, 'no_not_run_window': not not_run}
    rec['conditions'] = cond
    rec['pass'] = all(cond.values())
    rec['bootstrap_t_disagree'] = len(d) >= 2 and cond['bootstrap_upper'] != cond['t_upper']
    rec['status'] = ('pass' if rec['pass'] else 'not_analysable_not_run' if not_run else
                     'inconclusive_n_below_24' if len(common) < MIN_N else 'not_met')
    return rec


def analyse(cells_dir, order, model):
    cells = {c: load(cells_dir, c) for st in SETTINGS.values() for c in st.values()}
    sc = {c: {w: v for w, v in cells[c][0].items()} for c in cells}
    stat = {c: cells[c][2] for c in cells}
    out = {'job': 'QSP-R01-C4', 'model': model, 'data': 'R01 (Qwen) / R01-L (Llama); confirmation', 'frozen_order': order,
           'error_control': ERROR_CONTROL,
           'valid_windows': {c: [w for w in order if w in sc[c]] for c in sc}, 'hypotheses': {}, 'descriptive': {}}
    for name, (x, y, thr, op) in HYP[model].items():
        per = {s: test_setting(name, si, sc[st[x]], sc[st[y]], st[x], st[y], thr, op, order, stat[st[x]], stat[st[y]])
               for si, (s, st) in enumerate(SETTINGS.items())}
        sts = [v['status'] for v in per.values()]
        out['hypotheses'][name] = {'settings': per, 'pass': all(v['pass'] for v in per.values()), 'alpha_one_sided': ALPHA_ONE_SIDED,
                                   'status': 'pass' if all(v['pass'] for v in per.values()) else
                                   'not_analysable_not_run' if 'not_analysable_not_run' in sts else
                                   'inconclusive_n_below_24' if 'inconclusive_n_below_24' in sts else 'not_met'}
    seed = SEED + 500
    for s, st in SETTINGS.items():
        dsc = {}
        for lab, (x, y) in {'off_vs_bf16': ('off', 'bf16'), 'kmin_vs_bf16': ('kmin', 'bf16'), 'kmin_vs_off': ('kmin', 'off')}.items():
            A, B = sc[st[x]], sc[st[y]]
            common = [w for w in order if w in A and w in B]
            dsc[lab] = summarise([A[w]['mean'] - B[w]['mean'] for w in common], seed); seed += 1
        tri = [w for w in order if all(w in sc[st[k]] for k in ('off', 'kmin', 'bf16'))]
        if tri:
            mo, mk, mb = (math.fsum(sc[st[k]][w]['mean'] for w in tri) / len(tri) for k in ('off', 'kmin', 'bf16'))
            dsc['mean_nll'] = {'off': mo, 'kmin': mk, 'bf16': mb, 'n': len(tri)}
            dsc['ppl'] = {'off': math.exp(mo), 'kmin': math.exp(mk), 'bf16': math.exp(mb)}
            dsc['share_of_off_bf16_gap_removed'] = (mo - mk) / (mo - mb) if mo != mb else None
        both = [w for w in order if w in sc[st['base']] and w in sc[st['off']]]
        dsc['identity_base_on_vs_off'] = {'compared': len(both), 'bitwise_equal': sum(sc[st['base']][w]['nll'] == sc[st['off']][w]['nll'] for w in both)}
        out['descriptive'][s] = dsc
    X = [w for w in order if all(w in sc[c] for c in ('off_single', 'off_two', 'kmin_single', 'kmin_two'))]
    g = lambda c, w: sc[c][w]['mean']
    out['descriptive']['cross_setting'] = {
        'off_two_vs_off_single': summarise([g('off_two', w) - g('off_single', w) for w in X], seed),
        'kmin_two_vs_kmin_single': summarise([g('kmin_two', w) - g('kmin_single', w) for w in X], seed + 1),
        'two_chunk_excess_change': summarise([(g('kmin_two', w) - g('kmin_single', w)) - (g('off_two', w) - g('off_single', w)) for w in X], seed + 2)}
    pm, pk = PAGE[model]
    alloc = {}
    for c in cells:
        sh = cells[c][1]
        r = {'cache_shapes': [list(x) for x in sh]}
        if len(sh) == 1:
            x = sh[0]
            packed = 'bf16' not in c
            r.update(blocks=x[0], page_dim=x[-1] if packed else None, kv_bytes_per_layer=math.prod(x) * (1 if packed else 2), token_capacity_arith=x[0] * 32)
        alloc[c] = r
    out['descriptive']['allocation'] = {'cells': alloc, 'expected_page_bytes': {'meta8': pm, 'kmin_bf16': pk}, 'page_ratio': pk / pm,
                                        'note': 'measured cache shapes; token capacity is arithmetic (blocks x 32); no throughput or latency claim'}
    return out


def decide(q_complete, q_summary, l_complete, l_summary):
    """Joint C4 decision from both models' completeness verdicts and analyses (fixed before any R01 data)."""
    rec = {'qwen_verdict': (q_complete or {}).get('verdict'), 'llama_verdict': (l_complete or {}).get('verdict')}
    verdicts = (rec['qwen_verdict'], rec['llama_verdict'])
    if 'halted_identity' in verdicts:
        rec['label'] = 'halted_identity_divergence'
        rec['meaning'] = 'the current version and the C2 tree with the switch off differ in the engine; no confirmatory decision; investigate'
        return rec
    if verdicts != ('complete', 'complete'):
        rec['label'] = 'not_decidable_incomplete'
        rec['meaning'] = 'at least one model has no complete attempt (a re-attempt may be allowed under section 5; otherwise C4 is inconclusive)'
        return rec
    q1 = q_summary['hypotheses']['Q1']
    q2 = q_summary['hypotheses']['Q2']
    lh = l_summary['hypotheses']['L']
    s1, s3 = q1['pass'], lh['pass']
    s2 = s1 and q2['pass']
    rec.update(S1_better_than_current_quantized=s1, S2_within_3pct_of_bf16=s2 if s1 else 'not_tested_Q1_not_passed', S3_llama_no_regression_0_5pct=s3,
               Q1_status=q1['status'], Q2_status=q2['status'] if s1 else 'not_tested', L_status=lh['status'])
    if 'not_analysable_not_run' in (q1['status'], lh['status']):
        rec['label'] = 'not_decidable_incomplete'
    elif s1 and s3:
        rec['label'] = 'C4_passed'
    elif q1['status'] == 'inconclusive_n_below_24' or lh['status'] == 'inconclusive_n_below_24':
        rec['label'] = 'C4_inconclusive_n_below_24'
    elif s1 and not s3:
        rec['label'] = 'C4_not_passed_llama_regression_not_excluded'
    else:
        rec['label'] = 'C4_not_passed_qwen_superiority_not_shown'
    rec['never_claimed'] = 'equivalence, restoration or losslessness relative to BF16'
    return rec


if __name__ == '__main__':
    cells_dir, win, model, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
    order = [w['id'] for w in json.loads(win.read_text())['windows']]
    out.write_text(json.dumps(analyse(cells_dir, order, model), indent=2) + '\n')
    print('ANALYSIS_DONE')
