"""HOLDOUT-001 (H001R) pre-specified replication analysis (numpy only). Frozen before any validation-split window is scored.

Input: per-window mean token NLL for each arm (equal-length arrays, same window order).
All quantities share ONE paired-window bootstrap index matrix (seed and replicates from the registration).
Decision intervals are two-sided 97.5% percentile intervals (Bonferroni over the two primary Qwen hypotheses);
95% intervals are reported alongside and are descriptive only.
"""
import math
import numpy as np

DECISION_LEVEL, REPORT_LEVEL = 0.975, 0.95


def _q(x, level):
    a = (1 - level) / 2
    return [float(v) for v in np.quantile(x, [a, 1 - a])]


def analyze(means, seed, replicates, model):
    arms = list(means)
    n = len(means[arms[0]])
    assert n >= 8 and all(len(means[a]) == n for a in arms)
    m = {a: np.asarray(means[a], dtype=np.float64) for a in arms}
    ix = np.random.default_rng(seed).integers(0, n, size=(replicates, n))

    def contrast(a, b):
        d = m[a] - m[b]
        boot = d[ix].mean(1)
        pct = lambda v: 100 * math.expm1(v)
        return {'mean_delta_nll': float(d.mean()), 'ppl_change_pct': pct(float(d.mean())),
                'ci975_pct': [pct(v) for v in _q(boot, DECISION_LEVEL)], 'ci95_pct': [pct(v) for v in _q(boot, REPORT_LEVEL)],
                'windows_better': int((d < 0).sum()), 'windows_worse': int((d > 0).sum())}

    def fraction(num_a, num_b, den_a, den_b):
        num, den = m[num_a] - m[num_b], m[den_a] - m[den_b]
        bn, bd = num[ix].mean(1), den[ix].mean(1)
        ok = np.abs(bd) > 1e-12
        r = bn[ok] / bd[ok]
        return {'point': float(num.mean() / den.mean()), 'ci975': _q(r, DECISION_LEVEL), 'ci95': _q(r, REPORT_LEVEL),
                'denominator_ci975': _q(bd, DECISION_LEVEL), 'denominator_crosses_zero': bool(bd.min() <= 0 <= bd.max())}

    out = {'windows': n, 'ppl': {a: math.exp(float(m[a].mean())) for a in arms}, 'contrasts': {}, 'fractions': {}}
    pairs = [('range_real', 'rabit'), ('range_real', 'bf16'), ('rabit', 'bf16'), ('k_exact', 'range_real'),
             ('closed_exact', 'range_real'), ('open_exact', 'range_real'), ('k_exact', 'bf16'), ('closed_exact', 'bf16'),
             ('open_exact', 'bf16')]
    for a, b in pairs:
        if a in m and b in m:
            out['contrasts'][f'{a}_vs_{b}'] = contrast(a, b)
    if {'rabit', 'range_real', 'bf16'} <= set(m):
        out['fractions']['padding_exclusion_of_frozen_excess'] = fraction('rabit', 'range_real', 'rabit', 'bf16')
    if {'range_real', 'k_exact', 'closed_exact', 'open_exact'} <= set(m):
        out['fractions']['closed_exact_of_range_vs_exactK_gap'] = fraction('range_real', 'closed_exact', 'range_real', 'k_exact')
        out['fractions']['open_exact_of_range_vs_exactK_gap'] = fraction('range_real', 'open_exact', 'range_real', 'k_exact')
    out['decisions'] = decide(out, model) if set(m) >= {'bf16', 'rabit', 'range_real', 'k_exact', 'closed_exact', 'open_exact'} else None
    return out


def decide(r, model):
    c, f = r['contrasts'], r['fractions']
    if model == 'qwen2_5_7b':
        h1 = {'range_real_lower_than_rabit': c['range_real_vs_rabit']['ci975_pct'][1] < 0,
              'padding_fraction_lower_bound_ge_0.80': f['padding_exclusion_of_frozen_excess']['ci975'][0] >= 0.80}
        h2 = {'exactK_gap_exists': c['k_exact_vs_range_real']['ci975_pct'][1] < 0,
              'closed_exact_lower_than_range_real': c['closed_exact_vs_range_real']['ci975_pct'][1] < 0,
              'closed_fraction_lower_bound_ge_0.75': f['closed_exact_of_range_vs_exactK_gap']['ci975'][0] >= 0.75,
              'open_fraction_upper_bound_le_0.25': f['open_exact_of_range_vs_exactK_gap']['ci975'][1] <= 0.25}
        return {'H1_padding_exclusion': {'conditions': h1, 'criteria_met': all(h1.values())},
                'H2_closed_page_localization': {'conditions': h2, 'criteria_met': all(h2.values())}}
    s1 = {'range_real_vs_rabit_ci95_upper_le_+0.5pct': c['range_real_vs_rabit']['ci95_pct'][1] <= 0.5}
    return {'S1_padding_exclusion_noninferior': {'conditions': s1, 'met': all(s1.values()), 'primary_replication_criterion': False}}
