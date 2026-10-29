"""RUNPOD-QWEN-A001 pre-specified descriptive analysis (numpy only). Frozen before any new-arm window is scored.

Input: per-window mean token NLL for each scientific arm (equal-length arrays, same window order).
All contrasts share ONE paired-window bootstrap index matrix (seed and replicates from the registration).
Intervals are two-sided 95% percentile intervals. There is no decision rule: every number is reported.
"""
import math
import numpy as np

LEVEL = 0.95
CONTRASTS = [('meta_exact', 'rabit'), ('range_meta_exact', 'range_real'), ('range_real', 'rabit'), ('rabit', 'bf16'),
             ('range_real', 'bf16'), ('meta_exact', 'bf16'), ('range_meta_exact', 'bf16')]


def analyze(means, seed, replicates):
    arms = list(means)
    n = len(means[arms[0]])
    assert n == 32 and all(len(means[a]) == n for a in arms)
    m = {a: np.asarray(means[a], dtype=np.float64) for a in arms}
    ix = np.random.default_rng(seed).integers(0, n, size=(replicates, n))
    pct = lambda v: 100 * math.expm1(v)
    lo, hi = (1 - LEVEL) / 2, 1 - (1 - LEVEL) / 2
    out = {'windows': n, 'ppl': {a: math.exp(float(m[a].mean())) for a in arms}, 'contrasts': {}}
    for a, b in CONTRASTS:
        d = m[a] - m[b]
        boot = d[ix].mean(1)
        q = [float(v) for v in np.quantile(boot, [lo, hi])]
        out['contrasts'][f'{a}_vs_{b}'] = {
            'mean_delta_nll': float(d.mean()), 'ppl_change_pct': pct(float(d.mean())),
            'ci95_pct': [pct(v) for v in q], 'windows_better': int((d < 0).sum()), 'windows_worse': int((d > 0).sum()),
            'windows_equal': int((d == 0).sum())}
    # share of the frozen policy's excess over BF16 that META8 accounts for (descriptive; reported, not interpreted
    # when the denominator interval crosses zero)
    for name, num, den in [('meta8_share_of_frozen_excess', ('rabit', 'meta_exact'), ('rabit', 'bf16')),
                           ('meta8_share_of_padding_excluded_excess', ('range_real', 'range_meta_exact'), ('range_real', 'bf16'))]:
        nu, de = m[num[0]] - m[num[1]], m[den[0]] - m[den[1]]
        bn, bd = nu[ix].mean(1), de[ix].mean(1)
        ok = np.abs(bd) > 1e-12
        r = bn[ok] / bd[ok]
        out[name] = {'point': float(nu.mean() / de.mean()), 'ci95': [float(v) for v in np.quantile(r, [lo, hi])],
                     'denominator_ci95_nll': [float(v) for v in np.quantile(bd, [lo, hi])],
                     'denominator_crosses_zero': bool(np.quantile(bd, lo) <= 0 <= np.quantile(bd, hi))}
    return out
