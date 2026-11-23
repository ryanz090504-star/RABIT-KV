"""H1 primary mechanism metric (registered in ../REGISTRATION-H1.md, section 4). Pure torch; the same code runs on CPU (offline tests,
A001 reproduction) and on the GPU in the Pod.

For one layer of one window, given BF16 post-RoPE queries Q [QH, T, D] and keys K [KVH, T, D] of a fixed BF16 forward:
  pages     page j holds stream positions [32 j, 32 j + 32); a page is closed for the query at position p iff j < C(p) with
            C(p) = floor((p + 1 - R) / 32), R = 4 (the 4 newest tokens stay in BF16), i.e. as in the serving runtime
  K3        per (page, KV head, channel) over the page's 32 real tokens, in FP32: mn = min, mx = max,
            step = (mx - mn) * fp32(1/7) (the H100 kernel's scalar division), step = 1 where |step| < 1e-8,
            codes = round_half_even((k - mn) / step) clamped to [0, 7]   (IEEE FP32 division)
  scales    the K3 steps are stored with META8g64 (below) -> step_hat; identical in both arms
  arms      A 'meta8': mn_hat = META8g64(mn)  (the current serving storage of K minima)
            B 'bf16' : mn_hat = BF16(mn)      (the K-min switch; exact, since K is BF16)
            reconstruction k_hat = codes * step_hat + mn_hat (FP32). Inputs, codes, steps and pages are the same in both arms.
  META8g64  the per-page metadata tensor in storage order (KV head, channel), groups of 64 consecutive values: gmin, gmax,
            sc = (gmax - gmin) * fp32(1/255) (sc = 1 where |sc| < 1e-12), c8 = round_half_even((v - gmin) / sc) clamped [0, 255],
            decoded = c8 * BF16(sc) + BF16(gmin) (FP32)
  error     for query position p, query head h (KV head g = h // (QH / KVH)), closed-page token n:
            e_arm(p, h, n) = q[h, p] . (k_hat_arm[g, n] - k[g, n]) / sqrt(D)   (FP64)
            E_arm = sqrt(mean of e_arm^2 over all (p in P, h, n closed at p))
  primary   dE = E_meta8 - E_bf16 (attention-score RMS error reduction, logit units); secondary rel = dE / E_meta8 and
            dMSE = E_meta8^2 - E_bf16^2
  minima    over every closed page at the last query (pages 0 .. C(max P) - 1), every KV head and channel:
            ratio = |META8(mn) - mn| / step (FP64); frac_gt_half = fraction with ratio > 0.5; median = torch.median (lower
            median), p90 / p99 = torch.quantile (linear), max
"""
import math
import torch

G, R, MG = 32, 4, 64
DIV_RULE = 'gpu'   # registered: 'gpu' = x * fp32(1/7), x * fp32(1/255) (the H100 serving rule); 'ieee' only for test T1's exact reproduction
INV7 = torch.tensor(1.0, dtype=torch.float32) / torch.tensor(7.0, dtype=torch.float32)
INV255 = torch.tensor(1.0, dtype=torch.float32) / torch.tensor(255.0, dtype=torch.float32)


def closed_pages(p):
    return (p + 1 - R) // G


def meta8(v):
    """META8g64 round trip of a per-page metadata tensor [..., KVH, D] (storage order = last two dims flattened)."""
    shp = v.shape
    flat = v.reshape(*shp[:-2], -1).float()
    n = flat.shape[-1]
    assert n % MG == 0
    g = flat.reshape(*shp[:-2], n // MG, MG)
    gmin, gmax = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    sc = (gmax - gmin) * INV255.to(g.device) if DIV_RULE == 'gpu' else (gmax - gmin) / 255.0
    sc = torch.where(sc.abs() < 1e-12, torch.ones_like(sc), sc)
    c8 = torch.round((g - gmin) / sc).clamp(0, 255)
    out = c8 * sc.to(torch.bfloat16).float() + gmin.to(torch.bfloat16).float()
    return out.reshape(shp)


def k3_pages(k, npages):
    """k [KVH, T, D] BF16 -> dict of FP32 tensors per page [npages, KVH, D] (mn, step) and codes [npages, KVH, 32, D]."""
    x = k[:, :npages * G, :].float().reshape(k.shape[0], npages, G, k.shape[2]).permute(1, 0, 2, 3)   # [P, KVH, 32, D]
    mn, mx = x.amin(2), x.amax(2)
    step = (mx - mn) * INV7.to(x.device) if DIV_RULE == 'gpu' else (mx - mn) / 7.0
    step = torch.where(step.abs() < 1e-8, torch.ones_like(step), step)
    codes = torch.round((x - mn.unsqueeze(2)) / step.unsqueeze(2)).clamp(0, 7)
    return {'x': x, 'mn': mn, 'step': step, 'codes': codes}


def reconstruct(pg, arm):
    step_hat = meta8(pg['step'])
    mn_hat = meta8(pg['mn']) if arm == 'meta8' else pg['mn'].to(torch.bfloat16).float()
    return pg['codes'] * step_hat.unsqueeze(2) + mn_hat.unsqueeze(2)          # [P, KVH, 32, D]


def layer_metric(q, k, positions):
    """q [QH, T, D], k [KVH, T, D] (BF16, any device); positions: ascending query positions. Returns a dict of floats."""
    QH, D = q.shape[0], q.shape[2]
    KVH = k.shape[0]
    grp = QH // KVH
    pmax = max(positions)
    npages = closed_pages(pmax)
    pg = k3_pages(k, npages)
    ntok = npages * G
    kx = pg['x'].permute(1, 0, 2, 3).reshape(KVH, ntok, D).double()
    qp = q[:, positions, :].double()                                          # [QH, P, D]
    pos = torch.tensor(positions, device=k.device)
    mask = (torch.arange(ntok, device=k.device)[None, :] < (closed_pages(pos) * G)[:, None]).double()   # [P, ntok]
    count = mask.sum() * QH
    res = {}
    for arm in ('meta8', 'bf16'):
        kh = reconstruct(pg, arm).permute(1, 0, 2, 3).reshape(KVH, ntok, D).double()
        dk = (kh - kx).repeat_interleave(grp, 0)                              # [QH, ntok, D]
        e = torch.einsum('hpd,hnd->hpn', qp, dk) / math.sqrt(D)
        res[arm] = math.sqrt(float((e.pow(2) * mask[None]).sum() / count))
    ratio = ((meta8(pg['mn']) - pg['mn']).abs() / pg['step']).reshape(-1).double()
    bf_lossy = int((pg['mn'].to(torch.bfloat16).float() != pg['mn']).sum())
    qs = [ratio.median().item()] + torch.quantile(ratio, torch.tensor([0.9, 0.99], dtype=torch.float64, device=ratio.device)).tolist()   # lower median (torch.median), linear p90/p99
    return {'E_meta8': res['meta8'], 'E_bf16': res['bf16'], 'dE': res['meta8'] - res['bf16'],
            'rel': (res['meta8'] - res['bf16']) / res['meta8'] if res['meta8'] > 0 else 0.0,
            'dMSE': res['meta8'] ** 2 - res['bf16'] ** 2,
            'min_ratio_frac_gt_half': float((ratio > 0.5).double().mean()), 'min_ratio_median': float(qs[0]),
            'min_ratio_p90': float(qs[1]), 'min_ratio_p99': float(qs[2]), 'min_ratio_max': float(ratio.max()),
            'kmin_bf16_lossy': bf_lossy, 'closed_pages_last_query': npages, 'queries': len(positions),
            'score_terms': int(count)}
