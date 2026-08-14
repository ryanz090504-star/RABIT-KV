#!/usr/bin/env python3
import json, sys

def s(v, fmt='.2f', width=8):
    if v is None: return '     N/A'
    try: return f'{float(v):{width}{fmt}}'
    except: return str(v)[:width]

rows = []
with open(sys.argv[1]) as f:
    for line in f:
        r = json.loads(line)
        pn = r.get('policy_name','?')
        nb = r.get('nbits','?')
        ppl = r.get('ppl', 0)
        comp = s(r.get('compression_ratio', 1), '.1f', 5)
        ms = s(r.get('ms_per_token', 0), '.0f')
        cos = s(r.get('error_cosine_similarity'), '.4f', 8)
        snr = s(r.get('error_snr_db'), '.1f', 6)
        kl = s(r.get('kl_divergence'), '.4f', 8)
        t1 = s(r.get('top1_accuracy'), '.3f', 7)
        t5 = s(r.get('top5_accuracy'), '.3f', 7)
        nb_str = str(nb) if nb is not None else '16'
        print(f"{pn:20s} {nb_str:>3s}bit  PPL={ppl:8.2f}  Comp={comp}  ms/tok={ms}  Cos={cos}  SNR={snr}  KL={kl}  Top1={t1}  Top5={t5}")
