"""A001 independent local audit (written after launch, before any result was read; not part of the frozen record).
Does not trust any pool, hash or analysis reported by the Pod: rebuilds both canonical test token pools locally from the
pinned text and the locally kept tokenizer files (holdout-001/tokenizers, read-only), checks every per-token row against
them, compares bf16/rabit/range_real with the ORIGINAL attempt-008 artifacts (gpu-attempt-008/artifacts, not the image
copy), re-runs the frozen A001 audit on a COPY of the retrieved run, and recomputes the frozen analysis from per-token NLLs.
argv: <retrieved run-gpu dir> <output dir (must not exist)>. Run with the CPU torch venv."""
import hashlib, json, math, shutil, struct, subprocess, sys
from pathlib import Path

A = Path(__file__).resolve().parents[1]; IMG = A / 'image'; Q = A.parent
sys.path[:0] = [str(IMG), str(IMG / 'stage')]
import canonical_ppl_core as core, canonical_ppl_identity as ident, a001_analysis as aa
from transformers import AutoTokenizer

run, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
out.mkdir(parents=True, exist_ok=False)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
before = {p.relative_to(run).as_posix(): sha(p) for p in sorted(run.rglob('*')) if p.is_file()}
reg = json.loads((IMG / 'stage' / 'registration.json').read_text()); a = run / 'artifacts'
rd = lambda n: json.loads((a / n).read_text())
res = {'run': str(run), 'registration_sha256': sha(IMG / 'stage' / 'registration.json'), 'checks': {}, 'models': {}}
chk = res['checks']
prov = rd('provenance.json')
chk['run_registration_equals_frozen_registration'] = prov['registration'] == reg
chk['run_stage_files_equal_frozen_image'] = prov['files'] == json.loads((IMG / 'stage-sha256.json').read_text())
chk['run_complete'] = rd('complete.json')['completed'] and rd('complete.json')['attempt'] == 'RUNPOD-QWEN-A001' and rd('runner.json')['returncode'] == 0
CONTROLS = {'passthrough': 'bf16', 'oracle': 'rabit', 'oracle_range': 'range_real', 'oracle_meta': 'meta_exact', 'oracle_range_meta': 'range_meta_exact'}
ctxh = lambda pool, w: hashlib.sha256(struct.pack('<1024q', *pool[(w - 1) * 1152:(w - 1) * 1152 + 1024])).hexdigest()
for key, spec in ident.MODELS.items():
    tok = AutoTokenizer.from_pretrained(Q / 'holdout-001' / 'tokenizers' / key, local_files_only=True)
    pool = core.build_token_pool(core.wikitext_lines((IMG / 'stage' / 'wikitext.txt').read_text(encoding='utf-8')), tok, 32, 1024, 128)
    rows = rd(key + '-scores.json'); m = {}
    m['local_pool_equals_registration'] = core.pool_sha256(pool) == reg['token_pool_sha256'][key] == spec['token_pool_sha256']
    m['pod_pool_equals_local_pool'] = rd(key + '-tokens.json')['pool'] == pool
    m['arms'] = sorted(rows) == sorted(reg['arms'])
    n_rows = n_tok = 0; bad = []
    for arm, rr in rows.items():
        want = list(range(1, 33)) if arm in reg['scientific_arms'] else [1]
        if [r['window'] for r in rr] != want:
            bad.append((arm, 'windows'))
        for r in rr:
            w = r['window']; o = (w - 1) * 1152; nll = r['token_nll']; n_rows += 1; n_tok += len(nll)
            ok = (len(nll) == 128 and all(isinstance(v, float) and math.isfinite(v) and v >= 0 for v in nll)
                  and r['audit']['target_ids'] == pool[o + 1024:o + 1152] and r['audit']['context_sha256'] == ctxh(pool, w)
                  and abs(math.fsum(nll) - r['loss_sum']) < 1e-3 and r['audit']['arm'] == arm)
            if not ok:
                bad.append((arm, w))
    m['rows'], m['per_token_values'], m['bad_rows'] = n_rows, n_tok, bad[:20]
    m['every_row_on_registered_test_window'] = not bad and n_rows == 5 * 32 + 5
    m['window1_controls_bitwise'] = {c: rows[c][0]['token_nll'] == rows[ref][0]['token_nll'] for c, ref in CONTROLS.items()}
    orig = json.loads((Q / 'gpu-attempt-008' / 'artifacts' / (key + '-scores.json')).read_text())
    m['attempt008_original_bitwise'] = {arm: [r['token_nll'] for r in rows[arm]] == [r['token_nll'] for r in sorted(orig[arm], key=lambda z: z['window'])]
                                        for arm in ['bf16', 'rabit', 'range_real']}
    means = {arm: [math.fsum(r['token_nll']) / 128 for r in rows[arm]] for arm in reg['scientific_arms']}
    an = reg['analysis']; m['analysis_recomputed'] = aa.analyze(means, an['bootstrap_seed'], an['bootstrap_replicates'])
    res['models'][key] = m
cp = out / 'run-gpu-copy'; shutil.copytree(run, cp)
(cp / 'audit.json').unlink(missing_ok=True)
r = subprocess.run([sys.executable, '-B', str(IMG / 'audit.py'), str(cp), str(IMG)], capture_output=True, text=True)
chk['frozen_audit_local_rc0'] = r.returncode == 0
(out / 'frozen-audit-local.log').write_text(r.stdout + r.stderr)
if r.returncode == 0:
    loc = json.loads((cp / 'audit.json').read_text())
    for key, m in res['models'].items():
        chk[f'{key}: local frozen audit valid'] = loc['models'][key]['run_valid']
        chk[f'{key}: recomputed analysis equals local frozen audit'] = json.dumps(m['analysis_recomputed']) == json.dumps(loc['models'][key]['analysis'])
    pod_audit = run / 'audit.json'
    chk['pod_audit_present'] = pod_audit.exists()
    if pod_audit.exists():
        pa_ = json.loads(pod_audit.read_text())
        chk['pod_audit_equals_local_audit'] = {k: json.dumps(v['analysis']) == json.dumps(loc['models'][k]['analysis']) for k, v in pa_['models'].items()}
for key, m in res['models'].items():
    for k in ['local_pool_equals_registration', 'pod_pool_equals_local_pool', 'arms', 'every_row_on_registered_test_window']:
        chk[f'{key}: {k}'] = m[k]
    chk[f'{key}: controls'] = all(m['window1_controls_bitwise'].values())
    chk[f'{key}: attempt008 reproduction (original artifacts)'] = all(m['attempt008_original_bitwise'].values())
after = {p.relative_to(run).as_posix(): sha(p) for p in sorted(run.rglob('*')) if p.is_file()}
chk['retrieved_files_unmodified'] = before == after
flat = [v if isinstance(v, bool) else all(v.values()) for v in chk.values()]
res['passed'] = all(flat)
(out / 'independent-audit.json').write_text(json.dumps(res, indent=2, allow_nan=False), newline='\n')
print(json.dumps({'passed': res['passed'], 'failed': [k for k, v in chk.items() if not (v if isinstance(v, bool) else all(v.values()))]}, indent=1))
