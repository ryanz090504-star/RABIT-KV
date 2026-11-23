"""H1 capture (registered in ../REGISTRATION-H1.md): one model, its own 32 C4 windows, BF16 HF forward, post-RoPE Q/K per layer,
registered metric per (window, layer). No quantization runs inside the model; nothing is scored for quality.
  windows   the frozen C4 window file of this model (c4/windows/windows-r01.json or windows-r01l.json); every window's token_sha256
            (= sha256 of the int64 little-endian ids of context + targets) and the list_sha256 are checked fail-closed
  forward   input = context (1,024) + targets[0:127] (1,151 tokens), one causal forward, BF16 weights, attn_implementation 'sdpa',
            use_cache False. The registered sdpa function is wrapped to keep (layer_idx, query, key): the post-RoPE tensors the
            attention receives (query [QH, T, D], key [KVH, T, D], before any GQA repeat). The wrapper then calls the original function.
  metric    h1_metric.layer_metric(q, k, positions 1024..1150) per layer (the 127 decode queries of C4's targets 1..127)
  sanity    capture transparency (logits with and without the wrapper bitwise equal, first window); layer-0 post-RoPE K recomputed
            from the embeddings (input_layernorm -> k_proj -> RoPE) equals the captured K bitwise (first window); the metric on CPU
            equals the GPU metric for one (window, layer) within 1e-6 relative; per-window BF16 NLL of targets 0..127 (FP32 log-softmax)
            is saved for a local comparison with C4's bf16_single cell (provenance, not a gate here)
  saved     metrics-<model>.jsonl (one line per window x layer), nll-<model>.json, capture-<model>.json (checks, timings),
            tensors-<model>-<window>.pt for the first 2 windows in file order: q[:, 1024:1151] and k[:, 0:1151] (BF16) of layers
            {0, 1, L // 2, L - 1}, for an offline recomputation of the Pod's metric
argv: <model qwen|llama> <model dir> <windows json> <out dir> <deadline epoch> <device cuda|cpu> [max windows, tests only]"""
import hashlib, json, math, struct, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import h1_metric as M

MODEL, MDIR, WIN, OUT, DEADLINE, DEV = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
MAXW = int(sys.argv[7]) if len(sys.argv) > 7 else None
GEOM = {'qwen': {'layers': 28, 'qh': 28, 'kvh': 4, 'd': 128}, 'llama': {'layers': 32, 'qh': 32, 'kvh': 8, 'd': 128}}
CTX, EVAL = 1024, 128
POSITIONS = list(range(CTX, CTX + EVAL - 1))          # 1024 .. 1150
OUT.mkdir(parents=True, exist_ok=True)
t_start = time.time()
sha = lambda b: hashlib.sha256(b).hexdigest()
digest_ids = lambda ids: sha(struct.pack(f'<{len(ids)}q', *ids))

doc = json.loads(WIN.read_text())
wins = doc['windows']
assert len(wins) == 32
for w in wins:
    assert len(w['context']) == CTX and len(w['targets']) == EVAL
    assert digest_ids(w['context'] + w['targets']) == w['token_sha256'], ('window token hash', w['id'])
assert sha(json.dumps([[w['id'], w['token_sha256']] for w in wins]).encode()) == doc['list_sha256'], 'window list hash'
if MAXW:
    wins = wins[:MAXW]

from transformers import AutoModelForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

model = AutoModelForCausalLM.from_pretrained(str(MDIR), torch_dtype=torch.bfloat16, attn_implementation='sdpa').to(DEV).eval()
cfg = model.config
g = GEOM[MODEL] if not MAXW else {'layers': cfg.num_hidden_layers, 'qh': cfg.num_attention_heads, 'kvh': cfg.num_key_value_heads,
                                  'd': cfg.hidden_size // cfg.num_attention_heads}
assert (cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads, getattr(cfg, 'head_dim', cfg.hidden_size // cfg.num_attention_heads)) == \
    (g['layers'], g['qh'], g['kvh'], g['d']), 'model geometry'
t_load = time.time() - t_start

ORIG = ALL_ATTENTION_FUNCTIONS['sdpa']
CAP = {'on': False, 'q': {}, 'k': {}}


def capturing_sdpa(module, query, key, value, attention_mask, **kw):
    if CAP['on']:
        CAP['q'][module.layer_idx] = query[0]
        CAP['k'][module.layer_idx] = key[0]
    return ORIG(module, query, key, value, attention_mask, **kw)


def forward(ids, capture):
    CAP['on'] = capture; CAP['q'].clear(); CAP['k'].clear()
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids], device=DEV), use_cache=False)
    CAP['on'] = False
    return out.logits[0]


checks = {}
w0 = wins[0]
ids0 = w0['context'] + w0['targets'][:-1]
plain = forward(ids0, False)
ALL_ATTENTION_FUNCTIONS['sdpa'] = capturing_sdpa
cap = forward(ids0, True)
checks['capture_transparent_logits_bitwise'] = bool(torch.equal(plain, cap))
checks['captured_layers'] = sorted(CAP['k']) == list(range(g['layers']))
# independent recomputation of layer-0 post-RoPE K from the embeddings
with torch.no_grad():
    ids_t = torch.tensor([ids0], device=DEV)
    h0 = model.model.layers[0].input_layernorm(model.model.embed_tokens(ids_t))
    attn0 = model.model.layers[0].self_attn
    kk = attn0.k_proj(h0).view(1, len(ids0), -1, g['d']).transpose(1, 2)
    qq = attn0.q_proj(h0).view(1, len(ids0), -1, g['d']).transpose(1, 2)
    cos, sin = model.model.rotary_emb(kk, torch.arange(len(ids0), device=DEV)[None])
    qr, kr = apply_rotary_pos_emb(qq, kk, cos, sin)
checks['layer0_postrope_k_bitwise'] = bool(torch.equal(kr[0], CAP['k'][0]))
checks['layer0_postrope_q_bitwise'] = bool(torch.equal(qr[0], CAP['q'][0]))

mf = (OUT / f'metrics-{MODEL}.jsonl').open('w')
nll, times, not_run = {}, [], []
save_layers = sorted({0, 1, g['layers'] // 2, g['layers'] - 1})
for wi, w in enumerate(wins):
    if time.time() > DEADLINE:
        not_run.append(w['id']); continue
    t0 = time.time()
    ids = w['context'] + w['targets'][:-1]
    logits = forward(ids, True)
    lp = torch.log_softmax(logits[CTX - 1:CTX - 1 + EVAL].float(), -1)
    tok = torch.tensor(w['targets'], device=DEV)
    rows = (-lp.gather(1, tok[:, None])[:, 0]).double().cpu()
    nll[w['id']] = {'mean': float(rows.mean()), 'rows': [float(x) for x in rows]}
    for layer in range(g['layers']):
        q, k = CAP['q'][layer], CAP['k'][layer]
        assert q.shape == (g['qh'], len(ids), g['d']) and k.shape == (g['kvh'], len(ids), g['d']) and q.dtype == torch.bfloat16
        m = M.layer_metric(q, k, POSITIONS)
        mf.write(json.dumps({'model': MODEL, 'window': w['id'], 'layer': layer, **m}) + '\n')
        if wi == 0 and layer == 0:
            mc = M.layer_metric(q.cpu(), k.cpu(), POSITIONS)
            checks['cpu_equals_device_metric'] = all(abs(mc[x] - m[x]) <= 1e-6 * max(1e-12, abs(m[x])) for x in ('E_meta8', 'E_bf16', 'dE')) \
                and mc['min_ratio_frac_gt_half'] == m['min_ratio_frac_gt_half']
            checks['cpu_vs_device_dE'] = [mc['dE'], m['dE']]
    if wi < 2:
        torch.save({'model': MODEL, 'window': w['id'], 'positions': POSITIONS,
                    'layers': {L: {'q': CAP['q'][L][:, CTX:CTX + EVAL - 1].cpu().clone(), 'k': CAP['k'][L].cpu().clone()} for L in save_layers}},
                   OUT / f'tensors-{MODEL}-{w["id"]}.pt')
    mf.flush()
    times.append(round(time.time() - t0, 2))
mf.close()
(OUT / f'nll-{MODEL}.json').write_text(json.dumps(nll) + '\n')
rec = {'model': MODEL, 'model_dir': str(MDIR), 'windows_file': str(WIN), 'windows_list_sha256': doc['list_sha256'], 'windows_run': len(nll),
       'not_run': not_run, 'checks': checks, 'save_layers': save_layers, 'load_seconds': round(t_load, 1), 'window_seconds': times,
       'device': torch.cuda.get_device_name(0) if DEV == 'cuda' else 'cpu', 'torch': torch.__version__,
       'transformers': __import__('transformers').__version__, 'div_rule': M.DIV_RULE, 'positions': [POSITIONS[0], POSITIONS[-1]],
       'total_seconds': round(time.time() - t_start, 1)}
ok = all(v is True for k, v in checks.items() if not k.startswith('cpu_vs')) and not not_run
rec['all_checks_pass'] = ok
(OUT / f'capture-{MODEL}.json').write_text(json.dumps(rec, indent=1) + '\n')
print('H1_CAPTURE', MODEL, 'PASS' if ok else 'FAIL', json.dumps(checks), flush=True)
sys.exit(0 if ok else 4)
