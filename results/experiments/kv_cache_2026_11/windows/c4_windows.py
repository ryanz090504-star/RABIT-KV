"""QSP-R01 C4 window builder (from H01's build procedure as in C3's rebuild_windows.py). Stdlib + `tokenizers` only.

Source: WikiText-2 validation (valid.txt, sha f0737ed3..., 1,121,681 bytes; pinned pytorch/examples URL), non-empty stripped lines
in blocks of 64, each block tokenized with the model's own hash-pinned tokenizer.json (no special tokens). The H001R pool (the
first 32 x 1,152 tokens of blocks 0..used) must be reproduced (Qwen 32cc9e1c..., Llama 7b1140d7...); the tail starts at the
next block; M = len(tail) // 1,152 spans.

Index rules (N = 32):
  h01  h_k = floor((2k+1) M / 64), k = 0..31             (H01's rule; used here only to reproduce H01 in the offline test)
  r01  r_k = floor((h_k + h_{k+1}) / 2), k = 0..30, plus M - 1 (the reserve; registered in ../PLAN.md section 4)
For Qwen, M must be 196 and the r01 indices must equal the registered list, and every R01 index must be >= 3 spans away from
every H01 index (>= 2 unscored windows in between). For Llama no Llama window was ever scored on this tail, so the distances to
the Llama H01-style grid are recorded, not required. Run order: random.Random(20261113).shuffle(range(32)) (h01: 20261110).
Window ids: r<tail index:03d> (h01: v<index:03d>).

Modes:
  build   <model> <tokenizer.json> <valid.txt> <out windows json> <out spec json>
          the R01 build step, run ONCE locally after the protocol freeze; writes the window file and its hash spec
  rebuild <model> <model dir> <spec json> <out windows json> [<local valid.txt> (offline tests only)]
          in the Pod: fetch, rebuild, compare with the frozen spec window by window, fail-closed before anything is written
  pool    <model> <tokenizer.json> <valid.txt>  -> prints only whether the H001R pool is reproduced (offline test)
  h01     <tokenizer.json> <valid.txt> <out>    -> H01's window file by the h01 rule (offline test: must equal H01's file)"""
import hashlib, json, random, struct, sys, time, urllib.request
from pathlib import Path

URL = ('https://raw.githubusercontent.com/pytorch/examples/d5678bc8ac0cdd79dbd5e44d4130271018bcec4e/'
       'word_language_model/data/wikitext-2/valid.txt')
VALID_SHA, VALID_BYTES = 'f0737ed31fc1329026e95cb8b98e19c2a182c39c240ab909dc31abf2f8af58e8', 1121681
CTX, EVAL, N, BLOCK = 1024, 128, 32, 64
SPAN = CTX + EVAL
SEED = {'h01': 20261110, 'r01': 20261113}
MODELS = {
    'qwen': {'model_id': 'Qwen/Qwen2.5-7B-Instruct', 'revision': '16c174980d8a1492910551634b4969e69cdc2444',
             'tokenizer_sha256': 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539',
             'pool_sha256': '32cc9e1cf1476347de2237daec9c78096429072827a57c08a6d16b3418b31de0'},
    'llama': {'model_id': 'LLM-Research/Meta-Llama-3.1-8B-Instruct', 'revision': '359efdbb8af05b788a4ad4185215c6b8caa9052c',
              'tokenizer_sha256': '79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4',
              'pool_sha256': '7b1140d750fbbf6ee496fcb25564059704a66d59c36f25fac6372b3b7187bf04'},
}
QWEN_M = 196
QWEN_R01_REGISTERED = [6, 12, 18, 24, 30, 36, 42, 48, 55, 61, 67, 73, 79, 85, 91, 97, 104, 110, 116, 122, 128, 134, 140, 146, 153,
                       159, 165, 171, 177, 183, 189, 195]
sha = lambda b: hashlib.sha256(b).hexdigest()
digest_ids = lambda ids: sha(struct.pack(f'<{len(ids)}q', *ids))


def h01_indices(M):
    return [((2 * k + 1) * M) // (2 * N) for k in range(N)]


def r01_indices(M):
    h = h01_indices(M)
    return [(h[k] + h[k + 1]) // 2 for k in range(N - 1)] + [M - 1]


def fetch():
    err = None
    for attempt in range(3):   # a read-only GET; the content hash decides
        try:
            with urllib.request.urlopen(urllib.request.Request(URL, headers={'User-Agent': 'qsp-r01-c4'}), timeout=60) as r:
                raw = r.read()
            if len(raw) == VALID_BYTES and sha(raw) == VALID_SHA:
                return raw, attempt + 1
            err = 'content mismatch: %d bytes, sha %s' % (len(raw), sha(raw)[:16])
        except Exception as e:
            err = '%s: %s' % (type(e).__name__, str(e)[:200])
        time.sleep(5)
    raise SystemExit('validation text not obtained: ' + err)


def tokenizer(model, path):
    from tokenizers import Tokenizer
    assert sha(Path(path).read_bytes()) == MODELS[model]['tokenizer_sha256'], 'tokenizer.json differs from the pinned manifest'
    return Tokenizer.from_file(str(path))


def pool_and_tail(model, raw, tok):
    assert len(raw) == VALID_BYTES and sha(raw) == VALID_SHA, 'validation text differs'
    enc = lambda text: tok.encode(text, add_special_tokens=False).ids
    lines = [s.strip() for s in raw.decode('utf-8').splitlines() if s.strip()]
    blocks = [enc('\n'.join(lines[i:i + BLOCK])) for i in range(0, len(lines), BLOCK)]
    cum, used = 0, None
    for b, ids in enumerate(blocks):
        cum += len(ids)
        if cum >= N * SPAN:
            used = b
            break
    pool = [t for ids in blocks[:used + 1] for t in ids][:N * SPAN]
    assert digest_ids(pool) == MODELS[model]['pool_sha256'], 'H001R pool not reproduced'
    tail = [int(t) for ids in blocks[used + 1:] for t in ids]
    return {'lines': len(lines), 'blocks': len(blocks), 'used': used, 'first_block': used + 1, 'tail': tail}


def make_doc(model, raw, tok, rule):
    """The window document, exactly as H01's build_windows.py writes it for rule 'h01' (Qwen), or the C4 reserve for 'r01'."""
    vocab = tok.get_vocab_size(with_added_tokens=True)
    pt = pool_and_tail(model, raw, tok)
    tail = pt['tail']
    M = len(tail) // SPAN
    idx = h01_indices(M) if rule == 'h01' else r01_indices(M)
    checks = {}
    if rule == 'r01':
        h = h01_indices(M)
        dist = [min(abs(r - x) for x in h) for r in idx]
        if model == 'qwen':
            checks = {'qwen_M_is_196': M == QWEN_M, 'qwen_indices_equal_registered': idx == QWEN_R01_REGISTERED,
                      'qwen_min_distance_to_h01_ge_3': min(dist) >= 3}
            assert all(checks.values()), ('Qwen reserve differs from the registration', checks)
        assert len(set(idx)) == N and all(0 <= i < M for i in idx)
    order = list(range(N))
    random.Random(SEED[rule]).shuffle(order)
    prefix = 'v' if rule == 'h01' else 'r'
    wins = []
    for k in order:
        i = idx[k]
        seq = tail[i * SPAN:(i + 1) * SPAN]
        assert len(seq) == SPAN and all(0 <= t < vocab for t in seq)
        wins.append({'id': f'{prefix}{i:03d}', 'tail_index': i, 'selection_k': k, 'tail_offset': i * SPAN,
                     'context': seq[:CTX], 'targets': seq[CTX:], 'token_sha256': digest_ids(seq)})
    m = MODELS[model]
    if rule == 'h01':
        doc = {'job': 'QSP-H01', 'model_id': m['model_id'], 'model_revision': m['revision'],
               'source': {'split': 'wikitext-2 validation', 'sha256': VALID_SHA, 'lines': pt['lines'], 'blocks': pt['blocks'], 'block_lines': BLOCK},
               'exclusions': {'test_split': 'entire (canonical windows 1-32 incl. Q01 window 1 scored; early dev use unbounded)',
                              'h001r_blocks': [0, pt['used']], 'h001r_qwen_pool_sha256_reproduced': m['pool_sha256']},
               'tail': {'first_block': pt['first_block'], 'tokens': len(tail), 'windows_available': M, 'token_sha256': digest_ids(tail)},
               'selection': {'n': N, 'rule': 'index_k = floor((2k+1) M / (2N)), k = 0..N-1', 'run_order': f'random.Random({SEED[rule]}).shuffle(range(N))'},
               'ctx': CTX, 'eval': EVAL, 'windows': wins}
    else:
        doc = {'job': 'QSP-R01-C4', 'set': 'R01' if model == 'qwen' else 'R01-L', 'model_id': m['model_id'], 'model_revision': m['revision'],
               'source': {'split': 'wikitext-2 validation', 'sha256': VALID_SHA, 'lines': pt['lines'], 'blocks': pt['blocks'], 'block_lines': BLOCK},
               'exclusions': {'test_split': 'entire', 'h001r_blocks': [0, pt['used']], 'h001r_pool_sha256_reproduced': m['pool_sha256'],
                              'h01_grid_indices': h01_indices(M), 'min_distance_to_h01_grid': min(min(abs(r - x) for x in h01_indices(M)) for r in idx),
                              'h01_grid_scored_for_this_model': model == 'qwen'},
               'tail': {'first_block': pt['first_block'], 'tokens': len(tail), 'windows_available': M, 'token_sha256': digest_ids(tail)},
               'selection': {'n': N, 'rule': 'r_k = floor((h_k + h_{k+1}) / 2), k = 0..N-2, plus M - 1; h_k = floor((2k+1) M / (2N))',
                             'indices': idx, 'checks': checks, 'run_order': f'random.Random({SEED[rule]}).shuffle(range(N))'},
               'ctx': CTX, 'eval': EVAL, 'windows': wins}
    doc['list_sha256'] = sha(json.dumps([[w['id'], w['token_sha256']] for w in wins]).encode())
    return doc, vocab


def spec_of(doc, text, vocab):
    return {'job': doc['job'], 'set': doc.get('set'), 'model_revision': doc['model_revision'], 'file_sha256_lf': sha(text.encode()),
            'list_sha256': doc['list_sha256'], 'tail_token_sha256': doc['tail']['token_sha256'], 'tokenizer_len': vocab,
            'windows': [{'id': w['id'], 'token_sha256': w['token_sha256'], 'target_sha256': digest_ids(w['targets'])} for w in doc['windows']]}


def compare(doc, text, vocab, spec):
    sw = spec['windows']
    wins = doc['windows']
    checks = {'model_revision': doc['model_revision'] == spec['model_revision'],
              'tokenizer_len': vocab == spec['tokenizer_len'],
              'tail_token_sha256': doc['tail']['token_sha256'] == spec['tail_token_sha256'],
              'window_count': len(wins) == len(sw) == N,
              'run_order': [w['id'] for w in wins] == [s['id'] for s in sw],
              'token_sha256': all(w['token_sha256'] == s['token_sha256'] for w, s in zip(wins, sw)),
              'target_sha256': all(digest_ids(w['targets']) == s['target_sha256'] for w, s in zip(wins, sw)),
              'list_sha256': doc['list_sha256'] == spec['list_sha256'],
              'file_sha256_lf': sha(text.encode()) == spec['file_sha256_lf']}
    return checks


def main():
    mode = sys.argv[1]
    if mode == 'build':
        model, tj, valid, out, spec_out = sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]), Path(sys.argv[6])
        doc, vocab = make_doc(model, valid.read_bytes(), tokenizer(model, tj), 'r01')
        text = json.dumps(doc, indent=1)
        spec = spec_of(doc, text, vocab)
        with out.open('x', encoding='utf-8', newline='\n') as f:
            f.write(text)
        with spec_out.open('x', encoding='utf-8', newline='\n') as f:
            f.write(json.dumps(spec, indent=1) + '\n')
        print(json.dumps({'built': doc['set'], 'file_sha256_lf': spec['file_sha256_lf'], 'list_sha256': spec['list_sha256']}))
    elif mode == 'rebuild':
        model, mdir, spec_p, out = sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5])
        spec = json.loads(spec_p.read_text())
        if len(sys.argv) > 6:
            raw, attempts = Path(sys.argv[6]).read_bytes(), 0
        else:
            raw, attempts = fetch()
        doc, vocab = make_doc(model, raw, tokenizer(model, mdir / 'tokenizer.json'), 'r01')
        text = json.dumps(doc, indent=1)
        checks = compare(doc, text, vocab, spec)
        rec = {'fetch_attempts': attempts, 'tokenizers_vocab': vocab, 'checks': checks, 'file_sha256_lf': sha(text.encode())}
        print(json.dumps(rec), flush=True)
        if not all(checks.values()):
            raise SystemExit('window rebuild does not match the frozen list; nothing is scored')
        with out.open('x', encoding='utf-8', newline='\n') as f:
            f.write(text)
    elif mode == 'pool':
        model, tj, valid = sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
        pool_and_tail(model, valid.read_bytes(), tokenizer(model, tj))
        print(json.dumps({'model': model, 'h001r_pool_reproduced': True}))
    elif mode == 'h01':
        tj, valid, out = Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
        doc, vocab = make_doc('qwen', valid.read_bytes(), tokenizer('qwen', tj), 'h01')
        with out.open('x', encoding='utf-8', newline='\n') as f:
            f.write(json.dumps(doc, indent=1))
        print(json.dumps({'h01_file_sha256_lf': sha(json.dumps(doc, indent=1).encode())}))
    else:
        raise SystemExit('unknown mode')


if __name__ == '__main__':
    main()
