import hashlib, json, sys
sys.path.insert(0, r"C:\Users\ryanz\Documents\GitHub\benchquant\kvquant_full\RABIT-KV-upload\benchmarks\mlsys2027")
import canonical_ppl_identity as ident
audit = json.load(open("llama_cache_audit.json", encoding="utf-8"))
gm = json.load(open("git_manifests.json"))
def msha(files): return hashlib.sha256("\n".join(f"{p}\t{s}\t{h}" for p, s, h in sorted(files)).encode()).hexdigest()
print("write_probe:", audit["write_probe"])
print("llama repo dir:", audit["llama_repo_dir_listing"], "snapshots:", audit["llama_snapshots_listing"])
print("non-snapshot small files (llama / .lock):", {k: v for k, v in audit["non_snapshot_small_files"].items() if v} or "all empty", len(audit["non_snapshot_small_files"]))
for snap, rows in audit["llama_snapshots"].items():
    files = [[r["path"], r["size"], r["sha256"]] for r in rows]
    print(f"\nSNAPSHOT {snap}: {len(files)} files, total {sum(f[1] for f in files)} bytes, links={sum(r['is_link'] for r in rows)}")
    for r in rows: print(f'  {r["path"]:40} {r["size"]:>12} {r["sha256"]} {r["mtime_utc"]}')
    hist = msha(files); print("  HISTORICAL MANIFEST SHA256:", hist)
    print("  == pinned LLAMA_MANIFEST_SHA256:", hist == ident.LLAMA_MANIFEST_SHA256, "| files identical to LLAMA_FILES:", sorted(files) == sorted(ident.LLAMA_FILES))
    cand = gm["359efdbb8af05b788a4ad4185215c6b8caa9052c"]
    gf = {f[0]: f for f in cand["files"]}
    for p, s, h in files:
        g = gf.get(p); print(f"   vs git 359efdbb {p:40} {'MATCH' if g and g[1]==s and g[2]==h else 'DIFF'} ({g[3] if g else 'absent'})")
    print("  extra in git tree:", sorted(set(gf) - {f[0] for f in files}))
    print("  commits whose git-derived manifest equals the historical manifest:")
    for c, v in gm.items():
        eq = v["manifest_sha256"] == hist
        d = [f[0] for f in v["files"] if dict((x[0], x[2]) for x in files).get(f[0]) != f[2]]
        print(f"   {c} {'EQUAL' if eq else 'differs'} n={len(v['files'])} differing={d[:6]} | {v['meta']}")
