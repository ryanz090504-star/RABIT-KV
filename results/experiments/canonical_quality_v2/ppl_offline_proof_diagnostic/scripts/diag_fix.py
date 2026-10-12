import sys
P = r"C:\Users\ryanz\Documents\GitHub\benchquant\kvquant_full\RABIT-KV-upload\benchmarks\mlsys2027\canonical_ppl_offline_proofs.py"
src = open(P, encoding="utf-8").read()
old = "        parity = core.prefill_state_parity(model, ctx)"
assert src.count(old) == 1
ns = {"__name__": "patched", "__file__": P}
exec(compile(src.replace(old, "        raw.clear()\n" + old), P, "exec"), ns)
models = ns["tiny_models"]()
res = [ns["canonical_path_proofs"](n, m) for _ in range(int(sys.argv[1])) for n, m in models.items()]
print({"runs": len(res), "passed": sum(r["passed"] for r in res), "state_mismatches": sum(r["state_mismatches"] for r in res),
       "decoded_calls": sorted({r["decoded_calls"] for r in res})})
