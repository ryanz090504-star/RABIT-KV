"""Diagnostic (scratchpad only): replicate canonical_path_proofs with (a) the ORIGINAL id(self)-keyed spy and
(b) an identity-safe spy (raw stored on the object itself), and log where any mismatch happens."""
import sys

import torch

sys.path.insert(0, r"C:\Users\ryanz\Documents\GitHub\benchquant\kvquant_full\RABIT-KV-upload\benchmarks\mlsys2027")
import canonical_ppl_core as core  # noqa: E402
import canonical_ppl_offline_proofs as proofs  # noqa: E402
import canonical_rabit_quality as crq  # noqa: E402

append0, decoded0 = crq.CanonicalLayerState.append, crq.CanonicalLayerState.decoded


def run_once(model, tag):
    ctx, cont = proofs._tiny_ids(11)
    raw_by_id, events, phase = {}, [], {"name": "score"}

    def append(self, k, v):
        k16, v16 = k.detach().to(torch.bfloat16), v.detach().to(torch.bfloat16)
        # (a) original spy: keyed by id(self)
        pk, pv = raw_by_id.get(id(self), (None, None))
        fresh_object = not hasattr(self, "_spy_raw")
        if fresh_object and pk is not None:
            events.append(("ID_REUSE", phase["name"], id(self), int(pk.shape[0])))
        raw_by_id[id(self)] = (k16 if pk is None else torch.cat([pk, k16]), v16 if pv is None else torch.cat([pv, v16]))
        # (b) identity-safe spy: stored on the object
        if fresh_object:
            self._spy_raw = (k16, v16)
        else:
            self._spy_raw = (torch.cat([self._spy_raw[0], k16]), torch.cat([self._spy_raw[1], v16]))
        return append0(self, k, v)

    def decoded(self):
        dk, dv = decoded0(self)
        ra = crq.canonical_state(*raw_by_id[id(self)])
        rb = crq.canonical_state(*self._spy_raw)
        a_ok = torch.equal(dk, ra["decoded_k"]) and torch.equal(dv, ra["decoded_v"])
        b_ok = torch.equal(dk, rb["decoded_k"]) and torch.equal(dv, rb["decoded_v"])
        if not a_ok or not b_ok:
            events.append(("MISMATCH", phase["name"], "id_keyed_ok=%s" % a_ok, "object_keyed_ok=%s" % b_ok,
                           "n=%d" % self.n, "id_keyed_raw_tokens=%d" % raw_by_id[id(self)][0].shape[0],
                           "object_raw_tokens=%d" % self._spy_raw[0].shape[0]))
        return dk, dv

    crq.CanonicalLayerState.append, crq.CanonicalLayerState.decoded = append, decoded
    try:
        core.score(model, ctx, cont, "rabit")
        phase["name"] = "parity"
        core.prefill_state_parity(model, ctx)
    finally:
        crq.CanonicalLayerState.append, crq.CanonicalLayerState.decoded = append0, decoded0
    return events


models = proofs.tiny_models()
total = {"runs": 0, "runs_with_id_keyed_mismatch": 0, "runs_with_object_keyed_mismatch": 0, "id_reuse_events": 0}
for i in range(int(sys.argv[1])):
    for name, m in models.items():
        ev = run_once(m, name)
        total["runs"] += 1
        total["id_reuse_events"] += sum(e[0] == "ID_REUSE" for e in ev)
        mm = [e for e in ev if e[0] == "MISMATCH"]
        total["runs_with_id_keyed_mismatch"] += any("id_keyed_ok=False" in e for e in mm)
        total["runs_with_object_keyed_mismatch"] += any("object_keyed_ok=False" in e for e in mm)
        if ev:
            print(i, name, ev)
print(total)
