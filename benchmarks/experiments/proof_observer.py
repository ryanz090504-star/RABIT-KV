"""
PROOF-ONLY bookkeeping (stdlib only; never shipped to the container, never imported by scoring code).

IdentityObserver associates recorded items with OBSERVED OBJECTS by object identity (`is`) and never by id(obj).
It keeps a DIRECT reference to every observed object for its whole lifetime, so an observed object cannot be freed
while the observer exists and its memory address cannot be reused by a later object; and because lookup compares with
`is` (no hash, no __eq__, no address), state recorded for one object can never be returned for another one.

It replaces the id()-keyed dict of the first canonical_ppl_offline_proofs.py, which produced nondeterministic FALSE
mismatches when a new layer state was allocated at the address of a freed one
(results/mlsys2027/canonical_quality_v2/ppl_offline_proof_diagnostic/record.json).
"""

from __future__ import annotations


class IdentityObserver:
    def __init__(self):
        self._entries = []  # [object (retained direct reference), items]; the index is the monotonic sequence ID

    def _index(self, obj) -> int:
        for i, entry in enumerate(self._entries):
            if entry[0] is obj:
                return i
        return -1

    def record(self, obj, item) -> int:
        """Append `item` to the state of `obj` (registering it on first sight); returns its sequence ID."""
        i = self._index(obj)
        if i < 0:
            self._entries.append([obj, []])
            i = len(self._entries) - 1
        self._entries[i][1].append(item)
        return i

    def items(self, obj) -> list:
        """Everything recorded for exactly this object; KeyError if it was never observed."""
        i = self._index(obj)
        if i < 0:
            raise KeyError("object was never observed")
        return list(self._entries[i][1])

    def sequence_id(self, obj) -> int:
        i = self._index(obj)
        if i < 0:
            raise KeyError("object was never observed")
        return i

    def __len__(self) -> int:
        return len(self._entries)
