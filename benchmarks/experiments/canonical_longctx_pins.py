"""
canonical-quality-v2 LONG-CONTEXT suite -- FROZEN pins (stdlib only; constants only).

PROMPT_SETS: for each task the number of units and the SHA-256 over the ordered
'key<TAB>token count<TAB>int64-LE token-id sha256' lines of every prompt, computed OFFLINE with the pinned Llama
tokenizer files, the pinned WikiText-2 text and the pinned LongBench parquet files, and proven equal to the legacy
Exp12 prompts (canonical_longctx_offline_proofs.py). The container recomputes them before the model is loaded.

CONFORMANCE_16K: the accepted real-model 16k CUDA canonical <-> CUDA frozen-oracle diagnostic (no scoring).
"""

PROMPT_SETS = {
    "niah": {"units": 57, "prompt_set_sha256": "452dbe81bf7cc331f163dfd485dca8d4b7691613a7065a7aba79b00ee8e4f962"},
    "passage_retrieval": {"units": 200, "prompt_set_sha256": "c59428c8210bd4a49ff127995ca5e160b24481f06ab616afe34d532755b81260"},
    "hotpotqa": {"units": 100, "prompt_set_sha256": "01e6976cc58d161c2b06f084101fc4b416437433c6279f7eba80b09bb2a16b1a"},
}
CONFORMANCE_16K = {"evidence_commit": "2637457",
                   "record_sha256_lf": "96892a4f7bf55d47ca3c2bbd9d92999d1fe04d8d786806116e5a06fbbf4852ea",
                   "result_sha256": "faa3a82ec7b3440b84a5270f86c825c8236137ec4d03c5e54ea5ed27c69bbb16"}
