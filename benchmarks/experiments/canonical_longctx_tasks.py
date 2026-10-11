"""
canonical-quality-v2 LONG-CONTEXT suite -- FROZEN task definitions (stdlib only; no torch, no quantization).

The three task families of the accepted legacy Exp12 run (benchmarks/quality/{niah,passage_retrieval,hotpotqa}.py,
hashes pinned in results/mlsys2027/variance/manifest.json), reproduced WITHOUT their legacy quantizers: prompt
construction, truncation, example selection and scoring only. Equality with the legacy source is proven offline
(canonical_longctx_offline_proofs.py executes the legacy statements by AST and compares).

    NIAH               57 cases = contexts 4096 / 8192 / 16384 x 19 depths 0.05 .. 0.95; 16 new tokens
    Passage Retrieval  LongBench passage_retrieval_en rows [0, 200); max input 16384; 32 new tokens
    HotpotQA           LongBench-E hotpotqa_e, length >= 8000, filtered positions [0, 100); max input 16384; 32 new tokens

HotpotQA has TWO preregistered scorers, both applied to the identical prediction strings:
    qa_f1_score_legacy    PRIMARY   -- the accepted Exp12 scorer verbatim, INCLUDING its historical behaviour: the
                                       article pattern is written with doubled backslashes inside a raw string, so
                                       articles are NOT removed. Kept for apples-to-apples comparison with Exp12.
    qa_f1_score_official  SECONDARY -- standards-aligned LongBench qa_f1_score (THUDM/LongBench metrics.py): articles
                                       removed; an empty prediction scores 0.
Dataset file hashes below are NEWLY RECOVERED / FROZEN provenance (2026-10-03) for this suite; the accepted Exp12
evidence recorded only the dataset commits, not these SHA-256 values.
"""

from __future__ import annotations

import hashlib
import json
import re
import string
import struct
from collections import Counter

# ------------------------------------------------------------------------------------------------ NIAH
NIAH_CONTEXTS = (4096, 8192, 16384)
NIAH_DEPTHS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85,
               0.90, 0.95)
NIAH_MAX_NEW_TOKENS = 16
NIAH_SECRET = "RABIT-7291"
NIAH_INSTRUCTION = (
    "Read the following document carefully. It contains one hidden secret code. "
    "After the document, answer the question using only the exact secret code.\n\n"
    "DOCUMENT START\n"
)
NIAH_NEEDLE = (
    "\nIMPORTANT RECORD: The hidden secret code is RABIT-7291. "
    "Remember this exact code.\n"
)
NIAH_QUESTION = (
    "\nDOCUMENT END\n\n"
    "Question: What is the hidden secret code?\n"
    "Answer with only the exact code:"
)
WIKITEXT_URL = ("https://raw.githubusercontent.com/pytorch/examples/main/"
                "word_language_model/data/wikitext-2/test.txt")
WIKITEXT_SHA256 = "d790b833ef8cf03a90db7bf1271b7520b83c45ce07ba3c1a9699df81e239eca0"


def niah_cases() -> list:
    """(context_tokens, depth) in the accepted order: context-major, then depth."""
    return [(c, d) for c in NIAH_CONTEXTS for d in NIAH_DEPTHS]


def niah_filler_text(wikitext: str) -> str:
    return "\n".join(line.strip() for line in wikitext.splitlines() if line.strip())


def niah_parts(tokenizer, wikitext: str) -> dict:
    """Token ids of the fixed prompt parts and of the WikiText-2 filler (no special tokens)."""
    ids = lambda text: tokenizer(text, add_special_tokens=False)["input_ids"]  # noqa: E731
    parts = {"instruction": ids(NIAH_INSTRUCTION), "needle": ids(NIAH_NEEDLE), "question": ids(NIAH_QUESTION),
             "filler": ids(niah_filler_text(wikitext))}
    if not parts["filler"]:
        raise RuntimeError("WikiText-2 filler tokenization produced no tokens.")
    return parts


def niah_prompt_ids(parts: dict, context_tokens: int, needle_depth: float) -> list:
    fixed_tokens = len(parts["instruction"]) + len(parts["needle"]) + len(parts["question"])
    filler_needed = int(context_tokens) - fixed_tokens
    if filler_needed < 128:
        raise ValueError(f"context_tokens={context_tokens} is too small for the prompt template.")
    base = parts["filler"]
    repeats = (filler_needed + len(base) - 1) // len(base)
    filler_ids = (base * repeats)[:filler_needed]
    before_count = int(round(filler_needed * float(needle_depth)))
    before_count = max(0, min(before_count, filler_needed))
    prompt = (parts["instruction"] + filler_ids[:before_count] + parts["needle"] + filler_ids[before_count:]
              + parts["question"])
    if len(prompt) != int(context_tokens):
        raise RuntimeError(f"Prompt length mismatch: got {len(prompt)}, expected {context_tokens}.")
    return [int(x) for x in prompt]


def niah_score(answer: str) -> dict:
    match = re.search(r"RABIT-\d{4}", answer.upper())
    extracted = match.group(0) if match else ""
    return {"extracted_code": extracted, "correct": bool(extracted == NIAH_SECRET)}


# ------------------------------------------------------------------------------------------------ LongBench
LONGBENCH_REPO = "zai-org/LongBench"
MAX_INPUT_TOKENS = 16384
LONGBENCH_MAX_NEW_TOKENS = 32
DATASETS = {
    "passage_retrieval": {
        "revision": "915b0c6ec0b6dfae1cd44224b7d8995317837f27",
        "filename": "passage_retrieval_en/test-00000-of-00001.parquet", "bytes": 7029836,
        "sha256": "452f03dbb0e394de2b26d6e016bf1e715a3ebac9dc93844fde73c0e8e74cfd68",
        "rows": 200, "sample_start": 0, "samples": 200, "length_bucket": None},
    "hotpotqa": {
        "revision": "92b6c5fbfb0c97b91e92d9ef79802f95ce74b05e",
        "filename": "hotpotqa_e/test-00000-of-00001.parquet", "bytes": 7196922,
        "sha256": "44ca413b8c2435a771cd7987b1d9298ab8fce512684533994b5a55c67b539dc3",
        "rows": 300, "sample_start": 0, "samples": 100, "length_bucket": "8k+"},
}
PASSAGE_RETRIEVAL_TEMPLATE = (
    "Here are 30 paragraphs from Wikipedia, along with an abstract.\n"
    "Please determine which paragraph the abstract is from.\n\n"
    "{context}\n\n"
    "The following is an abstract.\n\n"
    "{input}\n\n"
    "Please enter the number of the paragraph that the abstract is from. "
    'The answer format must be like "Paragraph 1", "Paragraph 2", etc.\n\n'
    "The answer is: "
)
HOTPOTQA_TEMPLATE = (
    "Answer the question based on the given passages. "
    "Only give me the answer and do not output any other words.\n\n"
    "The following are given passages.\n"
    "{context}\n\n"
    "Answer the question based on the given passages. "
    "Only give me the answer and do not output any other words.\n\n"
    "Question: {input}\n"
    "Answer:"
)
TEMPLATES = {"passage_retrieval": PASSAGE_RETRIEVAL_TEMPLATE, "hotpotqa": HOTPOTQA_TEMPLATE}


def select_indices(task: str, lengths: list) -> list:
    """Dataset row indices in the accepted order. passage_retrieval: rows [0, 200). hotpotqa: rows with
    length >= 8000 (bucket 8k+), filtered positions [0, 100)."""
    d = DATASETS[task]
    if d["length_bucket"] is None:
        pool = list(range(len(lengths)))
    else:
        pool = [i for i, length in enumerate(lengths) if int(length) >= 8000]
    stop = d["sample_start"] + d["samples"]
    if stop > len(pool):
        raise ValueError(f"{task}: requested [{d['sample_start']}, {stop}) but only {len(pool)} examples")
    return pool[d["sample_start"]:stop]


def user_prompt(task: str, example: dict) -> str:
    return TEMPLATES[task].format(context=example["context"], input=example["input"])


def longbench_prompt_ids(task: str, example: dict, tokenizer) -> tuple:
    """(token ids, original token count): the official template through the tokenizer chat template, tokenized without
    special tokens, truncated to the first 8192 + last 8192 tokens if longer than 16384."""
    prompt = user_prompt(task, example)
    if hasattr(tokenizer, "apply_chat_template"):
        rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                 add_generation_prompt=True)
    else:
        rendered = prompt
    ids = [int(x) for x in tokenizer(rendered, add_special_tokens=False)["input_ids"]]
    return truncate_ids(ids, MAX_INPUT_TOKENS), len(ids)


def truncate_ids(ids: list, max_input_tokens: int) -> list:
    if len(ids) > max_input_tokens:
        first_count = max_input_tokens // 2
        last_count = max_input_tokens - first_count
        return list(ids[:first_count]) + list(ids[-last_count:])
    return list(ids)


def normalize_answers(value) -> list:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except Exception:  # noqa: BLE001
            pass
        return [value]
    return [str(value)]


def retrieval_score(prediction: str, ground_truth: str) -> float:
    """LongBench retrieval score (the accepted Exp12 scorer)."""
    matches = re.findall(r"Paragraph (\d+)", ground_truth)
    if not matches:
        return 0.0
    target_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    correct_numbers = sum(1 for number in numbers if str(number) == str(target_id))
    return float(correct_numbers / len(numbers))


# ---- HotpotQA scorer 1 (PRIMARY): the accepted Exp12 scorer, verbatim, including its historical article behaviour
def normalize_answer_legacy(value: str) -> str:
    def remove_articles(text):
        return re.sub(r"\\b(a|an|the)\\b", " ", text)  # as in Exp12: this pattern matches no article (kept on purpose)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punctuation(text):
        punctuation = set(string.punctuation)
        return "".join(character for character in text if character not in punctuation)

    return white_space_fix(remove_articles(remove_punctuation(value.lower())))


def qa_f1_score_legacy(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer_legacy(prediction).split()
    ground_truth_tokens = normalize_answer_legacy(ground_truth).split()
    if not prediction_tokens or not ground_truth_tokens:
        return float(prediction_tokens == ground_truth_tokens)
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(prediction_tokens)
    recall = same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


# ---- HotpotQA scorer 2 (SECONDARY): standards-aligned LongBench qa_f1_score (THUDM/LongBench metrics.py)
def normalize_answer_official(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def qa_f1_score_official(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer_official(prediction).split()
    ground_truth_tokens = normalize_answer_official(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return float((2 * precision * recall) / (precision + recall))


def score(task: str, prediction: str, answers: list) -> dict:
    """All preregistered scores of one prediction (max over the reference answers)."""
    if task == "passage_retrieval":
        return {"score": max(retrieval_score(prediction, a) for a in answers)}
    if task == "hotpotqa":
        return {"score": max(qa_f1_score_legacy(prediction, a) for a in answers),
                "score_official": max(qa_f1_score_official(prediction, a) for a in answers)}
    raise ValueError(task)


# ------------------------------------------------------------------------------------------------ identity helpers
def ids_sha256(ids: list) -> str:
    """SHA-256 of token ids as little-endian int64."""
    return hashlib.sha256(struct.pack(f"<{len(ids)}q", *[int(x) for x in ids])).hexdigest()
