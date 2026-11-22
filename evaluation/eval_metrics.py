"""
Visual-Doc evaluation helpers -- shared by Evaluation 1 and Evaluation 2.

Both retrieval arms (ColPali + MaxSim now, OCR/text RAG later) must be scored by
IDENTICAL code, otherwise a difference in the numbers could come from the
metric implementation instead of the retrieval method. Everything that turns a
ranked list into a number, or a model answer into a verdict, lives here.

Retrieval convention: every question has exactly ONE gold page, identified as
(doc_name, page_number). `rank_of_gold` is its 1-based position in the ranked
list (None if it is not in the top max_k).
"""
import math
import re

import pandas as pd

KS = (1, 3, 5, 10)
MAX_K = 10


# --------------------------------------------------------------------------
# Retrieval metrics
# --------------------------------------------------------------------------
def rank_of_gold(ranked_pairs, gold, max_k=MAX_K):
    """1-based rank of `gold` within the first max_k (doc_name, page) pairs, else None."""
    for i, pair in enumerate(list(ranked_pairs)[:max_k]):
        if tuple(pair) == tuple(gold):
            return i + 1
    return None


def _clean_rank(r):
    if r is None:
        return None
    if isinstance(r, float) and math.isnan(r):
        return None
    return int(r)


def hit_at_k(rank, k):
    rank = _clean_rank(rank)
    return rank is not None and rank <= k


def reciprocal_rank(rank, max_k=MAX_K):
    rank = _clean_rank(rank)
    return 1.0 / rank if rank is not None and rank <= max_k else 0.0


def ndcg_at_k(rank, k=5):
    """nDCG@k with ONE relevant page: DCG = 1/log2(rank+1) if rank<=k, IDCG = 1."""
    rank = _clean_rank(rank)
    return 1.0 / math.log2(rank + 1) if rank is not None and rank <= k else 0.0


def metrics_from_ranks(ranks):
    ranks = [_clean_rank(r) for r in ranks]
    n = len(ranks)
    out = {"n": n}
    for k in KS:
        out[f"Recall@{k}"] = sum(hit_at_k(r, k) for r in ranks) / n if n else float("nan")
    out["MRR@10"] = sum(reciprocal_rank(r) for r in ranks) / n if n else float("nan")
    out["nDCG@5"] = sum(ndcg_at_k(r, 5) for r in ranks) / n if n else float("nan")
    return out


def metrics_table(frame, group_col=None, rank_col="rank_of_gold"):
    """Recall@1/3/5/10, MRR@10 and nDCG@5, overall (group_col=None) or per group."""
    if group_col is None:
        return pd.DataFrame([metrics_from_ranks(frame[rank_col].tolist())], index=["ALL"])
    rows = {g: metrics_from_ranks(sub[rank_col].tolist()) for g, sub in frame.groupby(group_col)}
    return pd.DataFrame(rows).T.astype({"n": int})


# --------------------------------------------------------------------------
# Answer scoring
# --------------------------------------------------------------------------
def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9.% ]", " ", text)
    text = re.sub(r" +", " ", text)
    return text.strip()


def extract_keyphrases(gold_answer):
    """Verbatim copy of the Evaluation 1 notebook's helper (kept identical on purpose)."""
    numbers = re.findall(r"\d+(?:\.\d+)?%?", gold_answer)
    proper_nouns = re.findall(r"\b[A-Z][a-zA-Z-]+\b", gold_answer)
    seen = []
    for kp in numbers + proper_nouns:
        if kp not in seen:
            seen.append(kp)
    return seen if seen else [gold_answer]


def keyword_grounded_score(model_answer, gold_answer):
    """Verbatim copy of the ORIGINAL Evaluation 1 scorer (lenient: needs half the key phrases)."""
    norm_answer = normalize(model_answer)
    keyphrases = extract_keyphrases(gold_answer)
    hits = 0
    for kp in keyphrases:
        if normalize(kp) in norm_answer:
            hits += 1
    return hits >= max(1, len(keyphrases) // 2)


# Phrases a model uses when it says the page does not contain the answer.
ABSTAIN_PATTERNS = [
    r"\bnot (?:included|contained|present|shown|visible|available|provided|found|mentioned|listed|displayed)\b",
    r"\b(?:does|do|did)(?: not|n't) (?:contain|include|show|provide|mention|list|have|display)\b",
    r"\b(?:cannot|can't|can not|could not|couldn't|unable to) (?:find|answer|be answered|determine|locate|identify)\b",
    r"\bno (?:information|mention|data|reference)\b",
    r"\bnot (?:enough|sufficient) information\b",
    r"\bnot (?:on|in) (?:this|the|its|the provided|the given) (?:page|image|document)\b",
    r"\bis not present\b|\bare not present\b",
]


def is_abstention(model_answer):
    """True if the answer says the provided page(s) do not contain the answer."""
    a = model_answer.lower()
    return any(re.search(p, a) for p in ABSTAIN_PATTERNS)


def numbers_all_present(model_answer, gold_answer):
    """Every number in the gold answer must appear (as a whole number) in the answer."""
    nums = re.findall(r"\d+(?:\.\d+)?", gold_answer)
    if not nums:
        return True
    norm = normalize(model_answer)
    for n in nums:
        if not re.search(r"(?<![\d.])" + re.escape(n) + r"(?!\d|\.\d)", norm):
            return False
    return True


def strict_keyword_score(model_answer, gold_answer):
    """Provisional automatic verdict: NOT an abstention, ALL gold numbers present,
    and the original keyword check also passes. Tighter than keyword_grounded_score
    (which could pass an answer like 'Table 2 is not included on this page' because
    generic words such as 'document' or 'retrieval' matched)."""
    return (
        (not is_abstention(model_answer))
        and numbers_all_present(model_answer, gold_answer)
        and keyword_grounded_score(model_answer, gold_answer)
    )
