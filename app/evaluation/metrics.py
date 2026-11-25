import math


def is_relevant(hit: dict, gold: list[dict]) -> bool:
    """gold items look like {"doc": "1902.07987.pdf", "pages": [3,4]}; empty pages = any page."""
    for g in gold:
        if hit["filename"] == g["doc"] and (not g.get("pages") or hit["page"] in g["pages"]):
            return True
    return False


def recall_at_k(hits: list[dict], gold: list[dict], k: int = 5) -> float:
    """Share of gold items that show up in the top k."""
    if not gold:
        return 0.0
    top = hits[:k]
    found = sum(any(is_relevant(h, [g]) for h in top) for g in gold)
    return found / len(gold)


def reciprocal_rank(hits: list[dict], gold: list[dict]) -> float:
    for rank, h in enumerate(hits, start=1):
        if is_relevant(h, gold):
            return 1.0 / rank
    return 0.0


def citation_precision(cited_sources: list[dict], gold: list[dict]) -> float | None:
    """Of the sources the answer actually cited, how many point at the right doc/page."""
    if not cited_sources:
        return None
    good = sum(is_relevant(s, gold) for s in cited_sources)
    return good / len(cited_sources)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = max(0, min(len(vals) - 1, math.ceil(p / 100 * len(vals)) - 1))
    return vals[idx]


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0


def hallucination_rates(checks_list: list[dict]) -> dict:
    """Sentence level: share of claims that are not supported.
    Answer level: share of answers with at least one unsupported claim."""
    total_claims = sum(c["n_claims"] for c in checks_list)
    bad_claims = sum(len(c["flagged"]) for c in checks_list)
    bad_answers = sum(1 for c in checks_list if c["flagged"])
    return {
        "sentence_level": bad_claims / total_claims if total_claims else 0.0,
        "answer_level": bad_answers / len(checks_list) if checks_list else 0.0,
    }
