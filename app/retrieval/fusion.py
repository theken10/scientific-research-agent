def rrf(rankings: list[list[int]], k: int = 60):
    """Reciprocal rank fusion. Each ranking is a list of ids, best first.
    Returns [(id, fused_score)] sorted best first."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for pos, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + pos)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
