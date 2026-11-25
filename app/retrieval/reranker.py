from functools import lru_cache

from app.config import settings


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(settings.rerank_model)


def rerank(query: str, hits: list[dict], top_k: int = 5) -> list[dict]:
    if not hits:
        return hits
    scores = _model().predict([(query, h["text"]) for h in hits])
    for h, sc in zip(hits, scores):
        h["rerank_score"] = float(sc)
    return sorted(hits, key=lambda h: h["rerank_score"], reverse=True)[:top_k]
