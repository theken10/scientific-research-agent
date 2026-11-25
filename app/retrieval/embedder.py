from functools import lru_cache

from app.config import settings


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(settings.embed_model)


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    vecs = _model().encode(texts, batch_size=batch_size, normalize_embeddings=True,
                           show_progress_bar=False)
    return vecs.tolist()


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]


def dim() -> int:
    return _model().get_sentence_embedding_dimension()
