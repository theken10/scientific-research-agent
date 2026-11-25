from functools import lru_cache

from qdrant_client import QdrantClient, models

from app.config import settings
from app.retrieval.embedder import dim


@lru_cache(maxsize=1)
def client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def ensure_collection():
    c = client()
    existing = [col.name for col in c.get_collections().collections]
    if settings.collection not in existing:
        c.create_collection(
            collection_name=settings.collection,
            vectors_config=models.VectorParams(size=dim(), distance=models.Distance.COSINE),
        )


def upsert(ids, vectors, payloads, batch=128):
    c = client()
    for i in range(0, len(ids), batch):
        points = [
            models.PointStruct(id=ids[k], vector=vectors[k], payload=payloads[k])
            for k in range(i, min(i + batch, len(ids)))
        ]
        c.upsert(collection_name=settings.collection, points=points)


def search(vector, k=20):
    res = client().query_points(collection_name=settings.collection, query=vector, limit=k)
    return [(p.id, p.score) for p in res.points]
