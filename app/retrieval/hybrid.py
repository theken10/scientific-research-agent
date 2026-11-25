from app.db import Chunk, Document, Session
from app.retrieval.embedder import embed_query
from app.retrieval.fusion import rrf
from app.retrieval.keyword import keyword_search
from app.retrieval.reranker import rerank
from app.retrieval import vector_store


def load_chunks(ids: list[int]) -> list[dict]:
    if not ids:
        return []
    with Session() as s:
        rows = (
            s.query(Chunk, Document)
            .join(Document, Chunk.doc_id == Document.id)
            .filter(Chunk.id.in_(ids))
            .all()
        )
    by_id = {
        c.id: {
            "chunk_id": c.id, "doc_id": d.id, "title": d.title, "filename": d.filename,
            "page": c.page, "text": c.text,
        }
        for c, d in rows
    }
    return [by_id[i] for i in ids if i in by_id]


def search(query: str, top_k: int = 5, candidates: int = 20, use_rerank: bool = True) -> list[dict]:
    dense = vector_store.search(embed_query(query), candidates)
    sparse = keyword_search(query, candidates)

    dense_rank = {cid: r for r, (cid, _) in enumerate(dense, 1)}
    sparse_rank = {cid: r for r, (cid, _) in enumerate(sparse, 1)}

    fused = rrf([[c for c, _ in dense], [c for c, _ in sparse]])[:candidates]
    hits = load_chunks([cid for cid, _ in fused])
    fused_score = dict(fused)
    for h in hits:
        h["fused_score"] = fused_score[h["chunk_id"]]
        h["dense_rank"] = dense_rank.get(h["chunk_id"])
        h["sparse_rank"] = sparse_rank.get(h["chunk_id"])

    if use_rerank:
        return rerank(query, hits, top_k)
    return hits[:top_k]
