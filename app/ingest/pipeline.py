import hashlib
from pathlib import Path

from app.config import settings
from app.db import Chunk, Document, Session, init_db
from app.ingest.chunker import chunk_words
from app.ingest.parser import parse_pdf
from app.retrieval import vector_store
from app.retrieval.embedder import embed_texts


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ingest_pdf(path: str | Path) -> dict:
    path = Path(path)
    digest = _sha256(path)
    init_db()
    vector_store.ensure_collection()

    with Session() as s:
        existing = s.query(Document).filter_by(sha256=digest).first()
        if existing:
            return {"status": "skipped", "filename": existing.filename, "doc_id": existing.id}

        title, pages = parse_pdf(path)
        doc = Document(filename=path.name, title=title, sha256=digest, n_pages=len(pages))
        s.add(doc)
        s.flush()

        rows = []
        for page_no, text in pages:
            for j, piece in enumerate(chunk_words(text, settings.chunk_words, settings.chunk_overlap)):
                rows.append(Chunk(doc_id=doc.id, page=page_no, idx=j, text=piece))
        s.add_all(rows)
        s.flush()  # need the ids

        vectors = embed_texts([r.text for r in rows])
        vector_store.upsert(
            ids=[r.id for r in rows],
            vectors=vectors,
            payloads=[{"doc_id": doc.id, "page": r.page} for r in rows],
        )
        s.commit()
        return {"status": "ingested", "filename": doc.filename, "doc_id": doc.id,
                "pages": len(pages), "chunks": len(rows)}
