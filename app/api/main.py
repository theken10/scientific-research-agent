import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.db import Document, QueryLog, Session, init_db

DATA_DIR = Path("data/pdfs")


@asynccontextmanager
async def lifespan(app: FastAPI):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    init_db()
    yield


app = FastAPI(title="Scientific Research Agent", lifespan=lifespan)


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(5, ge=1, le=20)
    rerank: bool = True


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/documents")
def documents():
    with Session() as s:
        docs = s.query(Document).order_by(Document.id).all()
        return [{"id": d.id, "filename": d.filename, "title": d.title, "pages": d.n_pages} for d in docs]


@app.post("/ingest")
def ingest(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "only pdf files are supported")
    dest = DATA_DIR / Path(file.filename).name
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    from app.ingest.pipeline import ingest_pdf
    return ingest_pdf(dest)


@app.post("/search")
def search_endpoint(req: SearchRequest):
    from app.retrieval.hybrid import search
    t0 = time.perf_counter()
    hits = search(req.query, top_k=req.top_k, use_rerank=req.rerank)
    return {"latency_ms": round((time.perf_counter() - t0) * 1000, 1), "results": hits}


@app.post("/ask")
def ask(req: AskRequest):
    from app.agent.graph import run_agent
    result = run_agent(req.question)
    with Session() as s:
        s.add(QueryLog(question=req.question, answer=result["answer"],
                       latency_ms=result["latency_ms"],
                       supported_ratio=result["checks"].get("supported_ratio")))
        s.commit()
    return result
