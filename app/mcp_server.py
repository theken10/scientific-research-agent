"""MCP server so other agents/clients (Claude Desktop, Cursor, ...) can use the research tools.
Run:  python -m app.mcp_server            (stdio)
      MCP_TRANSPORT=streamable-http python -m app.mcp_server
"""
import os

from mcp.server.fastmcp import FastMCP

from app.db import Chunk, Document, Session

mcp = FastMCP("research-agent")


@mcp.tool()
def search_papers(query: str, top_k: int = 5) -> list[dict]:
    """Hybrid (keyword + semantic) search with reranking over the indexed papers.
    Returns passages with document title, filename and page number."""
    from app.retrieval.hybrid import search
    hits = search(query, top_k=min(top_k, 20))
    return [{k: h[k] for k in ("chunk_id", "title", "filename", "page", "text")} for h in hits]


@mcp.tool()
def ask_research_question(question: str) -> dict:
    """Run the full agent: retrieve, compare sources, answer with citations, then fact-check
    each sentence against its cited source."""
    from app.agent.graph import run_agent
    out = run_agent(question)
    return {
        "answer": out["answer"],
        "sources": out["sources"],
        "supported_ratio": out["checks"].get("supported_ratio"),
        "flagged_claims": [f["sentence"] for f in out["checks"].get("flagged", [])],
    }


@mcp.tool()
def get_passage(chunk_id: int) -> dict:
    """Fetch the full text of one chunk by id (ids come from search_papers)."""
    with Session() as s:
        row = s.query(Chunk, Document).join(Document, Chunk.doc_id == Document.id) \
               .filter(Chunk.id == chunk_id).first()
    if not row:
        return {"error": "chunk not found"}
    c, d = row
    return {"chunk_id": c.id, "title": d.title, "filename": d.filename, "page": c.page, "text": c.text}


@mcp.tool()
def list_documents() -> list[dict]:
    """List all indexed documents."""
    with Session() as s:
        return [{"id": d.id, "filename": d.filename, "title": d.title, "pages": d.n_pages}
                for d in s.query(Document).order_by(Document.id)]


if __name__ == "__main__":
    mcp.run(transport=os.getenv("MCP_TRANSPORT", "stdio"))
