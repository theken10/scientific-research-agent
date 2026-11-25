import re

from sqlalchemy import text

from app.db import Session


def _or_query(q: str) -> str:
    terms = [t.lower() for t in re.findall(r"[A-Za-z0-9]+", q) if len(t) > 2]
    # dedupe, keep order
    seen, uniq = set(), []
    for t in terms:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return " | ".join(uniq)


def keyword_search(query: str, k: int = 20):
    """Postgres full text search. OR-ed terms because questions are long and an AND query
    would almost never match anything."""
    tsq = _or_query(query)
    if not tsq:
        return []
    sql = text("""
        SELECT id, ts_rank_cd(to_tsvector('english', text), q) AS score
        FROM chunks, to_tsquery('english', :q) AS q
        WHERE to_tsvector('english', text) @@ q
        ORDER BY score DESC
        LIMIT :k
    """)
    with Session() as s:
        rows = s.execute(sql, {"q": tsq, "k": k}).all()
    return [(r.id, float(r.score)) for r in rows]
