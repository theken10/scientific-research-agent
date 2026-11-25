import re
import time
from typing import TypedDict

from langgraph.graph import END, StateGraph

from app.config import settings
from app.evaluation.checks import CITE, check_answer
from app.generation.llm import generate
from app.generation.prompts import answer_prompt, plan_prompt
from app.retrieval.hybrid import search

MAX_RETRIES = 1
MIN_SUPPORTED = 0.7
TOP_K = 6


class AgentState(TypedDict, total=False):
    question: str
    queries: list[str]
    hits: list[dict]
    answer: str
    checks: dict
    retries: int
    timings: dict


def plan(state: AgentState) -> AgentState:
    q = state["question"]
    queries = [q]
    if settings.plan_with_llm:
        try:
            raw = generate(plan_prompt(q))
            extra = [l.strip("-• \t") for l in raw.splitlines() if len(l.strip()) > 5][:3]
            queries += extra
        except Exception as e:
            print("[plan] llm failed, using the raw question:", e)
    return {"queries": queries, "retries": 0, "timings": {}}


def retrieve(state: AgentState) -> AgentState:
    t0 = time.perf_counter()
    merged: dict[int, dict] = {}
    for q in state["queries"]:
        for h in search(q, top_k=TOP_K):
            old = merged.get(h["chunk_id"])
            if old is None or h.get("rerank_score", 0) > old.get("rerank_score", 0):
                merged[h["chunk_id"]] = h
    hits = sorted(merged.values(), key=lambda h: h.get("rerank_score", 0), reverse=True)[:TOP_K]
    timings = dict(state.get("timings", {}), retrieval_ms=(time.perf_counter() - t0) * 1000)
    return {"hits": hits, "timings": timings}


def organise(state: AgentState) -> AgentState:
    """Group evidence by document so the model sees sources side by side and can compare them.
    Keeps rank order inside each doc, docs ordered by their best chunk."""
    hits = state["hits"]
    order: list[int] = []
    for h in hits:
        if h["doc_id"] not in order:
            order.append(h["doc_id"])
    grouped = sorted(hits, key=lambda h: (order.index(h["doc_id"]), h["page"]))
    return {"hits": grouped}


def generate_answer(state: AgentState) -> AgentState:
    t0 = time.perf_counter()
    if not state["hits"]:
        return {"answer": "I could not find anything relevant in the indexed documents."}
    strict = state.get("retries", 0) > 0
    answer = generate(answer_prompt(state["question"], state["hits"], strict=strict))
    timings = dict(state.get("timings", {}))
    timings["generation_ms"] = timings.get("generation_ms", 0) + (time.perf_counter() - t0) * 1000
    return {"answer": answer, "timings": timings}


def verify(state: AgentState) -> AgentState:
    t0 = time.perf_counter()
    sources = {i: h["text"] for i, h in enumerate(state["hits"], start=1)}
    checks = check_answer(state["answer"], sources)
    timings = dict(state.get("timings", {}), verify_ms=(time.perf_counter() - t0) * 1000)
    retries = state.get("retries", 0)
    return {"checks": checks, "timings": timings, "retries": retries + 1}


def route(state: AgentState) -> str:
    checks = state["checks"]
    needs_retry = checks["n_claims"] > 0 and checks["supported_ratio"] < MIN_SUPPORTED
    if needs_retry and state["retries"] <= MAX_RETRIES:
        return "retry"
    return "done"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("plan", plan)
    g.add_node("retrieve", retrieve)
    g.add_node("organise", organise)
    g.add_node("generate", generate_answer)
    g.add_node("verify", verify)

    g.set_entry_point("plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "organise")
    g.add_edge("organise", "generate")
    g.add_edge("generate", "verify")
    g.add_conditional_edges("verify", route, {"retry": "generate", "done": END})
    return g.compile()


_graph = None


def run_agent(question: str) -> dict:
    global _graph
    if _graph is None:
        _graph = build_graph()
    t0 = time.perf_counter()
    state = _graph.invoke({"question": question})
    total_ms = (time.perf_counter() - t0) * 1000

    used = {int(t) for t in CITE.findall(state.get("answer", ""))}
    sources = []
    for i, h in enumerate(state.get("hits", []), start=1):
        sources.append({
            "tag": f"S{i}", "title": h["title"], "filename": h["filename"], "page": h["page"],
            "cited": i in used, "snippet": h["text"][:300],
        })
    return {
        "question": question,
        "answer": state.get("answer", ""),
        "sources": sources,
        "checks": state.get("checks", {}),
        "queries": state.get("queries", []),
        "latency_ms": round(total_ms, 1),
        "timings": {k: round(v, 1) for k, v in state.get("timings", {}).items()},
    }
