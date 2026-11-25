# Scientific Research Agent

RAG + agent system for searching scientific papers, comparing sources, and answering with
page-level citations. Every answer is fact-checked sentence by sentence against the passage
it cites, and unsupported claims get flagged.

Example question: *"Explain the machine learning methods used in particle event reconstruction."*

## How it works

```
question
  -> plan        (original question, optionally LLM-written sub-queries)
  -> retrieve    (Qdrant dense + Postgres full-text, fused with RRF, cross-encoder rerank)
  -> organise    (group passages by paper so the model can compare them)
  -> generate    (LLM, must cite [S1], [S2]... after each claim)
  -> verify      (NLI check per sentence: supported / unsupported / uncited / bad_citation)
        |-- too many unsupported claims? -> regenerate once with a stricter prompt
  -> answer + sources (title, file, page) + flagged claims
```

| Part | Tech |
|---|---|
| API | FastAPI |
| Agent workflow | LangGraph |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector DB | Qdrant |
| Keyword search + metadata | PostgreSQL (GIN full-text index) |
| Reranker | cross-encoder ms-marco-MiniLM |
| LLM | Hugging Face Transformers (Qwen2.5-1.5B-Instruct default) or any OpenAI-compatible endpoint |
| Faithfulness check | NLI cross-encoder (deberta-v3-small), falls back to word overlap |
| Tool interface | MCP server |
| Deploy | Docker Compose |

## Run it

```bash
cp .env.example .env
docker compose up --build -d          # postgres + qdrant + api on :8000

python scripts/download_papers.py     # 5 arxiv papers into data/pdfs
docker compose exec api python scripts/ingest_dir.py data/pdfs
```

First run downloads the HF models, so give it a few minutes.

```bash
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "Explain the machine learning methods used in particle event reconstruction."}'
```

Other endpoints: `POST /ingest` (upload pdf), `POST /search`, `GET /documents`, `GET /health`.
Swagger UI: http://localhost:8000/docs

CPU-only is fine but the 1.5B model is slow (~20-60 s per answer). With a GPU it's a few seconds.
To use a bigger model behind vLLM or an institute LLM proxy, set `LLM_BACKEND=openai`,
`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` in `.env`.

## MCP server

```bash
python -m app.mcp_server
```

Tools: `search_papers`, `ask_research_question`, `get_passage`, `list_documents`.

Claude Desktop config example:
```json
{
  "mcpServers": {
    "research-agent": {
      "command": "python",
      "args": ["-m", "app.mcp_server"],
      "cwd": "/path/to/sci-research-agent"
    }
  }
}
```
(needs Postgres and Qdrant reachable, so `docker compose up -d postgres qdrant` first.)

## Evaluation

`app/evaluation/qa_dataset.jsonl` has 5 starter questions. **You must verify them by hand**:
open each PDF, note the pages that really answer the question, fill in `"pages"`, and grow the
set to 30+ questions. Don't report numbers from the starter file as results.

```bash
docker compose exec api python -m app.evaluation.run_eval \
    --data app/evaluation/qa_dataset.jsonl --generate
```
Writes `eval_results/report.json`. Metrics:

- **Recall@5, MRR** - does retrieval surface the gold doc/page
- **Citation correctness** - share of cited sources that point to the gold doc/page
- **Answer faithfulness** - mean share of claims entailed by their cited passage (NLI)
- **Hallucination rate** - sentence-level (unsupported claims / all claims) and answer-level
- **Latency** - retrieval p50/p95, end-to-end p50

Add `--no-rerank` to measure what reranking gives you; that comparison makes a good table
for your write-up.

## Tests

```bash
pip install pytest && pytest -q tests
```
Covers chunking, RRF fusion, and the metric functions (no models/DB needed).

## Known limitations

- Faithfulness uses a small NLI model, so it has errors in both directions. Spot-check flagged claims.
- PDF parsing is plain text extraction; tables, equations and two-column layouts can get messy.
- Page numbers are PDF page indexes, not the printed page numbers in the paper.
- Small local LLMs sometimes skip citation tags; those sentences show up as `uncited`.

## Ideas to extend

- Section-aware chunking (use headings from PyMuPDF blocks)
- Qdrant sparse vectors instead of Postgres FTS
- Streaming answers, simple web UI
- Compare embedding models (bge, e5) with the eval script
