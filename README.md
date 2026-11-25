# Scientific Research Agent

An agentic retrieval-augmented generation (RAG) prototype for exploring scientific papers with source-grounded answers and page-level citations.

## Overview

This project combines dense vector retrieval and PostgreSQL full-text search, Reciprocal Rank Fusion (RRF), cross-encoder reranking, answer generation, and citation/claim verification. A LangGraph workflow coordinates planning, retrieval, synthesis, and verification.

## Architecture

1. Ingest scientific PDFs and split their text into chunks with page metadata.
2. Index chunks in Qdrant for dense retrieval and PostgreSQL for keyword search.
3. Fuse retrieval results with RRF and rerank candidate passages.
4. Generate an answer with citations to retrieved sources.
5. Check answer claims against cited passages and retry once when unsupported claims are detected.

## Stack

- Python, FastAPI, LangGraph
- Qdrant and PostgreSQL full-text search
- Sentence Transformers and cross-encoder reranking
- Hugging Face / OpenAI-compatible LLM endpoints
- PyMuPDF, Docker Compose, MCP

## Setup

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and configure the model endpoint and service settings. Never commit `.env` or credentials.
3. Start the supporting services:

   ```bash
   docker compose up -d
   ```

4. Use the scripts in `scripts/` to download sample papers and ingest a PDF directory. Run the API using the command documented in `app/api/main.py` or the Makefile.

## Evaluation

The repository includes an evaluation harness and a small starter QA dataset. The starter set is for smoke-testing the evaluation pipeline only; it is not a statistically meaningful benchmark. Build a larger manually verified dataset, check the cited PDF pages, and report retrieval/citation/faithfulness metrics only after running that evaluation.

## Current limitations

- PDF text extraction can lose structure in tables, equations, and multi-column layouts.
- Page metadata represents extracted PDF page indexes and may not match printed page labels.
- The lightweight NLI/overlap-based verifier can produce false positives and false negatives; it is an experimental guardrail, not proof of factual correctness.
- The included QA set is small and requires expansion and manual ground-truth verification.
- Model quality and latency depend on the configured inference endpoint and hardware.

## Project status

Research prototype. It is not presented as a production-ready or scientifically validated research system. Evaluate it on real papers and a manually checked benchmark before making quantitative performance claims.
