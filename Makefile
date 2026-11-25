up:
	docker compose up --build -d

down:
	docker compose down

papers:
	python scripts/download_papers.py

ingest:
	docker compose exec api python scripts/ingest_dir.py data/pdfs

eval:
	docker compose exec api python -m app.evaluation.run_eval --data app/evaluation/qa_dataset.jsonl --generate

test:
	pytest -q tests

mcp:
	python -m app.mcp_server
