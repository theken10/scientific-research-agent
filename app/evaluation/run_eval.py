"""Usage:
    python -m app.evaluation.run_eval --data app/evaluation/qa_dataset.jsonl            # retrieval only
    python -m app.evaluation.run_eval --data app/evaluation/qa_dataset.jsonl --generate # + answer checks
"""
import argparse
import json
import time
from pathlib import Path

from app.evaluation import metrics
from app.retrieval.hybrid import search


def load(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--generate", action="store_true", help="also run the full agent")
    ap.add_argument("--out", default="eval_results/report.json")
    ap.add_argument("--no-rerank", action="store_true")
    args = ap.parse_args()

    rows = load(args.data)
    per_q, retr_lat = [], []
    checks_list, cite_prec, faith, agent_lat = [], [], [], []

    for r in rows:
        t0 = time.perf_counter()
        hits = search(r["question"], top_k=10, use_rerank=not args.no_rerank)
        lat = (time.perf_counter() - t0) * 1000
        retr_lat.append(lat)

        entry = {
            "id": r["id"],
            "recall@5": metrics.recall_at_k(hits, r["gold"], 5),
            "rr": metrics.reciprocal_rank(hits, r["gold"]),
            "retrieval_ms": round(lat, 1),
        }

        if args.generate:
            from app.agent.graph import run_agent
            out = run_agent(r["question"])
            cited = [s for s in out["sources"] if s["cited"]]
            cp = metrics.citation_precision(cited, r["gold"])
            entry["citation_precision"] = cp
            entry["supported_ratio"] = out["checks"].get("supported_ratio", 0.0)
            entry["agent_ms"] = out["latency_ms"]
            entry["answer"] = out["answer"]
            checks_list.append(out["checks"])
            cite_prec.append(cp)
            faith.append(entry["supported_ratio"])
            agent_lat.append(out["latency_ms"])
        per_q.append(entry)
        print(f"{r['id']}: recall@5={entry['recall@5']:.2f} rr={entry['rr']:.2f} {lat:.0f}ms")

    summary = {
        "n_questions": len(rows),
        "recall@5": round(metrics.mean([e["recall@5"] for e in per_q]), 3),
        "MRR": round(metrics.mean([e["rr"] for e in per_q]), 3),
        "retrieval_latency_ms_p50": round(metrics.percentile(retr_lat, 50), 1),
        "retrieval_latency_ms_p95": round(metrics.percentile(retr_lat, 95), 1),
    }
    if args.generate:
        summary["citation_correctness"] = round(metrics.mean(cite_prec), 3)
        summary["answer_faithfulness"] = round(metrics.mean(faith), 3)
        summary["hallucination_rate"] = {
            k: round(v, 3) for k, v in metrics.hallucination_rates(checks_list).items()
        }
        summary["end_to_end_latency_ms_p50"] = round(metrics.percentile(agent_lat, 50), 1)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "per_question": per_q}, indent=2))
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
