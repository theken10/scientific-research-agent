import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app.evaluation import metrics
from app.evaluation.checks import lexical_support, split_sentences
from app.ingest.chunker import chunk_words
from app.retrieval.fusion import rrf


def test_chunker_overlap_and_coverage():
    text = " ".join(f"w{i}" for i in range(500))
    chunks = chunk_words(text, size=100, overlap=20)
    assert chunks[0].split()[0] == "w0"
    assert chunks[1].split()[0] == "w80"
    assert chunks[-1].split()[-1] == "w499"
    assert all(len(c.split()) <= 100 for c in chunks)


def test_chunker_bad_overlap():
    with pytest.raises(ValueError):
        chunk_words("a b c", size=5, overlap=5)


def test_chunker_empty():
    assert chunk_words("   ") == []


def test_rrf_prefers_docs_in_both_lists():
    fused = rrf([[1, 2, 3], [3, 4, 1]])
    ids = [i for i, _ in fused]
    assert set(ids[:2]) == {1, 3}


def test_recall_and_mrr():
    hits = [{"filename": "a.pdf", "page": 1}, {"filename": "b.pdf", "page": 2}]
    gold = [{"doc": "b.pdf", "pages": [2]}]
    assert metrics.recall_at_k(hits, gold, 5) == 1.0
    assert metrics.recall_at_k(hits, gold, 1) == 0.0
    assert metrics.reciprocal_rank(hits, gold) == 0.5


def test_citation_precision():
    gold = [{"doc": "a.pdf", "pages": []}]
    cited = [{"filename": "a.pdf", "page": 3}, {"filename": "x.pdf", "page": 1}]
    assert metrics.citation_precision(cited, gold) == 0.5
    assert metrics.citation_precision([], gold) is None


def test_hallucination_rates():
    checks = [
        {"n_claims": 4, "flagged": [1]},
        {"n_claims": 2, "flagged": []},
    ]
    r = metrics.hallucination_rates(checks)
    assert r["sentence_level"] == pytest.approx(1 / 6)
    assert r["answer_level"] == 0.5


def test_percentile():
    assert metrics.percentile([10, 20, 30, 40, 50], 50) == 30
    assert metrics.percentile([], 95) == 0.0


def test_sentence_split_and_lexical():
    s = split_sentences("Graphs are used here [S1]. Another claim follows [S2].")
    assert len(s) == 2
    assert lexical_support("graph networks cluster calorimeter hits", "calorimeter hits clustering") > 0.3
    assert lexical_support("nothing in common", "calorimeter hits clustering") == 0.0
