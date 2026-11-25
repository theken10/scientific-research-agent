"""Post-generation checks: is each sentence of the answer backed by the source it cites?"""
import re
from functools import lru_cache

from app.config import settings

CITE = re.compile(r"\[S(\d+)\]")
STOP = set("the and for that with this from are was were been have has into than then also such "
           "which their these those using used use can may not but its our they them more most".split())


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\[])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def lexical_support(premise: str, claim: str) -> float:
    words = [w for w in re.findall(r"[a-z0-9]+", claim.lower()) if len(w) > 3 and w not in STOP]
    if not words:
        return 1.0
    prem = set(re.findall(r"[a-z0-9]+", premise.lower()))
    return sum(w in prem for w in words) / len(words)


@lru_cache(maxsize=1)
def _nli():
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(settings.nli_model)
        model = AutoModelForSequenceClassification.from_pretrained(settings.nli_model).eval()
        ent = [i for i, l in model.config.id2label.items() if l.lower().startswith("entail")][0]
        return tok, model, ent
    except Exception as e:  # no internet / model missing -> fall back to word overlap
        print(f"[checks] NLI model unavailable ({e}), using lexical overlap")
        return None


def entailment_score(premise: str, claim: str) -> float:
    nli = _nli()
    if nli is None:
        return lexical_support(premise, claim)
    import torch
    tok, model, ent = nli
    enc = tok(premise, claim, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)[0]
    return float(probs[ent])


def check_answer(answer: str, sources: dict[int, str], threshold: float | None = None) -> dict:
    """sources maps tag number (the N in [SN]) -> chunk text."""
    threshold = settings.faithfulness_threshold if threshold is None else threshold
    results = []
    for sent in split_sentences(answer):
        tags = [int(t) for t in CITE.findall(sent)]
        claim = CITE.sub("", sent).strip()
        if len(claim.split()) < 5:
            continue  # headings, fragments

        valid = [t for t in tags if t in sources]
        if not tags:
            status, score = "uncited", 0.0
        elif not valid:
            status, score = "bad_citation", 0.0
        else:
            score = max(entailment_score(sources[t], claim) for t in valid)
            status = "supported" if score >= threshold else "unsupported"
        results.append({"sentence": sent, "tags": tags, "status": status, "score": round(score, 3)})

    n = len(results)
    ok = sum(r["status"] == "supported" for r in results)
    return {
        "sentences": results,
        "n_claims": n,
        "supported_ratio": (ok / n) if n else 0.0,
        "flagged": [r for r in results if r["status"] != "supported"],
    }
