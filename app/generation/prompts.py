SYSTEM = (
    "You are a careful scientific research assistant. Answer ONLY from the numbered "
    "sources provided. After every sentence that states a fact, add the source tag(s) "
    "like [S1] or [S2][S3]. If the sources do not contain the answer, say so plainly. "
    "Do not use outside knowledge."
)


def build_context(hits: list[dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        blocks.append(f"[S{i}] ({h['title'][:80]}, p.{h['page']})\n{h['text']}")
    return "\n\n".join(blocks)


def answer_prompt(question: str, hits: list[dict], strict: bool = False) -> list[dict]:
    extra = ""
    if strict:
        extra = (
            "\nYour previous attempt had claims that the sources do not support. "
            "Only repeat what the sources literally say, keep it short, and cite every sentence."
        )
    user = (
        f"Sources:\n{build_context(hits)}\n\n"
        f"Question: {question}\n\n"
        "Write a clear answer. If several sources describe different methods, compare them "
        "(what is similar, what differs) and cite each one." + extra
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def plan_prompt(question: str) -> list[dict]:
    return [
        {"role": "system", "content": "You write short search queries for a scientific paper index."},
        {"role": "user", "content": (
            "Write up to 3 different short search queries (one per line, no numbering) that would "
            f"help answer this question:\n{question}")},
    ]
