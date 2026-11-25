def chunk_words(text: str, size: int = 200, overlap: int = 40):
    """Sliding window over words. Never crosses page boundaries (caller does it per page)
    so every chunk maps to exactly one page number."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = text.split()
    if not words:
        return []
    step = size - overlap
    out = []
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        # tiny tail chunk adds nothing, merge it into the previous one
        if out and len(piece) < overlap + 10:
            break
        out.append(" ".join(piece))
        if start + size >= len(words):
            break
    return out
