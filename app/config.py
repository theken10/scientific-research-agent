import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv(
        "DATABASE_URL", "postgresql+psycopg2://research:research@localhost:5432/research"
    )
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection: str = os.getenv("QDRANT_COLLECTION", "papers")

    embed_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    rerank_model: str = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    nli_model: str = os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-small")

    llm_backend: str = os.getenv("LLM_BACKEND", "hf")
    llm_model: str = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "http://localhost:8001/v1")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "450"))

    chunk_words: int = int(os.getenv("CHUNK_WORDS", "200"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "40"))

    plan_with_llm: bool = os.getenv("PLAN_WITH_LLM", "0") == "1"
    faithfulness_threshold: float = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.5"))


settings = Settings()
