from datetime import datetime

from sqlalchemy import (Column, DateTime, Float, ForeignKey, Integer, String, Text,
                        create_engine, text)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from app.config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True)
Session = sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    filename = Column(String(300), unique=True, nullable=False)
    title = Column(String(500))
    sha256 = Column(String(64), unique=True, nullable=False)
    n_pages = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True)  # same id is used as the qdrant point id
    doc_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    page = Column(Integer, nullable=False)
    idx = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)

    document = relationship("Document", back_populates="chunks")


class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True)
    question = Column(Text)
    answer = Column(Text)
    latency_ms = Column(Float)
    supported_ratio = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS chunks_fts_idx ON chunks "
            "USING GIN (to_tsvector('english', text))"
        ))
