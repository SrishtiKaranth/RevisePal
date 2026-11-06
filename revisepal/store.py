"""Persistent vector store (Chroma) holding note chunks and their embeddings."""
from __future__ import annotations

import chromadb

from .config import Settings
from .ingest import Chunk
from .llm import LLM


class NoteStore:
    def __init__(self, settings: Settings, llm: LLM, client=None):
        self.s = settings
        self.llm = llm
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.client = client or chromadb.PersistentClient(path=str(settings.chroma_dir))
        self.col = self.client.get_or_create_collection("notes", metadata={"hnsw:space": "cosine"})

    def add(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        # Re-uploading a file replaces its old chunks
        for source in {c.source for c in chunks}:
            self.delete_source(source)
        self.col.add(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.metadata() for c in chunks],
            embeddings=self.llm.embed([c.text for c in chunks], kind="document"),
        )
        return len(chunks)

    def delete_source(self, source: str):
        self.col.delete(where={"source": source})

    def sources(self) -> dict[str, int]:
        metas = self.col.get(include=["metadatas"])["metadatas"]
        counts: dict[str, int] = {}
        for m in metas:
            counts[m["source"]] = counts.get(m["source"], 0) + 1
        return dict(sorted(counts.items()))

    def search(self, query: str, k: int | None = None, source: str | None = None) -> list[tuple[Chunk, float]]:
        """Return [(chunk, similarity)] best first. similarity = 1 - cosine distance."""
        if self.col.count() == 0:
            return []
        res = self.col.query(
            query_embeddings=self.llm.embed([query], kind="query"),
            n_results=min(k or self.s.top_k, self.col.count()),
            where={"source": source} if source else None,
        )
        return [(_to_chunk(i, d, m), 1 - dist) for i, d, m, dist in
                zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])]

    def all_chunks(self, source: str | None = None) -> list[Chunk]:
        res = self.col.get(where={"source": source} if source else None, include=["documents", "metadatas"])
        return [_to_chunk(i, d, m) for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])]

    def get(self, chunk_id: str) -> Chunk | None:
        res = self.col.get(ids=[chunk_id], include=["documents", "metadatas"])
        return _to_chunk(res["ids"][0], res["documents"][0], res["metadatas"][0]) if res["ids"] else None


def _to_chunk(cid, doc, meta) -> Chunk:
    page = meta.get("page", -1)
    return Chunk(id=cid, text=doc, source=meta["source"], page=None if page == -1 else page,
                 section=meta.get("section", ""))
