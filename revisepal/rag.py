"""Grounded question answering over your notes."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .ingest import Chunk
from .llm import LLM
from .store import NoteStore

NOT_FOUND = "I couldn't find that in your notes."

SYSTEM = f"""You are a study assistant. Answer the student's question using ONLY the numbered
excerpts from their class notes below.
- Cite every claim with the excerpt number in square brackets, e.g. [1] or [2][3].
- If the excerpts do not contain the answer, reply exactly: "{NOT_FOUND}"
  Do not use outside knowledge to fill gaps.
- Be concise and clear, like a good tutor. Use short bullet points for lists or steps."""


@dataclass
class Answer:
    text: str
    sources: list[tuple[Chunk, float]]  # the excerpts shown to the model, numbered from 1
    cited: list[int]                    # excerpt numbers actually cited in the answer

    @property
    def found(self) -> bool:
        return NOT_FOUND.lower() not in self.text.lower()


def format_context(hits: list[tuple[Chunk, float]]) -> str:
    return "\n\n".join(f"[{i}] ({c.citation()})\n{c.text}" for i, (c, _) in enumerate(hits, 1))


def answer(question: str, store: NoteStore, llm: LLM, k: int | None = None,
           min_similarity: float = 0.0, source: str | None = None) -> Answer:
    hits = [h for h in store.search(question, k=k, source=source) if h[1] >= min_similarity]
    if not hits:
        return Answer(NOT_FOUND, [], [])
    user = f"Excerpts from my notes:\n\n{format_context(hits)}\n\nQuestion: {question}"
    text = llm.chat(SYSTEM, user).strip()
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(hits)})
    return Answer(text, hits, cited)
