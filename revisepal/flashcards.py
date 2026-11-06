"""Generate flashcards grounded in your notes, and review them with spaced repetition.

Grounding: the model must return, for every card, a short verbatim "evidence" quote
from the note chunk it was given. Cards whose evidence can't be found in that chunk
are rejected, so every saved card can be traced back to a specific place in the notes.
"""
from __future__ import annotations

import json
import random
import re
import uuid
from dataclasses import dataclass, asdict, field
from datetime import date, timedelta
from difflib import SequenceMatcher
from pathlib import Path

from .ingest import Chunk
from .llm import LLM
from .store import NoteStore

SYSTEM = """You write flashcards for a student from an excerpt of their class notes.
Rules:
- Use ONLY information stated in the excerpt. No outside facts.
- Each card tests one idea: a definition, a cause/effect, a comparison, a step, a formula.
- Questions must make sense on their own (no "according to the excerpt").
- Answers are short: one sentence or a few words.
- "evidence" is an exact quote (copied word for word, max ~25 words) from the excerpt that supports the answer.
- Skip trivia (dates of the lecture, page numbers, admin details).
Return JSON: {"cards": [{"question": "...", "answer": "...", "evidence": "..."}]}"""

# Leitner boxes: a card moves up a box when you get it right, back to box 1 when wrong.
INTERVAL_DAYS = {1: 0, 2: 1, 3: 3, 4: 7, 5: 14}


@dataclass
class Card:
    question: str
    answer: str
    evidence: str
    chunk_id: str
    source: str
    citation: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    box: int = 1
    due: str = field(default_factory=lambda: date.today().isoformat())
    reviews: int = 0
    correct: int = 0


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


def grounding_score(evidence: str, chunk_text: str) -> float:
    """Fraction of the evidence quote found verbatim (after normalisation) in the chunk."""
    ev, ch = _norm(evidence), _norm(chunk_text)
    if not ev:
        return 0.0
    if ev in ch:
        return 1.0
    m = SequenceMatcher(None, ch, ev, autojunk=False).find_longest_match(0, len(ch), 0, len(ev))
    return m.size / len(ev)


def cards_from_chunk(chunk: Chunk, llm: LLM, n: int = 3, min_grounding: float = 0.8) -> tuple[list[Card], int]:
    """Return (grounded cards, number rejected)."""
    user = f"Write up to {n} flashcards.\n\nExcerpt ({chunk.citation()}):\n{chunk.text}"
    try:
        data = llm.chat_json(SYSTEM, user)
    except ValueError:
        return [], 0
    raw = data.get("cards", []) if isinstance(data, dict) else []
    kept, rejected = [], 0
    for c in raw[:n]:
        if not isinstance(c, dict) or not c.get("question") or not c.get("answer"):
            rejected += 1
            continue
        if grounding_score(str(c.get("evidence", "")), chunk.text) < min_grounding:
            rejected += 1
            continue
        kept.append(Card(question=str(c["question"]).strip(), answer=str(c["answer"]).strip(),
                         evidence=str(c["evidence"]).strip(), chunk_id=chunk.id,
                         source=chunk.source, citation=chunk.citation()))
    return kept, rejected


def pick_chunks(store: NoteStore, n_chunks: int, topic: str = "", source: str | None = None,
                seed: int | None = None) -> list[Chunk]:
    """Topic given -> most relevant chunks; otherwise a random spread across the notes."""
    if topic.strip():
        return [c for c, _ in store.search(topic, k=n_chunks, source=source)]
    chunks = [c for c in store.all_chunks(source) if len(c.text) > 200]
    random.Random(seed).shuffle(chunks)
    return chunks[:n_chunks]


def generate(store: NoteStore, llm: LLM, n_cards: int = 10, topic: str = "", source: str | None = None,
             per_chunk: int = 2, progress=None) -> tuple[list[Card], dict]:
    chunks = pick_chunks(store, max(1, -(-n_cards // per_chunk)), topic, source)
    cards, rejected = [], 0
    for i, ch in enumerate(chunks):
        got, rej = cards_from_chunk(ch, llm, n=per_chunk)
        cards.extend(got)
        rejected += rej
        if progress:
            progress((i + 1) / len(chunks))
    cards = _dedupe(cards)[:n_cards]
    total = len(cards) + rejected
    stats = {"chunks": len(chunks), "kept": len(cards), "rejected": rejected,
             "grounded_rate": round(len(cards) / total, 3) if total else 0.0}
    return cards, stats


def _dedupe(cards: list[Card]) -> list[Card]:
    seen, out = set(), []
    for c in cards:
        key = _norm(c.question)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


class Deck:
    """Flashcards saved to a JSON file, with Leitner spaced repetition."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.cards: list[Card] = []
        if self.path.exists():
            self.cards = [Card(**c) for c in json.loads(self.path.read_text())]

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([asdict(c) for c in self.cards], indent=2))

    def add(self, cards: list[Card]):
        existing = {_norm(c.question) for c in self.cards}
        self.cards.extend(c for c in cards if _norm(c.question) not in existing)
        self.save()

    def due(self, today: date | None = None, source: str | None = None) -> list[Card]:
        today = (today or date.today()).isoformat()
        return sorted((c for c in self.cards if c.due <= today and (not source or c.source == source)),
                      key=lambda c: (c.box, c.due))

    def review(self, card_id: str, correct: bool, today: date | None = None):
        today = today or date.today()
        c = next(c for c in self.cards if c.id == card_id)
        c.reviews += 1
        if correct:
            c.correct += 1
            c.box = min(c.box + 1, 5)
        else:
            c.box = 1
        c.due = (today + timedelta(days=INTERVAL_DAYS[c.box])).isoformat()
        self.save()

    def remove(self, card_id: str):
        self.cards = [c for c in self.cards if c.id != card_id]
        self.save()

    def remove_source(self, source: str):
        self.cards = [c for c in self.cards if c.source != source]
        self.save()
