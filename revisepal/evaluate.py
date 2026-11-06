"""Evaluate retrieval and answer grounding on your own notes.

Uses the flashcards as a labelled test set: each card's question should retrieve
the chunk the card was written from. Reports hit@1, hit@k and MRR, and (optionally)
how often the generated answer cites that chunk.

    python -m revisepal.evaluate --k 5 --answers
"""
from __future__ import annotations

import argparse
import json
import time

from .config import Settings
from .flashcards import Deck
from .llm import LLM
from .rag import answer
from .store import NoteStore


def evaluate(store: NoteStore, llm: LLM, deck: Deck, k: int = 5, with_answers: bool = False,
             limit: int | None = None) -> dict:
    cards = deck.cards[:limit] if limit else deck.cards
    hits1 = hitsk = 0
    rr = 0.0
    ans_found = ans_cites_gold = 0
    latencies = []
    for card in cards:
        t0 = time.perf_counter()
        ranked = [c.id for c, _ in store.search(card.question, k=k)]
        latencies.append(time.perf_counter() - t0)
        if ranked[:1] == [card.chunk_id]:
            hits1 += 1
        if card.chunk_id in ranked:
            hitsk += 1
            rr += 1 / (ranked.index(card.chunk_id) + 1)
        if with_answers:
            a = answer(card.question, store, llm, k=k)
            ans_found += a.found
            gold_positions = {i for i, (c, _) in enumerate(a.sources, 1) if c.id == card.chunk_id}
            ans_cites_gold += bool(gold_positions & set(a.cited))
    n = len(cards) or 1
    out = {
        "n_questions": len(cards),
        "k": k,
        "hit@1": round(hits1 / n, 3),
        f"hit@{k}": round(hitsk / n, 3),
        "mrr": round(rr / n, 3),
        "median_retrieval_ms": round(1000 * sorted(latencies)[len(latencies) // 2], 1) if latencies else None,
    }
    if with_answers:
        out["answered_rate"] = round(ans_found / n, 3)
        out["cites_source_chunk_rate"] = round(ans_cites_gold / n, 3)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--answers", action="store_true", help="also generate answers (slower)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default="data/eval.json")
    args = p.parse_args()
    s = Settings()
    llm = LLM(s)
    res = evaluate(NoteStore(s, llm), llm, Deck(s.deck_path), k=args.k, with_answers=args.answers, limit=args.limit)
    res.update(chat_model=s.chat_model, embed_model=s.embed_model, chunk_chars=s.chunk_chars)
    print(json.dumps(res, indent=2))
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
