"""Offline tests: a fake OpenAI client stands in for Ollama, so no model is needed."""
import hashlib
import json
import re
from datetime import date, timedelta
from types import SimpleNamespace as NS

import numpy as np
import pytest

from revisepal.config import Settings
from revisepal.evaluate import evaluate
from revisepal.flashcards import Deck, generate, grounding_score
from revisepal.ingest import chunk_document
from revisepal.llm import LLM, parse_json
from revisepal.rag import NOT_FOUND, answer
from revisepal.store import NoteStore

NOTES = b"""# Gradient descent
Gradient descent is an iterative optimisation algorithm that updates parameters in the
direction of the negative gradient of the loss function. The learning rate controls the step size.
A learning rate that is too large can make training diverge, while one that is too small makes convergence slow.

# Overfitting
Overfitting happens when a model learns noise in the training data and performs poorly on unseen data.
Regularisation techniques such as L2 weight decay and dropout reduce overfitting. Early stopping halts
training when validation loss stops improving.

# Bias and variance
High bias models underfit the data because they are too simple. High variance models are too sensitive
to the training set. The bias-variance trade-off describes the balance between these two sources of error.
"""


def _embed(text, dim=256):
    v = np.zeros(dim)
    for w in re.findall(r"[a-z]+", text.lower()):
        if len(w) > 3:
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1
    n = np.linalg.norm(v)
    return (v / n if n else v + 1e-6).tolist()


class FakeClient:
    def __init__(self, bad_evidence=False):
        self.bad_evidence = bad_evidence
        self.embeddings = NS(create=self._emb)
        self.chat = NS(completions=NS(create=self._chat))

    def _emb(self, model, input):
        return NS(data=[NS(embedding=_embed(t)) for t in input])

    def _chat(self, model, messages, temperature=0, response_format=None):
        user = messages[-1]["content"]
        if "flashcards" in messages[0]["content"]:
            excerpt = user.split("):\n", 1)[1]
            sentences = [s.strip() for s in re.split(r"(?<=\.)\s+", excerpt) if len(s.split()) > 5]
            cards = [{"question": f"What does this describe: {s[:40]}?", "answer": s[:60],
                      "evidence": "the moon is made of cheese" if self.bad_evidence else s}
                     for s in sentences[:2]]
            content = "```json\n" + json.dumps({"cards": cards}) + "\n```"
        else:
            content = "Overfitting is learning noise in the training data [1]." if "[1]" in user else NOT_FOUND
        return NS(choices=[NS(message=NS(content=content))])


@pytest.fixture
def env(tmp_path):
    s = Settings(data_dir=tmp_path, embed_model="nomic-embed-text")
    llm = LLM(s, client=FakeClient())
    store = NoteStore(s, llm)
    store.add(chunk_document("ml_notes.md", NOTES, size=300, overlap=50))
    return s, llm, store


def test_chunking_keeps_sections_and_ids():
    chunks = chunk_document("ml_notes.md", NOTES, size=300, overlap=50)
    assert len(chunks) >= 3
    assert {c.section for c in chunks} >= {"Gradient descent", "Overfitting", "Bias and variance"}
    assert len({c.id for c in chunks}) == len(chunks)
    assert all(len(c.text) <= 300 + 60 for c in chunks)


def test_long_paragraph_is_split():
    text = ("Sentence number one is here. " * 200).encode()
    chunks = chunk_document("long.txt", text, size=500, overlap=50)
    assert len(chunks) > 5 and max(len(c.text) for c in chunks) <= 500


def test_reupload_replaces_chunks(env):
    s, llm, store = env
    before = store.sources()["ml_notes.md"]
    store.add(chunk_document("ml_notes.md", NOTES, size=300, overlap=50))
    assert store.sources()["ml_notes.md"] == before


def test_retrieval_finds_relevant_section(env):
    _, _, store = env
    top, sim = store.search("what is overfitting and how does dropout help", k=1)[0]
    assert top.section == "Overfitting" and sim > 0


def test_answer_cites_sources(env):
    _, llm, store = env
    a = answer("What is overfitting?", store, llm, k=3)
    assert a.found and a.cited == [1] and len(a.sources) == 3


def test_answer_not_found_when_nothing_relevant(env):
    _, llm, store = env
    a = answer("photosynthesis in plants", store, llm, min_similarity=0.95)
    assert not a.found and a.sources == []


def test_grounding_score():
    chunk = "Dropout randomly disables neurons during training to reduce overfitting."
    assert grounding_score("dropout randomly disables neurons", chunk) == 1.0
    assert grounding_score("Dropout randomly disables neurons, during training!", chunk) == 1.0
    assert grounding_score("the moon is made of cheese", chunk) < 0.5


def test_flashcards_grounded_and_rejected(env, tmp_path):
    s, llm, store = env
    cards, stats = generate(store, llm, n_cards=4)
    assert cards and stats["grounded_rate"] == 1.0
    assert all(grounding_score(c.evidence, store.get(c.chunk_id).text) >= 0.8 for c in cards)

    bad = LLM(s, client=FakeClient(bad_evidence=True))
    cards, stats = generate(store, bad, n_cards=4)
    assert cards == [] and stats["rejected"] > 0


def test_deck_leitner_schedule(env):
    s, llm, store = env
    deck = Deck(s.deck_path)
    cards, _ = generate(store, llm, n_cards=2)
    deck.add(cards)
    deck.add(cards)  # duplicates ignored
    assert len(deck.cards) == len(cards)
    today = date.today()
    c = deck.due()[0]
    deck.review(c.id, True, today)
    assert c.box == 2 and c.due == (today + timedelta(days=1)).isoformat()
    deck.review(c.id, False, today)
    assert c.box == 1 and c.due == today.isoformat()
    assert len(Deck(s.deck_path).cards) == len(cards)  # persisted


def test_evaluate_reports_metrics(env):
    s, llm, store = env
    deck = Deck(s.deck_path)
    deck.add(generate(store, llm, n_cards=6)[0])
    res = evaluate(store, llm, deck, k=3, with_answers=True)
    assert res["n_questions"] == len(deck.cards)
    assert 0 <= res["hit@1"] <= res["hit@3"] <= 1 and res["mrr"] > 0


def test_parse_json_tolerates_chatter():
    assert parse_json('Sure! Here you go: {"cards": []} hope it helps') == {"cards": []}


def test_pdf_ingest():
    from pypdf import PdfWriter
    import io
    w = PdfWriter()
    w.add_blank_page(200, 200)
    buf = io.BytesIO()
    w.write(buf)
    assert chunk_document("blank.pdf", buf.getvalue()) == []  # no text, no crash
