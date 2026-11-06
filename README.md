# RevisePal: a local study assistant for your class notes

Upload your lecture notes (PDF, Markdown, text) and:

- **Ask questions.** Answers come only from your notes, with numbered citations to the exact file, page and section. If the notes don't contain the answer, it says so instead of guessing.
- **Generate flashcards.** Cards are written from your notes. Each one carries a verbatim quote from the notes as evidence, and any card whose quote can't be found in the source excerpt is rejected automatically.
- **Review with spaced repetition.** It uses a 5-box Leitner schedule (due again after 0 / 1 / 3 / 7 / 14 days), and you can get a grounded "explain this more" for any card.
- **Measure retrieval quality.** It uses your own flashcards as a labelled test set and reports hit@k, MRR and citation accuracy.

Everything runs **locally** with [Ollama](https://ollama.com), so your notes never leave your machine. Any OpenAI-compatible endpoint also works.

## How it works

```
 notes (pdf/md/txt)
      │  parse + chunk (≈1000 chars, overlap, keeps file / page / heading)
      ▼
 embeddings (nomic-embed-text) ──► Chroma vector store (persistent, cosine)
                                         │ top-k retrieval
             ┌───────────────────────────┼─────────────────────────────┐
             ▼                           ▼                             ▼
   Q&A with citations          Flashcard generation             Evaluation
   (answer only from           (JSON output + verbatim          (card question →
    excerpts, [n] cites,        evidence quote, fuzzy-match       should retrieve its
    "not found" fallback)       grounding check rejects the       source chunk:
                                ungrounded cards)                 hit@1, hit@k, MRR)
                                         │
                                         ▼
                                Leitner spaced-repetition deck
```

| Component | Choice |
|---|---|
| LLM | `llama3.2:3b` via Ollama (configurable) |
| Embeddings | `nomic-embed-text` with `search_query:` / `search_document:` task prefixes |
| Vector DB | Chroma (persistent, cosine similarity) |
| UI | Streamlit |
| Tests | pytest, with a fake model client so the tests run offline |

## Run it

### Option A: native (fastest)

```bash
# 1. Install Ollama from https://ollama.com, then pull the models
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# 2. Install and start the app
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501 and upload some notes.

### Option B: Docker

```bash
docker compose up -d ollama
docker compose run --rm pull-models     # first time only, downloads the models
docker compose up -d app
```

### Using a different model

The default model is set with environment variables (see `.env.example`):

```bash
CHAT_MODEL=qwen2.5:7b streamlit run app.py          # bigger local model, better cards
# or a hosted OpenAI-compatible API:
LLM_BASE_URL=https://api.groq.com/openai/v1 LLM_API_KEY=... CHAT_MODEL=llama-3.1-8b-instant streamlit run app.py
```

## Evaluate

After generating and saving some flashcards:

```bash
python -m revisepal.evaluate --k 5 --answers
```

## Tests

```bash
pytest -q
```

## Project layout

```
revisepal/
  ingest.py      PDF/MD/TXT parsing, heading-aware chunking with overlap
  llm.py         OpenAI-compatible chat + embeddings, robust JSON parsing
  store.py       Chroma vector store: add / replace / delete / search
  rag.py         grounded Q&A with citations and a "not found" fallback
  flashcards.py  grounded card generation, evidence check, Leitner deck
  evaluate.py    retrieval + citation evaluation using cards as test set
app.py           Streamlit UI
tests/           offline tests with a fake model client
```

## Limitations and next steps

- Scanned PDFs need OCR first, because pypdf only reads embedded text.
- Small local models sometimes write weak questions. Use a 7B+ model for better cards.
- Ideas for next steps: hybrid BM25 + vector search, a re-ranker, an LLM judge for answer faithfulness, and a typed-answer quiz mode with automatic grading.
