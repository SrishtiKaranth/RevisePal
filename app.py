"""RevisePal - Streamlit UI.  Run:  streamlit run app.py"""
import streamlit as st

from revisepal.config import Settings
from revisepal.evaluate import evaluate
from revisepal.flashcards import Deck, generate
from revisepal.ingest import chunk_document
from revisepal.llm import LLM
from revisepal.rag import answer
from revisepal.store import NoteStore

st.set_page_config(page_title="RevisePal", page_icon="📚", layout="wide")


@st.cache_resource
def services():
    s = Settings()
    llm = LLM(s)
    return s, llm, NoteStore(s, llm)


settings, llm, store = services()
deck = Deck(settings.deck_path)

# ---------------- Sidebar: notes + settings ----------------
with st.sidebar:
    st.title("📚 RevisePal")
    st.caption(f"Chat: `{settings.chat_model}` · Embeddings: `{settings.embed_model}`")

    uploads = st.file_uploader("Add class notes", type=["pdf", "md", "txt"], accept_multiple_files=True)
    if uploads and st.button("Index notes", type="primary", use_container_width=True):
        for f in uploads:
            with st.spinner(f"Indexing {f.name}…"):
                try:
                    chunks = chunk_document(f.name, f.getvalue(), settings.chunk_chars, settings.chunk_overlap)
                    n = store.add(chunks)
                    st.success(f"{f.name}: {n} chunks")
                except Exception as e:
                    st.error(f"{f.name}: {e}")

    sources = store.sources()
    st.subheader("Your notes")
    if not sources:
        st.info("No notes yet. Upload PDFs, Markdown or text files.")
    for src, n in sources.items():
        c1, c2 = st.columns([4, 1])
        c1.write(f"📄 {src} · {n} chunks")
        if c2.button("✕", key=f"del-{src}", help="Remove these notes and their flashcards"):
            store.delete_source(src)
            deck.remove_source(src)
            st.rerun()

    with st.expander("Settings"):
        scope = st.selectbox("Search in", ["All notes", *sources.keys()])
        scope = None if scope == "All notes" else scope
        top_k = st.slider("Excerpts per answer", 2, 10, settings.top_k)
        min_sim = st.slider("Min. similarity", 0.0, 0.9, 0.0, 0.05,
                            help="Excerpts below this score are ignored. Raise it if answers drift off-topic.")

tab_ask, tab_make, tab_review, tab_eval = st.tabs(["💬 Ask", "🃏 Make flashcards", "🔁 Review", "📊 Evaluate"])

# ---------------- Ask ----------------
with tab_ask:
    st.session_state.setdefault("chat", [])
    for turn in st.session_state.chat:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("sources"):
                with st.expander("Sources"):
                    for i, (cit, sim, text, cited) in enumerate(turn["sources"], 1):
                        st.markdown(f"**[{i}] {cit}** · similarity {sim:.2f}{' · cited' if cited else ''}")
                        st.caption(text[:600] + ("…" if len(text) > 600 else ""))

    if q := st.chat_input("Ask a question about your notes", disabled=not sources):
        st.session_state.chat.append({"role": "user", "content": q})
        with st.spinner("Searching your notes…"):
            try:
                a = answer(q, store, llm, k=top_k, min_similarity=min_sim, source=scope)
                srcs = [(c.citation(), sim, c.text, i in a.cited) for i, (c, sim) in enumerate(a.sources, 1)]
                st.session_state.chat.append({"role": "assistant", "content": a.text, "sources": srcs})
            except Exception as e:
                st.session_state.chat.append({"role": "assistant", "content": f"⚠️ {e}"})
        st.rerun()
    if st.session_state.chat and st.button("Clear chat"):
        st.session_state.chat = []
        st.rerun()

# ---------------- Make flashcards ----------------
with tab_make:
    st.write("Cards are written only from your notes. Each one keeps a quote from the notes as evidence. "
             "Cards whose quote can't be found in the notes are rejected.")
    c1, c2 = st.columns([3, 1])
    topic = c1.text_input("Topic (optional)", placeholder="e.g. gradient descent. Leave empty for a mix")
    n_cards = c2.number_input("Cards", 3, 40, 10)
    if st.button("Generate", type="primary", disabled=not sources):
        bar = st.progress(0.0, "Writing cards…")
        try:
            cards, stats = generate(store, llm, n_cards=int(n_cards), topic=topic, source=scope,
                                    progress=lambda p: bar.progress(p, "Writing cards…"))
            st.session_state.draft = cards
            st.session_state.draft_stats = stats
        except Exception as e:
            st.error(e)
        bar.empty()

    draft = st.session_state.get("draft", [])
    if draft:
        s = st.session_state.draft_stats
        st.caption(f"{s['kept']} grounded cards from {s['chunks']} excerpts · {s['rejected']} rejected "
                   f"· grounding pass rate {s['grounded_rate']:.0%}")
        keep = []
        for i, c in enumerate(draft):
            with st.container(border=True):
                k = st.checkbox(f"**Q:** {c.question}", value=True, key=f"keep-{c.id}")
                st.markdown(f"**A:** {c.answer}")
                st.caption(f"“{c.evidence}”  —  {c.citation}")
                if k:
                    keep.append(c)
        if st.button(f"Add {len(keep)} cards to deck", type="primary"):
            deck.add(keep)
            st.session_state.draft = []
            st.success("Saved. Go to Review.")
            st.rerun()

# ---------------- Review ----------------
with tab_review:
    due = deck.due(source=scope)
    boxes = [sum(c.box == b for c in deck.cards) for b in range(1, 6)]
    m1, m2, m3 = st.columns(3)
    m1.metric("Cards in deck", len(deck.cards))
    m2.metric("Due now", len(due))
    m3.metric("Mastered (box 5)", boxes[4])
    if not due:
        st.info("Nothing due. Make some flashcards or come back later.")
    else:
        card = due[0]
        st.session_state.setdefault("flipped", None)
        with st.container(border=True):
            st.caption(f"Box {card.box} · {card.citation}")
            st.subheader(card.question)
            if st.session_state.flipped == card.id:
                st.info(card.answer)
                st.caption(f"From your notes: “{card.evidence}”")
                c1, c2 = st.columns(2)
                if c1.button("✗ Again", use_container_width=True):
                    deck.review(card.id, False)
                    st.session_state.flipped = None
                    st.rerun()
                if c2.button("✓ Got it", type="primary", use_container_width=True):
                    deck.review(card.id, True)
                    st.session_state.flipped = None
                    st.rerun()
            elif st.button("Show answer", type="primary", use_container_width=True):
                st.session_state.flipped = card.id
                st.rerun()
        if st.session_state.flipped == card.id and st.button("💡 Explain this more (from my notes)"):
            with st.spinner("Explaining…"):
                st.markdown(answer(f"Explain: {card.question}", store, llm, k=top_k, source=card.source).text)
        if st.button("🗑 Delete this card"):
            deck.remove(card.id)
            st.rerun()

# ---------------- Evaluate ----------------
with tab_eval:
    st.write("Uses your flashcards as a test set: each card's question should retrieve the excerpt "
             "it was written from.")
    k = st.slider("k", 1, 10, 5, key="eval-k")
    with_ans = st.checkbox("Also check that answers cite the right excerpt (slower)")
    if st.button("Run evaluation", disabled=not deck.cards):
        with st.spinner("Evaluating…"):
            st.json(evaluate(store, llm, deck, k=k, with_answers=with_ans))
