import glob
import os

import numpy as np
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ---------------- Settings ----------------
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE, CHUNK_OVERLAP, TOP_K = 1000, 200, 3

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")

client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)


# ---------------- RAG steps ----------------
@st.cache_resource
def get_embedder():
    return SentenceTransformer(EMBED_MODEL)


def embed(texts):
    return get_embedder().encode(texts, normalize_embeddings=True, show_progress_bar=False)


def split_text(text):
    step = CHUNK_SIZE - CHUNK_OVERLAP
    chunks = [text[i : i + CHUNK_SIZE].strip() for i in range(0, len(text), step)]
    return [c for c in chunks if c]


@st.cache_resource(show_spinner="Reading documents (first run only)...")
def build_index():
    """Load every PDF in the repo folder, chunk it, and embed it once."""
    chunks = []
    for path in glob.glob("*.pdf"):
        text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
        chunks += split_text(text)
    if not chunks:
        return [], None
    return chunks, embed(chunks)


def retrieve(question, chunks, vectors):
    q = embed([question])[0]
    best = np.argsort(vectors @ q)[::-1][:TOP_K]
    return [chunks[i] for i in best]


def generate(question, context):
    prompt = (
        "Answer the question using only the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    response = client.chat.completions.create(
        model=LLM_MODEL, messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# ---------------- UI ----------------
st.title("Academic Policy Chatbot")

chunks, vectors = build_index()
if not chunks:
    st.error("No PDF found. Put your PDF in the same folder as app.py.")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

for role, text in st.session_state.history:
    with st.chat_message(role):
        st.markdown(text)

if question := st.chat_input("Ask about the academic policy"):
    st.session_state.history.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            context = "\n\n".join(retrieve(question, chunks, vectors))
            answer = generate(question, context)
        st.markdown(answer)
    st.session_state.history.append(("assistant", answer))
