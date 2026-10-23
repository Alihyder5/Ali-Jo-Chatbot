import glob

import numpy as np
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

st.title("Academic Policy Chatbot")

client = Groq(api_key=st.secrets["GROQ_API_KEY"])


@st.cache_resource(show_spinner="Reading the PDF (first time only)...")
def load_data():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    chunks = []
    for path in glob.glob("*.pdf"):
        text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
        chunks += [text[i : i + 1000] for i in range(0, len(text), 800)]
    vectors = embedder.encode(chunks, normalize_embeddings=True)
    return embedder, chunks, vectors


embedder, chunks, vectors = load_data()

question = st.chat_input("Ask about the academic policy")

if question:
    st.chat_message("user").write(question)

    # 1. Find the 3 most relevant chunks
    q_vector = embedder.encode([question], normalize_embeddings=True)[0]
    best = np.argsort(vectors @ q_vector)[::-1][:3]
    context = "\n\n".join(chunks[i] for i in best)

    # 2. Ask Groq to answer using only those chunks
    prompt = (
        "Answer using only the context below. "
        "If the answer is not there, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    reply = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
    )
    st.chat_message("assistant").write(reply.choices[0].message.content)
