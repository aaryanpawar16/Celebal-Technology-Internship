"""Static knowledge RAG: chunk local docs, embed them, retrieve with FAISS."""
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config import DOCS_PATH, EMBEDDING_MODEL, TOP_K

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def load_chunks(path: str = DOCS_PATH) -> list[str]:
    with open(path, "r") as f:
        text = f.read()
    # Paragraphs are our chunks — simple and good enough for a small demo corpus.
    return [c.strip() for c in text.split("\n\n") if c.strip()]


def build_index(chunks: list[str]):
    model = get_model()
    embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine sim via normalized vectors
    index.add(embeddings)
    return index


def retrieve(query: str, index, chunks: list[str], k: int = TOP_K) -> list[str]:
    model = get_model()
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, ids = index.search(q_emb, k)
    return [chunks[i] for i in ids[0] if i != -1]
