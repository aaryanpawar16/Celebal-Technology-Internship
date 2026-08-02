"""
retrieval.py
------------
Query -> vector -> retrieval pipeline.

Includes:
  - VectorRetriever: pure semantic (embedding) search against the FAISS store
  - HybridRetriever: blends semantic similarity with a lightweight TF-IDF
    keyword score, useful when queries contain exact terms (names, codes,
    numbers) that embeddings alone can under-rank
  - CrossEncoderReranker: optional second-stage re-ranking of the top
    candidates using a cross-encoder model for higher-precision ordering
"""

from typing import List, Dict, Optional
import numpy as np

from embeddings import EmbeddingModel
from vector_store import FAISSVectorStore

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


class VectorRetriever:
    """Converts a question into a query vector and retrieves the closest chunks."""

    def __init__(self, embedder: EmbeddingModel, store: FAISSVectorStore):
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        query_vec = self.embedder.encode_one(query)
        results = self.store.search(query_vec, top_k=top_k)
        out = []
        for meta, score in results:
            item = dict(meta)
            item["score"] = score
            item["retrieval_method"] = "vector"
            out.append(item)
        return out


class HybridRetriever:
    """
    Combines vector similarity with TF-IDF keyword overlap.
    final_score = alpha * vector_score + (1 - alpha) * keyword_score
    alpha closer to 1.0 favors semantic match; closer to 0.0 favors exact terms.
    """

    def __init__(self, embedder: EmbeddingModel, store: FAISSVectorStore, alpha: float = 0.65):
        self.embedder = embedder
        self.store = store
        self.alpha = alpha
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for HybridRetriever's keyword scoring.")

    def retrieve(self, query: str, top_k: int = 3, candidate_pool: int = 20) -> List[Dict]:
        # Stage 1: cast a wider net with vector search
        query_vec = self.embedder.encode_one(query)
        candidates = self.store.search(query_vec, top_k=min(candidate_pool, len(self.store)))
        if not candidates:
            return []

        texts = [meta["chunk_text"] for meta, _ in candidates]
        vectorizer = TfidfVectorizer(stop_words="english")
        try:
            tfidf = vectorizer.fit_transform(texts)
            q_vec = vectorizer.transform([query])
            kw_scores = cosine_similarity(q_vec, tfidf).flatten()
        except ValueError:
            # e.g. empty vocabulary after stopword removal
            kw_scores = np.zeros(len(texts))

        blended = []
        for (meta, vec_score), kw_score in zip(candidates, kw_scores):
            final = self.alpha * vec_score + (1 - self.alpha) * float(kw_score)
            item = dict(meta)
            item["score"] = final
            item["vector_score"] = vec_score
            item["keyword_score"] = float(kw_score)
            item["retrieval_method"] = "hybrid"
            blended.append(item)

        blended.sort(key=lambda x: x["score"], reverse=True)
        return blended[:top_k]


class CrossEncoderReranker:
    """
    Optional second-stage re-ranker. Takes the top candidates from a first-stage
    retriever and re-scores (query, chunk) pairs jointly with a cross-encoder,
    which is typically more accurate than embedding cosine similarity alone.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder  # lazy import
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 3) -> List[Dict]:
        if not candidates:
            return []
        pairs = [(query, c["chunk_text"]) for c in candidates]
        scores = self.model.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
        return reranked[:top_k]
