"""
vector_store.py
----------------
Initializes a vector database (FAISS), stores chunk embeddings + metadata,
and configures it for fast similarity search.

FAISS is used here because it's local/embedded (no server to run), which
keeps this pipeline self-contained. Swap in Chroma/Pinecone/Weaviate by
implementing the same add()/search()/save()/load() interface if you need
persistence across processes or a managed/hosted store instead.
"""

import json
import os
from typing import List, Dict, Tuple

import numpy as np
import faiss


class FAISSVectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        # Inner-product index; works as cosine similarity when embeddings are
        # normalized (see embeddings.py: normalize_embeddings=True).
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: List[Dict] = []  # parallel list; metadata[i] <-> vector i

    def add(self, embeddings: np.ndarray, metadatas: List[Dict]):
        assert embeddings.shape[0] == len(metadatas), "embeddings/metadata count mismatch"
        assert embeddings.shape[1] == self.dim, f"expected dim {self.dim}, got {embeddings.shape[1]}"
        self.index.add(embeddings)
        self.metadata.extend(metadatas)

    def search(self, query_vec: np.ndarray, top_k: int = 3) -> List[Tuple[Dict, float]]:
        """Returns list of (metadata, similarity_score) sorted by relevance."""
        if self.index.ntotal == 0:
            return []
        query_vec = query_vec.reshape(1, -1).astype("float32")
        top_k = min(top_k, self.index.ntotal)
        scores, idxs = self.index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append((self.metadata[idx], float(score)))
        return results

    def save(self, dir_path: str):
        os.makedirs(dir_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(dir_path, "index.faiss"))
        with open(os.path.join(dir_path, "metadata.json"), "w") as f:
            json.dump({"dim": self.dim, "metadata": self.metadata}, f)

    @classmethod
    def load(cls, dir_path: str) -> "FAISSVectorStore":
        with open(os.path.join(dir_path, "metadata.json")) as f:
            data = json.load(f)
        store = cls(dim=data["dim"])
        store.index = faiss.read_index(os.path.join(dir_path, "index.faiss"))
        store.metadata = data["metadata"]
        return store

    def __len__(self):
        return self.index.ntotal
