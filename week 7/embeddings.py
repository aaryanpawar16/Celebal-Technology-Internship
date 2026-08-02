"""
embeddings.py
-------------
Maps chunked text strings into dense vector representations using a
pre-trained embedding model (sentence-transformers).

Default model: "all-MiniLM-L6-v2" -> 384-dimensional embeddings.
Swap MODEL_NAME for a larger model (e.g. "all-mpnet-base-v2", 768-dim) if you
need higher accuracy at the cost of speed/memory.
"""

from typing import List
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # must match MODEL_NAME; update if you change the model


class EmbeddingModel:
    def __init__(self, model_name: str = MODEL_NAME):
        from sentence_transformers import SentenceTransformer  # lazy import
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        """Encode a list of strings into an (N, dim) float32 numpy array."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=normalize,  # normalized -> cosine sim via dot product
            convert_to_numpy=True,
        )
        return embeddings.astype("float32")

    def encode_one(self, text: str, normalize: bool = True) -> np.ndarray:
        return self.encode([text], normalize=normalize)[0]
