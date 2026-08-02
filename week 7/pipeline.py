"""
pipeline.py
-----------
Ties the whole RAG system together end-to-end:

  ingest -> chunk -> embed -> store -> (query -> embed -> retrieve [-> rerank]) -> LLM prompt -> answer

Usage:
    from pipeline import RAGPipeline

    rag = RAGPipeline(retrieval_mode="hybrid", use_reranker=True)
    rag.ingest_pdf_bytes(pdf_bytes, source_name="handbook.pdf")
    rag.build_index()
    result = rag.ask("What is the refund policy?")
    print(result["answer"])
"""

import time
import requests
from typing import List, Dict, Optional

from ingestion import ingest_pdf, ingest_txt, ingest_hf_dataset
from chunking import build_chunks
from embeddings import EmbeddingModel
from vector_store import FAISSVectorStore
from retrieval import VectorRetriever, HybridRetriever, CrossEncoderReranker

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama2"


class RAGPipeline:
    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
        chunk_method: str = "sentence",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        retrieval_mode: str = "hybrid",   # "vector" | "hybrid"
        hybrid_alpha: float = 0.65,
        use_reranker: bool = False,
        ollama_model: str = OLLAMA_MODEL,
        ollama_url: str = OLLAMA_URL,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunk_method = chunk_method
        self.retrieval_mode = retrieval_mode
        self.hybrid_alpha = hybrid_alpha
        self.use_reranker = use_reranker
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url

        self.records: List[Dict] = []   # raw ingested records, pre-chunk
        self.chunks: List[Dict] = []    # chunked text
        self.embedder = EmbeddingModel(embedding_model_name)
        self.store: Optional[FAISSVectorStore] = None
        self.retriever = None
        self.reranker = CrossEncoderReranker() if use_reranker else None

    # ---------- Ingestion ----------
    def ingest_pdf_bytes(self, pdf_bytes: bytes, source_name: str = "uploaded.pdf", dpi: int = 300):
        self.records.extend(ingest_pdf(pdf_bytes, source_name=source_name, dpi=dpi))

    def ingest_txt_bytes(self, txt_bytes: bytes, source_name: str = "uploaded.txt"):
        self.records.extend(ingest_txt(txt_bytes, source_name=source_name))

    def ingest_hf(self, dataset_name: str, text_column: str, split: str = "train", **kwargs):
        self.records.extend(ingest_hf_dataset(dataset_name, text_column, split=split, **kwargs))

    # ---------- Index build ----------
    def build_index(self):
        """Chunk all ingested records, embed them, and load them into the vector store."""
        if not self.records:
            raise ValueError("No documents ingested yet. Call ingest_* first.")

        self.chunks = build_chunks(
            self.records, chunk_size=self.chunk_size, overlap=self.overlap, method=self.chunk_method
        )
        if not self.chunks:
            raise ValueError("Chunking produced no chunks (empty documents?).")

        texts = [c["chunk_text"] for c in self.chunks]
        embeddings = self.embedder.encode(texts)

        self.store = FAISSVectorStore(dim=self.embedder.dim)
        self.store.add(embeddings, self.chunks)

        if self.retrieval_mode == "hybrid":
            self.retriever = HybridRetriever(self.embedder, self.store, alpha=self.hybrid_alpha)
        else:
            self.retriever = VectorRetriever(self.embedder, self.store)

        return {"num_records": len(self.records), "num_chunks": len(self.chunks), "embedding_dim": self.embedder.dim}

    # ---------- Query ----------
    def retrieve(self, question: str, top_k: int = 3) -> List[Dict]:
        if self.retriever is None:
            raise ValueError("Index not built yet. Call build_index() first.")
        candidate_k = top_k * 4 if self.reranker else top_k
        results = self.retriever.retrieve(question, top_k=candidate_k)
        if self.reranker:
            results = self.reranker.rerank(question, results, top_k=top_k)
        return results

    def _build_prompt(self, question: str, chunks: List[Dict]) -> str:
        context = "\n---\n".join(
            f"(Source: {c.get('source')}, page {c.get('page_number')}) {c['chunk_text']}"
            for c in chunks
        )
        return (
            "You are a helpful assistant. Answer the question using ONLY the provided context. "
            "If the context doesn't contain the answer, say you don't know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )

    def _call_ollama(self, prompt: str) -> str:
        resp = requests.post(
            self.ollama_url,
            json={"model": self.ollama_model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("response") or data.get("message") or str(data)
        return f"[Ollama error {resp.status_code}] {resp.text}"

    def ask(self, question: str, top_k: int = 3, verbose: bool = True) -> Dict:
        """Full ask cycle: retrieve -> prompt -> generate. Returns answer + grounding sources."""
        start = time.time()
        retrieved = self.retrieve(question, top_k=top_k)
        prompt = self._build_prompt(question, retrieved)
        answer = self._call_ollama(prompt)
        elapsed = time.time() - start

        result = {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "source": c.get("source"),
                    "page_number": c.get("page_number"),
                    "score": c.get("score"),
                    "snippet": c["chunk_text"][:200],
                }
                for c in retrieved
            ],
            "latency_seconds": round(elapsed, 3),
        }
        if verbose:
            print(f"\nQ: {question}")
            print(f"A: {answer}")
            print(f"Grounded in {len(retrieved)} chunk(s), {elapsed:.2f}s")
        return result
