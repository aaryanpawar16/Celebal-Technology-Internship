# RAG Pipeline — Document Question Answering System

A modular, end-to-end Retrieval-Augmented Generation (RAG) pipeline: ingest documents (PDF/OCR, raw text, or Hugging Face datasets), chunk them cleanly, embed with a pre-trained sentence-transformer, store/search with FAISS, retrieve with vector or hybrid (vector + keyword) search, optionally re-rank with a cross-encoder, and generate grounded answers with a local Ollama model — with validation logging and a metrics report built in.

## Contents

| File | Purpose |
|---|---|
| `ingestion.py` | Load PDFs (OCR), raw `.txt` files, or Hugging Face datasets into a common record format |
| `chunking.py` | Split text into overlapping chunks — fixed-size (`char`) or sentence-aware (`sentence`) |
| `embeddings.py` | Wraps a pre-trained `sentence-transformers` model to embed text |
| `vector_store.py` | FAISS-backed vector database — add, search, save, load |
| `retrieval.py` | `VectorRetriever` (pure semantic), `HybridRetriever` (semantic + TF-IDF keyword), `CrossEncoderReranker` (optional second-stage re-ranking) |
| `pipeline.py` | `RAGPipeline` class tying ingestion → chunking → embedding → storage → retrieval → LLM prompt/generation together |
| `metrics.py` | Runs validation questions, logs results to JSON, generates a Markdown system metrics report |
| `run_example.py` | Runnable demo — builds an index from sample text, prints grounded answers, writes a validation log + metrics report |
| `app.py` | Streamlit UI wired to `RAGPipeline` — upload PDFs/text/HF datasets, build the index, chat, and run validation from the browser |
| `requirements.txt` | Python dependencies |

## Requirements

### Python packages

```bash
pip install -r requirements.txt
```

Installs: `streamlit`, `pytesseract`, `pdf2image`, `pillow`, `requests`, `scikit-learn`, `sentence-transformers`, `faiss-cpu`, `datasets`.

### External tools

- **Tesseract OCR** and **Poppler** — only needed if you're ingesting PDFs (see `ingestion.py`'s `TESSERACT_CMD` / `POPPLER_PATH` — set these as environment variables or edit the defaults in the file for your machine).
- **Ollama** — required for answer generation. Install from [ollama.com](https://ollama.com/), then:
  ```bash
  ollama pull llama2
  ollama run llama2
  ```
- Internet access on first run — `sentence-transformers` downloads the embedding model (`all-MiniLM-L6-v2`) from Hugging Face the first time it's used, then caches it locally.

## Quick start

```bash
pip install -r requirements.txt
ollama run llama2          # in a separate terminal, keep it running
python run_example.py
```

### Or run the Streamlit UI

```bash
pip install -r requirements.txt
ollama run llama2          # in a separate terminal, keep it running
streamlit run app.py
```

`app.py` gives you tabs to ingest a PDF, `.txt` file, or Hugging Face dataset; a "Build index" step; a chat panel with per-answer sources and relevance scores; and a validation panel that writes `validation_log.json` + `metrics_report.md` and renders the report inline.

This ingests a small sample "return policy" document, builds the index, asks four sample questions (printing grounded answers with retrieval latency), then writes:
- `validation_log.json` — every question's answer, sources, relevance scores, and latency
- `metrics_report.md` — chunking profile, embedding model/dimensions, vector store config, retrieval mode, LLM setup, and a validation summary table

## Using it with your own documents

```python
from pipeline import RAGPipeline
from metrics import run_validation, generate_metrics_report

rag = RAGPipeline(
    chunk_size=1000,
    overlap=200,
    chunk_method="sentence",       # or "char"
    retrieval_mode="hybrid",       # or "vector"
    hybrid_alpha=0.65,             # weight toward semantic vs keyword match
    use_reranker=False,            # True to add cross-encoder re-ranking
    ollama_model="llama2",
)

# Ingest — pick whichever source(s) you need
with open("handbook.pdf", "rb") as f:
    rag.ingest_pdf_bytes(f.read(), source_name="handbook.pdf")

with open("notes.txt", "rb") as f:
    rag.ingest_txt_bytes(f.read(), source_name="notes.txt")

rag.ingest_hf("squad", text_column="context", split="train", max_records=500)

# Build the index (chunk -> embed -> store)
stats = rag.build_index()
print(stats)  # {'num_records': ..., 'num_chunks': ..., 'embedding_dim': 384}

# Ask questions
result = rag.ask("What is the refund policy?", top_k=3)
print(result["answer"])
print(result["sources"])

# Validate + report
sample_questions = ["What is the refund policy?", "How long does shipping take?"]
validation_results = run_validation(rag, sample_questions, top_k=3, log_path="validation_log.json")
generate_metrics_report(rag, validation_results, report_path="metrics_report.md")
```

## How it works (pipeline)

```
Ingest (PDF/OCR, .txt, or HF dataset)
   → Chunk (sentence-aware or fixed-size, with overlap)
   → Embed (sentence-transformers, e.g. all-MiniLM-L6-v2 -> 384-dim vectors)
   → Store (FAISS IndexFlatIP, cosine similarity via normalized vectors)
   → On each question:
        - question -> query vector (same embedding model)
        - VectorRetriever: pure semantic top-k search
          or HybridRetriever: blends semantic score + TF-IDF keyword score
        - optional CrossEncoderReranker: re-scores (query, chunk) pairs jointly
        - top chunks + question -> single prompt -> Ollama -> grounded answer
   → Logged to validation_log.json, summarized in metrics_report.md
```

## Configuration reference

| Parameter | Where | Default | Notes |
|---|---|---|---|
| `chunk_size` / `overlap` | `RAGPipeline(...)` | `1000` / `200` | Characters per chunk / overlap between chunks |
| `chunk_method` | `RAGPipeline(...)` | `"sentence"` | `"sentence"` avoids cutting sentences mid-way; `"char"` is faster/simpler |
| `embedding_model_name` | `RAGPipeline(...)` | `"all-MiniLM-L6-v2"` | Swap for `"all-mpnet-base-v2"` (768-dim) for higher accuracy, slower |
| `retrieval_mode` | `RAGPipeline(...)` | `"hybrid"` | `"vector"` for pure semantic; `"hybrid"` blends in keyword matching |
| `hybrid_alpha` | `RAGPipeline(...)` | `0.65` | Weight toward vector score (`1-alpha` goes to keyword score) |
| `use_reranker` | `RAGPipeline(...)` | `False` | Adds a cross-encoder re-ranking pass over top candidates (slower, more precise) |
| `ollama_model` / `ollama_url` | `RAGPipeline(...)` | `"llama2"` / `http://localhost:11434/api/generate` | Change model to whatever you've pulled in Ollama |

## Known limitations

- First run needs internet access to download the embedding model (and re-ranker model, if enabled) from Hugging Face; after that they're cached locally.
- FAISS `IndexFlatIP` does exact (brute-force) search — fine up to tens of thousands of chunks, but for very large corpora consider an approximate index (e.g. `IndexIVFFlat`, `IndexHNSWFlat`).
- No persistence across pipeline restarts unless you explicitly call `store.save()` / `FAISSVectorStore.load()`.
- PDF ingestion still depends on locally installed Tesseract + Poppler binaries (same as the original app) — accuracy depends on scan quality and DPI.
- Ollama must be running locally before calling `rag.ask(...)`, or generation requests will fail/time out.

## Troubleshooting

- **`OSError` / connection errors loading the embedding model** — check internet access; the first run downloads `all-MiniLM-L6-v2` from Hugging Face.
- **`ImportError: scikit-learn is required for HybridRetriever`** — install `scikit-learn`, or switch to `retrieval_mode="vector"`.
- **`[Ollama error ...]` in answers** — make sure Ollama is running (`ollama run <model>`) and reachable at the configured `ollama_url`.
- **Empty `sources` in results** — the vector store may be empty; make sure `build_index()` ran successfully before calling `ask()`.
