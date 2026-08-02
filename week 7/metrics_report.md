# RAG System Metrics Report
_Generated: 2026-08-02T14:45:02.349677Z_

## Chunking Profile
- Method: `sentence`
- Chunk size: `1000` characters
- Overlap: `200` characters
- Total chunks produced: `7` (from `2` ingested records)

## Embedding Model
- Model: `all-MiniLM-L6-v2`
- Embedding dimension: `384`
- Similarity metric: cosine (via normalized embeddings + inner product)

## Vector Store
- Engine: FAISS (`IndexFlatIP`, exact search)
- Indexed vectors: `7`

## Retrieval Configuration
- Mode: `hybrid`
- Hybrid alpha (vector weight): `0.65` (keyword weight: `0.35`)
- Re-ranking enabled: `False`

## Language Model Setup
- Provider: Ollama (local)
- Model: `llama2`
- Endpoint: `http://localhost:11434/api/generate`

## Validation Run Summary
- Sample questions run: `2`
- Avg end-to-end latency: `59.718s`
- Min / Max latency: `58.361s` / `61.076s`
- Avg top retrieval relevance score: `0.187`
- Questions with no retrieved chunks: `0`

### Per-question results
| # | Question | Top Score | Latency (s) | Sources |
|---|---|---|---|---|
| 1 | What internships has this candidate completed? | 0.21605424284935 | 58.361 | Aaryan Pawar resume.pdf |
| 2 | What is the candidate's educational background? | 0.15835201740264893 | 61.076 | Aaryan Pawar resume.pdf |
