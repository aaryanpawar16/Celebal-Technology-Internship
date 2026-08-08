# Memory-Augmented Chatbot (RAG + Knowledge Graph + LangGraph)

Fully local demo — no API keys, no Neo4j required.

## Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Optional: use Ollama for real generated answers
```bash
ollama pull llama3
ollama serve
```
The app auto-detects Ollama at `http://localhost:11434` and uses it if running.
To use a different pulled model:
```bash
export OLLAMA_MODEL=mistral
```
If Ollama isn't running, the app still works — it falls back to composing an
answer directly from the retrieved context (no LLM required at all).

## How it works
- `rag_pipeline.py` — embeds `data/sample_docs.txt` with sentence-transformers and searches it with FAISS.
- `knowledge_graph.py` — small in-memory graph (NetworkX) of entity relationships, used instead of Neo4j.
- `memory_store.py` — stores per-user facts and chat history in `data/memory.json`.
- `tools.py` — free Wikipedia lookup for "live"/real-time-style questions.
- `graph_workflow.py` — LangGraph StateGraph that routes each question to memory, RAG, the knowledge graph, and/or the live tool (in parallel, merged via a reducer), then generates the answer.
- `llm.py` — generates the final answer. Tries Ollama first (local, no key). If Ollama isn't running, falls back to `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` if set, and finally to a local extractive fallback that needs no LLM at all.
- `evaluation.py` — simple word-overlap scores (context relevance, faithfulness, and correctness if you supply a reference answer).
- `app.py` — Streamlit chat UI, with a sidebar to add memory facts and an expander per turn showing retrieved context + evaluation scores.

## Answer generation priority
1. Ollama (`http://localhost:11434`, model set by `OLLAMA_MODEL`, default `llama3`)
2. Anthropic (`ANTHROPIC_API_KEY` env var)
3. OpenAI (`OPENAI_API_KEY` env var)
4. Local fallback — stitches retrieved context into a readable answer, no LLM needed

## Try it
- Static knowledge: "What is Retrieval-Augmented Generation?"
- Knowledge graph: "What is FAISS used for?"
- Live tool: "Who is the current president of the United States?"
- Memory: add a fact in the sidebar, then ask a question that could use it
- Multi-path: "How does NetworkX compare to Neo4j as a knowledge graph?"

Expand "Debug: retrieval + evaluation" under any answer to see exactly what
was retrieved and how it scored.

## Extending it
- Swap `data/sample_docs.txt` for your own scraped/cleaned corpus.
- Add more triples in `knowledge_graph.py` (or build an extractor to populate it automatically).
- Add more tools in `tools.py` and route to them in `graph_workflow.py`'s `route()` method.