"""
metrics.py
----------
Two responsibilities:

1. run_validation(): runs a batch of sample questions through the pipeline,
   records retrieval scores/sources/answers/latency, and writes a validation
   log (JSON) that can be inspected or diffed across pipeline versions.

2. generate_metrics_report(): writes a human-readable Markdown report
   documenting the chunking profile, embedding model + dimensions, vector
   store configuration, retrieval mode, and LLM setup -- plus a summary
   table of the validation run.
"""

import json
import os
import time
from datetime import datetime
from typing import List, Dict, Optional

from pipeline import RAGPipeline


def run_validation(
    rag: RAGPipeline,
    sample_questions: List[str],
    top_k: int = 3,
    log_path: str = "validation_log.json",
) -> List[Dict]:
    """
    Runs each sample question through the live pipeline and logs:
      - the question
      - the generated answer
      - retrieved chunk sources + relevance scores
      - retrieval + generation latency
    Writes results to `log_path` and also returns them in-memory.
    """
    results = []
    for q in sample_questions:
        t0 = time.time()
        result = rag.ask(q, top_k=top_k, verbose=False)
        result["timestamp"] = datetime.utcnow().isoformat() + "Z"
        result["total_latency_seconds"] = round(time.time() - t0, 3)
        results.append(result)

    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def summarize_validation(results: List[Dict]) -> Dict:
    """Aggregate simple stats over a validation run for the metrics report."""
    if not results:
        return {"num_questions": 0}

    latencies = [r["latency_seconds"] for r in results]
    avg_top_score = []
    zero_hit_count = 0
    for r in results:
        scores = [s["score"] for s in r["sources"] if s.get("score") is not None]
        if scores:
            avg_top_score.append(max(scores))
        else:
            zero_hit_count += 1

    return {
        "num_questions": len(results),
        "avg_latency_seconds": round(sum(latencies) / len(latencies), 3),
        "max_latency_seconds": round(max(latencies), 3),
        "min_latency_seconds": round(min(latencies), 3),
        "avg_top_retrieval_score": round(sum(avg_top_score) / len(avg_top_score), 3) if avg_top_score else None,
        "questions_with_no_retrieved_chunks": zero_hit_count,
    }


def generate_metrics_report(
    rag: RAGPipeline,
    validation_results: Optional[List[Dict]] = None,
    report_path: str = "metrics_report.md",
) -> str:
    """Writes a Markdown system metrics report and returns the report text."""
    index_stats = {
        "num_records": len(rag.records),
        "num_chunks": len(rag.chunks),
        "embedding_dim": rag.embedder.dim,
    }
    summary = summarize_validation(validation_results) if validation_results else {}

    lines = []
    lines.append("# RAG System Metrics Report")
    lines.append(f"_Generated: {datetime.utcnow().isoformat()}Z_\n")

    lines.append("## Chunking Profile")
    lines.append(f"- Method: `{rag.chunk_method}`")
    lines.append(f"- Chunk size: `{rag.chunk_size}` characters")
    lines.append(f"- Overlap: `{rag.overlap}` characters")
    lines.append(f"- Total chunks produced: `{index_stats['num_chunks']}` (from `{index_stats['num_records']}` ingested records)\n")

    lines.append("## Embedding Model")
    lines.append(f"- Model: `{rag.embedder.model_name}`")
    lines.append(f"- Embedding dimension: `{index_stats['embedding_dim']}`")
    lines.append("- Similarity metric: cosine (via normalized embeddings + inner product)\n")

    lines.append("## Vector Store")
    lines.append("- Engine: FAISS (`IndexFlatIP`, exact search)")
    lines.append(f"- Indexed vectors: `{len(rag.store) if rag.store else 0}`\n")

    lines.append("## Retrieval Configuration")
    lines.append(f"- Mode: `{rag.retrieval_mode}`")
    if rag.retrieval_mode == "hybrid":
        lines.append(f"- Hybrid alpha (vector weight): `{rag.hybrid_alpha}` (keyword weight: `{round(1 - rag.hybrid_alpha, 2)}`)")
    lines.append(f"- Re-ranking enabled: `{rag.use_reranker}`")
    if rag.use_reranker:
        lines.append("- Re-ranker model: `cross-encoder/ms-marco-MiniLM-L-6-v2`")
    lines.append("")

    lines.append("## Language Model Setup")
    lines.append(f"- Provider: Ollama (local)")
    lines.append(f"- Model: `{rag.ollama_model}`")
    lines.append(f"- Endpoint: `{rag.ollama_url}`\n")

    if summary:
        lines.append("## Validation Run Summary")
        lines.append(f"- Sample questions run: `{summary['num_questions']}`")
        lines.append(f"- Avg end-to-end latency: `{summary['avg_latency_seconds']}s`")
        lines.append(f"- Min / Max latency: `{summary['min_latency_seconds']}s` / `{summary['max_latency_seconds']}s`")
        lines.append(f"- Avg top retrieval relevance score: `{summary['avg_top_retrieval_score']}`")
        lines.append(f"- Questions with no retrieved chunks: `{summary['questions_with_no_retrieved_chunks']}`\n")

        lines.append("### Per-question results")
        lines.append("| # | Question | Top Score | Latency (s) | Sources |")
        lines.append("|---|---|---|---|---|")
        for i, r in enumerate(validation_results, 1):
            top_score = max((s["score"] for s in r["sources"] if s.get("score") is not None), default=None)
            sources = ", ".join(sorted({str(s.get("source")) for s in r["sources"]})) or "none"
            q_short = r["question"] if len(r["question"]) <= 60 else r["question"][:57] + "..."
            lines.append(f"| {i} | {q_short} | {top_score} | {r['latency_seconds']} | {sources} |")
        lines.append("")

    report_text = "\n".join(lines)
    with open(report_path, "w") as f:
        f.write(report_text)

    return report_text
