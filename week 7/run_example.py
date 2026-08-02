"""
run_example.py
--------------
Demonstrates the full pipeline end-to-end against a plain-text document
(no PDF/OCR dependency needed to try it out) and produces:
  - printed, grounded, context-aware answers for a set of domain queries
  - a validation_log.json with per-question retrieval/generation details
  - a metrics_report.md documenting chunking, embeddings, vector store, and LLM setup

Run:
    python run_example.py
"""

from pipeline import RAGPipeline
from metrics import run_validation, generate_metrics_report

SAMPLE_DOCUMENT = b"""
Our return policy allows customers to return unused items within 30 days of purchase
for a full refund. Items must be in their original packaging with proof of purchase.
Refunds are processed within 5-7 business days to the original payment method.

Shipping is free on orders over $50 within the continental United States. Orders under
$50 incur a flat $5.99 shipping fee. Expedited shipping options are available at checkout
for an additional cost, typically arriving within 1-2 business days.

Customer support is available Monday through Friday, 9 AM to 6 PM EST, via email or
live chat. For urgent issues outside these hours, customers can leave a voicemail and
expect a callback within one business day.
"""

SAMPLE_QUESTIONS = [
    "How many days do I have to return an item?",
    "Is shipping free, and under what conditions?",
    "What are the customer support hours?",
    "Do you offer same-day shipping?",  # intentionally not in the doc, tests grounding
]


def main():
    rag = RAGPipeline(
        chunk_size=300,
        overlap=50,
        chunk_method="sentence",
        retrieval_mode="hybrid",
        hybrid_alpha=0.65,
        use_reranker=False,   # set True to enable cross-encoder re-ranking
        ollama_model="llama2",
    )

    rag.ingest_txt_bytes(SAMPLE_DOCUMENT, source_name="policy.txt")
    build_stats = rag.build_index()
    print("Index built:", build_stats)

    # 1. Operational end-to-end QA with printed, grounded answers
    for q in SAMPLE_QUESTIONS:
        rag.ask(q, top_k=3, verbose=True)

    # 2. Documented validation logs
    validation_results = run_validation(rag, SAMPLE_QUESTIONS, top_k=3, log_path="validation_log.json")
    print("\nValidation log written to validation_log.json")

    # 3. System metrics report
    report = generate_metrics_report(rag, validation_results, report_path="metrics_report.md")
    print("\nMetrics report written to metrics_report.md")
    print("\n--- Metrics Report Preview ---\n")
    print(report[:800], "...")


if __name__ == "__main__":
    main()
