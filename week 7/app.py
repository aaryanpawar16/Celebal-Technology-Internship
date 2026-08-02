"""
app.py
------
Streamlit UI for the upgraded RAG pipeline (embeddings + FAISS + hybrid
retrieval), replacing the original TF-IDF-only prototype.

Run:
    streamlit run app.py
"""

import streamlit as st
from pipeline import RAGPipeline
from metrics import run_validation, generate_metrics_report

st.set_page_config(page_title="RAG Pipeline — Document Q&A", layout="wide")
st.title("Document Question Answering System (RAG)")
st.markdown(
    "Upload a PDF, `.txt` file, or load a Hugging Face dataset, then chat with it "
    "using embeddings + FAISS retrieval and a local Ollama model."
)

# --- Sidebar settings ---
with st.sidebar:
    st.header("Settings")
    chunk_size = st.number_input("Chunk size (chars)", min_value=200, max_value=5000, value=1000, step=100)
    overlap = st.number_input("Overlap (chars)", min_value=0, max_value=1000, value=200, step=50)
    chunk_method = st.selectbox("Chunk method", ["sentence", "char"], index=0)
    retrieval_mode = st.selectbox("Retrieval mode", ["hybrid", "vector"], index=0)
    hybrid_alpha = st.slider("Hybrid alpha (vector weight)", 0.0, 1.0, 0.65, 0.05,
                              disabled=(retrieval_mode != "hybrid"))
    use_reranker = st.checkbox("Enable cross-encoder re-ranking", value=False)
    top_k = st.number_input("Top K chunks", min_value=1, max_value=10, value=3)
    ollama_model = st.text_input("Ollama model", value="llama2")
    dpi = st.number_input("PDF OCR DPI", min_value=100, max_value=600, value=300, step=50)

# --- Session state ---
if "rag" not in st.session_state:
    st.session_state["rag"] = None
if "index_built" not in st.session_state:
    st.session_state["index_built"] = False
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


def get_pipeline() -> RAGPipeline:
    """(Re)build the pipeline object if settings changed."""
    key = (chunk_size, overlap, chunk_method, retrieval_mode, hybrid_alpha, use_reranker, ollama_model)
    if st.session_state.get("rag") is None or st.session_state.get("rag_key") != key:
        st.session_state["rag"] = RAGPipeline(
            chunk_size=chunk_size,
            overlap=overlap,
            chunk_method=chunk_method,
            retrieval_mode=retrieval_mode,
            hybrid_alpha=hybrid_alpha,
            use_reranker=use_reranker,
            ollama_model=ollama_model,
        )
        st.session_state["rag_key"] = key
        st.session_state["index_built"] = False
    return st.session_state["rag"]


# --- Document ingestion ---
st.subheader("1. Ingest documents")
tab_pdf, tab_txt, tab_hf = st.tabs(["PDF", "Text file", "Hugging Face dataset"])

rag = get_pipeline()

with tab_pdf:
    pdf_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_uploader")
    if pdf_file is not None and st.button("Ingest PDF"):
        with st.spinner("Running OCR..."):
            try:
                rag.ingest_pdf_bytes(pdf_file.getvalue(), source_name=pdf_file.name, dpi=dpi)
                st.success(f"Ingested {pdf_file.name}")
            except Exception as e:
                st.error(f"OCR error: {e}")

with tab_txt:
    txt_file = st.file_uploader("Choose a .txt file", type="txt", key="txt_uploader")
    if txt_file is not None and st.button("Ingest text file"):
        rag.ingest_txt_bytes(txt_file.getvalue(), source_name=txt_file.name)
        st.success(f"Ingested {txt_file.name}")

with tab_hf:
    col1, col2, col3 = st.columns(3)
    with col1:
        hf_name = st.text_input("Dataset name", placeholder="e.g. squad")
    with col2:
        hf_text_col = st.text_input("Text column", placeholder="e.g. context")
    with col3:
        hf_split = st.text_input("Split", value="train")
    hf_max = st.number_input("Max records", min_value=1, max_value=100000, value=500)
    if st.button("Ingest Hugging Face dataset"):
        if not hf_name or not hf_text_col:
            st.error("Dataset name and text column are required.")
        else:
            with st.spinner(f"Loading {hf_name}..."):
                try:
                    rag.ingest_hf(hf_name, text_column=hf_text_col, split=hf_split, max_records=hf_max)
                    st.success(f"Ingested {hf_name} ({hf_split})")
                except Exception as e:
                    st.error(f"Dataset load error: {e}")

st.caption(f"Records ingested so far: {len(rag.records)}")

# --- Build index ---
st.subheader("2. Build index")
if st.button("Build index (chunk + embed + store)", disabled=(len(rag.records) == 0)):
    with st.spinner("Chunking, embedding, and indexing..."):
        try:
            stats = rag.build_index()
            st.session_state["index_built"] = True
            st.success(
                f"Index built — {stats['num_chunks']} chunks from {stats['num_records']} "
                f"record(s), embedding dim {stats['embedding_dim']}."
            )
        except Exception as e:
            st.error(f"Index build error: {e}")

if st.session_state.get("index_built") and st.checkbox("Show sample chunks"):
    for c in rag.chunks[:5]:
        st.markdown(f"**Chunk {c['id']}** (source: `{c.get('source')}`, page {c.get('page_number')})")
        st.write(c["chunk_text"][:400])

# --- Chat ---
st.markdown("---")
st.subheader("3. Chat with your documents")
user_query = st.text_input("Ask a question")

if st.button("Submit", disabled=not st.session_state.get("index_built")) and user_query.strip():
    with st.spinner("Retrieving and generating..."):
        result = rag.ask(user_query, top_k=top_k, verbose=False)
    st.session_state["chat_history"].append({"role": "user", "text": user_query})
    st.session_state["chat_history"].append({
        "role": "assistant",
        "text": result["answer"],
        "sources": result["sources"],
        "latency": result["latency_seconds"],
    })

for msg in st.session_state["chat_history"][-20:]:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['text']}")
    else:
        st.markdown(f"**Assistant:** {msg['text']}")
        with st.expander(f"Sources & scores ({msg['latency']}s)"):
            for s in msg["sources"]:
                st.write(f"`{s['source']}` (page {s['page_number']}) — score {round(s['score'], 3) if s['score'] is not None else 'n/a'}")
                st.caption(s["snippet"])

# --- Validation & metrics ---
st.markdown("---")
st.subheader("4. Validation & metrics report")
sample_qs_raw = st.text_area(
    "Sample questions for validation (one per line)",
    placeholder="What is the refund policy?\nHow long does shipping take?",
)
if st.button("Run validation + generate metrics report", disabled=not st.session_state.get("index_built")):
    sample_questions = [q.strip() for q in sample_qs_raw.splitlines() if q.strip()]
    if not sample_questions:
        st.error("Enter at least one sample question.")
    else:
        with st.spinner("Running validation..."):
            results = run_validation(rag, sample_questions, top_k=top_k, log_path="validation_log.json")
            report = generate_metrics_report(rag, results, report_path="metrics_report.md")
        st.success("validation_log.json and metrics_report.md written.")
        st.markdown(report)

st.caption("Uses Ollama locally for generation. Ensure Ollama is running (e.g. `ollama run llama2`).")
