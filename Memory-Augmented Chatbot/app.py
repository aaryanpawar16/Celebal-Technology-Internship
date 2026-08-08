import streamlit as st
from graph_workflow import ChatbotGraph
import memory_store
import evaluation

st.set_page_config(page_title="Memory-Augmented Chatbot", layout="wide")
st.title("🧠 Memory-Augmented Chatbot")
st.caption("RAG + Knowledge Graph + Long-Term Memory + LangGraph — fully local, no API key required")

# --- Setup (cached so the index/graph only build once) ---
@st.cache_resource
def load_bot():
    return ChatbotGraph()

bot = load_bot()

# --- Sidebar: user + memory ---
st.sidebar.header("User")
user_id = st.sidebar.text_input("User ID", value="aaryan")

st.sidebar.header("Add a fact to memory")
new_fact = st.sidebar.text_input("e.g. 'prefers concise answers'")
if st.sidebar.button("Save fact") and new_fact:
    memory_store.remember_fact(user_id, new_fact)
    st.sidebar.success("Saved.")

st.sidebar.header("Known facts")
st.sidebar.write(memory_store.get_user_memory(user_id)["facts"] or "None yet")

# --- Chat state ---
if "chat" not in st.session_state:
    st.session_state.chat = []

# --- Chat history ---
for turn in st.session_state.chat:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        with st.expander("Debug: retrieval + evaluation"):
            st.text(turn["context"])
            st.json(turn["scores"])

# --- New question ---
question = st.chat_input("Ask me anything...")
if question:
    with st.chat_message("user"):
        st.write(question)

    state = {"user_id": user_id, "question": question, "context": [], "memory": "", "answer": ""}
    result = bot.graph.invoke(state)
    context_text = "\n".join(result["context"])

    with st.chat_message("assistant"):
        st.write(result["answer"])

    scores = evaluation.evaluate_turn(question, context_text, result["answer"])

    st.session_state.chat.append({
        "question": question,
        "answer": result["answer"],
        "context": context_text,
        "scores": scores,
    })
