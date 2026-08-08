"""LangGraph orchestration: routes a query to KG, RAG, and/or a live tool, then answers."""
import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END

import memory_store
import knowledge_graph as kg
import rag_pipeline as rag
import tools
import llm


class ChatState(TypedDict):
    user_id: str
    question: str
    context: Annotated[list[str], operator.add]  # parallel nodes append here safely
    memory: str
    answer: str


class ChatbotGraph:
    def __init__(self):
        self.kg = kg.build_graph()
        self.chunks = rag.load_chunks()
        self.index = rag.build_index(self.chunks)
        self.graph = self._build()

    # --- Nodes: each returns only the keys it changes ---
    def memory_node(self, state: ChatState) -> dict:
        return {"memory": memory_store.memory_as_context(state["user_id"])}

    def kg_node(self, state: ChatState) -> dict:
        entity = kg.find_mentioned_entity(self.kg, state["question"])
        facts = kg.query_entity(self.kg, entity) if entity else []
        return {"context": facts}

    def rag_node(self, state: ChatState) -> dict:
        results = rag.retrieve(state["question"], self.index, self.chunks)
        return {"context": results}

    def tool_node(self, state: ChatState) -> dict:
        q = state["question"]
        topic = q.split("about")[-1].strip() if "about" in q else q
        live_info = tools.wikipedia_lookup(topic)
        return {"context": [f"[Live] {live_info}"]}

    def answer_node(self, state: ChatState) -> dict:
        context_text = "\n".join(state["context"])
        answer = llm.generate_response(state["question"], context_text, state["memory"])
        memory_store.log_turn(state["user_id"], state["question"], answer)
        return {"answer": answer}

    # --- Router: decides which knowledge nodes to visit (can fan out to several) ---
    def route(self, state: ChatState) -> list[str]:
        branches = ["rag"]  # always check static knowledge
        if kg.find_mentioned_entity(self.kg, state["question"]):
            branches.append("kg")
        if tools.needs_live_lookup(state["question"]):
            branches.append("tool")
        return branches

    def _build(self):
        g = StateGraph(ChatState)
        g.add_node("memory", self.memory_node)
        g.add_node("kg", self.kg_node)
        g.add_node("rag", self.rag_node)
        g.add_node("tool", self.tool_node)
        g.add_node("answer", self.answer_node)

        g.set_entry_point("memory")
        g.add_conditional_edges("memory", self.route, {
            "rag": "rag", "kg": "kg", "tool": "tool",
        })
        # Every knowledge branch funnels into the answer node.
        g.add_edge("rag", "answer")
        g.add_edge("kg", "answer")
        g.add_edge("tool", "answer")
        g.add_edge("answer", END)
        return g.compile()

    def ask(self, user_id: str, question: str) -> str:
        state: ChatState = {"user_id": user_id, "question": question, "context": [], "memory": "", "answer": ""}
        result = self.graph.invoke(state)
        return result["answer"]
