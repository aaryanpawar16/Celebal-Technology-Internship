"""Simple long-term memory: stores per-user facts and chat history in a JSON file."""
import json
import os
from config import MEMORY_PATH


def _load_all():
    if not os.path.exists(MEMORY_PATH):
        return {}
    with open(MEMORY_PATH, "r") as f:
        return json.load(f)


def _save_all(data):
    with open(MEMORY_PATH, "w") as f:
        json.dump(data, f, indent=2)


def get_user_memory(user_id: str) -> dict:
    """Return this user's stored facts + history, creating an empty record if new."""
    data = _load_all()
    return data.get(user_id, {"facts": [], "history": []})


def remember_fact(user_id: str, fact: str):
    """Store a durable fact about the user (e.g. a stated preference)."""
    data = _load_all()
    user = data.setdefault(user_id, {"facts": [], "history": []})
    if fact not in user["facts"]:
        user["facts"].append(fact)
    _save_all(data)


def log_turn(user_id: str, question: str, answer: str):
    """Log one conversation turn to history."""
    data = _load_all()
    user = data.setdefault(user_id, {"facts": [], "history": []})
    user["history"].append({"question": question, "answer": answer})
    user["history"] = user["history"][-20:]  # keep it small
    _save_all(data)


def memory_as_context(user_id: str) -> str:
    """Format stored facts as a short text block for prompting."""
    user = get_user_memory(user_id)
    if not user["facts"]:
        return ""
    return "Known facts about the user: " + "; ".join(user["facts"])
