"""Dynamic tool: fetches real-time info from Wikipedia (free, no API key)."""
import wikipedia


def wikipedia_lookup(topic: str) -> str:
    try:
        return wikipedia.summary(topic, sentences=2)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"'{topic}' is ambiguous. Options include: {', '.join(e.options[:5])}"
    except Exception:
        return f"No live Wikipedia result found for '{topic}'."


def needs_live_lookup(text: str) -> bool:
    """Simple keyword trigger for when to use a live tool instead of static RAG."""
    triggers = ["current", "latest", "today", "now", "recent", "who is", "what is happening"]
    lowered = text.lower()
    return any(t in lowered for t in triggers)
