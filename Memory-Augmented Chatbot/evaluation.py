"""Lightweight evaluation: word-overlap based scores. No external API needed."""
import re


def _words(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def context_relevance(question: str, context: str) -> float:
    """How much of the question's vocabulary shows up in the retrieved context."""
    q, c = _words(question), _words(context)
    if not q:
        return 0.0
    return round(len(q & c) / len(q), 2)


def faithfulness(answer: str, context: str) -> float:
    """How much of the answer's vocabulary is actually grounded in the context."""
    a, c = _words(answer), _words(context)
    if not a:
        return 0.0
    return round(len(a & c) / len(a), 2)


def answer_correctness(answer: str, reference: str) -> float:
    """Overlap between the answer and a human-provided reference answer, if available."""
    a, r = _words(answer), _words(reference)
    if not r:
        return None
    return round(len(a & r) / len(r), 2)


def evaluate_turn(question: str, context: str, answer: str, reference: str = "") -> dict:
    return {
        "context_relevance": context_relevance(question, context),
        "faithfulness": faithfulness(answer, context),
        "answer_correctness": answer_correctness(answer, reference) if reference else None,
    }
