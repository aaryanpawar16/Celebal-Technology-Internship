"""Answer generation: tries Ollama (local) first, then cloud keys, then a local fallback."""
import requests
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, ANTHROPIC_API_KEY, OPENAI_API_KEY


def generate_response(question: str, context: str, memory: str) -> str:
    prompt = build_prompt(question, context, memory)

    ollama_answer = _call_ollama(prompt)
    if ollama_answer is not None:
        return ollama_answer

    if ANTHROPIC_API_KEY:
        return _call_anthropic(prompt)
    if OPENAI_API_KEY:
        return _call_openai(prompt)
    return _local_fallback(question, context, memory)


def build_prompt(question: str, context: str, memory: str) -> str:
    parts = []
    if memory:
        parts.append(memory)
    if context:
        parts.append(f"Relevant information:\n{context}")
    parts.append(f"Question: {question}\nAnswer clearly and concisely.")
    return "\n\n".join(parts)


def _call_ollama(prompt: str, timeout: int = 30) -> str | None:
    """Returns the model's reply, or None if Ollama isn't running/reachable."""
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()
    except requests.exceptions.RequestException:
        return None  # Ollama not running locally — caller falls back


def _call_anthropic(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def _call_openai(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def _local_fallback(question: str, context: str, memory: str) -> str:
    """No LLM available at all: just compose the retrieved info into a readable answer."""
    if not context:
        return "I don't have enough local information to answer that confidently."
    intro = f"{memory}\n\n" if memory else ""
    return f"{intro}Based on what I found:\n{context}"
