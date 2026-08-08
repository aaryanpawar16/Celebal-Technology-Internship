import os

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DOCS_PATH = os.path.join(DATA_DIR, "sample_docs.txt")
MEMORY_PATH = os.path.join(DATA_DIR, "memory.json")

# --- RAG settings ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3

# --- Ollama (local LLM, no API key needed) ---
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")

# --- Optional cloud LLM keys (only used if Ollama isn't running) ---
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# --- Ollama (free, fully local — no API key needed) ---
USE_OLLAMA = os.environ.get("USE_OLLAMA", "true").lower() == "true"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
