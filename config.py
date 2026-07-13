"""
config.py

All the settings for the RAG pipeline live here, in one place.
If you want to change the embedding model, batch size, or timeout,
you only need to edit this file (or your .env file) - nothing else.
"""

import os

from dotenv import load_dotenv
load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Embedding model ---
# To switch models later (e.g. to "text-embedding-3-large"),
# change this one line - no other code needs to change.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# --- Batching and reliability settings ---
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))
MAX_RETRY_ATTEMPTS = int(os.getenv("EMBEDDING_MAX_RETRIES", "3"))
REQUEST_TIMEOUT_SECONDS = int(os.getenv("EMBEDDING_TIMEOUT", "60"))

# --- Vector database path (not used yet, will be needed in the next stage) ---
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "vector_db")


def validate_config() -> None:
    """
    Make sure required settings are present before we try to call the API.

    Raises:
        ValueError: if OPENAI_API_KEY is missing, with a clear instruction
            on how to fix it.
    """
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is missing.\n"
            "Fix: copy '.env.example' to '.env' and set your real API key, "
            "e.g. OPENAI_API_KEY=sk-..."
        )