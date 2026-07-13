"""

The Embedding Stage of the RAG pipeline.

Pipeline so far:
    PDF -> Loader -> Cleaning -> Chunking -> Embeddings -> (Vector Database later)

This file turns text chunks into embedding vectors (lists of numbers) using
OpenAI's embedding API, so they can eventually be stored in a vector database
and searched by meaning instead of exact keywords.

Example:
    from document_loader import load_documents
    from embeddings import create_embeddings

    docs, chunks = load_documents("FODL.pdf")
    embedded_chunks = create_embeddings(chunks)

    print(embedded_chunks[0]["embedding"][:5])
"""

import hashlib
import logging
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_embedding_model() -> OpenAIEmbeddings:
    """
    Build an OpenAIEmbeddings client using the settings from config.py.

    Returns:
        A ready-to-use OpenAIEmbeddings instance.

    Raises:
        ValueError: if OPENAI_API_KEY is missing from the environment.
    """
    # Check the API key exists before we try to use it, so we fail early
    # with a clear message instead of a confusing error from the API.
    config.validate_config()

    return OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        api_key=config.OPENAI_API_KEY,
        timeout=config.REQUEST_TIMEOUT_SECONDS,
        max_retries=config.MAX_RETRY_ATTEMPTS,
    )


def _hash_text(text: str) -> str:
    """Create a short fingerprint for a piece of text, used to spot duplicates."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _remove_empty_chunks(chunks: List[Document]) -> List[Document]:
    """
    Drop chunks that have no real text in them.

    Sending empty text to the API wastes a request and gives a useless result.
    """
    valid_chunks = []
    for chunk in chunks:
        if chunk.page_content and chunk.page_content.strip():
            valid_chunks.append(chunk)
        else:
            logger.warning("Skipping an empty chunk.")
    return valid_chunks


def _remove_duplicate_chunks(chunks: List[Document]) -> List[Document]:
    """
    Drop chunks whose text is identical to a chunk we already saw.

    This avoids paying to embed the exact same text twice in one run.
    """
    seen_hashes = set()
    unique_chunks = []
    for chunk in chunks:
        text_hash = _hash_text(chunk.page_content)
        if text_hash in seen_hashes:
            logger.info("Skipping a duplicate chunk.")
            continue
        seen_hashes.add(text_hash)
        unique_chunks.append(chunk)
    return unique_chunks


def create_embeddings(chunks: List[Document]) -> List[Dict[str, Any]]:
    """
    Create an embedding vector for each chunk of text.

    Args:
        chunks: A list of LangChain Document objects (e.g. produced by
            document_loader.load_documents()).

    Returns:
        A list of dictionaries, one per chunk, each shaped like:
            {
                "text": "the original chunk text",
                "embedding": [0.012, -0.034, ...],
                "metadata": {...},
            }

    Raises:
        ValueError: if OPENAI_API_KEY is missing.
        RuntimeError: if a batch fails to embed even after retrying.
    """
    if not chunks:
        logger.warning("No chunks were given. Nothing to embed.")
        return []

    logger.info("Preparing %d chunk(s) for embedding...", len(chunks))

    # Step 1: Clean up the input before spending any API calls on it.
    chunks = _remove_empty_chunks(chunks)
    chunks = _remove_duplicate_chunks(chunks)

    if not chunks:
        logger.warning("No valid chunks left after cleaning. Nothing to embed.")
        return []

    embedder = get_embedding_model()
    results: List[Dict[str, Any]] = []

    # Step 2: Process the chunks in batches instead of one at a time.
    # One request with 100 chunks is much faster and cheaper than
    # 100 separate requests with one chunk each.
    total_batches = (len(chunks) + config.BATCH_SIZE - 1) // config.BATCH_SIZE

    for batch_index in range(total_batches):
        start = batch_index * config.BATCH_SIZE
        end = start + config.BATCH_SIZE
        batch = chunks[start:end]
        texts = [chunk.page_content for chunk in batch]

        logger.info("Embedding batch %d of %d (%d chunks)...", batch_index + 1, total_batches, len(batch))

        # OpenAIEmbeddings already retries automatically on temporary failures
        # (network errors, rate limits) using config.MAX_RETRY_ATTEMPTS.
        # We still wrap the call so a final failure gives a clear message.
        try:
            vectors = embedder.embed_documents(texts)
        except Exception as error:
            raise RuntimeError(
                f"Failed to create embeddings for batch {batch_index + 1}: {error}"
            ) from error

        for chunk, vector in zip(batch, vectors):
            results.append(
                {
                    "text": chunk.page_content,
                    "embedding": vector,
                    "metadata": chunk.metadata,
                }
            )

    logger.info("Done. Created %d embedding(s).", len(results))
    return results