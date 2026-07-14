"""
vector_store.py

The Vector Database Stage of the RAG pipeline.

Pipeline so far:
    PDF -> Loader -> Cleaning -> Chunking -> Embeddings -> Vector Database

This file takes chunks of text, embeds them, and saves them in a Chroma
vector database on disk, so they can be searched by meaning later
(that search step is the next stage, retrieval - not built yet).

Example:
    from document_loader import load_documents
    from vector_store import save_to_vector_store

    docs, chunks = load_documents("FODL.pdf")
    vector_db = save_to_vector_store(chunks)
"""

import logging
import os
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document

import config
from embeddings import get_embedding_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def save_to_vector_store(chunks: List[Document]) -> Chroma:
    """
    Embed a list of chunks and save them in a persistent Chroma database.

    Args:
        chunks: A list of LangChain Document objects (e.g. from
            document_loader.load_documents()).

    Returns:
        The Chroma vector store, already saved to disk at config.VECTOR_DB_PATH.

    Raises:
        ValueError: if OPENAI_API_KEY is missing.
        RuntimeError: if no chunks were given.
    """
    if not chunks:
        raise RuntimeError("No chunks were given. Nothing to store.")

    logger.info("Preparing to store %d chunk(s) in the vector database...", len(chunks))

    # Make sure the folder that will hold the database actually exists.
    os.makedirs(config.VECTOR_DB_PATH, exist_ok=True)

    # get_embedding_model() already checks that OPENAI_API_KEY is set.
    embedder = get_embedding_model()

    logger.info("Embedding and saving chunks to: %s", config.VECTOR_DB_PATH)

    # Chroma.from_documents() does two things in one step:
    #   1. Calls the embedding model on every chunk's text.
    #   2. Saves the resulting vectors (plus text and metadata) to disk.
    # Because it embeds internally, we pass in the raw chunks here rather
    # than calling embeddings.create_embeddings() first - that avoids
    # paying to embed the same text twice.
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        persist_directory=config.VECTOR_DB_PATH,
    )

    stored_count = vector_db._collection.count()
    logger.info("Stored %d chunk(s) in the vector database.", stored_count)

    return vector_db


def load_vector_store() -> Chroma:
    """
    Load an existing Chroma vector database from disk.

    Returns:
        The Chroma vector store, loaded from config.VECTOR_DB_PATH.

    Raises:
        ValueError: if OPENAI_API_KEY is missing.
        FileNotFoundError: if no database exists yet at config.VECTOR_DB_PATH.
    """
    if not os.path.exists(config.VECTOR_DB_PATH):
        raise FileNotFoundError(
            f"No vector database found at '{config.VECTOR_DB_PATH}'. "
            "Call save_to_vector_store() first to create one."
        )

    embedder = get_embedding_model()

    logger.info("Loading vector database from: %s", config.VECTOR_DB_PATH)

    return Chroma(
        persist_directory=config.VECTOR_DB_PATH,
        embedding_function=embedder,
    )