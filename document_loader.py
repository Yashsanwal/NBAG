"""
document_loader.py

 for loading documents into a RAG pipeline.

It can load:
    - PDF files (.pdf)
    - Text files (.txt)
    - Markdown files (.md)
    - CSV files (.csv)
    - Word documents (.docx)
    - Web pages (URLs)

"""

import logging
import os
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
    WebBaseLoader,
)

# Set up basic logging so we can print status messages instead of using print().
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def is_url(source: str) -> bool:
    """
    Check if the given source string is a web URL.

    Example: "https://example.com" -> True
              "myfile.pdf"          -> False
    """
    return source.startswith("http://") or source.startswith("https://")


def get_file_extension(file_path: str) -> str:
    """
    Get the lowercase file extension of a file path.

    Example: "notes/FODL.PDF" -> ".pdf"
    """
    _, extension = os.path.splitext(file_path)
    return extension.lower()


def get_loader_for_source(source: str):
    """
    Pick the correct LangChain loader based on the source.

    - If the source is a URL, use WebBaseLoader.
    - If it's a local file, look at its extension and pick a matching loader.

    Raises:
        FileNotFoundError: if a local file path does not exist.
        ValueError: if the file type is not supported.
    """
    # Case 1: source is a URL (web page)
    if is_url(source):
        logger.info("Source is a URL: %s", source)
        return WebBaseLoader(source)

    # Case 2: source is a local file, so it must actually exist
    if not os.path.exists(source):
        raise FileNotFoundError(f"Could not find file: '{source}'")

    extension = get_file_extension(source)
    logger.info("Source is a local file: %s (type: %s)", source, extension)

    # Pick the right loader based on the file extension.
    # To support a new file type later, just add another "elif" here.
    if extension == ".pdf":
        return PyPDFLoader(source)
    elif extension == ".txt":
        return TextLoader(source, encoding="utf-8")
    elif extension == ".md":
        return UnstructuredMarkdownLoader(source)
    elif extension == ".csv":
        return CSVLoader(source, encoding="utf-8")
    elif extension == ".docx":
        return UnstructuredWordDocumentLoader(source)
    else:
        raise ValueError(
            f"Unsupported file type '{extension}'. "
            "Supported types are: .pdf, .txt, .md, .csv, .docx, or a URL."
        )


def load_documents(
    source: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> Tuple[List[Document], List[Document]]:
    """
    Load a document (file path or URL) and split it into chunks.

    Args:
        source: Path to a file (.pdf, .txt, .md, .csv, .docx) or a URL.
        chunk_size: Max number of characters in each chunk.
        chunk_overlap: How many characters each chunk shares with the next one.

    Returns:
        A tuple: (original_documents, chunks)

    Raises:
        FileNotFoundError: if the local file does not exist.
        ValueError: if the file type is not supported.
        RuntimeError: if loading the document fails for another reason.
    """
    logger.info("Loading documents from: %s", source)

    # Step 1: Pick the correct loader and load the raw documents.
    try:
        loader = get_loader_for_source(source)
        documents = loader.load()
    except (FileNotFoundError, ValueError):
        # These are already clear error messages, so just pass them along.
        raise
    except Exception as error:
        # Catch anything else (network issues, broken files, etc.)
        raise RuntimeError(f"Failed to load '{source}': {error}") from error

    if not documents:
        logger.warning("No text was found in: %s", source)
        return documents, []

    logger.info("Loaded %d document(s).", len(documents))

    # Step 2: Split the documents into smaller chunks for embedding.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    logger.info("Split into %d chunk(s).", len(chunks))

    return documents, chunks


# This part only runs if you execute this file directly, e.g.:
#   python document_loader.py FODL.pdf
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python document_loader.py <file-path-or-url>")
        sys.exit(1)

    source_path = sys.argv[1]
    docs, chunks = load_documents(source_path)

    print(f"Documents loaded: {len(docs)}")
    print(f"Chunks created:   {len(chunks)}")
    if chunks:
        print("\nFirst chunk preview:")
        print(chunks[0].page_content[:300])