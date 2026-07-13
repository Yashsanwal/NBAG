from dotenv import load_dotenv
load_dotenv()      # Load environment variables from .env file



"""

A small end-to-end test of the pipeline so far:
    load a file -> split into chunks -> create embeddings -> print a summary.

Run it with:
    python main.py FODL.pdf
"""

import sys
import time

from document_loader import load_documents
from embeddings import create_embeddings


def main() -> None:
    """Load a file, embed its chunks, and print the results."""
    if len(sys.argv) != 2:
        print("Usage: python main.py <file-path-or-url>")
        sys.exit(1)

    source = sys.argv[1]
    start_time = time.time()

    # Step 1: Load and chunk the document.
    docs, chunks = load_documents(source)

    # Step 2: Create embeddings for the chunks.
    embedded_chunks = create_embeddings(chunks)

    elapsed_seconds = time.time() - start_time

    # Step 3: Print a summary so you can confirm everything worked.
    print("\n--- Embedding Test Results ---")
    print(f"Original documents:  {len(docs)}")
    print(f"Chunks created:      {len(chunks)}")
    print(f"Embeddings created:  {len(embedded_chunks)}")

    if embedded_chunks:
        first_embedding = embedded_chunks[0]["embedding"]
        print(f"Embedding dimension: {len(first_embedding)}")
        print(f"First 5 values:      {first_embedding[:5]}")

    print(f"Total time taken:    {elapsed_seconds:.2f} seconds")
    print(
        "API usage info:      langchain-openai does not expose token/cost "
        "counts from embed_documents(). Check your usage at "
        "platform.openai.com/usage if you need exact numbers."
    )


if __name__ == "__main__":
    main()