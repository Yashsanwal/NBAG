"""
A small end-to-end test of the pipeline so far:
    load a file -> split into chunks -> embed and store them
    in the vector database -> print a summary.

"""

import sys
import time

import config
from document_loader import load_documents
from vector_store import save_to_vector_store


def main() -> None:
    """Load a file, store its chunks in the vector database, and print the results."""
    if len(sys.argv) != 2:
        print("Usage: python main.py <file-path-or-url>")
        sys.exit(1)

    source = sys.argv[1]
    start_time = time.time()

    # Step 1: Load and chunk the document.
    docs, chunks = load_documents(source)

    # Step 2: Embed the chunks and save them in the Chroma vector database.
    vector_db = save_to_vector_store(chunks)

    elapsed_seconds = time.time() - start_time

    # Step 3: Print a summary so you can confirm everything worked.
    print("\n--- Vector Store Test Results ---")
    print(f"Original documents:  {len(docs)}")
    print(f"Chunks created:      {len(chunks)}")
    print(f"Chunks stored in DB: {vector_db._collection.count()}")

    # Fetch one stored record back (not a similarity search) just to show
    # what an embedding looks like once it's saved in the database.
    sample = vector_db.get(limit=1, include=["embeddings"])
    if len(sample["embeddings"]) > 0:
        first_embedding = sample["embeddings"][0]
        print(f"Embedding dimension: {len(first_embedding)}")
        print(f"First 5 values:      {list(first_embedding[:5])}")

    print(f"Total time taken:    {elapsed_seconds:.2f} seconds")
    print(f"Database saved at:   {config.VECTOR_DB_PATH}")


if __name__ == "__main__":
    main()