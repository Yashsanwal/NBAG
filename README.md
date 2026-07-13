# 🚀 Production RAG Pipeline

A production-ready Retrieval-Augmented Generation (RAG) pipeline built from scratch using Python, LangChain, OpenAI Embeddings, and ChromaDB.

The goal of this project is not just to build a working RAG application, but to understand every component of the pipeline and implement it using clean, modular, and production-style engineering practices.

---

## Features

- Multi-format document loading
  - PDF
  - TXT
  - Markdown
  - CSV
  - DOCX
  - URLs

- Intelligent document chunking

- OpenAI Embedding generation

- Duplicate chunk removal

- Empty chunk filtering

- Configurable batching

- Structured logging

- Modular project architecture

---

## Tech Stack

- Python
- LangChain
- OpenAI
- ChromaDB
- dotenv
- Logging

---

## Current Progress

- [x] Project setup
- [x] Document Loader
- [x] Text Chunking
- [x] Embedding Generation
- [ ] Vector Database
- [ ] Semantic Retrieval
- [ ] Prompt Engineering
- [ ] LLM Integration
- [ ] Hybrid Search
- [ ] FastAPI
- [ ] Docker
- [ ] Kubernetes

---

## Project Structure
rag-pipeline/ <br>
│             <br>
├── data/     <br>
├── src/       <br>
│ ├── document_loader.py     <br>
│ ├── embedding_model.py      <br>
│ ├── config.py               <br>
│<br>
├── .env.example             <br>
├── requirements.txt          <br>
├── pyproject.toml           <br>
└── README.md                 <br>
<br> 
<br>
<br>
<br>
---


## Learning Goals

This repository is being built incrementally to understand:

- Document ingestion
- Text chunking
- Embeddings
- Vector databases
- Retrieval
- Reranking
- Prompt Engineering
- Agentic RAG
- Production deployment

---

## Future Improvements

- Hybrid Search (BM25 + Vector Search)
- Rerankers
- GraphRAG
- FastAPI API
- Docker
- Kubernetes Deployment
- Monitoring
- CI/CD
