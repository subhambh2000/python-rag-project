# Python RAG Project — Obsidian Notes Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built **entirely from scratch in Python** — no LangChain, no LlamaIndex, no orchestration frameworks. Every stage of the pipeline is implemented manually, from markdown-aware chunking through grounded generation, to understand how RAG actually works under the hood.

The chatbot answers questions grounded in a personal knowledge base of **Obsidian notes**, retrieved from a local Qdrant vector store.

## Pipeline

| Module       | Responsibility |
|--------------|----------------|
| `chunker`    | Markdown-aware chunking — detects tables, applies overlap between chunks, injects filename metadata into each chunk |
| `embedder`   | Generates embeddings locally with GPU acceleration using `sentence-transformers` |
| `ingestor`   | Batches embeddings and upserts them into Qdrant using deterministic IDs (so re-ingestion is idempotent) |
| `retriever`  | Semantic search against Qdrant, with similarity threshold filtering and deduplication of results |
| `generator`  | Builds grounded prompts from retrieved context, streams the LLM response, and handles finish reasons |
| `chatbot`    | CLI chat loop with clean exit handling and a dual-gate hallucination prevention check |

## Tech stack

| Component         | Choice |
|--------------------|--------|
| Vector database     | Qdrant |
| Embedding model      | Qwen3-Embedding-0.6B — run locally on an NVIDIA GPU (`sentence-transformers` + PyTorch with CUDA) |
| LLM (generation)      | `llama-3.3-70b-versatile` via Groq API |
| Orchestration          | None — built manually, no LangChain/LlamaIndex |
| Data source              | Obsidian markdown notes |

## How it works

1. **Chunk** — Obsidian markdown notes are split into overlapping, table-aware chunks, each tagged with its source filename.
2. **Data/notes** — The folder where markdown notes will be saved
3. **Embed** — Chunks are embedded locally on an NVIDIA GPU (PyTorch + CUDA) using Qwen3-Embedding-0.6B.
4. **Ingest** — Embeddings are batch-upserted into Qdrant with deterministic IDs.
5. **Retrieve** — At query time, semantic search finds relevant chunks, filtered by similarity threshold and deduplicated.
6. **Generate** — Retrieved chunks are grounded into a prompt and streamed from `llama-3.3-70b-versatile` via the Groq API.
7. **Chat** — A CLI loop drives the conversation, with a dual-gate check to reduce hallucinated answers.

## Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/subhambh2000/python-rag-project.git
   cd python-rag-project
   ```
2. Start Qdrant locally (via Docker):
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```
3. Set your Groq API key as an environment variable:
   ```bash
   export GROQ_API_KEY=your_key_here
   ```
4. Point the ingestion pipeline at your Obsidian vault, run it to chunk/embed/ingest your notes, then launch the CLI chatbot.

## Motivation

This is Project 1 in a series of hands-on projects to learn RAG and AI engineering fundamentals by building each component manually rather than relying on high-level frameworks.

## License

MIT
