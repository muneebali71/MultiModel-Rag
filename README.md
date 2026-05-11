# Multimodal PDF RAG API
### Project Documentation

---

## Overview

This is a fully production-grade, end-to-end **Multimodal Retrieval-Augmented Generation (RAG)** system. It accepts PDF documents containing any combination of text, tables, images, and equations — processes them through a 7-layer AI pipeline — and exposes a clean REST API that answers natural language questions with cited, grounded responses.

The entire stack runs on **free-tier services** with no paid API keys required to get started.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| PDF Parsing | Unstructured (`hi_res` strategy) |
| Vision LLM | Groq — Llama 4 Scout (sees images + tables) |
| Answer LLM | Groq — Llama 3.1 8B Instant |
| Embeddings | FastEmbed — Dense + BM25 + ColBERT |
| Vector Database | Qdrant Cloud |
| API Framework | FastAPI + Uvicorn |
| Language | Python 3.11 |

---

## Project Structure

```
rag_project/
├── main.py                   FastAPI application entry point
├── config.py                 Environment-based configuration
├── logger.py                 Structured logging with @trace decorator
├── schemas.py                Pydantic request/response models
├── requirements.txt          All dependencies
├── .env                      API keys (never committed to version control)
│
├── api/routes/
│   ├── health.py             GET  /health
│   ├── upload.py             POST /api/upload
│   └── query.py              POST /api/ask
│
├── core/
│   ├── pipeline.py           Orchestrates all 7 pipeline layers
│   ├── retrieval.py          Hybrid search with ColBERT reranking
│   └── generation.py         Answer generation via Groq LLM
│
├── utils/
│   ├── pdf_parser.py         PDF extraction and structure-aware chunking
│   ├── chunker.py            Document enrichment and LangChain Document builder
│   ├── summarizer.py         Groq vision multimodal summarization
│   ├── embedder.py           Triple-model embedding pipeline
│   └── cleaning.py           Text normalization utilities
│
├── db/
│   └── qdrant_client.py      Qdrant collection management and vector indexing
│
└── logs/
    └── app.log               Auto-generated structured log output
```

---

## The 7-Layer Pipeline

When a PDF is uploaded, it passes through 7 sequential processing layers before becoming queryable.

```
PDF Upload
    │
    ▼
Layer 1 — PDF Extraction
    extract_document() via Unstructured hi_res
    Extracts: text blocks, tables (HTML), images (base64), equations
    │
    ▼
Layer 2 — Structure-Aware Chunking
    create_chunking_by_title()
    Groups elements by document headings and section boundaries
    Config: max 3000 chars, new chunk after 2400, combine under 500
    │
    ▼
Layer 3 — Content Type Separation
    separate_types()
    Each chunk split into: raw text / tables[] / images[]
    │
    ▼
Layer 4 — Document Building
    summarize_chunks()
    Builds LangChain Documents with flat, non-nested metadata
    │
    ▼
Layer 5 — Multimodal Vision Enrichment
    create_summary() via Groq Vision (Llama 4 Scout)
    Type-aware prompting: figures → visual analysis,
    tables → data extraction, text → concept + Q&A generation
    Graceful fallback to raw text if vision call fails
    │
    ▼
Layer 6 — Triple Embedding Generation
    generate_embeddings()
    Dense (384-dim semantic) + BM25 (sparse keyword) + ColBERT (128×n_tokens)
    All three generated in a single batch pass
    │
    ▼
Layer 7 — Qdrant Indexing
    build_qdrant_index() + index_chunks()
    Creates collection with 3 vector configs
    Uploads in batches with 3-attempt retry logic
    Vision summaries stored in payload for retrieval-time access
    │
    ▼
    Ready for Queries
    │
    ▼
Query Time — Hybrid Retrieval
    hybrid_search()
    Stage 1: Dense semantic prefetch (20 candidates)
    Stage 2: BM25 keyword prefetch (20 candidates)
    Stage 3: ColBERT late-interaction reranking → top_k results
    │
    ▼
Answer Generation
    generate_answer() via Groq (Llama 3.1 8B)
    Grounded, cited answers with [chunk_N] inline citations
    Anti-hallucination system prompt with 20 strict rules
```

---

## API Endpoints

### `GET /health`

Check system and Qdrant connection status.

**Response:**
```json
{
  "status": "ok",
  "qdrant": "ok",
  "collection": "Hybrid_search"
}
```

---

### `POST /api/upload`

Upload a PDF for processing. Processing runs as a background task — the endpoint returns immediately.

**Request:** `multipart/form-data` with a `file` field.

**cURL Example:**
```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@your_document.pdf"
```

**Response:**
```json
{
  "file_id": "a1b2c3d4",
  "filename": "your_document.pdf",
  "status": "processing",
  "message": "Poll /api/status/a1b2c3d4 to check progress."
}
```

---

### `GET /api/status/{file_id}`

Poll to check processing progress. Repeat until status is `done`.

**cURL Example:**
```bash
curl http://localhost:8000/api/status/a1b2c3d4
```

**Response (when complete):**
```json
{
  "file_id": "a1b2c3d4",
  "status": "done",
  "total_chunks": 18,
  "elapsed_sec": 42.3
}
```

**Status values:** `processing` → `done` | `failed`

---

### `POST /api/ask`

Ask a natural language question against the indexed document.

**Request body:**
```json
{
  "question": "What is scaled dot-product attention?",
  "top_k": 5
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is scaled dot-product attention?", "top_k": 5}'
```

**Response:**
```json
{
  "question": "What is scaled dot-product attention?",
  "answer": "Scaled dot-product attention computes a weighted sum of values... [chunk_5]",
  "sources": [
    {
      "chunk_index": 5,
      "score": 24.38,
      "icon": "🖼️",
      "has_images": true,
      "has_tables": false,
      "preview": "The attention function maps a query and a set of key-value pairs..."
    }
  ],
  "total_found": 5,
  "latency_sec": 1.24
}
```

**Source icons:** `📄` text only · `📊` contains table · `🖼️` contains image/figure

---

## Setup & Installation

### Prerequisites

- Python 3.11+
- Free Groq API key — [console.groq.com](https://console.groq.com) (no credit card)
- Free Qdrant Cloud cluster — [cloud.qdrant.io](https://cloud.qdrant.io) (free 1 GB)

### Step 1 — Install dependencies

```bash
cd rag_project
pip install -r requirements.txt
```

### Step 2 — Configure environment

Edit the `.env` file with your keys:

```
GROQ_API_KEY=your_key_from_console.groq.com
QDRANT_URL=https://your-cluster.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_key
```

All other settings are pre-configured with sensible defaults and can be overridden in `.env` if needed.

### Step 3 — Start the server

```bash
python main.py
```

The API will be live at:
- **API base:** `http://localhost:8000`
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc UI:** `http://localhost:8000/redoc`

---

## Configuration Reference

All settings are managed via environment variables in `.env`.

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | Your Groq API key (required) |
| `GROQ_VISION_MODEL` | `meta-llama/llama-4-scout-17b-16e-instruct` | Vision LLM for image/table enrichment |
| `GROQ_CHAT_MODEL` | `llama-3.1-8b-instant` | Chat LLM for answer generation |
| `QDRANT_URL` | — | Qdrant Cloud cluster URL (required) |
| `QDRANT_API_KEY` | — | Qdrant API key (required) |
| `QDRANT_COLLECTION` | `Hybrid_search` | Collection name |
| `DENSE_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Dense embedding model |
| `BM25_MODEL` | `Qdrant/bm25` | Sparse keyword embedding model |
| `COLBERT_MODEL` | `colbert-ir/colbertv2.0` | Late-interaction reranking model |
| `CHUNK_MAX_CHARACTERS` | `3000` | Maximum characters per chunk |
| `CHUNK_NEW_AFTER_N_CHARS` | `2400` | Start new chunk after N chars |
| `CHUNK_COMBINE_UNDER_N_CHARS` | `500` | Combine small chunks under N chars |
| `APP_HOST` | `0.0.0.0` | Server host |
| `APP_PORT` | `8000` | Server port |
| `UPLOAD_DIR` | `/tmp/rag_uploads` | Temporary upload directory |

---

## Retrieval Architecture

The hybrid retrieval system uses three complementary search strategies working in parallel, then combines their results via late-interaction reranking.

**Dense Semantic Search** uses a 384-dimensional sentence transformer to find chunks that are semantically similar to the query — capturing meaning even when exact keywords differ.

**BM25 Sparse Search** uses traditional TF-IDF keyword matching with Qdrant's IDF modifier — strong at finding chunks containing exact technical terms, model names, or specific values.

**ColBERT Reranking** takes the merged candidate pool (up to 40 chunks) and applies token-level late interaction to compute precise relevance scores. This is the final ranking step and produces the `top_k` results returned to the generation layer.

This three-stage approach consistently outperforms any single retrieval method — semantic search handles paraphrase and synonyms, BM25 handles exact terminology, and ColBERT provides fine-grained reranking.

---

## Answer Generation

Answers are generated by Groq's Llama 3.1 8B Instant model using a carefully engineered system prompt with 20 strict rules covering:

- **Grounding** — only information present in retrieved chunks may be used
- **Citations** — every factual claim must end with an inline `[chunk_N]` citation
- **Anti-redundancy** — the same fact may not be stated twice in different wording
- **Content-type handling** — separate rules for text, table, and image chunks
- **Fallback behavior** — exact response when the answer is not in the document

The context passed to the LLM includes both the raw chunk text and the AI-generated vision enrichment (when available), clearly labelled to prevent the model treating the same content as two independent sources.

---

## Logging & Observability

Every module uses a named structured logger (`rag.pipeline`, `rag.retrieval`, etc.). All functions decorated with `@trace` automatically log entry, exit, and execution time.

Logs are written to both console and `logs/app.log`. Key events tracked:

- Per-layer pipeline progress with chunk counts and timing
- Embedding model load events (singleton, loaded once)
- Qdrant connection status and batch upload progress
- Retrieval candidate counts and raw ColBERT scores
- Answer length and source count per query
- All errors with full exception messages

---

*Built with Python 3.11 · FastAPI · LangChain · FastEmbed · Qdrant · Groq*
