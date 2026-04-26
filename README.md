# Multimodal PDF RAG API

Upload PDFs containing text, tables, images and equations — then ask questions in natural language.

## Stack (100% Free)
- **Parsing**: Unstructured hi_res
- **Vision LLM**: Groq (llama-4-scout — sees images + tables)
- **Embeddings**: FastEmbed (dense + BM25 + ColBERT — all local)
- **Vector DB**: Qdrant Cloud (free tier)
- **Answer LLM**: Groq (llama-3.1-8b-instant)
- **API**: FastAPI

---

## Project Structure

```
rag_project/
├── main.py                  ← FastAPI entry point
├── config.py                ← settings from .env
├── logger.py                ← logging + @trace decorator
├── schemas.py               ← Pydantic request/response models
├── requirements.txt
├── .env                     ← your API keys (never commit this)
│
├── api/routes/
│   ├── health.py            ← GET  /health
│   ├── upload.py            ← POST /api/upload
│   └── query.py             ← POST /api/ask
│
├── core/
│   ├── pipeline.py          ← orchestrates all 7 layers
│   ├── retrieval.py         ← hybrid_search() ColBERT reranking
│   └── generation.py        ← generate_answer() with Groq
│
├── utils/
│   ├── pdf_parser.py        ← extract_document(), separate_types()
│   ├── chunker.py           ← summarize_chunks()
│   ├── summarizer.py        ← create_summary() Groq vision
│   └── embedder.py          ← generate_embeddings(), embed_query()
│
├── db/
│   └── qdrant_client.py     ← build_qdrant_index(), index_chunks()
│
└── logs/
    └── app.log              ← auto-generated
```

---

## Setup

### 1. Clone and install

```bash
cd rag_project
pip install -r requirements.txt
```

### 2. Fill in your .env

```bash
cp .env .env.local   # optional
```

Edit `.env`:
```
GROQ_API_KEY=your_key_from_console.groq.com
QDRANT_URL=https://your-cluster.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_key
```

Get keys free:
- Groq: https://console.groq.com (no credit card)
- Qdrant: https://cloud.qdrant.io (free 1GB cluster)

### 3. Run

```bash
python main.py
```

API is now live at `http://localhost:8000`
Swagger docs at `http://localhost:8000/docs`

---

## API Endpoints

### `GET /health`
Check system status.
```json
{"status": "ok", "qdrant": "ok", "collection": "Hybrid_search"}
```

### `POST /api/upload`
Upload a PDF. Processing runs in background.
```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@attention_paper.pdf"
```
Response:
```json
{
  "file_id": "a1b2c3d4",
  "filename": "attention_paper.pdf",
  "status": "processing",
  "message": "Poll /api/status/a1b2c3d4 to check progress."
}
```

### `GET /api/status/{file_id}`
Poll until status is `done`.
```bash
curl http://localhost:8000/api/status/a1b2c3d4
```
Response when done:
```json
{
  "file_id": "a1b2c3d4",
  "status": "done",
  "total_chunks": 18,
  "elapsed_sec": 42.3
}
```

### `POST /api/ask`
Ask a question.
```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is scaled dot-product attention?", "top_k": 5}'
```
Response:
```json
{
  "question": "What is scaled dot-product attention?",
  "answer": "Scaled dot-product attention computes... Sources used: [1]",
  "sources": [
    {
      "chunk_index": 5,
      "score": 24.38,
      "icon": "🖼️",
      "has_images": true,
      "has_tables": false,
      "preview": "The attention function maps a query..."
    }
  ],
  "total_found": 5,
  "latency_sec": 1.24
}
```

---

## Pipeline — 7 Layers

```
PDF Upload
    │
    ▼
L1  extract_document()        Unstructured hi_res → text/tables/images/formulas
    │
    ▼
L2  create_chunking_by_title() chunk_by_title → structure-aware chunks
    │
    ▼
L3  separate_types()           split each chunk → text / tables[] / images[]
    │
    ▼
L4  summarize_chunks()         build LangChain Documents with flat metadata
    │
    ▼
L5  create_summary()           Groq vision LLM sees image+table+text together
    │
    ▼
L6  generate_embeddings()      dense(384) + bm25(sparse) + colbert(128×n_tokens)
    │
    ▼
L7  build_qdrant_index()       create collection with 3 vector configs
    index_chunks()             upload in batches of 50
    │
    ▼
    Ready for queries
    │
    ▼
    hybrid_search()            dense(20) + bm25(20) → ColBERT reranks → top_k
    generate_answer()          Groq llama-3.1-8b answers with source citations
```

---

## Bugs Fixed from Original Notebook

| Bug | Impact | Fix |
|-----|--------|-----|
| `return` inside `for` loop in `hybrid_search` | Only 1 result returned always | Moved `return` outside loop |
| `create_summary` commented out | Images/tables had no descriptions | Enabled with Groq vision |
| Metadata stored as nested JSON string | Payload parsing failures | Flat metadata dict |
| Prefetch limit=10 | Only 20 candidates for reranking | Raised to 20 each (40 total) |
| Qdrant API key hardcoded | Security risk | Moved to .env |
