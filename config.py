import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ───────────────────────────────────────────────────────
GROQ_API_KEY         = os.getenv("GROQ_API_KEY")
GROQ_VISION_MODEL    = os.getenv("GROQ_VISION_MODEL",  "meta-llama/llama-4-scout-17b-16e-instruct")
GROQ_CHAT_MODEL      = os.getenv("GROQ_CHAT_MODEL",    "llama-3.1-8b-instant")

# ── Qdrant ────────────────────────────────────────────────────
QDRANT_URL           = os.getenv("QDRANT_URL")
QDRANT_API_KEY       = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION    = os.getenv("QDRANT_COLLECTION",  "Hybrid_search")

# ── Embedding models ──────────────────────────────────────────
DENSE_MODEL          = os.getenv("DENSE_MODEL",   "sentence-transformers/all-MiniLM-L6-v2")
BM25_MODEL           = os.getenv("BM25_MODEL",    "Qdrant/bm25")
COLBERT_MODEL        = os.getenv("COLBERT_MODEL", "colbert-ir/colbertv2.0")

# ── Chunking ──────────────────────────────────────────────────
CHUNK_MAX_CHARS      = int(os.getenv("CHUNK_MAX_CHARACTERS",       3000))
CHUNK_NEW_AFTER      = int(os.getenv("CHUNK_NEW_AFTER_N_CHARS",    2400))
CHUNK_COMBINE_UNDER  = int(os.getenv("CHUNK_COMBINE_UNDER_N_CHARS", 500))

# ── App ───────────────────────────────────────────────────────
APP_HOST             = os.getenv("APP_HOST",    "0.0.0.0")
APP_PORT             = int(os.getenv("APP_PORT", 8000))
LOG_LEVEL            = os.getenv("LOG_LEVEL",   "INFO")
UPLOAD_DIR           = os.getenv("UPLOAD_DIR",  "/tmp/rag_uploads")

#_______LLama_parser_____________________________________________
llama_api_key        =os.getenv("llama_key")
