import os
import time
from typing import Dict
from langchain_core.documents import Document

from utils.pdf_parser import extract_document, create_chunking_by_title
from utils.chunker import summarize_chunks
from utils.embedder import generate_embeddings
from utils.cleaning import clean_text
from db.qdrant_client import build_qdrant_index, index_chunks
from logger import setup_logger, trace
from config import UPLOAD_DIR


logger = setup_logger("rag.pipeline")

# In-memory status tracking per file_id
# In production replace with Redis or a DB table
_pipeline_status: Dict[str, dict] = {}


def get_status(file_id: str) -> dict:
    return _pipeline_status.get(file_id, {"status": "not_found"})


# ─────────────────────────────────────────────────────────────
# FULL PIPELINE — Layers 1 → 7
# ─────────────────────────────────────────────────────────────

def run_pipeline(pdf_path: str, file_id: str) -> None:
    """
    Orchestrates all 7 layers:
      L1 extract_document
      L2 create_chunking_by_title
      L3 separate_types (inside summarize_chunks)
      L4 summarize_chunks (build Documents)
      L5 create_summary (inside summarize_chunks — Groq vision)
      L6 generate_embeddings (dense + BM25 + ColBERT)
      L7 build_qdrant_index + index_chunks

    Called as a background task from the upload endpoint.
    Updates _pipeline_status so /status can be polled.
    """
    start = time.perf_counter()

    _pipeline_status[file_id] = {
        "status":       "processing",
        "total_chunks": None,
        "elapsed_sec":  None,
        "error":        None
    }

    try:
        # ── L1: Parse PDF ─────────────────────────────────────
        logger.info(f"[{file_id}] L1: Extracting elements from {pdf_path}")
        elements = extract_document(pdf_path)

        # ── L2: Chunk by title ────────────────────────────────
        logger.info(f"[{file_id}] L2: Chunking {len(elements)} elements")
        chunks = create_chunking_by_title(elements)

        # cleaing chunks
        # for chunk in chunks:
        #     clean_text(chunk)
        for chunk in chunks:
            if hasattr(chunk, "text"):
                chunk.text = clean_text(chunk.text)

        # ── L3+L4+L5: Summarize chunks (vision enrichment) ───
        logger.info(f"[{file_id}] L3-L5: Summarizing {len(chunks)} chunks")
        documents = summarize_chunks(chunks)

        # ── L6: Generate embeddings ───────────────────────────
        logger.info(f"[{file_id}] L6: Generating embeddings")
        dense_vecs, bm25_vecs, colbert_vecs, dense_size, colbert_size = \
            generate_embeddings(documents)

        # ── L7: Build Qdrant index + upload ───────────────────
        logger.info(f"[{file_id}] L7: Building Qdrant index")
        build_qdrant_index(dense_size, colbert_size)
        total = index_chunks(documents, dense_vecs, bm25_vecs, colbert_vecs)

        elapsed = round(time.perf_counter() - start, 2)
        logger.info(f"[{file_id}] ✓ Pipeline done in {elapsed}s — {total} chunks indexed")

        _pipeline_status[file_id] = {
            "status":       "done",
            "total_chunks": total,
            "elapsed_sec":  elapsed,
            "error":        None
        }

        # Clean up uploaded file
        try:
            os.remove(pdf_path)
        except Exception:
            pass

    except Exception as e:
        elapsed = round(time.perf_counter() - start, 2)
        logger.error(f"[{file_id}] ✗ Pipeline failed after {elapsed}s: {e}")
        _pipeline_status[file_id] = {
            "status":       "failed",
            "total_chunks": None,
            "elapsed_sec":  elapsed,
            "error":        str(e)
        }
        raise
