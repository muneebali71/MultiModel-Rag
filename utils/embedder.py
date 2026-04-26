from typing import List, Tuple
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
from langchain_core.documents import Document
from config import DENSE_MODEL, BM25_MODEL, COLBERT_MODEL
from logger import setup_logger, trace

logger = setup_logger("rag.embedder")

# ── Singleton model instances (loaded once, reused) ───────────
_dense_model:   TextEmbedding                 = None
_bm25_model:    SparseTextEmbedding           = None
_colbert_model: LateInteractionTextEmbedding  = None


def get_dense_model() -> TextEmbedding:
    global _dense_model
    if _dense_model is None:
        logger.info(f"Loading dense model: {DENSE_MODEL}")
        _dense_model = TextEmbedding(DENSE_MODEL)
    return _dense_model


def get_bm25_model() -> SparseTextEmbedding:
    global _bm25_model
    if _bm25_model is None:
        logger.info(f"Loading BM25 model: {BM25_MODEL}")
        _bm25_model = SparseTextEmbedding(BM25_MODEL)
    return _bm25_model


def get_colbert_model() -> LateInteractionTextEmbedding:
    global _colbert_model
    if _colbert_model is None:
        logger.info(f"Loading ColBERT model: {COLBERT_MODEL}")
        _colbert_model = LateInteractionTextEmbedding(COLBERT_MODEL)
    return _colbert_model


# ─────────────────────────────────────────────────────────────
# LAYER 6 — Generate all 3 embeddings in batch
# ─────────────────────────────────────────────────────────────

@trace
def generate_embeddings(
    documents: List[Document]
) -> Tuple[list, list, list, int, int]:
    """
    Generate dense + BM25 + ColBERT embeddings for all documents.
    Returns: (dense_vecs, bm25_vecs, colbert_vecs, dense_size, colbert_size)
    """
    texts = [doc.page_content for doc in documents]
    logger.info(f"Embedding {len(texts)} chunks with 3 models...")

    logger.info("  [1/3] Dense (semantic)...")
    dense_vecs   = list(get_dense_model().embed(texts))

    logger.info("  [2/3] BM25 (keyword)...")
    bm25_vecs    = list(get_bm25_model().embed(texts))

    logger.info("  [3/3] ColBERT (reranker)...")
    colbert_vecs = list(get_colbert_model().embed(texts))

    dense_size   = len(dense_vecs[0])
    colbert_size = len(colbert_vecs[0][0])

    logger.info(f"✓ Embeddings done — dense={dense_size}, colbert={colbert_size}")
    return dense_vecs, bm25_vecs, colbert_vecs, dense_size, colbert_size


# ─────────────────────────────────────────────────────────────
# Query embedding (used at retrieval time)
# ─────────────────────────────────────────────────────────────

def embed_query(query: str) -> Tuple:
    """Embed a single query with all 3 models for hybrid search."""
    dense_q   = next(get_dense_model().query_embed(query))
    bm25_q    = next(get_bm25_model().query_embed(query))
    colbert_q = next(get_colbert_model().query_embed(query))
    return dense_q, bm25_q, colbert_q
