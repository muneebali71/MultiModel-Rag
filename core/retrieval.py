from typing import List, Optional
from qdrant_client import models
from utils.embedder import embed_query
from db.qdrant_client import get_qdrant_client
from schemas import SourceSchema
from config import QDRANT_COLLECTION
from logger import setup_logger, trace

logger = setup_logger("rag.retrieval")

# ─────────────────────────────────────────────────────────────
# Tuning — raise these for better recall at cost of latency
# ─────────────────────────────────────────────────────────────
DENSE_PREFETCH  = 20   # candidates from semantic search
BM25_PREFETCH   = 20   # candidates from keyword search
# ColBERT reranks the merged pool (up to 60) and returns top_k


# ─────────────────────────────────────────────────────────────
# LAYER 7c — Hybrid search + ColBERT reranking
# ─────────────────────────────────────────────────────────────

@trace
def hybrid_search(
    query: str,
    top_k: int = 8,
    score_threshold: Optional[float] = None,
) -> List[dict]:
    """
    3-stage retrieval:
      Stage 1 — Dense semantic search    (DENSE_PREFETCH candidates)
      Stage 2 — BM25 keyword search      (BM25_PREFETCH  candidates)
      Stage 3 — ColBERT reranks merged pool → returns top_k

    Parameters
    ----------
    query            : user question
    top_k            : how many chunks to return (default 8)
    score_threshold  : optional hard floor on raw ColBERT score.
                       Chunks below this are dropped after reranking.
                       Leave None to return all top_k.

    Returns
    -------
    List of dicts with keys:
        content     : raw chunk text
        summary     : AI-enriched description (may be "")
        score       : raw ColBERT score  ← normalisation happens in generation.py
        has_images  : bool
        has_tables  : bool
        source      : SourceSchema (score = raw, updated to normalised later)
    """
    client = get_qdrant_client()
    logger.info(f"Hybrid search: '{query}'  top_k={top_k}")

    dense_q, bm25_q, colbert_q = embed_query(query)

    prefetch = [
        models.Prefetch(
            query=dense_q.tolist(),
            using="dense",
            limit=DENSE_PREFETCH,
        ),
        models.Prefetch(
            query=models.SparseVector(**bm25_q.as_object()),
            using="bm25",
            limit=BM25_PREFETCH,
        ),
    ]

    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        prefetch=prefetch,
        query=colbert_q.tolist(),
        using="colbert",
        with_payload=True,
        limit=top_k,
    )

    formatted: List[dict] = []
    for r in results.points:
        p         = r.payload
        raw_score = round(r.score, 6)

        if score_threshold is not None and raw_score < score_threshold:
            logger.debug(
                f"  chunk_{p['chunk_index']} dropped "
                f"(score {raw_score:.4f} < threshold {score_threshold})"
            )
            continue

        icon = "🖼️" if p["has_images"] else "📊" if p["has_tables"] else "📄"

        formatted.append({
            "content":    p["content"],
            "summary":    p.get("summary", ""),   # populated by qdrant_client fix
            "score":      raw_score,
            "has_images": p["has_images"],
            "has_tables": p["has_tables"],
            "source": SourceSchema(
                chunk_index=p["chunk_index"],
                score=raw_score,               # will be overwritten with normalised value
                icon=icon,
                has_images=p["has_images"],
                has_tables=p["has_tables"],
                preview=p["content"][:150] + "...",
            ),
        })

    logger.info(
        f"Retrieved {len(formatted)} chunks | "
        f"raw scores: {[f['score'] for f in formatted]}"
    )
    return formatted