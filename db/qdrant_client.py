from typing import List, Tuple
from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.models import PointStruct
from langchain_core.documents import Document
from config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION
from logger import setup_logger, trace
import time

logger = setup_logger("rag.qdrant")

_client: QdrantClient = None


def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        logger.info(f"Connecting to Qdrant: {QDRANT_URL}")
        _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        logger.info("✓ Qdrant connected")
    return _client


# ─────────────────────────────────────────────────────────────
# LAYER 7a — Create collection
# ─────────────────────────────────────────────────────────────

@trace
def build_qdrant_index(dense_size: int, colbert_size: int) -> None:
    """
    Create Qdrant collection with 3 vector configs:
      dense   → cosine semantic similarity
      bm25    → sparse IDF keyword matching
      colbert → late-interaction reranking (no HNSW — brute force only)
    """
    client = get_qdrant_client()

    try:
        client.delete_collection(QDRANT_COLLECTION)
        logger.info(f"Deleted old collection: {QDRANT_COLLECTION}")
    except Exception:
        pass

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config={
            "dense": models.VectorParams(
                size=dense_size,
                distance=models.Distance.COSINE,
            ),
            "colbert": models.VectorParams(
                size=colbert_size,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                ),
                hnsw_config=models.HnswConfigDiff(m=0),  # disable HNSW — rerank only
            ),
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(
                modifier=models.Modifier.IDF
            )
        },
    )
    logger.info(
        f"✓ Collection '{QDRANT_COLLECTION}' created "
        f"(dense={dense_size}, colbert={colbert_size})"
    )


# ─────────────────────────────────────────────────────────────
# LAYER 7b — Upload chunks with vectors
# ─────────────────────────────────────────────────────────────

@trace
def index_chunks(
    documents:    List[Document],
    dense_vecs:   list,
    bm25_vecs:    list,
    colbert_vecs: list,
    batch_size:   int = 2,
) -> int:
    """
    Upload all chunks with their 3 vectors into Qdrant.

    FIX: 'summary' is now stored in the payload so retrieval.py
    can pass it to generation.py for ENRICHED DESCRIPTION context.
    Without this the LLM never saw the vision-generated descriptions.
    """
    client = get_qdrant_client()
    points = []

    for i, (doc, dense, bm25, colbert) in enumerate(
        zip(documents, dense_vecs, bm25_vecs, colbert_vecs)
    ):
        meta = doc.metadata

        point = PointStruct(
            id=i,
            vector={
                "dense":   dense.tolist(),
                "bm25":    bm25.as_object(),
                "colbert": colbert.tolist(),
            },
            payload={
                "content":     doc.page_content,
                "chunk_index": meta.get("chunk_index", i),
                "has_images":  meta.get("has_images",  False),
                "has_tables":  meta.get("has_tables",  False),
                "image_count": meta.get("image_count", 0),
                "table_count": meta.get("table_count", 0),
                "raw_text":    meta.get("raw_text",    "")[:300],
                "types":       meta.get("types",       "text"),
                # ── THIS WAS MISSING — vision summaries never reached the LLM ──
                "summary":     meta.get("summary",     ""),
            },
        )
        points.append(point)

    total          = len(points)
    uploaded_total = 0

    for start in range(0, total, batch_size):
        batch = points[start : start + batch_size]

        for attempt in range(3):
            try:
                client.upsert(collection_name=QDRANT_COLLECTION, points=batch)
                uploaded_total += len(batch)
                logger.info(
                    f"Uploaded batch {start}–{start + len(batch)} "
                    f"({uploaded_total}/{total})"
                )
                break
            except Exception as e:
                wait = 2 * (attempt + 1)
                logger.warning(
                    f"Batch {start}–{start + len(batch)} failed "
                    f"(attempt {attempt + 1}/3): {e} — retrying in {wait}s"
                )
                time.sleep(wait)
        else:
            logger.error(f"Batch {start}–{start + len(batch)} permanently failed")
            raise RuntimeError("Indexing aborted — collection is incomplete!")

    if uploaded_total != total:
        raise RuntimeError(
            f"Upload mismatch: sent {uploaded_total}, expected {total}"
        )

    logger.info(f"✓ Indexing complete — {uploaded_total} chunks in '{QDRANT_COLLECTION}'")
    return uploaded_total


# ─────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────

def check_qdrant_health() -> dict:
    """Returns collection status for the /health endpoint."""
    try:
        client = get_qdrant_client()
        info   = client.get_collection(QDRANT_COLLECTION)
        return {
            "qdrant":     "ok",
            "collection": QDRANT_COLLECTION,
            "points":     info.points_count,
        }
    except Exception as e:
        return {"qdrant": f"error: {e}", "collection": QDRANT_COLLECTION}

