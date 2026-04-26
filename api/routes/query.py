import time
from fastapi import APIRouter, HTTPException
from schemas import QuestionRequest, AnswerResponse
from core.retrieval import hybrid_search
from core.generation import generate_answer
from logger import setup_logger

router = APIRouter(prefix="/api", tags=["query"])
logger = setup_logger("rag.query")

# ── Tuning ────────────────────────────────────────────────────
DEFAULT_TOP_K  = 8      # raised from 5
SCORE_FLOOR    = 0.0    # raw ColBERT floor; set to ~5.0 to gate very weak matches


@router.post("/ask", response_model=AnswerResponse)
async def ask(req: QuestionRequest):
    """
    Ask a question about the uploaded document.

    Pipeline:
      1. hybrid_search   → dense + BM25 + ColBERT rerank → raw chunks
      2. generate_answer → normalise scores [0.5,1.0], build context, call LLM
      3. return          → AnswerResponse with normalised scores
    """
    start = time.perf_counter()
    top_k = req.top_k or DEFAULT_TOP_K
    logger.info(f"[ask] '{req.question}'  top_k={top_k}")

    results = hybrid_search(
        query=req.question,
        top_k=top_k,
        score_threshold=SCORE_FLOOR if SCORE_FLOOR > 0.0 else None,
    )

    if not results:
        raise HTTPException(
            status_code=404,
            detail="No relevant content found. Upload and process a document first.",
        )

    answer, sources = generate_answer(question=req.question, results=results)

    elapsed = round(time.perf_counter() - start, 3)
    logger.info(f"[ask] done {elapsed}s | sources={len(sources)}")

    return AnswerResponse(
        question=req.question,
        answer=answer,
        sources=sources,
        total_found=len(sources),
        latency_sec=elapsed,
    )