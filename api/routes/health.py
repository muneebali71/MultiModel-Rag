from fastapi import APIRouter
from schemas import HealthResponse
from db.qdrant_client import check_qdrant_health
from logger import setup_logger

router = APIRouter(tags=["health"])
logger = setup_logger("rag.health")


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Returns system status — Qdrant connection + collection info."""
    logger.info("Health check requested")
    info = check_qdrant_health()
    return HealthResponse(
        status="ok",
        qdrant=info["qdrant"],
        collection=info["collection"]
    )
