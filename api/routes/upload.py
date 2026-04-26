import os
import uuid
import shutil
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from schemas import UploadResponse, ProcessingStatus, ProcessingStatusResponse
from core.pipeline import run_pipeline, get_status
from config import UPLOAD_DIR
from logger import setup_logger

router = APIRouter(prefix="/api", tags=["upload"])
logger = setup_logger("rag.upload")

os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload a PDF file.
    Processing runs in the background — poll /api/status/{file_id} to check progress.
    """
    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Save to disk
    file_id   = uuid.uuid4().hex[:8]
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(f"Uploaded: {file.filename} → {file_path} (id={file_id})")

    # Run pipeline in background so response returns immediately
    background_tasks.add_task(run_pipeline, file_path, file_id)

    return UploadResponse(
        file_id=file_id,
        filename=file.filename,
        status=ProcessingStatus.PROCESSING,
        message=f"PDF received. Processing started. Poll /api/status/{file_id} to check progress."
    )


@router.get("/status/{file_id}", response_model=ProcessingStatusResponse)
async def get_processing_status(file_id: str):
    """Poll this endpoint to check if your PDF has finished processing."""
    info = get_status(file_id)

    if info.get("status") == "not_found":
        raise HTTPException(status_code=404, detail=f"file_id '{file_id}' not found.")

    return ProcessingStatusResponse(
        file_id=file_id,
        status=ProcessingStatus(info["status"]),
        total_chunks=info.get("total_chunks"),
        elapsed_sec=info.get("elapsed_sec"),
        error=info.get("error")
    )
