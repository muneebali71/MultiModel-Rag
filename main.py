import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import health, upload, query
from logger import setup_logger
from config import APP_HOST, APP_PORT

logger = setup_logger("rag.main")

# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title="Multimodal PDF RAG API",
    description="Upload PDFs with text, tables, images and ask questions.",
    version="1.0.0",
    docs_url="/docs",      # Swagger UI at /docs
    redoc_url="/redoc"     # ReDoc UI at /redoc
)

# ── CORS (allows any frontend to call this API) ───────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(upload.router)
app.include_router(query.router)


# ── Startup / shutdown events ─────────────────────────────────
@app.on_event("startup")
async def startup():
    logger.info("=" * 50)
    logger.info("Multimodal PDF RAG API started")
    logger.info(f"Docs: http://{APP_HOST}:{APP_PORT}/docs")
    logger.info("=" * 50)


@app.on_event("shutdown")
async def shutdown():
    logger.info("API shutting down")


# ── Run directly ──────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=True,        # auto-reload on code changes during dev
        log_level="info"
    )
