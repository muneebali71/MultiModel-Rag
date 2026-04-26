import logging
import time
import functools
import os
from config import LOG_LEVEL

# ── Create logs dir if missing ────────────────────────────────
os.makedirs("logs", exist_ok=True)

# ── Configure root logger ─────────────────────────────────────
def setup_logger(name: str = "rag") -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # already set up

    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler
    file_handler = logging.FileHandler("logs/app.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


# ── Tracing decorator ─────────────────────────────────────────
# Wrap any function with @trace to auto-log start, end, time
def trace(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(f"rag.trace.{func.__name__}")
        start  = time.perf_counter()
        logger.info(f"START {func.__name__}")
        try:
            result  = func(*args, **kwargs)
            elapsed = round(time.perf_counter() - start, 3)
            logger.info(f"END   {func.__name__} ── {elapsed}s")
            return result
        except Exception as e:
            elapsed = round(time.perf_counter() - start, 3)
            logger.error(f"FAIL  {func.__name__} ── {elapsed}s ── {e}")
            raise
    return wrapper


# ── Async tracing decorator ───────────────────────────────────
def async_trace(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        logger = logging.getLogger(f"rag.trace.{func.__name__}")
        start  = time.perf_counter()
        logger.info(f"START {func.__name__}")
        try:
            result  = await func(*args, **kwargs)
            elapsed = round(time.perf_counter() - start, 3)
            logger.info(f"END   {func.__name__} ── {elapsed}s")
            return result
        except Exception as e:
            elapsed = round(time.perf_counter() - start, 3)
            logger.error(f"FAIL  {func.__name__} ── {elapsed}s ── {e}")
            raise
    return wrapper
