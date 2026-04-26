import os
from typing import List, Dict, Any
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from logger import setup_logger, trace
from config import (
    CHUNK_MAX_CHARS,
    CHUNK_NEW_AFTER,
    CHUNK_COMBINE_UNDER,
    UPLOAD_DIR
)

logger = setup_logger("rag.pdf_parser")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("./images", exist_ok=True)


# ─────────────────────────────────────────────────────────────
# LAYER 1 — Extract raw elements from PDF
# ─────────────────────────────────────────────────────────────

@trace
def extract_document(pdf_path: str) -> list:
    """
    Extract all elements from PDF using Unstructured hi_res strategy.
    Returns raw unstructured elements (text, tables, images, formulas).
    Exactly your notebook Cell 6.
    """
    logger.info(f"Extracting PDF: {pdf_path}")

    elements = partition_pdf(
        filename=pdf_path,
        infer_table_structure=True,
        strategy="hi_res",
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,       # store image as base64
        extract_image_block_output_dir="./images/"
    )

    # Log element type summary
    from collections import Counter
    counts = Counter(type(el).__name__ for el in elements)
    logger.info(f"Extracted {len(elements)} elements: {dict(counts)}")

    return elements


# ─────────────────────────────────────────────────────────────
# LAYER 2 — Chunk by title (structure-aware)
# ─────────────────────────────────────────────────────────────

@trace
def create_chunking_by_title(elements: list) -> list:
    """
    Group elements into chunks by document title/heading boundaries.
    Exactly your notebook Cell 14.
    """
    chunks = chunk_by_title(
        elements,
        max_characters=CHUNK_MAX_CHARS,
        new_after_n_chars=CHUNK_NEW_AFTER,
        combine_text_under_n_chars=CHUNK_COMBINE_UNDER
    )
    logger.info(f"Created {len(chunks)} chunks from {len(elements)} elements")
    return chunks


# ─────────────────────────────────────────────────────────────
# LAYER 3 — Separate element types within each chunk
# ─────────────────────────────────────────────────────────────

def separate_types(chunk) -> Dict[str, Any]:
    """
    Analyze a chunk and separate its content by type:
    text, tables (HTML), images (base64).
    Exactly your notebook Cell 21.
    """
    data = {
        "text":   chunk.text,
        "tables": [],
        "images": [],
        "types":  ["text"]
    }

    if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__

            if element_type == "Table":
                data["types"].append("table")
                table_html = getattr(element.metadata, "text_as_html", element.text)
                data["tables"].append(table_html)

            elif element_type == "Image":
                if hasattr(element, "metadata") and hasattr(element.metadata, "image_base64"):
                    if element.metadata.image_base64:
                        data["types"].append("image")
                        data["images"].append(element.metadata.image_base64)

    data["types"] = list(set(data["types"]))
    return data
