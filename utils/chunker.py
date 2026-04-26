from typing import List
from langchain_core.documents import Document
from utils.pdf_parser import separate_types
from utils.summarizer import create_summary
from logger import setup_logger, trace

logger = setup_logger("rag.chunker")


# ─────────────────────────────────────────────────────────────
# LAYER 4 — Build enriched LangChain Documents
# ─────────────────────────────────────────────────────────────

@trace
def summarize_chunks(chunks: list) -> List[Document]:
    """
    For each chunk:
      - Separate text / tables / images
      - If tables or images exist → call Groq vision for rich description
      - Else → use raw text
      - Build LangChain Document with FLAT metadata (not nested JSON)
    Fixes all bugs from original notebook.
    """
    total     = len(chunks)
    documents = []

    for i, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {i+1}/{total}")

        data = separate_types(chunk)
        logger.info(
            f"  types={data['types']} "
            f"tables={len(data['tables'])} "
            f"images={len(data['images'])}"
        )

        if data["tables"] or data["images"]:
            logger.info(f"  → Creating AI summary (vision+text)")
            try:
                enhanced_content = create_summary(
                    data["text"],
                    data["tables"],
                    data["images"]
                )
                logger.info(f"   Summary preview: {enhanced_content[:100]}...")
            except Exception as e:
                logger.error(f"   Summary failed: {e} — using raw text")
                enhanced_content = data["text"]
        else:
            logger.info(f"  → Using raw text (no tables/images)")
            enhanced_content = data["text"]

        # ── FLAT metadata — NOT nested JSON string ────────────
        # This fixes the original bug where metadata was
        # double-encoded as json.dumps inside json.dumps
        doc = Document(
            # page_content=enhanced_content,
            page_content =data["text"],
            metadata={
                "chunk_index": i,
                "has_images":  len(data["images"]) > 0,
                "has_tables":  len(data["tables"]) > 0,
                "image_count": len(data["images"]),
                "table_count": len(data["tables"]),
                "types":       str(data["types"]),
                "raw_text":    data["text"][:500],   # preview only

                "summary": enhanced_content
            }
        )
        documents.append(doc)

    logger.info(f"✓ Built {len(documents)} LangChain Documents")
    return documents


# ─────────────────────────────────────────────────────────────
# Export helper (optional — for debugging)
# ─────────────────────────────────────────────────────────────

def export_chunks_to_json(chunks: List[Document], filename: str = "chunks_export.json") -> list:
    """Export processed chunks to JSON for inspection."""
    import json

    export_data = []
    for i, doc in enumerate(chunks):
        export_data.append({
            "chunk_id":         i + 1,
            "enhanced_content": doc.page_content,
            "metadata":         doc.metadata
        })

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Exported {len(export_data)} chunks to {filename}")
    return export_data
