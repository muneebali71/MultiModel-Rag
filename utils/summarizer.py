from typing import List
from groq import Groq
from config import GROQ_API_KEY, GROQ_VISION_MODEL
from logger import setup_logger, trace

logger = setup_logger("rag.summarizer")

# Single Groq client instance
_groq_client: Groq = None

def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


# ─────────────────────────────────────────────────────────────
# LAYER 5 — Multimodal enrichment
# ─────────────────────────────────────────────────────────────

@trace
def create_summary(text: str, tables: List[str], images: List[str]) -> str:
    """
    Type-aware prompting:
    - Images  → explicit figure analysis prompt
    - Tables  → explicit data extraction prompt
    - Text    → concept + question generation prompt
    All combined into one searchable description.
    """
    client = get_groq_client()
    try:
        message_content = []

        # ── Detect what this chunk contains ───────────────────
        has_images = len(images) > 0
        has_tables = len(tables) > 0
        has_text   = bool(text.strip())

        # ── Build type-aware prompt ────────────────────────────
        prompt = "You are a RAG document indexing assistant.\n"
        prompt += "Your job: create a rich, searchable description so users can find this content.\n\n"

        # TEXT section
        if has_text:
            prompt += f"## TEXT IN THIS CHUNK\n{text}\n\n"

        # TABLE section — specific extraction instructions
        if has_tables:
            prompt += "## TABLES IN THIS CHUNK\n"
            for i, table in enumerate(tables):
                prompt += f"Table {i+1}:\n{table[:1000]}\n\n"

        # IMAGE instructions — specific figure analysis
        if has_images:
            prompt += f"## IMAGES IN THIS CHUNK\n"
            prompt += f"There are {len(images)} image(s) attached below.\n"
            prompt += "For EACH image, identify if it has a figure number (e.g. Figure 1, Fig. 2).\n\n"

        # ── Task section changes based on content type ─────────
        prompt += "## YOUR TASK\n"
        prompt += "Write a searchable description that includes ALL of the following:\n\n"

        prompt += "### 1. EXPLICIT REFERENCES (most important for search)\n"
        prompt += "- State the exact figure numbers visible (e.g. 'Figure 1 shows...', 'Figure 2 depicts...')\n"
        prompt += "- State the exact table numbers (e.g. 'Table 1 contains...', 'Table 2 shows...')\n"
        prompt += "- State exact section names mentioned in the text\n\n"

        if has_images:
            prompt += "### 2. VISUAL CONTENT ANALYSIS (for each image/figure)\n"
            prompt += "- What type of diagram/chart/figure is it? (architecture, flow, bar chart, etc.)\n"
            prompt += "- What are the main components, labels, arrows, boxes shown?\n"
            prompt += "- What concept or process does it illustrate?\n"
            prompt += "- What would a user search for to find this figure?\n\n"

        if has_tables:
            prompt += "### 3. TABLE DATA EXTRACTION\n"
            prompt += "- What are the column headers?\n"
            prompt += "- What are the key rows and their values?\n"
            prompt += "- What metrics, scores, or comparisons does it show?\n"
            prompt += "- What questions can this table answer?\n\n"

        if has_text:
            prompt += "### 4. KEY CONCEPTS FROM TEXT\n"
            prompt += "- Main topics and technical concepts\n"
            prompt += "- Key facts, numbers, formulas mentioned\n"
            prompt += "- What questions does this text answer?\n\n"

        prompt += "### 5. SEARCHABLE Q&A PAIRS\n"
        prompt += "Write 3-5 questions a user might ask that this chunk can answer.\n"
        prompt += "Format: Q: [question] A: [short answer]\n\n"

        prompt += "Output ONLY the description. Be specific. Use exact names, numbers, and labels."

        # ── Build messages ─────────────────────────────────────
        message_content = [{"type": "text", "text": prompt}]

        for img_b64 in images:
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
            })

        logger.info(
            f"Calling Groq vision: {len(images)} image(s), "
            f"{len(tables)} table(s), text_len={len(text)}"
        )

        response = client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[{"role": "user", "content": message_content}],
            temperature=0.1,
            max_tokens=800
        )

        summary = response.choices[0].message.content
        logger.info(f"Summary created ({len(summary)} chars)")
        return summary

    except Exception as e:
        logger.error(f"create_summary failed: {e}")
        # Graceful fallback — still returns something useful
        fallback = text[:300]
        if tables: fallback += f" [Contains {len(tables)} table(s)]"
        if images: fallback += f" [Contains {len(images)} image(s) — visual content]"
        return fallback

