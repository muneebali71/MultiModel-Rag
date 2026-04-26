from typing import List, Tuple
from groq import Groq
from config import GROQ_API_KEY, GROQ_CHAT_MODEL
from schemas import SourceSchema
from logger import setup_logger, trace

logger = setup_logger("rag.generation")

_groq_client: Groq = None


def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

# def _normalize_scores(results: List[dict]) -> List[dict]:
#     """
#     Min-max normalize raw ColBERT scores → [0.0, 1.0].
#     Returns new list of dicts — never mutates the caller's data.
#     Edge case: all scores identical → all set to 1.0.
#     """
#     scores = [r["score"] for r in results]
#     min_s  = min(scores)
#     max_s  = max(scores)
#     spread = max_s - min_s

#     normalized = []
#     for r in results:
#         r_copy          = dict(r)
#         r_copy["score"] = 1.0 if spread == 0 else round((r["score"] - min_s) / spread, 4)
#         normalized.append(r_copy)

#     return normalized


def _build_context(results: List[dict]) -> str:
    """
    Build a clearly structured context string.

    Each block:
        [chunk_N] (TYPE) relevance=X.XX
        RAW TEXT:
        <up to 600 chars of chunk text>
        ENRICHED DESCRIPTION:     ← only when summary is genuinely different
        <up to 500 chars of summary>

    Separating raw text and summary prevents the LLM treating the same
    content as two independent sources.
    """
    blocks = []
    for r in results:
        chunk_idx = r["source"].chunk_index
        score     = r["score"]

        if r["has_images"]:
            type_label = "TEXT + FIGURE/IMAGE"
        elif r["has_tables"]:
            type_label = "TEXT + TABLE"
        else:
            type_label = "TEXT"

        raw_text = r["content"].strip()

        block = (
            f"[chunk_{chunk_idx}] ({type_label}) relevance={score:.2f}\n"
            f"RAW TEXT:\n{raw_text[:600]}\n"
        )

        summary = (r.get("summary") or "").strip()
        # Only include summary when it genuinely adds different information.
        # Guard: first 80 chars identical means the summary is just a truncated
        # copy of the raw text (happens when vision model wasn't called).
        if summary and summary[:80] != raw_text[:80]:
            block += f"ENRICHED DESCRIPTION:\n{summary[:500]}\n"

        blocks.append(block)

    return "\n---\n".join(blocks)


def _system_prompt() -> str:
    return """\
You are a precise document Q&A assistant.

## GROUNDING 
1. Use ONLY information from the provided context chunks. No outside knowledge.
2. If the answer is absent from all chunks, reply EXACTLY with:
   "This information is not found in the provided document."
   Do not add apologies, guesses, or partial answers.
3. Do not repeat the same ideas/facts/concepts in different sentences.
4. Do not expand or reinterpret the meaning.
5. If the chunk contains a numbered list or bullet points, 
   reproduce ALL items completely. Do not merge or skip any
6. Preserve the structure (bullets, numbering) from the source

## CITATIONS 
7. Every factual sentence MUST end with its source in brackets, e.g. [chunk_5].
8. Only cite IDs that actually appear in the context — never invent them.
9. When the same fact appears in multiple chunks, cite all: [chunk_5][chunk_7].
10. No external knowledge
11. No extra citations

## SYNTHESIS 
12. Read ALL chunks before writing. Draw on every chunk that contains relevant info.
13 Combine chunks only when they describe different facts.
14 Do NOT rephrase the same fact from different chunks.
15 If two chunks describe the same idea, mention it once only.

## CONTENT TYPES 
16. TEXT + FIGURE/IMAGE chunks: describe only what the text SAYS about the figure.
    Do not invent visual details you cannot see.
17. TEXT + TABLE chunks: extract the specific values mentioned in the chunk text
    or ENRICHED DESCRIPTION.

## LENGTH & QUALITY 
18. Aim for 3-6 sentences for simple questions, up to 10 for complex ones.
19. Never truncate mid-thought. A complete answer is better than a partial one.

## ANTI-REDUNDANCY RULE 
20. If a concept has already been stated, do not mention it again in different wording.
"""


# ─────────────────────────────────────────────────────────────
# LAYER 8 — Main entry point
# ─────────────────────────────────────────────────────────────

@trace
def generate_answer(
    question: str,
    results:  List[dict],
    *,
    max_tokens: int = 700,
) -> Tuple[str, List[SourceSchema]]:
    """
    Generate a grounded, cited, multi-chunk answer.

    Parameters
    ----------
    question   : user question string
    results    : list of dicts from hybrid_search — each must contain:
                 content, score (raw ColBERT), has_images, has_tables,
                 summary, source (SourceSchema)
    max_tokens : LLM token budget (default 700)

    Returns
    -------
    (answer_text, list_of_SourceSchema with NORMALISED scores)
    """
    if not results:
        logger.warning("generate_answer: empty results")
        return "I could not find relevant information in the document.", []

    # ── Normalise scores BEFORE building SourceSchemas ────────
    # This is the single place normalisation happens.
    # query.py passes raw results; we normalise here and update
    # the source.score so the API response shows [0,1] values.
    # results = _normalize_scores(results)
    for r in results:
        r["source"].score = r["score"]   # keep SourceSchema in sync

    # ── Build LLM context ─────────────────────────────────────
    context = _build_context(results)
    logger.info(
        f"Generating: '{question}' | "
        f"{len(results)} chunks | "
        f"scores={[r['score'] for r in results]}"
    )

    user_prompt = (
        f"CONTEXT CHUNKS:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "Write your answer now. Cite every factual claim with its [chunk_N] inline."
    )

    try:
        response = get_groq_client().chat.completions.create(
            model=GROQ_CHAT_MODEL,
            messages=[
                {"role": "system", "content": _system_prompt()},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        logger.error(f"Groq API error: {exc}")
        raise

    answer  = response.choices[0].message.content.strip()
    sources = [r["source"] for r in results]

    logger.info(f"Answer: {len(answer)} chars | {len(sources)} sources cited")
    return answer, sources
