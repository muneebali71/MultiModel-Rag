# from pydantic import BaseModel, Field
# from typing import List, Optional
# from enum import Enum


# # ─────────────────────────────────────────────────────────────
# # ENUMS
# # ─────────────────────────────────────────────────────────────

# class ChunkType(str, Enum):
#     TEXT    = "text"
#     TABLE   = "table"
#     IMAGE   = "image"
#     MIXED   = "mixed"


# class ProcessingStatus(str, Enum):
#     PENDING    = "pending"
#     PROCESSING = "processing"
#     DONE       = "done"
#     FAILED     = "failed"


# # ─────────────────────────────────────────────────────────────
# # UPLOAD schemas
# # ─────────────────────────────────────────────────────────────

# class UploadResponse(BaseModel):
#     """Returned immediately after PDF is uploaded."""
#     file_id:  str
#     filename: str
#     status:   ProcessingStatus
#     message:  str


# class ProcessingStatusResponse(BaseModel):
#     """Returned when client polls /status/{file_id}"""
#     file_id:       str
#     status:        ProcessingStatus
#     total_chunks:  Optional[int]  = None
#     elapsed_sec:   Optional[float] = None
#     error:         Optional[str]  = None


# # ─────────────────────────────────────────────────────────────
# # CHUNK schemas  (internal + export)
# # ─────────────────────────────────────────────────────────────

# class ChunkMetadata(BaseModel):
#     chunk_index: int
#     has_images:  bool  = False
#     has_tables:  bool  = False
#     image_count: int   = 0
#     table_count: int   = 0
#     types:       str   = "text"
#     raw_text:    str   = ""


# class ChunkSchema(BaseModel):
#     chunk_id:         int
#     enhanced_content: str
#     metadata:         ChunkMetadata


# # ─────────────────────────────────────────────────────────────
# # SOURCE schema  (shown to user in answer)
# # ─────────────────────────────────────────────────────────────

# class SourceSchema(BaseModel):
#     """A single retrieved source shown alongside the answer."""
#     chunk_index: int
#     score:       float
#     icon:        str          # 🖼️ 📊 📄
#     has_images:  bool
#     has_tables:  bool
#     preview:     str          # first 150 chars of content


# # ─────────────────────────────────────────────────────────────
# # QUERY schemas
# # ─────────────────────────────────────────────────────────────

# class QuestionRequest(BaseModel):
#     """POST /ask request body."""
#     question: str  = Field(..., min_length=3, max_length=1000,
#                            example="What is scaled dot-product attention?")
#     top_k:    int  = Field(default=5, ge=1, le=20,
#                            description="Number of chunks to retrieve")


# class AnswerResponse(BaseModel):
#     """POST /ask response body."""
#     question:    str
#     answer:      str
#     sources:     List[SourceSchema]
#     total_found: int
#     latency_sec: float


# # ─────────────────────────────────────────────────────────────
# # HEALTH schema
# # ─────────────────────────────────────────────────────────────

# class HealthResponse(BaseModel):
#     status:     str
#     qdrant:     str
#     collection: str
#     version:    str = "1.0.0"



















































# from pydantic import BaseModel, Field
# from typing import List, Optional
# from enum import Enum


# # ─────────────────────────────────────────────────────────────
# # ENUMS
# # ─────────────────────────────────────────────────────────────

# class ChunkType(str, Enum):
#     TEXT   = "text"
#     TABLE  = "table"
#     IMAGE  = "image"
#     MIXED  = "mixed"


# class ProcessingStatus(str, Enum):
#     PENDING    = "pending"
#     PROCESSING = "processing"
#     DONE       = "done"
#     FAILED     = "failed"


# # ─────────────────────────────────────────────────────────────
# # UPLOAD schemas
# # ─────────────────────────────────────────────────────────────

# class UploadResponse(BaseModel):
#     """Returned immediately after a PDF is accepted."""
#     file_id:  str
#     filename: str
#     status:   ProcessingStatus
#     message:  str


# class ProcessingStatusResponse(BaseModel):
#     """Returned when the client polls /status/{file_id}."""
#     file_id:      str
#     status:       ProcessingStatus
#     total_chunks: Optional[int]   = None
#     elapsed_sec:  Optional[float] = None
#     error:        Optional[str]   = None


# # ─────────────────────────────────────────────────────────────
# # CHUNK schemas
# # ─────────────────────────────────────────────────────────────

# class ChunkMetadata(BaseModel):
#     chunk_index: int
#     has_images:  bool = False
#     has_tables:  bool = False
#     image_count: int  = 0
#     table_count: int  = 0
#     types:       str  = "text"
#     raw_text:    str  = ""


# class ChunkSchema(BaseModel):
#     chunk_id:         int
#     enhanced_content: str
#     metadata:         ChunkMetadata


# # ─────────────────────────────────────────────────────────────
# # SOURCE schema
# # ─────────────────────────────────────────────────────────────

# class SourceSchema(BaseModel):
#     """
#     A single retrieved source shown alongside the answer.
#     score is ALWAYS the normalised [0, 1] value by the time
#     it reaches the API response — raw scores are internal only.
#     """
#     chunk_index: int
#     score:       float   # normalised [0, 1] — set by generation.py
#     icon:        str     # 🖼️ 📊 📄
#     has_images:  bool
#     has_tables:  bool
#     preview:     str     # first 150 chars of raw content


# # ─────────────────────────────────────────────────────────────
# # QUERY schemas
# # ─────────────────────────────────────────────────────────────

# class QuestionRequest(BaseModel):
#     """POST /ask request body."""
#     question: str = Field(
#         ...,
#         min_length=3,
#         max_length=1000,
#         example="What is scaled dot-product attention?",
#     )
#     top_k: int = Field(
#         default=8,          # raised from 5 — more context = better answers
#         ge=1,
#         le=20,
#         description="Number of chunks to retrieve (default 8)",
#     )


# class AnswerResponse(BaseModel):
#     """POST /ask response body."""
#     question:    str
#     answer:      str
#     sources:     List[SourceSchema]   # scores are normalised [0, 1]
#     total_found: int
#     latency_sec: float


# # ─────────────────────────────────────────────────────────────
# # HEALTH schema
# # ─────────────────────────────────────────────────────────────

# class HealthResponse(BaseModel):
#     status:     str
#     qdrant:     str
#     collection: str
#     version:    str = "1.1.0"



































from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class ProcessingStatus(str, Enum):
    PENDING    = "pending"
    PROCESSING = "processing"
    DONE       = "done"
    FAILED     = "failed"


class UploadResponse(BaseModel):
    file_id:  str
    filename: str
    status:   ProcessingStatus
    message:  str


class ProcessingStatusResponse(BaseModel):
    file_id:      str
    status:       ProcessingStatus
    total_chunks: Optional[int]   = None
    elapsed_sec:  Optional[float] = None
    error:        Optional[str]   = None


class SourceSchema(BaseModel):
    """
    score is ALWAYS normalised [0.5, 1.0] by the time it reaches the response.
    Raw ColBERT scores (~5–30) are internal only.
    """
    chunk_index: int
    score:       float   # normalised [0.5, 1.0]
    icon:        str     # 🖼️  📊  💻  📄
    has_images:  bool
    has_tables:  bool
    preview:     str


class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        example="How does scaled dot-product attention work?",
    )
    top_k: int = Field(
        default=8,
        ge=1,
        le=20,
        description="Chunks to retrieve (default 8)",
    )


class AnswerResponse(BaseModel):
    question:    str
    answer:      str
    sources:     List[SourceSchema]
    total_found: int
    latency_sec: float


class HealthResponse(BaseModel):
    status:     str
    qdrant:     str
    collection: str
    version:    str = "1.2.0"