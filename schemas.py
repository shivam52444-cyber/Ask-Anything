"""
All request/response models. Keeping every API contract as a Pydantic model
gives us validation + structured, predictable output for free.
"""
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AnswerSource(str, Enum):
    PDF = "pdf"
    WEB_SEARCH = "web_search"
    LLM_KNOWLEDGE = "llm_internal_knowledge"
    ASK_HUMAN_PENDING = "ask_human_pending"
    NONE = "none"


class QueryRequest(BaseModel):
    session_id: str = Field(..., description="Client-generated session identifier.")
    query: str = Field(..., min_length=1, max_length=4000)
    document_id: str | None = Field(
        default=None, description="If set, restrict PDF lookup to this previously uploaded document."
    )


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: float
    page: int | None = None


class QueryResponse(BaseModel):
    success: bool = True
    session_id: str
    answer: str
    source: AnswerSource
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    requires_human_input: bool = False
    clarification_prompt: str | None = None
    trace_id: str | None = None


class UploadResponse(BaseModel):
    success: bool = True
    document_id: str
    filename: str
    num_chunks: int
    num_pages: int


class ErrorResponse(BaseModel):
    success: bool = False
    error_code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class EvalMetrics(BaseModel):
    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    k: int
