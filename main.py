"""
FastAPI application entrypoint.

Endpoints:
  POST /upload       -> upload a PDF, chunk + screen + embed + index it
  POST /query        -> ask a question (uses last-uploaded PDF for the
                         session if present, else web search / LLM knowledge)
  GET  /health        -> liveness check
"""
import os
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from langsmith import traceable

from app.config import get_settings
from app.exceptions import PDFProcessingError, register_exception_handlers
from app.logging_config import configure_logging, get_logger
from app.orchestrator import get_orchestrator
from app.rag.embeddings import get_embedding_client
from app.rag.pdf_processor import process_pdf
from app.rag.vector_store import get_vector_store
from app.schemas import QueryRequest, QueryResponse, UploadResponse
from app.security.lakera_guard import get_lakera_client
from app.security.nemo_guardrails import get_nemo_client
from app.session_manager import get_session_manager

configure_logging()
logger = get_logger(__name__)
settings = get_settings()

app = FastAPI(title="PDF + Web Search RAG Agent", version="1.0.0")
register_exception_handlers(app)

os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.chroma_persist_dir, exist_ok=True)


@app.get("/health")
async def health():
    return {"status": "ok"}


@traceable(name="ingest_pdf", run_type="chain")
async def _ingest_pdf(file_path: str) -> UploadResponse:
    document_id, chunks, num_pages = process_pdf(file_path)
    if not chunks:
        raise PDFProcessingError("PDF produced no usable text chunks.")

    lakera = get_lakera_client()
    nemo = get_nemo_client()
    safe_chunks = []
    for chunk in chunks:
        try:
            lakera.screen(chunk.text, source="pdf_chunk")
            await nemo.screen(chunk.text, source="pdf_chunk")
        except Exception as exc:
            # A flagged chunk is dropped (not indexed) rather than aborting the whole upload,
            # so one poisoned page doesn't block a legitimate document.
            logger.warning(
                "pdf_chunk_dropped_by_guardrails",
                extra={"chunk_id": chunk.chunk_id, "document_id": document_id, "reason": str(exc)},
            )
            continue
        safe_chunks.append(chunk)

    if not safe_chunks:
        raise PDFProcessingError("All extracted content was flagged as unsafe by guardrails.")

    embedder = get_embedding_client()
    embeddings = embedder.embed_texts([c.text for c in safe_chunks])

    store = get_vector_store()
    store.add_chunks(safe_chunks, embeddings)

    return UploadResponse(
        document_id=document_id,
        filename=os.path.basename(file_path),
        num_chunks=len(safe_chunks),
        num_pages=num_pages,
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(session_id: str, file: UploadFile = File(...)):
    if file.content_type != "application/pdf" and not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    dest_path = os.path.join(settings.upload_dir, f"{uuid.uuid4()}_{file.filename}")
    try:
        contents = await file.read()
        with open(dest_path, "wb") as f:
            f.write(contents)

        response = await _ingest_pdf(dest_path)
    finally:
        # We keep the extracted chunks in Chroma; the raw file itself isn't needed after ingest.
        if os.path.exists(dest_path):
            os.remove(dest_path)

    get_session_manager().set_active_document(session_id, response.document_id)
    logger.info(
        "pdf_uploaded",
        extra={"session_id": session_id, "document_id": response.document_id, "num_chunks": response.num_chunks},
    )
    return response


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    orchestrator = get_orchestrator()
    response = await orchestrator.handle_query(
        session_id=request.session_id,
        query=request.query,
        document_id=request.document_id,
    )
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
