"""
Custom exception hierarchy + FastAPI exception handlers.
Every exception carries a machine-readable `code` so the client can branch
on it without parsing free-text messages.
"""
from fastapi import Request, status
from fastapi.responses import JSONResponse

from app.logging_config import get_logger

logger = get_logger(__name__)


class AppError(Exception):
    """Base class for all expected/handled application errors."""

    code: str = "app_error"
    http_status: int = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class PromptInjectionDetected(AppError):
    code = "prompt_injection_detected"
    http_status = status.HTTP_400_BAD_REQUEST


class PDFProcessingError(AppError):
    code = "pdf_processing_error"
    http_status = status.HTTP_422_UNPROCESSABLE_ENTITY


class EmbeddingServiceError(AppError):
    code = "embedding_service_error"
    http_status = status.HTTP_502_BAD_GATEWAY


class VectorStoreError(AppError):
    code = "vector_store_error"
    http_status = status.HTTP_500_INTERNAL_SERVER_ERROR


class LLMServiceError(AppError):
    code = "llm_service_error"
    http_status = status.HTTP_502_BAD_GATEWAY


class MCPToolError(AppError):
    code = "mcp_tool_error"
    http_status = status.HTTP_502_BAD_GATEWAY


class SessionNotFoundError(AppError):
    code = "session_not_found"
    http_status = status.HTTP_404_NOT_FOUND


def register_exception_handlers(app) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError):
        logger.error(
            "handled_app_error",
            extra={"code": exc.code, "message": exc.message, "path": str(request.url), "details": exc.details},
        )
        return JSONResponse(
            status_code=exc.http_status,
            content={"success": False, "error_code": exc.code, "message": exc.message, "details": exc.details},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception):
        logger.exception("unhandled_exception", extra={"path": str(request.url)})
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error_code": "internal_server_error",
                "message": "An unexpected error occurred.",
                "details": {},
            },
        )
