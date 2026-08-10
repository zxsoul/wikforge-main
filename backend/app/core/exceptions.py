"""Custom exception classes and exception handlers.

错误响应统一为如下结构（envelope）：

```json
{
  "error": {
    "code": "NotFound",
    "message": "Document not found: 123",
    "details": {"resource": "Document", "id": "123"}
  }
}
```

包含三类全局处理器：

1. :class:`AppException`        → 业务异常，按声明的状态码返回标准化结构
2. :class:`RequestValidationError` → 422 + 标准化错误结构（包含字段级错误）
3. ``Exception`` 兜底           → 500，隐藏内部细节，统一写入日志
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import get_logger

logger = get_logger(__name__)


class AppException(Exception):
    """Base application exception.

    Attributes:
        message: 面向用户的可读信息
        status_code: HTTP 状态码
        detail: 结构化补充信息（字段级错误、资源 ID 等）
        code: 机器可读的错误代码，默认为子类名
    """

    code: str = "AppError"

    def __init__(
        self,
        message: str,
        status_code: int = 400,
        detail: dict[str, Any] | None = None,
        *,
        code: str | None = None,
    ):
        self.message = message
        self.status_code = status_code
        self.detail = detail or {}
        if code:
            self.code = code
        super().__init__(message)


class NotFoundException(AppException):
    """Resource not found."""

    code = "NotFound"

    def __init__(self, resource: str, resource_id: str):
        super().__init__(
            message=f"{resource} not found: {resource_id}",
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"resource": resource, "id": resource_id},
        )


class ForbiddenException(AppException):
    """Access denied."""

    code = "Forbidden"

    def __init__(self, message: str = "Access denied"):
        super().__init__(message=message, status_code=status.HTTP_403_FORBIDDEN)


class UnauthorizedException(AppException):
    """Authentication required."""

    code = "Unauthorized"

    def __init__(self, message: str = "Authentication required"):
        super().__init__(message=message, status_code=status.HTTP_401_UNAUTHORIZED)


class ValidationException(AppException):
    """Validation error."""

    code = "ValidationError"

    def __init__(self, message: str, errors: list[dict[str, Any]] | None = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"errors": errors or []},
        )


class ConflictException(AppException):
    """Resource conflict."""

    code = "Conflict"

    def __init__(self, message: str):
        super().__init__(message=message, status_code=status.HTTP_409_CONFLICT)


def _error_envelope(
    code: str, message: str, details: dict[str, Any] | list[Any] | None = None
) -> dict[str, Any]:
    """构造标准化错误响应体。"""
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
        }
    }


def register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers on the FastAPI app."""

    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
        # 业务异常：按声明状态码返回，info 级别日志记录方便排查
        logger.info(
            "app_exception",
            code=exc.code,
            message=exc.message,
            status_code=exc.status_code,
            path=request.url.path,
            method=request.method,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_envelope(exc.code, exc.message, exc.detail),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # FastAPI/Pydantic 请求体/查询参数校验失败：返回 422 + 字段级错误
        errors: list[dict[str, Any]] = []
        for err in exc.errors():
            errors.append(
                {
                    "loc": list(err.get("loc", [])),
                    "msg": err.get("msg", ""),
                    "type": err.get("type", ""),
                }
            )
        logger.info(
            "request_validation_error",
            path=request.url.path,
            method=request.method,
            errors=errors,
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_error_envelope(
                "ValidationError",
                "Request validation failed",
                {"errors": errors},
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        # FastAPI 内置 HTTPException（如 404、405 等）：保留状态码，统一信封
        logger.info(
            "http_exception",
            status_code=exc.status_code,
            detail=exc.detail,
            path=request.url.path,
            method=request.method,
        )
        message = exc.detail if isinstance(exc.detail, str) else "HTTP error"
        details: dict[str, Any] = {}
        if not isinstance(exc.detail, str) and exc.detail is not None:
            details = {"detail": exc.detail}
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_envelope(f"HTTP{exc.status_code}", message, details),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        # 未捕获异常：隐藏内部细节，仅返回通用错误，详细堆栈写入日志
        logger.exception(
            "unhandled_exception",
            path=request.url.path,
            method=request.method,
            error_type=type(exc).__name__,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error_envelope(
                "InternalServerError",
                "An internal server error occurred",
                {},
            ),
        )
