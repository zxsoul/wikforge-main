"""Logging configuration for the application.

使用 structlog 输出结构化日志：
- ``settings.DEBUG=True`` → 彩色 console 渲染，便于本地调试
- ``settings.DEBUG=False`` → JSON 渲染，便于生产环境采集
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from app.core.config import get_settings


def configure_logging() -> None:
    """Configure application-wide structured logging.

    1. 配置 stdlib logging 把根 logger 输出到 stdout
    2. 配置 structlog 与 stdlib logging 桥接，根据 ``settings.DEBUG``
       切换控制台渲染或 JSON 渲染
    3. 降低第三方库噪声（uvicorn.access、httpx、httpcore）
    """
    settings = get_settings()
    level = logging.DEBUG if settings.DEBUG else logging.INFO

    # ---- stdlib logging：仅作为最终输出 ----
    root = logging.getLogger()
    # 清理已有 handler，避免重复初始化导致多份输出
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)
    root.setLevel(level)

    # 降低三方库噪声
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # ---- structlog ----
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.DEBUG:
        renderer: Any = structlog.dev.ConsoleRenderer(colors=True)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


# 兼容旧调用（main.py 历史调用 setup_logging）
def setup_logging() -> None:
    """Backward compatible alias for :func:`configure_logging`."""
    configure_logging()


def get_logger(name: str | None = None) -> Any:
    """Get a structlog logger instance.

    返回 structlog 的 BoundLogger，调用方式与 stdlib logger 兼容
    （info/warning/error/debug 等方法）。
    """
    return structlog.get_logger(name)
