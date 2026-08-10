"""搜索 API 路由。

实现：
- POST /api/search: 复合搜索（多路召回 + RRF 融合 + Cross-Encoder 精排）

设计要点：
- 请求体由 :class:`SearchRequest` 使用 Pydantic 校验，
  ``page`` ≥ 1，``page_size`` 取 1-50，``query`` 长度 1-500
- 鉴权依赖 :func:`app.api.auth.get_current_user`，未登录返回 401
- 用户的可访问空间通过 :class:`Permission` 表查询
  （仅取 ``read`` / ``write`` 级别的 space-level 权限）
- 整体执行使用 ``asyncio.wait_for`` 控制在 5 秒内完成；超时返回 504
  并以统一错误信封返回错误信息（参见
  :mod:`app.core.exceptions`）
- 单路召回的 3 秒超时由 :class:`SearchService` 内部控制
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.core.database import get_db
from app.models.permission import AccessLevel, Permission, ResourceType
from app.models.user import User
from app.services.query_enhancer import QueryEnhancer, build_query_enhancer
from app.services.search_service import SearchService

logger = logging.getLogger(__name__)

# 整体搜索超时时间（秒）。需求 6.7：单次搜索 5 秒内返回。
SEARCH_TOTAL_TIMEOUT = 5.0

router = APIRouter(prefix="/api", tags=["search"])


# ─── Request / Response Schemas ────────────────────────────────────────


class SearchRequest(BaseModel):
    """搜索请求体。

    - ``query``: 用户查询文本，1-500 字符
    - ``page``: 页码，从 1 开始
    - ``page_size``: 每页结果数，默认 10，上限 50（需求 6.7）
    """

    query: str = Field(..., min_length=1, max_length=500, description="搜索查询文本")
    page: int = Field(default=1, ge=1, description="页码（从 1 开始）")
    page_size: int = Field(
        default=10,
        ge=1,
        le=50,
        description="每页结果数（默认 10，最大 50）",
    )


class SearchResultItem(BaseModel):
    """单条搜索结果。

    字段对应 :class:`app.services.search_service.SearchResult`，
    其中 ``score`` 严格夹紧到 [0.0, 1.0]，``highlight`` 长度 ≤ 200。
    """

    chunk_id: str
    document_id: str
    chunk_index: int
    title_chain: str
    source_file: str
    page_number: int = 0
    score: float = Field(..., ge=0.0, le=1.0, description="相关性分数 0-1")
    highlight: str = Field(
        ..., max_length=200, description="高亮匹配片段（最多 200 字符）"
    )


class SearchResponseSchema(BaseModel):
    """搜索响应体。"""

    results: list[SearchResultItem]
    total: int = Field(..., ge=0, description="融合后候选总数")
    page: int = Field(..., ge=1, description="当前页码")
    page_size: int = Field(..., ge=1, le=50, description="每页结果数")


# ─── Dependencies ──────────────────────────────────────────────────────


async def get_search_service() -> SearchService:
    """构造默认的 :class:`SearchService` 实例。

    单独抽出依赖工厂便于测试通过 ``dependency_overrides`` 注入 mock。
    """
    return SearchService()


async def get_query_enhancer() -> QueryEnhancer:
    """构造按 Settings 配置的 :class:`QueryEnhancer` 实例（任务 15.6 / 需求 7.6）。

    通过 :func:`app.services.query_enhancer.build_query_enhancer` 读取以下
    环境变量驱动的开关：

    - ``QUERY_ENHANCEMENT_ENABLE_REWRITE``
    - ``QUERY_ENHANCEMENT_ENABLE_HYDE``
    - ``QUERY_ENHANCEMENT_ENABLE_DECOMPOSITION``

    单独抽出依赖工厂便于测试通过 ``dependency_overrides`` 注入 mock 增强器。
    本依赖在 15.7 / 15.8 集成查询增强到搜索流程时使用；当前 :func:`search`
    端点尚未消费该实例，仅暴露依赖以便后续接入。
    """
    return build_query_enhancer()


async def _get_user_allowed_space_ids(
    user: User,
    db: AsyncSession,
) -> list[str]:
    """查询用户可访问的空间 ID 列表（仅取 read/write）。

    Admin 用户 (邮箱与 INITIAL_ADMIN_EMAIL 匹配) 拥有全部空间访问权,
    返回所有 spaces.id。

    Args:
        user: 当前登录用户
        db: 异步 DB 会话

    Returns:
        用户具备 read/write 权限的空间 ID 字符串列表
    """
    from app.api.auth import is_admin_user
    from app.models.space import Space

    # Admin 看到所有空间
    if is_admin_user(user):
        result = await db.execute(select(Space.id))
        return [str(sid) for sid in result.scalars().all()]

    stmt = select(Permission.resource_id).where(
        Permission.user_id == user.id,
        Permission.resource_type == ResourceType.space,
        Permission.access_level.in_([AccessLevel.read, AccessLevel.write]),
    )
    result = await db.execute(stmt)
    space_ids = result.scalars().all()
    return [str(sid) for sid in space_ids]


# ─── Endpoints ─────────────────────────────────────────────────────────


@router.post("/search", response_model=SearchResponseSchema)
async def search(
    body: SearchRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    search_service: SearchService = Depends(get_search_service),
):
    """执行复合搜索。

    - 多路召回（BM25 + Dense + Sparse） + RRF 融合 + Cross-Encoder 精排
    - 通过 Pre-Filtering 仅返回用户具备权限的文档块
    - 默认每页 10 条，最大 50 条
    - 整体执行在 5 秒内完成，超时返回 504

    Returns:
        :class:`SearchResponseSchema`：分页后的搜索结果
    """
    # 通过 Permission 表查询当前用户的可访问空间集合
    allowed_space_ids = await _get_user_allowed_space_ids(current_user, db)

    # 整体 5 秒超时（需求 6.7）。单路召回的 3 秒超时（需求 6.6）
    # 由 SearchService 内部用 asyncio.wait_for 控制。
    try:
        response = await asyncio.wait_for(
            search_service.search(
                query=body.query,
                user_id=str(current_user.id),
                allowed_space_ids=allowed_space_ids,
                page=body.page,
                page_size=body.page_size,
            ),
            timeout=SEARCH_TOTAL_TIMEOUT,
        )
    except asyncio.TimeoutError:
        # 全局异常处理器只处理 AppException 子类，这里走标准 504 信封
        logger.warning(
            "search_timeout",
            extra={
                "user_id": str(current_user.id),
                "query": body.query,
                "timeout_seconds": SEARCH_TOTAL_TIMEOUT,
            },
        )
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content={
                "error": {
                    "code": "SearchTimeout",
                    "message": (
                        f"搜索服务超时（>{SEARCH_TOTAL_TIMEOUT:.0f} 秒），请稍后重试"
                    ),
                    "details": {"timeout_seconds": SEARCH_TOTAL_TIMEOUT},
                }
            },
        )

    return SearchResponseSchema(
        results=[
            SearchResultItem(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                chunk_index=r.chunk_index,
                title_chain=r.title_chain,
                source_file=r.source_file,
                page_number=r.page_number,
                score=r.score,
                highlight=r.highlight,
            )
            for r in response.results
        ],
        total=response.total,
        page=response.page,
        page_size=response.page_size,
    )
