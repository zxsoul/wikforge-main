"""问答（QA）API 路由。

任务 16.3：在 :mod:`app.services.rag_service` 之上提供 SSE 流式问答接口。

接口列表：

- ``POST /api/qa/ask``：非流式，直接返回 :class:`RAGAnswer` 的 JSON 表示。
- ``POST /api/qa/ask/stream``：以 ``text/event-stream`` 推送流式答案。

SSE 事件格式（逐事件以空行分隔）::

    event: token
    data: {"text": "..."}

    event: sources
    data: {"sources": [...]}

    event: done
    data: {}

    event: error
    data: {"code": "...", "message": "..."}
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.core.database import get_db
from app.models.permission import AccessLevel, Permission, ResourceType
from app.models.user import User
from app.services.conversation_service import ConversationService
from app.services.rag_service import (
    DEFAULT_TOP_K,
    FIRST_TOKEN_TIMEOUT_SECONDS,
    MAX_TOP_K,
    MIN_TOP_K,
    STREAM_EVENT_ERROR,
    RAGService,
    RAGServiceError,
    Source,
    StreamEvent,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/qa", tags=["qa"])


# ─── 友好错误文案（任务 16.7 / 需求 8.7） ──────────────────────────────
#
# 需求 8.7：LLM 调用失败或响应超时时，应"向用户返回服务暂时不可用的错误
# 提示"。这里把 ``LLMGatewayError.reason`` 映射为面向终端用户的中文提示，
# 既保持"服务暂时不可用"的统一基调，又让用户大致了解问题类别（超时 / 鉴权
# / 限流 / 模型不可用 / 其它），便于运维和用户决定是否重试。
#
# 默认提示语在 :data:`_DEFAULT_LLM_FAILURE_MESSAGE` 中给出，所有未列出的
# reason（包括 ``unknown``）都退回到默认值。
_DEFAULT_LLM_FAILURE_MESSAGE = "服务暂时不可用，请稍后重试。"
_LLM_FAILURE_MESSAGES: dict[str, str] = {
    "timeout": "服务暂时不可用：模型响应超时，请稍后重试。",
    "rate_limit": "服务暂时不可用：当前请求过多，请稍后重试。",
    "auth": "服务暂时不可用：模型访问凭证异常，请联系管理员。",
    "model_unavailable": "服务暂时不可用：所选模型当前不可用，请联系管理员。",
}


def _format_llm_failure_message(reason: str) -> str:
    """根据 ``LLMGatewayError.reason`` 返回中文友好提示。

    - 命中已知 reason → 返回对应文案
    - 未命中或为 ``unknown`` → 返回默认提示
    - 不在响应里回显原始异常 message，避免把英文 stacktrace 暴露给终端用户
    """
    return _LLM_FAILURE_MESSAGES.get(reason, _DEFAULT_LLM_FAILURE_MESSAGE)


# ─── Schemas ───────────────────────────────────────────────────────────


class AskRequest(BaseModel):
    """问答请求体。"""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="用户问题（最多 2000 字符）",
    )
    top_k: int = Field(
        default=DEFAULT_TOP_K,
        ge=MIN_TOP_K,
        le=MAX_TOP_K,
        description="检索条数（1-20，默认 5）",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="LLM 采样温度（0-2）",
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        le=128_000,
        description="LLM 生成上限",
    )


class SourceItem(BaseModel):
    """API 输出的引用条目。"""

    index: int
    chunk_id: str
    document_id: str
    title_chain: str = ""
    source_file: str = ""
    page_number: int = 0
    score: float = 0.0
    #: 任务 16.4：标记该来源是否在 LLM 答案中被实际引用（``[index]`` 出现）。
    cited: bool = False


class AskResponse(BaseModel):
    """非流式问答响应体。"""

    answer: str
    sources: list[SourceItem]
    usage: dict


class ConversationStatusResponse(BaseModel):
    """会话状态响应体（任务 16.8 / 需求 8.8）。

    Attributes:
        conversation_id: 查询的会话 ID（透传）。
        exists: 会话当前是否仍存在于 Redis 中。``False`` 表示会话已过期或
            从未创建——需求 8.8 要求此时调用方按"新会话"处理。
        ttl_seconds: 剩余 TTL（秒）。Redis 协议：
            - ``-2``：会话不存在或已过期被删除
            - ``-1``：会话存在但未设置 TTL（理论上不应出现，因为本服务在
              每次 append 时都会 ``EXPIRE``）
            - ``>= 0``：剩余存活秒数
    """

    conversation_id: str
    exists: bool
    ttl_seconds: int


# ─── Dependencies ──────────────────────────────────────────────────────


def get_rag_service() -> RAGService:
    """构造默认 :class:`RAGService` 实例（FastAPI 依赖项）。

    单独抽出以便测试通过 ``app.dependency_overrides`` 注入 mock。
    """
    return RAGService()


def get_conversation_service() -> ConversationService:
    """构造默认 :class:`ConversationService` 实例（FastAPI 依赖项）。

    任务 16.8：会话状态查询路由所用，单独抽出以便测试注入 fakeredis。
    """
    return ConversationService()


async def get_user_allowed_space_ids(
    user: User, db: AsyncSession
) -> list[str]:
    """返回当前用户具备读/写权限的空间 ID 列表。

    Admin 用户 (邮箱与 INITIAL_ADMIN_EMAIL 匹配) 拥有全部空间访问权。
    """
    from app.api.auth import is_admin_user
    from app.models.space import Space

    if is_admin_user(user):
        result = await db.execute(select(Space.id))
        return [str(sid) for sid in result.scalars().all()]

    stmt = select(Permission.resource_id).where(
        Permission.user_id == user.id,
        Permission.resource_type == ResourceType.space,
        Permission.access_level.in_([AccessLevel.read, AccessLevel.write]),
    )
    result = await db.execute(stmt)
    return [str(sid) for sid in result.scalars().all()]


# ─── SSE 编码 ──────────────────────────────────────────────────────────


def _encode_sse(event: StreamEvent) -> str:
    """把 :class:`StreamEvent` 编码为 SSE wire format 字符串。"""
    payload = json.dumps(event.data, ensure_ascii=False)
    return f"event: {event.event}\ndata: {payload}\n\n"


# ─── Endpoints ─────────────────────────────────────────────────────────


@router.post("/ask", response_model=AskResponse)
async def ask(
    body: AskRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    rag_service: RAGService = Depends(get_rag_service),
) -> AskResponse:
    """非流式问答（需求 8.2）。"""
    allowed_space_ids = await get_user_allowed_space_ids(current_user, db)
    try:
        result = await rag_service.answer(
            query=body.question,
            user_id=str(current_user.id),
            allowed_space_ids=allowed_space_ids,
            top_k=body.top_k,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
        )
    except RAGServiceError as exc:
        logger.warning("QA-ask: RAG 失败 (reason=%s)", exc.reason)
        # 任务 16.7 / 需求 8.7：返回中文友好提示，不回显原始异常 message。
        # 当前对话上下文（conversation_id 维护）由 RAGService 在写回历史
        # 之前抛错，因此本次失败不会污染会话历史，符合"保留当前对话上下文
        # 不丢失"的要求。
        return AskResponse(
            answer=_format_llm_failure_message(exc.reason),
            sources=[],
            usage={},
        )

    return AskResponse(
        answer=result.answer,
        sources=[_source_to_schema(s) for s in result.sources],
        usage=dict(result.usage),
    )


@router.post("/ask/stream")
async def ask_stream(
    body: AskRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    rag_service: RAGService = Depends(get_rag_service),
) -> StreamingResponse:
    """SSE 流式问答（需求 8.3）。

    返回 ``text/event-stream``，事件类型见模块顶部说明。
    """
    allowed_space_ids = await get_user_allowed_space_ids(current_user, db)
    user_id = str(current_user.id)

    async def event_source():
        try:
            async for event in rag_service.answer_stream(
                query=body.question,
                user_id=user_id,
                allowed_space_ids=allowed_space_ids,
                top_k=body.top_k,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
                first_token_timeout=FIRST_TOKEN_TIMEOUT_SECONDS,
            ):
                # 任务 16.7 / 需求 8.7：把 LLM 失败的 error 事件 message 替换
                # 为中文友好提示。RAGService 已经在 ``code`` 字段保留了原始
                # reason（``timeout`` / ``rate_limit`` / ``auth`` /
                # ``model_unavailable`` / ``first_token_timeout`` 等），前端
                # 可据此决定是否显示重试按钮；终端用户看到的 message 始终为
                # 中文友好文案而非英文堆栈。
                yield _encode_sse(_humanize_error_event(event))
        except Exception:  # noqa: BLE001 - 兜底，避免连接挂死
            logger.error("QA-stream: 未预期错误", exc_info=True)
            yield _encode_sse(
                StreamEvent(
                    event=STREAM_EVENT_ERROR,
                    data={
                        "code": "internal_error",
                        "message": _DEFAULT_LLM_FAILURE_MESSAGE,
                    },
                )
            )

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/conversations/{conversation_id}/status",
    response_model=ConversationStatusResponse,
)
async def get_conversation_status(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    conversation_service: ConversationService = Depends(
        get_conversation_service
    ),
) -> ConversationStatusResponse:
    """查询会话是否仍处于活跃状态（任务 16.8 / 需求 8.8）。

    需求 8.8：会话超过 30 分钟无新消息后应被标记为过期，下次提问开启新
    会话。Redis 自动过期会在 TTL 到期时删除整个 List，因此本接口只需

    - 调 ``EXISTS`` 判断会话是否仍存在；
    - 调 ``TTL`` 返回剩余秒数。

    返回值的语义供前端使用：

    - ``exists=False``：会话已过期或从未创建，前端应丢弃旧 ``conversation_id``
      并在下次提问时让 :class:`RAGService` 自动开新会话（传入旧 ID 时
      :class:`RAGService` 也会按空历史处理，业务层无需特别区分）。
    - ``exists=True`` 且 ``ttl_seconds > 0``：会话活跃，可继续追问。

    本接口仅依赖鉴权用户身份完成调用合法性，不做"会话归属用户"校验
    （会话归属由 :class:`RAGService` 自身在写入历史时通过 ``conversation_id``
    隔离；引入归属校验需要持久化所有者信息，超出本任务范围）。
    """
    # 路径参数为空在 FastAPI 路由层不会到达，但 conversation_id 来自 URL，
    # 仍做一道兜底防御以避免 ConversationService 抛 ValueError 透传 5xx。
    if not conversation_id:
        return ConversationStatusResponse(
            conversation_id="", exists=False, ttl_seconds=-2
        )
    exists = await conversation_service.is_active(conversation_id)
    ttl_seconds = await conversation_service.ttl(conversation_id)
    logger.debug(
        "QA-status: user=%s conversation=%s exists=%s ttl=%s",
        current_user.id,
        conversation_id,
        exists,
        ttl_seconds,
    )
    return ConversationStatusResponse(
        conversation_id=conversation_id,
        exists=exists,
        ttl_seconds=int(ttl_seconds),
    )


# ─── helpers ───────────────────────────────────────────────────────────


def _humanize_error_event(event: StreamEvent) -> StreamEvent:
    """把 SSE ``error`` 事件的 ``message`` 字段重写为中文友好提示。

    - 仅作用于 ``error`` 事件，其它事件原样返回。
    - 保留 ``code``（即 ``LLMGatewayError.reason`` 或 ``first_token_timeout``）
      让前端按错误类别决定行为，例如对 ``timeout`` / ``first_token_timeout``
      展示重试按钮，对 ``auth`` / ``model_unavailable`` 提示联系管理员。
    - 不回显原始异常文本，避免英文堆栈或 provider 内部信息暴露给终端用户。
    """
    if event.event != STREAM_EVENT_ERROR:
        return event
    code = str(event.data.get("code", "")) or "unknown"
    # 首 token 超时同样映射到 ``timeout`` 文案，对用户而言两者都是"超时"。
    message_key = "timeout" if code == "first_token_timeout" else code
    new_data = dict(event.data)
    new_data["message"] = _format_llm_failure_message(message_key)
    return StreamEvent(event=event.event, data=new_data)


def _source_to_schema(source: Source) -> SourceItem:
    """把内部 :class:`Source` 映射为 API 输出的 :class:`SourceItem`。"""
    return SourceItem(
        index=source.index,
        chunk_id=source.chunk_id,
        document_id=source.document_id,
        title_chain=source.title_chain,
        source_file=source.source_file,
        page_number=source.page_number,
        score=source.score,
        cited=source.cited,
    )
