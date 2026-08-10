"""RAG 问答核心服务。

本模块实现 RAG 流程的核心逻辑（需求 8.2、8.3）：

    检索 Top-K 相关文档块 → 构建带上下文的 Prompt → 调用 LLM 生成答案

设计要点：
- 同时支持非流式（``answer``，需求 8.2）和流式（``answer_stream``，
  需求 8.3）两种调用方式。流式版本将首 token 等待时间限制在 5 秒内，
  超时则抛 ``RAGServiceError(reason="first_token_timeout")``。
- 依赖注入：``SearchService`` 与 ``LLMGateway`` 可在测试时被替换为 mock，
  避免触达真实的 OpenSearch / Qdrant / LLM 网关。
- 失败处理：LLM 调用失败统一抛出 :class:`RAGServiceError`，调用方可据此
  返回友好的错误响应；检索为空时则返回固定提示而不抛错。
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

from app.services.conversation_service import ConversationService
from app.services.llm_gateway import LLMGateway, LLMGatewayError
from app.services.search_service import SearchResult, SearchService

logger = logging.getLogger(__name__)

# ─── 常量 ────────────────────────────────────────────────────────────────

#: 默认检索条数（与需求 8.1 一致：默认 5，范围 1-20）。
DEFAULT_TOP_K = 5

#: top_k 允许的最小值。
MIN_TOP_K = 1

#: top_k 允许的最大值。
MAX_TOP_K = 20

#: 检索结果为空时返回给用户的提示语。
NO_CONTEXT_MESSAGE = "知识库中未找到相关内容。"

#: 默认相似度阈值（任务 16.6 / 需求 8.6）。
#:
#: 当 SearchService 返回的所有候选 chunk 的 ``score`` 都低于该值时，
#: RAGService 将退化为"未找到相关内容"的固定回复，不再调用 LLM，
#: 避免依据低相关上下文生成幻觉答案。
#:
#: 实例级阈值由 :class:`RAGService` 构造函数读取（默认从
#: ``Settings.SIMILARITY_THRESHOLD`` 取值，未注入 settings 时回退到本常量）。
DEFAULT_SIMILARITY_THRESHOLD = 0.5

#: 首 token 最长等待时间（秒）。
#: 原需求 8.3 写 ≤5 秒,但实际 LiteLLM proxy + 上游网关 + 思考型模型
#: (gpt-5.5/claude-sonnet) 的首 token 平均 8-15 秒,5 秒太苛刻会反复触发
#: ``first_token_timeout``。改为 30 秒,平衡用户体验和真实延迟。
#: 用户可通过环境变量 ``RAG_FIRST_TOKEN_TIMEOUT_SECONDS`` 覆盖。
import os as _os

FIRST_TOKEN_TIMEOUT_SECONDS = float(
    _os.environ.get("RAG_FIRST_TOKEN_TIMEOUT_SECONDS", "30.0")
)

#: 流式事件类型——常规 token。
STREAM_EVENT_TOKEN = "token"

#: 流式事件类型——引用列表（在 token 流结束后产出一次）。
STREAM_EVENT_SOURCES = "sources"

#: 流式事件类型——结束信号。
STREAM_EVENT_DONE = "done"

#: 流式事件类型——错误。
STREAM_EVENT_ERROR = "error"

#: System Prompt：约束 LLM 仅依据上下文作答并按统一格式标注引用。
#:
#: 任务 16.4：要求 LLM 在引用对应资料时**显式**使用 ``[i]`` 编号标注
#: （``i`` 对应 Prompt 中提供的资料编号，从 1 开始）。系统会基于回答文本
#: 用正则解析这些 ``[\d+]`` 标注，并据此判定每个 source 是否被实际引用。
SYSTEM_PROMPT = (
    "你是企业知识库助手，仅依据下方提供的编号资料回答用户问题。"
    "若资料不足以回答，请直接说明无法回答，不要编造信息。"
    "请在回答中**必须**使用 [n] 形式的方括号编号（例如 [1]、[2]）"
    "标注每段引用对应的资料编号，n 必须与下方提供的资料编号一致。"
    "同一资料可被多次引用，多个资料可在同一处合并引用（如 [1][2]）。"
)


#: 用于从 LLM 答案中解析引用编号 ``[n]`` 的正则。
#:
#: - 仅匹配方括号包裹的纯数字，例如 ``[1]``、``[12]``。
#: - 编号本身不限制范围（``9`` 也会被匹配到），由调用方再过滤超出实际范围
#:   的编号；这样即便 LLM 偶尔编造了 ``[99]`` 也不会影响输出。
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


# ─── 数据结构 ────────────────────────────────────────────────────────────


@dataclass
class Source:
    """RAG 答案的引用来源条目。

    每个 Source 对应检索到的一个 chunk，包含足够前端跳转和展示的元数据。

    任务 16.4：``cited`` 字段标记该 chunk 是否在 LLM 回答中被实际引用
    （即回答里出现了对应的 ``[i]`` 编号）。前端可以据此区分"被引用的来源"
    与"仅作为上下文提供但未被引用的来源"。
    """

    #: 在 Prompt 中分配给该 chunk 的序号（1-based），与回答中 ``[1]`` 等编号对应。
    index: int
    chunk_id: str
    document_id: str
    title_chain: str = ""
    source_file: str = ""
    page_number: int = 0
    score: float = 0.0
    #: LLM 答案中是否实际出现了 ``[index]`` 引用。
    cited: bool = False


@dataclass
class RAGAnswer:
    """RAG 问答的完整结果。

    Attributes:
        answer: LLM 生成的回答文本（检索为空时为固定提示语）。
        sources: 引用的 chunk 列表，按检索结果顺序排列。
        usage: LLM 调用的 token 用量，例如 ``{"prompt_tokens": ...}``。
            检索为空、未实际调用 LLM 时为空字典。
    """

    answer: str
    sources: list[Source] = field(default_factory=list)
    usage: dict = field(default_factory=dict)


@dataclass
class StreamEvent:
    """流式问答中的一个事件。

    Attributes:
        event: 事件类型，取值之一：``token`` / ``sources`` / ``done`` / ``error``。
        data: 事件载荷，按事件类型不同而结构不同：

            - ``token``：``{"text": str}``
            - ``sources``：``{"sources": list[dict]}``
            - ``done``：``{}``
            - ``error``：``{"code": str, "message": str}``
    """

    event: str
    data: dict


class RAGServiceError(Exception):
    """RAG 服务调用失败时抛出。

    主要场景：底层 :class:`LLMGateway` 调用失败（超时、鉴权、限流等），
    或其他不可恢复的错误。``reason`` 字段沿用 ``LLMGatewayError`` 的语义，
    便于上层根据原因生成不同的提示文案。
    """

    def __init__(self, message: str, reason: str = "unknown"):
        super().__init__(message)
        self.reason = reason


# ─── 服务实现 ────────────────────────────────────────────────────────────


class RAGService:
    """RAG 问答核心服务（不含会话与流式输出）。

    典型使用：

        >>> service = RAGService(search_service=..., llm_gateway=...)
        >>> result = await service.answer(
        ...     query="什么是 RAG？",
        ...     user_id="u-1",
        ...     allowed_space_ids=["s-1"],
        ...     top_k=5,
        ... )
        >>> result.answer
        '...'
        >>> result.sources[0].chunk_id
        '...'
    """

    def __init__(
        self,
        search_service: SearchService | None = None,
        llm_gateway: LLMGateway | None = None,
        conversation_service: ConversationService | None = None,
        similarity_threshold: float | None = None,
    ) -> None:
        """初始化 RAG 服务。

        Args:
            search_service: 文档检索服务。``None`` 时使用默认实例。
            llm_gateway: LLM 网关。``None`` 时使用默认实例。
            conversation_service: 会话历史服务（任务 16.5）。``None`` 时使用
                默认实例；当 ``answer`` / ``answer_stream`` 接收到 ``conversation_id``
                参数时，将通过该服务读取历史并把当前轮次写回。
            similarity_threshold: 相似度阈值（任务 16.6 / 需求 8.6）。
                ``None`` 时从 ``Settings.SIMILARITY_THRESHOLD`` 读取（默认 0.5）。
                所有候选 chunk 的 ``score`` 都低于该值时，将不调用 LLM 而直接
                返回 ``NO_CONTEXT_MESSAGE``。注意阈值仅作用于 RAG 输出的 chunks
                过滤，不会影响 SearchService 内部的排序与召回行为。
        """
        self._search_service = search_service or SearchService()
        self._llm_gateway = llm_gateway or LLMGateway()
        self._conversation_service = (
            conversation_service or ConversationService()
        )
        self._similarity_threshold = self._resolve_similarity_threshold(
            similarity_threshold
        )

    @property
    def similarity_threshold(self) -> float:
        """当前实例使用的相似度阈值。

        构造时确定后保持不变；外部可读取以记录日志或在测试中断言。
        """
        return self._similarity_threshold

    @staticmethod
    def _resolve_similarity_threshold(
        explicit: float | None,
    ) -> float:
        """决定本实例最终使用的相似度阈值。

        优先级：构造函数显式传入 > ``Settings.SIMILARITY_THRESHOLD`` > 模块默认。
        Settings 读取失败时（例如在最小化测试环境下）退回到默认值，确保该服务
        在缺少完整配置的场景下也能初始化成功。
        """
        if explicit is not None:
            return float(explicit)
        try:
            from app.core.config import get_settings

            return float(get_settings().SIMILARITY_THRESHOLD)
        except Exception:  # noqa: BLE001 - 配置不可用时退回默认
            return DEFAULT_SIMILARITY_THRESHOLD

    # ─── 主流程 ──────────────────────────────────────────────────────────

    async def answer(
        self,
        query: str,
        user_id: str,
        allowed_space_ids: list[str],
        top_k: int = DEFAULT_TOP_K,
        temperature: float = 1,
        max_tokens: int = 4096,
        conversation_id: str | None = None,
    ) -> RAGAnswer:
        """执行一次完整的 RAG 问答。

        流程：

        1. 调用 ``SearchService.search`` 拿到 Top-K 个相关 chunk。
        2. 若结果为空，返回固定提示，不调用 LLM。
        3. 否则按编号 ``[1]..[K]`` 拼接上下文，构造 Prompt。
           当传入 ``conversation_id`` 时，会把历史消息按 ``user`` /
           ``assistant`` 角色拼接进消息序列（任务 16.5）。
        4. 调用 ``LLMGateway.complete`` 生成答案。
        5. 把答案与对应的 sources/usage 一起返回。
        6. 当传入 ``conversation_id`` 时，把当前问题与答案 append 回会话历史。

        Args:
            query: 用户问题。
            user_id: 当前用户 ID（用于权限过滤）。
            allowed_space_ids: 用户可访问的空间 ID 列表。
            top_k: 检索条数。会被夹紧到 ``[MIN_TOP_K, MAX_TOP_K]`` 区间。
            temperature: LLM 采样温度。
            max_tokens: LLM 生成上限。
            conversation_id: 会话 ID。``None`` 表示单轮问答（不读取也不写入历史）。

        Returns:
            :class:`RAGAnswer`，包含答案文本、引用来源和 token 用量。

        Raises:
            RAGServiceError: LLM 调用失败时抛出。
        """
        top_k = self._clamp_top_k(top_k)

        # 1. 检索 Top-K
        search_response = await self._search_service.search(
            query=query,
            user_id=user_id,
            allowed_space_ids=allowed_space_ids,
            page=1,
            page_size=top_k,
        )
        results = self._apply_similarity_threshold(search_response.results)

        # 2. 检索为空 / 全部低于阈值：直接返回固定提示，不消耗 LLM 配额
        if not results:
            logger.info(
                "RAG: 无可用上下文 (user=%s, query_len=%d, threshold=%.2f)",
                user_id,
                len(query),
                self._similarity_threshold,
            )
            return RAGAnswer(answer=NO_CONTEXT_MESSAGE, sources=[], usage={})

        # 3. 构建 Prompt（含会话历史）
        sources = self._build_sources(results)
        history = await self._load_history(conversation_id)
        user_prompt = self._build_user_prompt(
            query, results, history=history
        )

        # 4. 调用 LLM
        try:
            llm_response = await self._llm_gateway.complete(
                prompt=user_prompt,
                system_prompt=SYSTEM_PROMPT,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except LLMGatewayError as exc:
            logger.error(
                "RAG: LLM 调用失败 (reason=%s): %s", exc.reason, exc
            )
            raise RAGServiceError(
                f"LLM 调用失败: {exc}", reason=exc.reason
            ) from exc

        # 5. 组装最终答案：根据答案文本标记 cited
        sources = self._mark_cited_sources(sources, llm_response.content)

        # 6. 把当前轮写回会话历史（仅在调用方提供 conversation_id 时）
        await self._persist_turn(
            conversation_id, query, llm_response.content
        )

        return RAGAnswer(
            answer=llm_response.content,
            sources=sources,
            usage=dict(llm_response.usage or {}),
        )

    # ─── 流式问答（需求 8.3） ──────────────────────────────────────────

    async def answer_stream(
        self,
        query: str,
        user_id: str,
        allowed_space_ids: list[str],
        top_k: int = DEFAULT_TOP_K,
        temperature: float = 1,
        max_tokens: int = 4096,
        first_token_timeout: float = FIRST_TOKEN_TIMEOUT_SECONDS,
        conversation_id: str | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """以流式方式产出 RAG 答案（异步生成器）。

        事件序列（正常路径）::

            token (n 次) → sources (1 次) → done (1 次)

        异常情况下会产出 ``error`` 事件并提前结束流：

        - 首 token 超过 ``first_token_timeout`` 秒未到达 → ``code=first_token_timeout``
        - LLM 调用失败 → ``code=<LLMGatewayError.reason>``

        检索为空时不调用 LLM，事件序列为：

            token("知识库中未找到相关内容。") → sources([]) → done

        Args:
            query: 用户问题。
            user_id: 当前用户 ID（用于权限过滤）。
            allowed_space_ids: 用户可访问的空间 ID 列表。
            top_k: 检索条数，被夹紧到 ``[MIN_TOP_K, MAX_TOP_K]``。
            temperature: LLM 采样温度。
            max_tokens: LLM 生成上限。
            first_token_timeout: 首 token 最长等待时间，默认 5 秒（需求 8.3）。

        Yields:
            :class:`StreamEvent`：依次为若干 ``token``、一条 ``sources``、一条 ``done``；
            异常时改为产出 ``error`` 后立即结束。
        """
        top_k = self._clamp_top_k(top_k)

        # 1. 检索 Top-K
        search_response = await self._search_service.search(
            query=query,
            user_id=user_id,
            allowed_space_ids=allowed_space_ids,
            page=1,
            page_size=top_k,
        )
        results = self._apply_similarity_threshold(search_response.results)

        # 2. 检索为空 / 全部低于阈值：直接给出固定提示，不调用 LLM
        if not results:
            logger.info(
                "RAG-stream: 无可用上下文 (user=%s, query_len=%d, threshold=%.2f)",
                user_id,
                len(query),
                self._similarity_threshold,
            )
            yield StreamEvent(
                event=STREAM_EVENT_TOKEN,
                data={"text": NO_CONTEXT_MESSAGE},
            )
            yield StreamEvent(
                event=STREAM_EVENT_SOURCES,
                data={"sources": []},
            )
            yield StreamEvent(event=STREAM_EVENT_DONE, data={})
            return

        # 3. 构建 Prompt（含会话历史）
        sources = self._build_sources(results)
        history = await self._load_history(conversation_id)
        user_prompt = self._build_user_prompt(
            query, results, history=history
        )
        # 任务 16.4：累积完整答案文本，待 token 流结束后用于解析 [n] 引用编号。
        answer_buffer: list[str] = []

        # 4. 启动 LLM 流式调用，并给"首 token"加超时
        token_iter = self._llm_gateway.stream(
            prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=temperature,
            max_tokens=max_tokens,
        ).__aiter__()

        try:
            try:
                first_token = await asyncio.wait_for(
                    token_iter.__anext__(),
                    timeout=first_token_timeout,
                )
            except asyncio.TimeoutError:
                logger.error(
                    "RAG-stream: 首 token 超时 %.1fs (user=%s)",
                    first_token_timeout,
                    user_id,
                )
                yield StreamEvent(
                    event=STREAM_EVENT_ERROR,
                    data={
                        "code": "first_token_timeout",
                        "message": (
                            f"首 token 在 {first_token_timeout:.0f} 秒内未返回"
                        ),
                    },
                )
                return
            except StopAsyncIteration:
                # LLM 直接返回空流：视作正常结束，但仍要补上 sources/done
                first_token = None
            except LLMGatewayError as exc:
                logger.error(
                    "RAG-stream: LLM 调用失败 (reason=%s): %s",
                    exc.reason,
                    exc,
                )
                yield StreamEvent(
                    event=STREAM_EVENT_ERROR,
                    data={"code": exc.reason, "message": str(exc)},
                )
                return

            if first_token is not None:
                answer_buffer.append(first_token)
                yield StreamEvent(
                    event=STREAM_EVENT_TOKEN,
                    data={"text": first_token},
                )

            # 5. 产出剩余 token
            try:
                async for token in token_iter:
                    answer_buffer.append(token)
                    yield StreamEvent(
                        event=STREAM_EVENT_TOKEN,
                        data={"text": token},
                    )
            except LLMGatewayError as exc:
                logger.error(
                    "RAG-stream: LLM 流中断 (reason=%s): %s",
                    exc.reason,
                    exc,
                )
                yield StreamEvent(
                    event=STREAM_EVENT_ERROR,
                    data={"code": exc.reason, "message": str(exc)},
                )
                return
        finally:
            # 主动关闭异步迭代器，避免 LLM 长连接悬挂
            aclose = getattr(token_iter, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:  # noqa: BLE001 - 关闭流的兜底
                    pass

        # 6. 流正常结束：基于完整答案标记 cited 后产出 sources，再产出 done
        full_answer = "".join(answer_buffer)
        sources = self._mark_cited_sources(sources, full_answer)
        # 任务 16.5：流式正常结束后写回会话历史（仅在传入 conversation_id 时）。
        await self._persist_turn(conversation_id, query, full_answer)
        yield StreamEvent(
            event=STREAM_EVENT_SOURCES,
            data={"sources": [self._source_to_dict(s) for s in sources]},
        )
        yield StreamEvent(event=STREAM_EVENT_DONE, data={})

    # ─── 辅助方法 ────────────────────────────────────────────────────────

    @staticmethod
    def parse_citations(answer_text: str) -> set[int]:
        """从答案文本中解析所有 ``[n]`` 形式的引用编号。

        - 仅识别方括号包裹的纯数字，例如 ``[1]``、``[12]``。
        - 不限制编号范围，调用方需自行根据实际 sources 数量过滤超界编号
          （如 LLM 偶尔编造 ``[99]`` 不应影响输出）。
        - 若答案为空或不含任何标注，返回空 set。

        Args:
            answer_text: LLM 生成的完整答案文本。

        Returns:
            出现过的所有引用编号的集合（去重）。
        """
        if not answer_text:
            return set()
        return {int(m) for m in _CITATION_PATTERN.findall(answer_text)}

    @staticmethod
    def _mark_cited_sources(
        sources: list[Source], answer_text: str
    ) -> list[Source]:
        """根据答案文本就地标记每个 source 的 ``cited`` 字段。

        - 只有"实际存在"且被引用的编号才会置 ``cited=True``。
        - LLM 引用的越界编号（如 ``[99]``）会被静默忽略。

        Args:
            sources: ``_build_sources`` 输出的 Source 列表。
            answer_text: LLM 生成的完整答案文本。

        Returns:
            原列表（同时已就地修改）。
        """
        cited_indices = RAGService.parse_citations(answer_text)
        valid_indices = {s.index for s in sources}
        effective = cited_indices & valid_indices
        for source in sources:
            source.cited = source.index in effective
        return sources

    @staticmethod
    def _clamp_top_k(top_k: int) -> int:
        """把 ``top_k`` 夹紧到合法区间。"""
        if top_k < MIN_TOP_K:
            return MIN_TOP_K
        if top_k > MAX_TOP_K:
            return MAX_TOP_K
        return top_k

    def _apply_similarity_threshold(
        self, results: list[SearchResult]
    ) -> list[SearchResult]:
        """按 ``similarity_threshold`` 过滤候选 chunk（任务 16.6 / 需求 8.6）。

        - 仅保留 ``score >= self._similarity_threshold`` 的 chunk；边界值
          （如分数恰好等于阈值）按"满足条件"处理，与需求中"低于"的措辞一致。
        - 阈值 ≤0 时直接全量返回，避免对 score 为 0 的合法结果误伤；同时也方
          便测试场景显式关闭过滤。
        - 仅作用于 RAG 输出层，不会影响 SearchService 内部已计算好的排序。
        - 当全部候选都低于阈值时返回空列表，调用方据此走"未找到相关信息"的
          退化路径。
        """
        if self._similarity_threshold <= 0:
            return list(results)
        return [r for r in results if r.score >= self._similarity_threshold]

    @staticmethod
    def _format_source_label(result: SearchResult, index: int) -> str:
        """生成 chunk 的引用标签，例如 ``[1] 文档:《xxx》章节:xxx 页:3``。

        - 文件名缺失时显示为 ``未知文档``。
        - 章节链缺失时省略 ``章节:`` 段。
        - 页码 ≤0 时省略 ``页:`` 段（很多文档没有真实页码）。
        """
        title = result.source_file or "未知文档"
        parts = [f"[{index}] 文档:《{title}》"]
        if result.title_chain:
            parts.append(f"章节:{result.title_chain}")
        if result.page_number and result.page_number > 0:
            parts.append(f"页:{result.page_number}")
        return " ".join(parts)

    @classmethod
    def _build_user_prompt(
        cls,
        query: str,
        results: list[SearchResult],
        history: list[dict] | None = None,
    ) -> str:
        """按需求 8.2 拼接 Prompt：上下文段 + 可选对话历史 + 问题段。

        - 每个 chunk 一段，前缀为 ``[i] ...`` 引用标签
        - 块之间用空行分隔，便于 LLM 分辨边界
        - 任务 16.5：``history`` 非空时插入"对话历史："段，``user`` / ``assistant``
          各自换行展示，让 LLM 看到多轮上下文。会话历史里只读取 ``role`` 和
          ``content`` 字段，其它字段（如 ``citations``）忽略。
        - 用户问题放在最后，并以 ``问题：`` 开头
        """
        context_blocks: list[str] = []
        for index, result in enumerate(results, start=1):
            label = cls._format_source_label(result, index)
            # 当前 SearchResult 仅暴露 highlight 作为可用文本片段
            content = (result.highlight or "").strip()
            context_blocks.append(f"{label}\n{content}")
        context = "\n\n".join(context_blocks)

        sections = [f"以下是检索到的参考资料：\n\n{context}"]

        history_section = cls._format_history(history)
        if history_section:
            sections.append(history_section)

        sections.append(f"问题：{query}")
        return "\n\n".join(sections)

    @staticmethod
    def _format_history(history: list[dict] | None) -> str:
        """把会话历史格式化为 Prompt 中的"对话历史："段。

        - ``history`` 为空（None 或 []）时返回空串，调用方据此跳过该段。
        - 仅认 ``role`` ∈ {``user``, ``assistant``} 的条目；其它角色（包括脏
          数据）会被静默忽略，避免让 LLM 看到未知角色。
        """
        if not history:
            return ""
        lines: list[str] = []
        for msg in history:
            role = msg.get("role") if isinstance(msg, dict) else None
            content = msg.get("content", "") if isinstance(msg, dict) else ""
            if role == "user":
                lines.append(f"用户：{content}")
            elif role == "assistant":
                lines.append(f"助手：{content}")
        if not lines:
            return ""
        return "以下是此前的对话历史（按时间顺序）：\n" + "\n".join(lines)

    async def _load_history(
        self, conversation_id: str | None
    ) -> list[dict]:
        """读取指定会话的历史消息；``None`` 时返回空列表。

        包装一层是为了让 ``answer`` / ``answer_stream`` 主流程更简洁，并集中
        处理 Redis 异常——历史读取失败不应阻塞当前问答，最差也只是退化为单轮。
        """
        if not conversation_id:
            return []
        try:
            return await self._conversation_service.get_history(
                conversation_id
            )
        except Exception:  # noqa: BLE001 - Redis 故障兜底
            logger.warning(
                "RAG: 读取会话历史失败，按单轮处理 (conversation_id=%s)",
                conversation_id,
                exc_info=True,
            )
            return []

    async def _persist_turn(
        self, conversation_id: str | None, query: str, answer: str
    ) -> None:
        """把当前轮（user + assistant）写回会话历史。

        - ``conversation_id`` 为 ``None`` 时不做任何事。
        - Redis 写入失败仅记录日志，不抛给调用方——答案本身已经返回成功，
          会话写入失败只是历史记忆受损，不应让用户感知到错误。
        """
        if not conversation_id:
            return
        try:
            await self._conversation_service.append(
                conversation_id, "user", query
            )
            await self._conversation_service.append(
                conversation_id, "assistant", answer
            )
        except Exception:  # noqa: BLE001 - Redis 故障兜底
            logger.warning(
                "RAG: 写入会话历史失败 (conversation_id=%s)",
                conversation_id,
                exc_info=True,
            )

    @staticmethod
    def _build_sources(results: list[SearchResult]) -> list[Source]:
        """把 SearchResult 列表转换为 :class:`Source` 列表。

        编号与 :meth:`_build_user_prompt` 中的引用标签一一对应。
        """
        sources: list[Source] = []
        for index, result in enumerate(results, start=1):
            sources.append(
                Source(
                    index=index,
                    chunk_id=result.chunk_id,
                    document_id=result.document_id,
                    title_chain=result.title_chain,
                    source_file=result.source_file,
                    page_number=result.page_number,
                    score=result.score,
                )
            )
        return sources

    @staticmethod
    def _source_to_dict(source: Source) -> dict:
        """把 :class:`Source` 序列化成可放进 SSE ``data`` 的纯 dict。"""
        return {
            "index": source.index,
            "chunk_id": source.chunk_id,
            "document_id": source.document_id,
            "title_chain": source.title_chain,
            "source_file": source.source_file,
            "page_number": source.page_number,
            "score": source.score,
            "cited": source.cited,
        }
