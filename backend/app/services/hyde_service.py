"""HyDE 服务（任务 15.2）。

需求 7.2：当 HyDE 功能启用时，搜索引擎应基于原始查询生成 1 至 3 个假设文档嵌入，
并将其作为补充向量参与语义检索。

本模块提供 ``HyDEService`` 单一职责组件：

- ``generate_hypothetical_embeddings(query)`` 统一入口，返回 ``≤ 3`` 条 dense
  向量（不包含原始查询本身的向量；原始查询的向量由调用方/SearchService 自行生成）
- 调用 ``LLMGateway`` 让模型针对 query 生成若干段 50-150 字的"假设回答"伪文档
- 调用 ``EmbeddingService`` 将每段伪文档转成 dense 向量
- 通过 ``asyncio.wait_for(timeout=3.0)`` 在 3 秒内强约束完成；超时即降级返回 ``[]``
- LLM / Embedding 失败时静默降级返回 ``[]`` 或仅保留成功段落，符合需求 7.5 的
  "查询增强失败应静默降级"约束

为什么要有独立的 ``HyDEService`` 模块？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- ``QueryEnhancer`` 负责协调 改写 / HyDE / 子查询分解 三种增强能力，并行调度
  + 总超时；
- ``HyDEService`` 仅聚焦 HyDE 这一项，便于单独单测、独立替换实现（例如未来
  切换到本地小模型或者基于检索结果再生成的 HyDE 变体），并保持与设计文档一致
  的 "1-3 个假设文档 / 3 秒超时" 硬约束。

调用关系：``SearchService → QueryEnhancer → HyDEService → LLMGateway / EmbeddingService``。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

from app.services.embedding_service import EmbeddingService
from app.services.llm_gateway import LLMGateway, LLMGatewayError

logger = logging.getLogger(__name__)

# 单次 HyDE 整体超时时间（秒）。
#
# 对应需求 7.5 给出的 5 秒总预算下，HyDE 通常是最耗时的一项（LLM 输出多段长文本
# + 多次 embedding），因此单独留 3 秒预算，给改写/分解保留余量。
HYDE_TIMEOUT_SECONDS: float = 3.0

# 输出假设文档数量上限。需求 7.2 要求 "1 至 3 个"。
MAX_HYPOTHETICAL_DOCUMENTS: int = 3

# 单段假设文档的目标字数下限（字符）。低于该长度的段落（例如 LLM 误输出标题）
# 会被丢弃，避免向量化噪声。
_MIN_DOC_LENGTH: int = 20

# 单段假设文档的硬上限（字符）。LLM 偶尔会生成长篇大论，超长输入既浪费 embedding
# 配额，也会被 EmbeddingService 在内部截断；这里提前截断以保持日志可观察性。
_MAX_DOC_LENGTH: int = 400

# LLM 调用使用的 max_tokens 上限。3 段 ~150 字的中文段落约 600 token，留一点冗余。
_HYDE_MAX_TOKENS: int = 1024

# LLM 采样温度。略高以鼓励内容多样性，但仍受 system prompt 控制。
_HYDE_TEMPERATURE: float = 0.5

# 系统提示词：要求 LLM 输出 JSON 数组，避免自然语言混杂导致解析失败。
_SYSTEM_PROMPT: str = (
    "你是一个专业的检索增强助手。你的任务是针对用户搜索查询，"
    "生成 1 到 3 段假设性的理想答案段落，用于补充向量检索 (HyDE)。\n\n"
    "生成要求：\n"
    "1. 每段假设文档应直接回答查询，长度 50-150 字\n"
    "2. 内容要专业、具体、信息密度高，像真实文档而非泛泛而谈\n"
    "3. 不同段落可以从不同角度或层次回答，避免完全重复\n"
    "4. 不要复述查询本身，不要包含不确定语气（例如 假设 / 可能）\n"
    "5. 不要输出标题、编号、Markdown 格式\n\n"
    '输出格式：严格输出 JSON 数组，例如 ["段落1", "段落2"]。'
    "数量在 1 到 3 之间。不要输出任何解释或额外文本。"
)


class HyDEService:
    """LLM 驱动的假设文档嵌入（HyDE）组件。

    使用方式::

        service = HyDEService(
            llm_gateway=gateway,
            embedding_service=embeddings,
        )
        vectors = await service.generate_hypothetical_embeddings("机器学习入门")
        # vectors 长度在 0 到 3 之间，每个向量为 dense 向量

    设计要点：

    - 输入 ``query`` 为空 / 仅空白时直接返回 ``[]``，**不调用** LLM
    - 通过 ``asyncio.wait_for`` 强制 3 秒内完成；超时返回 ``[]`` 并记录 warning
    - 任何 LLM 错误或解析错误都会被吞掉并降级为 ``[]``，不向上抛出
    - Embedding 阶段对每段独立处理：单段失败仅丢弃该段，其它成功段落仍返回；
      全部失败时返回 ``[]``
    - 输出向量数量严格限制在 ``MAX_HYPOTHETICAL_DOCUMENTS = 3`` 以内
    """

    def __init__(
        self,
        llm_gateway: LLMGateway | None = None,
        embedding_service: EmbeddingService | None = None,
        timeout: float = HYDE_TIMEOUT_SECONDS,
        max_documents: int = MAX_HYPOTHETICAL_DOCUMENTS,
    ) -> None:
        """初始化 HyDE 服务。

        Args:
            llm_gateway: LLM 网关。为 ``None`` 时构造默认实例（生产路径走 LiteLLM）。
                测试中应注入 mock，避免真实网络调用。
            embedding_service: Embedding 服务。为 ``None`` 时构造默认实例。
                测试中应注入 mock，避免真实 embedding 调用。
            timeout: 整体超时（秒），覆盖 LLM + 全部 embedding 调用。默认 ``3.0``。
            max_documents: 输出假设文档数量上限。默认 ``3`` 对应需求 7.2。
        """
        # LLM 网关本身的 timeout 设为与 HyDE timeout 一致，避免内部等待时间超过外层。
        self._llm = llm_gateway or LLMGateway(timeout=timeout)
        self._embedding_service = embedding_service or EmbeddingService()
        self._timeout = timeout
        self._max_documents = max(0, max_documents)

    async def generate_hypothetical_embeddings(self, query: str) -> list[list[float]]:
        """生成不超过 ``max_documents`` 个假设文档的 dense 向量。

        Args:
            query: 原始用户查询。空 / 仅空白会直接返回 ``[]`` 且不调用 LLM。

        Returns:
            dense 向量列表，长度 ``0 ≤ len ≤ max_documents``。
            发生超时、LLM 错误、全部 embedding 失败时统一返回 ``[]``。
        """
        # 输入校验：空字符串或纯空白直接短路，避免无谓的 LLM / embedding 调用。
        if not query or not query.strip():
            return []

        if self._max_documents == 0:
            return []

        try:
            return await asyncio.wait_for(
                self._generate_internal(query),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            # 超时是预期内的降级路径，记录 warning 即可，不向上抛。
            logger.warning(
                "HyDE 生成超时（%.1fs），降级返回空列表 query=%r",
                self._timeout,
                query,
            )
            return []
        except LLMGatewayError as exc:
            # LLM 网关错误（限流 / 鉴权 / 模型不可用等）也走降级。
            logger.warning("HyDE LLM 调用失败: %s, query=%r", exc, query)
            return []
        except Exception as exc:  # noqa: BLE001 - 兜底任何意外错误，保持服务可用
            logger.warning("HyDE 发生未知错误: %s, query=%r", exc, query)
            return []

    # ─── 内部辅助 ──────────────────────────────────────────────────

    async def _generate_internal(self, query: str) -> list[list[float]]:
        """完整的 HyDE 流水线：LLM 生成伪文档 → 解析 → embedding。

        被 ``asyncio.wait_for`` 包裹以受总超时约束。
        """
        # 1. 调 LLM 生成假设文档
        response = await self._llm.complete(
            prompt=self._build_user_prompt(query),
            system_prompt=_SYSTEM_PROMPT,
            temperature=_HYDE_TEMPERATURE,
            max_tokens=_HYDE_MAX_TOKENS,
        )

        content = (response.content or "").strip()
        if not content:
            return []

        documents = self._parse_documents(content)
        documents = self._normalize(documents)
        if not documents:
            return []

        # 2. 对每段独立 embedding，单段失败仅丢弃该段
        return await self._embed_documents(documents)

    def _build_user_prompt(self, query: str) -> str:
        """构建发送给 LLM 的用户消息。"""
        return (
            f"请为以下搜索查询生成 1 到 {self._max_documents} 段假设性的理想答案段落，"
            f"用于补充向量检索。\n"
            f"严格按照系统提示中描述的 JSON 数组格式输出。\n\n"
            f"查询：{query}"
        )

    def _parse_documents(self, content: str) -> list[str]:
        """解析 LLM 输出，优先按 JSON 数组解析，失败时退化到按段落分隔。

        Args:
            content: LLM 返回的原始文本

        Returns:
            原始段落列表（未做长度过滤 / 未截断数量）
        """
        # 优先尝试 JSON 数组解析。LLM 偶尔会在数组前后添加 ```json fence 或解释文本，
        # 因此先用正则提取首个 ``[...]`` 块再解析。
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match is not None:
            json_text = match.group(0)
            try:
                parsed = json.loads(json_text)
            except (json.JSONDecodeError, ValueError):
                parsed = None

            if isinstance(parsed, list):
                # 仅接受字符串元素；其它类型（dict / number）忽略，保持鲁棒性。
                return [item.strip() for item in parsed if isinstance(item, str)]

        # JSON 解析失败：退化为按段落分隔。优先用双换行分隔，单换行兜底。
        return self._parse_paragraphs(content)

    @staticmethod
    def _parse_paragraphs(content: str) -> list[str]:
        """按段落分隔解析 LLM 输出，作为 JSON 解析失败时的兜底。

        - 优先按双换行 ``\\n\\n`` 分段（典型 markdown 段落分隔）
        - 若没有双换行，则按单换行分段（每行视为一段）
        - 剥离常见的序号 / Markdown 强调符号
        """
        if not content:
            return []

        # 双换行优先；若无双换行则使用单换行
        if "\n\n" in content:
            raw_parts = content.split("\n\n")
        else:
            raw_parts = content.splitlines()

        results: list[str] = []
        for raw in raw_parts:
            line = raw.strip()
            if not line:
                continue

            # 剥离形如 "1. " / "1) " / "1、" / "段落 1：" 等前缀
            line = re.sub(r"^\s*\d+\s*[\.\)、:：]\s*", "", line)
            line = re.sub(r"^段落\s*\d+\s*[:：]\s*", "", line)
            # 剥离 "- " / "* " / "• " / "· " 项目符号前缀
            line = re.sub(r"^\s*[\-\*•·]\s*", "", line)
            # 去除可能出现的成对引号
            line = line.strip().strip('"').strip("'").strip()

            if line:
                results.append(line)
        return results

    def _normalize(self, documents: list[str]) -> list[str]:
        """长度过滤 + 截断 + 去重 + 数量上限。

        - 过短的段落（< ``_MIN_DOC_LENGTH`` 字符）丢弃，避免标题/噪声进 embedding
        - 过长的段落截断到 ``_MAX_DOC_LENGTH`` 字符，节省 embedding 配额
        - 重复段落只保留第一次出现，保持插入顺序
        - 最终长度严格 ``≤ self._max_documents``
        """
        seen: dict[str, None] = {}
        for doc in documents:
            cleaned = (doc or "").strip()
            if len(cleaned) < _MIN_DOC_LENGTH:
                continue
            if len(cleaned) > _MAX_DOC_LENGTH:
                cleaned = cleaned[:_MAX_DOC_LENGTH]
            if cleaned in seen:
                continue
            seen[cleaned] = None
            if len(seen) >= self._max_documents:
                break
        return list(seen.keys())

    async def _embed_documents(self, documents: list[str]) -> list[list[float]]:
        """对每段假设文档独立 embedding。

        单段失败 / 返回空向量时跳过该段，不影响其它段；全部失败时返回 ``[]``。
        采用串行调用：HyDE 段数最多 3，串行避免对 embedding 服务造成额外峰值
        压力，且语义清晰（任一段失败可立即记录 warning 而不影响其它段）。
        """
        vectors: list[list[float]] = []
        for idx, doc in enumerate(documents):
            try:
                result = await self._embedding_service.embed_query(doc)
            except Exception as exc:  # noqa: BLE001 - 单段失败不应中断整体
                logger.warning(
                    "HyDE 段落 #%d embedding 失败: %s",
                    idx,
                    exc,
                )
                continue

            dense = getattr(result, "dense_vector", None) or []
            if dense:
                vectors.append(list(dense))
            else:
                logger.warning("HyDE 段落 #%d embedding 返回空向量，已忽略", idx)
        return vectors
