"""查询改写服务（任务 15.1）。

需求 7.1：用户提交搜索查询后，搜索引擎应在 2 秒内对查询进行改写，
生成不超过 5 个语义相关的改写变体，并将改写变体与原始查询共同用于检索。

本模块提供 ``QueryRewriter`` 单一职责组件：

- ``rewrite(query)`` 统一入口，返回 ``≤ 5`` 条语义变体（不含原始查询本身）
- 调用 ``LLMGateway`` 让模型输出 JSON 数组，便于稳定解析；解析失败时退化到按行解析
- 通过 ``asyncio.wait_for(timeout=2.0)`` 在 2 秒内强约束完成；超时即降级返回 ``[]``
- LLM 抛出异常 / 返回非法格式时同样降级为空列表（由上层 ``QueryEnhancer`` 决定如何
  与原始查询合并）

为什么要有独立的 ``QueryRewriter`` 模块？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- ``QueryEnhancer`` 负责协调 改写 / HyDE / 子查询分解 三种增强能力，并行调度+总超时；
- ``QueryRewriter`` 仅聚焦"改写"这一项，便于单独单测、独立替换实现（例如未来切换到
  本地小模型或词典扩展），并保持与设计文档一致的"≤5 个变体 / 2 秒超时"硬约束。

调用关系：``SearchService → QueryEnhancer → QueryRewriter → LLMGateway``。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

from app.services.llm_gateway import LLMGateway, LLMGatewayError

logger = logging.getLogger(__name__)

# 单次改写整体超时时间（秒）。需求 7.1 要求 "2 秒内完成"。
REWRITE_TIMEOUT_SECONDS: float = 2.0

# 输出变体数量上限。需求 7.1 要求 "不超过 5 个"。
MAX_REWRITE_VARIANTS: int = 5

# LLM 调用使用的 max_tokens 上限。5 个中等长度查询变体足以容纳。
_REWRITE_MAX_TOKENS: int = 512

# LLM 采样温度。略高以鼓励措辞多样性，但仍受 system prompt 控制。
_REWRITE_TEMPERATURE: float = 0.7

# 单条变体的最短长度（字符），过短的输出（如 LLM 误输出标点）会被丢弃。
_MIN_VARIANT_LENGTH: int = 2

# 系统提示词：要求 LLM 输出 JSON 数组，避免自然语言混杂导致解析失败。
_SYSTEM_PROMPT: str = (
    "你是一个专业的搜索查询改写助手。你的任务是为用户的搜索查询生成语义等价的改写变体，"
    "用于提高检索召回率。\n\n"
    "改写要求：\n"
    "1. 保持与原查询完全相同的搜索意图\n"
    "2. 可以使用同义词替换、句式转换、近义表达\n"
    "3. 不要扩展或缩小查询范围，不要添加额外信息\n"
    "4. 每个变体都应是简洁、自然的中文搜索查询\n\n"
    '输出格式：严格输出 JSON 数组，例如 ["变体1", "变体2", "变体3"]。'
    "数量不超过 5 个，且不要包含原始查询。不要输出任何解释或额外文本。"
)


class QueryRewriter:
    """LLM 驱动的查询改写组件。

    使用方式::

        rewriter = QueryRewriter(llm_gateway=gateway)
        variants = await rewriter.rewrite("机器学习入门")
        # variants 长度在 0 到 5 之间，不包含原查询

    设计要点：

    - 输入 ``query`` 为空 / 仅空白时直接返回 ``[]``，**不调用** LLM
    - 通过 ``asyncio.wait_for`` 强制 2 秒内完成；超时返回 ``[]`` 并记录 warning
    - 任何 LLM 错误或解析错误都会被吞掉并降级为 ``[]``，不向上抛出，符合需求 7.5
      的"查询增强失败应静默降级"约束
    - 结果会去重、去空、移除与原始 query 完全相同的项（按 strip 后比较）
    """

    def __init__(
        self,
        llm_gateway: LLMGateway | None = None,
        timeout: float = REWRITE_TIMEOUT_SECONDS,
        max_variants: int = MAX_REWRITE_VARIANTS,
    ) -> None:
        """初始化查询改写器。

        Args:
            llm_gateway: LLM 网关。为 ``None`` 时构造默认实例（生产路径走 LiteLLM）。
                测试中应注入 mock，避免真实网络调用。
            timeout: 整体超时（秒）。默认 ``2.0`` 对应需求 7.1。
            max_variants: 输出变体数量上限。默认 ``5`` 对应需求 7.1。
        """
        # LLM 网关本身的 timeout 设为与改写 timeout 一致，避免内部等待时间超过外层。
        self._llm = llm_gateway or LLMGateway(timeout=timeout)
        self._timeout = timeout
        self._max_variants = max(0, max_variants)

    async def rewrite(self, query: str) -> list[str]:
        """生成不超过 ``max_variants`` 个语义改写变体。

        Args:
            query: 原始用户查询。空 / 仅空白会直接返回 ``[]`` 且不调用 LLM。

        Returns:
            语义变体列表，长度 ``0 ≤ len ≤ max_variants``。不包含原始查询本身。
            发生超时、LLM 错误、解析失败时统一返回 ``[]``。
        """
        # 输入校验：空字符串或纯空白直接短路，避免无谓的 LLM 调用与 token 消耗。
        if not query or not query.strip():
            return []

        if self._max_variants == 0:
            return []

        try:
            response = await asyncio.wait_for(
                self._llm.complete(
                    prompt=self._build_user_prompt(query),
                    system_prompt=_SYSTEM_PROMPT,
                    temperature=_REWRITE_TEMPERATURE,
                    max_tokens=_REWRITE_MAX_TOKENS,
                ),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            # 超时是预期内的降级路径，记录 warning 即可，不向上抛。
            logger.warning(
                "查询改写超时（%.1fs），降级返回空列表 query=%r",
                self._timeout,
                query,
            )
            return []
        except LLMGatewayError as exc:
            # LLM 网关错误（限流 / 鉴权 / 模型不可用等）也走降级。
            logger.warning("查询改写 LLM 调用失败: %s, query=%r", exc, query)
            return []
        except Exception as exc:  # noqa: BLE001 - 兜底任何意外错误，保持服务可用
            logger.warning("查询改写发生未知错误: %s, query=%r", exc, query)
            return []

        content = (response.content or "").strip()
        if not content:
            return []

        variants = self._parse_variants(content)
        return self._normalize(variants, original_query=query)

    # ─── 内部辅助 ──────────────────────────────────────────────────

    def _build_user_prompt(self, query: str) -> str:
        """构建发送给 LLM 的用户消息。"""
        return (
            f"请为以下搜索查询生成最多 {self._max_variants} 个语义改写变体。\n"
            f"严格按照系统提示中描述的 JSON 数组格式输出，不要包含原始查询。\n\n"
            f"原始查询：{query}"
        )

    def _parse_variants(self, content: str) -> list[str]:
        """解析 LLM 输出，优先按 JSON 数组解析，失败时退化到按行解析。

        Args:
            content: LLM 返回的原始文本

        Returns:
            原始变体列表（未去重 / 未去除原查询）
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

        # JSON 解析失败：退化为按行解析，剥离常见序号 / 项目符号前缀。
        return self._parse_lines(content)

    @staticmethod
    def _parse_lines(content: str) -> list[str]:
        """按行解析 LLM 输出，作为 JSON 解析失败时的兜底。"""
        results: list[str] = []
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            # 剥离形如 "1. " / "1) " / "1、" 的中英文序号前缀
            line = re.sub(r"^\s*\d+\s*[\.\)、]\s*", "", line)
            # 剥离 "- " / "* " / "• " / "· " 项目符号前缀
            line = re.sub(r"^\s*[\-\*•·]\s*", "", line)
            # 去除可能出现的成对引号
            line = line.strip().strip('"').strip("'").strip()

            if line:
                results.append(line)
        return results

    def _normalize(self, variants: list[str], original_query: str) -> list[str]:
        """去重 + 去空 + 去除与原查询相同的项 + 截断到 max_variants。

        保持首次出现顺序（dict 在 Python 3.7+ 保留插入顺序）。
        """
        original_norm = original_query.strip()
        seen: dict[str, None] = {}
        for variant in variants:
            cleaned = (variant or "").strip()
            if len(cleaned) < _MIN_VARIANT_LENGTH:
                continue
            if cleaned == original_norm:
                # 与原查询完全相同的变体没有检索价值，需求 7.4 由上层负责始终保留原查询，
                # 这里专注输出"补充变体"。
                continue
            if cleaned in seen:
                continue
            seen[cleaned] = None
            if len(seen) >= self._max_variants:
                break
        return list(seen.keys())
