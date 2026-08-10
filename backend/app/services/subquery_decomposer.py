"""子查询分解服务（任务 15.3）。

需求 7.3：当用户查询包含 2 个及以上可独立回答的子问题时，搜索引擎应将查询
分解为不超过 5 个子查询，分别检索后对结果进行去重合并，返回合并后的结果集。

本模块提供 ``SubqueryDecomposer`` 单一职责组件：

- ``decompose(query)`` 统一入口，返回 ``0 ≤ len ≤ 5`` 条子查询
  - 单一问题查询 → LLM 返回空数组 ``[]`` → 本组件返回 ``[]``
  - 多子问题查询 → LLM 返回 N 条独立子查询 → 本组件去重 / 截断后返回
- 调用 ``LLMGateway`` 让模型先判断"是否多子问题查询"，再输出 JSON 数组
- 通过 ``asyncio.wait_for(timeout=2.0)`` 在 2 秒内强约束完成；超时即降级返回 ``[]``
- LLM 抛出异常 / 返回非法格式时同样降级为空列表（由上层 ``QueryEnhancer`` 决定如何
  与原始查询合并），符合需求 7.5 的"查询增强失败应静默降级"约束

为什么要有独立的 ``SubqueryDecomposer`` 模块？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- ``QueryEnhancer`` 负责协调 改写 / HyDE / 子查询分解 三种增强能力，并行调度+总超时；
- ``SubqueryDecomposer`` 仅聚焦"子查询分解"这一项，便于单独单测、独立替换实现
  （例如未来切换到本地小模型或基于规则的拆分），并保持与设计文档一致的
  "≤5 个子查询 / 2 秒超时"硬约束。

调用关系：``SearchService → QueryEnhancer → SubqueryDecomposer → LLMGateway``。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

from app.services.llm_gateway import LLMGateway, LLMGatewayError

logger = logging.getLogger(__name__)

# 单次分解整体超时时间（秒）。
#
# 需求 7.5 给出 5 秒总预算，``QueryEnhancer`` 会并行触发 改写 / HyDE / 分解。
# 其中 HyDE 通常最耗时（独占 3 秒预算），改写与分解各分配 2 秒预算，并发执行。
DECOMPOSE_TIMEOUT_SECONDS: float = 2.0

# 输出子查询数量上限。需求 7.3 要求"不超过 5 个"。
MAX_SUB_QUERIES: int = 5

# LLM 调用使用的 max_tokens 上限。5 个中等长度子查询足以容纳。
_DECOMPOSE_MAX_TOKENS: int = 512

# LLM 采样温度。低温度以稳定判断"是否需要分解"，避免随机将单一查询过度拆分。
_DECOMPOSE_TEMPERATURE: float = 0.3

# 单条子查询的最短长度（字符），过短的输出（如 LLM 误输出标点）会被丢弃。
_MIN_SUBQUERY_LENGTH: int = 2

# 系统提示词：明确要求 LLM 先判断是否多子问题，再输出 JSON 数组；单一问题输出 ``[]``。
_SYSTEM_PROMPT: str = (
    "你是一个专业的搜索查询分析助手。你的任务是判断用户的搜索查询是否包含多个可"
    "独立回答的子问题，并在需要时将其分解为多个子查询，用于分别检索后合并结果。\n\n"
    "判断与分解要求：\n"
    "1. 多子问题信号：包含「和」「以及」「同时」「分别」「比较」「区别」「与」"
    "等连接词、含有多个独立疑问词、或显式枚举多个独立主题\n"
    "2. 仅当查询确实包含 2 个及以上可独立回答的子问题时才分解；否则视为单一问题\n"
    "3. 单一问题（即使较长或包含修饰语）→ 直接输出空数组 []\n"
    "4. 多子问题 → 将每个独立问题改写为简洁、自然的中文搜索查询，互相独立、不重复\n"
    "5. 子查询数量不超过 5 个，每条子查询应能单独检索并得到有意义结果\n"
    "6. 不要复述原始查询本身，不要添加额外解释或修饰\n\n"
    '输出格式：严格输出 JSON 数组。\n'
    '- 单一问题查询：输出 []\n'
    '- 多子问题查询：输出 ["子查询1", "子查询2", ...]\n'
    "不要输出任何解释或额外文本，不要使用 Markdown 代码块。"
)


class SubqueryDecomposer:
    """LLM 驱动的子查询分解组件。

    使用方式::

        decomposer = SubqueryDecomposer(llm_gateway=gateway)
        sub_queries = await decomposer.decompose("Python 和 Java 的区别是什么")
        # sub_queries 长度在 0 到 5 之间；单一问题返回 []

    设计要点：

    - 输入 ``query`` 为空 / 仅空白时直接返回 ``[]``，**不调用** LLM
    - 通过 ``asyncio.wait_for`` 强制 2 秒内完成；超时返回 ``[]`` 并记录 warning
    - 任何 LLM 错误或解析错误都会被吞掉并降级为 ``[]``，不向上抛出，符合需求 7.5
      的"查询增强失败应静默降级"约束
    - LLM 输出 ``[]`` 表示单一问题，本组件原样返回 ``[]``
    - 结果会去重、去空、移除与原始 query 完全相同的项（按 strip 后比较）
    - 仅返回 1 条子查询是无意义的（与原查询等价），此时同样视为单一问题返回 ``[]``
    """

    def __init__(
        self,
        llm_gateway: LLMGateway | None = None,
        timeout: float = DECOMPOSE_TIMEOUT_SECONDS,
        max_sub_queries: int = MAX_SUB_QUERIES,
    ) -> None:
        """初始化子查询分解器。

        Args:
            llm_gateway: LLM 网关。为 ``None`` 时构造默认实例（生产路径走 LiteLLM）。
                测试中应注入 mock，避免真实网络调用。
            timeout: 整体超时（秒）。默认 ``2.0`` 与改写一致。
            max_sub_queries: 输出子查询数量上限。默认 ``5`` 对应需求 7.3。
        """
        # LLM 网关本身的 timeout 设为与分解 timeout 一致，避免内部等待时间超过外层。
        self._llm = llm_gateway or LLMGateway(timeout=timeout)
        self._timeout = timeout
        self._max_sub_queries = max(0, max_sub_queries)

    async def decompose(self, query: str) -> list[str]:
        """识别多子问题查询并分解为不超过 ``max_sub_queries`` 个子查询。

        Args:
            query: 原始用户查询。空 / 仅空白会直接返回 ``[]`` 且不调用 LLM。

        Returns:
            子查询列表，长度 ``0 ≤ len ≤ max_sub_queries``。
            单一问题查询返回 ``[]``，表示无需分解。
            发生超时、LLM 错误、解析失败时统一返回 ``[]``。
        """
        # 输入校验：空字符串或纯空白直接短路，避免无谓的 LLM 调用与 token 消耗。
        if not query or not query.strip():
            return []

        if self._max_sub_queries == 0:
            return []

        try:
            response = await asyncio.wait_for(
                self._llm.complete(
                    prompt=self._build_user_prompt(query),
                    system_prompt=_SYSTEM_PROMPT,
                    temperature=_DECOMPOSE_TEMPERATURE,
                    max_tokens=_DECOMPOSE_MAX_TOKENS,
                ),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            # 超时是预期内的降级路径，记录 warning 即可，不向上抛。
            logger.warning(
                "子查询分解超时（%.1fs），降级返回空列表 query=%r",
                self._timeout,
                query,
            )
            return []
        except LLMGatewayError as exc:
            # LLM 网关错误（限流 / 鉴权 / 模型不可用等）也走降级。
            logger.warning("子查询分解 LLM 调用失败: %s, query=%r", exc, query)
            return []
        except Exception as exc:  # noqa: BLE001 - 兜底任何意外错误，保持服务可用
            logger.warning("子查询分解发生未知错误: %s, query=%r", exc, query)
            return []

        content = (response.content or "").strip()
        if not content:
            return []

        sub_queries = self._parse_sub_queries(content)
        normalized = self._normalize(sub_queries, original_query=query)

        # 仅 1 条子查询与原查询等价，没有分解价值，按"单一问题"处理返回空列表。
        # 需求 7.3 明确"包含 2 个及以上"才视为多子问题查询。
        if len(normalized) < 2:
            return []

        return normalized

    # ─── 内部辅助 ──────────────────────────────────────────────────

    def _build_user_prompt(self, query: str) -> str:
        """构建发送给 LLM 的用户消息。"""
        return (
            f"请分析以下搜索查询是否包含多个可独立回答的子问题。\n"
            f"如果是，请将其分解为最多 {self._max_sub_queries} 个子查询；"
            f"如果不是（即单一问题），请输出空数组 []。\n"
            f"严格按照系统提示中描述的 JSON 数组格式输出。\n\n"
            f"原始查询：{query}"
        )

    def _parse_sub_queries(self, content: str) -> list[str]:
        """解析 LLM 输出，优先按 JSON 数组解析，失败时退化到按行解析。

        Args:
            content: LLM 返回的原始文本

        Returns:
            原始子查询列表（未去重 / 未去除原查询）
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
                # 空数组 [] 走到这里会得到 []，正确表达"无需分解"。
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

    def _normalize(self, sub_queries: list[str], original_query: str) -> list[str]:
        """去重 + 去空 + 去除与原查询相同的项 + 截断到 max_sub_queries。

        保持首次出现顺序（dict 在 Python 3.7+ 保留插入顺序）。
        """
        original_norm = original_query.strip()
        seen: dict[str, None] = {}
        for sub_query in sub_queries:
            cleaned = (sub_query or "").strip()
            if len(cleaned) < _MIN_SUBQUERY_LENGTH:
                continue
            if cleaned == original_norm:
                # 与原查询完全相同的子查询没有检索价值，需求 7.4 由上层负责始终保留原查询，
                # 这里专注输出"补充子查询"。
                continue
            if cleaned in seen:
                continue
            seen[cleaned] = None
            if len(seen) >= self._max_sub_queries:
                break
        return list(seen.keys())
