"""Query Enhancer: Rewrite, HyDE, Sub-query Decomposition with timeout degradation.
查询增强器：支持查询重写、HyDE（假设性文档嵌入）以及带超时降级机制的子查询分解
Implements:
- Query rewriting: LLM generates up to 5 semantic variants (2s timeout)
查询重写（Query rewriting）：由大语言模型（LLM）生成最多 5 个语义变体（即换几种不同的问法），该操作设有 2 秒超时限制。
- HyDE: LLM generates 1-3 hypothetical documents, embed them for supplementary vector search
- Sub-query decomposition: detect multi-part queries, split into ≤5 sub-queries
- Original query always included in final results
- Overall timeout: 5 seconds, fall back to original query on timeout
全局超时与降级：整个增强过程设有 5 秒的全局超时限制。如果超时，系统将自动降级，直接回退使用用户的原始查询进行搜索，防止用户等待过久。
- Config: enable_rewrite, enable_hyde, enable_decomposition (all default True)
"""
# 标准库
import asyncio
import logging
from dataclasses import dataclass, field
# 第三方库
from app.services.embedding_service import EmbeddingService
from app.services.llm_gateway import LLMGateway, LLMGatewayError

logger = logging.getLogger(__name__)

# Timeout constants
REWRITE_TIMEOUT = 2.0  # seconds for query rewriting
HYDE_TIMEOUT = 3.0  # seconds for HyDE generation
DECOMPOSE_TIMEOUT = 2.0  # seconds for sub-query decomposition
OVERALL_TIMEOUT = 5.0  # seconds for entire enhancement process

# Limits
MAX_REWRITE_VARIANTS = 5
MAX_HYDE_DOCUMENTS = 3
MAX_SUB_QUERIES = 5


@dataclass
class QueryEnhancerConfig:
    """Configuration for query enhancement features.

    Attributes:
        enable_rewrite: Whether to enable query rewriting (default True)
        enable_hyde: Whether to enable HyDE (default True)
        enable_decomposition: Whether to enable sub-query decomposition (default True)
    """

    enable_rewrite: bool = True
    enable_hyde: bool = True
    enable_decomposition: bool = True

    @classmethod
    def from_settings(
        cls, settings: "object | None" = None
    ) -> "QueryEnhancerConfig":
        """从应用 ``Settings`` 构造查询增强配置（任务 15.6 / 需求 7.6）。

        将三个环境变量驱动的开关映射到 :class:`QueryEnhancerConfig`：

        - ``QUERY_ENHANCEMENT_ENABLE_REWRITE`` → :attr:`enable_rewrite`
        - ``QUERY_ENHANCEMENT_ENABLE_HYDE`` → :attr:`enable_hyde`
        - ``QUERY_ENHANCEMENT_ENABLE_DECOMPOSITION`` → :attr:`enable_decomposition`

        Args:
            settings: 可选注入，便于测试覆写。默认调用
                :func:`app.core.config.get_settings`。

        Returns:
            根据当前 Settings 构造的 :class:`QueryEnhancerConfig`，三个开关
            均默认 True。
        """
        # 局部导入避免 ``query_enhancer`` 在被测试模块单独导入时强依赖 Settings
        if settings is None:
            from app.core.config import get_settings

            settings = get_settings()
        return cls(
            enable_rewrite=bool(
                getattr(settings, "QUERY_ENHANCEMENT_ENABLE_REWRITE", True)
            ),
            #getattr 从对象中取属性值，如果没有就按默认值
            enable_hyde=bool(
                getattr(settings, "QUERY_ENHANCEMENT_ENABLE_HYDE", True)
            ),
            enable_decomposition=bool(
                getattr(settings, "QUERY_ENHANCEMENT_ENABLE_DECOMPOSITION", True)
            ),
        )


@dataclass
class EnhancedQuery:
    """查询增强结果。

    需求 7.4：查询增强过程必须始终包含原始查询，确保返回结果包含与原始查询直接匹配
    的内容。本数据类承担"原始查询保留"的契约：

    - ``original`` 字段恒定为构造时传入的原始查询字符串（即使为空字符串也保留）
    - ``all_text_queries`` 字段是供 BM25/向量检索使用的"文本查询合集"，**第一项始终为
      原始查询**，其后追加去重后的改写变体与子查询；即使所有 LLM 子模块失败/超时，
      该列表也至少包含原始查询一条
    - 提供别名属性 ``original_query`` / ``rewrites`` / ``hypothetical_embeddings``
      与任务描述（15.4）保持命名一致；原字段名保留以兼容既有调用方与单测

    Attributes:
        original: 原始用户查询，始终非空时直接保留；空字符串场景同样原样保留
        variants: 语义改写变体（去重后 ≤ 5），不包含原始查询
        hyde_embeddings: 假设文档生成的 dense 向量（≤ 3）
        sub_queries: 多子问题分解后的子查询（去重后 ≤ 5），不包含原始查询
        all_text_queries: 用于 BM25/向量检索的文本查询合集，去重后包含原始查询 +
            改写变体 + 子查询；首项恒定为 ``original``
    """

    original: str
    variants: list[str] = field(default_factory=list)
    hyde_embeddings: list[list[float]] = field(default_factory=list)
    sub_queries: list[str] = field(default_factory=list)
    all_text_queries: list[str] = field(default_factory=list)

    # ── 任务 15.4 中文别名属性 ─────────────────────────────────────
    # 任务描述里使用 ``original_query`` / ``rewrites`` / ``hypothetical_embeddings``
    # 三个名字；为了不破坏既有 ``original`` / ``variants`` / ``hyde_embeddings`` 调用
    # 方与已通过的 45 个单测，这里以只读 property 形式提供等价访问。

    @property
    def original_query(self) -> str:
        """``original`` 的别名，对齐任务 15.4 描述中的字段命名。"""
        return self.original

    @property
    def rewrites(self) -> list[str]:
        """``variants`` 的别名，对齐任务 15.4 描述中的字段命名。"""
        return self.variants

    @property
    def hypothetical_embeddings(self) -> list[list[float]]:
        """``hyde_embeddings`` 的别名，对齐任务 15.4 描述中的字段命名。"""
        return self.hyde_embeddings


class QueryEnhancer:
    """Query enhancement service with rewrite, HyDE, and decomposition.

    Enhances user queries to improve search recall through:
    1. Query rewriting: generates semantic variants
    2. HyDE: generates hypothetical document embeddings
    3. Sub-query decomposition: splits complex queries into sub-queries

    All enhancements are subject to timeout constraints and can be
    individually enabled/disabled via configuration.
    """

    def __init__(
        self,
        llm_gateway: LLMGateway | None = None,
        embedding_service: EmbeddingService | None = None,
        config: QueryEnhancerConfig | None = None,
    ):
        """Initialize the QueryEnhancer.

        Args:
            llm_gateway: LLM gateway for generating rewrites and hypothetical docs.
                        If None, a default instance with 2s timeout will be created.
            embedding_service: Service for embedding hypothetical documents.
                             If None, a default instance will be created.
            config: Enhancement configuration. If None, all features enabled.
        """
        self._llm = llm_gateway or LLMGateway(timeout=REWRITE_TIMEOUT)
        self._embedding_service = embedding_service or EmbeddingService()
        self._config = config or QueryEnhancerConfig()

    @property
    def config(self) -> QueryEnhancerConfig:
        """Get the current configuration."""
        return self._config

    @config.setter
    def config(self, value: QueryEnhancerConfig) -> None:
        """Set the configuration."""
        self._config = value

    async def enhance(self, query: str) -> EnhancedQuery:
        """Enhance a user query with rewrites, HyDE, and decomposition.

        需求 7.4：原始查询始终作为必选检索条件被保留——无论各子模块成功、超时还是
        抛异常，返回的 ``EnhancedQuery.all_text_queries`` 第一项恒为 ``query``。

        增强任务并发执行，受 5 秒总超时约束；超时或异常时降级返回仅含原始查询的结果。

        Args:
            query: 原始用户搜索查询

        Returns:
            ``EnhancedQuery``：``original`` 与 ``all_text_queries`` 始终包含原始查询。
        """
        # 空查询特殊处理：保留原始字符串，但不调用 LLM；``all_text_queries`` 不
        # 强行注入空串，避免下游误以为存在一条"空文本查询"。
        if not query or not query.strip():
            return EnhancedQuery(
                original=query,
                all_text_queries=[],
            )
        ## 降低兜底策略，通过异步时间限制判断问题预处理是否超时，超时采取兜底策略，直接返回原问题
        try:
            result = await asyncio.wait_for(
                self._enhance_internal(query),
                timeout=OVERALL_TIMEOUT,
            )
        except asyncio.TimeoutError:
            # 需求 7.5：整体超时降级。日志中带 ``event=query_enhancement_timeout``
            # 结构化关键字，便于日志聚合系统按事件名告警与统计。
            logger.warning(
                "event=query_enhancement_timeout timeout=%.1fs "
                "Query enhancement timed out, falling back to original query",
                OVERALL_TIMEOUT,
            )
            # 即便整体超时，也至少保留原始查询，满足需求 7.4 + 7.5
            return EnhancedQuery(
                original=query,
                all_text_queries=[query],
            )
        except Exception as e:
            # 需求 7.5：异常同样降级到仅原始查询，且打 ``event=query_enhancement_failed``
            logger.warning(
                "event=query_enhancement_failed error=%s "
                "Query enhancement failed, falling back to original query",
                e,
            )
            return EnhancedQuery(
                original=query,
                all_text_queries=[query],
            )

        # 内部成功路径：补齐 all_text_queries（去重，原始查询置首）
        result.all_text_queries = self._build_all_text_queries(
            original=query,
            rewrites=result.variants,
            sub_queries=result.sub_queries,
        )
        return result

    async def _enhance_internal(self, query: str) -> EnhancedQuery:
        """Internal enhancement logic that runs within the overall timeout.

        Runs enabled enhancement tasks concurrently.

        Args:
            query: The original user query

        Returns:
            EnhancedQuery with all successful enhancements
        """
        result = EnhancedQuery(original=query)

        # Build list of tasks to run concurrently
        tasks: dict[str, asyncio.Task] = {}

        if self._config.enable_rewrite:
            tasks["rewrite"] = asyncio.create_task(self._rewrite_query(query))

        if self._config.enable_hyde:
            tasks["hyde"] = asyncio.create_task(self._generate_hyde(query))

        if self._config.enable_decomposition:
            tasks["decompose"] = asyncio.create_task(self._decompose_query(query))

        if not tasks:
            return result

        # Wait for all tasks, handling individual failures gracefully
        done, pending = await asyncio.wait(
            tasks.values(),
            return_when=asyncio.ALL_COMPLETED,
        )

        # Cancel any pending tasks (shouldn't happen since we wait for all)
        for task in pending:
            task.cancel()

        # Collect results
        if "rewrite" in tasks:
            task = tasks["rewrite"]
            if task.done() and not task.cancelled() and task.exception() is None:
                result.variants = task.result()
            elif task.exception() is not None:
                logger.warning(f"Query rewrite failed: {task.exception()}")

        if "hyde" in tasks:
            task = tasks["hyde"]
            if task.done() and not task.cancelled() and task.exception() is None:
                result.hyde_embeddings = task.result()
            elif task.exception() is not None:
                logger.warning(f"HyDE generation failed: {task.exception()}")

        if "decompose" in tasks:
            task = tasks["decompose"]
            if task.done() and not task.cancelled() and task.exception() is None:
                result.sub_queries = task.result()
            elif task.exception() is not None:
                logger.warning(f"Query decomposition failed: {task.exception()}")

        return result

    async def _rewrite_query(self, query: str) -> list[str]:
        """Generate semantic rewrite variants of the query.

        Uses LLM to generate up to 5 semantically equivalent queries
        within a 2-second timeout.

        Args:
            query: Original user query

        Returns:
            List of rewrite variants (up to 5)
        """
        prompt = (
            f"请为以下搜索查询生成最多5个语义相关的改写变体。"
            f"每个变体应该表达相同的搜索意图但使用不同的措辞或角度。"
            f"只输出改写结果，每行一个，不要编号，不要解释。\n\n"
            f"原始查询：{query}"
        )

        system_prompt = (
            "你是一个搜索查询改写助手。你的任务是生成语义等价的查询变体，"
            "帮助提高搜索召回率。保持简洁，直接输出改写结果。"
        )

        try:
            response = await asyncio.wait_for(
                self._llm.complete(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=0.7,
                    max_tokens=512,
                ),
                timeout=REWRITE_TIMEOUT,
            )

            # Parse response: each line is a variant
            variants = self._parse_variants(response.content, MAX_REWRITE_VARIANTS)
            return variants

        except asyncio.TimeoutError:
            logger.warning(f"Query rewrite timed out after {REWRITE_TIMEOUT}s")
            return []
        except LLMGatewayError as e:
            logger.warning(f"Query rewrite LLM call failed: {e}")
            return []

    async def _generate_hyde(self, query: str) -> list[list[float]]:
        """Generate HyDE (Hypothetical Document Embeddings).

        Uses LLM to generate 1-3 hypothetical documents that would answer
        the query, then embeds them for supplementary vector search.

        Args:
            query: Original user query

        Returns:
            List of dense embedding vectors from hypothetical documents
        """
        prompt = (
            f"请根据以下搜索查询，生成1到3个假设性文档段落。"
            f"这些段落应该是回答该查询的理想文档内容。"
            f"每个段落100-200字，用空行分隔不同段落。\n\n"
            f"查询：{query}"
        )

        system_prompt = (
            "你是一个文档生成助手。根据用户的搜索查询，生成假设性的文档段落，"
            "这些段落代表了用户期望找到的理想答案内容。保持专业和准确。"
        )

        try:
            response = await asyncio.wait_for(
                self._llm.complete(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=0.5,
                    max_tokens=1024,
                ),
                timeout=HYDE_TIMEOUT,
            )

            # Parse hypothetical documents
            hypothetical_docs = self._parse_hypothetical_documents(
                response.content, MAX_HYDE_DOCUMENTS
            )

            if not hypothetical_docs:
                return []

            # Embed each hypothetical document
            embeddings = []
            for doc in hypothetical_docs:
                embedding_result = await self._embedding_service.embed_query(doc)
                if embedding_result.dense_vector:
                    embeddings.append(embedding_result.dense_vector)

            return embeddings

        except asyncio.TimeoutError:
            logger.warning(f"HyDE generation timed out after {HYDE_TIMEOUT}s")
            return []
        except LLMGatewayError as e:
            logger.warning(f"HyDE LLM call failed: {e}")
            return []
        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}")
            return []

    async def _decompose_query(self, query: str) -> list[str]:
        """Decompose a complex query into sub-queries.

        Detects multi-part queries and splits them into up to 5
        independent sub-queries for separate retrieval.

        Args:
            query: Original user query

        Returns:
            List of sub-queries (up to 5). Empty if query is simple.
        """
        prompt = (
            f"分析以下查询是否包含多个可独立回答的子问题。"
            f"如果是，请将其分解为不超过5个子查询，每行一个。"
            f"如果查询是单一问题，只输出'无需分解'。\n\n"
            f"查询：{query}"
        )

        system_prompt = (
            "你是一个查询分析助手。你的任务是判断用户查询是否包含多个独立的子问题，"
            "如果是则将其分解。只有当查询确实包含2个及以上可独立回答的子问题时才进行分解。"
            "直接输出分解结果，不要解释。"
        )

        try:
            response = await asyncio.wait_for(
                self._llm.complete(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=0.3,
                    max_tokens=512,
                ),
                timeout=DECOMPOSE_TIMEOUT,
            )

            # Check if decomposition is needed
            content = response.content.strip()
            if "无需分解" in content or not content:
                return []

            # Parse sub-queries
            sub_queries = self._parse_variants(content, MAX_SUB_QUERIES)
            # Only return if we got at least 2 sub-queries (otherwise not worth decomposing)
            if len(sub_queries) < 2:
                return []

            return sub_queries

        except asyncio.TimeoutError:
            logger.warning(f"Query decomposition timed out after {DECOMPOSE_TIMEOUT}s")
            return []
        except LLMGatewayError as e:
            logger.warning(f"Query decomposition LLM call failed: {e}")
            return []

    @staticmethod
    def _build_all_text_queries(
        original: str,
        rewrites: list[str],
        sub_queries: list[str],
    ) -> list[str]:
        """构建用于 BM25/向量检索的文本查询合集。

        需求 7.4 的核心实现：

        - 首项**恒为** ``original``（即使原始查询恰好出现在改写或子查询中也以原始
          查询的"原始文本"为准，不被替换）
        - 之后顺序追加去重后的改写变体与子查询
        - 与 ``original`` 完全相同（按 ``strip()`` 比较）的项视为重复，跳过
        - 各子列表内部已由各自服务做过去重，这里再次合并去重以兼容外部直接构造
          ``EnhancedQuery`` 的场景

        Args:
            original: 原始查询文本
            rewrites: 改写变体列表（不包含原始查询）
            sub_queries: 子查询列表（不包含原始查询）

        Returns:
            首项为原始查询的去重文本查询合集；当 ``original`` 为空白时返回空列表。
        """
        original_stripped = original.strip() if original else ""
        if not original_stripped:
            return []

        # 使用 dict 保持插入顺序去重（Python 3.7+ 保证）
        seen: dict[str, None] = {original: None}
        for candidate in (*rewrites, *sub_queries):
            if not candidate:
                continue
            cleaned = candidate.strip()
            if not cleaned:
                continue
            # 与原始查询去除首尾空白后等价的项视为重复
            if cleaned == original_stripped:
                continue
            if cleaned in seen:
                continue
            seen[cleaned] = None
        return list(seen.keys())

    def _parse_variants(self, content: str, max_count: int) -> list[str]:
        """Parse LLM response into a list of query variants.

        Handles various formats: numbered lists, bullet points, plain lines.

        Args:
            content: Raw LLM response text
            max_count: Maximum number of variants to return

        Returns:
            List of parsed, cleaned variants
        """
        if not content:
            return []

        lines = content.strip().split("\n")
        variants = []

        for line in lines:
            # Clean up the line: remove numbering, bullets, etc.
            cleaned = line.strip()
            if not cleaned:
                continue
            # 第一步：去掉数字序号（比如 "1.", "2)", "3、"）
            # Remove common prefixes: "1.", "1)", "-", "*", "•"
            for prefix in [".", ")", "、"]:
                if len(cleaned) > 2 and cleaned[0].isdigit() and prefix in cleaned[:4]:
                    idx = cleaned.index(prefix)
                    cleaned = cleaned[idx + 1:].strip()
                    break
            # 第二步：去掉项目符号（比如 "-", "*", "•", "·"）
            if cleaned.startswith(("-", "*", "•", "·")):
                cleaned = cleaned[1:].strip()
            # 第三步：过滤太短的内容（少于2个字符的废话不要）
            # Skip empty or too short results
            if len(cleaned) < 2:
                continue
            # 第四步：过滤大模型的“废话/自我解释”
            # Skip if it's the same as original or a meta-comment
            if cleaned.startswith(("注", "说明", "解释")):
                continue

            variants.append(cleaned)

            if len(variants) >= max_count:
                break

        return variants

    def _parse_hypothetical_documents(
        self, content: str, max_count: int
    ) -> list[str]:
        """Parse LLM response into hypothetical document paragraphs.

        Splits by double newlines (paragraph breaks).

        Args:
            content: Raw LLM response text
            max_count: Maximum number of documents to return

        Returns:
            List of hypothetical document paragraphs
        """
        if not content:
            return []

        # Split by double newlines (paragraph separator)
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        # If no double newlines, try single newlines with length threshold
        if len(paragraphs) <= 1 and "\n" in content:
            lines = content.strip().split("\n")
            # Group consecutive non-empty lines as paragraphs
            current_para = []
            paragraphs = []
            for line in lines:
                if line.strip():
                    current_para.append(line.strip())
                else:
                    if current_para:
                        paragraphs.append(" ".join(current_para))
                        current_para = []
            if current_para:
                paragraphs.append(" ".join(current_para))

        # Filter out very short paragraphs (likely headers or noise)
        documents = [p for p in paragraphs if len(p) >= 20]

        return documents[:max_count]


# ─── 模块级工厂 ────────────────────────────────────────────────────────


def build_query_enhancer(
    llm_gateway: LLMGateway | None = None,
    embedding_service: EmbeddingService | None = None,
    settings: "object | None" = None,
) -> QueryEnhancer:
    """从应用 ``Settings`` 构造 :class:`QueryEnhancer`（任务 15.6 / 需求 7.6）。

    依赖注入处（如 ``app/api/search.py`` 或 ``SearchService``）应通过本工厂
    创建增强器实例，确保 :class:`QueryEnhancerConfig` 的三个开关由环境变量
    ``QUERY_ENHANCEMENT_ENABLE_REWRITE`` / ``..._ENABLE_HYDE`` /
    ``..._ENABLE_DECOMPOSITION`` 驱动；而不是用 ``QueryEnhancer()`` 默认构造。

    Args:
        llm_gateway: 可选注入；为 None 时由 :class:`QueryEnhancer` 自身构造默认实例。
        embedding_service: 可选注入；为 None 时由 :class:`QueryEnhancer` 自身构造默认实例。
        settings: 可选注入；用于测试覆盖。默认调用 :func:`app.core.config.get_settings`。

    Returns:
        按当前 Settings 配置开关的 :class:`QueryEnhancer` 实例。
    """
    config = QueryEnhancerConfig.from_settings(settings)
    return QueryEnhancer(
        llm_gateway=llm_gateway,
        embedding_service=embedding_service,
        config=config,
    )
