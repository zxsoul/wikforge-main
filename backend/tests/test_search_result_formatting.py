"""搜索结果格式化（任务 14.8）单元测试。

覆盖范围：
- ``SearchService._clamp_score``：分数夹紧到 [0, 1]，含 NaN / 非数值兜底
- ``SearchService._format_results``：来源信息字段完整性
- ``SearchService._generate_highlight``：高亮片段长度、关键词窗口选择、
  多关键词命中、首/尾命中位置、空查询与无命中回退、``<mark>`` 包裹

对应需求：Requirements 6.4 —— 在最终结果中返回每个文档块的
相关性分数（0.0 至 1.0）、来源文档信息（文档标题/ID/块索引/页码）
和不超过 200 字符的高亮匹配片段。
"""

from __future__ import annotations

import math

import pytest

from app.services.search_service import (
    HIGHLIGHT_MARK_CLOSE,
    HIGHLIGHT_MARK_OPEN,
    HIGHLIGHT_MAX_CHARS,
    SearchHit,
    SearchService,
)


# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def service() -> SearchService:
    """构造一个不带任何外部依赖的 SearchService 实例。

    格式化与高亮逻辑是纯函数，不需要 embedding/检索后端。
    """

    class _StubEmbedding:
        async def embed_query(self, *_args, **_kwargs):  # pragma: no cover
            raise AssertionError("embed_query should not be called in formatting tests")

    return SearchService(embedding_service=_StubEmbedding())


def _make_hit(
    *,
    chunk_id: str = "chunk-1",
    document_id: str = "doc-1",
    chunk_index: int = 3,
    title_chain: str = "第一章 > 1.1 节",
    source_file: str = "report.pdf",
    page_number: int = 7,
    content: str = "示例内容",
    score: float = 0.5,
) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        document_id=document_id,
        space_id="space-1",
        chunk_index=chunk_index,
        title_chain=title_chain,
        source_file=source_file,
        page_number=page_number,
        content=content,
        score=score,
    )


# ─── 分数夹紧 ─────────────────────────────────────────────────────────


class TestClampScore:
    """覆盖 ``SearchService._clamp_score`` 的边界与异常输入。"""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            (0.5, 0.5),
            (1.0, 1.0),
            (0.0, 0.0),
            (1.7, 1.0),  # 超过上界
            (-0.3, 0.0),  # 低于下界
            (10.0, 1.0),  # 远超上界
            (-100.0, 0.0),  # 远低下界
        ],
    )
    def test_clamp_within_unit_interval(self, raw, expected):
        assert SearchService._clamp_score(raw) == expected

    def test_clamp_rounds_to_4_decimals(self):
        assert SearchService._clamp_score(0.123456789) == 0.1235

    def test_clamp_handles_nan(self):
        assert SearchService._clamp_score(float("nan")) == 0.0

    def test_clamp_handles_inf(self):
        assert SearchService._clamp_score(math.inf) == 1.0
        assert SearchService._clamp_score(-math.inf) == 0.0

    def test_clamp_handles_non_numeric(self):
        assert SearchService._clamp_score("not-a-number") == 0.0
        assert SearchService._clamp_score(None) == 0.0

    def test_format_results_clamps_scores(self, service):
        candidates = [
            _make_hit(chunk_id="a", score=2.5, content="hello world"),
            _make_hit(chunk_id="b", score=-1.0, content="hello world"),
            _make_hit(chunk_id="c", score=0.42, content="hello world"),
        ]
        results = service._format_results(candidates, query="hello")
        assert results[0].score == 1.0
        assert results[1].score == 0.0
        assert results[2].score == 0.42


# ─── 来源信息字段完整性 ───────────────────────────────────────────────


class TestSourceMetadata:
    """覆盖 ``_format_results`` 输出的来源字段。"""

    def test_all_source_fields_preserved(self, service):
        hit = _make_hit(
            chunk_id="ck-9",
            document_id="doc-99",
            chunk_index=5,
            title_chain="第三章 > 3.2 数据模型",
            source_file="设计文档.docx",
            page_number=12,
            content="向量数据库存储 Dense 与 Sparse 向量",
            score=0.7,
        )
        [result] = service._format_results([hit], query="向量")

        assert result.chunk_id == "ck-9"
        assert result.document_id == "doc-99"
        assert result.chunk_index == 5
        assert result.title_chain == "第三章 > 3.2 数据模型"
        assert result.source_file == "设计文档.docx"
        assert result.page_number == 12

    def test_format_results_empty_input(self, service):
        assert service._format_results([], query="hello") == []

    def test_format_results_default_page_number(self, service):
        hit = _make_hit(page_number=0, content="正文")
        [result] = service._format_results([hit], query="正文")
        # page_number 缺省时直接透传，不会变成 None
        assert result.page_number == 0


# ─── 高亮长度 ─────────────────────────────────────────────────────────


class TestHighlightLength:
    """高亮片段的长度上限与短内容直通。"""

    def test_short_content_returned_as_is_when_no_match(self, service):
        content = "短文本"
        assert service._generate_highlight(content, query="无关词") == content

    def test_short_content_with_match_wraps_marks(self, service):
        content = "搜索引擎"
        out = service._generate_highlight(content, query="搜索")
        assert out.startswith(f"{HIGHLIGHT_MARK_OPEN}搜索{HIGHLIGHT_MARK_CLOSE}")
        # 即使加上标签也应小于 HIGHLIGHT_MAX_CHARS
        assert len(out) <= HIGHLIGHT_MAX_CHARS

    def test_highlight_never_exceeds_200(self, service):
        content = "x" * 1000
        out = service._generate_highlight(content, query="x")
        assert len(out) <= HIGHLIGHT_MAX_CHARS

    def test_highlight_with_dense_matches_truncates_safely(self, service):
        # 整段文本都是关键词，每对 <mark> 都会消耗预算
        content = "搜索" * 200  # 400 字符
        out = service._generate_highlight(content, query="搜索")
        assert len(out) <= HIGHLIGHT_MAX_CHARS
        # 不应该出现未闭合的 <mark>
        assert out.count(HIGHLIGHT_MARK_OPEN) == out.count(HIGHLIGHT_MARK_CLOSE)


# ─── 命中位置窗口选择 ─────────────────────────────────────────────────


class TestHighlightWindow:
    """覆盖首字命中、末尾命中、中间命中三种关键场景。"""

    def test_match_at_beginning(self, service):
        content = "machine learning " + ("filler text " * 50)
        out = service._generate_highlight(content, query="machine")
        assert f"{HIGHLIGHT_MARK_OPEN}machine{HIGHLIGHT_MARK_CLOSE}" in out
        # 首字命中时窗口应从开头开始
        assert out.startswith(HIGHLIGHT_MARK_OPEN)

    def test_match_at_middle(self, service):
        prefix = "前缀文本" * 60  # 240 字符
        suffix = "后缀文本" * 60  # 240 字符
        content = prefix + "向量数据库" + suffix
        out = service._generate_highlight(content, query="向量数据库")
        assert f"{HIGHLIGHT_MARK_OPEN}向量数据" in out
        assert len(out) <= HIGHLIGHT_MAX_CHARS

    def test_match_at_end(self, service):
        content = ("padding text " * 50) + " final keyword"
        out = service._generate_highlight(content, query="keyword")
        assert f"{HIGHLIGHT_MARK_OPEN}keyword{HIGHLIGHT_MARK_CLOSE}" in out
        assert len(out) <= HIGHLIGHT_MAX_CHARS

    def test_window_prefers_higher_density_region(self, service):
        # 左侧只有 1 个命中，右侧有 3 个紧邻命中
        # 窗口选择应当落在右侧密集区
        left = "A" * 300 + " match " + "A" * 300
        right = "match match match"
        content = left + right
        out = service._generate_highlight(content, query="match")
        # 至少包含 2 个 mark 包裹（因为右侧有 3 个 match 紧邻）
        assert out.count(HIGHLIGHT_MARK_OPEN) >= 2


# ─── 多关键词高亮 ─────────────────────────────────────────────────────


class TestMultipleKeywords:
    """覆盖多关键词同时命中的高亮策略。"""

    def test_multiple_english_terms(self, service):
        content = "Hybrid search combines BM25 and dense vector retrieval"
        out = service._generate_highlight(content, query="BM25 dense")
        assert f"{HIGHLIGHT_MARK_OPEN}BM25{HIGHLIGHT_MARK_CLOSE}" in out
        # 'dense' 也应该被命中（大小写不敏感）
        assert HIGHLIGHT_MARK_OPEN + "dense" in out.lower()

    def test_multiple_chinese_terms(self, service):
        content = "复合搜索引擎包含 BM25、Dense 向量与 Sparse 向量"
        out = service._generate_highlight(content, query="向量 搜索")
        # 中文 2-gram 的高亮命中至少包含「向量」
        assert HIGHLIGHT_MARK_OPEN in out
        assert HIGHLIGHT_MARK_CLOSE in out
        # 多关键词均匹配：搜索 与 向量 都应该出现
        assert "向量" in out
        assert "搜" in out

    def test_overlapping_matches_merged(self, service):
        # 「搜索」与「索引」共享「索」字，命中区间会重叠 → 应当合并成一个 mark
        content = "搜索索引技术"
        out = service._generate_highlight(content, query="搜索 索引")
        # 不应出现嵌套或相邻 <mark>，重叠区间合并后只有一对
        assert out.count(HIGHLIGHT_MARK_OPEN) == 1
        assert out.count(HIGHLIGHT_MARK_CLOSE) == 1


# ─── 空 query / 无命中回退 ────────────────────────────────────────────


class TestFallbackBehavior:
    """空 query / 无命中时回退到 chunk 开头。"""

    def test_empty_query_returns_prefix(self, service):
        content = "abcdef" * 100  # 600 字符
        out = service._generate_highlight(content, query="")
        assert len(out) == HIGHLIGHT_MAX_CHARS
        assert HIGHLIGHT_MARK_OPEN not in out
        assert out == content[:HIGHLIGHT_MAX_CHARS]

    def test_whitespace_query_returns_prefix(self, service):
        content = "abcdef" * 100
        out = service._generate_highlight(content, query="   \n\t  ")
        assert len(out) == HIGHLIGHT_MAX_CHARS
        assert HIGHLIGHT_MARK_OPEN not in out

    def test_no_match_returns_prefix(self, service):
        content = "完全不相关的内容" * 60
        out = service._generate_highlight(content, query="machine")
        assert len(out) <= HIGHLIGHT_MAX_CHARS
        assert HIGHLIGHT_MARK_OPEN not in out
        # 回退到开头
        assert out == content[:HIGHLIGHT_MAX_CHARS]

    def test_empty_content(self, service):
        assert service._generate_highlight("", query="anything") == ""


# ─── 关键词抽取辅助函数 ────────────────────────────────────────────────


class TestQueryTermExtraction:
    """直接覆盖 ``_extract_query_terms`` 的边界。"""

    def test_extract_english_terms(self):
        terms = SearchService._extract_query_terms("BM25 dense Vector")
        assert "bm25" in terms
        assert "dense" in terms
        assert "vector" in terms

    def test_extract_chinese_2grams(self):
        terms = SearchService._extract_query_terms("搜索引擎")
        # 应包含「搜索」「索引」「引擎」三个 2-gram
        assert "搜索" in terms
        assert "索引" in terms
        assert "引擎" in terms

    def test_extract_single_chinese_char(self):
        terms = SearchService._extract_query_terms("书")
        assert terms == ["书"]

    def test_extract_dedup_preserves_order(self):
        terms = SearchService._extract_query_terms("搜索 搜索 BM25 BM25")
        # 去重但保留首次出现顺序
        assert terms.count("搜索") == 1
        assert terms.count("bm25") == 1
        assert terms.index("搜索") < terms.index("bm25")

    def test_extract_empty_query(self):
        assert SearchService._extract_query_terms("") == []
        assert SearchService._extract_query_terms("   ") == []
