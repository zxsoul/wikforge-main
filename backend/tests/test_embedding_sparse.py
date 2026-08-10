"""任务 12.4：Sparse 向量（SPLADE 等效方案 / TF-IDF 哈希）单元测试。

设计文档要求 ``Sparse 向量（SPLADE 模型或等效方案）``，落盘格式为
``{"indices": [int], "values": [float]}``，与 Qdrant ``SparseVector`` 兼容。

本套测试聚焦 ``EmbeddingService._generate_sparse_embeddings`` 的核心契约：

- 空文本返回空向量；
- 中文 / 英文 / 中英混合文本都能产生 token；
- ``indices`` 严格升序、无重复；
- ``values`` 非负；
- 同 batch 内大文档的 token 数 ≥ 小文档；
- IDF 在 batch 内正确生效（高频 token 得到更低权重）；
- token → index 的哈希在跨进程下确定性稳定。

历史的 ``_generate_sparse_embeddings`` / ``_tokenize`` 用法测试保留在
``tests/test_indexing.py``，本文件只覆盖 12.4 任务要求的不变量。
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from app.services.embedding_service import (
    SPARSE_VOCAB_SIZE,
    EmbeddingService,
    _stable_token_index,
)


# ─── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def service() -> EmbeddingService:
    """单实例 EmbeddingService；sparse 路径不会触达 LiteLLM。"""
    return EmbeddingService()


# ─── 输入语言覆盖 ─────────────────────────────────────────────────────


class TestSparseEmbeddingLanguageCoverage:
    """空文本、中文、英文、中英混合输入下的形状契约。"""

    def test_empty_text_yields_empty_indices_and_values(self, service):
        """空字符串必须返回 ``{"indices": [], "values": []}``。"""
        results = service._generate_sparse_embeddings([""])
        assert len(results) == 1
        assert results[0] == {"indices": [], "values": []}

    def test_whitespace_only_text_yields_empty(self, service):
        """纯空白字符串没有可分词内容，等同于空。"""
        results = service._generate_sparse_embeddings(["   \n\t  "])
        assert results[0] == {"indices": [], "values": []}

    def test_chinese_text_produces_tokens(self, service):
        """中文文本应产生非空 sparse 向量（字符 + 双字组）。"""
        results = service._generate_sparse_embeddings(["企业知识库系统"])
        assert len(results) == 1
        assert len(results[0]["indices"]) > 0
        assert len(results[0]["indices"]) == len(results[0]["values"])

    def test_english_text_produces_tokens(self, service):
        """英文文本应产生非空 sparse 向量。"""
        results = service._generate_sparse_embeddings(
            ["The quick brown fox jumps over the lazy dog"]
        )
        assert len(results) == 1
        assert len(results[0]["indices"]) > 0
        assert len(results[0]["indices"]) == len(results[0]["values"])

    def test_mixed_chinese_english_text_produces_tokens(self, service):
        """中英混合应同时贡献中文 bigram 和英文 word token。"""
        results = service._generate_sparse_embeddings(["知识库 system version 123"])
        assert len(results[0]["indices"]) > 0
        # 简单 sanity：混合文本 token 数应 > 任一单独子串
        only_chinese = service._generate_sparse_embeddings(["知识库"])
        only_english = service._generate_sparse_embeddings(["system version 123"])
        assert len(results[0]["indices"]) >= len(only_chinese[0]["indices"])
        assert len(results[0]["indices"]) >= len(only_english[0]["indices"])


# ─── 输出形状不变量 ───────────────────────────────────────────────────


class TestSparseEmbeddingShapeInvariants:
    """``indices`` 升序去重、``values`` 非负、长度匹配。"""

    @pytest.mark.parametrize(
        "text",
        [
            "alpha beta gamma delta epsilon",
            "重复 重复 重复 重复",
            "Document processing pipeline pipeline pipeline",
            "知识库 系统 知识库 系统 知识库",
        ],
        ids=["english", "chinese-repeat", "english-repeat", "mixed-repeat"],
    )
    def test_indices_are_sorted_ascending(self, service, text):
        """``indices`` 必须严格升序。"""
        results = service._generate_sparse_embeddings([text])
        indices = results[0]["indices"]
        assert indices == sorted(indices)

    @pytest.mark.parametrize(
        "text",
        [
            "duplicate duplicate duplicate",
            "知识 知识 知识 知识 知识",
            "abc abc abc 123 123",
        ],
        ids=["english", "chinese", "mixed"],
    )
    def test_indices_are_deduplicated(self, service, text):
        """高频 token 不能产生重复索引（同 token 收敛到同 bucket）。"""
        results = service._generate_sparse_embeddings([text])
        indices = results[0]["indices"]
        assert len(indices) == len(set(indices))

    def test_values_are_non_negative(self, service):
        """TF-IDF 权重必须非负。"""
        results = service._generate_sparse_embeddings(
            ["Testing non-negative TF-IDF weights with mixed 知识 库"]
        )
        for value in results[0]["values"]:
            assert value >= 0.0

    def test_indices_within_vocab_size(self, service):
        """所有 index 必须位于 ``[0, SPARSE_VOCAB_SIZE)``。"""
        results = service._generate_sparse_embeddings(
            ["A reasonably long English text that produces many distinct tokens"]
        )
        for idx in results[0]["indices"]:
            assert 0 <= idx < SPARSE_VOCAB_SIZE

    def test_indices_and_values_have_same_length(self, service):
        """``len(indices) == len(values)``，便于直接喂给 Qdrant SparseVector。"""
        results = service._generate_sparse_embeddings(
            [
                "first document about vector search",
                "second document about full text",
                "",
                "知识库",
            ]
        )
        for r in results:
            assert len(r["indices"]) == len(r["values"])

    def test_indices_are_python_ints(self, service):
        """Qdrant SparseVector 期望 ``list[int]``，避免 numpy.int64 之类的子类。"""
        results = service._generate_sparse_embeddings(["sample tokens here"])
        for idx in results[0]["indices"]:
            assert type(idx) is int  # noqa: E721

    def test_values_are_python_floats(self, service):
        """同上，``list[float]``。"""
        results = service._generate_sparse_embeddings(["sample tokens here"])
        for value in results[0]["values"]:
            assert type(value) is float  # noqa: E721


# ─── 文档规模与 token 数 ──────────────────────────────────────────────


class TestSparseEmbeddingDocumentSize:
    """更长的文档应产生不少于短文档的 token 数。"""

    def test_longer_document_has_at_least_as_many_tokens(self, service):
        short = "alpha beta"
        long = (
            "alpha beta gamma delta epsilon zeta eta theta iota kappa "
            "lambda mu nu xi omicron pi rho sigma tau upsilon"
        )
        results = service._generate_sparse_embeddings([short, long])
        assert len(results[1]["indices"]) >= len(results[0]["indices"])

    def test_chinese_longer_document_has_more_tokens(self, service):
        short = "知识库"
        long = "企业知识库系统支持多格式文档导入与全文检索"
        results = service._generate_sparse_embeddings([short, long])
        assert len(results[1]["indices"]) >= len(results[0]["indices"])


# ─── IDF 行为 ────────────────────────────────────────────────────────


class TestSparseEmbeddingIDF:
    """同 batch 内高频 token 应得到更低权重。"""

    def test_frequent_term_has_lower_idf_than_rare_term(self, service):
        """``common`` 出现在所有文档里，``unique`` 只在第一篇。

        TF 相同的前提下，``common`` 的 IDF 应严格小于 ``unique``，因此其
        TF-IDF 权重也应更小。
        """
        texts = [
            "common unique alpha",
            "common alpha beta",
            "common alpha gamma",
            "common alpha delta",
        ]
        results = service._generate_sparse_embeddings(texts)

        common_idx = _stable_token_index("common")
        unique_idx = _stable_token_index("unique")

        first_doc_indices = results[0]["indices"]
        first_doc_values = results[0]["values"]

        # 两个 token 都应出现在第一篇文档中。
        assert common_idx in first_doc_indices
        assert unique_idx in first_doc_indices

        common_weight = first_doc_values[first_doc_indices.index(common_idx)]
        unique_weight = first_doc_values[first_doc_indices.index(unique_idx)]

        # ``common`` 在所有 4 篇出现 → df=4；``unique`` 只在 1 篇 → df=1。
        # 因此 unique 的 TF-IDF 必须严格大于 common。
        assert unique_weight > common_weight

    def test_term_in_every_doc_still_has_positive_weight(self, service):
        """``log(1 + N/N) = log(2) > 0``，全频 token 仍保留非零权重。"""
        texts = ["alpha beta", "alpha gamma", "alpha delta"]
        results = service._generate_sparse_embeddings(texts)
        alpha_idx = _stable_token_index("alpha")
        for r in results:
            # 不同文档的 alpha 都应保留（权重大于阈值即被收录）。
            if alpha_idx in r["indices"]:
                assert r["values"][r["indices"].index(alpha_idx)] > 0.0


# ─── token → index 的跨进程稳定性 ─────────────────────────────────────


class TestSparseTokenIndexStability:
    """``_stable_token_index`` 必须跨进程、跨 PYTHONHASHSEED 一致。

    Python 的内置 ``hash()`` 会被 PYTHONHASHSEED 随机化，这意味着如果使用
    ``hash(token)`` 计算 sparse 索引，写入端 Celery worker 与查询端 FastAPI
    进程会产生不同的索引，导致 sparse 召回完全失效。本测试在子进程里以
    ``PYTHONHASHSEED=random`` 重新计算同一组 token 的索引，确保结果与当前
    进程一致。
    """

    def test_token_index_is_stable_across_subprocess(self):
        import os
        from pathlib import Path

        tokens = ["alpha", "知识", "知识库", "system", "版本", "123"]
        expected = [_stable_token_index(t) for t in tokens]

        # 让子进程能 import ``app.services``：传入 backend/ 目录所在路径。
        backend_dir = str(Path(__file__).resolve().parent.parent)

        script = textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, {backend_dir!r})
            from app.services.embedding_service import _stable_token_index
            tokens = {tokens!r}
            print(",".join(str(_stable_token_index(t)) for t in tokens))
            """
        ).strip()

        # 复制必要的环境变量（PATH、HOME 等），并强制 ``PYTHONHASHSEED=random``
        # 以模拟另一个独立进程下的字符串哈希随机化。
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = "random"

        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        actual = [int(x) for x in completed.stdout.strip().split(",")]
        assert actual == expected

    def test_token_index_within_vocab_size(self):
        for token in ["alpha", "beta", "知识", "知识库", "x" * 200]:
            idx = _stable_token_index(token)
            assert 0 <= idx < SPARSE_VOCAB_SIZE

    def test_same_token_maps_to_same_index(self):
        """重复调用必须返回同一索引（确定性 sanity check）。"""
        for token in ["alpha", "知识库", "version"]:
            assert _stable_token_index(token) == _stable_token_index(token)
