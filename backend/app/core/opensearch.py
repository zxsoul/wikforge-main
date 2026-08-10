"""OpenSearch 客户端与 ``chunks`` 索引管理。

任务 12.2：配置 OpenSearch 客户端和 Index 创建（chunks，IK 分词器映射）。

提供：

- ``get_opensearch_client``：单例客户端，使用
  ``OPENSEARCH_HOST`` / ``OPENSEARCH_PORT`` 等配置连接。
- ``reset_opensearch_client`` / ``close_opensearch_client``：
  释放或关闭单例，便于测试与运行期清理。
- ``ensure_index_exists``：幂等创建 ``chunks`` 索引，使用 IK 分词器
  （``ik_max_word`` / ``ik_smart``）；当容器未安装 IK 插件时优雅降级
  到 ``standard`` 分词器并打印警告。
- ``delete_index``：删除 ``chunks`` 索引（仅供测试与重置使用）。

索引字段映射严格遵循 design.md《OpenSearch 索引模型》。
"""

from __future__ import annotations

import logging
from typing import Any

from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.exceptions import RequestError, TransportError

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────

# 索引名称——用于存储 chunk 的全文检索文档。
INDEX_NAME = "chunks"

# IK 分词器名称（如未安装 IK 插件，会回退到 standard）。
_IK_MAX_WORD = "ik_max_word"
_IK_SMART = "ik_smart"

_opensearch_client: OpenSearch | None = None


# ─── Index mappings ──────────────────────────────────────────────────


def _build_index_body(
    *,
    text_analyzer: str,
    search_analyzer: str | None,
) -> dict[str, Any]:
    """根据所选分词器生成索引创建 body。

    Args:
        text_analyzer: ``content`` / ``title_chain`` 字段写入时使用的分词器。
        search_analyzer: ``content`` 字段查询时使用的分词器；为 ``None`` 时
            不显式设置 ``search_analyzer``（用于无 IK 插件场景，避免引用
            未注册的分析器）。

    Returns:
        合法的 OpenSearch ``create index`` 请求体。
    """
    content_field: dict[str, Any] = {
        "type": "text",
        "analyzer": text_analyzer,
    }
    if search_analyzer is not None:
        content_field["search_analyzer"] = search_analyzer

    return {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "document_id": {"type": "keyword"},
                "space_id": {"type": "keyword"},
                "content": content_field,
                "title_chain": {
                    "type": "text",
                    "analyzer": text_analyzer,
                },
                "source_file": {"type": "keyword"},
                "page_number": {"type": "integer"},
                "chunk_index": {"type": "integer"},
                "allowed_user_ids": {"type": "keyword"},
                "created_at": {"type": "date"},
            },
        },
    }


# 主映射：写入用 ik_max_word（更细粒度），查询用 ik_smart（更精确召回）。
INDEX_MAPPING: dict[str, Any] = _build_index_body(
    text_analyzer=_IK_MAX_WORD,
    search_analyzer=_IK_SMART,
)

# 降级映射：未安装 IK 插件时使用 OpenSearch 内置 standard 分词器。
INDEX_MAPPING_FALLBACK: dict[str, Any] = _build_index_body(
    text_analyzer="standard",
    search_analyzer=None,
)


# ─── Client lifecycle ────────────────────────────────────────────────


def get_opensearch_client() -> OpenSearch:
    """获取（或惰性创建）OpenSearch 客户端单例。

    使用 ``settings.OPENSEARCH_HOST`` / ``OPENSEARCH_PORT`` /
    ``OPENSEARCH_USER`` / ``OPENSEARCH_PASSWORD`` 配置 HTTPS 基础认证。
    Docker Compose 默认证书为自签，因此 ``verify_certs`` 关闭。
    """
    global _opensearch_client
    if _opensearch_client is None:
        settings = get_settings()
        # 生产 / 自签证书时由 settings.OPENSEARCH_USE_SSL 开关 HTTPS;
        # 开发部署 (plugins.security.disabled=true) 走 HTTP, 不传 http_auth。
        use_ssl = bool(settings.OPENSEARCH_USE_SSL)
        client_kwargs: dict = {
            "hosts": [
                {
                    "host": settings.OPENSEARCH_HOST,
                    "port": settings.OPENSEARCH_PORT,
                }
            ],
            "use_ssl": use_ssl,
            "connection_class": RequestsHttpConnection,
            "timeout": 30,
        }
        if use_ssl:
            client_kwargs["verify_certs"] = bool(settings.OPENSEARCH_VERIFY_CERTS)
            client_kwargs["ssl_show_warn"] = False
            client_kwargs["http_auth"] = (
                settings.OPENSEARCH_USER,
                settings.OPENSEARCH_PASSWORD,
            )
        _opensearch_client = OpenSearch(**client_kwargs)
    return _opensearch_client


def reset_opensearch_client() -> None:
    """丢弃缓存的客户端实例，主要供测试使用。

    与 ``close_opensearch_client`` 不同的是不会调用 ``client.close()``，
    适合 mock 场景下重置单例以便重新构造。
    """
    global _opensearch_client
    _opensearch_client = None


def close_opensearch_client() -> None:
    """关闭并清空 OpenSearch 客户端连接。"""
    global _opensearch_client
    if _opensearch_client is not None:
        try:
            _opensearch_client.close()
        except Exception:  # 关闭失败不应阻塞调用方
            logger.debug("OpenSearch 客户端关闭失败，已忽略", exc_info=True)
        _opensearch_client = None


# ─── Index management ───────────────────────────────────────────────


def _is_unknown_analyzer_error(exc: BaseException) -> bool:
    """判断异常是否表示『分词器未注册』，用于决定是否降级到 standard。

    OpenSearch 缺少 IK 插件时，``indices.create`` 会返回 400，
    错误信息形如 ``Unknown analyzer type [ik_max_word]`` 或
    ``analyzer [ik_smart] not found``。这两类信息都包含 ``analyzer``
    或 ``tokenizer`` 关键字。
    """
    if isinstance(exc, RequestError):
        info = getattr(exc, "info", None)
        if isinstance(info, dict):
            text = str(info).lower()
            if "analyzer" in text or "tokenizer" in text:
                return True
    message = str(exc).lower()
    return ("analyzer" in message or "tokenizer" in message) and (
        _IK_MAX_WORD in message or _IK_SMART in message or "unknown" in message or "not found" in message
    )


def ensure_index_exists() -> None:
    """幂等地确保 ``chunks`` 索引存在。

    流程：
    1. 若索引已存在，直接返回；
    2. 否则使用 IK 分词器创建索引；
    3. 若集群未安装 IK 插件（``Unknown analyzer`` / ``not found``），
       打印警告并退化到 ``standard`` 分词器再次尝试创建。

    重复调用是安全的——这是文档处理管线启动入口的入口前置条件。
    """
    client = get_opensearch_client()

    if client.indices.exists(index=INDEX_NAME):
        logger.debug("索引 '%s' 已存在，跳过创建", INDEX_NAME)
        return

    logger.info("正在创建索引 '%s'（IK 分词器）", INDEX_NAME)
    try:
        client.indices.create(index=INDEX_NAME, body=INDEX_MAPPING)
        logger.info("索引 '%s' 创建成功（IK 分词器）", INDEX_NAME)
        return
    except (RequestError, TransportError) as exc:
        if not _is_unknown_analyzer_error(exc):
            logger.error("创建索引 '%s' 失败：%s", INDEX_NAME, exc)
            raise
        logger.warning(
            "未检测到 IK 分词器插件，降级到 standard 分词器：%s", exc
        )
    except Exception as exc:  # noqa: BLE001 — 兼容部分客户端抛裸异常
        if not _is_unknown_analyzer_error(exc):
            logger.error("创建索引 '%s' 失败：%s", INDEX_NAME, exc)
            raise
        logger.warning(
            "未检测到 IK 分词器插件，降级到 standard 分词器：%s", exc
        )

    # 降级路径：使用 standard 分词器重试。
    try:
        client.indices.create(index=INDEX_NAME, body=INDEX_MAPPING_FALLBACK)
        logger.info("索引 '%s' 创建成功（standard 分词器，降级）", INDEX_NAME)
    except Exception as fallback_error:
        logger.error(
            "降级创建索引 '%s' 仍然失败：%s", INDEX_NAME, fallback_error
        )
        raise


def delete_index() -> None:
    """删除 ``chunks`` 索引。仅供测试与重置场景使用。

    若索引不存在则静默忽略，保持幂等。
    """
    client = get_opensearch_client()
    try:
        if client.indices.exists(index=INDEX_NAME):
            client.indices.delete(index=INDEX_NAME)
            logger.info("索引 '%s' 已删除", INDEX_NAME)
        else:
            logger.debug("索引 '%s' 不存在，无需删除", INDEX_NAME)
    except Exception as exc:
        logger.error("删除索引 '%s' 失败：%s", INDEX_NAME, exc)
        raise
