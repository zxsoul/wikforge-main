"""MinIO 客户端（boto3 S3 兼容接口）。

设计参考：design.md 的 MinIO 存储部分（任务 6.1）。

提供：
- :func:`get_minio_client`：单例 boto3 S3 客户端
- :func:`ensure_bucket_exists`：检查/创建配置中的 bucket
- :func:`generate_presigned_get_url`：生成对象 GET 预签名 URL（任务 11.10
  审核详情页并排预览使用）
- :func:`reset_minio_client`：测试用，重置单例缓存
"""

from __future__ import annotations

import logging

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError

from app.core.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

# 任务 11.10 审核详情页：预签名 URL 默认有效期 10 分钟，覆盖前端单次审核停留
# 时长，又不会让链接长期可被泄露。可由调用方按需覆盖（例如批量审核器需要更长
# 时间）。
DEFAULT_PRESIGNED_URL_EXPIRES_IN = 600

_s3_client = None


def get_minio_client():
    """获取或创建 MinIO（S3 兼容）客户端单例。"""
    global _s3_client
    if _s3_client is None:
        protocol = "https" if settings.MINIO_SECURE else "http"
        endpoint_url = f"{protocol}://{settings.MINIO_ENDPOINT}"

        _s3_client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=settings.MINIO_ACCESS_KEY,
            aws_secret_access_key=settings.MINIO_SECRET_KEY,
            config=BotoConfig(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
            ),
            region_name="us-east-1",
        )
    return _s3_client


def ensure_bucket_exists() -> None:
    """确保配置的 bucket 存在，不存在则创建。"""
    client = get_minio_client()
    bucket = settings.MINIO_BUCKET
    try:
        client.head_bucket(Bucket=bucket)
    except ClientError:
        client.create_bucket(Bucket=bucket)


def reset_minio_client() -> None:
    """重置 MinIO 客户端单例缓存（仅用于测试）。"""
    global _s3_client
    _s3_client = None


def generate_presigned_get_url(
    storage_path: str,
    expires_in: int = DEFAULT_PRESIGNED_URL_EXPIRES_IN,
) -> str | None:
    """为存储中的对象生成 GET 预签名 URL。

    任务 11.10「文档并排预览 API」用该 URL 让前端审核详情页直接拉取原文件
    （PDF/DOCX 等）做并排展示，无需经过后端再代理一次下载。生成失败（例如
    MinIO 不可达 / 凭据错误）时返回 ``None``，调用方应当退化到一个备用
    URL（如 ``/api/documents/{id}/download``）。

    Args:
        storage_path: ``Document.storage_path`` 中保存的对象 key。
        expires_in: 链接有效期，单位秒，默认 ``DEFAULT_PRESIGNED_URL_EXPIRES_IN``。

    Returns:
        预签名 URL 字符串，或 ``None`` 表示生成失败。
    """
    if not storage_path:
        return None
    try:
        client = get_minio_client()
        url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.MINIO_BUCKET, "Key": storage_path},
            ExpiresIn=expires_in,
        )
        return url
    except (BotoCoreError, ClientError, ValueError) as exc:  # noqa: BLE001
        # MinIO 在审核流程里属于「锦上添花」依赖：拉不到原文件时仍要让前端能
        # 渲染解析后 Markdown，所以只记 WARNING，不抛。
        logger.warning(
            "Failed to generate presigned URL for %s: %s", storage_path, exc
        )
        return None
    except Exception as exc:  # noqa: BLE001 — defensive: third-party errors vary
        logger.warning(
            "Unexpected error generating presigned URL for %s: %s",
            storage_path,
            exc,
        )
        return None
