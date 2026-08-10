"""IK Analyzer 远程词库分发路由（任务 13.7）。

OpenSearch 的 IK 插件支持 ``remote_ext_dict`` / ``remote_ext_stopwords``
配置项，指向一个 HTTP URL；插件每 60 秒发起 ``HEAD`` 请求，依据响应头中的
``Last-Modified`` / ``ETag`` 决定是否拉取新版本词典并热加载。

本路由暴露两个端点：

- ``GET  /api/ik-dict/{filename}``：返回 ``.dic`` 文件原始内容
- ``HEAD /api/ik-dict/{filename}``：仅返回头（IK 插件首选用法）

并支持条件请求：当客户端通过 ``If-Modified-Since`` 头携带上次拉取的时间，
本端会比对文件 mtime；未变更则返回 ``304 Not Modified``，避免无谓传输。

安全约束：
- ``filename`` 通过白名单（``custom_main.dic`` / ``custom_stopword.dic``）
  校验，禁止任意路径跳转（``..``、绝对路径）。
- 端点本身不要求鉴权 — IK 插件通常以匿名身份从内网拉取，且词库文件不含
  机密信息（仅是分词术语）。如部署在公网请通过反向代理限制源 IP。
"""

from __future__ import annotations

from email.utils import formatdate, parsedate_to_datetime
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException, Response

from app.services.dictionary_service import (
    IK_MAIN_DICT_FILE,
    IK_STOP_DICT_FILE,
)

router = APIRouter(prefix="/api/ik-dict", tags=["ik-dict"])


# 白名单：任何不在此集合里的 filename 一律 404，杜绝路径穿越。
_ALLOWED_FILES = frozenset({IK_MAIN_DICT_FILE, IK_STOP_DICT_FILE})


def _resolve_dict_path(filename: str) -> Path:
    """把 ``filename`` 解析为 ``IK_DICT_DIR`` 下的合法路径。

    任何非白名单文件名直接抛 404；白名单文件不存在时返回路径但调用方需自行
    处理 ``Path.exists()``（缺省返回空文件 200 — IK 插件能正常处理）。
    """
    if filename not in _ALLOWED_FILES:
        raise HTTPException(status_code=404, detail="Unknown IK dict file")
    # 模块级 IK_DICT_DIR 在 dictionary_service 加载时绑定，这里同步取最新值。
    from app.services import dictionary_service

    return dictionary_service.IK_DICT_DIR / filename


def _build_headers(path: Path) -> dict[str, str]:
    """构造 ``Last-Modified`` 等响应头。文件不存在时回落到当前时间。"""
    if path.exists():
        mtime = path.stat().st_mtime
    else:
        # 文件还没生成时，使用 0（Epoch）作为 Last-Modified，让 IK 拉到空内容；
        # 后续真正同步后 mtime 推进，IK 自然命中变更。
        mtime = 0.0
    last_modified = formatdate(mtime, usegmt=True)
    return {
        "Last-Modified": last_modified,
        # 简单 ETag：文件 mtime + 字节数。IK 插件主要看 Last-Modified，
        # 但部分代理 / 客户端会用 ETag 二次校验，这里给一个稳定值。
        "ETag": (
            f'W/"{int(mtime)}-{path.stat().st_size if path.exists() else 0}"'
        ),
        "Cache-Control": "no-cache",
        # IK 插件按行解析，明确声明字符集避免容器默认 latin-1。
        "Content-Type": "text/plain; charset=utf-8",
    }


def _is_not_modified(path: Path, if_modified_since: str | None) -> bool:
    """根据 ``If-Modified-Since`` 头判断是否应回 304。"""
    if not if_modified_since or not path.exists():
        return False
    try:
        client_dt = parsedate_to_datetime(if_modified_since)
    except (TypeError, ValueError):
        return False
    if client_dt is None:
        return False
    # HTTP-date 精度为秒，把 mtime 也按秒截断。文件 mtime ≤ 客户端时间表示
    # 自客户端上次拉取以来无变更。
    file_mtime = int(path.stat().st_mtime)
    client_mtime = int(client_dt.timestamp())
    return file_mtime <= client_mtime


@router.get("/{filename}")
async def get_ik_dict(
    filename: str,
    if_modified_since: str | None = Header(default=None, alias="If-Modified-Since"),
) -> Response:
    """返回 IK 远程词库内容（GET）。"""
    path = _resolve_dict_path(filename)

    if _is_not_modified(path, if_modified_since):
        return Response(status_code=304, headers=_build_headers(path))

    # 文件不存在时返回空内容 200 + Last-Modified=Epoch，IK 插件能正常处理。
    if not path.exists():
        return Response(content="", headers=_build_headers(path))

    try:
        content = path.read_bytes()
    except OSError as e:
        # 文件存在但读不出（权限 / 磁盘故障）— 给客户端 503 而不是 200 空内容，
        # 避免 IK 误把"读失败"当作"词典已清空"。
        raise HTTPException(
            status_code=503,
            detail=f"Failed to read IK dict file: {e}",
        ) from e

    return Response(content=content, headers=_build_headers(path))


@router.head("/{filename}")
async def head_ik_dict(
    filename: str,
    if_modified_since: str | None = Header(default=None, alias="If-Modified-Since"),
) -> Response:
    """返回 IK 远程词库的元数据头（HEAD）。

    IK 插件在轮询时优先发 HEAD，仅当 ``Last-Modified`` 改变才发 GET 拉取
    完整内容。FastAPI 不会自动从 GET 派生 HEAD，故显式声明。
    """
    path = _resolve_dict_path(filename)
    if _is_not_modified(path, if_modified_since):
        return Response(status_code=304, headers=_build_headers(path))
    return Response(status_code=200, headers=_build_headers(path))


# 暴露给 OpenAPI 仅作展示；运行期通过 ``main.py`` 注册 router。
__all__ = ["router"]
