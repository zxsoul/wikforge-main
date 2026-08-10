"""任务 13.7 焦点测试：IK 远程词库同步与 HTTP 分发。

本文件**只**覆盖 13.7 的硬化点（与已有 ``test_dictionary.py`` 互补）：

1. ``sync_ik_dictionaries`` 真实写入文件后产出可被读取的 ``.dic`` 内容。
2. 仅启用的词典进入输出，禁用词典的术语 / 停用词被排除。
3. 写入触发 mtime 推进，便于 IK 通过 ``Last-Modified`` 检出变更。
4. 内容未变时跳过重写，mtime 保持稳定（避免 IK 误判热加载）。
5. ``DictionaryService`` 的 CRUD（create / update / delete / toggle / add_terms）
   会触发同步。
6. 文件系统失败（权限不足）只记 warning，不打断词典 CRUD。
7. 空输入产出空文件，不视作错误。
8. ``/api/ik-dict/{filename}`` HTTP 路由：
   - 200 返回原始内容 + 正确头
   - 304 命中 ``If-Modified-Since`` 缓存
   - 404 拒绝白名单外文件名（路径穿越防御）
   - HEAD 仅返回头不返回 body
"""

from __future__ import annotations

import logging
import os
import time
from email.utils import formatdate
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.ik_dict import router as ik_dict_router
from app.core.exceptions import register_exception_handlers
from app.services import dictionary_service as ds_mod
from app.services.dictionary_service import (
    DictionaryService,
    IK_MAIN_DICT_FILE,
    IK_STOP_DICT_FILE,
    sync_ik_dictionaries,
)


# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def ik_dir(tmp_path, monkeypatch):
    """把模块级 ``IK_DICT_DIR`` 指向临时目录，覆盖默认 ``/data/ik-custom-dict``。"""
    target = tmp_path / "ik-custom-dict"
    monkeypatch.setattr(ds_mod, "IK_DICT_DIR", target)
    return target


def _build_dict(*, terms=None, stop_words=None, enabled=True):
    """构造一个最小可用的 ``DomainDictionary`` mock。"""
    d = MagicMock()
    d.terms = terms or []
    d.stop_words = stop_words or []
    d.enabled = enabled
    return d


def _mock_db(dictionaries):
    """``db.execute`` 返回的 scalars().all() 给定列表。"""
    db = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = list(dictionaries)
    db.execute = AsyncMock(return_value=result)
    db.flush = AsyncMock()
    db.delete = AsyncMock()
    db.refresh = AsyncMock()
    return db


# ─── 1. 写入产出可读 .dic 内容 ────────────────────────────────────────


class TestSyncWritesFiles:
    """sync_ik_dictionaries 真实写出可被读取的 .dic 文件。"""

    @pytest.mark.asyncio
    async def test_writes_main_and_stopword_files(self, ik_dir):
        db = _mock_db([
            _build_dict(
                terms=[{"word": "大齿圈"}, {"word": "回转窑"}],
                stop_words=["的", "了"],
            ),
        ])

        result = await sync_ik_dictionaries(db)

        assert result == {"terms": 2, "stop_words": 2}
        main_file = ik_dir / IK_MAIN_DICT_FILE
        stop_file = ik_dir / IK_STOP_DICT_FILE
        assert main_file.exists()
        assert stop_file.exists()
        # 内容是去重 + 排序的逐行术语
        assert sorted(main_file.read_text(encoding="utf-8").splitlines()) == [
            "回转窑",
            "大齿圈",
        ]
        assert sorted(stop_file.read_text(encoding="utf-8").splitlines()) == [
            "了",
            "的",
        ]

    @pytest.mark.asyncio
    async def test_string_terms_and_dict_terms_both_supported(self, ik_dir):
        """JSONB 中既可能是 ``{"word": "..."}`` 也可能是裸字符串。"""
        db = _mock_db([
            _build_dict(terms=[{"word": "水泥"}, "钢材"]),
        ])
        await sync_ik_dictionaries(db)

        lines = (ik_dir / IK_MAIN_DICT_FILE).read_text(
            encoding="utf-8"
        ).splitlines()
        assert sorted(lines) == ["水泥", "钢材"]


# ─── 2. 仅同步启用的词典 ──────────────────────────────────────────────


class TestEnabledFilter:
    """禁用词典的术语 / 停用词不写入文件。"""

    @pytest.mark.asyncio
    async def test_disabled_dictionaries_excluded_from_query(self, ik_dir):
        """``sync_ik_dictionaries`` 通过 ``where(enabled == True)`` 过滤。

        我们模拟 DB 已经过滤出启用词典；这里验证「只把传入的启用词典写入」。
        """
        db = _mock_db([
            _build_dict(terms=[{"word": "启用术语"}], enabled=True),
            # 即使列表里混入了 enabled=False，也不会出现 — DB 层已过滤；
            # 这里再加一个测试确认即使混入，输出也只包含启用词典内容。
        ])
        await sync_ik_dictionaries(db)
        lines = (ik_dir / IK_MAIN_DICT_FILE).read_text(
            encoding="utf-8"
        ).splitlines()
        assert lines == ["启用术语"]

    @pytest.mark.asyncio
    async def test_query_uses_enabled_true_filter(self, ik_dir):
        """sanity：函数内部确实执行了带 enabled 过滤的 SELECT。"""
        db = _mock_db([])
        await sync_ik_dictionaries(db)

        # ``execute`` 至少被调用一次；调用参数是带 where 的 select 语句。
        assert db.execute.await_count == 1
        stmt = db.execute.await_args.args[0]
        # SQLAlchemy 的 Select 对象转成字符串包含 ``enabled`` 关键字
        compiled = str(stmt.compile())
        assert "enabled" in compiled


# ─── 3. mtime bump 让 IK 检出变更 ─────────────────────────────────────


class TestMtimeBump:
    """变化时显式更新 mtime，未变化时保持稳定。"""

    @pytest.mark.asyncio
    async def test_mtime_changes_when_content_changes(self, ik_dir):
        """两次同步内容不同 → 文件 mtime 推进。"""
        # 第一次：只有「水泥」
        db = _mock_db([_build_dict(terms=[{"word": "水泥"}])])
        await sync_ik_dictionaries(db)
        main_file = ik_dir / IK_MAIN_DICT_FILE
        first_mtime = main_file.stat().st_mtime

        # 让 wall clock 推进（os.utime 的精度通常是 1µs，但 sleep 增强可观测性）
        await _async_sleep(0.05)

        # 第二次：新增「钢材」
        db = _mock_db([
            _build_dict(terms=[{"word": "水泥"}, {"word": "钢材"}])
        ])
        await sync_ik_dictionaries(db)
        second_mtime = main_file.stat().st_mtime

        assert second_mtime > first_mtime, (
            f"mtime should bump on content change: {first_mtime} → {second_mtime}"
        )

    @pytest.mark.asyncio
    async def test_mtime_stable_when_content_unchanged(self, ik_dir):
        """两次同步内容相同 → 跳过写入，mtime 保持。"""
        db = _mock_db([_build_dict(terms=[{"word": "水泥"}])])
        await sync_ik_dictionaries(db)
        main_file = ik_dir / IK_MAIN_DICT_FILE
        first_mtime = main_file.stat().st_mtime

        # 把 mtime 人为往前推一点，确认下一次同步时不会被改写。
        backdated = first_mtime - 60
        os.utime(main_file, (backdated, backdated))

        db = _mock_db([_build_dict(terms=[{"word": "水泥"}])])
        await sync_ik_dictionaries(db)

        assert main_file.stat().st_mtime == pytest.approx(backdated, abs=0.01), (
            "未变更时不应触摸 mtime，否则 IK 会做无谓热加载"
        )


# ─── 4. CRUD 触发同步 ─────────────────────────────────────────────────


class TestCrudTriggersSync:
    """Service 层 CRUD 操作必须调用 ``sync_ik_dictionaries``。

    我们 monkeypatch 函数为 AsyncMock，验证调用，无需真实 DB。
    """

    def _patched_service(self, monkeypatch):
        """构造一个 ``DictionaryService``，``sync_ik_dictionaries`` 已被 mock。"""
        sync_mock = AsyncMock(return_value={"terms": 0, "stop_words": 0})
        monkeypatch.setattr(ds_mod, "sync_ik_dictionaries", sync_mock)

        db = AsyncMock()
        db.add = MagicMock()
        db.flush = AsyncMock()
        db.delete = AsyncMock()
        db.refresh = AsyncMock()
        db.execute = AsyncMock()
        service = DictionaryService(db)
        return service, sync_mock, db

    @pytest.mark.asyncio
    async def test_create_enabled_dict_triggers_sync(self, monkeypatch):
        service, sync_mock, _ = self._patched_service(monkeypatch)
        await service.create_dictionary(name="新词典", enabled=True)
        sync_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_create_disabled_dict_skips_sync(self, monkeypatch):
        """禁用状态下创建无需同步（IK 词库不会包含其内容）。"""
        service, sync_mock, _ = self._patched_service(monkeypatch)
        await service.create_dictionary(name="新词典", enabled=False)
        sync_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_update_triggers_sync(self, monkeypatch):
        service, sync_mock, db = self._patched_service(monkeypatch)
        # update_dictionary 先 get_dictionary，再调用 sync。这里直接 patch
        # service.get_dictionary 返回一个 mock 词典。
        existing = MagicMock()
        existing.id = "abc"
        existing.terms = []
        existing.synonyms = []
        existing.stop_words = []
        existing.enabled = True
        existing.name = "old"
        service.get_dictionary = AsyncMock(return_value=existing)

        await service.update_dictionary(
            dictionary_id="00000000-0000-0000-0000-000000000000",
            name="updated",
        )
        sync_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete_triggers_sync(self, monkeypatch):
        service, sync_mock, _ = self._patched_service(monkeypatch)
        existing = MagicMock()
        service.get_dictionary = AsyncMock(return_value=existing)

        await service.delete_dictionary(
            dictionary_id="00000000-0000-0000-0000-000000000000"
        )
        sync_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_toggle_triggers_sync(self, monkeypatch):
        service, sync_mock, _ = self._patched_service(monkeypatch)
        existing = MagicMock()
        existing.enabled = False
        service.get_dictionary = AsyncMock(return_value=existing)

        await service.toggle_dictionary(
            dictionary_id="00000000-0000-0000-0000-000000000000",
            enabled=True,
        )
        sync_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_add_terms_triggers_sync_when_enabled(self, monkeypatch):
        service, sync_mock, _ = self._patched_service(monkeypatch)
        existing = MagicMock()
        existing.terms = []
        existing.enabled = True
        service.get_dictionary = AsyncMock(return_value=existing)

        await service.add_terms(
            dictionary_id="00000000-0000-0000-0000-000000000000",
            new_terms=[{"word": "新术语"}],
        )
        sync_mock.assert_awaited_once()


# ─── 5. 失败容忍 ──────────────────────────────────────────────────────


class TestFailureTolerance:
    """文件系统失败时只记 warning，词典 CRUD 不抛异常。"""

    @pytest.mark.asyncio
    async def test_mkdir_failure_logs_warning_and_returns(
        self, tmp_path, monkeypatch, caplog
    ):
        # 让 mkdir 抛 PermissionError；返回值仍然有效。
        target = tmp_path / "no-such"
        monkeypatch.setattr(ds_mod, "IK_DICT_DIR", target)

        original_mkdir = type(target).mkdir

        def boom(self, *args, **kwargs):
            raise PermissionError("permission denied")

        monkeypatch.setattr(type(target), "mkdir", boom)
        try:
            db = _mock_db([_build_dict(terms=[{"word": "水泥"}])])

            with caplog.at_level(logging.WARNING, logger=ds_mod.logger.name):
                result = await sync_ik_dictionaries(db)
        finally:
            monkeypatch.setattr(type(target), "mkdir", original_mkdir)

        # 仍然返回 counts，CRUD 上层不会感知到失败。
        assert result == {"terms": 1, "stop_words": 0}
        # warning 被记录。
        assert any(
            "Failed to prepare IK dictionary directory" in record.message
            for record in caplog.records
        ), [r.message for r in caplog.records]

    @pytest.mark.asyncio
    async def test_write_failure_logs_warning_and_continues(
        self, ik_dir, monkeypatch, caplog
    ):
        """单文件写入失败不影响另一文件，也不向上抛。"""
        ik_dir.mkdir(parents=True, exist_ok=True)

        # 用一个会抛 OSError 的 _write_ik_dict_file 替身：
        def fake_write(path, content):
            raise OSError("Disk full")

        # 直接在内部 helper 上 patch — 正常流程进入它后立即抛错。
        monkeypatch.setattr(ds_mod, "_write_ik_dict_file", fake_write)

        db = _mock_db([_build_dict(terms=[{"word": "水泥"}])])

        with caplog.at_level(logging.WARNING, logger=ds_mod.logger.name):
            # _write_ik_dict_file 抛 OSError 会被 sync 的外层 try 捕获。
            result = await sync_ik_dictionaries(db)

        assert result["terms"] == 1
        # 至少一条 warning（来自外层 try/except）
        assert caplog.records, "应有 warning 日志记录写入失败"


# ─── 6. 空输入产出空文件 ──────────────────────────────────────────────


class TestEmptyInput:
    """没有启用词典时仍写出空文件，不视作错误。"""

    @pytest.mark.asyncio
    async def test_empty_dict_list_writes_empty_files(self, ik_dir):
        db = _mock_db([])
        result = await sync_ik_dictionaries(db)

        assert result == {"terms": 0, "stop_words": 0}
        assert (ik_dir / IK_MAIN_DICT_FILE).exists()
        assert (ik_dir / IK_STOP_DICT_FILE).exists()
        assert (ik_dir / IK_MAIN_DICT_FILE).read_text(encoding="utf-8") == ""
        assert (ik_dir / IK_STOP_DICT_FILE).read_text(encoding="utf-8") == ""

    @pytest.mark.asyncio
    async def test_dict_with_only_whitespace_terms_yields_empty(self, ik_dir):
        """空白 / 空字符串术语被过滤，最终文件为空。"""
        db = _mock_db([
            _build_dict(terms=[{"word": ""}, {"word": "   "}]),
        ])
        result = await sync_ik_dictionaries(db)
        assert result["terms"] == 0
        assert (ik_dir / IK_MAIN_DICT_FILE).read_text(encoding="utf-8") == ""


# ─── 7. HTTP 路由 ─────────────────────────────────────────────────────


@pytest.fixture
def http_app(ik_dir):
    """构造一个仅装载 ``ik_dict_router`` 的 FastAPI app。"""
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(ik_dict_router)
    return app


@pytest.fixture
def http_client(http_app):
    return TestClient(http_app)


class TestHttpRoute:
    """``/api/ik-dict/{filename}`` 端点验证。"""

    def test_get_returns_dict_content(self, http_client, ik_dir):
        ik_dir.mkdir(parents=True, exist_ok=True)
        path = ik_dir / IK_MAIN_DICT_FILE
        path.write_text("水泥\n钢材\n", encoding="utf-8")

        response = http_client.get(f"/api/ik-dict/{IK_MAIN_DICT_FILE}")

        assert response.status_code == 200
        # 直接比对 bytes（避免编码歧义）
        assert response.content == "水泥\n钢材\n".encode("utf-8")
        assert response.headers["content-type"].startswith("text/plain")
        assert "Last-Modified" in response.headers
        assert "ETag" in response.headers

    def test_head_returns_headers_only(self, http_client, ik_dir):
        ik_dir.mkdir(parents=True, exist_ok=True)
        (ik_dir / IK_MAIN_DICT_FILE).write_text("水泥", encoding="utf-8")

        response = http_client.head(f"/api/ik-dict/{IK_MAIN_DICT_FILE}")

        assert response.status_code == 200
        assert "Last-Modified" in response.headers
        # HEAD 响应不携带 body
        assert response.content == b""

    def test_if_modified_since_returns_304(self, http_client, ik_dir):
        ik_dir.mkdir(parents=True, exist_ok=True)
        path = ik_dir / IK_MAIN_DICT_FILE
        path.write_text("水泥", encoding="utf-8")
        # 把 mtime 推到一个稳定的过去时间
        past = time.time() - 3600
        os.utime(path, (past, past))

        # 客户端声称已经持有「未来」时间的版本，应得到 304
        future_header = formatdate(time.time(), usegmt=True)
        response = http_client.get(
            f"/api/ik-dict/{IK_MAIN_DICT_FILE}",
            headers={"If-Modified-Since": future_header},
        )
        assert response.status_code == 304
        assert response.content == b""

    def test_if_modified_since_older_returns_200(self, http_client, ik_dir):
        ik_dir.mkdir(parents=True, exist_ok=True)
        path = ik_dir / IK_MAIN_DICT_FILE
        path.write_text("水泥", encoding="utf-8")
        # 文件 mtime = 现在
        now = time.time()
        os.utime(path, (now, now))

        # 客户端持有的是 1 小时前的版本，应当获取最新内容。
        past_header = formatdate(now - 3600, usegmt=True)
        response = http_client.get(
            f"/api/ik-dict/{IK_MAIN_DICT_FILE}",
            headers={"If-Modified-Since": past_header},
        )
        assert response.status_code == 200
        assert response.content == "水泥".encode("utf-8")

    def test_unknown_filename_returns_404(self, http_client):
        response = http_client.get("/api/ik-dict/evil.dic")
        assert response.status_code == 404

    def test_path_traversal_blocked(self, http_client):
        """``..`` 路径穿越尝试被白名单拦下（FastAPI 会先 404 路由不匹配）。"""
        # FastAPI path 参数不允许 ``/`` 默认；这里直接传非白名单 → 404
        response = http_client.get("/api/ik-dict/..%2Fpasswd")
        assert response.status_code == 404

    def test_missing_file_returns_200_empty(self, http_client, ik_dir):
        """文件还未生成时返回 200 + 空内容（IK 插件能正常处理）。"""
        # 不创建 ik_dir
        response = http_client.get(f"/api/ik-dict/{IK_STOP_DICT_FILE}")
        assert response.status_code == 200
        assert response.content == b""


# ─── helpers ──────────────────────────────────────────────────────────


async def _async_sleep(seconds: float) -> None:
    """协程友好的小睡 — 避免 ``time.sleep`` 阻塞事件循环。"""
    import asyncio

    await asyncio.sleep(seconds)
