"""Dictionary Toggle 启用/禁用语义集成测试（任务 13.8）。

覆盖 ``PATCH /api/admin/dictionaries/{id}/toggle`` 端点和 ``toggle_dictionary``
服务方法的关键语义：

1. **鉴权守门**：未登录 401，非管理员 403（与 ``admin_dictionaries`` 其它路由
   一致）。
2. **错误路径**：非法 UUID → 400；词典 ID 不存在 → 404。
3. **正确路径**：合法管理员调用返回 200 + 更新后的词典 payload，
   ``enabled`` 字段反映请求里的目标值。
4. **端到端 IK 同步**：``toggle_dictionary`` 必须调用 ``sync_ik_dictionaries``，
   并且：
   - 禁用（True → False）：被禁用词典的术语 / 停用词从 ``custom_main.dic`` /
     ``custom_stopword.dic`` 中移除（依靠 ``sync_ik_dictionaries`` 内部
     ``where(enabled == True)`` 过滤实现）。
   - 启用（False → True）：被启用词典的术语 / 停用词重新写入 IK 文件。
   - 同值（True → True / False → False）：仍触发一次 ``sync_ik_dictionaries``
     调用（no-op safety），保证幂等且不会让 IK 文件与 DB 状态偏离。

约束：
- HTTP 路由层用 TestClient + dependency_overrides 注入 mock DB / admin。
- 服务层端到端测试用 mock DB 但调用真实 ``sync_ik_dictionaries``，验证
  ``tmp_path`` 下的 ``.dic`` 文件内容。

与已有测试的关系：
- ``test_ik_sync.py``：覆盖 ``sync_ik_dictionaries`` 自身行为（写文件、mtime
  bump、enabled 过滤、CRUD 触发同步）。
- ``test_admin_dictionaries.py``：覆盖 CRUD（POST/GET/PUT/DELETE）。
- 本文件：聚焦 toggle PATCH 端点 + toggle 引发的 IK 文件实际变化。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.admin_dictionaries import router as admin_dictionaries_router
from app.api.auth import require_admin
from app.core.database import get_db
from app.core.exceptions import (
    ForbiddenException,
    UnauthorizedException,
    register_exception_handlers,
)
from app.models.domain_dictionary import DomainDictionary
from app.services import dictionary_service as ds_mod
from app.services.dictionary_service import (
    DictionaryService,
    IK_MAIN_DICT_FILE,
    IK_STOP_DICT_FILE,
)


# ─── Helpers ───────────────────────────────────────────────────────────


def _build_dictionary(
    *,
    dict_id: uuid.UUID | None = None,
    name: str = "水泥行业术语",
    description: str | None = "示例词典",
    terms: list | None = None,
    synonyms: list | None = None,
    stop_words: list | None = None,
    enabled: bool = True,
) -> DomainDictionary:
    """构造一个填满字段的 ``DomainDictionary`` ORM 实例。

    与 ``test_admin_dictionaries.py`` 同形态：直接塞进 id / 时间戳让
    ``DictionaryResponse`` 校验通过。
    """
    d = DomainDictionary(
        name=name,
        description=description,
        terms=terms if terms is not None else [],
        synonyms=synonyms if synonyms is not None else [],
        stop_words=stop_words if stop_words is not None else [],
        enabled=enabled,
    )
    d.id = dict_id or uuid.uuid4()
    d.created_at = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    d.updated_at = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    return d


def _scalar_result(value):
    """``await db.execute(...)`` 后 ``.scalar_one_or_none()`` 返回 *value*。"""
    r = MagicMock()
    r.scalar_one_or_none.return_value = value
    return r


def _scalars_all_result(values: list):
    """``await db.execute(...)`` 后 ``.scalars().all()`` 返回 *values*。"""
    r = MagicMock()
    r.scalars.return_value.all.return_value = list(values)
    return r


# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def mock_db() -> AsyncMock:
    db = AsyncMock()
    db.execute = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.delete = AsyncMock()
    db.refresh = AsyncMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    return db


@pytest.fixture
def admin_user() -> MagicMock:
    user = MagicMock()
    user.id = uuid.uuid4()
    user.email = "admin@wikforge.local"
    user.display_name = "Admin"
    return user


@pytest.fixture
def app(mock_db: AsyncMock, admin_user: MagicMock) -> FastAPI:
    application = FastAPI()
    register_exception_handlers(application)
    application.include_router(admin_dictionaries_router)

    async def _override_get_db():
        yield mock_db

    async def _override_require_admin():
        return admin_user

    application.dependency_overrides[get_db] = _override_get_db
    application.dependency_overrides[require_admin] = _override_require_admin
    return application


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


@pytest.fixture
def patched_service(monkeypatch):
    """把 ``DictionaryService`` 替换为 MagicMock，按需配置 toggle 行为。"""
    service = MagicMock()
    service.toggle_dictionary = AsyncMock(return_value=None)

    def _factory(_db):
        return service

    monkeypatch.setattr(
        "app.api.admin_dictionaries.DictionaryService",
        _factory,
    )
    return service


@pytest.fixture
def ik_dir(tmp_path, monkeypatch):
    """让模块级 ``IK_DICT_DIR`` 指向临时目录，避免触碰真实 ``/data/...``。

    与 ``test_ik_sync.ik_dir`` 同形态，确保端到端测试能直接读 ``.dic`` 文件
    校验内容。
    """
    target = tmp_path / "ik-custom-dict"
    monkeypatch.setattr(ds_mod, "IK_DICT_DIR", target)
    return target


# ─── Authorization ─────────────────────────────────────────────────────


class TestToggleAuthorization:
    """``PATCH /api/admin/dictionaries/{id}/toggle`` 鉴权守门。"""

    def _build_app_with_unauth(self, mock_db, exc):
        application = FastAPI()
        register_exception_handlers(application)
        application.include_router(admin_dictionaries_router)

        async def _override_get_db():
            yield mock_db

        async def _override_require_admin():
            raise exc

        application.dependency_overrides[get_db] = _override_get_db
        application.dependency_overrides[require_admin] = _override_require_admin
        return TestClient(application)

    def test_unauthenticated_returns_401(self, mock_db):
        """未登录访问 toggle 端点 → 401。"""
        client = self._build_app_with_unauth(
            mock_db, UnauthorizedException("缺少认证令牌")
        )
        response = client.patch(
            f"/api/admin/dictionaries/{uuid.uuid4()}/toggle",
            json={"enabled": False},
        )
        assert response.status_code == 401, response.text

    def test_non_admin_returns_403(self, mock_db):
        """普通登录用户访问 toggle 端点 → 403。"""
        client = self._build_app_with_unauth(
            mock_db, ForbiddenException("需要管理员权限")
        )
        response = client.patch(
            f"/api/admin/dictionaries/{uuid.uuid4()}/toggle",
            json={"enabled": False},
        )
        assert response.status_code == 403, response.text


# ─── Endpoint Behavior（路由层 + 服务层 mock）─────────────────────────


class TestToggleEndpoint:
    """toggle 端点的契约：参数透传、错误映射、返回 payload。"""

    def test_toggle_to_disabled_returns_updated_dictionary(
        self, client, patched_service
    ):
        """请求 ``enabled=False`` → 200 + ``enabled`` 在响应中为 False。"""
        dict_id = uuid.uuid4()
        patched_service.toggle_dictionary = AsyncMock(
            return_value=_build_dictionary(
                dict_id=dict_id,
                name="水泥行业术语",
                terms=[{"word": "大齿圈"}],
                enabled=False,
            )
        )

        response = client.patch(
            f"/api/admin/dictionaries/{dict_id}/toggle",
            json={"enabled": False},
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["id"] == str(dict_id)
        assert body["enabled"] is False
        # 路由把 dict_id 与 enabled 透传到服务层。
        kwargs = patched_service.toggle_dictionary.call_args.kwargs
        assert kwargs["dictionary_id"] == str(dict_id)
        assert kwargs["enabled"] is False

    def test_toggle_to_enabled_returns_updated_dictionary(
        self, client, patched_service
    ):
        """请求 ``enabled=True`` → 200 + ``enabled`` 在响应中为 True。"""
        dict_id = uuid.uuid4()
        patched_service.toggle_dictionary = AsyncMock(
            return_value=_build_dictionary(
                dict_id=dict_id,
                terms=[{"word": "回转窑"}],
                enabled=True,
            )
        )

        response = client.patch(
            f"/api/admin/dictionaries/{dict_id}/toggle",
            json={"enabled": True},
        )

        assert response.status_code == 200, response.text
        assert response.json()["enabled"] is True
        kwargs = patched_service.toggle_dictionary.call_args.kwargs
        assert kwargs["enabled"] is True

    def test_toggle_not_found_returns_404(self, client, patched_service):
        """词典不存在 → 服务层返回 None → 404。"""
        patched_service.toggle_dictionary = AsyncMock(return_value=None)

        response = client.patch(
            f"/api/admin/dictionaries/{uuid.uuid4()}/toggle",
            json={"enabled": False},
        )

        assert response.status_code == 404
        assert "Dictionary not found" in response.text

    def test_toggle_invalid_uuid_returns_400(self, client, patched_service):
        """非法 UUID 在路由层 ``_coerce_uuid`` 拦下 → 400，服务层不被调用。"""
        response = client.patch(
            "/api/admin/dictionaries/not-a-uuid/toggle",
            json={"enabled": False},
        )
        assert response.status_code == 400
        patched_service.toggle_dictionary.assert_not_called()

    def test_toggle_missing_enabled_field_returns_422(
        self, client, patched_service
    ):
        """``ToggleRequest.enabled`` 必填，缺失返回 422，服务层不被调用。"""
        response = client.patch(
            f"/api/admin/dictionaries/{uuid.uuid4()}/toggle",
            json={},
        )
        assert response.status_code == 422
        patched_service.toggle_dictionary.assert_not_called()


# ─── 端到端 IK 同步（真实写文件）─────────────────────────────────────


class TestToggleSyncBehavior:
    """``toggle_dictionary`` 必须触发 ``sync_ik_dictionaries`` 并真实改写
    ``.dic`` 文件，反映启用/禁用语义。

    使用 mock DB 配合 *真实* ``sync_ik_dictionaries`` 把 ``IK_DICT_DIR``
    重定向到 ``tmp_path``，从而能直接断言文件内容。
    """

    def _make_db_for_toggle(
        self,
        target_dict: DomainDictionary,
        *,
        other_enabled_dicts: list[DomainDictionary] | None = None,
    ) -> AsyncMock:
        """构造 toggle 流程专用的 mock DB。

        ``toggle_dictionary`` 的 execute 顺序：
        1. ``get_dictionary``：返回目标词典。
        2. ``sync_ik_dictionaries``：返回所有 ``enabled=True`` 的词典。

        第二步的返回结果**实时**反映 ``target_dict.enabled``，模拟 DB 层
        ``where(enabled == True)`` 过滤；这样即可无侵入地测试启用/禁用
        在 IK 文件上的实际效果。
        """
        other = other_enabled_dicts or []

        async def _execute(_stmt):
            # 第一次调用：get_dictionary 走 .scalar_one_or_none() 路径
            if not _execute.first_called:
                _execute.first_called = True
                return _scalar_result(target_dict)
            # 之后：sync_ik_dictionaries 走 .scalars().all() 路径，
            # 仅返回当前启用的词典（包含目标词典若仍启用）。
            enabled_dicts = list(other)
            if target_dict.enabled:
                enabled_dicts.append(target_dict)
            return _scalars_all_result(enabled_dicts)

        _execute.first_called = False  # type: ignore[attr-defined]

        db = AsyncMock()
        db.execute = AsyncMock(side_effect=_execute)
        db.flush = AsyncMock()
        db.refresh = AsyncMock()
        db.delete = AsyncMock()
        db.add = MagicMock()
        return db

    @pytest.mark.asyncio
    async def test_toggle_enabled_to_disabled_removes_terms_from_ik(
        self, ik_dir
    ):
        """启用 → 禁用：词典术语 / 停用词从 IK 文件中消失。"""
        target = _build_dictionary(
            terms=[{"word": "大齿圈"}, {"word": "回转窑"}],
            stop_words=["噪声词"],
            enabled=True,
        )
        # 另一启用词典：禁用 target 后它的术语仍应保留在文件里。
        other = _build_dictionary(
            dict_id=uuid.uuid4(),
            name="其它词典",
            terms=[{"word": "保留术语"}],
            stop_words=["其它停用词"],
            enabled=True,
        )

        db = self._make_db_for_toggle(
            target, other_enabled_dicts=[other]
        )
        service = DictionaryService(db)

        result = await service.toggle_dictionary(
            dictionary_id=str(target.id), enabled=False
        )

        assert result is target
        assert result.enabled is False  # 字段已更新
        # IK 文件物化：只剩 other 的术语 / 停用词。
        main_lines = (ik_dir / IK_MAIN_DICT_FILE).read_text(
            encoding="utf-8"
        ).splitlines()
        stop_lines = (ik_dir / IK_STOP_DICT_FILE).read_text(
            encoding="utf-8"
        ).splitlines()
        assert main_lines == ["保留术语"], main_lines
        assert "大齿圈" not in main_lines
        assert "回转窑" not in main_lines
        assert stop_lines == ["其它停用词"], stop_lines
        assert "噪声词" not in stop_lines

    @pytest.mark.asyncio
    async def test_toggle_disabled_to_enabled_re_adds_terms_to_ik(
        self, ik_dir
    ):
        """禁用 → 启用：词典术语 / 停用词重新出现在 IK 文件里。"""
        target = _build_dictionary(
            terms=[{"word": "新启用术语"}],
            stop_words=["新启用停用词"],
            enabled=False,
        )

        db = self._make_db_for_toggle(target)
        service = DictionaryService(db)

        result = await service.toggle_dictionary(
            dictionary_id=str(target.id), enabled=True
        )

        assert result is target
        assert result.enabled is True
        main_lines = (ik_dir / IK_MAIN_DICT_FILE).read_text(
            encoding="utf-8"
        ).splitlines()
        stop_lines = (ik_dir / IK_STOP_DICT_FILE).read_text(
            encoding="utf-8"
        ).splitlines()
        assert "新启用术语" in main_lines, main_lines
        assert "新启用停用词" in stop_lines, stop_lines

    @pytest.mark.asyncio
    async def test_toggle_same_value_still_triggers_sync(
        self, ik_dir, monkeypatch
    ):
        """同值切换（True → True / False → False）仍调用 ``sync_ik_dictionaries``。

        语义意义：管理员若发现 IK 文件与 DB 状态出现偏离（例如手工编辑
        ``.dic`` 后），可以通过「再次提交相同状态」强制同步。这个 no-op
        safety 也保证 toggle 端点不在路由层做"已是该状态就跳过"的隐式
        优化（这种隐式跳过会让 toggle 失去自愈能力）。
        """
        target = _build_dictionary(
            terms=[{"word": "保持术语"}],
            enabled=True,
        )

        # 监听 ``sync_ik_dictionaries`` 调用次数，但仍执行真实逻辑。
        original_sync = ds_mod.sync_ik_dictionaries
        sync_calls: list = []

        async def _spy_sync(db):
            sync_calls.append(db)
            return await original_sync(db)

        monkeypatch.setattr(ds_mod, "sync_ik_dictionaries", _spy_sync)

        db = self._make_db_for_toggle(target)
        service = DictionaryService(db)

        # True → True：语义上是 no-op，但同步仍应触发。
        result = await service.toggle_dictionary(
            dictionary_id=str(target.id), enabled=True
        )

        assert result is target
        assert result.enabled is True
        assert len(sync_calls) == 1, "no-op 切换也必须触发 IK 同步"
        # IK 文件仍包含术语（启用状态未变）。
        main_lines = (ik_dir / IK_MAIN_DICT_FILE).read_text(
            encoding="utf-8"
        ).splitlines()
        assert "保持术语" in main_lines

    @pytest.mark.asyncio
    async def test_toggle_disabled_to_disabled_still_triggers_sync(
        self, ik_dir, monkeypatch
    ):
        """已禁用词典再次切换到禁用：同步仍触发，IK 文件继续不含其术语。"""
        target = _build_dictionary(
            terms=[{"word": "禁用术语"}],
            enabled=False,
        )

        original_sync = ds_mod.sync_ik_dictionaries
        sync_calls: list = []

        async def _spy_sync(db):
            sync_calls.append(db)
            return await original_sync(db)

        monkeypatch.setattr(ds_mod, "sync_ik_dictionaries", _spy_sync)

        db = self._make_db_for_toggle(target)
        service = DictionaryService(db)

        result = await service.toggle_dictionary(
            dictionary_id=str(target.id), enabled=False
        )

        assert result is target
        assert result.enabled is False
        assert len(sync_calls) == 1, "no-op 切换也必须触发 IK 同步"
        # 禁用词典的术语**不在** IK 主词典文件里。
        main_content = (ik_dir / IK_MAIN_DICT_FILE).read_text(encoding="utf-8")
        assert "禁用术语" not in main_content

    @pytest.mark.asyncio
    async def test_toggle_missing_dictionary_skips_sync(self, ik_dir, monkeypatch):
        """词典不存在 → 服务返回 None，**不**触发 IK 同步。

        这是路由层 404 的下游保障：toggle 一个 ID 不存在的词典不应该把整
        个 IK 词库重写一遍（避免无意义的 mtime 变化）。
        """
        sync_calls: list = []

        async def _spy_sync(db):
            sync_calls.append(db)
            return {"terms": 0, "stop_words": 0}

        monkeypatch.setattr(ds_mod, "sync_ik_dictionaries", _spy_sync)

        db = AsyncMock()
        db.execute = AsyncMock(return_value=_scalar_result(None))
        db.flush = AsyncMock()
        db.refresh = AsyncMock()

        service = DictionaryService(db)
        result = await service.toggle_dictionary(
            dictionary_id=str(uuid.uuid4()), enabled=False
        )

        assert result is None
        assert sync_calls == [], "找不到词典时不应触发 IK 同步"
