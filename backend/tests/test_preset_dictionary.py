"""预置词典：通用中文停用词（任务 13.10）。

针对 ``ensure_preset_dictionaries`` 与 ``POST /api/admin/dictionaries/preset/init``
的聚焦测试，覆盖：

- 服务层语义
  * 首次调用创建一个名为「通用中文停用词」的词典
  * 重复调用幂等（已存在时不再插入）
  * 创建出的词典默认 ``enabled=True``，``stop_words == CHINESE_STOP_WORDS``
  * ``CHINESE_STOP_WORDS`` 全部通过 ``validate_term`` 校验，且包含常见停用词

- 路由层
  * ``POST /api/admin/dictionaries/preset/init`` 实际调用
    ``ensure_preset_dictionaries``，返回 201
  * 鉴权守门：未登录 401、非管理员 403

策略与 ``test_admin_dictionaries.py`` 一致：
- FastAPI ``TestClient`` + ``dependency_overrides`` 注入 ``mock_db``、覆写
  ``require_admin``。
- 通过 ``monkeypatch`` 把 ``ensure_preset_dictionaries`` 在
  ``app.services.dictionary_service`` 模块级替换为 mock，以便观察是否被调用。
"""

from __future__ import annotations

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
from app.services.dictionary_service import (
    CHINESE_STOP_WORDS,
    ensure_preset_dictionaries,
    validate_term,
)


# ─── Helpers ───────────────────────────────────────────────────────────


def _scalar_result(value):
    """Build a fake ``await db.execute(...)`` result whose
    ``.scalar_one_or_none()`` returns *value*."""
    r = MagicMock()
    r.scalar_one_or_none.return_value = value
    return r


# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def mock_db() -> AsyncMock:
    """SQLAlchemy ``AsyncSession`` mock。``execute`` 在每个用例里按需 patch。"""
    db = AsyncMock()
    db.execute = AsyncMock()
    # ``add`` 在 SQLAlchemy 中是同步方法
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.refresh = AsyncMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    return db


@pytest.fixture
def admin_user() -> MagicMock:
    user = MagicMock()
    user.email = "admin@wikforge.local"
    return user


@pytest.fixture
def app(mock_db: AsyncMock, admin_user: MagicMock) -> FastAPI:
    """FastAPI app，把 ``require_admin`` 与 ``get_db`` 替换成测试 stub。"""
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


# ─── 服务层：ensure_preset_dictionaries ────────────────────────────────


class TestEnsurePresetDictionaries:
    """``ensure_preset_dictionaries`` 的服务层语义。"""

    @pytest.mark.asyncio
    async def test_creates_dictionary_on_first_call(self, mock_db):
        """首次调用：词典不存在 → 写入一条「通用中文停用词」。"""
        mock_db.execute = AsyncMock(return_value=_scalar_result(None))

        await ensure_preset_dictionaries(mock_db)

        # 仅插入一次
        assert mock_db.add.call_count == 1
        added = mock_db.add.call_args.args[0]
        assert isinstance(added, DomainDictionary)
        assert added.name == "通用中文停用词"
        # 默认启用
        assert added.enabled is True
        # 停用词列表与预置常量一致
        assert added.stop_words == CHINESE_STOP_WORDS
        # terms / synonyms 留空
        assert added.terms == []
        assert added.synonyms == []
        # 描述非空（用户友好）
        assert added.description and added.description.strip()
        # flush 落表
        mock_db.flush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_idempotent_on_subsequent_calls(self, mock_db):
        """重复调用：词典已存在 → 不再插入，也不 flush。"""
        existing = DomainDictionary(
            name="通用中文停用词",
            description="已存在",
            terms=[],
            synonyms=[],
            stop_words=CHINESE_STOP_WORDS,
            enabled=True,
        )
        mock_db.execute = AsyncMock(return_value=_scalar_result(existing))

        await ensure_preset_dictionaries(mock_db)

        mock_db.add.assert_not_called()
        mock_db.flush.assert_not_called()

    @pytest.mark.asyncio
    async def test_two_consecutive_calls_create_only_once(self, mock_db):
        """连续两次调用：第一次插入，第二次幂等（端到端模拟）。"""
        # 第一次 execute 返回 None（不存在），第二次返回已插入的对象
        first_call_result = _scalar_result(None)

        # 用 side_effect 控制两次返回
        existing_after_first_call: list[DomainDictionary] = []

        async def _execute(*_args, **_kwargs):
            if existing_after_first_call:
                return _scalar_result(existing_after_first_call[0])
            return first_call_result

        # add() 时记录已插入的对象，下一次 execute 就能返回它
        def _add(obj):
            existing_after_first_call.append(obj)

        mock_db.execute = AsyncMock(side_effect=_execute)
        mock_db.add = MagicMock(side_effect=_add)

        await ensure_preset_dictionaries(mock_db)
        await ensure_preset_dictionaries(mock_db)

        # add 只在第一次被调用一次
        assert mock_db.add.call_count == 1
        # flush 也只调用一次（第二次幂等）
        assert mock_db.flush.await_count == 1


# ─── 预置常量内容审计 ──────────────────────────────────────────────────


class TestChineseStopWordsContent:
    """``CHINESE_STOP_WORDS`` 的内容质量审计。"""

    def test_contains_common_stop_words(self):
        """覆盖最常见的中文停用词 的/了/是/在/和。"""
        for word in ["的", "了", "是", "在", "和"]:
            assert word in CHINESE_STOP_WORDS, f"missing common stop word: {word}"

    def test_no_empty_or_blank_entries(self):
        """所有条目非空（防止编辑时引入空串）。"""
        for word in CHINESE_STOP_WORDS:
            assert word, "preset list contains empty entry"
            assert word.strip() == word, f"preset entry not stripped: {word!r}"

    def test_no_duplicates(self):
        """预置列表没有重复词。"""
        assert len(CHINESE_STOP_WORDS) == len(set(CHINESE_STOP_WORDS))

    def test_all_words_pass_validate_term(self):
        """每个停用词都能通过 ``validate_term``（确保未来拓展时不会引入非法字符）。"""
        for word in CHINESE_STOP_WORDS:
            is_valid, msg = validate_term(word)
            assert is_valid, f"stop word {word!r} fails validate_term: {msg}"


# ─── 路由：POST /api/admin/dictionaries/preset/init ────────────────────


class TestPresetInitEndpoint:
    """``POST /api/admin/dictionaries/preset/init`` 端点行为。"""

    def test_endpoint_invokes_ensure_preset_dictionaries(
        self, client, mock_db, monkeypatch
    ):
        """端点调用 ``ensure_preset_dictionaries`` 并返回 201。"""
        called_with: list = []

        async def _fake_ensure(db):
            called_with.append(db)

        monkeypatch.setattr(
            "app.services.dictionary_service.ensure_preset_dictionaries",
            _fake_ensure,
        )

        response = client.post("/api/admin/dictionaries/preset/init")

        assert response.status_code == 201, response.text
        body = response.json()
        assert "message" in body
        # ensure_preset_dictionaries 被以注入的 mock_db 调用
        assert len(called_with) == 1
        assert called_with[0] is mock_db

    def test_unauthenticated_returns_401(self, mock_db):
        """未登录访问 → 401。"""
        application = FastAPI()
        register_exception_handlers(application)
        application.include_router(admin_dictionaries_router)

        async def _override_get_db():
            yield mock_db

        async def _override_require_admin():
            raise UnauthorizedException("缺少认证令牌")

        application.dependency_overrides[get_db] = _override_get_db
        application.dependency_overrides[require_admin] = _override_require_admin

        with TestClient(application) as c:
            response = c.post("/api/admin/dictionaries/preset/init")

        assert response.status_code == 401, response.text
        # 未通过鉴权时不应触达 DB
        mock_db.execute.assert_not_called()
        mock_db.add.assert_not_called()

    def test_non_admin_returns_403(self, mock_db):
        """普通用户访问 → 403。"""
        application = FastAPI()
        register_exception_handlers(application)
        application.include_router(admin_dictionaries_router)

        async def _override_get_db():
            yield mock_db

        async def _override_require_admin():
            raise ForbiddenException("需要管理员权限")

        application.dependency_overrides[get_db] = _override_get_db
        application.dependency_overrides[require_admin] = _override_require_admin

        with TestClient(application) as c:
            response = c.post("/api/admin/dictionaries/preset/init")

        assert response.status_code == 403, response.text
        mock_db.execute.assert_not_called()
        mock_db.add.assert_not_called()
