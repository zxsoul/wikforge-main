"""Admin Dictionaries 同义词组管理 API 集成测试（任务 13.4）。

覆盖 ``/api/admin/dictionaries/{id}/synonyms`` 两个路由：

- ``POST /api/admin/dictionaries/{id}/synonyms``：upsert 同义词组（按
  ``primary`` 幂等替换），主术语 / 同义词字段校验失败返回 422，字典缺失
  返回 404，非法 UUID 返回 400。
- ``DELETE /api/admin/dictionaries/{id}/synonyms``：按 ``primary`` 删除同
  义词组，对不存在的 ``primary`` 幂等（不报错），字典缺失返回 404。
- 鉴权守门：两个路由都要求 ``require_admin``，未登录 401，非管理员 403。

策略与 ``test_admin_dictionaries_terms.py`` 完全一致：
- TestClient + ``dependency_overrides`` 注入 mock DB session。
- 通过 ``patched_service`` monkeypatch ``DictionaryService``，验证路由层
  对服务层的契约（参数透传、异常映射）。
- 另用真实服务层 + mock DB 走通一遍，覆盖核心语义：
  ``add_synonym_group`` 在 ``primary`` 已存在时替换而非重复追加；
  ``remove_synonym_group`` 对不存在的 ``primary`` 幂等。
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
    """构造一个填满字段的 ``DomainDictionary`` ORM 实例。"""
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
    """把 ``DictionaryService`` 替换为 MagicMock，按需配置返回值/异常。"""
    service = MagicMock()
    service.add_synonym_group = AsyncMock(return_value=None)
    service.remove_synonym_group = AsyncMock(return_value=None)

    def _factory(_db):
        return service

    monkeypatch.setattr(
        "app.api.admin_dictionaries.DictionaryService",
        _factory,
    )
    return service


# ─── Authorization ─────────────────────────────────────────────────────


class TestAuthorization:
    """``require_admin`` 守门：未登录 401 / 非管理员 403。"""

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

    @pytest.mark.parametrize("method", ["POST", "DELETE"])
    def test_unauthenticated_returns_401(self, mock_db, method):
        client = self._build_app_with_unauth(
            mock_db, UnauthorizedException("缺少认证令牌")
        )
        path = f"/api/admin/dictionaries/{uuid.uuid4()}/synonyms"
        body = (
            {"primary": "大齿圈", "synonyms": ["齿圈"]}
            if method == "POST"
            else {"primary": "大齿圈"}
        )
        response = client.request(method, path, json=body)
        assert response.status_code == 401, (method, response.text)

    @pytest.mark.parametrize("method", ["POST", "DELETE"])
    def test_non_admin_returns_403(self, mock_db, method):
        client = self._build_app_with_unauth(
            mock_db, ForbiddenException("需要管理员权限")
        )
        path = f"/api/admin/dictionaries/{uuid.uuid4()}/synonyms"
        body = (
            {"primary": "大齿圈", "synonyms": ["齿圈"]}
            if method == "POST"
            else {"primary": "大齿圈"}
        )
        response = client.request(method, path, json=body)
        assert response.status_code == 403, (method, response.text)


# ─── POST /api/admin/dictionaries/{id}/synonyms ───────────────────────


class TestAddSynonymGroup:
    """upsert 同义词组：200 / 404 / 422 / 400。"""

    def test_add_new_group_returns_updated_dictionary(
        self, client, patched_service
    ):
        dict_id = uuid.uuid4()
        new_synonyms = [{"primary": "大齿圈", "synonyms": ["齿圈", "主齿圈"]}]
        patched_service.add_synonym_group = AsyncMock(
            return_value=_build_dictionary(
                dict_id=dict_id, synonyms=new_synonyms
            )
        )

        response = client.post(
            f"/api/admin/dictionaries/{dict_id}/synonyms",
            json={"primary": "大齿圈", "synonyms": ["齿圈", "主齿圈"]},
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["id"] == str(dict_id)
        assert body["synonyms"] == new_synonyms

        kwargs = patched_service.add_synonym_group.call_args.kwargs
        assert kwargs["dictionary_id"] == str(dict_id)
        assert kwargs["primary"] == "大齿圈"
        assert kwargs["synonyms"] == ["齿圈", "主齿圈"]

    def test_add_replaces_existing_group_with_same_primary(
        self, client, patched_service
    ):
        """对同一 ``primary`` 二次 POST：服务层做替换，路由原样回显。"""
        dict_id = uuid.uuid4()
        # 服务层 upsert 后只有一组同义词，synonyms 是新列表
        replaced = [{"primary": "大齿圈", "synonyms": ["主齿圈"]}]
        patched_service.add_synonym_group = AsyncMock(
            return_value=_build_dictionary(dict_id=dict_id, synonyms=replaced)
        )

        response = client.post(
            f"/api/admin/dictionaries/{dict_id}/synonyms",
            json={"primary": "大齿圈", "synonyms": ["主齿圈"]},
        )

        assert response.status_code == 200, response.text
        body = response.json()
        # 同 primary 不出现两次
        primaries = [g["primary"] for g in body["synonyms"]]
        assert primaries.count("大齿圈") == 1
        # synonyms 列表已被替换
        assert body["synonyms"][0]["synonyms"] == ["主齿圈"]

    def test_add_accepts_empty_synonyms_list(self, client, patched_service):
        """``SynonymGroup`` 允许空 synonyms 列表（design.md）。"""
        dict_id = uuid.uuid4()
        patched_service.add_synonym_group = AsyncMock(
            return_value=_build_dictionary(
                dict_id=dict_id,
                synonyms=[{"primary": "大齿圈", "synonyms": []}],
            )
        )

        response = client.post(
            f"/api/admin/dictionaries/{dict_id}/synonyms",
            json={"primary": "大齿圈", "synonyms": []},
        )

        assert response.status_code == 200, response.text
        kwargs = patched_service.add_synonym_group.call_args.kwargs
        assert kwargs["synonyms"] == []

    def test_add_not_found_returns_404(self, client, patched_service):
        patched_service.add_synonym_group = AsyncMock(return_value=None)

        response = client.post(
            f"/api/admin/dictionaries/{uuid.uuid4()}/synonyms",
            json={"primary": "大齿圈", "synonyms": ["齿圈"]},
        )

        assert response.status_code == 404
        assert "Dictionary not found" in response.text

    def test_add_blank_primary_returns_422(self, client, patched_service):
        """``primary`` 强制 ``min_length=1``，空串触发 422。"""
        response = client.post(
            f"/api/admin/dictionaries/{uuid.uuid4()}/synonyms",
            json={"primary": "", "synonyms": ["齿圈"]},
        )

        assert response.status_code == 422
        patched_service.add_synonym_group.assert_not_called()

    def test_add_overlong_primary_returns_422(self, client, patched_service):
        """``primary`` 强制 ``max_length=30``，超长触发 422。"""
        response = client.post(
            f"/api/admin/dictionaries/{uuid.uuid4()}/synonyms",
            json={"primary": "x" * 31, "synonyms": ["齿圈"]},
        )

        assert response.status_code == 422
        patched_service.add_synonym_group.assert_not_called()

    def test_add_missing_primary_returns_422(self, client, patched_service):
        """``primary`` 是必填字段。"""
        response = client.post(
            f"/api/admin/dictionaries/{uuid.uuid4()}/synonyms",
            json={"synonyms": ["齿圈"]},
        )

        assert response.status_code == 422
        patched_service.add_synonym_group.assert_not_called()

    def test_add_missing_synonyms_returns_422(self, client, patched_service):
        """``synonyms`` 也是必填字段（即使列表可以为空）。"""
        response = client.post(
            f"/api/admin/dictionaries/{uuid.uuid4()}/synonyms",
            json={"primary": "大齿圈"},
        )

        assert response.status_code == 422
        patched_service.add_synonym_group.assert_not_called()

    def test_add_service_validation_failure_returns_422(
        self, client, patched_service
    ):
        """同义词在 Pydantic 层合法但 ``validate_term`` 失败 → 422。

        例如同义词包含控制字符，``add_synonym_group`` 抛 ``ValueError``，
        路由把它映射成 422，与其它写入路由保持一致。
        """
        patched_service.add_synonym_group = AsyncMock(
            side_effect=ValueError("同义词校验失败: 不能包含特殊控制字符")
        )

        response = client.post(
            f"/api/admin/dictionaries/{uuid.uuid4()}/synonyms",
            json={"primary": "大齿圈", "synonyms": ["齿圈"]},
        )

        assert response.status_code == 422
        assert "同义词校验失败" in response.text

    def test_add_invalid_uuid_returns_400(self, client, patched_service):
        response = client.post(
            "/api/admin/dictionaries/not-a-uuid/synonyms",
            json={"primary": "大齿圈", "synonyms": ["齿圈"]},
        )

        assert response.status_code == 400
        patched_service.add_synonym_group.assert_not_called()


# ─── DELETE /api/admin/dictionaries/{id}/synonyms ─────────────────────


class TestRemoveSynonymGroup:
    """删除同义词组：200 / 404 / 400 / 422 + 幂等性。"""

    def test_remove_returns_updated_dictionary(self, client, patched_service):
        dict_id = uuid.uuid4()
        # 删除 “大齿圈” 后剩下一个组
        remaining = [{"primary": "回转窑", "synonyms": ["窑炉"]}]
        patched_service.remove_synonym_group = AsyncMock(
            return_value=_build_dictionary(
                dict_id=dict_id, synonyms=remaining
            )
        )

        response = client.request(
            "DELETE",
            f"/api/admin/dictionaries/{dict_id}/synonyms",
            json={"primary": "大齿圈"},
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["id"] == str(dict_id)
        primaries = [g["primary"] for g in body["synonyms"]]
        assert "大齿圈" not in primaries
        assert primaries == ["回转窑"]

        kwargs = patched_service.remove_synonym_group.call_args.kwargs
        assert kwargs["dictionary_id"] == str(dict_id)
        assert kwargs["primary"] == "大齿圈"

    def test_remove_idempotent_on_missing_primary(
        self, client, patched_service
    ):
        """删除不存在的 ``primary`` 不报错，返回原字典。"""
        dict_id = uuid.uuid4()
        existing = [{"primary": "回转窑", "synonyms": ["窑炉"]}]
        patched_service.remove_synonym_group = AsyncMock(
            return_value=_build_dictionary(
                dict_id=dict_id, synonyms=existing
            )
        )

        response = client.request(
            "DELETE",
            f"/api/admin/dictionaries/{dict_id}/synonyms",
            json={"primary": "不存在的词"},
        )

        assert response.status_code == 200, response.text
        body = response.json()
        # 字典内容保持不变
        assert [g["primary"] for g in body["synonyms"]] == ["回转窑"]

    def test_remove_not_found_returns_404(self, client, patched_service):
        patched_service.remove_synonym_group = AsyncMock(return_value=None)

        response = client.request(
            "DELETE",
            f"/api/admin/dictionaries/{uuid.uuid4()}/synonyms",
            json={"primary": "大齿圈"},
        )

        assert response.status_code == 404
        assert "Dictionary not found" in response.text

    def test_remove_invalid_uuid_returns_400(self, client, patched_service):
        response = client.request(
            "DELETE",
            "/api/admin/dictionaries/not-a-uuid/synonyms",
            json={"primary": "大齿圈"},
        )

        assert response.status_code == 400
        patched_service.remove_synonym_group.assert_not_called()

    def test_remove_missing_primary_returns_422(
        self, client, patched_service
    ):
        """``primary`` 字段必填（``RemoveSynonymGroupRequest``），缺失 422。"""
        response = client.request(
            "DELETE",
            f"/api/admin/dictionaries/{uuid.uuid4()}/synonyms",
            json={},
        )

        assert response.status_code == 422
        patched_service.remove_synonym_group.assert_not_called()


# ─── 服务层语义验证（不 mock 服务层） ───────────────────────────────


class TestServiceLevelSemantics:
    """直接验证 ``DictionaryService.add_synonym_group`` /
    ``remove_synonym_group`` 的契约。

    路由层在上面已被覆盖；这里用真实服务层 + mock DB 走通一遍，确认
    upsert 替换与删除幂等的核心语义存在。这两条性质是任务 13.4 的核心
    需求（同 primary 替换、删除不存在的 primary 不报错）。
    """

    @pytest.fixture
    def dictionary_with_synonyms(self):
        return _build_dictionary(
            synonyms=[
                {"primary": "大齿圈", "synonyms": ["齿圈"]},
                {"primary": "回转窑", "synonyms": ["窑炉"]},
            ]
        )

    @pytest.fixture
    def service(self, mock_db, dictionary_with_synonyms):
        from app.services.dictionary_service import DictionaryService

        service = DictionaryService(mock_db)
        service.get_dictionary = AsyncMock(
            return_value=dictionary_with_synonyms
        )
        return service

    @pytest.mark.asyncio
    async def test_add_synonym_group_appends_when_primary_is_new(
        self, service, dictionary_with_synonyms
    ):
        """``primary`` 不存在时追加新组。"""
        result = await service.add_synonym_group(
            dictionary_id=str(dictionary_with_synonyms.id),
            primary="水泥",
            synonyms=["熟料", "灰泥"],
        )

        assert result is not None
        primaries = [g["primary"] for g in result.synonyms]
        assert primaries == ["大齿圈", "回转窑", "水泥"]
        new_group = result.synonyms[-1]
        assert new_group["synonyms"] == ["熟料", "灰泥"]

    @pytest.mark.asyncio
    async def test_add_synonym_group_replaces_existing_primary(
        self, service, dictionary_with_synonyms
    ):
        """``primary`` 已存在时替换该组的 synonyms 列表，不重复追加。"""
        result = await service.add_synonym_group(
            dictionary_id=str(dictionary_with_synonyms.id),
            primary="大齿圈",
            synonyms=["主齿圈", "齿轮圈"],
        )

        assert result is not None
        primaries = [g["primary"] for g in result.synonyms]
        # “大齿圈” 仍然只出现一次
        assert primaries.count("大齿圈") == 1
        # 列表整体长度未增加
        assert len(result.synonyms) == 2
        # synonyms 列表已被替换
        target = next(g for g in result.synonyms if g["primary"] == "大齿圈")
        assert target["synonyms"] == ["主齿圈", "齿轮圈"]

    @pytest.mark.asyncio
    async def test_add_synonym_group_accepts_empty_synonyms_list(
        self, service, dictionary_with_synonyms
    ):
        """空 synonyms 列表合法（design.md 允许空列表）。"""
        result = await service.add_synonym_group(
            dictionary_id=str(dictionary_with_synonyms.id),
            primary="新词",
            synonyms=[],
        )

        assert result is not None
        new_group = next(g for g in result.synonyms if g["primary"] == "新词")
        assert new_group["synonyms"] == []

    @pytest.mark.asyncio
    async def test_add_synonym_group_invalid_primary_raises(
        self, service, dictionary_with_synonyms
    ):
        """超长 primary 触发 ``validate_term`` 校验失败。"""
        with pytest.raises(ValueError, match="主术语校验失败"):
            await service.add_synonym_group(
                dictionary_id=str(dictionary_with_synonyms.id),
                primary="x" * 31,
                synonyms=[],
            )

    @pytest.mark.asyncio
    async def test_add_synonym_group_invalid_synonym_raises(
        self, service, dictionary_with_synonyms
    ):
        """同义词包含控制字符时触发 ``validate_term`` 校验失败。"""
        with pytest.raises(ValueError, match="同义词校验失败"):
            await service.add_synonym_group(
                dictionary_id=str(dictionary_with_synonyms.id),
                primary="合法词",
                synonyms=["含\x01控制字符"],
            )

    @pytest.mark.asyncio
    async def test_remove_synonym_group_removes_matching_primary(
        self, service, dictionary_with_synonyms
    ):
        """删除存在的 primary：组被移除，其它组保留。"""
        result = await service.remove_synonym_group(
            dictionary_id=str(dictionary_with_synonyms.id),
            primary="大齿圈",
        )

        assert result is not None
        primaries = [g["primary"] for g in result.synonyms]
        assert primaries == ["回转窑"]

    @pytest.mark.asyncio
    async def test_remove_synonym_group_idempotent_on_missing(
        self, service, dictionary_with_synonyms
    ):
        """删除不存在的 primary 不抛错，字典保持不变。"""
        result = await service.remove_synonym_group(
            dictionary_id=str(dictionary_with_synonyms.id),
            primary="幽灵词",
        )

        assert result is not None
        primaries = [g["primary"] for g in result.synonyms]
        assert primaries == ["大齿圈", "回转窑"]

    @pytest.mark.asyncio
    async def test_add_then_remove_round_trip(
        self, service, dictionary_with_synonyms
    ):
        """upsert 一个新 primary 后再 DELETE：回到无该 primary 的状态。"""
        await service.add_synonym_group(
            dictionary_id=str(dictionary_with_synonyms.id),
            primary="水泥",
            synonyms=["熟料"],
        )
        result = await service.remove_synonym_group(
            dictionary_id=str(dictionary_with_synonyms.id),
            primary="水泥",
        )

        assert result is not None
        primaries = [g["primary"] for g in result.synonyms]
        assert "水泥" not in primaries
        assert primaries == ["大齿圈", "回转窑"]
