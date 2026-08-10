"""Admin Dictionaries 术语增删 API 集成测试（任务 13.3）。

覆盖 ``/api/admin/dictionaries/{id}/terms`` 两个路由：

- ``POST /api/admin/dictionaries/{id}/terms``：追加术语，按 word 去重；成功
  返回 200 + 更新后的字典；字典缺失 404；术语字段非法 422。
- ``DELETE /api/admin/dictionaries/{id}/terms``：按 word 删除术语，对不存在
  的 word 幂等（不报错）；字典缺失 404。
- 鉴权守门：两个路由都要求 ``require_admin``，未登录 401，非管理员 403。
- 路径段非法 UUID 返回 400。

策略与 ``test_admin_dictionaries.py`` 完全一致：
- TestClient + ``dependency_overrides`` 注入 mock DB session。
- 通过 ``patched_service`` monkeypatch ``DictionaryService``，验证路由层
  对服务层的契约（参数透传、异常映射），同时通过一个真实服务行为子用例
  覆盖 ``add_terms`` upsert / ``remove_terms`` 幂等的语义。
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
    service.add_terms = AsyncMock(return_value=None)
    service.remove_terms = AsyncMock(return_value=None)

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
        path = f"/api/admin/dictionaries/{uuid.uuid4()}/terms"
        body = {"terms": []} if method == "POST" else {"words": []}
        response = client.request(method, path, json=body)
        assert response.status_code == 401, (method, response.text)

    @pytest.mark.parametrize("method", ["POST", "DELETE"])
    def test_non_admin_returns_403(self, mock_db, method):
        client = self._build_app_with_unauth(
            mock_db, ForbiddenException("需要管理员权限")
        )
        path = f"/api/admin/dictionaries/{uuid.uuid4()}/terms"
        body = {"terms": []} if method == "POST" else {"words": []}
        response = client.request(method, path, json=body)
        assert response.status_code == 403, (method, response.text)


# ─── POST /api/admin/dictionaries/{id}/terms ───────────────────────────


class TestAddTerms:
    """追加术语：200 / 404 / 422 / 400。"""

    def test_add_terms_returns_updated_dictionary(self, client, patched_service):
        dict_id = uuid.uuid4()
        new_terms = [
            {"word": "大齿圈", "pos": "n", "weight": 1.5},
            {"word": "回转窑"},
        ]
        patched_service.add_terms = AsyncMock(
            return_value=_build_dictionary(dict_id=dict_id, terms=new_terms)
        )

        response = client.post(
            f"/api/admin/dictionaries/{dict_id}/terms",
            json={"terms": new_terms},
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["id"] == str(dict_id)
        assert body["terms"] == new_terms

        kwargs = patched_service.add_terms.call_args.kwargs
        assert kwargs["dictionary_id"] == str(dict_id)
        # 路由层把 TermSchema 转回 dict，pos / weight 默认值已补全
        assert kwargs["new_terms"] == [
            {"word": "大齿圈", "pos": "n", "weight": 1.5},
            {"word": "回转窑", "pos": None, "weight": 1.0},
        ]

    def test_add_terms_idempotent_on_duplicate_word(
        self, client, patched_service
    ):
        """重复 word 不应在结果里出现两次（语义由服务层负责，路由透传）。

        通过让 mock 服务层返回去重后的结果，断言路由原样回显，证明
        ``AddTermsRequest`` 的契约允许传入重复 word，不会触发 422。
        """
        dict_id = uuid.uuid4()
        existing = [{"word": "水泥", "pos": "n", "weight": 1.0}]
        # 客户端传入与已有 word 重复的术语
        payload_terms = [
            {"word": "水泥", "pos": "n", "weight": 2.0},  # 与现有重名
            {"word": "熟料"},
        ]
        # 服务层 upsert 后返回去重结果（保留首个）
        merged = existing + [{"word": "熟料", "pos": None, "weight": 1.0}]
        patched_service.add_terms = AsyncMock(
            return_value=_build_dictionary(dict_id=dict_id, terms=merged)
        )

        response = client.post(
            f"/api/admin/dictionaries/{dict_id}/terms",
            json={"terms": payload_terms},
        )

        assert response.status_code == 200, response.text
        body = response.json()
        # 返回的 word 列表去重后只有一个 “水泥”
        words = [t["word"] for t in body["terms"]]
        assert words.count("水泥") == 1
        assert "熟料" in words

    def test_add_terms_not_found_returns_404(self, client, patched_service):
        patched_service.add_terms = AsyncMock(return_value=None)

        response = client.post(
            f"/api/admin/dictionaries/{uuid.uuid4()}/terms",
            json={"terms": [{"word": "大齿圈"}]},
        )

        assert response.status_code == 404
        assert "Dictionary not found" in response.text

    def test_add_terms_invalid_term_blank_word_returns_422(
        self, client, patched_service
    ):
        """``TermSchema.word`` 强制 ``min_length=1``，空串触发 422。"""
        response = client.post(
            f"/api/admin/dictionaries/{uuid.uuid4()}/terms",
            json={"terms": [{"word": ""}]},
        )

        assert response.status_code == 422
        patched_service.add_terms.assert_not_called()

    def test_add_terms_invalid_term_overlong_word_returns_422(
        self, client, patched_service
    ):
        """``TermSchema.word`` 强制 ``max_length=30``，超长触发 422。"""
        response = client.post(
            f"/api/admin/dictionaries/{uuid.uuid4()}/terms",
            json={"terms": [{"word": "x" * 31}]},
        )

        assert response.status_code == 422
        patched_service.add_terms.assert_not_called()

    def test_add_terms_invalid_term_non_float_weight_returns_422(
        self, client, patched_service
    ):
        """``weight`` 必须可强转为 float，传入字符串触发 422。"""
        response = client.post(
            f"/api/admin/dictionaries/{uuid.uuid4()}/terms",
            json={"terms": [{"word": "大齿圈", "weight": "重要"}]},
        )

        assert response.status_code == 422
        patched_service.add_terms.assert_not_called()

    def test_add_terms_service_validation_failure_returns_422(
        self, client, patched_service
    ):
        """术语在 Pydantic 层合法但服务层更深层校验失败 → 422。

        例如包含控制字符，``validate_term`` 会抛 ``ValueError``，路由把
        它映射成 422，与 ``create_dictionary`` 的语义保持一致。
        """
        patched_service.add_terms = AsyncMock(
            side_effect=ValueError("术语校验失败: 不能包含特殊控制字符")
        )

        response = client.post(
            f"/api/admin/dictionaries/{uuid.uuid4()}/terms",
            json={"terms": [{"word": "合法术语"}]},
        )

        assert response.status_code == 422
        assert "术语校验失败" in response.text

    def test_add_terms_invalid_uuid_returns_400(self, client, patched_service):
        response = client.post(
            "/api/admin/dictionaries/not-a-uuid/terms",
            json={"terms": [{"word": "大齿圈"}]},
        )

        assert response.status_code == 400
        patched_service.add_terms.assert_not_called()


# ─── DELETE /api/admin/dictionaries/{id}/terms ─────────────────────────


class TestRemoveTerms:
    """删除术语：200 / 404 / 400 + 幂等性。"""

    def test_remove_terms_returns_updated_dictionary(
        self, client, patched_service
    ):
        dict_id = uuid.uuid4()
        # 删除 “水泥” 后剩下的术语
        remaining = [{"word": "熟料", "pos": None, "weight": 1.0}]
        patched_service.remove_terms = AsyncMock(
            return_value=_build_dictionary(dict_id=dict_id, terms=remaining)
        )

        response = client.request(
            "DELETE",
            f"/api/admin/dictionaries/{dict_id}/terms",
            json={"words": ["水泥"]},
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["id"] == str(dict_id)
        words = [t["word"] for t in body["terms"]]
        assert "水泥" not in words
        assert words == ["熟料"]

        kwargs = patched_service.remove_terms.call_args.kwargs
        assert kwargs["dictionary_id"] == str(dict_id)
        assert kwargs["words"] == ["水泥"]

    def test_remove_terms_idempotent_on_missing_word(
        self, client, patched_service
    ):
        """删除不在字典里的 word 不应报错，返回原字典。"""
        dict_id = uuid.uuid4()
        existing = [{"word": "水泥", "pos": None, "weight": 1.0}]
        # 服务层对未命中 word 直接返回未变更字典（remove_terms 内部按
        # 集合差集过滤，不会抛出 KeyError 或 404）。
        patched_service.remove_terms = AsyncMock(
            return_value=_build_dictionary(dict_id=dict_id, terms=existing)
        )

        response = client.request(
            "DELETE",
            f"/api/admin/dictionaries/{dict_id}/terms",
            json={"words": ["不存在的词"]},
        )

        assert response.status_code == 200, response.text
        body = response.json()
        # 字典内容保持不变
        assert [t["word"] for t in body["terms"]] == ["水泥"]

    def test_remove_terms_idempotent_on_empty_words_list(
        self, client, patched_service
    ):
        """空 ``words`` 列表合法，服务层返回原字典。"""
        dict_id = uuid.uuid4()
        existing = [{"word": "水泥", "pos": None, "weight": 1.0}]
        patched_service.remove_terms = AsyncMock(
            return_value=_build_dictionary(dict_id=dict_id, terms=existing)
        )

        response = client.request(
            "DELETE",
            f"/api/admin/dictionaries/{dict_id}/terms",
            json={"words": []},
        )

        assert response.status_code == 200
        body = response.json()
        assert [t["word"] for t in body["terms"]] == ["水泥"]
        # 服务层依然被调用一次，由它决定空列表是 no-op
        patched_service.remove_terms.assert_awaited_once()

    def test_remove_terms_not_found_returns_404(self, client, patched_service):
        patched_service.remove_terms = AsyncMock(return_value=None)

        response = client.request(
            "DELETE",
            f"/api/admin/dictionaries/{uuid.uuid4()}/terms",
            json={"words": ["x"]},
        )

        assert response.status_code == 404
        assert "Dictionary not found" in response.text

    def test_remove_terms_invalid_uuid_returns_400(
        self, client, patched_service
    ):
        response = client.request(
            "DELETE",
            "/api/admin/dictionaries/not-a-uuid/terms",
            json={"words": ["x"]},
        )

        assert response.status_code == 400
        patched_service.remove_terms.assert_not_called()

    def test_remove_terms_missing_words_field_returns_422(
        self, client, patched_service
    ):
        """``words`` 字段必填（``RemoveTermsRequest``），缺失返回 422。"""
        response = client.request(
            "DELETE",
            f"/api/admin/dictionaries/{uuid.uuid4()}/terms",
            json={},
        )

        assert response.status_code == 422
        patched_service.remove_terms.assert_not_called()


# ─── 服务层语义验证（不 mock 服务层） ───────────────────────────────


class TestServiceLevelSemantics:
    """直接验证 ``DictionaryService.add_terms`` / ``remove_terms`` 的契约。

    路由层在上面已被覆盖；这里用真实服务层 + mock DB 走通一遍，确认
    upsert 去重与删除幂等的核心语义存在。这两条性质是任务 13.3 的核心
    需求（按 word 去重、删除不存在的 word 不报错）。
    """

    @pytest.fixture
    def dictionary_with_terms(self):
        return _build_dictionary(
            terms=[
                {"word": "水泥", "pos": "n", "weight": 1.0},
                {"word": "熟料", "pos": "n", "weight": 1.0},
            ]
        )

    @pytest.fixture
    def service(self, monkeypatch, mock_db, dictionary_with_terms):
        """构造一个 ``DictionaryService``，``get_dictionary`` 走 mock，
        ``sync_ik_dictionaries`` 不实际写文件。"""
        from app.services import dictionary_service as ds_mod
        from app.services.dictionary_service import DictionaryService

        service = DictionaryService(mock_db)
        service.get_dictionary = AsyncMock(return_value=dictionary_with_terms)
        # IK 同步会触碰文件系统 / 真实 DB，这里替成 no-op
        monkeypatch.setattr(
            ds_mod, "sync_ik_dictionaries", AsyncMock(return_value={})
        )
        return service

    @pytest.mark.asyncio
    async def test_add_terms_dedupes_by_word(
        self, service, dictionary_with_terms
    ):
        """重复 word 只保留首个，新 word 追加。"""
        result = await service.add_terms(
            dictionary_id=str(dictionary_with_terms.id),
            new_terms=[
                {"word": "水泥", "pos": "n", "weight": 99.0},  # 与现有重名
                {"word": "回转窑", "pos": "n", "weight": 1.0},
            ],
        )

        assert result is not None
        words = [t["word"] for t in result.terms]
        assert words.count("水泥") == 1
        assert "回转窑" in words
        # 现有 “水泥” 的 weight 不被覆盖（去重保留首个）
        cement = next(t for t in result.terms if t["word"] == "水泥")
        assert cement["weight"] == 1.0

    @pytest.mark.asyncio
    async def test_remove_terms_idempotent_on_missing(
        self, service, dictionary_with_terms
    ):
        """删除不存在的 word 不抛错，字典保持不变。"""
        result = await service.remove_terms(
            dictionary_id=str(dictionary_with_terms.id),
            words=["不存在的词", "也不存在"],
        )

        assert result is not None
        words = [t["word"] for t in result.terms]
        assert words == ["水泥", "熟料"]

    @pytest.mark.asyncio
    async def test_remove_terms_removes_only_matching(
        self, service, dictionary_with_terms
    ):
        """混合存在与不存在 word：只移除存在的，其它幂等无视。"""
        result = await service.remove_terms(
            dictionary_id=str(dictionary_with_terms.id),
            words=["水泥", "幽灵词"],
        )

        assert result is not None
        words = [t["word"] for t in result.terms]
        assert words == ["熟料"]
