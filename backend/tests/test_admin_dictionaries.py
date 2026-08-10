"""Admin Dictionaries CRUD API 集成测试（任务 13.2）。

覆盖 ``/api/admin/dictionaries`` 五个 CRUD 路由：

- ``POST /api/admin/dictionaries``：创建词典，验证 name 唯一性（409）、
  字段缺失（422）、术语校验失败（422），成功返回 201。
- ``GET /api/admin/dictionaries``：分页列表，含 ``total`` 和过滤参数。
- ``GET /api/admin/dictionaries/{id}``：单条查询，未找到返回 404，
  非法 UUID 返回 400。
- ``PUT /api/admin/dictionaries/{id}``：部分字段更新，name 冲突 409，
  未找到 404。
- ``DELETE /api/admin/dictionaries/{id}``：成功 204，未找到 404。
- 鉴权守门：所有路由要求 ``require_admin``，未登录 401，非管理员 403。

策略与 ``test_admin_reviews_list.py`` / ``test_admin_profiles.py`` 一致：
- FastAPI TestClient + ``dependency_overrides`` 注入 AsyncMock DB session。
- 通过覆盖 ``require_admin`` 依赖模拟「管理员 / 非管理员 / 未登录」三种场景。
- 对 ``DictionaryService`` 做 monkeypatch，避免连接真实 DB 与 IK 同步。
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
    """构造一个填满字段的 ``DomainDictionary`` ORM 实例。

    真实 DB 里 ``id`` / ``created_at`` / ``updated_at`` 由服务端默认值产生，
    测试里直接塞进对象即可让 ``DictionaryResponse`` Pydantic 校验通过。
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
    """``await db.execute(...)`` 的 mock，``.scalar_one_or_none()`` 返回 *value*。"""
    r = MagicMock()
    r.scalar_one_or_none.return_value = value
    return r


# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def mock_db() -> AsyncMock:
    """SQLAlchemy AsyncSession mock。``execute`` 在每个测试里按需 patch。"""
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
    """FastAPI app with ``require_admin`` overridden to return *admin_user*。"""
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
    """把 ``DictionaryService`` 整体替换为 MagicMock，在测试里按需配置返回值。

    返回 ``service`` 实例供 caller 配置（``service.create_dictionary.return_value =
    ...``）。``DictionaryService(db)`` 调用会被拦截并始终返回该实例。
    """
    service = MagicMock()
    # 默认所有 async 方法返回 None；具体测试里覆写。
    service.list_dictionaries = AsyncMock(return_value=([], 0))
    service.get_dictionary = AsyncMock(return_value=None)
    service.create_dictionary = AsyncMock(return_value=None)
    service.update_dictionary = AsyncMock(return_value=None)
    service.delete_dictionary = AsyncMock(return_value=False)

    def _factory(_db):
        return service

    monkeypatch.setattr(
        "app.api.admin_dictionaries.DictionaryService",
        _factory,
    )
    return service


# ─── Authorization ─────────────────────────────────────────────────────


class TestAuthorization:
    """``require_admin`` 守门：401 / 403 路径覆盖所有 5 个 CRUD 端点。"""

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

    @pytest.mark.parametrize(
        "method,path",
        [
            ("get", "/api/admin/dictionaries"),
            ("post", "/api/admin/dictionaries"),
            ("get", f"/api/admin/dictionaries/{uuid.uuid4()}"),
            ("put", f"/api/admin/dictionaries/{uuid.uuid4()}"),
            ("delete", f"/api/admin/dictionaries/{uuid.uuid4()}"),
        ],
    )
    def test_unauthenticated_returns_401(self, mock_db, method, path):
        """未登录访问任意 CRUD 端点 → 401。"""
        client = self._build_app_with_unauth(
            mock_db, UnauthorizedException("缺少认证令牌")
        )
        kwargs = {"json": {"name": "x"}} if method in ("post", "put") else {}
        response = getattr(client, method)(path, **kwargs)
        assert response.status_code == 401, (method, path, response.text)

    @pytest.mark.parametrize(
        "method,path",
        [
            ("get", "/api/admin/dictionaries"),
            ("post", "/api/admin/dictionaries"),
            ("get", f"/api/admin/dictionaries/{uuid.uuid4()}"),
            ("put", f"/api/admin/dictionaries/{uuid.uuid4()}"),
            ("delete", f"/api/admin/dictionaries/{uuid.uuid4()}"),
        ],
    )
    def test_non_admin_returns_403(self, mock_db, method, path):
        """普通用户访问任意 CRUD 端点 → 403。"""
        client = self._build_app_with_unauth(
            mock_db, ForbiddenException("需要管理员权限")
        )
        kwargs = {"json": {"name": "x"}} if method in ("post", "put") else {}
        response = getattr(client, method)(path, **kwargs)
        assert response.status_code == 403, (method, path, response.text)


# ─── POST /api/admin/dictionaries ──────────────────────────────────────


class TestCreate:
    """创建词典：201 + 唯一性 409 + 字段校验 422。"""

    def test_create_returns_201_with_minimal_payload(
        self, client, mock_db, patched_service
    ):
        """提供最小合法 payload，返回 201 并回显词典字段。"""
        new_id = uuid.uuid4()
        # name 唯一性预检查：先返回不存在
        mock_db.execute = AsyncMock(return_value=_scalar_result(None))
        patched_service.create_dictionary = AsyncMock(
            return_value=_build_dictionary(
                dict_id=new_id, name="测试词典", description=None
            )
        )

        response = client.post(
            "/api/admin/dictionaries",
            json={"name": "测试词典"},
        )

        assert response.status_code == 201, response.text
        body = response.json()
        assert body["id"] == str(new_id)
        assert body["name"] == "测试词典"
        assert body["terms"] == []
        assert body["synonyms"] == []
        assert body["stop_words"] == []
        assert body["enabled"] is True
        # 服务层被调用一次；唯一性预检查发生在它之前。
        patched_service.create_dictionary.assert_awaited_once()

    def test_create_with_full_payload_persists_all_fields(
        self, client, mock_db, patched_service
    ):
        """完整 payload：terms / synonyms / stop_words / description / enabled。"""
        new_id = uuid.uuid4()
        terms = [{"word": "大齿圈", "pos": "n", "weight": 1.0}]
        synonyms = [{"primary": "大齿圈", "synonyms": ["齿圈"]}]
        stop_words = ["的", "了"]
        mock_db.execute = AsyncMock(return_value=_scalar_result(None))
        patched_service.create_dictionary = AsyncMock(
            return_value=_build_dictionary(
                dict_id=new_id,
                name="水泥行业术语",
                description="水泥行业专业术语",
                terms=terms,
                synonyms=synonyms,
                stop_words=stop_words,
                enabled=False,
            )
        )

        response = client.post(
            "/api/admin/dictionaries",
            json={
                "name": "水泥行业术语",
                "description": "水泥行业专业术语",
                "terms": terms,
                "synonyms": synonyms,
                "stop_words": stop_words,
                "enabled": False,
            },
        )

        assert response.status_code == 201
        body = response.json()
        assert body["description"] == "水泥行业专业术语"
        assert body["terms"] == terms
        assert body["synonyms"] == synonyms
        assert body["stop_words"] == stop_words
        assert body["enabled"] is False

        call_kwargs = patched_service.create_dictionary.call_args.kwargs
        assert call_kwargs["name"] == "水泥行业术语"
        assert call_kwargs["description"] == "水泥行业专业术语"
        assert call_kwargs["terms"] == terms
        assert call_kwargs["synonyms"] == synonyms
        assert call_kwargs["stop_words"] == stop_words
        assert call_kwargs["enabled"] is False

    def test_create_duplicate_name_returns_409(
        self, client, mock_db, patched_service
    ):
        """name 已存在 → 唯一性预检查命中，返回 409，且不调用服务层。"""
        existing = _build_dictionary(name="重复词典")
        mock_db.execute = AsyncMock(return_value=_scalar_result(existing))

        response = client.post(
            "/api/admin/dictionaries",
            json={"name": "重复词典"},
        )

        assert response.status_code == 409
        assert "重复词典" in response.text
        patched_service.create_dictionary.assert_not_called()

    def test_create_missing_name_returns_422(self, client, mock_db, patched_service):
        """``name`` 是必填字段，缺失返回 422。"""
        mock_db.execute = AsyncMock()  # 不应触达 DB

        response = client.post("/api/admin/dictionaries", json={})

        assert response.status_code == 422
        # Pydantic 校验在依赖解析前完成；DB 与服务层均不应被调用。
        mock_db.execute.assert_not_called()
        patched_service.create_dictionary.assert_not_called()

    def test_create_blank_name_returns_422(self, client, mock_db, patched_service):
        """``name`` 为空字符串触发 ``min_length=1`` 校验，返回 422。"""
        mock_db.execute = AsyncMock()

        response = client.post("/api/admin/dictionaries", json={"name": ""})

        assert response.status_code == 422
        patched_service.create_dictionary.assert_not_called()

    def test_create_overlong_name_returns_422(
        self, client, mock_db, patched_service
    ):
        """``name`` 超过 100 字符 → 422。"""
        mock_db.execute = AsyncMock()

        response = client.post(
            "/api/admin/dictionaries",
            json={"name": "x" * 101},
        )

        assert response.status_code == 422
        patched_service.create_dictionary.assert_not_called()

    def test_create_invalid_term_returns_422(
        self, client, mock_db, patched_service
    ):
        """术语校验失败 → 服务层抛 ValueError → 422。

        ``TermSchema`` 已在 Pydantic 层挡住空串/超长 word，所以这里用合法的
        word 触发请求校验通过，但 mock 服务层在更深层校验时抛出 ValueError
        （例如包含控制字符等场景）。
        """
        mock_db.execute = AsyncMock(return_value=_scalar_result(None))
        patched_service.create_dictionary = AsyncMock(
            side_effect=ValueError("术语校验失败: 不能包含特殊控制字符")
        )

        response = client.post(
            "/api/admin/dictionaries",
            json={"name": "新词典", "terms": [{"word": "正常术语"}]},
        )

        assert response.status_code == 422
        assert "术语校验失败" in response.text


# ─── GET /api/admin/dictionaries ───────────────────────────────────────


class TestList:
    """列表接口：分页 + 过滤 + 空结果。"""

    def test_list_returns_dictionaries_with_total(
        self, client, patched_service
    ):
        d1 = _build_dictionary(name="词典 1")
        d2 = _build_dictionary(name="词典 2", enabled=False)
        patched_service.list_dictionaries = AsyncMock(return_value=([d1, d2], 2))

        response = client.get("/api/admin/dictionaries")

        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 2
        assert len(body["dictionaries"]) == 2
        names = [d["name"] for d in body["dictionaries"]]
        assert "词典 1" in names and "词典 2" in names

    def test_list_pagination_passes_skip_and_limit(
        self, client, patched_service
    ):
        """``skip`` / ``limit`` 直接透传给服务层。"""
        patched_service.list_dictionaries = AsyncMock(return_value=([], 42))

        response = client.get("/api/admin/dictionaries?skip=10&limit=5")

        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 42
        kwargs = patched_service.list_dictionaries.call_args.kwargs
        assert kwargs["skip"] == 10
        assert kwargs["limit"] == 5
        assert kwargs["enabled"] is None

    def test_list_filter_by_enabled(self, client, patched_service):
        patched_service.list_dictionaries = AsyncMock(return_value=([], 0))

        response = client.get("/api/admin/dictionaries?enabled=true")
        assert response.status_code == 200
        kwargs = patched_service.list_dictionaries.call_args.kwargs
        assert kwargs["enabled"] is True

    def test_list_invalid_pagination_returns_422(self, client):
        """``skip<0`` 或 ``limit>100`` 触发 Pydantic 校验 → 422。"""
        assert client.get("/api/admin/dictionaries?skip=-1").status_code == 422
        assert client.get("/api/admin/dictionaries?limit=0").status_code == 422
        assert client.get("/api/admin/dictionaries?limit=101").status_code == 422

    def test_list_empty_returns_zero_total(self, client, patched_service):
        patched_service.list_dictionaries = AsyncMock(return_value=([], 0))

        response = client.get("/api/admin/dictionaries")
        body = response.json()
        assert body["total"] == 0
        assert body["dictionaries"] == []


# ─── GET /api/admin/dictionaries/{id} ──────────────────────────────────


class TestGetSingle:
    """单条查询：成功 / 404 / 非法 UUID 400。"""

    def test_get_returns_dictionary(self, client, patched_service):
        dict_id = uuid.uuid4()
        patched_service.get_dictionary = AsyncMock(
            return_value=_build_dictionary(dict_id=dict_id, name="目标词典")
        )

        response = client.get(f"/api/admin/dictionaries/{dict_id}")

        assert response.status_code == 200
        body = response.json()
        assert body["id"] == str(dict_id)
        assert body["name"] == "目标词典"

    def test_get_not_found_returns_404(self, client, patched_service):
        patched_service.get_dictionary = AsyncMock(return_value=None)

        response = client.get(f"/api/admin/dictionaries/{uuid.uuid4()}")

        assert response.status_code == 404
        assert "Dictionary not found" in response.text

    def test_get_invalid_uuid_returns_400(self, client, patched_service):
        """非法 UUID 在路由层就被 ``_coerce_uuid`` 拦下 → 400。"""
        response = client.get("/api/admin/dictionaries/not-a-uuid")

        assert response.status_code == 400
        # 不应触达服务层
        patched_service.get_dictionary.assert_not_called()


# ─── PUT /api/admin/dictionaries/{id} ──────────────────────────────────


class TestUpdate:
    """更新词典：成功 / 404 / 唯一性 409 / 校验 422。"""

    def test_update_returns_updated_dictionary(
        self, client, mock_db, patched_service
    ):
        dict_id = uuid.uuid4()
        # name 唯一性检查：返回 None（无冲突）
        mock_db.execute = AsyncMock(return_value=_scalar_result(None))
        updated = _build_dictionary(
            dict_id=dict_id,
            name="新名字",
            description="改了",
            enabled=False,
        )
        patched_service.update_dictionary = AsyncMock(return_value=updated)

        response = client.put(
            f"/api/admin/dictionaries/{dict_id}",
            json={"name": "新名字", "description": "改了", "enabled": False},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["id"] == str(dict_id)
        assert body["name"] == "新名字"
        assert body["description"] == "改了"
        assert body["enabled"] is False

        call_kwargs = patched_service.update_dictionary.call_args.kwargs
        assert call_kwargs["dictionary_id"] == str(dict_id)
        assert call_kwargs["name"] == "新名字"
        assert call_kwargs["description"] == "改了"
        assert call_kwargs["enabled"] is False
        # terms / synonyms / stop_words 未提供时为 None（部分更新）
        assert call_kwargs["terms"] is None
        assert call_kwargs["synonyms"] is None
        assert call_kwargs["stop_words"] is None

    def test_update_partial_only_changes_provided_fields(
        self, client, mock_db, patched_service
    ):
        """只传 ``enabled=False`` 时，其它字段保持 None（不动）。"""
        dict_id = uuid.uuid4()
        # 没改 name，跳过唯一性检查（路由不会调用 db.execute）
        mock_db.execute = AsyncMock()
        patched_service.update_dictionary = AsyncMock(
            return_value=_build_dictionary(dict_id=dict_id, enabled=False)
        )

        response = client.put(
            f"/api/admin/dictionaries/{dict_id}",
            json={"enabled": False},
        )

        assert response.status_code == 200
        kwargs = patched_service.update_dictionary.call_args.kwargs
        assert kwargs["name"] is None
        assert kwargs["description"] is None
        assert kwargs["enabled"] is False
        # 因为没改 name，唯一性检查不应跑
        mock_db.execute.assert_not_called()

    def test_update_duplicate_name_returns_409(
        self, client, mock_db, patched_service
    ):
        """改 name 时撞库 → 409，且不调用服务层。"""
        dict_id = uuid.uuid4()
        existing = _build_dictionary(name="已有名字")
        mock_db.execute = AsyncMock(return_value=_scalar_result(existing))

        response = client.put(
            f"/api/admin/dictionaries/{dict_id}",
            json={"name": "已有名字"},
        )

        assert response.status_code == 409
        assert "已有名字" in response.text
        patched_service.update_dictionary.assert_not_called()

    def test_update_not_found_returns_404(
        self, client, mock_db, patched_service
    ):
        mock_db.execute = AsyncMock(return_value=_scalar_result(None))
        patched_service.update_dictionary = AsyncMock(return_value=None)

        response = client.put(
            f"/api/admin/dictionaries/{uuid.uuid4()}",
            json={"name": "随便"},
        )

        assert response.status_code == 404

    def test_update_invalid_term_returns_422(
        self, client, mock_db, patched_service
    ):
        """服务层 term 校验失败 → ValueError → 422。

        ``TermSchema`` 已在请求层挡住超长 word，本测试用合法 word 触发
        服务层更深层校验失败（例如控制字符）。
        """
        mock_db.execute = AsyncMock()  # 没改 name，唯一性检查不跑
        patched_service.update_dictionary = AsyncMock(
            side_effect=ValueError("术语校验失败: 不能包含特殊控制字符")
        )

        response = client.put(
            f"/api/admin/dictionaries/{uuid.uuid4()}",
            json={"terms": [{"word": "合法术语"}]},
        )

        assert response.status_code == 422
        assert "术语校验失败" in response.text

    def test_update_invalid_uuid_returns_400(self, client, patched_service):
        response = client.put(
            "/api/admin/dictionaries/not-a-uuid",
            json={"name": "x"},
        )

        assert response.status_code == 400
        patched_service.update_dictionary.assert_not_called()


# ─── DELETE /api/admin/dictionaries/{id} ──────────────────────────────


class TestDelete:
    """删除词典：成功 204 / 404 / 非法 UUID 400。"""

    def test_delete_returns_204_on_success(self, client, patched_service):
        patched_service.delete_dictionary = AsyncMock(return_value=True)

        response = client.delete(f"/api/admin/dictionaries/{uuid.uuid4()}")

        assert response.status_code == 204
        # 204 不应有 body
        assert response.content == b""

    def test_delete_not_found_returns_404(self, client, patched_service):
        patched_service.delete_dictionary = AsyncMock(return_value=False)

        response = client.delete(f"/api/admin/dictionaries/{uuid.uuid4()}")

        assert response.status_code == 404
        assert "Dictionary not found" in response.text

    def test_delete_invalid_uuid_returns_400(self, client, patched_service):
        response = client.delete("/api/admin/dictionaries/not-a-uuid")

        assert response.status_code == 400
        patched_service.delete_dictionary.assert_not_called()
