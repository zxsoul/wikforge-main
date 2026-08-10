"""Admin Dictionaries 导入/导出 API 集成测试（任务 13.5）。

覆盖 ``/api/admin/dictionaries/{id}/{export,import}/{json,csv}`` 四个路由：

- ``GET .../export/json``：返回完整字典字段（name/description/terms/
  synonyms/stop_words/enabled），``application/json``。
- ``GET .../export/csv``：``text/csv`` + ``Content-Disposition: attachment``，
  内容首行是 ``word,pos,weight`` 表头，后续是 terms 各行。
- ``POST .../import/json``：把传入的 terms/synonyms/stop_words 与现有
  字典合并（按 word / primary / 字符串去重，不覆盖现有项）。
- ``POST .../import/csv``：multipart 上传 CSV 文件，按行解析后合并 terms；
  非 UTF-8 编码返回 400；单行格式错误（如 weight 非数）跳过该行不影响整体。
- 鉴权守门：四个路由都要求 ``require_admin``，未登录 401，非管理员 403。
- 路径段非法 UUID 返回 400；字典缺失 404。

策略与 ``test_admin_dictionaries_terms.py`` / ``test_admin_dictionaries_synonyms.py``
保持一致：TestClient + ``dependency_overrides`` 注入 mock DB session，
通过 ``patched_service`` monkeypatch ``DictionaryService`` 验证路由层契约
（参数透传、Content-Type、合并语义）。CSV 解析等服务层语义另写一组真实
``DictionaryService`` 子测试。
"""

from __future__ import annotations

import csv
import io
import json
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
    """把 ``DictionaryService`` 替换为 MagicMock。

    路由对服务层的依赖：
    - ``get_dictionary``：返回 ORM 实例或 None（None → 404）
    - ``export_as_json`` / ``export_as_csv``：导出格式化（同步）
    - ``import_from_json`` / ``import_from_csv``：解析输入（同步）
    - ``update_dictionary``：合并后写回（异步）
    """
    service = MagicMock()
    service.get_dictionary = AsyncMock(return_value=None)
    service.export_as_json = MagicMock(return_value={})
    service.export_as_csv = MagicMock(return_value="")
    service.import_from_json = MagicMock(
        return_value={"terms": [], "synonyms": [], "stop_words": []}
    )
    service.import_from_csv = MagicMock(return_value=[])
    service.update_dictionary = AsyncMock(return_value=None)

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

    @pytest.mark.parametrize(
        "method,suffix",
        [
            ("GET", "export/json"),
            ("GET", "export/csv"),
            ("POST", "import/json"),
            ("POST", "import/csv"),
        ],
    )
    def test_unauthenticated_returns_401(self, mock_db, method, suffix):
        client = self._build_app_with_unauth(
            mock_db, UnauthorizedException("缺少认证令牌")
        )
        path = f"/api/admin/dictionaries/{uuid.uuid4()}/{suffix}"
        kwargs = {}
        if suffix == "import/json":
            kwargs["json"] = {}
        elif suffix == "import/csv":
            kwargs["files"] = {"file": ("a.csv", b"word\n", "text/csv")}
        response = client.request(method, path, **kwargs)
        assert response.status_code == 401, (suffix, response.text)

    @pytest.mark.parametrize(
        "method,suffix",
        [
            ("GET", "export/json"),
            ("GET", "export/csv"),
            ("POST", "import/json"),
            ("POST", "import/csv"),
        ],
    )
    def test_non_admin_returns_403(self, mock_db, method, suffix):
        client = self._build_app_with_unauth(
            mock_db, ForbiddenException("需要管理员权限")
        )
        path = f"/api/admin/dictionaries/{uuid.uuid4()}/{suffix}"
        kwargs = {}
        if suffix == "import/json":
            kwargs["json"] = {}
        elif suffix == "import/csv":
            kwargs["files"] = {"file": ("a.csv", b"word\n", "text/csv")}
        response = client.request(method, path, **kwargs)
        assert response.status_code == 403, (suffix, response.text)


# ─── GET /api/admin/dictionaries/{id}/export/json ─────────────────────


class TestExportJson:
    """JSON 导出：200 包含完整字段 / 404 / 400。"""

    def test_export_json_returns_full_dictionary(self, client, patched_service):
        dict_id = uuid.uuid4()
        terms = [{"word": "水泥", "pos": "n", "weight": 1.0}]
        synonyms = [{"primary": "大齿圈", "synonyms": ["齿圈", "主齿圈"]}]
        stop_words = ["的", "了"]
        dictionary = _build_dictionary(
            dict_id=dict_id,
            name="水泥行业术语",
            description="示例词典",
            terms=terms,
            synonyms=synonyms,
            stop_words=stop_words,
            enabled=True,
        )
        patched_service.get_dictionary = AsyncMock(return_value=dictionary)
        # 路由调用 ``service.export_as_json`` 拿到导出体
        patched_service.export_as_json = MagicMock(
            return_value={
                "name": dictionary.name,
                "description": dictionary.description,
                "terms": terms,
                "synonyms": synonyms,
                "stop_words": stop_words,
                "enabled": True,
            }
        )

        response = client.get(
            f"/api/admin/dictionaries/{dict_id}/export/json"
        )

        assert response.status_code == 200, response.text
        # FastAPI 默认使用 application/json
        assert response.headers["content-type"].startswith("application/json")

        body = response.json()
        # 完整字段都在
        assert body["name"] == "水泥行业术语"
        assert body["description"] == "示例词典"
        assert body["terms"] == terms
        assert body["synonyms"] == synonyms
        assert body["stop_words"] == stop_words
        assert body["enabled"] is True

        # 服务层用 ORM 实例调用一次 export_as_json
        patched_service.export_as_json.assert_called_once_with(dictionary)
        # 透传 dictionary_id（字符串）给 get_dictionary
        assert patched_service.get_dictionary.call_args.args[0] == str(dict_id)

    def test_export_json_not_found_returns_404(self, client, patched_service):
        patched_service.get_dictionary = AsyncMock(return_value=None)

        response = client.get(
            f"/api/admin/dictionaries/{uuid.uuid4()}/export/json"
        )

        assert response.status_code == 404
        assert "Dictionary not found" in response.text
        patched_service.export_as_json.assert_not_called()

    def test_export_json_invalid_uuid_returns_400(self, client, patched_service):
        response = client.get(
            "/api/admin/dictionaries/not-a-uuid/export/json"
        )

        assert response.status_code == 400
        patched_service.get_dictionary.assert_not_called()


# ─── GET /api/admin/dictionaries/{id}/export/csv ──────────────────────


class TestExportCsv:
    """CSV 导出：200 + 表头/行 / 404 / 400。"""

    def test_export_csv_returns_header_and_rows(self, client, patched_service):
        dict_id = uuid.uuid4()
        terms = [
            {"word": "水泥", "pos": "n", "weight": 1.0},
            {"word": "熟料", "pos": "n", "weight": 2.5},
        ]
        dictionary = _build_dictionary(dict_id=dict_id, terms=terms)
        patched_service.get_dictionary = AsyncMock(return_value=dictionary)
        # 用真实 csv 模块产出表头 + 行，确保契约可被解析回 dict
        csv_content = "word,pos,weight\r\n水泥,n,1.0\r\n熟料,n,2.5\r\n"
        patched_service.export_as_csv = MagicMock(return_value=csv_content)

        response = client.get(
            f"/api/admin/dictionaries/{dict_id}/export/csv"
        )

        assert response.status_code == 200, response.text
        # Content-Type 是 text/csv（PlainTextResponse 设的 media_type）
        assert response.headers["content-type"].startswith("text/csv")
        # Attachment 头：ASCII filename 兜底 + RFC 5987 filename* 包含原始
        # 中文名（百分号编码），整个 header 必须是 latin-1 可编码。
        disposition = response.headers["content-disposition"]
        assert "attachment" in disposition
        assert "filename=" in disposition
        # RFC 5987 形式：filename*=UTF-8'' 后跟百分号编码的中文
        assert "filename*=UTF-8''" in disposition
        # 验证百分号编码后还能 round-trip 回原始中文名
        from urllib.parse import unquote

        # 取 filename*=UTF-8'' 后面那段（到行尾或 ;）
        marker = "filename*=UTF-8''"
        encoded_part = disposition.split(marker, 1)[1].split(";", 1)[0]
        assert unquote(encoded_part) == "水泥行业术语.csv"

        body = response.text
        # 解析回来：第一行表头、之后是数据行
        reader = csv.reader(io.StringIO(body))
        rows = [r for r in reader if r]
        assert rows[0] == ["word", "pos", "weight"]
        assert rows[1] == ["水泥", "n", "1.0"]
        assert rows[2] == ["熟料", "n", "2.5"]

        patched_service.export_as_csv.assert_called_once_with(dictionary)

    def test_export_csv_not_found_returns_404(self, client, patched_service):
        patched_service.get_dictionary = AsyncMock(return_value=None)

        response = client.get(
            f"/api/admin/dictionaries/{uuid.uuid4()}/export/csv"
        )

        assert response.status_code == 404
        patched_service.export_as_csv.assert_not_called()

    def test_export_csv_invalid_uuid_returns_400(self, client, patched_service):
        response = client.get(
            "/api/admin/dictionaries/not-a-uuid/export/csv"
        )

        assert response.status_code == 400
        patched_service.get_dictionary.assert_not_called()


# ─── POST /api/admin/dictionaries/{id}/import/json ────────────────────


class TestImportJson:
    """JSON 导入：200 合并不重复 / 404 / 400 / 422。"""

    def test_import_json_merges_with_existing(self, client, patched_service):
        """合并语义：现有 word/primary/stop_word 不被覆盖，新项追加。"""
        dict_id = uuid.uuid4()
        existing = _build_dictionary(
            dict_id=dict_id,
            terms=[{"word": "水泥", "pos": "n", "weight": 1.0}],
            synonyms=[{"primary": "大齿圈", "synonyms": ["齿圈"]}],
            stop_words=["的"],
        )
        patched_service.get_dictionary = AsyncMock(return_value=existing)
        # import_from_json 直接回显（路由层负责合并）
        payload = {
            "terms": [
                {"word": "水泥", "pos": "n", "weight": 99.0},  # 与现有重名
                {"word": "熟料", "pos": "n", "weight": 1.0},
            ],
            "synonyms": [
                # 与现有 primary 重名
                {"primary": "大齿圈", "synonyms": ["主齿圈"]},
                # 新组
                {"primary": "回转窑", "synonyms": ["窑炉"]},
            ],
            "stop_words": ["的", "了"],
        }
        patched_service.import_from_json = MagicMock(return_value=payload)

        # 最终 update_dictionary 接收合并结果，回显为更新后的字典
        merged_terms = [
            {"word": "水泥", "pos": "n", "weight": 1.0},  # 保留首个
            {"word": "熟料", "pos": "n", "weight": 1.0},
        ]
        merged_synonyms = [
            {"primary": "大齿圈", "synonyms": ["齿圈"]},  # 保留首个
            {"primary": "回转窑", "synonyms": ["窑炉"]},
        ]
        merged_stop_words = ["的", "了"]
        patched_service.update_dictionary = AsyncMock(
            return_value=_build_dictionary(
                dict_id=dict_id,
                terms=merged_terms,
                synonyms=merged_synonyms,
                stop_words=merged_stop_words,
            )
        )

        response = client.post(
            f"/api/admin/dictionaries/{dict_id}/import/json",
            json=payload,
        )

        assert response.status_code == 200, response.text
        body = response.json()
        # 词不重复
        words = [t["word"] for t in body["terms"]]
        assert words.count("水泥") == 1
        assert "熟料" in words
        # primary 不重复
        primaries = [sg["primary"] for sg in body["synonyms"]]
        assert primaries.count("大齿圈") == 1
        assert "回转窑" in primaries
        # stop_words 不重复
        assert body["stop_words"].count("的") == 1
        assert "了" in body["stop_words"]

        # update_dictionary 被传入合并后的列表（路由层把现有 + 新增合并）
        kwargs = patched_service.update_dictionary.call_args.kwargs
        assert kwargs["dictionary_id"] == str(dict_id)
        assert [t["word"] for t in kwargs["terms"]] == ["水泥", "熟料"]
        # 现有 “水泥” 的 weight 不被新值（99.0）覆盖
        assert kwargs["terms"][0]["weight"] == 1.0
        assert [sg["primary"] for sg in kwargs["synonyms"]] == [
            "大齿圈",
            "回转窑",
        ]
        # 现有 “大齿圈” 的 synonyms 不被新值（["主齿圈"]）覆盖
        assert kwargs["synonyms"][0]["synonyms"] == ["齿圈"]
        assert kwargs["stop_words"] == ["的", "了"]

    def test_import_json_not_found_returns_404(self, client, patched_service):
        patched_service.get_dictionary = AsyncMock(return_value=None)

        response = client.post(
            f"/api/admin/dictionaries/{uuid.uuid4()}/import/json",
            json={"terms": [], "synonyms": [], "stop_words": []},
        )

        assert response.status_code == 404
        patched_service.import_from_json.assert_not_called()
        patched_service.update_dictionary.assert_not_called()

    def test_import_json_invalid_uuid_returns_400(
        self, client, patched_service
    ):
        response = client.post(
            "/api/admin/dictionaries/not-a-uuid/import/json",
            json={"terms": [], "synonyms": [], "stop_words": []},
        )

        assert response.status_code == 400
        patched_service.get_dictionary.assert_not_called()

    def test_import_json_service_validation_failure_returns_422(
        self, client, patched_service
    ):
        """update_dictionary 抛 ``ValueError`` 时映射成 422，与 CRUD 一致。"""
        dict_id = uuid.uuid4()
        patched_service.get_dictionary = AsyncMock(
            return_value=_build_dictionary(dict_id=dict_id)
        )
        patched_service.import_from_json = MagicMock(
            return_value={
                "terms": [{"word": "x"}],
                "synonyms": [],
                "stop_words": [],
            }
        )
        patched_service.update_dictionary = AsyncMock(
            side_effect=ValueError("术语校验失败: 不能包含特殊控制字符")
        )

        response = client.post(
            f"/api/admin/dictionaries/{dict_id}/import/json",
            json={
                "terms": [{"word": "x"}],
                "synonyms": [],
                "stop_words": [],
            },
        )

        assert response.status_code == 422
        assert "术语校验失败" in response.text


# ─── POST /api/admin/dictionaries/{id}/import/csv ─────────────────────


class TestImportCsv:
    """CSV 导入（multipart）：200 合并 / 404 / 400 / 编码错误 400。"""

    def test_import_csv_from_multipart_merges_terms(
        self, client, patched_service
    ):
        dict_id = uuid.uuid4()
        existing = _build_dictionary(
            dict_id=dict_id,
            terms=[{"word": "水泥", "pos": "n", "weight": 1.0}],
            synonyms=[{"primary": "大齿圈", "synonyms": ["齿圈"]}],
            stop_words=["的"],
        )
        patched_service.get_dictionary = AsyncMock(return_value=existing)
        # CSV 解析后只产出 terms（无同义词/停用词）
        parsed_terms = [
            {"word": "水泥", "pos": "n", "weight": 99.0},  # 重复
            {"word": "熟料", "pos": "n", "weight": 1.0},
        ]
        patched_service.import_from_csv = MagicMock(return_value=parsed_terms)

        merged = [
            {"word": "水泥", "pos": "n", "weight": 1.0},  # 保留首个
            {"word": "熟料", "pos": "n", "weight": 1.0},
        ]
        patched_service.update_dictionary = AsyncMock(
            return_value=_build_dictionary(
                dict_id=dict_id,
                terms=merged,
                # synonyms / stop_words 不被 CSV 导入触碰
                synonyms=existing.synonyms,
                stop_words=existing.stop_words,
            )
        )

        csv_bytes = "word,pos,weight\n水泥,n,99.0\n熟料,n,1.0\n".encode(
            "utf-8"
        )
        response = client.post(
            f"/api/admin/dictionaries/{dict_id}/import/csv",
            files={"file": ("terms.csv", csv_bytes, "text/csv")},
        )

        assert response.status_code == 200, response.text
        body = response.json()
        words = [t["word"] for t in body["terms"]]
        assert words == ["水泥", "熟料"]
        # 同义词/停用词保持原样（CSV 导入不影响这些字段）
        assert body["synonyms"] == [
            {"primary": "大齿圈", "synonyms": ["齿圈"]}
        ]
        assert body["stop_words"] == ["的"]

        # 路由把解析出的 terms 与现有合并后传给 update_dictionary
        kwargs = patched_service.update_dictionary.call_args.kwargs
        assert kwargs["dictionary_id"] == str(dict_id)
        assert [t["word"] for t in kwargs["terms"]] == ["水泥", "熟料"]
        # 重复 word 保留首个的 weight
        assert kwargs["terms"][0]["weight"] == 1.0
        # 仅 terms 被传入，synonyms / stop_words 不在更新范围里
        assert "synonyms" not in kwargs or kwargs.get("synonyms") is None
        assert "stop_words" not in kwargs or kwargs.get("stop_words") is None

        # 服务层 import_from_csv 收到 utf-8 解码后的字符串
        decoded = patched_service.import_from_csv.call_args.args[0]
        assert "水泥,n,99.0" in decoded
        assert "熟料,n,1.0" in decoded

    def test_import_csv_not_found_returns_404(self, client, patched_service):
        patched_service.get_dictionary = AsyncMock(return_value=None)

        response = client.post(
            f"/api/admin/dictionaries/{uuid.uuid4()}/import/csv",
            files={"file": ("a.csv", b"word\n", "text/csv")},
        )

        assert response.status_code == 404
        patched_service.import_from_csv.assert_not_called()
        patched_service.update_dictionary.assert_not_called()

    def test_import_csv_invalid_uuid_returns_400(self, client, patched_service):
        response = client.post(
            "/api/admin/dictionaries/not-a-uuid/import/csv",
            files={"file": ("a.csv", b"word\n", "text/csv")},
        )

        assert response.status_code == 400
        patched_service.get_dictionary.assert_not_called()

    def test_import_csv_non_utf8_returns_400(self, client, patched_service):
        """CSV 必须 UTF-8 编码；GBK/Latin-1 等触发 400 而不是 500。"""
        dict_id = uuid.uuid4()
        patched_service.get_dictionary = AsyncMock(
            return_value=_build_dictionary(dict_id=dict_id)
        )

        # GBK 编码包含非 UTF-8 字节序列
        gbk_bytes = "word,pos,weight\n水泥,n,1.0\n".encode("gbk")
        response = client.post(
            f"/api/admin/dictionaries/{dict_id}/import/csv",
            files={"file": ("a.csv", gbk_bytes, "text/csv")},
        )

        assert response.status_code == 400, response.text
        assert "UTF-8" in response.text
        patched_service.import_from_csv.assert_not_called()
        patched_service.update_dictionary.assert_not_called()

    def test_import_csv_missing_file_returns_422(self, client, patched_service):
        """缺少 file 字段，FastAPI 返回 422 校验错误。"""
        response = client.post(
            f"/api/admin/dictionaries/{uuid.uuid4()}/import/csv",
        )

        assert response.status_code == 422
        patched_service.get_dictionary.assert_not_called()


# ─── 服务层语义验证（不 mock 服务层） ───────────────────────────────


class TestServiceLevelSemantics:
    """直接验证 ``DictionaryService`` 的导入/导出契约。

    路由层在上面已被覆盖；这里走真实服务 + mock DB 对 export_as_json /
    export_as_csv / import_from_json / import_from_csv 的语义做兜底，
    保证：
    - export_as_json 包含 design.md 列出的所有字段
    - export_as_csv 第一行是固定表头
    - import_from_csv 单行格式错误（如 weight 非数）跳过该行
    - import_from_json 跳过非法术语（含控制字符）但保留同义词/停用词
    """

    @pytest.fixture
    def service(self, mock_db):
        from app.services.dictionary_service import DictionaryService

        return DictionaryService(mock_db)

    @pytest.fixture
    def filled_dictionary(self):
        return _build_dictionary(
            terms=[
                {"word": "水泥", "pos": "n", "weight": 1.0},
                {"word": "熟料", "pos": "n", "weight": 2.0},
            ],
            synonyms=[
                {"primary": "大齿圈", "synonyms": ["齿圈", "主齿圈"]},
            ],
            stop_words=["的", "了"],
        )

    def test_export_as_json_contains_all_fields(self, service, filled_dictionary):
        exported = service.export_as_json(filled_dictionary)

        # design.md 列出的字段：name/description/terms/synonyms/stop_words/enabled
        assert set(exported.keys()) == {
            "name",
            "description",
            "terms",
            "synonyms",
            "stop_words",
            "enabled",
        }
        assert exported["name"] == filled_dictionary.name
        assert exported["description"] == filled_dictionary.description
        assert exported["terms"] == filled_dictionary.terms
        assert exported["synonyms"] == filled_dictionary.synonyms
        assert exported["stop_words"] == filled_dictionary.stop_words
        assert exported["enabled"] is True
        # JSON 可序列化（不抛 TypeError）
        json.dumps(exported, ensure_ascii=False)

    def test_export_as_csv_has_header_and_rows(self, service, filled_dictionary):
        csv_content = service.export_as_csv(filled_dictionary)

        rows = [r for r in csv.reader(io.StringIO(csv_content)) if r]
        # 第一行是固定表头
        assert rows[0] == ["word", "pos", "weight"]
        # 之后两行对应 filled_dictionary.terms 顺序
        assert rows[1] == ["水泥", "n", "1.0"]
        assert rows[2] == ["熟料", "n", "2.0"]

    def test_import_from_csv_skips_malformed_row(self, service):
        """单行 weight 非数值不应让整体导入失败：跳过该行，其它行保留。"""
        csv_content = (
            "word,pos,weight\n"
            "水泥,n,1.0\n"
            "熟料,n,not-a-number\n"  # 这一行 weight 不合法
            "回转窑,n,2.0\n"
        )

        terms = service.import_from_csv(csv_content)

        words = [t["word"] for t in terms]
        # 中间那行被跳过，其它保留
        assert "水泥" in words
        assert "回转窑" in words
        assert "熟料" not in words

    def test_import_from_csv_skips_invalid_term(self, service):
        """单行 word 非法（含控制字符 / 超长）也被跳过。"""
        csv_content = (
            "word,pos,weight\n"
            "水泥,n,1.0\n"
            f"{'x' * 31},n,1.0\n"  # 超长，validate_term 拒绝
            "熟料,n,1.0\n"
        )

        terms = service.import_from_csv(csv_content)

        words = [t["word"] for t in terms]
        assert words == ["水泥", "熟料"]

    def test_import_from_json_skips_invalid_term(self, service):
        """非法 word 被丢弃，但 synonyms / stop_words 原样保留。"""
        payload = {
            "terms": [
                {"word": "水泥"},
                {"word": "含\x01控制字符"},  # validate_term 拒绝
                {"word": "x" * 31},  # 超长，拒绝
                "熟料",  # 字符串形式也支持
            ],
            "synonyms": [{"primary": "大齿圈", "synonyms": ["齿圈"]}],
            "stop_words": ["的"],
        }

        result = service.import_from_json(payload)

        words = [t["word"] for t in result["terms"]]
        assert words == ["水泥", "熟料"]
        assert result["synonyms"] == [
            {"primary": "大齿圈", "synonyms": ["齿圈"]}
        ]
        assert result["stop_words"] == ["的"]
