"""Admin Profile API 集成测试（任务 8.9 - 8.14）。

覆盖：
- CRUD：create / get / list / update / delete
- name 唯一性 → 409
- 缺失/非法字段 → 422
- 删除 generic-text → 400
- toggle 启用/禁用；禁用 generic-text → 400
- export 返回所有 profile
- import 新建 + 更新（按 name 匹配）
- 版本历史：每次 update 写入 ProfileVersion
- 预览：mock parser & matcher，返回 blocks + features

策略：
- FastAPI TestClient + dependency_overrides 注入 AsyncMock DB session
- monkeypatch ``_get_admin_user_id`` 跳过真实查询
- 不连接真实 DB / Qdrant / 解析器
"""

from __future__ import annotations

import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.admin_profiles import router as admin_profiles_router
from app.core.database import get_db
from app.core.exceptions import register_exception_handlers
from app.models.document_profile import DocumentProfile
from app.models.profile_version import ProfileVersion
from app.services.parsers.base import Block, ParsedDocument


# ─── Mock helpers ──────────────────────────────────────────────────────


def scalar_result(value):
    """Mock for ``await db.execute(...)``; ``.scalar_one_or_none()`` returns *value*."""
    r = MagicMock()
    r.scalar_one_or_none.return_value = value
    return r


def scalars_all_result(values):
    """Mock for ``(await db.execute(...)).scalars().all()`` returning *values*."""
    r = MagicMock()
    scalars = MagicMock()
    scalars.all.return_value = list(values)
    r.scalars.return_value = scalars
    return r


def build_profile(
    *,
    name: str = "custom-profile",
    priority: int = 5,
    enabled: bool = True,
    version: int = 1,
    profile_id: uuid.UUID | None = None,
    description: str | None = "示例 Profile",
) -> DocumentProfile:
    """Build a fully-populated ``DocumentProfile`` ORM instance for tests.

    所有字段都被预填充以避免 Pydantic 校验失败（``id`` / ``created_at`` /
    ``updated_at`` 在真实 DB 中由服务端默认产生，测试里手动塞值即可）。
    """
    p = DocumentProfile(
        name=name,
        description=description,
        priority=priority,
        enabled=enabled,
        match_rules={"filename_regex": [], "content_regex": [], "min_content_match_count": 1},
        heading_rules=[],
        boilerplate={"detection_mode": "statistical", "statistical_threshold": 0.5, "manual_patterns": []},
        tables={"cross_page_merge": True, "row_level_chunking": False, "collapse_merged_cells": "describe"},
        chunking={
            "min_tokens": 256,
            "max_tokens": 800,
            "overlap_tokens": 80,
            "respect_heading_level": 1,
            "protect_patterns": [],
        },
        version=version,
    )
    p.id = profile_id or uuid.uuid4()
    p.created_at = datetime(2024, 1, 1, 12, 0, 0)
    p.updated_at = datetime(2024, 6, 1, 12, 0, 0)
    p.domain_dictionary_id = None
    return p


# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def mock_db() -> AsyncMock:
    """AsyncSession mock。

    ``refresh`` 的 side_effect 模拟 SQLAlchemy 在 flush 后填充服务端默认值
    （id / 时间戳 / version），让 ``ProfileResponse`` 校验能通过。
    """
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.delete = AsyncMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()

    async def fake_refresh(obj):
        if getattr(obj, "id", None) is None:
            obj.id = uuid.uuid4()
        if getattr(obj, "created_at", None) is None:
            obj.created_at = datetime(2024, 1, 1, 0, 0, 0)
        if getattr(obj, "updated_at", None) is None:
            obj.updated_at = datetime(2024, 1, 1, 0, 0, 0)
        if getattr(obj, "version", None) is None:
            obj.version = 1
        if getattr(obj, "domain_dictionary_id", None) is None:
            obj.domain_dictionary_id = None

    db.refresh = AsyncMock(side_effect=fake_refresh)
    return db


@pytest.fixture
def app(mock_db: AsyncMock) -> FastAPI:
    application = FastAPI()
    register_exception_handlers(application)
    application.include_router(admin_profiles_router)

    async def _override_get_db():
        yield mock_db

    application.dependency_overrides[get_db] = _override_get_db
    return application


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def stub_admin_user(monkeypatch):
    """默认让 ``_get_admin_user_id`` 返回固定 UUID，使版本快照路径可执行。

    个别测试可重新打桩为 ``None`` 来跳过快照，或者改成抛异常验证健壮性。
    """
    admin_uid = uuid.uuid4()
    monkeypatch.setattr(
        "app.api.admin_profiles._get_admin_user_id",
        AsyncMock(return_value=admin_uid),
    )
    return admin_uid


# ─── CRUD: list / get ──────────────────────────────────────────────────


class TestListProfiles:
    """``GET /api/admin/profiles``"""

    def test_list_returns_profiles_with_total(self, client, mock_db):
        profiles = [build_profile(name="a", priority=10), build_profile(name="b", priority=0)]
        # 第一次 execute 用作 count，第二次用作分页
        mock_db.execute = AsyncMock(
            side_effect=[scalars_all_result(profiles), scalars_all_result(profiles)]
        )

        response = client.get("/api/admin/profiles")
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 2
        assert len(body["profiles"]) == 2
        assert {p["name"] for p in body["profiles"]} == {"a", "b"}

    def test_list_filters_by_enabled(self, client, mock_db):
        profiles = [build_profile(name="enabled-only", enabled=True)]
        mock_db.execute = AsyncMock(
            side_effect=[scalars_all_result(profiles), scalars_all_result(profiles)]
        )

        response = client.get("/api/admin/profiles?enabled=true")
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 1
        assert body["profiles"][0]["enabled"] is True


class TestGetProfile:
    """``GET /api/admin/profiles/{id}``"""

    def test_get_existing(self, client, mock_db):
        p = build_profile(name="existing")
        mock_db.execute = AsyncMock(return_value=scalar_result(p))

        response = client.get(f"/api/admin/profiles/{p.id}")
        assert response.status_code == 200
        assert response.json()["name"] == "existing"

    def test_get_not_found(self, client, mock_db):
        mock_db.execute = AsyncMock(return_value=scalar_result(None))
        response = client.get(f"/api/admin/profiles/{uuid.uuid4()}")
        assert response.status_code == 404


# ─── CRUD: create ──────────────────────────────────────────────────────


class TestCreateProfile:
    """``POST /api/admin/profiles``"""

    def test_create_minimal(self, client, mock_db):
        # 唯一性检查 → None
        mock_db.execute = AsyncMock(return_value=scalar_result(None))

        payload = {"name": "new-profile", "description": "新建测试", "priority": 3}
        response = client.post("/api/admin/profiles", json=payload)
        assert response.status_code == 201, response.text
        body = response.json()
        assert body["name"] == "new-profile"
        assert body["description"] == "新建测试"
        assert body["priority"] == 3
        assert body["version"] == 1

        # add 至少被调用一次（profile + 初始 ProfileVersion 快照）
        added_objs = [c.args[0] for c in mock_db.add.call_args_list]
        assert any(isinstance(o, DocumentProfile) for o in added_objs)
        assert any(isinstance(o, ProfileVersion) for o in added_objs)

    def test_create_with_full_config(self, client, mock_db):
        mock_db.execute = AsyncMock(return_value=scalar_result(None))

        payload = {
            "name": "full-profile",
            "description": "完整配置",
            "priority": 10,
            "enabled": True,
            "match_rules": {
                "filename_regex": [r".*\.pdf"],
                "content_regex": [r"^Chapter"],
                "min_content_match_count": 1,
            },
            "heading_rules": [{"pattern": r"^#\s+", "level": 1, "strip_pattern": False}],
            "boilerplate": {
                "detection_mode": "manual",
                "statistical_threshold": 0.5,
                "manual_patterns": [r"^Page \d+$"],
            },
            "tables": {
                "cross_page_merge": False,
                "row_level_chunking": True,
                "collapse_merged_cells": "repeat",
            },
            "chunking": {
                "min_tokens": 128,
                "max_tokens": 1024,
                "overlap_tokens": 64,
                "respect_heading_level": 2,
                "protect_patterns": [r"\d+kg"],
            },
        }
        response = client.post("/api/admin/profiles", json=payload)
        assert response.status_code == 201
        body = response.json()
        assert body["match_rules"]["filename_regex"] == [r".*\.pdf"]
        assert body["heading_rules"][0]["pattern"] == r"^#\s+"
        assert body["chunking"]["max_tokens"] == 1024

    def test_create_duplicate_name_returns_409(self, client, mock_db):
        existing = build_profile(name="dup")
        mock_db.execute = AsyncMock(return_value=scalar_result(existing))

        response = client.post("/api/admin/profiles", json={"name": "dup"})
        assert response.status_code == 409
        assert "dup" in response.text

    def test_create_missing_name_returns_422(self, client, mock_db):
        # 不必 stub execute —— Pydantic 校验在调用 endpoint 前失败
        response = client.post("/api/admin/profiles", json={"description": "no name"})
        assert response.status_code == 422

    def test_create_empty_name_returns_422(self, client, mock_db):
        response = client.post("/api/admin/profiles", json={"name": ""})
        assert response.status_code == 422


# ─── CRUD: update ──────────────────────────────────────────────────────


class TestUpdateProfile:
    """``PUT /api/admin/profiles/{id}``"""

    def test_update_description_only(self, client, mock_db):
        existing = build_profile(name="existing", version=2)
        mock_db.execute = AsyncMock(return_value=scalar_result(existing))

        response = client.put(
            f"/api/admin/profiles/{existing.id}",
            json={"description": "更新后描述", "change_note": "minor edit"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["description"] == "更新后描述"
        # 版本号自增
        assert body["version"] == 3
        assert existing.version == 3

    def test_update_creates_version_snapshot(self, client, mock_db, stub_admin_user):
        """任务 8.12：每次 update 都写入 ProfileVersion 快照。"""
        existing = build_profile(name="versioned", version=1)
        mock_db.execute = AsyncMock(return_value=scalar_result(existing))

        client.put(
            f"/api/admin/profiles/{existing.id}",
            json={"priority": 99, "change_note": "提升优先级"},
        )

        added_objs = [c.args[0] for c in mock_db.add.call_args_list]
        versions = [o for o in added_objs if isinstance(o, ProfileVersion)]
        assert len(versions) == 1
        snapshot = versions[0]
        assert snapshot.profile_id == existing.id
        # version 在写快照前已自增
        assert snapshot.version == 2
        assert snapshot.changed_by == stub_admin_user
        assert snapshot.change_note == "提升优先级"
        # 快照 payload 包含核心字段
        assert snapshot.snapshot["name"] == "versioned"
        assert snapshot.snapshot["priority"] == 99

    def test_update_name_uniqueness_check(self, client, mock_db):
        existing = build_profile(name="old-name")
        conflict = build_profile(name="taken")
        mock_db.execute = AsyncMock(
            side_effect=[
                scalar_result(existing),  # find by id
                scalar_result(conflict),  # name conflict
            ]
        )

        response = client.put(
            f"/api/admin/profiles/{existing.id}",
            json={"name": "taken"},
        )
        assert response.status_code == 409

    def test_update_not_found(self, client, mock_db):
        mock_db.execute = AsyncMock(return_value=scalar_result(None))
        response = client.put(
            f"/api/admin/profiles/{uuid.uuid4()}",
            json={"description": "x"},
        )
        assert response.status_code == 404


# ─── CRUD: delete ──────────────────────────────────────────────────────


class TestDeleteProfile:
    """``DELETE /api/admin/profiles/{id}``"""

    def test_delete_custom_profile(self, client, mock_db):
        target = build_profile(name="custom")
        mock_db.execute = AsyncMock(return_value=scalar_result(target))

        response = client.delete(f"/api/admin/profiles/{target.id}")
        assert response.status_code == 204
        mock_db.delete.assert_awaited_once_with(target)

    def test_delete_generic_text_blocked(self, client, mock_db):
        """generic-text 是默认兜底 Profile，不允许删除（任务 8.5/8.6）。"""
        target = build_profile(name="generic-text")
        mock_db.execute = AsyncMock(return_value=scalar_result(target))

        response = client.delete(f"/api/admin/profiles/{target.id}")
        assert response.status_code == 400
        assert "generic-text" in response.text
        mock_db.delete.assert_not_called()

    def test_delete_not_found(self, client, mock_db):
        mock_db.execute = AsyncMock(return_value=scalar_result(None))
        response = client.delete(f"/api/admin/profiles/{uuid.uuid4()}")
        assert response.status_code == 404


# ─── Toggle ────────────────────────────────────────────────────────────


class TestToggleProfile:
    """``PATCH /api/admin/profiles/{id}/toggle``"""

    def test_disable_profile(self, client, mock_db):
        target = build_profile(name="active", enabled=True)
        mock_db.execute = AsyncMock(return_value=scalar_result(target))

        response = client.patch(
            f"/api/admin/profiles/{target.id}/toggle", json={"enabled": False}
        )
        assert response.status_code == 200
        assert response.json()["enabled"] is False
        assert target.enabled is False

    def test_enable_profile(self, client, mock_db):
        target = build_profile(name="paused", enabled=False)
        mock_db.execute = AsyncMock(return_value=scalar_result(target))

        response = client.patch(
            f"/api/admin/profiles/{target.id}/toggle", json={"enabled": True}
        )
        assert response.status_code == 200
        assert response.json()["enabled"] is True

    def test_disable_generic_text_blocked(self, client, mock_db):
        """禁用 generic-text 会破坏兜底链，应被拒绝（任务 8.5）。"""
        target = build_profile(name="generic-text", enabled=True)
        mock_db.execute = AsyncMock(return_value=scalar_result(target))

        response = client.patch(
            f"/api/admin/profiles/{target.id}/toggle", json={"enabled": False}
        )
        assert response.status_code == 400
        assert "generic-text" in response.text

    def test_enable_generic_text_allowed(self, client, mock_db):
        """启用 generic-text 是允许的（确保它始终可作为兜底使用）。"""
        target = build_profile(name="generic-text", enabled=False)
        mock_db.execute = AsyncMock(return_value=scalar_result(target))

        response = client.patch(
            f"/api/admin/profiles/{target.id}/toggle", json={"enabled": True}
        )
        assert response.status_code == 200


# ─── Import / Export ───────────────────────────────────────────────────


class TestExportProfiles:
    """``GET /api/admin/profiles/export/all``"""

    def test_export_returns_all_profiles(self, client, mock_db):
        profiles = [
            build_profile(name="generic-text", priority=0),
            build_profile(name="chinese-technical-spec", priority=10),
            build_profile(name="scanned-pdf", priority=5),
        ]
        mock_db.execute = AsyncMock(return_value=scalars_all_result(profiles))

        response = client.get("/api/admin/profiles/export/all")
        assert response.status_code == 200
        body = response.json()
        assert body["count"] == 3
        names = {p["name"] for p in body["profiles"]}
        assert names == {"generic-text", "chinese-technical-spec", "scanned-pdf"}
        assert "exported_at" in body

    def test_export_empty(self, client, mock_db):
        mock_db.execute = AsyncMock(return_value=scalars_all_result([]))
        response = client.get("/api/admin/profiles/export/all")
        assert response.status_code == 200
        assert response.json()["count"] == 0


class TestImportProfiles:
    """``POST /api/admin/profiles/import``"""

    def test_import_creates_new_profiles(self, client, mock_db):
        # 两个新 Profile：每次 name 查询都返回 None
        mock_db.execute = AsyncMock(
            side_effect=[scalar_result(None), scalar_result(None)]
        )

        payload = {
            "profiles": [
                {"name": "imported-a", "priority": 5, "description": "导入 A"},
                {"name": "imported-b", "priority": 7, "description": "导入 B"},
            ]
        }
        response = client.post("/api/admin/profiles/import", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["imported"] == 2
        assert body["updated"] == 0
        assert body["errors"] == []

        added_objs = [c.args[0] for c in mock_db.add.call_args_list]
        new_profiles = [o for o in added_objs if isinstance(o, DocumentProfile)]
        assert len(new_profiles) == 2
        assert {p.name for p in new_profiles} == {"imported-a", "imported-b"}

    def test_import_updates_existing_by_name(self, client, mock_db):
        """name 匹配时执行更新，并写入版本快照。"""
        existing = build_profile(name="existing-pf", priority=1, version=4)
        mock_db.execute = AsyncMock(return_value=scalar_result(existing))

        payload = {
            "profiles": [
                {
                    "name": "existing-pf",
                    "priority": 99,
                    "description": "导入更新",
                    "match_rules": {"filename_regex": [r".*new.*"], "content_regex": [], "min_content_match_count": 1},
                }
            ]
        }
        response = client.post("/api/admin/profiles/import", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["imported"] == 0
        assert body["updated"] == 1
        # 现有 ORM 实例已被原地更新
        assert existing.priority == 99
        assert existing.description == "导入更新"
        assert existing.version == 5

        # 版本快照写入
        added_objs = [c.args[0] for c in mock_db.add.call_args_list]
        versions = [o for o in added_objs if isinstance(o, ProfileVersion)]
        assert len(versions) == 1
        assert "Imported" in (versions[0].change_note or "")

    def test_import_mixed_create_and_update(self, client, mock_db):
        existing = build_profile(name="existing-pf", priority=1)
        mock_db.execute = AsyncMock(
            side_effect=[
                scalar_result(existing),  # 第一个：已存在
                scalar_result(None),       # 第二个：新建
            ]
        )

        payload = {
            "profiles": [
                {"name": "existing-pf", "priority": 99},
                {"name": "fresh-one", "priority": 3},
            ]
        }
        response = client.post("/api/admin/profiles/import", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["imported"] == 1
        assert body["updated"] == 1

    def test_import_skips_entries_without_name(self, client, mock_db):
        # 没有 name 的条目直接进入 errors，不会触发 execute
        mock_db.execute = AsyncMock()

        payload = {
            "profiles": [
                {"description": "missing name"},
                {"name": ""},  # 空字符串等价于缺失
            ]
        }
        response = client.post("/api/admin/profiles/import", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["imported"] == 0
        assert body["updated"] == 0
        assert len(body["errors"]) == 2
        assert all("name" in e for e in body["errors"])


# ─── Versions ──────────────────────────────────────────────────────────


class TestProfileVersions:
    """``GET /api/admin/profiles/{id}/versions``"""

    def test_list_versions_descending(self, client, mock_db):
        pid = uuid.uuid4()
        v3 = MagicMock(spec=ProfileVersion)
        v3.id = uuid.uuid4()
        v3.profile_id = pid
        v3.version = 3
        v3.snapshot = {"name": "x", "priority": 7}
        v3.changed_by = uuid.uuid4()
        v3.change_note = "third"
        v3.created_at = datetime(2024, 6, 3)

        v2 = MagicMock(spec=ProfileVersion)
        v2.id = uuid.uuid4()
        v2.profile_id = pid
        v2.version = 2
        v2.snapshot = {"name": "x", "priority": 5}
        v2.changed_by = uuid.uuid4()
        v2.change_note = "second"
        v2.created_at = datetime(2024, 6, 2)

        mock_db.execute = AsyncMock(return_value=scalars_all_result([v3, v2]))

        response = client.get(f"/api/admin/profiles/{pid}/versions")
        assert response.status_code == 200
        body = response.json()
        assert len(body) == 2
        assert body[0]["version"] == 3
        assert body[1]["version"] == 2
        assert body[0]["change_note"] == "third"


# ─── Preview ───────────────────────────────────────────────────────────


class TestPreviewProfile:
    """``POST /api/admin/profiles/{id}/preview``"""

    def test_preview_returns_blocks_and_features(self, client, mock_db, monkeypatch):
        target = build_profile(name="preview-me")
        mock_db.execute = AsyncMock(return_value=scalar_result(target))

        # mock 解析器选择
        sample_doc = ParsedDocument(
            blocks=[
                Block(type="heading", text="第一章 总则", page_number=1, style={"level": 1}),
                Block(type="paragraph", text="一、内容", page_number=1),
            ],
            metadata={"page_count": 1},
        )
        fake_parser = MagicMock()
        fake_parser.parse = AsyncMock(return_value=sample_doc)
        fake_registry = MagicMock()
        fake_registry.select = MagicMock(return_value=fake_parser)
        fake_registry.plugins = ["dummy"]  # 让 _ensure_default_parsers 直接 return

        monkeypatch.setattr(
            "app.services.parsers.registry.get_parser_registry",
            lambda: fake_registry,
        )

        files = {"file": ("规范样本.txt", b"sample content", "text/plain")}
        response = client.post(
            f"/api/admin/profiles/{target.id}/preview",
            files=files,
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["matched_profile"] == "preview-me"
        assert len(body["blocks"]) == 2
        assert body["blocks"][0]["text"] == "第一章 总则"

        feats = body["features"]
        assert feats["filename"] == "规范样本.txt"
        assert feats["page_count"] == 1
        # 中式编号被识别
        assert "chinese_chapter" in feats["numbering_patterns"]
        fake_parser.parse.assert_awaited_once()

    def test_preview_profile_not_found(self, client, mock_db):
        mock_db.execute = AsyncMock(return_value=scalar_result(None))
        files = {"file": ("a.txt", b"abc", "text/plain")}
        response = client.post(
            f"/api/admin/profiles/{uuid.uuid4()}/preview",
            files=files,
        )
        assert response.status_code == 404

    def test_preview_no_parser_for_extension(self, client, mock_db, monkeypatch):
        target = build_profile(name="preview-me")
        mock_db.execute = AsyncMock(return_value=scalar_result(target))

        fake_registry = MagicMock()
        fake_registry.plugins = ["dummy"]
        fake_registry.select = MagicMock(side_effect=ValueError("No parser available"))

        monkeypatch.setattr(
            "app.services.parsers.registry.get_parser_registry",
            lambda: fake_registry,
        )

        files = {"file": ("weird.xyz", b"abc", "application/octet-stream")}
        response = client.post(
            f"/api/admin/profiles/{target.id}/preview",
            files=files,
        )
        assert response.status_code == 400
