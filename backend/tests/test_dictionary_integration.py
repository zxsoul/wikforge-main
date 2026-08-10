"""领域词典生命周期集成测试（任务 13.11）。

任务 13.x 各子任务的单元测试已经各自覆盖：

- ``test_dictionary.py`` — ``validate_term`` / IK 文件生成 / 导入导出 /
  候选提取（13.1、13.5、13.6、13.9、13.10 的纯函数面）。
- ``test_admin_dictionaries.py`` 等 — REST API 路由层契约（13.2 ~ 13.5）。
- ``test_term_validation_property.py`` — 术语校验性质测试（13.6）。
- ``test_ik_sync.py`` — ``sync_ik_dictionaries`` 自身行为 + IK HTTP 端点（13.7）。
- ``test_dictionary_toggle.py`` — toggle 端点 + 单独词典启用/禁用对 IK 的影响
  （13.8）。
- ``test_candidate_extraction.py`` / ``test_preset_dictionary.py`` — 13.9 /
  13.10。

本文件**不重复**上述覆盖；它把真实 ``DictionaryService`` 串成端到端的
管理员工作流，验证多个 CRUD 操作之间的状态一致性 — 即「**词典 DB 状态
与 IK 文件内容始终保持同步**」这一跨任务的总体不变量。

通过一个轻量级的 ``FakeAsyncSession`` 模拟 JSONB 列 + ``where(id == ...)`` /
``where(enabled == True)`` 两种查询模式，让真实的 ``DictionaryService``
直接读写它，避免 PostgreSQL 容器依赖；IK 同步逻辑则**不打桩**，让
``IK_DICT_DIR`` 重定向到 ``tmp_path`` 真实落盘 ``.dic`` 文件供断言。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.models.domain_dictionary import DomainDictionary
from app.services import dictionary_service as ds_mod
from app.services.dictionary_service import (
    DictionaryService,
    IK_MAIN_DICT_FILE,
    IK_STOP_DICT_FILE,
)


# ─── In-memory FakeAsyncSession ───────────────────────────────────────


class FakeAsyncSession:
    """轻量 ``AsyncSession`` 替身，专供 ``DictionaryService`` 使用。

    支持 ``DictionaryService`` 内部用到的两种 ``select`` 形态：

    1. ``select(DomainDictionary).where(DomainDictionary.id == uuid)``
       → ``get_dictionary``。
    2. ``select(DomainDictionary).where(DomainDictionary.enabled == True)``
       → ``sync_ik_dictionaries`` / ``extract_candidate_terms``。

    其它 ``select`` 形态（如 ``list_dictionaries`` 的无 where 查询）也能
    工作：返回全部词典。``execute`` 返回的结果对象同时实现
    ``.scalar_one_or_none()`` 和 ``.scalars().all()`` 两个调用。
    """

    def __init__(self) -> None:
        self.store: list[DomainDictionary] = []

    # SQLAlchemy 的 add 是同步的
    def add(self, obj: DomainDictionary) -> None:
        if getattr(obj, "id", None) is None:
            obj.id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        if getattr(obj, "created_at", None) is None:
            obj.created_at = now
        # updated_at 每次 add / flush 都刷新
        obj.updated_at = now
        self.store.append(obj)

    async def execute(self, stmt: Any):
        matches = self._filter(stmt)
        return _make_result(matches)

    async def flush(self) -> None:
        # 真实 DB 会触发 onupdate；这里同步刷一下 updated_at。
        for obj in self.store:
            obj.updated_at = datetime.now(timezone.utc)

    async def refresh(self, obj: DomainDictionary) -> None:
        # In-memory 模式下对象引用即"最新状态"，无需重载。
        return None

    async def delete(self, obj: DomainDictionary) -> None:
        # SQLAlchemy 的 ``await db.delete(obj)`` 标记删除；这里直接移除。
        try:
            self.store.remove(obj)
        except ValueError:
            pass

    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        return None

    # ─── 内部：根据 where 子句过滤 store ───────────────────────────

    def _filter(self, stmt: Any) -> list[DomainDictionary]:
        where = getattr(stmt, "whereclause", None)
        if where is None:
            return list(self.store)
        # 单一 BinaryExpression：左侧是 InstrumentedAttribute，右侧通常是
        # ``BindParameter``（``col == 'foo'``）或 ``True_`` / ``False_``
        # （``col == True`` 在 SQLAlchemy 中编译为 ``IS TRUE``）。复合
        # where 不在 service 的调用中出现，遇到时 fail-fast。
        col_key = getattr(getattr(where, "left", None), "key", None)
        right = getattr(where, "right", None)
        if col_key is None or right is None:
            raise NotImplementedError(
                f"FakeAsyncSession 不支持的 where 子句: {where!r}"
            )

        # 提取右值。BindParameter 走 .value；True_/False_ 走类型判断
        # （它们没有 .value 属性，会触发 AttributeError）。
        right_cls = right.__class__.__name__
        if right_cls == "True_":
            target: Any = True
        elif right_cls == "False_":
            target = False
        elif hasattr(right, "value"):
            target = right.value
        else:  # pragma: no cover - 防御
            raise NotImplementedError(
                f"FakeAsyncSession 不支持的右操作数: {right!r}"
            )

        if col_key == "id":
            return [d for d in self.store if d.id == target]
        if col_key == "enabled":
            return [d for d in self.store if d.enabled is target]
        if col_key == "name":
            return [d for d in self.store if d.name == target]
        raise NotImplementedError(
            f"FakeAsyncSession 未实现的过滤列: {col_key!r}"
        )


def _make_result(items: list[DomainDictionary]):
    """构造既能 ``.scalar_one_or_none()`` 又能 ``.scalars().all()`` 的结果。"""
    r = MagicMock()
    r.scalar_one_or_none.return_value = items[0] if items else None
    r.scalars.return_value.all.return_value = list(items)
    return r


# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def ik_dir(tmp_path, monkeypatch):
    """让 ``IK_DICT_DIR`` 指向 ``tmp_path``，避免触碰真实 ``/data/...``。"""
    target = tmp_path / "ik-custom-dict"
    monkeypatch.setattr(ds_mod, "IK_DICT_DIR", target)
    return target


@pytest.fixture
def session() -> FakeAsyncSession:
    return FakeAsyncSession()


@pytest.fixture
def service(session: FakeAsyncSession) -> DictionaryService:
    return DictionaryService(session)


# ─── 辅助断言 ─────────────────────────────────────────────────────────


def _read_ik(ik_dir) -> tuple[list[str], list[str]]:
    """读取 ``custom_main.dic`` 与 ``custom_stopword.dic`` 已排序去重的行。"""
    main_path = ik_dir / IK_MAIN_DICT_FILE
    stop_path = ik_dir / IK_STOP_DICT_FILE
    main_lines = (
        main_path.read_text(encoding="utf-8").splitlines()
        if main_path.exists()
        else []
    )
    stop_lines = (
        stop_path.read_text(encoding="utf-8").splitlines()
        if stop_path.exists()
        else []
    )
    return sorted(main_lines), sorted(stop_lines)


# ─── 集成测试 1：单词典完整生命周期 ───────────────────────────────


@pytest.mark.asyncio
async def test_single_dictionary_full_lifecycle_keeps_ik_in_sync(
    ik_dir, service: DictionaryService
):
    """走完一个词典从 0 到删除的全部状态变迁，每一步检查 IK 文件。

    工作流：
    1. ``create_dictionary`` 时禁用 → 不应触发 IK 写入（IK 文件还没生成）。
    2. ``toggle_dictionary(True)`` 启用 → IK 主词典 / 停用词文件出现内容。
    3. ``add_terms`` 追加术语 → IK 主词典扩展。
    4. ``add_synonym_group`` 仅写 DB（IK 主词典不感知同义词）。
    5. ``update_dictionary`` 替换 stop_words → IK 停用词文件更新。
    6. ``toggle_dictionary(False)`` 禁用 → IK 文件清空。
    7. ``delete_dictionary`` → IK 文件保持为空。
    """
    # 1. 创建禁用词典：sync_ik 不被触发，IK 目录甚至可能不存在。
    created = await service.create_dictionary(
        name="水泥行业术语",
        description="试运行词典",
        terms=[{"word": "大齿圈", "pos": "n", "weight": 1.0}],
        stop_words=["的"],
        enabled=False,
    )
    assert created.enabled is False
    assert not (ik_dir / IK_MAIN_DICT_FILE).exists(), (
        "禁用状态下 create 不应触发 IK 同步（IK 目录尚未生成）"
    )

    # 2. 启用：IK 文件出现术语 / 停用词。
    toggled_on = await service.toggle_dictionary(
        dictionary_id=str(created.id), enabled=True
    )
    assert toggled_on is not None and toggled_on.enabled is True
    main_lines, stop_lines = _read_ik(ik_dir)
    assert main_lines == ["大齿圈"]
    assert stop_lines == ["的"]

    # 3. 追加术语：IK 主词典扩展（注意 add_terms 内部去重，重复词不会再写）。
    updated_with_terms = await service.add_terms(
        dictionary_id=str(created.id),
        new_terms=[
            {"word": "回转窑", "pos": "n", "weight": 1.0},
            {"word": "大齿圈", "pos": "n", "weight": 1.0},  # 重复
        ],
    )
    assert updated_with_terms is not None
    assert {t["word"] for t in updated_with_terms.terms} == {"大齿圈", "回转窑"}
    main_lines, stop_lines = _read_ik(ik_dir)
    assert main_lines == ["回转窑", "大齿圈"]
    assert stop_lines == ["的"]

    # 4. 同义词组只动 DB，对 IK 主词典 / 停用词文件无影响。
    main_before, stop_before = _read_ik(ik_dir)
    main_mtime_before = (ik_dir / IK_MAIN_DICT_FILE).stat().st_mtime
    syn_updated = await service.add_synonym_group(
        dictionary_id=str(created.id),
        primary="大齿圈",
        synonyms=["齿圈", "主齿圈"],
    )
    assert syn_updated is not None
    assert any(
        sg.get("primary") == "大齿圈" and sg.get("synonyms") == ["齿圈", "主齿圈"]
        for sg in syn_updated.synonyms
    ), syn_updated.synonyms
    main_after, stop_after = _read_ik(ik_dir)
    assert main_after == main_before, "同义词不应改写 IK 主词典"
    assert stop_after == stop_before, "同义词不应改写 IK 停用词"
    assert (ik_dir / IK_MAIN_DICT_FILE).stat().st_mtime == pytest.approx(
        main_mtime_before, abs=0.01
    ), "未变化的 IK 主词典 mtime 不应被打扰"

    # 5. update_dictionary 替换 stop_words：IK 停用词文件刷新。
    updated_full = await service.update_dictionary(
        dictionary_id=str(created.id),
        stop_words=["了", "在"],  # 完全替换原有的 "的"
    )
    assert updated_full is not None
    main_lines, stop_lines = _read_ik(ik_dir)
    assert main_lines == ["回转窑", "大齿圈"]
    assert stop_lines == ["了", "在"]
    assert "的" not in stop_lines

    # 6. 禁用词典：IK 文件清空（没有其它启用词典）。
    toggled_off = await service.toggle_dictionary(
        dictionary_id=str(created.id), enabled=False
    )
    assert toggled_off is not None and toggled_off.enabled is False
    main_lines, stop_lines = _read_ik(ik_dir)
    assert main_lines == []
    assert stop_lines == []

    # 7. 删除已禁用词典：IK 仍为空。
    deleted = await service.delete_dictionary(str(created.id))
    assert deleted is True
    assert await service.get_dictionary(str(created.id)) is None
    main_lines, stop_lines = _read_ik(ik_dir)
    assert main_lines == []
    assert stop_lines == []


# ─── 集成测试 2：多词典隔离 + 删除回收 ────────────────────────────


@pytest.mark.asyncio
async def test_multi_dictionary_lifecycle_preserves_isolation(
    ik_dir, service: DictionaryService, session: FakeAsyncSession
):
    """两个词典并存时，单个词典的禁用 / 删除不应影响另一个词典在 IK 中的内容。

    工作流：
    1. 同时启用 A、B → IK 同时含两者术语 / 停用词。
    2. 禁用 A → IK 仅剩 B。
    3. 删除已启用的 B → IK 完全清空（DB 中 A 仍存在但为禁用）。
    4. 重新启用 A → IK 重新出现 A 的内容（验证启用回路自愈）。
    5. 删除 A → IK 清空且 DB 也为空。
    """
    # 1. 创建两个启用词典。
    dict_a = await service.create_dictionary(
        name="水泥行业术语",
        terms=[{"word": "大齿圈", "pos": "n", "weight": 1.0}],
        stop_words=["甲停用"],
        enabled=True,
    )
    dict_b = await service.create_dictionary(
        name="冶金行业术语",
        terms=[{"word": "回转窑", "pos": "n", "weight": 1.0}],
        stop_words=["乙停用"],
        enabled=True,
    )

    main_lines, stop_lines = _read_ik(ik_dir)
    assert main_lines == ["回转窑", "大齿圈"]
    assert stop_lines == ["乙停用", "甲停用"]

    # 2. 禁用 A → 仅保留 B 的内容。
    await service.toggle_dictionary(
        dictionary_id=str(dict_a.id), enabled=False
    )
    main_lines, stop_lines = _read_ik(ik_dir)
    assert main_lines == ["回转窑"]
    assert stop_lines == ["乙停用"]
    assert "大齿圈" not in main_lines
    assert "甲停用" not in stop_lines

    # 3. 删除 B：DB 里 A 仍在但已禁用，IK 应当为空。
    assert await service.delete_dictionary(str(dict_b.id)) is True
    main_lines, stop_lines = _read_ik(ik_dir)
    assert main_lines == []
    assert stop_lines == []
    # A 还在 DB 中。
    assert len(session.store) == 1
    assert session.store[0].id == dict_a.id

    # 4. 重新启用 A：IK 恢复 A 的内容（B 已删除，故只剩 A）。
    await service.toggle_dictionary(
        dictionary_id=str(dict_a.id), enabled=True
    )
    main_lines, stop_lines = _read_ik(ik_dir)
    assert main_lines == ["大齿圈"]
    assert stop_lines == ["甲停用"]

    # 5. 删除 A → IK 完全清空，DB 也为空。
    assert await service.delete_dictionary(str(dict_a.id)) is True
    main_lines, stop_lines = _read_ik(ik_dir)
    assert main_lines == []
    assert stop_lines == []
    assert session.store == []
