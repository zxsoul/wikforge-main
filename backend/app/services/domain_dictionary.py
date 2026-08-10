"""Domain Dictionary 领域级数据结构。

提供 DomainDictionary、Term、SynonymGroup 三个 dataclass，配套：

- ``to_dict`` / ``from_dict`` 用于在 PostgreSQL JSONB 列与 Python 对象之间
  双向转换（与 ``app.models.domain_dictionary.DomainDictionary`` 模型的
  ``terms`` / ``synonyms`` / ``stop_words`` JSONB 字段对齐）。
- 术语合法性校验：长度 1-30 字符，不含控制字符（对应需求 20.5）。
- ``SynonymGroup`` 完整性校验：主术语不能出现在同义词列表中，同义词不允许重复。
- 往返保持（round-trip preservation）：``from_dict(to_dict(x)) == x``。

注：``app.services.dictionary_service`` 中也存在 ``Term`` / ``SynonymGroup`` 的
轻量版本（仅用于 IK 词典文件生成，不做强校验），二者短期并存，后续任务可整合。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "DomainDictionary",
    "Term",
    "SynonymGroup",
    "validate_word",
    "WORD_MIN_LENGTH",
    "WORD_MAX_LENGTH",
]

# 字数限制：单个术语最少 1 字符、最多 30 字符（需求 20.5）。
WORD_MIN_LENGTH = 1
WORD_MAX_LENGTH = 30

# 控制字符：C0 (0x00-0x1F 排除制表/换行/回车) 与 DEL/C1 (0x7F-0x9F)。
# 普通空白（\t、\n、\r）允许，但术语两端会被 ``strip``。
_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def validate_word(word: Any) -> tuple[bool, str]:
    """校验一个术语字符串，返回 ``(is_valid, error_message)``。

    规则（需求 20.5）：

    - 必须为非空字符串
    - 去除首尾空白后长度 1-30
    - 不能包含控制字符
    """
    if not isinstance(word, str):
        return False, "术语必须为字符串"

    if not word or not word.strip():
        return False, "术语不能为空"

    stripped = word.strip()
    if len(stripped) < WORD_MIN_LENGTH or len(stripped) > WORD_MAX_LENGTH:
        return (
            False,
            f"术语长度必须在 {WORD_MIN_LENGTH}-{WORD_MAX_LENGTH} 字符之间，"
            f"当前长度: {len(stripped)}",
        )

    if _CONTROL_CHAR_PATTERN.search(stripped):
        return False, "术语不能包含特殊控制字符"

    return True, ""


def _ensure_valid_word(word: Any, label: str = "术语") -> str:
    """校验并返回去除首尾空白的术语，校验失败抛出 ``ValueError``。"""
    is_valid, error = validate_word(word)
    if not is_valid:
        raise ValueError(f"{label}校验失败: {error}")
    return word.strip()


@dataclass
class Term:
    """领域词典中的单条术语。

    对应 design.md 6 节中的 ``Term`` dataclass：

    .. code-block:: python

        @dataclass
        class Term:
            word: str
            pos: str | None  # 词性
            weight: float = 1.0
    """

    word: str
    pos: str | None = None
    weight: float = 1.0

    def __post_init__(self) -> None:
        self.word = _ensure_valid_word(self.word, label="术语")
        # weight 标准化为 float，便于 JSON 序列化稳定性。
        self.weight = float(self.weight)

    def to_dict(self) -> dict[str, Any]:
        """序列化为可写入 JSONB 的字典。"""
        return {
            "word": self.word,
            "pos": self.pos,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: Any) -> Term:
        """从 JSONB 字典或字符串反序列化。

        兼容历史数据：若 JSONB 中存的是纯字符串，则按 ``Term(word=...)``
        构造，``pos`` 为 ``None``，``weight`` 取默认值 1.0。
        """
        if isinstance(data, str):
            return cls(word=data)
        if not isinstance(data, dict):
            raise TypeError(
                f"Term.from_dict 期望 dict 或 str，得到 {type(data).__name__}"
            )
        return cls(
            word=data["word"],
            pos=data.get("pos"),
            weight=float(data.get("weight", 1.0)),
        )


@dataclass
class SynonymGroup:
    """一组同义词。

    对应 design.md 6 节中的 ``SynonymGroup`` dataclass：

    .. code-block:: python

        @dataclass
        class SynonymGroup:
            primary: str
            synonyms: list[str]  # 如 ["大齿圈", "齿圈", "主齿圈"]
    """

    primary: str
    synonyms: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.primary = _ensure_valid_word(self.primary, label="主术语")

        cleaned: list[str] = []
        seen: set[str] = {self.primary}
        for syn in self.synonyms:
            stripped = _ensure_valid_word(syn, label="同义词")
            if stripped == self.primary:
                raise ValueError(
                    f"同义词不能与主术语相同: {stripped}"
                )
            if stripped in seen:
                raise ValueError(f"同义词列表存在重复: {stripped}")
            seen.add(stripped)
            cleaned.append(stripped)
        self.synonyms = cleaned

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary": self.primary,
            "synonyms": list(self.synonyms),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SynonymGroup:
        if not isinstance(data, dict):
            raise TypeError(
                f"SynonymGroup.from_dict 期望 dict，得到 {type(data).__name__}"
            )
        return cls(
            primary=data["primary"],
            synonyms=list(data.get("synonyms", [])),
        )


@dataclass
class DomainDictionary:
    """领域词典聚合根。

    对应 design.md 6 节中的 ``DomainDictionary`` dataclass：

    .. code-block:: python

        @dataclass
        class DomainDictionary:
            id: str
            name: str           # "水泥行业术语"
            description: str
            terms: list[Term]
            synonyms: list[SynonymGroup]
            stop_words: list[str]
            enabled: bool

    与 SQLAlchemy ORM 模型 ``app.models.domain_dictionary.DomainDictionary``
    的关系：``terms`` / ``synonyms`` / ``stop_words`` 三个字段对应 JSONB 列，
    使用 :meth:`to_jsonb_payload` 即可获得入库所需的字典数据；从行记录
    重建领域对象使用 :meth:`from_orm`。
    """

    id: str
    name: str
    description: str = ""
    terms: list[Term] = field(default_factory=list)
    synonyms: list[SynonymGroup] = field(default_factory=list)
    stop_words: list[str] = field(default_factory=list)
    enabled: bool = True

    def __post_init__(self) -> None:
        # 校验停用词。terms / synonyms 已在自身 __post_init__ 校验。
        cleaned_stop_words: list[str] = []
        for sw in self.stop_words:
            stripped = _ensure_valid_word(sw, label="停用词")
            cleaned_stop_words.append(stripped)
        self.stop_words = cleaned_stop_words

    # ─── 序列化 ───────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """完整序列化为字典（含 ``id`` / ``name`` / ``enabled`` 等）。"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "terms": [t.to_dict() for t in self.terms],
            "synonyms": [sg.to_dict() for sg in self.synonyms],
            "stop_words": list(self.stop_words),
            "enabled": self.enabled,
        }

    def to_jsonb_payload(self) -> dict[str, Any]:
        """提取仅写入 JSONB 列的部分（terms / synonyms / stop_words）。

        与 ``app.models.domain_dictionary.DomainDictionary`` 的 JSONB 字段
        一一对应，便于通过 ``setattr`` 或字典展开赋给 ORM 实例。
        """
        return {
            "terms": [t.to_dict() for t in self.terms],
            "synonyms": [sg.to_dict() for sg in self.synonyms],
            "stop_words": list(self.stop_words),
        }

    # ─── 反序列化 ─────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DomainDictionary:
        if not isinstance(data, dict):
            raise TypeError(
                f"DomainDictionary.from_dict 期望 dict，得到 {type(data).__name__}"
            )
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", "") or "",
            terms=[Term.from_dict(t) for t in data.get("terms", []) or []],
            synonyms=[
                SynonymGroup.from_dict(s)
                for s in data.get("synonyms", []) or []
            ],
            stop_words=list(data.get("stop_words", []) or []),
            enabled=bool(data.get("enabled", True)),
        )

    @classmethod
    def from_orm(cls, model: Any) -> DomainDictionary:
        """从 SQLAlchemy ORM 实例构建领域对象。

        ``model`` 期望具有 ``id`` / ``name`` / ``description`` / ``terms`` /
        ``synonyms`` / ``stop_words`` / ``enabled`` 属性，其中后三者为 JSONB
        反序列化后的 list。
        """
        return cls.from_dict(
            {
                "id": str(model.id),
                "name": model.name,
                "description": model.description or "",
                "terms": list(model.terms or []),
                "synonyms": list(model.synonyms or []),
                "stop_words": list(model.stop_words or []),
                "enabled": bool(model.enabled),
            }
        )
