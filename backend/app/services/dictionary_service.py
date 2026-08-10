"""Domain Dictionary service for terminology management and IK sync.

Provides:
- CRUD operations for domain dictionaries
- Term validation (1-30 chars, no control characters)
- IK analyzer remote dictionary sync (hot-reload via URL)
- Enable/disable logic (disabled dictionaries removed from IK custom dict)
- Candidate term extraction (word frequency + unrecognized word detection)
- Import/export in CSV and JSON formats
"""

import csv
import io
import logging
import os
import re
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.domain_dictionary import DomainDictionary

logger = logging.getLogger(__name__)

# ─── Data Structures ───────────────────────────────────────────────────


@dataclass
class Term:
    """A single term entry in a domain dictionary."""

    word: str
    pos: str | None = None  # Part of speech (optional)
    weight: float = 1.0


@dataclass
class SynonymGroup:
    """A group of synonymous terms."""

    primary: str
    synonyms: list[str] = field(default_factory=list)


# ─── Validation ────────────────────────────────────────────────────────

# Control characters regex (C0, C1 control chars, excluding normal whitespace)
_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def validate_term(word: str) -> tuple[bool, str]:
    """Validate a term string.

    Rules:
    - Length must be 1-30 characters
    - Must not contain control characters

    Returns:
        (is_valid, error_message)
    """
    if not word or not word.strip():
        return False, "术语不能为空"

    word = word.strip()
    if len(word) < 1 or len(word) > 30:
        return False, f"术语长度必须在 1-30 字符之间，当前长度: {len(word)}"

    if _CONTROL_CHAR_PATTERN.search(word):
        return False, "术语不能包含特殊控制字符"

    return True, ""


# ─── IK Sync ──────────────────────────────────────────────────────────

# IK Analyzer 远程词库热更新工作流：
#
#   1. 管理员通过 DictionaryService 增删改启用的词典；CRUD 方法在事务提交后
#      调用 ``sync_ik_dictionaries`` 把所有 ``enabled=True`` 词典里的术语 /
#      停用词写入 ``IK_DICT_DIR`` 下的两个 ``.dic`` 文件。
#   2. ``backend/app/api/ik_dict.py`` 暴露 ``/api/ik-dict/{filename}`` 端点
#      把 ``.dic`` 文件作为 HTTP 资源服务给 IK 插件，并设置 ``Last-Modified``
#      头，让 IK 插件能基于 ``If-Modified-Since`` 命中 304 / 触发热加载。
#   3. OpenSearch 容器里的 IK 插件配置 ``remote_ext_dict`` 指向该 URL，
#      每 ~60 秒轮询；文件 ``Last-Modified`` / ``ETag`` 变化时插件重新加载
#      词典，无需重启 OpenSearch。
#
# ``IK_DICT_DIR`` 是模块级常量，导入时从 ``Settings.IK_DICT_DIR`` 读取，
# 测试可以通过 ``monkeypatch.setattr("app.services.dictionary_service.IK_DICT_DIR", ...)``
# 或 ``unittest.mock.patch`` 暂时改写。
IK_DICT_DIR: Path = Path(get_settings().IK_DICT_DIR)
IK_MAIN_DICT_FILE = "custom_main.dic"
IK_STOP_DICT_FILE = "custom_stopword.dic"


def generate_ik_dict_content(terms: list[Term]) -> str:
    """Generate IK dictionary file content from terms.

    Each line is a single word for IK to recognize.
    """
    lines = [term.word for term in terms if term.word.strip()]
    return "\n".join(sorted(set(lines)))


def generate_ik_stopword_content(stop_words: list[str]) -> str:
    """Generate IK stopword file content."""
    lines = [w.strip() for w in stop_words if w.strip()]
    return "\n".join(sorted(set(lines)))


async def sync_ik_dictionaries(db: AsyncSession) -> dict[str, int]:
    """Sync all enabled dictionaries to IK remote dictionary files.

    IK analyzer 通过 ``remote_ext_dict`` 配置项轮询远程 URL（默认 60 秒一次），
    通过 ``Last-Modified`` / ``ETag`` 决定是否重新加载。本函数仅负责把所有
    ``enabled=True`` 的词典内容物化到 ``IK_DICT_DIR`` 下的两个文件：

    - ``custom_main.dic`` — 自定义主词典（一行一个词）
    - ``custom_stopword.dic`` — 自定义停用词

    特性：
    - 仅同步启用的词典；禁用词典自动从输出中排除（任务 13.7 - 启用/禁用语义）。
    - 内容相同则**不重写**且不更新 mtime，避免 IK 插件做无意义热加载。
    - 内容变化时**显式 bump mtime**到当前时间，让 IK 的 ``Last-Modified``
      校验确实命中变更（即便内容差异极小也能触发）。
    - 文件系统失败（权限不足 / 磁盘满）只记 warning 不抛异常，词典 CRUD 不受影响。
    - 输入为空时仍写入空文件（清空旧内容），不视作错误。

    Returns:
        ``{"terms": N, "stop_words": M}`` — 写入文件的去重后行数。
    """
    result = await db.execute(
        select(DomainDictionary).where(DomainDictionary.enabled == True)  # noqa: E712
    )
    dictionaries = result.scalars().all()

    all_terms: list[Term] = []
    all_stop_words: list[str] = []

    for dictionary in dictionaries:
        # Parse terms from JSONB
        for term_data in (dictionary.terms or []):
            if isinstance(term_data, dict):
                word = term_data.get("word", "")
                if word.strip():
                    all_terms.append(Term(
                        word=word,
                        pos=term_data.get("pos"),
                        weight=term_data.get("weight", 1.0),
                    ))
            elif isinstance(term_data, str):
                if term_data.strip():
                    all_terms.append(Term(word=term_data))

        # Parse stop words
        for sw in (dictionary.stop_words or []):
            if isinstance(sw, str) and sw.strip():
                all_stop_words.append(sw.strip())

    # Generate dictionary files
    main_content = generate_ik_dict_content(all_terms)
    stop_content = generate_ik_stopword_content(all_stop_words)

    # 实际写入由 ``_write_ik_dict_file`` 处理：内容比较 + 原子写 + bump mtime。
    # 任何 IO 错误只记 warning，绝不打断词典 CRUD（任务要求"失败可容忍"）。
    try:
        IK_DICT_DIR.mkdir(parents=True, exist_ok=True)
        main_changed = _write_ik_dict_file(
            IK_DICT_DIR / IK_MAIN_DICT_FILE, main_content
        )
        stop_changed = _write_ik_dict_file(
            IK_DICT_DIR / IK_STOP_DICT_FILE, stop_content
        )
        logger.info(
            "IK dictionaries synced: %d terms, %d stop words "
            "(main_changed=%s, stop_changed=%s)",
            len(set(t.word for t in all_terms if t.word.strip())),
            len(set(all_stop_words)),
            main_changed,
            stop_changed,
        )
    except OSError as e:
        # mkdir 失败（权限/挂载错）— 写入根本没开始，词典 CRUD 仍然成功。
        logger.warning("Failed to prepare IK dictionary directory: %s", e)

    return {"terms": len(all_terms), "stop_words": len(all_stop_words)}


def _write_ik_dict_file(path: Path, content: str) -> bool:
    """把词库内容写入 ``path``，仅在内容变化时更新文件并 bump mtime。

    返回 ``True`` 表示文件被改写（IK 下次轮询会触发热加载），``False`` 表示
    内容未变跳过写入。任何 ``OSError``（写权限不足、磁盘满等）只记 warning，
    不向上传播 — 调用方据此判断"沿用旧文件"也是可接受的退化行为。

    使用临时文件 + ``replace`` 做原子替换：避免 IK 插件读到半截内容。
    """
    try:
        if path.exists():
            try:
                existing = path.read_text(encoding="utf-8")
            except OSError as e:
                # 读失败时直接重写：宁可多写一次也不要因为读不出旧内容跳过同步。
                logger.warning("Failed to read existing IK dict %s: %s", path, e)
                existing = None
            if existing == content:
                # 内容相同，IK 不需要重新加载，跳过写入和 mtime 更新。
                return False

        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.replace(path)

        # 显式 bump mtime 到当前时间。某些文件系统（NFS / overlayfs）的
        # mtime 粒度只有 1~2 秒，连续两次同步可能落在同一秒导致 IK 无法
        # 通过 Last-Modified 检出变更；这里强制把 mtime 推到 ``time.time()``，
        # 保证 HTTP 路由后续返回的 ``Last-Modified`` 头会真实变化。
        now = time.time()
        try:
            os.utime(path, (now, now))
        except OSError as e:  # 权限不足等
            logger.warning("Failed to bump mtime for IK dict %s: %s", path, e)
        return True
    except OSError as e:
        logger.warning("Failed to write IK dict %s: %s", path, e)
        return False


# ─── Dictionary Service ────────────────────────────────────────────────


class DictionaryService:
    """Service for domain dictionary CRUD and management."""

    def __init__(self, db: AsyncSession):
        self.db = db

    # ─── CRUD ──────────────────────────────────────────────────────

    async def list_dictionaries(
        self,
        enabled: bool | None = None,
        skip: int = 0,
        limit: int = 50,
    ) -> tuple[list[DomainDictionary], int]:
        """List dictionaries with optional filtering."""
        query = select(DomainDictionary)
        if enabled is not None:
            query = query.where(DomainDictionary.enabled == enabled)

        # Count
        count_query = select(DomainDictionary)
        if enabled is not None:
            count_query = count_query.where(DomainDictionary.enabled == enabled)
        count_result = await self.db.execute(count_query)
        total = len(count_result.scalars().all())

        # Paginate
        query = query.order_by(DomainDictionary.created_at.desc())
        query = query.offset(skip).limit(limit)
        result = await self.db.execute(query)
        dictionaries = list(result.scalars().all())

        return dictionaries, total

    async def get_dictionary(self, dictionary_id: str) -> DomainDictionary | None:
        """Get a single dictionary by ID."""
        result = await self.db.execute(
            select(DomainDictionary).where(
                DomainDictionary.id == uuid.UUID(dictionary_id)
            )
        )
        return result.scalar_one_or_none()

    async def create_dictionary(
        self,
        name: str,
        description: str | None = None,
        terms: list[dict] | None = None,
        synonyms: list[dict] | None = None,
        stop_words: list[str] | None = None,
        enabled: bool = True,
    ) -> DomainDictionary:
        """Create a new domain dictionary."""
        # Validate terms if provided
        if terms:
            for term_data in terms:
                word = term_data.get("word", "")
                is_valid, error = validate_term(word)
                if not is_valid:
                    raise ValueError(f"术语校验失败: {error}")

        dictionary = DomainDictionary(
            name=name,
            description=description,
            terms=terms or [],
            synonyms=synonyms or [],
            stop_words=stop_words or [],
            enabled=enabled,
        )
        self.db.add(dictionary)
        await self.db.flush()
        await self.db.refresh(dictionary)

        # Sync to IK if enabled
        if enabled:
            await sync_ik_dictionaries(self.db)

        return dictionary

    async def update_dictionary(
        self,
        dictionary_id: str,
        name: str | None = None,
        description: str | None = None,
        terms: list[dict] | None = None,
        synonyms: list[dict] | None = None,
        stop_words: list[str] | None = None,
        enabled: bool | None = None,
    ) -> DomainDictionary | None:
        """Update an existing dictionary."""
        dictionary = await self.get_dictionary(dictionary_id)
        if not dictionary:
            return None

        # Validate terms if provided
        if terms is not None:
            for term_data in terms:
                word = term_data.get("word", "")
                is_valid, error = validate_term(word)
                if not is_valid:
                    raise ValueError(f"术语校验失败: {error}")

        if name is not None:
            dictionary.name = name
        if description is not None:
            dictionary.description = description
        if terms is not None:
            dictionary.terms = terms
        if synonyms is not None:
            dictionary.synonyms = synonyms
        if stop_words is not None:
            dictionary.stop_words = stop_words
        if enabled is not None:
            dictionary.enabled = enabled

        await self.db.flush()
        await self.db.refresh(dictionary)

        # Sync to IK
        await sync_ik_dictionaries(self.db)

        return dictionary

    async def delete_dictionary(self, dictionary_id: str) -> bool:
        """Delete a dictionary."""
        dictionary = await self.get_dictionary(dictionary_id)
        if not dictionary:
            return False

        await self.db.delete(dictionary)
        await self.db.flush()

        # Re-sync IK (removed dictionary terms)
        await sync_ik_dictionaries(self.db)

        return True

    # ─── Term Management ───────────────────────────────────────────

    async def add_terms(
        self, dictionary_id: str, new_terms: list[dict]
    ) -> DomainDictionary | None:
        """Add terms to a dictionary."""
        dictionary = await self.get_dictionary(dictionary_id)
        if not dictionary:
            return None

        # Validate new terms
        for term_data in new_terms:
            word = term_data.get("word", "")
            is_valid, error = validate_term(word)
            if not is_valid:
                raise ValueError(f"术语校验失败: {error}")

        # Merge terms (avoid duplicates by word)
        existing_words = {
            t["word"] if isinstance(t, dict) else t
            for t in (dictionary.terms or [])
        }
        current_terms = list(dictionary.terms or [])

        for term_data in new_terms:
            word = term_data.get("word", "")
            if word not in existing_words:
                current_terms.append(term_data)
                existing_words.add(word)

        dictionary.terms = current_terms
        await self.db.flush()
        await self.db.refresh(dictionary)

        if dictionary.enabled:
            await sync_ik_dictionaries(self.db)

        return dictionary

    async def remove_terms(
        self, dictionary_id: str, words: list[str]
    ) -> DomainDictionary | None:
        """Remove terms from a dictionary by word."""
        dictionary = await self.get_dictionary(dictionary_id)
        if not dictionary:
            return None

        words_to_remove = set(words)
        current_terms = dictionary.terms or []
        filtered_terms = [
            t for t in current_terms
            if (t.get("word") if isinstance(t, dict) else t) not in words_to_remove
        ]

        dictionary.terms = filtered_terms
        await self.db.flush()
        await self.db.refresh(dictionary)

        if dictionary.enabled:
            await sync_ik_dictionaries(self.db)

        return dictionary

    # ─── Synonym Management ────────────────────────────────────────

    async def add_synonym_group(
        self, dictionary_id: str, primary: str, synonyms: list[str]
    ) -> DomainDictionary | None:
        """Upsert a synonym group on a dictionary.

        语义（任务 13.4）：
        - ``primary`` 不存在时追加新组。
        - ``primary`` 已存在时**替换**该组的 ``synonyms`` 列表（idempotent
          upsert），而不是追加重复组。这与前端「编辑同义词组」的直觉一致，
          也避免了重复组导致 IK 同义词扩展时一对多映射混乱。
        - ``primary`` 与每个 synonym 都按 ``validate_term`` 校验，失败抛
          ``ValueError`` 由路由层映射成 422。
        """
        dictionary = await self.get_dictionary(dictionary_id)
        if not dictionary:
            return None

        # Validate primary and synonyms
        is_valid, error = validate_term(primary)
        if not is_valid:
            raise ValueError(f"主术语校验失败: {error}")
        for syn in synonyms:
            is_valid, error = validate_term(syn)
            if not is_valid:
                raise ValueError(f"同义词校验失败: {error}")

        current_synonyms = list(dictionary.synonyms or [])
        new_group = {"primary": primary, "synonyms": list(synonyms)}
        replaced = False
        for idx, sg in enumerate(current_synonyms):
            if isinstance(sg, dict) and sg.get("primary") == primary:
                current_synonyms[idx] = new_group
                replaced = True
                break
        if not replaced:
            current_synonyms.append(new_group)
        dictionary.synonyms = current_synonyms

        await self.db.flush()
        await self.db.refresh(dictionary)
        return dictionary

    async def remove_synonym_group(
        self, dictionary_id: str, primary: str
    ) -> DomainDictionary | None:
        """Remove a synonym group by its primary term."""
        dictionary = await self.get_dictionary(dictionary_id)
        if not dictionary:
            return None

        current_synonyms = dictionary.synonyms or []
        filtered = [
            sg for sg in current_synonyms
            if sg.get("primary") != primary
        ]
        dictionary.synonyms = filtered

        await self.db.flush()
        await self.db.refresh(dictionary)
        return dictionary

    # ─── Enable/Disable ────────────────────────────────────────────

    async def toggle_dictionary(
        self, dictionary_id: str, enabled: bool
    ) -> DomainDictionary | None:
        """Enable or disable a dictionary.

        When disabled, the dictionary's terms are removed from IK custom dict.
        """
        dictionary = await self.get_dictionary(dictionary_id)
        if not dictionary:
            return None

        dictionary.enabled = enabled
        await self.db.flush()
        await self.db.refresh(dictionary)

        # Re-sync IK (will include/exclude based on enabled status)
        await sync_ik_dictionaries(self.db)

        return dictionary

    # ─── Import/Export ─────────────────────────────────────────────

    def export_as_json(self, dictionary: DomainDictionary) -> dict:
        """Export a dictionary as JSON."""
        return {
            "name": dictionary.name,
            "description": dictionary.description,
            "terms": dictionary.terms or [],
            "synonyms": dictionary.synonyms or [],
            "stop_words": dictionary.stop_words or [],
            "enabled": dictionary.enabled,
        }

    def export_as_csv(self, dictionary: DomainDictionary) -> str:
        """Export dictionary terms as CSV (word,pos,weight)."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["word", "pos", "weight"])

        for term_data in (dictionary.terms or []):
            if isinstance(term_data, dict):
                writer.writerow([
                    term_data.get("word", ""),
                    term_data.get("pos", ""),
                    term_data.get("weight", 1.0),
                ])
            elif isinstance(term_data, str):
                writer.writerow([term_data, "", 1.0])

        return output.getvalue()

    def import_from_csv(self, csv_content: str) -> list[dict]:
        """Parse CSV content into term list.

        Expected format: word,pos,weight
        """
        reader = csv.reader(io.StringIO(csv_content))
        terms = []
        header_skipped = False

        for row in reader:
            if not row:
                continue
            # Skip header row
            if not header_skipped and row[0].lower() in ("word", "词", "术语"):
                header_skipped = True
                continue
            header_skipped = True

            # 单行解析失败（如 weight 非数值）只跳过该行，不影响整体导入。
            try:
                word = row[0].strip() if len(row) > 0 else ""
                pos = row[1].strip() if len(row) > 1 else None
                weight = (
                    float(row[2]) if len(row) > 2 and row[2].strip() else 1.0
                )
            except (ValueError, IndexError) as exc:
                logger.warning(f"Skipping malformed CSV row {row!r}: {exc}")
                continue

            if word:
                is_valid, error = validate_term(word)
                if is_valid:
                    terms.append({
                        "word": word,
                        "pos": pos if pos else None,
                        "weight": weight,
                    })
                else:
                    logger.warning(f"Skipping invalid term '{word}': {error}")

        return terms

    def import_from_json(self, json_data: dict) -> dict:
        """Parse JSON import data.

        Returns dict with terms, synonyms, stop_words.
        """
        terms = json_data.get("terms", [])
        synonyms = json_data.get("synonyms", [])
        stop_words = json_data.get("stop_words", [])

        # Validate terms
        valid_terms = []
        for term_data in terms:
            if isinstance(term_data, dict):
                word = term_data.get("word", "")
            elif isinstance(term_data, str):
                word = term_data
                term_data = {"word": word, "pos": None, "weight": 1.0}
            else:
                continue

            is_valid, _ = validate_term(word)
            if is_valid:
                valid_terms.append(term_data)

        return {
            "terms": valid_terms,
            "synonyms": synonyms,
            "stop_words": stop_words,
        }

    # ─── Candidate Term Extraction ─────────────────────────────────

    async def extract_candidate_terms(
        self,
        documents_content: list[str],
        min_frequency: int = 3,
        min_length: int = 2,
        max_length: int = 10,
        top_n: int = 50,
    ) -> list[dict]:
        """Extract candidate terms from document content.

        Uses word frequency statistics and unrecognized word detection.
        Terms that already exist in enabled dictionaries are excluded.

        Args:
            documents_content: List of document text content
            min_frequency: Minimum occurrence count to be a candidate
            min_length: Minimum character length for candidates
            max_length: Maximum character length for candidates
            top_n: Maximum number of candidates to return

        Returns:
            List of candidate term dicts with word and frequency
        """
        # Get existing terms from all enabled dictionaries
        result = await self.db.execute(
            select(DomainDictionary).where(DomainDictionary.enabled == True)  # noqa: E712
        )
        dictionaries = result.scalars().all()

        existing_words: set[str] = set()
        for d in dictionaries:
            for term_data in (d.terms or []):
                word: str = ""
                if isinstance(term_data, dict):
                    word = term_data.get("word", "") or ""
                elif isinstance(term_data, str):
                    word = term_data
                word = word.strip()
                if word:
                    existing_words.add(word)
            for sw in (d.stop_words or []):
                if isinstance(sw, str) and sw.strip():
                    existing_words.add(sw.strip())

        # Extract word frequencies using simple Chinese word segmentation
        # Uses character n-grams as a basic approach
        word_counter: Counter = Counter()

        for content in documents_content:
            if not content:
                continue
            # Extract potential terms using regex for Chinese characters
            # and common patterns
            words = _extract_chinese_words(content, min_length, max_length)
            word_counter.update(words)

        # Filter candidates
        candidates = []
        for word, freq in word_counter.most_common(top_n * 3):
            if freq < min_frequency:
                continue
            if word in existing_words:
                continue
            if len(word) < min_length or len(word) > max_length:
                continue

            is_valid, _ = validate_term(word)
            if not is_valid:
                continue

            candidates.append({"word": word, "frequency": freq})
            if len(candidates) >= top_n:
                break

        return candidates


def _extract_chinese_words(
    text: str, min_length: int = 2, max_length: int = 10
) -> list[str]:
    """Extract potential Chinese words from text using n-gram approach.

    This is a simple extraction method that identifies sequences of
    Chinese characters as potential terms.
    """
    # Pattern for sequences of Chinese characters
    chinese_pattern = re.compile(r"[\u4e00-\u9fff]+")
    words: list[str] = []

    for match in chinese_pattern.finditer(text):
        segment = match.group()
        # Generate n-grams of various lengths
        for n in range(min_length, min(max_length + 1, len(segment) + 1)):
            for i in range(len(segment) - n + 1):
                word = segment[i : i + n]
                words.append(word)

    return words


# ─── Preset Dictionary ─────────────────────────────────────────────────

# Common Chinese stop words
CHINESE_STOP_WORDS = [
    "的", "了", "是", "在", "和", "有", "不", "这", "人", "我",
    "他", "她", "它", "们", "你", "也", "就", "都", "而", "及",
    "与", "或", "但", "如", "对", "从", "到", "把", "被", "让",
    "给", "向", "往", "由", "以", "为", "因", "所", "其", "那",
    "这个", "那个", "什么", "怎么", "哪里", "哪个", "为什么",
    "可以", "能够", "应该", "已经", "正在", "将要", "曾经",
    "一个", "一些", "一种", "一样", "一直", "一起",
    "没有", "不是", "不能", "不会", "不要",
    "然后", "因此", "所以", "但是", "虽然", "如果", "即使",
    "而且", "并且", "或者", "以及", "还是",
    "这样", "那样", "这些", "那些", "自己", "大家",
    "非常", "十分", "比较", "更加", "最", "很", "太",
    "吗", "呢", "吧", "啊", "呀", "哦", "嗯",
    "上", "下", "中", "前", "后", "左", "右", "里", "外",
]


async def ensure_preset_dictionaries(db: AsyncSession) -> None:
    """Ensure preset dictionaries exist in the database.

    Creates the default Chinese stop words dictionary if it doesn't exist.
    """
    result = await db.execute(
        select(DomainDictionary).where(
            DomainDictionary.name == "通用中文停用词"
        )
    )
    existing = result.scalar_one_or_none()

    if not existing:
        preset = DomainDictionary(
            name="通用中文停用词",
            description="通用中文停用词词典，包含常见的中文停用词，用于过滤搜索噪声",
            terms=[],
            synonyms=[],
            stop_words=CHINESE_STOP_WORDS,
            enabled=True,
        )
        db.add(preset)
        await db.flush()
        logger.info("Created preset dictionary: 通用中文停用词")
