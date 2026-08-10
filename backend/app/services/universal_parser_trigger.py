"""Universal Parser 触发条件与编排辅助（任务 10.9）。

本模块把「LLM 通用兜底解析」的触发判定与候选 Profile 持久化抽离成纯业务函数，
让 Celery 管线（``app.tasks.pipeline``）可以在不感知具体规则的前提下完成调度。

设计要点：

- 所有判定函数都是同步、无副作用的纯函数，方便在 Celery worker 与单元测试里
  共用；不引入任何 Celery 依赖。
- 编排函数 ``run_universal_parser_and_persist_candidate`` 负责调用
  ``UniversalParser.parse`` → ``suggest_profile`` → ``save_candidate`` 这条链路，
  捕获候选 envelope 校验失败（``ValueError``）只记日志、保证 ``processed_document``
  仍能流回管线，避免文档卡死。
- ``db is None`` 时直接跳过持久化，仅返回 LLM 解析结果；调用方在 worker 里没有
  AsyncSession 时也能复用本函数。

触发规则（来自 design.md → Universal Parser → 触发条件）：
1. ``profile_matched`` 为假（``profile_matcher`` 兜底到 ``generic-text``）。
2. 解析质量分低于 ``settings.QUALITY_FALLBACK_THRESHOLD``（默认 0.7）。

任意一条命中即触发。
"""

from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any

from app.core.config import get_settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.services.parsers.base import ParsedDocument
    from app.services.universal_parser import UniversalParser

logger = logging.getLogger(__name__)


# 触发原因码：与判定函数返回值保持一致，方便 worker 把它们写到 metadata 与日志里。
TRIGGER_NO_PROFILE_MATCH = "no_profile_match"
TRIGGER_QUALITY_BELOW_THRESHOLD = "quality_below_threshold"

# 与 ProfileMatcher 的兜底 Profile 名保持一致；在这里硬编码避免循环依赖。
GENERIC_PROFILE_NAME = "generic-text"


def is_no_profile_match(
    profile_id: str | None,
    profile_name: str | None,
) -> bool:
    """是否属于「没有匹配到具体 Profile」。

    判定条件（任意一条命中即视为没有匹配）：

    - ``profile_id is None``：``profile_match`` 任务在没有命中规则时已经把 ID 置空。
    - ``profile_name == "generic-text"``：兜底 Profile 不算「真匹配」，无论它有没有
      被赋予一个具体 ID。

    Args:
        profile_id: ``profile_match`` 任务返回的 profile UUID 字符串，``None`` 表示
            匹配失败已落到兜底。
        profile_name: 匹配到的 Profile 名字。

    Returns:
        True 如果应该被视为「没有匹配到具体 Profile」。
    """
    if profile_id is None:
        return True
    if isinstance(profile_name, str) and profile_name == GENERIC_PROFILE_NAME:
        return True
    return False


def is_quality_below_threshold(
    quality_score: float | None,
    threshold: float | None = None,
) -> bool:
    """质量分是否严格低于阈值。

    ``quality_score is None`` 表示评分阶段还没运行（任务 11 才接入），此时返回
    ``False`` —— 不能因为「还没评分」就强行触发兜底。

    Args:
        quality_score: 解析质量综合分（0~1），``None`` 表示尚未计算。
        threshold: 触发阈值。``None`` 时从 ``settings.QUALITY_FALLBACK_THRESHOLD``
            读取（默认 0.7）。

    Returns:
        True 当 ``quality_score is not None and quality_score < effective_threshold``。
    """
    if quality_score is None:
        return False
    effective_threshold = threshold
    if effective_threshold is None:
        effective_threshold = float(get_settings().QUALITY_FALLBACK_THRESHOLD)
    return quality_score < effective_threshold


def should_run_universal_parser(
    profile_id: str | None,
    profile_name: str | None,
    quality_score: float | None = None,
    threshold: float | None = None,
) -> tuple[bool, list[str]]:
    """综合判定是否应该运行 Universal Parser，并返回所有命中的触发原因。

    返回值的 ``reasons`` 列表保持稳定顺序，便于在日志 / metadata 里直接断言：

    1. ``"no_profile_match"`` 优先于 ``"quality_below_threshold"``。
    2. 一个原因都没有命中 → ``(False, [])``。
    3. 任意一个命中 → ``(True, [...])``。

    Args:
        profile_id: ``profile_match`` 返回的 ID（``None`` 表示兜底）。
        profile_name: ``profile_match`` 返回的 Profile 名。
        quality_score: 当前文档的解析质量分。Task 11 实现 QualityScorer 之后才会
            在管线里被填充；在此之前调用方应传 ``None``。
        threshold: 显式覆盖阈值。``None`` 时回落到 settings。

    Returns:
        ``(should_run, reasons)`` 二元组。
    """
    reasons: list[str] = []
    if is_no_profile_match(profile_id, profile_name):
        reasons.append(TRIGGER_NO_PROFILE_MATCH)
    if is_quality_below_threshold(quality_score, threshold):
        reasons.append(TRIGGER_QUALITY_BELOW_THRESHOLD)
    return (bool(reasons), reasons)


# ─── 编排：parse → suggest_profile → save_candidate ──────────────────────


def _serialize_processed_document(processed: Any) -> Any:
    """把 ``ProcessedDocument`` 序列化为 dict，方便跨 Celery 任务边界传输。

    ``ProcessedDocument`` 是 dataclass，``asdict`` 能递归处理嵌套 dataclass。
    如果传入的不是 dataclass（例如调用方在测试里传了 dict），原样返回。
    """
    if processed is None:
        return None
    if is_dataclass(processed):
        return asdict(processed)
    return processed


async def run_universal_parser_and_persist_candidate(
    parsed_doc: "ParsedDocument",
    *,
    db: "AsyncSession | None",
    llm_gateway: Any | None = None,
    threshold: float | None = None,  # noqa: ARG001 — reserved for future quality-score plumbing
    parser: "UniversalParser | None" = None,
) -> dict[str, Any]:
    """运行 LLM 兜底解析并把候选 Profile 持久化。

    流程：

    1. 实例化 ``UniversalParser``（调用方可注入 ``parser`` / ``llm_gateway``，
       便于测试隔离）。
    2. ``await parser.parse(parsed_doc)`` 得到 ``ProcessedDocument``。
    3. ``await parser.suggest_profile(processed)`` 生成候选 envelope。
    4. 当 ``db`` 不为 ``None`` 时，调用 ``profile_candidate_service.save_candidate``
       持久化。如果 envelope 校验失败（``ValueError``）只记 WARNING 日志、把
       ``candidate_profile_id`` 设为 ``None``，让 ``processed_document`` 继续流向
       下游避免文档卡住。
    5. 持久化成功后调用 ``db.commit()`` 落库，与 FastAPI ``get_db`` 的事务模型
       保持一致。

    Args:
        parsed_doc: 上游解析器产出的 IR。
        db: AsyncSession；``None`` 时跳过持久化（仅返回 LLM 解析结果）。
        llm_gateway: 可选的 LLMGateway，留给调用方注入定制网关 / mock。
        threshold: 当前未使用，保留参数位以便后续 Task 11 把质量分纳入决策。
        parser: 可选的 ``UniversalParser`` 实例，主要给单元测试注入 mock。

    Returns:
        ``{
            "processed_document": <serialized ProcessedDocument | None>,
            "candidate_profile_id": <uuid str | None>,
            "trigger_reasons": [],
        }``

        ``trigger_reasons`` 由调用方填充（管线任务知道触发了哪几条规则），本函数
        只负责执行 LLM + 持久化；保留键以便管线侧统一处理。
    """
    # 局部导入，避免在模块顶层引入 UniversalParser → 触发 LLM Gateway 等重型依赖。
    if parser is None:
        from app.services.universal_parser import UniversalParser

        parser = UniversalParser(llm_gateway=llm_gateway)

    processed = await parser.parse(parsed_doc)
    candidate_envelope = await parser.suggest_profile(processed)

    candidate_profile_id: str | None = None

    if db is not None:
        # 局部导入，与上同理；同时避免循环导入风险。
        from app.services.profile_candidate_service import save_candidate

        try:
            saved = await save_candidate(db, candidate_envelope)
            await db.commit()
            candidate_profile_id = str(saved.id) if getattr(saved, "id", None) else None
        except ValueError as exc:
            # envelope 形状非法 / 名字冲突无法解决 —— 不阻塞文档处理，只警告。
            logger.warning(
                "universal_parser candidate envelope rejected: %s",
                exc,
            )
            try:
                await db.rollback()
            except Exception as rollback_exc:  # noqa: BLE001 — defensive
                logger.warning(
                    "universal_parser db rollback failed after envelope rejection: %s",
                    rollback_exc,
                )

    return {
        "processed_document": _serialize_processed_document(processed),
        "candidate_profile_id": candidate_profile_id,
        "trigger_reasons": [],
    }
