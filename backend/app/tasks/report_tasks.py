"""工作汇报场景的 Celery 定时任务。

两个 Beat 任务（调度在 ``app.core.celery_app`` 的 ``beat_schedule`` 注册）：

1. ``reports.auto_generate_daily``
   每天傍晚自动为「当天有上传工作材料」的 (space, user) 组合生成日报。
   员工即使忘记点「生成日报」，主管第二天早上也能看到全员日报。
   单个员工失败不影响其它人（与 pipeline 的隔离语义一致）。

2. ``alerts.detect_stalled_work``
   每晚汇总各空间最近 7 天日报，执行进度停滞检测，产出预警记录。

同步任务内通过 ``asyncio.new_event_loop()`` 运行异步服务，与
``app.tasks.pipeline`` 中的既有风格保持一致。
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, time, timezone

from sqlalchemy import select

from app.core.celery_app import celery_app

logger = logging.getLogger(__name__)


def _run_async(coro):
    """在同步 Celery 任务里运行异步协程（与 pipeline.py 同一风格）。"""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(
    name="reports.auto_generate_daily",
    soft_time_limit=570,
    time_limit=600,
)
def auto_generate_daily_reports(target_date: str | None = None) -> dict:
    """为当天有上传记录的所有 (space, user) 自动生成日报。

    Args:
        target_date: ``YYYY-MM-DD``，缺省为今天（UTC）。支持传参便于
            补跑历史日期或演示。

    Returns:
        ``{"date": ..., "generated": n, "skipped": n, "errors": [...]}``
    """
    from app.core.database import AsyncSessionLocal
    from app.models.daily_report import DailyReport, DailyReportStatus
    from app.models.document import Document, DocumentStatus
    from app.models.user import User
    from app.services.report_service import DailyReportService

    day = (
        date.fromisoformat(target_date)
        if target_date
        else datetime.now(timezone.utc).date()
    )
    day_start = datetime.combine(day, time.min, tzinfo=timezone.utc)
    day_end = datetime.combine(day, time.max, tzinfo=timezone.utc)

    async def _run() -> dict:
        summary = {"date": day.isoformat(), "generated": 0, "skipped": 0, "errors": []}
        async with AsyncSessionLocal() as session:
            # 找出当天有「已完成入库」上传记录的 (space_id, uploaded_by) 组合
            stmt = (
                select(Document.space_id, Document.uploaded_by)
                .where(
                    Document.status == DocumentStatus.completed,
                    Document.created_at >= day_start,
                    Document.created_at <= day_end,
                )
                .distinct()
            )
            pairs = (await session.execute(stmt)).all()

        for space_id, user_id in pairs:
            async with AsyncSessionLocal() as session:
                try:
                    # 已有完成的日报则跳过（幂等；force 重算走 API）
                    existing = await session.execute(
                        select(DailyReport).where(
                            DailyReport.space_id == space_id,
                            DailyReport.user_id == user_id,
                            DailyReport.report_date == day,
                            DailyReport.status == DailyReportStatus.completed,
                        )
                    )
                    if existing.scalar_one_or_none():
                        summary["skipped"] += 1
                        continue

                    user = (
                        await session.execute(
                            select(User).where(User.id == user_id)
                        )
                    ).scalar_one_or_none()
                    if not user:
                        continue

                    service = DailyReportService(session)
                    await service.generate_daily_report(
                        user=user,
                        space_id=space_id,
                        report_date=day,
                        force=False,
                    )
                    summary["generated"] += 1
                except Exception as exc:  # noqa: BLE001 — 单人失败不阻塞其他人
                    logger.warning(
                        "auto daily report failed: space=%s user=%s error=%s",
                        space_id,
                        user_id,
                        exc,
                        exc_info=True,
                    )
                    summary["errors"].append(
                        {
                            "space_id": str(space_id),
                            "user_id": str(user_id),
                            "error": str(exc)[:500],
                        }
                    )
                    await session.rollback()
        return summary

    return _run_async(_run())


@celery_app.task(
    name="alerts.detect_stalled_work",
    soft_time_limit=570,
    time_limit=600,
)
def detect_stalled_work(lookback_days: int | None = None) -> dict:
    """全空间进度停滞检测（每日 Beat 任务入口）。

    Args:
        lookback_days: 分析窗口天数，缺省用 AlertService 默认（7 天）。

    Returns:
        AlertService.detect_all_spaces 的汇总 dict。
    """
    from app.core.database import AsyncSessionLocal
    from app.services.alert_service import DEFAULT_LOOKBACK_DAYS, AlertService

    days = lookback_days or DEFAULT_LOOKBACK_DAYS

    async def _run() -> dict:
        async with AsyncSessionLocal() as session:
            service = AlertService(session)
            return await service.detect_all_spaces(lookback_days=days)

    return _run_async(_run())
