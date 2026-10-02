"""进度预警相关的 API 路由（工作汇报场景 · 主管端）。

- ``GET  /api/alerts``：主管查看空间内的进度预警列表
- ``POST /api/alerts/{alert_id}/acknowledge``：主管确认预警（已知晓）
- ``POST /api/alerts/detect``：管理员手动触发一次全空间停滞检测
  （日常检测由 Celery Beat 任务 ``alerts.detect_stalled_work`` 每日执行，
  此接口用于演示与排错）

鉴权：查询/确认需要目标空间的主管权限（空间创建者或系统管理员）；
手动触发检测仅管理员。
"""

import logging
import uuid
from datetime import date, datetime

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user, require_admin
from app.core.database import get_db
from app.models.progress_alert import AlertStatus, ProgressAlert
from app.models.user import User
from app.services.alert_service import AlertService
from app.services.report_service import DailyReportService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["alerts"])


# ─── Request / Response Schemas ──────────────────────────────────────


class AlertResponse(BaseModel):
    """单条进度预警。"""

    id: str
    space_id: str
    work_item: str
    reason: str
    severity: str
    status: str
    evidence: dict
    first_seen_date: date
    last_active_date: date
    acknowledged_by: str | None
    acknowledged_at: datetime | None
    created_at: datetime


class AlertListResponse(BaseModel):
    """预警列表。"""

    alerts: list[AlertResponse]
    total: int


class DetectRequest(BaseModel):
    """手动触发检测请求。"""

    lookback_days: int = Field(7, ge=2, le=30, description="分析窗口（天）")


class DetectResponse(BaseModel):
    """检测执行结果汇总。"""

    spaces: int
    alerts_new: int
    errors: list[dict]


# ─── Helpers ──────────────────────────────────────────────────────────


def _to_response(alert: ProgressAlert) -> AlertResponse:
    return AlertResponse(
        id=str(alert.id),
        space_id=str(alert.space_id),
        work_item=alert.work_item,
        reason=alert.reason,
        severity=alert.severity.value,
        status=alert.status.value,
        evidence=alert.evidence or {},
        first_seen_date=alert.first_seen_date,
        last_active_date=alert.last_active_date,
        acknowledged_by=(
            str(alert.acknowledged_by) if alert.acknowledged_by else None
        ),
        acknowledged_at=alert.acknowledged_at,
        created_at=alert.created_at,
    )


# ─── Endpoints ────────────────────────────────────────────────────────


@router.get("/api/alerts", response_model=AlertListResponse)
async def list_alerts(
    space_id: str = Query(..., description="目标空间 ID"),
    status: str | None = Query(
        None, description="按状态过滤：open / acknowledged / resolved"
    ),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AlertListResponse:
    """主管查看空间内进度预警（需空间主管权限，否则 403）。"""
    # 空间级主管校验（与全员日报同一套判定：空间创建者或管理员）
    await DailyReportService(db).require_space_manager(
        current_user, uuid.UUID(space_id)
    )

    service = AlertService(db)
    alerts, total = await service.list_alerts(
        space_id=uuid.UUID(space_id),
        status=AlertStatus(status) if status else None,
        skip=skip,
        limit=limit,
    )
    return AlertListResponse(
        alerts=[_to_response(a) for a in alerts], total=total
    )


@router.post(
    "/api/alerts/{alert_id}/acknowledge", response_model=AlertResponse
)
async def acknowledge_alert(
    alert_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AlertResponse:
    """主管确认预警：状态 open → acknowledged，记录确认人与时间。"""
    service = AlertService(db)
    # 先查出预警拿到 space_id 做权限校验
    from sqlalchemy import select

    from app.models.progress_alert import ProgressAlert as PA

    row = (
        await db.execute(select(PA).where(PA.id == uuid.UUID(alert_id)))
    ).scalar_one_or_none()
    if row is None:
        from app.core.exceptions import NotFoundException

        raise NotFoundException("ProgressAlert", alert_id)
    await DailyReportService(db).require_space_manager(
        current_user, row.space_id
    )

    alert = await service.acknowledge(
        alert_id=uuid.UUID(alert_id), operator=current_user
    )
    return _to_response(alert)


@router.post("/api/alerts/detect", response_model=DetectResponse)
async def trigger_detection(
    request: DetectRequest,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> DetectResponse:
    """管理员手动触发一次全空间停滞检测（演示/排错用）。

    日常的每日检测由 Celery Beat 任务 ``alerts.detect_stalled_work``
    自动执行，本接口同步执行同样逻辑并返回结果汇总。
    """
    service = AlertService(db)
    summary = await service.detect_all_spaces(
        lookback_days=request.lookback_days
    )
    return DetectResponse(**summary)
