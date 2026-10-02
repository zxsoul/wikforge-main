"""工作日报相关的 API 路由（工作汇报场景 · 员工端 + 主管端）。

- ``POST /api/reports/generate``：员工为自己生成某天的日报（幂等）
- ``GET  /api/reports/my``：员工查看自己的历史日报
- ``GET  /api/reports/team``：主管查看空间内全员日报（空间创建者/管理员）
- ``GET  /api/reports/{report_id}``：查看单份日报（本人或空间主管）

鉴权：全员接口依赖 :func:`get_current_user`；越权防护在
:class:`DailyReportService` 内完成——员工生成自己的日报需要空间 write
权限（与上传材料一致），「全员视图」要求空间创建者（主管）或系统管理员，
普通成员无法查看他人日报。
"""

import logging
import uuid
from datetime import date, datetime

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.core.database import get_db
from app.models.daily_report import DailyReport
from app.models.user import User
from app.services.report_service import DailyReportService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["reports"])


# ─── Request / Response Schemas ──────────────────────────────────────


class GenerateReportRequest(BaseModel):
    """日报生成请求。"""

    space_id: str = Field(..., description="目标空间（部门/项目）ID")
    report_date: date | None = Field(
        None, description="日报日期，缺省为今天（UTC）"
    )
    force: bool = Field(
        False, description="已存在日报时是否基于最新材料重新生成"
    )


class DailyReportResponse(BaseModel):
    """单份日报。"""

    id: str
    space_id: str
    user_id: str
    author: str
    report_date: date
    title: str
    content: str
    source_document_ids: list[str]
    document_id: str | None
    status: str
    error_detail: str | None
    generated_at: datetime | None
    created_at: datetime


class DailyReportListResponse(BaseModel):
    """日报列表。"""

    reports: list[DailyReportResponse]
    total: int


# ─── Helpers ──────────────────────────────────────────────────────────


def _to_response(report: DailyReport) -> DailyReportResponse:
    author = ""
    if report.user:
        author = report.user.display_name or report.user.email
    return DailyReportResponse(
        id=str(report.id),
        space_id=str(report.space_id),
        user_id=str(report.user_id),
        author=author,
        report_date=report.report_date,
        title=report.title,
        content=report.content,
        source_document_ids=report.source_document_ids or [],
        document_id=str(report.document_id) if report.document_id else None,
        status=report.status.value,
        error_detail=report.error_detail,
        generated_at=report.generated_at,
        created_at=report.created_at,
    )


# ─── Endpoints ────────────────────────────────────────────────────────


@router.post("/api/reports/generate", response_model=DailyReportResponse)
async def generate_report(
    request: GenerateReportRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DailyReportResponse:
    """生成当日（或指定日）我的工作日报。

    聚合当天本人上传且已入库的全部工作材料（截图/日志/文档），LLM 合成
    四段式日报并回流入库。同一天重复调用默认返回已有日报（幂等），
    传 ``force=true`` 可基于最新材料重新合成。
    """
    service = DailyReportService(db)
    report = await service.generate_daily_report(
        user=current_user,
        space_id=uuid.UUID(request.space_id),
        report_date=request.report_date or date.today(),
        force=request.force,
    )
    return _to_response(report)


@router.get("/api/reports/my", response_model=DailyReportListResponse)
async def list_my_reports(
    space_id: str | None = Query(None, description="按空间过滤"),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DailyReportListResponse:
    """员工查看自己的历史日报列表。"""
    service = DailyReportService(db)
    reports, total = await service.list_my_reports(
        user_id=current_user.id,
        space_id=uuid.UUID(space_id) if space_id else None,
        skip=skip,
        limit=limit,
    )
    return DailyReportListResponse(
        reports=[_to_response(r) for r in reports], total=total
    )


@router.get("/api/reports/team", response_model=DailyReportListResponse)
async def list_team_reports(
    space_id: str = Query(..., description="目标空间 ID"),
    report_date: date | None = Query(None, description="按日期过滤"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DailyReportListResponse:
    """主管查看空间内全员日报（需空间主管权限，否则 403）。"""
    service = DailyReportService(db)
    reports, total = await service.list_team_reports(
        requester=current_user,
        space_id=uuid.UUID(space_id),
        report_date=report_date,
        skip=skip,
        limit=limit,
    )
    return DailyReportListResponse(
        reports=[_to_response(r) for r in reports], total=total
    )


@router.get("/api/reports/{report_id}", response_model=DailyReportResponse)
async def get_report(
    report_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DailyReportResponse:
    """查看单份日报详情：本人或空间主管（write 权限）。"""
    service = DailyReportService(db)
    report = await service.get_report(
        requester=current_user, report_id=uuid.UUID(report_id)
    )
    return _to_response(report)
