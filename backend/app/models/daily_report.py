"""DailyReport model for AI-synthesized daily work reports.

工作汇报场景的核心表：员工每天上传的截图 / 日志 / 文档先走常规文档
管线入库，随后由 ``DailyReportService`` 按 (space, user, date) 聚合
当日 chunk，调用 LLM 合成结构化日报，并把日报 Markdown 再次回流入库
（作为新的 Document 走完整 pipeline），供主管端 RAG 问答检索。

设计要点：
- ``(space_id, user_id, report_date)`` 唯一约束保证同一天同一人只有
  一份日报，重复生成走 ``force`` 覆盖语义。
- ``source_document_ids`` 记录日报依据的原始文档，保证可追溯（工作留痕）。
- ``document_id`` 指向回流入库的日报文档本身，检索侧通过它命中日报。
"""

import enum
import uuid
from datetime import date, datetime

from sqlalchemy import Date, DateTime, Enum, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDMixin


class DailyReportStatus(str, enum.Enum):
    """Daily report generation status."""

    generating = "generating"
    completed = "completed"
    failed = "failed"


class DailyReport(Base, UUIDMixin, TimestampMixin):
    """AI 合成的员工日报。"""

    __tablename__ = "daily_reports"
    __table_args__ = (
        UniqueConstraint(
            "space_id",
            "user_id",
            "report_date",
            name="uq_daily_report_space_user_date",
        ),
    )

    space_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("spaces.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    report_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    # 依据的原始文档 ID 列表（员工当日上传的截图/日志/文档），用于留痕追溯。
    source_document_ids: Mapped[list] = mapped_column(
        JSONB, nullable=False, default=list, server_default="[]"
    )
    # 回流入库的日报 Document（走完整 pipeline 后主管端可检索到）。
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
    )
    status: Mapped[DailyReportStatus] = mapped_column(
        Enum(DailyReportStatus, name="daily_report_status"),
        default=DailyReportStatus.generating,
        server_default="generating",
        nullable=False,
        index=True,
    )
    error_detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    generated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    space = relationship("Space", lazy="selectin")
    user = relationship("User", lazy="selectin")
    document = relationship("Document", lazy="selectin")
