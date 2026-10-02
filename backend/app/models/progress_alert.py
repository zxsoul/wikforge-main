"""ProgressAlert model for stalled-work early warning.

主管端「进度预警」工作流的持久化表。Celery Beat 每日定时任务
``alerts.detect_stalled_work`` 汇总空间内最近 N 天的日报，交给 LLM
提取工作项及其进度轨迹，识别「连续多天无进展 / 从日报中消失 / 风险
反复出现」的工作项，写入本表。

设计要点：
- 去重靠查询端保证：同一 (space_id, work_item) 存在 open 状态的预警时
  不再新建，只刷新 ``last_active_date`` 与 ``evidence``。
- ``severity`` 三档（info/warning/critical），由 LLM 按停滞天数给出，
  主管端面板按 severity 排序展示。
- 预警只提示、不考核——定位是「帮主管发现被忽略的进度死角」。
"""

import enum
import uuid
from datetime import date, datetime

from sqlalchemy import Date, DateTime, Enum, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDMixin


class AlertSeverity(str, enum.Enum):
    """Alert severity levels."""

    info = "info"
    warning = "warning"
    critical = "critical"


class AlertStatus(str, enum.Enum):
    """Alert lifecycle status."""

    open = "open"
    acknowledged = "acknowledged"
    resolved = "resolved"


class ProgressAlert(Base, UUIDMixin, TimestampMixin):
    """进度停滞预警记录。"""

    __tablename__ = "progress_alerts"
    __table_args__ = (
        Index("ix_progress_alert_space_status", "space_id", "status"),
    )

    space_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("spaces.id", ondelete="CASCADE"), nullable=False
    )
    # 被判定停滞的工作项名称（如 "支付模块联调"）。
    work_item: Mapped[str] = mapped_column(String(500), nullable=False)
    # LLM 给出的停滞判定理由（面向主管的中文描述）。
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[AlertSeverity] = mapped_column(
        Enum(AlertSeverity, name="alert_severity"),
        default=AlertSeverity.warning,
        server_default="warning",
        nullable=False,
    )
    status: Mapped[AlertStatus] = mapped_column(
        Enum(AlertStatus, name="alert_status"),
        default=AlertStatus.open,
        server_default="open",
        nullable=False,
        index=True,
    )
    # 判定证据：{"dates": [...], "excerpts": [...], "stalled_days": n}
    evidence: Mapped[dict] = mapped_column(
        JSONB, nullable=False, default=dict, server_default="{}"
    )
    first_seen_date: Mapped[date] = mapped_column(Date, nullable=False)
    last_active_date: Mapped[date] = mapped_column(Date, nullable=False)
    acknowledged_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    acknowledged_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    space = relationship("Space", lazy="selectin")
    acknowledger = relationship("User", lazy="selectin")
