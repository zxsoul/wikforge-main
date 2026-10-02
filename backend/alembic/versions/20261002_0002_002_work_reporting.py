"""add daily_reports and progress_alerts (work reporting scenario)

工作汇报场景换皮新增两张表：

- ``daily_reports``：AI 合成的员工日报（按 space+user+date 唯一），
  记录来源文档与回流入库的日报文档 ID，支撑工作留痕与主管端检索。
- ``progress_alerts``：进度停滞预警，由 Celery Beat 每日检测任务产出，
  主管端确认（acknowledge）后进入跟进状态。

Revision ID: 002
Revises: 001
Create Date: 2026-10-02
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


DAILY_REPORT_STATUS = postgresql.ENUM(
    "generating", "completed", "failed",
    name="daily_report_status",
    create_type=False,
)
ALERT_SEVERITY = postgresql.ENUM(
    "info", "warning", "critical",
    name="alert_severity",
    create_type=False,
)
ALERT_STATUS = postgresql.ENUM(
    "open", "acknowledged", "resolved",
    name="alert_status",
    create_type=False,
)


def upgrade() -> None:
    bind = op.get_bind()

    DAILY_REPORT_STATUS.create(bind, checkfirst=True)
    ALERT_SEVERITY.create(bind, checkfirst=True)
    ALERT_STATUS.create(bind, checkfirst=True)

    op.create_table(
        "daily_reports",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "space_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("spaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("report_date", sa.Date(), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("content", sa.Text(), nullable=False, server_default=""),
        sa.Column(
            "source_document_ids",
            postgresql.JSONB(),
            nullable=False,
            server_default="[]",
        ),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "status",
            DAILY_REPORT_STATUS,
            nullable=False,
            server_default="generating",
        ),
        sa.Column("error_detail", sa.Text(), nullable=True),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "space_id", "user_id", "report_date",
            name="uq_daily_report_space_user_date",
        ),
    )
    op.create_index(
        "ix_daily_reports_report_date", "daily_reports", ["report_date"]
    )
    op.create_index("ix_daily_reports_status", "daily_reports", ["status"])

    op.create_table(
        "progress_alerts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "space_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("spaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("work_item", sa.String(length=500), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column(
            "severity", ALERT_SEVERITY, nullable=False, server_default="warning"
        ),
        sa.Column(
            "status", ALERT_STATUS, nullable=False, server_default="open"
        ),
        sa.Column(
            "evidence", postgresql.JSONB(), nullable=False, server_default="{}"
        ),
        sa.Column("first_seen_date", sa.Date(), nullable=False),
        sa.Column("last_active_date", sa.Date(), nullable=False),
        sa.Column(
            "acknowledged_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "ix_progress_alert_space_status",
        "progress_alerts",
        ["space_id", "status"],
    )
    op.create_index("ix_progress_alerts_status", "progress_alerts", ["status"])


def downgrade() -> None:
    op.drop_index("ix_progress_alerts_status", table_name="progress_alerts")
    op.drop_index(
        "ix_progress_alert_space_status", table_name="progress_alerts"
    )
    op.drop_table("progress_alerts")

    op.drop_index("ix_daily_reports_status", table_name="daily_reports")
    op.drop_index("ix_daily_reports_report_date", table_name="daily_reports")
    op.drop_table("daily_reports")

    bind = op.get_bind()
    ALERT_STATUS.drop(bind, checkfirst=True)
    ALERT_SEVERITY.drop(bind, checkfirst=True)
    DAILY_REPORT_STATUS.drop(bind, checkfirst=True)
