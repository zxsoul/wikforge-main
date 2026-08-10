"""SearchFeedback model for user feedback on search quality."""

import uuid

from sqlalchemy import ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, CreatedAtMixin, UUIDMixin


class SearchFeedback(Base, UUIDMixin, CreatedAtMixin):
    """User feedback on search results for quality iteration.

    设计参考：design.md Feedback Loop（反馈迭代层）。
    - ``user_id`` 允许为空：保留匿名 / 已注销用户反馈，便于全量分析。
    - 复合索引 ``(related_profile_id, created_at)`` 支撑按 Profile + 时间窗的聚合分析。
    - 复合索引 ``(user_id, created_at)`` 支撑用户视角的反馈历史浏览。
    """

    __tablename__ = "search_feedbacks"
    __table_args__ = (
        Index(
            "ix_search_feedback_profile_created",
            "related_profile_id",
            "created_at",
        ),
        Index(
            "ix_search_feedback_user_created",
            "user_id",
            "created_at",
        ),
    )

    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    query: Mapped[str] = mapped_column(Text, nullable=False)
    returned_results: Mapped[list] = mapped_column(JSONB, nullable=False)
    feedback_type: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # "thumbs_up" | "thumbs_down" | "issue"
    issue_category: Mapped[str | None] = mapped_column(
        String(30), nullable=True
    )  # "irrelevant" | "missing_info" | "citation_error" | "format" | "other"
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    related_profile_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_profiles.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Relationships
    user = relationship("User", lazy="selectin")
    related_profile = relationship("DocumentProfile", lazy="selectin")
