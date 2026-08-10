"""DocumentReview model for quality review workflow."""

import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, CreatedAtMixin, UUIDMixin


class ReviewStatus(str, enum.Enum):
    """Review status enum."""

    pending = "pending"
    approved = "approved"
    corrected = "corrected"
    rejected = "rejected"


class DocumentReview(Base, UUIDMixin, CreatedAtMixin):
    """Review record for document parsing quality assessment."""

    __tablename__ = "document_reviews"

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    quality_score: Mapped[dict] = mapped_column(JSONB, nullable=False)
    status: Mapped[ReviewStatus] = mapped_column(
        Enum(ReviewStatus, name="review_status"),
        default=ReviewStatus.pending,
        server_default="pending",
        nullable=False,
    )
    reviewer_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True
    )
    reviewer_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    document = relationship("Document", back_populates="reviews", lazy="selectin")
    reviewer = relationship("User", lazy="selectin")
