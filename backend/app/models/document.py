"""Document model for file metadata and processing state."""

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDMixin


class DocumentStatus(str, enum.Enum):
    """Document processing status enum.

    设计参考：design.md PostgreSQL ER 图、文档处理状态机
    （待处理→解析中→清洗中→分块中→向量化中→入库中→已完成/失败）。
    """

    pending = "pending"
    parsing = "parsing"
    cleaning = "cleaning"
    chunking = "chunking"
    embedding = "embedding"
    indexing = "indexing"
    completed = "completed"
    failed = "failed"


class Document(Base, UUIDMixin, TimestampMixin):
    """Document represents an imported file and its processing state."""

    __tablename__ = "documents"

    space_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("spaces.id", ondelete="CASCADE"), nullable=False
    )
    folder_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("folders.id", ondelete="SET NULL"),
        nullable=True,
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    file_type: Mapped[str] = mapped_column(String(20), nullable=False)
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    storage_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus, name="document_status"),
        default=DocumentStatus.pending,
        server_default="pending",
        nullable=False,
        index=True,
    )
    retry_count: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=False
    )
    error_detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    current_stage: Mapped[str | None] = mapped_column(String(50), nullable=True)
    progress_percent: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=False
    )
    matched_profile_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_profiles.id", ondelete="SET NULL"),
        nullable=True,
    )
    # quality_score 使用 JSONB 以承载多维分数（overall + components + issues），
    # 与 design.md 中 ParseQualityScore 数据结构一致。
    quality_score: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    uploaded_by: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False
    )
    last_status_update: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    space = relationship("Space", back_populates="documents", lazy="selectin")
    folder = relationship("Folder", back_populates="documents", lazy="selectin")
    uploader = relationship("User", lazy="selectin")
    tags = relationship(
        "DocumentTag",
        back_populates="document",
        lazy="selectin",
        cascade="all, delete-orphan",
    )
    matched_profile = relationship("DocumentProfile", lazy="selectin")
    reviews = relationship(
        "DocumentReview",
        back_populates="document",
        lazy="selectin",
        cascade="all, delete-orphan",
    )
