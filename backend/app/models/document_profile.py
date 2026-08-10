"""DocumentProfile model for configurable document parsing strategies."""

import uuid

from sqlalchemy import Boolean, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDMixin


class DocumentProfile(Base, UUIDMixin, TimestampMixin):
    """Profile defining parsing strategy for a category of documents."""

    __tablename__ = "document_profiles"

    name: Mapped[str] = mapped_column(
        String(100), unique=True, nullable=False, index=True
    )
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    priority: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=False
    )
    enabled: Mapped[bool] = mapped_column(
        Boolean, default=True, server_default="true", nullable=False
    )
    match_rules: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default="{}"
    )
    heading_rules: Mapped[list] = mapped_column(
        JSONB, nullable=False, server_default="[]"
    )
    boilerplate: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default="{}"
    )
    tables: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default="{}"
    )
    chunking: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default="{}"
    )
    domain_dictionary_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("domain_dictionaries.id", ondelete="SET NULL"),
        nullable=True,
    )
    version: Mapped[int] = mapped_column(
        Integer, default=1, server_default="1", nullable=False
    )

    # Relationships
    domain_dictionary = relationship("DomainDictionary", lazy="selectin")
    versions = relationship(
        "ProfileVersion",
        back_populates="profile",
        lazy="selectin",
        cascade="all, delete-orphan",
    )
