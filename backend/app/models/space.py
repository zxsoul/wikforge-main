"""Space model for top-level document organization."""

import uuid

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDMixin


class Space(Base, UUIDMixin, TimestampMixin):
    """Space is the top-level organizational unit for documents."""

    __tablename__ = "spaces"

    name: Mapped[str] = mapped_column(
        String(50), unique=True, nullable=False, index=True
    )
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Relationships
    creator = relationship("User", back_populates="spaces", lazy="selectin")
    folders = relationship(
        "Folder", back_populates="space", lazy="selectin", cascade="all, delete-orphan"
    )
    documents = relationship(
        "Document", back_populates="space", lazy="selectin", cascade="all, delete-orphan"
    )
