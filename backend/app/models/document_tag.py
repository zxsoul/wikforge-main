"""DocumentTag model for document labeling."""

import uuid

from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, UUIDMixin


class DocumentTag(Base, UUIDMixin):
    """Tag associated with a document for categorization."""

    __tablename__ = "document_tags"
    __table_args__ = (
        UniqueConstraint("document_id", "tag_name", name="uq_document_tag"),
    )

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    tag_name: Mapped[str] = mapped_column(String(30), nullable=False, index=True)

    # Relationships
    document = relationship("Document", back_populates="tags", lazy="selectin")
