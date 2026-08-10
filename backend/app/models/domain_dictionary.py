"""DomainDictionary model for industry-specific terminology management."""

from sqlalchemy import Boolean, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class DomainDictionary(Base, UUIDMixin, TimestampMixin):
    """Domain-specific dictionary for terminology, synonyms, and stop words."""

    __tablename__ = "domain_dictionaries"

    name: Mapped[str] = mapped_column(
        String(100), unique=True, nullable=False, index=True
    )
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    terms: Mapped[list] = mapped_column(
        JSONB, nullable=False, server_default="[]"
    )
    synonyms: Mapped[list] = mapped_column(
        JSONB, nullable=False, server_default="[]"
    )
    stop_words: Mapped[list] = mapped_column(
        JSONB, nullable=False, server_default="[]"
    )
    enabled: Mapped[bool] = mapped_column(
        Boolean, default=True, server_default="true", nullable=False
    )
