"""Chat session and message models for RAG conversations."""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, CreatedAtMixin, UUIDMixin


class ChatSession(Base, UUIDMixin, CreatedAtMixin):
    """A conversation session between a user and the RAG engine."""

    __tablename__ = "chat_sessions"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    last_active_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    is_expired: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="false", nullable=False
    )

    # Relationships
    user = relationship("User", back_populates="chat_sessions", lazy="selectin")
    messages = relationship(
        "ChatMessage",
        back_populates="session",
        lazy="selectin",
        cascade="all, delete-orphan",
        order_by="ChatMessage.sequence_number",
    )


class ChatMessage(Base, UUIDMixin, CreatedAtMixin):
    """A single message within a chat session."""

    __tablename__ = "chat_messages"
    __table_args__ = (
        # 复合索引：会话内按 sequence_number 排序消息（设计要求）
        Index(
            "ix_chat_messages_session_sequence",
            "session_id",
            "sequence_number",
        ),
    )

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # "user" | "assistant" | "system"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    sequence_number: Mapped[int] = mapped_column(Integer, nullable=False)

    # Relationships
    session = relationship("ChatSession", back_populates="messages", lazy="selectin")
