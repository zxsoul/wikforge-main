"""User model for authentication and identity."""

from datetime import datetime

from sqlalchemy import DateTime, Index, Integer, String, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDMixin


class User(Base, UUIDMixin, TimestampMixin):
    """User account model supporting local and OIDC authentication.

    设计参考：design.md PostgreSQL ER 图。
    - 本地账号：使用 ``email`` + ``password_hash`` 登录。
    - OIDC 账号：使用 ``oidc_provider`` + ``oidc_subject`` 唯一标识，
      允许 ``password_hash`` 为空。
    - 登录失败保护：``failed_login_count`` 计数，``locked_until`` 标记
      解锁时间（带时区）。
    """

    __tablename__ = "users"

    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )
    password_hash: Mapped[str | None] = mapped_column(String(255), nullable=True)
    display_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    oidc_provider: Mapped[str | None] = mapped_column(String(64), nullable=True)
    oidc_subject: Mapped[str | None] = mapped_column(String(255), nullable=True)
    failed_login_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        server_default="0",
        nullable=False,
    )
    locked_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Relationships
    spaces = relationship("Space", back_populates="creator", lazy="selectin")
    permissions = relationship("Permission", back_populates="user", lazy="selectin")
    chat_sessions = relationship("ChatSession", back_populates="user", lazy="selectin")

    __table_args__ = (
        # OIDC 身份的复合唯一约束（仅在 provider 与 subject 同时非空时生效），
        # 通过 PostgreSQL partial unique index 实现，避免本地账号的 NULL 冲突。
        Index(
            "uq_users_oidc_provider_subject",
            "oidc_provider",
            "oidc_subject",
            unique=True,
            postgresql_where=text(
                "oidc_provider IS NOT NULL AND oidc_subject IS NOT NULL"
            ),
        ),
    )
