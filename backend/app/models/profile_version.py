"""ProfileVersion model for tracking profile change history."""

import uuid

from sqlalchemy import ForeignKey, Integer, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, CreatedAtMixin, UUIDMixin


class ProfileVersion(Base, UUIDMixin, CreatedAtMixin):
    """Version history entry for a DocumentProfile.

    设计参考：design.md ProfileVersion 数据模型。
    - 每个 Profile 的版本号在该 Profile 范围内唯一。
    - ``changed_by`` 允许为空：包含系统自动产生的版本（如初始化、批量导入），
      此时不归属于具体管理员。
    """

    __tablename__ = "profile_versions"
    __table_args__ = (
        UniqueConstraint("profile_id", "version", name="uq_profile_version"),
    )

    profile_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_profiles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    snapshot: Mapped[dict] = mapped_column(JSONB, nullable=False)
    changed_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    change_note: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    profile = relationship("DocumentProfile", back_populates="versions", lazy="selectin")
    changer = relationship("User", lazy="selectin")
