"""Folder model for hierarchical document organization within spaces."""

import uuid

from sqlalchemy import ForeignKey, Index, Integer, String, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, CreatedAtMixin, UUIDMixin


class Folder(Base, UUIDMixin, CreatedAtMixin):
    """Folder supports up to 10 levels of nesting within a space.

    设计参考：design.md PostgreSQL ER 图。
    - ``depth`` 默认值 ``1`` 表示根目录（与 services 层 ``create_folder`` 默认行为
      保持一致），最大允许 10 级。该约束在服务层强校验。
    - 同级目录不允许重名（``space_id``、``parent_id``、``name`` 三元组唯一）。
      由于 PostgreSQL 中 ``NULL != NULL``，所以使用一条全列唯一约束（覆盖
      ``parent_id`` 非空场景）+ 一条 partial unique index 单独覆盖
      ``parent_id IS NULL`` 的根目录场景。
    """

    __tablename__ = "folders"
    __table_args__ = (
        UniqueConstraint(
            "space_id", "parent_id", "name", name="uq_folder_sibling_name"
        ),
        # Partial unique index：同一空间下的根目录（parent_id IS NULL）名称唯一
        Index(
            "uq_folder_sibling_name_root",
            "space_id",
            "name",
            unique=True,
            postgresql_where=text("parent_id IS NULL"),
        ),
    )

    space_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("spaces.id", ondelete="CASCADE"), nullable=False
    )
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("folders.id", ondelete="CASCADE"), nullable=True
    )
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    depth: Mapped[int] = mapped_column(
        Integer, default=1, server_default="1", nullable=False
    )

    # Relationships
    space = relationship("Space", back_populates="folders", lazy="selectin")
    parent = relationship(
        "Folder", remote_side="Folder.id", back_populates="children", lazy="selectin"
    )
    children = relationship(
        "Folder", back_populates="parent", lazy="selectin", cascade="all, delete-orphan"
    )
    documents = relationship(
        "Document", back_populates="folder", lazy="selectin", cascade="all, delete-orphan"
    )
