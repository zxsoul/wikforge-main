"""Permission model for ABAC access control."""

import enum
import uuid

from sqlalchemy import Enum, ForeignKey, Index, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDMixin


class AccessLevel(str, enum.Enum):
    """Access level for permissions."""

    invisible = "invisible"
    read = "read"
    write = "write"


class ResourceType(str, enum.Enum):
    """Type of resource the permission applies to."""

    space = "space"
    document = "document"


class Permission(Base, UUIDMixin, TimestampMixin):
    """Permission defines user access to a resource (space or document).

    设计参考：design.md ABAC 模型与 Permission Service。
    - ``(resource_id, resource_type, user_id)`` 三元组唯一，避免重复授权。
    - 增加 ``(user_id, resource_type)`` 复合索引以加速搜索时
      Pre-Filtering 构建用户可见空间/文档列表。
    """

    __tablename__ = "permissions"
    __table_args__ = (
        UniqueConstraint(
            "resource_id",
            "resource_type",
            "user_id",
            name="uq_permission_resource_user",
        ),
        Index("ix_permission_user_resource_type", "user_id", "resource_type"),
    )

    resource_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )
    resource_type: Mapped[ResourceType] = mapped_column(
        Enum(ResourceType, name="resource_type"), nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    access_level: Mapped[AccessLevel] = mapped_column(
        Enum(AccessLevel, name="access_level"), nullable=False
    )

    # Relationships
    user = relationship("User", back_populates="permissions", lazy="selectin")
