"""SQLAlchemy models package - import all models for Alembic discovery."""

from app.models.base import Base, CreatedAtMixin, TimestampMixin, UUIDMixin
from app.models.chat import ChatMessage, ChatSession
from app.models.document import Document, DocumentStatus
from app.models.document_profile import DocumentProfile
from app.models.document_review import DocumentReview, ReviewStatus
from app.models.document_tag import DocumentTag
from app.models.domain_dictionary import DomainDictionary
from app.models.folder import Folder
from app.models.parser_plugin_config import ParserPluginConfig
from app.models.permission import AccessLevel, Permission, ResourceType
from app.models.profile_version import ProfileVersion
from app.models.search_feedback import SearchFeedback
from app.models.space import Space
from app.models.user import User

__all__ = [
    "Base",
    "UUIDMixin",
    "TimestampMixin",
    "CreatedAtMixin",
    "User",
    "Space",
    "Folder",
    "Document",
    "DocumentStatus",
    "DocumentTag",
    "Permission",
    "AccessLevel",
    "ResourceType",
    "ChatSession",
    "ChatMessage",
    "DocumentProfile",
    "ProfileVersion",
    "DomainDictionary",
    "ParserPluginConfig",
    "DocumentReview",
    "ReviewStatus",
    "SearchFeedback",
]
