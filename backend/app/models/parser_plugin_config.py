"""ParserPluginConfig model for plugin registration and configuration."""

from sqlalchemy import Boolean, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class ParserPluginConfig(Base, UUIDMixin, TimestampMixin):
    """Configuration for a registered parser plugin.

    设计参考：design.md Parser Plugin（格式插件层）+ ParserRegistry。
    - ``import_path`` 形如 ``app.services.parsers.pdf_parser:PdfParser``，
      限制 255 字符即可覆盖常规 Python 模块路径。
    - ``supported_extensions`` 存储为 JSONB list[str]，例如 ``[".pdf", ".PDF"]``。
    - ``priority`` 用于多插件命中时的选择策略。
    """

    __tablename__ = "parser_plugin_configs"

    name: Mapped[str] = mapped_column(
        String(100), unique=True, nullable=False, index=True
    )
    import_path: Mapped[str] = mapped_column(String(255), nullable=False)
    supported_extensions: Mapped[list] = mapped_column(
        JSONB, nullable=False, server_default="[]"
    )
    priority: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=False
    )
    enabled: Mapped[bool] = mapped_column(
        Boolean, default=True, server_default="true", nullable=False
    )
    config: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default="{}"
    )
