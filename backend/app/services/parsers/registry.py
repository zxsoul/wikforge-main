"""Parser plugin registry with registration, selection, and hot-reload support.

The ParserRegistry manages all available parser plugins and selects the
appropriate one based on file extension, MIME type, and priority.
"""

import importlib
import logging
from typing import Any

from app.services.parsers.base import ParserPlugin

logger = logging.getLogger(__name__)


class ParserRegistry:
    """Registry for parser plugins with hot-reload support.

    Manages plugin lifecycle: registration, unregistration, selection by
    file type, and hot-reloading from database configuration.
    """

    def __init__(self) -> None:
        self._plugins: list[ParserPlugin] = []

    @property
    def plugins(self) -> list[ParserPlugin]:
        """Return a copy of the registered plugins list."""
        return list(self._plugins)

    def register(self, plugin: ParserPlugin) -> None:
        """Register a parser plugin.

        Args:
            plugin: Plugin instance implementing ParserPlugin protocol

        Raises:
            ValueError: If a plugin with the same name is already registered
        """
        if any(p.name == plugin.name for p in self._plugins):
            raise ValueError(f"Plugin '{plugin.name}' is already registered")
        self._plugins.append(plugin)
        # Keep sorted by priority (highest first) for selection
        self._plugins.sort(key=lambda p: p.priority, reverse=True)
        logger.info(f"Registered parser plugin: {plugin.name} (priority={plugin.priority})")

    def unregister(self, name: str) -> None:
        """Unregister a parser plugin by name.

        Args:
            name: Name of the plugin to remove

        Raises:
            ValueError: If no plugin with the given name is registered
        """
        original_count = len(self._plugins)
        self._plugins = [p for p in self._plugins if p.name != name]
        if len(self._plugins) == original_count:
            raise ValueError(f"Plugin '{name}' is not registered")
        logger.info(f"Unregistered parser plugin: {name}")

    def select(self, file_path: str, mime_type: str = "") -> ParserPlugin:
        """Select the best parser plugin for a given file.

        Iterates through plugins sorted by priority (highest first) and
        returns the first one that can parse the file.

        Args:
            file_path: Path to the file
            mime_type: MIME type of the file (optional)

        Returns:
            The highest-priority plugin that can handle the file

        Raises:
            ValueError: If no plugin can handle the file
        """
        for plugin in self._plugins:
            if plugin.can_parse(file_path, mime_type):
                return plugin
        raise ValueError(
            f"No parser plugin found for file: {file_path} (mime_type={mime_type})"
        )

    def get_plugin(self, name: str) -> ParserPlugin | None:
        """Get a registered plugin by name.

        Args:
            name: Plugin name to look up

        Returns:
            The plugin instance, or None if not found
        """
        for plugin in self._plugins:
            if plugin.name == name:
                return plugin
        return None

    def clear(self) -> None:
        """Remove all registered plugins."""
        self._plugins.clear()
        logger.info("Cleared all parser plugins")

    def reload_from_configs(self, configs: list[dict[str, Any]]) -> None:
        """Hot-reload plugins from configuration list.

        Clears existing plugins and loads new ones from the provided
        configuration dictionaries (typically from ParserPluginConfig table).

        Each config dict should have:
            - name: Plugin name
            - import_path: Python import path (e.g., "app.services.parsers.pdf_parser.PdfParser")
            - supported_extensions: List of file extensions
            - priority: Integer priority
            - enabled: Boolean flag
            - config: Additional plugin configuration dict

        Args:
            configs: List of plugin configuration dictionaries
        """
        self.clear()
        for cfg in configs:
            if not cfg.get("enabled", True):
                logger.info(f"Skipping disabled plugin: {cfg.get('name', 'unknown')}")
                continue
            try:
                plugin = self._load_plugin(cfg)
                self.register(plugin)
            except Exception as e:
                logger.error(
                    f"Failed to load plugin '{cfg.get('name', 'unknown')}' "
                    f"from '{cfg.get('import_path', '')}': {e}"
                )

    def reload_from_db_records(self, records: list[Any]) -> None:
        """Hot-reload plugins from ParserPluginConfig ORM records.

        Convenience adapter that maps SQLAlchemy/ORM records (or anything
        attribute-accessible) to the dict shape expected by
        :meth:`reload_from_configs`.

        Args:
            records: Iterable of objects exposing ``name``, ``import_path``,
                ``supported_extensions``, ``priority``, ``enabled``, ``config``.
        """
        configs: list[dict[str, Any]] = []
        for r in records:
            configs.append(
                {
                    "name": getattr(r, "name"),
                    "import_path": getattr(r, "import_path"),
                    "supported_extensions": list(getattr(r, "supported_extensions", []) or []),
                    "priority": int(getattr(r, "priority", 0) or 0),
                    "enabled": bool(getattr(r, "enabled", True)),
                    "config": dict(getattr(r, "config", {}) or {}),
                }
            )
        self.reload_from_configs(configs)

    async def load_from_database(self, session: Any) -> int:
        """Load and register plugins from the ``parser_plugin_configs`` table.

        Reads all enabled records from ``ParserPluginConfig`` (sorted by
        priority desc) using an async SQLAlchemy session, then hot-reloads
        the registry.

        Args:
            session: Async SQLAlchemy session (``AsyncSession``)

        Returns:
            Number of plugins successfully registered.
        """
        from sqlalchemy import select

        from app.models.parser_plugin_config import ParserPluginConfig

        stmt = (
            select(ParserPluginConfig)
            .where(ParserPluginConfig.enabled.is_(True))
            .order_by(ParserPluginConfig.priority.desc())
        )
        result = await session.execute(stmt)
        records = list(result.scalars().all())
        self.reload_from_db_records(records)
        return len(self._plugins)

    def _load_plugin(self, config: dict[str, Any]) -> ParserPlugin:
        """Load a plugin instance from its import path.

        Args:
            config: Plugin configuration dict with import_path and config

        Returns:
            Instantiated plugin

        Raises:
            ImportError: If the module or class cannot be imported
            TypeError: If the loaded class doesn't implement ParserPlugin
        """
        import_path = config["import_path"]
        # Accept both "pkg.module.Class" and "pkg.module:Class" forms
        if ":" in import_path:
            module_path, class_name = import_path.split(":", 1)
        else:
            module_path, class_name = import_path.rsplit(".", 1)

        module = importlib.import_module(module_path)
        plugin_class = getattr(module, class_name)

        # Instantiate with optional config
        plugin_config = config.get("config", {})
        plugin = plugin_class(**plugin_config) if plugin_config else plugin_class()

        # Override priority from config if specified
        if "priority" in config:
            plugin.priority = config["priority"]

        return plugin


# Global registry instance
_registry: ParserRegistry | None = None


def get_parser_registry() -> ParserRegistry:
    """Get or create the global parser registry instance."""
    global _registry
    if _registry is None:
        _registry = ParserRegistry()
    return _registry


def reset_parser_registry() -> None:
    """Reset the global parser registry (for testing)."""
    global _registry
    _registry = None
