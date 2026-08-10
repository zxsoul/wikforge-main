"""Parser plugin protocol and intermediate representation data structures.

Defines:
- ParserPlugin: Protocol class that all parser plugins must implement
- ParsedDocument: Output of a parser (list of blocks + metadata + assets)
- Block: Minimal content unit (paragraph, heading, table, image, etc.)
- Asset: Attached binary resource (image, formula screenshot, etc.)
"""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class Block:
    """Minimal content unit produced by a parser.

    Attributes:
        type: Block type - "paragraph", "heading", "table", "image", "formula", "list"
        text: Textual content of the block
        bbox: Bounding box (x0, y0, x1, y1) in normalized coordinates, or None
        page_number: Page number where this block appears (1-indexed)
        style: Style metadata (font_size, bold, italic, heading_level, etc.)
        raw: Raw parser-specific data for debugging
    """

    type: str
    text: str
    bbox: tuple[float, float, float, float] | None = None
    page_number: int = 1
    style: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)


@dataclass
class Asset:
    """Attached binary resource extracted from a document.

    Attributes:
        id: Unique identifier for this asset
        type: Asset type - "image", "formula", "diagram"
        data: Raw binary data
        mime_type: MIME type of the asset (e.g., "image/png")
        page_number: Page where the asset was found
        bbox: Bounding box in normalized coordinates
        description: Optional text description
    """

    id: str
    type: str
    data: bytes
    mime_type: str
    page_number: int = 1
    bbox: tuple[float, float, float, float] | None = None
    description: str = ""


@dataclass
class ParsedDocument:
    """Intermediate representation output by a parser plugin.

    Attributes:
        blocks: Ordered list of content blocks
        metadata: File-level metadata (page_count, author, title, etc.)
        assets: Extracted binary assets (images, formulas)
    """

    blocks: list[Block] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    assets: list[Asset] = field(default_factory=list)


@runtime_checkable
class ParserPlugin(Protocol):
    """Protocol defining the interface for parser plugins.

    All parser plugins must implement this interface. Plugins are registered
    in the ParserRegistry and selected based on file extension and priority.

    Attributes:
        name: Human-readable plugin name
        supported_extensions: List of file extensions this plugin handles (e.g., ["pdf"])
        priority: Higher priority plugins are selected first (default 0)
    """

    name: str
    supported_extensions: list[str]
    priority: int

    def can_parse(self, file_path: str, mime_type: str) -> bool:
        """Check if this plugin can parse the given file.

        Args:
            file_path: Path to the file to check
            mime_type: MIME type of the file

        Returns:
            True if this plugin can handle the file
        """
        ...

    async def parse(self, file_path: str) -> ParsedDocument:
        """Parse a file and return the intermediate representation.

        Args:
            file_path: Path to the file to parse

        Returns:
            ParsedDocument containing blocks, metadata, and assets

        Raises:
            ParseError: If the file cannot be parsed (corrupted, password-protected, etc.)
        """
        ...


class ParseError(Exception):
    """Raised when a parser cannot process a file."""

    def __init__(self, message: str, reason: str = "unknown"):
        """Initialize ParseError.

        Args:
            message: Human-readable error description
            reason: Machine-readable reason code:
                - "corrupted": File is corrupted or malformed
                - "password_protected": File requires a password
                - "unsupported_version": File format version not supported
                - "empty": File contains no extractable content
                - "unknown": Unknown error
        """
        super().__init__(message)
        self.reason = reason
