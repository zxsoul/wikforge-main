"""Parser plugin system for document format conversion.

This package provides:
- ParserPlugin protocol (interface for all parsers)
- ParsedDocument, Block, Asset data structures (intermediate representation)
- ParserRegistry (plugin registration, selection, hot-reload)
- Built-in parsers: PDF, DOCX, PPTX, HTML, Markdown, Plain Text
"""

from app.services.parsers.base import Asset, Block, ParsedDocument, ParserPlugin
from app.services.parsers.registry import ParserRegistry

__all__ = [
    "Asset",
    "Block",
    "ParsedDocument",
    "ParserPlugin",
    "ParserRegistry",
]
