"""Markdown and Plain Text parser plugin.

Handles .md, .txt, and other plain text files.
Markdown files are parsed with heading detection; plain text uses
paragraph-based splitting.
"""

import logging
import os
import re

from app.services.parsers.base import Block, ParsedDocument, ParseError

logger = logging.getLogger(__name__)


class TextParser:
    """Parser for Markdown and Plain Text files.

    Handles:
    - Markdown (.md): Parses headings, code blocks, tables, lists
    - Plain Text (.txt): Splits by paragraphs (double newlines)
    """

    name: str = "text-parser"
    supported_extensions: list[str] = ["md", "txt", "markdown", "text"]
    priority: int = 50

    def can_parse(self, file_path: str, mime_type: str) -> bool:
        """Check if this parser can handle the file."""
        ext = os.path.splitext(file_path)[1].lower().lstrip(".")
        return ext in self.supported_extensions or mime_type in (
            "text/plain",
            "text/markdown",
            "text/x-markdown",
        )

    async def parse(self, file_path: str) -> ParsedDocument:
        """Parse a text or markdown file.

        Args:
            file_path: Path to the text file

        Returns:
            ParsedDocument with extracted content blocks

        Raises:
            ParseError: If the file cannot be read
        """
        if not os.path.exists(file_path):
            raise ParseError(f"File not found: {file_path}", reason="corrupted")

        try:
            # Try UTF-8 first, then fall back to other encodings
            content = self._read_file(file_path)
        except Exception as e:
            raise ParseError(
                f"Cannot read text file: {file_path}: {e}",
                reason="corrupted",
            )

        if not content.strip():
            raise ParseError(
                f"Text file is empty: {file_path}",
                reason="empty",
            )

        ext = os.path.splitext(file_path)[1].lower().lstrip(".")
        if ext in ("md", "markdown"):
            blocks = self._parse_markdown(content)
        else:
            blocks = self._parse_plain_text(content)

        metadata: dict = {
            "source": file_path,
            "page_count": 1,
            "char_count": len(content),
        }

        return ParsedDocument(blocks=blocks, metadata=metadata, assets=[])

    def _read_file(self, file_path: str) -> str:
        """Read file content with encoding detection."""
        encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin-1"]
        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()
            except (UnicodeDecodeError, UnicodeError):
                continue
        # Last resort: read with errors replaced
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    def _parse_markdown(self, content: str) -> list[Block]:
        """Parse Markdown content into blocks."""
        blocks: list[Block] = []
        lines = content.split("\n")
        current_text = ""
        current_type = "paragraph"
        current_style: dict = {}
        in_code_block = False
        code_block_text = ""

        for line in lines:
            # Handle fenced code blocks
            if line.strip().startswith("```"):
                if in_code_block:
                    # End of code block
                    in_code_block = False
                    blocks.append(Block(
                        type="paragraph",
                        text=code_block_text.strip(),
                        page_number=1,
                        style={"code_block": True},
                    ))
                    code_block_text = ""
                else:
                    # Start of code block - flush current text
                    if current_text.strip():
                        blocks.append(Block(
                            type=current_type,
                            text=current_text.strip(),
                            page_number=1,
                            style=current_style,
                        ))
                        current_text = ""
                        current_type = "paragraph"
                        current_style = {}
                    in_code_block = True
                continue

            if in_code_block:
                code_block_text += line + "\n"
                continue

            stripped = line.strip()

            # Empty line = paragraph break
            if not stripped:
                if current_text.strip():
                    blocks.append(Block(
                        type=current_type,
                        text=current_text.strip(),
                        page_number=1,
                        style=current_style,
                    ))
                    current_text = ""
                    current_type = "paragraph"
                    current_style = {}
                continue

            # Headings (ATX style: # Heading)
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
            if heading_match:
                if current_text.strip():
                    blocks.append(Block(
                        type=current_type,
                        text=current_text.strip(),
                        page_number=1,
                        style=current_style,
                    ))
                    current_text = ""
                    current_type = "paragraph"
                    current_style = {}

                level = len(heading_match.group(1))
                heading_text = heading_match.group(2).strip()
                blocks.append(Block(
                    type="heading",
                    text=heading_text,
                    page_number=1,
                    style={"heading_level": level},
                ))
                continue

            # Table rows (lines with pipes)
            if stripped.startswith("|") and stripped.endswith("|"):
                # Check if it's a separator row
                if re.match(r"^\|[\s\-:|]+\|$", stripped):
                    current_text += line + "\n"
                    current_type = "table"
                    continue
                if current_type != "table" and current_text.strip():
                    blocks.append(Block(
                        type=current_type,
                        text=current_text.strip(),
                        page_number=1,
                        style=current_style,
                    ))
                    current_text = ""
                    current_style = {}
                current_type = "table"
                current_text += line + "\n"
                continue

            # List items
            if re.match(r"^[\-\*\+]\s+", stripped) or re.match(r"^\d+\.\s+", stripped):
                if current_type != "list" and current_text.strip():
                    blocks.append(Block(
                        type=current_type,
                        text=current_text.strip(),
                        page_number=1,
                        style=current_style,
                    ))
                    current_text = ""
                    current_style = {}
                current_type = "list"
                current_text += line + "\n"
                continue

            # Regular paragraph text
            if current_type in ("table", "list"):
                # End of table/list
                if current_text.strip():
                    blocks.append(Block(
                        type=current_type,
                        text=current_text.strip(),
                        page_number=1,
                        style=current_style,
                    ))
                    current_text = ""
                    current_style = {}
                current_type = "paragraph"

            current_text += line + "\n"

        # Flush remaining content
        if in_code_block and code_block_text.strip():
            blocks.append(Block(
                type="paragraph",
                text=code_block_text.strip(),
                page_number=1,
                style={"code_block": True},
            ))
        elif current_text.strip():
            blocks.append(Block(
                type=current_type,
                text=current_text.strip(),
                page_number=1,
                style=current_style,
            ))

        return blocks

    def _parse_plain_text(self, content: str) -> list[Block]:
        """Parse plain text content into paragraph blocks."""
        blocks: list[Block] = []

        # Split by double newlines (paragraphs)
        paragraphs = re.split(r"\n\s*\n", content)

        for para in paragraphs:
            text = para.strip()
            if not text:
                continue

            blocks.append(Block(
                type="paragraph",
                text=text,
                page_number=1,
                style={},
            ))

        return blocks
