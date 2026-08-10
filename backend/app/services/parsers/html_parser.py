"""HTML parser plugin using trafilatura for content extraction.

Extracts main body content from HTML files, removing navigation,
ads, and other boilerplate elements.
"""

import logging
import os

from app.services.parsers.base import Block, ParsedDocument, ParseError

logger = logging.getLogger(__name__)


class HtmlParser:
    """HTML parser using trafilatura for main content extraction.

    Trafilatura excels at extracting the main textual content from
    web pages while removing boilerplate (navigation, ads, footers).
    """

    name: str = "html-parser"
    supported_extensions: list[str] = ["html", "htm"]
    priority: int = 100

    def can_parse(self, file_path: str, mime_type: str) -> bool:
        """Check if this parser can handle the file."""
        ext = os.path.splitext(file_path)[1].lower().lstrip(".")
        return ext in self.supported_extensions or mime_type in (
            "text/html",
            "application/xhtml+xml",
        )

    async def parse(self, file_path: str) -> ParsedDocument:
        """Parse an HTML file using trafilatura.

        Args:
            file_path: Path to the HTML file

        Returns:
            ParsedDocument with extracted content blocks

        Raises:
            ParseError: If the file cannot be parsed
        """
        if not os.path.exists(file_path):
            raise ParseError(f"File not found: {file_path}", reason="corrupted")

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                html_content = f.read()
        except Exception as e:
            raise ParseError(
                f"Cannot read HTML file: {file_path}: {e}",
                reason="corrupted",
            )

        if not html_content.strip():
            raise ParseError(
                f"HTML file is empty: {file_path}",
                reason="empty",
            )

        try:
            import trafilatura
        except ImportError:
            raise ParseError(
                "trafilatura is not installed",
                reason="unknown",
            )

        # Extract main content using trafilatura
        extracted = trafilatura.extract(
            html_content,
            include_tables=True,
            include_links=True,
            include_images=True,
            output_format="txt",
        )

        if not extracted or not extracted.strip():
            # Fallback: try to extract any text content
            extracted = trafilatura.extract(
                html_content,
                include_tables=True,
                no_fallback=False,
            )

        if not extracted or not extracted.strip():
            raise ParseError(
                f"No content extracted from HTML: {file_path}",
                reason="empty",
            )

        # Parse extracted text into blocks
        blocks = self._text_to_blocks(extracted)

        metadata: dict = {
            "source": file_path,
            "page_count": 1,
        }

        # Try to extract title from HTML
        title = self._extract_title(html_content)
        if title:
            metadata["title"] = title

        return ParsedDocument(blocks=blocks, metadata=metadata, assets=[])

    def _text_to_blocks(self, text: str) -> list[Block]:
        """Convert extracted text into structured blocks."""
        blocks: list[Block] = []
        paragraphs = text.split("\n\n")

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Simple heuristic: lines that are short and don't end with
            # punctuation might be headings
            lines = para.split("\n")
            if len(lines) == 1 and len(para) < 100 and not para.endswith((".", "。", "!", "！", "?", "？")):
                blocks.append(Block(
                    type="heading",
                    text=para,
                    page_number=1,
                    style={"heading_level": 2},
                ))
            elif "|" in para and para.count("|") >= 3:
                # Likely a table
                blocks.append(Block(
                    type="table",
                    text=para,
                    page_number=1,
                    style={},
                ))
            else:
                blocks.append(Block(
                    type="paragraph",
                    text=para,
                    page_number=1,
                    style={},
                ))

        return blocks

    def _extract_title(self, html_content: str) -> str | None:
        """Extract title from HTML <title> tag."""
        import re

        match = re.search(r"<title[^>]*>(.*?)</title>", html_content, re.IGNORECASE | re.DOTALL)
        if match:
            title = match.group(1).strip()
            # Remove HTML entities
            title = title.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
            return title if title else None
        return None
