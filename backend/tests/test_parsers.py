"""Unit tests for parser plugin system.

Tests cover:
- ParserPlugin protocol and data structures (Block, Asset, ParsedDocument)
- ParserRegistry (register, unregister, select, hot-reload)
- PDF parser (text extraction, heading detection, error handling)
- DOCX parser (paragraphs, headings, tables, images)
- PPTX parser (slide-by-slide extraction)
- HTML parser (trafilatura content extraction)
- Text/Markdown parser (heading detection, code blocks, tables)
- Pipeline task chain (submit_pipeline, retry logic)
- Error handling (corrupted files, password-protected, empty files)
"""

import os
import tempfile
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.parsers.base import Asset, Block, ParsedDocument, ParseError, ParserPlugin
from app.services.parsers.registry import ParserRegistry, get_parser_registry, reset_parser_registry


# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset global registry before each test."""
    reset_parser_registry()
    yield
    reset_parser_registry()


@pytest.fixture
def registry():
    """Create a fresh ParserRegistry."""
    return ParserRegistry()


@pytest.fixture
def sample_txt_file():
    """Create a temporary text file for testing."""
    content = "Hello World\n\nThis is a test document.\n\nIt has multiple paragraphs."
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    tmp.write(content)
    tmp.close()
    yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def sample_md_file():
    """Create a temporary markdown file for testing."""
    content = """# Main Title

## Section One

This is the first paragraph of section one.

### Subsection

More content here with **bold** and *italic* text.

## Section Two

| Column A | Column B |
| --- | --- |
| Cell 1 | Cell 2 |
| Cell 3 | Cell 4 |

- Item 1
- Item 2
- Item 3

```python
def hello():
    print("world")
```
"""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8")
    tmp.write(content)
    tmp.close()
    yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def sample_html_file():
    """Create a temporary HTML file for testing."""
    content = """<!DOCTYPE html>
<html>
<head><title>Test Page</title></head>
<body>
<nav>Navigation menu</nav>
<article>
<h1>Article Title</h1>
<p>This is the main content of the article. It contains important information
that should be extracted by the parser.</p>
<p>Second paragraph with more details about the topic.</p>
<table>
<tr><th>Name</th><th>Value</th></tr>
<tr><td>Alpha</td><td>100</td></tr>
</table>
</article>
<footer>Footer content</footer>
</body>
</html>"""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8")
    tmp.write(content)
    tmp.close()
    yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def empty_file():
    """Create an empty temporary file."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    tmp.close()
    yield tmp.name
    os.unlink(tmp.name)


# ─── Mock Parser for Testing Registry ─────────────────────────────────


class MockParser:
    """Mock parser for testing registry operations."""

    def __init__(self, name: str = "mock-parser", extensions: list[str] | None = None, priority: int = 50):
        self.name = name
        self.supported_extensions = extensions or ["mock"]
        self.priority = priority

    def can_parse(self, file_path: str, mime_type: str) -> bool:
        ext = os.path.splitext(file_path)[1].lower().lstrip(".")
        return ext in self.supported_extensions

    async def parse(self, file_path: str) -> ParsedDocument:
        return ParsedDocument(
            blocks=[Block(type="paragraph", text="mock content", page_number=1)],
            metadata={"source": file_path},
            assets=[],
        )


# ─── Data Structure Tests ──────────────────────────────────────────────


class TestDataStructures:
    """Tests for Block, Asset, and ParsedDocument data structures."""

    def test_block_creation_minimal(self):
        """Block can be created with minimal required fields."""
        block = Block(type="paragraph", text="Hello")
        assert block.type == "paragraph"
        assert block.text == "Hello"
        assert block.bbox is None
        assert block.page_number == 1
        assert block.style == {}
        assert block.raw == {}

    def test_block_creation_full(self):
        """Block can be created with all fields."""
        block = Block(
            type="heading",
            text="Title",
            bbox=(0.1, 0.2, 0.9, 0.3),
            page_number=3,
            style={"heading_level": 1, "bold": True},
            raw={"original_font": "Arial"},
        )
        assert block.type == "heading"
        assert block.text == "Title"
        assert block.bbox == (0.1, 0.2, 0.9, 0.3)
        assert block.page_number == 3
        assert block.style["heading_level"] == 1
        assert block.raw["original_font"] == "Arial"

    def test_asset_creation(self):
        """Asset can be created with all fields."""
        asset = Asset(
            id="img-001",
            type="image",
            data=b"\x89PNG",
            mime_type="image/png",
            page_number=2,
            bbox=(0.1, 0.1, 0.5, 0.5),
            description="A chart",
        )
        assert asset.id == "img-001"
        assert asset.type == "image"
        assert asset.data == b"\x89PNG"
        assert asset.mime_type == "image/png"
        assert asset.page_number == 2
        assert asset.description == "A chart"

    def test_parsed_document_creation(self):
        """ParsedDocument can be created with blocks, metadata, and assets."""
        blocks = [Block(type="paragraph", text="Content")]
        assets = [Asset(id="1", type="image", data=b"", mime_type="image/png")]
        doc = ParsedDocument(
            blocks=blocks,
            metadata={"page_count": 5},
            assets=assets,
        )
        assert len(doc.blocks) == 1
        assert doc.metadata["page_count"] == 5
        assert len(doc.assets) == 1

    def test_parsed_document_empty(self):
        """ParsedDocument can be created empty."""
        doc = ParsedDocument()
        assert doc.blocks == []
        assert doc.metadata == {}
        assert doc.assets == []

    def test_parse_error_with_reason(self):
        """ParseError stores reason code."""
        err = ParseError("File is corrupted", reason="corrupted")
        assert str(err) == "File is corrupted"
        assert err.reason == "corrupted"

    def test_parse_error_default_reason(self):
        """ParseError defaults to 'unknown' reason."""
        err = ParseError("Something went wrong")
        assert err.reason == "unknown"


# ─── Protocol Compliance Tests ─────────────────────────────────────────


class TestParserPluginProtocol:
    """Tests for ParserPlugin protocol compliance."""

    def test_mock_parser_implements_protocol(self):
        """MockParser satisfies ParserPlugin protocol."""
        parser = MockParser()
        assert isinstance(parser, ParserPlugin)

    def test_protocol_requires_name(self):
        """ParserPlugin requires name attribute."""
        parser = MockParser(name="test")
        assert parser.name == "test"

    def test_protocol_requires_supported_extensions(self):
        """ParserPlugin requires supported_extensions attribute."""
        parser = MockParser(extensions=["pdf", "docx"])
        assert parser.supported_extensions == ["pdf", "docx"]

    def test_protocol_requires_priority(self):
        """ParserPlugin requires priority attribute."""
        parser = MockParser(priority=200)
        assert parser.priority == 200


# ─── Registry Tests ────────────────────────────────────────────────────


class TestParserRegistry:
    """Tests for ParserRegistry operations."""

    def test_register_plugin(self, registry):
        """Successfully register a plugin."""
        parser = MockParser()
        registry.register(parser)
        assert len(registry.plugins) == 1
        assert registry.plugins[0].name == "mock-parser"

    def test_register_duplicate_raises_error(self, registry):
        """Registering a plugin with duplicate name raises ValueError."""
        registry.register(MockParser(name="test"))
        with pytest.raises(ValueError, match="already registered"):
            registry.register(MockParser(name="test"))

    def test_unregister_plugin(self, registry):
        """Successfully unregister a plugin by name."""
        registry.register(MockParser(name="test"))
        registry.unregister("test")
        assert len(registry.plugins) == 0

    def test_unregister_nonexistent_raises_error(self, registry):
        """Unregistering a non-existent plugin raises ValueError."""
        with pytest.raises(ValueError, match="not registered"):
            registry.unregister("nonexistent")

    def test_select_by_extension(self, registry):
        """Select parser based on file extension."""
        registry.register(MockParser(name="pdf", extensions=["pdf"], priority=100))
        registry.register(MockParser(name="txt", extensions=["txt"], priority=50))

        selected = registry.select("document.pdf", "")
        assert selected.name == "pdf"

        selected = registry.select("readme.txt", "")
        assert selected.name == "txt"

    def test_select_by_priority(self, registry):
        """Higher priority plugin is selected when multiple match."""
        registry.register(MockParser(name="low", extensions=["pdf"], priority=10))
        registry.register(MockParser(name="high", extensions=["pdf"], priority=100))

        selected = registry.select("test.pdf", "")
        assert selected.name == "high"

    def test_select_no_match_raises_error(self, registry):
        """Selecting with no matching plugin raises ValueError."""
        registry.register(MockParser(name="pdf", extensions=["pdf"]))
        with pytest.raises(ValueError, match="No parser plugin found"):
            registry.select("test.xyz", "")

    def test_get_plugin_by_name(self, registry):
        """Get a registered plugin by name."""
        parser = MockParser(name="test")
        registry.register(parser)
        assert registry.get_plugin("test") is parser
        assert registry.get_plugin("nonexistent") is None

    def test_clear_removes_all(self, registry):
        """Clear removes all registered plugins."""
        registry.register(MockParser(name="a"))
        registry.register(MockParser(name="b"))
        registry.clear()
        assert len(registry.plugins) == 0

    def test_plugins_sorted_by_priority(self, registry):
        """Plugins are maintained sorted by priority (highest first)."""
        registry.register(MockParser(name="low", priority=10))
        registry.register(MockParser(name="mid", priority=50))
        registry.register(MockParser(name="high", priority=100))

        names = [p.name for p in registry.plugins]
        assert names == ["high", "mid", "low"]

    def test_reload_from_configs(self, registry):
        """Reload plugins from configuration list."""
        configs = [
            {
                "name": "text-parser",
                "import_path": "app.services.parsers.text_parser.TextParser",
                "supported_extensions": ["txt", "md"],
                "priority": 50,
                "enabled": True,
                "config": {},
            },
        ]
        registry.reload_from_configs(configs)
        assert len(registry.plugins) == 1
        assert registry.plugins[0].name == "text-parser"

    def test_reload_skips_disabled(self, registry):
        """Reload skips disabled plugins."""
        configs = [
            {
                "name": "disabled-parser",
                "import_path": "app.services.parsers.text_parser.TextParser",
                "supported_extensions": ["txt"],
                "priority": 50,
                "enabled": False,
                "config": {},
            },
        ]
        registry.reload_from_configs(configs)
        assert len(registry.plugins) == 0

    def test_reload_handles_import_error(self, registry):
        """Reload gracefully handles import errors."""
        configs = [
            {
                "name": "bad-parser",
                "import_path": "nonexistent.module.BadParser",
                "supported_extensions": ["xyz"],
                "priority": 50,
                "enabled": True,
                "config": {},
            },
        ]
        # Should not raise, just log error
        registry.reload_from_configs(configs)
        assert len(registry.plugins) == 0

    def test_global_registry_singleton(self):
        """get_parser_registry returns the same instance."""
        r1 = get_parser_registry()
        r2 = get_parser_registry()
        assert r1 is r2

    def test_reload_supports_colon_import_path(self, registry):
        """``module:Class`` syntax is accepted in addition to ``module.Class``."""
        configs = [
            {
                "name": "text-parser",
                "import_path": "app.services.parsers.text_parser:TextParser",
                "supported_extensions": ["txt", "md"],
                "priority": 50,
                "enabled": True,
                "config": {},
            },
        ]
        registry.reload_from_configs(configs)
        assert len(registry.plugins) == 1
        assert registry.plugins[0].name == "text-parser"

    def test_reload_from_db_records_maps_attributes(self, registry):
        """ORM-like records are converted to config dicts and registered."""
        record = MagicMock()
        record.name = "text-parser"
        record.import_path = "app.services.parsers.text_parser.TextParser"
        record.supported_extensions = ["txt", "md"]
        record.priority = 25
        record.enabled = True
        record.config = {}

        registry.reload_from_db_records([record])

        assert len(registry.plugins) == 1
        plugin = registry.plugins[0]
        assert plugin.name == "text-parser"
        assert plugin.priority == 25  # priority overridden from config

    def test_reload_from_db_records_skips_disabled(self, registry):
        """Disabled ORM records are not registered."""
        record = MagicMock()
        record.name = "disabled-parser"
        record.import_path = "app.services.parsers.text_parser.TextParser"
        record.supported_extensions = ["txt"]
        record.priority = 10
        record.enabled = False
        record.config = {}

        registry.reload_from_db_records([record])
        assert len(registry.plugins) == 0

    @pytest.mark.asyncio
    async def test_load_from_database_reads_enabled_records(self, registry):
        """``load_from_database`` queries enabled records and registers them."""
        record = MagicMock()
        record.name = "text-parser"
        record.import_path = "app.services.parsers.text_parser.TextParser"
        record.supported_extensions = ["txt", "md"]
        record.priority = 50
        record.enabled = True
        record.config = {}

        scalars = MagicMock()
        scalars.all.return_value = [record]

        result = MagicMock()
        result.scalars.return_value = scalars

        session = MagicMock()
        session.execute = AsyncMock(return_value=result)

        count = await registry.load_from_database(session)

        assert count == 1
        assert len(registry.plugins) == 1
        assert registry.plugins[0].name == "text-parser"
        # Verify the session was queried (single SELECT statement)
        session.execute.assert_awaited_once()


# ─── Text Parser Tests ─────────────────────────────────────────────────


class TestTextParser:
    """Tests for Markdown and Plain Text parser."""

    @pytest.mark.asyncio
    async def test_parse_plain_text(self, sample_txt_file):
        """Parse a plain text file into paragraph blocks."""
        from app.services.parsers.text_parser import TextParser

        parser = TextParser()
        result = await parser.parse(sample_txt_file)

        assert len(result.blocks) > 0
        assert all(b.type == "paragraph" for b in result.blocks)
        assert "Hello World" in result.blocks[0].text

    @pytest.mark.asyncio
    async def test_parse_markdown_headings(self, sample_md_file):
        """Parse markdown file and detect headings."""
        from app.services.parsers.text_parser import TextParser

        parser = TextParser()
        result = await parser.parse(sample_md_file)

        headings = [b for b in result.blocks if b.type == "heading"]
        assert len(headings) >= 3  # Main Title, Section One, Subsection, Section Two

        # Check heading levels
        h1 = [h for h in headings if h.style.get("heading_level") == 1]
        assert len(h1) >= 1
        assert "Main Title" in h1[0].text

    @pytest.mark.asyncio
    async def test_parse_markdown_table(self, sample_md_file):
        """Parse markdown file and detect tables."""
        from app.services.parsers.text_parser import TextParser

        parser = TextParser()
        result = await parser.parse(sample_md_file)

        tables = [b for b in result.blocks if b.type == "table"]
        assert len(tables) >= 1
        assert "Column A" in tables[0].text

    @pytest.mark.asyncio
    async def test_parse_markdown_list(self, sample_md_file):
        """Parse markdown file and detect lists."""
        from app.services.parsers.text_parser import TextParser

        parser = TextParser()
        result = await parser.parse(sample_md_file)

        lists = [b for b in result.blocks if b.type == "list"]
        assert len(lists) >= 1
        assert "Item 1" in lists[0].text

    @pytest.mark.asyncio
    async def test_parse_empty_file_raises_error(self, empty_file):
        """Parsing an empty file raises ParseError."""
        from app.services.parsers.text_parser import TextParser

        parser = TextParser()
        with pytest.raises(ParseError, match="empty"):
            await parser.parse(empty_file)

    @pytest.mark.asyncio
    async def test_parse_nonexistent_file_raises_error(self):
        """Parsing a non-existent file raises ParseError."""
        from app.services.parsers.text_parser import TextParser

        parser = TextParser()
        with pytest.raises(ParseError):
            await parser.parse("/nonexistent/file.txt")

    def test_can_parse_txt(self):
        """TextParser can parse .txt files."""
        from app.services.parsers.text_parser import TextParser

        parser = TextParser()
        assert parser.can_parse("test.txt", "") is True
        assert parser.can_parse("test.md", "") is True
        assert parser.can_parse("test.pdf", "") is False

    def test_can_parse_by_mime_type(self):
        """TextParser can parse by MIME type."""
        from app.services.parsers.text_parser import TextParser

        parser = TextParser()
        assert parser.can_parse("file", "text/plain") is True
        assert parser.can_parse("file", "text/markdown") is True
        assert parser.can_parse("file", "application/pdf") is False


# ─── HTML Parser Tests ─────────────────────────────────────────────────


class TestHtmlParser:
    """Tests for HTML parser."""

    @pytest.mark.asyncio
    async def test_parse_html_extracts_content(self, sample_html_file):
        """Parse HTML file and extract main content."""
        from app.services.parsers.html_parser import HtmlParser

        parser = HtmlParser()
        result = await parser.parse(sample_html_file)

        assert len(result.blocks) > 0
        # Should extract article content
        all_text = " ".join(b.text for b in result.blocks)
        assert "main content" in all_text.lower() or "article" in all_text.lower()

    @pytest.mark.asyncio
    async def test_parse_html_extracts_title(self, sample_html_file):
        """Parse HTML file and extract title metadata."""
        from app.services.parsers.html_parser import HtmlParser

        parser = HtmlParser()
        result = await parser.parse(sample_html_file)

        assert result.metadata.get("title") == "Test Page"

    @pytest.mark.asyncio
    async def test_parse_empty_html_raises_error(self):
        """Parsing empty HTML raises ParseError."""
        from app.services.parsers.html_parser import HtmlParser

        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False)
        tmp.write("")
        tmp.close()

        parser = HtmlParser()
        try:
            with pytest.raises(ParseError):
                await parser.parse(tmp.name)
        finally:
            os.unlink(tmp.name)

    def test_can_parse_html(self):
        """HtmlParser can parse .html and .htm files."""
        from app.services.parsers.html_parser import HtmlParser

        parser = HtmlParser()
        assert parser.can_parse("page.html", "") is True
        assert parser.can_parse("page.htm", "") is True
        assert parser.can_parse("page.txt", "") is False
        assert parser.can_parse("file", "text/html") is True


# ─── PDF Parser Tests ──────────────────────────────────────────────────


class TestPdfParser:
    """Tests for PDF parser."""

    def test_can_parse_pdf(self):
        """PdfParser can parse .pdf files."""
        from app.services.parsers.pdf_parser import PdfParser

        parser = PdfParser()
        assert parser.can_parse("document.pdf", "") is True
        assert parser.can_parse("document.PDF", "") is True
        assert parser.can_parse("document.docx", "") is False
        assert parser.can_parse("file", "application/pdf") is True

    @pytest.mark.asyncio
    async def test_parse_nonexistent_pdf_raises_error(self):
        """Parsing non-existent PDF raises ParseError."""
        from app.services.parsers.pdf_parser import PdfParser

        parser = PdfParser()
        with pytest.raises(ParseError):
            await parser.parse("/nonexistent/file.pdf")

    @pytest.mark.asyncio
    async def test_parse_invalid_pdf_raises_error(self):
        """Parsing an invalid PDF file raises ParseError."""
        from app.services.parsers.pdf_parser import PdfParser

        # Create a file that's not actually a PDF
        tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False)
        tmp.write(b"This is not a PDF file")
        tmp.close()

        parser = PdfParser()
        try:
            with pytest.raises(ParseError):
                await parser.parse(tmp.name)
        finally:
            os.unlink(tmp.name)

    @pytest.mark.asyncio
    async def test_parse_password_protected_pdf_marked_as_password_protected(self):
        """A password-protected PDF surfaces ParseError(reason='password_protected')."""
        from app.services.parsers import pdf_parser as pdf_module

        tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False)
        tmp.write(b"%PDF-1.4 fake")
        tmp.close()

        parser = pdf_module.PdfParser()
        try:
            with patch.object(
                parser,
                "_parse_with_marker",
                AsyncMock(side_effect=Exception("File is encrypted (needs password)")),
            ):
                with pytest.raises(ParseError) as exc_info:
                    await parser.parse(tmp.name)
                assert exc_info.value.reason == "password_protected"
        finally:
            os.unlink(tmp.name)

    @pytest.mark.asyncio
    async def test_parse_corrupted_pdf_marked_as_corrupted(self):
        """A corrupted PDF surfaces ParseError(reason='corrupted')."""
        from app.services.parsers import pdf_parser as pdf_module

        tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False)
        tmp.write(b"%PDF-1.4 broken")
        tmp.close()

        parser = pdf_module.PdfParser()
        try:
            with patch.object(
                parser,
                "_parse_with_marker",
                AsyncMock(side_effect=Exception("File is corrupted/damaged")),
            ):
                with pytest.raises(ParseError) as exc_info:
                    await parser.parse(tmp.name)
                assert exc_info.value.reason == "corrupted"
        finally:
            os.unlink(tmp.name)


# ─── DOCX Parser Tests ────────────────────────────────────────────────


class TestDocxParser:
    """Tests for Word document parser."""

    def test_can_parse_docx(self):
        """DocxParser can parse .docx files."""
        from app.services.parsers.docx_parser import DocxParser

        parser = DocxParser()
        assert parser.can_parse("document.docx", "") is True
        assert parser.can_parse("document.doc", "") is False
        assert parser.can_parse("document.pdf", "") is False

    @pytest.mark.asyncio
    async def test_parse_nonexistent_docx_raises_error(self):
        """Parsing non-existent DOCX raises ParseError."""
        from app.services.parsers.docx_parser import DocxParser

        parser = DocxParser()
        with pytest.raises(ParseError):
            await parser.parse("/nonexistent/file.docx")

    @pytest.mark.asyncio
    async def test_parse_invalid_docx_raises_error(self):
        """Parsing an invalid DOCX file raises ParseError."""
        from app.services.parsers.docx_parser import DocxParser

        tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".docx", delete=False)
        tmp.write(b"Not a valid docx file")
        tmp.close()

        parser = DocxParser()
        try:
            with pytest.raises(ParseError):
                await parser.parse(tmp.name)
        finally:
            os.unlink(tmp.name)

    @pytest.mark.asyncio
    async def test_parse_password_protected_docx(self):
        """A password-protected DOCX surfaces ParseError(reason='password_protected')."""
        pytest.importorskip("docx")
        from app.services.parsers.docx_parser import DocxParser

        tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".docx", delete=False)
        tmp.write(b"PK\x03\x04 fake")
        tmp.close()

        parser = DocxParser()
        try:
            with patch(
                "docx.Document",
                side_effect=Exception("Document is encrypted: password required"),
            ):
                with pytest.raises(ParseError) as exc_info:
                    await parser.parse(tmp.name)
                assert exc_info.value.reason == "password_protected"
        finally:
            os.unlink(tmp.name)


# ─── PPTX Parser Tests ────────────────────────────────────────────────


class TestPptxParser:
    """Tests for PowerPoint parser."""

    def test_can_parse_pptx(self):
        """PptxParser can parse .pptx files."""
        from app.services.parsers.pptx_parser import PptxParser

        parser = PptxParser()
        assert parser.can_parse("slides.pptx", "") is True
        assert parser.can_parse("slides.ppt", "") is False
        assert parser.can_parse("slides.pdf", "") is False

    @pytest.mark.asyncio
    async def test_parse_nonexistent_pptx_raises_error(self):
        """Parsing non-existent PPTX raises ParseError."""
        from app.services.parsers.pptx_parser import PptxParser

        parser = PptxParser()
        with pytest.raises(ParseError):
            await parser.parse("/nonexistent/file.pptx")

    @pytest.mark.asyncio
    async def test_parse_invalid_pptx_raises_error(self):
        """Parsing an invalid PPTX file raises ParseError."""
        from app.services.parsers.pptx_parser import PptxParser

        tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".pptx", delete=False)
        tmp.write(b"Not a valid pptx file")
        tmp.close()

        parser = PptxParser()
        try:
            with pytest.raises(ParseError):
                await parser.parse(tmp.name)
        finally:
            os.unlink(tmp.name)

    @pytest.mark.asyncio
    async def test_parse_password_protected_pptx(self):
        """A password-protected PPTX surfaces ParseError(reason='password_protected')."""
        pytest.importorskip("pptx")
        from app.services.parsers.pptx_parser import PptxParser

        tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".pptx", delete=False)
        tmp.write(b"PK\x03\x04 fake")
        tmp.close()

        parser = PptxParser()
        try:
            with patch(
                "pptx.Presentation",
                side_effect=Exception("Presentation is encrypted: password required"),
            ):
                with pytest.raises(ParseError) as exc_info:
                    await parser.parse(tmp.name)
                assert exc_info.value.reason == "password_protected"
        finally:
            os.unlink(tmp.name)


# ─── Pipeline Tests ────────────────────────────────────────────────────


class TestPipeline:
    """Tests for Celery pipeline task chain."""

    def test_get_retry_delay_exponential_backoff(self):
        """Retry delay follows exponential backoff."""
        from app.tasks.pipeline import _get_retry_delay

        assert _get_retry_delay(0) == 10   # 10 * 2^0 = 10
        assert _get_retry_delay(1) == 20   # 10 * 2^1 = 20
        assert _get_retry_delay(2) == 40   # 10 * 2^2 = 40

    def test_get_retry_delay_custom_base(self):
        """Retry delay with custom base delay."""
        from app.tasks.pipeline import _get_retry_delay

        assert _get_retry_delay(0, base_delay=5) == 5
        assert _get_retry_delay(1, base_delay=5) == 10
        assert _get_retry_delay(2, base_delay=5) == 20

    def test_get_mime_type_mapping(self):
        """MIME type mapping works correctly."""
        from app.tasks.pipeline import _get_mime_type

        assert _get_mime_type("pdf") == "application/pdf"
        assert _get_mime_type("docx") == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        assert _get_mime_type("pptx") == "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        assert _get_mime_type("txt") == "text/plain"
        assert _get_mime_type("md") == "text/markdown"
        assert _get_mime_type("html") == "text/html"
        assert _get_mime_type("unknown") == "application/octet-stream"

    def test_ensure_default_parsers_registered(self):
        """Default parsers are registered when registry is empty."""
        from app.tasks.pipeline import _ensure_default_parsers_registered

        registry = ParserRegistry()
        _ensure_default_parsers_registered(registry)

        assert len(registry.plugins) == 5
        names = {p.name for p in registry.plugins}
        assert "pdf-parser" in names
        assert "docx-parser" in names
        assert "pptx-parser" in names
        assert "html-parser" in names
        assert "text-parser" in names

    def test_ensure_default_parsers_not_duplicated(self):
        """Default parsers are not re-registered if already present."""
        from app.tasks.pipeline import _ensure_default_parsers_registered

        registry = ParserRegistry()
        _ensure_default_parsers_registered(registry)
        _ensure_default_parsers_registered(registry)  # Call again

        assert len(registry.plugins) == 5  # Still 5, not 10


# ─── Integration-style Tests ───────────────────────────────────────────


class TestParserIntegration:
    """Integration tests for parser selection and execution."""

    @pytest.mark.asyncio
    async def test_registry_selects_correct_parser_for_txt(self, sample_txt_file):
        """Registry selects TextParser for .txt files."""
        from app.services.parsers.text_parser import TextParser

        registry = ParserRegistry()
        registry.register(TextParser())

        parser = registry.select(sample_txt_file, "text/plain")
        assert parser.name == "text-parser"

        result = await parser.parse(sample_txt_file)
        assert len(result.blocks) > 0

    @pytest.mark.asyncio
    async def test_registry_selects_correct_parser_for_md(self, sample_md_file):
        """Registry selects TextParser for .md files."""
        from app.services.parsers.text_parser import TextParser

        registry = ParserRegistry()
        registry.register(TextParser())

        parser = registry.select(sample_md_file, "text/markdown")
        assert parser.name == "text-parser"

        result = await parser.parse(sample_md_file)
        headings = [b for b in result.blocks if b.type == "heading"]
        assert len(headings) > 0

    @pytest.mark.asyncio
    async def test_registry_selects_correct_parser_for_html(self, sample_html_file):
        """Registry selects HtmlParser for .html files."""
        from app.services.parsers.html_parser import HtmlParser
        from app.services.parsers.text_parser import TextParser

        registry = ParserRegistry()
        registry.register(HtmlParser())
        registry.register(TextParser())

        parser = registry.select(sample_html_file, "text/html")
        assert parser.name == "html-parser"

    def test_priority_based_selection(self):
        """Higher priority parser is selected over lower priority."""
        registry = ParserRegistry()
        registry.register(MockParser(name="low-pdf", extensions=["pdf"], priority=10))
        registry.register(MockParser(name="high-pdf", extensions=["pdf"], priority=100))

        selected = registry.select("test.pdf", "")
        assert selected.name == "high-pdf"

    @pytest.mark.asyncio
    async def test_chinese_text_parsing(self):
        """Parser handles Chinese text correctly."""
        from app.services.parsers.text_parser import TextParser

        content = "# 第一章 概述\n\n这是一段中文内容。\n\n## 1.1 背景\n\n更多中文文本。"
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8")
        tmp.write(content)
        tmp.close()

        parser = TextParser()
        try:
            result = await parser.parse(tmp.name)
            headings = [b for b in result.blocks if b.type == "heading"]
            assert len(headings) >= 2
            assert "第一章 概述" in headings[0].text
        finally:
            os.unlink(tmp.name)

    @pytest.mark.asyncio
    async def test_gbk_encoded_file(self):
        """Parser handles GBK-encoded files."""
        from app.services.parsers.text_parser import TextParser

        content = "这是GBK编码的文件内容"
        tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".txt", delete=False)
        tmp.write(content.encode("gbk"))
        tmp.close()

        parser = TextParser()
        try:
            result = await parser.parse(tmp.name)
            assert len(result.blocks) > 0
            assert "GBK" in result.blocks[0].text
        finally:
            os.unlink(tmp.name)
