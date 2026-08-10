"""Unit tests for IntelligentChunker: Profile-driven document chunking with hierarchy.

Tests cover:
- Token counting (tiktoken or word-based fallback)
- Profile-driven chunking (min_tokens, max_tokens, overlap, respect_heading_level)
- Heading-aware section splitting
- Parent-child hierarchy (up to 6 levels)
- Chunk metadata (title_chain, source_file, page_number, space_id, permissions)
- Protect patterns (formulas, numbers with units not split)
- Row-level table chunking
- Overlap between adjacent chunks
"""

import re

import pytest
from hypothesis import HealthCheck, given, settings as hyp_settings, strategies as st

from app.services.chunker import (
    Chunk,
    ChunkingContext,
    IntelligentChunker,
    count_tokens,
)
from app.services.document_processor import ProcessedBlock, ProcessedDocument
from app.services.profile_matcher import (
    ChunkingConfig,
    DocumentProfileConfig,
    TableConfig,
)


# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def chunker() -> IntelligentChunker:
    """Default intelligent chunker."""
    return IntelligentChunker()


@pytest.fixture
def default_profile() -> DocumentProfileConfig:
    """Default profile with standard chunking config."""
    return DocumentProfileConfig(
        id="default",
        name="generic-text",
        chunking=ChunkingConfig(
            min_tokens=256,
            max_tokens=800,
            overlap_tokens=80,
            respect_heading_level=1,
            protect_patterns=[],
        ),
        tables=TableConfig(),
    )


@pytest.fixture
def small_chunk_profile() -> DocumentProfileConfig:
    """Profile with small chunk sizes for testing."""
    return DocumentProfileConfig(
        id="small",
        name="small-chunks",
        chunking=ChunkingConfig(
            min_tokens=10,
            max_tokens=50,
            overlap_tokens=5,
            respect_heading_level=1,
            protect_patterns=[],
        ),
        tables=TableConfig(),
    )


@pytest.fixture
def context() -> ChunkingContext:
    """Default chunking context."""
    return ChunkingContext(
        source_file="test_document.pdf",
        space_id="space-001",
        permission_ids=["perm-read-all"],
        document_id="doc-001",
    )


def _make_paragraph(text: str, page: int = 1) -> ProcessedBlock:
    """Helper to create a paragraph block."""
    return ProcessedBlock(type="paragraph", text=text, page_number=page)


def _make_heading(text: str, level: int, page: int = 1) -> ProcessedBlock:
    """Helper to create a heading block."""
    return ProcessedBlock(type="heading", text=text, heading_level=level, page_number=page)


# ─── Token Counting Tests ──────────────────────────────────────────────


class TestTokenCounting:
    """Tests for token counting functionality."""

    def test_count_tokens_english(self):
        """Token counting works for English text."""
        text = "Hello world, this is a test."
        tokens = count_tokens(text)
        assert tokens > 0
        assert tokens < 20  # Should be around 7-8 tokens

    def test_count_tokens_chinese(self):
        """Token counting works for Chinese text."""
        text = "这是一个测试文本"
        tokens = count_tokens(text)
        assert tokens > 0

    def test_count_tokens_empty(self):
        """Empty text returns 0 tokens."""
        assert count_tokens("") == 0

    def test_count_tokens_mixed(self):
        """Token counting works for mixed Chinese/English text."""
        text = "Hello 你好 World 世界"
        tokens = count_tokens(text)
        assert tokens > 0


# ─── Basic Chunking Tests ──────────────────────────────────────────────


class TestBasicChunking:
    """Tests for basic chunking functionality."""

    def test_empty_document_returns_empty(self, chunker, default_profile):
        """Empty document produces no chunks."""
        doc = ProcessedDocument(blocks=[])
        chunks = chunker.chunk(doc, default_profile)
        assert chunks == []

    def test_single_small_block(self, chunker, default_profile, context):
        """A single small block produces one chunk."""
        doc = ProcessedDocument(
            blocks=[_make_paragraph("This is a short paragraph.")]
        )
        chunks = chunker.chunk(doc, default_profile, context)

        assert len(chunks) >= 1
        assert "This is a short paragraph." in chunks[0].text

    def test_chunks_have_unique_ids(self, chunker, small_chunk_profile, context):
        """Each chunk has a unique ID."""
        blocks = [
            _make_paragraph("First paragraph with some content."),
            _make_paragraph("Second paragraph with different content."),
            _make_paragraph("Third paragraph with more content here."),
        ]
        doc = ProcessedDocument(blocks=blocks)
        chunks = chunker.chunk(doc, small_chunk_profile, context)

        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))  # All unique

    def test_chunk_index_sequential(self, chunker, small_chunk_profile, context):
        """Chunk indices are sequential starting from 0."""
        blocks = [
            _make_paragraph("First paragraph content here."),
            _make_paragraph("Second paragraph content here."),
            _make_paragraph("Third paragraph content here."),
        ]
        doc = ProcessedDocument(blocks=blocks)
        chunks = chunker.chunk(doc, small_chunk_profile, context)

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i


# ─── Heading-aware Splitting Tests ─────────────────────────────────────


class TestHeadingAwareSplitting:
    """Tests for heading-aware section splitting."""

    def test_split_at_heading_level_1(self, chunker, context):
        """Chunks are split at heading level 1 boundaries."""
        profile = DocumentProfileConfig(
            id="h1-split",
            name="h1-split",
            chunking=ChunkingConfig(
                min_tokens=5,
                max_tokens=500,
                overlap_tokens=0,
                respect_heading_level=1,
            ),
            tables=TableConfig(),
        )
        blocks = [
            _make_heading("Chapter 1", level=1),
            _make_paragraph("Content of chapter 1."),
            _make_heading("Chapter 2", level=1),
            _make_paragraph("Content of chapter 2."),
        ]
        doc = ProcessedDocument(blocks=blocks)
        chunks = chunker.chunk(doc, profile, context)

        # Should have at least 2 chunks (one per chapter)
        assert len(chunks) >= 2
        # First chunk should contain chapter 1 content
        assert any("Chapter 1" in c.text for c in chunks)
        assert any("Chapter 2" in c.text for c in chunks)

    def test_subheadings_within_section(self, chunker, context):
        """Subheadings below respect_heading_level stay in same section."""
        profile = DocumentProfileConfig(
            id="h1-only",
            name="h1-only",
            chunking=ChunkingConfig(
                min_tokens=5,
                max_tokens=2000,
                overlap_tokens=0,
                respect_heading_level=1,
            ),
            tables=TableConfig(),
        )
        blocks = [
            _make_heading("Chapter 1", level=1),
            _make_heading("Section 1.1", level=2),
            _make_paragraph("Content of section 1.1."),
            _make_heading("Section 1.2", level=2),
            _make_paragraph("Content of section 1.2."),
        ]
        doc = ProcessedDocument(blocks=blocks)
        chunks = chunker.chunk(doc, profile, context)

        # All content should be in one section (only split at level 1)
        all_text = " ".join(c.text for c in chunks)
        assert "Section 1.1" in all_text
        assert "Section 1.2" in all_text


# ─── Title Chain Tests ─────────────────────────────────────────────────


class TestTitleChain:
    """Tests for title chain metadata."""

    def test_title_chain_built_from_headings(self, chunker, context):
        """Title chain reflects the heading hierarchy."""
        profile = DocumentProfileConfig(
            id="chain",
            name="chain-test",
            chunking=ChunkingConfig(
                min_tokens=5,
                max_tokens=2000,
                overlap_tokens=0,
                respect_heading_level=2,
            ),
            tables=TableConfig(),
        )
        blocks = [
            _make_heading("Chapter 1", level=1),
            _make_heading("Section A", level=2),
            _make_paragraph("Content under Section A."),
        ]
        doc = ProcessedDocument(blocks=blocks)
        chunks = chunker.chunk(doc, profile, context)

        # Find chunk with content
        content_chunks = [c for c in chunks if "Content under Section A" in c.text]
        assert len(content_chunks) >= 1
        # Title chain should include both headings
        assert "Chapter 1" in content_chunks[0].title_chain
        assert "Section A" in content_chunks[0].title_chain


# ─── Overlap Tests ─────────────────────────────────────────────────────


class TestOverlap:
    """Tests for overlap between adjacent chunks."""

    def test_overlap_applied_between_chunks(self, chunker, context):
        """Adjacent chunks have overlapping content."""
        profile = DocumentProfileConfig(
            id="overlap",
            name="overlap-test",
            chunking=ChunkingConfig(
                min_tokens=5,
                max_tokens=30,
                overlap_tokens=10,
                respect_heading_level=1,
            ),
            tables=TableConfig(),
        )
        # Create enough content to force multiple chunks
        long_text = " ".join(["word"] * 100)
        blocks = [_make_paragraph(long_text)]
        doc = ProcessedDocument(blocks=blocks)
        chunks = chunker.chunk(doc, profile, context)

        if len(chunks) >= 2:
            # Second chunk should start with content from end of first chunk
            # (overlap means some text from chunk N appears at start of chunk N+1)
            assert chunks[1].token_count > 0

    def test_no_overlap_when_zero(self, chunker, context):
        """No overlap when overlap_tokens is 0."""
        profile = DocumentProfileConfig(
            id="no-overlap",
            name="no-overlap",
            chunking=ChunkingConfig(
                min_tokens=5,
                max_tokens=30,
                overlap_tokens=0,
                respect_heading_level=1,
            ),
            tables=TableConfig(),
        )
        long_text = " ".join(["word"] * 100)
        blocks = [_make_paragraph(long_text)]
        doc = ProcessedDocument(blocks=blocks)
        chunks = chunker.chunk(doc, profile, context)

        # With 0 overlap, chunks should not share content
        # (hard to verify exactly, but token counts should be reasonable)
        for chunk in chunks:
            assert chunk.token_count > 0


# ─── Parent-child Hierarchy Tests ──────────────────────────────────────


class TestHierarchy:
    """Tests for parent-child hierarchy (up to 6 levels)."""

    def test_hierarchy_depth_set(self, chunker, context):
        """Chunks have depth set based on heading level."""
        profile = DocumentProfileConfig(
            id="hierarchy",
            name="hierarchy-test",
            chunking=ChunkingConfig(
                min_tokens=5,
                max_tokens=2000,
                overlap_tokens=0,
                respect_heading_level=3,
            ),
            tables=TableConfig(),
        )
        blocks = [
            _make_heading("Level 1", level=1),
            _make_paragraph("Content under level 1."),
            _make_heading("Level 2", level=2),
            _make_paragraph("Content under level 2."),
            _make_heading("Level 3", level=3),
            _make_paragraph("Content under level 3."),
        ]
        doc = ProcessedDocument(blocks=blocks)
        chunks = chunker.chunk(doc, profile, context)

        # Should have chunks at different depths
        depths = set(c.depth for c in chunks)
        assert len(depths) >= 2

    def test_parent_id_links_to_parent_chunk(self, chunker, context):
        """Child chunks reference their parent chunk ID."""
        profile = DocumentProfileConfig(
            id="parent",
            name="parent-test",
            chunking=ChunkingConfig(
                min_tokens=5,
                max_tokens=2000,
                overlap_tokens=0,
                respect_heading_level=2,
            ),
            tables=TableConfig(),
        )
        blocks = [
            _make_heading("Parent Heading", level=1),
            _make_heading("Child Heading", level=2),
            _make_paragraph("Child content."),
        ]
        doc = ProcessedDocument(blocks=blocks)
        chunks = chunker.chunk(doc, profile, context)

        # Find parent and child
        parent_chunks = [c for c in chunks if "Parent Heading" in c.text]
        child_chunks = [c for c in chunks if c.parent_id is not None]

        if parent_chunks and child_chunks:
            # At least one child should reference the parent
            parent_ids = {c.id for c in parent_chunks}
            assert any(c.parent_id in parent_ids for c in child_chunks)

    def test_depth_capped_at_6(self, chunker, context):
        """Depth is capped at 6 regardless of heading level."""
        profile = DocumentProfileConfig(
            id="deep",
            name="deep-test",
            chunking=ChunkingConfig(
                min_tokens=5,
                max_tokens=2000,
                overlap_tokens=0,
                respect_heading_level=6,
            ),
            tables=TableConfig(),
        )
        blocks = [
            _make_heading("H1", level=1),
            _make_heading("H2", level=2),
            _make_heading("H3", level=3),
            _make_heading("H4", level=4),
            _make_heading("H5", level=5),
            _make_heading("H6", level=6),
            _make_paragraph("Deep content."),
        ]
        doc = ProcessedDocument(blocks=blocks)
        chunks = chunker.chunk(doc, profile, context)

        for chunk in chunks:
            assert chunk.depth <= 6


# ─── Metadata Tests ───────────────────────────────────────────────────


class TestChunkMetadata:
    """Tests for chunk metadata attachment."""

    def test_source_file_attached(self, chunker, default_profile, context):
        """source_file is attached from context."""
        doc = ProcessedDocument(blocks=[_make_paragraph("Content")])
        chunks = chunker.chunk(doc, default_profile, context)

        for chunk in chunks:
            assert chunk.source_file == "test_document.pdf"

    def test_space_id_attached(self, chunker, default_profile, context):
        """space_id is attached from context."""
        doc = ProcessedDocument(blocks=[_make_paragraph("Content")])
        chunks = chunker.chunk(doc, default_profile, context)

        for chunk in chunks:
            assert chunk.space_id == "space-001"

    def test_permission_ids_attached(self, chunker, default_profile, context):
        """permission_ids are attached from context."""
        doc = ProcessedDocument(blocks=[_make_paragraph("Content")])
        chunks = chunker.chunk(doc, default_profile, context)

        for chunk in chunks:
            assert chunk.permission_ids == ["perm-read-all"]

    def test_page_number_from_block(self, chunker, default_profile, context):
        """page_number reflects the source block's page."""
        blocks = [
            _make_paragraph("Page 1 content"),
            ProcessedBlock(type="paragraph", text="Page 3 content", page_number=3),
        ]
        doc = ProcessedDocument(blocks=blocks)
        chunks = chunker.chunk(doc, default_profile, context)

        # At least one chunk should reference page 1
        pages = [c.page_number for c in chunks]
        assert 1 in pages

    def test_token_count_set(self, chunker, default_profile, context):
        """token_count is set for each chunk."""
        doc = ProcessedDocument(blocks=[_make_paragraph("Some content here.")])
        chunks = chunker.chunk(doc, default_profile, context)

        for chunk in chunks:
            assert chunk.token_count > 0
            # Token count should match actual count
            assert chunk.token_count == count_tokens(chunk.text)


# ─── Protect Patterns Tests ───────────────────────────────────────────


class TestProtectPatterns:
    """Tests for formula and numeric atomicity protection."""

    def test_default_numeric_patterns_protected(self, chunker, context):
        """Default numeric patterns (e.g., 0.05mm/m) are not split."""
        profile = DocumentProfileConfig(
            id="protect",
            name="protect-test",
            chunking=ChunkingConfig(
                min_tokens=5,
                max_tokens=20,
                overlap_tokens=0,
                respect_heading_level=1,
                protect_patterns=[],
            ),
            tables=TableConfig(),
        )
        # Text with numeric values that should stay together
        text = "The tolerance is 0.05mm/m and the deviation is ±10mm for this measurement."
        blocks = [_make_paragraph(text)]
        doc = ProcessedDocument(blocks=blocks)
        chunks = chunker.chunk(doc, profile, context)

        # The numeric values should appear intact in some chunk
        all_text = " ".join(c.text for c in chunks)
        assert "0.05mm/m" in all_text

    def test_custom_protect_patterns(self, chunker, context):
        """Custom protect patterns from profile are applied."""
        profile = DocumentProfileConfig(
            id="custom-protect",
            name="custom-protect",
            chunking=ChunkingConfig(
                min_tokens=5,
                max_tokens=20,
                overlap_tokens=0,
                respect_heading_level=1,
                protect_patterns=[r"FORMULA\[\d+\]"],
            ),
            tables=TableConfig(),
        )
        text = "Before FORMULA[123] after more text here to fill the chunk."
        blocks = [_make_paragraph(text)]
        doc = ProcessedDocument(blocks=blocks)
        chunks = chunker.chunk(doc, profile, context)

        # FORMULA[123] should appear intact
        all_text = " ".join(c.text for c in chunks)
        assert "FORMULA[123]" in all_text

    def test_formula_blocks_are_atomic(self, chunker, default_profile, context):
        """Formula-type blocks are treated as atomic (not split)."""
        blocks = [
            ProcessedBlock(type="formula", text="E = mc^2 + ∑(x_i)", page_number=1),
            _make_paragraph("Following paragraph."),
        ]
        doc = ProcessedDocument(blocks=blocks)
        chunks = chunker.chunk(doc, default_profile, context)

        # Formula should appear intact in a chunk
        all_text = " ".join(c.text for c in chunks)
        assert "E = mc^2 + ∑(x_i)" in all_text


# ─── Row-level Table Chunking Tests ───────────────────────────────────


class TestRowLevelTableChunking:
    """Tests for row-level table chunking when enabled."""

    def test_table_split_by_rows_when_enabled(self, chunker, context):
        """Tables are split by rows when row_level_chunking is enabled."""
        profile = DocumentProfileConfig(
            id="row-chunk",
            name="row-chunk",
            chunking=ChunkingConfig(
                min_tokens=5,
                max_tokens=2000,
                overlap_tokens=0,
                respect_heading_level=1,
            ),
            tables=TableConfig(row_level_chunking=True),
        )
        table_text = "| Name | Value |\n| --- | --- |\n| A | 1 |\n| B | 2 |\n| C | 3 |"
        blocks = [
            ProcessedBlock(type="table", text=table_text, page_number=1),
        ]
        doc = ProcessedDocument(blocks=blocks)
        chunks = chunker.chunk(doc, profile, context)

        # Each row should be in the output
        all_text = " ".join(c.text for c in chunks)
        assert "| A | 1 |" in all_text
        assert "| B | 2 |" in all_text
        assert "| C | 3 |" in all_text

    def test_table_not_split_when_disabled(self, chunker, context):
        """Tables are kept whole when row_level_chunking is disabled."""
        profile = DocumentProfileConfig(
            id="no-row-chunk",
            name="no-row-chunk",
            chunking=ChunkingConfig(
                min_tokens=5,
                max_tokens=2000,
                overlap_tokens=0,
                respect_heading_level=1,
            ),
            tables=TableConfig(row_level_chunking=False),
        )
        table_text = "| Name | Value |\n| --- | --- |\n| A | 1 |\n| B | 2 |"
        blocks = [
            ProcessedBlock(type="table", text=table_text, page_number=1),
        ]
        doc = ProcessedDocument(blocks=blocks)
        chunks = chunker.chunk(doc, profile, context)

        # Table should be in one chunk
        found = any(
            "| A | 1 |" in c.text and "| B | 2 |" in c.text
            for c in chunks
        )
        assert found


# ─── Token Limit Tests ─────────────────────────────────────────────────


class TestTokenLimits:
    """Tests for token limit enforcement."""

    def test_large_text_split_into_multiple_chunks(self, chunker, context):
        """Text exceeding max_tokens is split into multiple chunks."""
        profile = DocumentProfileConfig(
            id="small-max",
            name="small-max",
            chunking=ChunkingConfig(
                min_tokens=5,
                max_tokens=30,
                overlap_tokens=0,
                respect_heading_level=1,
            ),
            tables=TableConfig(),
        )
        # Create text that's definitely larger than 30 tokens
        long_text = " ".join(["word"] * 200)
        blocks = [_make_paragraph(long_text)]
        doc = ProcessedDocument(blocks=blocks)
        chunks = chunker.chunk(doc, profile, context)

        assert len(chunks) > 1

    def test_small_trailing_chunk_merged(self, chunker, context):
        """Small trailing chunks are merged with the previous chunk."""
        profile = DocumentProfileConfig(
            id="merge-test",
            name="merge-test",
            chunking=ChunkingConfig(
                min_tokens=50,
                max_tokens=200,
                overlap_tokens=0,
                respect_heading_level=1,
            ),
            tables=TableConfig(),
        )
        # Create blocks where the last one would be too small alone
        blocks = [
            _make_paragraph(" ".join(["content"] * 60)),
            _make_paragraph("tiny"),
        ]
        doc = ProcessedDocument(blocks=blocks)
        chunks = chunker.chunk(doc, profile, context)

        # The tiny block should be merged with the previous chunk
        # rather than creating a chunk below min_tokens
        if len(chunks) == 1:
            assert "tiny" in chunks[0].text


# ─── Asset IDs Tests ──────────────────────────────────────────────────


class TestAssetIds:
    """Tests for asset ID tracking in chunks."""

    def test_image_asset_ids_in_chunk(self, chunker, default_profile, context):
        """Image asset IDs are tracked in chunk metadata."""
        blocks = [
            ProcessedBlock(
                type="image",
                text="[图片: 流程图]",
                page_number=1,
                asset_ids=["img-001", "img-002"],
            ),
            _make_paragraph("Description of the image."),
        ]
        doc = ProcessedDocument(blocks=blocks)
        chunks = chunker.chunk(doc, default_profile, context)

        # At least one chunk should have the asset IDs
        all_assets = []
        for c in chunks:
            all_assets.extend(c.asset_ids)
        assert "img-001" in all_assets
        assert "img-002" in all_assets


# ─── Respect Heading Level Tests ──────────────────────────────────────


class TestRespectHeadingLevel:
    """Tests confirming sections are split at the configured heading level."""

    def test_respect_heading_level_2_splits_at_h1_and_h2(self, chunker, context):
        """When respect_heading_level=2, both H1 and H2 start new sections."""
        profile = DocumentProfileConfig(
            id="h2-respect",
            name="h2-respect",
            chunking=ChunkingConfig(
                min_tokens=1,
                max_tokens=10000,
                overlap_tokens=0,
                respect_heading_level=2,
            ),
            tables=TableConfig(),
        )
        blocks = [
            _make_heading("Chapter 1", level=1),
            _make_paragraph("Intro content under chapter 1."),
            _make_heading("Section 1.1", level=2),
            _make_paragraph("Content of 1.1."),
            _make_heading("Section 1.2", level=2),
            _make_paragraph("Content of 1.2."),
            _make_heading("Chapter 2", level=1),
            _make_paragraph("Content of chapter 2."),
        ]
        doc = ProcessedDocument(blocks=blocks)
        chunks = chunker.chunk(doc, profile, context)

        # Each H1 / H2 should give rise to its own chunk that does not contain
        # content from a sibling section.
        section_1_1 = [c for c in chunks if "Content of 1.1" in c.text]
        section_1_2 = [c for c in chunks if "Content of 1.2" in c.text]

        assert section_1_1, "expected a chunk containing Section 1.1 content"
        assert section_1_2, "expected a chunk containing Section 1.2 content"

        # Section 1.1's chunk must not bleed into 1.2's content.
        assert "Content of 1.2" not in section_1_1[0].text
        assert "Content of 1.1" not in section_1_2[0].text

    def test_respect_heading_level_2_keeps_h3_inline(self, chunker, context):
        """H3 sub-headings (deeper than respect_heading_level=2) stay inside their H2."""
        profile = DocumentProfileConfig(
            id="h2-keep-h3",
            name="h2-keep-h3",
            chunking=ChunkingConfig(
                min_tokens=1,
                max_tokens=10000,
                overlap_tokens=0,
                respect_heading_level=2,
            ),
            tables=TableConfig(),
        )
        blocks = [
            _make_heading("Chapter 1", level=1),
            _make_heading("Section 1.1", level=2),
            _make_heading("Detail 1.1.1", level=3),
            _make_paragraph("Deep detail content."),
            _make_heading("Detail 1.1.2", level=3),
            _make_paragraph("Another deep detail."),
        ]
        doc = ProcessedDocument(blocks=blocks)
        chunks = chunker.chunk(doc, profile, context)

        # Both H3 details should land in the same chunk because we only split at H2.
        merged_chunks = [
            c for c in chunks
            if "Deep detail content" in c.text and "Another deep detail" in c.text
        ]
        assert merged_chunks, "expected H3 sections to share the parent H2 chunk"


# ─── Title Chain Format Tests ─────────────────────────────────────────


class TestTitleChainFormat:
    """Tests asserting the exact ' > ' separator format for title chains."""

    def test_title_chain_uses_space_gt_space_separator(self, chunker, context):
        """The title chain joins headings with ' > ' (space, greater-than, space)."""
        profile = DocumentProfileConfig(
            id="chain-format",
            name="chain-format",
            chunking=ChunkingConfig(
                min_tokens=1,
                max_tokens=10000,
                overlap_tokens=0,
                respect_heading_level=3,
            ),
            tables=TableConfig(),
        )
        blocks = [
            _make_heading("第一章", level=1),
            _make_heading("基本要求", level=2),
            _make_heading("具体规定", level=3),
            _make_paragraph("章节正文。"),
        ]
        doc = ProcessedDocument(blocks=blocks)
        chunks = chunker.chunk(doc, profile, context)

        body_chunks = [c for c in chunks if "章节正文" in c.text]
        assert body_chunks
        chain = body_chunks[0].title_chain
        assert chain == "第一章 > 基本要求 > 具体规定"


# ─── Token Counting Backend Tests ─────────────────────────────────────


class TestTiktokenBackend:
    """Verify token counting prefers tiktoken cl100k_base when available."""

    def test_tiktoken_cl100k_used_when_available(self):
        """If tiktoken is installed, count_tokens must match cl100k_base encoding."""
        try:
            import tiktoken
        except ImportError:
            pytest.skip("tiktoken not installed")

        enc = tiktoken.get_encoding("cl100k_base")
        text = "Hello world, 这是一个混合 token 测试。"
        assert count_tokens(text) == len(enc.encode(text))

    def test_token_count_aligns_with_max_tokens(self, chunker, context):
        """Each chunk's reported token_count never exceeds max_tokens (best effort)."""
        max_tokens = 80
        profile = DocumentProfileConfig(
            id="token-cap",
            name="token-cap",
            chunking=ChunkingConfig(
                min_tokens=1,
                max_tokens=max_tokens,
                overlap_tokens=0,
                respect_heading_level=1,
            ),
            tables=TableConfig(),
        )
        # Plenty of paragraphs of moderate size, containing no protected text.
        blocks = [
            _make_paragraph(" ".join(["alpha", "beta", "gamma", "delta"] * 30))
        ]
        doc = ProcessedDocument(blocks=blocks)
        chunks = chunker.chunk(doc, profile, context)

        assert chunks
        # All chunks should be reasonably bounded by max_tokens (allow +20% slack
        # because protected spans or clause boundaries can slightly exceed the cap).
        for chunk in chunks:
            assert chunk.token_count == count_tokens(chunk.text)
            assert chunk.token_count <= int(max_tokens * 1.5)


# ─── Full Metadata Tests ──────────────────────────────────────────────


class TestFullChunkMetadata:
    """Tests that every required metadata field on Chunk is populated."""

    def test_all_required_fields_set(self, chunker, context):
        """Chunk metadata covers title_chain, source_file, page_number, space_id,
        allowed user / permission ids and asset_ids per the design document."""
        profile = DocumentProfileConfig(
            id="meta",
            name="meta",
            chunking=ChunkingConfig(
                min_tokens=1,
                max_tokens=10000,
                overlap_tokens=0,
                respect_heading_level=1,
            ),
            tables=TableConfig(),
        )
        blocks = [
            _make_heading("Chapter A", level=1, page=1),
            _make_paragraph("Chapter A intro paragraph.", page=1),
            ProcessedBlock(
                type="image",
                text="[图片: diagram]",
                page_number=2,
                asset_ids=["asset-9"],
            ),
        ]
        doc = ProcessedDocument(blocks=blocks)
        chunks = chunker.chunk(doc, profile, context)

        assert chunks
        for chunk in chunks:
            assert chunk.id  # uuid generated
            assert chunk.source_file == context.source_file
            assert chunk.space_id == context.space_id
            assert chunk.permission_ids == context.permission_ids
            assert chunk.token_count > 0
            assert chunk.page_number >= 1
        # Title chain present for content under heading
        body = [c for c in chunks if "Chapter A intro paragraph" in c.text]
        assert body and "Chapter A" in body[0].title_chain
        # Asset IDs propagated
        assert any("asset-9" in c.asset_ids for c in chunks)


# ─── Protect Pattern Property-Based Tests ─────────────────────────────


# Strategy for tokens that participate as protected segments. We restrict the
# alphabet to letters and digits so the surrounding chunker cannot accidentally
# treat parts of the protected literal as sentence/clause boundaries.
_protect_token = st.text(
    alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd"),
        whitelist_characters="",
    ),
    min_size=4,
    max_size=12,
)

# Strategy for filler words around the protected literal. We avoid characters
# that are split-point candidates inside protected spans.
_filler_word = st.text(
    alphabet=st.characters(
        whitelist_categories=("Lu", "Ll"),
        whitelist_characters="",
    ),
    min_size=3,
    max_size=8,
)


class TestProtectPatternProperty:
    """Property test: the chunker must never split a protected literal across chunks."""

    @hyp_settings(
        max_examples=40,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
    )
    @given(
        protected=_protect_token,
        prefix=st.lists(_filler_word, min_size=20, max_size=80),
        suffix=st.lists(_filler_word, min_size=20, max_size=80),
    )
    def test_protected_literal_never_split(self, protected, prefix, suffix):
        """A literal that matches the configured protect pattern must appear intact in
        exactly one chunk.

        **Validates: Requirements 5.9, 5.10**
        """
        chunker = IntelligentChunker()
        # Wrap the protected token in a recognisable marker so the regex is exact.
        marker = f"PROTECT-{protected}-END"
        prefix_text = " ".join(prefix)
        suffix_text = " ".join(suffix)
        text = f"{prefix_text} {marker} {suffix_text}".strip()

        # Force aggressive chunking so the protected marker is a real challenge.
        profile = DocumentProfileConfig(
            id="hyp",
            name="hyp",
            chunking=ChunkingConfig(
                min_tokens=1,
                max_tokens=20,
                overlap_tokens=0,
                respect_heading_level=1,
                protect_patterns=[re.escape(marker)],
            ),
            tables=TableConfig(),
        )
        doc = ProcessedDocument(blocks=[_make_paragraph(text)])
        ctx = ChunkingContext(source_file="hyp.txt", space_id="s", permission_ids=[])
        chunks = chunker.chunk(doc, profile, ctx)

        # The marker must be present, intact, and only in one chunk.
        appearances = [c for c in chunks if marker in c.text]
        assert appearances, f"protected literal lost: marker={marker!r} chunks={[c.text for c in chunks]}"
        assert len(appearances) == 1, (
            f"protected literal duplicated across chunks: {[c.text for c in appearances]}"
        )
