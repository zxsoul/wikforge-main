"""Document Processor: cleaning, noise removal, heading detection, and Markdown conversion.

Implements Profile-driven document processing:
- Basic noise removal (whitespace/blank line compression)
- Statistical watermark/header/footer detection (≥50% frequency = noise)
- Profile-driven boilerplate removal (regex patterns)
- Profile-driven heading level identification (regex + level mapping)
- Markdown format unification (paragraphs, bold, italic, links)
- Table conversion (standard Markdown tables + complex table text fallback)
- Cross-page table merging (same header/column structure across adjacent pages)
- Large table row-level chunking
- Formula and numeric atomicity protection
- Multimodal LLM image description generation (stub, configurable)
"""

import logging
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from app.services.parsers.base import Block, ParsedDocument
from app.services.profile_matcher import (
    BoilerplateConfig,
    DocumentProfileConfig,
    HeadingRule,
    TableConfig,
)

# Type alias for an injectable image-description callable.
# Receives the image Block and returns a Markdown string describing the image.
ImageDescriber = Callable[[Block], str]

logger = logging.getLogger(__name__)


@dataclass
class ProcessedBlock:
    """A block after cleaning and structural recognition.

    Attributes:
        type: Block type after processing ("heading", "paragraph", "table", "image", "formula")
        text: Cleaned text content (Markdown formatted)
        heading_level: Heading level (1-6) if type is "heading", else 0
        page_number: Original page number
        is_noise: Whether this block was identified as noise
        asset_ids: Referenced asset IDs (images, formulas)
        original_text: Original text before cleaning
    """

    type: str
    text: str
    heading_level: int = 0
    page_number: int = 1
    is_noise: bool = False
    asset_ids: list[str] = field(default_factory=list)
    original_text: str = ""


@dataclass
class ProcessedDocument:
    """Result of document processing (cleaning + structural recognition).

    Attributes:
        blocks: Processed blocks in order
        metadata: Document metadata
        markdown: Full Markdown content
        noise_removed_count: Number of blocks removed as noise
        headings_detected: Number of headings identified
    """

    blocks: list[ProcessedBlock] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    markdown: str = ""
    noise_removed_count: int = 0
    headings_detected: int = 0


class DocumentProcessor:
    """Profile-driven document processor.

    Applies cleaning, structural recognition, and Markdown conversion
    based on the matched DocumentProfile configuration.
    """

    def __init__(
        self,
        enable_llm_image_description: bool = False,
        image_describer: ImageDescriber | None = None,
    ):
        """Initialize the document processor.

        Args:
            enable_llm_image_description: Whether to enable LLM-based image descriptions.
                When True, ``image_describer`` is invoked for each image block.
            image_describer: Optional callable that receives an image Block and returns a
                Markdown description string. This is the integration point for the
                multimodal LLM Gateway (see ``app.services.llm_gateway.LLMGateway``).
                Production code is expected to inject a closure that calls
                ``LLMGateway.complete_multimodal``. Tests inject deterministic stubs.
        """
        self._enable_llm_image_description = enable_llm_image_description
        self._image_describer = image_describer

    def process(
        self,
        parsed_doc: ParsedDocument,
        profile: DocumentProfileConfig,
    ) -> ProcessedDocument:
        """Process a parsed document using the given profile.

        Steps:
        1. Basic noise removal (whitespace compression)
        2. Statistical noise detection (watermarks, headers, footers)
        3. Profile-driven boilerplate removal
        4. Profile-driven heading identification
        5. Cross-page table merging
        6. Markdown conversion
        7. Image description generation (stub)

        Args:
            parsed_doc: The parsed document intermediate representation
            profile: The matched document profile configuration

        Returns:
            ProcessedDocument with cleaned blocks and Markdown output
        """
        blocks = parsed_doc.blocks
        if not blocks:
            return ProcessedDocument(metadata=parsed_doc.metadata)

        # Step 1: Basic noise removal
        blocks = self._basic_noise_removal(blocks)

        # Step 2: Statistical noise detection
        noise_texts = self._detect_statistical_noise(blocks, profile.boilerplate)

        # Step 3: Profile-driven boilerplate removal
        boilerplate_patterns = self._compile_boilerplate_patterns(profile.boilerplate)

        # Step 4: Cross-page table merging
        blocks = self._merge_cross_page_tables(blocks, profile.tables)

        # Step 5: Process each block
        processed_blocks: list[ProcessedBlock] = []
        noise_count = 0

        for block in blocks:
            # Check if block is noise
            if self._is_noise_block(block, noise_texts, boilerplate_patterns):
                noise_count += 1
                continue

            # Identify headings
            processed = self._process_block(block, profile)
            processed_blocks.append(processed)

        # Step 6: Generate Markdown
        markdown = self._generate_markdown(processed_blocks, profile)

        headings_detected = sum(1 for b in processed_blocks if b.type == "heading")

        return ProcessedDocument(
            blocks=processed_blocks,
            metadata=parsed_doc.metadata,
            markdown=markdown,
            noise_removed_count=noise_count,
            headings_detected=headings_detected,
        )

    # ─── Step 1: Basic Noise Removal ──────────────────────────────────

    def _basic_noise_removal(self, blocks: list[Block]) -> list[Block]:
        """Remove basic noise: compress whitespace and blank lines.

        - Multiple consecutive spaces → single space
        - Multiple blank lines → single blank line
        - Leading/trailing whitespace trimmed per line

        Args:
            blocks: Input blocks

        Returns:
            Blocks with cleaned text
        """
        cleaned = []
        for block in blocks:
            text = block.text
            # Compress multiple spaces to single space (per line)
            text = re.sub(r"[^\S\n]+", " ", text)
            # Compress multiple blank lines to single blank line
            text = re.sub(r"\n{3,}", "\n\n", text)
            # Trim each line
            lines = [line.strip() for line in text.split("\n")]
            text = "\n".join(lines)
            # Trim overall
            text = text.strip()

            cleaned.append(
                Block(
                    type=block.type,
                    text=text,
                    bbox=block.bbox,
                    page_number=block.page_number,
                    style=block.style,
                    raw=block.raw,
                )
            )
        return cleaned

    # ─── Step 2: Statistical Noise Detection ──────────────────────────

    def _detect_statistical_noise(
        self, blocks: list[Block], boilerplate_config: BoilerplateConfig
    ) -> set[str]:
        """Detect noise using statistical methods.

        Identifies text that appears at the same position across pages
        with frequency ≥ threshold (default 50%).

        Args:
            blocks: All blocks in the document
            boilerplate_config: Boilerplate detection configuration

        Returns:
            Set of text strings identified as noise
        """
        if boilerplate_config.detection_mode == "manual":
            return set()

        threshold = boilerplate_config.statistical_threshold

        # Group blocks by page
        pages: dict[int, list[Block]] = {}
        for block in blocks:
            pages.setdefault(block.page_number, []).append(block)

        if len(pages) < 3:
            return set()

        total_pages = len(pages)
        noise_texts: set[str] = set()

        # Check first block of each page (potential header)
        first_texts: list[str] = []
        for page_num in sorted(pages.keys()):
            page_blocks = pages[page_num]
            if page_blocks and page_blocks[0].text.strip():
                first_texts.append(page_blocks[0].text.strip())

        if first_texts:
            counter = Counter(first_texts)
            for text, count in counter.items():
                if count / total_pages >= threshold:
                    noise_texts.add(text)

        # Check last block of each page (potential footer)
        last_texts: list[str] = []
        for page_num in sorted(pages.keys()):
            page_blocks = pages[page_num]
            if page_blocks and page_blocks[-1].text.strip():
                last_texts.append(page_blocks[-1].text.strip())

        if last_texts:
            counter = Counter(last_texts)
            for text, count in counter.items():
                if count / total_pages >= threshold:
                    noise_texts.add(text)

        # Check blocks at same bbox position across pages (watermarks)
        if any(b.bbox for b in blocks):
            self._detect_positional_noise(blocks, pages, total_pages, threshold, noise_texts)

        return noise_texts

    def _detect_positional_noise(
        self,
        blocks: list[Block],
        pages: dict[int, list[Block]],
        total_pages: int,
        threshold: float,
        noise_texts: set[str],
    ) -> None:
        """Detect noise by position (same text at same bbox across pages).

        Args:
            blocks: All blocks
            pages: Blocks grouped by page
            total_pages: Total number of pages
            threshold: Frequency threshold
            noise_texts: Set to add detected noise texts to (mutated)
        """
        # Group by approximate bbox position
        position_texts: dict[tuple, list[str]] = {}
        for block in blocks:
            if block.bbox and block.text.strip():
                # Round bbox to reduce floating point noise
                rounded_bbox = tuple(round(v, 1) for v in block.bbox)
                position_texts.setdefault(rounded_bbox, []).append(block.text.strip())

        for _bbox, texts in position_texts.items():
            if len(texts) >= 3:
                counter = Counter(texts)
                for text, count in counter.items():
                    if count / total_pages >= threshold:
                        noise_texts.add(text)

    # ─── Step 3: Profile-driven Boilerplate Removal ───────────────────

    def _compile_boilerplate_patterns(
        self, boilerplate_config: BoilerplateConfig
    ) -> list[re.Pattern]:
        """Compile boilerplate regex patterns from profile config.

        Args:
            boilerplate_config: Boilerplate configuration

        Returns:
            List of compiled regex patterns
        """
        if boilerplate_config.detection_mode == "statistical":
            return []

        patterns = []
        for pattern_str in boilerplate_config.manual_patterns:
            try:
                patterns.append(re.compile(pattern_str, re.MULTILINE))
            except re.error:
                logger.warning(f"Invalid boilerplate pattern: {pattern_str}")
        return patterns

    def _is_noise_block(
        self,
        block: Block,
        noise_texts: set[str],
        boilerplate_patterns: list[re.Pattern],
    ) -> bool:
        """Check if a block is noise.

        Args:
            block: Block to check
            noise_texts: Statistically detected noise texts
            boilerplate_patterns: Compiled boilerplate regex patterns

        Returns:
            True if the block is noise
        """
        text = block.text.strip()
        if not text:
            return True

        # Check against statistically detected noise
        if text in noise_texts:
            return True

        # Check against boilerplate patterns
        for pattern in boilerplate_patterns:
            if pattern.fullmatch(text):
                return True

        return False

    # ─── Step 4: Heading Identification ───────────────────────────────

    def _identify_heading(
        self, block: Block, heading_rules: list[HeadingRule]
    ) -> tuple[bool, int, str]:
        """Identify if a block is a heading using profile rules.

        Args:
            block: Block to check
            heading_rules: List of heading rules from profile

        Returns:
            Tuple of (is_heading, level, cleaned_text)
        """
        text = block.text.strip()

        # Check if block is already typed as heading
        if block.type == "heading":
            # Try to determine level from style
            level = block.style.get("heading_level", 0)
            if level > 0:
                return True, min(level, 6), text

        # Apply heading rules
        for rule in heading_rules:
            try:
                match = re.match(rule.pattern, text)
                if match:
                    cleaned_text = text
                    if rule.strip_pattern:
                        cleaned_text = re.sub(rule.pattern, "", text).strip()
                    return True, rule.level, cleaned_text
            except re.error:
                logger.warning(f"Invalid heading rule pattern: {rule.pattern}")

        # Check for Markdown-style headings
        md_match = re.match(r"^(#{1,6})\s+(.+)$", text)
        if md_match:
            level = len(md_match.group(1))
            return True, level, md_match.group(2)

        return False, 0, text

    # ─── Step 5: Cross-page Table Merging ─────────────────────────────

    def _merge_cross_page_tables(
        self, blocks: list[Block], table_config: TableConfig
    ) -> list[Block]:
        """Merge tables that span across adjacent pages.

        Detects tables with the same header or column structure on adjacent
        pages and merges them into a single table.

        Args:
            blocks: All blocks
            table_config: Table processing configuration

        Returns:
            Blocks with cross-page tables merged
        """
        if not table_config.cross_page_merge:
            return blocks

        result: list[Block] = []
        i = 0
        while i < len(blocks):
            block = blocks[i]
            if block.type != "table":
                result.append(block)
                i += 1
                continue

            # Look ahead for continuation tables on next page
            merged_text = block.text
            current_page = block.page_number
            header = self._extract_table_header(block.text)

            j = i + 1
            while j < len(blocks):
                next_block = blocks[j]
                if next_block.type == "table" and next_block.page_number == current_page + 1:
                    next_header = self._extract_table_header(next_block.text)
                    if header and next_header and self._headers_match(header, next_header):
                        # Merge: append rows without header
                        rows = next_block.text.strip().split("\n")
                        # Skip header row and separator row
                        data_rows = rows[2:] if len(rows) > 2 else rows
                        if data_rows:
                            merged_text += "\n" + "\n".join(data_rows)
                        current_page = next_block.page_number
                        j += 1
                        continue
                break

            result.append(
                Block(
                    type="table",
                    text=merged_text,
                    bbox=block.bbox,
                    page_number=block.page_number,
                    style=block.style,
                    raw=block.raw,
                )
            )
            i = j

        return result

    def _extract_table_header(self, table_text: str) -> list[str] | None:
        """Extract column headers from a Markdown table.

        Args:
            table_text: Markdown table text

        Returns:
            List of header cell texts, or None if not a valid table
        """
        lines = table_text.strip().split("\n")
        if not lines:
            return None

        header_line = lines[0]
        if "|" not in header_line:
            return None

        cells = [cell.strip() for cell in header_line.split("|") if cell.strip()]
        return cells if cells else None

    def _headers_match(self, header1: list[str], header2: list[str]) -> bool:
        """Check if two table headers match (same columns).

        Args:
            header1: First header cells
            header2: Second header cells

        Returns:
            True if headers match
        """
        if len(header1) != len(header2):
            return False
        return all(h1.lower() == h2.lower() for h1, h2 in zip(header1, header2))

    # ─── Block Processing ─────────────────────────────────────────────

    def _process_block(
        self, block: Block, profile: DocumentProfileConfig
    ) -> ProcessedBlock:
        """Process a single block: identify type, clean text, convert format.

        Args:
            block: Input block
            profile: Document profile

        Returns:
            ProcessedBlock with identified type and cleaned text
        """
        text = block.text.strip()
        original_text = text

        # Check for heading
        is_heading, level, heading_text = self._identify_heading(
            block, profile.heading_rules
        )
        if is_heading and level > 0:
            return ProcessedBlock(
                type="heading",
                text=heading_text,
                heading_level=level,
                page_number=block.page_number,
                original_text=original_text,
            )

        # Table blocks
        if block.type == "table":
            table_md = self._convert_table_to_markdown(text, profile.tables)
            return ProcessedBlock(
                type="table",
                text=table_md,
                page_number=block.page_number,
                original_text=original_text,
            )

        # Image blocks
        if block.type == "image":
            description = self._generate_image_description(block)
            return ProcessedBlock(
                type="image",
                text=description,
                page_number=block.page_number,
                asset_ids=[block.raw.get("asset_id", "")] if block.raw.get("asset_id") else [],
                original_text=original_text,
            )

        # Formula blocks
        if block.type == "formula":
            return ProcessedBlock(
                type="formula",
                text=text,
                page_number=block.page_number,
                original_text=original_text,
            )

        # Default: paragraph
        paragraph_md = self._convert_paragraph_to_markdown(text, block.style)
        return ProcessedBlock(
            type="paragraph",
            text=paragraph_md,
            page_number=block.page_number,
            original_text=original_text,
        )

    # ─── Table Conversion ─────────────────────────────────────────────

    def _convert_table_to_markdown(self, text: str, table_config: TableConfig) -> str:
        """Convert a table block to Markdown format.

        If the table is already in Markdown format, validate and return.
        If it's complex (merged cells, nested), convert to text description.

        Args:
            text: Table text content
            table_config: Table configuration

        Returns:
            Markdown table or text description
        """
        # If already looks like a Markdown table, validate it
        if "|" in text and "\n" in text:
            lines = text.strip().split("\n")
            if all("|" in line for line in lines):
                # Ensure separator row exists
                if len(lines) >= 2:
                    if re.match(r"^\|[\s\-:|]+\|$", lines[1]):
                        return text
                    # Insert separator after header
                    header_cells = [c.strip() for c in lines[0].split("|") if c.strip()]
                    separator = "| " + " | ".join(["---"] * len(header_cells)) + " |"
                    return lines[0] + "\n" + separator + "\n" + "\n".join(lines[1:])

        # Try to parse as structured table data
        if self._is_complex_table(text):
            return self._table_to_text_description(text, table_config)

        # Simple text that might be tab-separated
        return self._text_to_markdown_table(text)

    def _is_complex_table(self, text: str) -> bool:
        """Check if a table has complex structure (merged cells, nested).

        Args:
            text: Table text

        Returns:
            True if the table is complex
        """
        # Heuristic: if rows have very different column counts, it's complex
        lines = text.strip().split("\n")
        if not lines:
            return False

        col_counts = []
        for line in lines:
            if "|" in line:
                cols = len([c for c in line.split("|") if c.strip()])
                col_counts.append(cols)
            elif "\t" in line:
                cols = len(line.split("\t"))
                col_counts.append(cols)

        if not col_counts:
            return False

        # If column counts vary significantly, it's complex
        if max(col_counts) - min(col_counts) > 1:
            return True

        return False

    def _table_to_text_description(self, text: str, table_config: TableConfig) -> str:
        """Convert a complex table to text description.

        Args:
            text: Table text
            table_config: Table configuration

        Returns:
            Text description of the table
        """
        lines = text.strip().split("\n")
        description_parts = ["[表格内容]"]
        for i, line in enumerate(lines):
            cleaned = line.strip()
            if cleaned:
                description_parts.append(f"行{i + 1}: {cleaned}")
        return "\n".join(description_parts)

    def _text_to_markdown_table(self, text: str) -> str:
        """Convert tab-separated or space-separated text to Markdown table.

        Args:
            text: Text content

        Returns:
            Markdown table string
        """
        lines = text.strip().split("\n")
        if not lines:
            return text

        # Try tab-separated
        rows = []
        for line in lines:
            if "\t" in line:
                cells = [c.strip() for c in line.split("\t")]
            else:
                cells = [line.strip()]
            rows.append(cells)

        if not rows or len(rows[0]) <= 1:
            return text

        # Normalize column count
        max_cols = max(len(row) for row in rows)
        for row in rows:
            while len(row) < max_cols:
                row.append("")

        # Build Markdown table
        header = "| " + " | ".join(rows[0]) + " |"
        separator = "| " + " | ".join(["---"] * max_cols) + " |"
        body_lines = ["| " + " | ".join(row) + " |" for row in rows[1:]]

        return "\n".join([header, separator] + body_lines)

    # ─── Row-level Table Chunking ─────────────────────────────────────

    def split_table_by_rows(self, table_text: str) -> list[str]:
        """Split a large Markdown table into row-level chunks.

        Each chunk contains the header + separator + one data row.
        Used when Profile.tables.row_level_chunking is enabled.

        Args:
            table_text: Markdown table text

        Returns:
            List of table chunks (each with header + one row)
        """
        lines = table_text.strip().split("\n")
        if len(lines) < 3:
            return [table_text]

        header = lines[0]
        separator = lines[1]
        data_rows = lines[2:]

        chunks = []
        for row in data_rows:
            if row.strip():
                chunk = f"{header}\n{separator}\n{row}"
                chunks.append(chunk)

        return chunks if chunks else [table_text]

    # ─── Image Description ────────────────────────────────────────────

    def _generate_image_description(self, block: Block) -> str:
        """Generate a text description for an image block.

        If LLM image description is enabled and an ``image_describer`` callable was
        injected at construction time, that callable is invoked to obtain a description
        (intended to wrap a multimodal LLM call). Failures fall back to the embedded
        block text or a placeholder.

        Args:
            block: Image block

        Returns:
            Text description of the image
        """
        if self._enable_llm_image_description and self._image_describer is not None:
            try:
                description = self._image_describer(block)
                if description:
                    return description
            except Exception as exc:  # noqa: BLE001 - we want best-effort fallback
                logger.warning(
                    "LLM image description failed (page %s): %s",
                    block.page_number,
                    exc,
                )

        # Fallback: use existing description or placeholder
        description = block.text.strip() or block.raw.get("description", "")
        if description:
            # If the existing text already looks like a Markdown image marker, keep it.
            if description.startswith("[图片"):
                return description
            return f"[图片: {description}]"
        return "[图片]"

    # ─── Paragraph Conversion ─────────────────────────────────────────

    def _convert_paragraph_to_markdown(self, text: str, style: dict) -> str:
        """Convert a paragraph to Markdown format preserving inline formatting.

        Preserves:
        - Bold text
        - Italic text
        - Links

        Args:
            text: Paragraph text
            style: Style metadata

        Returns:
            Markdown-formatted paragraph
        """
        # If style indicates bold/italic, wrap accordingly
        if style.get("bold") and not text.startswith("**"):
            text = f"**{text}**"
        elif style.get("italic") and not text.startswith("*"):
            text = f"*{text}*"

        return text

    # ─── Markdown Generation ──────────────────────────────────────────

    def _generate_markdown(
        self, blocks: list[ProcessedBlock], profile: DocumentProfileConfig
    ) -> str:
        """Generate unified Markdown from processed blocks.

        Args:
            blocks: Processed blocks
            profile: Document profile

        Returns:
            Complete Markdown string
        """
        parts: list[str] = []

        for block in blocks:
            if block.is_noise:
                continue

            if block.type == "heading":
                prefix = "#" * block.heading_level
                parts.append(f"{prefix} {block.text}")
                parts.append("")  # Blank line after heading

            elif block.type == "table":
                parts.append(block.text)
                parts.append("")  # Blank line after table

            elif block.type == "formula":
                parts.append(f"${block.text}$")
                parts.append("")

            elif block.type == "image":
                parts.append(block.text)
                parts.append("")

            else:  # paragraph
                parts.append(block.text)
                parts.append("")  # Blank line between paragraphs

        # Clean up multiple blank lines
        markdown = "\n".join(parts)
        markdown = re.sub(r"\n{3,}", "\n\n", markdown)
        return markdown.strip()
