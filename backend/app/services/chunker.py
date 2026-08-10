"""Intelligent Chunker: Profile-driven document chunking with hierarchy support.

Implements:
- Token-based splitting (min_tokens, max_tokens from Profile.chunking)
- Overlap between adjacent chunks (overlap_tokens)
- Heading-level respect (don't split across headings at or above respect_heading_level)
- Parent-child hierarchy (up to 6 levels)
- Chunk metadata (title_chain, source_file, page_number, space_id, permission_ids)
- Protect patterns (formulas, numbers with units not split)
- Token counting (tiktoken cl100k_base)
- Row-level table chunking
"""

import logging
import re
import uuid
from dataclasses import dataclass, field

from app.services.document_processor import ProcessedBlock, ProcessedDocument
from app.services.profile_matcher import ChunkingConfig, DocumentProfileConfig, TableConfig

logger = logging.getLogger(__name__)

# ─── Token Counter ─────────────────────────────────────────────────────

_TIKTOKEN_ENCODING = None


def _get_encoding():
    """Get or create the tiktoken encoding (lazy singleton)."""
    global _TIKTOKEN_ENCODING
    if _TIKTOKEN_ENCODING is None:
        try:
            import tiktoken
            _TIKTOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
        except (ImportError, Exception) as e:
            logger.warning(f"tiktoken not available, using word-based approximation: {e}")
            _TIKTOKEN_ENCODING = None
    return _TIKTOKEN_ENCODING


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken (cl100k_base) or word approximation.

    Args:
        text: Text to count tokens for

    Returns:
        Number of tokens
    """
    encoding = _get_encoding()
    if encoding is not None:
        return len(encoding.encode(text))
    # Fallback: approximate 1 token ≈ 0.75 words for English,
    # for Chinese roughly 1 char ≈ 1 token
    words = text.split()
    # Count CJK characters separately
    cjk_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    non_cjk_words = len(words)
    return cjk_chars + int(non_cjk_words * 1.3)


# ─── Data Structures ───────────────────────────────────────────────────


@dataclass
class Chunk:
    """A document chunk ready for embedding and indexing.

    Attributes:
        id: Unique chunk identifier
        text: Chunk text content
        token_count: Number of tokens in the chunk
        chunk_index: Position index in the document
        page_number: Starting page number
        heading_level: Heading level of the chunk (0 for non-heading content)
        depth: Hierarchy depth (1-6)
        parent_id: Parent chunk ID (for hierarchy)
        title_chain: Concatenated heading titles (e.g., "H1 > H2 > H3")
        source_file: Original filename
        space_id: Space ID the document belongs to
        permission_ids: Permission identifiers inherited from document
        asset_ids: Referenced asset IDs (images, formulas)
        metadata: Additional metadata
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    token_count: int = 0
    chunk_index: int = 0
    page_number: int = 1
    heading_level: int = 0
    depth: int = 1
    parent_id: str | None = None
    title_chain: str = ""
    source_file: str = ""
    space_id: str = ""
    permission_ids: list[str] = field(default_factory=list)
    asset_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class ChunkingContext:
    """Context for the chunking process.

    Attributes:
        source_file: Original filename
        space_id: Space ID
        permission_ids: Permission identifiers
        document_id: Document ID
    """

    source_file: str = ""
    space_id: str = ""
    permission_ids: list[str] = field(default_factory=list)
    document_id: str = ""


# ─── Intelligent Chunker ──────────────────────────────────────────────


class IntelligentChunker:
    """Profile-driven intelligent document chunker.

    Splits processed documents into chunks respecting:
    - Token limits (min_tokens, max_tokens)
    - Heading boundaries (respect_heading_level)
    - Overlap between adjacent chunks
    - Parent-child hierarchy (up to 6 levels)
    - Protected patterns (formulas, numbers with units)
    """

    def __init__(self):
        """Initialize the chunker."""
        self._protect_patterns: list[re.Pattern] = []

    def chunk(
        self,
        processed_doc: ProcessedDocument,
        profile: DocumentProfileConfig,
        context: ChunkingContext | None = None,
    ) -> list[Chunk]:
        """Chunk a processed document using profile configuration.

        Args:
            processed_doc: The processed document
            profile: Document profile with chunking configuration
            context: Additional context (source file, space, permissions)

        Returns:
            List of chunks with metadata
        """
        if not processed_doc.blocks:
            return []

        config = profile.chunking
        table_config = profile.tables
        ctx = context or ChunkingContext()

        # Compile protect patterns
        self._protect_patterns = self._compile_protect_patterns(config)

        # Step 1: Build sections based on heading hierarchy
        sections = self._build_sections(processed_doc.blocks, config)

        # Step 2: Split sections into chunks respecting token limits
        raw_chunks = self._split_sections(sections, config, table_config)

        # Step 3: Apply overlap
        raw_chunks = self._apply_overlap(raw_chunks, config)

        # Step 4: Build parent-child hierarchy
        chunks = self._build_hierarchy(raw_chunks)

        # Step 5: Attach metadata
        self._attach_metadata(chunks, ctx)

        return chunks

    # ─── Section Building ─────────────────────────────────────────────

    def _build_sections(
        self, blocks: list[ProcessedBlock], config: ChunkingConfig
    ) -> list[dict]:
        """Build sections from blocks based on heading hierarchy.

        A section is a group of blocks under a heading. Sections are split
        at headings at or above respect_heading_level.

        Args:
            blocks: Processed blocks
            config: Chunking configuration

        Returns:
            List of section dicts with heading info and content blocks
        """
        sections: list[dict] = []
        current_section: dict = {
            "heading": None,
            "heading_level": 0,
            "blocks": [],
            "title_chain": [],
            "page_number": 1,
        }

        # Track heading stack for title chain
        heading_stack: list[tuple[int, str]] = []  # (level, text)

        for block in blocks:
            if block.type == "heading" and block.heading_level <= config.respect_heading_level:
                # Start a new section at this heading level
                if current_section["blocks"] or current_section["heading"]:
                    sections.append(current_section)

                # Update heading stack
                heading_stack = self._update_heading_stack(
                    heading_stack, block.heading_level, block.text
                )

                current_section = {
                    "heading": block.text,
                    "heading_level": block.heading_level,
                    "blocks": [],
                    "title_chain": [text for _, text in heading_stack],
                    "page_number": block.page_number,
                }
            else:
                current_section["blocks"].append(block)
                if not current_section["blocks"] and not current_section["heading"]:
                    current_section["page_number"] = block.page_number

        # Don't forget the last section
        if current_section["blocks"] or current_section["heading"]:
            sections.append(current_section)

        return sections

    def _update_heading_stack(
        self, stack: list[tuple[int, str]], level: int, text: str
    ) -> list[tuple[int, str]]:
        """Update the heading stack when a new heading is encountered.

        Pops headings at the same or lower level, then pushes the new one.

        Args:
            stack: Current heading stack
            level: New heading level
            text: New heading text

        Returns:
            Updated heading stack
        """
        # Remove headings at same or deeper level
        new_stack = [(l, t) for l, t in stack if l < level]
        new_stack.append((level, text))
        return new_stack

    # ─── Section Splitting ────────────────────────────────────────────

    def _split_sections(
        self,
        sections: list[dict],
        config: ChunkingConfig,
        table_config: TableConfig,
    ) -> list[Chunk]:
        """Split sections into chunks respecting token limits.

        Args:
            sections: Document sections
            config: Chunking configuration
            table_config: Table configuration

        Returns:
            List of chunks (without overlap applied yet)
        """
        chunks: list[Chunk] = []
        chunk_index = 0

        # 预处理: 合并 "只有 heading 没有 blocks" 的 section 到下一个 section,
        # 避免短文档的 H1 被独立切成无意义的小 chunk。
        # 例: "# Wikforge 是什么\n正文..." 会先被切成两个 section, 这里再合回去。
        sections = self._merge_heading_only_sections(sections)

        for section in sections:
            title_chain = " > ".join(section["title_chain"]) if section["title_chain"] else ""

            # Collect text segments from blocks
            segments = self._blocks_to_segments(section["blocks"], table_config)

            # If section heading exists, include it
            if section["heading"]:
                heading_prefix = "#" * section["heading_level"] + " " + section["heading"]
                segments.insert(0, {
                    "text": heading_prefix,
                    "page_number": section["page_number"],
                    "asset_ids": [],
                    "is_protected": False,
                })

            if not segments:
                continue

            # Split segments into chunks by token count
            section_chunks = self._split_segments_by_tokens(
                segments, config, title_chain, section["page_number"]
            )

            for chunk in section_chunks:
                chunk.chunk_index = chunk_index
                chunk.heading_level = section["heading_level"]
                chunks.append(chunk)
                chunk_index += 1

        return chunks

    def _merge_heading_only_sections(self, sections: list[dict]) -> list[dict]:
        """合并 "只有 heading 没有正文 blocks" 的 section 到下一个 section。

        chunker 之前的 ``_split_into_sections`` 在遇到 ``respect_heading_level``
        以下的标题时会强制切 section, 这导致 ``# H1\\n正文`` 这样的常见结构被
        切成 ``[只有 H1, 正文]`` 两个 section, 进而被切成 2 个独立 chunk
        (其中一个只是标题行, 严重碎片化)。

        本方法把没有 blocks 的 heading-only section 的 heading 作为下一个
        section 的 title_chain 前缀, 让它们合成一个有意义的 chunk。

        如果最后一个 section 是 heading-only, 保持原样 emit
        (确实是个空标题, chunker 后续会自然丢弃或合并到上一个)。
        """
        if not sections:
            return sections

        merged: list[dict] = []
        pending_heading: dict | None = None

        for section in sections:
            has_content = bool(section.get("blocks"))
            if not has_content and section.get("heading"):
                # 只有 heading: 攒着等下一个 section 合并
                pending_heading = section
                continue

            if pending_heading and has_content:
                # 把上一个 heading-only section 的标题挂到当前 section 头部
                # 通过 title_chain 体现层级 (供前端 / RAG 上下文显示)
                pending_chain = pending_heading.get("title_chain", [])
                section_chain = section.get("title_chain", [])
                # 如果当前 section 已经包含了 pending heading (因为它们在同一栈中),
                # 不重复添加
                if pending_chain and (
                    not section_chain or section_chain[: len(pending_chain)] != pending_chain
                ):
                    section = {**section, "title_chain": pending_chain + section_chain}
                pending_heading = None

            merged.append(section)

        # 文档末尾的孤立 heading: 保留为独立 section, chunker 后续 min_tokens
        # 合并逻辑会试图把它合到上一个 chunk
        if pending_heading is not None:
            merged.append(pending_heading)

        return merged

    def _blocks_to_segments(
        self, blocks: list[ProcessedBlock], table_config: TableConfig
    ) -> list[dict]:
        """Convert blocks to text segments for chunking.

        Handles row-level table chunking if enabled.

        Args:
            blocks: Processed blocks
            table_config: Table configuration

        Returns:
            List of segment dicts with text, page_number, asset_ids
        """
        from app.services.document_processor import DocumentProcessor

        segments: list[dict] = []
        processor = DocumentProcessor()

        for block in blocks:
            if block.type == "table" and table_config.row_level_chunking:
                # Split table by rows
                row_chunks = processor.split_table_by_rows(block.text)
                for row_chunk in row_chunks:
                    segments.append({
                        "text": row_chunk,
                        "page_number": block.page_number,
                        "asset_ids": block.asset_ids,
                        "is_protected": True,  # Table rows are atomic
                    })
            else:
                segments.append({
                    "text": block.text,
                    "page_number": block.page_number,
                    "asset_ids": block.asset_ids,
                    "is_protected": block.type in ("formula", "table"),
                })

        return segments

    def _split_segments_by_tokens(
        self,
        segments: list[dict],
        config: ChunkingConfig,
        title_chain: str,
        default_page: int,
    ) -> list[Chunk]:
        """Split segments into chunks respecting token limits.

        Args:
            segments: Text segments
            config: Chunking configuration
            title_chain: Title chain for metadata
            default_page: Default page number

        Returns:
            List of chunks
        """
        chunks: list[Chunk] = []
        current_texts: list[str] = []
        current_tokens = 0
        current_page = default_page
        current_assets: list[str] = []

        for segment in segments:
            seg_text = segment["text"]
            seg_tokens = count_tokens(seg_text)
            seg_page = segment["page_number"]
            seg_assets = segment.get("asset_ids", [])
            is_protected = segment.get("is_protected", False)

            # If this single segment exceeds max_tokens and is not protected,
            # we need to split it further
            if seg_tokens > config.max_tokens and not is_protected:
                # Flush current buffer first
                if current_texts:
                    chunk_text = "\n\n".join(current_texts)
                    if count_tokens(chunk_text) >= config.min_tokens or not chunks:
                        chunks.append(Chunk(
                            text=chunk_text,
                            token_count=count_tokens(chunk_text),
                            page_number=current_page,
                            title_chain=title_chain,
                            asset_ids=current_assets.copy(),
                        ))
                    current_texts = []
                    current_tokens = 0
                    current_assets = []

                # Split the large segment
                sub_chunks = self._split_large_text(
                    seg_text, config, title_chain, seg_page
                )
                chunks.extend(sub_chunks)
                continue

            # Check if adding this segment would exceed max_tokens
            potential_tokens = current_tokens + seg_tokens + (2 if current_texts else 0)
            if potential_tokens > config.max_tokens and current_texts:
                # Flush current buffer
                chunk_text = "\n\n".join(current_texts)
                chunks.append(Chunk(
                    text=chunk_text,
                    token_count=count_tokens(chunk_text),
                    page_number=current_page,
                    title_chain=title_chain,
                    asset_ids=current_assets.copy(),
                ))
                current_texts = []
                current_tokens = 0
                current_assets = []

            # Add segment to current buffer
            if not current_texts:
                current_page = seg_page
            current_texts.append(seg_text)
            current_tokens += seg_tokens + (2 if len(current_texts) > 1 else 0)
            current_assets.extend(seg_assets)

        # Flush remaining buffer
        if current_texts:
            chunk_text = "\n\n".join(current_texts)
            token_count = count_tokens(chunk_text)
            # If too small and we have previous chunks, merge with last
            if token_count < config.min_tokens and chunks:
                last_chunk = chunks[-1]
                merged_text = last_chunk.text + "\n\n" + chunk_text
                merged_tokens = count_tokens(merged_text)
                if merged_tokens <= config.max_tokens:
                    last_chunk.text = merged_text
                    last_chunk.token_count = merged_tokens
                    last_chunk.asset_ids.extend(current_assets)
                else:
                    chunks.append(Chunk(
                        text=chunk_text,
                        token_count=token_count,
                        page_number=current_page,
                        title_chain=title_chain,
                        asset_ids=current_assets.copy(),
                    ))
            else:
                chunks.append(Chunk(
                    text=chunk_text,
                    token_count=token_count,
                    page_number=current_page,
                    title_chain=title_chain,
                    asset_ids=current_assets.copy(),
                ))

        return chunks

    def _split_large_text(
        self,
        text: str,
        config: ChunkingConfig,
        title_chain: str,
        page_number: int,
    ) -> list[Chunk]:
        """Split a large text block into multiple chunks respecting protect patterns.

        Args:
            text: Large text to split
            config: Chunking configuration
            title_chain: Title chain for metadata
            page_number: Page number

        Returns:
            List of chunks
        """
        # Find protected spans
        protected_spans = self._find_protected_spans(text)

        # Split by sentences/paragraphs, respecting protected spans
        split_points = self._find_split_points(text, protected_spans)

        chunks: list[Chunk] = []
        current_start = 0
        current_texts: list[str] = []
        current_tokens = 0

        for point in split_points:
            segment = text[current_start:point].strip()
            if not segment:
                current_start = point
                continue

            seg_tokens = count_tokens(segment)

            if current_tokens + seg_tokens > config.max_tokens and current_texts:
                chunk_text = " ".join(current_texts)
                chunks.append(Chunk(
                    text=chunk_text,
                    token_count=count_tokens(chunk_text),
                    page_number=page_number,
                    title_chain=title_chain,
                ))
                current_texts = []
                current_tokens = 0

            current_texts.append(segment)
            current_tokens += seg_tokens
            current_start = point

        # Handle remaining text
        remaining = text[current_start:].strip()
        if remaining:
            current_texts.append(remaining)

        if current_texts:
            chunk_text = " ".join(current_texts)
            chunks.append(Chunk(
                text=chunk_text,
                token_count=count_tokens(chunk_text),
                page_number=page_number,
                title_chain=title_chain,
            ))

        return chunks

    def _find_protected_spans(self, text: str) -> list[tuple[int, int]]:
        """Find spans in text that should not be split.

        Args:
            text: Text to analyze

        Returns:
            List of (start, end) tuples for protected spans
        """
        spans: list[tuple[int, int]] = []
        for pattern in self._protect_patterns:
            for match in pattern.finditer(text):
                spans.append((match.start(), match.end()))
        return sorted(spans)

    def _find_split_points(
        self, text: str, protected_spans: list[tuple[int, int]]
    ) -> list[int]:
        """Find valid split points in text, avoiding protected spans.

        Prefers splitting at:
        1. Paragraph boundaries (double newline)
        2. Sentence boundaries (. ! ? followed by space)
        3. Clause boundaries (, ; :)

        Args:
            text: Text to find split points in
            protected_spans: Spans that must not be split

        Returns:
            Sorted list of valid split point indices
        """
        # Find all potential split points
        points: list[int] = []

        # Paragraph boundaries
        for match in re.finditer(r"\n\n", text):
            points.append(match.end())

        # Sentence boundaries
        for match in re.finditer(r"[.!?。！？]\s+", text):
            points.append(match.end())

        # Clause boundaries (lower priority)
        for match in re.finditer(r"[,;:，；：]\s*", text):
            points.append(match.end())

        # Filter out points inside protected spans
        valid_points = []
        for point in sorted(set(points)):
            is_protected = any(start <= point <= end for start, end in protected_spans)
            if not is_protected:
                valid_points.append(point)

        # If no valid points found, split at word boundaries
        if not valid_points:
            for match in re.finditer(r"\s+", text):
                point = match.end()
                is_protected = any(start <= point <= end for start, end in protected_spans)
                if not is_protected:
                    valid_points.append(point)

        return sorted(set(valid_points))

    # ─── Overlap ──────────────────────────────────────────────────────

    def _apply_overlap(self, chunks: list[Chunk], config: ChunkingConfig) -> list[Chunk]:
        """Apply overlap between adjacent chunks.

        Prepends the last N tokens from the previous chunk to the current chunk.

        Args:
            chunks: List of chunks
            config: Chunking configuration

        Returns:
            Chunks with overlap applied
        """
        if config.overlap_tokens <= 0 or len(chunks) <= 1:
            return chunks

        for i in range(1, len(chunks)):
            prev_text = chunks[i - 1].text
            overlap_text = self._extract_tail_tokens(prev_text, config.overlap_tokens)
            if overlap_text:
                chunks[i].text = overlap_text + "\n\n" + chunks[i].text
                chunks[i].token_count = count_tokens(chunks[i].text)

        return chunks

    def _extract_tail_tokens(self, text: str, max_tokens: int) -> str:
        """Extract the last N tokens from text.

        Tries to break at sentence or word boundaries.

        Args:
            text: Source text
            max_tokens: Maximum tokens to extract

        Returns:
            Tail text with approximately max_tokens tokens
        """
        encoding = _get_encoding()
        if encoding is not None:
            tokens = encoding.encode(text)
            if len(tokens) <= max_tokens:
                return ""
            tail_tokens = tokens[-max_tokens:]
            return encoding.decode(tail_tokens).strip()

        # Fallback: word-based approximation
        words = text.split()
        if len(words) <= max_tokens:
            return ""
        tail_words = words[-max_tokens:]
        return " ".join(tail_words)

    # ─── Hierarchy ────────────────────────────────────────────────────

    def _build_hierarchy(self, chunks: list[Chunk]) -> list[Chunk]:
        """Build parent-child hierarchy for chunks (up to 6 levels).

        Chunks under the same heading section share a parent.
        The hierarchy is based on heading levels.

        Args:
            chunks: Flat list of chunks

        Returns:
            Chunks with parent_id and depth set
        """
        if not chunks:
            return chunks

        # Track heading chunks as potential parents
        # heading_level -> most recent chunk at that level
        level_parents: dict[int, str] = {}

        for chunk in chunks:
            level = chunk.heading_level

            if level > 0:
                # This chunk starts a new section at this level
                # Its parent is the most recent chunk at a higher level
                parent_id = None
                for parent_level in range(1, level):
                    if parent_level in level_parents:
                        parent_id = level_parents[parent_level]

                chunk.parent_id = parent_id
                chunk.depth = min(level, 6)
                level_parents[level] = chunk.id

                # Clear deeper levels
                for deeper in list(level_parents.keys()):
                    if deeper > level:
                        del level_parents[deeper]
            else:
                # Content chunk: parent is the most recent heading chunk
                parent_id = None
                for check_level in sorted(level_parents.keys(), reverse=True):
                    parent_id = level_parents[check_level]
                    break

                chunk.parent_id = parent_id
                chunk.depth = min(max(level_parents.keys(), default=0) + 1, 6) if level_parents else 1

        return chunks

    # ─── Metadata ─────────────────────────────────────────────────────

    def _attach_metadata(self, chunks: list[Chunk], context: ChunkingContext) -> None:
        """Attach metadata to all chunks.

        Args:
            chunks: List of chunks to update
            context: Chunking context with source info
        """
        for chunk in chunks:
            chunk.source_file = context.source_file
            chunk.space_id = context.space_id
            chunk.permission_ids = context.permission_ids.copy()

    # ─── Protect Patterns ─────────────────────────────────────────────

    def _compile_protect_patterns(self, config: ChunkingConfig) -> list[re.Pattern]:
        """Compile protect patterns from chunking config.

        These patterns identify text that should not be split across chunk boundaries.

        Args:
            config: Chunking configuration

        Returns:
            List of compiled regex patterns
        """
        patterns: list[re.Pattern] = []

        # Always protect common numeric patterns
        default_patterns = [
            r"\d+[.,]\d+\s*(?:mm|cm|m|km|kg|g|mg|t|MPa|kPa|Pa|°C|°F|%|‰)",  # numbers with units
            r"[±＋]\s*\d+[.,]?\d*\s*(?:mm|cm|m|km|kg|g|mg|t|MPa|kPa|Pa|°C|°F|%)",  # ± values
            r"\d+(?:[.,]\d+)?(?:\s*[~～\-]\s*\d+(?:[.,]\d+)?)\s*(?:mm|cm|m|°|%)",  # ranges
            r"△\s*=\s*[^,\n]+",  # formulas with delta
            r"\d+[.,]\d+\s*[A-Za-z/]+",  # generic number+unit
        ]

        for pattern_str in default_patterns + config.protect_patterns:
            try:
                patterns.append(re.compile(pattern_str))
            except re.error:
                logger.warning(f"Invalid protect pattern: {pattern_str}")

        return patterns
