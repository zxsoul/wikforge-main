"""Profile Matcher service for automatic document profile matching.

Implements:
- Data classes: MatchRules, HeadingRule, BoilerplateConfig, TableConfig, ChunkingConfig, DocumentProfileConfig
- DocumentFeatures: extracted features from parsed documents
- ProfileMatcher: feature extraction + profile matching logic

Design:
- ProfileMatcher.match(parsed_doc, filename) → DocumentProfileConfig
- Feature extraction from first N pages: filename patterns, numbering patterns,
  header/footer repetition, table density
- Priority matching: highest priority wins, same priority → most recently updated
- Default fallback: generic-text profile when nothing matches
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

from app.services.parsers.base import Block, ParsedDocument

logger = logging.getLogger(__name__)

# Number of pages to sample for feature extraction
FEATURE_EXTRACTION_PAGES = 5


@dataclass
class MatchRules:
    """Rules for matching a document to a profile."""

    filename_regex: list[str] = field(default_factory=list)
    content_regex: list[str] = field(default_factory=list)
    min_content_match_count: int = 1


@dataclass
class HeadingRule:
    """Rule for identifying headings in document content."""

    pattern: str
    level: int
    strip_pattern: bool = False


@dataclass
class BoilerplateConfig:
    """Configuration for boilerplate/noise detection and removal."""

    detection_mode: str = "statistical"  # "statistical" | "manual" | "both"
    statistical_threshold: float = 0.5
    manual_patterns: list[str] = field(default_factory=list)


@dataclass
class TableConfig:
    """Configuration for table processing."""

    cross_page_merge: bool = True
    row_level_chunking: bool = False
    collapse_merged_cells: str = "describe"  # "describe" | "repeat"


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""

    min_tokens: int = 256
    max_tokens: int = 800
    overlap_tokens: int = 80
    respect_heading_level: int = 1
    protect_patterns: list[str] = field(default_factory=list)


@dataclass
class DocumentProfileConfig:
    """In-memory representation of a DocumentProfile for matching and processing.

    This is the service-layer data class, distinct from the SQLAlchemy ORM model.
    """

    id: str
    name: str
    description: str = ""
    priority: int = 0
    enabled: bool = True
    match_rules: MatchRules = field(default_factory=MatchRules)
    heading_rules: list[HeadingRule] = field(default_factory=list)
    boilerplate: BoilerplateConfig = field(default_factory=BoilerplateConfig)
    tables: TableConfig = field(default_factory=TableConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    domain_dictionary_id: str | None = None
    version: int = 1
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class DocumentFeatures:
    """Features extracted from a parsed document for profile matching."""

    filename: str = ""
    # Content from first N pages joined
    sample_text: str = ""
    # Detected numbering patterns
    numbering_patterns: list[str] = field(default_factory=list)
    # Header/footer repetition ratio (0-1)
    header_footer_repetition: float = 0.0
    # Table density (tables / total blocks)
    table_density: float = 0.0
    # Total page count
    page_count: int = 0
    # Whether document appears to be scanned (very low text per page)
    appears_scanned: bool = False
    # Average text length per page
    avg_text_per_page: float = 0.0


def profile_from_dict(data: dict) -> DocumentProfileConfig:
    """Convert a dictionary (e.g., from DB JSONB fields) to a DocumentProfileConfig.

    Args:
        data: Dictionary with profile fields (as stored in DB or from JSON import)

    Returns:
        DocumentProfileConfig instance
    """
    match_rules_data = data.get("match_rules", {})
    match_rules = MatchRules(
        filename_regex=match_rules_data.get("filename_regex", []),
        content_regex=match_rules_data.get("content_regex", []),
        min_content_match_count=match_rules_data.get("min_content_match_count", 1),
    )

    heading_rules_data = data.get("heading_rules", [])
    heading_rules = [
        HeadingRule(
            pattern=rule.get("pattern", ""),
            level=rule.get("level", 1),
            strip_pattern=rule.get("strip_pattern", False),
        )
        for rule in heading_rules_data
    ]

    boilerplate_data = data.get("boilerplate", {})
    boilerplate = BoilerplateConfig(
        detection_mode=boilerplate_data.get("detection_mode", "statistical"),
        statistical_threshold=boilerplate_data.get("statistical_threshold", 0.5),
        manual_patterns=boilerplate_data.get("manual_patterns", []),
    )

    tables_data = data.get("tables", {})
    tables = TableConfig(
        cross_page_merge=tables_data.get("cross_page_merge", True),
        row_level_chunking=tables_data.get("row_level_chunking", False),
        collapse_merged_cells=tables_data.get("collapse_merged_cells", "describe"),
    )

    chunking_data = data.get("chunking", {})
    chunking = ChunkingConfig(
        min_tokens=chunking_data.get("min_tokens", 256),
        max_tokens=chunking_data.get("max_tokens", 800),
        overlap_tokens=chunking_data.get("overlap_tokens", 80),
        respect_heading_level=chunking_data.get("respect_heading_level", 1),
        protect_patterns=chunking_data.get("protect_patterns", []),
    )

    return DocumentProfileConfig(
        id=str(data.get("id", "")),
        name=data.get("name", ""),
        description=data.get("description", ""),
        priority=data.get("priority", 0),
        enabled=data.get("enabled", True),
        match_rules=match_rules,
        heading_rules=heading_rules,
        boilerplate=boilerplate,
        tables=tables,
        chunking=chunking,
        domain_dictionary_id=data.get("domain_dictionary_id"),
        version=data.get("version", 1),
        created_at=data.get("created_at"),
        updated_at=data.get("updated_at"),
    )


def profile_to_dict(profile: DocumentProfileConfig) -> dict:
    """Convert a DocumentProfileConfig to a serializable dictionary.

    Args:
        profile: DocumentProfileConfig instance

    Returns:
        Dictionary suitable for JSON serialization or DB storage
    """
    return {
        "id": profile.id,
        "name": profile.name,
        "description": profile.description,
        "priority": profile.priority,
        "enabled": profile.enabled,
        "match_rules": {
            "filename_regex": profile.match_rules.filename_regex,
            "content_regex": profile.match_rules.content_regex,
            "min_content_match_count": profile.match_rules.min_content_match_count,
        },
        "heading_rules": [
            {
                "pattern": rule.pattern,
                "level": rule.level,
                "strip_pattern": rule.strip_pattern,
            }
            for rule in profile.heading_rules
        ],
        "boilerplate": {
            "detection_mode": profile.boilerplate.detection_mode,
            "statistical_threshold": profile.boilerplate.statistical_threshold,
            "manual_patterns": profile.boilerplate.manual_patterns,
        },
        "tables": {
            "cross_page_merge": profile.tables.cross_page_merge,
            "row_level_chunking": profile.tables.row_level_chunking,
            "collapse_merged_cells": profile.tables.collapse_merged_cells,
        },
        "chunking": {
            "min_tokens": profile.chunking.min_tokens,
            "max_tokens": profile.chunking.max_tokens,
            "overlap_tokens": profile.chunking.overlap_tokens,
            "respect_heading_level": profile.chunking.respect_heading_level,
            "protect_patterns": profile.chunking.protect_patterns,
        },
        "domain_dictionary_id": profile.domain_dictionary_id,
        "version": profile.version,
    }


class ProfileMatcher:
    """Matches parsed documents to the most appropriate DocumentProfile.

    The matcher:
    1. Extracts features from the parsed document (first N pages)
    2. Iterates over all enabled profiles, testing match rules
    3. Returns the highest-priority matching profile
    4. Falls back to 'generic-text' if nothing matches
    """

    def __init__(self, profiles: list[DocumentProfileConfig] | None = None):
        """Initialize with a list of profiles.

        Args:
            profiles: List of available profiles. If None, must be set later.
        """
        self._profiles: list[DocumentProfileConfig] = profiles or []

    @property
    def profiles(self) -> list[DocumentProfileConfig]:
        return self._profiles

    @profiles.setter
    def profiles(self, value: list[DocumentProfileConfig]) -> None:
        self._profiles = value

    def extract_features(self, parsed_doc: ParsedDocument, filename: str = "") -> DocumentFeatures:
        """Extract features from a parsed document for profile matching.

        Analyzes the first N pages of the document to extract:
        - Filename
        - Sample text content
        - Numbering patterns (Chinese, Arabic, circled numbers, etc.)
        - Header/footer repetition ratio
        - Table density
        - Whether document appears scanned

        Args:
            parsed_doc: The parsed document intermediate representation
            filename: Original filename of the document

        Returns:
            DocumentFeatures with extracted characteristics
        """
        blocks = parsed_doc.blocks
        if not blocks:
            return DocumentFeatures(filename=filename)

        # Determine page range for sampling
        max_page = max(b.page_number for b in blocks) if blocks else 1
        sample_page_limit = min(FEATURE_EXTRACTION_PAGES, max_page)

        # Filter blocks from first N pages
        sample_blocks = [b for b in blocks if b.page_number <= sample_page_limit]
        all_text = "\n".join(b.text for b in sample_blocks if b.text.strip())

        # Detect numbering patterns
        numbering_patterns = self._detect_numbering_patterns(all_text)

        # Calculate header/footer repetition
        header_footer_repetition = self._calculate_header_footer_repetition(blocks, max_page)

        # Calculate table density
        table_blocks = [b for b in sample_blocks if b.type == "table"]
        table_density = len(table_blocks) / len(sample_blocks) if sample_blocks else 0.0

        # Detect if document appears scanned (very little text per page)
        text_lengths_per_page: dict[int, int] = {}
        for b in blocks:
            text_lengths_per_page.setdefault(b.page_number, 0)
            text_lengths_per_page[b.page_number] += len(b.text.strip())

        avg_text_per_page = (
            sum(text_lengths_per_page.values()) / len(text_lengths_per_page)
            if text_lengths_per_page
            else 0.0
        )
        # A scanned PDF typically has very little extractable text
        appears_scanned = avg_text_per_page < 50 and max_page > 0

        return DocumentFeatures(
            filename=filename,
            sample_text=all_text,
            numbering_patterns=numbering_patterns,
            header_footer_repetition=header_footer_repetition,
            table_density=table_density,
            page_count=max_page,
            appears_scanned=appears_scanned,
            avg_text_per_page=avg_text_per_page,
        )

    def match(self, parsed_doc: ParsedDocument, filename: str = "") -> DocumentProfileConfig:
        """Match a parsed document to the best-fitting profile.

        Process:
        1. Extract features from the document
        2. Test each enabled profile's match rules against features
        3. Among matching profiles, select by highest priority
        4. If priority is tied, select the most recently updated
        5. If nothing matches, return the default 'generic-text' profile

        Args:
            parsed_doc: The parsed document intermediate representation
            filename: Original filename of the document

        Returns:
            The best matching DocumentProfileConfig
        """
        features = self.extract_features(parsed_doc, filename)
        return self._match_with_features(features)

    def _match_with_features(self, features: DocumentFeatures) -> DocumentProfileConfig:
        """Internal matching logic using pre-extracted features."""
        matched_profiles: list[DocumentProfileConfig] = []

        for profile in self._profiles:
            if not profile.enabled:
                continue
            if self._profile_matches(profile, features):
                matched_profiles.append(profile)

        if not matched_profiles:
            return self._get_default_profile()

        # Sort by priority (descending), then by updated_at (most recent first)
        matched_profiles.sort(
            key=lambda p: (
                p.priority,
                p.updated_at.timestamp() if p.updated_at else 0,
            ),
            reverse=True,
        )

        selected = matched_profiles[0]
        logger.info(
            f"Profile matched: '{selected.name}' (priority={selected.priority}) "
            f"for file '{features.filename}'"
        )
        return selected

    def _profile_matches(self, profile: DocumentProfileConfig, features: DocumentFeatures) -> bool:
        """Test if a profile's match rules are satisfied by the document features.

        A profile matches if:
        - Any filename_regex matches the filename, OR
        - At least min_content_match_count content_regex patterns match the sample text

        If both filename_regex and content_regex are empty, the profile does NOT match
        (it's the generic fallback and should only be used as default).

        Args:
            profile: Profile to test
            features: Extracted document features

        Returns:
            True if the profile matches
        """
        rules = profile.match_rules

        # If no rules defined, this profile doesn't actively match anything
        # (it's likely the generic-text fallback)
        if not rules.filename_regex and not rules.content_regex:
            return False

        # Check filename patterns
        if rules.filename_regex and features.filename:
            for pattern in rules.filename_regex:
                try:
                    if re.search(pattern, features.filename, re.IGNORECASE):
                        return True
                except re.error:
                    logger.warning(f"Invalid filename regex in profile '{profile.name}': {pattern}")

        # Check content patterns
        if rules.content_regex and features.sample_text:
            match_count = 0
            for pattern in rules.content_regex:
                try:
                    if re.search(pattern, features.sample_text, re.MULTILINE):
                        match_count += 1
                except re.error:
                    logger.warning(f"Invalid content regex in profile '{profile.name}': {pattern}")

            if match_count >= rules.min_content_match_count:
                return True

        return False

    def _get_default_profile(self) -> DocumentProfileConfig:
        """Return the default 'generic-text' profile.

        Looks for a profile named 'generic-text' in the loaded profiles.
        If not found, returns a hardcoded minimal default.
        """
        for profile in self._profiles:
            if profile.name == "generic-text":
                return profile

        # Hardcoded fallback if generic-text is not in the profile list
        logger.warning("No 'generic-text' profile found, using hardcoded default")
        return DocumentProfileConfig(
            id="default",
            name="generic-text",
            description="通用文本文档 - 默认兜底 Profile",
            priority=0,
            enabled=True,
            match_rules=MatchRules(),
            heading_rules=[
                HeadingRule(pattern=r"^#{1,6}\s+", level=0, strip_pattern=False),
            ],
            boilerplate=BoilerplateConfig(),
            tables=TableConfig(),
            chunking=ChunkingConfig(),
        )

    def _detect_numbering_patterns(self, text: str) -> list[str]:
        """Detect numbering patterns in the sample text.

        Looks for:
        - Chinese numbering: 一、二、三 / (一)(二)(三)
        - Arabic numbering: 1. 2. 3. / (1)(2)(3)
        - Circled numbers: ①②③
        - Chapter markers: 第一章、第二章

        Args:
            text: Sample text to analyze

        Returns:
            List of detected pattern names
        """
        patterns_found = []

        pattern_checks = [
            ("chinese_chapter", r"第[一二三四五六七八九十百]+[章节条款]"),
            ("chinese_numbering", r"^[一二三四五六七八九十]+[、．.]"),
            ("chinese_paren_numbering", r"^\([一二三四五六七八九十]+\)"),
            ("arabic_dot_numbering", r"^\d+[、．.]"),
            ("arabic_paren_numbering", r"^\(\d+\)"),
            ("circled_numbering", r"^[①②③④⑤⑥⑦⑧⑨⑩]"),
            ("decimal_section", r"^\d+\.\d+"),
        ]

        for name, pattern in pattern_checks:
            try:
                if re.search(pattern, text, re.MULTILINE):
                    patterns_found.append(name)
            except re.error:
                pass

        return patterns_found

    def _calculate_header_footer_repetition(
        self, blocks: list[Block], max_page: int
    ) -> float:
        """Calculate header/footer repetition ratio.

        Looks at blocks at the top and bottom of each page. If the same text
        appears in the same position across many pages, it's likely a header/footer.

        Args:
            blocks: All blocks in the document
            max_page: Total number of pages

        Returns:
            Repetition ratio (0.0 to 1.0)
        """
        if max_page < 3:
            return 0.0

        # Group blocks by page
        pages: dict[int, list[Block]] = {}
        for block in blocks:
            pages.setdefault(block.page_number, []).append(block)

        # Check first block of each page (potential header)
        first_texts: list[str] = []
        # Check last block of each page (potential footer)
        last_texts: list[str] = []

        for page_num in sorted(pages.keys()):
            page_blocks = pages[page_num]
            if page_blocks:
                first_text = page_blocks[0].text.strip()
                if first_text:
                    first_texts.append(first_text)
                last_text = page_blocks[-1].text.strip()
                if last_text:
                    last_texts.append(last_text)

        # Calculate repetition
        max_repetition = 0.0

        if first_texts:
            counter = Counter(first_texts)
            most_common_count = counter.most_common(1)[0][1]
            max_repetition = max(max_repetition, most_common_count / len(first_texts))

        if last_texts:
            counter = Counter(last_texts)
            most_common_count = counter.most_common(1)[0][1]
            max_repetition = max(max_repetition, most_common_count / len(last_texts))

        return max_repetition
