"""Quality Scorer service for document parsing quality assessment.

Implements multi-dimensional scoring:
- Text retention rate (visible chars after cleaning / original visible chars)
- Heading detection rate (detected headings / expected headings)
- Table completeness (cell fill rate, cross-page merge success)
- Numeric protection rate (sampling key numerics for integrity)
- Boilerplate removal rate (watermark/header removal coverage)

Scoring weights:
  text_retention=0.30, heading_detection=0.25, table_completeness=0.20,
  numeric_protection=0.15, boilerplate_removal=0.10

Also implements review queue enqueue logic (score < 0.7 → auto-enqueue).
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass, field

from app.services.document_processor import ProcessedBlock, ProcessedDocument
from app.services.parsers.base import Block, ParsedDocument
from app.services.profile_matcher import DocumentProfileConfig

logger = logging.getLogger(__name__)

# Scoring weights
WEIGHT_TEXT_RETENTION = 0.30
WEIGHT_HEADING_DETECTION = 0.25
WEIGHT_TABLE_COMPLETENESS = 0.20
WEIGHT_NUMERIC_PROTECTION = 0.15
WEIGHT_BOILERPLATE_REMOVAL = 0.10

# Default review queue threshold
DEFAULT_REVIEW_THRESHOLD = 0.7

# Numeric protection sampling cap. We cap to keep the per-document scoring
# cost bounded on long technical specs (which can carry hundreds of
# numeric tokens); 50 unique values is empirically enough to surface a
# regression in number preservation while remaining cheap to substring-check.
NUMERIC_SAMPLE_SIZE = 50

# When the numeric_protection score drops below this threshold the
# scorer records a human-readable issue describing what went wrong.
# Aligned with the dimension's strictness: numbers are typically
# expected to round-trip exactly, so even a 10% loss is worth flagging.
NUMERIC_PROTECTION_ISSUE_THRESHOLD = 0.9

# When recording a numeric_protection issue we include up to this many
# concrete missing values to help reviewers locate the regression
# without overwhelming the issue message.
NUMERIC_ISSUE_EXAMPLE_LIMIT = 3

# ─── Boilerplate Removal Constants ──────────────────────────────────────
#
# Aligned with the statistical detector in ``document_processor.py``
# (see ``_detect_statistical_noise``): a piece of text is considered
# boilerplate when it appears at the same position on at least 50% of
# pages. Mirroring the constant here keeps the scorer's "expected noise"
# estimate consistent with what the cleaner actually targets.
BOILERPLATE_DETECTION_THRESHOLD = 0.5

# When the boilerplate_removal score drops below this threshold the
# scorer records a human-readable issue. This is intentionally less
# strict than NUMERIC_PROTECTION_ISSUE_THRESHOLD because partial removal
# of repeated headers/footers is still useful — only a clear shortfall
# (< 70%) is worth flagging for human review.
BOILERPLATE_ISSUE_THRESHOLD = 0.7

# Statistical first/last-block detection needs a minimum number of pages
# before the "appears on ≥50% of pages" frequency rule produces a
# meaningful signal. With fewer pages the noise estimate is too noisy
# (e.g. a 2-page doc with the same first block on both pages would
# spuriously count as 100% boilerplate), so we skip the dimension.
MIN_PAGES_FOR_DETECTION = 3


@dataclass
class ParseQualityScore:
    """Quality score for a parsed document.

    Canonical schema (per design.md "Quality Scorer + Review Queue"):
      - overall: 加权综合分，取值范围 [0, 1]
      - components: 各维度子分（text_retention / heading_detection /
        table_completeness / numeric_protection / boilerplate_removal 等）
      - issues: 检测到的问题码 / 人类可读描述

    Construction-time guarantees (via ``__post_init__``):
      - overall 自动夹紧到 [0, 1]（防止子分实现 bug 让综合分越界）
      - components 缺省 / 显式传 None 时退化为空 dict
      - issues 缺省 / 显式传 None 时退化为空 list

    Persistence helpers:
      - ``to_dict()``  → 用于 DocumentReview.quality_score(JSONB) 序列化
      - ``from_dict()`` → 反序列化 JSONB 行回到数据类实例（往返一致）
    """

    overall: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # 防御 None：dataclass 的 default_factory 仅在字段未传时生效，
        # 显式传 None 时仍会落到字段上，这里统一兜底成默认空集合。
        if self.components is None:  # type: ignore[truthy-bool]
            self.components = {}
        if self.issues is None:  # type: ignore[truthy-bool]
            self.issues = []

        # overall 必须落在 [0, 1]；同时把非数值或 NaN 也归一化掉，
        # 避免下游 JSONB 序列化或阈值比较时拿到非法值。
        try:
            value = float(self.overall)
        except (TypeError, ValueError):
            value = 0.0
        if value != value:  # NaN check
            value = 0.0
        if value < 0.0:
            value = 0.0
        elif value > 1.0:
            value = 1.0
        self.overall = value

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary.

        Used by the document processing pipeline to persist the score
        into ``DocumentReview.quality_score`` (JSONB column).
        """
        return {
            "overall": round(self.overall, 4),
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "issues": list(self.issues),
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "ParseQualityScore":
        """Reconstruct a ``ParseQualityScore`` from its persisted dict form.

        Symmetric to :meth:`to_dict`. Tolerates missing keys (treats as
        default values) and ``None`` (returns a default-initialised score),
        which matches how the JSONB column may be NULL for unscored docs.
        """
        if not data:
            return cls()
        overall = data.get("overall", 0.0)
        components_raw = data.get("components") or {}
        components = {str(k): float(v) for k, v in components_raw.items()}
        issues_raw = data.get("issues") or []
        issues = [str(item) for item in issues_raw]
        return cls(overall=overall, components=components, issues=issues)


def _visible_char_count(text: str) -> int:
    """Count visible (non-whitespace) characters in text."""
    return len(re.sub(r"\s", "", text))


def _extract_numbers(text: str) -> list[str]:
    """Extract numeric values with units from text.

    Matches patterns like: 0.05mm/m, 0.002D, ±10mm, 55°~65°, 100kg, 3.14
    """
    pattern = r"[±]?\d+(?:\.\d+)?(?:\s*[~～\-]\s*[±]?\d+(?:\.\d+)?)?(?:\s*[a-zA-Z°℃%‰/]+(?:/[a-zA-Z]+)?)*"
    return re.findall(pattern, text)


def _normalize_boilerplate_text(text: str) -> str:
    """Normalise text for boilerplate frequency comparison.

    Repeated headers/footers/watermarks often differ across pages only
    by whitespace or letter case (e.g. ``"Page 1 "`` vs ``" page  1"``).
    Without normalisation those near-duplicates would be tallied as
    distinct strings and slip below the 50% frequency threshold,
    causing the scorer to under-estimate expected noise.

    Steps:
    1. Strip surrounding whitespace.
    2. Collapse internal runs of whitespace to a single space.
    3. Lower-case (case-insensitive comparison).
    """
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip().lower()


def _count_repeated_above_threshold(
    texts: list[str], total_pages: int, threshold: float
) -> int:
    """Sum the per-text occurrences of every text appearing on
    at least ``threshold`` fraction of pages.

    Returns 0 for an empty input. The caller is responsible for
    providing already-normalised text (see :func:`_normalize_boilerplate_text`).
    """
    if not texts or total_pages <= 0:
        return 0
    expected = 0
    for _text, count in Counter(texts).items():
        if count / total_pages >= threshold:
            expected += count
    return expected


class QualityScorer:
    """Multi-dimensional quality scorer for parsed documents.

    Computes quality scores across 5 dimensions and produces
    a weighted overall score.
    """

    def __init__(
        self,
        weight_text_retention: float = WEIGHT_TEXT_RETENTION,
        weight_heading_detection: float = WEIGHT_HEADING_DETECTION,
        weight_table_completeness: float = WEIGHT_TABLE_COMPLETENESS,
        weight_numeric_protection: float = WEIGHT_NUMERIC_PROTECTION,
        weight_boilerplate_removal: float = WEIGHT_BOILERPLATE_REMOVAL,
        review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
    ):
        """Initialize the quality scorer with configurable weights.

        Args:
            weight_text_retention: Weight for text retention score (default 0.30)
            weight_heading_detection: Weight for heading detection score (default 0.25)
            weight_table_completeness: Weight for table completeness score (default 0.20)
            weight_numeric_protection: Weight for numeric protection score (default 0.15)
            weight_boilerplate_removal: Weight for boilerplate removal score (default 0.10)
            review_threshold: Threshold below which documents enter review queue (default 0.7)
        """
        self.weight_text_retention = weight_text_retention
        self.weight_heading_detection = weight_heading_detection
        self.weight_table_completeness = weight_table_completeness
        self.weight_numeric_protection = weight_numeric_protection
        self.weight_boilerplate_removal = weight_boilerplate_removal
        self.review_threshold = review_threshold

    def score(
        self,
        original: ParsedDocument,
        processed: ProcessedDocument,
        profile: DocumentProfileConfig,
    ) -> ParseQualityScore:
        """Compute multi-dimensional quality score.

        Args:
            original: The original parsed document (before cleaning)
            processed: The processed document (after cleaning/conversion)
            profile: The document profile used for processing

        Returns:
            ParseQualityScore with overall score, component scores, and issues
        """
        issues: list[str] = []

        # Compute individual dimension scores
        text_retention = self._score_text_retention(original, processed, issues)
        heading_detection = self._score_heading_detection(original, processed, profile, issues)
        table_completeness = self._score_table_completeness(original, processed, issues)
        numeric_protection = self._score_numeric_protection(original, processed, issues)
        boilerplate_removal = self._score_boilerplate_removal(original, processed, issues)

        # Weighted overall score
        overall = (
            self.weight_text_retention * text_retention
            + self.weight_heading_detection * heading_detection
            + self.weight_table_completeness * table_completeness
            + self.weight_numeric_protection * numeric_protection
            + self.weight_boilerplate_removal * boilerplate_removal
        )

        components = {
            "text_retention": text_retention,
            "heading_detection": heading_detection,
            "table_completeness": table_completeness,
            "numeric_protection": numeric_protection,
            "boilerplate_removal": boilerplate_removal,
        }

        return ParseQualityScore(
            overall=overall,
            components=components,
            issues=issues,
        )

    def needs_review(self, score: ParseQualityScore) -> bool:
        """Check if a document needs human review based on its quality score.

        Args:
            score: The computed quality score

        Returns:
            True if the score is below the review threshold
        """
        return score.overall < self.review_threshold

    # ─── Dimension 1: Text Retention Rate ─────────────────────────────

    def _score_text_retention(
        self,
        original: ParsedDocument,
        processed: ProcessedDocument,
        issues: list[str],
    ) -> float:
        """Score text retention: visible chars after cleaning / original visible chars.

        Algorithm::

            retention = min(visible_chars(processed) / max(visible_chars(original), 1), 1.0)

        Visible chars are non-whitespace characters across all blocks' text
        (delegated to :func:`_visible_char_count`). The score is capped at
        1.0 because the metric measures *preservation*; some Markdown
        renderings legitimately add characters (e.g. ``**`` markers) and
        we don't want that to read as "more than perfect".

        Special cases (recorded as issue codes for downstream filtering):

        - ``original_empty``: original has no visible text at all → return
          1.0 (nothing to retain, vacuously preserved).
        - ``text_lost``: original has visible text but processed has none
          → return 0.0 and flag the document as a candidate for review.

        Args:
            original: Original parsed document.
            processed: Processed document. Blocks with ``is_noise=True``
                are excluded from the processed character count, so noise
                removal is *not* penalised by this dimension.
            issues: List to append issue codes to.

        Returns:
            Score in ``[0.0, 1.0]``.
        """
        # Original visible character count (across all original blocks).
        original_text = "".join(block.text for block in original.blocks)
        original_chars = _visible_char_count(original_text)

        # Processed visible character count, excluding intentionally
        # removed noise so that successful boilerplate removal does not
        # double-penalise text retention.
        processed_text = "".join(
            block.text for block in processed.blocks if not block.is_noise
        )
        processed_chars = _visible_char_count(processed_text)

        # Special case: original had nothing to retain. Treat as full
        # retention so this dimension does not drag down the overall score
        # for a document that simply has no text body (e.g. image-only).
        if original_chars == 0:
            issues.append("original_empty")
            return 1.0

        # Special case: original had content but everything was lost
        # after processing. This is a strong signal the document needs
        # review and should not be quietly scored low-but-nonzero.
        if processed_chars == 0:
            issues.append("text_lost")
            return 0.0

        # Cap at 1.0 — the dimension measures preservation, not growth.
        ratio = processed_chars / original_chars
        if ratio > 1.0:
            ratio = 1.0

        return ratio

    # ─── Dimension 2: Heading Detection Rate ──────────────────────────

    def _score_heading_detection(
        self,
        original: ParsedDocument,
        processed: ProcessedDocument,
        profile: DocumentProfileConfig,
        issues: list[str],
    ) -> float:
        """Score heading detection: detected headings / expected headings.

        Expected headings are estimated from:
        - Blocks already typed as "heading" in the original
        - Blocks matching heading rules in the profile

        Args:
            original: Original parsed document
            processed: Processed document
            profile: Document profile with heading rules
            issues: List to append issues to

        Returns:
            Score between 0.0 and 1.0
        """
        # Estimate expected headings from original document
        expected_headings = 0
        for block in original.blocks:
            if block.type == "heading":
                expected_headings += 1
            elif profile.heading_rules:
                # Check if block text matches any heading rule
                text = block.text.strip()
                for rule in profile.heading_rules:
                    try:
                        if re.match(rule.pattern, text):
                            expected_headings += 1
                            break
                    except re.error:
                        pass

        if expected_headings == 0:
            # No headings expected - give full score
            return 1.0

        # Count detected headings in processed output
        detected_headings = processed.headings_detected

        ratio = min(detected_headings / expected_headings, 1.0)

        if ratio < 0.7:
            issues.append(
                f"标题识别率偏低 ({ratio:.1%})，预期 {expected_headings} 个标题，"
                f"仅识别到 {detected_headings} 个"
            )

        return ratio

    # ─── Dimension 3: Table Completeness ──────────────────────────────

    def _score_table_completeness(
        self,
        original: ParsedDocument,
        processed: ProcessedDocument,
        issues: list[str],
    ) -> float:
        """Score table completeness: cell fill rate and cross-page merge success.

        Evaluates:
        - Whether all original tables are preserved
        - Cell fill rate (non-empty cells / total cells)
        - Cross-page table merge success (adjacent page tables merged)

        Args:
            original: Original parsed document
            processed: Processed document
            issues: List to append issues to

        Returns:
            Score between 0.0 and 1.0
        """
        # Count original tables
        original_tables = [b for b in original.blocks if b.type == "table"]
        if not original_tables:
            # No tables in document - full score
            return 1.0

        # Count processed tables
        processed_tables = [b for b in processed.blocks if b.type == "table"]

        # Table preservation score
        if not processed_tables:
            issues.append("原始文档包含表格但处理后未保留任何表格")
            return 0.0

        # Calculate cell fill rate across all processed tables
        total_cells = 0
        filled_cells = 0

        for table_block in processed_tables:
            cells_info = self._count_table_cells(table_block.text)
            total_cells += cells_info["total"]
            filled_cells += cells_info["filled"]

        cell_fill_rate = filled_cells / total_cells if total_cells > 0 else 1.0

        # Cross-page merge success: check if adjacent-page tables were merged
        merge_score = self._evaluate_cross_page_merge(original_tables, processed_tables)

        # Combined score: 60% cell fill rate + 40% merge success
        score = 0.6 * cell_fill_rate + 0.4 * merge_score

        if score < 0.7:
            issues.append(
                f"表格完整率偏低 ({score:.1%})，"
                f"单元格填充率 {cell_fill_rate:.1%}，跨页合并率 {merge_score:.1%}"
            )

        return score

    def _count_table_cells(self, table_text: str) -> dict[str, int]:
        """Count total and filled cells in a Markdown table.

        Args:
            table_text: Markdown table text

        Returns:
            Dict with 'total' and 'filled' cell counts
        """
        lines = table_text.strip().split("\n")
        total = 0
        filled = 0

        for line in lines:
            # Skip separator lines
            if re.match(r"^\|[\s\-:|]+\|$", line.strip()):
                continue
            if "|" in line:
                cells = [c.strip() for c in line.split("|")]
                # Remove empty first/last from leading/trailing |
                cells = [c for c in cells if c is not None]
                # Filter out the empty strings from split
                actual_cells = cells[1:-1] if len(cells) > 2 else cells
                for cell in actual_cells:
                    total += 1
                    if cell.strip():
                        filled += 1

        return {"total": max(total, 1), "filled": filled}

    def _evaluate_cross_page_merge(
        self,
        original_tables: list[Block],
        processed_tables: list[ProcessedBlock],
    ) -> float:
        """Evaluate cross-page table merge success.

        If original has tables on adjacent pages, check if they were merged.

        Args:
            original_tables: Original table blocks
            processed_tables: Processed table blocks

        Returns:
            Score between 0.0 and 1.0
        """
        if len(original_tables) <= 1:
            return 1.0

        # Detect adjacent-page table pairs in original
        adjacent_pairs = 0
        for i in range(len(original_tables) - 1):
            if original_tables[i + 1].page_number == original_tables[i].page_number + 1:
                adjacent_pairs += 1

        if adjacent_pairs == 0:
            return 1.0

        # If processed has fewer tables than original, merging likely happened
        merged_count = len(original_tables) - len(processed_tables)
        merge_ratio = min(merged_count / adjacent_pairs, 1.0) if adjacent_pairs > 0 else 1.0

        return max(merge_ratio, 0.0)

    # ─── Dimension 4: Numeric Protection Rate ─────────────────────────

    def _score_numeric_protection(
        self,
        original: ParsedDocument,
        processed: ProcessedDocument,
        issues: list[str],
    ) -> float:
        """Score numeric protection: sampling key numerics for integrity.

        Algorithm::

            1. Concatenate text from all original blocks.
            2. Extract numeric tokens via :func:`_extract_numbers` — this
               covers plain numbers (``3.14``), numbers with units
               (``25°C``, ``100kPa``, ``0.05mm/m``), tolerances (``±10mm``),
               ranges (``55°~65°``) and percentages/permilles (``5%``,
               ``0.1‰``).
            3. Deduplicate while preserving first-appearance order. Without
               this step a document that mentions ``100kPa`` 200 times would
               consume the entire sample with one value, masking real
               regressions on other numbers.
            4. Sample the first ``NUMERIC_SAMPLE_SIZE`` unique tokens. The
               cap keeps scoring cost bounded on long specs while still
               being statistically sufficient to detect preservation drift.
            5. For each sampled token, check whether its exact substring
               survives in the (non-noise) processed text.
            6. ``ratio = preserved / sampled``; if it falls below
               ``NUMERIC_PROTECTION_ISSUE_THRESHOLD`` (0.9), append a
               human-readable issue listing the ratio, sample size, lost
               count and a few concrete missing-value examples.

        Edge cases:

        - Original has no numeric tokens at all → return ``1.0`` (vacuous
          protection: there is nothing to protect, so the dimension does
          not penalise the document).
        - Sampling does *not* re-deduplicate against processed text: if
          the original mentions a value once and processed preserves it
          once, that value counts as preserved.

        Args:
            original: Original parsed document.
            processed: Processed document. Noise blocks are excluded so
                that numbers appearing only inside removed boilerplate are
                not double-counted as preserved.
            issues: List to append issue messages to.

        Returns:
            Score in ``[0.0, 1.0]``.
        """
        # 1) Aggregate original text and extract numeric tokens.
        original_text = " ".join(block.text for block in original.blocks)
        original_numbers = _extract_numbers(original_text)

        # Edge case: nothing to protect → vacuous full score.
        if not original_numbers:
            return 1.0

        # 2) Deduplicate while preserving first-appearance order. dict.fromkeys
        #    is the idiomatic order-preserving dedupe in Python 3.7+ and
        #    avoids the false-confidence we'd get by scoring the same value
        #    repeatedly.
        unique_numbers = list(dict.fromkeys(original_numbers))

        # 3) Cap the sample to bound scoring cost on long documents.
        sample_size = min(len(unique_numbers), NUMERIC_SAMPLE_SIZE)
        sample = unique_numbers[:sample_size]

        # 4) Build the searchable processed text (excluding intentionally
        #    removed noise so that numbers in removed boilerplate don't
        #    spuriously inflate the score).
        processed_text = " ".join(
            block.text for block in processed.blocks if not block.is_noise
        )

        # 5) Substring check for each sampled value, recording which ones
        #    were lost so we can surface them in the issue message.
        preserved_count = 0
        missing_examples: list[str] = []
        for number in sample:
            if number in processed_text:
                preserved_count += 1
            elif len(missing_examples) < NUMERIC_ISSUE_EXAMPLE_LIMIT:
                missing_examples.append(number)

        ratio = preserved_count / sample_size if sample_size > 0 else 1.0

        # 6) Flag when the ratio drops below the strict 0.9 threshold.
        if ratio < NUMERIC_PROTECTION_ISSUE_THRESHOLD:
            lost_count = sample_size - preserved_count
            message = (
                f"数值保护率偏低 ({ratio:.1%})，抽样 {sample_size} 个数值中 "
                f"{lost_count} 个未在处理结果中找到"
            )
            if missing_examples:
                message += f"，例如：{', '.join(missing_examples)}"
            issues.append(message)

        return ratio

    # ─── Dimension 5: Boilerplate Removal Rate ────────────────────────

    def _score_boilerplate_removal(
        self,
        original: ParsedDocument,
        processed: ProcessedDocument,
        issues: list[str],
    ) -> float:
        """Score boilerplate removal: watermark/header/footer removal coverage.

        Algorithm::

            ratio = min(noise_removed / expected_noise_blocks, 1.0)

        ``expected_noise_blocks`` is estimated by inspecting the first
        and last block of each page in the *original* document and
        counting any text that repeats on ``≥ BOILERPLATE_DETECTION_THRESHOLD``
        (50%) of the pages. This mirrors the statistical detector in
        :class:`DocumentProcessor._detect_statistical_noise`, so the
        scorer's "expected noise" matches what the cleaner actually
        targets.

        Implementation details / hardening:

        - Block text is normalised before comparison: surrounding
          whitespace stripped, internal whitespace collapsed, comparison
          done case-insensitively. This prevents minor rendering
          differences (``"Page 1 "`` vs ``" page  1"``) from masking
          repeated boilerplate.
        - When a page has only a single non-empty block, that block is
          counted only once (as the first-block candidate) — without this
          guard a single-block page would contribute the same text to
          both the first-block and last-block tallies and overstate the
          expected noise count.

        Edge cases (each returns ``1.0`` and does not record an issue):

        - Original document has zero blocks → vacuously perfect.
        - Original spans fewer than ``MIN_PAGES_FOR_DETECTION`` pages →
          the frequency rule is too noisy to be useful; the dimension
          declines to penalise the document.
        - No expected boilerplate detected → nothing to remove, so the
          dimension does not penalise the document. ``noise_removed`` is
          *not* compared against zero here: aggressive removal of
          per-page unique noise (e.g. random page numbers) is still a
          legitimate signal that we don't want to read as "score went
          above 1.0".

        Issue reporting: when ``ratio < BOILERPLATE_ISSUE_THRESHOLD``
        (0.7), append a Chinese message containing the ratio, the
        expected noise count and the actual ``noise_removed_count``.

        Args:
            original: Original parsed document.
            processed: Processed document; ``noise_removed_count`` is the
                authoritative count of blocks the cleaner removed as
                noise.
            issues: List to append issue messages to.

        Returns:
            Score in ``[0.0, 1.0]``.
        """
        # Edge case: empty original document → nothing to score.
        total_blocks = len(original.blocks)
        if total_blocks == 0:
            return 1.0

        # Group original blocks by page to inspect first/last positions.
        pages: dict[int, list[Block]] = {}
        for block in original.blocks:
            pages.setdefault(block.page_number, []).append(block)

        # Edge case: too few pages for the frequency rule to be reliable.
        # The 50% threshold can't distinguish "boilerplate" from "the
        # document only has 2 pages and they happen to start the same way".
        total_pages = len(pages)
        if total_pages < MIN_PAGES_FOR_DETECTION:
            return 1.0

        # Collect the first and last *normalised* non-empty block text on
        # each page. Pages with a single non-empty block contribute only
        # once (avoids double-counting the same block in both buckets).
        first_texts: list[str] = []
        last_texts: list[str] = []
        for page_num in sorted(pages.keys()):
            non_empty = [b for b in pages[page_num] if b.text and b.text.strip()]
            if not non_empty:
                continue
            first_texts.append(_normalize_boilerplate_text(non_empty[0].text))
            if len(non_empty) > 1:
                last_texts.append(_normalize_boilerplate_text(non_empty[-1].text))

        expected_noise_blocks = _count_repeated_above_threshold(
            first_texts, total_pages, BOILERPLATE_DETECTION_THRESHOLD
        ) + _count_repeated_above_threshold(
            last_texts, total_pages, BOILERPLATE_DETECTION_THRESHOLD
        )

        # No statistically detected boilerplate → vacuous full score.
        if expected_noise_blocks == 0:
            return 1.0

        # Cap at 1.0 — over-removing is a different concern (would show
        # up in text_retention) and shouldn't read as ">100% removed".
        noise_removed = processed.noise_removed_count
        removal_ratio = min(noise_removed / expected_noise_blocks, 1.0)

        if removal_ratio < BOILERPLATE_ISSUE_THRESHOLD:
            issues.append(
                f"噪声去除率偏低 ({removal_ratio:.1%})，"
                f"预期去除 {expected_noise_blocks} 个噪声块，实际去除 {noise_removed} 个"
            )

        return removal_ratio
