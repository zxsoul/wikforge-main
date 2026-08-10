"""Universal Parser: LLM-based fallback parser for documents without matching profiles.

Implements:
- UniversalParser: Multimodal LLM-driven document parsing
- Page-to-image conversion:
    * PDF → pdf2image (Poppler)
    * Office (DOCX/PPTX/XLSX/ODT/RTF) → LibreOffice (`soffice --headless --convert-to pdf`)
      then pdf2image
- Per-page multimodal LLM call (image + text → structured Markdown)
- Page result merging (preserve page numbers, merge cross-page tables, deduplicate)
- Candidate Profile generation from parsing results
- LLM model selection (GPT-4o / Qwen-VL / MiniCPM-V via LiteLLM)
- Degradation: on LLM failure, fall back to plain text + fixed-size chunks

Trigger conditions:
- No profile match (profile_matcher returns default/generic)
- Quality score < threshold (default 0.7)
"""

import asyncio
import io
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field

from app.core.config import get_settings
from app.services.document_processor import ProcessedBlock, ProcessedDocument
from app.services.llm_gateway import LLMGateway, LLMGatewayError
from app.services.parsers.base import Block, ParsedDocument

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Design alignment: ``design.md`` references the LLM-fallback output as
# ``StructuredDocument``. In this codebase we reuse the IR produced by Task 9's
# ``DocumentProcessor`` (``app.services.document_processor.ProcessedDocument``)
# as the structured-output type. The alias below keeps the design vocabulary
# discoverable while avoiding type drift between the two pipelines.
# ──────────────────────────────────────────────────────────────────────────────
StructuredDocument = ProcessedDocument

# Default chunk size for degradation mode (plain text fallback)
FALLBACK_CHUNK_SIZE = 500  # characters

# File extensions / file_type values that LibreOffice can convert to PDF.
OFFICE_FILE_TYPES = {
    "docx", "doc",
    "pptx", "ppt",
    "xlsx", "xls",
    "odt", "ods", "odp",
    "rtf",
}


@dataclass
class PageResult:
    """Result of parsing a single page via LLM.

    Attributes:
        page_number: The page number (1-indexed)
        markdown: Structured Markdown content for this page
        headings: Detected headings with levels
        tables: Detected tables (Markdown format)
        success: Whether the LLM call succeeded
        error: Error message if failed
    """

    page_number: int = 1
    markdown: str = ""
    headings: list[dict] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    success: bool = True
    error: str = ""


SYSTEM_PROMPT = """You are a document structure analysis expert. Given a page image and the raw text
extracted from that page, produce a faithful structured Markdown rendition of the page.

Hard requirements:
1. Preserve heading hierarchy. Use `#` / `##` / `###` … to mark heading levels exactly as
   they appear visually (font size, numbering, bold). Never invent headings.
2. Preserve tables. Render every table using GitHub-flavored Markdown table syntax
   (`| col | col |`). For merged cells or nested tables that cannot be expressed in
   Markdown, expand them into the closest equivalent rows and add a brief textual note.
3. Preserve formulas, units, and numeric values verbatim. Do not normalize, round, or
   translate numbers, units (mm, °, %), Greek letters, or formulas.
4. Remove obvious noise: watermarks, repeated page headers/footers, page numbers
   appearing alone on a line, and "confidential" stamps. Keep all body content.
5. Keep the content in its original language. Do not translate.
6. Output ONLY the Markdown body for the page. Do NOT wrap the output in code fences,
   do NOT add explanations, prefaces, or trailing commentary.
"""

PAGE_PROMPT_TEMPLATE = """Convert page {page_number} of this document into faithful structured Markdown.

Use the page image as the source of truth for layout (headings, tables, columns, formulas).
The raw text below was extracted by a non-LLM parser and may be reordered or noisy — use it
to disambiguate hard-to-read characters and to recover numbers, but trust the image for structure.

Raw text extracted from this page (may be partial / out of order):
---
{raw_text}
---

Return only the Markdown body for this page."""


# Pre-compiled regex used to strip surrounding ```markdown / ``` fences if the LLM
# wraps its output in a code block despite the system prompt asking it not to.
_FENCED_OUTPUT_RE = re.compile(
    r"^\s*```(?:markdown|md)?\s*\n(?P<body>.*?)\n?```\s*$",
    re.DOTALL | re.IGNORECASE,
)


class UniversalParser:
    """LLM-based universal document parser for fallback scenarios.

    Used when:
    - No Document Profile matches the document
    - Parse quality score is below threshold (default 0.7)

    The parser uses multimodal LLM to understand document structure by
    processing each page with both the page image and raw text.
    On failure, degrades to plain text extraction with fixed-size chunking.
    """

    # Known multimodal models (used as a hint for model selection helpers).
    # The actual call dispatch is delegated to LiteLLM; this list is for UI/config
    # validation and for the configurable `prefer_vision_model` selection logic.
    KNOWN_VISION_MODELS = (
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-4-vision-preview",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-5-sonnet-20241022",
        "qwen-vl-max",
        "qwen-vl-plus",
        "minicpm-v",
    )

    # 任务 10.7：常见纯文本模型。仅用于 UI 展示与 ``is_known_text_model`` 校验，
    # 实际派发依然走 LiteLLM —— 列表里没有的模型不会被拒绝。
    KNOWN_TEXT_MODELS = (
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "qwen-max",
        "qwen-plus",
        "qwen-turbo",
        "deepseek-chat",
        "ollama/llama3.1",
        "ollama/qwen2",
    )

    def __init__(
        self,
        llm_gateway: LLMGateway | None = None,
        model: str | None = None,
        vision_model: str | None = None,
        text_model: str | None = None,
        page_timeout: float | None = None,
        max_raw_text_chars: int | None = None,
        fallback_chunk_chars: int | None = None,
    ):
        """Initialize the Universal Parser.

        Args:
            llm_gateway: LLM Gateway instance. If None, creates one with default settings.
            model: 兼容旧调用方的别名。如果调用方既没传 ``text_model`` 也没传
                ``vision_model``，``model`` 会承担两个职责：
                - 作为 ``self.model`` 传给新建的 ``LLMGateway``（保持原行为）
                - 当 ``text_model`` 未显式提供时，回填 ``self.text_model``（兼容
                  10.7 之前“一个 model 同时管文本兜底”的写法）
                这样既不破坏旧测试，又允许新调用方用 ``text_model`` / ``vision_model``
                做更细粒度的覆盖。
            vision_model: 多模态调用使用的模型（任务 10.7）。解析顺序：
                显式参数 → ``settings.UNIVERSAL_PARSER_VISION_MODEL`` → ``None``。
                ``None`` 时不会向 gateway 传 ``model=``，使用 gateway 默认模型。
                接受任意 LiteLLM 兼容标识符（GPT-4o / Qwen-VL / MiniCPM-V /
                ``ollama/minicpm-v:latest`` 等），不在 ``KNOWN_VISION_MODELS`` 中的
                值不会被拒绝 —— 列表只用于 UI 展示。
            text_model: 纯文本兜底调用使用的模型（任务 10.7）。解析顺序：
                显式参数 → ``settings.UNIVERSAL_PARSER_TEXT_MODEL`` →
                ``model`` 形参（旧写法兜底） → ``None``。
            page_timeout: Timeout per page LLM call in seconds. Default 60s.
            max_raw_text_chars: Maximum number of characters from the per-page raw text
                that will be embedded into the LLM prompt. When ``None`` the value is
                taken from ``settings.UNIVERSAL_PARSER_MAX_RAW_TEXT_CHARS``. Truncating
                here protects the LLM call from token-limit errors on noisy parsers.
            fallback_chunk_chars: Character size for fixed-size chunks emitted by the
                degradation path (任务 10.8). When ``None`` the value is taken from
                ``settings.UNIVERSAL_PARSER_FALLBACK_CHUNK_CHARS`` (default 500). Floor
                at 1 to avoid an infinite loop on degenerate configs.
        """
        settings = get_settings()
        self.model = model or settings.LITELLM_MODEL

        # 任务 10.7：vision_model 解析顺序
        # 1) 显式构造参数优先；
        # 2) 否则读 settings.UNIVERSAL_PARSER_VISION_MODEL（空字符串视为未设置）；
        # 3) 都没有就保持 None，让 _parse_page 不传 model= 给 gateway。
        if vision_model is not None:
            self.vision_model: str | None = vision_model or None
        else:
            settings_vision = getattr(settings, "UNIVERSAL_PARSER_VISION_MODEL", "")
            # 防御：测试里常用 ``MagicMock(...)`` 注入 settings；未显式声明的属性会返回
            # 子 MagicMock 而非默认值。只接受字符串，避免污染下游 LiteLLM 调用。
            if not isinstance(settings_vision, str):
                settings_vision = ""
            self.vision_model = settings_vision or None

        # 任务 10.7：text_model 解析顺序
        # 1) 显式构造参数优先；
        # 2) settings.UNIVERSAL_PARSER_TEXT_MODEL（空字符串视为未设置）；
        # 3) 旧 ``model`` 参数兜底（保持向后兼容：之前的调用方用 model=...
        #    希望覆盖文本路径）。注意：``self.model`` 同时也被用来构造 gateway，
        #    这里再让它兼任 text_model 的兜底来源；如果调用方明确传了 text_model
        #    或者环境变量，该兜底就不生效。
        # 4) 否则 None，_parse_page 不传 model= 给 gateway。
        if text_model is not None:
            self.text_model: str | None = text_model or None
        else:
            settings_text = getattr(settings, "UNIVERSAL_PARSER_TEXT_MODEL", "")
            if not isinstance(settings_text, str):
                settings_text = ""
            if settings_text:
                self.text_model = settings_text
            elif model is not None:
                # 旧 API：``model`` 隐式承担 text_model 的角色。
                self.text_model = model or None
            else:
                self.text_model = None

        # page_timeout 解析: 显式参数 → settings → 默认 120s
        # qwen-vl-plus 单页推理 + base64 图片传输平均 30-60s, 慢的页可能 90s+,
        # 所以默认 120s 比 60s 更稳。可通过 UNIVERSAL_PARSER_PAGE_TIMEOUT 覆盖。
        if page_timeout is None:
            page_timeout = float(
                getattr(settings, "UNIVERSAL_PARSER_PAGE_TIMEOUT", 120.0)
            )
        self.page_timeout = max(page_timeout, 1.0)

        # 任务 10.2: page rasterization knobs come from settings so deployments
        # can tune DPI / LibreOffice timeout without code changes.
        self._page_dpi = int(getattr(settings, "UNIVERSAL_PARSER_PAGE_DPI", 150))
        self._libreoffice_timeout = int(
            getattr(settings, "UNIVERSAL_PARSER_LIBREOFFICE_TIMEOUT", 60)
        )

        # 任务 10.3: per-page raw-text budget. Override via constructor for tests
        # / specialized profiles, otherwise read the global setting.
        if max_raw_text_chars is None:
            max_raw_text_chars = int(
                getattr(settings, "UNIVERSAL_PARSER_MAX_RAW_TEXT_CHARS", 3000)
            )
        # Defensive: never let a misconfigured value drop the budget to 0/negative,
        # which would feed an empty prompt to the LLM and surface as 'empty content'.
        self.max_raw_text_chars = max(max_raw_text_chars, 1)

        # 任务 10.8: degradation chunk size. Override via constructor for tests
        # / specialized profiles, otherwise read the global setting. Floor at 1
        # so a misconfigured 0 never produces an infinite loop in
        # ``_chunk_text_into_blocks``.
        if fallback_chunk_chars is None:
            fallback_chunk_chars = int(
                getattr(settings, "UNIVERSAL_PARSER_FALLBACK_CHUNK_CHARS", FALLBACK_CHUNK_SIZE)
            )
        self.fallback_chunk_chars = max(fallback_chunk_chars, 1)

        # Per-`parse()` cache: maps source Office file path → converted PDF path.
        # Avoids invoking LibreOffice once per page on multi-page documents.
        # Populated by ``_office_pdf_cache`` and cleared on context exit.
        self._office_pdf_cache: dict[str, str] = {}

        if llm_gateway is not None:
            self.llm = llm_gateway
        else:
            self.llm = LLMGateway(model=self.model, timeout=page_timeout)

    async def parse(self, parsed_doc: ParsedDocument) -> ProcessedDocument:
        """Parse a document using multimodal LLM, with degradation on failure.

        任务 10.8 — Per-page partial-success policy:

        - Every page is attempted; a single page failure does not abort the
          remaining pages. Per-page exceptions are logged at WARNING with the
          page number and a short reason code (``timeout`` / ``rate_limit`` /
          ``empty`` / ``unknown``).
        - When at least one page succeeds, successful pages flow through
          ``_merge_page_results`` and failed pages are appended as
          fixed-size character chunks derived from the original
          ``parsed_doc.blocks``. The final ``ProcessedDocument.metadata`` gains a
          ``"universal_parser"`` envelope that records ``successful_pages``,
          ``degraded_pages``, ``failed_pages`` and a per-page ``page_errors``
          map keyed by page number.
        - When **all** pages fail, the whole document falls back to
          ``_degrade_to_plain_text`` (the original behavior). The metadata
          envelope is still emitted so operators can spot the failure mode.

        Args:
            parsed_doc: The parsed document intermediate representation.

        Returns:
            ProcessedDocument with structured blocks, Markdown, and the
            per-page failure-status envelope under ``metadata["universal_parser"]``.
        """
        if not parsed_doc.blocks:
            return ProcessedDocument(metadata=parsed_doc.metadata)

        # Group blocks by page
        pages = self._group_blocks_by_page(parsed_doc.blocks)
        all_page_numbers = sorted(pages.keys())

        # 任务 10.2 + 10.8: convert any Office source to PDF exactly once and run
        # every page through the LLM. ``page_errors`` records the reason code for
        # each failure; ``page_results`` keeps the successful results.
        page_results, page_errors = await self._process_pages(parsed_doc, pages)

        successful_pages = [r.page_number for r in page_results]
        failed_pages = sorted(page_errors.keys())

        # 任务 10.8 — Summary log so operators can spot misbehaving LLM deployments.
        logger.info(
            "universal_parser_completed pages_total=%d pages_succeeded=%d "
            "pages_degraded=%d pages_lost=%d",
            len(all_page_numbers),
            len(successful_pages),
            len(failed_pages),  # degraded == failed: each failed page produces a degraded section
            0,  # lost stays 0; with partial-success policy we never silently lose pages
        )

        # All pages failed → whole-document plain-text fallback (matches the
        # legacy behavior). Decorate metadata with the failure envelope so the
        # caller can still observe what went wrong.
        if not page_results:
            degraded = self._degrade_to_plain_text(parsed_doc)
            self._stamp_universal_parser_metadata(
                degraded,
                successful_pages=[],
                degraded_pages=[],  # whole-doc fallback is not a per-page degradation
                failed_pages=failed_pages,
                page_errors=page_errors,
                whole_doc_degraded=True,
            )
            return degraded

        # Mixed / all-success path: merge successes, append per-page degraded
        # sections for any failures, and decorate metadata.
        merged = self._merge_page_results(page_results, parsed_doc.metadata)

        if failed_pages:
            degraded_blocks = self._degrade_pages(
                parsed_doc, failed_pages, self.fallback_chunk_chars
            )
            if degraded_blocks:
                merged.blocks.extend(degraded_blocks)
                merged.markdown = self._blocks_to_markdown(merged.blocks)

        # ``degraded_pages`` records the pages that actually contributed
        # degraded blocks (a failed page with no original blocks contributes
        # nothing and therefore is not counted here).
        degraded_pages = self._degraded_pages_with_blocks(parsed_doc, failed_pages)
        self._stamp_universal_parser_metadata(
            merged,
            successful_pages=successful_pages,
            degraded_pages=degraded_pages,
            failed_pages=failed_pages,
            page_errors=page_errors,
            whole_doc_degraded=False,
        )
        return merged

    async def _process_pages(
        self,
        parsed_doc: ParsedDocument,
        pages: dict[int, list[Block]],
    ) -> tuple[list[PageResult], dict[int, str]]:
        """Run the per-page LLM loop and return (successes, failure-reason map).

        任务 10.8 — extracted helper. Iterates over every page, invokes
        ``_parse_page`` under the ``_office_pdf_scope`` so multi-page Office
        documents only convert once. Each per-page exception is caught,
        translated to a short reason code and recorded in the returned map; the
        loop never aborts on a single page.

        Reason codes:
        - ``LLMGatewayError.reason`` is propagated verbatim when available
          (``timeout`` / ``rate_limit`` / ``auth`` / ``model_unavailable`` /
          ``empty`` / ``unknown``).
        - Any non-LLM exception becomes ``"unknown"``.
        """
        page_results: list[PageResult] = []
        page_errors: dict[int, str] = {}

        with self._office_pdf_scope():
            for page_num in sorted(pages.keys()):
                page_blocks = pages[page_num]
                raw_text = "\n".join(b.text for b in page_blocks if b.text.strip())
                page_image = self._get_page_image(parsed_doc, page_num)

                try:
                    result = await self._parse_page(
                        page_number=page_num,
                        raw_text=raw_text,
                        page_image=page_image,
                    )
                    page_results.append(result)
                except LLMGatewayError as e:
                    reason = getattr(e, "reason", None) or "unknown"
                    logger.warning(
                        "universal_parser page=%d reason=%s error=%s",
                        page_num,
                        reason,
                        e,
                    )
                    page_errors[page_num] = reason
                except Exception as e:  # noqa: BLE001 — defensive bucket
                    logger.warning(
                        "universal_parser page=%d reason=unknown error=%s",
                        page_num,
                        e,
                    )
                    page_errors[page_num] = "unknown"

        return page_results, page_errors

    def _degrade_pages(
        self,
        parsed_doc: ParsedDocument,
        page_numbers: list[int],
        fallback_chunk_chars: int,
    ) -> list[ProcessedBlock]:
        """Produce degraded fixed-size chunks for the listed failed pages.

        任务 10.8 — extracted helper. For each page in ``page_numbers``:

        - Concatenate the original ``parsed_doc.blocks`` for that page (the
          native parser's output) into a single text body.
        - Split the body into ``fallback_chunk_chars``-sized character chunks
          using the same chunking primitive as ``_degrade_to_plain_text``.
        - Emit each non-empty chunk as a paragraph block anchored to the
          original ``page_number``.

        Pages with no original blocks (or only whitespace) emit nothing — the
        caller is responsible for not counting them under ``degraded_pages``.
        """
        if not page_numbers or fallback_chunk_chars < 1:
            return []

        # Build a per-page lookup once; ``parsed_doc.blocks`` is iterated only
        # in document order so this is O(n).
        blocks_by_page: dict[int, list[Block]] = {}
        for block in parsed_doc.blocks:
            blocks_by_page.setdefault(block.page_number, []).append(block)

        degraded: list[ProcessedBlock] = []
        for page_number in page_numbers:
            originals = blocks_by_page.get(page_number, [])
            page_text = "\n".join(b.text for b in originals if b.text.strip())
            if not page_text.strip():
                # Nothing to fall back to for this page — emit nothing. The
                # caller will not count this page under ``degraded_pages``.
                continue

            for chunk_text in self._chunk_text_to_char_blocks(
                page_text, fallback_chunk_chars
            ):
                degraded.append(
                    ProcessedBlock(
                        type="paragraph",
                        text=chunk_text,
                        page_number=page_number,
                    )
                )

        return degraded

    def _degraded_pages_with_blocks(
        self, parsed_doc: ParsedDocument, failed_pages: list[int]
    ) -> list[int]:
        """Return the subset of ``failed_pages`` that has original blocks to degrade.

        Mirrors the defensive check in ``_degrade_pages`` so the metadata
        envelope's ``degraded_pages`` list never includes pages that produced
        no blocks.
        """
        if not failed_pages:
            return []
        page_text_present: dict[int, bool] = {}
        for block in parsed_doc.blocks:
            if not block.text or not block.text.strip():
                continue
            page_text_present[block.page_number] = True
        return [p for p in failed_pages if page_text_present.get(p, False)]

    @staticmethod
    def _stamp_universal_parser_metadata(
        document: ProcessedDocument,
        *,
        successful_pages: list[int],
        degraded_pages: list[int],
        failed_pages: list[int],
        page_errors: dict[int, str],
        whole_doc_degraded: bool,
    ) -> None:
        """Attach the ``metadata["universal_parser"]`` failure envelope.

        ``ProcessedDocument.metadata`` is often aliased to ``parsed_doc.metadata``
        (both ``_merge_page_results`` and ``_degrade_to_plain_text`` pass the
        original dict through by reference). To avoid mutating the caller's
        input, we shallow-copy the dict before stamping when we detect that
        no envelope is present yet — subsequent calls in the same parse() are
        safe to mutate.

        The envelope is keyed by the single namespace ``"universal_parser"`` so
        downstream consumers can introspect partial failures without colliding
        with arbitrary upstream metadata.
        """
        if document.metadata is None:
            document.metadata = {}
        else:
            # Shallow-copy so we do not surprise callers who reuse parsed_doc.
            document.metadata = dict(document.metadata)
        document.metadata["universal_parser"] = {
            "successful_pages": list(successful_pages),
            "degraded_pages": list(degraded_pages),
            "failed_pages": list(failed_pages),
            "page_errors": dict(page_errors),
            "whole_doc_degraded": whole_doc_degraded,
        }

    async def suggest_profile(self, result: ProcessedDocument) -> dict:
        """Generate a candidate Document Profile from parsing results.

        Analyzes the parsed result to extract:
        - Heading patterns (regex for each level)
        - Noise patterns (repeated text across pages)
        - Chunking recommendations (based on content density)

        The candidate profile is returned as a two-level envelope so the storage
        layer (任务 10.6) can distinguish "candidate" from "saved" profiles
        without having to invent extra columns:

        ```python
        {
            "profile": <profile_to_dict-compatible dict>,
            "metadata": {
                "status": "pending_approval",
                "source": "universal_parser",
                "evidence": {
                    "page_count": int,
                    "heading_count": int,
                    "table_count": int,
                    "boilerplate_candidates": int,
                    "avg_block_chars": float,
                },
            },
        }
        ```

        The inner ``profile`` dict round-trips through
        ``profile_matcher.profile_from_dict`` → ``profile_to_dict`` so the
        storage layer can hand it straight to ``ProfileMatcher`` without any
        extra translation.

        Args:
            result: The processed document from LLM parsing

        Returns:
            Dict with ``profile`` and ``metadata`` keys (see above).
        """
        heading_rules = self._extract_heading_patterns(result)
        noise_patterns = self._extract_noise_patterns(result)
        chunking_config = self._recommend_chunking(result)

        # ── Evidence: shared between the profile name (page count) and the
        # metadata envelope (so reviewers can sanity-check the suggestion).
        page_numbers = {b.page_number for b in result.blocks if b.page_number}
        page_count = max(page_numbers) if page_numbers else 0
        heading_count = sum(1 for b in result.blocks if b.type == "heading")
        table_blocks = [b for b in result.blocks if b.type == "table"]
        table_count = len(table_blocks)
        total_chars = sum(len(b.text or "") for b in result.blocks)
        avg_block_chars = (
            total_chars / len(result.blocks) if result.blocks else 0.0
        )

        # Candidate profile name carries enough signal to be human-readable in
        # the review UI: ``auto-generated-{file_type|generic}-{n_pages}p``.
        file_type = (result.metadata or {}).get("file_type") or "generic"
        file_type = str(file_type).strip().lower() or "generic"
        profile_name = f"auto-generated-{file_type}-{page_count}p"

        tables_config = self._recommend_tables(result)

        # The candidate profile dict below is shaped exactly like the output of
        # ``profile_to_dict`` so the storage layer can pass it through
        # ``profile_from_dict`` without any extra mapping.
        profile_dict = {
            "name": profile_name,
            "description": "Automatically generated by Universal Parser",
            "priority": 0,
            "enabled": False,
            "match_rules": {
                "filename_regex": [],
                "content_regex": [],
                "min_content_match_count": 1,
            },
            "heading_rules": heading_rules,
            "boilerplate": {
                "detection_mode": "both",
                "statistical_threshold": 0.5,
                "manual_patterns": noise_patterns,
            },
            "tables": tables_config,
            "chunking": chunking_config,
            "domain_dictionary_id": None,
        }

        return {
            "profile": profile_dict,
            "metadata": {
                "status": "pending_approval",
                "source": "universal_parser",
                "evidence": {
                    "page_count": page_count,
                    "heading_count": heading_count,
                    "table_count": table_count,
                    "boilerplate_candidates": len(noise_patterns),
                    "avg_block_chars": avg_block_chars,
                },
            },
        }

    def should_trigger(
        self,
        profile_matched: bool,
        quality_score: float | None = None,
        threshold: float | None = None,
    ) -> bool:
        """Check if Universal Parser should be triggered.

        Trigger conditions:
        - No profile matched (using default/generic profile)
        - Quality score below threshold

        Args:
            profile_matched: Whether a specific profile was matched
            quality_score: The parse quality score (0-1), or None if not computed
            threshold: Quality threshold. Defaults to settings.QUALITY_FALLBACK_THRESHOLD.

        Returns:
            True if Universal Parser should be triggered
        """
        settings = get_settings()
        threshold = threshold if threshold is not None else settings.QUALITY_FALLBACK_THRESHOLD

        # Trigger if no profile matched
        if not profile_matched:
            return True

        # Trigger if quality score is below threshold
        if quality_score is not None and quality_score < threshold:
            return True

        return False

    # ─── 任务 10.7：模型校验 / UI 提示辅助 ──────────────────────────────
    # 这两个 classmethod 都是纯字符串校验，不会触发任何网络调用，仅供管理后台
    # 渲染下拉列表 / 给运行期日志做提示。``KNOWN_*`` 元组以外的标识符不会被拒绝。

    @classmethod
    def _normalize_model_name(cls, name: str) -> str:
        """归一化模型名：去 provider 前缀 + 去 ``:tag`` 版本号 + 转小写。

        例：
        - ``"gpt-4o"`` → ``"gpt-4o"``
        - ``"OpenAI/GPT-4o"`` → ``"gpt-4o"``
        - ``"ollama/minicpm-v:latest"`` → ``"minicpm-v"``
        - ``"ollama/qwen2:7b"`` → ``"qwen2"``
        """
        if not isinstance(name, str):
            return ""
        candidate = name.strip()
        if not candidate:
            return ""
        # provider 前缀（``ollama/``、``openrouter/``）：取最后一段
        if "/" in candidate:
            candidate = candidate.rsplit("/", 1)[-1]
        # tag（``:latest`` / ``:7b``）：去掉
        if ":" in candidate:
            candidate = candidate.split(":", 1)[0]
        return candidate.lower()

    @classmethod
    def _is_known_in(cls, name: str, catalog: tuple[str, ...]) -> bool:
        normalized = cls._normalize_model_name(name)
        if not normalized:
            return False
        # 前缀匹配：``gpt-4o-2024-05-13`` 也算是 ``gpt-4o`` 系列。
        for known in catalog:
            known_norm = cls._normalize_model_name(known)
            if not known_norm:
                continue
            if normalized == known_norm or normalized.startswith(known_norm):
                return True
        return False

    @classmethod
    def is_known_vision_model(cls, name: str) -> bool:
        """``name`` 是否在已知多模态模型目录里（信息性提示，不做强制校验）。"""
        return cls._is_known_in(name, cls.KNOWN_VISION_MODELS)

    @classmethod
    def is_known_text_model(cls, name: str) -> bool:
        """``name`` 是否在已知纯文本模型目录里（信息性提示，不做强制校验）。"""
        return cls._is_known_in(name, cls.KNOWN_TEXT_MODELS)

    # ─── Internal Methods ─────────────────────────────────────────────

    def _group_blocks_by_page(self, blocks: list[Block]) -> dict[int, list[Block]]:
        """Group blocks by their page number.

        Args:
            blocks: List of blocks from parsed document

        Returns:
            Dict mapping page_number → list of blocks on that page
        """
        pages: dict[int, list[Block]] = {}
        for block in blocks:
            pages.setdefault(block.page_number, []).append(block)
        return pages

    def _get_page_image(self, parsed_doc: ParsedDocument, page_number: int) -> bytes | None:
        """Get the image for a specific page.

        Conversion strategy:
        - PDF: ``pdf2image`` directly (任务 10.2)
        - Office (DOCX/PPTX/XLSX/ODT/RTF): convert to PDF via LibreOffice
          (``soffice --headless --convert-to pdf``) then ``pdf2image``. The PDF
          is cached for the lifetime of a ``parse()`` call (see ``_office_pdf_scope``)
          so multi-page documents are converted exactly once.
        - Other formats: returns None (text-only mode)

        This helper is intentionally exception-safe: any failure (missing binary,
        invalid page number, corrupted file, missing dependency) is logged and
        returns ``None`` so the caller can fall back to text-only LLM input.

        Args:
            parsed_doc: The parsed document
            page_number: Page number to get image for (1-indexed)

        Returns:
            Image bytes (PNG) or None if not available
        """
        if page_number is None or page_number < 1:
            return None

        file_path = parsed_doc.metadata.get("file_path") or ""
        file_type = (parsed_doc.metadata.get("file_type") or "").lower()

        if not file_path:
            return None
        # We deliberately do NOT touch the filesystem if the path is empty;
        # ``os.path.exists`` would otherwise return False and fall through.
        try:
            if not os.path.exists(file_path):
                return None
        except (TypeError, ValueError):
            return None

        # Discover the file extension from either the explicit metadata or the path.
        ext = file_type
        if not ext and "." in file_path:
            ext = file_path.rsplit(".", 1)[-1].lower()

        # PDF: direct conversion.
        if ext == "pdf":
            return self._pdf_page_to_image(file_path, page_number)

        # Office: route through LibreOffice → PDF → pdf2image.
        if ext in OFFICE_FILE_TYPES:
            return self._office_page_to_image(file_path, page_number)

        return None

    def _pdf_page_to_image(self, file_path: str, page_number: int) -> bytes | None:
        """Convert a single PDF page to a PNG byte string.

        Uses ``pdf2image.convert_from_path`` with the configured DPI. The function
        is exception-safe — any failure (Poppler missing, invalid page, corrupted
        PDF) is logged at WARNING and returns ``None``.

        Args:
            file_path: Path to the PDF file.
            page_number: Page number (1-indexed). Must be ≥ 1.

        Returns:
            PNG image bytes, or ``None`` on failure.
        """
        if page_number < 1:
            logger.warning("PDF page number must be ≥ 1, got %s", page_number)
            return None

        try:
            from pdf2image import convert_from_path  # type: ignore
        except ImportError:
            logger.warning("pdf2image not installed; cannot rasterize PDF page")
            return None

        try:
            images = convert_from_path(
                file_path,
                first_page=page_number,
                last_page=page_number,
                dpi=self._page_dpi,
                fmt="png",
            )
        except Exception as e:  # noqa: BLE001 — best-effort conversion
            # pdf2image raises a variety of exceptions when Poppler is missing or
            # the PDF is corrupted; treat all of them uniformly.
            logger.warning(
                "Failed to convert PDF page %s of %s: %s", page_number, file_path, e
            )
            return None

        if not images:
            return None

        return self._encode_pil_image_to_png(images[0])

    def _office_page_to_image(self, file_path: str, page_number: int) -> bytes | None:
        """Convert an Office document page to a PNG image via LibreOffice.

        Pipeline:
        1. ``soffice --headless --convert-to pdf <file>`` into a temp directory
           (cached per ``parse()`` call so multi-page documents only convert once)
        2. ``pdf2image.convert_from_path`` on the produced PDF for the requested page

        LibreOffice is invoked with a ``--user-profile`` pointing at a temp dir to
        avoid stomping on the host user's profile (and to allow concurrent worker calls).
        Caches and binaries are auto-discovered: ``soffice`` is the canonical name on
        Linux/Mac LibreOffice installs; ``libreoffice`` is also accepted.

        Args:
            file_path: Path to the source Office document (.docx/.pptx/.xlsx/...)
            page_number: Page number (1-indexed). For spreadsheets this maps to sheet index.

        Returns:
            PNG image bytes for the requested page, or ``None`` on failure.
        """
        if page_number < 1:
            logger.warning("Office page number must be ≥ 1, got %s", page_number)
            return None

        try:
            from pdf2image import convert_from_path  # type: ignore
        except ImportError:
            logger.warning("pdf2image not installed; cannot rasterize Office page")
            return None

        pdf_path = self._office_to_pdf(file_path)
        if not pdf_path:
            return None

        try:
            images = convert_from_path(
                pdf_path,
                first_page=page_number,
                last_page=page_number,
                dpi=self._page_dpi,
                fmt="png",
            )
        except Exception as e:  # noqa: BLE001 — best-effort conversion
            logger.warning(
                "Failed to rasterize converted PDF page %s: %s", page_number, e
            )
            return None

        if not images:
            return None

        return self._encode_pil_image_to_png(images[0])

    @staticmethod
    def _encode_pil_image_to_png(image) -> bytes | None:
        """Encode a PIL.Image to PNG bytes, swallowing any encoder errors."""
        try:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return buffer.getvalue()
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to encode page image to PNG: %s", e)
            return None

    @contextmanager
    def _office_pdf_scope(self):
        """Context manager that owns the temp directory for Office→PDF conversions.

        Within the ``with`` block, ``_office_to_pdf`` will create or reuse a single
        PDF per source path and store the cache in ``self._office_pdf_cache``. On
        exit the temp dir (and therefore the cached PDF) is deleted.
        """
        tmpdir = tempfile.mkdtemp(prefix="wikforge-office-")
        self._office_pdf_cache = {}
        # Stash the temp dir on the instance so ``_office_to_pdf`` can reuse it
        # without changing its signature.
        self._office_tmpdir = tmpdir
        try:
            yield
        finally:
            self._office_pdf_cache = {}
            self._office_tmpdir = None
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _office_to_pdf(self, file_path: str) -> str | None:
        """Convert an Office document to PDF via LibreOffice headless.

        Uses the temp directory from ``_office_pdf_scope`` and caches the result by
        absolute source path. Returns ``None`` on any failure (LibreOffice missing,
        timeout, non-zero exit, no PDF produced) and logs at WARNING.

        Args:
            file_path: Path to the Office document.

        Returns:
            Path to the converted PDF, or ``None`` on failure.
        """
        # If parse() didn't open a scope (e.g. helpers called directly), spin up a
        # one-shot temp dir on the fly. The cache is still keyed so it's safe.
        tmpdir = getattr(self, "_office_tmpdir", None)
        if tmpdir is None:
            tmpdir = tempfile.mkdtemp(prefix="wikforge-office-oneshot-")
            self._office_tmpdir = tmpdir

        cached = self._office_pdf_cache.get(file_path)
        if cached and os.path.exists(cached):
            return cached

        soffice_bin = shutil.which("soffice") or shutil.which("libreoffice")
        if not soffice_bin:
            logger.warning(
                "LibreOffice (soffice/libreoffice) not on PATH; cannot rasterize Office page"
            )
            return None

        user_profile = os.path.join(tmpdir, "lo-profile")
        try:
            os.makedirs(user_profile, exist_ok=True)
        except OSError as e:
            logger.warning("Failed to prepare LibreOffice user profile dir: %s", e)
            return None

        cmd = [
            soffice_bin,
            "--headless",
            "--norestore",
            "--nologo",
            "--nofirststartwizard",
            f"-env:UserInstallation=file://{user_profile}",
            "--convert-to",
            "pdf",
            "--outdir",
            tmpdir,
            file_path,
        ]
        try:
            proc = subprocess.run(  # noqa: S603 — invoking trusted binary
                cmd,
                capture_output=True,
                timeout=self._libreoffice_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                "LibreOffice conversion timed out after %ss for %s",
                self._libreoffice_timeout,
                file_path,
            )
            return None
        except OSError as e:
            logger.warning("LibreOffice invocation failed: %s", e)
            return None
        except Exception as e:  # noqa: BLE001 — never raise out of helper
            logger.warning("LibreOffice unexpected error: %s", e)
            return None

        if proc.returncode != 0:
            stderr = (proc.stderr or b"").decode(errors="replace")[:500]
            logger.warning(
                "LibreOffice conversion exited with %s: %s",
                proc.returncode,
                stderr,
            )
            return None

        # Locate the produced PDF. LibreOffice usually names it <basename>.pdf,
        # but some versions sanitize the basename — pick the newest PDF as a fallback.
        base = os.path.splitext(os.path.basename(file_path))[0]
        candidate = os.path.join(tmpdir, f"{base}.pdf")
        if not os.path.exists(candidate):
            try:
                produced = [
                    f for f in os.listdir(tmpdir) if f.lower().endswith(".pdf")
                ]
            except OSError:
                produced = []
            if not produced:
                logger.warning("LibreOffice produced no PDF for %s", file_path)
                return None
            candidate = os.path.join(tmpdir, produced[0])

        self._office_pdf_cache[file_path] = candidate
        return candidate

    async def _parse_page(
        self,
        page_number: int,
        raw_text: str,
        page_image: bytes | None = None,
    ) -> PageResult:
        """Parse a single page using multimodal LLM.

        Behavior contract (任务 10.3):

        - When ``page_image`` is provided, dispatches a multimodal LLM call (image + text)
          via ``LLMGateway.complete_multimodal``. When no image is available — e.g. plain
          text / Markdown / HTML sources, or when rasterization failed — gracefully degrades
          to a text-only ``LLMGateway.complete`` call so the document is still parsed.
        - Honors ``self.vision_model``: when set (任务 10.7) the multimodal call is routed
          to that model via the gateway's ``model=`` kwarg, leaving the gateway-level default
          untouched. Text-only calls always use the gateway default.
        - Caps the raw text embedded in the prompt at ``self.max_raw_text_chars`` to avoid
          token-limit errors on noisy parsers. Truncation is hard, not soft, so the prompt
          is bounded regardless of upstream size.
        - Wraps the LLM call in ``asyncio.wait_for(timeout=self.page_timeout)`` so a single
          slow / hung model invocation cannot stall the whole document. On timeout, raises
          ``LLMGatewayError(reason="timeout")`` so the outer ``parse()`` degrades to plain
          text fallback (consistent with existing behavior in 10.1/10.2).
        - Strips surrounding ```markdown / ``` fences if the LLM ignored the system prompt
          and wrapped its output in a code block.
        - Treats an empty response as a hard failure (raises ``LLMGatewayError(reason="empty")``);
          this preserves the orchestration contract of ``parse()``, which only degrades when
          ``_parse_page`` raises. Returning ``success=False`` here would be silently merged
          into the document and the caller has no way to know that an entire page was lost.

        Args:
            page_number: Page number (1-indexed)
            raw_text: Raw text extracted from the page by the upstream native parser
            page_image: Optional page image (PNG bytes)

        Returns:
            PageResult with structured Markdown, headings, and tables

        Raises:
            LLMGatewayError: If the LLM call fails, times out, or returns empty content.
        """
        # Hard-truncate raw text to the configured budget. This is the single source of
        # truth for prompt size — never embed unbounded raw_text into the prompt.
        truncated_raw_text = raw_text[: self.max_raw_text_chars]
        prompt = PAGE_PROMPT_TEMPLATE.format(
            page_number=page_number,
            raw_text=truncated_raw_text,
        )

        try:
            if page_image:
                # Multimodal call. Pass ``model`` only when an explicit vision_model is
                # configured so default deployments keep using the gateway's own model.
                multimodal_kwargs: dict = {
                    "prompt": prompt,
                    "images": [page_image],
                    "system_prompt": SYSTEM_PROMPT,
                    "temperature": 0.1,
                    "max_tokens": 4096,
                }
                if self.vision_model:
                    multimodal_kwargs["model"] = self.vision_model

                response = await asyncio.wait_for(
                    self.llm.complete_multimodal(**multimodal_kwargs),
                    timeout=self.page_timeout,
                )
            else:
                # Text-only fallback for sources without a usable page image.
                # 任务 10.7：当 ``self.text_model`` 显式配置时透传给 gateway，
                # 否则不传 ``model``，让 gateway 用自身的默认模型。
                text_kwargs: dict = {
                    "prompt": prompt,
                    "system_prompt": SYSTEM_PROMPT,
                    "temperature": 0.1,
                    "max_tokens": 4096,
                }
                if self.text_model:
                    text_kwargs["model"] = self.text_model

                response = await asyncio.wait_for(
                    self.llm.complete(**text_kwargs),
                    timeout=self.page_timeout,
                )
        except asyncio.TimeoutError as e:
            # Re-raise as a domain error so the outer ``parse()`` degrade path picks it up.
            logger.warning(
                "Per-page LLM call timed out after %ss (page %s)",
                self.page_timeout,
                page_number,
            )
            raise LLMGatewayError(
                f"LLM call for page {page_number} timed out after "
                f"{self.page_timeout} seconds",
                reason="timeout",
            ) from e

        # Defend against unexpected gateway responses (None, missing attribute, non-string).
        # Anything that isn't a non-empty string is treated as a hard failure so the outer
        # ``parse()`` can degrade rather than silently emitting an empty page.
        content = getattr(response, "content", None)
        if not isinstance(content, str):
            raise LLMGatewayError(
                f"LLM returned non-string content for page {page_number}",
                reason="unknown",
            )

        markdown = self._strip_markdown_fence(content).strip()

        if not markdown:
            raise LLMGatewayError(
                f"LLM returned empty content for page {page_number}",
                reason="empty",
            )

        # Extract structured signals from the cleaned Markdown so downstream steps
        # (page merging, profile suggestion, quality scoring) don't have to re-parse it.
        headings = self._extract_headings_from_markdown(markdown)
        tables = self._extract_tables_from_markdown(markdown)

        return PageResult(
            page_number=page_number,
            markdown=markdown,
            headings=headings,
            tables=tables,
            success=True,
        )

    @staticmethod
    def _strip_markdown_fence(content: str) -> str:
        """Remove a single surrounding ```markdown / ``` fence if present.

        Some models — even when instructed otherwise — wrap their entire response in a
        Markdown code fence. Stripping here keeps the rest of the pipeline (heading
        extraction, table merging) free of accidental fence noise.

        Only strips when the *entire* content is wrapped; mid-document fences (e.g. an
        actual code block inside the page) are left untouched.
        """
        if not content:
            return content
        match = _FENCED_OUTPUT_RE.match(content)
        if match:
            return match.group("body")
        return content

    def _extract_headings_from_markdown(self, markdown: str) -> list[dict]:
        """Extract headings from Markdown content.

        Args:
            markdown: Markdown text

        Returns:
            List of dicts with 'level' and 'text' keys
        """
        headings = []
        for match in re.finditer(r"^(#{1,6})\s+(.+)$", markdown, re.MULTILINE):
            level = len(match.group(1))
            text = match.group(2).strip()
            headings.append({"level": level, "text": text})
        return headings

    def _extract_tables_from_markdown(self, markdown: str) -> list[str]:
        """Extract table blocks from Markdown content.

        Args:
            markdown: Markdown text

        Returns:
            List of table strings in Markdown format
        """
        tables = []
        lines = markdown.split("\n")
        current_table: list[str] = []
        in_table = False

        for line in lines:
            if "|" in line and line.strip().startswith("|"):
                in_table = True
                current_table.append(line)
            else:
                if in_table and current_table:
                    tables.append("\n".join(current_table))
                    current_table = []
                in_table = False

        # Don't forget last table
        if current_table:
            tables.append("\n".join(current_table))

        return tables

    def _merge_page_results(
        self, page_results: list[PageResult], metadata: dict
    ) -> ProcessedDocument:
        """Merge per-page LLM results into a single ProcessedDocument.

        任务 10.4 — Page-level consolidation contract:

        - Inputs are sorted defensively by ``page_number`` so callers may pass an
          unordered list (e.g. when concurrent page calls finish out of order).
        - ``PageResult`` entries with ``success is False`` or empty / whitespace-only
          ``markdown`` are skipped silently — they cannot contribute structure and
          would otherwise pollute the output.
        - Page numbers are preserved on every emitted block. The ``page_number``
          is taken from the ``PageResult`` itself so blocks remain anchored to the
          page where they were produced.
        - Short repeated paragraphs (``len(normalized) < 100``) that appear on a
          *different* page than their first occurrence are treated as boilerplate
          (running headers / footers / page numbers / "confidential" stamps) and
          removed. The first occurrence is kept on its original page.
          Long repeated paragraphs (``≥ 100`` chars) are likely real body content
          (boilerplate clauses, repeated definitions) and are preserved on every
          page where they appear.
        - After deduplication, ``_merge_cross_page_tables`` collapses adjacent-page
          tables that share an identical header into a single block anchored to
          the page where the table started.
        - ``noise_removed_count`` reflects the number of paragraphs dropped as
          short repeated boilerplate; ``headings_detected`` counts the surviving
          heading blocks.

        Args:
            page_results: List of per-page parsing results (any order).
            metadata: Document metadata, copied verbatim onto the output.

        Returns:
            Merged ``ProcessedDocument`` with merged Markdown and structural blocks.
        """
        all_blocks: list[ProcessedBlock] = []
        # Map: normalized paragraph text → page number where it was first seen.
        # Repeated short occurrences on *other* pages are dropped; same-page
        # repeats are preserved (they are not cross-page boilerplate).
        first_seen_page: dict[str, int] = {}
        noise_removed_count = 0

        sorted_results = sorted(page_results, key=lambda r: r.page_number)

        for result in sorted_results:
            if not result.success:
                continue
            if not result.markdown or not result.markdown.strip():
                continue

            page_blocks = self._markdown_to_blocks(
                result.markdown, result.page_number
            )

            for block in page_blocks:
                if block.type == "paragraph":
                    normalized = block.text.strip().lower()
                    if normalized:
                        prior_page = first_seen_page.get(normalized)
                        if prior_page is None:
                            first_seen_page[normalized] = block.page_number
                        elif (
                            prior_page != block.page_number
                            and len(normalized) < 100
                        ):
                            # Short, cross-page repeat → boilerplate. Drop it.
                            noise_removed_count += 1
                            continue
                        # else: same-page repeat OR long repeat → keep.

                all_blocks.append(block)

        # Merge cross-page tables (same header on consecutive pages).
        all_blocks = self._merge_cross_page_tables(all_blocks)

        # Generate combined Markdown
        markdown = self._blocks_to_markdown(all_blocks)

        headings_detected = sum(1 for b in all_blocks if b.type == "heading")

        return ProcessedDocument(
            blocks=all_blocks,
            metadata=metadata,
            markdown=markdown,
            noise_removed_count=noise_removed_count,
            headings_detected=headings_detected,
        )

    def _markdown_to_blocks(self, markdown: str, page_number: int) -> list[ProcessedBlock]:
        """Convert Markdown text to ProcessedBlock list.

        Args:
            markdown: Markdown content
            page_number: Page number for all blocks

        Returns:
            List of ProcessedBlocks
        """
        blocks: list[ProcessedBlock] = []
        lines = markdown.split("\n")
        current_table: list[str] = []
        in_table = False

        for line in lines:
            # Table detection
            if "|" in line and line.strip().startswith("|"):
                in_table = True
                current_table.append(line)
                continue
            elif in_table:
                # End of table
                if current_table:
                    blocks.append(ProcessedBlock(
                        type="table",
                        text="\n".join(current_table),
                        page_number=page_number,
                    ))
                    current_table = []
                in_table = False

            line = line.strip()
            if not line:
                continue

            # Heading detection
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if heading_match:
                level = len(heading_match.group(1))
                text = heading_match.group(2).strip()
                blocks.append(ProcessedBlock(
                    type="heading",
                    text=text,
                    heading_level=level,
                    page_number=page_number,
                ))
                continue

            # Regular paragraph
            blocks.append(ProcessedBlock(
                type="paragraph",
                text=line,
                page_number=page_number,
            ))

        # Flush remaining table
        if current_table:
            blocks.append(ProcessedBlock(
                type="table",
                text="\n".join(current_table),
                page_number=page_number,
            ))

        return blocks

    def _merge_cross_page_tables(
        self, blocks: list[ProcessedBlock]
    ) -> list[ProcessedBlock]:
        """Merge tables that span across consecutive pages.

        任务 10.4 — Cross-page table merging contract:

        - Two table blocks merge when:
            1. They share an identical header signature (see ``_get_table_header``),
               compared case-insensitively and tolerant of whitespace.
            2. The candidate table sits on ``prev_table.page_number + 1``. Tables
               on the same page or with a gap > 1 do **not** merge.
        - Merging is transitive: a table on pages 1, 2, 3 with the same header
          collapses into a single block, anchored to page 1.
        - The merged block keeps the first table's header + separator and appends
          the *data* rows (rows after the separator) from each continuation.
        - Empty continuations (header only, no data rows) extend the page-anchor
          state but contribute no rows.
        - Non-table blocks between two mergeable tables are preserved in their
          original relative order *after* the merged block.
        - When the page-number sequence breaks or the header changes, scanning
          stops and the next table starts a fresh merge group.

        Args:
            blocks: All blocks from all pages, in document order.

        Returns:
            New list of blocks with cross-page tables merged in place.
        """
        if not blocks:
            return list(blocks)

        # ``consumed`` records the indices of continuation tables that were folded
        # into an earlier table. We rewrite the anchor in-place and filter at the end.
        rewritten = list(blocks)
        consumed: set[int] = set()

        n = len(rewritten)
        for i in range(n):
            if i in consumed:
                continue
            block = rewritten[i]
            if block.type != "table":
                continue

            header = self._get_table_header(block.text)
            if not header:
                continue

            merged_text = block.text
            current_page = block.page_number

            # Scan forward for continuation tables. Non-table blocks are skipped
            # over (they remain in the output after this anchor). The page-number
            # invariant `cand.page_number == current_page + 1` is what enforces
            # adjacency: a non-adjacent table breaks the chain.
            for j in range(i + 1, n):
                if j in consumed:
                    continue
                cand = rewritten[j]
                if cand.type != "table":
                    continue

                if cand.page_number != current_page + 1:
                    # Page gap (or out-of-order) — stop extending this group.
                    break

                cand_header = self._get_table_header(cand.text)
                if not cand_header or cand_header != header:
                    break

                # Append data rows (everything after the header + separator).
                rows = cand.text.strip().split("\n")
                data_rows = rows[2:] if len(rows) > 2 else []
                if data_rows:
                    merged_text = merged_text + "\n" + "\n".join(data_rows)

                consumed.add(j)
                current_page = cand.page_number

            rewritten[i] = ProcessedBlock(
                type="table",
                text=merged_text,
                page_number=block.page_number,
            )

        return [b for idx, b in enumerate(rewritten) if idx not in consumed]

    def _get_table_header(self, table_text: str) -> str | None:
        """Extract a normalized header signature from a Markdown table.

        Used by ``_merge_cross_page_tables`` to compare headers across pages.
        Normalization is intentionally aggressive so visually identical headers
        survive whitespace and case differences:

        - leading / trailing pipe characters are dropped before splitting cells
        - each cell is trimmed of surrounding whitespace and lowercased
        - cells are rejoined with a single ``|`` separator

        Returns ``None`` when ``table_text`` does not contain a valid header
        line (no ``|`` characters), so the caller can skip non-table content.

        Args:
            table_text: Markdown table text (may include trailing rows).

        Returns:
            Normalized header signature string, or ``None``.
        """
        lines = table_text.strip().split("\n")
        if not lines:
            return None
        header = lines[0].strip()
        if "|" not in header:
            return None
        # Strip outer pipes so empty leading/trailing cells don't leak through.
        stripped = header.strip("|")
        cells = [cell.strip().lower() for cell in stripped.split("|")]
        return "|".join(cells)

    def _blocks_to_markdown(self, blocks: list[ProcessedBlock]) -> str:
        """Convert ProcessedBlocks back to Markdown string.

        Args:
            blocks: List of processed blocks

        Returns:
            Markdown string
        """
        parts: list[str] = []

        for block in blocks:
            if block.type == "heading":
                prefix = "#" * block.heading_level
                parts.append(f"{prefix} {block.text}")
                parts.append("")
            elif block.type == "table":
                parts.append(block.text)
                parts.append("")
            else:
                parts.append(block.text)
                parts.append("")

        markdown = "\n".join(parts)
        # Clean up multiple blank lines
        markdown = re.sub(r"\n{3,}", "\n\n", markdown)
        return markdown.strip()

    def _degrade_to_plain_text(self, parsed_doc: ParsedDocument) -> ProcessedDocument:
        """Fallback: extract plain text and create fixed-size chunks.

        Used when LLM parsing fails for **every** page (任务 10.8 — whole-doc
        degradation path). The per-page partial-success path uses
        ``_degrade_pages`` instead, which shares the underlying chunking via
        ``_chunk_text_to_char_blocks``.

        The chunk size is taken from ``self.fallback_chunk_chars`` (set from
        ``settings.UNIVERSAL_PARSER_FALLBACK_CHUNK_CHARS`` at construction time,
        with an optional constructor override).

        Args:
            parsed_doc: The original parsed document.

        Returns:
            ProcessedDocument with plain-text paragraph blocks and Markdown.
        """
        # Collect all text
        full_text = "\n".join(
            block.text for block in parsed_doc.blocks if block.text.strip()
        )

        if not full_text.strip():
            return ProcessedDocument(metadata=parsed_doc.metadata)

        chunk_size = self.fallback_chunk_chars
        # Split into fixed-size chunks. We iterate the chunk offsets directly
        # so we can map each chunk back to its original page via
        # ``_estimate_page_number``; ``_chunk_text_to_char_blocks`` returns
        # text without offsets so this loop owns the offset arithmetic.
        blocks: list[ProcessedBlock] = []
        for i in range(0, len(full_text), chunk_size):
            chunk_text = full_text[i : i + chunk_size].strip()
            if chunk_text:
                page_number = self._estimate_page_number(
                    parsed_doc.blocks, i, full_text
                )
                blocks.append(ProcessedBlock(
                    type="paragraph",
                    text=chunk_text,
                    page_number=page_number,
                ))

        markdown = "\n\n".join(b.text for b in blocks)

        return ProcessedDocument(
            blocks=blocks,
            metadata=parsed_doc.metadata,
            markdown=markdown,
            noise_removed_count=0,
            headings_detected=0,
        )

    @staticmethod
    def _chunk_text_to_char_blocks(text: str, chunk_chars: int) -> list[str]:
        """Split ``text`` into stripped chunks of at most ``chunk_chars`` characters.

        Shared chunking primitive used by both the whole-doc fallback
        (``_degrade_to_plain_text``) and the per-page degradation
        (``_degrade_pages``). Empty / whitespace-only chunks are dropped so the
        output never contains blank blocks.

        ``chunk_chars`` is floored at 1 by the caller; this helper trusts the
        invariant (no extra defensive check on the hot path).
        """
        if not text or chunk_chars < 1:
            return []
        chunks: list[str] = []
        for i in range(0, len(text), chunk_chars):
            piece = text[i : i + chunk_chars].strip()
            if piece:
                chunks.append(piece)
        return chunks

    def _estimate_page_number(
        self, blocks: list[Block], char_offset: int, full_text: str
    ) -> int:
        """Estimate the page number for a character offset in the full text.

        Args:
            blocks: Original blocks with page numbers
            char_offset: Character offset in the concatenated text
            full_text: The full concatenated text

        Returns:
            Estimated page number
        """
        if not blocks:
            return 1

        # Simple estimation: map character offset to block index
        current_offset = 0
        for block in blocks:
            block_len = len(block.text) + 1  # +1 for newline
            if current_offset + block_len > char_offset:
                return block.page_number
            current_offset += block_len

        return blocks[-1].page_number

    # ─── Profile Suggestion Helpers ───────────────────────────────────

    # Heading numbering patterns we recognize in candidate profiles. Order matters:
    # when more than one pattern clears the match threshold we prefer the most
    # specific (longest pattern), since e.g. "Chapter 12" also satisfies the
    # decimal-section pattern but the chapter pattern is the better candidate.
    _HEADING_PATTERN_CATALOG: tuple[tuple[str, str], ...] = (
        (r"^[一二三四五六七八九十]+[、.]", "Chinese numbering"),
        (r"^\([一二三四五六七八九十]+\)", "Chinese parenthetical"),
        (r"^\d+[.、]", "Numeric"),
        (r"^\(\d+\)", "Numeric parenthetical"),
        (r"^[①②③④⑤⑥⑦⑧⑨⑩]", "Circled numbers"),
        (r"^[A-Z]\.", "Letter numbering"),
        # 任务 10.5: chapter-style headings ("Chapter 1", "第3章").
        (r"^(?:Chapter|第)\s*\d+\s*(?:章)?", "Chapter numbering"),
    )

    # Match threshold for a pattern to be emitted as a candidate heading rule.
    # 60% means a single stray heading at the same level can no longer pollute
    # an otherwise-uniform group.
    _HEADING_MATCH_THRESHOLD = 0.6

    # Boilerplate "shortcut" regexes — when a short paragraph matches any of
    # these on at least 2 pages we emit it even if it does not clear the
    # frequency threshold below. These cover the obvious page-marker shapes
    # that still leak through after LLM noise removal.
    _BOILERPLATE_SHORTCUT_REGEXES: tuple[re.Pattern[str], ...] = (
        re.compile(r"^page\s*\d+$", re.IGNORECASE),
        re.compile(r"^\d+\s*/\s*\d+$"),
        re.compile(r"^confidential", re.IGNORECASE),
        re.compile(r"copyright", re.IGNORECASE),
    )

    # Default chunking ``protect_patterns``: cross-page numeric atoms that the
    # downstream chunker MUST keep intact (units, tolerances, formula deltas).
    # Always emitted on candidate profiles so reviewers don't have to hand-add
    # them, and so the chunker does not split numbers like "30 mm" mid-token.
    _DEFAULT_PROTECT_PATTERNS: tuple[str, ...] = (
        r"\d+(?:\.\d+)?\s*(?:mm|cm|m|kg|g|°|%)",
        r"[△▽]\s*=\s*[\d.]+",
        r"±\s*[\d.]+",
    )

    def _extract_heading_patterns(self, result: ProcessedDocument) -> list[dict]:
        """Extract heading patterns from processed document for profile suggestion.

        Headings are grouped by ``heading_level`` (1..6); each group is fed to
        ``_infer_heading_pattern`` which returns the most-specific numbering
        pattern that ≥ 60% of the group satisfies, or ``None`` if no pattern
        clears the threshold.

        Args:
            result: Processed document

        Returns:
            List of ``HeadingRule``-shaped dicts ordered by heading level.
        """
        heading_rules: list[dict] = []
        heading_texts_by_level: dict[int, list[str]] = {}

        for block in result.blocks:
            if block.type == "heading" and block.heading_level > 0:
                heading_texts_by_level.setdefault(block.heading_level, []).append(
                    block.text
                )

        for level, texts in sorted(heading_texts_by_level.items()):
            pattern = self._infer_heading_pattern(texts)
            if pattern:
                heading_rules.append({
                    "pattern": pattern,
                    "level": level,
                    "strip_pattern": False,
                })

        return heading_rules

    def _infer_heading_pattern(self, texts: list[str]) -> str | None:
        """Infer a regex pattern from a list of heading texts.

        Checks each pattern in ``_HEADING_PATTERN_CATALOG`` and emits the most
        specific one that matches at least 60% of the input texts. "Most
        specific" is approximated by pattern length — longer regexes generally
        impose more constraints than shorter ones.

        Args:
            texts: List of heading texts at the same level.

        Returns:
            Regex pattern string or ``None`` when nothing clears the threshold.
        """
        if not texts:
            return None

        threshold = math.ceil(len(texts) * self._HEADING_MATCH_THRESHOLD)
        # ``ceil(60%)`` ensures small groups (1–2 headings) require all of them
        # to match before we commit to a pattern.

        winners: list[str] = []
        for pattern, _name in self._HEADING_PATTERN_CATALOG:
            try:
                compiled = re.compile(pattern)
            except re.error:
                continue
            matches = sum(1 for t in texts if compiled.match(t))
            if matches >= threshold:
                winners.append(pattern)

        if not winners:
            return None

        # Prefer the most specific (longest) pattern so e.g. ``Chapter \d+``
        # wins over a generic decimal-section regex when both apply.
        winners.sort(key=len, reverse=True)
        return winners[0]

    def _extract_noise_patterns(self, result: ProcessedDocument) -> list[str]:
        r"""Extract noise patterns from processed document.

        A short paragraph (``len(stripped) < 100``) is emitted as a candidate
        boilerplate pattern when either:

        - it appears on at least ``max(3, ceil(0.3 * total_pages))`` distinct
          pages (cross-page repetition is a strong boilerplate signal), OR
        - it matches one of the obvious-boilerplate "shortcut" regexes
          (``page \d+``, ``\d+/\d+``, ``confidential``, ``copyright``) on at
          least 2 pages.

        Output is capped at 20 patterns and sorted by descending page count so
        the most-impactful patterns come first — `manual_patterns` is matched
        line-by-line downstream, so order influences match-cost on hot paths.

        Args:
            result: Processed document.

        Returns:
            List of regex patterns (each a regex-escaped, anchored string).
        """
        # Map: normalized short paragraph text → set of pages it appears on.
        text_pages: dict[str, set[int]] = {}
        for block in result.blocks:
            if block.type != "paragraph":
                continue
            normalized = (block.text or "").strip()
            if not normalized or len(normalized) >= 100:
                continue
            text_pages.setdefault(normalized, set()).add(block.page_number)

        if not text_pages:
            return []

        # Total page count drives the dynamic threshold: small documents (≤3
        # pages) keep the floor of 3 pages; larger documents require 30% of
        # pages so a footer that appears on 15 of 50 pages still wins.
        all_pages: set[int] = set()
        for pages in text_pages.values():
            all_pages.update(pages)
        total_pages = len(all_pages) or 1
        frequency_threshold = max(3, math.ceil(0.3 * total_pages))

        scored: list[tuple[int, str]] = []
        for text, pages in text_pages.items():
            page_count = len(pages)

            # Frequency wins automatically.
            if page_count >= frequency_threshold:
                scored.append((page_count, text))
                continue

            # Shortcut regex match: emit even on 2 pages so obvious page-number
            # noise survives short documents.
            if page_count >= 2 and any(
                regex.match(text) for regex in self._BOILERPLATE_SHORTCUT_REGEXES
            ):
                scored.append((page_count, text))

        if not scored:
            return []

        # Stable, deterministic order: highest page count first, ties broken
        # alphabetically so the output is reproducible across runs.
        scored.sort(key=lambda item: (-item[0], item[1]))

        patterns = [f"^{re.escape(text)}$" for _count, text in scored]
        return patterns[:20]

    def _recommend_chunking(self, result: ProcessedDocument) -> dict:
        """Recommend chunking parameters based on document structure.

        Decision tree:

        - ``heading_count > 20`` (dense): respect H1, ``min=384`` / ``max=1024``,
          overlap 96.
        - ``heading_count < 5`` (sparse): respect H3, ``min=256`` / ``max=800``,
          overlap 80.
        - otherwise: respect H2, ``min=256`` / ``max=800``, overlap 80.

        Override: when the average block size is below 100 characters the
        document is mostly short snippets (slides, list-heavy material), so we
        halve ``min``/``max`` while keeping the dense / sparse / default
        ``respect_heading_level`` decision intact.

        Always includes ``protect_patterns`` covering common units, tolerances,
        and formula deltas so the chunker never splits numeric atoms.

        Args:
            result: Processed document.

        Returns:
            Dict matching ``ChunkingConfig`` shape.
        """
        total_chars = sum(len(b.text or "") for b in result.blocks)
        total_blocks = len(result.blocks)
        heading_count = sum(1 for b in result.blocks if b.type == "heading")

        avg_block_chars = total_chars / total_blocks if total_blocks else 0.0

        if heading_count > 20:
            respect_heading_level = 1
            min_tokens = 384
            max_tokens = 1024
            overlap_tokens = 96
        elif heading_count < 5:
            respect_heading_level = 3
            min_tokens = 256
            max_tokens = 800
            overlap_tokens = 80
        else:
            respect_heading_level = 2
            min_tokens = 256
            max_tokens = 800
            overlap_tokens = 80

        if total_blocks > 0 and avg_block_chars < 100:
            # Halve token budgets for short-block documents but preserve the
            # dense / sparse / default heading-level choice.
            min_tokens //= 2
            max_tokens //= 2

        return {
            "min_tokens": min_tokens,
            "max_tokens": max_tokens,
            "overlap_tokens": overlap_tokens,
            "respect_heading_level": respect_heading_level,
            "protect_patterns": list(self._DEFAULT_PROTECT_PATTERNS),
        }

    def _recommend_tables(self, result: ProcessedDocument) -> dict:
        """Recommend ``TableConfig`` settings based on table density.

        ``row_level_chunking`` is enabled when at least one table has more than
        20 data rows, since large tables benefit from per-row chunking
        (otherwise they overflow the chunker's max-token budget). For
        documents without large tables we keep row-level chunking off so small
        tables stay together as a single block.

        Args:
            result: Processed document.

        Returns:
            Dict matching ``TableConfig`` shape.
        """
        large_table_present = False
        for block in result.blocks:
            if block.type != "table":
                continue
            row_count = self._count_table_rows(block.text or "")
            if row_count > 20:
                large_table_present = True
                break

        return {
            "cross_page_merge": True,
            "row_level_chunking": large_table_present,
            "collapse_merged_cells": "describe",
        }

    @staticmethod
    def _count_table_rows(table_text: str) -> int:
        """Count the data rows in a Markdown table.

        Skips the header row and the ``| --- |`` separator row, and ignores
        empty lines. Returns 0 when ``table_text`` does not look like a table.
        """
        lines = [
            line for line in (table_text or "").splitlines() if line.strip()
        ]
        if len(lines) < 2:
            return 0
        # Standard Markdown table: header + separator + data rows.
        # Defensive: treat any line whose non-pipe characters are only ``-``,
        # ``:`` and whitespace as a separator.
        def _is_separator(line: str) -> bool:
            stripped = line.strip().strip("|")
            cells = [c.strip() for c in stripped.split("|") if c.strip()]
            if not cells:
                return False
            return all(set(c) <= set("-: ") for c in cells)

        # Header line must contain pipes.
        if "|" not in lines[0]:
            return 0

        if len(lines) >= 2 and _is_separator(lines[1]):
            return max(0, len(lines) - 2)
        # No separator detected: treat first line as header, rest as data.
        return max(0, len(lines) - 1)
