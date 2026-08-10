"""Focused tests for ``IndexingService._bulk_index_opensearch`` (任务 12.6).

These tests harden the OpenSearch-side write path:

- Empty payload list → no bulk call (and therefore no network round-trip).
- Single payload → exactly one ``bulk`` call whose ``_source`` contains
  every field defined by the OpenSearch index model in ``design.md``,
  with ``_id`` equal to ``chunk_id`` so the write is an idempotent upsert.
- ``len(payloads) > OPENSEARCH_BATCH_SIZE`` → the helper is called once
  per batch with at most ``OPENSEARCH_BATCH_SIZE`` actions per call, all
  with ``refresh=True``.
- Bulk reports per-item errors → ``IndexingError`` is raised and the
  error message reports the concrete error/success counts so callers
  can attribute the failure accurately.
- Partial-error batch → only the failed items are counted toward the
  error message; successful items contribute to the reported success
  count.

Validates: Requirements 4
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ``qdrant_client`` is an optional dev dependency; ``app.services.indexing_service``
# imports it eagerly via ``app.core.qdrant``. The Qdrant-side test module
# installs lightweight stand-ins that satisfy that import — reuse them here so
# the OpenSearch-side tests don't pull in the real SDK either.
from tests import test_indexing_qdrant_write as _qd_stubs  # noqa: F401

from app.core.opensearch import INDEX_NAME as OS_INDEX_NAME  # noqa: E402
from app.services.indexing_service import (  # noqa: E402
    OPENSEARCH_BATCH_SIZE,
    ChunkPayload,
    IndexingError,
    IndexingService,
)


# ─── helpers ─────────────────────────────────────────────────────────


def _make_payload(**overrides: Any) -> ChunkPayload:
    """Build a ``ChunkPayload`` with every field populated.

    The defaults mimic what the upstream chunker emits so the assertions
    on ``_source`` exercise non-trivial values for every key in the
    design.md OpenSearch model.
    """
    base: dict[str, Any] = dict(
        chunk_id=str(uuid.uuid4()),
        document_id=str(uuid.uuid4()),
        space_id=str(uuid.uuid4()),
        chunk_index=0,
        title_chain="第一章 > 1.1 概述",
        source_file="handbook.pdf",
        page_number=42,
        content="这是一段需要被索引的中文内容 with mixed English",
        parent_chunk_id=None,
        depth=2,
        token_count=128,
        allowed_user_ids=[str(uuid.uuid4()), str(uuid.uuid4())],
        access_level="read",
    )
    base.update(overrides)
    return ChunkPayload(**base)


def _make_service() -> IndexingService:
    service = IndexingService()
    service._opensearch = MagicMock(name="opensearch_client")
    # Qdrant is irrelevant for these tests; populate the slot anyway so
    # accidental access doesn't trigger lazy initialization.
    service._qdrant = MagicMock(name="qdrant_client")
    return service


# ─── empty-input contract ───────────────────────────────────────────


def test_bulk_index_empty_payload_is_noop_and_does_not_call_bulk():
    """Empty list must not open a bulk connection or refresh the index."""
    service = _make_service()

    with patch("opensearchpy.helpers.bulk") as mock_bulk:
        service._bulk_index_opensearch([])

    mock_bulk.assert_not_called()
    # Sanity: no methods on the client either.
    service._opensearch.assert_not_called()


# ─── single-payload _source shape ───────────────────────────────────


def test_bulk_index_single_payload_writes_design_md_fields():
    """One payload → one bulk call, one action, ``_source`` matches design.md."""
    service = _make_service()
    payload = _make_payload(
        chunk_index=7,
        title_chain="第一章 > 1.1 > 1.1.2",
        source_file="manual.pdf",
        page_number=11,
        content="完整的 chunk 文本内容",
    )

    with patch(
        "opensearchpy.helpers.bulk", return_value=(1, [])
    ) as mock_bulk:
        service._bulk_index_opensearch([payload])

    assert mock_bulk.call_count == 1
    args, kwargs = mock_bulk.call_args

    # Positional contract: bulk(client, actions, ...)
    assert args[0] is service._opensearch
    actions = list(args[1])
    assert len(actions) == 1

    # refresh=True so subsequent searches see the new document immediately.
    assert kwargs["refresh"] is True

    action = actions[0]
    assert action["_index"] == OS_INDEX_NAME
    # _id must equal chunk_id so re-indexing the same chunk is an upsert,
    # not a duplicate. This is the source of OpenSearch idempotency.
    assert action["_id"] == payload.chunk_id

    source = action["_source"]
    expected_keys = {
        "chunk_id",
        "document_id",
        "space_id",
        "content",
        "title_chain",
        "source_file",
        "page_number",
        "chunk_index",
        "allowed_user_ids",
        "created_at",
    }
    assert set(source.keys()) == expected_keys

    assert source["chunk_id"] == payload.chunk_id
    assert source["document_id"] == payload.document_id
    assert source["space_id"] == payload.space_id
    assert source["content"] == payload.content
    assert source["title_chain"] == payload.title_chain
    assert source["source_file"] == payload.source_file
    assert source["page_number"] == payload.page_number
    assert source["chunk_index"] == payload.chunk_index
    assert source["allowed_user_ids"] == list(payload.allowed_user_ids)

    # created_at must be a parseable ISO-8601 timestamp.
    parsed = datetime.fromisoformat(source["created_at"])
    assert parsed.tzinfo is not None  # timezone-aware


def test_bulk_index_uses_chunk_id_as_id_for_idempotent_upsert():
    """Re-indexing a chunk must produce the same ``_id`` so OpenSearch
    treats the second write as an in-place update."""
    service = _make_service()
    chunk_id = str(uuid.uuid4())
    payload = _make_payload(chunk_id=chunk_id)

    with patch("opensearchpy.helpers.bulk", return_value=(1, [])) as mock_bulk:
        # First write
        service._bulk_index_opensearch([payload])
        # Second write — same chunk_id, different content
        payload_again = _make_payload(chunk_id=chunk_id, content="updated content")
        service._bulk_index_opensearch([payload_again])

    assert mock_bulk.call_count == 2
    first_action = list(mock_bulk.call_args_list[0].args[1])[0]
    second_action = list(mock_bulk.call_args_list[1].args[1])[0]
    assert first_action["_id"] == second_action["_id"] == chunk_id


# ─── batching ───────────────────────────────────────────────────────


def test_bulk_index_splits_into_batches_of_opensearch_batch_size():
    """``len(payloads) > OPENSEARCH_BATCH_SIZE`` → multiple bulk calls,
    each carrying at most ``OPENSEARCH_BATCH_SIZE`` actions."""
    service = _make_service()

    n = OPENSEARCH_BATCH_SIZE * 2 + 13
    payloads = [_make_payload() for _ in range(n)]

    with patch(
        "opensearchpy.helpers.bulk", return_value=(OPENSEARCH_BATCH_SIZE, [])
    ) as mock_bulk:
        service._bulk_index_opensearch(payloads)

    expected_calls = (n + OPENSEARCH_BATCH_SIZE - 1) // OPENSEARCH_BATCH_SIZE
    assert mock_bulk.call_count == expected_calls

    seen_ids: list[str] = []
    for call in mock_bulk.call_args_list:
        actions = list(call.args[1])
        assert 1 <= len(actions) <= OPENSEARCH_BATCH_SIZE
        # refresh=True must be set on every batch, not just the last one.
        assert call.kwargs["refresh"] is True
        seen_ids.extend(action["_id"] for action in actions)

    # IDs across all batches must reproduce the input order exactly,
    # ensuring no payload is dropped or duplicated by the slicing logic.
    assert seen_ids == [p.chunk_id for p in payloads]


def test_bulk_index_exact_batch_size_emits_single_call():
    """Boundary: exactly ``OPENSEARCH_BATCH_SIZE`` payloads → one bulk call."""
    service = _make_service()
    payloads = [_make_payload() for _ in range(OPENSEARCH_BATCH_SIZE)]

    with patch(
        "opensearchpy.helpers.bulk",
        return_value=(OPENSEARCH_BATCH_SIZE, []),
    ) as mock_bulk:
        service._bulk_index_opensearch(payloads)

    assert mock_bulk.call_count == 1
    actions = list(mock_bulk.call_args_list[0].args[1])
    assert len(actions) == OPENSEARCH_BATCH_SIZE


# ─── error reporting ────────────────────────────────────────────────


def test_bulk_index_raises_indexing_error_with_concrete_counts_when_all_fail():
    """When every action fails, the error message must report the failure
    count so the caller can log and surface it to the pipeline."""
    service = _make_service()
    payloads = [_make_payload() for _ in range(3)]

    fake_errors = [
        {"index": {"_id": p.chunk_id, "error": "mapper_parsing_exception"}}
        for p in payloads
    ]

    with patch(
        "opensearchpy.helpers.bulk", return_value=(0, fake_errors)
    ):
        with pytest.raises(IndexingError) as exc_info:
            service._bulk_index_opensearch(payloads)

    message = str(exc_info.value)
    assert "3 error" in message
    assert "0 succeeded" in message


def test_bulk_index_partial_errors_count_only_failed_items():
    """Mixed success/failure batch: error message counts ONLY the failed
    items, not the entire batch."""
    service = _make_service()
    payloads = [_make_payload() for _ in range(5)]

    # Simulate 2 of 5 actions failing.
    fake_errors = [
        {"index": {"_id": payloads[1].chunk_id, "error": "version_conflict"}},
        {"index": {"_id": payloads[4].chunk_id, "error": "mapper_parsing"}},
    ]

    with patch(
        "opensearchpy.helpers.bulk", return_value=(3, fake_errors)
    ):
        with pytest.raises(IndexingError) as exc_info:
            service._bulk_index_opensearch(payloads)

    message = str(exc_info.value)
    # Exactly 2 errors, not 5.
    assert "2 error" in message
    # 3 successes contribute to the success count.
    assert "3 succeeded" in message
    # The total action count (5) is also surfaced for context.
    assert "5 action" in message


def test_bulk_index_errors_in_later_batch_still_raise_indexing_error():
    """Error in batch 2 must still raise even though batch 1 succeeded —
    we don't silently ignore later failures."""
    service = _make_service()
    payloads = [_make_payload() for _ in range(OPENSEARCH_BATCH_SIZE + 2)]

    # First batch succeeds entirely, second batch has 1 failure.
    with patch(
        "opensearchpy.helpers.bulk",
        side_effect=[
            (OPENSEARCH_BATCH_SIZE, []),
            (1, [{"index": {"_id": "x", "error": "boom"}}]),
        ],
    ):
        with pytest.raises(IndexingError) as exc_info:
            service._bulk_index_opensearch(payloads)

    message = str(exc_info.value)
    assert "1 error" in message
    # Both successful items (100 + 1) are counted.
    assert f"{OPENSEARCH_BATCH_SIZE + 1} succeeded" in message
