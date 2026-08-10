"""Focused tests for ``IndexingService.index_chunks`` dual-write transaction (任务 12.7).

These tests harden the dual-write contract between Qdrant and OpenSearch:

1. Both writes succeed → returns counts and never deletes anything.
2. Qdrant fails → ``IndexingError`` raised, OpenSearch never contacted,
   ``_delete_qdrant_points`` never called (nothing was committed).
3. OpenSearch fails after Qdrant succeeded → rollback triggered with the
   exact set of point IDs that was just written.
4. Rollback itself fails → original OpenSearch error still surfaces as
   ``IndexingError``; rollback failure is logged and reflected in the
   error message so operators can reconcile orphaned points.
5. Empty input → no-op, no client calls.
6. Mismatched payload/embedding lengths → ``IndexingError`` raised
   **before** any backend write.

These complement the existing ``test_indexing_qdrant_write.py`` and
``test_indexing_opensearch_bulk.py`` modules — those exercise each side in
isolation; this module exercises their interaction.

Validates: Requirements 4
"""

from __future__ import annotations

import logging
import uuid
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ``qdrant_client`` is an optional dev dependency; reuse the lightweight
# stand-ins installed by the Qdrant-side test module so importing
# ``app.services.indexing_service`` doesn't require the real SDK.
from tests import test_indexing_qdrant_write as _qd_stubs  # noqa: F401

from app.services.embedding_service import EmbeddingResult  # noqa: E402
from app.services.indexing_service import (  # noqa: E402
    ChunkPayload,
    IndexingError,
    IndexingService,
)


DENSE_DIM = 1024


# ─── helpers ─────────────────────────────────────────────────────────


def _make_payload(chunk_id: str | None = None, **overrides: Any) -> ChunkPayload:
    base: dict[str, Any] = dict(
        chunk_id=chunk_id or str(uuid.uuid4()),
        document_id=str(uuid.uuid4()),
        space_id=str(uuid.uuid4()),
        chunk_index=0,
        title_chain="H1 > H2",
        source_file="manual.pdf",
        page_number=1,
        content="hello world",
        parent_chunk_id=None,
        depth=2,
        token_count=12,
        allowed_user_ids=[str(uuid.uuid4())],
        access_level="read",
    )
    base.update(overrides)
    return ChunkPayload(**base)


def _make_embedding(chunk_id: str) -> EmbeddingResult:
    return EmbeddingResult(
        chunk_id=chunk_id,
        dense_vector=[0.01] * DENSE_DIM,
        sparse_indices=[3, 17],
        sparse_values=[0.5, 0.7],
    )


def _make_service() -> IndexingService:
    """Build an IndexingService with both backend slots pre-populated by
    mocks so lazy initialization never reaches the real client factories."""
    service = IndexingService()
    service._qdrant = MagicMock(name="qdrant_client")
    service._opensearch = MagicMock(name="opensearch_client")
    return service


def _patch_qdrant_upsert(service: IndexingService, side_effect=None):
    """Patch the per-call Qdrant upsert helper so we can assert / fail it
    without going through the SDK stub."""
    return patch.object(service, "_upsert_qdrant", side_effect=side_effect)


def _patch_opensearch_bulk(service: IndexingService, side_effect=None):
    return patch.object(service, "_bulk_index_opensearch", side_effect=side_effect)


def _patch_rollback(service: IndexingService, side_effect=None):
    return patch.object(service, "_delete_qdrant_points", side_effect=side_effect)


# ─── 1. happy path ──────────────────────────────────────────────────


def test_index_chunks_both_succeed_returns_counts_and_no_rollback():
    """Both writes succeed → counts equal len(payloads), rollback is never called."""
    service = _make_service()

    payloads = [_make_payload() for _ in range(3)]
    embeddings = [_make_embedding(p.chunk_id) for p in payloads]

    with _patch_qdrant_upsert(service) as mock_qd, \
         _patch_opensearch_bulk(service) as mock_os, \
         _patch_rollback(service) as mock_rollback:
        # Qdrant returns the count of points it indexed.
        mock_qd.return_value = len(payloads)
        # OpenSearch bulk is a no-op on success.
        mock_os.return_value = None

        result = service.index_chunks(payloads, embeddings)

    assert result == {"qdrant_count": 3, "opensearch_count": 3}
    mock_qd.assert_called_once_with(payloads, embeddings)
    mock_os.assert_called_once_with(payloads)
    mock_rollback.assert_not_called()


# ─── 2. Qdrant fails → no OpenSearch attempt, no rollback ───────────


def test_index_chunks_qdrant_failure_skips_opensearch_and_rollback():
    """When the very first write (Qdrant) fails, OpenSearch must not be
    contacted and rollback must not run — there is nothing to undo."""
    service = _make_service()

    payloads = [_make_payload()]
    embeddings = [_make_embedding(payloads[0].chunk_id)]

    with _patch_qdrant_upsert(service, side_effect=RuntimeError("boom")) as mock_qd, \
         _patch_opensearch_bulk(service) as mock_os, \
         _patch_rollback(service) as mock_rollback:
        with pytest.raises(IndexingError, match="Qdrant write failed"):
            service.index_chunks(payloads, embeddings)

    mock_qd.assert_called_once()
    mock_os.assert_not_called()
    mock_rollback.assert_not_called()


def test_index_chunks_qdrant_validation_error_propagates_unchanged():
    """An ``IndexingError`` raised by ``_upsert_qdrant`` (e.g. non-UUID
    chunk_id) should surface as-is, *not* be re-wrapped, and must not
    trigger an OpenSearch call or rollback."""
    service = _make_service()

    payloads = [_make_payload()]
    embeddings = [_make_embedding(payloads[0].chunk_id)]

    original = IndexingError("chunk_id 'xyz' is not a valid UUID")

    with _patch_qdrant_upsert(service, side_effect=original), \
         _patch_opensearch_bulk(service) as mock_os, \
         _patch_rollback(service) as mock_rollback:
        with pytest.raises(IndexingError) as exc_info:
            service.index_chunks(payloads, embeddings)

    # The message must be the original one (not re-wrapped as "Qdrant write failed: ...").
    assert "not a valid UUID" in str(exc_info.value)
    assert "Qdrant write failed" not in str(exc_info.value)
    mock_os.assert_not_called()
    mock_rollback.assert_not_called()


# ─── 3. OpenSearch fails after Qdrant → rollback with correct IDs ───


def test_index_chunks_opensearch_failure_rolls_back_qdrant_with_exact_ids():
    """OpenSearch fails after Qdrant succeeded → rollback receives the
    exact list of chunk IDs that were just written, in order."""
    service = _make_service()

    payloads = [_make_payload() for _ in range(4)]
    embeddings = [_make_embedding(p.chunk_id) for p in payloads]
    expected_ids = [p.chunk_id for p in payloads]

    with _patch_qdrant_upsert(service) as mock_qd, \
         _patch_opensearch_bulk(service, side_effect=RuntimeError("opensearch down")), \
         _patch_rollback(service) as mock_rollback:
        mock_qd.return_value = len(payloads)

        with pytest.raises(IndexingError) as exc_info:
            service.index_chunks(payloads, embeddings)

    assert "OpenSearch write failed" in str(exc_info.value)
    assert "Qdrant rolled back" in str(exc_info.value)
    mock_rollback.assert_called_once_with(expected_ids)


def test_index_chunks_opensearch_failure_preserves_original_error_via_cause():
    """The raised ``IndexingError`` should chain the original OpenSearch
    exception as its ``__cause__`` so logs and tracebacks point at the
    real failure (not at our wrapper)."""
    service = _make_service()

    payloads = [_make_payload()]
    embeddings = [_make_embedding(payloads[0].chunk_id)]
    original = RuntimeError("bulk reported errors")

    with patch.object(service, "_upsert_qdrant", return_value=1), \
         _patch_opensearch_bulk(service, side_effect=original), \
         _patch_rollback(service):
        with pytest.raises(IndexingError) as exc_info:
            service.index_chunks(payloads, embeddings)

    assert exc_info.value.__cause__ is original


# ─── 4. rollback failure → original error still raised ──────────────


def test_index_chunks_rollback_failure_still_raises_original_error(caplog):
    """If rollback itself fails, we must still raise an ``IndexingError``
    describing the OpenSearch failure. The rollback error is logged and
    surfaced in the message so operators can reconcile orphaned Qdrant
    points, but it must not mask the root cause."""
    service = _make_service()

    payloads = [_make_payload(), _make_payload()]
    embeddings = [_make_embedding(p.chunk_id) for p in payloads]

    opensearch_err = RuntimeError("opensearch 503")
    rollback_err = RuntimeError("qdrant delete refused")

    with patch.object(service, "_upsert_qdrant", return_value=2), \
         _patch_opensearch_bulk(service, side_effect=opensearch_err), \
         _patch_rollback(service, side_effect=rollback_err) as mock_rollback, \
         caplog.at_level(logging.ERROR, logger="app.services.indexing_service"):
        with pytest.raises(IndexingError) as exc_info:
            service.index_chunks(payloads, embeddings)

    # Rollback was attempted.
    mock_rollback.assert_called_once()

    msg = str(exc_info.value)
    # The original OpenSearch error is the headline.
    assert "OpenSearch write failed" in msg
    assert "opensearch 503" in msg
    # The rollback failure is reflected in the message (not as the cause).
    assert "Qdrant rollback FAILED" in msg
    assert "qdrant delete refused" in msg
    # And the cause is the original OpenSearch error, not the rollback error.
    assert exc_info.value.__cause__ is opensearch_err

    # The rollback failure was logged at ERROR level.
    assert any(
        "Qdrant rollback failed" in rec.getMessage()
        for rec in caplog.records
    )


# ─── 5. empty / no-op input ─────────────────────────────────────────


def test_index_chunks_empty_payloads_is_noop():
    """Empty payloads → no client calls, returns zero counts."""
    service = _make_service()

    with _patch_qdrant_upsert(service) as mock_qd, \
         _patch_opensearch_bulk(service) as mock_os, \
         _patch_rollback(service) as mock_rollback:
        result = service.index_chunks([], [])

    assert result == {"qdrant_count": 0, "opensearch_count": 0}
    mock_qd.assert_not_called()
    mock_os.assert_not_called()
    mock_rollback.assert_not_called()


def test_index_chunks_empty_embeddings_is_noop():
    """Empty embeddings (with non-empty payloads) is also a no-op — we
    treat both being empty as the canonical empty case, but defensively
    short-circuit if either side is empty so we don't fall into the
    length-mismatch branch and raise spuriously."""
    service = _make_service()

    with _patch_qdrant_upsert(service) as mock_qd, \
         _patch_opensearch_bulk(service) as mock_os:
        result = service.index_chunks([_make_payload()], [])

    assert result == {"qdrant_count": 0, "opensearch_count": 0}
    mock_qd.assert_not_called()
    mock_os.assert_not_called()


# ─── 6. mismatched lengths → fail before any write ──────────────────


def test_index_chunks_mismatched_lengths_fail_before_any_backend_write():
    """Length mismatch is a programming error. We must reject the call
    before contacting either backend so we never leave one store
    partially populated."""
    service = _make_service()

    payloads = [_make_payload(), _make_payload()]
    embeddings = [_make_embedding(payloads[0].chunk_id)]  # 1 vs 2 — mismatch

    with _patch_qdrant_upsert(service) as mock_qd, \
         _patch_opensearch_bulk(service) as mock_os, \
         _patch_rollback(service) as mock_rollback:
        with pytest.raises(IndexingError, match="Payload count"):
            service.index_chunks(payloads, embeddings)

    mock_qd.assert_not_called()
    mock_os.assert_not_called()
    mock_rollback.assert_not_called()
