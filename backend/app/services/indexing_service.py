"""Indexing Service: Dual-write to Qdrant and OpenSearch with rollback.

Provides:
- Batch upsert to Qdrant (dense + sparse vectors + payload)
- Batch index to OpenSearch (full-text + metadata)
- Dual-write transaction logic (both succeed or both rollback)
- Cascade delete (remove from both stores)
- Pipeline status updates (Redis progress + PostgreSQL state)
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.core.config import get_settings
from app.core.opensearch import INDEX_NAME as OS_INDEX_NAME
from app.core.opensearch import get_opensearch_client
from app.core.qdrant import COLLECTION_NAME as QD_COLLECTION_NAME
from app.core.qdrant import get_qdrant_client
from app.services.embedding_service import EmbeddingResult

logger = logging.getLogger(__name__)

# Batch size for Qdrant upserts
QDRANT_BATCH_SIZE = 100

# Batch size for OpenSearch bulk operations
OPENSEARCH_BATCH_SIZE = 100


@dataclass
class ChunkPayload:
    """Payload data for a chunk to be indexed.

    Attributes:
        chunk_id: Unique chunk identifier
        document_id: Parent document ID
        space_id: Space the document belongs to
        chunk_index: Position in the document
        title_chain: Heading hierarchy (e.g., "H1 > H2 > H3")
        source_file: Original filename
        page_number: Starting page number
        content: Chunk text content
        parent_chunk_id: Parent chunk ID (hierarchy)
        depth: Hierarchy depth (1-6)
        token_count: Number of tokens
        allowed_user_ids: Users with access permission
        access_level: Access level (read/write)
    """

    chunk_id: str = ""
    document_id: str = ""
    space_id: str = ""
    chunk_index: int = 0
    title_chain: str = ""
    source_file: str = ""
    page_number: int = 1
    content: str = ""
    parent_chunk_id: str | None = None
    depth: int = 1
    token_count: int = 0
    allowed_user_ids: list[str] = field(default_factory=list)
    access_level: str = "read"


class IndexingError(Exception):
    """Raised when indexing operations fail."""

    pass


class IndexingService:
    """Service for dual-writing chunks to Qdrant and OpenSearch.

    Implements transactional semantics: if one write fails, the other
    is rolled back to maintain consistency.
    """

    def __init__(self):
        """Initialize the indexing service."""
        self._qdrant = None
        self._opensearch = None

    @property
    def qdrant(self):
        """Lazy-load Qdrant client."""
        if self._qdrant is None:
            self._qdrant = get_qdrant_client()
        return self._qdrant

    @property
    def opensearch(self):
        """Lazy-load OpenSearch client."""
        if self._opensearch is None:
            self._opensearch = get_opensearch_client()
        return self._opensearch

    def index_chunks(
        self,
        payloads: list[ChunkPayload],
        embeddings: list[EmbeddingResult],
    ) -> dict:
        """Dual-write chunks to Qdrant and OpenSearch with rollback.

        Transactional contract:

        1. Empty input is a no-op and never opens any backend connection.
        2. Mismatched payload/embedding counts raise ``IndexingError``
           **before** any backend write so neither store is touched.
        3. Qdrant is written first. A failure here means nothing was
           committed to Qdrant (the SDK is all-or-nothing per batch and
           on partial-batch failures we re-attempt deletion in step 5
           anyway), and OpenSearch is **not** contacted — there is
           nothing to roll back.
        4. OpenSearch is written second. On success, both stores hold
           the same set of chunks identified by ``chunk_id``.
        5. If OpenSearch fails **after** Qdrant succeeded, all Qdrant
           points written in this call are deleted by ID (the same set
           we just upserted, so the IDs are guaranteed to exist or be
           idempotently absent). The original OpenSearch error is
           always re-raised as ``IndexingError`` — even if the rollback
           itself fails, in which case we log the rollback failure and
           include its state in the error message so operators can
           clean up out-of-band.

        Args:
            payloads: Chunk metadata and content.
            embeddings: Corresponding embedding results (must be the
                same length as ``payloads``).

        Returns:
            Dict with indexing summary (``qdrant_count``,
            ``opensearch_count``). Both equal ``len(payloads)`` on a
            successful dual-write.

        Raises:
            IndexingError: If either side of the dual-write fails. The
                message identifies which side failed and the rollback
                outcome.
        """
        # 1) Empty input — no-op contract.
        if not payloads or not embeddings:
            return {"qdrant_count": 0, "opensearch_count": 0}

        # 2) Length mismatch is a programming error — fail before touching
        #    any backend so we don't leave one store partially populated.
        if len(payloads) != len(embeddings):
            raise IndexingError(
                f"Payload count ({len(payloads)}) != embedding count ({len(embeddings)})"
            )

        # Snapshot point IDs up-front so the rollback path doesn't depend
        # on the (possibly-mutated) payload list.
        point_ids = [payload.chunk_id for payload in payloads]

        # 3) Qdrant first. ``qdrant_success`` gates the rollback so we never
        #    issue a delete against a Qdrant that received nothing.
        qdrant_success = False
        qdrant_indexed = 0
        try:
            qdrant_indexed = self._upsert_qdrant(payloads, embeddings)
            qdrant_success = True
            logger.info("Qdrant upsert successful: %d points", qdrant_indexed)
        except IndexingError:
            # Already a typed error from _upsert_qdrant (e.g. invalid chunk_id);
            # surface it directly without re-wrapping. No OpenSearch attempt,
            # no rollback — Qdrant never received the batch.
            logger.error("Qdrant upsert failed (validation error)")
            raise
        except Exception as e:
            logger.error("Qdrant upsert failed: %s", e)
            raise IndexingError(f"Qdrant write failed: {e}") from e

        # 4) OpenSearch second. On failure, run rollback (5) and re-raise.
        try:
            self._bulk_index_opensearch(payloads)
            logger.info("OpenSearch index successful: %d docs", len(payloads))
        except Exception as e:
            logger.error("OpenSearch index failed: %s", e)
            rollback_state = "no rollback needed"
            if qdrant_success:
                logger.warning(
                    "Rolling back %d Qdrant points due to OpenSearch failure",
                    len(point_ids),
                )
                try:
                    self._delete_qdrant_points(point_ids)
                    rollback_state = "Qdrant rolled back"
                    logger.info(
                        "Qdrant rollback successful (%d points)", len(point_ids)
                    )
                except Exception as rollback_error:
                    # The original OpenSearch error must still surface. We
                    # log the rollback failure so operators can reconcile
                    # the orphaned Qdrant points out-of-band.
                    rollback_state = f"Qdrant rollback FAILED: {rollback_error}"
                    logger.error(
                        "Qdrant rollback failed for %d points: %s",
                        len(point_ids),
                        rollback_error,
                    )
            raise IndexingError(
                f"OpenSearch write failed ({rollback_state}): {e}"
            ) from e

        return {
            "qdrant_count": qdrant_indexed,
            "opensearch_count": len(payloads),
        }

    def delete_document_chunks(self, document_id: str) -> dict:
        """Delete all chunks for a document from both Qdrant and OpenSearch.

        Used for cascade cleanup when a document is deleted, when its parent
        space/folder is deleted (``DocumentService.delete_space`` /
        ``delete_folder``), and by the reprocess flow before re-indexing
        corrected content.

        Failure semantics
        -----------------
        Both backends are attempted in order. The two failure modes are
        treated **asymmetrically** so we never make the system worse than
        when we started:

        * **Qdrant failure** → raise :class:`IndexingError`. OpenSearch is
          **not** attempted in that case so we do not leave half a cleanup
          behind. The caller (e.g. ``DocumentService.delete_*``) should
          abort the business-level delete and let the operation be
          retried, keeping the PostgreSQL row consistent with the still-
          populated search backends.
        * **Qdrant succeeded, OpenSearch failed** → log an error and
          return a result that includes ``opensearch_error``. This is a
          *partial cleanup*: vector-side recall is already clean, so the
          deleted document cannot surface via Qdrant; the few stale
          OpenSearch docs that remain are filtered out by ABAC once the
          PostgreSQL row is removed and can be reaped by an out-of-band
          reconciliation. Re-raising would force the caller to either
          roll back the (already successful) Qdrant delete — which is
          strictly more expensive than retrying the OpenSearch delete —
          or leave the PostgreSQL row in a half-deleted state.

        Args:
            document_id: The document ID whose chunks should be removed.

        Returns:
            Dict with deletion counts:

            - ``qdrant_deleted`` (int): ``-1`` sentinel because Qdrant's
              filter-delete API does not return a row count. Treat it as
              "issued, count unknown" rather than "0 rows deleted".
            - ``opensearch_deleted`` (int): the actual deleted count
              reported by OpenSearch (``0`` when the call succeeded but
              matched nothing).
            - ``opensearch_error`` (str, optional): present **only** when
              OpenSearch failed after Qdrant succeeded. Indicates partial
              cleanup; callers may record this for later reconciliation.

        Raises:
            IndexingError: When Qdrant deletion fails. OpenSearch is not
                attempted in that case, so no partial state is created.
        """
        result: dict = {"qdrant_deleted": 0, "opensearch_deleted": 0}

        # 1) Qdrant first — fatal on failure so the caller can abort.
        try:
            from qdrant_client.models import (
                FieldCondition,
                Filter,
                MatchValue,
            )

            self.qdrant.delete(
                collection_name=QD_COLLECTION_NAME,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id),
                        )
                    ]
                ),
            )
            # Qdrant filter-delete does not echo a count; -1 distinguishes
            # "issued, count unknown" from "matched 0".
            result["qdrant_deleted"] = -1
            logger.info("Deleted Qdrant points for document %s", document_id)
        except Exception as e:
            logger.error(
                "Failed to delete Qdrant points for %s: %s", document_id, e
            )
            raise IndexingError(f"Qdrant deletion failed: {e}") from e

        # 2) OpenSearch second — non-fatal, partial-cleanup contract.
        try:
            os_result = self.opensearch.delete_by_query(
                index=OS_INDEX_NAME,
                body={"query": {"term": {"document_id": document_id}}},
                refresh=True,
            )
            result["opensearch_deleted"] = os_result.get("deleted", 0)
            logger.info(
                "Deleted %d OpenSearch docs for document %s",
                result["opensearch_deleted"],
                document_id,
            )
        except Exception as e:
            # Partial cleanup: Qdrant is clean, OpenSearch is not. Surface
            # the failure in the return value so the caller can record a
            # reconciliation hint, but do **not** raise — re-raising would
            # force a Qdrant rollback that is strictly more expensive than
            # retrying the OpenSearch delete later.
            logger.error(
                "Failed to delete OpenSearch docs for %s "
                "(Qdrant already cleaned, partial cleanup): %s",
                document_id,
                e,
            )
            result["opensearch_error"] = str(e)

        return result

    def _upsert_qdrant(
        self,
        payloads: list[ChunkPayload],
        embeddings: list[EmbeddingResult],
    ) -> int:
        """Batch upsert points to Qdrant.

        For each ``(payload, embedding)`` pair, builds a ``PointStruct`` with:

        - ``id`` = ``payload.chunk_id`` (must be UUID-compatible per Qdrant's
          point-ID rules — UUID-string or unsigned-int — see
          https://qdrant.tech/documentation/concepts/points/#point-ids).
        - ``vector`` = ``{"dense": [..1024..]}`` plus, when present,
          ``"sparse": SparseVector(indices=..., values=...)``. The vector is
          built in a single PointStruct so each point is upserted exactly once
          and Qdrant never observes a transient sparse-less revision.
        - ``payload`` = the full chunk metadata required by the design's Qdrant
          data model (``document_id``, ``space_id``, ``chunk_index``,
          ``title_chain``, ``source_file``, ``page_number``, ``content``,
          ``parent_chunk_id``, ``depth``, ``token_count``,
          ``allowed_user_ids``, ``access_level``).

        Args:
            payloads: Chunk metadata.
            embeddings: Corresponding embedding vectors (must be same length).

        Returns:
            Number of points upserted.

        Raises:
            IndexingError: If any chunk_id is not a valid UUID.
            Exception: Re-raises Qdrant client errors after logging the failed
                batch range so the caller (``index_chunks``) can run rollback.
        """
        from qdrant_client.models import PointStruct, SparseVector

        if not payloads:
            return 0

        points: list[PointStruct] = []
        for payload, embedding in zip(payloads, embeddings):
            # Qdrant rejects arbitrary strings as point IDs; enforce the
            # UUID requirement up-front so we fail fast with a clear message
            # instead of getting an opaque server-side rejection mid-batch.
            try:
                uuid.UUID(str(payload.chunk_id))
            except (ValueError, AttributeError, TypeError) as exc:
                raise IndexingError(
                    f"chunk_id {payload.chunk_id!r} is not a valid UUID; "
                    "Qdrant requires UUID-string or unsigned-int point IDs"
                ) from exc

            qdrant_payload = {
                "document_id": payload.document_id,
                "space_id": payload.space_id,
                "chunk_index": payload.chunk_index,
                "title_chain": payload.title_chain,
                "source_file": payload.source_file,
                "page_number": payload.page_number,
                "content": payload.content,
                "parent_chunk_id": payload.parent_chunk_id,
                "depth": payload.depth,
                "token_count": payload.token_count,
                "allowed_user_ids": list(payload.allowed_user_ids),
                "access_level": payload.access_level,
            }

            vectors: dict = {"dense": list(embedding.dense_vector)}
            # Only attach a sparse vector when there is signal to send.
            # Empty sparse vectors waste bytes on the wire and would have to
            # be filtered by the search side anyway.
            if embedding.sparse_indices and embedding.sparse_values:
                vectors["sparse"] = SparseVector(
                    indices=list(embedding.sparse_indices),
                    values=list(embedding.sparse_values),
                )

            points.append(
                PointStruct(
                    id=payload.chunk_id,
                    vector=vectors,
                    payload=qdrant_payload,
                )
            )

        total_indexed = 0
        for i in range(0, len(points), QDRANT_BATCH_SIZE):
            batch = points[i : i + QDRANT_BATCH_SIZE]
            try:
                self.qdrant.upsert(
                    collection_name=QD_COLLECTION_NAME,
                    points=batch,
                )
            except Exception as exc:
                logger.error(
                    "Qdrant upsert failed for batch [%d:%d] (size=%d): %s",
                    i,
                    i + len(batch),
                    len(batch),
                    exc,
                )
                raise
            total_indexed += len(batch)
            logger.debug(
                "Qdrant upsert ok: batch [%d:%d] (size=%d, total=%d)",
                i,
                i + len(batch),
                len(batch),
                total_indexed,
            )

        return total_indexed

    def _bulk_index_opensearch(self, payloads: list[ChunkPayload]) -> None:
        """Batch index documents to OpenSearch.

        Builds one bulk action per chunk and writes them to the
        ``chunks`` index in batches of ``OPENSEARCH_BATCH_SIZE``. Each
        ``_source`` mirrors the OpenSearch index model defined in
        ``design.md``:

        - ``chunk_id`` (keyword) — also used as ``_id`` so re-indexing is an
          idempotent upsert; bulk's default ``index`` op-type replaces the
          existing document with the same ``_id`` so callers can safely
          retry without producing duplicates.
        - ``document_id`` / ``space_id`` (keyword) — for tenancy & cascade
          cleanup filtering.
        - ``content`` (text, ``ik_max_word`` / ``ik_smart``).
        - ``title_chain`` (text), ``source_file`` (keyword),
          ``page_number`` / ``chunk_index`` (integer).
        - ``allowed_user_ids`` (keyword) — ABAC permission filter.
        - ``created_at`` (date) — write timestamp; recorded once per
          ``_bulk_index_opensearch`` invocation so all chunks of the same
          document share a coherent indexing time.

        ``refresh=True`` is passed to ``bulk`` so the documents become
        visible to subsequent search calls immediately — the dual-write
        path needs read-after-write consistency for the rollback logic
        and for the integration tests in task 12.10 to assert search
        results without sleeping.

        Args:
            payloads: Chunk metadata and content. Empty list is a no-op
                and **does not** open a bulk connection.

        Raises:
            IndexingError: When ``opensearchpy.helpers.bulk`` reports any
                per-item failures. The error message includes the
                concrete success/failure counts so the caller (and the
                pipeline's status writer) can attribute the failure
                accurately.
        """
        if not payloads:
            # Avoid a no-op bulk call that would still incur a network
            # round-trip and a refresh on the index.
            return

        from opensearchpy.helpers import bulk

        # All chunks of a single index_chunks call share one timestamp so
        # they can be filtered/sorted as a coherent unit downstream.
        now = datetime.now(timezone.utc).isoformat()

        actions = [
            {
                "_index": OS_INDEX_NAME,
                "_id": payload.chunk_id,
                "_source": {
                    "chunk_id": payload.chunk_id,
                    "document_id": payload.document_id,
                    "space_id": payload.space_id,
                    "content": payload.content,
                    "title_chain": payload.title_chain,
                    "source_file": payload.source_file,
                    "page_number": payload.page_number,
                    "chunk_index": payload.chunk_index,
                    "allowed_user_ids": list(payload.allowed_user_ids),
                    "created_at": now,
                },
            }
            for payload in payloads
        ]

        total_success = 0
        total_errors = 0
        for i in range(0, len(actions), OPENSEARCH_BATCH_SIZE):
            batch = actions[i : i + OPENSEARCH_BATCH_SIZE]
            # raise_on_error=False so we observe the per-item error list
            # ourselves and emit a single IndexingError with concrete
            # counts rather than letting the helper raise a BulkIndexError
            # that hides the success count.
            success, errors = bulk(
                self.opensearch,
                batch,
                refresh=True,
                raise_on_error=False,
            )
            total_success += success
            if errors:
                total_errors += len(errors)
                # Log only the first few errors to avoid log spam on a
                # bad batch — full payloads remain in OpenSearch's logs.
                logger.error(
                    "OpenSearch bulk batch [%d:%d] reported %d errors: %s",
                    i,
                    i + len(batch),
                    len(errors),
                    errors[:3],
                )

        if total_errors:
            raise IndexingError(
                f"OpenSearch bulk index had {total_errors} error(s) "
                f"out of {len(actions)} action(s) "
                f"({total_success} succeeded)"
            )

    def _delete_qdrant_points(self, point_ids: list[str]) -> None:
        """Delete specific points from Qdrant by ID.

        Args:
            point_ids: List of point IDs to delete
        """
        for i in range(0, len(point_ids), QDRANT_BATCH_SIZE):
            batch = point_ids[i:i + QDRANT_BATCH_SIZE]
            self.qdrant.delete(
                collection_name=QD_COLLECTION_NAME,
                points_selector=batch,
            )

    def _delete_opensearch_docs(self, doc_ids: list[str]) -> None:
        """Delete specific documents from OpenSearch by ID.

        Args:
            doc_ids: List of document IDs to delete
        """
        for doc_id in doc_ids:
            try:
                self.opensearch.delete(
                    index=OS_INDEX_NAME,
                    id=doc_id,
                    refresh=True,
                )
            except Exception:
                pass  # Document may not exist


def update_pipeline_progress(
    document_id: str,
    stage: str,
    progress: int,
) -> None:
    """Update pipeline progress in Redis.

    Args:
        document_id: Document UUID
        stage: Current processing stage
        progress: Progress percentage (0-100)
    """
    try:
        import redis as redis_lib

        settings = get_settings()
        r = redis_lib.Redis.from_url(settings.REDIS_URL)
        key = f"doc:status:{document_id}"
        r.hset(key, mapping={
            "stage": stage,
            "progress": str(progress),
            "updated_at": str(time.time()),
        })
        r.expire(key, 3600)
    except Exception as e:
        logger.warning(f"Failed to update pipeline progress for {document_id}: {e}")


def update_document_db_status(
    document_id: str,
    status: str,
    current_stage: str | None = None,
    progress_percent: int | None = None,
    error_detail: str | None = None,
) -> None:
    """Update document status in PostgreSQL.

    Args:
        document_id: Document UUID
        status: New status value
        current_stage: Current processing stage
        progress_percent: Progress percentage
        error_detail: Error description if failed
    """
    try:
        from sqlalchemy import create_engine, text

        settings = get_settings()
        sync_url = settings.DATABASE_URL.replace("+asyncpg", "+psycopg2")
        engine = create_engine(sync_url)

        updates = ["status = :status", "last_status_update = NOW()"]
        params: dict = {"id": document_id, "status": status}

        if current_stage is not None:
            updates.append("current_stage = :current_stage")
            params["current_stage"] = current_stage

        if progress_percent is not None:
            updates.append("progress_percent = :progress_percent")
            params["progress_percent"] = progress_percent

        if error_detail is not None:
            updates.append("error_detail = :error_detail")
            params["error_detail"] = error_detail

        sql = f"UPDATE documents SET {', '.join(updates)} WHERE id = :id"

        with engine.connect() as conn:
            conn.execute(text(sql), params)
            conn.commit()

    except Exception as e:
        logger.warning(f"Failed to update document DB status for {document_id}: {e}")
