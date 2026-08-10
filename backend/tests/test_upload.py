"""Unit tests for upload service (file upload, URL import, status management, retry).

Tests cover:
- File format validation (PDF, DOCX, PPTX, TXT, MD, HTML)
- File size validation (100MB limit)
- Batch upload limit (max 50 files)
- MinIO client initialization and bucket auto-creation
- MinIO storage
- URL import with trafilatura
- Document status state machine transitions
- State machine closure property test (hypothesis)
- Progress tracking via Redis
- Retry logic with permanent failure marking (max 3 retries)
"""

import uuid
from datetime import datetime, timezone
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from botocore.exceptions import ClientError
from hypothesis import given, settings as hyp_settings
from hypothesis import strategies as st

from app.core import minio as minio_module
from app.core.exceptions import NotFoundException, ValidationException
from app.models.document import Document, DocumentStatus
from app.services.upload_service import (
    MAX_BATCH_SIZE,
    MAX_FILE_SIZE,
    MAX_RETRY_COUNT,
    SUPPORTED_EXTENSIONS,
    VALID_TRANSITIONS,
    UploadService,
)


# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def mock_db():
    """Create a mock async database session."""
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.refresh = AsyncMock()
    return db


@pytest.fixture
def service(mock_db):
    """Create an UploadService instance with mocked DB."""
    return UploadService(db=mock_db)


@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    redis = AsyncMock()
    redis.hset = AsyncMock()
    redis.hgetall = AsyncMock(return_value={})
    redis.expire = AsyncMock()
    return redis


# ─── Helper Functions ──────────────────────────────────────────────────


def make_upload_file(
    filename: str = "test.pdf",
    content: bytes = b"fake pdf content",
    content_type: str = "application/pdf",
) -> MagicMock:
    """Create a mock UploadFile."""
    file = AsyncMock()
    file.filename = filename
    file.content_type = content_type
    file.read = AsyncMock(return_value=content)
    return file


def make_document(
    status: DocumentStatus = DocumentStatus.pending,
    retry_count: int = 0,
    space_id: uuid.UUID | None = None,
) -> MagicMock:
    """Create a mock Document object."""
    doc = MagicMock(spec=Document)
    doc.id = uuid.uuid4()
    doc.space_id = space_id or uuid.uuid4()
    doc.folder_id = None
    doc.title = "test.pdf"
    doc.file_type = "pdf"
    doc.file_size = 1024
    doc.storage_path = "space/uuid/test.pdf"
    doc.status = status
    doc.retry_count = retry_count
    doc.error_detail = None
    doc.current_stage = status.value
    doc.progress_percent = 0
    doc.uploaded_by = uuid.uuid4()
    doc.created_at = datetime.now(timezone.utc)
    doc.updated_at = datetime.now(timezone.utc)
    return doc


# ─── File Format Validation Tests ──────────────────────────────────────


class TestFileFormatValidation:
    """Tests for file format validation."""

    @pytest.mark.asyncio
    @patch("app.services.upload_service.get_redis")
    @patch("app.services.upload_service.get_minio_client")
    @patch("app.services.upload_service.ensure_bucket_exists")
    async def test_upload_pdf_success(self, mock_bucket, mock_minio, mock_get_redis, service, mock_db, mock_redis):
        """Successfully upload a PDF file."""
        mock_get_redis.return_value = mock_redis
        mock_client = MagicMock()
        mock_minio.return_value = mock_client

        file = make_upload_file(filename="document.pdf", content=b"pdf content")
        space_id = uuid.uuid4()
        user_id = uuid.uuid4()

        docs = await service.upload_files(
            files=[file],
            space_id=space_id,
            folder_id=None,
            uploaded_by=user_id,
        )

        mock_db.add.assert_called_once()
        mock_client.put_object.assert_called_once()
        added_doc = mock_db.add.call_args[0][0]
        assert added_doc.file_type == "pdf"
        assert added_doc.status == DocumentStatus.pending

    @pytest.mark.asyncio
    @patch("app.services.upload_service.get_redis")
    @patch("app.services.upload_service.get_minio_client")
    @patch("app.services.upload_service.ensure_bucket_exists")
    async def test_upload_all_supported_formats(self, mock_bucket, mock_minio, mock_get_redis, service, mock_db, mock_redis):
        """All supported formats are accepted."""
        mock_get_redis.return_value = mock_redis
        mock_client = MagicMock()
        mock_minio.return_value = mock_client

        user_id = uuid.uuid4()
        space_id = uuid.uuid4()

        for ext in SUPPORTED_EXTENSIONS:
            mock_db.reset_mock()
            file = make_upload_file(filename=f"test.{ext}", content=b"content")
            docs = await service.upload_files(
                files=[file],
                space_id=space_id,
                folder_id=None,
                uploaded_by=user_id,
            )
            mock_db.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_unsupported_format_rejected(self, service, mock_db):
        """Unsupported file format raises ValidationException."""
        file = make_upload_file(filename="test.exe", content=b"content")

        with pytest.raises(ValidationException, match="不支持的文件格式"):
            await service.upload_files(
                files=[file],
                space_id=uuid.uuid4(),
                folder_id=None,
                uploaded_by=uuid.uuid4(),
            )

    @pytest.mark.asyncio
    async def test_upload_no_extension_rejected(self, service, mock_db):
        """File without extension raises ValidationException."""
        file = make_upload_file(filename="noextension", content=b"content")

        with pytest.raises(ValidationException, match="不支持的文件格式"):
            await service.upload_files(
                files=[file],
                space_id=uuid.uuid4(),
                folder_id=None,
                uploaded_by=uuid.uuid4(),
            )


# ─── File Size Validation Tests ────────────────────────────────────────


class TestFileSizeValidation:
    """Tests for file size validation."""

    @pytest.mark.asyncio
    async def test_upload_file_exceeds_100mb(self, service, mock_db):
        """File larger than 100MB raises ValidationException."""
        # Create content just over 100MB
        large_content = b"x" * (MAX_FILE_SIZE + 1)
        file = make_upload_file(filename="large.pdf", content=large_content)

        with pytest.raises(ValidationException, match="超出限制"):
            await service.upload_files(
                files=[file],
                space_id=uuid.uuid4(),
                folder_id=None,
                uploaded_by=uuid.uuid4(),
            )

    @pytest.mark.asyncio
    async def test_upload_empty_file_rejected(self, service, mock_db):
        """Empty file raises ValidationException."""
        file = make_upload_file(filename="empty.pdf", content=b"")

        with pytest.raises(ValidationException, match="空文件"):
            await service.upload_files(
                files=[file],
                space_id=uuid.uuid4(),
                folder_id=None,
                uploaded_by=uuid.uuid4(),
            )

    @pytest.mark.asyncio
    @patch("app.services.upload_service.get_redis")
    @patch("app.services.upload_service.get_minio_client")
    @patch("app.services.upload_service.ensure_bucket_exists")
    async def test_upload_exactly_100mb_succeeds(
        self, mock_bucket, mock_minio, mock_get_redis, service, mock_db, mock_redis
    ):
        """File of exactly 100MB is accepted (boundary test)."""
        mock_get_redis.return_value = mock_redis
        mock_client = MagicMock()
        mock_minio.return_value = mock_client

        # 边界值：恰好 100MB 应当通过
        boundary_content = b"x" * MAX_FILE_SIZE
        file = make_upload_file(filename="boundary.pdf", content=boundary_content)

        await service.upload_files(
            files=[file],
            space_id=uuid.uuid4(),
            folder_id=None,
            uploaded_by=uuid.uuid4(),
        )

        mock_db.add.assert_called_once()
        added_doc = mock_db.add.call_args[0][0]
        assert added_doc.file_size == MAX_FILE_SIZE


# ─── Batch Upload Tests ────────────────────────────────────────────────


class TestBatchUpload:
    """Tests for batch upload limits."""

    @pytest.mark.asyncio
    async def test_upload_exceeds_50_files(self, service, mock_db):
        """Uploading more than 50 files raises ValidationException."""
        files = [make_upload_file(filename=f"file{i}.pdf") for i in range(51)]

        with pytest.raises(ValidationException, match="50"):
            await service.upload_files(
                files=files,
                space_id=uuid.uuid4(),
                folder_id=None,
                uploaded_by=uuid.uuid4(),
            )

    @pytest.mark.asyncio
    async def test_upload_zero_files(self, service, mock_db):
        """Uploading zero files raises ValidationException."""
        with pytest.raises(ValidationException, match="至少上传一个"):
            await service.upload_files(
                files=[],
                space_id=uuid.uuid4(),
                folder_id=None,
                uploaded_by=uuid.uuid4(),
            )

    @pytest.mark.asyncio
    @patch("app.services.upload_service.get_redis")
    @patch("app.services.upload_service.get_minio_client")
    @patch("app.services.upload_service.ensure_bucket_exists")
    async def test_upload_exactly_50_files(self, mock_bucket, mock_minio, mock_get_redis, service, mock_db, mock_redis):
        """Uploading exactly 50 files succeeds."""
        mock_get_redis.return_value = mock_redis
        mock_client = MagicMock()
        mock_minio.return_value = mock_client

        files = [make_upload_file(filename=f"file{i}.pdf", content=b"content") for i in range(50)]

        docs = await service.upload_files(
            files=files,
            space_id=uuid.uuid4(),
            folder_id=None,
            uploaded_by=uuid.uuid4(),
        )

        assert mock_db.add.call_count == 50


# ─── URL Import Tests ──────────────────────────────────────────────────


class TestUrlImport:
    """Tests for URL import functionality."""

    @pytest.mark.asyncio
    async def test_import_empty_url(self, service, mock_db):
        """Empty URL raises ValidationException."""
        with pytest.raises(ValidationException, match="不能为空"):
            await service.import_url(
                url="",
                space_id=uuid.uuid4(),
                folder_id=None,
                uploaded_by=uuid.uuid4(),
            )

    @pytest.mark.asyncio
    async def test_import_invalid_url_scheme(self, service, mock_db):
        """URL without http/https raises ValidationException."""
        with pytest.raises(ValidationException, match="http://"):
            await service.import_url(
                url="ftp://example.com",
                space_id=uuid.uuid4(),
                folder_id=None,
                uploaded_by=uuid.uuid4(),
            )

    @pytest.mark.asyncio
    @patch("app.services.upload_service.trafilatura")
    @patch("app.services.upload_service.get_redis")
    @patch("app.services.upload_service.get_minio_client")
    @patch("app.services.upload_service.ensure_bucket_exists")
    async def test_import_url_success(self, mock_bucket, mock_minio, mock_get_redis, mock_traf, service, mock_db, mock_redis):
        """Successfully import content from a URL."""
        mock_get_redis.return_value = mock_redis
        mock_client = MagicMock()
        mock_minio.return_value = mock_client
        mock_traf.fetch_url.return_value = "<html><body>Hello World</body></html>"
        mock_traf.extract.return_value = "Hello World extracted content"

        doc = await service.import_url(
            url="https://example.com/article",
            space_id=uuid.uuid4(),
            folder_id=None,
            uploaded_by=uuid.uuid4(),
        )

        mock_db.add.assert_called_once()
        added_doc = mock_db.add.call_args[0][0]
        assert added_doc.file_type == "html"
        assert added_doc.status == DocumentStatus.pending

    @pytest.mark.asyncio
    @patch("app.services.upload_service.trafilatura")
    async def test_import_url_fetch_fails(self, mock_traf, service, mock_db):
        """URL that cannot be fetched raises ValidationException."""
        mock_traf.fetch_url.return_value = None

        with pytest.raises(ValidationException, match="无法访问"):
            await service.import_url(
                url="https://nonexistent.example.com",
                space_id=uuid.uuid4(),
                folder_id=None,
                uploaded_by=uuid.uuid4(),
            )

    @pytest.mark.asyncio
    @patch("app.services.upload_service.trafilatura")
    async def test_import_url_no_content_extracted(self, mock_traf, service, mock_db):
        """URL with no extractable content raises ValidationException."""
        mock_traf.fetch_url.return_value = "<html></html>"
        mock_traf.extract.return_value = ""

        with pytest.raises(ValidationException, match="无法从 URL 提取"):
            await service.import_url(
                url="https://example.com/empty",
                space_id=uuid.uuid4(),
                folder_id=None,
                uploaded_by=uuid.uuid4(),
            )


# ─── Status Management Tests ──────────────────────────────────────────


class TestStatusManagement:
    """Tests for document status state machine."""

    @pytest.mark.asyncio
    @patch("app.services.upload_service.get_redis")
    async def test_valid_transition_pending_to_parsing(self, mock_get_redis, service, mock_db, mock_redis):
        """Valid transition from pending to parsing."""
        mock_get_redis.return_value = mock_redis
        doc = make_document(status=DocumentStatus.pending)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = doc
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await service.transition_status(doc.id, DocumentStatus.parsing)
        assert doc.status == DocumentStatus.parsing

    @pytest.mark.asyncio
    @patch("app.services.upload_service.get_redis")
    async def test_valid_transition_parsing_to_cleaning(self, mock_get_redis, service, mock_db, mock_redis):
        """Valid transition from parsing to cleaning."""
        mock_get_redis.return_value = mock_redis
        doc = make_document(status=DocumentStatus.parsing)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = doc
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await service.transition_status(doc.id, DocumentStatus.cleaning)
        assert doc.status == DocumentStatus.cleaning

    @pytest.mark.asyncio
    @patch("app.services.upload_service.get_redis")
    async def test_valid_transition_to_completed(self, mock_get_redis, service, mock_db, mock_redis):
        """Valid transition from indexing to completed sets progress to 100."""
        mock_get_redis.return_value = mock_redis
        doc = make_document(status=DocumentStatus.indexing)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = doc
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await service.transition_status(doc.id, DocumentStatus.completed)
        assert doc.status == DocumentStatus.completed
        assert doc.progress_percent == 100

    @pytest.mark.asyncio
    @patch("app.services.upload_service.get_redis")
    async def test_transition_to_failed_increments_retry(self, mock_get_redis, service, mock_db, mock_redis):
        """Transitioning to failed increments retry_count."""
        mock_get_redis.return_value = mock_redis
        doc = make_document(status=DocumentStatus.parsing)
        doc.retry_count = 1

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = doc
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await service.transition_status(
            doc.id, DocumentStatus.failed, error_detail="Parse error"
        )
        assert doc.status == DocumentStatus.failed
        assert doc.retry_count == 2
        assert doc.error_detail == "Parse error"

    @pytest.mark.asyncio
    async def test_invalid_transition_raises_error(self, service, mock_db):
        """Invalid status transition raises ValidationException."""
        doc = make_document(status=DocumentStatus.completed)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = doc
        mock_db.execute = AsyncMock(return_value=mock_result)

        with pytest.raises(ValidationException, match="无效的状态转换"):
            await service.transition_status(doc.id, DocumentStatus.parsing)

    @pytest.mark.asyncio
    async def test_transition_document_not_found(self, service, mock_db):
        """Transitioning non-existent document raises NotFoundException."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        with pytest.raises(NotFoundException):
            await service.transition_status(uuid.uuid4(), DocumentStatus.parsing)

    def test_state_machine_completeness(self, service):
        """All statuses have defined transitions."""
        for status in DocumentStatus:
            assert status in VALID_TRANSITIONS

    def test_completed_has_no_forward_transitions(self, service):
        """Completed status has no forward transitions (terminal state)."""
        assert VALID_TRANSITIONS[DocumentStatus.completed] == []

    def test_failed_can_transition_to_pending(self, service):
        """Failed status can transition back to pending (retry)."""
        assert DocumentStatus.pending in VALID_TRANSITIONS[DocumentStatus.failed]

    @pytest.mark.asyncio
    async def test_completed_cannot_transition_anywhere(self, service, mock_db):
        """已完成是终态：尝试任意转换都应抛出 ValidationException。"""
        doc = make_document(status=DocumentStatus.completed)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = doc
        mock_db.execute = AsyncMock(return_value=mock_result)

        for target in DocumentStatus:
            if target == DocumentStatus.completed:
                # 同状态本身也不在白名单里；理论上也属于"无效转换"
                pass
            with pytest.raises(ValidationException, match="无效的状态转换"):
                await service.transition_status(doc.id, target)

    @pytest.mark.asyncio
    @patch("app.services.upload_service.get_redis")
    async def test_failed_to_pending_clears_error_detail(
        self, mock_get_redis, service, mock_db, mock_redis
    ):
        """failed → pending 是合法转换（用于重试），但 transition_status 不会清空 error_detail；
        清空发生在 retry_document 中。这里验证 transition_status 行为本身合法。"""
        mock_get_redis.return_value = mock_redis
        doc = make_document(status=DocumentStatus.failed, retry_count=1)
        doc.error_detail = "previous error"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = doc
        mock_db.execute = AsyncMock(return_value=mock_result)

        await service.transition_status(doc.id, DocumentStatus.pending)
        assert doc.status == DocumentStatus.pending

    # ─── PBT: 状态机封闭性 ────────────────────────────────────────

    @hyp_settings(deadline=None, max_examples=80)
    @given(
        src=st.sampled_from(list(DocumentStatus)),
        dst=st.sampled_from(list(DocumentStatus)),
    )
    def test_state_machine_closure_property(self, src, dst):
        """**Validates: Requirements 3.1, 3.5**

        属性：对状态机中任意 (src, dst) 组合，
        - 若 dst 在 VALID_TRANSITIONS[src] 中：transition_status 不抛异常
        - 否则：transition_status 必须抛 ValidationException

        即状态机外的所有转换都被严格拒绝（封闭性）。
        """
        import asyncio

        # 数据结构层断言
        valid = VALID_TRANSITIONS.get(src, [])
        is_legal = dst in valid

        if src == DocumentStatus.completed:
            assert valid == []
            assert is_legal is False
        elif src == DocumentStatus.failed:
            assert valid == [DocumentStatus.pending]
        else:
            assert DocumentStatus.failed in valid
        for d in valid:
            assert isinstance(d, DocumentStatus)

        # 行为层断言：调用 transition_status 验证一致性
        async def _run() -> None:
            db = AsyncMock()
            db.add = MagicMock()
            db.flush = AsyncMock()
            db.refresh = AsyncMock()
            svc = UploadService(db=db)

            doc = make_document(status=src)
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = doc
            db.execute = AsyncMock(return_value=mock_result)

            with patch("app.services.upload_service.get_redis") as mock_get_redis:
                mock_get_redis.return_value = AsyncMock(
                    hset=AsyncMock(), expire=AsyncMock(), hgetall=AsyncMock(return_value={})
                )
                if is_legal:
                    await svc.transition_status(doc.id, dst)
                    assert doc.status == dst
                else:
                    with pytest.raises(ValidationException, match="无效的状态转换"):
                        await svc.transition_status(doc.id, dst)

        asyncio.run(_run())


# ─── Progress Query Tests ──────────────────────────────────────────────


class TestProgressQuery:
    """Tests for document progress query."""

    @pytest.mark.asyncio
    @patch("app.services.upload_service.get_redis")
    async def test_get_progress_from_redis(self, mock_get_redis, service, mock_db):
        """Get progress from Redis cache."""
        mock_redis = AsyncMock()
        mock_redis.hgetall = AsyncMock(return_value={
            "stage": "parsing",
            "progress": "45",
            "updated_at": "2024-01-01T00:00:00+00:00",
        })
        mock_get_redis.return_value = mock_redis

        doc_id = uuid.uuid4()
        progress = await service.get_document_progress(doc_id)

        assert progress["stage"] == "parsing"
        assert progress["progress"] == 45
        assert progress["document_id"] == str(doc_id)

    @pytest.mark.asyncio
    @patch("app.services.upload_service.get_redis")
    async def test_get_progress_fallback_to_db(self, mock_get_redis, service, mock_db):
        """Falls back to database when Redis has no data."""
        mock_redis = AsyncMock()
        mock_redis.hgetall = AsyncMock(return_value={})
        mock_get_redis.return_value = mock_redis

        doc = make_document(status=DocumentStatus.chunking)
        doc.current_stage = "chunking"
        doc.progress_percent = 60

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = doc
        mock_db.execute = AsyncMock(return_value=mock_result)

        progress = await service.get_document_progress(doc.id)

        assert progress["stage"] == "chunking"
        assert progress["progress"] == 60


# ─── Retry Logic Tests ─────────────────────────────────────────────────


class TestRetryLogic:
    """Tests for document retry and permanent failure logic."""

    @pytest.mark.asyncio
    @patch("app.services.upload_service.get_redis")
    async def test_retry_failed_document(self, mock_get_redis, service, mock_db, mock_redis):
        """Successfully retry a failed document."""
        mock_get_redis.return_value = mock_redis
        doc = make_document(status=DocumentStatus.failed, retry_count=1)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = doc
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await service.retry_document(doc.id)
        assert doc.status == DocumentStatus.pending
        assert doc.progress_percent == 0
        assert doc.error_detail is None

    @pytest.mark.asyncio
    async def test_retry_non_failed_document_raises_error(self, service, mock_db):
        """Retrying a non-failed document raises ValidationException."""
        doc = make_document(status=DocumentStatus.parsing)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = doc
        mock_db.execute = AsyncMock(return_value=mock_result)

        with pytest.raises(ValidationException, match="只有状态为'失败'"):
            await service.retry_document(doc.id)

    @pytest.mark.asyncio
    async def test_retry_permanently_failed_raises_error(self, service, mock_db):
        """Retrying a permanently failed document (3+ retries) raises ValidationException."""
        doc = make_document(status=DocumentStatus.failed, retry_count=3)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = doc
        mock_db.execute = AsyncMock(return_value=mock_result)

        with pytest.raises(ValidationException, match="永久失败"):
            await service.retry_document(doc.id)

    @pytest.mark.asyncio
    async def test_retry_document_not_found(self, service, mock_db):
        """Retrying non-existent document raises NotFoundException."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        with pytest.raises(NotFoundException):
            await service.retry_document(uuid.uuid4())

    def test_is_permanently_failed_true(self, service):
        """Document with 3+ retries and failed status is permanently failed."""
        doc = make_document(status=DocumentStatus.failed, retry_count=3)
        assert service.is_permanently_failed(doc) is True

    def test_is_permanently_failed_false_low_retry(self, service):
        """Document with < 3 retries is not permanently failed."""
        doc = make_document(status=DocumentStatus.failed, retry_count=2)
        assert service.is_permanently_failed(doc) is False

    def test_is_permanently_failed_false_not_failed(self, service):
        """Document not in failed status is not permanently failed."""
        doc = make_document(status=DocumentStatus.pending, retry_count=5)
        assert service.is_permanently_failed(doc) is False


# ─── Storage Path Tests ────────────────────────────────────────────────


class TestStoragePath:
    """Tests for storage path generation."""

    def test_generate_storage_path_format(self, service):
        """Storage path follows {space_id}/{uuid}/{filename} format."""
        space_id = uuid.uuid4()
        path = service._generate_storage_path(space_id, "test.pdf")

        parts = path.split("/")
        assert len(parts) == 3
        assert parts[0] == str(space_id)
        # Middle part should be a valid UUID
        uuid.UUID(parts[1])
        assert parts[2] == "test.pdf"

    def test_get_file_extension(self, service):
        """File extension extraction works correctly."""
        assert service._get_file_extension("test.pdf") == "pdf"
        assert service._get_file_extension("test.DOCX") == "docx"
        assert service._get_file_extension("my.file.txt") == "txt"
        assert service._get_file_extension("noext") == ""

    def test_extract_title_from_url(self, service):
        """Title extraction from URL works correctly."""
        assert service._extract_title_from_url("https://example.com/article") == "article"
        assert service._extract_title_from_url("https://example.com/path/page.html") == "page.html"
        assert service._extract_title_from_url("https://example.com/") == "example.com"
        assert service._extract_title_from_url("https://example.com") == "example.com"


# ─── MinIO Client Tests ────────────────────────────────────────────────


class TestMinIOClient:
    """Tests for MinIO 客户端单例与 bucket 自动创建（任务 6.1）。"""

    def setup_method(self) -> None:
        """每个用例前重置 MinIO 客户端单例缓存。"""
        minio_module.reset_minio_client()

    def teardown_method(self) -> None:
        """每个用例后再次重置，避免污染其他测试。"""
        minio_module.reset_minio_client()

    @patch("app.core.minio.boto3")
    def test_get_minio_client_singleton(self, mock_boto3):
        """get_minio_client 应返回单例：多次调用只创建一次。"""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        c1 = minio_module.get_minio_client()
        c2 = minio_module.get_minio_client()

        assert c1 is c2
        assert mock_boto3.client.call_count == 1

    @patch("app.core.minio.boto3")
    def test_get_minio_client_uses_settings(self, mock_boto3):
        """客户端应使用 settings 中的 endpoint、credentials 构造。"""
        mock_boto3.client.return_value = MagicMock()
        minio_module.get_minio_client()

        call_kwargs = mock_boto3.client.call_args.kwargs
        # 服务名固定为 s3
        assert mock_boto3.client.call_args.args[0] == "s3"
        assert call_kwargs["endpoint_url"].startswith(("http://", "https://"))
        assert call_kwargs["aws_access_key_id"] == minio_module.settings.MINIO_ACCESS_KEY
        assert call_kwargs["aws_secret_access_key"] == minio_module.settings.MINIO_SECRET_KEY

    @patch("app.core.minio.boto3")
    def test_ensure_bucket_exists_when_present(self, mock_boto3):
        """bucket 已存在：head_bucket 成功，不应调用 create_bucket。"""
        mock_client = MagicMock()
        mock_client.head_bucket.return_value = {}
        mock_boto3.client.return_value = mock_client

        minio_module.ensure_bucket_exists()

        mock_client.head_bucket.assert_called_once_with(
            Bucket=minio_module.settings.MINIO_BUCKET
        )
        mock_client.create_bucket.assert_not_called()

    @patch("app.core.minio.boto3")
    def test_ensure_bucket_exists_creates_when_missing(self, mock_boto3):
        """bucket 不存在：head_bucket 抛 ClientError，应触发 create_bucket。"""
        mock_client = MagicMock()
        mock_client.head_bucket.side_effect = ClientError(
            error_response={"Error": {"Code": "404", "Message": "Not Found"}},
            operation_name="HeadBucket",
        )
        mock_boto3.client.return_value = mock_client

        minio_module.ensure_bucket_exists()

        mock_client.create_bucket.assert_called_once_with(
            Bucket=minio_module.settings.MINIO_BUCKET
        )
