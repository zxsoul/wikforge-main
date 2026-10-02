"""Daily report synthesis service —— 员工端「零填报」日报的核心编排。

流程（对应工作汇报场景的「机器采、机器写、人只看」）：

1. **聚合**：按 (space, user, report_date) 捞出员工当日上传且已处理完成的
   文档（截图 / 日志 / 文档），排除已生成的日报文档本身，避免日报回流
   后污染下一次聚合（检索结果自引用问题）。
2. **取料**：从 OpenSearch ``chunks`` 索引按 document_id 取回正文片段，
   按文档数均摊长度预算，保证 LLM 上下文可控。
3. **合成**：调用 :class:`LLMGateway` 生成固定四段式 Markdown 日报
   （今日完成 / 进行中与进度 / 阻塞与风险 / 明日计划）。
4. **回流**：日报 Markdown 存入 MinIO 并创建 Document，走与手工上传完全
   一致的 pipeline（解析 → 清洗 → 分块 → 向量化 → 双索引），主管端
   RAG 问答因此可以直接检索到日报内容，无需任何特殊检索逻辑。

失败语义：LLM 调用失败时日报行落为 ``failed``，不影响原始材料入库；
主管端的兜底是仍然可以直接检索原始材料（查询增强 + 多路召回链路不变）。
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import uuid
from datetime import date, datetime, time, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.exceptions import ForbiddenException, NotFoundException, ValidationException
from app.core.minio import ensure_bucket_exists, get_minio_client
from app.core.opensearch import INDEX_NAME, get_opensearch_client
from app.core.redis import get_redis
from app.models.daily_report import DailyReport, DailyReportStatus
from app.models.document import Document, DocumentStatus
from app.models.permission import ResourceType
from app.models.space import Space
from app.models.user import User
from app.services.llm_gateway import LLMGateway, LLMGatewayError
from app.services.permission_service import Action, PermissionService

try:
    from app.tasks.pipeline import submit_pipeline as _submit_pipeline
except Exception:  # pragma: no cover — Celery 不可用时 (单元测试) 不触发任务
    _submit_pipeline = None

logger = logging.getLogger(__name__)
settings = get_settings()

# 单个文档贡献给日报合成的最大字符数（按文档数均摊前的单文档上限）
MAX_CHARS_PER_DOCUMENT = 4000
# 喂给 LLM 的材料总字符数上限，超出部分按时间顺序截断
MAX_TOTAL_MATERIAL_CHARS = 16000
# 每个文档最多取回的 chunk 数
MAX_CHUNKS_PER_DOCUMENT = 12
# 日报文档标题前缀，用于聚合时排除日报自身、前端展示标识
REPORT_TITLE_PREFIX = "工作日报"

REPORT_SYSTEM_PROMPT = """你是一名严谨的工作汇报助手。根据员工当天上传的工作材料
（截图文字、日志、文档片段），合成一份客观、结构化的中文工作日报。

硬性要求：
- 只陈述材料中出现的事实，禁止编造材料中不存在的工作内容、数据或结论。
- 每个要点尽量保留可核对的信息：文件名、模块名、数据指标、时间节点。
- 无法从材料判断的部分直接写「材料中未体现」，不要硬凑。
- 输出固定四段式 Markdown，不要输出其他任何内容：

# 工作日报 · {author} · {date}

## 一、今日完成
（按事项分条列出，每条附材料依据）

## 二、进行中事项与进度
（事项 + 当前进度/状态）

## 三、阻塞与风险
（没有则写「无」）

## 四、明日计划
（材料中未体现则写「材料中未体现」）
"""


class DailyReportService:
    """日报合成与查询服务。"""

    def __init__(self, db: AsyncSession):
        self.db = db

    # ─── 生成 ─────────────────────────────────────────────────────────

    async def generate_daily_report(
        self,
        *,
        user: User,
        space_id: uuid.UUID,
        report_date: date,
        force: bool = False,
    ) -> DailyReport:
        """为指定员工合成某天的日报（幂等：默认已存在则直接返回）。

        Args:
            user: 日报归属员工
            space_id: 目标空间（部门/项目）
            report_date: 日报日期（按 UTC 日界聚合上传记录）
            force: True 时基于最新材料重新合成并覆盖旧日报

        Raises:
            ForbiddenException: 用户在该空间无写权限（不能替别人/无权空间生成）
            ValidationException: 当日没有可用的工作材料
        """
        await self._require_space_action(user, space_id, Action.write)

        existing = await self._get_report(space_id, user.id, report_date)
        if existing and existing.status == DailyReportStatus.completed and not force:
            return existing

        report = existing or DailyReport(
            space_id=space_id,
            user_id=user.id,
            report_date=report_date,
            title=self._build_title(user, report_date),
            content="",
        )
        report.status = DailyReportStatus.generating
        report.error_detail = None
        self.db.add(report)
        await self.db.flush()

        try:
            source_docs = await self._get_source_documents(
                user_id=user.id, space_id=space_id, report_date=report_date
            )
            if not source_docs:
                raise ValidationException(
                    f"{report_date.isoformat()} 当天没有已入库的工作材料，"
                    "请先上传截图、日志或文档"
                )

            materials = await self._collect_materials(source_docs)
            content = await self._synthesize(user, report_date, materials)

            report.content = content
            report.source_document_ids = [str(d.id) for d in source_docs]
            report.title = self._build_title(user, report_date)
            report.status = DailyReportStatus.completed
            report.generated_at = datetime.now(timezone.utc)
            await self.db.flush()

            # 日报回流入库（失败不阻塞：日报行已落库，主管仍可查看文本）
            try:
                document = await self._ingest_report_document(report, content)
                report.document_id = document.id
                await self.db.flush()
            except Exception:
                logger.warning(
                    "daily_report ingest failed: report_id=%s",
                    report.id,
                    exc_info=True,
                )
        except Exception as exc:
            report.status = DailyReportStatus.failed
            report.error_detail = str(exc)[:2000]
            # 先提交失败状态：get_db 依赖在异常时会回滚未提交的事务，
            # 这里主动 commit 让前端能看到「生成失败」并可点「重新生成」。
            await self.db.commit()
            raise

        await self.db.commit()
        await self.db.refresh(report)
        return report

    # ─── 查询 ─────────────────────────────────────────────────────────

    async def list_my_reports(
        self,
        *,
        user_id: uuid.UUID,
        space_id: uuid.UUID | None = None,
        skip: int = 0,
        limit: int = 20,
    ) -> tuple[list[DailyReport], int]:
        """员工查看自己的历史日报。"""
        stmt = select(DailyReport).where(DailyReport.user_id == user_id)
        count_stmt = select(func.count(DailyReport.id)).where(
            DailyReport.user_id == user_id
        )
        if space_id:
            stmt = stmt.where(DailyReport.space_id == space_id)
            count_stmt = count_stmt.where(DailyReport.space_id == space_id)
        stmt = stmt.order_by(DailyReport.report_date.desc()).offset(skip).limit(limit)

        total = (await self.db.execute(count_stmt)).scalar_one()
        rows = (await self.db.execute(stmt)).scalars().all()
        return list(rows), total

    async def list_team_reports(
        self,
        *,
        requester: User,
        space_id: uuid.UUID,
        report_date: date | None = None,
        skip: int = 0,
        limit: int = 50,
    ) -> tuple[list[DailyReport], int]:
        """主管查看空间（部门/项目）内全员日报。

        鉴权：系统管理员或空间创建者（主管）。普通成员只有 read/write
        材料权限，无法查看他人日报——天然实现「成员看自己、主管看全员」。
        """
        await self.require_space_manager(requester, space_id)

        stmt = select(DailyReport).where(DailyReport.space_id == space_id)
        count_stmt = select(func.count(DailyReport.id)).where(
            DailyReport.space_id == space_id
        )
        if report_date:
            stmt = stmt.where(DailyReport.report_date == report_date)
            count_stmt = count_stmt.where(DailyReport.report_date == report_date)
        stmt = stmt.order_by(
            DailyReport.report_date.desc(), DailyReport.created_at.desc()
        ).offset(skip).limit(limit)

        total = (await self.db.execute(count_stmt)).scalar_one()
        rows = (await self.db.execute(stmt)).scalars().all()
        return list(rows), total

    async def get_report(
        self, *, requester: User, report_id: uuid.UUID
    ) -> DailyReport:
        """查看单份日报：本人或空间主管（空间创建者/管理员）。"""
        stmt = select(DailyReport).where(DailyReport.id == report_id)
        report = (await self.db.execute(stmt)).scalar_one_or_none()
        if not report:
            raise NotFoundException("DailyReport", str(report_id))

        if report.user_id != requester.id:
            await self.require_space_manager(requester, report.space_id)
        return report

    # ─── 内部：材料聚合 ────────────────────────────────────────────────

    async def _get_source_documents(
        self, *, user_id: uuid.UUID, space_id: uuid.UUID, report_date: date
    ) -> list[Document]:
        """捞出员工当日上传且处理完成的原始材料文档。

        排除两类文档：
        - 标题以「工作日报」开头的回流日报（避免自引用污染）；
        - 已被任意日报引用过的回流文档（双保险，按 document_id 排除）。
        """
        day_start = datetime.combine(report_date, time.min, tzinfo=timezone.utc)
        day_end = datetime.combine(report_date, time.max, tzinfo=timezone.utc)

        report_doc_ids = select(DailyReport.document_id).where(
            DailyReport.space_id == space_id,
            DailyReport.document_id.is_not(None),
        )

        stmt = (
            select(Document)
            .where(
                Document.space_id == space_id,
                Document.uploaded_by == user_id,
                Document.status == DocumentStatus.completed,
                Document.created_at >= day_start,
                Document.created_at <= day_end,
                ~Document.title.startswith(REPORT_TITLE_PREFIX),
                ~Document.id.in_(report_doc_ids),
            )
            .order_by(Document.created_at.asc())
        )
        rows = (await self.db.execute(stmt)).scalars().all()
        return list(rows)

    async def _collect_materials(self, documents: list[Document]) -> str:
        """从 OpenSearch 取回各文档正文片段，拼成 LLM 输入材料。

        OpenSearch 客户端是同步的，用 ``asyncio.to_thread`` 避免阻塞事件循环
        （与 search_service 中的处理方式一致）。
        """
        doc_ids = [str(d.id) for d in documents]
        chunks_by_doc = await asyncio.to_thread(
            self._fetch_chunks_sync, doc_ids, MAX_CHUNKS_PER_DOCUMENT
        )

        per_doc_budget = max(
            500, min(MAX_CHARS_PER_DOCUMENT, MAX_TOTAL_MATERIAL_CHARS // max(len(documents), 1))
        )
        sections: list[str] = []
        total = 0
        for doc in documents:
            text = "\n".join(chunks_by_doc.get(str(doc.id), []))[:per_doc_budget]
            if not text.strip():
                continue
            section = f"### 材料：{doc.title}（{doc.file_type}，上传于 {doc.created_at:%H:%M}）\n{text}"
            if total + len(section) > MAX_TOTAL_MATERIAL_CHARS:
                break
            sections.append(section)
            total += len(section)

        if not sections:
            raise ValidationException(
                "当日材料尚未完成索引或内容为空，请稍后重试"
            )
        return "\n\n".join(sections)

    @staticmethod
    def _fetch_chunks_sync(
        document_ids: list[str], max_chunks_per_doc: int
    ) -> dict[str, list[str]]:
        """同步查询 OpenSearch，按 document_id 分组返回 chunk 正文。"""
        client = get_opensearch_client()
        result: dict[str, list[str]] = {doc_id: [] for doc_id in document_ids}
        size = min(1000, max_chunks_per_doc * max(len(document_ids), 1))
        resp = client.search(
            index=INDEX_NAME,
            body={
                "size": size,
                "_source": ["document_id", "content"],
                "query": {"terms": {"document_id": document_ids}},
            },
        )
        for hit in resp.get("hits", {}).get("hits", []):
            src = hit.get("_source", {})
            doc_id = src.get("document_id")
            content = src.get("content") or ""
            if doc_id in result and content.strip():
                if len(result[doc_id]) < max_chunks_per_doc:
                    result[doc_id].append(content.strip())
        return result

    # ─── 内部：LLM 合成 ────────────────────────────────────────────────

    async def _synthesize(
        self, user: User, report_date: date, materials: str
    ) -> str:
        author = user.display_name or user.email
        system_prompt = REPORT_SYSTEM_PROMPT.format(
            author=author, date=report_date.isoformat()
        )
        gateway = LLMGateway()
        try:
            resp = await gateway.complete(
                prompt=(
                    f"以下是 {author} 在 {report_date.isoformat()} 上传的"
                    f"工作材料，请合成当日工作日报：\n\n{materials}"
                ),
                system_prompt=system_prompt,
                max_tokens=2048,
            )
        except LLMGatewayError as exc:
            raise ValidationException(
                f"日报合成失败（LLM 网关: {exc.reason}），请稍后重试"
            ) from exc
        content = (resp.content or "").strip()
        if not content:
            raise ValidationException("日报合成失败：模型返回空内容，请稍后重试")
        return content

    # ─── 内部：日报回流入库 ────────────────────────────────────────────

    async def _ingest_report_document(
        self, report: DailyReport, content: str
    ) -> Document:
        """把日报 Markdown 作为新 Document 走完整 pipeline 入库。

        与 ``UploadService.import_url`` 保持同一语义：先提交 DB 再入队，
        保证 worker 拉取时一定能从 DB 读到文档记录。
        """
        content_bytes = content.encode("utf-8")
        storage_path = f"{report.space_id}/{uuid.uuid4()}/{report.title}.md"

        client = get_minio_client()
        ensure_bucket_exists()
        client.upload_fileobj(
            Fileobj=io.BytesIO(content_bytes),
            Bucket=settings.MINIO_BUCKET,
            Key=storage_path,
            ExtraArgs={"ContentType": "text/markdown"},
        )

        document = Document(
            space_id=report.space_id,
            folder_id=None,
            title=report.title,
            file_type="md",
            file_size=len(content_bytes),
            storage_path=storage_path,
            status=DocumentStatus.pending,
            retry_count=0,
            uploaded_by=report.user_id,
        )
        self.db.add(document)
        await self.db.flush()
        # 先提交，让 worker 可见；日报行的 document_id 关联由调用方随后 flush
        await self.db.commit()

        if _submit_pipeline is not None:
            try:
                await asyncio.to_thread(_submit_pipeline, str(document.id))
            except Exception:
                logger.warning(
                    "daily_report pipeline submit failed: document_id=%s",
                    document.id,
                    exc_info=True,
                )
        return document

    # ─── 内部：通用 ────────────────────────────────────────────────────

    async def _get_report(
        self, space_id: uuid.UUID, user_id: uuid.UUID, report_date: date
    ) -> DailyReport | None:
        stmt = select(DailyReport).where(
            DailyReport.space_id == space_id,
            DailyReport.user_id == user_id,
            DailyReport.report_date == report_date,
        )
        return (await self.db.execute(stmt)).scalar_one_or_none()

    async def _require_space_action(
        self, user: User, space_id: uuid.UUID, action: Action
    ) -> None:
        """空间级 ABAC 校验；管理员邮箱绕过（与 require_admin 同一口径）。"""
        if self._is_admin(user):
            return
        redis = await get_redis()
        perm = PermissionService(self.db, redis)
        allowed = await perm.check_access(
            user_id=user.id,
            resource_id=space_id,
            resource_type=ResourceType.space,
            action=action,
        )
        if not allowed:
            raise ForbiddenException("没有该空间的操作权限")

    async def require_space_manager(
        self, user: User, space_id: uuid.UUID
    ) -> None:
        """「主管」判定：系统管理员或空间创建者。

        为什么不用 write 权限：员工上传材料本身就需要 write，如果全员视图
        也用 write 守门，任何成员都能看所有人的日报。空间创建者即团队主管，
        是现有模型里唯一能把「主管」和「成员」区分开的信号；管理员在
        「系统管理 → 权限配置」里把空间转授给主管即完成授权。
        """
        if self._is_admin(user):
            return
        stmt = select(Space.created_by).where(Space.id == space_id)
        creator_id = (await self.db.execute(stmt)).scalar_one_or_none()
        if creator_id is None:
            raise NotFoundException("Space", str(space_id))
        if creator_id != user.id:
            raise ForbiddenException("需要该空间的主管权限")

    @staticmethod
    def _is_admin(user: User) -> bool:
        admin_email = os.environ.get("INITIAL_ADMIN_EMAIL", "").strip().lower()
        return bool(admin_email) and (user.email or "").strip().lower() == admin_email

    @staticmethod
    def _build_title(user: User, report_date: date) -> str:
        author = user.display_name or user.email.split("@")[0]
        return f"{REPORT_TITLE_PREFIX}-{author}-{report_date.isoformat()}"
