"""管理员系统监控 API。

返回:
- 文档处理队列状态 (按 DocumentStatus 聚合的 4 类计数)
- 系统资源 (CPU / 内存 / 磁盘) 使用率

定位: 单机内网部署的轻量监控端点, 不替代 Prometheus / Grafana。
前端 ``/admin/monitoring`` 页面每 30 秒轮询一次。
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import require_admin
from app.core.database import get_db
from app.models.document import Document, DocumentStatus
from app.models.user import User

router = APIRouter(prefix="/api/admin/monitoring", tags=["admin", "monitoring"])


# ─── Schemas ────────────────────────────────────────────────────────────


class QueueStatus(BaseModel):
    """文档处理队列计数 (前端用 4 个卡片显示)。"""

    pending: int
    processing: int
    completed: int
    failed: int


class ResourceUsage(BaseModel):
    """系统资源使用率 (基于 psutil)。"""

    cpu_percent: float
    memory_percent: float
    storage_percent: float
    memory_used_gb: float
    memory_total_gb: float
    storage_used_gb: float
    storage_total_gb: float


class MonitoringResponse(BaseModel):
    queue: QueueStatus
    resources: ResourceUsage
    updated_at: datetime


# ─── Helpers ────────────────────────────────────────────────────────────


# 把 8 种 DocumentStatus 归到前端期望的 4 类
_PROCESSING_STATES = {
    DocumentStatus.parsing,
    DocumentStatus.cleaning,
    DocumentStatus.chunking,
    DocumentStatus.embedding,
    DocumentStatus.indexing,
}


async def _query_queue_status(db: AsyncSession) -> QueueStatus:
    """Aggregate document counts by status from PostgreSQL."""
    stmt = select(Document.status, func.count(Document.id)).group_by(Document.status)
    rows = (await db.execute(stmt)).all()

    counts: dict[DocumentStatus, int] = {row[0]: int(row[1]) for row in rows}
    pending = counts.get(DocumentStatus.pending, 0)
    completed = counts.get(DocumentStatus.completed, 0)
    failed = counts.get(DocumentStatus.failed, 0)
    processing = sum(counts.get(s, 0) for s in _PROCESSING_STATES)

    return QueueStatus(
        pending=pending,
        processing=processing,
        completed=completed,
        failed=failed,
    )


def _query_resource_usage() -> ResourceUsage:
    """Probe local system metrics via psutil.

    在容器内运行时, 这些数值反映容器视角 (cgroup) 而非宿主机整体。
    对于单机部署足够: 当容器到达资源上限时这些指标会先饱和。
    """
    import psutil

    bytes_per_gb = 1024**3

    # CPU: 0.1 秒采样窗口 (interval=None 会用上次调用作为基线, 第一次返回 0)
    # 0.1 不会阻塞健康轮询太久, 又能给出有意义的瞬时值
    cpu_percent = float(psutil.cpu_percent(interval=0.1))

    mem = psutil.virtual_memory()
    memory_percent = float(mem.percent)
    memory_total_gb = round(mem.total / bytes_per_gb, 2)
    memory_used_gb = round(mem.used / bytes_per_gb, 2)

    # Disk: 容器内 / 路径反映容器写入空间
    disk = psutil.disk_usage("/")
    storage_percent = float(disk.percent)
    storage_total_gb = round(disk.total / bytes_per_gb, 2)
    storage_used_gb = round(disk.used / bytes_per_gb, 2)

    return ResourceUsage(
        cpu_percent=cpu_percent,
        memory_percent=memory_percent,
        storage_percent=storage_percent,
        memory_used_gb=memory_used_gb,
        memory_total_gb=memory_total_gb,
        storage_used_gb=storage_used_gb,
        storage_total_gb=storage_total_gb,
    )


# ─── Endpoint ───────────────────────────────────────────────────────────


@router.get("", response_model=MonitoringResponse)
async def get_monitoring(
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> MonitoringResponse:
    """聚合返回队列状态 + 资源使用率。"""
    queue = await _query_queue_status(db)
    resources = _query_resource_usage()
    return MonitoringResponse(
        queue=queue,
        resources=resources,
        updated_at=datetime.now(timezone.utc),
    )
