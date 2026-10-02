"""Progress alert service —— 主管端「进度预警」工作流的核心。

背景（工作汇报场景的差异化能力）：传统周报/口头汇报里，主管的注意力
容易被「声音大」的事项占据，一些工作项悄悄停滞多日无人察觉。本服务
利用日报已持久化、可检索的特性，每日定时（Celery Beat）汇总空间内
最近 N 天的全员日报，交给 LLM 提取工作项并对比其跨天进度轨迹，识别：

- **停滞**：工作项连续 ≥2 天被提及但没有任何进展描述；
- **消失**：工作项之前持续推进，最近 ≥2 天从所有日报中消失且未见完成结论；
- **风险反复**：同一阻塞/风险连续多天出现且未见解除。

去重语义：同一 (space_id, work_item) 已有 open 预警时不新建，仅刷新
``last_active_date`` 与 ``evidence``；主管确认（acknowledge）后若再次
检出会重新 open，形成「预警 → 关注 → 再停滞 → 再预警」的闭环。
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import date, datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundException
from app.models.daily_report import DailyReport, DailyReportStatus
from app.models.progress_alert import (
    AlertSeverity,
    AlertStatus,
    ProgressAlert,
)
from app.models.space import Space
from app.models.user import User
from app.services.llm_gateway import LLMGateway, LLMGatewayError

logger = logging.getLogger(__name__)

# 预警分析窗口（天）
DEFAULT_LOOKBACK_DAYS = 7
# 喂给 LLM 的日报材料总字符数上限
MAX_REPORT_CHARS = 24000
# 每个空间一次最多产出的新预警数，防止 LLM 过度报警
MAX_ALERTS_PER_RUN = 10

ALERT_SYSTEM_PROMPT = """你是一名项目进度分析助手。输入是某个团队最近若干天的全员
工作日报（Markdown），你的任务是找出「进度异常」的工作项，帮助主管发现被
忽略的进度死角。

只报告以下三类异常，每类都要有日报原文依据，禁止臆测：
1. stalled —— 某工作项连续 2 天以上被提及，但日报中没有体现任何实质进展
   （措辞重复、没有新的产出物/数据/结论）。
2. disappeared —— 某工作项之前在持续推进，最近 2 天以上从所有日报中消失，
   且没有任何日报说明它已完成或暂停。
3. recurring_risk —— 同一阻塞或风险连续 2 天以上出现，且未见解除迹象。

严重度判定：停滞/消失 2-3 天为 warning，4 天及以上为 critical；
首次出现但影响面大的风险为 info。

严格只输出 JSON（不要输出 Markdown 代码块标记、不要任何额外文字）：
{"alerts": [{"work_item": "工作项名称", "category": "stalled|disappeared|recurring_risk",
"severity": "info|warning|critical", "reason": "给主管看的中文说明，50-120字，含判断依据",
"evidence_dates": ["YYYY-MM-DD", ...], "excerpts": ["日报原文关键句", ...],
"stalled_days": 2}]}

如果没有异常，输出 {"alerts": []}。最多输出 10 条。
"""


class AlertService:
    """进度预警的检测与查询服务。"""

    def __init__(self, db: AsyncSession):
        self.db = db

    # ─── 检测 ─────────────────────────────────────────────────────────

    async def detect_for_space(
        self,
        *,
        space_id: uuid.UUID,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        reference_date: date | None = None,
    ) -> list[ProgressAlert]:
        """对单个空间执行一次停滞检测，返回本次新建/刷新的预警。

        Args:
            space_id: 目标空间
            lookback_days: 汇总最近多少天的日报
            reference_date: 窗口右端（默认今天，便于测试注入）
        """
        end_date = reference_date or datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=lookback_days - 1)

        reports = await self._get_reports_in_window(space_id, start_date, end_date)
        distinct_days = {r.report_date for r in reports}
        if len(distinct_days) < 2:
            logger.info(
                "alert detection skipped: space=%s days_with_reports=%d",
                space_id,
                len(distinct_days),
            )
            return []

        payload = self._build_llm_payload(reports)
        findings = await self._analyze(payload)

        alerts: list[ProgressAlert] = []
        for finding in findings[:MAX_ALERTS_PER_RUN]:
            alert = await self._upsert_alert(
                space_id=space_id,
                finding=finding,
                today=end_date,
            )
            if alert:
                alerts.append(alert)

        await self.db.commit()
        return alerts

    async def detect_all_spaces(
        self,
        *,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    ) -> dict:
        """Beat 任务入口：遍历全部空间执行检测。

        单个空间失败不影响其它空间（与 pipeline 的「单文档失败不阻塞
        其它文档」语义一致）。
        """
        spaces = (await self.db.execute(select(Space))).scalars().all()
        summary = {"spaces": len(spaces), "alerts_new": 0, "errors": []}
        for space in spaces:
            try:
                alerts = await self.detect_for_space(
                    space_id=space.id, lookback_days=lookback_days
                )
                summary["alerts_new"] += len(alerts)
            except Exception as exc:  # noqa: BLE001 — 逐空间隔离
                logger.warning(
                    "alert detection failed: space=%s error=%s",
                    space.id,
                    exc,
                    exc_info=True,
                )
                summary["errors"].append(
                    {"space_id": str(space.id), "error": str(exc)[:500]}
                )
                await self.db.rollback()
        return summary

    # ─── 查询与处置 ────────────────────────────────────────────────────

    async def list_alerts(
        self,
        *,
        space_id: uuid.UUID,
        status: AlertStatus | None = None,
        skip: int = 0,
        limit: int = 50,
    ) -> tuple[list[ProgressAlert], int]:
        """主管端预警列表：critical 在前，其次按最近活跃日期倒序。"""
        stmt = select(ProgressAlert).where(ProgressAlert.space_id == space_id)
        count_stmt = select(func.count(ProgressAlert.id)).where(
            ProgressAlert.space_id == space_id
        )
        if status:
            stmt = stmt.where(ProgressAlert.status == status)
            count_stmt = count_stmt.where(ProgressAlert.status == status)

        severity_rank = {
            AlertSeverity.critical: 0,
            AlertSeverity.warning: 1,
            AlertSeverity.info: 2,
        }
        stmt = stmt.offset(skip).limit(limit)

        total = (await self.db.execute(count_stmt)).scalar_one()
        rows = list((await self.db.execute(stmt)).scalars().all())
        rows.sort(
            key=lambda a: (
                severity_rank.get(a.severity, 3),
                -a.last_active_date.toordinal(),
            )
        )
        return rows, total

    async def acknowledge(
        self, *, alert_id: uuid.UUID, operator: User
    ) -> ProgressAlert:
        """主管确认预警（已知晓，进入跟进状态）。"""
        stmt = select(ProgressAlert).where(ProgressAlert.id == alert_id)
        alert = (await self.db.execute(stmt)).scalar_one_or_none()
        if not alert:
            raise NotFoundException("ProgressAlert", str(alert_id))

        alert.status = AlertStatus.acknowledged
        alert.acknowledged_by = operator.id
        alert.acknowledged_at = datetime.now(timezone.utc)
        await self.db.flush()
        await self.db.commit()
        await self.db.refresh(alert)
        return alert

    # ─── 内部 ─────────────────────────────────────────────────────────

    async def _get_reports_in_window(
        self, space_id: uuid.UUID, start: date, end: date
    ) -> list[DailyReport]:
        stmt = (
            select(DailyReport)
            .where(
                DailyReport.space_id == space_id,
                DailyReport.status == DailyReportStatus.completed,
                DailyReport.report_date >= start,
                DailyReport.report_date <= end,
            )
            .order_by(DailyReport.report_date.asc(), DailyReport.user_id.asc())
        )
        return list((await self.db.execute(stmt)).scalars().all())

    @staticmethod
    def _build_llm_payload(reports: list[DailyReport]) -> str:
        """把窗口内日报拼成 LLM 输入，超出预算时优先保留最近的日期。"""
        sections: list[str] = []
        total = 0
        for report in reversed(reports):  # 最新的优先
            author = report.user.display_name or report.user.email if report.user else "未知成员"
            section = (
                f"## {report.report_date.isoformat()} · {author}\n"
                f"{report.content}"
            )
            if total + len(section) > MAX_REPORT_CHARS:
                break
            sections.append(section)
            total += len(section)
        sections.reverse()  # 恢复时间正序，便于 LLM 判断「连续天数」
        return "\n\n".join(sections)

    async def _analyze(self, payload: str) -> list[dict]:
        """调用 LLM 分析日报，鲁棒解析 JSON 输出。

        解析失败 / 调用失败都返回空列表——预警是增强能力，任何异常都
        不应该影响主链路（与查询增强的「失败回退原始检索」同一哲学）。
        """
        gateway = LLMGateway()
        try:
            resp = await gateway.complete(
                prompt=(
                    "以下是某团队最近的全员工作日报，请找出进度异常的工作项："
                    f"\n\n{payload}"
                ),
                system_prompt=ALERT_SYSTEM_PROMPT,
                max_tokens=4096,
            )
        except LLMGatewayError as exc:
            logger.warning("alert analysis LLM call failed: %s", exc)
            return []

        return self._parse_findings(resp.content or "")

    @staticmethod
    def _parse_findings(content: str) -> list[dict]:
        """从模型输出中提取 alerts JSON，容忍代码块包裹和前后噪声。"""
        text = content.strip()
        # 去掉 ```json ... ``` 包裹
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
        # 截取第一个 { 到最后一个 }，容忍模型输出前后多余文字
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            logger.warning("alert analysis: no JSON object found in output")
            return []
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            logger.warning("alert analysis: JSON parse failed: %s", exc)
            return []

        findings = data.get("alerts")
        if not isinstance(findings, list):
            return []
        # 过滤缺字段的脏数据
        return [
            f for f in findings
            if isinstance(f, dict) and f.get("work_item") and f.get("reason")
        ]

    async def _upsert_alert(
        self, *, space_id: uuid.UUID, finding: dict, today: date
    ) -> ProgressAlert | None:
        """同一工作项已有 open 预警则刷新证据，否则新建。"""
        work_item = str(finding["work_item"])[:500]
        stmt = select(ProgressAlert).where(
            ProgressAlert.space_id == space_id,
            ProgressAlert.work_item == work_item,
            ProgressAlert.status == AlertStatus.open,
        )
        existing = (await self.db.execute(stmt)).scalar_one_or_none()

        evidence = {
            "category": finding.get("category", "stalled"),
            "dates": finding.get("evidence_dates", []),
            "excerpts": finding.get("excerpts", [])[:5],
            "stalled_days": finding.get("stalled_days"),
        }
        try:
            severity = AlertSeverity(finding.get("severity", "warning"))
        except ValueError:
            severity = AlertSeverity.warning

        if existing:
            existing.reason = str(finding["reason"])
            existing.severity = severity
            existing.evidence = evidence
            existing.last_active_date = today
            await self.db.flush()
            return existing

        alert = ProgressAlert(
            space_id=space_id,
            work_item=work_item,
            reason=str(finding["reason"]),
            severity=severity,
            status=AlertStatus.open,
            evidence=evidence,
            first_seen_date=today,
            last_active_date=today,
        )
        self.db.add(alert)
        await self.db.flush()
        return alert
