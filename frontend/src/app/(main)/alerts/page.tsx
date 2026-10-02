"use client";

/**
 * 主管端 · 进度中心
 *
 * 两个页签：
 * 1. 进度预警 —— 系统每晚自动汇总最近 7 天全员日报，检出停滞 / 消失 /
 *    风险反复的工作项。主管只需在空闲时扫一眼并「确认」进入跟进。
 * 2. 全员日报 —— 按日期查看空间内所有成员的 AI 日报，可展开原文；
 *    更细颗粒度的追问走「AI 问答」（日报已回流入库，支持引用溯源）。
 *
 * 权限：需要目标空间的 write 权限（主管由管理员授权），普通成员访问
 * 会看到 403 提示。
 */

import { useCallback, useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  AlertTriangle,
  BellRing,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Flame,
  Info,
  Loader2,
  Users,
} from "lucide-react";

import { apiClient } from "@/lib/api-client";
import {
  alertsApi,
  reportsApi,
  type DailyReport,
  type ProgressAlert,
} from "@/lib/reports-api";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { useToast } from "@/components/ui/toast";

interface Space {
  id: string;
  name: string;
}

const SEVERITY_META: Record<
  ProgressAlert["severity"],
  { label: string; className: string; icon: typeof Flame }
> = {
  critical: {
    label: "严重",
    className: "bg-red-100 text-red-700",
    icon: Flame,
  },
  warning: {
    label: "警告",
    className: "bg-amber-100 text-amber-700",
    icon: AlertTriangle,
  },
  info: {
    label: "提示",
    className: "bg-blue-100 text-blue-700",
    icon: Info,
  },
};

const ALERT_STATUS_LABEL: Record<ProgressAlert["status"], string> = {
  open: "待处理",
  acknowledged: "已确认",
  resolved: "已解决",
};

export default function AlertsPage() {
  const [spaces, setSpaces] = useState<Space[]>([]);
  const [spaceId, setSpaceId] = useState("");
  const [tab, setTab] = useState<"alerts" | "reports">("alerts");

  const [alerts, setAlerts] = useState<ProgressAlert[]>([]);
  const [showAcknowledged, setShowAcknowledged] = useState(false);
  const [teamReports, setTeamReports] = useState<DailyReport[]>([]);
  const [reportDate, setReportDate] = useState("");
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const [loading, setLoading] = useState(false);
  const [forbidden, setForbidden] = useState(false);
  const { addToast } = useToast();

  useEffect(() => {
    (async () => {
      try {
        const data = await apiClient.get<Space[]>("/api/spaces");
        setSpaces(data);
        if (data.length > 0) setSpaceId(data[0].id);
      } catch {
        addToast({ type: "error", message: "加载空间列表失败" });
      }
    })();
  }, [addToast]);

  const loadAlerts = useCallback(
    async (sid: string) => {
      if (!sid) return;
      setLoading(true);
      setForbidden(false);
      try {
        const { alerts } = await alertsApi.list(
          sid,
          showAcknowledged ? undefined : "open"
        );
        setAlerts(alerts);
      } catch (err) {
        if (err instanceof Error && "status" in err && err.status === 403) {
          setForbidden(true);
        } else {
          addToast({ type: "error", message: "加载预警失败" });
        }
      } finally {
        setLoading(false);
      }
    },
    [showAcknowledged, addToast]
  );

  const loadTeamReports = useCallback(
    async (sid: string) => {
      if (!sid) return;
      setLoading(true);
      setForbidden(false);
      try {
        const { reports } = await reportsApi.listTeam(
          sid,
          reportDate || undefined
        );
        setTeamReports(reports);
      } catch (err) {
        if (err instanceof Error && "status" in err && err.status === 403) {
          setForbidden(true);
        } else {
          addToast({ type: "error", message: "加载全员日报失败" });
        }
      } finally {
        setLoading(false);
      }
    },
    [reportDate, addToast]
  );

  useEffect(() => {
    if (tab === "alerts") loadAlerts(spaceId);
    else loadTeamReports(spaceId);
  }, [spaceId, tab, loadAlerts, loadTeamReports]);

  const handleAcknowledge = async (alertId: string) => {
    try {
      await alertsApi.acknowledge(alertId);
      addToast({ type: "success", message: "已确认，进入跟进状态" });
      loadAlerts(spaceId);
    } catch {
      addToast({ type: "error", message: "确认失败" });
    }
  };

  return (
    <div className="mx-auto max-w-4xl space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">进度中心</h1>
          <p className="text-sm text-muted-foreground">
            无需口头汇报，碎片时间即可掌握团队每日进展
          </p>
        </div>
        {spaces.length > 1 && (
          <select
            className="rounded-md border bg-background px-3 py-2 text-sm"
            value={spaceId}
            onChange={(e) => setSpaceId(e.target.value)}
          >
            {spaces.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name}
              </option>
            ))}
          </select>
        )}
      </div>

      {/* 页签 */}
      <div className="flex gap-2 border-b">
        <button
          className={`flex items-center gap-1 border-b-2 px-4 py-2 text-sm font-medium transition-colors ${
            tab === "alerts"
              ? "border-primary text-primary"
              : "border-transparent text-muted-foreground hover:text-foreground"
          }`}
          onClick={() => setTab("alerts")}
        >
          <BellRing className="h-4 w-4" /> 进度预警
        </button>
        <button
          className={`flex items-center gap-1 border-b-2 px-4 py-2 text-sm font-medium transition-colors ${
            tab === "reports"
              ? "border-primary text-primary"
              : "border-transparent text-muted-foreground hover:text-foreground"
          }`}
          onClick={() => setTab("reports")}
        >
          <Users className="h-4 w-4" /> 全员日报
        </button>
      </div>

      {forbidden ? (
        <Card className="p-6 text-sm text-muted-foreground">
          当前账号不是该空间的主管（空间创建者）。请用创建该空间的账号登录，
          或联系管理员处理。
        </Card>
      ) : loading ? (
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" /> 加载中…
        </div>
      ) : tab === "alerts" ? (
        <>
          <label className="flex items-center gap-2 text-sm text-muted-foreground">
            <input
              type="checkbox"
              checked={showAcknowledged}
              onChange={(e) => setShowAcknowledged(e.target.checked)}
            />
            显示已确认 / 已解决
          </label>
          {alerts.length === 0 ? (
            <Card className="p-6 text-sm text-muted-foreground">
              暂无待处理的进度预警。系统每晚 21:23 自动分析最近 7 天日报，
              检出停滞、消失或风险反复的工作项。
            </Card>
          ) : (
            <div className="space-y-3">
              {alerts.map((alert) => {
                const meta = SEVERITY_META[alert.severity];
                return (
                  <Card key={alert.id} className="p-4">
                    <div className="flex items-start justify-between gap-4">
                      <div className="space-y-1">
                        <div className="flex items-center gap-2">
                          <meta.icon className="h-4 w-4 shrink-0" />
                          <span className="font-semibold">
                            {alert.work_item}
                          </span>
                          <span
                            className={`rounded-full px-2 py-0.5 text-xs ${meta.className}`}
                          >
                            {meta.label}
                          </span>
                          <span className="text-xs text-muted-foreground">
                            {ALERT_STATUS_LABEL[alert.status]}
                          </span>
                        </div>
                        <p className="text-sm text-muted-foreground">
                          {alert.reason}
                        </p>
                        <p className="text-xs text-muted-foreground">
                          首次发现 {alert.first_seen_date} · 最近活跃{" "}
                          {alert.last_active_date}
                          {alert.evidence?.stalled_days
                            ? ` · 已停滞约 ${alert.evidence.stalled_days} 天`
                            : ""}
                        </p>
                      </div>
                      {alert.status === "open" && (
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => handleAcknowledge(alert.id)}
                        >
                          <CheckCircle2 className="mr-1 h-4 w-4" />
                          确认
                        </Button>
                      )}
                    </div>
                  </Card>
                );
              })}
            </div>
          )}
        </>
      ) : (
        <>
          <div className="flex items-center gap-2 text-sm">
            <span className="text-muted-foreground">按日期筛选：</span>
            <input
              type="date"
              className="rounded-md border bg-background px-2 py-1"
              value={reportDate}
              onChange={(e) => setReportDate(e.target.value)}
            />
            {reportDate && (
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setReportDate("")}
              >
                清除
              </Button>
            )}
          </div>
          {teamReports.length === 0 ? (
            <Card className="p-6 text-sm text-muted-foreground">
              该条件下暂无日报。
            </Card>
          ) : (
            <div className="space-y-2">
              {teamReports.map((report) => {
                const expanded = expandedId === report.id;
                return (
                  <Card key={report.id} className="p-4">
                    <button
                      className="flex w-full items-center justify-between text-left"
                      onClick={() =>
                        setExpandedId(expanded ? null : report.id)
                      }
                    >
                      <div className="flex items-center gap-2 text-sm">
                        {expanded ? (
                          <ChevronDown className="h-4 w-4" />
                        ) : (
                          <ChevronRight className="h-4 w-4" />
                        )}
                        <span className="font-medium">{report.author}</span>
                        <span className="text-muted-foreground">
                          {report.report_date}
                        </span>
                      </div>
                      <span
                        className={
                          report.status === "completed"
                            ? "text-xs text-green-600"
                            : report.status === "failed"
                              ? "text-xs text-destructive"
                              : "text-xs text-muted-foreground"
                        }
                      >
                        {report.status === "completed"
                          ? "已完成"
                          : report.status === "failed"
                            ? "生成失败"
                            : "生成中"}
                      </span>
                    </button>
                    {expanded && (
                      <>
                        <Separator className="my-3" />
                        <article className="prose prose-sm max-w-none dark:prose-invert">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {report.content}
                          </ReactMarkdown>
                        </article>
                      </>
                    )}
                  </Card>
                );
              })}
            </div>
          )}
        </>
      )}
    </div>
  );
}
