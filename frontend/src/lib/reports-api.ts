/**
 * 工作汇报场景 API 封装：日报（员工端/主管端）与进度预警（主管端）。
 *
 * 后端路由：
 * - POST /api/reports/generate      员工生成自己的日报
 * - GET  /api/reports/my            员工历史日报
 * - GET  /api/reports/team          主管查看空间全员日报（需空间 write 权限）
 * - GET  /api/reports/{id}          日报详情
 * - GET  /api/alerts                主管查看进度预警
 * - POST /api/alerts/{id}/acknowledge  主管确认预警
 */

import { apiClient } from "@/lib/api-client";

// ─── Types ───────────────────────────────────────────────────────────

export interface DailyReport {
  id: string;
  space_id: string;
  user_id: string;
  author: string;
  report_date: string;
  title: string;
  content: string;
  source_document_ids: string[];
  document_id: string | null;
  status: "generating" | "completed" | "failed";
  error_detail: string | null;
  generated_at: string | null;
  created_at: string;
}

export interface DailyReportList {
  reports: DailyReport[];
  total: number;
}

export interface ProgressAlert {
  id: string;
  space_id: string;
  work_item: string;
  reason: string;
  severity: "info" | "warning" | "critical";
  status: "open" | "acknowledged" | "resolved";
  evidence: {
    category?: string;
    dates?: string[];
    excerpts?: string[];
    stalled_days?: number | null;
  };
  first_seen_date: string;
  last_active_date: string;
  acknowledged_by: string | null;
  acknowledged_at: string | null;
  created_at: string;
}

export interface ProgressAlertList {
  alerts: ProgressAlert[];
  total: number;
}

// ─── Reports API ─────────────────────────────────────────────────────

export const reportsApi = {
  /** 生成（或获取已有的）当日日报；force=true 基于最新材料重新合成 */
  generate: (spaceId: string, force = false, reportDate?: string) =>
    apiClient.post<DailyReport>("/api/reports/generate", {
      space_id: spaceId,
      report_date: reportDate ?? null,
      force,
    }),

  /** 员工：我的历史日报 */
  listMine: (spaceId?: string, skip = 0, limit = 20) => {
    const params = new URLSearchParams();
    if (spaceId) params.set("space_id", spaceId);
    params.set("skip", String(skip));
    params.set("limit", String(limit));
    return apiClient.get<DailyReportList>(`/api/reports/my?${params}`);
  },

  /** 主管：空间全员日报（需要空间 write 权限） */
  listTeam: (spaceId: string, reportDate?: string, skip = 0, limit = 50) => {
    const params = new URLSearchParams({ space_id: spaceId });
    if (reportDate) params.set("report_date", reportDate);
    params.set("skip", String(skip));
    params.set("limit", String(limit));
    return apiClient.get<DailyReportList>(`/api/reports/team?${params}`);
  },

  get: (reportId: string) =>
    apiClient.get<DailyReport>(`/api/reports/${reportId}`),
};

// ─── Alerts API ──────────────────────────────────────────────────────

export const alertsApi = {
  /** 主管：空间进度预警列表 */
  list: (spaceId: string, status?: string, skip = 0, limit = 50) => {
    const params = new URLSearchParams({ space_id: spaceId });
    if (status) params.set("status", status);
    params.set("skip", String(skip));
    params.set("limit", String(limit));
    return apiClient.get<ProgressAlertList>(`/api/alerts?${params}`);
  },

  /** 主管：确认预警（已知晓，进入跟进状态） */
  acknowledge: (alertId: string) =>
    apiClient.post<ProgressAlert>(`/api/alerts/${alertId}/acknowledge`),
};
