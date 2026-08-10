"use client";

import * as React from "react";
import {
  AlertCircle,
  CheckCircle2,
  Clock,
  Lightbulb,
  Loader2,
  RefreshCw,
  Sparkles,
  ThumbsDown,
  ThumbsUp,
  Wand2,
} from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/components/ui/toast";
import { cn } from "@/lib/utils";
import { apiClient, ApiClientError } from "@/lib/api-client";

// ─── Types ─────────────────────────────────────────────────────────────

interface AggregationResponse {
  total_count: number;
  thumbs_up_count: number;
  thumbs_down_count: number;
  issue_count: number;
  by_issue_category: Record<string, number>;
  by_profile: Record<string, number>;
  by_document: Record<string, number>;
  by_date: Record<string, number>;
}

interface ErrorPattern {
  profile_id: string;
  profile_name: string;
  issue_category: string;
  occurrence_count: number;
  sample_queries: string[];
  first_seen: string | null;
  last_seen: string | null;
}

interface Suggestion {
  type: string;
  target_id: string;
  target_name: string;
  recommendation: Record<string, unknown>;
  evidence: string[];
  confidence: number;
  description: string;
}

interface ReprocessProgress {
  task_id: string;
  total_documents: number;
  processed_documents: number;
  status: string;
  progress_percent: number;
  created_at: string;
  error: string | null;
}

interface AnalysisFilters {
  profile_id: string;
  document_id: string;
  feedback_type: string;
  issue_category: string;
  start_date: string;
  end_date: string;
}

const ISSUE_CATEGORY_LABELS: Record<string, string> = {
  irrelevant: "结果不相关",
  missing_info: "缺少关键信息",
  citation_error: "引用错误",
  format: "格式问题",
  other: "其他",
  general_negative: "一般差评",
};

const SUGGESTION_TYPE_LABELS: Record<string, string> = {
  adjust_chunking: "调整分块策略",
  add_term: "新增领域术语",
  update_boilerplate: "更新模板规则",
  adjust_heading_rules: "调整标题规则",
  add_query_template: "新增查询模板",
};

const FEEDBACK_TYPE_OPTIONS: Array<{ value: string; label: string }> = [
  { value: "", label: "全部" },
  { value: "thumbs_up", label: "赞同" },
  { value: "thumbs_down", label: "反对" },
  { value: "issue", label: "问题" },
];

const ISSUE_CATEGORY_OPTIONS: Array<{ value: string; label: string }> = [
  { value: "", label: "全部" },
  { value: "irrelevant", label: "结果不相关" },
  { value: "missing_info", label: "缺少关键信息" },
  { value: "citation_error", label: "引用错误" },
  { value: "format", label: "格式问题" },
  { value: "other", label: "其他" },
];

type TabKey =
  | "overview"
  | "profile"
  | "document"
  | "category"
  | "patterns"
  | "suggestions"
  | "reprocess";

const TABS: Array<{ key: TabKey; label: string }> = [
  { key: "overview", label: "总览" },
  { key: "profile", label: "按 Profile" },
  { key: "document", label: "按文档" },
  { key: "category", label: "按问题类型" },
  { key: "patterns", label: "错误模式" },
  { key: "suggestions", label: "优化建议" },
  { key: "reprocess", label: "重处理进度" },
];

// ─── Utilities ─────────────────────────────────────────────────────────

function buildAnalysisQuery(filters: AnalysisFilters): string {
  const params = new URLSearchParams();
  Object.entries(filters).forEach(([key, value]) => {
    const trimmed = (value ?? "").trim();
    if (trimmed) params.append(key, trimmed);
  });
  const qs = params.toString();
  return qs ? `?${qs}` : "";
}

function formatPercent(value: number, total: number): string {
  if (!total) return "0%";
  return `${Math.round((value / total) * 100)}%`;
}

function formatDateTime(value: string | null): string {
  if (!value) return "—";
  try {
    return new Date(value).toLocaleString("zh-CN");
  } catch {
    return value;
  }
}

// ─── Page ──────────────────────────────────────────────────────────────

export default function AdminFeedbackPage() {
  const { addToast } = useToast();
  const [activeTab, setActiveTab] = React.useState<TabKey>("overview");

  const [filters, setFilters] = React.useState<AnalysisFilters>({
    profile_id: "",
    document_id: "",
    feedback_type: "",
    issue_category: "",
    start_date: "",
    end_date: "",
  });
  const [pendingFilters, setPendingFilters] = React.useState<AnalysisFilters>(
    filters
  );

  const [analysis, setAnalysis] = React.useState<AggregationResponse | null>(
    null
  );
  const [analysisLoading, setAnalysisLoading] = React.useState(false);

  const [patterns, setPatterns] = React.useState<ErrorPattern[]>([]);
  const [patternsLoading, setPatternsLoading] = React.useState(false);
  const [minOccurrences, setMinOccurrences] = React.useState(3);
  const [daysLookback, setDaysLookback] = React.useState(30);

  const [suggestions, setSuggestions] = React.useState<Suggestion[]>([]);
  const [suggestionsLoading, setSuggestionsLoading] = React.useState(false);
  const [applyingSuggestion, setApplyingSuggestion] = React.useState<
    string | null
  >(null);

  const [taskIds, setTaskIds] = React.useState<string[]>([]);
  const [progressMap, setProgressMap] = React.useState<
    Record<string, ReprocessProgress>
  >({});
  const [progressLoading, setProgressLoading] = React.useState(false);
  const [taskInput, setTaskInput] = React.useState("");

  // ─── Data fetchers ──────────────────────────────────────────────────

  const fetchAnalysis = React.useCallback(async () => {
    setAnalysisLoading(true);
    try {
      const data = await apiClient.get<AggregationResponse>(
        `/api/admin/feedback/analysis${buildAnalysisQuery(filters)}`
      );
      setAnalysis(data);
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({
          type: "error",
          message: "加载反馈聚合失败",
          description: err.message,
        });
      }
    } finally {
      setAnalysisLoading(false);
    }
  }, [filters, addToast]);

  const fetchPatterns = React.useCallback(async () => {
    setPatternsLoading(true);
    try {
      const params = new URLSearchParams({
        min_occurrences: String(minOccurrences),
        days_lookback: String(daysLookback),
      });
      const data = await apiClient.get<ErrorPattern[]>(
        `/api/admin/feedback/patterns?${params.toString()}`
      );
      setPatterns(data);
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({
          type: "error",
          message: "加载错误模式失败",
          description: err.message,
        });
      }
    } finally {
      setPatternsLoading(false);
    }
  }, [minOccurrences, daysLookback, addToast]);

  const fetchSuggestions = React.useCallback(async () => {
    setSuggestionsLoading(true);
    try {
      const data = await apiClient.get<Suggestion[]>(
        "/api/admin/feedback/suggestions"
      );
      setSuggestions(data);
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({
          type: "error",
          message: "加载优化建议失败",
          description: err.message,
        });
      }
    } finally {
      setSuggestionsLoading(false);
    }
  }, [addToast]);

  const fetchProgress = React.useCallback(
    async (taskId: string): Promise<ReprocessProgress | null> => {
      try {
        return await apiClient.get<ReprocessProgress>(
          `/api/admin/feedback/reprocess/${taskId}`
        );
      } catch (err) {
        if (err instanceof ApiClientError) {
          addToast({
            type: "error",
            message: `加载任务 ${taskId} 进度失败`,
            description: err.message,
          });
        }
        return null;
      }
    },
    [addToast]
  );

  const refreshAllProgress = React.useCallback(async () => {
    if (taskIds.length === 0) return;
    setProgressLoading(true);
    try {
      const updates = await Promise.all(
        taskIds.map((id) => fetchProgress(id))
      );
      const next: Record<string, ReprocessProgress> = {};
      updates.forEach((entry, idx) => {
        if (entry) next[taskIds[idx]] = entry;
      });
      setProgressMap((prev) => ({ ...prev, ...next }));
    } finally {
      setProgressLoading(false);
    }
  }, [taskIds, fetchProgress]);

  // ─── Initial loads & polling ────────────────────────────────────────

  React.useEffect(() => {
    void fetchAnalysis();
  }, [fetchAnalysis]);

  React.useEffect(() => {
    if (activeTab === "patterns") void fetchPatterns();
  }, [activeTab, fetchPatterns]);

  React.useEffect(() => {
    if (activeTab === "suggestions") void fetchSuggestions();
  }, [activeTab, fetchSuggestions]);

  React.useEffect(() => {
    if (activeTab !== "reprocess" || taskIds.length === 0) return;
    // 仅对仍在 pending/in_progress 的任务做轮询，已完成的不再刷新
    const hasActive = taskIds.some((id) => {
      const entry = progressMap[id];
      return !entry || ["pending", "in_progress"].includes(entry.status);
    });
    if (!hasActive) return;
    const handle = setInterval(() => {
      void refreshAllProgress();
    }, 5000);
    return () => clearInterval(handle);
  }, [activeTab, taskIds, progressMap, refreshAllProgress]);

  // ─── Actions ────────────────────────────────────────────────────────

  const handleApplyFilters = () => {
    setFilters(pendingFilters);
  };

  const handleResetFilters = () => {
    const empty: AnalysisFilters = {
      profile_id: "",
      document_id: "",
      feedback_type: "",
      issue_category: "",
      start_date: "",
      end_date: "",
    };
    setPendingFilters(empty);
    setFilters(empty);
  };

  const handleApplySuggestion = async (suggestion: Suggestion) => {
    const key = `${suggestion.type}:${suggestion.target_id}`;
    setApplyingSuggestion(key);
    try {
      let response: { success: boolean; message: string; reprocessing_task_id: string | null };
      if (suggestion.type === "add_term") {
        const newTerms = Array.isArray(suggestion.recommendation?.new_terms)
          ? (suggestion.recommendation.new_terms as Array<Record<string, unknown>>)
          : [];
        response = await apiClient.post(
          "/api/admin/feedback/apply/dictionary",
          {
            dictionary_id: suggestion.target_id,
            new_terms: newTerms,
          }
        );
      } else {
        response = await apiClient.post(
          "/api/admin/feedback/apply/profile",
          {
            profile_id: suggestion.target_id,
            updates: suggestion.recommendation,
          }
        );
      }
      addToast({
        type: "success",
        message: "已应用建议",
        description: response.message,
      });
      if (response.reprocessing_task_id) {
        setTaskIds((prev) =>
          prev.includes(response.reprocessing_task_id!)
            ? prev
            : [response.reprocessing_task_id!, ...prev]
        );
        const progress = await fetchProgress(response.reprocessing_task_id);
        if (progress) {
          setProgressMap((prev) => ({
            ...prev,
            [response.reprocessing_task_id!]: progress,
          }));
        }
        setActiveTab("reprocess");
      }
      // 刷新建议列表，已应用的项目下一次生成时会消失
      void fetchSuggestions();
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({
          type: "error",
          message: "应用建议失败",
          description: err.message,
        });
      }
    } finally {
      setApplyingSuggestion(null);
    }
  };

  const handleAttachTaskId = async () => {
    const id = taskInput.trim();
    if (!id) return;
    setTaskInput("");
    setTaskIds((prev) => (prev.includes(id) ? prev : [id, ...prev]));
    const progress = await fetchProgress(id);
    if (progress) {
      setProgressMap((prev) => ({ ...prev, [id]: progress }));
    }
  };

  // ─── Render ─────────────────────────────────────────────────────────

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">反馈分析</h1>
        <p className="text-muted-foreground mt-1">
          聚合检索反馈、识别错误模式并一键应用优化建议
        </p>
      </div>

      <FilterBar
        filters={pendingFilters}
        onChange={setPendingFilters}
        onApply={handleApplyFilters}
        onReset={handleResetFilters}
        loading={analysisLoading}
      />

      <TabBar active={activeTab} onChange={setActiveTab} />

      {activeTab === "overview" && (
        <OverviewSection analysis={analysis} loading={analysisLoading} />
      )}
      {activeTab === "profile" && (
        <BreakdownSection
          title="按 Profile 聚合"
          description="按文档解析策略 (Profile) 维度展示反馈数量"
          data={analysis?.by_profile ?? {}}
          total={analysis?.total_count ?? 0}
          emptyHint="所选条件下没有匹配的 Profile 反馈"
          loading={analysisLoading}
        />
      )}
      {activeTab === "document" && (
        <BreakdownSection
          title="按文档聚合"
          description="按 returned_results 中出现的文档/分块 ID 统计命中"
          data={analysis?.by_document ?? {}}
          total={analysis?.total_count ?? 0}
          emptyHint="所选条件下没有任何被引用的文档"
          loading={analysisLoading}
        />
      )}
      {activeTab === "category" && (
        <BreakdownSection
          title="按问题类型聚合"
          description="按 issue_category 展示反馈分布，可识别主要痛点"
          data={analysis?.by_issue_category ?? {}}
          total={analysis?.total_count ?? 0}
          emptyHint="所选条件下没有问题类型反馈"
          labelMap={ISSUE_CATEGORY_LABELS}
          loading={analysisLoading}
        />
      )}
      {activeTab === "patterns" && (
        <PatternsSection
          patterns={patterns}
          loading={patternsLoading}
          minOccurrences={minOccurrences}
          daysLookback={daysLookback}
          onChangeThreshold={setMinOccurrences}
          onChangeLookback={setDaysLookback}
          onRefresh={fetchPatterns}
        />
      )}
      {activeTab === "suggestions" && (
        <SuggestionsSection
          suggestions={suggestions}
          loading={suggestionsLoading}
          onRefresh={fetchSuggestions}
          onApply={handleApplySuggestion}
          applyingKey={applyingSuggestion}
        />
      )}
      {activeTab === "reprocess" && (
        <ReprocessSection
          taskIds={taskIds}
          progressMap={progressMap}
          loading={progressLoading}
          taskInput={taskInput}
          onChangeTaskInput={setTaskInput}
          onAttachTask={handleAttachTaskId}
          onRefresh={refreshAllProgress}
        />
      )}
    </div>
  );
}

// ─── Filter bar ────────────────────────────────────────────────────────

function FilterBar({
  filters,
  onChange,
  onApply,
  onReset,
  loading,
}: {
  filters: AnalysisFilters;
  onChange: (next: AnalysisFilters) => void;
  onApply: () => void;
  onReset: () => void;
  loading: boolean;
}) {
  const update = <K extends keyof AnalysisFilters>(
    key: K,
    value: AnalysisFilters[K]
  ) => onChange({ ...filters, [key]: value });

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">筛选条件</CardTitle>
        <CardDescription>
          支持多维过滤，所有过滤条件之间为「AND」关系
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          <div className="space-y-1.5">
            <Label htmlFor="filter-profile">Profile ID</Label>
            <Input
              id="filter-profile"
              placeholder="UUID"
              value={filters.profile_id}
              onChange={(e) => update("profile_id", e.target.value)}
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="filter-document">文档 / Chunk ID</Label>
            <Input
              id="filter-document"
              placeholder="UUID 或字符串"
              value={filters.document_id}
              onChange={(e) => update("document_id", e.target.value)}
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="filter-feedback-type">反馈类型</Label>
            <select
              id="filter-feedback-type"
              className={cn(
                "flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              )}
              value={filters.feedback_type}
              onChange={(e) => update("feedback_type", e.target.value)}
            >
              {FEEDBACK_TYPE_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="filter-issue">问题类型</Label>
            <select
              id="filter-issue"
              className={cn(
                "flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              )}
              value={filters.issue_category}
              onChange={(e) => update("issue_category", e.target.value)}
            >
              {ISSUE_CATEGORY_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="filter-start">开始时间</Label>
            <Input
              id="filter-start"
              type="datetime-local"
              value={filters.start_date}
              onChange={(e) => update("start_date", e.target.value)}
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="filter-end">结束时间</Label>
            <Input
              id="filter-end"
              type="datetime-local"
              value={filters.end_date}
              onChange={(e) => update("end_date", e.target.value)}
            />
          </div>
        </div>
        <div className="mt-4 flex items-center gap-2">
          <Button onClick={onApply} disabled={loading}>
            {loading ? (
              <Loader2 className="mr-1 h-4 w-4 animate-spin" />
            ) : (
              <RefreshCw className="mr-1 h-4 w-4" />
            )}
            应用筛选
          </Button>
          <Button variant="outline" onClick={onReset} disabled={loading}>
            重置
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

// ─── Tabs ──────────────────────────────────────────────────────────────

function TabBar({
  active,
  onChange,
}: {
  active: TabKey;
  onChange: (next: TabKey) => void;
}) {
  return (
    <div
      role="tablist"
      aria-label="反馈分析视图"
      className="flex flex-wrap gap-1 border-b"
    >
      {TABS.map((tab) => {
        const isActive = tab.key === active;
        return (
          <button
            key={tab.key}
            role="tab"
            type="button"
            aria-selected={isActive}
            onClick={() => onChange(tab.key)}
            className={cn(
              "rounded-t-md px-3 py-2 text-sm font-medium transition-colors",
              "border border-b-0 -mb-px",
              isActive
                ? "border-border bg-background text-foreground"
                : "border-transparent text-muted-foreground hover:text-foreground"
            )}
          >
            {tab.label}
          </button>
        );
      })}
    </div>
  );
}

// ─── Overview ──────────────────────────────────────────────────────────

function OverviewSection({
  analysis,
  loading,
}: {
  analysis: AggregationResponse | null;
  loading: boolean;
}) {
  const total = analysis?.total_count ?? 0;
  const stats = [
    {
      label: "反馈总数",
      value: total,
      icon: Sparkles,
      tone: "text-foreground",
    },
    {
      label: "赞同 (thumbs_up)",
      value: analysis?.thumbs_up_count ?? 0,
      icon: ThumbsUp,
      tone: "text-emerald-600",
      sub: total ? formatPercent(analysis?.thumbs_up_count ?? 0, total) : undefined,
    },
    {
      label: "反对 (thumbs_down)",
      value: analysis?.thumbs_down_count ?? 0,
      icon: ThumbsDown,
      tone: "text-amber-600",
      sub: total
        ? formatPercent(analysis?.thumbs_down_count ?? 0, total)
        : undefined,
    },
    {
      label: "问题反馈 (issue)",
      value: analysis?.issue_count ?? 0,
      icon: AlertCircle,
      tone: "text-red-600",
      sub: total ? formatPercent(analysis?.issue_count ?? 0, total) : undefined,
    },
  ];

  return (
    <div className="space-y-4">
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {stats.map((s) => (
          <Card key={s.label}>
            <CardHeader className="flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                {s.label}
              </CardTitle>
              <s.icon className={cn("h-4 w-4", s.tone)} />
            </CardHeader>
            <CardContent>
              <div className={cn("text-2xl font-bold", s.tone)}>
                {loading ? <Loader2 className="h-6 w-6 animate-spin" /> : s.value}
              </div>
              {s.sub && (
                <p className="text-xs text-muted-foreground mt-1">
                  占比 {s.sub}
                </p>
              )}
            </CardContent>
          </Card>
        ))}
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">每日反馈趋势</CardTitle>
          <CardDescription>
            过去时间段内各日反馈数量（按 by_date 聚合）
          </CardDescription>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="py-8 text-center text-sm text-muted-foreground">
              <Loader2 className="mx-auto h-5 w-5 animate-spin" />
            </div>
          ) : (
            <DateBars data={analysis?.by_date ?? {}} />
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function DateBars({ data }: { data: Record<string, number> }) {
  const entries = Object.entries(data).sort(([a], [b]) => a.localeCompare(b));
  if (entries.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">所选条件下暂无反馈数据</p>
    );
  }
  const max = Math.max(...entries.map(([, v]) => v));
  return (
    <div className="space-y-2">
      {entries.map(([date, count]) => {
        const ratio = max > 0 ? count / max : 0;
        return (
          <div key={date} className="flex items-center gap-3 text-sm">
            <span className="w-24 shrink-0 font-mono text-xs text-muted-foreground">
              {date}
            </span>
            <div className="relative flex-1 overflow-hidden rounded bg-muted">
              <div
                className="h-5 bg-primary/70"
                style={{ width: `${Math.max(ratio * 100, 2)}%` }}
              />
            </div>
            <span className="w-10 text-right font-mono">{count}</span>
          </div>
        );
      })}
    </div>
  );
}

// ─── Breakdown (按 profile / document / category) ──────────────────────

function BreakdownSection({
  title,
  description,
  data,
  total,
  emptyHint,
  labelMap,
  loading,
}: {
  title: string;
  description: string;
  data: Record<string, number>;
  total: number;
  emptyHint: string;
  labelMap?: Record<string, string>;
  loading: boolean;
}) {
  const entries = Object.entries(data).sort(([, a], [, b]) => b - a);
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="py-8 text-center text-sm text-muted-foreground">
            <Loader2 className="mx-auto h-5 w-5 animate-spin" />
          </div>
        ) : entries.length === 0 ? (
          <p className="text-sm text-muted-foreground">{emptyHint}</p>
        ) : (
          <div className="space-y-2">
            {entries.map(([key, count]) => {
              const max = entries[0][1] || 1;
              const ratio = count / max;
              const label = labelMap?.[key] ?? key;
              const percent = total ? formatPercent(count, total) : "—";
              return (
                <div
                  key={key}
                  className="grid grid-cols-[1fr_auto] gap-x-3 gap-y-1 text-sm"
                >
                  <span className="truncate font-medium" title={key}>
                    {label}
                  </span>
                  <span className="font-mono text-xs text-muted-foreground">
                    {count}（{percent}）
                  </span>
                  <div className="col-span-2 h-2 overflow-hidden rounded bg-muted">
                    <div
                      className="h-2 bg-primary/70"
                      style={{ width: `${Math.max(ratio * 100, 2)}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// ─── Patterns ──────────────────────────────────────────────────────────

function PatternsSection({
  patterns,
  loading,
  minOccurrences,
  daysLookback,
  onChangeThreshold,
  onChangeLookback,
  onRefresh,
}: {
  patterns: ErrorPattern[];
  loading: boolean;
  minOccurrences: number;
  daysLookback: number;
  onChangeThreshold: (n: number) => void;
  onChangeLookback: (n: number) => void;
  onRefresh: () => void;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">错误模式</CardTitle>
        <CardDescription>
          同一 Profile 下相同问题类型重复出现的反馈聚合
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-4 md:grid-cols-3">
          <div className="space-y-1.5">
            <Label htmlFor="patterns-threshold">最小出现次数</Label>
            <Input
              id="patterns-threshold"
              type="number"
              min={1}
              value={minOccurrences}
              onChange={(e) =>
                onChangeThreshold(Math.max(1, Number(e.target.value) || 1))
              }
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="patterns-lookback">回看天数</Label>
            <Input
              id="patterns-lookback"
              type="number"
              min={1}
              max={365}
              value={daysLookback}
              onChange={(e) =>
                onChangeLookback(
                  Math.min(365, Math.max(1, Number(e.target.value) || 1))
                )
              }
            />
          </div>
          <div className="flex items-end">
            <Button onClick={onRefresh} disabled={loading}>
              {loading ? (
                <Loader2 className="mr-1 h-4 w-4 animate-spin" />
              ) : (
                <RefreshCw className="mr-1 h-4 w-4" />
              )}
              重新分析
            </Button>
          </div>
        </div>

        {loading ? (
          <div className="py-8 text-center text-sm text-muted-foreground">
            <Loader2 className="mx-auto h-5 w-5 animate-spin" />
          </div>
        ) : patterns.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            未检测到任何达到阈值的错误模式
          </p>
        ) : (
          <div className="space-y-3">
            {patterns.map((p) => (
              <Card key={`${p.profile_id}-${p.issue_category}`}>
                <CardContent className="space-y-2 pt-4">
                  <div className="flex flex-wrap items-baseline gap-2">
                    <span className="font-semibold">{p.profile_name}</span>
                    <span className="rounded bg-amber-100 px-2 py-0.5 text-xs text-amber-900 dark:bg-amber-950 dark:text-amber-200">
                      {ISSUE_CATEGORY_LABELS[p.issue_category] ??
                        p.issue_category}
                    </span>
                    <span className="text-sm text-muted-foreground">
                      重复出现 {p.occurrence_count} 次
                    </span>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    首次：{formatDateTime(p.first_seen)}　最近：
                    {formatDateTime(p.last_seen)}
                  </p>
                  {p.sample_queries.length > 0 && (
                    <div className="space-y-1">
                      <p className="text-xs font-medium text-muted-foreground">
                        样本查询
                      </p>
                      <ul className="ml-4 list-disc space-y-0.5 text-sm">
                        {p.sample_queries.slice(0, 5).map((q, idx) => (
                          <li key={idx} className="truncate" title={q}>
                            {q}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// ─── Suggestions ───────────────────────────────────────────────────────

function SuggestionsSection({
  suggestions,
  loading,
  onRefresh,
  onApply,
  applyingKey,
}: {
  suggestions: Suggestion[];
  loading: boolean;
  onRefresh: () => void;
  onApply: (s: Suggestion) => void;
  applyingKey: string | null;
}) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-start justify-between">
        <div>
          <CardTitle className="text-base">优化建议</CardTitle>
          <CardDescription>
            系统基于错误模式生成的优化建议，确认后可一键应用
          </CardDescription>
        </div>
        <Button variant="outline" size="sm" onClick={onRefresh} disabled={loading}>
          {loading ? (
            <Loader2 className="mr-1 h-4 w-4 animate-spin" />
          ) : (
            <RefreshCw className="mr-1 h-4 w-4" />
          )}
          刷新
        </Button>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="py-8 text-center text-sm text-muted-foreground">
            <Loader2 className="mx-auto h-5 w-5 animate-spin" />
          </div>
        ) : suggestions.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            暂无优化建议。建议会在错误模式累计后自动生成。
          </p>
        ) : (
          <div className="space-y-3">
            {suggestions.map((s) => {
              const key = `${s.type}:${s.target_id}`;
              const isApplying = applyingKey === key;
              return (
                <Card key={key}>
                  <CardContent className="space-y-3 pt-4">
                    <div className="flex flex-wrap items-baseline gap-2">
                      <Lightbulb className="h-4 w-4 text-amber-500" />
                      <span className="font-semibold">
                        {SUGGESTION_TYPE_LABELS[s.type] ?? s.type}
                      </span>
                      <span className="rounded bg-muted px-2 py-0.5 text-xs">
                        目标：{s.target_name}
                      </span>
                      <span className="text-xs text-muted-foreground">
                        置信度 {(s.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                    <p className="text-sm">{s.description}</p>
                    {s.evidence.length > 0 && (
                      <details className="text-sm">
                        <summary className="cursor-pointer text-xs font-medium text-muted-foreground">
                          查看 {s.evidence.length} 条证据
                        </summary>
                        <ul className="ml-4 mt-1 list-disc space-y-0.5">
                          {s.evidence.slice(0, 8).map((e, idx) => (
                            <li key={idx} className="truncate" title={e}>
                              {e}
                            </li>
                          ))}
                        </ul>
                      </details>
                    )}
                    <details className="text-sm">
                      <summary className="cursor-pointer text-xs font-medium text-muted-foreground">
                        查看建议详情
                      </summary>
                      <pre className="mt-1 overflow-auto rounded bg-muted p-2 text-xs">
                        {JSON.stringify(s.recommendation, null, 2)}
                      </pre>
                    </details>
                    <div>
                      <Button
                        size="sm"
                        onClick={() => onApply(s)}
                        disabled={isApplying || applyingKey !== null}
                      >
                        {isApplying ? (
                          <Loader2 className="mr-1 h-4 w-4 animate-spin" />
                        ) : (
                          <Wand2 className="mr-1 h-4 w-4" />
                        )}
                        一键应用
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// ─── Reprocess ─────────────────────────────────────────────────────────

const STATUS_LABELS: Record<string, { text: string; tone: string }> = {
  pending: { text: "等待中", tone: "text-muted-foreground" },
  in_progress: { text: "处理中", tone: "text-blue-600" },
  completed: { text: "已完成", tone: "text-emerald-600" },
  failed: { text: "失败", tone: "text-red-600" },
};

function ReprocessSection({
  taskIds,
  progressMap,
  loading,
  taskInput,
  onChangeTaskInput,
  onAttachTask,
  onRefresh,
}: {
  taskIds: string[];
  progressMap: Record<string, ReprocessProgress>;
  loading: boolean;
  taskInput: string;
  onChangeTaskInput: (next: string) => void;
  onAttachTask: () => void;
  onRefresh: () => void;
}) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-start justify-between">
        <div>
          <CardTitle className="text-base">重处理进度</CardTitle>
          <CardDescription>
            一键应用建议后会自动出现进度条；也可手动输入 task_id 跟踪
          </CardDescription>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={onRefresh}
          disabled={loading || taskIds.length === 0}
        >
          {loading ? (
            <Loader2 className="mr-1 h-4 w-4 animate-spin" />
          ) : (
            <RefreshCw className="mr-1 h-4 w-4" />
          )}
          刷新
        </Button>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-end gap-2">
          <div className="flex-1 space-y-1.5">
            <Label htmlFor="reprocess-task-id">手动添加 task_id</Label>
            <Input
              id="reprocess-task-id"
              placeholder="reprocess-task-..."
              value={taskInput}
              onChange={(e) => onChangeTaskInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  onAttachTask();
                }
              }}
            />
          </div>
          <Button onClick={onAttachTask} disabled={!taskInput.trim()}>
            添加
          </Button>
        </div>

        {taskIds.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            暂无重处理任务。可以从「优化建议」一键应用以触发重处理。
          </p>
        ) : (
          <div className="space-y-3">
            {taskIds.map((id) => {
              const progress = progressMap[id];
              if (!progress) {
                return (
                  <Card key={id}>
                    <CardContent className="flex items-center gap-2 pt-4">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      <span className="font-mono text-sm">{id}</span>
                      <span className="text-sm text-muted-foreground">
                        正在加载进度...
                      </span>
                    </CardContent>
                  </Card>
                );
              }
              const meta =
                STATUS_LABELS[progress.status] ?? {
                  text: progress.status,
                  tone: "text-foreground",
                };
              return (
                <Card key={id}>
                  <CardContent className="space-y-2 pt-4">
                    <div className="flex flex-wrap items-baseline justify-between gap-2">
                      <span className="font-mono text-sm">{id}</span>
                      <span
                        className={cn(
                          "inline-flex items-center gap-1 text-sm font-medium",
                          meta.tone
                        )}
                      >
                        {progress.status === "completed" ? (
                          <CheckCircle2 className="h-4 w-4" />
                        ) : progress.status === "failed" ? (
                          <AlertCircle className="h-4 w-4" />
                        ) : (
                          <Clock className="h-4 w-4" />
                        )}
                        {meta.text}
                      </span>
                    </div>
                    <div className="space-y-1">
                      <div className="flex items-baseline justify-between text-xs text-muted-foreground">
                        <span>
                          {progress.processed_documents} /{" "}
                          {progress.total_documents} 个文档
                        </span>
                        <span>{progress.progress_percent.toFixed(1)}%</span>
                      </div>
                      <div className="h-2 overflow-hidden rounded bg-muted">
                        <div
                          className={cn(
                            "h-2",
                            progress.status === "failed"
                              ? "bg-red-500"
                              : progress.status === "completed"
                              ? "bg-emerald-500"
                              : "bg-primary/70"
                          )}
                          style={{
                            width: `${Math.min(
                              100,
                              Math.max(progress.progress_percent, 2)
                            )}%`,
                          }}
                        />
                      </div>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      创建于：{formatDateTime(progress.created_at)}
                    </p>
                    {progress.error && (
                      <p className="text-xs text-red-600">
                        错误：{progress.error}
                      </p>
                    )}
                  </CardContent>
                </Card>
              );
            })}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
