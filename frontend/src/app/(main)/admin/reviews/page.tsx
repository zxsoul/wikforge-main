"use client";

import * as React from "react";
import Link from "next/link";
import { Filter } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/components/ui/toast";
import { apiClient, ApiClientError } from "@/lib/api-client";

interface QualityScore {
  overall?: number;
  components?: Record<string, number>;
  issues?: string[];
}

interface ReviewListItem {
  review_id: string;
  document_id: string;
  document_title: string;
  space_id: string;
  profile_id: string | null;
  profile_name: string | null;
  quality_score: QualityScore;
  status: string;
  created_at: string;
  reviewed_at: string | null;
}

interface ReviewListResponse {
  items: ReviewListItem[];
  page: number;
  page_size: number;
  total: number;
}

type StatusFilter = "pending" | "approved" | "corrected" | "rejected";
type SortBy = "quality_score_asc" | "created_at_desc";

const STATUS_OPTIONS: { value: StatusFilter; label: string }[] = [
  { value: "pending", label: "待审核" },
  { value: "corrected", label: "已修正" },
  { value: "approved", label: "已通过" },
  { value: "rejected", label: "已驳回" },
];

const COMPONENT_LABELS: Record<string, string> = {
  text_retention: "文本保留",
  heading_detection: "标题识别",
  table_completeness: "表格完整",
  numeric_protection: "数值保护",
  noise_removal: "噪声去除",
};

function formatDate(dateStr: string): string {
  return new Date(dateStr).toLocaleString("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatScore(score: number | undefined): string {
  if (score == null || Number.isNaN(score)) return "-";
  return (score * 100).toFixed(0);
}

function scoreColor(score: number | undefined): string {
  if (score == null) return "text-muted-foreground";
  if (score >= 0.8) return "text-green-600 dark:text-green-400";
  if (score >= 0.6) return "text-yellow-600 dark:text-yellow-400";
  return "text-red-600 dark:text-red-400";
}

export default function AdminReviewsPage() {
  const { addToast } = useToast();
  const [items, setItems] = React.useState<ReviewListItem[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [status, setStatus] = React.useState<StatusFilter>("pending");
  const [sortBy, setSortBy] = React.useState<SortBy>("quality_score_asc");
  const [page, setPage] = React.useState(1);
  const [pageSize] = React.useState(20);
  const [total, setTotal] = React.useState(0);

  const fetchReviews = React.useCallback(async () => {
    try {
      setLoading(true);
      const params = new URLSearchParams({
        status,
        sort_by: sortBy,
        page: String(page),
        page_size: String(pageSize),
      });
      const data = await apiClient.get<ReviewListResponse>(
        `/api/admin/reviews?${params.toString()}`
      );
      setItems(data.items || []);
      setTotal(data.total || 0);
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({
          type: "error",
          message: "加载审核队列失败",
          description: err.message,
        });
      }
    } finally {
      setLoading(false);
    }
  }, [status, sortBy, page, pageSize, addToast]);

  React.useEffect(() => {
    void fetchReviews();
  }, [fetchReviews]);

  const totalPages = Math.max(1, Math.ceil(total / pageSize));

  const componentKeys = React.useMemo(() => {
    const keys = new Set<string>();
    for (const item of items) {
      const comps = item.quality_score?.components ?? {};
      Object.keys(comps).forEach((k) => keys.add(k));
    }
    // 优先显示已知字段，再追加未知字段，保持顺序稳定
    const known = Object.keys(COMPONENT_LABELS).filter((k) => keys.has(k));
    const extra = Array.from(keys).filter((k) => !COMPONENT_LABELS[k]);
    return [...known, ...extra];
  }, [items]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">审核队列</h1>
        <p className="text-muted-foreground mt-1">
          管理质量分较低的文档,按各维度分数审核、修正或驳回
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-1.5 text-sm text-muted-foreground">
          <Filter className="h-4 w-4" />
          <span>状态</span>
        </div>
        <div className="flex items-center gap-1">
          {STATUS_OPTIONS.map((opt) => {
            const active = status === opt.value;
            return (
              <button
                key={opt.value}
                type="button"
                onClick={() => {
                  setStatus(opt.value);
                  setPage(1);
                }}
                className={
                  "px-3 py-1.5 text-sm rounded-md border transition-colors " +
                  (active
                    ? "bg-primary text-primary-foreground border-primary"
                    : "bg-background hover:bg-accent border-input")
                }
              >
                {opt.label}
              </button>
            );
          })}
        </div>

        <div className="ml-auto flex items-center gap-2 text-sm">
          <span className="text-muted-foreground">排序</span>
          <select
            value={sortBy}
            onChange={(e) => {
              setSortBy(e.target.value as SortBy);
              setPage(1);
            }}
            className="h-9 rounded-md border border-input bg-background px-2 text-sm"
          >
            <option value="quality_score_asc">质量分（升序）</option>
            <option value="created_at_desc">创建时间（降序）</option>
          </select>
        </div>
      </div>

      {loading ? (
        <div className="text-center py-12 text-muted-foreground">加载中...</div>
      ) : items.length === 0 ? (
        <div className="text-center py-12 text-muted-foreground">
          当前筛选下暂无审核记录
        </div>
      ) : (
        <div className="border rounded-lg overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-muted/50">
              <tr>
                <th className="text-left px-4 py-3 font-medium">文档</th>
                <th className="text-left px-4 py-3 font-medium">Profile</th>
                <th className="text-left px-4 py-3 font-medium">综合分</th>
                {componentKeys.map((key) => (
                  <th
                    key={key}
                    className="text-left px-3 py-3 font-medium whitespace-nowrap"
                  >
                    {COMPONENT_LABELS[key] ?? key}
                  </th>
                ))}
                <th className="text-left px-4 py-3 font-medium">问题</th>
                <th className="text-left px-4 py-3 font-medium whitespace-nowrap">
                  创建时间
                </th>
                <th className="text-right px-4 py-3 font-medium">操作</th>
              </tr>
            </thead>
            <tbody className="divide-y">
              {items.map((item) => {
                const overall = item.quality_score?.overall;
                const components = item.quality_score?.components ?? {};
                const issues = item.quality_score?.issues ?? [];
                return (
                  <tr key={item.review_id} className="hover:bg-muted/30">
                    <td className="px-4 py-3">
                      <Link
                        href={`/admin/reviews/${item.review_id}`}
                        className="font-medium text-primary hover:underline"
                      >
                        {item.document_title || "未命名文档"}
                      </Link>
                    </td>
                    <td className="px-4 py-3 text-muted-foreground">
                      {item.profile_name ?? "-"}
                    </td>
                    <td className={"px-4 py-3 font-semibold " + scoreColor(overall)}>
                      {formatScore(overall)}
                    </td>
                    {componentKeys.map((key) => {
                      const v = components[key];
                      return (
                        <td
                          key={key}
                          className={
                            "px-3 py-3 whitespace-nowrap " +
                            scoreColor(typeof v === "number" ? v : undefined)
                          }
                        >
                          {formatScore(typeof v === "number" ? v : undefined)}
                        </td>
                      );
                    })}
                    <td className="px-4 py-3 text-xs text-muted-foreground">
                      {issues.length === 0 ? (
                        <span>-</span>
                      ) : (
                        <span title={issues.join("\n")}>
                          {issues.length} 项
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-xs text-muted-foreground whitespace-nowrap">
                      {formatDate(item.created_at)}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <Link href={`/admin/reviews/${item.review_id}`}>
                        <Button variant="ghost" size="sm">
                          查看
                        </Button>
                      </Link>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {!loading && total > 0 && (
        <div className="flex items-center justify-between text-sm text-muted-foreground">
          <div>
            共 {total} 条 · 第 {page} / {totalPages} 页
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              disabled={page <= 1}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
            >
              上一页
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={page >= totalPages}
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            >
              下一页
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
