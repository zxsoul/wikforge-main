"use client";

import * as React from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { ChevronLeft, ExternalLink, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useToast } from "@/components/ui/toast";
import { apiClient, ApiClientError } from "@/lib/api-client";

interface QualityScoreResponse {
  overall: number;
  components: Record<string, number>;
  issues: string[];
}

interface PreviewResponse {
  review_id: string;
  document_id: string;
  document_title: string;
  original_file_url: string;
  parsed_markdown: string;
  quality_score: QualityScoreResponse;
  status: string;
}

const COMPONENT_LABELS: Record<string, string> = {
  text_retention: "文本保留率",
  heading_detection: "标题识别率",
  table_completeness: "表格完整率",
  numeric_protection: "数值保护率",
  noise_removal: "噪声去除率",
};

const STATUS_LABELS: Record<string, string> = {
  pending: "待审核",
  approved: "已通过",
  corrected: "已修正",
  rejected: "已驳回",
};

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

function statusBadgeClass(status: string): string {
  switch (status) {
    case "approved":
      return "bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-300";
    case "corrected":
      return "bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300";
    case "rejected":
      return "bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300";
    default:
      return "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/40 dark:text-yellow-300";
  }
}

export default function ReviewDetailPage() {
  const params = useParams<{ id: string }>();
  const reviewId = params.id;
  const router = useRouter();
  const { addToast } = useToast();

  const [preview, setPreview] = React.useState<PreviewResponse | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [editing, setEditing] = React.useState(false);
  const [correctedMarkdown, setCorrectedMarkdown] = React.useState("");
  const [reviewerNote, setReviewerNote] = React.useState("");
  const [submitting, setSubmitting] = React.useState(false);
  const [actionDialog, setActionDialog] = React.useState<
    "approve" | "reject" | null
  >(null);
  const [actionNote, setActionNote] = React.useState("");

  const fetchPreview = React.useCallback(async () => {
    try {
      setLoading(true);
      const data = await apiClient.get<PreviewResponse>(
        `/api/admin/reviews/${reviewId}/preview`
      );
      setPreview(data);
      setCorrectedMarkdown(data.parsed_markdown ?? "");
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({
          type: "error",
          message: "加载审核详情失败",
          description: err.message,
        });
      }
    } finally {
      setLoading(false);
    }
  }, [reviewId, addToast]);

  React.useEffect(() => {
    void fetchPreview();
  }, [fetchPreview]);

  const handleSubmitCorrection = async () => {
    if (!correctedMarkdown.trim()) {
      addToast({ type: "error", message: "修正内容不能为空" });
      return;
    }
    setSubmitting(true);
    try {
      const res = await apiClient.post<{ message: string }>(
        `/api/admin/reviews/${reviewId}/correct`,
        {
          corrected_markdown: correctedMarkdown,
          reviewer_note: reviewerNote || null,
        }
      );
      addToast({
        type: "success",
        message: "修正已提交",
        description: res.message,
      });
      setEditing(false);
      setReviewerNote("");
      void fetchPreview();
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({
          type: "error",
          message: "提交修正失败",
          description: err.message,
        });
      }
    } finally {
      setSubmitting(false);
    }
  };

  const handleAction = async (action: "approve" | "reject") => {
    setSubmitting(true);
    try {
      const res = await apiClient.post<{ message: string }>(
        `/api/admin/reviews/${reviewId}/${action}`,
        { reviewer_note: actionNote || null }
      );
      addToast({
        type: "success",
        message: action === "approve" ? "已通过" : "已驳回",
        description: res.message,
      });
      setActionDialog(null);
      setActionNote("");
      void fetchPreview();
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({
          type: "error",
          message: "操作失败",
          description: err.message,
        });
      }
    } finally {
      setSubmitting(false);
    }
  };

  if (loading) {
    return (
      <div className="text-center py-12 text-muted-foreground">加载中...</div>
    );
  }

  if (!preview) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        审核记录不存在
        <div className="mt-4">
          <Button onClick={() => router.push("/admin/reviews")}>
            返回列表
          </Button>
        </div>
      </div>
    );
  }

  const score = preview.quality_score;
  const isPending = preview.status === "pending";
  const canCorrect =
    preview.status === "pending" || preview.status === "rejected";

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Link href="/admin/reviews">
            <Button variant="ghost" size="sm">
              <ChevronLeft className="mr-1 h-4 w-4" />
              返回列表
            </Button>
          </Link>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => void fetchPreview()}
          >
            <RefreshCw className="mr-1 h-4 w-4" />
            刷新
          </Button>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            disabled={!canCorrect || submitting}
            onClick={() => setEditing((e) => !e)}
          >
            {editing ? "取消修正" : "修正"}
          </Button>
          <Button
            variant="outline"
            disabled={!isPending || submitting}
            onClick={() => void fetchPreview()}
          >
            重新解析
          </Button>
          <Button
            variant="default"
            disabled={!isPending || submitting}
            onClick={() => setActionDialog("approve")}
          >
            审核通过
          </Button>
          <Button
            variant="destructive"
            disabled={!isPending || submitting}
            onClick={() => setActionDialog("reject")}
          >
            审核拒绝
          </Button>
        </div>
      </div>

      <div>
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-bold">{preview.document_title}</h1>
          <span
            className={
              "inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium " +
              statusBadgeClass(preview.status)
            }
          >
            {STATUS_LABELS[preview.status] ?? preview.status}
          </span>
        </div>
        <p className="text-muted-foreground mt-1 text-sm">
          审核 ID: {preview.review_id} · 文档 ID: {preview.document_id}
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">质量分</CardTitle>
          <CardDescription>
            综合分 = 30% 文本 + 25% 标题 + 20% 表格 + 15% 数值 + 10% 噪声
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 grid-cols-2 md:grid-cols-3 lg:grid-cols-6">
            <div className="space-y-1">
              <div className="text-xs text-muted-foreground">综合分</div>
              <div
                className={"text-2xl font-bold " + scoreColor(score.overall)}
              >
                {formatScore(score.overall)}
              </div>
            </div>
            {Object.entries(COMPONENT_LABELS).map(([key, label]) => {
              const v = score.components?.[key];
              return (
                <div key={key} className="space-y-1">
                  <div className="text-xs text-muted-foreground">{label}</div>
                  <div
                    className={
                      "text-2xl font-bold " +
                      scoreColor(typeof v === "number" ? v : undefined)
                    }
                  >
                    {formatScore(typeof v === "number" ? v : undefined)}
                  </div>
                </div>
              );
            })}
          </div>

          {score.issues && score.issues.length > 0 && (
            <div className="mt-6">
              <div className="text-sm font-medium mb-2">检测问题</div>
              <ul className="space-y-1 text-sm text-muted-foreground list-disc list-inside">
                {score.issues.map((issue, idx) => (
                  <li key={idx}>{issue}</li>
                ))}
              </ul>
            </div>
          )}
        </CardContent>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle className="text-base">原文件预览</CardTitle>
              <CardDescription>
                通过预签名 URL 加载,在新窗口可下载完整文件
              </CardDescription>
            </div>
            <a
              href={preview.original_file_url}
              target="_blank"
              rel="noreferrer"
              className="text-sm text-primary hover:underline inline-flex items-center gap-1"
            >
              在新窗口打开
              <ExternalLink className="h-3 w-3" />
            </a>
          </CardHeader>
          <CardContent>
            <div className="border rounded-md overflow-hidden bg-muted/20">
              <iframe
                src={preview.original_file_url}
                title="原文件预览"
                className="w-full h-[600px]"
              />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">
              {editing ? "修正解析结果" : "解析后 Markdown"}
            </CardTitle>
            <CardDescription>
              {editing
                ? "调整标题层级、修正表格内容、补充缺失文本,提交后将重新触发分块和向量化"
                : "管线产出的清洗后 Markdown,点击「修正」可编辑后提交"}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {editing ? (
              <>
                <Textarea
                  rows={20}
                  value={correctedMarkdown}
                  onChange={(e) => setCorrectedMarkdown(e.target.value)}
                  placeholder="在此编辑修正后的 Markdown..."
                  className="font-mono text-xs"
                />
                <div className="space-y-2">
                  <Label htmlFor="reviewer-note">审核备注（可选）</Label>
                  <Textarea
                    id="reviewer-note"
                    rows={2}
                    value={reviewerNote}
                    onChange={(e) => setReviewerNote(e.target.value)}
                    placeholder="说明修正的原因或调整范围,便于后续 Profile 优化分析"
                    maxLength={2000}
                  />
                </div>
                <div className="flex justify-end gap-2">
                  <Button
                    variant="outline"
                    onClick={() => {
                      setEditing(false);
                      setCorrectedMarkdown(preview.parsed_markdown ?? "");
                      setReviewerNote("");
                    }}
                    disabled={submitting}
                  >
                    取消
                  </Button>
                  <Button
                    onClick={handleSubmitCorrection}
                    disabled={submitting}
                  >
                    {submitting ? "提交中..." : "提交修正"}
                  </Button>
                </div>
              </>
            ) : (
              <pre className="border rounded-md bg-muted/20 p-3 text-xs whitespace-pre-wrap break-words max-h-[600px] overflow-auto">
                {preview.parsed_markdown || "[暂无解析结果]"}
              </pre>
            )}
          </CardContent>
        </Card>
      </div>

      <Dialog
        open={actionDialog !== null}
        onOpenChange={(open) => {
          if (!open) {
            setActionDialog(null);
            setActionNote("");
          }
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              {actionDialog === "approve" ? "审核通过" : "审核驳回"}
            </DialogTitle>
            <DialogDescription>
              {actionDialog === "approve"
                ? "通过后该文档将进入下游索引流程"
                : "驳回后可重新修正,或交由文档作者重新上传"}
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-2">
            <Label htmlFor="action-note">备注（可选）</Label>
            <Textarea
              id="action-note"
              rows={3}
              value={actionNote}
              onChange={(e) => setActionNote(e.target.value)}
              placeholder="可填写决策原因"
            />
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setActionDialog(null);
                setActionNote("");
              }}
              disabled={submitting}
            >
              取消
            </Button>
            <Button
              variant={actionDialog === "reject" ? "destructive" : "default"}
              onClick={() => actionDialog && handleAction(actionDialog)}
              disabled={submitting}
            >
              {submitting
                ? "处理中..."
                : actionDialog === "approve"
                  ? "确认通过"
                  : "确认驳回"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
