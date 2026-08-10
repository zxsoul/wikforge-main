"use client";

import * as React from "react";
import Link from "next/link";
import {
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Check,
  X,
  Pencil,
} from "lucide-react";
import { Button } from "@/components/ui/button";
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
import {
  ProfileForm,
  defaultProfileValue,
  type ProfileFormValue,
} from "@/components/admin/profile-form";

interface ProfileResponse {
  id: string;
  name: string;
  description: string | null;
  priority: number;
  enabled: boolean;
  match_rules: Record<string, unknown>;
  heading_rules: Array<Record<string, unknown>>;
  boilerplate: Record<string, unknown>;
  tables: Record<string, unknown>;
  chunking: Record<string, unknown>;
  domain_dictionary_id: string | null;
  version: number;
  created_at: string;
  updated_at: string;
}

interface CandidateEvidence {
  page_count?: number;
  heading_count?: number;
  table_count?: number;
  boilerplate_candidates?: number;
  avg_block_chars?: number;
}

interface CandidateMetadata {
  status: string;
  source: string;
  evidence: CandidateEvidence;
}

interface CandidateProfileItem {
  profile: ProfileResponse;
  metadata: CandidateMetadata;
}

interface CandidateListResponse {
  candidates: CandidateProfileItem[];
  total: number;
}

const CANDIDATE_DESCRIPTION_PREFIX = "[CANDIDATE] ";

function stripCandidatePrefix(description: string | null): string {
  if (!description) return "";
  if (description.startsWith(CANDIDATE_DESCRIPTION_PREFIX)) {
    return description.slice(CANDIDATE_DESCRIPTION_PREFIX.length);
  }
  return description;
}

function profileToFormValue(p: ProfileResponse): ProfileFormValue {
  const matchRules = (p.match_rules ?? {}) as {
    filename_regex?: string[];
    content_regex?: string[];
    min_content_match_count?: number;
  };
  const boilerplate = (p.boilerplate ?? {}) as {
    detection_mode?: string;
    statistical_threshold?: number;
    manual_patterns?: string[];
  };
  const tables = (p.tables ?? {}) as {
    cross_page_merge?: boolean;
    row_level_chunking?: boolean;
    collapse_merged_cells?: string;
  };
  const chunking = (p.chunking ?? {}) as {
    min_tokens?: number;
    max_tokens?: number;
    overlap_tokens?: number;
    respect_heading_level?: number;
    protect_patterns?: string[];
  };

  return {
    ...defaultProfileValue,
    name: p.name,
    description: stripCandidatePrefix(p.description),
    priority: p.priority,
    enabled: p.enabled,
    match_rules: {
      filename_regex: matchRules.filename_regex ?? [],
      content_regex: matchRules.content_regex ?? [],
      min_content_match_count: matchRules.min_content_match_count ?? 1,
    },
    heading_rules: (p.heading_rules ?? []).map((r) => ({
      pattern: String((r as { pattern?: unknown }).pattern ?? ""),
      level: Number((r as { level?: unknown }).level ?? 1),
      strip_pattern: Boolean((r as { strip_pattern?: unknown }).strip_pattern),
    })),
    boilerplate: {
      detection_mode: boilerplate.detection_mode ?? "statistical",
      statistical_threshold: boilerplate.statistical_threshold ?? 0.5,
      manual_patterns: boilerplate.manual_patterns ?? [],
    },
    tables: {
      cross_page_merge: tables.cross_page_merge ?? true,
      row_level_chunking: tables.row_level_chunking ?? false,
      collapse_merged_cells: tables.collapse_merged_cells ?? "describe",
    },
    chunking: {
      min_tokens: chunking.min_tokens ?? 256,
      max_tokens: chunking.max_tokens ?? 800,
      overlap_tokens: chunking.overlap_tokens ?? 80,
      respect_heading_level: chunking.respect_heading_level ?? 1,
      protect_patterns: chunking.protect_patterns ?? [],
    },
    domain_dictionary_id: p.domain_dictionary_id,
  };
}

function formatDate(dateStr: string): string {
  return new Date(dateStr).toLocaleString("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

const PAGE_SIZE = 20;

export default function CandidateProfilesPage() {
  const { addToast } = useToast();
  const [items, setItems] = React.useState<CandidateProfileItem[]>([]);
  const [total, setTotal] = React.useState(0);
  const [page, setPage] = React.useState(1);
  const [loading, setLoading] = React.useState(true);

  // 当前正在编辑的候选 ID 与表单值
  const [editingId, setEditingId] = React.useState<string | null>(null);
  const [formValue, setFormValue] = React.useState<ProfileFormValue>({
    ...defaultProfileValue,
  });
  const [submitting, setSubmitting] = React.useState<string | null>(null);

  // 拒绝二次确认
  const [rejectingId, setRejectingId] = React.useState<string | null>(null);

  const fetchCandidates = React.useCallback(async () => {
    try {
      setLoading(true);
      const skip = (page - 1) * PAGE_SIZE;
      const params = new URLSearchParams({
        skip: String(skip),
        limit: String(PAGE_SIZE),
      });
      const data = await apiClient.get<CandidateListResponse>(
        `/api/admin/profiles/candidates?${params.toString()}`
      );
      setItems(data.candidates ?? []);
      setTotal(data.total ?? 0);
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({
          type: "error",
          message: "加载候选 Profile 失败",
          description: err.message,
        });
      }
    } finally {
      setLoading(false);
    }
  }, [page, addToast]);

  React.useEffect(() => {
    void fetchCandidates();
  }, [fetchCandidates]);

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  const startEdit = (item: CandidateProfileItem) => {
    setEditingId(item.profile.id);
    setFormValue(profileToFormValue(item.profile));
  };

  const cancelEdit = () => {
    setEditingId(null);
    setFormValue({ ...defaultProfileValue });
  };

  /**
   * 保存编辑：直接 PUT 更新候选 Profile 的内容，但保留候选状态。
   * 后端会保留 description 中的 [CANDIDATE] 前缀（除非 enabled 翻转），
   * 这里我们手动加回前缀，避免管理员误把 enabled 改成 true。
   */
  const handleSave = async (changeNote?: string) => {
    if (!editingId) return;
    if (!formValue.name.trim()) {
      addToast({ type: "error", message: "请填写 Profile 名称" });
      return;
    }
    setSubmitting(editingId);
    try {
      const trimmedDesc = formValue.description.trim();
      const description = trimmedDesc
        ? `${CANDIDATE_DESCRIPTION_PREFIX}${trimmedDesc}`
        : CANDIDATE_DESCRIPTION_PREFIX.trim();
      const payload = {
        name: formValue.name.trim(),
        description,
        priority: formValue.priority,
        enabled: false,
        match_rules: formValue.match_rules,
        heading_rules: formValue.heading_rules,
        boilerplate: formValue.boilerplate,
        tables: formValue.tables,
        chunking: formValue.chunking,
        domain_dictionary_id: formValue.domain_dictionary_id,
        change_note: changeNote ?? null,
      };
      await apiClient.put<ProfileResponse>(
        `/api/admin/profiles/${editingId}`,
        payload
      );
      addToast({ type: "success", message: "已保存修改" });
      setEditingId(null);
      await fetchCandidates();
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({
          type: "error",
          message: "保存失败",
          description: err.message,
        });
      }
    } finally {
      setSubmitting(null);
    }
  };

  const handleApprove = async (item: CandidateProfileItem) => {
    setSubmitting(item.profile.id);
    try {
      await apiClient.post(
        `/api/admin/profiles/candidates/${item.profile.id}/approve`,
        {
          change_note: "审核通过候选 Profile",
          enabled: true,
        }
      );
      addToast({ type: "success", message: "已批准候选 Profile" });
      if (editingId === item.profile.id) {
        setEditingId(null);
      }
      await fetchCandidates();
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({
          type: "error",
          message: "批准失败",
          description: err.message,
        });
      }
    } finally {
      setSubmitting(null);
    }
  };

  const handleReject = async (id: string) => {
    setSubmitting(id);
    try {
      await apiClient.post(`/api/admin/profiles/candidates/${id}/reject`, {});
      addToast({ type: "success", message: "已拒绝候选 Profile" });
      if (editingId === id) {
        setEditingId(null);
      }
      setRejectingId(null);
      await fetchCandidates();
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({
          type: "error",
          message: "拒绝失败",
          description: err.message,
        });
      }
    } finally {
      setSubmitting(null);
    }
  };

  return (
    <div className="space-y-6 max-w-5xl">
      <div className="flex items-center gap-2">
        <Link href="/admin/profiles">
          <Button variant="ghost" size="sm">
            <ChevronLeft className="mr-1 h-4 w-4" />
            返回 Profile 列表
          </Button>
        </Link>
      </div>
      <div>
        <h1 className="text-2xl font-bold">候选 Profile 审核</h1>
        <p className="text-muted-foreground mt-1 text-sm">
          这些 Profile 由通用解析器（Universal Parser）针对未匹配文档自动推断生成，
          需要管理员确认后才会启用。可以编辑后再批准，或直接拒绝。
        </p>
      </div>

      {loading ? (
        <div className="text-center py-12 text-muted-foreground">加载中...</div>
      ) : items.length === 0 ? (
        <div className="text-center py-12 text-muted-foreground">
          暂无待审核的候选 Profile
        </div>
      ) : (
        <div className="space-y-4">
          {items.map((item) => {
            const isEditing = editingId === item.profile.id;
            const evidence = item.metadata.evidence ?? {};
            const isBusy = submitting === item.profile.id;
            return (
              <Card key={item.profile.id}>
                <CardHeader>
                  <div className="flex items-start justify-between gap-4">
                    <div className="space-y-1.5">
                      <CardTitle className="text-base flex items-center gap-2">
                        <span className="inline-flex items-center rounded-full bg-amber-100 dark:bg-amber-900/40 text-amber-800 dark:text-amber-300 px-2 py-0.5 text-xs font-medium">
                          候选
                        </span>
                        <span>{item.profile.name}</span>
                      </CardTitle>
                      <CardDescription>
                        {stripCandidatePrefix(item.profile.description) ||
                          "（无描述）"}
                      </CardDescription>
                      <div className="flex flex-wrap gap-2 text-xs text-muted-foreground pt-1">
                        <span>来源：{item.metadata.source || "未知"}</span>
                        <span>·</span>
                        <span>状态：{item.metadata.status}</span>
                        <span>·</span>
                        <span>
                          创建时间：{formatDate(item.profile.created_at)}
                        </span>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() =>
                          isEditing ? cancelEdit() : startEdit(item)
                        }
                        disabled={isBusy}
                      >
                        <Pencil className="mr-1 h-4 w-4" />
                        {isEditing ? "收起编辑" : "编辑"}
                      </Button>
                      <Button
                        size="sm"
                        onClick={() => handleApprove(item)}
                        disabled={isBusy}
                      >
                        <Check className="mr-1 h-4 w-4" />
                        批准
                      </Button>
                      <Button
                        variant="destructive"
                        size="sm"
                        onClick={() => setRejectingId(item.profile.id)}
                        disabled={isBusy}
                      >
                        <X className="mr-1 h-4 w-4" />
                        拒绝
                      </Button>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  {/* 证据指标 */}
                  <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
                    <EvidenceCell
                      label="页数"
                      value={evidence.page_count}
                    />
                    <EvidenceCell
                      label="标题数"
                      value={evidence.heading_count}
                    />
                    <EvidenceCell
                      label="表格数"
                      value={evidence.table_count}
                    />
                    <EvidenceCell
                      label="噪声候选数"
                      value={evidence.boilerplate_candidates}
                    />
                    <EvidenceCell
                      label="平均块字符"
                      value={evidence.avg_block_chars}
                      precision={1}
                    />
                  </div>

                  {/* 关键参数预览 */}
                  <div className="border rounded-md bg-muted/30 p-3 text-sm space-y-2">
                    <div className="flex items-center gap-2 font-medium">
                      <ChevronDown className="h-3.5 w-3.5" />
                      LLM 推断的初始 Profile JSON
                    </div>
                    <pre className="text-xs overflow-x-auto whitespace-pre-wrap max-h-72 leading-relaxed">
                      {JSON.stringify(
                        {
                          name: item.profile.name,
                          priority: item.profile.priority,
                          match_rules: item.profile.match_rules,
                          heading_rules: item.profile.heading_rules,
                          boilerplate: item.profile.boilerplate,
                          tables: item.profile.tables,
                          chunking: item.profile.chunking,
                          domain_dictionary_id:
                            item.profile.domain_dictionary_id,
                        },
                        null,
                        2
                      )}
                    </pre>
                  </div>

                  {/* 内嵌编辑表单 */}
                  {isEditing && (
                    <div className="border-t pt-4">
                      <ProfileForm
                        value={formValue}
                        onChange={setFormValue}
                        onSubmit={handleSave}
                        submitLabel="保存修改"
                        showChangeNote
                        submitting={isBusy}
                      />
                    </div>
                  )}
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}

      {/* 分页 */}
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
              <ChevronLeft className="mr-1 h-4 w-4" />
              上一页
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={page >= totalPages}
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            >
              下一页
              <ChevronRight className="ml-1 h-4 w-4" />
            </Button>
          </div>
        </div>
      )}

      {/* 拒绝二次确认 */}
      <Dialog
        open={rejectingId !== null}
        onOpenChange={(open) => !open && setRejectingId(null)}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>拒绝候选 Profile</DialogTitle>
            <DialogDescription>
              确认拒绝该候选 Profile？拒绝后会从数据库中删除，无法恢复。
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setRejectingId(null)}>
              取消
            </Button>
            <Button
              variant="destructive"
              onClick={() => rejectingId && void handleReject(rejectingId)}
              disabled={rejectingId !== null && submitting === rejectingId}
            >
              确认拒绝
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

interface EvidenceCellProps {
  label: string;
  value: number | undefined;
  precision?: number;
}

function EvidenceCell({ label, value, precision = 0 }: EvidenceCellProps) {
  let display: string;
  if (value == null || Number.isNaN(value)) {
    display = "-";
  } else if (precision > 0) {
    display = value.toFixed(precision);
  } else {
    display = String(value);
  }
  return (
    <div className="rounded-md border bg-background px-3 py-2">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="text-base font-semibold">{display}</div>
    </div>
  );
}
