"use client";

import * as React from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { ChevronLeft, History, Trash2, Upload } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
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

interface PreviewResponse {
  blocks: Array<{
    type: string;
    text: string;
    page_number: number | null;
    style: Record<string, unknown> | null;
  }>;
  features: Record<string, unknown>;
  matched_profile: string | null;
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
    name: p.name,
    description: p.description ?? "",
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

export default function EditProfilePage() {
  const params = useParams<{ id: string }>();
  const profileId = params.id;
  const router = useRouter();
  const { addToast } = useToast();
  const [value, setValue] = React.useState<ProfileFormValue>({
    ...defaultProfileValue,
  });
  const [profile, setProfile] = React.useState<ProfileResponse | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [submitting, setSubmitting] = React.useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = React.useState(false);
  const [previewFile, setPreviewFile] = React.useState<File | null>(null);
  const [previewing, setPreviewing] = React.useState(false);
  const [preview, setPreview] = React.useState<PreviewResponse | null>(null);

  const fetchProfile = React.useCallback(async () => {
    try {
      setLoading(true);
      const data = await apiClient.get<ProfileResponse>(
        `/api/admin/profiles/${profileId}`
      );
      setProfile(data);
      setValue(profileToFormValue(data));
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({
          type: "error",
          message: "加载 Profile 失败",
          description: err.message,
        });
      }
    } finally {
      setLoading(false);
    }
  }, [profileId, addToast]);

  React.useEffect(() => {
    void fetchProfile();
  }, [fetchProfile]);

  const handleSubmit = async (changeNote?: string) => {
    if (!value.name.trim()) {
      addToast({ type: "error", message: "请填写 Profile 名称" });
      return;
    }
    setSubmitting(true);
    try {
      const payload = {
        name: value.name.trim(),
        description: value.description || null,
        priority: value.priority,
        enabled: value.enabled,
        match_rules: value.match_rules,
        heading_rules: value.heading_rules,
        boilerplate: value.boilerplate,
        tables: value.tables,
        chunking: value.chunking,
        domain_dictionary_id: value.domain_dictionary_id,
        change_note: changeNote ?? null,
      };
      await apiClient.put<ProfileResponse>(
        `/api/admin/profiles/${profileId}`,
        payload
      );
      addToast({ type: "success", message: "已保存修改" });
      void fetchProfile();
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({
          type: "error",
          message: "保存失败",
          description: err.message,
        });
      }
    } finally {
      setSubmitting(false);
    }
  };

  const handleDelete = async () => {
    try {
      await apiClient.delete(`/api/admin/profiles/${profileId}`);
      addToast({ type: "success", message: "已删除" });
      router.push("/admin/profiles");
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({
          type: "error",
          message: "删除失败",
          description: err.message,
        });
      }
    }
  };

  const handlePreview = async () => {
    if (!previewFile) return;
    setPreviewing(true);
    try {
      const formData = new FormData();
      formData.append("file", previewFile);

      const baseUrl =
        process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const token = apiClient.getAccessToken();
      const response = await fetch(
        `${baseUrl}/api/admin/profiles/${profileId}/preview`,
        {
          method: "POST",
          headers: token ? { Authorization: `Bearer ${token}` } : undefined,
          body: formData,
        }
      );
      if (!response.ok) {
        throw new Error(`预览失败: ${response.statusText}`);
      }
      const data = (await response.json()) as PreviewResponse;
      setPreview(data);
      addToast({ type: "success", message: "预览生成成功" });
    } catch (err) {
      addToast({
        type: "error",
        message: "预览失败",
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setPreviewing(false);
    }
  };

  if (loading) {
    return (
      <div className="text-center py-12 text-muted-foreground">加载中...</div>
    );
  }

  if (!profile) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        Profile 不存在
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-4xl">
      <div className="flex items-center justify-between gap-2">
        <Link href="/admin/profiles">
          <Button variant="ghost" size="sm">
            <ChevronLeft className="mr-1 h-4 w-4" />
            返回列表
          </Button>
        </Link>
        <div className="flex items-center gap-2">
          <Link href={`/admin/profiles/${profileId}/versions`}>
            <Button variant="outline" size="sm">
              <History className="mr-1 h-4 w-4" />
              版本历史
            </Button>
          </Link>
          <Button
            variant="destructive"
            size="sm"
            onClick={() => setDeleteDialogOpen(true)}
            disabled={profile.name === "generic-text"}
          >
            <Trash2 className="mr-1 h-4 w-4" />
            删除
          </Button>
        </div>
      </div>
      <div>
        <h1 className="text-2xl font-bold">{profile.name}</h1>
        <p className="text-muted-foreground mt-1 text-sm">
          当前版本 v{profile.version} · 最近更新{" "}
          {new Date(profile.updated_at).toLocaleString("zh-CN")}
        </p>
      </div>

      <ProfileForm
        value={value}
        onChange={setValue}
        onSubmit={handleSubmit}
        submitLabel="保存修改"
        showChangeNote
        submitting={submitting}
      />

      {/* 预览解析结果(任务 23.3) */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">预览解析结果</CardTitle>
          <CardDescription>
            上传样本文档,使用当前 Profile 配置解析并查看分块结果
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center gap-2">
            <input
              type="file"
              accept=".pdf,.docx,.pptx,.txt,.md,.html"
              onChange={(e) => setPreviewFile(e.target.files?.[0] ?? null)}
              className="text-sm"
            />
            <Button
              type="button"
              variant="outline"
              onClick={handlePreview}
              disabled={!previewFile || previewing}
            >
              <Upload className="mr-1 h-4 w-4" />
              {previewing ? "解析中..." : "运行预览"}
            </Button>
          </div>
          {preview && (
            <div className="space-y-3">
              <div className="text-sm">
                <span className="font-medium">特征:</span>{" "}
                <code className="text-xs bg-muted px-1.5 py-0.5 rounded">
                  {JSON.stringify(preview.features, null, 0)}
                </code>
              </div>
              <div>
                <p className="text-sm font-medium mb-2">
                  解析块({preview.blocks.length} 个)
                </p>
                <div className="border rounded-md max-h-96 overflow-y-auto divide-y text-sm">
                  {preview.blocks.slice(0, 100).map((block, idx) => (
                    <div key={idx} className="px-3 py-2">
                      <div className="text-xs text-muted-foreground mb-0.5">
                        #{idx + 1} · {block.type}
                        {block.page_number != null
                          ? ` · 第 ${block.page_number} 页`
                          : ""}
                      </div>
                      <div className="line-clamp-3 whitespace-pre-wrap">
                        {block.text}
                      </div>
                    </div>
                  ))}
                  {preview.blocks.length > 100 && (
                    <div className="px-3 py-2 text-xs text-muted-foreground">
                      已截断,共 {preview.blocks.length} 块,仅显示前 100 块
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>删除 Profile</DialogTitle>
            <DialogDescription>
              确认删除 Profile <strong>{profile.name}</strong>?该操作不可撤销。
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setDeleteDialogOpen(false)}
            >
              取消
            </Button>
            <Button variant="destructive" onClick={handleDelete}>
              确认删除
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
