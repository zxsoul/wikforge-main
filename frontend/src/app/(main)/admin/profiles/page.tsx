"use client";

import * as React from "react";
import Link from "next/link";
import {
  Plus,
  Search,
  Upload,
  Download,
  ToggleLeft,
  ToggleRight,
  FileText,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Card,
  CardContent,
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

interface Profile {
  id: string;
  name: string;
  description: string | null;
  priority: number;
  enabled: boolean;
  match_rules: Record<string, unknown>;
  version: number;
  created_at: string;
  updated_at: string;
}

interface ProfileListResponse {
  profiles: Profile[];
  total: number;
}

export default function AdminProfilesPage() {
  const { addToast } = useToast();
  const [profiles, setProfiles] = React.useState<Profile[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [searchQuery, setSearchQuery] = React.useState("");
  const [importDialogOpen, setImportDialogOpen] = React.useState(false);
  const [importFile, setImportFile] = React.useState<File | null>(null);

  const fetchProfiles = React.useCallback(async () => {
    try {
      setLoading(true);
      const data = await apiClient.get<ProfileListResponse>(
        "/api/admin/profiles"
      );
      setProfiles(data.profiles || []);
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({
          type: "error",
          message: "加载 Profile 列表失败",
          description: err.message,
        });
      }
    } finally {
      setLoading(false);
    }
  }, [addToast]);

  React.useEffect(() => {
    fetchProfiles();
  }, [fetchProfiles]);

  const filteredProfiles = profiles.filter(
    (p) =>
      p.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (p.description || "").toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleToggle = async (profile: Profile) => {
    try {
      await apiClient.patch(`/api/admin/profiles/${profile.id}/toggle`, {
        enabled: !profile.enabled,
      });
      addToast({
        type: "success",
        message: profile.enabled ? "已禁用" : "已启用",
      });
      fetchProfiles();
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({
          type: "error",
          message: "切换状态失败",
          description: err.message,
        });
      }
    }
  };

  const handleExport = async () => {
    try {
      const data = await apiClient.get<Record<string, unknown>>(
        "/api/admin/profiles/export/all"
      );
      const blob = new Blob([JSON.stringify(data, null, 2)], {
        type: "application/json",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `profiles-export-${new Date().toISOString().slice(0, 10)}.json`;
      a.click();
      URL.revokeObjectURL(url);
      addToast({ type: "success", message: "导出成功" });
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({
          type: "error",
          message: "导出失败",
          description: err.message,
        });
      }
    }
  };

  const handleImport = async () => {
    if (!importFile) return;
    try {
      const text = await importFile.text();
      const json = JSON.parse(text);
      const result = await apiClient.post<{
        imported: number;
        updated: number;
        errors: string[];
      }>("/api/admin/profiles/import", json);
      addToast({
        type: "success",
        message: `导入完成：新增 ${result.imported}，更新 ${result.updated}`,
      });
      if (result.errors.length > 0) {
        addToast({
          type: "error",
          message: `${result.errors.length} 个错误`,
          description: result.errors.join("; "),
        });
      }
      setImportDialogOpen(false);
      setImportFile(null);
      fetchProfiles();
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({
          type: "error",
          message: "导入失败",
          description: err.message,
        });
      } else {
        addToast({ type: "error", message: "JSON 文件格式错误" });
      }
    }
  };

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleString("zh-CN", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Profile 管理</h1>
        <p className="text-muted-foreground mt-1">
          管理文档解析策略配置，支持创建、编辑、启用/禁用、导入导出
        </p>
      </div>

      {/* Toolbar */}
      <div className="flex items-center justify-between gap-4">
        <div className="relative w-72">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="搜索 Profile..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-9"
          />
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" onClick={() => setImportDialogOpen(true)}>
            <Upload className="mr-2 h-4 w-4" />
            导入
          </Button>
          <Button variant="outline" onClick={handleExport}>
            <Download className="mr-2 h-4 w-4" />
            导出
          </Button>
          <Link href="/admin/profiles/new">
            <Button>
              <Plus className="mr-2 h-4 w-4" />
              新建 Profile
            </Button>
          </Link>
        </div>
      </div>

      {/* Profile List */}
      {loading ? (
        <div className="text-center py-12 text-muted-foreground">加载中...</div>
      ) : filteredProfiles.length === 0 ? (
        <div className="text-center py-12 text-muted-foreground">
          {searchQuery ? "未找到匹配的 Profile" : "暂无 Profile，点击上方按钮创建"}
        </div>
      ) : (
        <div className="border rounded-lg overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-muted/50">
              <tr>
                <th className="text-left px-4 py-3 font-medium">名称</th>
                <th className="text-left px-4 py-3 font-medium">优先级</th>
                <th className="text-left px-4 py-3 font-medium">状态</th>
                <th className="text-left px-4 py-3 font-medium">版本</th>
                <th className="text-left px-4 py-3 font-medium">最近更新</th>
                <th className="text-right px-4 py-3 font-medium">操作</th>
              </tr>
            </thead>
            <tbody className="divide-y">
              {filteredProfiles.map((profile) => (
                <tr key={profile.id} className="hover:bg-muted/30">
                  <td className="px-4 py-3">
                    <div>
                      <Link
                        href={`/admin/profiles/${profile.id}`}
                        className="font-medium text-primary hover:underline"
                      >
                        {profile.name}
                      </Link>
                      {profile.description && (
                        <p className="text-xs text-muted-foreground mt-0.5 line-clamp-1">
                          {profile.description}
                        </p>
                      )}
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <span className="inline-flex items-center rounded-full bg-secondary px-2 py-0.5 text-xs font-medium">
                      {profile.priority}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <button
                      onClick={() => handleToggle(profile)}
                      className="inline-flex items-center gap-1.5"
                      title={profile.enabled ? "点击禁用" : "点击启用"}
                    >
                      {profile.enabled ? (
                        <>
                          <ToggleRight className="h-5 w-5 text-green-600" />
                          <span className="text-xs text-green-700">启用</span>
                        </>
                      ) : (
                        <>
                          <ToggleLeft className="h-5 w-5 text-muted-foreground" />
                          <span className="text-xs text-muted-foreground">
                            禁用
                          </span>
                        </>
                      )}
                    </button>
                  </td>
                  <td className="px-4 py-3 text-muted-foreground">
                    v{profile.version}
                  </td>
                  <td className="px-4 py-3 text-muted-foreground text-xs">
                    {formatDate(profile.updated_at)}
                  </td>
                  <td className="px-4 py-3 text-right">
                    <div className="flex items-center justify-end gap-1">
                      <Link href={`/admin/profiles/${profile.id}`}>
                        <Button variant="ghost" size="sm">
                          编辑
                        </Button>
                      </Link>
                      <Link href={`/admin/profiles/${profile.id}/versions`}>
                        <Button variant="ghost" size="sm">
                          历史
                        </Button>
                      </Link>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Import Dialog */}
      <Dialog open={importDialogOpen} onOpenChange={setImportDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>导入 Profile</DialogTitle>
            <DialogDescription>
              上传 JSON 文件导入 Profile 配置，同名 Profile 将被更新
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div className="border-2 border-dashed rounded-lg p-6 text-center">
              <FileText className="mx-auto h-8 w-8 text-muted-foreground mb-2" />
              <input
                type="file"
                accept=".json"
                onChange={(e) => setImportFile(e.target.files?.[0] || null)}
                className="text-sm"
              />
              {importFile && (
                <p className="text-sm text-muted-foreground mt-2">
                  已选择: {importFile.name}
                </p>
              )}
            </div>
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setImportDialogOpen(false);
                setImportFile(null);
              }}
            >
              取消
            </Button>
            <Button onClick={handleImport} disabled={!importFile}>
              导入
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
