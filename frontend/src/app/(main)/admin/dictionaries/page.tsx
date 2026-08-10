"use client";

import * as React from "react";
import Link from "next/link";
import {
  Plus,
  Search,
  ToggleLeft,
  ToggleRight,
  Trash2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
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

interface Dictionary {
  id: string;
  name: string;
  description: string | null;
  terms: Array<Record<string, unknown>>;
  synonyms: Array<Record<string, unknown>>;
  stop_words: string[];
  enabled: boolean;
  created_at: string;
  updated_at: string;
}

interface DictionaryListResponse {
  dictionaries: Dictionary[];
  total: number;
}

export default function AdminDictionariesPage() {
  const { addToast } = useToast();
  const [dictionaries, setDictionaries] = React.useState<Dictionary[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [searchQuery, setSearchQuery] = React.useState("");
  const [createOpen, setCreateOpen] = React.useState(false);
  const [createName, setCreateName] = React.useState("");
  const [createDescription, setCreateDescription] = React.useState("");
  const [creating, setCreating] = React.useState(false);
  const [deleteTarget, setDeleteTarget] = React.useState<Dictionary | null>(
    null
  );

  const fetchDictionaries = React.useCallback(async () => {
    try {
      setLoading(true);
      const data = await apiClient.get<DictionaryListResponse>(
        "/api/admin/dictionaries"
      );
      setDictionaries(data.dictionaries || []);
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({
          type: "error",
          message: "加载词典列表失败",
          description: err.message,
        });
      }
    } finally {
      setLoading(false);
    }
  }, [addToast]);

  React.useEffect(() => {
    void fetchDictionaries();
  }, [fetchDictionaries]);

  const filtered = dictionaries.filter(
    (d) =>
      d.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (d.description ?? "").toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleToggle = async (d: Dictionary) => {
    try {
      await apiClient.patch(`/api/admin/dictionaries/${d.id}/toggle`, {
        enabled: !d.enabled,
      });
      addToast({
        type: "success",
        message: d.enabled ? "已禁用词典" : "已启用词典",
      });
      void fetchDictionaries();
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

  const handleCreate = async () => {
    if (!createName.trim()) {
      addToast({ type: "error", message: "请填写词典名称" });
      return;
    }
    setCreating(true);
    try {
      await apiClient.post("/api/admin/dictionaries", {
        name: createName.trim(),
        description: createDescription || null,
        terms: [],
        synonyms: [],
        stop_words: [],
        enabled: true,
      });
      addToast({ type: "success", message: "已创建词典" });
      setCreateOpen(false);
      setCreateName("");
      setCreateDescription("");
      void fetchDictionaries();
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({
          type: "error",
          message: "创建失败",
          description: err.message,
        });
      }
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async () => {
    if (!deleteTarget) return;
    try {
      await apiClient.delete(`/api/admin/dictionaries/${deleteTarget.id}`);
      addToast({ type: "success", message: "已删除" });
      setDeleteTarget(null);
      void fetchDictionaries();
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

  const formatDate = (s: string) =>
    new Date(s).toLocaleString("zh-CN", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    });

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">词典管理</h1>
        <p className="text-muted-foreground mt-1">
          管理领域术语、同义词组与停用词,启用后同步到 OpenSearch IK 分词器
        </p>
      </div>

      <div className="flex items-center justify-between gap-4">
        <div className="relative w-72">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="搜索词典..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-9"
          />
        </div>
        <Button onClick={() => setCreateOpen(true)}>
          <Plus className="mr-2 h-4 w-4" />
          新建词典
        </Button>
      </div>

      {loading ? (
        <div className="text-center py-12 text-muted-foreground">加载中...</div>
      ) : filtered.length === 0 ? (
        <div className="text-center py-12 text-muted-foreground">
          {searchQuery ? "未找到匹配的词典" : "暂无词典,点击上方按钮创建"}
        </div>
      ) : (
        <div className="border rounded-lg overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-muted/50">
              <tr>
                <th className="text-left px-4 py-3 font-medium">名称</th>
                <th className="text-left px-4 py-3 font-medium">术语数</th>
                <th className="text-left px-4 py-3 font-medium">同义词组</th>
                <th className="text-left px-4 py-3 font-medium">停用词</th>
                <th className="text-left px-4 py-3 font-medium">状态</th>
                <th className="text-left px-4 py-3 font-medium">最近更新</th>
                <th className="text-right px-4 py-3 font-medium">操作</th>
              </tr>
            </thead>
            <tbody className="divide-y">
              {filtered.map((d) => (
                <tr key={d.id} className="hover:bg-muted/30">
                  <td className="px-4 py-3">
                    <Link
                      href={`/admin/dictionaries/${d.id}`}
                      className="font-medium text-primary hover:underline"
                    >
                      {d.name}
                    </Link>
                    {d.description && (
                      <p className="text-xs text-muted-foreground mt-0.5 line-clamp-1">
                        {d.description}
                      </p>
                    )}
                  </td>
                  <td className="px-4 py-3 text-muted-foreground">
                    {d.terms?.length ?? 0}
                  </td>
                  <td className="px-4 py-3 text-muted-foreground">
                    {d.synonyms?.length ?? 0}
                  </td>
                  <td className="px-4 py-3 text-muted-foreground">
                    {d.stop_words?.length ?? 0}
                  </td>
                  <td className="px-4 py-3">
                    <button
                      onClick={() => handleToggle(d)}
                      className="inline-flex items-center gap-1.5"
                      title={d.enabled ? "点击禁用" : "点击启用"}
                    >
                      {d.enabled ? (
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
                  <td className="px-4 py-3 text-xs text-muted-foreground">
                    {formatDate(d.updated_at)}
                  </td>
                  <td className="px-4 py-3 text-right">
                    <div className="flex items-center justify-end gap-1">
                      <Link href={`/admin/dictionaries/${d.id}`}>
                        <Button variant="ghost" size="sm">
                          编辑
                        </Button>
                      </Link>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setDeleteTarget(d)}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <Dialog open={createOpen} onOpenChange={setCreateOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>新建词典</DialogTitle>
            <DialogDescription>
              创建后可在编辑页面添加术语、同义词组和停用词
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="dict-name">名称</Label>
              <Input
                id="dict-name"
                value={createName}
                onChange={(e) => setCreateName(e.target.value)}
                placeholder="例如:化工领域术语"
                maxLength={100}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="dict-description">描述</Label>
              <Textarea
                id="dict-description"
                rows={2}
                value={createDescription}
                onChange={(e) => setCreateDescription(e.target.value)}
                placeholder="可选,该词典适用的领域和文档类型"
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setCreateOpen(false)}>
              取消
            </Button>
            <Button onClick={handleCreate} disabled={creating}>
              {creating ? "创建中..." : "创建"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={deleteTarget !== null}
        onOpenChange={(open) => !open && setDeleteTarget(null)}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>删除词典</DialogTitle>
            <DialogDescription>
              确认删除词典 <strong>{deleteTarget?.name}</strong>?该操作不可撤销,
              并会从 IK 分词器中移除关联词条。
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteTarget(null)}>
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
