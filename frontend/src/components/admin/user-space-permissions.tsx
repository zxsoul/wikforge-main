"use client";

import * as React from "react";
import Link from "next/link";
import { ChevronLeft, Save } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { useToast } from "@/components/ui/toast";
import { apiClient, ApiClientError } from "@/lib/api-client";

interface SpacePermissionItem {
  space_id: string;
  space_name: string;
  description: string | null;
  // null 表示无权限记录（等同 invisible / 默认不可见）
  access_level: string | null;
}

interface SpacePermissionListResponse {
  user_id: string;
  items: SpacePermissionItem[];
}

interface SetResult {
  space_id: string;
  access_level: string;
  status: string;
  detail: string | null;
}

interface SetResponse {
  user_id: string;
  results: SetResult[];
}

const LEVEL_LABELS: Record<string, string> = {
  read: "可读",
  write: "可写",
  invisible: "不可见",
};

function isVisible(level: string | null): boolean {
  return level === "read" || level === "write";
}

export function UserSpacePermissions({ userId }: { userId: string }) {
  const { addToast } = useToast();
  const [items, setItems] = React.useState<SpacePermissionItem[]>([]);
  const [checked, setChecked] = React.useState<Record<string, boolean>>({});
  const [loading, setLoading] = React.useState(true);
  const [saving, setSaving] = React.useState(false);

  const fetchPermissions = React.useCallback(async () => {
    try {
      setLoading(true);
      const data = await apiClient.get<SpacePermissionListResponse>(
        `/api/admin/users/${userId}/space-permissions`
      );
      const list = data.items || [];
      setItems(list);
      const initial: Record<string, boolean> = {};
      for (const item of list) {
        initial[item.space_id] = isVisible(item.access_level);
      }
      setChecked(initial);
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({
          type: "error",
          message: "加载空间权限失败",
          description: err.message,
        });
      }
    } finally {
      setLoading(false);
    }
  }, [userId, addToast]);

  React.useEffect(() => {
    fetchPermissions();
  }, [fetchPermissions]);

  const toggle = (spaceId: string) => {
    setChecked((prev) => ({ ...prev, [spaceId]: !prev[spaceId] }));
  };

  const visibleCount = items.filter((i) => checked[i.space_id]).length;

  const handleSave = async () => {
    // 只提交发生变化的空间：
    // 勾选 -> 原为 write 保持 write，否则 read；未勾选 -> invisible
    const changes = items
      .filter((item) => isVisible(item.access_level) !== !!checked[item.space_id])
      .map((item) => ({
        space_id: item.space_id,
        access_level: checked[item.space_id]
          ? item.access_level === "write"
            ? "write"
            : "read"
          : "invisible",
      }));

    if (changes.length === 0) {
      addToast({ type: "success", message: "没有需要保存的变更" });
      return;
    }

    try {
      setSaving(true);
      const resp = await apiClient.put<SetResponse>(
        `/api/admin/users/${userId}/space-permissions`,
        { items: changes }
      );
      const failed = (resp.results || []).filter((r) => r.status !== "ok");
      if (failed.length === 0) {
        addToast({
          type: "success",
          message: "空间权限已保存",
          description: "检索权限同步已触发，问答检索将按新权限过滤",
        });
      } else {
        addToast({
          type: "error",
          message: `部分空间保存失败（${failed.length}/${changes.length}）`,
          description: failed[0]?.detail || undefined,
        });
      }
      await fetchPermissions();
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({
          type: "error",
          message: "保存空间权限失败",
          description: err.message,
        });
      }
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="text-center py-12 text-muted-foreground">加载中...</div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <Link
          href="/admin/users"
          className="inline-flex items-center text-sm text-muted-foreground hover:text-foreground"
        >
          <ChevronLeft className="mr-1 h-4 w-4" />
          返回用户列表
        </Link>
      </div>

      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-base">空间权限分配</CardTitle>
              <CardDescription>
                勾选该用户可见的空间（共 {items.length} 个空间，已选{" "}
                {visibleCount} 个）。保存后检索与问答将立即按新权限过滤。
              </CardDescription>
            </div>
            <Button onClick={handleSave} disabled={saving}>
              <Save className="mr-2 h-4 w-4" />
              {saving ? "保存中..." : "保存权限"}
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="rounded-md border">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b bg-muted/50">
                  <th className="px-4 py-3 text-center font-medium w-16">可见</th>
                  <th className="px-4 py-3 text-left font-medium">空间名称</th>
                  <th className="px-4 py-3 text-left font-medium">描述</th>
                  <th className="px-4 py-3 text-left font-medium w-28">
                    当前权限
                  </th>
                </tr>
              </thead>
              <tbody>
                {items.length === 0 ? (
                  <tr>
                    <td
                      colSpan={4}
                      className="px-4 py-8 text-center text-muted-foreground"
                    >
                      暂无空间
                    </td>
                  </tr>
                ) : (
                  items.map((item) => (
                    <tr key={item.space_id} className="border-b last:border-0">
                      <td className="px-4 py-3 text-center">
                        <input
                          type="checkbox"
                          className="h-4 w-4 rounded border-gray-300 text-primary focus:ring-primary"
                          checked={!!checked[item.space_id]}
                          onChange={() => toggle(item.space_id)}
                        />
                      </td>
                      <td className="px-4 py-3 font-medium">
                        {item.space_name}
                      </td>
                      <td className="px-4 py-3 text-muted-foreground">
                        {item.description || "—"}
                      </td>
                      <td className="px-4 py-3 text-muted-foreground">
                        {item.access_level
                          ? LEVEL_LABELS[item.access_level] || item.access_level
                          : "未设置"}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
