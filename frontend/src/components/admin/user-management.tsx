"use client";

import * as React from "react";
import { Search, ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useToast } from "@/components/ui/toast";
import { apiClient, ApiClientError } from "@/lib/api-client";
import { useDebounce } from "@/hooks/use-debounce";

interface User {
  id: string;
  email: string;
  display_name: string;
  role: string;
  created_at: string;
  last_login_at: string | null;
}

interface PaginatedResponse {
  items: User[];
  total: number;
  page: number;
  page_size: number;
}

const ROLES = [
  { value: "admin", label: "管理员" },
  { value: "editor", label: "编辑者" },
  { value: "viewer", label: "查看者" },
];

export function UserManagement() {
  const { addToast } = useToast();
  const [users, setUsers] = React.useState<User[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [searchQuery, setSearchQuery] = React.useState("");
  const [page, setPage] = React.useState(1);
  const [total, setTotal] = React.useState(0);
  const pageSize = 20;

  const debouncedSearch = useDebounce(searchQuery, 300);

  const fetchUsers = React.useCallback(async () => {
    try {
      setLoading(true);
      const params = new URLSearchParams({
        page: page.toString(),
        page_size: pageSize.toString(),
      });
      if (debouncedSearch) {
        params.set("search", debouncedSearch);
      }
      const data = await apiClient.get<PaginatedResponse>(
        `/api/admin/users?${params.toString()}`
      );
      setUsers(data.items || []);
      setTotal(data.total || 0);
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({ type: "error", message: "加载用户列表失败", description: err.message });
      }
    } finally {
      setLoading(false);
    }
  }, [page, debouncedSearch, addToast]);

  React.useEffect(() => {
    fetchUsers();
  }, [fetchUsers]);

  React.useEffect(() => {
    setPage(1);
  }, [debouncedSearch]);

  const totalPages = Math.ceil(total / pageSize);

  const handleRoleChange = async (userId: string, newRole: string) => {
    try {
      await apiClient.put(`/api/admin/users/${userId}/role`, { role: newRole });
      setUsers((prev) =>
        prev.map((u) => (u.id === userId ? { ...u, role: newRole } : u))
      );
      addToast({ type: "success", message: "角色更新成功" });
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({ type: "error", message: "更新角色失败", description: err.message });
      }
    }
  };

  return (
    <div className="space-y-6">
      {/* Search */}
      <div className="relative w-80">
        <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
        <Input
          placeholder="按用户名或邮箱搜索..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="pl-9"
        />
      </div>

      {/* User table */}
      <div className="rounded-md border">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b bg-muted/50">
              <th className="px-4 py-3 text-left font-medium">用户名</th>
              <th className="px-4 py-3 text-left font-medium">邮箱</th>
              <th className="px-4 py-3 text-left font-medium">角色</th>
              <th className="px-4 py-3 text-left font-medium">注册时间</th>
              <th className="px-4 py-3 text-left font-medium">最近登录</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td colSpan={5} className="px-4 py-8 text-center text-muted-foreground">
                  加载中...
                </td>
              </tr>
            ) : users.length === 0 ? (
              <tr>
                <td colSpan={5} className="px-4 py-8 text-center text-muted-foreground">
                  {debouncedSearch ? "未找到匹配的用户" : "暂无用户"}
                </td>
              </tr>
            ) : (
              users.map((user) => (
                <tr key={user.id} className="border-b last:border-0">
                  <td className="px-4 py-3 font-medium">{user.display_name}</td>
                  <td className="px-4 py-3 text-muted-foreground">
                    {user.email}
                  </td>
                  <td className="px-4 py-3">
                    <select
                      className="h-8 rounded-md border border-input bg-background px-2 text-xs"
                      value={user.role}
                      onChange={(e) => handleRoleChange(user.id, e.target.value)}
                    >
                      {ROLES.map((role) => (
                        <option key={role.value} value={role.value}>
                          {role.label}
                        </option>
                      ))}
                    </select>
                  </td>
                  <td className="px-4 py-3 text-muted-foreground">
                    {new Date(user.created_at).toLocaleDateString("zh-CN")}
                  </td>
                  <td className="px-4 py-3 text-muted-foreground">
                    {user.last_login_at
                      ? new Date(user.last_login_at).toLocaleDateString("zh-CN")
                      : "从未登录"}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <p className="text-sm text-muted-foreground">
            共 {total} 位用户，第 {page}/{totalPages} 页
          </p>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page <= 1}
            >
              <ChevronLeft className="h-4 w-4" />
              上一页
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              disabled={page >= totalPages}
            >
              下一页
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
