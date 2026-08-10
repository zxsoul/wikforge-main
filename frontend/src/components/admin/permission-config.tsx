"use client";

import * as React from "react";
import { Save } from "lucide-react";
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

interface ModulePermission {
  module: string;
  module_label: string;
  description: string;
}

interface RolePermissions {
  role: string;
  role_label: string;
  modules: string[];
}

const DEFAULT_MODULES: ModulePermission[] = [
  { module: "documents", module_label: "文档管理", description: "文档上传、编辑、删除" },
  { module: "spaces", module_label: "空间管理", description: "空间创建、编辑、成员管理" },
  { module: "search", module_label: "知识检索", description: "全文搜索和语义搜索" },
  { module: "chat", module_label: "AI 问答", description: "RAG 对话式问答" },
  { module: "admin_users", module_label: "用户管理", description: "用户列表、角色分配" },
  { module: "admin_permissions", module_label: "权限配置", description: "角色权限设置" },
  { module: "admin_llm", module_label: "模型配置", description: "LLM 模型和参数管理" },
  { module: "admin_monitoring", module_label: "系统监控", description: "队列状态和资源监控" },
];

const DEFAULT_ROLES = [
  { role: "admin", role_label: "管理员" },
  { role: "editor", role_label: "编辑者" },
  { role: "viewer", role_label: "查看者" },
];

export function PermissionConfig() {
  const { addToast } = useToast();
  const [modules, setModules] = React.useState<ModulePermission[]>(DEFAULT_MODULES);
  const [rolePermissions, setRolePermissions] = React.useState<RolePermissions[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [saving, setSaving] = React.useState(false);

  const fetchPermissions = React.useCallback(async () => {
    try {
      setLoading(true);
      const data = await apiClient.get<{
        modules: ModulePermission[];
        roles: RolePermissions[];
      }>("/api/admin/permissions/config");
      if (data.modules?.length) setModules(data.modules);
      if (data.roles?.length) {
        setRolePermissions(data.roles);
      } else {
        // Initialize with defaults
        setRolePermissions(
          DEFAULT_ROLES.map((r) => ({
            ...r,
            modules:
              r.role === "admin"
                ? DEFAULT_MODULES.map((m) => m.module)
                : r.role === "editor"
                ? ["documents", "spaces", "search", "chat"]
                : ["search", "chat"],
          }))
        );
      }
    } catch (err) {
      // Use defaults on error
      setRolePermissions(
        DEFAULT_ROLES.map((r) => ({
          ...r,
          modules:
            r.role === "admin"
              ? DEFAULT_MODULES.map((m) => m.module)
              : r.role === "editor"
              ? ["documents", "spaces", "search", "chat"]
              : ["search", "chat"],
        }))
      );
      if (err instanceof ApiClientError && err.status !== 404) {
        addToast({ type: "error", message: "加载权限配置失败", description: (err as ApiClientError).message });
      }
    } finally {
      setLoading(false);
    }
  }, [addToast]);

  React.useEffect(() => {
    fetchPermissions();
  }, [fetchPermissions]);

  const togglePermission = (role: string, module: string) => {
    setRolePermissions((prev) =>
      prev.map((rp) => {
        if (rp.role !== role) return rp;
        const hasModule = rp.modules.includes(module);
        return {
          ...rp,
          modules: hasModule
            ? rp.modules.filter((m) => m !== module)
            : [...rp.modules, module],
        };
      })
    );
  };

  const handleSave = async () => {
    try {
      setSaving(true);
      await apiClient.put("/api/admin/permissions/config", {
        roles: rolePermissions,
      });
      addToast({ type: "success", message: "权限配置保存成功" });
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({ type: "error", message: "保存权限配置失败", description: err.message });
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
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-base">角色权限矩阵</CardTitle>
              <CardDescription>
                配置各角色对功能模块的访问权限
              </CardDescription>
            </div>
            <Button onClick={handleSave} disabled={saving}>
              <Save className="mr-2 h-4 w-4" />
              {saving ? "保存中..." : "保存配置"}
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="rounded-md border overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b bg-muted/50">
                  <th className="px-4 py-3 text-left font-medium min-w-[200px]">
                    功能模块
                  </th>
                  {rolePermissions.map((rp) => (
                    <th
                      key={rp.role}
                      className="px-4 py-3 text-center font-medium min-w-[100px]"
                    >
                      {rp.role_label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {modules.map((mod) => (
                  <tr key={mod.module} className="border-b last:border-0">
                    <td className="px-4 py-3">
                      <div>
                        <p className="font-medium">{mod.module_label}</p>
                        <p className="text-xs text-muted-foreground">
                          {mod.description}
                        </p>
                      </div>
                    </td>
                    {rolePermissions.map((rp) => (
                      <td key={rp.role} className="px-4 py-3 text-center">
                        <input
                          type="checkbox"
                          className="h-4 w-4 rounded border-gray-300 text-primary focus:ring-primary"
                          checked={rp.modules.includes(mod.module)}
                          onChange={() => togglePermission(rp.role, mod.module)}
                        />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
