"use client";

import { PermissionConfig } from "@/components/admin/permission-config";

export default function AdminPermissionsPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">权限配置</h1>
        <p className="text-muted-foreground mt-1">
          按角色配置功能模块的访问权限
        </p>
      </div>
      <PermissionConfig />
    </div>
  );
}
