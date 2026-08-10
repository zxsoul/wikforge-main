"use client";

import { UserManagement } from "@/components/admin/user-management";

export default function AdminUsersPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">用户管理</h1>
        <p className="text-muted-foreground mt-1">
          管理用户列表、搜索用户、分配角色
        </p>
      </div>
      <UserManagement />
    </div>
  );
}
