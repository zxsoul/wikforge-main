"use client";

import { SpaceManagement } from "@/components/admin/space-management";

export default function AdminSpacesPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">空间管理</h1>
        <p className="text-muted-foreground mt-1">
          管理知识空间、成员分配和角色设置
        </p>
      </div>
      <SpaceManagement />
    </div>
  );
}
