"use client";

import { useParams } from "next/navigation";
import { UserSpacePermissions } from "@/components/admin/user-space-permissions";

export default function AdminUserSpacePermissionsPage() {
  const params = useParams<{ id: string }>();
  const userId = params.id;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">用户空间权限</h1>
        <p className="text-muted-foreground mt-1">
          为该用户分配可访问的空间（空间级权限）
        </p>
      </div>
      <UserSpacePermissions userId={userId} />
    </div>
  );
}
