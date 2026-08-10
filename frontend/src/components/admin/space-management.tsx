"use client";

import * as React from "react";
import { Plus, Pencil, Trash2, Users, Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Card,
  CardContent,
  CardDescription,
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

interface Space {
  id: string;
  name: string;
  description: string;
  document_count: number;
  member_count: number;
  created_at: string;
}

interface SpaceMember {
  user_id: string;
  email: string;
  display_name: string;
  role: "admin" | "editor" | "viewer";
}

type MemberRole = "admin" | "editor" | "viewer";

const ROLE_LABELS: Record<MemberRole, string> = {
  admin: "管理员",
  editor: "编辑者",
  viewer: "查看者",
};

export function SpaceManagement() {
  const { addToast } = useToast();
  const [spaces, setSpaces] = React.useState<Space[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [searchQuery, setSearchQuery] = React.useState("");

  // Dialog states
  const [createDialogOpen, setCreateDialogOpen] = React.useState(false);
  const [editDialogOpen, setEditDialogOpen] = React.useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = React.useState(false);
  const [memberDialogOpen, setMemberDialogOpen] = React.useState(false);
  const [selectedSpace, setSelectedSpace] = React.useState<Space | null>(null);

  // Form states
  const [formName, setFormName] = React.useState("");
  const [formDescription, setFormDescription] = React.useState("");
  const [members, setMembers] = React.useState<SpaceMember[]>([]);
  const [newMemberEmail, setNewMemberEmail] = React.useState("");
  const [newMemberRole, setNewMemberRole] = React.useState<MemberRole>("viewer");

  const fetchSpaces = React.useCallback(async () => {
    try {
      setLoading(true);
      const data = await apiClient.get<{ items: Space[] }>("/api/spaces");
      setSpaces(data.items || []);
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({ type: "error", message: "加载空间列表失败", description: err.message });
      }
    } finally {
      setLoading(false);
    }
  }, [addToast]);

  React.useEffect(() => {
    fetchSpaces();
  }, [fetchSpaces]);

  const filteredSpaces = spaces.filter(
    (s) =>
      s.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      s.description.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleCreate = async () => {
    try {
      await apiClient.post("/api/spaces", {
        name: formName,
        description: formDescription,
      });
      addToast({ type: "success", message: "空间创建成功" });
      setCreateDialogOpen(false);
      setFormName("");
      setFormDescription("");
      fetchSpaces();
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({ type: "error", message: "创建空间失败", description: err.message });
      }
    }
  };

  const handleEdit = async () => {
    if (!selectedSpace) return;
    try {
      await apiClient.put(`/api/spaces/${selectedSpace.id}`, {
        name: formName,
        description: formDescription,
      });
      addToast({ type: "success", message: "空间更新成功" });
      setEditDialogOpen(false);
      fetchSpaces();
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({ type: "error", message: "更新空间失败", description: err.message });
      }
    }
  };

  const handleDelete = async () => {
    if (!selectedSpace) return;
    try {
      await apiClient.delete(`/api/spaces/${selectedSpace.id}`);
      addToast({ type: "success", message: "空间删除成功" });
      setDeleteDialogOpen(false);
      setSelectedSpace(null);
      fetchSpaces();
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({ type: "error", message: "删除空间失败", description: err.message });
      }
    }
  };

  const openEditDialog = (space: Space) => {
    setSelectedSpace(space);
    setFormName(space.name);
    setFormDescription(space.description);
    setEditDialogOpen(true);
  };

  const openDeleteDialog = (space: Space) => {
    setSelectedSpace(space);
    setDeleteDialogOpen(true);
  };

  const openMemberDialog = async (space: Space) => {
    setSelectedSpace(space);
    setMemberDialogOpen(true);
    try {
      const data = await apiClient.get<{ items: SpaceMember[] }>(
        `/api/spaces/${space.id}/members`
      );
      setMembers(data.items || []);
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({ type: "error", message: "加载成员列表失败", description: err.message });
      }
    }
  };

  const handleAddMember = async () => {
    if (!selectedSpace || !newMemberEmail) return;
    try {
      await apiClient.post(`/api/spaces/${selectedSpace.id}/members`, {
        email: newMemberEmail,
        role: newMemberRole,
      });
      addToast({ type: "success", message: "成员添加成功" });
      setNewMemberEmail("");
      // Refresh members
      const data = await apiClient.get<{ items: SpaceMember[] }>(
        `/api/spaces/${selectedSpace.id}/members`
      );
      setMembers(data.items || []);
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({ type: "error", message: "添加成员失败", description: err.message });
      }
    }
  };

  const handleUpdateMemberRole = async (userId: string, role: MemberRole) => {
    if (!selectedSpace) return;
    try {
      await apiClient.put(`/api/spaces/${selectedSpace.id}/members/${userId}`, {
        role,
      });
      setMembers((prev) =>
        prev.map((m) => (m.user_id === userId ? { ...m, role } : m))
      );
      addToast({ type: "success", message: "角色更新成功" });
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({ type: "error", message: "更新角色失败", description: err.message });
      }
    }
  };

  const handleRemoveMember = async (userId: string) => {
    if (!selectedSpace) return;
    try {
      await apiClient.delete(
        `/api/spaces/${selectedSpace.id}/members/${userId}`
      );
      setMembers((prev) => prev.filter((m) => m.user_id !== userId));
      addToast({ type: "success", message: "成员移除成功" });
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({ type: "error", message: "移除成员失败", description: err.message });
      }
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="relative w-72">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="搜索空间..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-9"
          />
        </div>
        <Button onClick={() => setCreateDialogOpen(true)}>
          <Plus className="mr-2 h-4 w-4" />
          创建空间
        </Button>
      </div>

      {/* Space list */}
      {loading ? (
        <div className="text-center py-12 text-muted-foreground">加载中...</div>
      ) : filteredSpaces.length === 0 ? (
        <div className="text-center py-12 text-muted-foreground">
          {searchQuery ? "未找到匹配的空间" : "暂无空间，点击上方按钮创建"}
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {filteredSpaces.map((space) => (
            <Card key={space.id}>
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between">
                  <div>
                    <CardTitle className="text-base">{space.name}</CardTitle>
                    <CardDescription className="mt-1 line-clamp-2">
                      {space.description || "暂无描述"}
                    </CardDescription>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="flex items-center gap-4 text-sm text-muted-foreground mb-4">
                  <span>{space.document_count} 篇文档</span>
                  <span>{space.member_count} 位成员</span>
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => openMemberDialog(space)}
                  >
                    <Users className="mr-1 h-3 w-3" />
                    成员
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => openEditDialog(space)}
                  >
                    <Pencil className="mr-1 h-3 w-3" />
                    编辑
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => openDeleteDialog(space)}
                    className="text-destructive hover:text-destructive"
                  >
                    <Trash2 className="mr-1 h-3 w-3" />
                    删除
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Create Dialog */}
      <Dialog open={createDialogOpen} onOpenChange={setCreateDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>创建空间</DialogTitle>
            <DialogDescription>
              创建一个新的知识空间来组织文档
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="space-name">空间名称</Label>
              <Input
                id="space-name"
                placeholder="输入空间名称（最多 50 个字符）"
                maxLength={50}
                value={formName}
                onChange={(e) => setFormName(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="space-desc">描述</Label>
              <Input
                id="space-desc"
                placeholder="输入空间描述（最多 200 个字符）"
                maxLength={200}
                value={formDescription}
                onChange={(e) => setFormDescription(e.target.value)}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setCreateDialogOpen(false)}>
              取消
            </Button>
            <Button onClick={handleCreate} disabled={!formName.trim()}>
              创建
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Edit Dialog */}
      <Dialog open={editDialogOpen} onOpenChange={setEditDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>编辑空间</DialogTitle>
            <DialogDescription>修改空间信息</DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="edit-name">空间名称</Label>
              <Input
                id="edit-name"
                maxLength={50}
                value={formName}
                onChange={(e) => setFormName(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="edit-desc">描述</Label>
              <Input
                id="edit-desc"
                maxLength={200}
                value={formDescription}
                onChange={(e) => setFormDescription(e.target.value)}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setEditDialogOpen(false)}>
              取消
            </Button>
            <Button onClick={handleEdit} disabled={!formName.trim()}>
              保存
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Dialog (sub-task 2) */}
      <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>确认删除空间</DialogTitle>
            <DialogDescription>
              此操作不可撤销，将永久删除该空间及其所有内容。
            </DialogDescription>
          </DialogHeader>
          {selectedSpace && (
            <div className="rounded-md border border-destructive/20 bg-destructive/5 p-4">
              <p className="text-sm font-medium">
                即将删除空间：
                <span className="font-bold">{selectedSpace.name}</span>
              </p>
              <p className="text-sm text-muted-foreground mt-1">
                该空间包含{" "}
                <span className="font-semibold text-destructive">
                  {selectedSpace.document_count}
                </span>{" "}
                篇文档，删除后将无法恢复。
              </p>
            </div>
          )}
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

      {/* Member Management Dialog */}
      <Dialog open={memberDialogOpen} onOpenChange={setMemberDialogOpen}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>
              成员管理 - {selectedSpace?.name}
            </DialogTitle>
            <DialogDescription>管理空间成员及其角色</DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            {/* Add member */}
            <div className="flex items-end gap-2">
              <div className="flex-1 space-y-1">
                <Label htmlFor="member-email">添加成员</Label>
                <Input
                  id="member-email"
                  placeholder="输入用户邮箱"
                  value={newMemberEmail}
                  onChange={(e) => setNewMemberEmail(e.target.value)}
                />
              </div>
              <select
                className="h-10 rounded-md border border-input bg-background px-3 text-sm"
                value={newMemberRole}
                onChange={(e) => setNewMemberRole(e.target.value as MemberRole)}
              >
                <option value="viewer">查看者</option>
                <option value="editor">编辑者</option>
                <option value="admin">管理员</option>
              </select>
              <Button onClick={handleAddMember} disabled={!newMemberEmail.trim()}>
                添加
              </Button>
            </div>

            {/* Member list */}
            <div className="max-h-64 overflow-y-auto space-y-2">
              {members.length === 0 ? (
                <p className="text-sm text-muted-foreground text-center py-4">
                  暂无成员
                </p>
              ) : (
                members.map((member) => (
                  <div
                    key={member.user_id}
                    className="flex items-center justify-between rounded-md border p-3"
                  >
                    <div>
                      <p className="text-sm font-medium">
                        {member.display_name}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {member.email}
                      </p>
                    </div>
                    <div className="flex items-center gap-2">
                      <select
                        className="h-8 rounded-md border border-input bg-background px-2 text-xs"
                        value={member.role}
                        onChange={(e) =>
                          handleUpdateMemberRole(
                            member.user_id,
                            e.target.value as MemberRole
                          )
                        }
                      >
                        <option value="viewer">{ROLE_LABELS.viewer}</option>
                        <option value="editor">{ROLE_LABELS.editor}</option>
                        <option value="admin">{ROLE_LABELS.admin}</option>
                      </select>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleRemoveMember(member.user_id)}
                        className="h-8 w-8 p-0 text-destructive hover:text-destructive"
                      >
                        <Trash2 className="h-3 w-3" />
                      </Button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
