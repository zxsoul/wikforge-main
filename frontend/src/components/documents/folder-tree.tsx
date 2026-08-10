"use client";

import { useEffect, useState, useCallback } from "react";
import {
  ChevronRight,
  ChevronDown,
  Folder,
  FolderOpen,
  Plus,
  Loader2,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { apiClient } from "@/lib/api-client";

export interface Space {
  id: string;
  name: string;
  description?: string;
}

export interface FolderNode {
  id: string;
  name: string;
  space_id: string;
  parent_id: string | null;
  depth: number;
  children?: FolderNode[];
}

interface TreeNode {
  id: string;
  name: string;
  type: "space" | "folder";
  spaceId: string;
  folderId?: string;
  children?: TreeNode[];
  isExpanded?: boolean;
  isLoading?: boolean;
}

interface FolderTreeProps {
  onSelect: (spaceId: string, folderId?: string) => void;
  selectedSpaceId?: string;
  selectedFolderId?: string;
}

export function FolderTree({
  onSelect,
  selectedSpaceId,
  selectedFolderId,
}: FolderTreeProps) {
  const [spaces, setSpaces] = useState<Space[]>([]);
  const [treeData, setTreeData] = useState<TreeNode[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [newSpaceName, setNewSpaceName] = useState("");
  const [isCreatingSpace, setIsCreatingSpace] = useState(false);
  const [showNewSpaceInput, setShowNewSpaceInput] = useState(false);
  const [newFolderTarget, setNewFolderTarget] = useState<{
    spaceId: string;
    parentId?: string;
  } | null>(null);
  const [newFolderName, setNewFolderName] = useState("");

  const fetchSpaces = useCallback(async () => {
    setIsLoading(true);
    try {
      const data = await apiClient.get<Space[]>("/api/spaces");
      setSpaces(data);
      setTreeData(
        data.map((s) => ({
          id: s.id,
          name: s.name,
          type: "space" as const,
          spaceId: s.id,
          children: undefined,
          isExpanded: s.id === selectedSpaceId,
        }))
      );
    } catch {
      // Handle error silently
    } finally {
      setIsLoading(false);
    }
  }, [selectedSpaceId]);

  useEffect(() => {
    fetchSpaces();
  }, [fetchSpaces]);

  const buildTreeNodes = useCallback(
    (folders: FolderNode[], spaceId: string): TreeNode[] => {
      return folders.map((f) => ({
        id: f.id,
        name: f.name,
        type: "folder" as const,
        spaceId,
        folderId: f.id,
        children: f.children ? buildTreeNodes(f.children, spaceId) : undefined,
        isExpanded: false,
      }));
    },
    []
  );

  const loadFolderTree = useCallback(
    async (spaceId: string) => {
      try {
        const data = await apiClient.get<FolderNode[]>(
          `/api/spaces/${spaceId}/tree`
        );
        return buildTreeNodes(data, spaceId);
      } catch {
        return [];
      }
    },
    [buildTreeNodes]
  );

  const toggleNode = useCallback(
    async (nodeId: string) => {
      setTreeData((prev) => {
        const toggle = (nodes: TreeNode[]): TreeNode[] =>
          nodes.map((node) => {
            if (node.id === nodeId) {
              return { ...node, isExpanded: !node.isExpanded };
            }
            if (node.children) {
              return { ...node, children: toggle(node.children) };
            }
            return node;
          });
        return toggle(prev);
      });

      // Load children for space nodes that haven't been loaded
      const findNode = (nodes: TreeNode[]): TreeNode | undefined => {
        for (const n of nodes) {
          if (n.id === nodeId) return n;
          if (n.children) {
            const found = findNode(n.children);
            if (found) return found;
          }
        }
        return undefined;
      };

      const node = findNode(treeData);
      if (node && node.type === "space" && !node.children) {
        setTreeData((prev) => {
          const setLoading = (nodes: TreeNode[]): TreeNode[] =>
            nodes.map((n) =>
              n.id === nodeId ? { ...n, isLoading: true } : n
            );
          return setLoading(prev);
        });

        const children = await loadFolderTree(nodeId);
        setTreeData((prev) => {
          const setChildren = (nodes: TreeNode[]): TreeNode[] =>
            nodes.map((n) =>
              n.id === nodeId
                ? { ...n, children, isLoading: false, isExpanded: true }
                : n
            );
          return setChildren(prev);
        });
      }
    },
    [treeData, loadFolderTree]
  );

  const handleCreateSpace = async () => {
    if (!newSpaceName.trim()) return;
    setIsCreatingSpace(true);
    try {
      await apiClient.post("/api/spaces", { name: newSpaceName.trim() });
      setNewSpaceName("");
      setShowNewSpaceInput(false);
      await fetchSpaces();
    } catch {
      // Handle error
    } finally {
      setIsCreatingSpace(false);
    }
  };

  const handleCreateFolder = async () => {
    if (!newFolderName.trim() || !newFolderTarget) return;
    try {
      await apiClient.post(`/api/spaces/${newFolderTarget.spaceId}/folders`, {
        name: newFolderName.trim(),
        parent_id: newFolderTarget.parentId || null,
      });
      setNewFolderName("");
      setNewFolderTarget(null);
      // Reload tree for this space
      const children = await loadFolderTree(newFolderTarget.spaceId);
      setTreeData((prev) =>
        prev.map((n) =>
          n.id === newFolderTarget.spaceId
            ? { ...n, children, isExpanded: true }
            : n
        )
      );
    } catch {
      // Handle error
    }
  };

  const renderNode = (node: TreeNode, depth: number = 0) => {
    const isSelected =
      node.type === "space"
        ? selectedSpaceId === node.spaceId && !selectedFolderId
        : selectedFolderId === node.folderId;

    const hasChildren =
      node.children === undefined || (node.children && node.children.length > 0);

    return (
      <div key={node.id}>
        <div
          className={cn(
            "group flex cursor-pointer items-center gap-1 rounded-md px-2 py-1.5 text-sm transition-colors hover:bg-accent",
            isSelected && "bg-accent font-medium"
          )}
          style={{ paddingLeft: `${depth * 16 + 8}px` }}
          onClick={() => {
            onSelect(node.spaceId, node.folderId);
          }}
        >
          {/* Expand/collapse toggle */}
          <button
            className="shrink-0 rounded p-0.5 hover:bg-muted"
            onClick={(e) => {
              e.stopPropagation();
              toggleNode(node.id);
            }}
          >
            {node.isLoading ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : node.isExpanded ? (
              <ChevronDown className="h-3.5 w-3.5" />
            ) : (
              <ChevronRight className="h-3.5 w-3.5" />
            )}
          </button>

          {/* Icon */}
          {node.isExpanded ? (
            <FolderOpen className="h-4 w-4 shrink-0 text-muted-foreground" />
          ) : (
            <Folder className="h-4 w-4 shrink-0 text-muted-foreground" />
          )}

          {/* Name */}
          <span className="truncate">{node.name}</span>

          {/* Add folder button */}
          {depth < 10 && (
            <button
              className="ml-auto hidden shrink-0 rounded p-0.5 hover:bg-muted group-hover:block"
              onClick={(e) => {
                e.stopPropagation();
                setNewFolderTarget({
                  spaceId: node.spaceId,
                  parentId: node.type === "folder" ? node.folderId : undefined,
                });
              }}
              title="新建子目录"
            >
              <Plus className="h-3.5 w-3.5" />
            </button>
          )}
        </div>

        {/* Children */}
        {node.isExpanded && node.children && (
          <div>
            {node.children.map((child) => renderNode(child, depth + 1))}
          </div>
        )}

        {/* New folder input */}
        {newFolderTarget &&
          newFolderTarget.spaceId === node.spaceId &&
          ((node.type === "space" && !newFolderTarget.parentId) ||
            newFolderTarget.parentId === node.folderId) && (
            <div
              className="flex items-center gap-1 px-2 py-1"
              style={{ paddingLeft: `${(depth + 1) * 16 + 8}px` }}
            >
              <Input
                value={newFolderName}
                onChange={(e) => setNewFolderName(e.target.value)}
                placeholder="目录名称"
                className="h-7 text-xs"
                autoFocus
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleCreateFolder();
                  if (e.key === "Escape") setNewFolderTarget(null);
                }}
              />
              <Button
                size="sm"
                className="h-7 px-2 text-xs"
                onClick={handleCreateFolder}
              >
                确定
              </Button>
            </div>
          )}
      </div>
    );
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between px-2">
        <span className="text-xs font-medium uppercase text-muted-foreground">
          空间与目录
        </span>
        <Button
          variant="ghost"
          size="icon"
          className="h-6 w-6"
          onClick={() => setShowNewSpaceInput(true)}
          title="新建空间"
        >
          <Plus className="h-3.5 w-3.5" />
        </Button>
      </div>

      {/* New space input */}
      {showNewSpaceInput && (
        <div className="flex items-center gap-1 px-2">
          <Input
            value={newSpaceName}
            onChange={(e) => setNewSpaceName(e.target.value)}
            placeholder="空间名称"
            className="h-7 text-xs"
            autoFocus
            onKeyDown={(e) => {
              if (e.key === "Enter") handleCreateSpace();
              if (e.key === "Escape") setShowNewSpaceInput(false);
            }}
          />
          <Button
            size="sm"
            className="h-7 px-2 text-xs"
            onClick={handleCreateSpace}
            disabled={isCreatingSpace}
          >
            确定
          </Button>
        </div>
      )}

      {/* Tree */}
      <div className="space-y-0.5">
        {treeData.length === 0 ? (
          <p className="px-2 py-4 text-center text-xs text-muted-foreground">
            暂无空间，点击 + 创建
          </p>
        ) : (
          treeData.map((node) => renderNode(node))
        )}
      </div>
    </div>
  );
}
