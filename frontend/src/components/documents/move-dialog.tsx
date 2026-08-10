"use client";

import { useState, useEffect, useCallback } from "react";
import {
  ChevronRight,
  ChevronDown,
  Folder,
  FolderOpen,
  Loader2,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { apiClient } from "@/lib/api-client";

interface Space {
  id: string;
  name: string;
}

interface FolderNode {
  id: string;
  name: string;
  children?: FolderNode[];
}

interface MoveTarget {
  spaceId: string;
  folderId?: string;
  label: string;
}

interface MoveDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  documentId: string;
  documentTitle: string;
  onMoveComplete?: () => void;
}

export function MoveDialog({
  open,
  onOpenChange,
  documentId,
  documentTitle,
  onMoveComplete,
}: MoveDialogProps) {
  const [spaces, setSpaces] = useState<Space[]>([]);
  const [folderTrees, setFolderTrees] = useState<
    Record<string, FolderNode[]>
  >({});
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set());
  const [selectedTarget, setSelectedTarget] = useState<MoveTarget | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isMoving, setIsMoving] = useState(false);
  const [loadingSpaces, setLoadingSpaces] = useState<Set<string>>(new Set());

  useEffect(() => {
    if (!open) return;
    const fetchSpaces = async () => {
      setIsLoading(true);
      try {
        const data = await apiClient.get<Space[]>("/api/spaces");
        setSpaces(data);
      } catch {
        // Handle error
      } finally {
        setIsLoading(false);
      }
    };
    fetchSpaces();
    setSelectedTarget(null);
    setExpandedNodes(new Set());
    setFolderTrees({});
  }, [open]);

  const loadFolders = useCallback(
    async (spaceId: string) => {
      if (folderTrees[spaceId]) return;
      setLoadingSpaces((prev) => new Set(prev).add(spaceId));
      try {
        const data = await apiClient.get<FolderNode[]>(
          `/api/spaces/${spaceId}/tree`
        );
        setFolderTrees((prev) => ({ ...prev, [spaceId]: data }));
      } catch {
        setFolderTrees((prev) => ({ ...prev, [spaceId]: [] }));
      } finally {
        setLoadingSpaces((prev) => {
          const next = new Set(prev);
          next.delete(spaceId);
          return next;
        });
      }
    },
    [folderTrees]
  );

  const toggleExpand = useCallback(
    (nodeId: string, spaceId?: string) => {
      setExpandedNodes((prev) => {
        const next = new Set(prev);
        if (next.has(nodeId)) {
          next.delete(nodeId);
        } else {
          next.add(nodeId);
          if (spaceId) loadFolders(spaceId);
        }
        return next;
      });
    },
    [loadFolders]
  );

  const handleMove = async () => {
    if (!selectedTarget) return;
    setIsMoving(true);
    try {
      await apiClient.patch(`/api/documents/${documentId}/move`, {
        space_id: selectedTarget.spaceId,
        folder_id: selectedTarget.folderId || null,
      });
      onMoveComplete?.();
      onOpenChange(false);
    } catch {
      // Handle error
    } finally {
      setIsMoving(false);
    }
  };

  const renderFolderNode = (
    node: FolderNode,
    spaceId: string,
    depth: number
  ) => {
    const isExpanded = expandedNodes.has(node.id);
    const isSelected =
      selectedTarget?.spaceId === spaceId &&
      selectedTarget?.folderId === node.id;
    const hasChildren = node.children && node.children.length > 0;

    return (
      <div key={node.id}>
        <div
          className={cn(
            "flex cursor-pointer items-center gap-1 rounded-md px-2 py-1.5 text-sm transition-colors hover:bg-accent",
            isSelected && "bg-accent ring-1 ring-primary"
          )}
          style={{ paddingLeft: `${depth * 16 + 8}px` }}
          onClick={() =>
            setSelectedTarget({
              spaceId,
              folderId: node.id,
              label: node.name,
            })
          }
        >
          {hasChildren ? (
            <button
              className="shrink-0 rounded p-0.5 hover:bg-muted"
              onClick={(e) => {
                e.stopPropagation();
                toggleExpand(node.id);
              }}
            >
              {isExpanded ? (
                <ChevronDown className="h-3.5 w-3.5" />
              ) : (
                <ChevronRight className="h-3.5 w-3.5" />
              )}
            </button>
          ) : (
            <span className="w-5" />
          )}
          {isExpanded ? (
            <FolderOpen className="h-4 w-4 shrink-0 text-muted-foreground" />
          ) : (
            <Folder className="h-4 w-4 shrink-0 text-muted-foreground" />
          )}
          <span className="truncate">{node.name}</span>
        </div>
        {isExpanded &&
          hasChildren &&
          node.children!.map((child) =>
            renderFolderNode(child, spaceId, depth + 1)
          )}
      </div>
    );
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>移动文档</DialogTitle>
          <DialogDescription className="truncate">
            将 &ldquo;{documentTitle}&rdquo; 移动到目标位置
          </DialogDescription>
        </DialogHeader>

        <ScrollArea className="h-72">
          {isLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
            </div>
          ) : spaces.length === 0 ? (
            <p className="py-8 text-center text-sm text-muted-foreground">
              暂无可用空间
            </p>
          ) : (
            <div className="space-y-1 pr-4">
              {spaces.map((space) => {
                const isExpanded = expandedNodes.has(space.id);
                const isSelected =
                  selectedTarget?.spaceId === space.id &&
                  !selectedTarget?.folderId;
                const isLoadingFolders = loadingSpaces.has(space.id);

                return (
                  <div key={space.id}>
                    <div
                      className={cn(
                        "flex cursor-pointer items-center gap-1 rounded-md px-2 py-1.5 text-sm font-medium transition-colors hover:bg-accent",
                        isSelected && "bg-accent ring-1 ring-primary"
                      )}
                      onClick={() =>
                        setSelectedTarget({
                          spaceId: space.id,
                          folderId: undefined,
                          label: space.name,
                        })
                      }
                    >
                      <button
                        className="shrink-0 rounded p-0.5 hover:bg-muted"
                        onClick={(e) => {
                          e.stopPropagation();
                          toggleExpand(space.id, space.id);
                        }}
                      >
                        {isLoadingFolders ? (
                          <Loader2 className="h-3.5 w-3.5 animate-spin" />
                        ) : isExpanded ? (
                          <ChevronDown className="h-3.5 w-3.5" />
                        ) : (
                          <ChevronRight className="h-3.5 w-3.5" />
                        )}
                      </button>
                      {isExpanded ? (
                        <FolderOpen className="h-4 w-4 shrink-0 text-muted-foreground" />
                      ) : (
                        <Folder className="h-4 w-4 shrink-0 text-muted-foreground" />
                      )}
                      <span className="truncate">{space.name}</span>
                    </div>
                    {isExpanded &&
                      folderTrees[space.id]?.map((folder) =>
                        renderFolderNode(folder, space.id, 1)
                      )}
                  </div>
                );
              })}
            </div>
          )}
        </ScrollArea>

        {selectedTarget && (
          <p className="text-sm text-muted-foreground">
            目标位置：{selectedTarget.label}
          </p>
        )}

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            取消
          </Button>
          <Button
            onClick={handleMove}
            disabled={!selectedTarget || isMoving}
          >
            {isMoving ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                移动中...
              </>
            ) : (
              "确认移动"
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
