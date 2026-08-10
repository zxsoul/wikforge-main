"use client";

import { useState, useCallback } from "react";
import { Upload, PanelLeftClose, PanelLeft, Trash2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { FileUpload } from "@/components/documents/file-upload";
import {
  DocumentList,
  type Document,
} from "@/components/documents/document-list";
import { FolderTree } from "@/components/documents/folder-tree";
import { TagManagerDialog } from "@/components/documents/tag-manager";
import { TagFilter } from "@/components/documents/tag-manager";
import { MoveDialog } from "@/components/documents/move-dialog";
import { DeleteDialog } from "@/components/documents/delete-dialog";
import { apiClient, ApiClientError } from "@/lib/api-client";
import { useToast } from "@/components/ui/toast";

export default function DocumentsPage() {
  const { addToast } = useToast();
  // Navigation state
  const [selectedSpaceId, setSelectedSpaceId] = useState<string | undefined>();
  const [selectedFolderId, setSelectedFolderId] = useState<
    string | undefined
  >();
  const [tagFilter, setTagFilter] = useState<string | undefined>();

  // UI state
  const [showUpload, setShowUpload] = useState(false);
  const [showTreePanel, setShowTreePanel] = useState(true);
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  // Dialog state
  const [moveDoc, setMoveDoc] = useState<Document | null>(null);
  const [deleteDoc, setDeleteDoc] = useState<Document | null>(null);
  const [tagDoc, setTagDoc] = useState<Document | null>(null);

  const handleFolderSelect = useCallback(
    (spaceId: string, folderId?: string) => {
      setSelectedSpaceId(spaceId);
      setSelectedFolderId(folderId);
    },
    []
  );

  const handleUploadComplete = useCallback(() => {
    setRefreshTrigger((prev) => prev + 1);
  }, []);

  const handleMoveComplete = useCallback(() => {
    setRefreshTrigger((prev) => prev + 1);
    setMoveDoc(null);
  }, []);

  const handleDeleteComplete = useCallback(() => {
    setRefreshTrigger((prev) => prev + 1);
    setDeleteDoc(null);
  }, []);

  const handleTagsChanged = useCallback(() => {
    setRefreshTrigger((prev) => prev + 1);
  }, []);

  return (
    <div className="flex h-full flex-col gap-4">
      {/* Header */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold lg:text-3xl">文档管理</h1>
          <p className="text-sm text-muted-foreground">
            管理您的知识库文档
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            className="lg:hidden"
            onClick={() => setShowTreePanel(!showTreePanel)}
          >
            {showTreePanel ? (
              <PanelLeftClose className="mr-1 h-4 w-4" />
            ) : (
              <PanelLeft className="mr-1 h-4 w-4" />
            )}
            目录
          </Button>
          <Button
            onClick={() => setShowUpload(!showUpload)}
            size="sm"
          >
            <Upload className="mr-1 h-4 w-4" />
            上传文档
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={async () => {
              if (!confirm("确认清理所有处理失败的文档?")) return;
              try {
                const r = await apiClient.post<{ deleted: number; ids: string[] }>(
                  "/api/documents/batch-delete",
                  { status: "failed" }
                );
                addToast({
                  type: "success",
                  message: `已清理 ${r.deleted} 个失败文档`,
                });
                setRefreshTrigger((v) => v + 1);
              } catch (err) {
                const msg = err instanceof ApiClientError ? err.message : "清理失败";
                addToast({ type: "error", message: "清理失败", description: msg });
              }
            }}
            title="一键清理 status=failed 的文档"
          >
            <Trash2 className="mr-1 h-4 w-4" />
            清理失败
          </Button>
        </div>
      </div>

      {/* Upload area (collapsible) */}
      {showUpload && (
        <div className="rounded-lg border p-4">
          {!selectedSpaceId ? (
            <div className="text-center text-sm text-muted-foreground py-6">
              请先在左侧选择要上传的知识空间
            </div>
          ) : (
            <FileUpload
              spaceId={selectedSpaceId}
              folderId={selectedFolderId}
              onUploadComplete={handleUploadComplete}
            />
          )}
        </div>
      )}

      {/* Main content: tree + document list */}
      <div className="flex min-h-0 flex-1 gap-4">
        {/* Folder tree panel */}
        {showTreePanel && (
          <aside
            className={cn(
              "shrink-0 overflow-y-auto rounded-lg border p-3",
              "w-full sm:w-56 lg:w-64",
              "max-h-[calc(100vh-280px)]",
              // On mobile, show as full width above the list
              "sm:block"
            )}
          >
            <FolderTree
              onSelect={handleFolderSelect}
              selectedSpaceId={selectedSpaceId}
              selectedFolderId={selectedFolderId}
            />
          </aside>
        )}

        {/* Document list area */}
        <div className="min-w-0 flex-1 space-y-3">
          {/* Tag filter */}
          <TagFilter selectedTag={tagFilter} onTagSelect={setTagFilter} />

          {/* Document table */}
          <DocumentList
            spaceId={selectedSpaceId}
            folderId={selectedFolderId}
            tagFilter={tagFilter}
            refreshTrigger={refreshTrigger}
            onMoveDocument={setMoveDoc}
            onDeleteDocument={setDeleteDoc}
            onManageTags={setTagDoc}
          />
        </div>
      </div>

      {/* Move dialog */}
      {moveDoc && (
        <MoveDialog
          open={!!moveDoc}
          onOpenChange={(open) => !open && setMoveDoc(null)}
          documentId={moveDoc.id}
          documentTitle={moveDoc.title}
          onMoveComplete={handleMoveComplete}
        />
      )}

      {/* Delete dialog */}
      {deleteDoc && (
        <DeleteDialog
          open={!!deleteDoc}
          onOpenChange={(open) => !open && setDeleteDoc(null)}
          documentId={deleteDoc.id}
          documentTitle={deleteDoc.title}
          onDeleteComplete={handleDeleteComplete}
        />
      )}

      {/* Tag manager dialog */}
      {tagDoc && (
        <TagManagerDialog
          open={!!tagDoc}
          onOpenChange={(open) => !open && setTagDoc(null)}
          documentId={tagDoc.id}
          documentTitle={tagDoc.title}
          initialTags={tagDoc.tags || []}
          onTagsChanged={handleTagsChanged}
        />
      )}
    </div>
  );
}
