"use client";

import { useState } from "react";
import { AlertTriangle, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { apiClient } from "@/lib/api-client";

interface DeleteDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  documentId: string;
  documentTitle: string;
  onDeleteComplete?: () => void;
}

export function DeleteDialog({
  open,
  onOpenChange,
  documentId,
  documentTitle,
  onDeleteComplete,
}: DeleteDialogProps) {
  const [isDeleting, setIsDeleting] = useState(false);

  const handleDelete = async () => {
    setIsDeleting(true);
    try {
      await apiClient.delete(`/api/documents/${documentId}`);
      onDeleteComplete?.();
      onOpenChange(false);
    } catch {
      // Handle error
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-destructive" />
            确认删除
          </DialogTitle>
          <DialogDescription>
            此操作不可撤销，请确认是否继续。
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-3 py-2">
          <p className="text-sm">
            确定要删除文档 <strong>&ldquo;{documentTitle}&rdquo;</strong> 吗？
          </p>
          <div className="rounded-md bg-destructive/10 p-3">
            <p className="text-sm text-destructive">
              ⚠️ 删除后，以下内容将被永久移除：
            </p>
            <ul className="mt-2 list-inside list-disc space-y-1 text-sm text-destructive/80">
              <li>文档原始文件</li>
              <li>文档的所有标签</li>
              <li>已生成的文档块和向量数据</li>
              <li>相关的搜索索引</li>
            </ul>
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            取消
          </Button>
          <Button
            variant="destructive"
            onClick={handleDelete}
            disabled={isDeleting}
          >
            {isDeleting ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                删除中...
              </>
            ) : (
              "确认删除"
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
