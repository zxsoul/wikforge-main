"use client";

import { useState, useCallback } from "react";
import { X, Plus, Tag } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { apiClient } from "@/lib/api-client";

const MAX_TAGS = 20;
const MAX_TAG_LENGTH = 30;

interface TagManagerDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  documentId: string;
  documentTitle: string;
  initialTags: string[];
  onTagsChanged?: () => void;
}

export function TagManagerDialog({
  open,
  onOpenChange,
  documentId,
  documentTitle,
  initialTags,
  onTagsChanged,
}: TagManagerDialogProps) {
  const [tags, setTags] = useState<string[]>(initialTags);
  const [newTag, setNewTag] = useState("");
  const [error, setError] = useState<string | null>(null);

  const addTag = useCallback(async () => {
    const tagName = newTag.trim();
    setError(null);

    if (!tagName) return;

    if (tagName.length > MAX_TAG_LENGTH) {
      setError(`标签名称不能超过 ${MAX_TAG_LENGTH} 个字符`);
      return;
    }

    if (tags.length >= MAX_TAGS) {
      setError(`每个文档最多 ${MAX_TAGS} 个标签`);
      return;
    }

    if (tags.includes(tagName)) {
      setError("标签已存在");
      return;
    }

    try {
      await apiClient.post(`/api/documents/${documentId}/tags`, {
        tag_name: tagName,
      });
      setTags((prev) => [...prev, tagName]);
      setNewTag("");
      onTagsChanged?.();
    } catch {
      setError("添加标签失败");
    }
  }, [newTag, tags, documentId, onTagsChanged]);

  const removeTag = useCallback(
    async (tagName: string) => {
      setError(null);
      try {
        await apiClient.delete(
          `/api/documents/${documentId}/tags/${encodeURIComponent(tagName)}`
        );
        setTags((prev) => prev.filter((t) => t !== tagName));
        onTagsChanged?.();
      } catch {
        setError("删除标签失败");
      }
    },
    [documentId, onTagsChanged]
  );

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>管理标签</DialogTitle>
          <DialogDescription className="truncate">
            文档：{documentTitle}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          {/* Add tag input */}
          <div className="flex items-center gap-2">
            <Input
              value={newTag}
              onChange={(e) => setNewTag(e.target.value)}
              placeholder="输入标签名称"
              maxLength={MAX_TAG_LENGTH}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  addTag();
                }
              }}
            />
            <Button size="sm" onClick={addTag}>
              <Plus className="mr-1 h-3.5 w-3.5" />
              添加
            </Button>
          </div>

          {/* Error */}
          {error && (
            <p className="text-sm text-destructive">{error}</p>
          )}

          {/* Tag list */}
          <div className="flex flex-wrap gap-2">
            {tags.length === 0 ? (
              <p className="text-sm text-muted-foreground">暂无标签</p>
            ) : (
              tags.map((tag) => (
                <span
                  key={tag}
                  className="inline-flex items-center gap-1 rounded-full bg-secondary px-3 py-1 text-sm"
                >
                  <Tag className="h-3 w-3" />
                  {tag}
                  <button
                    className="ml-0.5 rounded-full p-0.5 hover:bg-muted"
                    onClick={() => removeTag(tag)}
                  >
                    <X className="h-3 w-3" />
                  </button>
                </span>
              ))
            )}
          </div>

          <p className="text-xs text-muted-foreground">
            {tags.length}/{MAX_TAGS} 个标签
          </p>
        </div>
      </DialogContent>
    </Dialog>
  );
}

/* Tag filter component for the document list */
interface TagFilterProps {
  selectedTag?: string;
  onTagSelect: (tag: string | undefined) => void;
}

export function TagFilter({ selectedTag, onTagSelect }: TagFilterProps) {
  const [tags, setTags] = useState<string[]>([]);
  const [isLoaded, setIsLoaded] = useState(false);

  const loadTags = useCallback(async () => {
    if (isLoaded) return;
    try {
      const data = await apiClient.get<string[]>("/api/tags");
      setTags(data);
      setIsLoaded(true);
    } catch {
      // Silently handle
    }
  }, [isLoaded]);

  // Load tags on first render
  useState(() => {
    loadTags();
  });

  if (tags.length === 0) return null;

  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="text-xs text-muted-foreground">标签筛选:</span>
      {selectedTag && (
        <Button
          variant="ghost"
          size="sm"
          className="h-6 px-2 text-xs"
          onClick={() => onTagSelect(undefined)}
        >
          清除筛选
        </Button>
      )}
      {tags.map((tag) => (
        <button
          key={tag}
          className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs transition-colors ${
            selectedTag === tag
              ? "bg-primary text-primary-foreground"
              : "bg-secondary hover:bg-secondary/80"
          }`}
          onClick={() => onTagSelect(selectedTag === tag ? undefined : tag)}
        >
          <Tag className="h-2.5 w-2.5" />
          {tag}
        </button>
      ))}
    </div>
  );
}
