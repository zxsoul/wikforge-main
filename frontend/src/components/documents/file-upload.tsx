"use client";

import { useCallback, useState, useRef } from "react";
import { Upload, X, FileText, AlertCircle, CheckCircle2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

const ALLOWED_EXTENSIONS = [".pdf", ".docx", ".pptx", ".txt", ".md", ".html"];
const ALLOWED_MIME_TYPES = [
  "application/pdf",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "application/vnd.openxmlformats-officedocument.presentationml.presentation",
  "text/plain",
  "text/markdown",
  "text/html",
];
const MAX_FILE_SIZE = 100 * 1024 * 1024; // 100MB
const MAX_FILE_COUNT = 20;

export interface UploadFile {
  id: string;
  file: File;
  progress: number;
  status: "pending" | "uploading" | "success" | "error";
  error?: string;
}

interface FileUploadProps {
  /** 上传到哪个 space (必传; 未选择 space 时禁用上传) */
  spaceId?: string;
  /** 上传到哪个 folder (可选) */
  folderId?: string;
  onUploadComplete?: () => void;
}

function validateFile(file: File): string | null {
  const ext = "." + file.name.split(".").pop()?.toLowerCase();
  if (!ALLOWED_EXTENSIONS.includes(ext)) {
    return `不支持的文件格式 "${ext}"，支持格式：${ALLOWED_EXTENSIONS.join("、")}`;
  }
  if (file.size > MAX_FILE_SIZE) {
    return `文件大小 ${(file.size / 1024 / 1024).toFixed(1)}MB 超过 100MB 限制`;
  }
  return null;
}

function generateId(): string {
  return Math.random().toString(36).substring(2, 10);
}

export function FileUpload({ spaceId, folderId, onUploadComplete }: FileUploadProps) {
  const [files, setFiles] = useState<UploadFile[]>([]);
  const [isDragOver, setIsDragOver] = useState(false);
  const [globalError, setGlobalError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const addFiles = useCallback(
    (newFiles: FileList | File[]) => {
      setGlobalError(null);
      const fileArray = Array.from(newFiles);

      if (files.length + fileArray.length > MAX_FILE_COUNT) {
        setGlobalError(`最多同时上传 ${MAX_FILE_COUNT} 个文件`);
        return;
      }

      const uploadFiles: UploadFile[] = [];
      for (const file of fileArray) {
        const error = validateFile(file);
        uploadFiles.push({
          id: generateId(),
          file,
          progress: 0,
          status: error ? "error" : "pending",
          error: error || undefined,
        });
      }

      setFiles((prev) => [...prev, ...uploadFiles]);
    },
    [files.length]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragOver(false);
      if (e.dataTransfer.files.length > 0) {
        addFiles(e.dataTransfer.files);
      }
    },
    [addFiles]
  );

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files && e.target.files.length > 0) {
        addFiles(e.target.files);
      }
      // Reset input so same file can be selected again
      e.target.value = "";
    },
    [addFiles]
  );

  const removeFile = useCallback((id: string) => {
    setFiles((prev) => prev.filter((f) => f.id !== id));
  }, []);

  const uploadFiles = useCallback(async () => {
    const pendingFiles = files.filter((f) => f.status === "pending");
    if (pendingFiles.length === 0) return;

    if (!spaceId) {
      setGlobalError("请先在左侧选择要上传的知识空间");
      return;
    }

    const API_BASE_URL =
      process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    const token = localStorage.getItem("access_token");

    // 拼 query 参数: ?space_id=xxx&folder_id=yyy (folder_id 可选)
    const qs = new URLSearchParams({ space_id: spaceId });
    if (folderId) qs.set("folder_id", folderId);
    const uploadUrl = `${API_BASE_URL}/api/documents/upload?${qs.toString()}`;

    for (const uploadFile of pendingFiles) {
      setFiles((prev) =>
        prev.map((f) =>
          f.id === uploadFile.id ? { ...f, status: "uploading", progress: 0 } : f
        )
      );

      try {
        const formData = new FormData();
        formData.append("files", uploadFile.file);

        const xhr = new XMLHttpRequest();
        await new Promise<void>((resolve, reject) => {
          xhr.upload.addEventListener("progress", (e) => {
            if (e.lengthComputable) {
              const progress = Math.round((e.loaded / e.total) * 100);
              setFiles((prev) =>
                prev.map((f) =>
                  f.id === uploadFile.id ? { ...f, progress } : f
                )
              );
            }
          });

          xhr.addEventListener("load", () => {
            if (xhr.status >= 200 && xhr.status < 300) {
              setFiles((prev) =>
                prev.map((f) =>
                  f.id === uploadFile.id
                    ? { ...f, status: "success", progress: 100 }
                    : f
                )
              );
              resolve();
            } else {
              reject(new Error(`上传失败: ${xhr.statusText}`));
            }
          });

          xhr.addEventListener("error", () => {
            reject(new Error("网络错误"));
          });

          xhr.open("POST", uploadUrl);
          if (token) {
            xhr.setRequestHeader("Authorization", `Bearer ${token}`);
          }
          xhr.send(formData);
        });
      } catch (err) {
        setFiles((prev) =>
          prev.map((f) =>
            f.id === uploadFile.id
              ? {
                  ...f,
                  status: "error",
                  error: err instanceof Error ? err.message : "上传失败",
                }
              : f
          )
        );
      }
    }

    onUploadComplete?.();
  }, [files, onUploadComplete, spaceId, folderId]);

  const clearCompleted = useCallback(() => {
    setFiles((prev) => prev.filter((f) => f.status !== "success"));
  }, []);

  const hasPending = files.some((f) => f.status === "pending");
  const hasCompleted = files.some((f) => f.status === "success");
  const isUploading = files.some((f) => f.status === "uploading");

  return (
    <div className="space-y-4">
      {/* Drop zone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={cn(
          "flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed p-8 transition-colors",
          isDragOver
            ? "border-primary bg-primary/5"
            : "border-muted-foreground/25 hover:border-primary/50 hover:bg-muted/50"
        )}
      >
        <Upload className="mb-3 h-10 w-10 text-muted-foreground" />
        <p className="text-sm font-medium">拖拽文件到此处，或点击选择文件</p>
        <p className="mt-1 text-xs text-muted-foreground">
          支持 PDF、DOCX、PPTX、TXT、MD、HTML，单文件最大 100MB，最多 20 个文件
        </p>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept={ALLOWED_MIME_TYPES.join(",")}
          onChange={handleFileSelect}
          className="hidden"
        />
      </div>

      {/* Global error */}
      {globalError && (
        <div className="flex items-center gap-2 rounded-md bg-destructive/10 px-3 py-2 text-sm text-destructive">
          <AlertCircle className="h-4 w-4 shrink-0" />
          <span>{globalError}</span>
        </div>
      )}

      {/* File list */}
      {files.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">
              文件列表 ({files.length})
            </span>
            <div className="flex gap-2">
              {hasCompleted && (
                <Button variant="ghost" size="sm" onClick={clearCompleted}>
                  清除已完成
                </Button>
              )}
              {hasPending && (
                <Button size="sm" onClick={uploadFiles} disabled={isUploading}>
                  {isUploading ? "上传中..." : "开始上传"}
                </Button>
              )}
            </div>
          </div>

          <div className="max-h-60 space-y-1 overflow-y-auto">
            {files.map((f) => (
              <div
                key={f.id}
                className="flex items-center gap-3 rounded-md border px-3 py-2"
              >
                <FileText className="h-4 w-4 shrink-0 text-muted-foreground" />
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm">{f.file.name}</p>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-muted-foreground">
                      {(f.file.size / 1024 / 1024).toFixed(1)} MB
                    </span>
                    {f.status === "uploading" && (
                      <div className="flex flex-1 items-center gap-2">
                        <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
                          <div
                            className="h-full rounded-full bg-primary transition-all"
                            style={{ width: `${f.progress}%` }}
                          />
                        </div>
                        <span className="text-xs text-muted-foreground">
                          {f.progress}%
                        </span>
                      </div>
                    )}
                    {f.status === "error" && (
                      <span className="text-xs text-destructive">
                        {f.error}
                      </span>
                    )}
                    {f.status === "success" && (
                      <span className="flex items-center gap-1 text-xs text-green-600">
                        <CheckCircle2 className="h-3 w-3" />
                        已完成
                      </span>
                    )}
                  </div>
                </div>
                {f.status !== "uploading" && (
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6 shrink-0"
                    onClick={() => removeFile(f.id)}
                  >
                    <X className="h-3 w-3" />
                  </Button>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
