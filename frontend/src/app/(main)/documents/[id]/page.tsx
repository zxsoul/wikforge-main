"use client";

import * as React from "react";
import { useParams, useSearchParams } from "next/navigation";
import { ArrowLeft, FileText, AlertCircle, Loader2 } from "lucide-react";
import { cn, generateUUID } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { apiClient, ApiClientError } from "@/lib/api-client";

interface DocumentDetail {
  id: string;
  title: string;
  file_type: string;
  file_size: number;
  status: string;
  space_id: string;
  created_at: string;
  updated_at: string;
}

interface DocumentChunk {
  chunk_id: string;
  chunk_index: number;
  content: string;
  page_number: number | null;
  title_chain: string | null;
}

interface DocumentChunks {
  document_id: string;
  total: number;
  chunks: DocumentChunk[];
}

const STATUS_LABEL: Record<string, string> = {
  completed: "已入库",
  parsing: "解析中",
  embedding: "向量化中",
  indexing: "索引中",
  pending: "排队中",
  failed: "失败",
};

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export default function DocumentDetailPage() {
  const params = useParams();
  const searchParams = useSearchParams();
  const documentId = params.id as string;
  const highlightParam = searchParams.get("highlight");
  const highlightIndex =
    highlightParam !== null ? Number(highlightParam) : null;

  const [detail, setDetail] = React.useState<DocumentDetail | null>(null);
  const [chunks, setChunks] = React.useState<DocumentChunk[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const [doc, chunkData] = await Promise.all([
          apiClient.get<DocumentDetail>(`/api/documents/${documentId}`),
          apiClient.get<DocumentChunks>(
            `/api/documents/${documentId}/chunks`
          ),
        ]);
        if (cancelled) return;
        setDetail(doc);
        setChunks(chunkData.chunks || []);
      } catch (err) {
        if (cancelled) return;
        if (err instanceof ApiClientError) {
          if (err.status === 404) setError("文档不存在或已被删除");
          else if (err.status === 403) setError("无权访问该文档");
          else setError(`加载失败: ${err.message}`);
        } else {
          setError("网络错误，请检查连接后重试");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [documentId]);

  // 高亮目标分块渲染后滚动到可视区域
  React.useEffect(() => {
    if (highlightIndex === null || chunks.length === 0) return;
    const el = window.document.getElementById(
      `chunk-${highlightIndex}`
    );
    el?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [highlightIndex, chunks]);

  return (
    <div className="flex h-[calc(100vh-theme(spacing.16))] -m-6 flex-col">
      {/* 头部 */}
      <div className="flex items-center gap-3 border-b px-6 py-3">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => (window.location.href = "/documents")}
        >
          <ArrowLeft className="mr-1 h-4 w-4" />
          返回文档列表
        </Button>
        {detail && (
          <div className="flex min-w-0 flex-1 items-center gap-3">
            <FileText className="h-5 w-5 shrink-0 text-muted-foreground" />
            <div className="min-w-0">
              <h1 className="truncate text-base font-semibold">
                {detail.title}
              </h1>
              <p className="text-xs text-muted-foreground">
                {detail.file_type.toUpperCase()} ·{" "}
                {formatSize(detail.file_size)} ·{" "}
                {STATUS_LABEL[detail.status] || detail.status} ·{" "}
                {chunks.length > 0 ? `${chunks.length} 个分块 · ` : ""}
                上传于 {new Date(detail.created_at).toLocaleString()}
              </p>
            </div>
          </div>
        )}
      </div>

      {/* 内容区 */}
      <ScrollArea className="flex-1">
        <div className="mx-auto max-w-3xl px-6 py-6">
          {loading && (
            <div className="flex items-center justify-center py-20 text-muted-foreground">
              <Loader2 className="mr-2 h-5 w-5 animate-spin" />
              加载文档内容...
            </div>
          )}

          {error && (
            <div className="flex items-center justify-center gap-2 py-20 text-destructive">
              <AlertCircle className="h-5 w-5" />
              {error}
            </div>
          )}

          {!loading && !error && chunks.length === 0 && (
            <div className="py-20 text-center text-muted-foreground">
              该文档暂无可展示的内容（可能仍在处理中或处理失败）
            </div>
          )}

          {!loading &&
            !error &&
            chunks.map((chunk) => {
              const highlighted = chunk.chunk_index === highlightIndex;
              return (
                <Card
                  key={chunk.chunk_id || generateUUID()}
                  id={`chunk-${chunk.chunk_index}`}
                  className={cn(
                    "mb-3 p-4 transition-colors",
                    highlighted &&
                      "border-primary bg-primary/5 ring-1 ring-primary"
                  )}
                >
                  {(chunk.title_chain || chunk.page_number) && (
                    <div className="mb-1 text-xs text-muted-foreground">
                      {chunk.title_chain && <span>{chunk.title_chain}</span>}
                      {chunk.page_number != null && (
                        <span className="ml-2">第 {chunk.page_number} 页</span>
                      )}
                      <span className="ml-2 text-muted-foreground/60">
                        #{chunk.chunk_index}
                      </span>
                    </div>
                  )}
                  <p className="whitespace-pre-wrap text-sm leading-6">
                    {chunk.content}
                  </p>
                </Card>
              );
            })}
        </div>
      </ScrollArea>
    </div>
  );
}
