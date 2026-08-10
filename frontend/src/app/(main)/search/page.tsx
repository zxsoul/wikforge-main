"use client";

import * as React from "react";
import { Search, FileText, Loader2, SearchX } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { apiClient } from "@/lib/api-client";
import { useDebounce } from "@/hooks/use-debounce";

interface SearchResult {
  id: string;
  document_id: string;
  document_title: string;
  content: string;
  highlight: string;
  score: number;
  chunk_index: number;
  title_chain?: string;
  page_number?: number;
}

export default function SearchPage() {
  const [query, setQuery] = React.useState("");
  const [results, setResults] = React.useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = React.useState(false);
  const [hasSearched, setHasSearched] = React.useState(false);
  const debouncedQuery = useDebounce(query, 300);

  // Trigger search when debounced query changes
  React.useEffect(() => {
    if (!debouncedQuery.trim()) {
      setResults([]);
      setHasSearched(false);
      return;
    }

    performSearch(debouncedQuery);
  }, [debouncedQuery]);

  const performSearch = async (searchQuery: string) => {
    setIsSearching(true);
    setHasSearched(true);

    try {
      const data = await apiClient.post<{ results: SearchResult[] }>(
        "/api/search",
        { query: searchQuery, page_size: 20 }
      );
      setResults(data.results || []);
    } catch {
      setResults([]);
    } finally {
      setIsSearching(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      performSearch(query.trim());
    }
  };

  const navigateToDocument = (result: SearchResult) => {
    const params = new URLSearchParams({
      highlight: String(result.chunk_index),
    });
    window.location.href = `/documents/${result.document_id}?${params.toString()}`;
  };

  return (
    <div className="mx-auto max-w-4xl space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold">搜索</h1>
        <p className="mt-1 text-muted-foreground">
          搜索知识库中的文档和内容
        </p>
      </div>

      {/* Search input */}
      <form onSubmit={handleSubmit} className="flex gap-2">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="输入关键词搜索文档..."
            className="pl-10"
          />
        </div>
        <Button type="submit" disabled={!query.trim() || isSearching}>
          {isSearching ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            "搜索"
          )}
        </Button>
      </form>

      {/* Results */}
      <div>
        {isSearching && (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="mr-2 h-5 w-5 animate-spin text-muted-foreground" />
            <span className="text-sm text-muted-foreground">搜索中...</span>
          </div>
        )}

        {!isSearching && hasSearched && results.length === 0 && (
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <SearchX className="mb-3 h-10 w-10 text-muted-foreground/50" />
            <h3 className="text-lg font-medium">未找到匹配内容</h3>
            <p className="mt-1 text-sm text-muted-foreground">
              请尝试调整关键词或使用更通用的搜索词
            </p>
          </div>
        )}

        {!isSearching && results.length > 0 && (
          <div className="space-y-1">
            <p className="mb-4 text-sm text-muted-foreground">
              找到 {results.length} 条结果
            </p>
            <div className="space-y-3">
              {results.map((result) => (
                <button
                  key={result.id}
                  onClick={() => navigateToDocument(result)}
                  className="flex w-full items-start gap-4 rounded-lg border p-4 text-left transition-colors hover:bg-accent"
                >
                  <FileText className="mt-0.5 h-5 w-5 shrink-0 text-muted-foreground" />
                  <div className="flex-1 overflow-hidden">
                    <div className="flex items-center gap-2">
                      <h4 className="font-medium truncate">
                        {result.document_title}
                      </h4>
                      <span className="shrink-0 rounded-full bg-muted px-2 py-0.5 text-xs text-muted-foreground">
                        {(result.score * 100).toFixed(0)}% 相关
                      </span>
                    </div>
                    {result.title_chain && (
                      <p className="mt-0.5 text-xs text-muted-foreground truncate">
                        {result.title_chain}
                      </p>
                    )}
                    <p
                      className="mt-1.5 text-sm text-muted-foreground line-clamp-3"
                      dangerouslySetInnerHTML={{
                        __html: highlightSnippet(result.highlight, 200),
                      }}
                    />
                    {result.page_number != null && (
                      <span className="mt-1.5 inline-block text-xs text-muted-foreground/70">
                        第 {result.page_number} 页
                      </span>
                    )}
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}

        {!hasSearched && !isSearching && (
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <Search className="mb-3 h-10 w-10 text-muted-foreground/50" />
            <h3 className="text-lg font-medium">开始搜索</h3>
            <p className="mt-1 text-sm text-muted-foreground">
              输入关键词搜索知识库中的文档内容
            </p>
            <p className="mt-3 text-xs text-muted-foreground">
              提示：在任意页面按{" "}
              <kbd className="rounded border bg-muted px-1.5 py-0.5 text-xs">
                ⌘K
              </kbd>{" "}
              可快速搜索
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * Truncate highlight snippet to max characters while preserving HTML tags
 */
function highlightSnippet(html: string, maxChars: number): string {
  if (!html) return "";
  // Strip HTML to count text length
  const textOnly = html.replace(/<[^>]*>/g, "");
  if (textOnly.length <= maxChars) return html;
  // Truncate the raw HTML at approximately the right point
  // This is a simple approach - for production, a proper HTML-aware truncation would be better
  let textCount = 0;
  let result = "";
  let inTag = false;

  for (const char of html) {
    if (char === "<") {
      inTag = true;
      result += char;
    } else if (char === ">") {
      inTag = false;
      result += char;
    } else if (inTag) {
      result += char;
    } else {
      textCount++;
      if (textCount > maxChars) {
        result += "...";
        break;
      }
      result += char;
    }
  }

  return result;
}
