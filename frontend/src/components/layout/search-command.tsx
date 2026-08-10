"use client";

import * as React from "react";
import { Search, FileText, Loader2, ExternalLink } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useSearchStore } from "@/stores/search-store";
import { apiClient } from "@/lib/api-client";

interface SearchResult {
  id: string;
  document_id: string;
  document_title: string;
  content: string;
  highlight: string;
  score: number;
  chunk_index: number;
}

export function SearchCommand() {
  const { isOpen, query, results, isSearching, close, setQuery, setResults, setIsSearching } =
    useSearchStore();
  const inputRef = React.useRef<HTMLInputElement>(null);
  const debounceRef = React.useRef<NodeJS.Timeout | null>(null);
  const [selectedIndex, setSelectedIndex] = React.useState(-1);

  // 全局快捷键监听 Cmd+K / Ctrl+K
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        useSearchStore.getState().open();
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, []);

  // 打开时聚焦输入框
  React.useEffect(() => {
    if (isOpen) {
      setTimeout(() => inputRef.current?.focus(), 0);
      setSelectedIndex(-1);
    }
  }, [isOpen]);

  // Reset selection when results change
  React.useEffect(() => {
    setSelectedIndex(-1);
  }, [results]);

  // 搜索防抖
  const handleSearch = React.useCallback(
    (value: string) => {
      setQuery(value);

      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }

      if (!value.trim()) {
        setResults([]);
        setIsSearching(false);
        return;
      }

      setIsSearching(true);
      debounceRef.current = setTimeout(async () => {
        try {
          const data = await apiClient.post<{ results: SearchResult[] }>(
            "/api/search",
            { query: value, page_size: 20 }
          );
          setResults(data.results);
        } catch {
          setResults([]);
        } finally {
          setIsSearching(false);
        }
      }, 300);
    },
    [setQuery, setResults, setIsSearching]
  );

  const navigateToResult = (result: SearchResult) => {
    close();
    const params = new URLSearchParams({
      highlight: String(result.chunk_index),
    });
    window.location.href = `/documents/${result.document_id}?${params.toString()}`;
  };

  const handleInputKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setSelectedIndex((prev) =>
        prev < results.length - 1 ? prev + 1 : prev
      );
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSelectedIndex((prev) => (prev > 0 ? prev - 1 : -1));
    } else if (e.key === "Enter" && selectedIndex >= 0) {
      e.preventDefault();
      navigateToResult(results[selectedIndex]);
    }
  };

  const openFullSearch = () => {
    close();
    window.location.href = `/search?q=${encodeURIComponent(query)}`;
  };

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && close()}>
      <DialogContent className="top-[20%] translate-y-0 sm:max-w-[640px]">
        <DialogHeader className="sr-only">
          <DialogTitle>搜索知识库</DialogTitle>
        </DialogHeader>
        <div className="flex items-center border-b px-3 pb-3">
          <Search className="mr-2 h-4 w-4 shrink-0 opacity-50" />
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => handleSearch(e.target.value)}
            onKeyDown={handleInputKeyDown}
            placeholder="搜索文档、知识..."
            className="flex h-10 w-full rounded-md bg-transparent py-3 text-sm outline-none placeholder:text-muted-foreground disabled:cursor-not-allowed disabled:opacity-50"
          />
          {isSearching && (
            <Loader2 className="ml-2 h-4 w-4 animate-spin opacity-50" />
          )}
        </div>

        {/* Results */}
        <div className="max-h-[400px] overflow-y-auto">
          {results.length > 0 ? (
            <div className="space-y-1 p-2">
              {results.slice(0, 10).map((result, index) => (
                <button
                  key={result.id}
                  className={`flex w-full items-start gap-3 rounded-md px-3 py-2 text-left text-sm hover:bg-accent ${
                    selectedIndex === index ? "bg-accent" : ""
                  }`}
                  onClick={() => navigateToResult(result)}
                >
                  <FileText className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
                  <div className="flex-1 overflow-hidden">
                    <p className="font-medium truncate">
                      {result.document_title}
                    </p>
                    <p
                      className="mt-0.5 text-xs text-muted-foreground line-clamp-2"
                      dangerouslySetInnerHTML={{ __html: result.highlight }}
                    />
                  </div>
                  <span className="shrink-0 text-xs text-muted-foreground">
                    {(result.score * 100).toFixed(0)}%
                  </span>
                </button>
              ))}
              {results.length > 10 && (
                <button
                  onClick={openFullSearch}
                  className="flex w-full items-center justify-center gap-1.5 rounded-md px-3 py-2 text-sm text-muted-foreground hover:bg-accent hover:text-accent-foreground"
                >
                  <ExternalLink className="h-3.5 w-3.5" />
                  查看全部 {results.length} 条结果
                </button>
              )}
            </div>
          ) : query && !isSearching ? (
            <div className="py-6 text-center text-sm text-muted-foreground">
              未找到匹配内容，请尝试调整关键词
            </div>
          ) : !query ? (
            <div className="py-6 text-center text-sm text-muted-foreground">
              输入关键词开始搜索
            </div>
          ) : null}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between border-t px-3 pt-3 text-xs text-muted-foreground">
          <span>按 ESC 关闭</span>
          <span>↑↓ 导航 · Enter 打开</span>
        </div>
      </DialogContent>
    </Dialog>
  );
}
