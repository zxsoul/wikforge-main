"use client";

import * as React from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { ChevronLeft, Eye, EyeOff, Plus, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { useToast } from "@/components/ui/toast";
import { apiClient, ApiClientError } from "@/lib/api-client";

interface DictionaryResponse {
  id: string;
  name: string;
  description: string | null;
  terms: Array<Record<string, unknown>>;
}

interface CandidateTerm {
  word: string;
  frequency: number;
}

const DEFAULT_PARAMS = {
  minFrequency: 3,
  minLength: 2,
  maxLength: 10,
  topN: 50,
};

function ignoredKey(dictId: string) {
  return `wikforge:dict:${dictId}:ignored-candidates`;
}

function loadIgnored(dictId: string): Set<string> {
  if (typeof window === "undefined") return new Set();
  try {
    const raw = window.localStorage.getItem(ignoredKey(dictId));
    if (!raw) return new Set();
    const parsed = JSON.parse(raw);
    return new Set(Array.isArray(parsed) ? parsed.map((s) => String(s)) : []);
  } catch {
    return new Set();
  }
}

function saveIgnored(dictId: string, ignored: Set<string>) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(
      ignoredKey(dictId),
      JSON.stringify(Array.from(ignored))
    );
  } catch {
    // 存储失败时静默,不打断主流程
  }
}

export default function CandidateTermsPage() {
  const params = useParams<{ id: string }>();
  const dictId = params.id;
  const { addToast } = useToast();

  const [dictionary, setDictionary] = React.useState<DictionaryResponse | null>(
    null
  );
  const [loadingMeta, setLoadingMeta] = React.useState(true);

  const [sampleText, setSampleText] = React.useState("");
  const [minFrequency, setMinFrequency] = React.useState(
    String(DEFAULT_PARAMS.minFrequency)
  );
  const [minLength, setMinLength] = React.useState(
    String(DEFAULT_PARAMS.minLength)
  );
  const [maxLength, setMaxLength] = React.useState(
    String(DEFAULT_PARAMS.maxLength)
  );
  const [topN, setTopN] = React.useState(String(DEFAULT_PARAMS.topN));

  const [extracting, setExtracting] = React.useState(false);
  const [candidates, setCandidates] = React.useState<CandidateTerm[]>([]);
  const [showIgnored, setShowIgnored] = React.useState(false);

  const [ignored, setIgnored] = React.useState<Set<string>>(new Set());
  const [adding, setAdding] = React.useState<string | null>(null);
  const [bulkAdding, setBulkAdding] = React.useState(false);

  // 加载词典基本信息和已忽略候选缓存
  React.useEffect(() => {
    if (!dictId) return;
    setIgnored(loadIgnored(dictId));
  }, [dictId]);

  React.useEffect(() => {
    if (!dictId) return;
    let cancelled = false;
    (async () => {
      try {
        setLoadingMeta(true);
        const data = await apiClient.get<DictionaryResponse>(
          `/api/admin/dictionaries/${dictId}`
        );
        if (!cancelled) setDictionary(data);
      } catch (err) {
        if (!cancelled && err instanceof ApiClientError) {
          addToast({
            type: "error",
            message: "加载词典失败",
            description: err.message,
          });
        }
      } finally {
        if (!cancelled) setLoadingMeta(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [dictId, addToast]);

  const handleExtract = async () => {
    const text = sampleText.trim();
    if (!text) {
      addToast({ type: "error", message: "请粘贴或输入文档样本文本" });
      return;
    }

    const minFreq = Number(minFrequency) || DEFAULT_PARAMS.minFrequency;
    const minLen = Number(minLength) || DEFAULT_PARAMS.minLength;
    const maxLen = Number(maxLength) || DEFAULT_PARAMS.maxLength;
    const top = Number(topN) || DEFAULT_PARAMS.topN;

    if (minLen >= maxLen) {
      addToast({
        type: "error",
        message: "最小词长应小于最大词长",
      });
      return;
    }

    setExtracting(true);
    try {
      // 后端按"段落"列表接收,把粘贴内容按空行切段以提升频次统计准确度
      const docs = text
        .split(/\n{2,}/)
        .map((s) => s.trim())
        .filter(Boolean);
      const data = await apiClient.post<CandidateTerm[]>(
        "/api/admin/dictionaries/candidates/extract",
        {
          documents_content: docs.length > 0 ? docs : [text],
          min_frequency: minFreq,
          min_length: minLen,
          max_length: maxLen,
          top_n: top,
        }
      );
      setCandidates(Array.isArray(data) ? data : []);
      addToast({
        type: "success",
        message: `已提取 ${data.length} 个候选术语`,
      });
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({
          type: "error",
          message: "提取失败",
          description: err.message,
        });
      }
    } finally {
      setExtracting(false);
    }
  };

  const handleAdd = async (word: string) => {
    setAdding(word);
    try {
      await apiClient.post(`/api/admin/dictionaries/${dictId}/terms`, {
        terms: [{ word, pos: null, weight: 1.0 }],
      });
      addToast({ type: "success", message: `已加入「${word}」` });
      setCandidates((prev) => prev.filter((c) => c.word !== word));
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({
          type: "error",
          message: "加入失败",
          description: err.message,
        });
      }
    } finally {
      setAdding(null);
    }
  };

  const handleIgnore = (word: string) => {
    setIgnored((prev) => {
      const next = new Set(prev);
      next.add(word);
      saveIgnored(dictId, next);
      return next;
    });
    addToast({ type: "success", message: `已忽略「${word}」` });
  };

  const handleUnignore = (word: string) => {
    setIgnored((prev) => {
      const next = new Set(prev);
      next.delete(word);
      saveIgnored(dictId, next);
      return next;
    });
  };

  const handleBulkAdd = async () => {
    const visible = candidates.filter((c) => !ignored.has(c.word));
    if (visible.length === 0) return;

    setBulkAdding(true);
    try {
      await apiClient.post(`/api/admin/dictionaries/${dictId}/terms`, {
        terms: visible.map((c) => ({
          word: c.word,
          pos: null,
          weight: 1.0,
        })),
      });
      addToast({
        type: "success",
        message: `已批量加入 ${visible.length} 个术语`,
      });
      const visibleSet = new Set(visible.map((c) => c.word));
      setCandidates((prev) => prev.filter((c) => !visibleSet.has(c.word)));
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({
          type: "error",
          message: "批量加入失败",
          description: err.message,
        });
      }
    } finally {
      setBulkAdding(false);
    }
  };

  const handleClearIgnored = () => {
    setIgnored(new Set());
    saveIgnored(dictId, new Set());
    addToast({ type: "success", message: "已清空忽略列表" });
  };

  const visibleCandidates = showIgnored
    ? candidates
    : candidates.filter((c) => !ignored.has(c.word));
  const hiddenCount = candidates.length - visibleCandidates.length;

  if (loadingMeta) {
    return (
      <div className="text-center py-12 text-muted-foreground">加载中...</div>
    );
  }

  if (!dictionary) {
    return (
      <div className="space-y-4">
        <Link href="/admin/dictionaries">
          <Button variant="ghost" size="sm">
            <ChevronLeft className="mr-1 h-4 w-4" />
            返回列表
          </Button>
        </Link>
        <div className="text-center py-12 text-muted-foreground">词典不存在</div>
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-4xl">
      <div className="flex items-center justify-between gap-2">
        <Link href={`/admin/dictionaries/${dictId}`}>
          <Button variant="ghost" size="sm">
            <ChevronLeft className="mr-1 h-4 w-4" />
            返回词典
          </Button>
        </Link>
      </div>

      <div>
        <div className="flex items-center gap-2">
          <Sparkles className="h-5 w-5 text-amber-500" />
          <h1 className="text-2xl font-bold">候选术语审核</h1>
        </div>
        <p className="text-muted-foreground mt-1 text-sm">
          基于词频统计从文档样本中抽取未识别的候选术语,审核后一键加入「
          {dictionary.name}」词典。
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">提取候选术语</CardTitle>
          <CardDescription>
            粘贴一段或多段文档正文(用空行分段),系统将按词频统计推荐候选术语。
            已存在于启用词典的术语会自动过滤。
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="sample-text">文档样本</Label>
            <Textarea
              id="sample-text"
              rows={8}
              value={sampleText}
              onChange={(e) => setSampleText(e.target.value)}
              placeholder="将文档正文(可多段,用空行分隔)粘贴到此处..."
              className="font-mono text-sm"
            />
            <p className="text-xs text-muted-foreground">
              当前长度:{sampleText.length} 字符
            </p>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="space-y-1">
              <Label htmlFor="min-freq" className="text-xs">
                最小频次
              </Label>
              <Input
                id="min-freq"
                type="number"
                min={1}
                value={minFrequency}
                onChange={(e) => setMinFrequency(e.target.value)}
              />
            </div>
            <div className="space-y-1">
              <Label htmlFor="min-len" className="text-xs">
                最小词长
              </Label>
              <Input
                id="min-len"
                type="number"
                min={1}
                value={minLength}
                onChange={(e) => setMinLength(e.target.value)}
              />
            </div>
            <div className="space-y-1">
              <Label htmlFor="max-len" className="text-xs">
                最大词长
              </Label>
              <Input
                id="max-len"
                type="number"
                min={2}
                value={maxLength}
                onChange={(e) => setMaxLength(e.target.value)}
              />
            </div>
            <div className="space-y-1">
              <Label htmlFor="top-n" className="text-xs">
                返回数量
              </Label>
              <Input
                id="top-n"
                type="number"
                min={1}
                max={500}
                value={topN}
                onChange={(e) => setTopN(e.target.value)}
              />
            </div>
          </div>

          <div className="flex justify-end">
            <Button onClick={handleExtract} disabled={extracting}>
              <Sparkles className="mr-1 h-4 w-4" />
              {extracting ? "提取中..." : "提取候选术语"}
            </Button>
          </div>
        </CardContent>
      </Card>

      {candidates.length > 0 && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between gap-2">
              <div>
                <CardTitle className="text-base">
                  候选术语 ({visibleCandidates.length}
                  {hiddenCount > 0 ? ` / ${candidates.length}` : ""})
                </CardTitle>
                <CardDescription>
                  按出现频次降序,点击「加入」直接写入词典,点击「忽略」仅本地隐藏
                </CardDescription>
              </div>
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowIgnored((v) => !v)}
                  disabled={ignored.size === 0}
                  title={
                    ignored.size === 0 ? "没有已忽略的候选术语" : undefined
                  }
                >
                  {showIgnored ? (
                    <>
                      <EyeOff className="mr-1 h-4 w-4" />
                      隐藏已忽略
                    </>
                  ) : (
                    <>
                      <Eye className="mr-1 h-4 w-4" />
                      显示已忽略 ({ignored.size})
                    </>
                  )}
                </Button>
                <Button
                  size="sm"
                  onClick={handleBulkAdd}
                  disabled={bulkAdding || visibleCandidates.length === 0}
                >
                  <Plus className="mr-1 h-4 w-4" />
                  {bulkAdding
                    ? "批量加入中..."
                    : `全部加入 (${visibleCandidates.length})`}
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            {visibleCandidates.length === 0 ? (
              <div className="text-center py-6 text-sm text-muted-foreground">
                {hiddenCount > 0
                  ? "全部候选已被忽略,可点击右上角查看"
                  : "暂无候选术语"}
              </div>
            ) : (
              <div className="border rounded-md max-h-[480px] overflow-y-auto">
                <table className="w-full text-sm">
                  <thead className="bg-muted/50 sticky top-0">
                    <tr>
                      <th className="text-left px-3 py-2 font-medium w-12">
                        #
                      </th>
                      <th className="text-left px-3 py-2 font-medium">术语</th>
                      <th className="text-left px-3 py-2 font-medium">频次</th>
                      <th className="text-left px-3 py-2 font-medium">状态</th>
                      <th className="text-right px-3 py-2 font-medium">
                        操作
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y">
                    {visibleCandidates.map((c, idx) => {
                      const isIgnored = ignored.has(c.word);
                      const isAdding = adding === c.word;
                      return (
                        <tr
                          key={c.word}
                          className={`hover:bg-muted/30 ${
                            isIgnored ? "opacity-50" : ""
                          }`}
                        >
                          <td className="px-3 py-2 text-muted-foreground">
                            {idx + 1}
                          </td>
                          <td className="px-3 py-2 font-medium">{c.word}</td>
                          <td className="px-3 py-2 text-muted-foreground">
                            {c.frequency}
                          </td>
                          <td className="px-3 py-2">
                            {isIgnored ? (
                              <span className="inline-flex items-center rounded-full bg-muted px-2 py-0.5 text-xs">
                                已忽略
                              </span>
                            ) : (
                              <span className="inline-flex items-center rounded-full bg-amber-100 dark:bg-amber-900/40 text-amber-800 dark:text-amber-300 px-2 py-0.5 text-xs">
                                待审核
                              </span>
                            )}
                          </td>
                          <td className="px-3 py-2 text-right">
                            <div className="flex items-center justify-end gap-1">
                              {isIgnored ? (
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  onClick={() => handleUnignore(c.word)}
                                >
                                  恢复
                                </Button>
                              ) : (
                                <>
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => handleIgnore(c.word)}
                                    disabled={isAdding}
                                  >
                                    忽略
                                  </Button>
                                  <Button
                                    size="sm"
                                    onClick={() => handleAdd(c.word)}
                                    disabled={isAdding}
                                  >
                                    <Plus className="mr-1 h-3.5 w-3.5" />
                                    {isAdding ? "加入中..." : "加入"}
                                  </Button>
                                </>
                              )}
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}

            {ignored.size > 0 && (
              <div className="flex justify-end pt-3">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleClearIgnored}
                >
                  清空忽略列表 ({ignored.size})
                </Button>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
