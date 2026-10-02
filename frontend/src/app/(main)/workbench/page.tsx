"use client";

/**
 * 员工端 · 今日工作台
 *
 * 员工日常只需把截图 / 日志 / 文档拖进「文档管理」上传，本页负责：
 * 1. 一键生成当日工作日报（AI 聚合当天全部已入库材料合成）；
 * 2. 查看今日日报内容（四段式 Markdown 渲染）；
 * 3. 浏览自己的历史日报（工作留痕）。
 *
 * 日报每天在 18:47 也会由系统定时任务自动生成，手动生成只是为了
 * 「马上看到结果」。
 */

import { useCallback, useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  CalendarDays,
  FileText,
  Loader2,
  RefreshCw,
  Sparkles,
  Upload,
} from "lucide-react";
import Link from "next/link";

import { apiClient } from "@/lib/api-client";
import { reportsApi, type DailyReport } from "@/lib/reports-api";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { useToast } from "@/components/ui/toast";

interface Space {
  id: string;
  name: string;
  description?: string | null;
}

const STATUS_LABEL: Record<DailyReport["status"], string> = {
  generating: "生成中",
  completed: "已完成",
  failed: "生成失败",
};

export default function WorkbenchPage() {
  const [spaces, setSpaces] = useState<Space[]>([]);
  const [spaceId, setSpaceId] = useState<string>("");
  const [todayReport, setTodayReport] = useState<DailyReport | null>(null);
  const [history, setHistory] = useState<DailyReport[]>([]);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const { addToast } = useToast();

  // 加载空间列表，默认选中第一个
  useEffect(() => {
    (async () => {
      try {
        const data = await apiClient.get<Space[]>("/api/spaces");
        setSpaces(data);
        if (data.length > 0) setSpaceId(data[0].id);
      } catch {
        addToast({ type: "error", message: "加载空间列表失败" });
      }
    })();
  }, []);

  const loadReports = useCallback(
    async (sid: string) => {
      if (!sid) return;
      setLoading(true);
      try {
        const { reports } = await reportsApi.listMine(sid, 0, 30);
        const today = new Date().toISOString().slice(0, 10);
        setTodayReport(
          reports.find((r) => r.report_date === today) ?? null
        );
        setHistory(reports.filter((r) => r.report_date !== today));
      } catch {
        addToast({ type: "error", message: "加载日报失败" });
      } finally {
        setLoading(false);
      }
    },
    [addToast]
  );

  useEffect(() => {
    loadReports(spaceId);
  }, [spaceId, loadReports]);

  const handleGenerate = async (force: boolean) => {
    if (!spaceId) return;
    setGenerating(true);
    try {
      const report = await reportsApi.generate(spaceId, force);
      setTodayReport(report);
      addToast({
        type: "success",
        message: force ? "日报已重新生成" : "日报生成成功",
        description: "日报已自动入库，主管可以随时检索到",
      });
      loadReports(spaceId);
    } catch (err) {
      addToast({
        type: "error",
        message: "日报生成失败",
        description:
          err instanceof Error ? err.message : "请确认今天已上传工作材料",
      });
    } finally {
      setGenerating(false);
    }
  };

  return (
    <div className="mx-auto max-w-4xl space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">今日工作台</h1>
          <p className="text-sm text-muted-foreground">
            上传工作材料，AI 自动为你生成当日工作日报
          </p>
        </div>
        {spaces.length > 1 && (
          <select
            className="rounded-md border bg-background px-3 py-2 text-sm"
            value={spaceId}
            onChange={(e) => setSpaceId(e.target.value)}
          >
            {spaces.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name}
              </option>
            ))}
          </select>
        )}
      </div>

      {/* 今日日报 */}
      <Card className="p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <CalendarDays className="h-5 w-5 text-primary" />
            <h2 className="text-lg font-semibold">
              今日日报 · {new Date().toLocaleDateString("zh-CN")}
            </h2>
            {todayReport && (
              <span
                className={
                  todayReport.status === "completed"
                    ? "rounded-full bg-green-100 px-2 py-0.5 text-xs text-green-700"
                    : todayReport.status === "failed"
                      ? "rounded-full bg-red-100 px-2 py-0.5 text-xs text-red-700"
                      : "rounded-full bg-yellow-100 px-2 py-0.5 text-xs text-yellow-700"
                }
              >
                {STATUS_LABEL[todayReport.status]}
              </span>
            )}
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" asChild>
              <Link href="/documents">
                <Upload className="mr-1 h-4 w-4" />
                去上传材料
              </Link>
            </Button>
            {todayReport ? (
              <Button
                size="sm"
                variant="outline"
                disabled={generating}
                onClick={() => handleGenerate(true)}
              >
                {generating ? (
                  <Loader2 className="mr-1 h-4 w-4 animate-spin" />
                ) : (
                  <RefreshCw className="mr-1 h-4 w-4" />
                )}
                重新生成
              </Button>
            ) : (
              <Button
                size="sm"
                disabled={generating || !spaceId}
                onClick={() => handleGenerate(false)}
              >
                {generating ? (
                  <Loader2 className="mr-1 h-4 w-4 animate-spin" />
                ) : (
                  <Sparkles className="mr-1 h-4 w-4" />
                )}
                生成今日日报
              </Button>
            )}
          </div>
        </div>

        <Separator className="my-4" />

        {loading ? (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" /> 加载中…
          </div>
        ) : todayReport?.status === "completed" ? (
          <article className="prose prose-sm max-w-none dark:prose-invert">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {todayReport.content}
            </ReactMarkdown>
          </article>
        ) : todayReport?.status === "failed" ? (
          <p className="text-sm text-destructive">
            生成失败：{todayReport.error_detail || "未知错误"}
            ，可点击「重新生成」重试。
          </p>
        ) : (
          <p className="text-sm text-muted-foreground">
            今天还没有日报。把截图、日志或文档上传到团队空间后，点击
            「生成今日日报」即可。每天 18:47 系统也会自动生成。
          </p>
        )}
      </Card>

      {/* 历史日报 */}
      <Card className="p-6">
        <div className="flex items-center gap-2">
          <FileText className="h-5 w-5 text-primary" />
          <h2 className="text-lg font-semibold">历史日报</h2>
        </div>
        <Separator className="my-4" />
        {history.length === 0 ? (
          <p className="text-sm text-muted-foreground">暂无历史日报</p>
        ) : (
          <ul className="divide-y">
            {history.map((r) => (
              <li
                key={r.id}
                className="flex items-center justify-between py-2 text-sm"
              >
                <span className="font-medium">{r.report_date}</span>
                <span
                  className={
                    r.status === "completed"
                      ? "text-green-600"
                      : r.status === "failed"
                        ? "text-destructive"
                        : "text-muted-foreground"
                  }
                >
                  {STATUS_LABEL[r.status]}
                </span>
              </li>
            ))}
          </ul>
        )}
      </Card>
    </div>
  );
}
