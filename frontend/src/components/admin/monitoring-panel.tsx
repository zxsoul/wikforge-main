"use client";

import * as React from "react";
import {
  Clock,
  Loader2,
  CheckCircle,
  XCircle,
  Cpu,
  HardDrive,
  MemoryStick,
  RefreshCw,
} from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useToast } from "@/components/ui/toast";
import { apiClient, ApiClientError } from "@/lib/api-client";

interface QueueStatus {
  pending: number;
  processing: number;
  completed: number;
  failed: number;
}

interface ResourceUsage {
  cpu_percent: number;
  memory_percent: number;
  storage_percent: number;
  memory_used_gb: number;
  memory_total_gb: number;
  storage_used_gb: number;
  storage_total_gb: number;
}

interface MonitoringData {
  queue: QueueStatus;
  resources: ResourceUsage;
  updated_at: string;
}

export function MonitoringPanel() {
  const { addToast } = useToast();
  const [data, setData] = React.useState<MonitoringData | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [lastRefresh, setLastRefresh] = React.useState<Date>(new Date());

  const fetchMonitoring = React.useCallback(async () => {
    try {
      const result = await apiClient.get<MonitoringData>(
        "/api/admin/monitoring"
      );
      setData(result);
      setLastRefresh(new Date());
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({
          type: "error",
          message: "加载监控数据失败",
          description: err.message,
        });
      }
    } finally {
      setLoading(false);
    }
  }, [addToast]);

  React.useEffect(() => {
    fetchMonitoring();

    // Auto-refresh every 30 seconds
    const interval = setInterval(fetchMonitoring, 30000);
    return () => clearInterval(interval);
  }, [fetchMonitoring]);

  const handleManualRefresh = () => {
    setLoading(true);
    fetchMonitoring();
  };

  if (loading && !data) {
    return (
      <div className="text-center py-12 text-muted-foreground">加载中...</div>
    );
  }

  const queue = data?.queue || { pending: 0, processing: 0, completed: 0, failed: 0 };
  const resources = data?.resources || {
    cpu_percent: 0,
    memory_percent: 0,
    storage_percent: 0,
    memory_used_gb: 0,
    memory_total_gb: 0,
    storage_used_gb: 0,
    storage_total_gb: 0,
  };

  return (
    <div className="space-y-6">
      {/* Refresh indicator */}
      <div className="flex items-center justify-between">
        <p className="text-sm text-muted-foreground">
          上次刷新: {lastRefresh.toLocaleTimeString("zh-CN")}
          <span className="ml-2">（每 30 秒自动刷新）</span>
        </p>
        <Button variant="outline" size="sm" onClick={handleManualRefresh}>
          <RefreshCw className="mr-1 h-3 w-3" />
          手动刷新
        </Button>
      </div>

      {/* Queue Status */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">文档处理队列</CardTitle>
          <CardDescription>当前文档处理任务状态</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-4">
            <QueueCard
              icon={<Clock className="h-5 w-5 text-yellow-500" />}
              label="待处理"
              count={queue.pending}
              color="text-yellow-600"
            />
            <QueueCard
              icon={<Loader2 className="h-5 w-5 text-blue-500 animate-spin" />}
              label="处理中"
              count={queue.processing}
              color="text-blue-600"
            />
            <QueueCard
              icon={<CheckCircle className="h-5 w-5 text-green-500" />}
              label="已完成"
              count={queue.completed}
              color="text-green-600"
            />
            <QueueCard
              icon={<XCircle className="h-5 w-5 text-red-500" />}
              label="失败"
              count={queue.failed}
              color="text-red-600"
            />
          </div>
        </CardContent>
      </Card>

      {/* Resource Usage */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">系统资源</CardTitle>
          <CardDescription>服务器资源使用情况</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-6 md:grid-cols-3">
            <ResourceCard
              icon={<Cpu className="h-5 w-5" />}
              label="CPU 使用率"
              percent={resources.cpu_percent}
              detail={`${resources.cpu_percent.toFixed(1)}%`}
            />
            <ResourceCard
              icon={<MemoryStick className="h-5 w-5" />}
              label="内存使用率"
              percent={resources.memory_percent}
              detail={`${resources.memory_used_gb.toFixed(1)} / ${resources.memory_total_gb.toFixed(1)} GB`}
            />
            <ResourceCard
              icon={<HardDrive className="h-5 w-5" />}
              label="存储使用率"
              percent={resources.storage_percent}
              detail={`${resources.storage_used_gb.toFixed(1)} / ${resources.storage_total_gb.toFixed(1)} GB`}
            />
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function QueueCard({
  icon,
  label,
  count,
  color,
}: {
  icon: React.ReactNode;
  label: string;
  count: number;
  color: string;
}) {
  return (
    <div className="flex items-center gap-3 rounded-lg border p-4">
      {icon}
      <div>
        <p className="text-sm text-muted-foreground">{label}</p>
        <p className={`text-2xl font-bold ${color}`}>{count}</p>
      </div>
    </div>
  );
}

function ResourceCard({
  icon,
  label,
  percent,
  detail,
}: {
  icon: React.ReactNode;
  label: string;
  percent: number;
  detail: string;
}) {
  const getBarColor = (p: number) => {
    if (p >= 90) return "bg-red-500";
    if (p >= 70) return "bg-yellow-500";
    return "bg-green-500";
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        {icon}
        <span className="text-sm font-medium">{label}</span>
      </div>
      <div className="space-y-1">
        <div className="flex justify-between text-sm">
          <span className="text-muted-foreground">{detail}</span>
          <span className="font-medium">{percent.toFixed(1)}%</span>
        </div>
        <div className="h-2 w-full rounded-full bg-muted">
          <div
            className={`h-full rounded-full transition-all ${getBarColor(percent)}`}
            style={{ width: `${Math.min(100, percent)}%` }}
          />
        </div>
      </div>
    </div>
  );
}
