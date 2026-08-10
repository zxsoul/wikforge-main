"use client";

import { MonitoringPanel } from "@/components/admin/monitoring-panel";

export default function AdminMonitoringPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">系统监控</h1>
        <p className="text-muted-foreground mt-1">
          文档处理队列状态和系统资源使用情况
        </p>
      </div>
      <MonitoringPanel />
    </div>
  );
}
