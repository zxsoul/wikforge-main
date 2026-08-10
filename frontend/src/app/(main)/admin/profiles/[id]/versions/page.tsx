"use client";

import * as React from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { ChevronLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useToast } from "@/components/ui/toast";
import { apiClient, ApiClientError } from "@/lib/api-client";

interface VersionResponse {
  id: string;
  profile_id: string;
  version: number;
  snapshot: Record<string, unknown>;
  changed_by: string;
  change_note: string | null;
  created_at: string;
}

function diffKeys(
  current: Record<string, unknown>,
  previous: Record<string, unknown>
): string[] {
  const keys = new Set<string>([
    ...Object.keys(current),
    ...Object.keys(previous),
  ]);
  const changed: string[] = [];
  keys.forEach((k) => {
    if (JSON.stringify(current[k]) !== JSON.stringify(previous[k])) {
      changed.push(k);
    }
  });
  return changed;
}

export default function ProfileVersionsPage() {
  const params = useParams<{ id: string }>();
  const profileId = params.id;
  const { addToast } = useToast();
  const [versions, setVersions] = React.useState<VersionResponse[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [diffOpen, setDiffOpen] = React.useState(false);
  const [diffPair, setDiffPair] = React.useState<{
    current: VersionResponse;
    previous: VersionResponse | null;
  } | null>(null);
  const [rollingBack, setRollingBack] = React.useState<string | null>(null);

  const fetchVersions = React.useCallback(async () => {
    try {
      setLoading(true);
      const data = await apiClient.get<VersionResponse[]>(
        `/api/admin/profiles/${profileId}/versions`
      );
      setVersions(data);
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({
          type: "error",
          message: "加载版本历史失败",
          description: err.message,
        });
      }
    } finally {
      setLoading(false);
    }
  }, [profileId, addToast]);

  React.useEffect(() => {
    void fetchVersions();
  }, [fetchVersions]);

  const handleRollback = async (version: VersionResponse) => {
    if (
      !window.confirm(
        `确认将 Profile 回滚到版本 v${version.version}?将作为新版本写入。`
      )
    ) {
      return;
    }
    setRollingBack(version.id);
    try {
      const snap = version.snapshot;
      const payload = {
        name: snap.name,
        description: snap.description,
        priority: snap.priority,
        enabled: snap.enabled,
        match_rules: snap.match_rules,
        heading_rules: snap.heading_rules,
        boilerplate: snap.boilerplate,
        tables: snap.tables,
        chunking: snap.chunking,
        domain_dictionary_id: snap.domain_dictionary_id,
        change_note: `回滚自 v${version.version}`,
      };
      await apiClient.put(`/api/admin/profiles/${profileId}`, payload);
      addToast({ type: "success", message: `已回滚到 v${version.version}` });
      void fetchVersions();
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({
          type: "error",
          message: "回滚失败",
          description: err.message,
        });
      }
    } finally {
      setRollingBack(null);
    }
  };

  const openDiff = (idx: number) => {
    const current = versions[idx];
    const previous = versions[idx + 1] ?? null;
    setDiffPair({ current, previous });
    setDiffOpen(true);
  };

  return (
    <div className="space-y-6 max-w-4xl">
      <div className="flex items-center gap-2">
        <Link href={`/admin/profiles/${profileId}`}>
          <Button variant="ghost" size="sm">
            <ChevronLeft className="mr-1 h-4 w-4" />
            返回 Profile
          </Button>
        </Link>
      </div>
      <div>
        <h1 className="text-2xl font-bold">版本历史</h1>
        <p className="text-muted-foreground mt-1">
          展示最近 20 次变更记录,可查看差异并回滚到指定版本
        </p>
      </div>

      {loading ? (
        <div className="text-center py-12 text-muted-foreground">加载中...</div>
      ) : versions.length === 0 ? (
        <div className="text-center py-12 text-muted-foreground">
          暂无版本记录
        </div>
      ) : (
        <div className="space-y-3">
          {versions.map((v, idx) => {
            const previous = versions[idx + 1] ?? null;
            const changedKeys = previous
              ? diffKeys(
                  v.snapshot,
                  (previous.snapshot ?? {}) as Record<string, unknown>
                )
              : ["initial"];
            return (
              <Card key={v.id}>
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <CardTitle className="text-base">
                        v{v.version}
                        {idx === 0 && (
                          <span className="ml-2 inline-flex items-center rounded-full bg-primary/10 text-primary px-2 py-0.5 text-xs">
                            当前
                          </span>
                        )}
                      </CardTitle>
                      <CardDescription className="mt-1">
                        {new Date(v.created_at).toLocaleString("zh-CN")}
                      </CardDescription>
                    </div>
                    <div className="flex items-center gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => openDiff(idx)}
                      >
                        查看差异
                      </Button>
                      {idx !== 0 && (
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleRollback(v)}
                          disabled={rollingBack === v.id}
                        >
                          {rollingBack === v.id ? "回滚中..." : "回滚到此版本"}
                        </Button>
                      )}
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="text-sm space-y-2">
                  {v.change_note && (
                    <div className="text-muted-foreground">
                      <span className="font-medium text-foreground">
                        说明:
                      </span>{" "}
                      {v.change_note}
                    </div>
                  )}
                  <div className="text-xs text-muted-foreground">
                    变更字段:{" "}
                    {changedKeys.length === 0
                      ? "无差异"
                      : changedKeys.join(", ")}
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}

      <Dialog open={diffOpen} onOpenChange={setDiffOpen}>
        <DialogContent className="max-w-4xl">
          <DialogHeader>
            <DialogTitle>
              版本差异
              {diffPair &&
                ` v${diffPair.current.version}${
                  diffPair.previous
                    ? ` ← v${diffPair.previous.version}`
                    : " (初始版本)"
                }`}
            </DialogTitle>
            <DialogDescription>
              左侧为旧版本快照,右侧为新版本快照
            </DialogDescription>
          </DialogHeader>
          {diffPair && (
            <div className="grid grid-cols-2 gap-4 max-h-[60vh] overflow-auto">
              <div>
                <div className="text-xs font-medium text-muted-foreground mb-1">
                  {diffPair.previous
                    ? `v${diffPair.previous.version}`
                    : "无前置版本"}
                </div>
                <pre className="text-xs bg-muted p-3 rounded-md overflow-auto whitespace-pre-wrap">
                  {diffPair.previous
                    ? JSON.stringify(diffPair.previous.snapshot, null, 2)
                    : "(初始版本)"}
                </pre>
              </div>
              <div>
                <div className="text-xs font-medium text-muted-foreground mb-1">
                  v{diffPair.current.version}
                </div>
                <pre className="text-xs bg-muted p-3 rounded-md overflow-auto whitespace-pre-wrap">
                  {JSON.stringify(diffPair.current.snapshot, null, 2)}
                </pre>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
