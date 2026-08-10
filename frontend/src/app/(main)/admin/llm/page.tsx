"use client";

import { ExternalLink } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

/**
 * 模型配置 / LLM Gateway 入口。
 *
 * Wikforge 把多模型路由、API Key、限流、配额、调用日志等能力
 * 全部委托给独立部署的 LiteLLM Proxy (compose 服务名: litellm)。
 * 该服务自带 Admin UI, 比在 Wikforge 里重做一份更稳, 因此这里只
 * 提供入口跳转, 不再实现独立的模型管理界面。
 */
export default function AdminLLMPage() {
  // 通过 NEXT_PUBLIC_LITELLM_UI_URL 覆盖, 默认指向同主机的 4000 端口
  const litellmUrl =
    process.env.NEXT_PUBLIC_LITELLM_UI_URL ||
    (typeof window !== "undefined"
      ? `${window.location.protocol}//${window.location.hostname}:4000/ui`
      : "http://localhost:4000/ui");

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">模型配置</h1>
        <p className="text-muted-foreground mt-1">
          通过 LiteLLM Proxy Admin UI 管理模型与 API Key
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>LiteLLM Admin UI</CardTitle>
          <CardDescription>
            Wikforge 后端通过 LiteLLM Proxy 统一调用各家大模型。请在 LiteLLM
            自带的管理界面配置:
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <ul className="list-disc list-inside text-sm text-muted-foreground space-y-1">
            <li>添加 / 删除模型 (gpt / claude / qwen / glm / deepseek 等)</li>
            <li>配置上游 API Key 与 base URL</li>
            <li>设置 Fallback 链, 单上游故障自动切备用</li>
            <li>查看调用日志、Token 用量、Cost Tracking</li>
            <li>生成虚拟 Key, 给团队不同成员分发</li>
          </ul>
          <div className="flex items-center gap-3 pt-2">
            <Button asChild>
              <a
                href={litellmUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2"
              >
                打开 LiteLLM Admin UI
                <ExternalLink className="h-4 w-4" />
              </a>
            </Button>
            <span className="text-xs text-muted-foreground break-all">
              {litellmUrl}
            </span>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>当前 Wikforge 使用的模型</CardTitle>
          <CardDescription>
            来自后端 .env 配置, 修改后需重启 wikforge-api / wikforge-worker
            容器
          </CardDescription>
        </CardHeader>
        <CardContent>
          <dl className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div>
              <dt className="text-muted-foreground">Chat 模型</dt>
              <dd className="font-mono">LITELLM_MODEL (默认 gpt-5.5)</dd>
            </div>
            <div>
              <dt className="text-muted-foreground">Embedding 模型</dt>
              <dd className="font-mono">
                EMBEDDING_MODEL (默认 text-embedding-v4)
              </dd>
            </div>
            <div>
              <dt className="text-muted-foreground">Vision 模型 (PDF 兜底)</dt>
              <dd className="font-mono">
                UNIVERSAL_PARSER_VISION_MODEL (留空走 LITELLM_MODEL)
              </dd>
            </div>
            <div>
              <dt className="text-muted-foreground">查询增强</dt>
              <dd className="font-mono">
                QUERY_ENHANCEMENT_ENABLE_REWRITE / HYDE / DECOMPOSITION
              </dd>
            </div>
          </dl>
        </CardContent>
      </Card>
    </div>
  );
}
