"use client";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

/**
 * 模型配置说明页。
 *
 * 2026-09 架构调整：Wikforge 后端已移除 LiteLLM Proxy 中间层，
 * 改为通过 openai SDK 直连 OpenAI 兼容端点（CPA 网关 / 阿里百炼）。
 * 模型与 API Key 全部通过后端 .env 配置，本页展示当前生效的配置项
 * 及修改方式，不再提供独立的模型管理界面。
 */
export default function AdminLLMPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">模型配置</h1>
        <p className="text-muted-foreground mt-1">
          后端直连 OpenAI 兼容 API，配置通过 .env 管理
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>直连架构说明</CardTitle>
          <CardDescription>
            Wikforge 后端通过 openai SDK 直接调用上游模型服务，不经过任何
            中间代理层。
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <ul className="list-disc list-inside text-sm text-muted-foreground space-y-1">
            <li>Chat 模型：走 CHAT_API_BASE 指向的 OpenAI 兼容端点（默认 CPA 网关）</li>
            <li>Embedding 模型：走 EMBEDDING_API_BASE 指向的端点（默认阿里百炼）</li>
            <li>Rerank 精排：直连阿里 DashScope gte-rerank-v2（独立端点）</li>
            <li>任何 OpenAI 兼容服务（OpenAI / DeepSeek / 通义 / Kimi 等）均可接入</li>
          </ul>
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
              <dd className="font-mono">CHAT_MODEL (默认 gpt-5.5)</dd>
            </div>
            <div>
              <dt className="text-muted-foreground">Chat 上游</dt>
              <dd className="font-mono">CHAT_API_BASE / CHAT_API_KEY</dd>
            </div>
            <div>
              <dt className="text-muted-foreground">Embedding 模型</dt>
              <dd className="font-mono">
                EMBEDDING_MODEL (默认 text-embedding-v4)
              </dd>
            </div>
            <div>
              <dt className="text-muted-foreground">Embedding 上游</dt>
              <dd className="font-mono">
                EMBEDDING_API_BASE / EMBEDDING_API_KEY (默认阿里百炼)
              </dd>
            </div>
            <div>
              <dt className="text-muted-foreground">Vision 模型 (PDF 兜底)</dt>
              <dd className="font-mono">
                UNIVERSAL_PARSER_VISION_MODEL (留空走 CHAT_MODEL)
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
