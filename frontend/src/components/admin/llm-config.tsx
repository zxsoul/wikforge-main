"use client";

import * as React from "react";
import { Plus, Trash2, Power, PowerOff } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
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
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useToast } from "@/components/ui/toast";
import { apiClient, ApiClientError } from "@/lib/api-client";

interface LLMModel {
  id: string;
  name: string;
  provider: string;
  model_id: string;
  is_active: boolean;
}

interface ApiKey {
  id: string;
  name: string;
  key_masked: string;
  provider: string;
  is_enabled: boolean;
  created_at: string;
}

interface ModelParams {
  temperature: number;
  max_tokens: number;
  top_p: number;
}

/**
 * Masks an API key to show only the last 4 characters.
 * Sub-task 6: API Key 掩码显示
 */
function maskApiKey(key: string): string {
  if (key.length <= 4) return key;
  return "•".repeat(key.length - 4) + key.slice(-4);
}

export function LLMConfig() {
  const { addToast } = useToast();
  const [models, setModels] = React.useState<LLMModel[]>([]);
  const [apiKeys, setApiKeys] = React.useState<ApiKey[]>([]);
  const [params, setParams] = React.useState<ModelParams>({
    temperature: 0.7,
    max_tokens: 4096,
    top_p: 0.9,
  });
  const [selectedModel, setSelectedModel] = React.useState<string>("");
  const [loading, setLoading] = React.useState(true);
  const [addKeyDialogOpen, setAddKeyDialogOpen] = React.useState(false);
  const [newKeyName, setNewKeyName] = React.useState("");
  const [newKeyValue, setNewKeyValue] = React.useState("");
  const [newKeyProvider, setNewKeyProvider] = React.useState("openai");

  const fetchConfig = React.useCallback(async () => {
    try {
      setLoading(true);
      const [modelsData, keysData, paramsData] = await Promise.all([
        apiClient.get<{ items: LLMModel[] }>("/api/admin/llm/models"),
        apiClient.get<{ items: ApiKey[] }>("/api/admin/llm/api-keys"),
        apiClient.get<ModelParams>("/api/admin/llm/params"),
      ]);
      setModels(modelsData.items || []);
      setApiKeys(keysData.items || []);
      if (paramsData) setParams(paramsData);
      const activeModel = (modelsData.items || []).find((m) => m.is_active);
      if (activeModel) setSelectedModel(activeModel.id);
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({ type: "error", message: "加载 LLM 配置失败", description: err.message });
      }
    } finally {
      setLoading(false);
    }
  }, [addToast]);

  React.useEffect(() => {
    fetchConfig();
  }, [fetchConfig]);

  const handleModelSelect = async (modelId: string) => {
    try {
      await apiClient.put("/api/admin/llm/models/active", { model_id: modelId });
      setSelectedModel(modelId);
      setModels((prev) =>
        prev.map((m) => ({ ...m, is_active: m.id === modelId }))
      );
      addToast({ type: "success", message: "模型切换成功" });
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({ type: "error", message: "切换模型失败", description: err.message });
      }
    }
  };

  const handleAddKey = async () => {
    try {
      await apiClient.post("/api/admin/llm/api-keys", {
        name: newKeyName,
        key: newKeyValue,
        provider: newKeyProvider,
      });
      addToast({ type: "success", message: "API Key 添加成功" });
      setAddKeyDialogOpen(false);
      setNewKeyName("");
      setNewKeyValue("");
      fetchConfig();
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({ type: "error", message: "添加 API Key 失败", description: err.message });
      }
    }
  };

  const handleDeleteKey = async (keyId: string) => {
    try {
      await apiClient.delete(`/api/admin/llm/api-keys/${keyId}`);
      setApiKeys((prev) => prev.filter((k) => k.id !== keyId));
      addToast({ type: "success", message: "API Key 删除成功" });
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({ type: "error", message: "删除 API Key 失败", description: err.message });
      }
    }
  };

  const handleToggleKey = async (keyId: string, enabled: boolean) => {
    try {
      await apiClient.put(`/api/admin/llm/api-keys/${keyId}`, {
        is_enabled: enabled,
      });
      setApiKeys((prev) =>
        prev.map((k) => (k.id === keyId ? { ...k, is_enabled: enabled } : k))
      );
      addToast({
        type: "success",
        message: enabled ? "API Key 已启用" : "API Key 已禁用",
      });
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({ type: "error", message: "操作失败", description: err.message });
      }
    }
  };

  const handleParamsChange = async (newParams: ModelParams) => {
    setParams(newParams);
  };

  const handleSaveParams = async () => {
    try {
      await apiClient.put("/api/admin/llm/params", params);
      addToast({ type: "success", message: "参数保存成功" });
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({ type: "error", message: "保存参数失败", description: err.message });
      }
    }
  };

  if (loading) {
    return (
      <div className="text-center py-12 text-muted-foreground">加载中...</div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Model Selection */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">模型选择</CardTitle>
          <CardDescription>选择当前使用的 LLM 模型</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
            {models.map((model) => (
              <div
                key={model.id}
                className={`cursor-pointer rounded-lg border p-4 transition-colors ${
                  selectedModel === model.id
                    ? "border-primary bg-primary/5"
                    : "hover:border-primary/50"
                }`}
                onClick={() => handleModelSelect(model.id)}
              >
                <p className="font-medium text-sm">{model.name}</p>
                <p className="text-xs text-muted-foreground mt-1">
                  {model.provider} / {model.model_id}
                </p>
                {selectedModel === model.id && (
                  <span className="inline-block mt-2 text-xs text-primary font-medium">
                    ✓ 当前使用
                  </span>
                )}
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* API Key Management (sub-task 6: masked display) */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-base">API Key 管理</CardTitle>
              <CardDescription>
                管理各模型提供商的 API Key（仅显示最后 4 位）
              </CardDescription>
            </div>
            <Button size="sm" onClick={() => setAddKeyDialogOpen(true)}>
              <Plus className="mr-1 h-3 w-3" />
              新增
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {apiKeys.length === 0 ? (
            <p className="text-sm text-muted-foreground text-center py-4">
              暂无 API Key，点击上方按钮添加
            </p>
          ) : (
            <div className="space-y-3">
              {apiKeys.map((key) => (
                <div
                  key={key.id}
                  className="flex items-center justify-between rounded-md border p-3"
                >
                  <div className="flex items-center gap-3">
                    <div
                      className={`h-2 w-2 rounded-full ${
                        key.is_enabled ? "bg-green-500" : "bg-gray-300"
                      }`}
                    />
                    <div>
                      <p className="text-sm font-medium">{key.name}</p>
                      <p className="text-xs text-muted-foreground font-mono">
                        {key.key_masked || maskApiKey("placeholder1234")}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-muted-foreground">
                      {key.provider}
                    </span>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-8 w-8 p-0"
                      onClick={() => handleToggleKey(key.id, !key.is_enabled)}
                      title={key.is_enabled ? "禁用" : "启用"}
                    >
                      {key.is_enabled ? (
                        <Power className="h-4 w-4 text-green-600" />
                      ) : (
                        <PowerOff className="h-4 w-4 text-muted-foreground" />
                      )}
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-8 w-8 p-0 text-destructive hover:text-destructive"
                      onClick={() => handleDeleteKey(key.id)}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Model Parameters */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-base">模型参数</CardTitle>
              <CardDescription>调整 LLM 生成参数</CardDescription>
            </div>
            <Button size="sm" onClick={handleSaveParams}>
              保存参数
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid gap-6 md:grid-cols-3">
            {/* Temperature */}
            <div className="space-y-2">
              <Label>
                Temperature:{" "}
                <span className="font-mono text-primary">
                  {params.temperature}
                </span>
              </Label>
              <input
                type="range"
                min="0"
                max="2"
                step="0.1"
                value={params.temperature}
                onChange={(e) =>
                  handleParamsChange({
                    ...params,
                    temperature: parseFloat(e.target.value),
                  })
                }
                className="w-full"
              />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>0（精确）</span>
                <span>2（创意）</span>
              </div>
            </div>

            {/* Max Tokens */}
            <div className="space-y-2">
              <Label htmlFor="max-tokens">
                Max Tokens:{" "}
                <span className="font-mono text-primary">
                  {params.max_tokens}
                </span>
              </Label>
              <Input
                id="max-tokens"
                type="number"
                min={1}
                max={128000}
                value={params.max_tokens}
                onChange={(e) =>
                  handleParamsChange({
                    ...params,
                    max_tokens: Math.min(
                      128000,
                      Math.max(1, parseInt(e.target.value) || 1)
                    ),
                  })
                }
              />
              <p className="text-xs text-muted-foreground">范围: 1 - 128000</p>
            </div>

            {/* Top P */}
            <div className="space-y-2">
              <Label>
                Top P:{" "}
                <span className="font-mono text-primary">{params.top_p}</span>
              </Label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={params.top_p}
                onChange={(e) =>
                  handleParamsChange({
                    ...params,
                    top_p: parseFloat(e.target.value),
                  })
                }
                className="w-full"
              />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>0</span>
                <span>1</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Add API Key Dialog */}
      <Dialog open={addKeyDialogOpen} onOpenChange={setAddKeyDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>新增 API Key</DialogTitle>
            <DialogDescription>
              添加模型提供商的 API Key
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="key-name">名称</Label>
              <Input
                id="key-name"
                placeholder="例如：OpenAI Production"
                value={newKeyName}
                onChange={(e) => setNewKeyName(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="key-provider">提供商</Label>
              <select
                id="key-provider"
                className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                value={newKeyProvider}
                onChange={(e) => setNewKeyProvider(e.target.value)}
              >
                <option value="openai">OpenAI</option>
                <option value="anthropic">Anthropic (Claude)</option>
                <option value="dashscope">通义千问</option>
                <option value="ollama">Ollama</option>
              </select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="key-value">API Key</Label>
              <Input
                id="key-value"
                type="password"
                placeholder="输入 API Key"
                value={newKeyValue}
                onChange={(e) => setNewKeyValue(e.target.value)}
              />
            </div>
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setAddKeyDialogOpen(false)}
            >
              取消
            </Button>
            <Button
              onClick={handleAddKey}
              disabled={!newKeyName.trim() || !newKeyValue.trim()}
            >
              添加
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
