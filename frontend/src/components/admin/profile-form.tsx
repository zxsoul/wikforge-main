"use client";

import * as React from "react";
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

export interface MatchRules {
  filename_regex: string[];
  content_regex: string[];
  min_content_match_count: number;
}

export interface HeadingRule {
  pattern: string;
  level: number;
  strip_pattern: boolean;
}

export interface BoilerplateConfig {
  detection_mode: string;
  statistical_threshold: number;
  manual_patterns: string[];
}

export interface TableConfig {
  cross_page_merge: boolean;
  row_level_chunking: boolean;
  collapse_merged_cells: string;
}

export interface ChunkingConfig {
  min_tokens: number;
  max_tokens: number;
  overlap_tokens: number;
  respect_heading_level: number;
  protect_patterns: string[];
}

export interface ProfileFormValue {
  name: string;
  description: string;
  priority: number;
  enabled: boolean;
  match_rules: MatchRules;
  heading_rules: HeadingRule[];
  boilerplate: BoilerplateConfig;
  tables: TableConfig;
  chunking: ChunkingConfig;
  domain_dictionary_id: string | null;
}

export const defaultProfileValue: ProfileFormValue = {
  name: "",
  description: "",
  priority: 0,
  enabled: true,
  match_rules: {
    filename_regex: [],
    content_regex: [],
    min_content_match_count: 1,
  },
  heading_rules: [],
  boilerplate: {
    detection_mode: "statistical",
    statistical_threshold: 0.5,
    manual_patterns: [],
  },
  tables: {
    cross_page_merge: true,
    row_level_chunking: false,
    collapse_merged_cells: "describe",
  },
  chunking: {
    min_tokens: 256,
    max_tokens: 800,
    overlap_tokens: 80,
    respect_heading_level: 1,
    protect_patterns: [],
  },
  domain_dictionary_id: null,
};

interface ProfileFormProps {
  value: ProfileFormValue;
  onChange: (value: ProfileFormValue) => void;
  onSubmit: (changeNote?: string) => void | Promise<void>;
  submitLabel: string;
  showChangeNote?: boolean;
  submitting?: boolean;
}

function linesToArray(text: string): string[] {
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
}

export function ProfileForm({
  value,
  onChange,
  onSubmit,
  submitLabel,
  showChangeNote = false,
  submitting = false,
}: ProfileFormProps) {
  const [changeNote, setChangeNote] = React.useState("");

  const update = <K extends keyof ProfileFormValue>(
    key: K,
    next: ProfileFormValue[K]
  ) => {
    onChange({ ...value, [key]: next });
  };

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        void onSubmit(showChangeNote ? changeNote || undefined : undefined);
      }}
      className="space-y-6"
    >
      {/* 基本信息 */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">基本信息</CardTitle>
          <CardDescription>Profile 名称、描述、优先级和启用状态</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="name">名称 *</Label>
            <Input
              id="name"
              required
              maxLength={100}
              value={value.name}
              onChange={(e) => update("name", e.target.value)}
              placeholder="例如:chinese-technical-spec"
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="description">描述</Label>
            <Textarea
              id="description"
              rows={2}
              value={value.description}
              onChange={(e) => update("description", e.target.value)}
              placeholder="该 Profile 适用的文档类型与场景"
            />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="priority">优先级</Label>
              <Input
                id="priority"
                type="number"
                value={value.priority}
                onChange={(e) =>
                  update("priority", Number(e.target.value) || 0)
                }
              />
              <p className="text-xs text-muted-foreground">
                数值越大越优先匹配
              </p>
            </div>
            <div className="space-y-2">
              <Label htmlFor="enabled">启用状态</Label>
              <div className="flex items-center gap-2 pt-2">
                <input
                  id="enabled"
                  type="checkbox"
                  checked={value.enabled}
                  onChange={(e) => update("enabled", e.target.checked)}
                  className="h-4 w-4"
                />
                <span className="text-sm">{value.enabled ? "启用" : "禁用"}</span>
              </div>
            </div>
          </div>
          <div className="space-y-2">
            <Label htmlFor="domain_dictionary_id">关联词典 ID</Label>
            <Input
              id="domain_dictionary_id"
              value={value.domain_dictionary_id ?? ""}
              onChange={(e) =>
                update("domain_dictionary_id", e.target.value || null)
              }
              placeholder="可选,留空表示不关联词典"
            />
          </div>
        </CardContent>
      </Card>

      {/* 匹配规则 */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">匹配规则</CardTitle>
          <CardDescription>
            通过文件名和内容正则匹配文档,每行一个表达式
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="filename_regex">文件名正则</Label>
            <Textarea
              id="filename_regex"
              rows={3}
              value={value.match_rules.filename_regex.join("\n")}
              onChange={(e) =>
                update("match_rules", {
                  ...value.match_rules,
                  filename_regex: linesToArray(e.target.value),
                })
              }
              placeholder=".*技术规范.*&#10;.*spec.*\\.pdf"
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="content_regex">内容正则</Label>
            <Textarea
              id="content_regex"
              rows={3}
              value={value.match_rules.content_regex.join("\n")}
              onChange={(e) =>
                update("match_rules", {
                  ...value.match_rules,
                  content_regex: linesToArray(e.target.value),
                })
              }
              placeholder="^第[一二三四五六七八九十]+章"
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="min_content_match_count">最小内容命中数</Label>
            <Input
              id="min_content_match_count"
              type="number"
              min={0}
              value={value.match_rules.min_content_match_count}
              onChange={(e) =>
                update("match_rules", {
                  ...value.match_rules,
                  min_content_match_count: Number(e.target.value) || 0,
                })
              }
            />
          </div>
        </CardContent>
      </Card>

      {/* 标题规则 */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">标题规则</CardTitle>
          <CardDescription>
            按正则识别标题层级,可添加多条规则,数字越小层级越高
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {value.heading_rules.map((rule, idx) => (
            <div
              key={idx}
              className="grid grid-cols-12 gap-2 items-start border rounded-md p-3"
            >
              <div className="col-span-7 space-y-1">
                <Label className="text-xs">正则模式</Label>
                <Input
                  value={rule.pattern}
                  onChange={(e) => {
                    const next = [...value.heading_rules];
                    next[idx] = { ...rule, pattern: e.target.value };
                    update("heading_rules", next);
                  }}
                />
              </div>
              <div className="col-span-2 space-y-1">
                <Label className="text-xs">层级</Label>
                <Input
                  type="number"
                  min={1}
                  max={6}
                  value={rule.level}
                  onChange={(e) => {
                    const next = [...value.heading_rules];
                    next[idx] = {
                      ...rule,
                      level: Number(e.target.value) || 1,
                    };
                    update("heading_rules", next);
                  }}
                />
              </div>
              <div className="col-span-2 space-y-1">
                <Label className="text-xs">剥离模式</Label>
                <div className="flex items-center pt-2">
                  <input
                    type="checkbox"
                    checked={rule.strip_pattern}
                    onChange={(e) => {
                      const next = [...value.heading_rules];
                      next[idx] = {
                        ...rule,
                        strip_pattern: e.target.checked,
                      };
                      update("heading_rules", next);
                    }}
                    className="h-4 w-4"
                  />
                </div>
              </div>
              <div className="col-span-1 flex items-end justify-end">
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() =>
                    update(
                      "heading_rules",
                      value.heading_rules.filter((_, i) => i !== idx)
                    )
                  }
                >
                  删除
                </Button>
              </div>
            </div>
          ))}
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() =>
              update("heading_rules", [
                ...value.heading_rules,
                { pattern: "", level: 1, strip_pattern: false },
              ])
            }
          >
            添加规则
          </Button>
        </CardContent>
      </Card>

      {/* 噪声模式 */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">噪声模式</CardTitle>
          <CardDescription>
            页眉页脚水印检测,statistical 自动统计,manual 使用正则
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="detection_mode">检测模式</Label>
              <select
                id="detection_mode"
                value={value.boilerplate.detection_mode}
                onChange={(e) =>
                  update("boilerplate", {
                    ...value.boilerplate,
                    detection_mode: e.target.value,
                  })
                }
                className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              >
                <option value="statistical">统计学检测</option>
                <option value="manual">手动正则</option>
                <option value="hybrid">混合模式</option>
              </select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="statistical_threshold">统计阈值</Label>
              <Input
                id="statistical_threshold"
                type="number"
                step={0.05}
                min={0}
                max={1}
                value={value.boilerplate.statistical_threshold}
                onChange={(e) =>
                  update("boilerplate", {
                    ...value.boilerplate,
                    statistical_threshold: Number(e.target.value) || 0,
                  })
                }
              />
              <p className="text-xs text-muted-foreground">
                同位置同文本出现频率超过该阈值视为噪声
              </p>
            </div>
          </div>
          <div className="space-y-2">
            <Label htmlFor="manual_patterns">手动正则模式</Label>
            <Textarea
              id="manual_patterns"
              rows={3}
              value={value.boilerplate.manual_patterns.join("\n")}
              onChange={(e) =>
                update("boilerplate", {
                  ...value.boilerplate,
                  manual_patterns: linesToArray(e.target.value),
                })
              }
              placeholder="第\\s*\\d+\\s*页&#10;CONFIDENTIAL"
            />
          </div>
        </CardContent>
      </Card>

      {/* 表格策略 */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">表格策略</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-3 gap-4">
            <div className="space-y-2">
              <Label htmlFor="cross_page_merge" className="block">
                跨页合并
              </Label>
              <input
                id="cross_page_merge"
                type="checkbox"
                checked={value.tables.cross_page_merge}
                onChange={(e) =>
                  update("tables", {
                    ...value.tables,
                    cross_page_merge: e.target.checked,
                  })
                }
                className="h-4 w-4"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="row_level_chunking" className="block">
                按行分块
              </Label>
              <input
                id="row_level_chunking"
                type="checkbox"
                checked={value.tables.row_level_chunking}
                onChange={(e) =>
                  update("tables", {
                    ...value.tables,
                    row_level_chunking: e.target.checked,
                  })
                }
                className="h-4 w-4"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="collapse_merged_cells">合并单元格</Label>
              <select
                id="collapse_merged_cells"
                value={value.tables.collapse_merged_cells}
                onChange={(e) =>
                  update("tables", {
                    ...value.tables,
                    collapse_merged_cells: e.target.value,
                  })
                }
                className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              >
                <option value="describe">文本化描述</option>
                <option value="repeat">重复填充</option>
                <option value="ignore">忽略</option>
              </select>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 分块参数 */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">分块参数</CardTitle>
          <CardDescription>Token 数量按 tiktoken 计算</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="min_tokens">最小 Token 数</Label>
              <Input
                id="min_tokens"
                type="number"
                min={1}
                value={value.chunking.min_tokens}
                onChange={(e) =>
                  update("chunking", {
                    ...value.chunking,
                    min_tokens: Number(e.target.value) || 1,
                  })
                }
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="max_tokens">最大 Token 数</Label>
              <Input
                id="max_tokens"
                type="number"
                min={1}
                value={value.chunking.max_tokens}
                onChange={(e) =>
                  update("chunking", {
                    ...value.chunking,
                    max_tokens: Number(e.target.value) || 1,
                  })
                }
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="overlap_tokens">重叠 Token 数</Label>
              <Input
                id="overlap_tokens"
                type="number"
                min={0}
                value={value.chunking.overlap_tokens}
                onChange={(e) =>
                  update("chunking", {
                    ...value.chunking,
                    overlap_tokens: Number(e.target.value) || 0,
                  })
                }
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="respect_heading_level">尊重标题层级</Label>
              <Input
                id="respect_heading_level"
                type="number"
                min={1}
                max={6}
                value={value.chunking.respect_heading_level}
                onChange={(e) =>
                  update("chunking", {
                    ...value.chunking,
                    respect_heading_level: Number(e.target.value) || 1,
                  })
                }
              />
              <p className="text-xs text-muted-foreground">
                分块边界不会跨越该层级及以上的标题
              </p>
            </div>
          </div>
          <div className="space-y-2">
            <Label htmlFor="protect_patterns">保护模式(原子性正则)</Label>
            <Textarea
              id="protect_patterns"
              rows={3}
              value={value.chunking.protect_patterns.join("\n")}
              onChange={(e) =>
                update("chunking", {
                  ...value.chunking,
                  protect_patterns: linesToArray(e.target.value),
                })
              }
              placeholder="\\$[^$]+\\$&#10;\\d+(\\.\\d+)?\\s*(MPa|kg|mm)"
            />
          </div>
        </CardContent>
      </Card>

      {/* 变更说明 */}
      {showChangeNote && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">变更说明</CardTitle>
            <CardDescription>
              将记录到版本历史,便于后续追溯
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Textarea
              rows={2}
              value={changeNote}
              onChange={(e) => setChangeNote(e.target.value)}
              placeholder="例如:调整中式技术规范的章节正则"
            />
          </CardContent>
        </Card>
      )}

      <div className="flex justify-end gap-2">
        <Button type="submit" disabled={submitting}>
          {submitting ? "保存中..." : submitLabel}
        </Button>
      </div>
    </form>
  );
}
