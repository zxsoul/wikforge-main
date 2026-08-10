"use client";

import Link from "next/link";
import {
  FolderTree,
  Users,
  Shield,
  Brain,
  Activity,
  FileCog,
  BookOpen,
  ClipboardCheck,
  Sparkles,
  MessageSquareWarning,
} from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

const adminModules = [
  {
    title: "空间管理",
    description: "创建、编辑、删除空间，管理成员和角色",
    href: "/admin/spaces",
    icon: FolderTree,
  },
  {
    title: "用户管理",
    description: "用户列表、搜索、角色分配",
    href: "/admin/users",
    icon: Users,
  },
  {
    title: "权限配置",
    description: "按角色配置功能模块访问权限",
    href: "/admin/permissions",
    icon: Shield,
  },
  {
    title: "模型配置",
    description: "LLM 模型选择、API Key 管理、参数调整",
    href: "/admin/llm",
    icon: Brain,
  },
  {
    title: "Profile 管理",
    description: "文档解析策略、匹配规则、版本历史",
    href: "/admin/profiles",
    icon: FileCog,
  },
  {
    title: "候选 Profile 审核",
    description: "审核 LLM 生成的候选 Profile，可编辑后批准或拒绝",
    href: "/admin/profiles/candidates",
    icon: Sparkles,
  },
  {
    title: "审核队列",
    description: "解析质量低于阈值的文档审核与修正",
    href: "/admin/reviews",
    icon: ClipboardCheck,
  },
  {
    title: "词典管理",
    description: "领域术语、同义词组、停用词与 IK 同步",
    href: "/admin/dictionaries",
    icon: BookOpen,
  },
  {
    title: "反馈分析",
    description: "聚合反馈、识别错误模式、一键应用优化建议、跟踪重处理进度",
    href: "/admin/feedback",
    icon: MessageSquareWarning,
  },
  {
    title: "系统监控",
    description: "队列状态、资源使用率实时监控",
    href: "/admin/monitoring",
    icon: Activity,
  },
];

export default function AdminPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">系统管理</h1>
        <p className="text-muted-foreground mt-1">
          管理系统配置、用户和权限
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {adminModules.map((mod) => (
          <Link key={mod.href} href={mod.href}>
            <Card className="h-full transition-colors hover:border-primary/50 hover:shadow-md cursor-pointer">
              <CardHeader>
                <div className="flex items-center gap-3">
                  <div className="rounded-lg bg-primary/10 p-2">
                    <mod.icon className="h-5 w-5 text-primary" />
                  </div>
                  <CardTitle className="text-base">{mod.title}</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <CardDescription>{mod.description}</CardDescription>
              </CardContent>
            </Card>
          </Link>
        ))}
      </div>
    </div>
  );
}
