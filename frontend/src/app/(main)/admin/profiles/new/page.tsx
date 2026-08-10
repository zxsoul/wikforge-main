"use client";

import * as React from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { ChevronLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/components/ui/toast";
import { apiClient, ApiClientError } from "@/lib/api-client";
import {
  ProfileForm,
  defaultProfileValue,
  type ProfileFormValue,
} from "@/components/admin/profile-form";

interface ProfileResponse {
  id: string;
  name: string;
}

export default function NewProfilePage() {
  const router = useRouter();
  const { addToast } = useToast();
  const [value, setValue] = React.useState<ProfileFormValue>({
    ...defaultProfileValue,
  });
  const [submitting, setSubmitting] = React.useState(false);

  const handleSubmit = async () => {
    if (!value.name.trim()) {
      addToast({ type: "error", message: "请填写 Profile 名称" });
      return;
    }
    setSubmitting(true);
    try {
      const payload = {
        name: value.name.trim(),
        description: value.description || null,
        priority: value.priority,
        enabled: value.enabled,
        match_rules: value.match_rules,
        heading_rules: value.heading_rules,
        boilerplate: value.boilerplate,
        tables: value.tables,
        chunking: value.chunking,
        domain_dictionary_id: value.domain_dictionary_id,
      };
      const created = await apiClient.post<ProfileResponse>(
        "/api/admin/profiles",
        payload
      );
      addToast({ type: "success", message: "已创建 Profile" });
      router.push(`/admin/profiles/${created.id}`);
    } catch (err) {
      if (err instanceof ApiClientError) {
        addToast({
          type: "error",
          message: "创建失败",
          description: err.message,
        });
      }
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="space-y-6 max-w-4xl">
      <div className="flex items-center gap-2">
        <Link href="/admin/profiles">
          <Button variant="ghost" size="sm">
            <ChevronLeft className="mr-1 h-4 w-4" />
            返回列表
          </Button>
        </Link>
      </div>
      <div>
        <h1 className="text-2xl font-bold">新建 Profile</h1>
        <p className="text-muted-foreground mt-1">
          配置文档解析策略,保存后可在列表中启用并应用到匹配文档
        </p>
      </div>

      <ProfileForm
        value={value}
        onChange={setValue}
        onSubmit={handleSubmit}
        submitLabel="创建 Profile"
        submitting={submitting}
      />
    </div>
  );
}
