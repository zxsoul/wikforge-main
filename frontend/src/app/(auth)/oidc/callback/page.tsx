"use client";

import { Suspense, useCallback, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Loader2 } from "lucide-react";
import { apiClient } from "@/lib/api-client";
import { useAuthStore } from "@/stores/auth-store";

/**
 * OIDC 回调页面
 * 处理 IdP 授权后的回调，交换 code 获取 token
 */
function OidcCallbackContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const setUser = useAuthStore((state) => state.setUser);
  const [error, setError] = useState("");

  const handleCallback = useCallback(async (code: string) => {
    try {
      const data = await apiClient.get<{
        access_token: string;
        refresh_token: string;
        token_type: string;
      }>(`/api/auth/oidc/callback?code=${encodeURIComponent(code)}`, { skipAuth: true });

      apiClient.setTokens(data.access_token, data.refresh_token);

      // Fetch user info after setting tokens
      const user = await apiClient.get<{ id: string; email: string; display_name: string }>("/api/auth/me");
      setUser(user);

      router.push("/dashboard");
    } catch {
      setError("OIDC 认证失败，请重试");
    }
  }, [router, setUser]);

  useEffect(() => {
    const code = searchParams.get("code");
    const errorParam = searchParams.get("error");

    if (errorParam) {
      setError(`认证失败: ${searchParams.get("error_description") || errorParam}`);
      return;
    }

    if (!code) {
      setError("缺少授权码参数");
      return;
    }

    handleCallback(code);
  }, [searchParams, handleCallback]);

  if (error) {
    return (
      <div className="flex flex-col items-center gap-4 text-center">
        <div className="rounded-md bg-destructive/10 p-4 text-sm text-destructive">
          {error}
        </div>
        <a href="/login" className="text-sm text-primary underline-offset-4 hover:underline">
          返回登录页
        </a>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center gap-2">
      <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      <p className="text-sm text-muted-foreground">正在完成认证...</p>
    </div>
  );
}

export default function OidcCallbackPage() {
  return (
    <Suspense
      fallback={
        <div className="flex flex-col items-center gap-2">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
          <p className="text-sm text-muted-foreground">正在完成认证...</p>
        </div>
      }
    >
      <OidcCallbackContent />
    </Suspense>
  );
}
