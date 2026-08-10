"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Eye, EyeOff, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { OidcButton } from "@/components/auth/oidc-button";
import { useAuthStore } from "@/stores/auth-store";
import { ApiClientError } from "@/lib/api-client";

export function LoginForm() {
  const router = useRouter();
  const login = useAuthStore((state) => state.login);

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [lockoutMessage, setLockoutMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLockoutMessage("");

    if (!email || !password) {
      setError("请输入邮箱和密码");
      return;
    }

    setIsLoading(true);
    try {
      await login(email, password);
      router.push("/dashboard");
    } catch (err) {
      if (err instanceof ApiClientError) {
        const detail = err.detail as { detail?: string; locked_until?: string; remaining_seconds?: number } | undefined;
        if (err.status === 423 || detail?.locked_until) {
          const remaining = detail?.remaining_seconds;
          if (remaining) {
            const minutes = Math.ceil(remaining / 60);
            setLockoutMessage(`账号已被临时锁定，请 ${minutes} 分钟后重试`);
          } else {
            setLockoutMessage("账号已被临时锁定，请稍后重试");
          }
        } else if (err.status === 401) {
          setError("邮箱或密码错误");
        } else {
          setError(detail?.detail || "登录失败，请稍后重试");
        }
      } else {
        setError("网络错误，请检查连接后重试");
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {error && (
        <div className="rounded-md bg-destructive/10 p-3 text-sm text-destructive">
          {error}
        </div>
      )}

      {lockoutMessage && (
        <div className="rounded-md bg-yellow-500/10 border border-yellow-500/20 p-3 text-sm text-yellow-600 dark:text-yellow-400">
          {lockoutMessage}
        </div>
      )}

      <div className="space-y-2">
        <Label htmlFor="email">邮箱</Label>
        <Input
          id="email"
          type="email"
          placeholder="name@example.com"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          disabled={isLoading}
          autoComplete="email"
          required
        />
      </div>

      <div className="space-y-2">
        <Label htmlFor="password">密码</Label>
        <div className="relative">
          <Input
            id="password"
            type={showPassword ? "text" : "password"}
            placeholder="输入密码"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            disabled={isLoading}
            autoComplete="current-password"
            required
          />
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="absolute right-0 top-0 h-10 w-10 text-muted-foreground hover:text-foreground"
            onClick={() => setShowPassword(!showPassword)}
            tabIndex={-1}
            aria-label={showPassword ? "隐藏密码" : "显示密码"}
          >
            {showPassword ? (
              <EyeOff className="h-4 w-4" />
            ) : (
              <Eye className="h-4 w-4" />
            )}
          </Button>
        </div>
      </div>

      <Button type="submit" className="w-full" disabled={isLoading}>
        {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
        登录
      </Button>

      <div className="relative my-4">
        <div className="absolute inset-0 flex items-center">
          <span className="w-full border-t" />
        </div>
        <div className="relative flex justify-center text-xs uppercase">
          <span className="bg-card px-2 text-muted-foreground">或</span>
        </div>
      </div>

      <OidcButton />

      <p className="text-center text-sm text-muted-foreground">
        还没有账号？{" "}
        <Link href="/register" className="text-primary underline-offset-4 hover:underline">
          注册
        </Link>
      </p>
    </form>
  );
}
