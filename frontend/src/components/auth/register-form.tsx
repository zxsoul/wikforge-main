"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Eye, EyeOff, Loader2, Check, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { apiClient, ApiClientError } from "@/lib/api-client";
import { cn } from "@/lib/utils";

interface PasswordStrength {
  hasUppercase: boolean;
  hasLowercase: boolean;
  hasNumber: boolean;
  hasSpecial: boolean;
  hasMinLength: boolean;
  hasMaxLength: boolean;
  categoriesCount: number;
  isValid: boolean;
}

function checkPasswordStrength(password: string): PasswordStrength {
  const hasUppercase = /[A-Z]/.test(password);
  const hasLowercase = /[a-z]/.test(password);
  const hasNumber = /[0-9]/.test(password);
  const hasSpecial = /[^A-Za-z0-9]/.test(password);
  const hasMinLength = password.length >= 8;
  const hasMaxLength = password.length <= 64;

  const categoriesCount = [hasUppercase, hasLowercase, hasNumber, hasSpecial].filter(Boolean).length;
  const isValid = hasMinLength && hasMaxLength && categoriesCount >= 3;

  return {
    hasUppercase,
    hasLowercase,
    hasNumber,
    hasSpecial,
    hasMinLength,
    hasMaxLength,
    categoriesCount,
    isValid,
  };
}

function PasswordStrengthIndicator({ password }: { password: string }) {
  const strength = checkPasswordStrength(password);

  if (!password) return null;

  const rules = [
    { label: "8-64 个字符", met: strength.hasMinLength && strength.hasMaxLength },
    { label: "大写字母", met: strength.hasUppercase },
    { label: "小写字母", met: strength.hasLowercase },
    { label: "数字", met: strength.hasNumber },
    { label: "特殊字符", met: strength.hasSpecial },
  ];

  const strengthLevel = strength.categoriesCount;
  const strengthLabel =
    strengthLevel <= 1 ? "弱" : strengthLevel === 2 ? "一般" : strengthLevel === 3 ? "强" : "很强";
  const strengthColor =
    strengthLevel <= 1
      ? "bg-destructive"
      : strengthLevel === 2
        ? "bg-yellow-500"
        : strengthLevel === 3
          ? "bg-green-500"
          : "bg-green-600";

  return (
    <div className="space-y-2 mt-2">
      {/* Strength bar */}
      <div className="flex gap-1">
        {[1, 2, 3, 4].map((level) => (
          <div
            key={level}
            className={cn(
              "h-1.5 flex-1 rounded-full transition-colors",
              level <= strengthLevel ? strengthColor : "bg-muted"
            )}
          />
        ))}
      </div>
      <p className="text-xs text-muted-foreground">
        密码强度：{strengthLabel}（需满足至少 3 类字符）
      </p>

      {/* Rules checklist */}
      <ul className="space-y-1">
        {rules.map((rule) => (
          <li key={rule.label} className="flex items-center gap-1.5 text-xs">
            {rule.met ? (
              <Check className="h-3 w-3 text-green-500" />
            ) : (
              <X className="h-3 w-3 text-muted-foreground" />
            )}
            <span className={rule.met ? "text-green-600 dark:text-green-400" : "text-muted-foreground"}>
              {rule.label}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function RegisterForm() {
  const router = useRouter();

  const [email, setEmail] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const passwordStrength = checkPasswordStrength(password);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    if (!email || !password || !confirmPassword) {
      setError("请填写所有必填字段");
      return;
    }

    if (password !== confirmPassword) {
      setError("两次输入的密码不一致");
      return;
    }

    if (!passwordStrength.isValid) {
      setError("密码不满足复杂度要求：8-64 个字符，须包含至少三类字符");
      return;
    }

    setIsLoading(true);
    try {
      await apiClient.post(
        "/api/auth/register",
        { email, password, display_name: displayName || email.split("@")[0] },
        { skipAuth: true }
      );
      // Registration successful, redirect to login
      router.push("/login?registered=true");
    } catch (err) {
      if (err instanceof ApiClientError) {
        const detail = err.detail as { detail?: string } | undefined;
        if (err.status === 409) {
          setError("该邮箱已被注册");
        } else if (err.status === 422) {
          setError(detail?.detail || "输入信息格式不正确");
        } else {
          setError(detail?.detail || "注册失败，请稍后重试");
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
        <Label htmlFor="display-name">显示名称（可选）</Label>
        <Input
          id="display-name"
          type="text"
          placeholder="您的名称"
          value={displayName}
          onChange={(e) => setDisplayName(e.target.value)}
          disabled={isLoading}
          autoComplete="name"
        />
      </div>

      <div className="space-y-2">
        <Label htmlFor="password">密码</Label>
        <div className="relative">
          <Input
            id="password"
            type={showPassword ? "text" : "password"}
            placeholder="设置密码"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            disabled={isLoading}
            autoComplete="new-password"
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
        <PasswordStrengthIndicator password={password} />
      </div>

      <div className="space-y-2">
        <Label htmlFor="confirm-password">确认密码</Label>
        <div className="relative">
          <Input
            id="confirm-password"
            type={showConfirmPassword ? "text" : "password"}
            placeholder="再次输入密码"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            disabled={isLoading}
            autoComplete="new-password"
            required
          />
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="absolute right-0 top-0 h-10 w-10 text-muted-foreground hover:text-foreground"
            onClick={() => setShowConfirmPassword(!showConfirmPassword)}
            tabIndex={-1}
            aria-label={showConfirmPassword ? "隐藏密码" : "显示密码"}
          >
            {showConfirmPassword ? (
              <EyeOff className="h-4 w-4" />
            ) : (
              <Eye className="h-4 w-4" />
            )}
          </Button>
        </div>
        {confirmPassword && password !== confirmPassword && (
          <p className="text-xs text-destructive">两次输入的密码不一致</p>
        )}
      </div>

      <Button type="submit" className="w-full" disabled={isLoading}>
        {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
        注册
      </Button>

      <p className="text-center text-sm text-muted-foreground">
        已有账号？{" "}
        <Link href="/login" className="text-primary underline-offset-4 hover:underline">
          登录
        </Link>
      </p>
    </form>
  );
}
