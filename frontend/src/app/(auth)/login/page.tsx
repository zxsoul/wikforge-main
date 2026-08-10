"use client";

import { Suspense } from "react";
import { useSearchParams } from "next/navigation";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { LoginForm } from "@/components/auth/login-form";
import { WikforgeLogo } from "@/components/brand/logo";

function LoginContent() {
  const searchParams = useSearchParams();
  const registered = searchParams.get("registered");

  return (
    <Card>
      <CardHeader className="text-center space-y-3">
        <div className="flex justify-center">
          <WikforgeLogo size={48} />
        </div>
        <CardTitle className="text-2xl">登录 Wikforge</CardTitle>
        <CardDescription>输入您的账号信息以访问知识库</CardDescription>
      </CardHeader>
      <CardContent>
        {registered && (
          <div className="mb-4 rounded-md bg-green-500/10 border border-green-500/20 p-3 text-sm text-green-600 dark:text-green-400">
            注册成功！请使用您的账号登录。
          </div>
        )}
        <LoginForm />
      </CardContent>
    </Card>
  );
}

export default function LoginPage() {
  return (
    <Suspense>
      <LoginContent />
    </Suspense>
  );
}
