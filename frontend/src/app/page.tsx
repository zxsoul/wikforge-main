import Link from "next/link";
import { WikforgeLogo } from "@/components/brand/logo";

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center gap-6 p-24">
      <WikforgeLogo size={80} />
      <h1 className="text-4xl font-bold tracking-tight">Wikforge</h1>
      <p className="text-muted-foreground">企业级知识库系统</p>
      <div className="flex gap-3">
        <Link
          href="/login"
          className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90"
        >
          登录
        </Link>
        <Link
          href="/register"
          className="rounded-md border border-input bg-background px-4 py-2 text-sm font-medium hover:bg-accent hover:text-accent-foreground"
        >
          注册
        </Link>
        <Link
          href="/dashboard"
          className="rounded-md border border-input bg-background px-4 py-2 text-sm font-medium hover:bg-accent hover:text-accent-foreground"
        >
          进入应用
        </Link>
      </div>
    </main>
  );
}
