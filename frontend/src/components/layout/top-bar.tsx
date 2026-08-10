"use client";

import { Search, Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import { Button } from "@/components/ui/button";
import { useSearchStore } from "@/stores/search-store";
import { UserMenu } from "@/components/auth/user-menu";

export function TopBar() {
  const { resolvedTheme, setTheme } = useTheme();
  const openSearch = useSearchStore((state) => state.open);

  return (
    <header className="flex h-14 items-center justify-between border-b bg-background px-6">
      {/* Search trigger */}
      <Button
        variant="outline"
        className="relative h-9 w-full max-w-sm justify-start text-sm text-muted-foreground"
        onClick={openSearch}
      >
        <Search className="mr-2 h-4 w-4" />
        <span>搜索知识库...</span>
        <kbd className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 select-none rounded border bg-muted px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground">
          ⌘K
        </kbd>
      </Button>

      {/* Actions */}
      <div className="flex items-center gap-2">
        <Button
          variant="ghost"
          size="icon"
          className="relative"
          onClick={() => setTheme(resolvedTheme === "dark" ? "light" : "dark")}
          aria-label="切换主题"
        >
          <Sun className="h-4 w-4 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
          <Moon className="absolute h-4 w-4 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
        </Button>

        {/* User info + logout */}
        <UserMenu />
      </div>
    </header>
  );
}
