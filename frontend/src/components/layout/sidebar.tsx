"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  FileText,
  Search,
  MessageSquare,
  Settings,
  FolderTree,
  LayoutDashboard,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
  TooltipProvider,
} from "@/components/ui/tooltip";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { useSidebarStore } from "@/stores/sidebar-store";
import { WikforgeLogo } from "@/components/brand/logo";

const navItems = [
  {
    title: "概览",
    href: "/dashboard",
    icon: LayoutDashboard,
  },
  {
    title: "文档管理",
    href: "/documents",
    icon: FileText,
  },
  {
    title: "知识空间",
    href: "/spaces",
    icon: FolderTree,
  },
  {
    title: "搜索",
    href: "/search",
    icon: Search,
  },
  {
    title: "AI 问答",
    href: "/chat",
    icon: MessageSquare,
  },
];

const adminItems = [
  {
    title: "系统管理",
    href: "/admin",
    icon: Settings,
  },
];

export function Sidebar() {
  const pathname = usePathname();
  const { isCollapsed, toggle } = useSidebarStore();

  return (
    <TooltipProvider delayDuration={0}>
      <aside
        className={cn(
          "relative flex h-full flex-col border-r bg-sidebar transition-all duration-300",
          isCollapsed ? "w-16" : "w-64"
        )}
      >
        {/* Logo */}
        <div className="flex h-14 items-center border-b px-4">
          {isCollapsed ? (
            <Link href="/" className="mx-auto" aria-label="Wikforge">
              <WikforgeLogo size={28} />
            </Link>
          ) : (
            <Link href="/" className="flex items-center gap-2">
              <WikforgeLogo size={28} />
              <span className="text-lg font-bold tracking-tight text-sidebar-foreground">
                Wikforge
              </span>
            </Link>
          )}
          <Button
            variant="ghost"
            size="icon"
            className={cn(
              "h-8 w-8 text-sidebar-foreground",
              isCollapsed ? "mx-auto" : "ml-auto"
            )}
            onClick={toggle}
            aria-label={isCollapsed ? "展开侧边栏" : "收起侧边栏"}
          >
            {isCollapsed ? (
              <ChevronRight className="h-4 w-4" />
            ) : (
              <ChevronLeft className="h-4 w-4" />
            )}
          </Button>
        </div>

        {/* Navigation */}
        <ScrollArea className="flex-1 px-2 py-4">
          <nav className="flex flex-col gap-1">
            {navItems.map((item) => {
              const isActive = pathname.startsWith(item.href);
              const NavLink = (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                    isActive
                      ? "bg-sidebar-accent text-sidebar-accent-foreground"
                      : "text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
                    isCollapsed && "justify-center px-2"
                  )}
                >
                  <item.icon className="h-4 w-4 shrink-0" />
                  {!isCollapsed && <span>{item.title}</span>}
                </Link>
              );

              if (isCollapsed) {
                return (
                  <Tooltip key={item.href}>
                    <TooltipTrigger asChild>{NavLink}</TooltipTrigger>
                    <TooltipContent side="right">
                      {item.title}
                    </TooltipContent>
                  </Tooltip>
                );
              }

              return NavLink;
            })}

            <Separator className="my-2" />

            {adminItems.map((item) => {
              const isActive = pathname.startsWith(item.href);
              const NavLink = (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                    isActive
                      ? "bg-sidebar-accent text-sidebar-accent-foreground"
                      : "text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
                    isCollapsed && "justify-center px-2"
                  )}
                >
                  <item.icon className="h-4 w-4 shrink-0" />
                  {!isCollapsed && <span>{item.title}</span>}
                </Link>
              );

              if (isCollapsed) {
                return (
                  <Tooltip key={item.href}>
                    <TooltipTrigger asChild>{NavLink}</TooltipTrigger>
                    <TooltipContent side="right">
                      {item.title}
                    </TooltipContent>
                  </Tooltip>
                );
              }

              return NavLink;
            })}
          </nav>
        </ScrollArea>
      </aside>
    </TooltipProvider>
  );
}
