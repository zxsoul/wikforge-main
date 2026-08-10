import { cn } from "@/lib/utils";

/**
 * Wikforge 品牌 Logo (SVG)。
 *
 * 设计语言:
 * - 蓝紫色块 + 白色 W 字母, 与默认 shadcn 中性色拉开区别
 * - 圆角 (rounded-lg) 风格统一卡片设计
 * - 自适应大小, 通过 ``size`` prop 控制 (单位: px)
 *
 * 也作为 favicon 的 SVG 源, 通过 src/app/icon.tsx 渲染成 PNG/ICO。
 */
export function WikforgeLogo({
  size = 32,
  className,
  withText = false,
}: {
  size?: number;
  className?: string;
  withText?: boolean;
}) {
  return (
    <div className={cn("inline-flex items-center gap-2", className)}>
      <svg
        width={size}
        height={size}
        viewBox="0 0 64 64"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        aria-hidden="true"
      >
        <defs>
          <linearGradient id="wf-grad" x1="0" y1="0" x2="64" y2="64" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#4F46E5" />
            <stop offset="100%" stopColor="#7C3AED" />
          </linearGradient>
        </defs>
        {/* 圆角方块底 */}
        <rect width="64" height="64" rx="14" fill="url(#wf-grad)" />
        {/* 字母 W: 通过两条 V 形 path 组成,粗描边 + 圆角端点 */}
        <path
          d="M14 20 L22 46 L32 28 L42 46 L50 20"
          stroke="white"
          strokeWidth="5"
          strokeLinecap="round"
          strokeLinejoin="round"
          fill="none"
        />
      </svg>
      {withText && (
        <span className="text-lg font-bold tracking-tight">Wikforge</span>
      )}
    </div>
  );
}
