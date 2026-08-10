/**
 * Playwright 配置（任务 25.9）。
 *
 * 用法:
 *   npm install -D @playwright/test
 *   npx playwright install chromium
 *   npx playwright test
 *
 * 设计要点:
 * - 默认本地运行 dev server（http://localhost:3000），由 webServer 自动启动
 * - 仅启用 Chromium，覆盖核心用户流程；移动端 viewport 通过 project 切换
 * - CI 模式下自动重试 2 次以容忍网络抖动
 * - 测试文件放在 e2e/ 目录，artifacts（截图/trace/视频）保存到 e2e-results/
 */

import { defineConfig, devices } from '@playwright/test';

const PORT = Number(process.env.PORT ?? 3000);
const BASE_URL = process.env.PLAYWRIGHT_BASE_URL ?? `http://localhost:${PORT}`;

export default defineConfig({
  testDir: './e2e',
  // 单测超时 30s（网络慢时可调）
  timeout: 30_000,
  // 全套超时 5 分钟
  globalTimeout: 5 * 60 * 1000,
  expect: {
    timeout: 5_000,
  },
  // CI 中失败时重试一次，本地不重试
  retries: process.env.CI ? 2 : 0,
  // CI 模式只跑一次
  workers: process.env.CI ? 1 : undefined,
  reporter: process.env.CI
    ? [['github'], ['html', { outputFolder: 'e2e-results/html', open: 'never' }]]
    : [['list'], ['html', { outputFolder: 'e2e-results/html', open: 'never' }]],
  outputDir: 'e2e-results/artifacts',
  use: {
    baseURL: BASE_URL,
    headless: true,
    // 失败时保留 trace / 截图，便于排错
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
    actionTimeout: 10_000,
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
  // 自动启动 dev server（避免开发者忘记开）
  webServer: process.env.PLAYWRIGHT_SKIP_WEBSERVER
    ? undefined
    : {
        command: 'npm run dev',
        url: BASE_URL,
        reuseExistingServer: !process.env.CI,
        timeout: 120_000,
      },
});
