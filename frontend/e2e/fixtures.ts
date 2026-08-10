/**
 * 共享 fixture：登录态、API mock（任务 25.9）。
 *
 * E2E 不依赖真实后端：通过 ``page.route`` 拦截 API 请求，
 * 让前端在 Playwright 环境下也能稳定运行关键路径。
 */

import { test as base, expect, Page } from '@playwright/test';

export type Fixtures = {
  /** 登录后的 Page（已注入 token，并完成跳转） */
  loggedInPage: Page;
};

/** 默认凭据（CI 与本地都用这套） */
export const DEFAULT_USER = {
  email: process.env.E2E_USERNAME ?? 'e2e@wikforge.local',
  password: process.env.E2E_PASSWORD ?? 'e2e-pass-123',
};

/** 模拟成功登录响应 */
export async function mockAuth(page: Page) {
  await page.route('**/api/auth/login', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        access_token: 'e2e-access-token',
        refresh_token: 'e2e-refresh-token',
        token_type: 'bearer',
        expires_in: 1800,
        user: {
          id: 'e2e-user-id',
          email: DEFAULT_USER.email,
          display_name: 'E2E User',
        },
      }),
    });
  });
}

/** 模拟搜索 API */
export async function mockSearch(page: Page) {
  await page.route('**/api/search', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        results: [
          {
            chunk_id: 'c-e2e-1',
            document_id: 'doc-e2e-1',
            chunk_index: 0,
            title_chain: 'E2E 测试文档',
            source_file: 'e2e.pdf',
            page_number: 1,
            score: 0.92,
            highlight: 'E2E 测试通过 <mark>关键词</mark> 命中。',
          },
        ],
        total: 1,
        page: 1,
        page_size: 10,
      }),
    });
  });
}

/** 模拟文档上传 */
export async function mockUpload(page: Page) {
  await page.route('**/api/documents/upload', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        documents: [
          {
            id: 'doc-e2e-uploaded',
            title: 'e2e_sample.txt',
            status: 'pending',
            file_type: 'txt',
            progress_percent: 0,
          },
        ],
      }),
    });
  });
}

/** 模拟 RAG 流式响应（SSE） */
export async function mockRAG(page: Page) {
  await page.route('**/api/rag/chat', async (route) => {
    const body = [
      'data: {"type":"token","content":"E2E"}\n\n',
      'data: {"type":"token","content":" 答案"}\n\n',
      'data: {"type":"token","content":" [1]"}\n\n',
      'data: {"type":"done","content":"E2E 答案 [1]"}\n\n',
    ].join('');
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body,
    });
  });
}

/** 完整 mock：一次性把上述 API 全装上 */
export async function mockAllAPIs(page: Page) {
  await mockAuth(page);
  await mockSearch(page);
  await mockUpload(page);
  await mockRAG(page);
}

/** 扩展 test：``loggedInPage`` 自动完成登录 */
export const test = base.extend<Fixtures>({
  loggedInPage: async ({ page }, use) => {
    await mockAllAPIs(page);
    // 直接通过 localStorage 注入 token，跳过 UI 登录流程，避免不同实现差异
    await page.addInitScript(() => {
      localStorage.setItem('access_token', 'e2e-access-token');
      localStorage.setItem('refresh_token', 'e2e-refresh-token');
    });
    await page.goto('/');
    await use(page);
  },
});

export { expect };
