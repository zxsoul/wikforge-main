# Frontend E2E Tests (任务 25.9)

基于 Playwright 的前端端到端测试，覆盖核心用户流程：登录 → 上传 → 搜索 → 问答。

## 安装

```bash
npm install -D @playwright/test
npx playwright install chromium
```

## 运行

```bash
# 启动后端（另一个终端）
docker compose up -d
cd backend && uvicorn app.main:app

# 运行 E2E
cd frontend
npm run e2e
```

或一次性跑所有：

```bash
npx playwright test
```

## 调试

```bash
# 打开 UI 模式
npx playwright test --ui

# 单条用例
npx playwright test e2e/critical-path.spec.ts

# 失败保留 trace
npx playwright show-trace e2e-results/artifacts/.../trace.zip
```

## 环境变量

- `PLAYWRIGHT_BASE_URL`：覆盖 baseURL（默认 http://localhost:3000）
- `PLAYWRIGHT_SKIP_WEBSERVER=1`：跳过自动启动 dev server
- `E2E_USERNAME` / `E2E_PASSWORD`：登录凭据（默认 e2e@wikforge.local / e2e-pass-123）

## 文件结构

```
frontend/
├── playwright.config.ts       # Playwright 配置
├── e2e/
│   ├── README.md             # 本文档
│   ├── fixtures.ts           # 共享 fixture（登录态、API mock）
│   └── critical-path.spec.ts # 关键路径：登录 → 上传 → 搜索 → 问答
└── e2e-results/              # 运行产物（gitignore）
```
