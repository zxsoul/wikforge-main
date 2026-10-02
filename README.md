# 前言
本项目为学习项目，开源项目来自github,上传初心是认为该项目的内容非常适合刚学习完langchian的rag的人，这个项目中基本是通过手搓实现的rag，并没有langchain中的简易封装，所以作者本人非常喜欢这个项目，不仅学到rag，还能明白各种维护操作，更加符合工程思维

## 💼 产品定位：智能工作汇报与进度洞察系统

Wikforge 面向 ToB 团队管理场景，解决「主管依赖 3-7 天一次的口头汇报会掌握项目进度、
周期长、双方耗时、信息失真」的问题，把进度汇报日常化：

- **员工端（零填报）**：日常工作中随手把截图、日志、文档上传到团队空间，
  系统自动完成解析入库；每天 18:47 定时任务（或手动一键）由 AI 聚合当天
  全部材料，合成四段式工作日报（今日完成 / 进行中与进度 / 阻塞与风险 /
  明日计划），日报再次回流入库，形成可检索、可追溯的工作留痕。
- **主管端（碎片时间掌控进度）**：无需召集汇报会，主管在任意空闲时刻用
  自然语言向 RAG 问答询问「今天某部门完成了哪些内容」「某模块进度如何」，
  多路召回 + 精排保证答案带引用、可溯源。
- **进度预警工作流**：每晚 21:23 定时任务汇总空间内最近 7 天全员日报，
  LLM 对比工作项的跨天进度轨迹，识别「连续多天无进展 / 从日报中消失 /
  风险反复出现」的事项并生成预警，防止团队注意力过度聚焦而忽略进度死角。

底层是一套完整的手搓 RAG 引擎（多路召回 / 精排降级 / 查询增强 / 权限隔离），
工作汇报场景是它的落地承载，二者共用同一条检索与问答链路。

## ✨ 核心能力

```
📸 材料采集           截图 / 日志 / 文档即传即入库: PDF / DOCX / Markdown / HTML / 源代码 + LLM 视觉兜底
📝 AI 日报合成        按「员工+日期」聚合当日材料 → LLM 四段式日报 → 回流入库（工作留痕）
🚨 进度预警           Celery Beat 每日汇总 7 天日报 → LLM 跨天进度对比 → 停滞/消失/风险预警
🔍 复合搜索           BM25 + Dense Vector + Sparse Vector + RRF 融合 + Cross-Encoder 重排
🎯 Profile 系统       自动匹配文档类型 (通用文本 / 中式技术规范 / 扫描版 PDF)
💡 查询增强           LLM 改写 / HyDE 假设文档 / 多子查询分解, 三档独立开关
🤖 流式 RAG           SSE 输出 + 引用标注 + 会话记忆, 首 token < 5s
🔁 反馈闭环           错误模式聚合 → 优化建议 → 一键应用 → 批量重处理
📚 领域词典           术语标准化 + 同义词扩展 + 候选词审核
🔐 权限隔离           Pre-Filtering 在向量层与全文层同时生效；员工看自己、主管（空间创建者）看全员
🛡️ 审核队列           解析质量评分 + 人工修正 + Profile 反向优化
📊 后台管理           空间 / 用户 / 权限 / Profile / 词典 / 反馈 / 监控 / LLM 网关
```


## 🚀 Quick Start

```bash
# 1. 配置环境变量
cp .env.example .env
make secrets         # 生成强随机密钥, 拷贝到 .env
vim .env             # 至少填:
                     #   CPA_API_BASE / CPA_API_KEY        (Chat 上游)
                     #   DASHSCOPE_API_KEY                 (Embedding 上游, 阿里百炼)
                     #   INITIAL_ADMIN_PASSWORD            (建议随机, 启动后自动播种 admin)

# 2. 一键拉起 11 个服务
make first-run

# 3. 验证 (走一遍 登录→上传→搜索→RAG)
INITIAL_ADMIN_PASSWORD=$(grep ^INITIAL_ADMIN_PASSWORD .env | cut -d= -f2) \
  ./scripts/smoke-test.sh

# 4. 访问 (账号密码见下方"服务入口与凭证")
open http://localhost                # 前端
open http://localhost:8000/docs      # API 文档 (Swagger)
open http://localhost:5555           # Flower (Celery 监控)
open http://localhost:9001           # MinIO Console
open http://localhost:6333/dashboard # Qdrant Dashboard
```

完整部署 / 升级 / 备份 / 排错请看 [`docs/deploy.md`](docs/deploy.md)。

## 🔑 服务入口与凭证

> **不要把真实密码写进 README**——以下表格只列出**入口地址**和**`.env` 中对应的环境变量名**。
> 启动后从你本机的 `.env`（已在 `.gitignore`，不入库）查实际值。

| 服务 | 入口 | 用户名 | 密码 (`.env` key) |
|---|---|---|---|
| **Wikforge 主系统** | http://localhost | `INITIAL_ADMIN_EMAIL` | `INITIAL_ADMIN_PASSWORD` |
| **API 文档** (Swagger) | http://localhost:8000/docs | — (登录后复用主系统 JWT) | — |
| **Flower** (Celery 监控) | http://localhost:5555 | `FLOWER_USERNAME` | `FLOWER_PASSWORD` |
| **MinIO Console** | http://localhost:9001 | `MINIO_ACCESS_KEY` | `MINIO_SECRET_KEY` |
| **Qdrant Dashboard** | http://localhost:6333/dashboard | — (内网信任,无鉴权) | — |
| **OpenSearch** | http://localhost:9200 | `OPENSEARCH_USER` | `OPENSEARCH_PASSWORD` |
| **PostgreSQL** | `localhost:15432` | `POSTGRES_USER` (默认 `wikforge`) | `POSTGRES_PASSWORD` |
| **Redis** | `localhost:16379` | — | `REDIS_PASSWORD` (默认空) |

**快速查当前真实凭证** (本机执行):

```bash
# 列出所有登录相关的 env (会打印密码,只在私人终端跑)
grep -E '^(INITIAL_ADMIN|LITELLM_(MASTER|UI)|FLOWER|MINIO|POSTGRES|OPENSEARCH|REDIS)_' .env

# 或者只看主系统密码
grep ^INITIAL_ADMIN_PASSWORD .env
```

**重置某个密码**:

1. 改 `.env` 里对应的 key
2. `docker compose up -d --force-recreate <service>` 让新值生效
3. 主系统 admin 密码改后还需要 `make reset-admin` 或在数据库里重置 (见 `docs/deploy.md`)


## 🏗️ 架构

```mermaid
flowchart LR
    User([👤 用户])
    Browser[🌐 浏览器]

    subgraph Wikforge["🚀 Wikforge"]
        FE[Next.js 14<br/>前端]
        API[FastAPI<br/>API Server]
        Worker[Celery Worker<br/>concurrency=1<br/>PDF/Embed/Index]
        Beat[Celery Beat<br/>定时任务]
        Flower[Flower<br/>队列监控]
    end

    subgraph Storage["💾 存储层"]
        PG[(PostgreSQL<br/>元数据)]
        Redis[(Redis<br/>AOF 持久化<br/>Broker/Cache)]
        OS[(OpenSearch<br/>BM25)]
        QD[(Qdrant<br/>向量)]
        MinIO[(MinIO<br/>对象存储)]
    end

    subgraph Upstream["☁️ 上游模型"]
        CPA[CPA 网关<br/>gpt-5.5 / claude / qwen]
        DS[阿里百炼<br/>text-embedding-v3/v4]
    end

    User --> Browser
    Browser <--> FE
    FE <--> API
    API <--> PG & Redis & OS & QD & MinIO
    API -.触发任务.-> Worker
    Beat -.定时调度.-> Worker
    Flower -.读取.-> Redis
    Worker <--> PG & Redis & OS & QD & MinIO
    API --> CPA & DS
    Worker --> CPA & DS

    classDef fe fill:#3B82F6,stroke:#1E40AF,color:#fff
    classDef api fill:#10B981,stroke:#047857,color:#fff
    classDef worker fill:#F59E0B,stroke:#B45309,color:#fff
    classDef llm fill:#8B5CF6,stroke:#5B21B6,color:#fff
    classDef store fill:#1F2937,stroke:#111827,color:#fff
    classDef upstream fill:#EC4899,stroke:#9D174D,color:#fff

    class FE fe
    class API api
    class Worker,Beat,Flower worker
    class PG,Redis,OS,QD,MinIO store
    class CPA,DS upstream
```

## 📥 文档处理管线

```mermaid
flowchart LR
    Upload[📤 上传<br/>API] --> Parse[🔍 parse_document<br/>原生解析]
    Parse --> Profile[🎯 profile_match<br/>规则匹配]
    Profile --> Universal[🤖 universal_parser_check<br/>LLM 兜底]
    Universal --> Process[⚙️ process_document<br/>清洗+评分]
    Process -- 质量低 --> Review[👁️ 审核队列]
    Process --> Chunk[✂️ chunk_document<br/>分块]
    Chunk --> Embed[🧮 embed_chunks<br/>向量化]
    Embed --> Index[📦 index_chunks<br/>双索引入库]
    Index --> Done([✅ completed])

    Review -. 人工修正 .-> Process

    classDef ok fill:#22C55E,stroke:#15803D,color:#fff
    classDef llm fill:#8B5CF6,stroke:#5B21B6,color:#fff
    classDef warn fill:#F97316,stroke:#9A3412,color:#fff

    class Done ok
    class Universal,Embed llm
    class Review warn
```

## 📝 日报合成与进度预警管线

```mermaid
flowchart TB
    subgraph Employee["👤 员工端"]
        M[截图/日志/文档<br/>即传即入库] --> AGG[按 员工+日期 聚合<br/>当日 chunk]
        AGG --> GEN[LLM 四段式合成<br/>今日完成/进行中/风险/计划]
        GEN --> RE[日报回流入库<br/>走完整 pipeline]
    end

    RE --> RAGQ[主管端 RAG 进度问答<br/>自然语言 + 引用溯源]

    subgraph Beat["⏰ Celery Beat 定时"]
        AUTO[18:47 自动补生成<br/>全员日报]
        DET[21:23 汇总 7 天日报<br/>LLM 跨天进度对比]
    end

    AUTO --> GEN
    DET --> ALERT[(progress_alerts<br/>停滞/消失/风险预警)]
    ALERT --> PANEL[主管端预警面板<br/>确认 → 跟进闭环]

    classDef emp fill:#3B82F6,stroke:#1E40AF,color:#fff
    classDef llm fill:#8B5CF6,stroke:#5B21B6,color:#fff
    classDef warn fill:#F97316,stroke:#9A3412,color:#fff

    class M,AGG,RE emp
    class GEN,DET llm
    class ALERT,PANEL warn
```

**关键设计**

- **幂等与防自引用**：`(space_id, user_id, report_date)` 唯一约束，同日重复生成默认返回已有
  日报；聚合材料时排除标题为「工作日报-*」的回流文档，避免日报喂给下一次日报合成。
- **异常降级**：日报 LLM 合成失败仅标记该日报行 failed，原始材料检索链路不受影响；
  预警 LLM 输出做鲁棒 JSON 解析（容忍代码块包裹/噪声），解析失败当轮静默跳过。
- **权限模型复用**：员工只能生成/查看自己的日报；主管由管理员授予空间 `write` 权限后
  可看全员日报与预警，普通成员 `read` 权限天然被 ABAC 拦截（403）。

## 🔍 检索与问答管线

```mermaid
flowchart TB
    Q[用户查询] --> Enh{查询增强}
    Enh -->|改写| R1[改写×N]
    Enh -->|HyDE| R2[假设文档]
    Enh -->|分解| R3[子查询×N]

    R1 & R2 & R3 --> Multi{多路召回}
    Multi -->|BM25| OS[(OpenSearch)]
    Multi -->|Dense| QD1[(Qdrant Dense)]
    Multi -->|Sparse| QD2[(Qdrant Sparse)]

    OS & QD1 & QD2 --> RRF[🔀 RRF Fusion<br/>k=60]
    RRF --> Rerank[🎯 Cross-Encoder<br/>BGE-Reranker]
    Rerank --> Filter[阈值过滤]
    Filter -->|有结果| LLM[💬 LLM 合成<br/>+ 引用标注]
    Filter -->|无结果| Fallback[📭 兜底回复]
    LLM --> Stream[📡 SSE 流式输出]

    classDef enh fill:#A855F7,stroke:#6B21A8,color:#fff
    classDef rec fill:#3B82F6,stroke:#1E40AF,color:#fff
    classDef llm fill:#8B5CF6,stroke:#5B21B6,color:#fff

    class Enh,R1,R2,R3 enh
    class Multi,RRF,Rerank rec
    class LLM,Stream llm
```

## 📦 技术栈

| 层 | 组件 | 版本 | 职责 |
|---|---|---|---|
| **API** | FastAPI | 0.115+ | 异步 HTTP / OpenAPI 文档 |
| **ORM** | SQLAlchemy | 2.0 (async) | 类型安全的 DB 访问 |
| **Worker** | Celery | 5.4 | 文档处理流水线 |
| **迁移** | Alembic | 1.13+ | 数据库版本管理 |
| **前端** | Next.js | 14 (App Router) | React Server Components |
| **样式** | Tailwind + shadcn/ui | 3.4 | 设计系统 |
| **状态** | Zustand | 5 | 轻量状态管理 |
| **数据库** | PostgreSQL | 16 | 主数据 + JSONB 配置 |
| **缓存** | Redis | 7 | Celery broker / 进度 / 会话 |
| **全文** | OpenSearch | 2.17 | BM25 + 中文分词 (IK 可选) |
| **向量** | Qdrant | 1.14 | Dense (1024d) + Sparse (TF-IDF) |
| **存储** | MinIO | 2025 | S3 兼容对象存储 |
| **LLM** | openai SDK 直连 | 1.x | 直连任意 OpenAI 兼容端点 (2026-09 移除 LiteLLM Proxy，省 ~2GB 内存) |

## 🧰 常用命令

```bash
make help            # 查看全部命令 (19 个)
make ps              # 服务状态
make logs            # 跟踪所有日志
make logs-api        # 只看 api
make logs-worker     # 只看 worker
make psql            # 进 PostgreSQL CLI
make shell-api       # 进 api 容器 bash
make migrate         # 手动跑 alembic upgrade head
make seed            # 手动 init_db (admin + 默认 Profile)
make verify          # 跑 verify_compose 完整健康检查
make secrets         # 生成一组强随机密钥
make first-run       # 新机器: 启动 + 等待 healthy + 提示访问
make smoke           # 端到端冒烟 (登录→上传→搜索→RAG)
make backup          # 备份 postgres + minio + qdrant + opensearch 到 backups/
make clean           # 清 Docker 悬挂镜像 / build cache
make clean-data      # 清业务数据 (保留 admin / Profile)
make down            # 停止 (保留 volume)
make reset           # 完全清理 (会丢数据!)
```

## 🗂️ 目录结构

```
wikforge/
├── backend/              # Python / FastAPI
│   ├── app/
│   │   ├── api/          # 路由层 (auth/documents/search/qa/reports/alerts/admin_*)
│   │   ├── services/     # 业务逻辑 (含 report_service 日报合成 / alert_service 进度预警)
│   │   ├── tasks/        # Celery 任务 (pipeline.py 是核心; report_tasks.py 日报/预警定时任务)
│   │   ├── models/       # SQLAlchemy ORM (含 daily_report / progress_alert)
│   │   ├── core/         # 基础设施 (db/redis/qdrant/opensearch/minio)
│   │   └── scripts/      # init_db / api-entrypoint
│   ├── alembic/          # 数据库迁移
│   ├── tests/            # 单元 + 集成测试 (2054 个)
│   └── eval/             # 检索质量评估 (Recall@K / MRR / NDCG)
├── frontend/             # Next.js 14
│   ├── src/app/          # App Router 页面 (workbench 员工工作台 / alerts 主管进度中心)
│   ├── src/components/   # UI 组件
│   ├── src/lib/          # api-client / reports-api / utils
│   └── src/stores/       # Zustand stores
├── scripts/              # verify_compose / smoke-test / backup
├── secrets/              # 本地凭证速查 (gitignored,不入库)
├── docs/                 # 部署文档 / 架构图 / 资源
└── docker-compose.yml    # 10 服务编排 (api / worker / beat / flower + 6 存储)
```

## 🛣️ Roadmap

> 总计 55 项, 详情参见 [`docs/ROADMAP.md`](docs/ROADMAP.md)。

### 已完成 (本轮)

- ✅ **B2** 真 Cross-Encoder reranker (BAAI/bge-reranker-base, module-level singleton 缓存)
- ✅ **B3** Bigram fallback 加日志告警
- ✅ **A4 / A5** Embedding 走 LiteLLM Proxy + Redis 缓存
- ✅ **C1 / C2** presigned URL 下载 + multipart 上传
- ✅ **D2** 备份脚本 (postgres + minio + qdrant + opensearch)
- ✅ **E1 / E2** 修改密码 + 修改邮箱/显示名 + Settings 页
- ✅ **F4** Redis AOF 持久化
- ✅ **PDF OOM 修复** marker 模型缓存到共享 volume + `PDF_PARSER_MODE=auto` 小文件走 fitz
- ✅ **retry 入队修复** retry 真的 enqueue Celery (之前只改 db status)

### 近期 (P1)

- [ ] **A1 / A2** `list_spaces` / `list_documents` 加权限过滤
- [ ] **B1** OpenSearch 装 IK 中文分词器 (中文召回 +30~50%)
- [ ] **B4-B5** Profile 自动匹配 / LLM 兜底实测
- [ ] **A6** UploadService commit 边界重构

### 中期 (P2)

- [ ] **C6** 升级到 query_points API, 解锁 qdrant-client 1.15+
- [ ] **C9** qdrant collection / opensearch index 启动时自动 ensure
- [ ] **D3** nginx 反代 (TLS 终止 + SSE proxy_buffering off)
- [ ] **D9** 全局 health-check API + 前端 dashboard

### 远期 (P3)

- [ ] **E3-E12** 用户体验扩展 (版本管理 / 审计 / 批量 / i18n / 移动端)
- [ ] **F1-F7** 性能 / HA (gunicorn / Postgres 池 / OpenSearch JVM / Qdrant HNSW 调优)
- [ ] **G1-G5** CI/CD 集成测试 / 检索质量自动评估

## 📚 文档

| 文档 | 说明 |
|---|---|
| [`docs/deploy.md`](docs/deploy.md) | 部署 / 升级 / 备份 / 排错完整手册 |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | 55 项修复 / 优化清单 |
| [`backend/tests/integration/README.md`](backend/tests/integration/README.md) | 集成测试运行说明 |
| [`frontend/e2e/README.md`](frontend/e2e/README.md) | Playwright E2E 说明 |
| [`scripts/README.md`](scripts/README.md) | 运维脚本说明 |

