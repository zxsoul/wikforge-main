# Wikforge 部署手册

适用范围: macOS / Linux 单机部署。生产部署到 Kubernetes 见 `docs/deploy-k8s.md` (TODO)。

## 0. 前置依赖

| 依赖 | 版本 | 安装 |
|---|---|---|
| Docker Engine + Compose v2 | 24+ | macOS: Docker Desktop / Linux: `apt install docker-ce docker-compose-plugin` |
| jq | 任意 | `brew install jq` / `apt install jq` |
| GNU Make (可选,用 Makefile) | 任意 | macOS 自带 |
| openssl (生成密钥) | 任意 | 系统自带 |

资源建议: 8GB+ 内存,首次部署需 6GB 磁盘 (镜像 + 容器 + 卷)。

## 1. 国内拉镜像加速 (强烈建议)

国内直连 Docker Hub 偶发 EOF 错误。在 Docker Desktop -> Settings -> Docker Engine 加上 (合并到现有 JSON,不要覆盖):

```json
{
  "registry-mirrors": [
    "https://docker.1ms.run",
    "https://docker.xuanyuan.me",
    "https://docker.m.daocloud.io",
    "https://dockerproxy.com"
  ]
}
```

点 "Apply & restart"。Linux 直接编辑 `/etc/docker/daemon.json` 后 `systemctl restart docker`。

## 2. 首次部署 (一键)

```bash
# 1. 拷贝项目并进入目录
git clone <repo> wikforge && cd wikforge

# 2. 准备环境变量
cp .env.example .env
make secrets        # 复制输出, 替换 .env 中对应字段
vim .env            # 至少填:
                    # POSTGRES_PASSWORD / MINIO_SECRET_KEY / JWT_SECRET_KEY
                    # LITELLM_MASTER_KEY / LITELLM_SALT_KEY / LITELLM_UI_PASSWORD
                    # CPA_API_KEY / DASHSCOPE_API_KEY (按需)

# 3. 一键启动
make first-run
```

`first-run` 会:
1. `docker compose up -d` 启动所有服务 (首次拉镜像 + 构建,5-15 分钟)
2. 轮询等待全部容器 healthy
3. api 容器 entrypoint 自动跑 `alembic upgrade head` + `init_db` 播种 admin

## 3. 访问入口

| 服务 | URL | 凭证 |
|---|---|---|
| Wikforge 前端 | http://localhost:${FRONTEND_PORT} | `INITIAL_ADMIN_EMAIL` / `INITIAL_ADMIN_PASSWORD` |
| API 文档 (Swagger) | http://localhost:${API_PORT}/docs | - |
| LiteLLM Admin UI | http://localhost:${LITELLM_PORT}/ui | `LITELLM_UI_USERNAME` / `LITELLM_UI_PASSWORD` 或直接用 `LITELLM_MASTER_KEY` |
| MinIO 控制台 | http://localhost:${MINIO_CONSOLE_PORT} | `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` |
| Qdrant Dashboard | http://localhost:${QDRANT_PORT}/dashboard | - |

## 4. 日常运维

```bash
make ps              # 查看所有服务状态
make logs            # 跟踪所有日志
make logs-api        # 只看 api
make logs-worker     # 只看 worker
make psql            # 进 PostgreSQL CLI
make shell-api       # 进 api 容器 bash
make verify          # 完整健康检查
make down            # 停止 (保留数据)
make up              # 再次启动
make reset           # 完全清理 (会丢失所有数据!)
```

## 5. 升级 / 重新部署

```bash
git pull
docker compose pull            # 拉最新基础镜像
docker compose build --pull    # 重建应用镜像
docker compose up -d           # api 容器 entrypoint 自动跑增量 alembic
```

## 6. 常见问题排查

### 6.1 端口冲突

宿主机已有 5432 / 6379 / 80 / 8000 等端口的占用进程。改 `.env` 中对应 `*_PORT` 变量为不冲突的端口 (例如本项目默认已经用 15432 / 16379 避开 PG / Redis)。

```bash
lsof -nP -iTCP:5432 -sTCP:LISTEN   # 查谁占用
```

### 6.2 镜像拉取超时 / EOF

按 1 节配置镜像加速器。Docker Desktop 重启后再 `docker compose pull` 重试。

### 6.3 admin 邮箱被拒

pydantic `EmailStr` 不接受 `.local` / `.test` / `.example` 等保留域。改用 `admin@wikforge.com` 这类合规邮箱。

### 6.4 前端 "网络错误"

浏览器 console 看 Network 面板:

- **CORS preflight 400**: `.env` 中 `CORS_ORIGINS` 没包含浏览器实际访问的 origin。注意 `http://localhost` 与 `http://localhost:80` 是两个字符串。
- **connection refused**: api 容器没起 / 健康检查失败,看 `make logs-api`。
- **401 Unauthorized**: token 过期,刷新页面或重新登录。

### 6.5 文档上传后处理失败

看 worker 日志 `make logs-worker`:

- **embedding 调用失败**: `DASHSCOPE_API_KEY` 没配 / 错误。`make secrets` 输出不会覆盖此值,你手填的不要被覆盖。
- **LiteLLM 调用失败**: 进 LiteLLM UI Test 页验证 `gpt-5.5` / `text-embedding-v3` 直接能调通。
- **OCR / LibreOffice 超时**: 调大 `UNIVERSAL_PARSER_LIBREOFFICE_TIMEOUT` 或 worker `--time-limit`。

### 6.6 Postgres 健康检查反复失败

通常是 `POSTGRES_PASSWORD` 改了但 volume 里的旧密码未更新。`make reset` 重置 (会丢数据) 或手动:

```bash
docker compose stop postgres
docker volume rm wikforge_postgres_data
docker compose up -d postgres
```

## 7. 备份与恢复

### 备份 (建议每天 cron)

```bash
# 数据库
docker compose exec -T postgres pg_dump -U wikforge wikforge > backups/wikforge-$(date +%F).sql

# MinIO (对象存储)
docker run --rm -v wikforge_minio_data:/data -v $(pwd)/backups:/bak alpine \
    tar czf /bak/minio-$(date +%F).tar.gz -C /data .

# Qdrant (向量库)
docker run --rm -v wikforge_qdrant_data:/data -v $(pwd)/backups:/bak alpine \
    tar czf /bak/qdrant-$(date +%F).tar.gz -C /data .
```

### 恢复

```bash
# 数据库
make down && docker volume rm wikforge_postgres_data
make up                     # 让 postgres 重建空库
sleep 30                    # 等就绪
docker compose exec -T postgres psql -U wikforge -d wikforge < backups/wikforge-2026-05-17.sql

# MinIO / Qdrant 同理: docker volume rm -> 重新启动 -> tar -x 解压回 volume
```

## 8. 安全清单 (上线前必看)

- [ ] `make secrets` 输出全部替换到 `.env`,所有 `change-me-in-production` 不再出现
- [ ] `INITIAL_ADMIN_PASSWORD` 改为强口令,登录后立即在 UI 中再改一次
- [ ] `JWT_SECRET_KEY` 是 32+ 字节随机串
- [ ] `LITELLM_MASTER_KEY` 改为强随机串,只通过环境变量传递,不写入代码
- [ ] `MINIO_ACCESS_KEY` 不再使用默认 `minioadmin`
- [ ] `OPENSEARCH_PASSWORD` 满足复杂度要求
- [ ] 关闭 `DEBUG=true`
- [ ] 调整 `CORS_ORIGINS` 只列生产域名,不留 `http://localhost*`
- [ ] 反向代理 (nginx/Traefik) 终止 TLS,不让任何端口暴露到公网
- [ ] 防火墙只放行 80 / 443,其它端口仅限内网
- [ ] 配置自动备份 + 异地存储

## 9. 端口清单

| 端口 | 服务 | 暴露范围 |
|---|---|---|
| 80 | Frontend (nginx) | 公网 |
| 8000 | API (FastAPI) | 公网 (建议反代) |
| 4000 | LiteLLM Proxy | 内网 |
| 9001 | MinIO Console | 内网 |
| 9000 | MinIO API (S3) | 内网 |
| 9200 | OpenSearch | 内网 |
| 6333 | Qdrant HTTP | 内网 |
| 6334 | Qdrant gRPC | 内网 |
| 15432 | Postgres | 内网 (运维需要) |
| 16379 | Redis | 内网 (运维需要) |
