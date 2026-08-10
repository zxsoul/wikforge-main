# scripts/

本目录存放与项目运维相关的辅助脚本。

## verify_compose.sh

用于验证 `docker-compose.yml` 中定义的全部服务（`postgres`、`redis`、`opensearch`、`qdrant`、`minio`、`api`、`worker`、`frontend`）能成功启动并通过容器内的健康检查。归属任务：`enterprise-knowledge-base / Task 1.10`。

### 何时使用

- 首次拉起本地开发环境后，做一次端到端的"基础设施 + 应用容器全部就绪"验证。
- CI 中作为 smoke check：在每次合并前确认 compose 编排没有被破坏。
- 修改了 `docker-compose.yml`、Dockerfile 或健康检查命令后做回归。

### 前置条件

1. **Docker Engine 已安装且 daemon 正在运行**
   - macOS：启动 Docker Desktop，确认任务栏图标处于运行状态。
   - Linux：`systemctl start docker` 并确认当前用户在 `docker` 组。
   - 自检：`docker info` 能正常返回，没有 "Cannot connect to the Docker daemon"。
2. **Docker Compose v2**（脚本使用 `docker compose` 子命令而非 `docker-compose`）
   - 自检：`docker compose version` 输出 v2.x。
3. **`jq` 已安装**（用于解析 `docker compose ps --format json` 输出）
   - macOS：`brew install jq`
   - Debian/Ubuntu：`sudo apt-get install jq`
4. **`.env` 文件存在**（推荐）
   - `cp .env.example .env`，按需调整参数。
   - 不创建 `.env` 也可以运行：`docker-compose.yml` 中所有变量都带有默认值，但密码会落到不安全的占位值，仅适合本地试跑。
5. **磁盘和内存充足**
   - OpenSearch 默认堆内存 512MB；首次启动各镜像合计需要约 5GB 磁盘空间。

### 如何运行

```bash
# 在仓库根目录执行
./scripts/verify_compose.sh
```

或显式从 scripts 目录调用，脚本会自动切回到项目根：

```bash
bash scripts/verify_compose.sh
```

可通过环境变量调整行为：

| 变量              | 默认值               | 说明                                     |
| ----------------- | -------------------- | ---------------------------------------- |
| `COMPOSE_FILE`    | `docker-compose.yml` | 使用的 compose 文件路径                  |
| `TIMEOUT_SECONDS` | `300`                | 健康检查最大等待秒数（默认 5 分钟）      |
| `POLL_INTERVAL`   | `10`                 | 轮询间隔秒数                             |
| `LOG_TAIL_LINES`  | `50`                 | 失败时每个服务打印的日志行数             |

示例：放宽到 10 分钟超时：

```bash
TIMEOUT_SECONDS=600 ./scripts/verify_compose.sh
```

### 行为与退出码

脚本会依次完成：

1. 检查 `docker`、`docker compose`、`jq` 与 daemon 可用性。
2. 执行 `docker compose -f $COMPOSE_FILE config --quiet` 校验 compose 语法。
3. 执行 `docker compose up -d` 启动所有服务（如有未拉取的镜像会自动拉取，首次启动较慢）。
4. 每 `POLL_INTERVAL` 秒读取 `docker compose ps --format json`，解析每个服务的 `State` 和 `Health`。
5. 状态有变化时刷新输出一张服务状态表（绿色 healthy / 黄色 starting 或无 healthcheck / 红色 unhealthy）。
6. 当全部期望服务都进入 `healthy` 时，打印总结表并以退出码 `0` 结束。
7. 如果在 `TIMEOUT_SECONDS` 内仍未全部 healthy，打印当前状态表 + 每个服务最近 `LOG_TAIL_LINES` 行日志，以退出码 `1` 结束。

退出码：

- `0`：全部服务 healthy。
- `1`：前置依赖缺失、compose 校验失败、`up -d` 失败，或健康检查超时/部分服务 unhealthy。

### 排错指南

- **`Cannot connect to the Docker daemon`**：Docker Desktop 未启动或 daemon 已挂掉，先恢复 daemon 再重试。
- **`docker compose: 'compose' is not a docker command`**：使用的是旧版 `docker-compose`，请安装 Compose v2 或将命令替换为 `docker-compose` 后重试（脚本仅支持 v2）。
- **某服务长期 `starting`**：通常是镜像拉取慢、容器初始化慢（OpenSearch 第一次启动可能 60-120 秒）。可适当调大 `TIMEOUT_SECONDS`。如果失败仍然继续，看脚本输出的对应服务日志：
  - `opensearch`：常见是堆内存不足，调小 `OPENSEARCH_JAVA_OPTS`；或宿主机 `vm.max_map_count` 太低，需要 `sudo sysctl -w vm.max_map_count=262144`。
  - `postgres` / `redis` / `minio`：检查端口是否被宿主机其他进程占用，必要时改 `.env` 中对应端口。
  - `api` / `worker`：依赖未就绪或环境变量不全；查看 traceback 修正后 `docker compose up -d --build api worker`。
  - `frontend`：尚未实现或镜像未构建；可暂时 `docker compose up -d --scale frontend=0` 跳过。
- **`mc ready local` 一直失败**：MinIO 启动较慢或镜像版本切换导致 `mc` 子命令未就绪，等待 1-2 个轮询周期通常即可恢复。
- **想强制重置环境**：`docker compose down -v` 会同时删除 volumes，下次脚本运行相当于全新初始化。

### 与 Task 1.10 的对应关系

Task 1.10 验证目标分为两类：

- **静态/离线验证**：通过 CI 或本地 `docker compose config --quiet`、`python3 -m compileall backend/app`、`.env.example` 与 `${VAR}` 一致性检查即可完成，不依赖 daemon。
- **运行时验证**：必须有可用的 Docker daemon。本脚本即该步骤的可执行交付物，能在任何安装好 Docker 的机器上一键复现。
