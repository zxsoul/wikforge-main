#!/bin/bash
# =============================================================================
# wikforge api 容器入口脚本
#
# 功能:
#   1. 等待 postgres 就绪 (compose 已用 depends_on:condition:service_healthy,
#      此处再做一次 200ms 轮询作为兜底)
#   2. 跑 alembic upgrade head, 幂等
#   3. 跑 init_db 播种 admin + 默认 Profile, 幂等
#   4. exec 替换为 uvicorn 进程, 让信号能正常传递给 FastAPI
#
# 失败处理:
#   - 任何一步失败立即退出, compose 会按 restart_policy 重启容器
#   - 幂等设计意味着重启不会污染数据
# =============================================================================

set -euo pipefail

echo "[entrypoint] waiting for postgres ${POSTGRES_HOST:-postgres}:5432 ..."
until python -c "
import os, socket, sys
s = socket.socket()
s.settimeout(1)
try:
    s.connect((os.environ.get('POSTGRES_HOST', 'postgres'), 5432))
    s.close()
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null; do
    sleep 0.5
done
echo "[entrypoint] postgres is reachable"

echo "[entrypoint] running alembic migrations ..."
alembic upgrade head

# init_db 是幂等的(用户/profile 走 SELECT 后再 INSERT),失败也不应阻塞 API 启动
# 但我们仍然让失败可见 —— 失败时记 warning,继续启动
echo "[entrypoint] seeding initial admin + default profiles ..."
if ! python -m app.scripts.init_db; then
    echo "[entrypoint] WARN: init_db failed, continuing anyway"
fi

echo "[entrypoint] starting uvicorn ..."
exec uvicorn app.main:app --host "${API_HOST:-0.0.0.0}" --port "${API_PORT:-8000}" --reload
