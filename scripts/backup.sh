#!/bin/bash
# Wikforge 一键备份脚本
#
# 备份内容:
#   - PostgreSQL (元数据 / 用户 / 权限 / Profile / 词典) -> .sql
#   - MinIO (文档原文件) -> .tar.gz
#   - Qdrant (向量) -> .tar.gz
#   - OpenSearch index 'chunks' -> .tar.gz (元数据 + 索引快照)
#
# 输出: backups/wikforge-YYYY-MM-DD-HHMMSS/ 目录下 4 个文件
# 默认保留最近 7 天的备份,更老的自动删除
#
# Cron 示例 (每天凌晨 2 点跑):
#   0 2 * * * cd /path/to/wikforge && ./scripts/backup.sh >> backups/cron.log 2>&1
#
# 恢复见 docs/deploy.md 第 7 节。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TS=$(date +%F-%H%M%S)
OUT_DIR="backups/wikforge-$TS"
mkdir -p "$OUT_DIR"

# 通过 docker compose 拿到生效的环境变量, 避免 .env 中带特殊字符的值 (如 OPENSEARCH_JAVA_OPTS)
# 干扰 shell source。
PG_USER=$(docker compose exec -T postgres printenv POSTGRES_USER 2>/dev/null | tr -d '\r' || echo "wikforge")
PG_DB=$(docker compose exec -T postgres printenv POSTGRES_DB 2>/dev/null | tr -d '\r' || echo "wikforge")

echo "[1/4] 备份 PostgreSQL ..."
docker compose exec -T postgres pg_dump \
    -U "${PG_USER:-wikforge}" \
    -d "${PG_DB:-wikforge}" \
    --no-owner --no-privileges \
    > "$OUT_DIR/postgres.sql"
echo "  ✓ $(du -h "$OUT_DIR/postgres.sql" | cut -f1)"

echo "[2/4] 备份 MinIO ..."
docker run --rm \
    -v wikforge_minio_data:/data:ro \
    -v "$(pwd)/$OUT_DIR:/bak" \
    alpine \
    tar czf /bak/minio.tar.gz -C /data . 2>/dev/null
echo "  ✓ $(du -h "$OUT_DIR/minio.tar.gz" | cut -f1)"

echo "[3/4] 备份 Qdrant ..."
docker run --rm \
    -v wikforge_qdrant_data:/data:ro \
    -v "$(pwd)/$OUT_DIR:/bak" \
    alpine \
    tar czf /bak/qdrant.tar.gz -C /data . 2>/dev/null
echo "  ✓ $(du -h "$OUT_DIR/qdrant.tar.gz" | cut -f1)"

echo "[4/4] 备份 OpenSearch ..."
docker run --rm \
    -v wikforge_opensearch_data:/data:ro \
    -v "$(pwd)/$OUT_DIR:/bak" \
    alpine \
    tar czf /bak/opensearch.tar.gz -C /data . 2>/dev/null
echo "  ✓ $(du -h "$OUT_DIR/opensearch.tar.gz" | cut -f1)"

echo ""
echo "✓ 备份完成: $OUT_DIR ($(du -sh "$OUT_DIR" | cut -f1))"

# 保留最近 7 天的备份, 更老的自动删除
KEEP_DAYS="${BACKUP_KEEP_DAYS:-7}"
echo ""
echo "清理 ${KEEP_DAYS} 天前的旧备份 ..."
find backups -maxdepth 1 -type d -name 'wikforge-*' -mtime "+${KEEP_DAYS}" -print -exec rm -rf {} \; 2>/dev/null || true

echo "✓ 完成"
