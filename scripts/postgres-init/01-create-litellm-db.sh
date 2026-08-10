#!/bin/bash
# 自动为 LiteLLM Proxy 创建独立数据库,避免污染 wikforge 主库。
# Postgres 镜像会在首次初始化时执行 /docker-entrypoint-initdb.d/*.sh。
set -e

LITELLM_DB="${LITELLM_DB:-litellm}"

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    SELECT 'CREATE DATABASE $LITELLM_DB OWNER $POSTGRES_USER'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$LITELLM_DB')\gexec
EOSQL

echo "[postgres-init] LiteLLM database '$LITELLM_DB' ensured"
