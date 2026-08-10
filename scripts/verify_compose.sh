#!/usr/bin/env bash
# =============================================================================
# verify_compose.sh
#
# 用途：
#   验证 docker-compose.yml 能完整启动并通过所有服务的健康检查。
#   归属任务：enterprise-knowledge-base / Task 1.10。
#
# 步骤：
#   1) 检查 docker / jq 是否可用，且 Docker daemon 正在运行
#   2) `docker compose config --quiet` 校验 compose 语法
#   3) `docker compose up -d` 启动全部服务
#   4) 最长 5 分钟内每 10 秒轮询 `docker compose ps --format json`，
#      检查每个服务的 Health 字段是否为 "healthy"
#   5) 输出每个服务的最终状态
#   6) 失败时打印各服务最近 50 行日志
#
# 退出码：
#   0 - 全部服务健康
#   1 - 超时 / 任意服务不健康 / 前置依赖缺失 / 启动失败
# =============================================================================
set -euo pipefail

# ---------- 配置 ----------
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-300}"   # 5 分钟
POLL_INTERVAL="${POLL_INTERVAL:-10}"        # 10 秒
LOG_TAIL_LINES="${LOG_TAIL_LINES:-50}"

# ---------- 输出工具 ----------
if [ -t 1 ]; then
    C_RED=$'\033[31m'; C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'
    C_BLUE=$'\033[34m'; C_BOLD=$'\033[1m'; C_RESET=$'\033[0m'
else
    C_RED=""; C_GREEN=""; C_YELLOW=""; C_BLUE=""; C_BOLD=""; C_RESET=""
fi
log_info()  { printf "%s[INFO]%s %s\n"  "$C_BLUE"   "$C_RESET" "$*"; }
log_ok()    { printf "%s[ OK ]%s %s\n"  "$C_GREEN"  "$C_RESET" "$*"; }
log_warn()  { printf "%s[WARN]%s %s\n"  "$C_YELLOW" "$C_RESET" "$*" >&2; }
log_error() { printf "%s[ERR ]%s %s\n"  "$C_RED"    "$C_RESET" "$*" >&2; }

# ---------- 切换到脚本父目录（项目根） ----------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [ ! -f "$COMPOSE_FILE" ]; then
    log_error "未找到 compose 文件: $PROJECT_ROOT/$COMPOSE_FILE"
    exit 1
fi

# ---------- 前置依赖检查 ----------
check_prerequisites() {
    if ! command -v docker >/dev/null 2>&1; then
        log_error "未检测到 docker 命令。请先安装 Docker Desktop / Docker Engine。"
        exit 1
    fi
    if ! docker compose version >/dev/null 2>&1; then
        log_error "未检测到 'docker compose' 子命令。请升级到 Docker Compose v2。"
        exit 1
    fi
    if ! command -v jq >/dev/null 2>&1; then
        log_error "未检测到 jq 命令。macOS: 'brew install jq'；Debian/Ubuntu: 'apt-get install jq'。"
        exit 1
    fi
    if ! docker info >/dev/null 2>&1; then
        log_error "无法连接到 Docker daemon。请先启动 Docker Desktop 或 'systemctl start docker'。"
        exit 1
    fi
    log_ok "前置依赖检查通过 (docker / docker compose / jq / daemon)"
}

# ---------- 解析 ps 输出（兼容两种格式） ----------
# 新版本 compose 输出 NDJSON（每行一个对象），旧版本输出 JSON 数组。
# 统一转换为 NDJSON。
ps_as_ndjson() {
    local raw
    raw="$(docker compose -f "$COMPOSE_FILE" ps --all --format json 2>/dev/null || true)"
    if [ -z "$raw" ]; then
        return 0
    fi
    if printf '%s' "$raw" | head -c 1 | grep -q '\['; then
        printf '%s' "$raw" | jq -c '.[]'
    else
        printf '%s' "$raw"
    fi
}

# ---------- 主流程 ----------
main() {
    log_info "项目根目录: $PROJECT_ROOT"
    log_info "Compose 文件: $COMPOSE_FILE"

    check_prerequisites

    log_info "校验 compose 语法 ..."
    if ! docker compose -f "$COMPOSE_FILE" config --quiet; then
        log_error "compose 语法校验失败"
        exit 1
    fi
    log_ok "compose 语法校验通过"

    # 期望服务列表（来自 compose 文件本身，避免硬编码）
    local services
    services="$(docker compose -f "$COMPOSE_FILE" config --services | sort)"
    local expected_count
    expected_count="$(printf '%s\n' "$services" | wc -l | tr -d ' ')"
    log_info "期望服务数量: $expected_count"
    printf '%s\n' "$services" | sed 's/^/    - /'

    log_info "执行 docker compose up -d (不会触发 docker pull/build 之外的额外网络操作) ..."
    if ! docker compose -f "$COMPOSE_FILE" up -d; then
        log_error "docker compose up -d 失败"
        dump_logs "$services"
        exit 1
    fi

    log_info "开始健康检查轮询: 总时长 ${TIMEOUT_SECONDS}s，间隔 ${POLL_INTERVAL}s"
    local deadline
    deadline=$(( $(date +%s) + TIMEOUT_SECONDS ))

    local last_summary=""
    while :; do
        local now
        now=$(date +%s)
        local remaining=$(( deadline - now ))

        local ndjson
        ndjson="$(ps_as_ndjson)"

        # 汇总每个服务的状态/健康
        # 输出格式: <service>\t<state>\t<health>
        local summary=""
        if [ -n "$ndjson" ]; then
            summary="$(printf '%s' "$ndjson" \
                | jq -r '[(.Service // .Name // "?"), (.State // "?"), (.Health // "")] | @tsv' \
                | sort -u || true)"
        fi

        if [ "$summary" != "$last_summary" ]; then
            printf "\n%s当前状态 (剩余 %ss)%s\n" "$C_BOLD" "$remaining" "$C_RESET"
            print_status_table "$services" "$summary"
            last_summary="$summary"
        fi

        # 判定终态
        if all_healthy "$services" "$summary"; then
            printf "\n"
            log_ok "全部 $expected_count 个服务已健康"
            print_status_table "$services" "$summary"
            exit 0
        fi

        if [ "$now" -ge "$deadline" ]; then
            printf "\n"
            log_error "健康检查超时（${TIMEOUT_SECONDS}s）"
            print_status_table "$services" "$summary"
            dump_logs "$services"
            exit 1
        fi

        sleep "$POLL_INTERVAL"
    done
}

# ---------- 状态表渲染 ----------
print_status_table() {
    local services="$1"
    local summary="$2"
    printf "  %-14s %-12s %-12s\n" "SERVICE" "STATE" "HEALTH"
    printf "  %-14s %-12s %-12s\n" "-------" "-----" "------"
    local svc state health line
    while IFS= read -r svc; do
        [ -z "$svc" ] && continue
        line="$(printf '%s\n' "$summary" | awk -v s="$svc" -F'\t' '$1==s{print; exit}')"
        if [ -z "$line" ]; then
            state="not-created"
            health="-"
        else
            state="$(printf '%s' "$line" | awk -F'\t' '{print $2}')"
            health="$(printf '%s' "$line" | awk -F'\t' '{print $3}')"
            [ -z "$health" ] && health="(no-healthcheck)"
        fi
        local color="$C_YELLOW"
        case "$health" in
            healthy)              color="$C_GREEN" ;;
            unhealthy|"")         color="$C_RED" ;;
            "(no-healthcheck)")   color="$C_YELLOW" ;;
        esac
        printf "  %-14s %-12s %s%-12s%s\n" "$svc" "$state" "$color" "$health" "$C_RESET"
    done <<EOF
$services
EOF
}

# ---------- 终态判定：所有期望服务必须 healthy ----------
all_healthy() {
    local services="$1"
    local summary="$2"
    [ -z "$summary" ] && return 1
    local svc line health
    while IFS= read -r svc; do
        [ -z "$svc" ] && continue
        line="$(printf '%s\n' "$summary" | awk -v s="$svc" -F'\t' '$1==s{print; exit}')"
        [ -z "$line" ] && return 1
        health="$(printf '%s' "$line" | awk -F'\t' '{print $3}')"
        if [ "$health" != "healthy" ]; then
            return 1
        fi
    done <<EOF
$services
EOF
    return 0
}

# ---------- 失败时输出每个服务的日志尾巴 ----------
dump_logs() {
    local services="$1"
    log_warn "下面输出各服务最近 ${LOG_TAIL_LINES} 行日志，便于排错："
    local svc
    while IFS= read -r svc; do
        [ -z "$svc" ] && continue
        printf "\n%s========== logs: %s ==========%s\n" "$C_BOLD" "$svc" "$C_RESET"
        docker compose -f "$COMPOSE_FILE" logs --tail "$LOG_TAIL_LINES" "$svc" 2>&1 || true
    done <<EOF
$services
EOF
}

main "$@"
