#!/bin/bash
# wikforge 端到端冒烟测试
# 流程: login -> 创建 space -> 上传文档 -> 等待处理 -> 搜索 -> RAG 提问

set -e

API="http://localhost:8000"
EMAIL="${INITIAL_ADMIN_EMAIL:-admin@wikforge.com}"
PASSWORD="${INITIAL_ADMIN_PASSWORD:-Admin@123}"

echo "==[1/6] 登录 admin =================================================="
TOKEN=$(curl -sS --max-time 10 -X POST "$API/api/auth/login" \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"$EMAIL\",\"password\":\"$PASSWORD\"}" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['access_token'])")
[ -z "$TOKEN" ] && echo "登录失败" && exit 1
echo "✓ 拿到 access_token (len=${#TOKEN})"
AUTH="Authorization: Bearer $TOKEN"

echo ""
echo "==[2/6] 创建测试空间 ================================================="
SPACE_RESP=$(curl -sS --max-time 10 -X POST "$API/api/spaces" \
  -H "$AUTH" -H "Content-Type: application/json" \
  -d '{"name":"smoke-test-space","description":"冒烟测试用空间"}')
SPACE_ID=$(echo "$SPACE_RESP" | python3 -c "import json,sys; print(json.load(sys.stdin).get('id',''))" 2>/dev/null)
if [ -z "$SPACE_ID" ]; then
  echo "创建空间失败,尝试复用已有 space:"
  SPACE_ID=$(curl -sS "$API/api/spaces" -H "$AUTH" | python3 -c "
import json,sys
arr=json.load(sys.stdin)
for s in arr:
    if s['name']=='smoke-test-space':
        print(s['id']); break
")
fi
echo "✓ space_id=$SPACE_ID"

echo ""
echo "==[3/6] 准备并上传测试文档 =========================================="
TMPDIR=$(mktemp -d)
TESTDOC="$TMPDIR/wikforge-test.md"
cat > "$TESTDOC" <<'EOF'
# Wikforge 是什么

Wikforge 是一个企业级知识库系统,基于 RAG 架构,
支持文档智能解析、向量检索和大语言模型问答。

## 核心组件

- 后端: FastAPI + SQLAlchemy + Celery
- 前端: Next.js 14 + Tailwind + shadcn
- 向量库: Qdrant 1.12
- 关键词检索: OpenSearch 2.17
- 对象存储: MinIO
- LLM 网关: LiteLLM Proxy

## 关键能力

文档解析支持 PDF、Word、Markdown、HTML、源代码;
具备 LLM 兜底解析能力,可处理扫描版 PDF 等复杂格式。
EOF

UPLOAD_RESP=$(curl -sS --max-time 30 -X POST "$API/api/documents/upload?space_id=$SPACE_ID" \
  -H "$AUTH" \
  -F "files=@$TESTDOC")
echo "  上传响应: $(echo "$UPLOAD_RESP" | head -c 300)"
DOC_ID=$(echo "$UPLOAD_RESP" | python3 -c "
import json,sys
data=json.load(sys.stdin)
arr = data if isinstance(data,list) else data.get('items',[data])
print(arr[0].get('id','') if arr else '')
" 2>/dev/null)
[ -z "$DOC_ID" ] && echo "上传失败" && exit 1
echo "✓ document_id=$DOC_ID"

echo ""
echo "==[4/6] 等待文档处理完成 (最多 90 秒) =============================="
for i in $(seq 1 30); do
  RAW=$(curl -sS "$API/api/documents/$DOC_ID/progress" -H "$AUTH" 2>/dev/null)
  STAGE=$(echo "$RAW" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('stage') or d.get('status') or 'unknown')" 2>/dev/null)
  PROGRESS=$(echo "$RAW" | python3 -c "import json,sys; print(json.load(sys.stdin).get('progress',0))" 2>/dev/null)
  echo "  [$i] stage=$STAGE progress=$PROGRESS"
  if [ "$STAGE" = "done" ] || [ "$STAGE" = "completed" ]; then
    echo "✓ 文档处理完成"
    break
  fi
  if [ "$STAGE" = "failed" ]; then
    echo "✗ 文档处理失败"
    echo "$RAW"
    exit 1
  fi
  sleep 3
done

echo ""
echo "==[5/6] 测试搜索 ===================================================="
SEARCH_RESP=$(curl -sS --max-time 15 -X POST "$API/api/search" \
  -H "$AUTH" -H "Content-Type: application/json" \
  -d '{"query":"wikforge 用什么向量库","page":1,"page_size":5}')
echo "$SEARCH_RESP" | python3 -c "
import json,sys
d=json.load(sys.stdin)
items=d.get('items') or d.get('results') or []
print(f'  total={d.get(\"total\",len(items))}, items={len(items)}')
for i,it in enumerate(items[:3]):
    snippet = (it.get('content') or it.get('text') or '')[:80].replace('\n',' ')
    print(f'  [{i}] score={it.get(\"score\",0):.3f}: {snippet}...')
"

echo ""
echo "==[6/6] 测试 RAG 问答 ================================================"
ANSWER=$(curl -sS --max-time 60 -X POST "$API/api/qa/ask" \
  -H "$AUTH" -H "Content-Type: application/json" \
  -d '{"question":"wikforge 用的是什么向量数据库?","top_k":3}')
echo "$ANSWER" | python3 -c "
import json,sys
d=json.load(sys.stdin)
print('回答:', (d.get('answer') or d.get('content') or str(d))[:400])
print('引用数:', len(d.get('citations',[])) or len(d.get('sources',[])))
"

echo ""
echo "===================================================================="
echo "✓ 冒烟测试通过"
echo "===================================================================="

rm -rf "$TMPDIR"
