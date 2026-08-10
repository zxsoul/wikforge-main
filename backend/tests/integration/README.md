# 后端集成测试（任务 25）

## 概览

| 任务 | 文件 | 内容 |
|------|------|------|
| 25.1 | `conftest.py` + `docker-compose.test.yml` | 集成测试环境（端点 fixture） |
| 25.2 | `test_e2e_document_import.py` | 文档导入 → 入库 → 可搜索 |
| 25.3 | `test_e2e_search.py` | 多种查询类型 → 验证排序与相关性 |
| 25.4 | `test_e2e_rag.py` | RAG 多场景 → 验证答案质量与引用 |
| 25.5 | `test_e2e_permissions.py` | 权限设置 → 无权限文档不出现 |
| 25.8 | `test_performance.py` | SLO 时延门槛 |

## 运行

集成测试默认跳过，需显式开关：

```bash
# 仅跑集成测试（mock 后端，无需容器）
pytest backend/tests/integration -m integration --run-integration

# 跑性能基准
pytest backend/tests/integration -m benchmark --run-integration

# 单元 + 集成全跑
pytest backend/tests --run-integration
```

## 真实容器模式（可选）

```bash
docker compose -f docker-compose.test.yml up -d
pytest backend/tests/integration -m integration --run-integration
docker compose -f docker-compose.test.yml down -v
```

或用 testcontainers 自动管理：

```bash
WIKFORGE_USE_TESTCONTAINERS=1 pytest backend/tests/integration --run-integration
```

注：本目录下大多数测试已通过 mock 后端覆盖端到端逻辑，**无需**容器。容器模式
保留给未来真正需要打通真实 OpenSearch / Qdrant 路径时使用。

## 覆盖率（任务 25.10）

需要先安装 pytest-cov（已在 dev extras 中）：

```bash
pip install -e ".[dev]"
```

然后：

```bash
pytest --cov=app --cov-report=term-missing --cov-report=html backend/tests
open backend/htmlcov/index.html
```

CI 中可直接：

```bash
pytest --cov=app --cov-report=xml --cov-fail-under=60 backend/tests
```

覆盖率配置见 `backend/pyproject.toml` 的 `[tool.coverage.*]` 段。
