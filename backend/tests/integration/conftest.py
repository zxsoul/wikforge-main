"""集成测试共享 fixtures（任务 25.1）。

设计原则：
- 提供两套基础 fixture：``testcontainers``（如可用，自动拉起容器）
  与 ``docker-compose-test``（依赖外部已启动的 docker-compose.test.yml）。
- 在两者都不可用时优雅降级：``compose_endpoints`` 直接返回环境变量配置，
  让 CI / 本地开发者能用任意已就绪的服务。
- 集成测试模块统一在 ``pytest.mark.integration`` 标记下，并通过
  ``--run-integration`` 命令行开关显式启用，避免污染常规单元测试 CI。

Notes
-----
- testcontainers fixture 仅在依赖与 Docker 可用时返回真实端点；
  否则跳过相应测试。
- compose_endpoints 通过环境变量 ``WIKFORGE_TEST_*`` 覆盖默认端点，
  开发者可用：
      docker compose -f docker-compose.test.yml up -d
      WIKFORGE_TEST_POSTGRES_PORT=15432 pytest -m integration
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from typing import Iterator

import pytest


# ─── 命令行选项与标记 ─────────────────────────────────────────────────


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="启用集成测试（默认跳过）",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: 标记为集成测试，依赖外部容器或 testcontainers",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """未加 ``--run-integration`` 时跳过整目录。"""
    if config.getoption("--run-integration"):
        return
    skip_marker = pytest.mark.skip(reason="加 --run-integration 启用集成测试")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_marker)


# ─── 端点数据结构 ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class ServiceEndpoints:
    """测试服务端点集合。"""

    postgres_dsn: str
    redis_url: str
    opensearch_url: str
    qdrant_url: str
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str


def _is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    """探测端口是否可连接。"""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# ─── 基于 docker-compose.test.yml 的端点 ──────────────────────────────


@pytest.fixture(scope="session")
def compose_endpoints() -> ServiceEndpoints:
    """从环境变量读取已启动的测试容器端点。

    用法：
        docker compose -f docker-compose.test.yml up -d
        pytest -m integration --run-integration

    若任何端口不可达，将跳过依赖此 fixture 的测试。
    """
    host = os.getenv("WIKFORGE_TEST_HOST", "127.0.0.1")
    pg_port = int(os.getenv("WIKFORGE_TEST_POSTGRES_PORT", "15432"))
    redis_port = int(os.getenv("WIKFORGE_TEST_REDIS_PORT", "16379"))
    os_port = int(os.getenv("WIKFORGE_TEST_OPENSEARCH_PORT", "19200"))
    qdrant_port = int(os.getenv("WIKFORGE_TEST_QDRANT_PORT", "16333"))
    minio_port = int(os.getenv("WIKFORGE_TEST_MINIO_PORT", "19000"))

    for service, port in [
        ("postgres", pg_port),
        ("redis", redis_port),
        ("opensearch", os_port),
        ("qdrant", qdrant_port),
        ("minio", minio_port),
    ]:
        if not _is_port_open(host, port):
            pytest.skip(
                f"集成测试需要 {service}:{port} 可用，"
                f"先执行 docker compose -f docker-compose.test.yml up -d"
            )

    return ServiceEndpoints(
        postgres_dsn=(
            f"postgresql+asyncpg://wikforge_test:wikforge_test_secret@"
            f"{host}:{pg_port}/wikforge_test"
        ),
        redis_url=f"redis://{host}:{redis_port}/0",
        opensearch_url=f"http://{host}:{os_port}",
        qdrant_url=f"http://{host}:{qdrant_port}",
        minio_endpoint=f"{host}:{minio_port}",
        minio_access_key="minioadmin",
        minio_secret_key="minioadmin",
    )


# ─── 基于 testcontainers 的可选 fixture ───────────────────────────────


@pytest.fixture(scope="session")
def testcontainers_endpoints() -> Iterator[ServiceEndpoints]:
    """使用 testcontainers-python 临时拉起服务（仅当依赖与 Docker 可用时）。

    比 ``compose_endpoints`` 更隔离：每个 pytest 进程独立容器、自动清理。
    缺点：启动慢（首次拉镜像可能数分钟），不适合本地频繁迭代。
    """
    try:
        from testcontainers.compose import DockerCompose
    except ImportError:
        pytest.skip("testcontainers 未安装：pip install testcontainers")

    compose_file = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "docker-compose.test.yml"
        )
    )
    if not os.path.exists(compose_file):
        pytest.skip(f"找不到 {compose_file}")

    with DockerCompose(
        filepath=os.path.dirname(compose_file),
        compose_file_name="docker-compose.test.yml",
        pull=False,
    ) as compose:
        compose.wait_for("http://localhost:19200/_cluster/health")
        yield ServiceEndpoints(
            postgres_dsn=(
                "postgresql+asyncpg://wikforge_test:wikforge_test_secret@"
                "127.0.0.1:15432/wikforge_test"
            ),
            redis_url="redis://127.0.0.1:16379/0",
            opensearch_url="http://127.0.0.1:19200",
            qdrant_url="http://127.0.0.1:16333",
            minio_endpoint="127.0.0.1:19000",
            minio_access_key="minioadmin",
            minio_secret_key="minioadmin",
        )


# ─── 通用：自动选择 testcontainers 或 compose ─────────────────────────


@pytest.fixture(scope="session")
def integration_endpoints(request: pytest.FixtureRequest) -> ServiceEndpoints:
    """根据环境变量自动选择端点提供者。

    - 默认：使用 ``compose_endpoints``（需先 ``docker compose up``）
    - 设置 ``WIKFORGE_USE_TESTCONTAINERS=1``：使用 ``testcontainers_endpoints``
    """
    if os.getenv("WIKFORGE_USE_TESTCONTAINERS") == "1":
        return request.getfixturevalue("testcontainers_endpoints")
    return request.getfixturevalue("compose_endpoints")
