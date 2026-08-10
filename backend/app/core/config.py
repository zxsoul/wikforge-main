"""Application configuration management using Pydantic Settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    APP_NAME: str = "Wikforge"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    API_PREFIX: str = "/api"

    # Server
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # PostgreSQL
    POSTGRES_HOST: str = "postgres"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "wikforge"
    POSTGRES_PASSWORD: str = "wikforge_secret"
    POSTGRES_DB: str = "wikforge"

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # Redis
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""
    REDIS_DB: int = 0

    @property
    def REDIS_URL(self) -> str:
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # OpenSearch
    OPENSEARCH_HOST: str = "opensearch"
    OPENSEARCH_PORT: int = 9200
    OPENSEARCH_USER: str = "admin"
    OPENSEARCH_PASSWORD: str = "Admin@123"
    # 是否启用 HTTPS。默认 false 适配开发环境 (compose 中 plugins.security.disabled=true)。
    # 生产建议启用 SSL + 真实证书,通过 OPENSEARCH_VERIFY_CERTS 控制证书校验。
    OPENSEARCH_USE_SSL: bool = False
    OPENSEARCH_VERIFY_CERTS: bool = False

    @property
    def OPENSEARCH_URL(self) -> str:
        scheme = "https" if self.OPENSEARCH_USE_SSL else "http"
        return f"{scheme}://{self.OPENSEARCH_HOST}:{self.OPENSEARCH_PORT}"

    # Qdrant
    QDRANT_HOST: str = "qdrant"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: str = ""

    # MinIO
    MINIO_HOST: str = "minio"
    MINIO_PORT: int = 9000
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "wikforge-documents"
    MINIO_SECURE: bool = False

    @property
    def MINIO_ENDPOINT(self) -> str:
        return f"{self.MINIO_HOST}:{self.MINIO_PORT}"

    # JWT
    JWT_SECRET_KEY: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # OIDC
    OIDC_DISCOVERY_URL: str = ""
    OIDC_CLIENT_ID: str = ""
    OIDC_CLIENT_SECRET: str = ""
    OIDC_REDIRECT_URI: str = ""

    # LiteLLM
    LITELLM_API_BASE: str = ""
    LITELLM_API_KEY: str = ""
    LITELLM_MODEL: str = "gpt-4o"

    # LLM 调用超时（任务 16.7 / 需求 8.7）
    # ``LLMGateway`` 在构造时未显式传入 ``timeout`` 时使用此值。LLM 调用
    # 超过该时长后，``LLMGateway`` 抛出 ``LLMGatewayError(reason="timeout")``，
    # ``RAGService`` 进一步映射为 ``RAGServiceError(reason="timeout")``，
    # API 层再返回"服务暂时不可用"的中文提示。需求 8.7 要求该值"可配置，
    # 默认 60 秒"——若上层希望对一次性长任务放宽限制，可通过环境变量
    # ``LLM_TIMEOUT`` 调整。
    LLM_TIMEOUT: float = 60.0

    # Embedding (任务 12.3) — Dense 向量生成
    # ``EMBEDDING_MODEL`` 留空表示沿用 ``LITELLM_MODEL``，便于在没有专用 embedding
    # 网关时直接复用主 LiteLLM 配置。生产环境建议显式配置（如
    # ``text-embedding-3-large`` / ``bge-large-zh`` / ``ollama/bge-m3``）。
    EMBEDDING_MODEL: str = ""
    # Dense 向量维度。必须与 Qdrant Collection ``document_chunks.dense.size``
    # 保持一致；超出/不足由 EmbeddingService 自行截断/补零。
    EMBEDDING_DIMENSIONS: int = 1024
    # 单次 LiteLLM aembedding 调用的总超时（秒，含重试前的单次等待）。
    EMBEDDING_TIMEOUT: float = 30.0
    # 单条文本喂给 embedding API 前允许的最大字符数。超过则截断，避免触发模型
    # 上下文上限或导致整批请求被拒。1024-dim 中文 embedding 模型常见上下文 ~8k tokens，
    # 留出余量按字符近似为 6000。
    EMBEDDING_MAX_INPUT_CHARS: int = 6000
    # 单批次失败时的最大重试次数（除首次调用外的额外尝试）。
    EMBEDDING_MAX_RETRIES: int = 2

    # Celery
    CELERY_BROKER_URL: str = ""
    CELERY_RESULT_BACKEND: str = ""

    @property
    def celery_broker_url(self) -> str:
        return self.CELERY_BROKER_URL or self.REDIS_URL

    @property
    def celery_result_backend(self) -> str:
        return self.CELERY_RESULT_BACKEND or self.REDIS_URL

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:80"]

    # Quality thresholds
    QUALITY_FALLBACK_THRESHOLD: float = 0.7
    REVIEW_QUEUE_THRESHOLD: float = 0.7

    # RAG 相似度阈值（任务 16.6 / 需求 8.6）
    # 当问答检索的所有候选 chunk 的相似度分数都低于该阈值时，
    # RAGService 不再调用 LLM，而是直接返回 NO_CONTEXT_MESSAGE，
    # 避免基于低相关上下文产生幻觉回答。范围 0-1，默认 0.5。
    SIMILARITY_THRESHOLD: float = 0.5

    # IK Analyzer 远程词库（任务 13.7）
    # IK 插件配置 ``remote_ext_dict`` / ``remote_ext_stopwords`` 指向
    # ``${API_BASE}/api/ik-dict/custom_main.dic`` 等 URL，插件每 60 秒轮询
    # ``Last-Modified`` / ``ETag``，变更时热加载词典。这里的目录是 API 进程
    # 写入 ``.dic`` 文件的位置，HTTP 路由会从同一目录读取。
    # 默认放在 ``/data/ik-custom-dict``（容器内）；本地开发可指向 tmp 目录。
    IK_DICT_DIR: str = "/data/ik-custom-dict"

    # Universal Parser (LLM 兜底解析) — 任务 10.2
    # PDF/Office 页面光栅化 DPI；DPI 越高图像越清晰，但 LLM 输入 token 消耗也越大。
    UNIVERSAL_PARSER_PAGE_DPI: int = 150
    # LibreOffice headless 转 PDF 的整体超时（秒）。仅作用于单个文档的一次转换。
    UNIVERSAL_PARSER_LIBREOFFICE_TIMEOUT: int = 60
    # 单页 LLM 调用超时 (秒)。qwen-vl-plus 平均 30-60s/页, 慢页 90s+,
    # 所以默认 120s 比早期 60s 更稳; 单文档 20 页时最大约 40 分钟。
    UNIVERSAL_PARSER_PAGE_TIMEOUT: float = 120.0
    # 任务 10.3：单页喂给 LLM 的原始文本最大字符数。原始文本越长越容易触发上下文
    # 溢出 / 让模型偏离图像内容；过短会丢失上下文。3000 字符约等于 ~1500 tokens。
    UNIVERSAL_PARSER_MAX_RAW_TEXT_CHARS: int = 3000

    # 任务 10.7：Universal Parser 模型选择。两者都接受任意 LiteLLM 兼容标识符，例如
    # ``gpt-4o`` / ``gpt-4o-mini`` / ``qwen-vl-max`` / ``minicpm-v`` /
    # ``ollama/minicpm-v:latest`` / ``claude-3-5-sonnet-20241022``。
    # 留空表示沿用 ``LITELLM_MODEL``，由 LLMGateway 决定具体模型。
    # ``UNIVERSAL_PARSER_VISION_MODEL`` 仅作用于带页面图像的多模态调用；
    # ``UNIVERSAL_PARSER_TEXT_MODEL`` 仅作用于无图像时的纯文本兜底调用。
    UNIVERSAL_PARSER_VISION_MODEL: str = ""
    UNIVERSAL_PARSER_TEXT_MODEL: str = ""

    # 任务 10.8：LLM 失败降级时的纯文本固定大小分块字符数。降级路径不再依赖 LLM，
    # 仅按字符数切分原始文本以保证文档至少可被检索。最小值 1（构造函数会兜底）。
    UNIVERSAL_PARSER_FALLBACK_CHUNK_CHARS: int = 500

    # 任务 15.6 / 需求 7：查询增强子模块的全局开关。三者均默认 True，
    # 可按功能独立启用 / 禁用：
    # - ``QUERY_ENHANCEMENT_ENABLE_REWRITE``: LLM 生成 ≤5 个语义改写变体
    # - ``QUERY_ENHANCEMENT_ENABLE_HYDE``: LLM 生成 1-3 个假设文档并嵌入
    # - ``QUERY_ENHANCEMENT_ENABLE_DECOMPOSITION``: 多子问题查询拆为 ≤5 个子查询
    # 关闭对应开关后，``QueryEnhancer.enhance()`` 不会再为该子模块创建 asyncio
    # 任务，也不会触发 LLM / Embedding 调用，便于在故障演练或离线环境下定位问题。
    QUERY_ENHANCEMENT_ENABLE_REWRITE: bool = True
    QUERY_ENHANCEMENT_ENABLE_HYDE: bool = True
    QUERY_ENHANCEMENT_ENABLE_DECOMPOSITION: bool = True

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
