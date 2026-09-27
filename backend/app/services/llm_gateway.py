"""LLM Gateway: Unified interface for LLM access via OpenAI-compatible APIs.

直连架构（2026-09 重构）
------------------------
早期版本通过独立部署的 LiteLLM Proxy 容器做多模型路由，该容器常驻 ~2GB
内存，对只使用 1~2 个上游的部署是纯浪费。现改为在后端进程内通过
``openai`` SDK 直连任意 OpenAI 兼容端点（CPA 网关 / OpenAI / DeepSeek /
阿里百炼兼容模式等），功能等价、内存归零。

Provides:
- LLMGateway: Async wrapper around the OpenAI-compatible Chat Completions API
- Multimodal support (text + image inputs)
- Streaming support via async generator
- Timeout handling and error management
- Unsupported-parameter fallback (e.g. gpt-5 系列只接受 temperature=1)
"""

import asyncio
import base64
import logging
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

from app.core.config import get_settings

logger = logging.getLogger(__name__)


#: 默认 LLM 调用超时时间（秒）。
#:
#: 任务 16.7 / 需求 8.7：默认 60 秒。``LLMGateway`` 在构造时未显式传入
#: ``timeout`` 时，会优先读取 ``Settings.LLM_TIMEOUT``，读取失败（例如测试
#: 中使用未列出该字段的 MagicMock 配置）则回退到这里的常量。这样既支持
#: 通过环境变量调整，又保证旧调用方在最小化配置下也能工作。
DEFAULT_LLM_TIMEOUT = 60.0


@dataclass
class LLMResponse:
    """Response from an LLM call.

    Attributes:
        content: The generated text content
        model: The model that generated the response
        usage: Token usage information (prompt_tokens, completion_tokens, total_tokens)
        finish_reason: Why the generation stopped ("stop", "length", "error")
    """

    content: str = ""
    model: str = ""
    usage: dict = field(default_factory=dict)
    finish_reason: str = "stop"


class LLMGatewayError(Exception):
    """Raised when an LLM call fails."""

    def __init__(self, message: str, reason: str = "unknown"):
        """Initialize LLMGatewayError.

        Args:
            message: Human-readable error description
            reason: Machine-readable reason code:
                - "timeout": Call exceeded timeout
                - "rate_limit": Rate limit exceeded
                - "auth": Authentication failed
                - "model_unavailable": Model not available
                - "unknown": Unknown error
        """
        super().__init__(message)
        self.reason = reason


#: (base_url, api_key) -> AsyncOpenAI 客户端缓存。
#: httpx 连接池随客户端复用，避免每次调用都重建 TCP/TLS 连接。
_client_cache: dict[tuple[str, str], object] = {}


def get_openai_client(api_base: str, api_key: str):
    """Return a cached ``openai.AsyncOpenAI`` client for the given endpoint.

    供 ``LLMGateway`` 与 ``EmbeddingService`` 共用：同一上游端点在所有
    调用方之间共享同一个客户端（及其 httpx 连接池）。
    """
    key = (api_base or "", api_key or "")
    client = _client_cache.get(key)
    if client is None:
        try:
            import openai
        except ImportError as exc:
            raise LLMGatewayError(
                "openai package is not installed",
                reason="unknown",
            ) from exc
        client = openai.AsyncOpenAI(
            base_url=api_base or None,
            api_key=api_key or None,
            # 客户端自身超时由每次调用的 timeout 参数控制，这里不重复设置。
            max_retries=0,  # 重试策略由调用方（EmbeddingService 等）自行实现
        )
        _client_cache[key] = client
    return client


def _classify_error(error_msg: str) -> str:
    """Map a raw upstream error message to an ``LLMGatewayError.reason`` code."""
    msg = error_msg.lower()
    if "rate_limit" in msg or "429" in msg:
        return "rate_limit"
    if "auth" in msg or "401" in msg or "403" in msg or "api key" in msg:
        return "auth"
    if "not found" in msg or "404" in msg:
        return "model_unavailable"
    if "timeout" in msg or "timed out" in msg:
        return "timeout"
    return "unknown"


class LLMGateway:
    """Unified LLM gateway over any OpenAI-compatible Chat Completions API.

    Supports:
    - CPA 网关（gpt-5.5 等，默认上游）
    - OpenAI (GPT-4o, GPT-4-turbo, etc.)
    - DeepSeek / 通义千问 / Kimi 等一切 OpenAI 兼容端点

    Usage:
        gateway = LLMGateway(model="gpt-5.5")
        response = await gateway.complete("Describe this document")
        response = await gateway.complete_multimodal("Describe this image", [image_bytes])
    """

    # Supported multimodal models for reference
    MULTIMODAL_MODELS = [
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-4-vision-preview",
        "gpt-5.5",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-5-sonnet-20241022",
        "qwen-vl-max",
        "qwen-vl-plus",
        "minicpm-v",
    ]

    def __init__(
        self,
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
    ):
        """Initialize the LLM Gateway.

        Args:
            model: Model identifier (e.g., "gpt-5.5", "qwen-vl-max").
                   Defaults to settings.CHAT_MODEL.
            api_base: API base URL. Defaults to settings.CHAT_API_BASE.
            api_key: API key. Defaults to settings.CHAT_API_KEY.
            timeout: 单次 LLM 调用的超时时间（秒）。``None`` 时回落到
                ``Settings.LLM_TIMEOUT``（默认 60s，对应需求 8.7）；超过该
                时长后抛 ``LLMGatewayError(reason="timeout")``，由调用方映射
                为对用户的友好错误提示。
        """
        settings = get_settings()
        self.model = model or settings.CHAT_MODEL
        self.api_base = api_base or settings.CHAT_API_BASE
        self.api_key = api_key or settings.CHAT_API_KEY
        # 任务 16.7 / 需求 8.7：默认 60 秒、可通过 ``LLM_TIMEOUT`` 环境变量
        # 调整；显式传入的 ``timeout`` 始终优先于 Settings。
        self.timeout = self._resolve_timeout(timeout, settings)

    @staticmethod
    def _resolve_timeout(explicit: float | None, settings) -> float:
        """决定本实例最终使用的 LLM 调用超时（秒）。

        优先级：构造函数显式传入 > ``Settings.LLM_TIMEOUT`` > 模块默认。

        Settings 上不存在该字段，或字段值不是合法数值（例如测试用
        ``MagicMock`` 未列出该字段、或环境变量给了非数值），都会退回到
        :data:`DEFAULT_LLM_TIMEOUT`，确保 LLMGateway 在最小化配置下也能
        初始化成功。这里做严格的类型检查（而非 try/except 包 ``float()``），
        是因为 ``MagicMock`` 实例的 ``__float__`` 默认会返回 ``1.0``，依赖
        异常分支会得到错误的超时值。
        """
        if explicit is not None:
            return float(explicit)
        value = getattr(settings, "LLM_TIMEOUT", None)
        # 排除 bool（``isinstance(True, int)`` 为真）以及 MagicMock 等非数值。
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return DEFAULT_LLM_TIMEOUT
        if value <= 0:
            return DEFAULT_LLM_TIMEOUT
        return float(value)

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 1,
        max_tokens: int = 4096,
        model: str | None = None,
    ) -> LLMResponse:
        """Make a text-only completion call.

        Args:
            prompt: User prompt text
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            model: Optional per-call model override. When ``None`` the gateway's
                configured ``self.model`` is used.

        Returns:
            LLMResponse with generated content

        Raises:
            LLMGatewayError: If the call fails or times out
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return await self._call_chat_api(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
        )

    async def complete_multimodal(
        self,
        prompt: str,
        images: list[bytes],
        system_prompt: str | None = None,
        temperature: float = 1,
        max_tokens: int = 4096,
        image_mime_type: str = "image/png",
        model: str | None = None,
    ) -> LLMResponse:
        """Make a multimodal completion call with text + images.

        Args:
            prompt: User prompt text
            images: List of image bytes to include
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            image_mime_type: MIME type of the images (default "image/png")
            model: Optional per-call model override (e.g. swap to a vision-capable
                model for image inputs without re-instantiating the gateway).
                When ``None`` the gateway's configured ``self.model`` is used.

        Returns:
            LLMResponse with generated content

        Raises:
            LLMGatewayError: If the call fails or times out
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Build multimodal content array
        content: list[dict] = []

        # Add images first
        for image_data in images:
            b64_image = base64.b64encode(image_data).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{image_mime_type};base64,{b64_image}",
                },
            })

        # Add text prompt
        content.append({"type": "text", "text": prompt})

        messages.append({"role": "user", "content": content})

        return await self._call_chat_api(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
        )

    async def stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 1,
        max_tokens: int = 4096,
    ) -> AsyncGenerator[str, None]:
        """Stream a text completion, yielding tokens as they arrive.

        Args:
            prompt: User prompt text
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate

        Yields:
            Individual tokens/chunks as they are generated

        Raises:
            LLMGatewayError: If the call fails or times out
        """
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        start_time = time.perf_counter()

        try:
            client = get_openai_client(self.api_base, self.api_key)
            kwargs = self._build_kwargs(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                model=self.model,
            )
            kwargs["stream"] = True

            logger.info(
                "LLM stream started: model=%s messages=%d prompt_len=%d timeout=%.1f",
                self.model,
                len(messages),
                len(prompt),
                self.timeout,
            )
            response = await asyncio.wait_for(
                self._create_with_param_fallback(client, kwargs),
                timeout=self.timeout + 5,
            )

            token_count = 0
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    token_count += 1
                    yield chunk.choices[0].delta.content

            logger.info(
                "LLM stream completed: model=%s tokens=%d elapsed_ms=%d",
                self.model,
                token_count,
                int((time.perf_counter() - start_time) * 1000),
            )

        except asyncio.TimeoutError:
            logger.error(f"LLM stream timed out after {self.timeout}s (model={self.model})")
            raise LLMGatewayError(
                f"LLM stream timed out after {self.timeout} seconds",
                reason="timeout",
            )
        except LLMGatewayError:
            raise
        except Exception as e:
            error_msg = str(e)
            reason = _classify_error(error_msg)
            logger.error(f"LLM stream failed (model={self.model}): {error_msg}")
            raise LLMGatewayError(
                f"LLM stream failed: {error_msg}",
                reason=reason,
            )

    def _build_kwargs(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        model: str,
    ) -> dict:
        """Assemble Chat Completions request parameters."""
        return {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": self.timeout,
        }

    async def _create_with_param_fallback(self, client, kwargs: dict):
        """Call ``chat.completions.create`` with unsupported-parameter fallback.

        替代原 LiteLLM ``drop_params=True`` 的行为：部分上游（如 gpt-5 系列）
        只接受 ``temperature=1``，或要求用 ``max_completion_tokens`` 取代
        ``max_tokens``。直接透传会被 400 拒绝，这里按报错信息逐步降级：

        1. 报 ``max_tokens`` 不支持 → 改用 ``max_completion_tokens``
        2. 报 ``temperature`` 不支持 → 移除该参数（用上游默认值）

        每个参数最多降级一次，其余错误原样抛出。
        """
        try:
            import openai
        except ImportError as exc:
            raise LLMGatewayError("openai package is not installed", reason="unknown") from exc

        attempt_kwargs = dict(kwargs)
        for _ in range(3):  # 原始调用 + 最多两次参数降级
            try:
                return await client.chat.completions.create(**attempt_kwargs)
            except openai.BadRequestError as exc:
                msg = str(exc).lower()
                if "max_tokens" in msg and "max_tokens" in attempt_kwargs:
                    logger.warning(
                        "upstream rejects max_tokens, retrying with max_completion_tokens: %s",
                        exc,
                    )
                    attempt_kwargs["max_completion_tokens"] = attempt_kwargs.pop("max_tokens")
                    continue
                if "temperature" in msg and "temperature" in attempt_kwargs:
                    logger.warning(
                        "upstream rejects temperature, retrying without it: %s",
                        exc,
                    )
                    attempt_kwargs.pop("temperature")
                    continue
                raise

    async def _call_chat_api(
        self,
        messages: list[dict],
        temperature: float = 1,
        max_tokens: int = 4096,
        model: str | None = None,
    ) -> LLMResponse:
        """Internal method to call the Chat Completions API with timeout handling.

        Args:
            messages: Chat messages in OpenAI format
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            model: Optional per-call model override.

        Returns:
            LLMResponse

        Raises:
            LLMGatewayError: On failure or timeout
        """
        effective_model = model or self.model
        start_time = time.perf_counter()
        try:
            client = get_openai_client(self.api_base, self.api_key)
            kwargs = self._build_kwargs(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                model=effective_model,
            )

            # 不记录 messages 原文，只记录数量和长度，避免日志中出现用户问题或知识库片段。
            prompt_chars = 0
            for message in messages:
                content = message.get("content", "")
                if isinstance(content, str):
                    prompt_chars += len(content)
                elif isinstance(content, list):
                    # 多模态消息里可能包含 base64 图片，统计文本部分即可，避免
                    # 为了日志长度把大图片内容转成字符串。
                    prompt_chars += sum(
                        len(str(part.get("text", "")))
                        for part in content
                        if isinstance(part, dict) and part.get("type") == "text"
                    )
                else:
                    prompt_chars += len(str(content))
            logger.info(
                "LLM call started: model=%s messages=%d prompt_chars=%d timeout=%.1f",
                effective_model,
                len(messages),
                prompt_chars,
                self.timeout,
            )

            # Use asyncio timeout as additional safety net
            response = await asyncio.wait_for(
                self._create_with_param_fallback(client, kwargs),
                timeout=self.timeout + 5,  # Extra 5s buffer beyond the SDK's own timeout
            )

            # Extract response content
            choice = response.choices[0]
            content = choice.message.content or ""
            finish_reason = choice.finish_reason or "stop"

            usage = {}
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            logger.info(
                "LLM call completed: model=%s finish_reason=%s total_tokens=%s elapsed_ms=%d",
                response.model or effective_model,
                finish_reason,
                usage.get("total_tokens"),
                int((time.perf_counter() - start_time) * 1000),
            )

            return LLMResponse(
                content=content,
                model=response.model or effective_model,
                usage=usage,
                finish_reason=finish_reason,
            )

        except asyncio.TimeoutError:
            logger.error(f"LLM call timed out after {self.timeout}s (model={effective_model})")
            raise LLMGatewayError(
                f"LLM call timed out after {self.timeout} seconds",
                reason="timeout",
            )
        except LLMGatewayError:
            raise
        except Exception as e:
            error_msg = str(e)
            reason = _classify_error(error_msg)
            logger.error(f"LLM call failed (model={effective_model}): {error_msg}")
            raise LLMGatewayError(
                f"LLM call failed: {error_msg}",
                reason=reason,
            )
