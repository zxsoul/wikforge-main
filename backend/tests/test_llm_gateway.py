"""任务 16.1：LLM Gateway 封装单元测试。

针对 ``app.services.llm_gateway.LLMGateway``，验证其通过 ``openai`` SDK
直连任意 OpenAI 兼容端点（CPA 网关 / OpenAI / DeepSeek / 通义千问等）
时的行为，以及配置透传与错误映射逻辑。

2026-09 重构：不再经过 LiteLLM Proxy / litellm SDK。测试通过 patch
``app.services.llm_gateway.get_openai_client`` 注入可控的客户端 stub，
精确控制 ``chat.completions.create`` 的返回值与异常，无需真实网络。

覆盖点（对应需求 8.1）：
- 配置默认值与显式覆盖（model / api_base / api_key / timeout）
- ``complete`` 把 model / messages / temperature / max_tokens 完整透传
- api_base / api_key 作为客户端级参数传给 ``get_openai_client``
- 不同 provider 的 model 字符串（``gpt-4o`` / ``claude-3-5-sonnet`` /
  ``qwen-vl-max`` / ``ollama/llama3``）原样透传，无需修改业务代码
- 单次调用的 ``model`` 形参可覆盖网关默认 model
- ``complete_multimodal`` 构造图片 + 文本的多模态 messages
- ``stream`` 异步迭代 token
- 上游不支持 ``temperature`` / ``max_tokens`` 时的参数降级重试
- 错误映射：``rate_limit`` / ``auth`` / ``model_unavailable`` / ``timeout``
  / ``unknown`` 全部归一为 ``LLMGatewayError(reason=...)``
- ``LLMResponse`` 把 usage / finish_reason / model 正确回填
"""

from __future__ import annotations

import asyncio
import base64
import sys
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.llm_gateway import LLMGateway, LLMGatewayError, LLMResponse


# ─── openai client stub ───────────────────────────────────────────────


@pytest.fixture
def openai_client_stub():
    """Patch ``get_openai_client``，注入可控的 AsyncOpenAI 客户端 stub。

    yield 出 ``(client, get_client_mock)``：
    - ``client.chat.completions.create`` 是 AsyncMock，可设置返回值/异常；
    - ``get_client_mock`` 可断言客户端以哪个 (api_base, api_key) 被获取。
    """
    client = MagicMock()
    client.chat.completions.create = AsyncMock()
    with patch(
        # embedding_service 在调用时才从 llm_gateway import get_openai_client，
        # 因此 patch 源头对两个模块都生效。
        "app.services.llm_gateway.get_openai_client",
        return_value=client,
    ) as get_client_mock:
        yield client, get_client_mock


def _make_completion_response(
    content: str = "hello",
    *,
    model: str = "gpt-4o",
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> MagicMock:
    """构造一个最小化的 Chat Completions 响应。"""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens

    response = MagicMock()
    response.choices = [choice]
    response.model = model
    response.usage = usage
    return response


def _make_stream_response(chunks: list[str]) -> AsyncIterator[MagicMock]:
    """构造一个异步可迭代的流式响应。"""

    async def _gen():
        for token in chunks:
            chunk = MagicMock()
            delta = MagicMock()
            delta.content = token
            choice = MagicMock()
            choice.delta = delta
            chunk.choices = [choice]
            yield chunk

    return _gen()


def _make_settings(
    model: str = "gpt-4o",
    api_base: str = "https://api.example.com/v1",
    api_key: str = "sk-test-123",
) -> MagicMock:
    """构造符合新配置命名（CHAT_*）的 stub Settings。"""
    return MagicMock(
        CHAT_MODEL=model,
        CHAT_API_BASE=api_base,
        CHAT_API_KEY=api_key,
    )


# ─── 配置默认值 ───────────────────────────────────────────────────────


class TestLLMGatewayConfig:
    """构造函数从 Settings 读取默认值，并允许显式覆盖。"""

    @patch("app.services.llm_gateway.get_settings")
    def test_defaults_from_settings(self, mock_settings):
        """未传入参数时，使用 Settings 中的 Chat 配置。"""
        mock_settings.return_value = _make_settings(
            model="gpt-4o",
            api_base="https://api.openai.com/v1",
            api_key="sk-test",
        )

        gateway = LLMGateway()

        assert gateway.model == "gpt-4o"
        assert gateway.api_base == "https://api.openai.com/v1"
        assert gateway.api_key == "sk-test"
        assert gateway.timeout == 60.0

    @patch("app.services.llm_gateway.get_settings")
    def test_explicit_args_override_settings(self, mock_settings):
        """显式参数优先于 Settings。"""
        mock_settings.return_value = _make_settings(api_base="https://default", api_key="default")

        gateway = LLMGateway(
            model="claude-3-5-sonnet-20241022",
            api_base="https://api.anthropic.com",
            api_key="anthropic-key",
            timeout=15.0,
        )

        assert gateway.model == "claude-3-5-sonnet-20241022"
        assert gateway.api_base == "https://api.anthropic.com"
        assert gateway.api_key == "anthropic-key"
        assert gateway.timeout == 15.0


# ─── complete：参数透传 ──────────────────────────────────────────────


@pytest.fixture
def gateway(openai_client_stub):
    """提供一个用 stub settings + stub openai client 配置好的 gateway。"""
    with patch("app.services.llm_gateway.get_settings") as mock_settings:
        mock_settings.return_value = _make_settings()
        yield LLMGateway()


class TestLLMGatewayComplete:
    """``complete`` 把调用参数透传给 ``chat.completions.create``。"""

    @pytest.mark.asyncio
    async def test_complete_passes_model_and_messages(
        self, gateway: LLMGateway, openai_client_stub
    ):
        """model / messages / temperature / max_tokens 必须全部透传。"""
        client, get_client_mock = openai_client_stub
        client.chat.completions.create.return_value = _make_completion_response(
            content="hi there", model="gpt-4o"
        )

        response = await gateway.complete(
            prompt="What is RAG?",
            system_prompt="You are a helpful assistant.",
            temperature=0.2,
            max_tokens=512,
        )

        # api_base / api_key 是客户端级参数
        get_client_mock.assert_called_with(
            "https://api.example.com/v1", "sk-test-123"
        )

        client.chat.completions.create.assert_awaited_once()
        kwargs = client.chat.completions.create.await_args.kwargs

        assert kwargs["model"] == "gpt-4o"
        assert kwargs["temperature"] == 0.2
        assert kwargs["max_tokens"] == 512
        assert kwargs["timeout"] == 60.0

        # messages 必须包含 system + user 两条
        messages = kwargs["messages"]
        assert messages == [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is RAG?"},
        ]

        # 返回值结构正确
        assert isinstance(response, LLMResponse)
        assert response.content == "hi there"
        assert response.model == "gpt-4o"
        assert response.finish_reason == "stop"
        assert response.usage["total_tokens"] == 15

    @pytest.mark.asyncio
    async def test_complete_without_system_prompt(
        self, gateway: LLMGateway, openai_client_stub
    ):
        """缺省 system_prompt 时，messages 仅包含 user 一条。"""
        client, _ = openai_client_stub
        client.chat.completions.create.return_value = _make_completion_response()

        await gateway.complete(prompt="hello")

        messages = client.chat.completions.create.await_args.kwargs["messages"]
        assert messages == [{"role": "user", "content": "hello"}]

    @pytest.mark.asyncio
    async def test_complete_per_call_model_override(
        self, gateway: LLMGateway, openai_client_stub
    ):
        """单次调用的 ``model`` 可覆盖网关默认值，不修改 ``self.model``。"""
        client, _ = openai_client_stub
        client.chat.completions.create.return_value = _make_completion_response(
            model="ollama/llama3"
        )

        await gateway.complete(prompt="hi", model="ollama/llama3")

        assert client.chat.completions.create.await_args.kwargs["model"] == "ollama/llama3"
        # 网关默认 model 不变
        assert gateway.model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_complete_uses_empty_api_base_and_key_as_is(self, openai_client_stub):
        """空的 api_base / api_key 原样传给 get_openai_client（SDK 用默认端点）。"""
        client, get_client_mock = openai_client_stub
        with patch("app.services.llm_gateway.get_settings") as mock_settings:
            mock_settings.return_value = _make_settings(api_base="", api_key="")
            gw = LLMGateway()

        client.chat.completions.create.return_value = _make_completion_response()
        await gw.complete(prompt="hi")

        get_client_mock.assert_called_with("", "")


# ─── 多 provider 透传（需求 8.1 核心） ────────────────────────────────


class TestLLMGatewayProviderRouting:
    """需求 8.1：不同 provider 的 model 字符串原样透传，无需修改业务代码。"""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "model_id",
        [
            "gpt-4o",  # OpenAI
            "claude-3-5-sonnet-20241022",  # Anthropic Claude (经兼容网关)
            "qwen-vl-max",  # 通义千问
            "deepseek-chat",  # DeepSeek
            "ollama/llama3",  # Ollama 本地
        ],
    )
    async def test_model_string_passes_through(self, openai_client_stub, model_id):
        """各 provider 的 model 字符串原样传给 ``chat.completions.create``。"""
        client, _ = openai_client_stub
        with patch("app.services.llm_gateway.get_settings") as mock_settings:
            mock_settings.return_value = _make_settings(model=model_id, api_base="", api_key="")
            gw = LLMGateway()

        client.chat.completions.create.return_value = _make_completion_response(
            model=model_id
        )

        response = await gw.complete(prompt="ping")

        kwargs = client.chat.completions.create.await_args.kwargs
        assert kwargs["model"] == model_id
        assert response.model == model_id

    @pytest.mark.asyncio
    async def test_switch_provider_via_settings_only(self, openai_client_stub):
        """切换 provider 只需改 Settings，业务代码（complete 调用）保持不变。"""
        client, get_client_mock = openai_client_stub
        # provider A：OpenAI
        with patch("app.services.llm_gateway.get_settings") as mock_settings:
            mock_settings.return_value = _make_settings(model="gpt-4o", api_base="", api_key="")
            gw_a = LLMGateway()

        # provider B：通义千问
        with patch("app.services.llm_gateway.get_settings") as mock_settings:
            mock_settings.return_value = _make_settings(
                model="qwen-max",
                api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_key="ali-key",
            )
            gw_b = LLMGateway()

        client.chat.completions.create.side_effect = [
            _make_completion_response(model="gpt-4o"),
            _make_completion_response(model="qwen-max"),
        ]

        # 完全相同的业务代码
        await gw_a.complete(prompt="hi")
        await gw_b.complete(prompt="hi")

        first_call = client.chat.completions.create.await_args_list[0].kwargs
        second_call = client.chat.completions.create.await_args_list[1].kwargs

        assert first_call["model"] == "gpt-4o"
        assert second_call["model"] == "qwen-max"

        # 第二次以百炼端点获取客户端
        assert get_client_mock.call_args_list[1].args == (
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "ali-key",
        )


# ─── 多模态 ───────────────────────────────────────────────────────────


class TestLLMGatewayMultimodal:
    """``complete_multimodal`` 把图像编码为 data URL 并构造混合 content。"""

    @pytest.mark.asyncio
    async def test_multimodal_messages_include_image_and_text(
        self, gateway: LLMGateway, openai_client_stub
    ):
        """图像在前、文本在后；图像被 base64 + data URL 包装。"""
        client, _ = openai_client_stub
        client.chat.completions.create.return_value = _make_completion_response(
            content="image described", model="gpt-4o"
        )

        image_bytes = b"\x89PNG\r\n\x1a\n-fake"
        await gateway.complete_multimodal(
            prompt="Describe this",
            images=[image_bytes],
            system_prompt="You see images.",
            image_mime_type="image/png",
        )

        kwargs = client.chat.completions.create.await_args.kwargs
        messages = kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "You see images."}

        user_msg = messages[1]
        assert user_msg["role"] == "user"
        content = user_msg["content"]
        assert isinstance(content, list)
        assert len(content) == 2

        # 图像段
        assert content[0]["type"] == "image_url"
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        assert content[0]["image_url"]["url"] == f"data:image/png;base64,{b64}"

        # 文本段
        assert content[1] == {"type": "text", "text": "Describe this"}

    @pytest.mark.asyncio
    async def test_multimodal_per_call_model_override(
        self, gateway: LLMGateway, openai_client_stub
    ):
        """多模态调用支持单次切换到 vision 模型。"""
        client, _ = openai_client_stub
        client.chat.completions.create.return_value = _make_completion_response(
            model="qwen-vl-max"
        )

        await gateway.complete_multimodal(
            prompt="ocr",
            images=[b"png-bytes"],
            model="qwen-vl-max",
        )

        assert client.chat.completions.create.await_args.kwargs["model"] == "qwen-vl-max"
        # 网关默认仍为 gpt-4o
        assert gateway.model == "gpt-4o"


# ─── 流式 ─────────────────────────────────────────────────────────────


class TestLLMGatewayStream:
    """``stream`` 应异步迭代 token，并以 stream=True 调用。"""

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self, gateway: LLMGateway, openai_client_stub):
        """token 应按 chunk 顺序产出，跳过空 delta。"""
        client, _ = openai_client_stub
        client.chat.completions.create.return_value = _make_stream_response(
            ["Hello", ", ", "world", "!"]
        )

        tokens: list[str] = []
        async for tok in gateway.stream(prompt="hi"):
            tokens.append(tok)

        assert tokens == ["Hello", ", ", "world", "!"]
        kwargs = client.chat.completions.create.await_args.kwargs
        assert kwargs["stream"] is True
        assert kwargs["model"] == "gpt-4o"


# ─── 参数降级（替代 LiteLLM drop_params） ─────────────────────────────


class TestUnsupportedParamFallback:
    """上游 400 拒绝 ``temperature`` / ``max_tokens`` 时自动降级重试。"""

    @staticmethod
    def _bad_request(message: str):
        import httpx
        import openai

        request = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
        response = httpx.Response(400, request=request)
        return openai.BadRequestError(message, response=response, body=None)

    @pytest.mark.asyncio
    async def test_temperature_dropped_on_400(
        self, gateway: LLMGateway, openai_client_stub
    ):
        """报 temperature 不支持 → 移除该参数后重试并成功。"""
        client, _ = openai_client_stub
        client.chat.completions.create.side_effect = [
            self._bad_request("Unsupported value: 'temperature' does not support 0.2"),
            _make_completion_response(content="ok"),
        ]

        response = await gateway.complete(prompt="hi", temperature=0.2)

        assert response.content == "ok"
        assert client.chat.completions.create.await_count == 2
        second_kwargs = client.chat.completions.create.await_args_list[1].kwargs
        assert "temperature" not in second_kwargs

    @pytest.mark.asyncio
    async def test_max_tokens_switched_on_400(
        self, gateway: LLMGateway, openai_client_stub
    ):
        """报 max_tokens 不支持 → 改用 max_completion_tokens 重试。"""
        client, _ = openai_client_stub
        client.chat.completions.create.side_effect = [
            self._bad_request("Unsupported parameter: 'max_tokens' is not supported"),
            _make_completion_response(content="ok"),
        ]

        response = await gateway.complete(prompt="hi", max_tokens=512)

        assert response.content == "ok"
        second_kwargs = client.chat.completions.create.await_args_list[1].kwargs
        assert "max_tokens" not in second_kwargs
        assert second_kwargs["max_completion_tokens"] == 512

    @pytest.mark.asyncio
    async def test_other_400_raises(
        self, gateway: LLMGateway, openai_client_stub
    ):
        """与参数无关的 400 不降级，原样映射为 LLMGatewayError。"""
        client, _ = openai_client_stub
        client.chat.completions.create.side_effect = self._bad_request(
            "Invalid messages format"
        )

        with pytest.raises(LLMGatewayError):
            await gateway.complete(prompt="hi")
        assert client.chat.completions.create.await_count == 1


# ─── 错误映射 ─────────────────────────────────────────────────────────


class TestLLMGatewayErrorMapping:
    """异常被归一为 ``LLMGatewayError`` 并设置正确的 ``reason``。"""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "raised, expected_reason",
        [
            (RuntimeError("HTTP 429 rate_limit exceeded"), "rate_limit"),
            (RuntimeError("AuthenticationError 401: invalid api key"), "auth"),
            (RuntimeError("Model not found 404"), "model_unavailable"),
            (RuntimeError("Connection timeout while reading"), "timeout"),
            (RuntimeError("some unexpected boom"), "unknown"),
        ],
    )
    async def test_error_classification(
        self,
        gateway: LLMGateway,
        openai_client_stub,
        raised: Exception,
        expected_reason: str,
    ):
        """常见上游错误信息应映射到对应 ``reason``。"""
        client, _ = openai_client_stub
        client.chat.completions.create.side_effect = raised

        with pytest.raises(LLMGatewayError) as exc_info:
            await gateway.complete(prompt="hi")

        assert exc_info.value.reason == expected_reason

    @pytest.mark.asyncio
    async def test_asyncio_timeout_maps_to_timeout_reason(
        self, gateway: LLMGateway, openai_client_stub
    ):
        """asyncio.wait_for 抛 TimeoutError → reason='timeout'。"""
        client, _ = openai_client_stub

        async def _hang(**_kwargs):
            await asyncio.sleep(10)
            return _make_completion_response()

        client.chat.completions.create.side_effect = _hang
        gateway.timeout = 0.05  # 收紧到 50ms 触发 wait_for 超时

        with pytest.raises(LLMGatewayError) as exc_info:
            await gateway.complete(prompt="hi")

        assert exc_info.value.reason == "timeout"

    @pytest.mark.asyncio
    async def test_missing_openai_raises_gateway_error(self):
        """openai 包未安装时，应抛 ``LLMGatewayError`` 而非 ImportError。"""
        with patch("app.services.llm_gateway.get_settings") as mock_settings:
            mock_settings.return_value = _make_settings(
                api_base="https://no-cache-test.invalid/v1",
                api_key="sk-no-cache",
            )
            gw = LLMGateway()

        original = sys.modules.get("openai")
        sys.modules["openai"] = None  # import 时触发 ImportError
        try:
            with pytest.raises(LLMGatewayError) as exc_info:
                await gw.complete(prompt="hi")
            assert "openai" in str(exc_info.value).lower()
        finally:
            if original is not None:
                sys.modules["openai"] = original
            else:
                sys.modules.pop("openai", None)

    @pytest.mark.asyncio
    async def test_stream_error_classification(
        self, gateway: LLMGateway, openai_client_stub
    ):
        """流式接口的错误也应被归一为 ``LLMGatewayError``。"""
        client, _ = openai_client_stub
        client.chat.completions.create.side_effect = RuntimeError("HTTP 429 rate_limit")

        with pytest.raises(LLMGatewayError) as exc_info:
            async for _ in gateway.stream(prompt="hi"):
                pass

        assert exc_info.value.reason == "rate_limit"
