"""Universal Parser 模型配置 API 测试（任务 10.7）。

覆盖 ``GET /api/admin/universal-parser/config`` 的两类形态：
- 未设置环境变量 → ``vision_model`` / ``text_model`` 为空字符串。
- 设置环境变量 → 返回该值，并附带 ``known_*_models`` 目录。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.admin_universal_parser import router as admin_universal_parser_router
from app.services.universal_parser import UniversalParser


@pytest.fixture
def client():
    application = FastAPI()
    application.include_router(admin_universal_parser_router)
    return TestClient(application)


def _make_settings_mock(*, vision: str = "", text: str = "") -> MagicMock:
    return MagicMock(
        UNIVERSAL_PARSER_VISION_MODEL=vision,
        UNIVERSAL_PARSER_TEXT_MODEL=text,
    )


class TestGetUniversalParserConfig:
    def test_returns_empty_strings_when_unset(self, client):
        with patch(
            "app.api.admin_universal_parser.get_settings",
            return_value=_make_settings_mock(),
        ):
            response = client.get("/api/admin/universal-parser/config")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["vision_model"] == ""
        assert body["text_model"] == ""
        # 已知模型目录必须是非空 list
        assert isinstance(body["known_vision_models"], list)
        assert isinstance(body["known_text_models"], list)
        assert len(body["known_vision_models"]) > 0
        assert len(body["known_text_models"]) > 0

    def test_returns_configured_models(self, client):
        with patch(
            "app.api.admin_universal_parser.get_settings",
            return_value=_make_settings_mock(
                vision="qwen-vl-max",
                text="qwen-max",
            ),
        ):
            response = client.get("/api/admin/universal-parser/config")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["vision_model"] == "qwen-vl-max"
        assert body["text_model"] == "qwen-max"

    def test_known_lists_match_universal_parser_catalog(self, client):
        """``known_*_models`` 必须与 ``UniversalParser`` 的常量保持一致。"""
        with patch(
            "app.api.admin_universal_parser.get_settings",
            return_value=_make_settings_mock(),
        ):
            response = client.get("/api/admin/universal-parser/config")
        body = response.json()
        assert body["known_vision_models"] == list(UniversalParser.KNOWN_VISION_MODELS)
        assert body["known_text_models"] == list(UniversalParser.KNOWN_TEXT_MODELS)

    def test_endpoint_is_get_only(self, client):
        """不应提供写接口（任务 10.7 明确不实现 POST/PUT）。"""
        with patch(
            "app.api.admin_universal_parser.get_settings",
            return_value=_make_settings_mock(),
        ):
            post = client.post(
                "/api/admin/universal-parser/config",
                json={"vision_model": "x"},
            )
            put = client.put(
                "/api/admin/universal-parser/config",
                json={"vision_model": "x"},
            )
        # 405 Method Not Allowed
        assert post.status_code == 405
        assert put.status_code == 405
