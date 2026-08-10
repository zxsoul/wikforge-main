"""Admin API：Universal Parser 运行期模型配置（任务 10.7）。

仅暴露只读的 ``GET /api/admin/universal-parser/config``，让管理后台可以渲染
模型下拉列表 / 展示当前生效的模型。运行期切换模型仍然走标准的 ``.env`` 重载
流程（重启 worker），不在这里提供写接口。
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.services.universal_parser import UniversalParser

router = APIRouter(
    prefix="/api/admin/universal-parser",
    tags=["admin-universal-parser"],
)


class UniversalParserConfigResponse(BaseModel):
    """Universal Parser 当前生效的模型配置 + 已知模型目录。

    - ``vision_model`` / ``text_model`` 直接来自 ``settings``，空字符串表示
      “未覆盖、使用 ``LITELLM_MODEL`` 默认值”，UI 应据此展示 “(默认)” 占位。
    - ``known_vision_models`` / ``known_text_models`` 是信息性目录，UI 用来
      构建下拉列表；运行期实际派发的模型不限于此列表（LiteLLM 兼容即可）。
    """

    vision_model: str = Field(
        ...,
        description="多模态调用使用的模型；空字符串表示沿用 LITELLM_MODEL 默认值。",
    )
    text_model: str = Field(
        ...,
        description="纯文本兜底调用使用的模型；空字符串表示沿用 LITELLM_MODEL 默认值。",
    )
    known_vision_models: list[str] = Field(
        ...,
        description="已知的多模态模型标识符目录，仅作 UI 提示，不强制校验。",
    )
    known_text_models: list[str] = Field(
        ...,
        description="已知的纯文本模型标识符目录，仅作 UI 提示，不强制校验。",
    )


@router.get("/config", response_model=UniversalParserConfigResponse)
async def get_universal_parser_config() -> UniversalParserConfigResponse:
    """返回 Universal Parser 当前生效的模型配置与已知模型目录（任务 10.7）。

    纯只读：从 ``Settings`` 读取，无副作用、无网络调用。运行期切换模型请通过
    修改 ``.env`` / 环境变量后重启 worker 完成。
    """
    settings = get_settings()
    vision_setting = getattr(settings, "UNIVERSAL_PARSER_VISION_MODEL", "") or ""
    text_setting = getattr(settings, "UNIVERSAL_PARSER_TEXT_MODEL", "") or ""

    return UniversalParserConfigResponse(
        vision_model=vision_setting,
        text_model=text_setting,
        known_vision_models=list(UniversalParser.KNOWN_VISION_MODELS),
        known_text_models=list(UniversalParser.KNOWN_TEXT_MODELS),
    )
