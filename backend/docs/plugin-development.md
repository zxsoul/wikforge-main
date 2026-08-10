# 解析器插件开发指南

## 概述

Wikforge 使用插件化架构来支持不同文件格式的解析。每个解析器插件负责将特定格式的文件转换为统一的中间表示（Intermediate Representation），供后续的清洗、分块、向量化流程使用。

## 插件接口

所有解析器插件必须实现 `ParserPlugin` 协议：

```python
from app.services.parsers.base import ParserPlugin, ParsedDocument, Block, Asset, ParseError

class MyCustomParser:
    """自定义解析器示例"""

    name: str = "my-custom-parser"
    supported_extensions: list[str] = ["xyz", "abc"]
    priority: int = 100  # 数值越大优先级越高

    def can_parse(self, file_path: str, mime_type: str) -> bool:
        """判断是否能处理该文件"""
        import os
        ext = os.path.splitext(file_path)[1].lower().lstrip(".")
        return ext in self.supported_extensions

    async def parse(self, file_path: str) -> ParsedDocument:
        """解析文件，返回中间表示"""
        # 实现解析逻辑
        blocks = []
        # ... 提取内容 ...
        return ParsedDocument(
            blocks=blocks,
            metadata={"source": file_path},
            assets=[],
        )
```

## 数据结构

### Block（内容块）

```python
@dataclass
class Block:
    type: str       # "paragraph" | "heading" | "table" | "image" | "formula" | "list"
    text: str       # 文本内容
    bbox: tuple[float, float, float, float] | None  # 位置信息 (x0, y0, x1, y1)，归一化到 0-1
    page_number: int  # 页码（从 1 开始）
    style: dict     # 样式信息（font_size, bold, italic, heading_level 等）
    raw: dict       # 原始解析数据（调试用）
```

### Asset（附件资源）

```python
@dataclass
class Asset:
    id: str         # 唯一标识
    type: str       # "image" | "formula" | "diagram"
    data: bytes     # 二进制数据
    mime_type: str  # MIME 类型
    page_number: int
    bbox: tuple[float, float, float, float] | None
    description: str  # 可选的文本描述
```

### ParsedDocument（解析结果）

```python
@dataclass
class ParsedDocument:
    blocks: list[Block]    # 有序内容块列表
    metadata: dict         # 文件级元数据（page_count, author, title 等）
    assets: list[Asset]    # 提取的二进制资源
```

## 错误处理

使用 `ParseError` 异常来报告解析失败：

```python
from app.services.parsers.base import ParseError

# 文件损坏
raise ParseError("文件格式无效", reason="corrupted")

# 密码保护
raise ParseError("文件需要密码", reason="password_protected")

# 内容为空
raise ParseError("无法提取任何内容", reason="empty")

# 格式版本不支持
raise ParseError("不支持的文件版本", reason="unsupported_version")
```

`reason` 字段用于区分永久性错误（不重试）和临时性错误（可重试）：
- `corrupted`、`password_protected`、`empty`：永久性错误，不会重试
- `unknown`：临时性错误，会按指数退避策略重试（最多 3 次）

## 注册插件

### 方式一：数据库配置（推荐）

在 `parser_plugin_configs` 表中添加记录：

```sql
INSERT INTO parser_plugin_configs (id, name, import_path, supported_extensions, priority, enabled, config)
VALUES (
    gen_random_uuid(),
    'my-custom-parser',
    'app.services.parsers.my_parser.MyCustomParser',
    '["xyz", "abc"]',
    100,
    true,
    '{}'
);
```

系统启动时会自动加载所有 `enabled=true` 的插件。

### 方式二：代码注册

```python
from app.services.parsers.registry import get_parser_registry

registry = get_parser_registry()
registry.register(MyCustomParser())
```

### 方式三：热加载

通过管理 API 或直接修改数据库配置后，调用热加载：

```python
registry = get_parser_registry()
registry.reload_from_configs(configs)  # configs 从数据库读取
```

## 完整示例：CSV 解析器

```python
"""CSV 文件解析器插件示例"""

import csv
import os

from app.services.parsers.base import Block, ParsedDocument, ParseError


class CsvParser:
    """CSV 文件解析器

    将 CSV 文件转换为表格类型的 Block。
    """

    name: str = "csv-parser"
    supported_extensions: list[str] = ["csv", "tsv"]
    priority: int = 80

    def can_parse(self, file_path: str, mime_type: str) -> bool:
        ext = os.path.splitext(file_path)[1].lower().lstrip(".")
        return ext in self.supported_extensions or mime_type in (
            "text/csv",
            "text/tab-separated-values",
        )

    async def parse(self, file_path: str) -> ParsedDocument:
        if not os.path.exists(file_path):
            raise ParseError(f"文件不存在: {file_path}", reason="corrupted")

        try:
            ext = os.path.splitext(file_path)[1].lower().lstrip(".")
            delimiter = "\t" if ext == "tsv" else ","

            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter=delimiter)
                rows = list(reader)
        except Exception as e:
            raise ParseError(f"无法读取 CSV 文件: {e}", reason="corrupted")

        if not rows:
            raise ParseError("CSV 文件为空", reason="empty")

        # 转换为 Markdown 表格
        md_rows = []
        for i, row in enumerate(rows):
            md_rows.append("| " + " | ".join(row) + " |")
            if i == 0:
                md_rows.append("| " + " | ".join(["---"] * len(row)) + " |")

        table_text = "\n".join(md_rows)

        blocks = [
            Block(
                type="table",
                text=table_text,
                page_number=1,
                style={"rows": len(rows), "cols": len(rows[0]) if rows else 0},
                raw={},
            )
        ]

        metadata = {
            "source": file_path,
            "page_count": 1,
            "row_count": len(rows),
            "col_count": len(rows[0]) if rows else 0,
        }

        return ParsedDocument(blocks=blocks, metadata=metadata, assets=[])
```

## 插件优先级

当多个插件支持同一文件格式时，系统按 `priority` 从高到低选择第一个 `can_parse` 返回 `True` 的插件。

建议的优先级范围：
- 100+：专用高质量解析器（如 Marker PDF 解析器）
- 50-99：通用解析器（如纯文本解析器）
- 1-49：兜底解析器

## 测试插件

```python
import pytest
from app.services.parsers.my_parser import MyCustomParser

@pytest.mark.asyncio
async def test_my_parser_basic():
    parser = MyCustomParser()

    # 测试 can_parse
    assert parser.can_parse("test.xyz", "") is True
    assert parser.can_parse("test.pdf", "") is False

    # 测试解析
    result = await parser.parse("tests/fixtures/sample.xyz")
    assert len(result.blocks) > 0
    assert result.blocks[0].type in ("paragraph", "heading", "table")

@pytest.mark.asyncio
async def test_my_parser_corrupted_file():
    parser = MyCustomParser()
    with pytest.raises(ParseError) as exc_info:
        await parser.parse("tests/fixtures/corrupted.xyz")
    assert exc_info.value.reason == "corrupted"
```

## 注意事项

1. **异步接口**：`parse` 方法是 `async` 的，如果底层库是同步的，可以使用 `asyncio.to_thread` 包装
2. **内存管理**：大文件解析时注意内存使用，避免一次性加载整个文件到内存
3. **超时控制**：Celery 任务有 60 秒硬超时，确保解析在此时间内完成
4. **错误分类**：正确区分永久性错误和临时性错误，避免无意义的重试
5. **位置信息**：`bbox` 归一化到 0-1 范围，便于跨页面比较
6. **页码**：从 1 开始计数
