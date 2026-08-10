# 解析器插件开发指南

> 适用于 `enterprise-knowledge-base` 后端的 Parser Plugin 系统。
> 对应 spec：`/.kiro/specs/enterprise-knowledge-base/`，任务 7。

## 总览

文档处理管线第一道防线是 **Parser Plugin（格式解析层）**：把原始文件转成结构化的中间表示（IR），不做业务理解。后续的 Profile Matcher、清洗、分块、向量化都依赖这个 IR。

插件以 **可热加载** 的方式注册到 `ParserRegistry`：

- 内置插件直接放在 `backend/app/services/parsers/` 下，由 pipeline 在首次使用时自动注册。
- 第三方/自定义插件通过 `parser_plugin_configs` 表声明（含 `import_path`），运行时由 `ParserRegistry.load_from_database()` 加载。

```
原始文件 ──▶ ParserRegistry.select(file_path, mime_type) ──▶ ParserPlugin.parse() ──▶ ParsedDocument
                                                              │
                                                              └──▶ blocks / metadata / assets
```

## 协议规范：`ParserPlugin`

定义位置：`backend/app/services/parsers/base.py`

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class ParserPlugin(Protocol):
    name: str                          # 插件唯一名称（用于注册表 / 配置表 / 日志）
    supported_extensions: list[str]    # 不带点的扩展名，例如 ["pdf"]、["html", "htm"]
    priority: int                      # 多插件命中时按优先级降序选择，默认 0

    def can_parse(self, file_path: str, mime_type: str) -> bool:
        """根据文件路径或 MIME 判断是否能处理。建议两者都看。"""

    async def parse(self, file_path: str) -> ParsedDocument:
        """把文件转为 ParsedDocument。出错时抛 ParseError。"""
```

约束：

- `parse` 必须是 **async**（即使内部是同步操作也要写成 `async def`）。
- `name` 在整个 Registry 中必须唯一，重复注册会抛 `ValueError`。
- 不能在 `parse` 中执行下载、写盘等带副作用的逻辑（pipeline 已经把文件下载到本地临时路径再传入）。

## 中间表示：`ParsedDocument` / `Block` / `Asset`

```python
@dataclass
class ParsedDocument:
    blocks: list[Block]      # 顺序保留的内容块
    metadata: dict           # 文件元数据：page_count、title、author、source
    assets: list[Asset]      # 二进制附件：图片、公式截图

@dataclass
class Block:
    type: str                # "paragraph" | "heading" | "table" | "image" | "formula" | "list"
    text: str
    bbox: tuple[float, float, float, float] | None = None  # 归一化坐标 (x0, y0, x1, y1)
    page_number: int = 1
    style: dict = {}         # heading_level / bold / italic / font_size / code_block 等
    raw: dict = {}           # 原始解析数据（仅用于排错，不进入下游管线决策）

@dataclass
class Asset:
    id: str                  # uuid，用于 Block 引用
    type: str                # "image" | "formula" | "diagram"
    data: bytes
    mime_type: str           # 例如 "image/png"
    page_number: int = 1
    bbox: tuple[float, float, float, float] | None = None
    description: str = ""
```

写解析器时的几个原则：

1. **保持顺序**：blocks 必须按文档自然阅读顺序输出，下游清洗依赖前后关系判断噪声。
2. **保留页码**：`page_number` 必须是 1-indexed；非分页格式（DOCX/HTML/MD）填 `1` 即可。
3. **保留标题层级**：能识别出标题就标 `type="heading"` 并写 `style["heading_level"]`（1–6）。
4. **图片占位**：发现图片时既要追加 `Asset`，也要插入 `type="image"` 的 `Block`（`text=f"[Image: {asset.id}]"`），方便后续的多模态描述生成。
5. **表格统一为 Markdown**：`type="table"` 的 block，`text` 用 GitHub 风格 Markdown 表格存储，便于后续 chunker 处理。
6. **空文档要抛错**：抽取后没有内容时抛 `ParseError(reason="empty")`，由 pipeline 走失败处理。

## 错误处理：`ParseError`

```python
raise ParseError(message, reason="corrupted")
```

`reason` 取值（与 pipeline 的失败处理一一对应）：

| reason                | 含义                          | pipeline 行为     |
| --------------------- | ----------------------------- | ----------------- |
| `corrupted`           | 文件损坏、非合法格式          | 直接标记失败，不重试 |
| `password_protected`  | 文件加密 / 需要密码           | 直接标记失败，不重试 |
| `empty`               | 解析后无任何可用内容          | 直接标记失败，不重试 |
| `unsupported_version` | 格式版本不支持                | 直接标记失败，不重试 |
| `unknown`             | 其他错误                      | 触发指数退避重试  |

非 `ParseError` 的异常会被视作瞬时错误，进入 Celery 的指数退避重试链（最多 3 次，初始 10 秒）。

## 注册流程

### 内置插件

直接在 `backend/app/services/parsers/` 添加模块即可，pipeline 中的 `_ensure_default_parsers_registered` 会自动注册预置 5 个解析器。

如要把新插件加入 "默认集合"，在 `pipeline.py` 的 `_ensure_default_parsers_registered` 列表里追加；不过更推荐走数据库配置（下文）。

### 通过数据库注册（推荐）

第三方/自定义插件不需要改代码，只需要：

1. 把插件代码放在 Python 路径里能 import 到的位置（容器内、wheel 包均可）。
2. 在 `parser_plugin_configs` 表插入一条记录：

   ```sql
   INSERT INTO parser_plugin_configs (
       id, name, import_path, supported_extensions, priority, enabled, config
   ) VALUES (
       gen_random_uuid(),
       'csv-parser',
       'app.services.parsers.csv_parser:CsvParser',  -- 也支持 'pkg.mod.Class'
       '["csv"]',
       80,
       true,
       '{"delimiter": ","}'
   );
   ```

3. 触发热加载：

   ```python
   from app.services.parsers.registry import get_parser_registry

   registry = get_parser_registry()
   await registry.load_from_database(session)   # AsyncSession
   ```

`load_from_database` 会清空现有插件、按 `priority desc` 拉取所有 `enabled=true` 的记录、按 `import_path` 反射加载。也提供同步版本：

- `registry.reload_from_db_records(records)`：传入 ORM/MagicMock 等 attribute 风格对象。
- `registry.reload_from_configs(dicts)`：传入字典风格配置（与 `ParserPluginConfig` 表字段同名即可）。

`config` 字段会作为 **kwargs 传给插件构造函数**，例如 `CsvParser(delimiter=',')`。

## 最小示例：CSV 解析器

新建 `backend/app/services/parsers/csv_parser.py`：

```python
"""CSV 解析器示例（演示自定义插件）。"""

import csv
import os

from app.services.parsers.base import Block, ParsedDocument, ParseError


class CsvParser:
    """把 CSV 文件转换为单一 Markdown 表格 Block。"""

    name: str = "csv-parser"
    supported_extensions: list[str] = ["csv"]
    priority: int = 80

    def __init__(self, delimiter: str = ",", encoding: str = "utf-8") -> None:
        self.delimiter = delimiter
        self.encoding = encoding

    def can_parse(self, file_path: str, mime_type: str) -> bool:
        ext = os.path.splitext(file_path)[1].lower().lstrip(".")
        return ext in self.supported_extensions or mime_type == "text/csv"

    async def parse(self, file_path: str) -> ParsedDocument:
        if not os.path.exists(file_path):
            raise ParseError(f"file not found: {file_path}", reason="corrupted")

        try:
            with open(file_path, "r", encoding=self.encoding, newline="") as f:
                rows = list(csv.reader(f, delimiter=self.delimiter))
        except (UnicodeDecodeError, csv.Error) as e:
            raise ParseError(f"invalid CSV: {e}", reason="corrupted")

        rows = [r for r in rows if any(c.strip() for c in r)]
        if not rows:
            raise ParseError("CSV has no rows", reason="empty")

        header = rows[0]
        body = rows[1:]
        md_lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * len(header)) + " |",
        ]
        for row in body:
            # 列数对不齐时补空字符串
            padded = list(row) + [""] * (len(header) - len(row))
            md_lines.append("| " + " | ".join(c.replace("|", "\\|") for c in padded) + " |")

        block = Block(
            type="table",
            text="\n".join(md_lines),
            page_number=1,
            style={"row_count": len(body), "col_count": len(header)},
        )

        return ParsedDocument(
            blocks=[block],
            metadata={"source": file_path, "page_count": 1, "row_count": len(body)},
            assets=[],
        )
```

注册到数据库：

```sql
INSERT INTO parser_plugin_configs (
    id, name, import_path, supported_extensions, priority, enabled, config
) VALUES (
    gen_random_uuid(),
    'csv-parser',
    'app.services.parsers.csv_parser:CsvParser',
    '["csv"]',
    80,
    true,
    '{"delimiter": ",", "encoding": "utf-8"}'
);
```

热加载后，`registry.select("data.csv", "text/csv")` 就会拿到 `CsvParser` 实例。

## 测试要点

新插件至少需要覆盖以下用例（参考 `backend/tests/test_parsers.py`）：

1. **协议合规**：`isinstance(parser, ParserPlugin)` 返回 `True`。
2. **`can_parse` 路由**：扩展名命中 / 不命中，MIME 命中 / 不命中。
3. **happy path**：构造一份 fixture 文件，断言关键 block 的 `type`、`text`、`page_number`、`style.heading_level`。
4. **空文件**：抛 `ParseError(reason="empty")`。
5. **损坏文件**：构造一段非法字节，抛 `ParseError(reason="corrupted")`。
6. **密码保护**（如适用）：用 `unittest.mock.patch` mock 第三方库抛带 "password" / "encrypted" 的异常，断言 `reason == "password_protected"`。
7. **不要依赖真实 Marker / LibreOffice 等重型外部组件**：用 mock 替代第三方 SDK，保持单测秒级完成。

集成层面，新插件应：

- 在 `parser_plugin_configs` 表插入一条 `enabled=true` 的记录。
- 通过 `registry.load_from_database()` 热加载后能被 `select()` 命中。
- 在 pipeline 的 `parse_document` Celery 任务里，正常生成 `ParsedDocument` 并传给 `profile_match`。

## 常见陷阱

- **插件构造函数有必填参数但 `config` 没传**：`_load_plugin` 用 `**config` 实例化，缺参会直接抛错并被记录为 `Failed to load plugin`。在协议层面建议把所有参数都给默认值。
- **`supported_extensions` 写成大写**：所有扩展名匹配都做了 `lower()`，请用小写。
- **在 `parse` 内 `print` 调试信息**：用 `logging`，pipeline 跑在 Celery worker 里，stdout 不一定能看到。
- **修改 `ParsedDocument` / `Block` 数据结构**：会影响所有下游组件，请走 design doc 流程而不是直接改。
