"""Universal Parser 「无 Profile」场景的合成测试文档夹具（任务 10.10）。

本模块只暴露**纯 Python 构造器**，每个函数返回一个 ``ParsedDocument`` 实例；
不读取磁盘上的真实 PDF / DOCX，也不依赖 Poppler / LibreOffice / litellm，让
测试在 CI（无 OS 依赖）上稳定运行。

设计原则：

- 文档结构刻意不命中 Profile 系统的内置启发式（中文编号 / Chapter / 数字
  小节），因此在没有自定义 Profile 时 ``ProfileMatcher`` 必然回落到
  ``generic-text`` 兜底，从而触发 LLM 通用解析。
- 每个函数都返回一份「足够大、但又足够便宜」的多页文档：分页结构真实
  存在（需要走「按页 → LLM」的循环），单页内容长度短（避免无意义的耗时）。
- 元数据里塞入 ``file_type`` / ``file_path`` / ``page_count`` 等 metadata 项，
  让下游的候选 Profile 命名 / metadata envelope 校验也能在这些夹具上跑通。

公开 API：

- :func:`make_unknown_format_document` —— 多页、英文 + 数字 + 内联表格，
  没有可识别的标题前缀，最贴近「Profile 完全失配」场景。
- :func:`make_scanned_pdf_like_document` —— 多页、绝大多数块只有 ``style.is_image``
  / 空文本，模拟 OCR 缺失的扫描件 PDF。
- :func:`make_chinese_unknown_layout_document` —— 中文内容但没有 ``一、`` /
  ``第N章`` / ``N.`` 等编号，使用自定义组织内部标题。
"""

from __future__ import annotations

from app.services.parsers.base import Block, ParsedDocument

__all__ = [
    "make_unknown_format_document",
    "make_scanned_pdf_like_document",
    "make_chinese_unknown_layout_document",
]


# ─── 多页通用辅助 ──────────────────────────────────────────────────────


# 跨页重复的「页眉」文本：刻意短（<100 字符）且明确不命中任何编号 / 标题模式，
# 这样 ``ProfileMatcher`` 不会因为它把文档当成结构化文档；同时 UniversalParser
# 的 boilerplate 检测会在多次重复后把它识别为噪声。
_REPEATED_HEADER = "Internal Working Document — Subject to Update"


def _heading_block(text: str, page: int) -> Block:
    """生成一个不带 markdown 标题前缀的「视觉标题」块。

    Profile 兜底分支专门测试的是「没有可被规则识别的标题」的场景，因此这里
    type 故意不是 ``"heading"``，避免被任何 heading 规则走偏。这些文本以纯
    paragraph 的形式出现，保证 ProfileMatcher 不会被它们误导。
    """
    return Block(type="paragraph", text=text, page_number=page)


# ─── 1) 未知格式 / 无可识别标题 ────────────────────────────────────────


def make_unknown_format_document() -> ParsedDocument:
    """多页文档，**没有任何 Profile 内置规则能识别其标题或编号**。

    具体特征：
    - 标题文本是「Overview & Goals」/「How It Works」/「Operational Notes」/
      「Closing Remarks」之类的英文短语，刻意避开 ``Chapter 1`` / ``1.`` /
      ``一、`` 等所有内置编号模式。
    - 每页都重复一行很短的页眉（``_REPEATED_HEADER``），用来给 boilerplate
      检测提供材料。
    - 第二页插一个 inline Markdown 表格（在原始 ``ParsedDocument`` 里以
      ``type="table"`` 出现），便于跨页表格 / table_count 路径覆盖。
    - metadata 标识 ``file_type="txt"``，反映这是一种纯文本格式。

    注：这里强制保留多页结构（4 页）—— UniversalParser.parse 会按页分组
    并对每页发起独立 LLM 调用，用 1 页文档无法测出按页循环的行为。
    """
    blocks: list[Block] = []

    # 第 1 页：开场白 + 一段散文
    blocks.append(Block(type="paragraph", text=_REPEATED_HEADER, page_number=1))
    blocks.append(_heading_block("Overview & Goals", 1))
    blocks.append(
        Block(
            type="paragraph",
            text=(
                "This memo summarises the current state of the team's "
                "investigation into how the system has been behaving "
                "during peak load windows over the past quarter."
            ),
            page_number=1,
        )
    )
    blocks.append(
        Block(
            type="paragraph",
            text=(
                "Findings have been compiled from operator interviews, "
                "log samples, and the on-call runbook archive."
            ),
            page_number=1,
        )
    )

    # 第 2 页：标题 + 表格
    blocks.append(Block(type="paragraph", text=_REPEATED_HEADER, page_number=2))
    blocks.append(_heading_block("How It Works", 2))
    blocks.append(
        Block(
            type="paragraph",
            text=(
                "The pipeline streams events into a buffered queue which a "
                "worker pool drains using a backoff strategy."
            ),
            page_number=2,
        )
    )
    blocks.append(
        Block(
            type="table",
            text=(
                "| Stage | Avg Latency | P95 Latency |\n"
                "| --- | --- | --- |\n"
                "| ingest | 12ms | 41ms |\n"
                "| transform | 23ms | 78ms |\n"
                "| persist | 9ms | 30ms |"
            ),
            page_number=2,
        )
    )

    # 第 3 页：标题 + 散文
    blocks.append(Block(type="paragraph", text=_REPEATED_HEADER, page_number=3))
    blocks.append(_heading_block("Operational Notes", 3))
    blocks.append(
        Block(
            type="paragraph",
            text=(
                "Operators should expect transient spikes during the daily "
                "import window, and follow the documented escalation path "
                "if backpressure persists for more than five minutes."
            ),
            page_number=3,
        )
    )

    # 第 4 页：收尾
    blocks.append(Block(type="paragraph", text=_REPEATED_HEADER, page_number=4))
    blocks.append(_heading_block("Closing Remarks", 4))
    blocks.append(
        Block(
            type="paragraph",
            text=(
                "Further investigation is scheduled for the following sprint, "
                "with a separate review meeting planned to revisit retention "
                "policies and downstream consumer guarantees."
            ),
            page_number=4,
        )
    )

    return ParsedDocument(
        blocks=blocks,
        metadata={
            "file_type": "txt",
            "file_path": "/synthetic/unknown_format.txt",
            "page_count": 4,
        },
        assets=[],
    )


# ─── 2) 类扫描件 PDF（OCR 缺失） ────────────────────────────────────────


def make_scanned_pdf_like_document() -> ParsedDocument:
    """多页文档，**绝大多数块文本为空 / 仅含图像样式**。

    模拟「OCR 缺失的扫描件 PDF」——native 解析器抓不到正文，只能把页面
    拍成图片走 multimodal LLM 路径。为了让 UniversalParser 的 ``_parse_page``
    在测试里仍然有图像输入，调用方需要 patch ``_get_page_image`` 直接返回
    伪造的 PNG 字节。

    具体特征：
    - 所有页面都至少有一个 ``style={"is_image": True}`` 的空文本块。
    - 单页 raw_text 长度极短（可能为空字符串），让
      ``ProfileMatcher.extract_features`` 把 ``appears_scanned`` 标记为 True。
    - metadata.``appears_scanned=True`` 是测试断言用的提示，不会改变 Parser
      行为。
    """
    blocks: list[Block] = []
    for page in range(1, 4):  # 3 页
        # 页面纯图像块：text 为空字符串，style 标记 is_image。
        blocks.append(
            Block(
                type="image",
                text="",
                page_number=page,
                style={"is_image": True},
            )
        )
        # 偶尔有一两个噪声字符串混进来（OCR 漏字时常见）。
        if page == 2:
            blocks.append(
                Block(
                    type="paragraph",
                    text="???",
                    page_number=page,
                    style={"is_image": True},
                )
            )

    return ParsedDocument(
        blocks=blocks,
        metadata={
            "file_type": "pdf",
            "file_path": "/synthetic/scanned.pdf",
            "page_count": 3,
            "appears_scanned": True,
        },
        assets=[],
    )


# ─── 3) 中文 + 自定义组织内部标题 ──────────────────────────────────────


def make_chinese_unknown_layout_document() -> ParsedDocument:
    """多页中文文档，**用自定义组织内部标题**（不命中任何内置编号模式）。

    具体特征：
    - 标题用 ``技术总览`` / ``维护要点`` / ``例外处理`` / ``附录信息`` 这样的
      自由文本，刻意避开 ``一、`` / ``第 N 章`` / ``N.`` 内置编号。
    - 每页重复一段短的「保密」标记，模拟跨页页眉。
    - metadata 标识 ``file_type="docx"``，候选 Profile 名应该体现为
      ``auto-generated-docx-{N}p``。
    """
    repeat_marker = "本文档仅供内部参考"
    blocks: list[Block] = []

    # 第 1 页：技术总览
    blocks.append(Block(type="paragraph", text=repeat_marker, page_number=1))
    blocks.append(_heading_block("技术总览", 1))
    blocks.append(
        Block(
            type="paragraph",
            text=(
                "本部分介绍系统的整体设计理念，重点关注吞吐量、可靠性与"
                "运维成本之间的权衡，并给出具体的取舍依据。"
            ),
            page_number=1,
        )
    )

    # 第 2 页：维护要点
    blocks.append(Block(type="paragraph", text=repeat_marker, page_number=2))
    blocks.append(_heading_block("维护要点", 2))
    blocks.append(
        Block(
            type="paragraph",
            text=(
                "日常维护需要按照固定节奏巡检，包括但不限于队列堆积、慢"
                "查询、磁盘占用与备份完整性。"
            ),
            page_number=2,
        )
    )

    # 第 3 页：例外处理
    blocks.append(Block(type="paragraph", text=repeat_marker, page_number=3))
    blocks.append(_heading_block("例外处理", 3))
    blocks.append(
        Block(
            type="paragraph",
            text=(
                "当核心组件出现不可恢复故障时，应先切换到备份链路，再按"
                "升级流程联系当班负责人。"
            ),
            page_number=3,
        )
    )

    # 第 4 页：附录
    blocks.append(Block(type="paragraph", text=repeat_marker, page_number=4))
    blocks.append(_heading_block("附录信息", 4))
    blocks.append(
        Block(
            type="paragraph",
            text="附录提供供查阅的术语表与外部参考链接列表。",
            page_number=4,
        )
    )

    return ParsedDocument(
        blocks=blocks,
        metadata={
            "file_type": "docx",
            "file_path": "/synthetic/chinese_unknown.docx",
            "page_count": 4,
        },
        assets=[],
    )
