# WikForge 项目精要 · 面试速学版（手机阅读）

> 项目：企业知识库 + RAG 问答系统
> 技术栈：FastAPI + Celery + PostgreSQL / OpenSearch / Qdrant / Redis / MinIO + LiteLLM + Next.js
> 用途：通勤/手机上学习，回到电脑后按文末清单精读源码

---

## 一、这个项目是什么

一句话：**企业级知识库系统，上传文档 → 解析入库 → 员工提问 → RAG 检索增强生成回答，带权限控制和质量审核。**

两条主线：

- **入库线**：文档上传 → 解析（插件/LLM 兜底）→ 清洗 → 分块 → 向量化 → 双写索引（Qdrant + OpenSearch）
- **问答线**：提问 → 查询增强（改写/HyDE/子查询分解）→ 混合检索 → 精排 → LLM 生成 → 流式返回 + 引用

---

## 二、最核心：LLM 网关（面试必讲）

文件：`backend/app/services/llm_gateway.py`（约 480 行）

### 2.1 它是什么

项目自建的**多模型统一调用层**。底层走 LiteLLM，上面包了一层生产级逻辑。全项目所有调 LLM 的地方（RAG 生成、查询改写、HyDE、文档解析）都必须经过它。

### 2.2 和你用过的 ChatOpenAI 对比

| 维度 | 你的 ChatOpenAI | 本项目 LLMGateway |
|---|---|---|
| 供应商 | 只支持 OpenAI 协议 | 100+ 厂商（GPT/Claude/Qwen/Ollama），改环境变量即切换 |
| 底层 | OpenAI SDK | LiteLLM（协议翻译层） |
| 返回值 | LangChain `AIMessage` | 自定义 `LLMResponse`（content/model/usage/finish_reason） |
| 超时 | 单个 timeout 参数 | 双层：LiteLLM timeout + `asyncio.wait_for` 兜底 |
| 错误处理 | 抛 SDK 原生异常 | 统一翻译成 `LLMGatewayError`，带 reason 错误码（timeout/rate_limit/auth/model_unavailable） |
| 日志 | 无/框架默认 | 每次调用记录 model、token 用量、耗时；**不记 prompt 原文**（隐私合规） |
| 多模态 | 手写 HumanMessage content 数组 | `complete_multimodal()` 收口，传图片 bytes 即可 |
| 测试 | 需 patch LangChain 内部 | 依赖注入，测试传 mock gateway |

### 2.3 关键代码片段（记住这三个就够讲）

**① 模型名规范化——一句代码实现多厂商统一：**

```python
# 没有 "/" 就自动加 openai/ 前缀，统一走 OpenAI 兼容协议
normalized_model = self.model if "/" in self.model else f"openai/{self.model}"

# 加上这个，自动丢弃上游模型不支持的参数（如 gpt-5 只收 temperature=1）
litellm.drop_params = True
```

**② 双层超时——SDK 卡住也有兜底：**

```python
response = await asyncio.wait_for(
    litellm.acompletion(**kwargs),   # kwargs 里有 timeout=self.timeout
    timeout=self.timeout + 5,        # 外层再加 5 秒保险
)
```

**③ 错误分类——把各家异常翻译成业务错误码：**

```python
except Exception as e:
    if "rate_limit" in error_msg or "429" in error_msg:
        reason = "rate_limit"
    elif "auth" in error_msg or "401" in error_msg:
        reason = "auth"
    elif "not found" in error_msg or "404" in error_msg:
        reason = "model_unavailable"
    raise LLMGatewayError(f"LLM call failed: {error_msg}", reason=reason)
```

### 2.4 面试话术（背这个）

> "我的 demo 用 LangChain 的 ChatOpenAI 直接调模型，解决的是'方便调一个模型'。这个项目把模型调用收敛成自建网关：用 LiteLLM 抹平多厂商协议差异，自己掌控超时、错误分类、脱敏日志、多模态和依赖注入。本质上是把 LiteLLM Proxy 这类网关产品的职责，以轻量代码形式内嵌进了服务。"

**加分反问点**：这个网关没实现自动重试、token 计数、缓存、tool calling、结构化输出——被问"怎么改进"时答这些（重试可用 LiteLLM 的 `num_retries` 或 tenacity）。

---

## 三、LiteLLM 是什么（一句话分清层级）

- **LangChain**：应用编排**框架**（Chain/Agent/Memory/Tool 全家桶），多模型支持只是副产品
- **LiteLLM**：单一职责的**协议翻译库**——把 100+ 厂商 API 统一成 OpenAI Chat Completions 格式，核心就 `completion` / `acompletion` 两个函数
- 它还有一个 Proxy 形态（跑成独立网关服务），本项目用的是 **SDK 形态**
- 本项目对 LangChain 的依赖**只有** `langchain-text-splitters`（切文档用），调模型完全不经过 LangChain

---

## 四、LangChain 概念对照表（你已有知识的迁移）

| 你学过的 | 本项目对应物 |
|---|---|
| `ChatOpenAI` / `.invoke()` | `llm_gateway.py` → `LLMGateway.complete()` |
| `.stream()` | `LLMGateway.stream()` 异步生成器 |
| Retriever | `search_service.py`（多路召回 + RRF 融合 + Cross-Encoder 精排） |
| `RecursiveCharacterTextSplitter` | `chunker.py`（保留 langchain-text-splitters） |
| Embeddings 接口 | `embedding_service.py`（稠密 + 稀疏向量） |
| LCEL Chain | `rag_engine.py` 的 `chat()` 手写 async 编排 |
| Memory | `conversation_service.py` |
| 查询改写 / HyDE / 子查询分解 | `query_rewriter.py` / `hyde_service.py` / `subquery_decomposer.py`（总控在 `query_enhancer.py`，带超时降级） |
| Document Loader | `parsers/` 插件体系 + `universal_parser.py`（LLM 多模态兜底） |

**学习方法**：每读一个模块就问自己"这在 LangChain 里叫什么"——概念是同一套，区别只是从"框架帮你串"变成"工程师手写串"。

---

## 五、企业为什么不用 LangChain 全栈（高频面试题）

**好处**：依赖可控（LangChain 升级常 breaking）、模型自由（改环境变量换厂商）、可测试（mock 注入）、透明（数据流一目了然，不用学 LCEL 语法）、性能（无框架转换开销）。

**代价**：重试/缓存/tool calling 等要自己写。

**行业实情**：
- POC/demo 用 LangChain 多，因为快
- 认真的生产系统**不用或轻度使用是主流**（像本项目只取 text-splitter）
- Agent 场景 LangGraph 采用度在上升；也有很多团队直接用厂商原生 SDK 或 Dify 类平台

**话术**："我用 LangChain 做过 RAG，也理解为什么生产项目会绕开它的全栈、只取所需"——这个组合比只会调框架立体得多。

---

## 六、问答链路（面试第二大重点）

`POST /rag/chat` 进来后（`api/rag.py` → `rag_engine.py`）：

```
用户提问
  → 查询增强 query_enhancer.py（三路并行，各自带超时降级）：
  │     ├── query_rewriter.py     改写 query（结合会话历史）
  │     ├── hyde_service.py       生成假想答案，用它去检索
  │     └── subquery_decomposer.py 复杂问题拆子查询
  → 混合检索 search_service.py：
  │     ├── 稠密向量召回（Qdrant）
  │     ├── BM25 关键词召回（OpenSearch，IK 中文分词）
  │     ├── RRF 融合多路结果
  │     └── Cross-Encoder 精排
  │     （全程被 permission_service.py 按 ABAC 权限过滤）
  → 生成 rag_service.py：
        ├── complete() 非流式，或 stream() 逐 token 推送（SSE）
        └── 组装引用 citations（回答可溯源到文档片段）
```

**这条链路比 LangChain 默认 RAG 强在哪**（可以直接说）：多路召回+精排而不是单次向量检索、查询增强带超时降级（增强失败不影响主流程）、权限过滤在检索层完成、回答带引用。

---

## 七、入库流水线（第三大重点）

`app/tasks/pipeline.py`（Celery 任务链）：

```
上传 upload_service.py
  → 解析 parsers/（pdf/docx/pptx/html/txt 插件，registry.py 热更新）
  │     └── 无匹配 Profile 时 → universal_parser.py 用多模态 LLM 兜底解析
  → 清洗 document_processor.py（去噪、标题识别、转 Markdown）
  → 分块 chunker.py（按 Profile 策略、保留文档层级）
  → 向量化 embedding_service.py（稠密 + 稀疏）
  → 双写索引 indexing_service.py（Qdrant + OpenSearch，失败回滚）
  → 质量评分 quality_scorer.py → 低分进人工审核 review_queue.py
```

亮点：插件化解析器、LLM 兜底解析、双写回滚、质量评分+人工审核闭环、watchdog 捞回卡死任务。

---

## 八、回到电脑后的精读清单（按优先级）

1. ★ `backend/app/services/llm_gateway.py` + `backend/tests/test_llm_gateway.py`
2. ★ `backend/app/services/rag_engine.py`（问答主编排 `chat()` 方法）
3. ★ `backend/app/services/search_service.py`（混合检索+精排）
4. `backend/app/services/query_enhancer.py`（超时降级设计）
5. `backend/app/tasks/pipeline.py`（入库流水线）
6. `backend/app/services/indexing_service.py`（双写回滚）
7. `backend/eval/`（RAG 效果评估，加分项）
8. `docker-compose.yml` + `backend/app/core/config.py`（全局组件视图）

---

## 九、30 秒电梯陈述（面试开场可背）

> "这个项目是企业知识库 RAG 系统。文档侧：多格式插件解析 + LLM 多模态兜底，分块后稠密稀疏双向量，双写 Qdrant 和 OpenSearch 并支持回滚。问答侧：查询改写/HyDE/子查询分解三路增强，多路召回 RRF 融合加 Cross-Encoder 精排，检索层做 ABAC 权限过滤，最后经自建 LLM 网关流式生成带引用的回答。全系统不依赖 LangChain 编排，模型调用通过 LiteLLM 网关统一管控超时、错误和日志，另有质量评分、人工审核和 RAG 效果评估的生产闭环。"
