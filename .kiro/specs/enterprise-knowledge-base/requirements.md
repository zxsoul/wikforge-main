# 需求文档

## 简介

Wikforge 是一个企业级知识库系统，核心能力包括文档导入、清洗、转换、向量化，以及高准确率的知识检索和 RAG（检索增强生成）问答。系统面向千人级用户、十万篇文档规模，强调召回准确率，采用多路召回 + 融合排序 + 精排的复合搜索架构。

## 术语表

- **System（系统）**: Wikforge 企业级知识库系统整体
- **Document_Manager（文档管理器）**: 负责文档导入、组织、状态追踪的子系统
- **Pipeline（处理管线）**: 基于 Celery 的异步文档处理流水线，包含解析、清洗、分块、向量化等步骤
- **Search_Engine（搜索引擎）**: 负责多路召回、融合排序、精排的复合搜索子系统
- **RAG_Engine（RAG 引擎）**: 基于检索增强生成的对话式问答子系统
- **Auth_Service（认证服务）**: 负责用户认证（本地账号 + OIDC）和 JWT 令牌管理的子系统
- **Permission_Service（权限服务）**: 基于 ABAC 模型的权限控制子系统
- **LLM_Gateway（LLM 网关）**: 通过 LiteLLM 聚合路由调用多种大语言模型的网关服务
- **Chunk（文档块）**: 文档经智能分块后的最小检索单元，附带元数据
- **Space（空间）**: 文档的顶层组织单元，用于隔离不同团队或项目的知识
- **RRF（倒数排名融合）**: Reciprocal Rank Fusion，用于融合多路召回结果的排序算法
- **HyDE（假设文档嵌入）**: Hypothetical Document Embeddings，一种查询增强技术
- **Cross_Encoder（交叉编码器）**: 用于精排的重排序模型
- **Pre_Filtering（预过滤）**: 在向量检索阶段通过元数据过滤实现权限控制的机制
- **Document_Profile（文档档案）**: 描述某一类文档解析策略的配置集合，包含标题识别规则、噪声模式、分块策略、领域词典等
- **Profile_Matcher（档案匹配器）**: 基于文档特征（文件名、内容模式、结构特征）自动匹配合适 Document_Profile 的组件
- **Universal_Parser（通用解析器）**: 基于多模态大模型的兜底解析器，用于处理未匹配到 Document_Profile 或质量评分过低的文档
- **Parse_Quality_Score（解析质量分）**: 对文档解析结果的量化评分（0-1），综合文本保留率、结构识别率、表格完整率、数值保护率等维度
- **Domain_Dictionary（领域词典）**: 针对特定行业或业务领域的专业术语词库，用于 IK 分词器自定义词典和查询改写
- **Parser_Plugin（解析器插件）**: 针对特定文件格式（如 PDF、DOCX）的原生解析组件，以插件形式注册，支持热加载扩展
- **Feedback_Loop（反馈回流）**: 用户对检索和回答质量的反馈机制，用于迭代 Document_Profile 和领域词典

## 需求

### 需求 1：文档导入

**用户故事：** 作为知识库管理员，我希望能够导入多种格式的文档，以便将企业已有知识资产纳入知识库。

#### 验收标准

1. WHEN 用户上传 PDF 文件（大小不超过 100MB）, THE Document_Manager SHALL 使用 Marker 解析器提取文本内容，并保留标题层级、段落分隔和表格结构
2. WHEN 用户上传 Word 文件（.docx，大小不超过 100MB）, THE Document_Manager SHALL 使用 python-docx 解析器提取文本内容，并保留标题层级、段落分隔和表格结构
3. WHEN 用户上传 PPT 文件（.pptx，大小不超过 100MB）, THE Document_Manager SHALL 按幻灯片页码顺序提取每页的文本内容，并保留页码与文本的对应关系
4. WHEN 用户提供网页 URL, THE Document_Manager SHALL 在 30 秒超时时间内抓取网页正文内容并转换为纯文本格式（去除 HTML 标签、脚本和样式）
5. WHEN 用户选择批量导入, THE Document_Manager SHALL 接受最多 50 个文件，并为每个文件创建独立的处理任务
6. THE Document_Manager SHALL 将导入的原始文件存储到 MinIO 对象存储中，并返回存储路径标识
7. IF 文件格式不受支持, THEN THE Document_Manager SHALL 拒绝该文件并返回错误信息，错误信息中包含当前支持的文件格式列表（PDF、DOCX、PPTX、TXT、MD、HTML）
8. IF 上传文件大小超过 100MB, THEN THE Document_Manager SHALL 拒绝该文件并返回错误信息，指明文件大小超出限制及允许的最大值
9. IF 网页 URL 在 30 秒内无法访问或返回非成功状态, THEN THE Document_Manager SHALL 终止抓取并返回错误信息，指明 URL 不可达或响应异常
10. IF 文件内容无法正常解析（如文件损坏或受密码保护）, THEN THE Document_Manager SHALL 将该文件标记为解析失败，并返回错误信息指明失败原因

### 需求 2：文档组织

**用户故事：** 作为知识库用户，我希望能够通过空间、目录和标签组织文档，以便快速定位所需知识。

#### 验收标准

1. THE Document_Manager SHALL 支持创建、编辑和删除空间（Space）作为文档的顶层组织单元，空间名称长度为 1 至 50 个字符且在系统内唯一
2. WHEN 用户在空间内创建目录, THE Document_Manager SHALL 支持最多 10 级目录嵌套结构，目录名称长度为 1 至 50 个字符且在同一父级下唯一
3. THE Document_Manager SHALL 支持为文档添加 1 至 20 个标签，每个标签名称长度为 1 至 30 个字符
4. WHEN 用户移动文档到不同空间或目录, THE Document_Manager SHALL 更新文档的组织关系并保持文档内容及其标签不变
5. THE Document_Manager SHALL 支持按空间、目录、标签进行文档筛选和浏览，筛选结果按分页展示，每页默认显示 20 条记录
6. IF 用户删除包含文档或子目录的空间或目录, THEN THE Document_Manager SHALL 提示用户确认，并在用户确认后将该空间或目录及其所有子目录和文档一并删除
7. IF 用户添加的标签数量超过 20 个或标签名称超过 30 个字符, THEN THE Document_Manager SHALL 拒绝操作并显示错误信息指明具体限制

### 需求 3：文档处理状态追踪

**用户故事：** 作为知识库管理员，我希望能够实时了解文档处理进度，以便掌握知识入库状态。

#### 验收标准

1. WHEN 文档进入处理管线, THE Document_Manager SHALL 将文档状态初始化为"待处理"，并按处理流程依次流转为以下状态之一：待处理、解析中、清洗中、分块中、向量化中、已完成、失败
2. WHILE 文档处于处理中状态（解析中、清洗中、分块中、向量化中）, THE Document_Manager SHALL 每 5 秒更新一次处理进度信息，进度信息包含：当前处理阶段、已耗时时长（秒）、以及当前阶段的完成百分比（0-100）
3. IF 文档处理过程中发生错误, THEN THE Document_Manager SHALL 将文档状态标记为"失败"，并记录错误详情，错误详情包含：失败所在阶段、错误发生时间、以及错误原因描述
4. WHEN 用户查看文档列表, THE Document_Manager SHALL 显示每个文档的当前处理状态、文档名称、以及最近一次状态更新时间
5. WHEN 用户对状态为"失败"的文档触发重新处理操作, THE Document_Manager SHALL 将该文档状态重置为"待处理"并重新进入处理管线
6. IF 同一文档累计处理失败次数达到 3 次, THEN THE Document_Manager SHALL 将该文档标记为"永久失败"，不再允许自动重试，并向用户展示需人工介入的提示信息

### 需求 4：异步文档处理管线

**用户故事：** 作为系统，我需要异步处理导入的文档，以便不阻塞用户操作并高效利用计算资源。

#### 验收标准

1. WHEN 文档导入完成, THE Pipeline SHALL 按照"解析 → 清洗 → 智能分块 → Embedding → 入库"的顺序依次执行处理步骤，单文档端到端处理时间不超过 300 秒
2. THE Pipeline SHALL 通过 Celery 任务队列异步执行所有处理步骤，每个步骤作为独立任务提交，单步骤执行超时时间不超过 60 秒
3. WHEN 执行智能分块时, THE Pipeline SHALL 按标题层级切分文档并保留父子块的层级关系，每个 Chunk 大小在 128 至 1024 个 token 之间，相邻 Chunk 之间保留不超过 64 个 token 的重叠内容，层级深度最多支持 6 级
4. THE Pipeline SHALL 为每个 Chunk 附加以下元数据：标题链（各级标题以分隔符连接）、文档来源（原始文件名）、页码（起始页码）、所属空间 ID、权限标识（继承自文档权限）
5. WHEN 执行 Embedding 步骤时, THE Pipeline SHALL 同时生成 Dense 向量和 Sparse 向量（SPLADE），Dense 向量维度为 1024 维，Sparse 向量以索引-权重稀疏格式存储
6. WHEN 所有步骤执行完成, THE Pipeline SHALL 将 Chunk 及其向量写入 Qdrant，将全文索引写入 OpenSearch，两者写入在同一逻辑事务中完成，若任一写入失败则两者均不生效
7. IF 管线中任一步骤失败, THEN THE Pipeline SHALL 记录失败步骤名称、错误信息及失败时间戳，并支持从失败步骤自动重试，最多重试 3 次，重试间隔按指数退避策略递增（初始间隔 10 秒）
8. WHEN 管线状态发生变更（开始、步骤完成、失败、全部完成）, THE Pipeline SHALL 更新文档处理状态，使用户可通过接口查询当前处理进度和所处步骤
9. IF 重试次数耗尽仍未成功, THEN THE Pipeline SHALL 将文档处理状态标记为"失败"，记录最终错误信息，并支持用户手动触发从失败步骤重新执行

### 需求 5：文档清洗与格式标准化

**用户故事：** 作为系统，我需要对解析后的文档进行清洗和标准化，以便提高后续分块和检索的质量。

#### 验收标准

1. WHEN 文档解析完成, THE Pipeline SHALL 去除噪声内容，包括：连续多个空白字符压缩为单个空格、段落间空行压缩为不超过 1 个空行、去除页眉页脚文本、去除水印文本及重复出现的背景文字
2. WHEN 文档存在重复性噪声文本（如水印、页眉页脚）, THE Pipeline SHALL 支持基于统计学方法自动检测（同一文本在文档页面的相同位置出现频率 ≥50% 时判定为噪声），无需预先配置
3. WHEN 文档清洗完成, THE Pipeline SHALL 将文档内容统一转换为 Markdown 格式，保留原文的段落分隔和行内格式（加粗、斜体、链接）
4. WHEN 转换为 Markdown 格式时, THE Pipeline SHALL 根据匹配到的 Document_Profile 识别并保留文档中的标题层级结构，支持至少 6 级嵌套；默认 Profile 应能识别中文技术规范常见编号（一/二/三、(一)/(二)、1/2/3、(1)/(2)、①/②）和英文标题层级（H1-H6）
5. WHEN 转换为 Markdown 格式时, THE Pipeline SHALL 将文档中的表格结构转换为 Markdown 表格格式，保留行列关系和单元格文本内容
6. WHEN 文档中存在跨页表格（相同表头或列结构在相邻页面延续）, THE Pipeline SHALL 自动识别并合并为单个完整表格
7. IF 表格包含合并单元格或嵌套表格等无法用标准 Markdown 表格表达的结构, THEN THE Pipeline SHALL 将该表格转换为等效的文本描述形式并保留原始数据内容
8. WHEN 文档包含图片、流程图或示意图, THE Pipeline SHALL 通过多模态 LLM 为图片生成文本描述，并将描述与图片在原文档中的位置信息关联
9. WHEN 文档包含数学公式或工程公式（如 △=0.25m + (2～3)）, THE Pipeline SHALL 保护公式作为不可分割的原子单元，在分块和清洗过程中不被拆分
10. WHEN 文档包含带单位的数值（如 0.05mm/m、0.002D、±10mm、55°~65°）, THE Pipeline SHALL 将数值与单位作为整体保留，不得在两者之间插入换行或分块边界
11. IF 文档清洗后剩余文本去除空白字符后长度为 0, THEN THE Pipeline SHALL 将文档标记为"清洗失败"并记录失败原因
12. WHEN 文档清洗与转换完成, THE Pipeline SHALL 验证输出的 Markdown 内容非空且保留了原文档至少 80% 的可见文本字符数（噪声字符除外）

### 需求 6：复合搜索 - 多路召回

**用户故事：** 作为知识库用户，我希望搜索能够同时利用关键词匹配和语义理解，以便获得高召回率的搜索结果。

#### 验收标准

1. WHEN 用户提交搜索查询, THE Search_Engine SHALL 并发执行以下三路召回：BM25 全文检索（OpenSearch）、Dense 向量检索（Qdrant）、Sparse 向量检索（Qdrant SPLADE），每路召回返回不超过 50 个候选文档块
2. WHEN 三路召回完成, THE Search_Engine SHALL 使用 RRF 算法（k=60）融合三路召回结果，生成不超过 100 个候选文档块的统一候选集
3. WHEN 候选集生成后, THE Search_Engine SHALL 使用 Cross_Encoder 重排序模型对候选集中前 20 个文档块进行精排，并按精排分数降序排列最终结果
4. THE Search_Engine SHALL 在最终结果中返回每个文档块的相关性分数（0.0 至 1.0）、来源文档信息（文档标题、文档 ID、块位置索引）和不超过 200 字符的高亮匹配片段
5. IF 用户具有权限限制, THEN THE Search_Engine SHALL 在向量检索阶段通过 Pre_Filtering 机制过滤无权限的文档块，确保未授权文档块不出现在召回结果中
6. IF 任一路召回在 3 秒内未返回结果, THEN THE Search_Engine SHALL 跳过该路召回，使用已返回的召回结果继续执行融合与精排流程
7. WHEN 搜索完成, THE Search_Engine SHALL 在 5 秒内返回最终结果，默认返回不超过 10 个文档块，支持分页参数调整返回数量（最大 50 个）

### 需求 7：查询增强

**用户故事：** 作为知识库用户，我希望系统能够理解我的搜索意图并优化查询，以便即使查询表述不精确也能获得相关结果。

#### 验收标准

1. WHEN 用户提交搜索查询, THE Search_Engine SHALL 在 2 秒内对查询进行改写，生成不超过 5 个语义相关的改写变体，并将改写变体与原始查询共同用于检索
2. WHERE HyDE 功能启用, WHEN 用户提交搜索查询, THE Search_Engine SHALL 基于原始查询生成 1 至 3 个假设文档嵌入，并将其作为补充向量参与语义检索
3. WHEN 用户查询包含 2 个及以上可独立回答的子问题, THE Search_Engine SHALL 将查询分解为不超过 5 个子查询，分别检索后对结果进行去重合并，返回合并后的结果集
4. THE Search_Engine SHALL 在查询增强过程中将原始查询作为必选检索条件纳入最终检索，确保返回结果始终包含与原始查询直接匹配的内容
5. IF 查询增强过程（改写、HyDE 生成或子查询分解）在 5 秒内未完成或发生错误, THEN THE Search_Engine SHALL 回退至使用用户原始查询直接执行检索，并向用户返回检索结果而不显示错误信息

### 需求 8：RAG 对话式问答

**用户故事：** 作为知识库用户，我希望能够通过自然语言对话获取知识库中的信息，以便快速获得准确答案。

#### 验收标准

1. WHEN 用户提交问题, THE RAG_Engine SHALL 通过 Search_Engine 检索相似度得分最高的前 K 个文档块（K 为可配置参数，默认值为 5，范围 1-20），并将检索到的文档块作为上下文提交给 LLM 生成回答
2. THE RAG_Engine SHALL 以流式方式输出 LLM 生成的回答内容，首个 token 应在用户提交问题后 5 秒内返回
3. THE RAG_Engine SHALL 在回答中标注引用来源，包括来源文档名称和文档块的定位标识（如章节标题或段落序号）
4. WHILE 用户处于同一对话会话中, THE RAG_Engine SHALL 维护最近 20 轮对话记录作为上下文以支持多轮追问，超出部分按时间顺序丢弃最早的记录
5. THE RAG_Engine SHALL 通过 LLM_Gateway 调用大语言模型，支持 OpenAI、Claude、通义千问、Ollama 四种模型接口
6. IF 所有检索结果的相似度得分均低于可配置阈值（默认 0.5，范围 0-1）, THEN THE RAG_Engine SHALL 向用户返回未找到相关信息的提示，而非生成无依据的回答
7. IF LLM_Gateway 调用失败或响应超时（超时阈值为可配置参数，默认 60 秒）, THEN THE RAG_Engine SHALL 向用户返回服务暂时不可用的错误提示，并保留当前对话上下文不丢失
8. IF 对话会话超过 30 分钟无新消息, THEN THE RAG_Engine SHALL 将该会话标记为过期，用户下次提问时开启新的对话会话

### 需求 9：认证

**用户故事：** 作为企业用户，我希望能够通过企业账号或本地账号安全登录系统，以便访问知识库资源。

#### 验收标准

1. THE Auth_Service SHALL 支持本地账号注册和登录（邮箱 + 密码），其中密码长度为 8 至 64 个字符，须包含大写字母、小写字母、数字和特殊字符中的至少三类，密码使用 bcrypt 哈希存储
2. THE Auth_Service SHALL 支持符合 OIDC 标准协议的身份提供商认证，包括但不限于 Keycloak、Okta、Azure AD，凡实现 OIDC Discovery 端点的身份提供商均可接入
3. WHEN 用户认证成功, THE Auth_Service SHALL 签发 JWT Access Token（有效期 30 分钟）和 Refresh Token（有效期 7 天）
4. WHEN Access Token 过期且 Refresh Token 仍有效, THE Auth_Service SHALL 接受 Refresh Token 签发新的 Access Token，客户端无需用户重新输入凭证
5. IF Refresh Token 已过期或已被撤销, THEN THE Auth_Service SHALL 拒绝刷新请求并返回错误信息指示用户需重新认证
6. IF 同一账号在 30 分钟内连续 5 次登录失败, THEN THE Auth_Service SHALL 锁定该账号 15 分钟，锁定期间的登录尝试应返回错误信息指示账号已被临时锁定及剩余锁定时间
7. IF 用户注册时提供的邮箱已存在或密码不满足复杂度要求, THEN THE Auth_Service SHALL 拒绝注册请求并返回错误信息指示具体失败原因

### 需求 10：权限控制

**用户故事：** 作为知识库管理员，我希望能够精细控制用户对空间和文档的访问权限，以便保护敏感信息。

#### 验收标准

1. THE Permission_Service SHALL 基于 ABAC（基于属性的访问控制）模型实现权限判定，单次权限判定响应时间不超过 50ms
2. THE Permission_Service SHALL 支持空间级权限设置，包含以下权限类型：不可见（空间对用户隐藏）、可见只读（用户可浏览和阅读）、可见可写（用户可浏览、阅读和编辑）；新建空间默认权限为不可见
3. THE Permission_Service SHALL 支持文档级权限设置（不可见、可读、可写），当文档级权限与空间级权限冲突时，以文档级权限为准；未单独设置文档级权限的文档继承所属空间的权限
4. WHEN 管理员设置或变更权限, THE Permission_Service SHALL 在 3 秒内将权限元数据同步写入 Qdrant payload 以支持检索时 Pre_Filtering
5. IF 权限元数据同步写入 Qdrant 失败, THEN THE Permission_Service SHALL 回滚本次权限变更操作，保留变更前的权限状态，并向管理员返回包含失败原因的错误提示
6. WHEN 权限发生变更, THE Permission_Service SHALL 通过异步任务在 60 秒内完成所有受影响文档块的向量元数据更新
7. THE Permission_Service SHALL 确保用户在搜索结果中仅能看到有权限访问的文档块，无权限的文档块不出现在搜索结果列表中
8. IF 用户尝试访问无权限的文档或空间, THEN THE Permission_Service SHALL 拒绝访问请求并返回无权限访问的错误提示，不暴露该资源的任何内容信息

### 需求 11：前端 - 文档管理界面

**用户故事：** 作为知识库管理员，我希望有一个直观的界面来管理文档的上传、导入和处理状态，以便高效管理知识资产。

#### 验收标准

1. THE System SHALL 提供文件上传组件，支持拖拽上传和文件选择器，支持同时选择最多 20 个文件上传，单个文件大小不超过 100MB，支持的文件格式包括 PDF、DOCX、PPTX、TXT、MD、HTML
2. WHEN 用户发起文件上传时, THE System SHALL 显示每个文件的上传进度百分比，并在上传完成后 2 秒内更新文档列表中该文档的处理状态
3. IF 用户上传的文件格式不在支持列表中或文件大小超过 100MB, THEN THE System SHALL 拒绝该文件上传并在上传区域显示错误提示，说明被拒绝的原因（文件格式不支持或文件过大）
4. THE System SHALL 在文档列表中显示每个文档的处理状态（待处理、处理中、已完成、失败），状态变更后 5 秒内在界面上刷新显示
5. THE System SHALL 提供空间和目录的树形导航结构，支持至少 10 层嵌套目录，点击目录节点后在 1 秒内加载并显示该目录下的文档列表
6. THE System SHALL 支持文档的标签管理：每个文档最多添加 20 个标签，标签名称长度为 1 至 30 个字符，支持按标签筛选文档列表
7. THE System SHALL 采用 Next.js + Tailwind CSS + shadcn/ui 技术栈实现，所有交互组件符合 shadcn/ui 设计规范，页面在 1920×1080 和 1366×768 分辨率下布局无溢出或重叠

### 需求 12：前端 - 知识检索与问答界面

**用户故事：** 作为知识库用户，我希望有一个统一的界面进行搜索和 AI 对话，以便快速获取所需信息。

#### 验收标准

1. THE System SHALL 提供全局搜索快捷键（macOS 为 Cmd+K，Windows/Linux 为 Ctrl+K），在任意页面触发搜索面板，搜索面板应在按键后 300ms 内显示
2. WHEN 搜索返回结果时, THE System SHALL 在搜索结果中高亮显示匹配的关键词片段，每条结果展示不超过 200 字符的上下文片段，单次搜索最多展示 20 条结果
3. IF 搜索无匹配结果, THEN THE System SHALL 显示空状态提示信息，告知用户未找到匹配内容并建议调整关键词
4. THE System SHALL 提供对话式 AI 问答界面，支持流式逐字显示回答内容，用户单次输入长度不超过 2000 字符
5. IF AI 回答请求在 30 秒内未返回任何内容或连接中断, THEN THE System SHALL 显示错误提示信息并提供重试操作入口
6. THE System SHALL 在 AI 回答中以可点击链接形式展示引用来源，每条引用包含文档标题和所在段落标识
7. WHEN 用户点击引用来源链接, THE System SHALL 跳转到对应文档并高亮显示被引用的片段
8. THE System SHALL 支持深色和浅色主题切换，默认跟随操作系统主题设置，用户可手动切换并持久化保存偏好

### 需求 13：前端 - 后台管理界面

**用户故事：** 作为系统管理员，我希望有一个后台管理界面来配置空间、权限和模型参数，以便灵活管理系统。

#### 验收标准

1. THE System SHALL 提供空间管理界面，支持以下操作：创建空间（名称最长 50 个字符、描述最长 200 个字符）、编辑空间信息、删除空间，以及为空间分配成员并设置成员角色（如管理员、编辑者、查看者）
2. WHEN 管理员执行删除空间操作时, THE System SHALL 显示二次确认对话框，明确提示将被删除的空间名称及其包含的文档数量，仅在管理员确认后执行删除
3. THE System SHALL 提供用户和权限管理界面，包含：用户列表（支持分页，每页默认 20 条，支持按用户名或邮箱搜索）、角色分配（将用户分配至预定义角色）、以及按角色配置功能模块的访问权限
4. THE System SHALL 提供 LLM 模型配置界面，支持：从可用模型列表中选择模型、管理 API Key（新增、删除、启用/禁用）、调整模型参数（temperature 范围 0-2、max_tokens 范围 1-128000、top_p 范围 0-1）
5. WHILE 显示已保存的 API Key 时, THE System SHALL 仅展示 API Key 的最后 4 位字符，其余部分以掩码形式显示，且不提供查看完整 Key 的功能
6. THE System SHALL 提供系统监控面板，展示：文档处理队列状态（待处理数量、处理中数量、已完成数量、失败数量）以及系统资源使用情况（CPU 使用率、内存使用率、存储使用率），监控数据每 30 秒自动刷新一次
7. WHEN 管理员执行创建、编辑、删除或配置操作后, THE System SHALL 在页面顶部显示操作结果通知（成功或失败），通知在 5 秒后自动消失，失败通知需包含错误原因描述

### 需求 14：部署

**用户故事：** 作为运维人员，我希望能够通过 Docker Compose 一键部署整个系统，以便快速搭建和维护知识库环境。

#### 验收标准

1. THE System SHALL 提供 Docker Compose 配置文件，包含以下容器：postgres、opensearch、qdrant、redis、minio、api、worker、frontend
2. WHEN 执行 docker compose up 命令, THE System SHALL 在 5 分钟内完成所有服务的启动和初始化，所有 8 个容器进入 running 状态且健康检查通过
3. THE System SHALL 为每个服务提供健康检查配置（检查间隔不超过 30 秒，超时不超过 60 秒，重试次数不少于 3 次），并通过 depends_on 条件确保依赖服务健康检查通过后再启动下游服务
4. THE System SHALL 通过环境变量文件（.env）集中管理所有可配置参数，并提供 .env.example 示例文件列出所有必填和可选参数及其说明
5. THE System SHALL 提供数据卷挂载配置以确保 PostgreSQL、OpenSearch、Qdrant、MinIO 的数据持久化，容器重启后数据不丢失
6. IF 任一服务启动失败或健康检查未在超时时间内通过, THEN THE System SHALL 停止启动依赖该服务的下游容器，并通过容器日志输出错误信息指明失败的服务名称和原因
7. THE System SHALL 在 Docker Compose 配置中为以下服务定义宿主机端口映射：frontend（80）、api（8000）、postgres（5432）、opensearch（9200）、qdrant（6333）、redis（6379）、minio（9000），且端口值可通过 .env 文件覆盖

### 需求 15：文档 Profile 管理与自动匹配

**用户故事：** 作为知识库管理员，我希望通过配置化的 Profile 来描述不同类型文档的解析策略，以便在新增文档类型时无需修改代码即可获得高质量的解析效果。

#### 验收标准

1. THE Pipeline SHALL 支持 Document_Profile 配置，每个 Profile 包含以下字段：名称、描述、匹配规则（文件名正则、内容正则、结构特征）、标题识别规则（正则 + 层级映射）、噪声模式（水印/页眉/页脚正则）、表格处理策略、分块参数、领域词典引用
2. WHEN 文档进入解析阶段, THE Profile_Matcher SHALL 根据文档特征自动匹配最合适的 Document_Profile，匹配过程不超过 500ms
3. IF 多个 Profile 匹配同一文档, THEN THE Profile_Matcher SHALL 按 Profile 的优先级（数值越大优先级越高）选择最合适的 Profile，优先级相同的选择最近更新的 Profile
4. IF 无任何 Profile 匹配文档, THEN THE Pipeline SHALL 使用系统默认 Profile（通用文本解析）或触发 Universal_Parser 兜底解析
5. THE System SHALL 预置至少 3 类默认 Profile：通用文本文档、中式技术规范文档（一/二/三编号）、通用 PDF 扫描文档
6. THE System SHALL 提供 Profile 管理界面（后台管理），支持创建、编辑、删除、启用/禁用、导入/导出 Profile（JSON 格式）
7. WHEN 管理员在 Profile 编辑界面修改规则, THE System SHALL 提供"预览解析结果"功能，可上传样本文档实时查看 Profile 应用效果
8. WHEN Profile 被修改并保存, THE System SHALL 记录变更历史（变更时间、变更人、变更前后差异），保留最近 20 个版本

### 需求 16：LLM 通用解析兜底

**用户故事：** 作为系统，我需要在无合适 Profile 或解析质量不达标时调用多模态大模型进行兜底解析，以便确保任何文档都能被处理入库。

#### 验收标准

1. IF 文档无匹配的 Document_Profile 或 Parse_Quality_Score 低于阈值（默认 0.7）, THEN THE Pipeline SHALL 自动调用 Universal_Parser 进行兜底解析
2. THE Universal_Parser SHALL 通过 LLM_Gateway 调用多模态大模型（支持图片+文本输入），将文档页面转换为结构化 Markdown
3. WHEN Universal_Parser 执行解析时, THE Pipeline SHALL 将文档按页切分并逐页调用 LLM，每页超时时间不超过 60 秒
4. WHEN Universal_Parser 完成解析, THE System SHALL 基于解析结果自动生成候选 Document_Profile（包含识别到的标题模式、噪声模式、表格结构），并推荐给管理员在后台确认保存
5. IF Universal_Parser 调用失败或超时, THEN THE Pipeline SHALL 记录失败原因并允许回退到基础解析器（仅提取纯文本，不保留结构）
6. THE Universal_Parser SHALL 在配置中支持选择不同的多模态 LLM 模型（如 GPT-4o、Qwen-VL、MiniCPM-V 等）

### 需求 17：解析质量评分与人工审核

**用户故事：** 作为知识库管理员，我希望系统自动评估每份文档的解析质量，并将可疑结果推送给我审核，以便及时发现和修正问题。

#### 验收标准

1. WHEN 文档解析与清洗完成, THE Pipeline SHALL 计算 Parse_Quality_Score，评分维度包含：文本保留率（权重 30%）、标题层级识别率（权重 25%）、表格完整率（权重 20%）、数值保护率（权重 15%）、噪声去除率（权重 10%）
2. IF Parse_Quality_Score 低于配置阈值（默认 0.7）, THEN THE Pipeline SHALL 将文档加入审核队列，标记为"待审核"状态
3. THE System SHALL 提供审核界面（后台管理），显示待审核文档列表，支持按评分排序、按 Profile 筛选、按空间筛选
4. WHEN 管理员打开审核界面, THE System SHALL 并排展示原始文档（PDF 预览或原文件下载）与解析后的 Markdown 结果，高亮显示识别到的标题、表格、公式
5. THE System SHALL 允许管理员在审核界面直接编辑解析结果（修正标题层级、调整分块边界、修正表格内容）并保存
6. WHEN 管理员修正解析结果并保存, THE Pipeline SHALL 重新执行分块、向量化和入库流程，使用修正后的内容
7. WHEN 管理员完成审核, THE System SHALL 收集修正数据作为 Profile 优化样本（存储原文本、修正后文本、使用的 Profile），供后续 Profile 调优参考

### 需求 18：检索质量反馈与持续迭代

**用户故事：** 作为知识库用户，我希望能对搜索和问答结果进行反馈，以便系统不断优化检索质量。

#### 验收标准

1. WHEN 用户查看搜索结果或 AI 回答, THE System SHALL 在每条结果/回答旁显示反馈按钮（点赞/点踩/标注问题）
2. WHEN 用户标注问题, THE System SHALL 提供问题类型选项：结果不相关、缺少关键信息、引用错误、格式问题、其他；并允许用户填写文字说明（不超过 500 字符）
3. THE System SHALL 将反馈数据记录到 PostgreSQL，包含：查询内容、返回结果、用户 ID、反馈类型、反馈时间、文字说明、相关的 Document_Profile
4. THE System SHALL 提供反馈分析面板（后台管理），支持按以下维度聚合：按 Document_Profile、按文档、按查询类型、按时间范围
5. WHEN 反馈分析面板显示错误样本, THE System SHALL 自动识别错误模式（如相同 Profile 下多次出现相同类型错误）并给出优化建议（如"该 Profile 的分块大小建议调整为 X"）
6. THE System SHALL 支持管理员基于反馈数据一键更新 Document_Profile 或领域词典，并在更新后触发受影响文档的重新处理
7. WHEN Profile 或词典被更新, THE System SHALL 异步重新处理受影响的文档（批量重新分块、重新向量化、重新入库），处理过程不阻塞用户查询

### 需求 19：插件化扩展机制

**用户故事：** 作为系统开发者，我希望以插件形式扩展解析器、分块器和清洗器，以便在遇到全新格式或特殊处理需求时快速扩展系统能力。

#### 验收标准

1. THE System SHALL 定义以下插件接口：Parser_Plugin（输入文件 → 输出原始文本块和元数据）、Cleaner_Plugin（输入文本 → 输出清洗后文本）、Chunker_Plugin（输入文本 → 输出分块列表）
2. THE System SHALL 通过配置文件注册插件，配置项包含：插件名称、Python 导入路径、支持的文件格式列表、优先级
3. WHEN 系统启动, THE System SHALL 自动加载所有已注册的插件，失败的插件记录错误日志但不阻塞系统启动
4. THE System SHALL 支持插件热加载，无需重启服务即可加载新插件或更新插件配置
5. WHEN 处理特定格式文档, THE Pipeline SHALL 根据文件类型自动选择优先级最高的对应插件
6. THE System SHALL 预置以下插件：PDF 解析器（基于 Marker）、DOCX 解析器（基于 python-docx）、PPTX 解析器、HTML 解析器（基于 trafilatura）、通用清洗器、智能分块器
7. THE System SHALL 提供插件开发文档和示例代码，指导开发者扩展新插件

### 需求 20：领域词典管理

**用户故事：** 作为知识库管理员，我希望能够维护行业专业术语词典，以便提升专业文档的分词和检索质量。

#### 验收标准

1. THE System SHALL 支持多个 Domain_Dictionary 并存，每个词典包含：词典名称、描述、术语列表（词 + 词性 + 权重）、同义词组、停用词列表
2. THE System SHALL 将启用的领域词典同步到 OpenSearch IK 分词器的自定义词库，并支持热更新（无需重启 OpenSearch）
3. WHEN Document_Profile 关联某个领域词典, THE Search_Engine SHALL 在检索时使用该词典进行分词和查询改写
4. THE System SHALL 提供词典管理界面（后台管理），支持添加/删除/编辑术语、导入/导出词典（CSV 或 JSON 格式）
5. WHEN 管理员添加术语, THE System SHALL 校验术语格式（长度 1-30 字符，不含特殊控制字符），校验通过后立即生效
6. THE System SHALL 支持从用户反馈或文档内容中自动提取候选术语（基于词频和未识别词统计），推荐给管理员审核加入词典
