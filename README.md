<div align="center">

<h1>AiAgentPractice</h1>

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/DeepSeek-API-4A90D9?style=flat-square&logo=openai&logoColor=white" />
  <img src="https://img.shields.io/badge/FlashRAG-RUC%20NLP-FF6B6B?style=flat-square" />
  <img src="https://img.shields.io/badge/LangChain-v1.x-1C3C3C?style=flat-square&logo=langchain&logoColor=white" />
  <img src="https://img.shields.io/badge/LangGraph-1.2.9-4B8BBE?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" />
</p>

<p>基于 DeepSeek · FlashRAG · LangChain · LangGraph 的 AI Agent 实战系列</p>
<p>以 <b>RAG 基础手写 → 研究级 RAG 框架 → LLM 应用框架 → Agent 编排框架</b> 为主线，循序渐进覆盖 AI Agent 核心技术栈</p>
<p>每个文件均配有详细中文注释，适合 AI Agent 入门与进阶学习</p>

</div>

---

## 📋 目录

- [项目概览](#-项目概览)
- [目录结构](#-目录结构)
- [文件介绍](#-文件介绍)
- [学习路径](#-学习路径)
- [环境依赖](#️-环境依赖)
- [快速开始](#-快速开始)

---

## 🗺 项目概览

| #   | 文件                                                                                               | 核心技术                                                                                         | 亮点                                                                          |
| --- | ------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| 01  | RAG 基础实现（DeepSeek + SentenceTransformers + FAISS）                                                | RAG · FAISS · SentenceTransformers · DeepSeek API · python-dotenv · dataclass               | 不依赖任何 RAG 框架，手写完整检索增强生成全链路；嵌入模型四级自动降级策略（ST → HF → TF-IDF → 随机）             |
| 02  | FlashRAG 全流程实战向导                                                                                | FlashRAG · Dense/BM25 检索 · CrossReranker · Refiner · Pipeline · YAML 配置                    | RUC NLP 团队出品的研究级 RAG 框架；涵盖索引构建 → 检索 → 重排序 → 精炼 → 生成 → 评估完整链路，共 12 章节      |
| 03  | LangChain 全流程实战向导                                                                               | LangChain v1.x · LCEL · Chat Model · Prompt · RAG · Agent · Memory · Structured Output      | GitHub ⭐ 142k+ LLM 应用框架；覆盖 Chat Model / LCEL 管道 / RAG / Tool & Agent 共 13 章节 |
| 04  | LangGraph 全流程实战向导                                                                               | LangGraph 1.2.9 · StateGraph · Node · Edge · Streaming · Checkpointer · SubGraph · 多 Agent | 有向图驱动的 AI Agent 编排框架；覆盖 State / Node / Edge / 持久化 / 时间旅行 / 多 Agent 共 14 章节  |

---

## 📁 目录结构

```
AiAgentPractice/
├── 📓 01.RAGBasicProcessWithoutFlashRAG_SentenceTransformersEmbedding_FaissVectorStore.ipynb
├── 📄 02.FlashRAG_WorkflowGuide.md
├── 📄 03.LangChain_WorkflowGuide.md
├── 📄 04.LangGraph_WorkflowGuide.md
│
├── 📂 model_cache/                    # 模型权重缓存目录（gitignored）
│   ├── models--sentence-transformers--all-MiniLM-L6-v2/   # [自动生成] 英文语义嵌入模型（Notebook 01）
│   └── models--shibing624--text2vec-base-chinese/          # [自动生成] 中文语义嵌入模型（Notebook 01）
│
├── 📄 .env                            # API Key 配置文件（gitignored，需自行创建）
├── 📄 .gitignore
├── 📄 LICENSE
└── 📄 README.md
```

---

## 📚 文件介绍

### 01. RAG 基础实现

> `01.RAGBasicProcessWithoutFlashRAG_SentenceTransformersEmbedding_FaissVectorStore.ipynb`

**不依赖任何 RAG 框架**，从零手写完整 **RAG（Retrieval-Augmented Generation）** 系统，涵盖文档分块、向量化、FAISS 索引构建、相似度检索、Prompt 构造及 DeepSeek 生成回答全链路。嵌入模型按优先级自动降级（SentenceTransformers → HuggingFace Transformers → TF-IDF → 随机向量），保证在任意环境下均可运行。


| 章节                              | 内容                                                                                     |
| ------------------------------- | -------------------------------------------------------------------------------------- |
| 导入依赖库                           | requests · numpy · sentence-transformers · faiss · python-dotenv · dataclasses         |
| Document 文档数据结构定义               | Python dataclass 设计 · id / content / metadata 字段 · 自动生成 `__init__` / `__repr__`      |
| DeepSeekClient — DeepSeek API 客户端 | OpenAI 兼容 `/chat/completions` REST API · Bearer Token 鉴权 · 多轮对话消息列表                  |
| DocumentProcessor — 文档分块处理器      | 段落感知两步切分策略 · chunk\_size 贪心合并 · overlap 支持 · 不在段落内部硬截断                               |
| EmbeddingModel — 嵌入模型包装器         | 四级自动降级策略（ST → HF → TF-IDF → 随机）· L2 归一化 · 统一 `encode()` 接口                          |
| VectorStore — 向量索引与检索            | FAISS `IndexFlatIP` 内积索引 · `add()` 批量入库 · `search()` Top-K 相似检索                      |
| RAGSystem — RAG 系统主类             | 文档入库 → 问题向量化 → Top-K 检索 → Prompt 拼接 → DeepSeek 生成回答完整闭环                             |
| 演示与运行                           | `.env` 文件结构说明 · `load_dotenv()` 原理与文件名无关性详解 · 端到端问答流程演示                              |


> **RAG 核心流程** · 原始文档经 `DocumentProcessor` 切分为固定大小的文本块，由 `EmbeddingModel` 向量化后存入 `VectorStore`（FAISS 索引）；用户提问时，将问题向量化并在 FAISS 中检索 Top-K 相似块，将检索结果与问题拼接为 Prompt，最终由 DeepSeek LLM 生成上下文感知的回答。
>
> 参考：*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*（Lewis et al., NeurIPS 2020）

---

### 02. FlashRAG 全流程实战向导

> `02.FlashRAG_WorkflowGuide.md`

**FlashRAG** 是由中国人民大学 NLP 团队开源的研究级 RAG 框架，所有 API 均经官方仓库 `main` 分支源码对照验证。本向导按照框架真实执行路径，完整覆盖从知识库构建到结果评估的端到端全链路，共 12 章节。


| 章节              | 内容                                                                             |
| --------------- | ------------------------------------------------------------------------------ |
| 一、安装与环境配置       | `flashrag-dev` · `faiss-cpu` · `bm25s` · `datasets` · `openai` · `tiktoken`   |
| 二、架构概览          | 八步数据流（分块 → JSONL → 索引 → 检索 → 重排 → 精炼 → 生成 → 评估）· 组件一览表                       |
| 三、配置系统          | `Config` 类 · YAML 驱动配置 · `basic_config.yaml` 默认值覆盖机制 · 路径规则                   |
| 四、知识库构建         | `RecursiveCharacterTextSplitter` 文档分块 · JSONL 语料格式 · `Index_Builder.build_index()` |
| 五、检索器           | `DenseRetriever`（向量检索）· `BM25Retriever`（稀疏检索）· `MultiRetrieverRouter`（多路混合）  |
| 六、重排序器          | `CrossReranker` · `BiReranker` · `reranker.rerank(queries, docs)` 返回值解析        |
| 七、精炼器           | `ExtractiveRefiner` · `AbstractiveRefiner` · `LLMLinguaRefiner` · 三种压缩策略对比    |
| 八、生成器           | `OpenaiGenerator`（API 调用）· `HFCausalLMGenerator`（本地模型）· `PromptTemplate`      |
| 九、Pipeline 实战   | `SequentialPipeline`（Naive RAG）· `FLAREPipeline`（主动检索）· 自定义 Pipeline 实现       |
| 十、结果评估          | `Evaluator` · EM / F1 / ROUGE 指标 · `dataset.save_result()` 结果持久化              |
| 十一、完整端到端流程      | 一键跑通：配置 → 索引 → 检索 → 生成 → 评估全链路                                               |
| 十二、总结           | 组件速查表 · 常见错误排查 · 与 LangChain 对比                                               |


> **FlashRAG 架构要点** · 以 YAML 配置文件为驱动，`Config` 类统一管理所有参数并设定默认值；`Index_Builder` 负责语料 JSONL 化与索引构建；Pipeline 类将检索器、重排序器、精炼器、生成器串联为完整流水线，支持 `Naive / FLARE / SelfRAG` 多种 Pipeline 模式；`Evaluator` 提供 EM、F1、ROUGE 等标准评测指标。
>
> 参考：*FlashRAG: A Modular Toolkit for Efficient Retrieval-Augmented Generation Research*（Jin et al., RUC NLP, 2024）

---

### 03. LangChain 全流程实战向导

> `03.LangChain_WorkflowGuide.md`

**LangChain**（GitHub ⭐ 142k+）是目前最流行的 LLM 应用开发框架，所有 API 均经官方 `master` 分支源码对照验证，对应 **LangChain v1.x / 2026**。本向导以统一 **Runnable 接口 + LCEL 管道语法（`|`）** 为核心，覆盖从简单问答到复杂多步 Agent 的全部关键模块，共 13 章节。

> **版本说明**：2025 年 10 月 LangChain v1.0 正式发布；遗留类（`AgentExecutor` / `LLMChain` 等）移至 `langchain-classic` 维护；v0.3.x 维护至 2026 年 12 月。


| 章节                | 内容                                                                                                                                    |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| 一、安装与环境配置         | `langchain` · `langchain-openai` · `langchain-community` · `langchain-chroma` · `python-dotenv`                                       |
| 二、LangChain 架构概览  | Runnable 接口 · LCEL 组合范式 · 核心组件一览表 · 包结构说明                                                                                           |
| 三、Chat Model      | `init_chat_model()` 统一初始化 · `ChatOpenAI` · 流式输出 · 批量调用                                                                               |
| 四、Prompt Template | `ChatPromptTemplate` · `from_messages()` · 变量插值 · Few-Shot 少样本提示词                                                                     |
| 五、Output Parser   | `StrOutputParser` · `JsonOutputParser` · `CommaSeparatedListOutputParser`                                                             |
| 六、LCEL 链式组合      | `\|` 管道操作符 · `RunnablePassthrough` · `RunnableLambda` · `RunnableParallel` · 串联多个管道                                                      |
| 七、对话记忆            | `InMemoryChatMessageHistory` · `RunnableWithMessageHistory` · 消息裁剪与窗口控制                                                               |
| 八、RAG             | 文档加载 · `RecursiveCharacterTextSplitter` · Chroma 向量库 · 完整 RAG 链 · 高级检索器                                                               |
| 九、Tool 与 Agent    | `@tool` 装饰器 · `StructuredTool` · 内置工具 · MCP 工具接入 · `create_agent()` · 工具异常处理                                                          |
| 十、结构化输出           | `with_structured_output()` · Pydantic 模型约束 · JSON Schema 输出                                                                         |
| 十一、完整端到端流程        | RAG + Agent 一键跑通：文档入库 → 检索 → LLM 生成 → 工具调用                                                                                          |
| 十二、补充知识点          | LLMs vs Chat Model · Token · 消息类型 · 异步调用 · 多平台接入 · PromptTemplate · RunnablePassthrough.assign · RunnableParallel · RunnableBranch · create_stuff_documents_chain · Callbacks · 旧链迁移 |
| 十三、总结             | 组件速查表 · LCEL vs 旧式 Chain · 框架选型建议                                                                                                    |


> **LCEL 核心优势** · 所有组件（Chat Model / Prompt / Parser / Retriever / Tool）均实现统一 `Runnable` 接口（`invoke` / `batch` / `stream`），通过 `|` 管道操作符自由组合；LCEL 链天然支持流式输出、批量处理、异步调用及 LangSmith 追踪，无需修改代码即可切换执行模式。
>
> 参考：[LangChain 官方文档](https://python.langchain.com/docs/introduction/)（LangChain v1.x / 2026）

---

### 04. LangGraph 全流程实战向导

> `04.LangGraph_WorkflowGuide.md`

**LangGraph**（GitHub ⭐ 37k+）是构建有状态、多 Actor LLM 应用的低阶编排框架，所有 API 均经官方 `main` 分支源码对照验证，对应 **LangGraph 1.2.9 / 2026**。本向导以 **有向图（StateGraph）** 为核心抽象，深度覆盖从单节点图到复杂多 Agent 系统的全部关键特性，共 14 章节。

> **核心理念**：LangGraph 将 AI 工作流建模为 **节点（Node）** 和 **边（Edge）** 的有向图，天然支持循环、条件分支、多 Agent 并行协作以及状态持久化与时间旅行回溯。


| 章节                   | 内容                                                                                            |
| -------------------- | --------------------------------------------------------------------------------------------- |
| 一、安装与环境配置            | `langgraph==1.2.9` · `langgraph-checkpoint-sqlite` · `langgraph-supervisor` · `python-dotenv` |
| 二、架构概览               | 四大核心要素（State / Node / Edge / Graph）· `StateGraph` 执行本质 · 最小完整示例（Hello LangGraph）           |
| 三、State（状态）          | `TypedDict` / `Pydantic` 状态定义 · `Annotated` + `Reducer` 状态合并策略 · `StateSchema` 状态模式          |
| 四、Node（节点）           | 函数签名规范 · `functools.partial` 注入参数 · `RetryPolicy` 自动重试 · `CachePolicy` 结果缓存                  |
| 五、Edge（边）            | 普通边（无条件跳转）· 条件边（`add_conditional_edges`）· 条件入口点（`set_conditional_entry_point`）             |
| 六、Graph 构建流程         | `StateGraph` → `add_node` → `add_edge` → `compile()` · 可视化图结构 · 链式 Builder 写法               |
| 七、特殊 API             | `Send`（动态扇出并行）· `Command`（节点内跳转 + 状态更新）· `RuntimeContext`（运行时上下文注入）                         |
| 八、Streaming 流式输出     | `stream()` 四种模式（values / updates / messages / debug）· `interrupt_before` / `interrupt_after` 节点级暂停 |
| 九、持久化（Checkpointer）  | `InMemorySaver` · `SqliteSaver` · `thread_id` 多会话隔离 · 中断恢复                                   |
| 十、时间旅行（Time Travel）  | `get_state()` · `get_state_history()` · `update_state()` · 历史状态回放与分支重执行                     |
| 十一、子图（SubGraph）      | State 共享与不共享两种嵌套模式 · 父子图状态传递协议 · `CompiledGraph` 作为父图节点                                     |
| 十二、多 Agent 系统        | `create_agent()` 创建单 Agent · Supervisor 模式（`langgraph-supervisor`）· Agent 四种 Stream 模式 · `handoff` Agent 间控制权移交  |
| 十三、完整端到端实战           | 旅行预订助手：State 定义 → 多节点图 → 工具调用 → 条件路由 → 持久化一键跑通                                              |
| 十四、总结                | 核心知识点回顾 · 框架选型决策树 · 知识体系全景图 · 常用命令速查                                                        |


> **LangGraph 与 LangChain Agent 的区别** · LangChain Agent（LCEL 链）适合线性的单轮工具调用场景；LangGraph 以有向图为核心，支持**循环**（节点可多次执行）、**条件分支**（条件边动态路由）、**状态持久化**（Checkpointer 跨会话恢复）及**多 Agent 协作**（子图 + Supervisor），是构建复杂、有状态 AI Agent 应用的首选方案。
>
> 参考：[LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)（LangGraph 1.2.9 / 2026）

---

## 🛤 学习路径

```
RAG 基础实现层
  01 RAG 基础（DeepSeek + SentenceTransformers + FAISS，不依赖任何框架，手写全链路）
       ↓
RAG 框架层
  02 FlashRAG（RUC NLP 研究级 RAG 框架）
     ├── 知识库构建（文档分块 → JSONL → Dense/BM25 索引）
     ├── 检索 + 重排序 + 精炼（CrossReranker / Refiner）
     └── Pipeline 封装（Naive / FLARE / SelfRAG）→ 结果评估
       ↓
LLM 应用框架层
  03 LangChain v1.x（GitHub ⭐ 142k+ LLM 应用框架）
     ├── Chat Model · Prompt Template · LCEL 管道（|）
     ├── 对话记忆 · RAG 链 · Tool & Agent
     └── 结构化输出 · 异步 · Callbacks
       ↓
Agent 编排框架层
  04 LangGraph 1.2.9（有向图 Agent 编排框架）
     ├── State · Node · Edge · Graph 构建
     ├── Streaming · Checkpointer 持久化 · 时间旅行
     └── SubGraph 子图 · 多 Agent（Supervisor 模式）
```

建议按编号顺序学习：先通过 Notebook 01 不借助任何框架手写完整 RAG 链路，深入理解 RAG 每个环节的原理（01），再学习专为 RAG 研究设计的 FlashRAG 框架，掌握工业级 RAG Pipeline 的标准化组件（02），然后学习 LangChain 提供的通用 LLM 应用开发能力（03），最后进阶到 LangGraph 构建复杂有状态的多 Agent 系统（04）。

---

## ⚙️ 环境依赖

> **说明**：各文件所需依赖相互独立，按需安装对应组件依赖即可。所有向导均通过 `OPENAI_API_KEY` + `OPENAI_BASE_URL`（兼容 DeepSeek）或直接 `DEEPSEEK_API_KEY` 调用大语言模型。

**核心依赖** · Python 3.10+

**Notebook 01 — RAG 基础实现**

| 包名                    | 用途                                |
| --------------------- | --------------------------------- |
| `sentence-transformers` | 语义嵌入模型（首选，`all-MiniLM-L6-v2` 等）  |
| `faiss-cpu`           | FAISS 向量索引库（CPU 版）                |
| `numpy`               | 嵌入向量矩阵运算                          |
| `requests`            | 向 DeepSeek REST API 发送 HTTP 请求    |
| `python-dotenv`       | 从 `.env` 文件加载 `DEEPSEEK_API_KEY` |

**Guide 02 — FlashRAG**

| 包名                        | 用途                                        |
| ------------------------- | ----------------------------------------- |
| `flashrag-dev`            | FlashRAG 框架本体（PyPI 包名，加 `--pre` 安装预发布版）  |
| `faiss-cpu`               | Dense 向量检索索引（推荐 `conda install` 以避免兼容问题） |
| `bm25s`                   | BM25 稀疏检索后端                               |
| `datasets`                | HuggingFace 数据集加载（FlashRAG 内部依赖）           |
| `openai`                  | OpenAI Python SDK（兼容 DeepSeek 等 API 端点）   |
| `tiktoken`                | OpenAI tokenizer（`OpenaiGenerator` 内部依赖）  |
| `langchain-text-splitters` | 文档分块工具（FlashRAG 无内置分块）                    |

**Guide 03 — LangChain v1.x**

| 包名                     | 用途                                          |
| ---------------------- | ------------------------------------------- |
| `langchain`            | 框架本体（当前为 v1.x，含 LCEL / Agent / Chain 高级抽象） |
| `langchain-openai`     | OpenAI / DeepSeek 模型集成（`ChatOpenAI` 等）      |
| `langchain-community`  | 社区集成（FAISS / BM25Retriever / 文档加载器等）        |
| `langchain-chroma`     | Chroma 向量数据库集成（RAG 章节）                      |
| `langchain-classic`    | 遗留 API 兼容包（`AgentExecutor` / `LLMChain` 等）  |
| `python-dotenv`        | 加载 API Key 环境变量                             |

**Guide 04 — LangGraph 1.2.9**

| 包名                          | 用途                               |
| --------------------------- | -------------------------------- |
| `langgraph==1.2.9`          | LangGraph 图框架本体（指定版本确保兼容性）      |
| `langchain`                 | LangGraph 的 LLM 集成依赖             |
| `langchain-openai`          | OpenAI / DeepSeek 模型适配器          |
| `langgraph-checkpoint-sqlite` | SQLite 持久化支持（本地开发 Checkpointer）  |
| `langgraph-supervisor`      | Supervisor 多 Agent 编排支持           |
| `python-dotenv`             | 加载 `OPENAI_API_KEY` 等环境变量        |

---

## 🚀 快速开始

**1. 克隆仓库**

```bash
git clone git@github.com:lisirui-ai/AiAgentPractice.git
cd AiAgentPractice
```

**2. 配置 API Key**

在项目根目录创建 `.env` 文件（已加入 `.gitignore`，不会提交到 Git）：

```bash
# .env
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# LangChain / LangGraph 向导使用 OpenAI 兼容端点（可指向 DeepSeek）
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.deepseek.com/v1
```

**3. 安装依赖**

根据需要学习的章节，按需安装对应依赖：

```bash
# Notebook 01 — RAG 基础实现
pip install sentence-transformers faiss-cpu numpy requests python-dotenv

# Guide 02 — FlashRAG（faiss 推荐用 conda 安装）
pip install flashrag-dev --pre bm25s datasets openai tiktoken langchain-text-splitters
conda install -c pytorch faiss-cpu=1.8.0  # 推荐，避免兼容问题

# Guide 03 — LangChain v1.x
pip install langchain langchain-openai langchain-community langchain-chroma langchain-classic python-dotenv

# Guide 04 — LangGraph 1.2.9
pip install langgraph==1.2.9 langchain langchain-openai langgraph-checkpoint-sqlite langgraph-supervisor python-dotenv
```

**4. 运行 Notebook**

```bash
jupyter notebook
```

打开 `01.RAGBasicProcessWithoutFlashRAG_SentenceTransformersEmbedding_FaissVectorStore.ipynb` 开始学习。

**5. 阅读工作流向导**

`02.FlashRAG_WorkflowGuide.md` / `03.LangChain_WorkflowGuide.md` / `04.LangGraph_WorkflowGuide.md` 均为带完整可运行代码块的 Markdown 向导，可直接在 IDE 中阅读，也可将代码片段粘贴至 Notebook 中逐步执行。

> ⚠️ Notebook 01 中模型（`sentence-transformers/all-MiniLM-L6-v2` 等）在首次运行时会**自动下载**并缓存至 `model_cache/` 目录，无需手动下载。

---

## 📄 License

[MIT](LICENSE) © 2026
