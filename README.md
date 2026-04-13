# LAWQA

一个面向法律咨询场景的 Agentic RAG 项目。它会先判断问题是否足够明确，再根据问题自动制定检索计划，调用本地法律知识库、查询改写、联网搜索等工具，最后生成回答并经过审计节点检查。

## 项目结构

- `main.py`：FastAPI 示例接口，提供 `/` 和 `/hello/{name}`。
- `app/main_graph.py`：主工作流入口，编排整条 LangGraph 流程。
- `app/node_ambiguity_judgment.py`：判断问题是否过于模糊，需要先追问。
- `app/node_task_planner.py`：根据用户问题生成多步工具调用计划。
- `app/node_tool_executor.py`：按计划执行工具，并把结果写回状态。
- `app/node_auditor.py`：审计工具输出是否相关、是否足够回答问题。
- `app/node_generator.py`：整合所有线索，生成最终法律意见。
- `app/node_router.py`：根据审计结果和循环次数决定下一步流向。
- `app/tool_query_rewrite.py`：把口语化问题改写成适合检索的关键词。
- `app/tool_retrieve_rerank.py`：在本地 Chroma 知识库中召回并重排法条片段。
- `app/tool_scout_search.py`：调用百度千帆网页搜索，补充实时新闻事实。
- `app/vectorize_chroma.py`：把本地法律语料向量化并写入 ChromaDB。
- `app/llm_factory.py`：统一创建大模型客户端。
- `app/state.py`：定义整个 Agent 流程共享的状态结构。

## 工作流程

1. 先判断用户问题是否清晰。
2. 清晰则交给规划器生成工具调用计划。
3. 依次执行查询改写、本地检索、联网搜索等工具。
4. 审计每一步输出质量。
5. 低质量结果会被打回重做，超过循环上限后直接进入生成器兜底输出。
6. 最后由生成器整合法条、事实和新闻，输出法律意见。

## 运行环境

- Python 3.10+
- 本地向量数据库：ChromaDB
- 向量模型：`bge-m3`
- 重排模型：`bge-reranker-v2-m3`
- 大模型服务：DeepSeek、OpenAI 兼容接口、百度千帆搜索

## 安装依赖

仓库里没有单独的依赖文件，下面这些包是从代码里能看出来必须要装的：

```bash
pip install fastapi uvicorn python-dotenv langgraph langchain-core langchain-openai pydantic chromadb sentence-transformers torch pandas requests tqdm
```

## 环境变量

需要在项目根目录放一个 `.env` 文件，至少包含：

```env
DEEPSEEK_API_KEY=你的DeepSeekKey
LINO_API_KEY=你的OpenAI兼容接口Key
QIANFAN_API_KEY=你的百度千帆Key
```

可选配置：

```env
MAX_LOOPS=3
PASSING_SCORE=3
RECALL_LIMIT=15
RERANK_LIMIT=5
PLANNER_TEMP=0.0
GENERATOR_TEMP=0.3
```

## 本地知识库构建

先把法律语料向量化并写入 ChromaDB：

```bash
python app/vectorize_chroma.py
```

脚本默认会读取：

- `../data/data_set/pkulaw_combined_articles_chunked.csv`
- `../models/bge-m3/Xorbits/bge-m3`
- `../data/chroma_local_db`

如果你的目录结构不同，需要先改代码里的路径。

## 启动方式

### 运行法律 Agent 主流程

```bash
python app/main_graph.py
```

`app/main_graph.py` 里自带了两个测试样例：一个模糊提问，一个复杂法律问题。

### 测试 FastAPI 示例接口

```bash
uvicorn main:app --reload
```

访问：

- `GET /`
- `GET /hello/{name}`

## 说明

- 查询改写和本地检索都默认读取全局状态，不需要手动传参数。
- 审计节点会根据工具输出质量决定是否打回重做。
- 如果联网搜索不可用，系统仍然可以依赖本地知识库完成回答。
