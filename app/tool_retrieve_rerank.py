from typing import List, Dict, Any
from langchain_core.tools import tool
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

from state import AgentState
import config
# ==========================================
# 1. 相对路径配置区
# ==========================================
CHROMA_DB_PATH = "../data/chroma_local_db"
EMBEDDING_MODEL_PATH = "../models/bge-m3/Xorbits/bge-m3"
RERANKER_MODEL_PATH = "../models/bge-reranker-v2-m3"

COLLECTION_NAME = "pkulaw_knowledge_base"

# ==========================================
# 2. 全局模型与数据库初始化
# ==========================================
print("🔄 [初始化] 正在连接本地 ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

print("🧠 [初始化] 正在加载本地 BGE-M3 Embedding 模型...")
# 【修改点 1】直接使用 SentenceTransformer 加载模型，不走 Chroma 的包装器
embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)

# 【修改点 2】获取 Collection 时，不要传 embedding_function，绕过配置冲突
collection = chroma_client.get_collection(name=COLLECTION_NAME)

print("🎯 [初始化] 正在加载本地 BGE-Reranker-v2 模型...")
cross_encoder_model = CrossEncoder(RERANKER_MODEL_PATH)
print("✅ 所有本地依赖加载完毕！\n" + "=" * 40)


# ==========================================
# 3. LangChain 工具定义
# ==========================================
@tool
def retrieve_and_rerank_tool(state: AgentState) -> List[dict]:
    """
    作用：本地法律知识库的向量检索与精排工具。
    调用指南：直接调用 `retrieve_and_rerank_tool()` 即可，不需要传参数。
    它会自动判断使用改写后的词还是原始提问进行检索。
    """
    # 动态参数判断：优先使用改写后的词，如果没有，退回到原始请求
    query_to_search = state.get('rewrite_request') or state['original_request']

    # 粗排和精排参数在这里可以写死，或者从 state 里动态读取（如果你以后需要高级控制）
    recall_limit = config.RECALL_LIMIT
    rerank_limit = config.RERANK_LIMIT

    print(f"\n[Tool 2] 启动检索任务 | 粗排: Top-{recall_limit} | 精排: Top-{rerank_limit}\n"
          f"| 使用的检索词: '{query_to_search}'")

    # --------------------------------------------------
    # 步骤 A: 向量检索 (粗排 - Recall)
    # --------------------------------------------------
    # 【修改点 3】手动将文本编码为向量，并转为 list 格式
    query_vector = embedding_model.encode(query_to_search).tolist()

    # 【修改点 4】使用 query_embeddings 而不是 query_texts 进行纯向量搜索
    search_results = collection.query(
        query_embeddings=[query_vector],
        n_results=recall_limit
    )

    candidate_docs = []
    if search_results and search_results['documents'] and len(search_results['documents'][0]) > 0:
        docs = search_results['documents'][0]
        metas = search_results['metadatas'][0] if search_results['metadatas'] else [{}] * len(docs)
        ids = search_results['ids'][0]

        for doc, meta, doc_id in zip(docs, metas, ids):
            candidate_docs.append({
                "id": doc_id,
                "content": doc,
                **meta
            })

    print(f"  - 粗排阶段：成功从 ChromaDB 召回 {len(candidate_docs)} 条候选切片。")

    if not candidate_docs:
        print("  - 警告：未检索到任何相关文档！")
        return []

    # --------------------------------------------------
    # 步骤 B: 语义重排 (精排 - Rerank)
    # --------------------------------------------------
    rerank_pairs = [[query_to_search, doc['content']] for doc in candidate_docs]
    scores = cross_encoder_model.predict(rerank_pairs)

    for i, score in enumerate(scores):
        candidate_docs[i]['rerank_score'] = float(score)

    reranked_results = sorted(candidate_docs, key=lambda x: x['rerank_score'], reverse=True)

    # --------------------------------------------------
    # 步骤 C: 截取 Top-K 返回
    # --------------------------------------------------
    final_results = reranked_results[:rerank_limit]
    print(f"  - 精排阶段：已筛选出得分最高的 {len(final_results)} 条结果。")

    return final_results


# ==========================================
# 4. 本地测试入口 (Main 函数)
# ==========================================
if __name__ == "__main__":
    import json

    print("🚀 开始独立测试 retrieve_and_rerank_tool 工具...")
    test_query = "船舶所有权注销 报废船舶 海事管理 登记程序"

    test_args = {
        "optimized_query": test_query,
        "recall_limit": 100,
        "rerank_limit": 10
    }

    try:
        results = retrieve_and_rerank_tool.invoke(test_args)

        print("\n🏆 --- 最终精排输出结果 ---")
        print(json.dumps(results, indent=4, ensure_ascii=False))

    except Exception as e:
        print(f"\n❌ 运行出错: {e}")