import numpy as np
import torch
from typing import List, Dict, Any
from langchain_core.tools import tool
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import time
import csv
import json
import os
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
# 2. 硬件检测与全局模型初始化
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔄 [初始化] 硬件加速: {device.upper()} | 连接 ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

print(f"🧠 [初始化] 加载 Embedding 模型 ({device.upper()})...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH, device=device)
collection = chroma_client.get_collection(name=COLLECTION_NAME)

print(f"🎯 [初始化] 加载 Reranker 模型 ({device.upper()})...")
cross_encoder_model = CrossEncoder(RERANKER_MODEL_PATH, device=device)
print("✅ 本地模型加载完毕！\n" + "=" * 40)


# ==========================================
# 3. LangChain 工具定义
# ==========================================
@tool
def retrieve_and_rerank_tool(state: AgentState) -> Dict[str, Any]:
    """
    作用：本地法律知识库检索与精排。
    字段精简：仅返回 id, title, content, rerank_score
    """
    start_time = time.time()
    # 优先使用改写后的词
    query_to_search = state.get('rewrite_request') or state.get('original_request', '')
    recall_limit = config.RECALL_LIMIT
    rerank_limit = config.RERANK_LIMIT

    # A: 粗排
    query_vector = embedding_model.encode(query_to_search).tolist()
    search_results = collection.query(query_embeddings=[query_vector], n_results=recall_limit)

    candidate_docs = []
    if search_results and search_results['documents'] and len(search_results['documents'][0]) > 0:
        docs = search_results['documents'][0]
        metas = search_results['metadatas'][0] if search_results['metadatas'] else [{}] * len(docs)
        ids = search_results['ids'][0]
        for doc, meta, doc_id in zip(docs, metas, ids):
            candidate_docs.append({"id": doc_id, "content": doc, **meta})

    if not candidate_docs:
        return {"retrieval_results": [], "timing_stats": state.get('timing_stats', {})}

    # B: 精排
    rerank_pairs = [[query_to_search, doc['content']] for doc in candidate_docs]
    scores = cross_encoder_model.predict(rerank_pairs)
    for i, score in enumerate(scores):
        candidate_docs[i]['rerank_score'] = float(score)
    reranked_results = sorted(candidate_docs, key=lambda x: x['rerank_score'], reverse=True)

    # C: 截取并精简字段
    top_k_results = reranked_results[:rerank_limit]
    final_results = []
    for res in top_k_results:
        final_results.append({
            "id": res["id"],
            "title": res.get("law_title", "未知法律"),
            "content": res["content"],
            "rerank_score": res["rerank_score"]
        })

    elapsed_time = round(time.time() - start_time, 4)
    timing_stats = state.get('timing_stats', {}).copy()
    timing_stats['retrieval'] = timing_stats.get('retrieval', 0) + elapsed_time

    return {"retrieval_results": final_results, "timing_stats": timing_stats}


# ==========================================
# 4. 批量评测脚本 (从 Rewrite CSV 读取 -> 检索 -> 存入 Retrieve CSV)
# ==========================================
if __name__ == "__main__":
    # 配置路径
    REWRITE_CSV = "../output/rewrite_test_result.csv"
    SOURCE_JSONL = "../data/test_set/web.jsonl"  # 用于加载标准答案
    RETRIEVE_CSV = "../output/retrieve_test_result.csv"
    EVAL_TOP_K = 5

    print(f"🚀 开始全链路检索质量评测...")

    # 1. 预加载标准答案 (以 ID 为键)
    ground_truth_map = {}
    if os.path.exists(SOURCE_JSONL):
        with open(SOURCE_JSONL, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                ground_truth_map[str(item['id'])] = [str(gid) for gid in item.get('golden_doc_ids', [])]
        print(f"📖 成功加载了 {len(ground_truth_map)} 条标准答案。")

    if not os.path.exists(REWRITE_CSV):
        print(f"❌ 找不到改写结果文件: {REWRITE_CSV}")
        exit()

    # 指标累加器
    all_metrics = {"hit": [], "recall": [], "precision": [], "mrr": []}

    # 2. 准备读写
    with open(REWRITE_CSV, 'r', encoding='utf-8-sig') as f_in, \
            open(RETRIEVE_CSV, 'w', encoding='utf-8-sig', newline='') as f_out:

        reader = csv.DictReader(f_in)

        # 🌟 动态构造表头：包含 id, title, content, score
        fieldnames = reader.fieldnames + []
        for i in range(1, EVAL_TOP_K + 1):
            fieldnames.extend([f"top_{i}_id", f"top_{i}_title", f"top_{i}_content", f"top_{i}_score"])
        fieldnames.extend(["hit@5", "recall@5", "precision@5", "mrr", "retrieval_time_sec"])

        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            q_id = row['id']
            # 这里的 rewritten_query 是上一个脚本生成的
            query_to_use = row.get('rewritten_query') or row.get('original_question')
            golden_ids = set(ground_truth_map.get(q_id, []))

            # 🌟 完美补全测试状态，防止 Pydantic 报错
            test_state = {
                "original_request": row.get('original_question', ''),
                "rewrite_request": row.get('rewritten_query', ''),
                "clarification_question": None,
                "plan": [],
                "intermediate_steps": [],
                "verification_history": [],
                "final_response": "",
                "loop_count": 0,
                "timing_stats": {}
            }

            try:
                # 执行检索工具
                tool_output = retrieve_and_rerank_tool.invoke({"state": test_state})
                results = tool_output.get("retrieval_results", [])
                timing = tool_output.get("timing_stats", {})

                # 提取 Top-K 信息
                retrieved_ids = [str(r['id']) for r in results[:EVAL_TOP_K]]

                # 计算指标
                correct_indices = [idx for idx, rid in enumerate(retrieved_ids) if rid in golden_ids]
                hit_k = 1 if correct_indices else 0
                recall_k = len(correct_indices) / len(golden_ids) if golden_ids else 0
                precision_k = len(correct_indices) / EVAL_TOP_K
                mrr = 1.0 / (correct_indices[0] + 1) if correct_indices else 0

                # 填充 CSV 数据行
                new_row = row.copy()
                for i in range(EVAL_TOP_K):
                    if i < len(results):
                        res = results[i]
                        new_row[f"top_{i + 1}_id"] = res["id"]
                        new_row[f"top_{i + 1}_title"] = res["title"]
                        new_row[f"top_{i + 1}_content"] = res["content"].replace("\n", " ")  # 换行转空格，防止 CSV 错位
                        new_row[f"top_{i + 1}_score"] = round(res["rerank_score"], 4)
                    else:
                        new_row[f"top_{i + 1}_id"] = ""
                        new_row[f"top_{i + 1}_title"] = ""
                        new_row[f"top_{i + 1}_content"] = ""
                        new_row[f"top_{i + 1}_score"] = ""

                new_row.update({
                    "hit@5": hit_k,
                    "recall@5": round(recall_k, 4),
                    "precision@5": round(precision_k, 4),
                    "mrr": round(mrr, 4),
                    "retrieval_time_sec": timing.get("retrieval", 0)
                })

                writer.writerow(new_row)

                # 汇总
                all_metrics["hit"].append(hit_k)
                all_metrics["recall"].append(recall_k)
                all_metrics["precision"].append(precision_k)
                all_metrics["mrr"].append(mrr)

                print(f"✅ ID {q_id} 评测完成 | Hit: {hit_k} | Recall: {recall_k:.2f}")

            except Exception as e:
                print(f"❌ ID {q_id} 运行时报错: {e}")
                writer.writerow(row)

    # 输出汇总结果
    print("\n" + "=" * 50)
    print("🏆 最终检索质量汇总报告 (Top-5):")
    for k, v in all_metrics.items():
        print(f"  - 平均 {k.upper()}: {np.mean(v):.4f}")
    print(f"💾 详细数据请查看: {RETRIEVE_CSV}")