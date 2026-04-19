import numpy as np
import torch
from typing import List, Dict, Any
from langchain_core.tools import tool
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import time
import csv
import json
from state import AgentState
import config

# ==========================================
# 1. 相对路径配置区
# ==========================================
CHROMA_DB_PATH = "../../data/chroma_local_db"
EMBEDDING_MODEL_PATH = "../../models/bge-m3/Xorbits/bge-m3"
RERANKER_MODEL_PATH = "../../models/bge-reranker-v2-m3"

COLLECTION_NAME = "pkulaw_knowledge_base"

# ==========================================
# 2. 硬件检测与全局模型初始化
# ==========================================
print("\n=== 💻 硬件加速状态自检 ===")
# 自动检测可用硬件：如果有 CUDA 就用显卡，否则用 CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"1. PyTorch 是否检测到了 GPU (CUDA)? : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"2. 显卡型号: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ 警告：当前环境未识别到 GPU，模型将使用 CPU 缓慢运行！请检查 PyTorch 版本。")
print("=============================\n")

print("🔄 [初始化] 正在连接本地 ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# 【核心修改 1】显式指定 device，强制让模型跑在检测到的硬件上
print(f"🧠 [初始化] 正在加载本地 BGE-M3 Embedding 模型 (运行在 {device.upper()})...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH, device=device)

collection = chroma_client.get_collection(name=COLLECTION_NAME)

# 【核心修改 2】显式指定 device
print(f"🎯 [初始化] 正在加载本地 BGE-Reranker-v2 模型 (运行在 {device.upper()})...")
cross_encoder_model = CrossEncoder(RERANKER_MODEL_PATH, device=device)

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
    # ⏱️ 开始计时
    start_time = time.time()

    # 动态参数判断：优先使用改写后的词，如果没有，退回到原始请求
    query_to_search = state.get('rewrite_request') or state.get('original_request', '')

    # 粗排和精排参数在这里可以写死，或者从 state 里动态读取（如果你以后需要高级控制）
    recall_limit = config.RECALL_LIMIT
    rerank_limit = config.RERANK_LIMIT

    print(f"\n[Tool 2] 启动检索任务 | 粗排: Top-{recall_limit} | 精排: Top-{rerank_limit}\n"
          f"| 使用的检索词: '{query_to_search}'")

    # --------------------------------------------------
    # 步骤 A: 向量检索 (粗排 - Recall)
    # --------------------------------------------------
    # 手动将文本编码为向量，并转为 list 格式
    query_vector = embedding_model.encode(query_to_search).tolist()

    # 使用 query_embeddings 而不是 query_texts 进行纯向量搜索
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
    top_k_results = reranked_results[:rerank_limit]

    final_results = []
    for res in top_k_results:
        final_results.append({
            "id": res["id"],
            "title": res.get("law_title", "未知法律"),  # 将原始的 law_title 映射为统一的 title
            "content": res["content"],
            "rerank_score": res["rerank_score"]  # 保留给底层评测代码用
        })

    print(f"  - 精排阶段：已筛选出得分最高的 {len(final_results)} 条结果。")

    # ⏱️ 计时结束，累计到 timing_stats
    end_time = time.time()
    elapsed_time = round(end_time - start_time, 4)

    timing_stats = state.get('timing_stats', {}).copy()
    timing_stats['retrieval'] = timing_stats.get('retrieval', 0) + elapsed_time
    print(f"  - 检索耗时: {elapsed_time}s | 累计检索时间: {timing_stats['retrieval']}s")

    # 返回结果时，也要更新 timing_stats
    return {
        "retrieval_results": final_results,
        "timing_stats": timing_stats
    }


# # ==========================================
# # 4. 本地测试入口 (Main 函数)
# # ==========================================
# if __name__ == "__main__":
#     import json
#
#     print("🚀 开始独立测试 retrieve_and_rerank_tool 工具...")
#
#     # 【核心修改 3】模拟真实的 AgentState 字典结构
#     test_state: AgentState = {
#         "original_request": "我想了解一下报废船舶的注销登记程序是什么？",
#         "rewrite_request": "船舶所有权注销 报废船舶 海事管理 登记程序",  # 工具会优先读取这个
#         "clarification_question": None,
#         "plan": [],
#         "intermediate_steps": [],
#         "verification_history": [],
#         "final_response": "",
#         "loop_count": 0
#     }
#
#     try:
#         # 直接传入完整的 state
#         results = retrieve_and_rerank_tool.invoke({"state": test_state})
#
#         print("\n🏆 --- 最终精排输出结果 ---")
#         print(json.dumps(results, indent=4, ensure_ascii=False))
#
#     except Exception as e:
#         print(f"\n❌ 运行出错: {e}")

# ==========================================
# 4. 本地测试入口 (Main 函数)
# ==========================================
if __name__ == "__main__":
    print("🚀 开始批量测试+评测 retrieve_and_rerank_tool 工具...")

    # 🔧 配置
    test_file = "../../data/test_set/mini_testset.jsonl"
    output_file = "../../output/暂时没用/retrieve_test_result.csv"
    eval_top_k = 5  # 评测指标计算到 top5

    print(f"📂 测试文件: {test_file}")
    print(f"📝 结果将输出到: {output_file}")
    print(f"🔍 评测指标将计算 Top-{eval_top_k} 的 hit/recall/precision/MRR")

    # 统计变量
    total = 0
    success = 0
    total_cost_time = 0.0
    error_cases = []

    # 评测指标的全局统计
    all_hit = {k: [] for k in range(1, eval_top_k + 1)}
    all_recall = {k: [] for k in range(1, eval_top_k + 1)}
    all_precision = {k: [] for k in range(1, eval_top_k + 1)}
    all_mrr = []

    try:
        with open(test_file, 'r', encoding='utf-8') as f, \
                open(output_file, 'w', encoding='utf-8', newline='') as out_f:

            # 🔧 动态生成 CSV 表头
            header = ['id']
            # 检索结果列
            for i in range(1, config.RERANK_LIMIT + 1):
                header.append(f'result_{i}_doc_id')
                header.append(f'result_{i}_score')
            # 评测相关列
            header.append('golden_doc_ids')
            for k in range(1, eval_top_k + 1):
                header.append(f'hit@{k}')
                header.append(f'recall@{k}')
                header.append(f'precision@{k}')
            header.append('mrr')
            header.append('cost_time')

            writer = csv.writer(out_f)
            writer.writerow(header)

            lines = f.readlines()
            total = len(lines)
            print(f"🔍 共加载 {total} 个测试用例\n")

            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    q_id = sample['id']
                    question = sample['question']
                    # 读取 golden 文档 id
                    golden_doc_ids = sample.get('golden_doc_ids', [])
                    # 统一转成字符串，防止类型不一致
                    golden_docs = {str(g) for g in golden_doc_ids}

                    print(f"\n{'=' * 60}")
                    print(f"【测试用例 {i + 1}/{total} | ID: {q_id}】")
                    print(f"用户问题: {question}")
                    print(f"Golden 正确文档: {golden_doc_ids}")

                    # 构造状态
                    test_state = {
                        "original_request": question,
                        "rewrite_request": None,  # 用原始问题检索
                        "clarification_question": None,
                        "plan": [],
                        "intermediate_steps": [],
                        "verification_history": [],
                        "final_response": "",
                        "loop_count": 0
                    }

                    # ⏱️ 计时开始
                    start_time = time.time()

                    # 调用检索工具
                    results = retrieve_and_rerank_tool.invoke({"state": test_state})

                    # ⏱️ 计时结束
                    end_time = time.time()
                    cost_time = round(end_time - start_time, 4)
                    total_cost_time += cost_time

                    # 🔍 计算评测指标
                    # 取出 top5 的检索结果 id，统一转字符串
                    retrieved_top5 = [str(res['id']) for res in results[:eval_top_k]]
                    # 找到正确结果的位置（排名从1开始）
                    correct_positions = [
                        pos + 1 for pos, doc_id in enumerate(retrieved_top5)
                        if doc_id in golden_docs
                    ]

                    # 初始化指标
                    hit = {}
                    recall = {}
                    precision = {}
                    mrr = 0.0

                    if golden_docs:  # 只有有 golden 的时候才计算
                        # 计算 MRR：第一个正确结果的排名的倒数
                        if correct_positions:
                            mrr = 1.0 / correct_positions[0]
                        else:
                            mrr = 0.0

                        # 计算 1-5 的 hit/recall/precision
                        for k in range(1, eval_top_k + 1):
                            # 前k个里的正确结果数量
                            correct_in_k = len([p for p in correct_positions if p <= k])

                            # hit@k：前k个里有没有至少一个正确的
                            hit[k] = 1 if correct_in_k > 0 else 0

                            # recall@k：前k个里的正确数 / 总正确数
                            recall[k] = correct_in_k / len(golden_docs)

                            # precision@k：前k个里的正确数 / k
                            precision[k] = correct_in_k / k

                            # 加入全局统计
                            all_hit[k].append(hit[k])
                            all_recall[k].append(recall[k])
                            all_precision[k].append(precision[k])

                        all_mrr.append(mrr)

                    print(f"\n📊 本次评测指标:")
                    if golden_docs:
                        for k in range(1, eval_top_k + 1):
                            print(
                                f"  hit@{k}: {hit[k]}, recall@{k}: {round(recall[k], 4)}, precision@{k}: {round(precision[k], 4)}")
                        print(f"  MRR: {round(mrr, 4)}")
                    print(f"  耗时: {cost_time}s")

                    # 📝 写入 CSV
                    row = [q_id]
                    # 写入检索结果
                    for res in results:
                        row.append(res['id'])
                        row.append(round(res['rerank_score'], 6))
                    # 补空结果列
                    while len(row) < 1 + 2 * config.RERANK_LIMIT:
                        row.append('')
                        row.append('')
                    # 写入 golden
                    row.append(','.join(map(str, golden_doc_ids)))
                    # 写入指标
                    for k in range(1, eval_top_k + 1):
                        row.append(hit.get(k, ''))
                        row.append(round(recall.get(k, ''), 6) if recall.get(k) else '')
                        row.append(round(precision.get(k, ''), 6) if precision.get(k) else '')
                    row.append(round(mrr, 6))
                    # 写入耗时
                    row.append(cost_time)

                    writer.writerow(row)
                    out_f.flush()  # 实时写入，防止崩溃丢数据

                    # 统计
                    success += 1

                    # 🧹 清空 GPU 缓存，防止内存溢出
                    if device == "cuda":
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"\n⚠️  处理测试用例 {i + 1} 出错: {str(e)}")
                    error_cases.append({"id": q_id, "error": str(e)})
                    # 写入空行
                    row = [q_id] + [''] * (len(header) - 1)
                    writer.writerow(row)
                    out_f.flush()
                    continue

        # 全部跑完，输出汇总报告
        print("\n\n" + "=" * 60)
        print("🏆 批量测试+评测完成！汇总报告:")
        print(f"  总测试用例: {total}")
        print(f"  成功检索+评测: {success} 个")
        print(f"  失败: {len(error_cases)} 个")
        if success > 0:
            avg_time = round(total_cost_time / success, 4)
            print(f"  平均检索耗时: {avg_time}s/个")

        print("\n📈 整个测试集的平均检索评估指标:")
        for k in range(1, eval_top_k + 1):
            if all_hit[k]:
                avg_hit = round(np.mean(all_hit[k]), 4)
                avg_recall = round(np.mean(all_recall[k]), 4)
                avg_precision = round(np.mean(all_precision[k]), 4)
                print(f"  hit@{k}: {avg_hit} | recall@{k}: {avg_recall} | precision@{k}: {avg_precision}")
        if all_mrr:
            avg_mrr = round(np.mean(all_mrr), 4)
            print(f"  Mean Reciprocal Rank (MRR): {avg_mrr}")

        print(f"\n📄 所有结果+评测指标已保存到: {output_file}")

        if error_cases:
            print("\n❌ 失败的测试用例:")
            for err in error_cases:
                print(f"  - ID: {err['id']} | 错误: {err['error']}")

    except FileNotFoundError:
        print(f"\n❌ 错误：找不到测试文件 {test_file}，请检查路径是否正确！")
        # fallback 到原来的单个测试
        print("\n🔄  fallback 到原来的单个测试...")
        # 【核心修改 3】模拟真实的 AgentState 字典结构
        test_state: AgentState = {
            "original_request": "我想了解一下报废船舶的注销登记程序是什么？",
            "rewrite_request": "船舶所有权注销 报废船舶 海事管理 登记程序",  # 工具会优先读取这个
            "clarification_question": None,
            "plan": [],
            "intermediate_steps": [],
            "verification_history": [],
            "final_response": "",
            "loop_count": 0
        }

        try:
            # 直接传入完整的 state
            results = retrieve_and_rerank_tool.invoke({"state": test_state})

            print("\n🏆 --- 最终精排输出结果 ---")
            print(json.dumps(results, indent=4, ensure_ascii=False))

        except Exception as e:
            print(f"\n❌ 运行出错: {e}")
    except Exception as e:
        print(f"\n❌ 运行出错: {str(e)}")