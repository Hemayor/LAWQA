import os
import json
import csv
from collections import defaultdict
import numpy as np
import config

# ==========================================
# 1. 配置文件路径
# ==========================================
RECALL_LIMIT = config.RECALL_LIMIT
RERANK_LIMIT = config.RERANK_LIMIT

GROUND_TRUTH_FILE = "../data/test_set/legal_test_set_final.jsonl"
# ⚠️ 请确保这里是你最新的包含 Agent 执行记录的 CSV 文件
RAG_RESULT_FILE = "../output/暂时没用/baseline_evaluation_results.csv"

OUTPUT_REPORT_TXT = f"../output/agenticrag_generation_with_complexity_classifier/recall{RECALL_LIMIT}_rerank{RERANK_LIMIT}_generation_evaluation_report.txt"
OUTPUT_REPORT_CSV = f"../output/agenticrag_generation_with_complexity_classifier/recall{RECALL_LIMIT}_rerank{RERANK_LIMIT}_generation_evaluation_detailed.csv"
# OUTPUT_REPORT_TXT = f"../output/baserag/新的完整报告/recall{RECALL_LIMIT}_rerank{RERANK_LIMIT}_generation_evaluation_report.txt"
# OUTPUT_REPORT_CSV = f"../output/baserag/新的完整报告/recall{RECALL_LIMIT}_rerank{RERANK_LIMIT}_generation_evaluation_detailed.csv"
# 评测的最大 K 值
MAX_K = 10


def calculate_metrics(retrieved_ids, golden_ids):
    """计算单条数据的 Hit@K, Recall@K, Precision@K 和 MRR"""
    golden_set = set(golden_ids)
    if not golden_set:
        return None

    metrics = {}
    mrr = 0.0
    for rank, doc_id in enumerate(retrieved_ids):
        if doc_id in golden_set:
            mrr = 1.0 / (rank + 1)
            break
    metrics['MRR'] = mrr

    for k in range(1, MAX_K + 1):
        top_k_retrieved = retrieved_ids[:k]
        hits_at_k = len([doc_id for doc_id in top_k_retrieved if doc_id in golden_set])
        metrics[f'Hit@{k}'] = 1 if hits_at_k > 0 else 0
        metrics[f'Recall@{k}'] = hits_at_k / len(golden_set)
        metrics[f'Precision@{k}'] = hits_at_k / k

    return metrics


def safe_mean(lst):
    """安全求平均"""
    return np.mean(lst) if lst else 0.0


def main():
    print("🚀 开始进行 Agentic RAG 评测指标计算...")

    # 1. 加载 Ground Truth
    if not os.path.exists(GROUND_TRUTH_FILE):
        print(f"❌ 找不到标准答案文件: {GROUND_TRUTH_FILE}")
        return

    ground_truth_map = {}
    with open(GROUND_TRUTH_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                q_id = str(data['id']).strip()
                ground_truth_map[q_id] = {
                    'category': data.get('category', 'unknown'),
                    'golden_ids': [str(gid).strip() for gid in data.get('golden_doc_ids', [])]
                }
    print(f"✅ 成功加载 {len(ground_truth_map)} 条标准答案。")

    # 2. 准备数据收集器
    stats = defaultdict(lambda: defaultdict(list))
    sample_counts = defaultdict(int)
    detailed_csv_data = []

    time_token_keys = [
        'planning_time_seconds', 'rewrite_time_seconds', 'retrieval_time_seconds',
        'generation_time_seconds', 'web_search_time_seconds', 'reflection_time_seconds',
        'other_tools_time_seconds', 'end_to_end_time_seconds',
        'prompt_tokens', 'completion_tokens', 'total_tokens'
    ]

    # 3. 解析 RAG 结果并计算
    if not os.path.exists(RAG_RESULT_FILE):
        print(f"❌ 找不到 RAG 结果文件: {RAG_RESULT_FILE}")
        return

    processed_count = 0
    with open(RAG_RESULT_FILE, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            q_id = str(row['id']).strip()
            if q_id not in ground_truth_map:
                continue

            gt_info = ground_truth_map[q_id]
            category = gt_info['category']
            golden_ids = gt_info['golden_ids']

            retrieved_ids = []
            try:
                steps = json.loads(row.get('intermediate_steps', '[]'))
                seen_ids = set()

                # 🌟 修复核心：强大的自适应结构解析
                for step in reversed(steps):
                    if step.get('tool_name') == 'retrieve_and_rerank_tool':
                        tool_output = step.get('tool_output')

                        results = []
                        # 兼容 Agent 输出直接为 List 的情况
                        if isinstance(tool_output, list):
                            results = tool_output
                        # 兼容 Baseline 输出为 Dict，法条在 retrieval_results 中的情况
                        elif isinstance(tool_output, dict):
                            results = tool_output.get('retrieval_results', [])

                        if results:
                            for res in results:
                                if 'id' in res:
                                    doc_id = str(res['id']).strip()
                                    if doc_id not in seen_ids:
                                        retrieved_ids.append(doc_id)
                                        seen_ids.add(doc_id)
                            break  # 只取最后一次成功检索的结果

            except Exception as e:
                print(f"⚠️ 解析 ID {q_id} 的 intermediate_steps 出错: {e}")

            # 计算检索指标
            retrieval_metrics = calculate_metrics(retrieved_ids, golden_ids)

            # 构造详细级别的单行数据
            row_detail = {
                "id": q_id,
                "category": category,
                "original_request": row.get('original_request', '')
            }

            if retrieval_metrics:
                row_detail['MRR'] = round(retrieval_metrics['MRR'], 4)
                for k in range(1, MAX_K + 1):
                    row_detail[f'Hit@{k}'] = retrieval_metrics[f'Hit@{k}']
                    row_detail[f'Recall@{k}'] = round(retrieval_metrics[f'Recall@{k}'], 4)
                    row_detail[f'Precision@{k}'] = round(retrieval_metrics[f'Precision@{k}'], 4)
            else:
                row_detail['MRR'] = 0.0
                for k in range(1, MAX_K + 1):
                    row_detail[f'Hit@{k}'] = 0
                    row_detail[f'Recall@{k}'] = 0.0
                    row_detail[f'Precision@{k}'] = 0.0

            for key in time_token_keys:
                try:
                    row_detail[key] = round(float(row.get(key, 0)), 4)
                except ValueError:
                    row_detail[key] = 0.0

            detailed_csv_data.append(row_detail)

            # 更新 TXT 报告所需的全局统计数据
            categories_to_update = [category, 'ALL']
            for cat in categories_to_update:
                sample_counts[cat] += 1
                if retrieval_metrics:
                    for metric_name, value in retrieval_metrics.items():
                        stats[cat][metric_name].append(value)
                for key in time_token_keys:
                    stats[cat][key].append(row_detail[key])

            processed_count += 1

    print(f"✅ 成功处理并计算了 {processed_count} 条测试用例。")

    # 4. 生成 TXT 分类汇总报告
    os.makedirs(os.path.dirname(OUTPUT_REPORT_TXT), exist_ok=True)
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("📊 Agentic RAG 系统评测效果报告")
    report_lines.append("=" * 60 + "\n")

    categories = sorted([c for c in sample_counts.keys() if c != 'ALL']) + ['ALL']

    for cat in categories:
        count = sample_counts[cat]
        if count == 0: continue

        is_all = (cat == 'ALL')
        title = "🌟 【全局总体平均指标 (ALL)】" if is_all else f"📁 【分类: {cat}】"
        report_lines.append(title)
        report_lines.append(f"  测试用例数: {count}")

        report_lines.append("  [检索质量指标]")
        mrr_val = safe_mean(stats[cat]['MRR'])
        report_lines.append(f"    - MRR: {mrr_val:.4f}")

        hit_str = "    - Hit@K:       "
        recall_str = "    - Recall@K:    "
        precision_str = "    - Precision@K: "

        for k in range(1, MAX_K + 1):
            hit_val = safe_mean(stats[cat].get(f'Hit@{k}', []))
            recall_val = safe_mean(stats[cat].get(f'Recall@{k}', []))
            precision_val = safe_mean(stats[cat].get(f'Precision@{k}', []))

            hit_str += f" | @{k}:{hit_val:.2f}"
            recall_str += f" | @{k}:{recall_val:.2f}"
            precision_str += f" | @{k}:{precision_val:.2f}"

        report_lines.extend([hit_str, recall_str, precision_str])

        report_lines.append("\n  [性能与消耗指标 (平均值)]")
        report_lines.append(f"    - 端到端总耗时: {safe_mean(stats[cat]['end_to_end_time_seconds']):.2f} 秒")
        report_lines.append(f"    - 规划耗时: {safe_mean(stats[cat]['planning_time_seconds']):.2f} 秒")
        report_lines.append(f"    - 改写耗时: {safe_mean(stats[cat]['rewrite_time_seconds']):.2f} 秒")
        report_lines.append(f"    - 检索工具耗时: {safe_mean(stats[cat]['retrieval_time_seconds']):.2f} 秒")
        report_lines.append(f"    - 联网搜索耗时: {safe_mean(stats[cat]['web_search_time_seconds']):.2f} 秒")
        report_lines.append(f"    - 反思耗时: {safe_mean(stats[cat]['reflection_time_seconds']):.2f} 秒")
        report_lines.append(f"    - LLM 生成耗时: {safe_mean(stats[cat]['generation_time_seconds']):.2f} 秒")
        report_lines.append(
            f"    - 总消耗 Token: {safe_mean(stats[cat]['total_tokens']):.0f} (Prompt: {safe_mean(stats[cat]['prompt_tokens']):.0f}, Completion: {safe_mean(stats[cat]['completion_tokens']):.0f})")
        report_lines.append("-" * 60 + "\n")

    with open(OUTPUT_REPORT_TXT, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    print(f"📄 分类汇总文本报告已生成: {OUTPUT_REPORT_TXT}")

    # 5. 生成详细级别 (Row-level) CSV 报告
    if detailed_csv_data:
        headers = list(detailed_csv_data[0].keys())
        with open(OUTPUT_REPORT_CSV, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(detailed_csv_data)
        print(f"📈 每题详细测评 CSV 已生成: {OUTPUT_REPORT_CSV}")


if __name__ == "__main__":
    main()