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

RAG_RESULT_FILE = "../output/暂时没用/baseline_evaluation_results.csv"

# OUTPUT_REPORT_TXT = f"../output/baserag_generation/recall{RECALL_LIMIT}_rerank{RERANK_LIMIT}_generation_evaluation_report.txt"
#
# OUTPUT_REPORT_CSV = f"../output/baserag_generation/recall{RECALL_LIMIT}_rerank{RERANK_LIMIT}_generation_evaluation_summary.csv"
OUTPUT_REPORT_TXT = f"../output/agenticrag_generation/recall{RECALL_LIMIT}_rerank{RERANK_LIMIT}_generation_evaluation_report.txt"

OUTPUT_REPORT_CSV = f"../output/agenticrag_generation/recall{RECALL_LIMIT}_rerank{RERANK_LIMIT}_generation_evaluation_summary.csv"
# 评测的最大 K 值
MAX_K = 10


def calculate_metrics(retrieved_ids, golden_ids):
    """
    计算单条数据的 Hit@K, Recall@K, Precision@K 和 MRR
    """
    golden_set = set(golden_ids)

    # 如果没有标准答案(如 clarification 题)，返回 None，但不阻断后续的时间统计
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
    """安全求平均，防止遇到空列表报错"""
    return np.mean(lst) if lst else 0.0


def main():
    print("🚀 开始进行 RAG 评测指标计算...")

    # 1. 加载 Ground Truth
    if not os.path.exists(GROUND_TRUTH_FILE):
        print(f"❌ 找不到标准答案文件: {GROUND_TRUTH_FILE}")
        return

    ground_truth_map = {}
    with open(GROUND_TRUTH_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                q_id = str(data['id'])
                ground_truth_map[q_id] = {
                    'category': data.get('category', 'unknown'),
                    'golden_ids': [str(gid) for gid in data.get('golden_doc_ids', [])]
                }
    print(f"✅ 成功加载 {len(ground_truth_map)} 条标准答案。")

    # 2. 准备数据收集器
    stats = defaultdict(lambda: defaultdict(list))
    # 专门用一个字典存各分类的样本总数
    sample_counts = defaultdict(int)

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
            q_id = str(row['id'])
            if q_id not in ground_truth_map:
                continue

            gt_info = ground_truth_map[q_id]
            category = gt_info['category']
            golden_ids = gt_info['golden_ids']

            retrieved_ids = []
            try:
                steps = json.loads(row.get('intermediate_steps', '[]'))
                for step in steps:
                    if step.get('tool_name') == 'retrieve_and_rerank_tool':
                        results = step.get('tool_output', {}).get('retrieval_results', [])
                        retrieved_ids = [str(res['id']) for res in results]
                        break
            except Exception as e:
                pass

            retrieval_metrics = calculate_metrics(retrieved_ids, golden_ids)

            # 🌟 核心修复：无论有没有检索指标，都要给分类计数，并记录时间和 Token
            categories_to_update = [category, 'ALL']
            for cat in categories_to_update:
                sample_counts[cat] += 1

                if retrieval_metrics:
                    for metric_name, value in retrieval_metrics.items():
                        stats[cat][metric_name].append(value)

                for key in time_token_keys:
                    try:
                        val = float(row.get(key, 0))
                        stats[cat][key].append(val)
                    except ValueError:
                        pass

            processed_count += 1

    print(f"✅ 成功处理并计算了 {processed_count} 条测试用例。")

    # 4. 生成报告
    os.makedirs(os.path.dirname(OUTPUT_REPORT_TXT), exist_ok=True)
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("📊 RAG 系统基线评测效果报告 (Baseline RAG Evaluation)")
    report_lines.append("=" * 60 + "\n")

    summary_csv_data = []
    categories = sorted([c for c in sample_counts.keys() if c != 'ALL']) + ['ALL']

    for cat in categories:
        count = sample_counts[cat]
        if count == 0: continue

        is_all = (cat == 'ALL')
        title = "🌟 【全局总体平均指标 (ALL)】" if is_all else f"📁 【分类: {cat}】"
        report_lines.append(title)
        report_lines.append(f"  测试用例数: {count}")

        cat_summary = {"Category": cat, "Sample_Count": count}

        # 格式化检索指标 (对于 clarification，由于没有分数，全部输出为 0)
        report_lines.append("  [检索质量指标]")
        mrr_val = safe_mean(stats[cat]['MRR'])
        report_lines.append(f"    - MRR: {mrr_val:.4f}")
        cat_summary["MRR"] = round(mrr_val, 4)

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

            cat_summary[f"Hit@{k}"] = round(hit_val, 4)
            cat_summary[f"Recall@{k}"] = round(recall_val, 4)
            cat_summary[f"Precision@{k}"] = round(precision_val, 4)

        report_lines.extend([hit_str, recall_str, precision_str])

        # 格式化性能指标
        report_lines.append("\n  [性能与消耗指标 (平均值)]")
        report_lines.append(f"    - 端到端总耗时: {safe_mean(stats[cat]['end_to_end_time_seconds']):.2f} 秒")
        report_lines.append(f"    - 检索工具耗时: {safe_mean(stats[cat]['retrieval_time_seconds']):.2f} 秒")
        report_lines.append(f"    - LLM 生成耗时: {safe_mean(stats[cat]['generation_time_seconds']):.2f} 秒")
        report_lines.append(
            f"    - 总消耗 Token: {safe_mean(stats[cat]['total_tokens']):.0f} (Prompt: {safe_mean(stats[cat]['prompt_tokens']):.0f}, Completion: {safe_mean(stats[cat]['completion_tokens']):.0f})")
        report_lines.append("-" * 60 + "\n")

        for key in time_token_keys:
            cat_summary[f"Avg_{key}"] = round(safe_mean(stats[cat][key]), 4)

        summary_csv_data.append(cat_summary)

    # 5. 写入 TXT 报告
    with open(OUTPUT_REPORT_TXT, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    print(f"📄 可读文本报告已生成: {OUTPUT_REPORT_TXT}")

    # 6. 写入 CSV 汇总表
    if summary_csv_data:
        headers = list(summary_csv_data[0].keys())
        with open(OUTPUT_REPORT_CSV, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(summary_csv_data)
        print(f"📈 结构化汇总表已生成: {OUTPUT_REPORT_CSV}")


if __name__ == "__main__":
    main()