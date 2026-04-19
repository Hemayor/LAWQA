import os
import csv
import json
from typing import Dict, Any
from pydantic import BaseModel, Field
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入你项目里的 LLM 工厂
from app.llm_factory import LLMFactory

# ==========================================
# 1. 全局与评测文件路径配置
# ==========================================
GROUND_TRUTH_FILE = "../data/test_set/legal_test_set_final.jsonl"

CSV_FILES = {
    "Baseline_RAG": "../output/baserag/baseline_evaluation_results(50_5).csv",
    "Agentic_RAG": "../output/agenticrag/baseline_evaluation_results(50_5).csv",
    "Adaptive_RAG": "../output/agenticragwithcomplexity/baseline_evaluation_results(50_5).csv"
}

OUTPUT_REPORT_DIR = "../output/generation_eval/"

# ⚠️ 测试模式开关：设为 True 时每个文件只测 3 条，方便调试；正式跑全量请改为 False
DEBUG_MODE = False

# 建议并发数
MAX_WORKERS = 100


# ==========================================
# 2. 定义 Judge LLM 的结构化输出模型
# ==========================================
class LegalGenerationEvaluation(BaseModel):
    """用于大模型法官结构化输出的 Schema"""
    faithfulness_score: int = Field(
        description="事实忠诚度打分 (1-5): 最终回复是否严格忠实于检索到的线索，没有产生幻觉？")
    relevance_score: int = Field(
        description="意图识别与响应度打分 (1-5): 策略(澄清/回答)是否绝对符合题型要求？结论是否与标准答案高度一致？")
    execution_score: int = Field(description="执行合理性打分 (1-5): 实际执行的工具动作及召回的线索结果是否精准、高质量？")
    synthesis_score: int = Field(
        description="法律综合推理打分 (1-5): 是否将法条与用户具体案情深度结合，或提出精准的澄清清单？")
    reasoning: str = Field(description="详细的裁判理由，按 1,2,3,4 点解释上述打分原因。")


# ==========================================
# 3. 裁判提示词构造器 (解耦正交版 + 深度考察中间步骤版)
# ==========================================
def build_judge_prompt(request: str, plan_str: str, context_str: str, actual_output: str, ground_truth: str,
                       category: str) -> str:
    """
    构造法律领域的 LLM-as-a-Judge 提示词，实现指标正交化，并强迫裁判通过 intermediate_steps (Context) 评估执行效果。
    """
    is_ambiguous = (category == "clarification")
    question_type_str = "【模糊提问 (必须发起澄清追问)】" if is_ambiguous else "【清晰提问 (必须直接给出法律意见)】"
    expected_behavior = "系统必须列出需要补充的案件要素清单（即发起澄清），绝对不能强行给出确定的最终处罚/判决结果。" if is_ambiguous else "系统必须结合检索到的法条直接给出明确的法律结论，绝对不能反问或推诿。"

    return f"""你是一个严苛、客观且逻辑缜密的高级法律科技系统评测专家。
你的任务是根据案件背景、系统的实际执行步骤以及【标准参考答案】，对 AI 法律助手的最终输出质量进行四维打分。

【客户原始诉求】
{request}

【🔥 金标准题型判定 (最高指令)】
此题的官方分类为: {question_type_str}
裁判强制指引：根据测试集定义，本题正确的处理策略是 —— {expected_behavior}

【标准参考答案 (Ground Truth)】
{ground_truth}

【系统规划】
{plan_str}

【系统实际执行步骤与工具召回结果 (Context)】
{context_str}

【系统实际输出 (Actual Output)】
{actual_output}

---
【解耦评测打分标准 (1-5分)】
⚠️ 裁判须知：以下四个指标必须相互独立（正交）。不要因为某一项表现差（例如策略选错了）就恶意压低其他维度的分数！一码归一码！

1. 事实忠诚度 (Faithfulness) - 核心：只评估是否产生幻觉。
   - 评估重点：无论系统的回答策略对错，它输出的内容是否严格来源于 Context？
   - 1分: 满嘴跑火车，凭空捏造法条，或在未检索到信息时瞎编事实。
   - 5分: 输出的法条引用、法律逻辑 100% 忠实于 Context，严谨准确。

2. 意图识别与响应度 (Relevance & Intent) - 核心：严格依据【金标准题型判定】打分。
   - 评估重点：系统的策略（该问还是该答）是否选对？结论是否对齐标准答案？
   - 1分: 策略完全方向性错误（如该澄清却强行作答，或该作答却反问）；或结论与标准答案完全相反。
   - 5分: 策略极其精准！严格遵循了【金标准题型判定】的要求，且核心结论/澄清清单与【标准答案】高度一致。

3. 执行合理性 (Execution Efficiency) - 核心：严格根据【系统实际执行步骤与工具召回结果 (Context)】打分！不要看表面计划！
   - 评估重点：系统实际调用工具后，**带回来的线索质量到底好不好？** 检索方向是否准确？
   - 1分: 找出来的法条或新闻与用户提问毫无关联，全是一堆“垃圾信息”，说明工具执行效果极差。
   - 5分: 工具调用精准，且召回的结果质量极高，直接命中了解决该问题所需的关键法条或核心事实。

4. 法律综合推理 (Legal Synthesis) - 核心：只评估生成的文本专业深度。
   - 评估重点：输出的文本内容是否展现了法律逻辑？
   - 1分: 只是机械地复制粘贴法条；或者提出的澄清问题毫无专业素养（比如只会问“你在哪”）。
   - 5分: 展现了深度的“法条适用”逻辑推理；若是澄清清单，则精准指出用户提问缺失的关键法律要件。

⚠️ 【强制输出格式规则】
请务必严格返回如下结构的 JSON 对象！绝对不要擅自增加嵌套层级，必须保持键名完全一致：
{{
    "faithfulness_score": <1到5的整数>,
    "relevance_score": <1到5的整数>,
    "execution_score": <1到5的整数>,
    "synthesis_score": <1到5的整数>,
    "reasoning": "<详细的裁判理由。请务必证明你做到了独立打分不连坐>"
}}
"""


# ==========================================
# 4. 辅助函数：提取数据 (包含 category 提取与 Context 清洗)
# ==========================================
def load_ground_truth(filepath: str) -> Dict[str, Dict[str, str]]:
    """加载 JSONL 格式的 Ground Truth 数据集，同时提取 category"""
    gt_map = {}
    if not os.path.exists(filepath):
        print(f"❌ 找不到标准答案文件: {filepath}")
        return gt_map

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                q_id = str(data.get('id', '')).strip()
                gt_map[q_id] = {
                    "ground_truth": data.get('ground_truth', '无标准答案'),
                    "category": data.get('category', 'unknown')
                }
    return gt_map


def extract_clean_context(intermediate_steps_str: str) -> str:
    """将冗长的 JSON 记录提取为纯文本线索，让裁判直观看到实际执行结果"""
    if not intermediate_steps_str or intermediate_steps_str == "[]":
        return "【无可用执行步骤或召回线索】"

    try:
        steps = json.loads(intermediate_steps_str)
        context_text = ""
        for i, step in enumerate(steps):
            tool_name = step.get('tool_name', 'Unknown Tool')
            context_text += f"\n[执行动作 {i + 1}: 调用工具 `{tool_name}`]\n"

            output = step.get('tool_output', {})
            # 兼容 RAG 召回格式
            if isinstance(output, dict) and 'retrieval_results' in output:
                results = output['retrieval_results']
                if not results:
                    context_text += " -> 结果: 未检索到任何相关内容。\n"
                for idx, res in enumerate(results):
                    context_text += f" -> 召回法条 {idx + 1}: {res.get('title', '')} | 内容节选: {res.get('content', '')[:120]}...\n"
            # 兼容 Web 搜索格式
            elif isinstance(output, dict) and 'search_results' in output:
                results = output['search_results']
                if not results:
                    context_text += " -> 结果: 未搜索到相关新闻。\n"
                for idx, res in enumerate(results):
                    context_text += f" -> 召回新闻 {idx + 1}: {res.get('title', '')} | 摘要: {res.get('content', '')[:120]}...\n"
            elif isinstance(output, list):
                for idx, res in enumerate(output):
                    context_text += f" -> 召回内容 {idx + 1}: {res.get('title', '')} | {res.get('content', '')[:120]}...\n"

        return context_text if context_text else str(steps)[:500] + "..."
    except:
        return "【中间步骤日志解析失败，请参考原始字符串】: " + intermediate_steps_str[:300] + "..."


# ==========================================
# 5. 多线程单例处理函数
# ==========================================
def process_single_case(row: Dict[str, Any], structured_judge, ground_truth_map: Dict[str, dict]) -> Dict[str, Any]:
    """供线程池调用的单行处理函数"""
    q_id = str(row.get('id', '')).strip()

    request = row.get('original_request', '')
    plan = row.get('original_plan', '')

    # 提取经过清洗的 intermediate_steps，即实际召回内容
    context = extract_clean_context(row.get('intermediate_steps', ''))

    clarification = row.get('clarification_question', '')
    response = row.get('final_response', '')

    if clarification and str(clarification).strip():
        actual_output = f"【系统判定此题模糊，并发起了如下澄清追问】\n{clarification}"
    else:
        actual_output = f"【系统判定此题清晰，并给出了如下最终意见】\n{response}"

    gt_data = ground_truth_map.get(q_id, {"ground_truth": "无标准答案", "category": "unknown"})
    ground_truth = gt_data["ground_truth"]
    category = gt_data["category"]

    prompt = build_judge_prompt(request, plan, context, actual_output, ground_truth, category)

    try:
        eval_result = structured_judge.invoke(prompt)

        row["faithfulness_score"] = eval_result.faithfulness_score
        row["relevance_score"] = eval_result.relevance_score
        row["exec_score"] = eval_result.execution_score
        row["synth_score"] = eval_result.synthesis_score
        row["judge_reasoning"] = eval_result.reasoning
        row["_success"] = True
        row["_avg_score"] = np.mean([
            eval_result.faithfulness_score,
            eval_result.relevance_score,
            eval_result.execution_score,
            eval_result.synthesis_score
        ])
    except Exception as e:
        print(f"\n❌ ID: {q_id} 评测失败: {e}")
        row["_success"] = False

    return row


# ==========================================
# 6. 核心评估总控制流
# ==========================================
def run_evaluation():
    print(f"⚖️ 启动 LLM-as-a-Judge 生成质量评估引擎 (并发数: {MAX_WORKERS})...")
    os.makedirs(OUTPUT_REPORT_DIR, exist_ok=True)

    print("📖 加载标准答案数据集...")
    gt_map = load_ground_truth(GROUND_TRUTH_FILE)
    print(f"✅ 成功加载 {len(gt_map)} 条标准答案与题型标签。")

    judge_llm = LLMFactory.get_gpt()
    judge_llm.temperature = 0.0
    structured_judge = judge_llm.with_structured_output(LegalGenerationEvaluation, method="json_mode")

    global_stats = {}

    for system_name, csv_path in CSV_FILES.items():
        if not os.path.exists(csv_path):
            print(f"⚠️ 找不到文件跳过: {csv_path}")
            continue

        print(f"\n📂 正在评估系统: {system_name}")
        scores = {"faithfulness": [], "relevance": [], "execution": [], "synthesis": []}
        out_csv_path = os.path.join(OUTPUT_REPORT_DIR, f"{system_name}_judged.csv")

        with open(csv_path, 'r', encoding='utf-8-sig') as f_in:
            reader = list(csv.DictReader(f_in))
            if not reader:
                continue
            fieldnames = list(reader[0].keys()) + ["faithfulness_score", "relevance_score", "exec_score", "synth_score",
                                                   "judge_reasoning"]

        rows_to_process = reader[:3] if DEBUG_MODE else reader
        print(f"  -> 准备处理 {len(rows_to_process)} 条数据...")

        with open(out_csv_path, 'w', encoding='utf-8-sig', newline='') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {executor.submit(process_single_case, row, structured_judge, gt_map): row for row in
                           rows_to_process}

                for future in as_completed(futures):
                    processed_row = future.result()

                    is_success = processed_row.pop("_success", False)
                    avg_score = processed_row.pop("_avg_score", 0)

                    writer.writerow(processed_row)
                    f_out.flush()

                    q_id = processed_row.get("id", "未知")
                    if is_success:
                        print(f"  -> 审判 ID: {q_id} 完成 (综合得分: {avg_score:.1f}/5.0)")
                        scores["faithfulness"].append(processed_row["faithfulness_score"])
                        scores["relevance"].append(processed_row["relevance_score"])
                        scores["execution"].append(processed_row["exec_score"])
                        scores["synthesis"].append(processed_row["synth_score"])

        global_stats[system_name] = {k: np.mean(v) if v else 0 for k, v in scores.items()}

    # ==========================================
    # 7. 打印全局对比战报
    # ==========================================
    print("\n" + "=" * 60)
    print("🏆 生成质量评估战报 (LLM-as-a-Judge Average Scores)")
    print("=" * 60)
    for sys_name, stat in global_stats.items():
        print(f"\n🔹 系统: {sys_name}")
        print(f"   - 事实忠诚度 (Faithfulness): {stat['faithfulness']:.2f} / 5.0")
        print(f"   - 意图与响应度 (Relevance):  {stat['relevance']:.2f} / 5.0")
        print(f"   - 执行合理性 (Execution):    {stat['execution']:.2f} / 5.0")
        print(f"   - 法律推理深 (Synthesis):    {stat['synthesis']:.2f} / 5.0")

        avg_total = np.mean(list(stat.values()))
        print(f"   ⭐ 综合生成质量分:           {avg_total:.2f} / 5.0")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation()