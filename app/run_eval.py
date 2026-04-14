import json
import csv
import time
import os

# 从你的主运行文件中导入调用函数
# 请确保 run_law_agent 在 main_graph.py 中可用
from app.main_graph import run_law_agent
from langchain_community.callbacks.manager import get_openai_callback
# ==========================================
# 1. 配置文件路径
# ==========================================
INPUT_FILE = "../data/test_set/mini_testset2.jsonl"
OUTPUT_FILE = "../output/evaluation_results.csv"

# ==========================================
# 2. 定义 CSV 的表头 (严格对齐下方的 row_data)
# ==========================================
CSV_HEADERS = [
    "id",
    "original_request",
    "clarification_question",
    "rewrite_request",
    "plan",
    "intermediate_steps",
    "verification_history",
    "final_response",
    "loop_count",
    "planning_time_seconds",      # 👈 规划时间（累计）
    "rewrite_time_seconds",       # 👈 改写时间（累计）
    "retrieval_time_seconds",     # 👈 检索时间（累计）
    "generation_time_seconds",    # 👈 生成时间（累计）
    "web_search_time_seconds",    # 👈 网络搜索时间（累计）
    "other_tools_time_seconds",   # 👈 其他工具时间（累计）
    "end_to_end_time_seconds",    # 👈 端到端总时间
    "prompt_tokens",              # 👈 必须包含输入Token
    "completion_tokens",          # 👈 必须包含输出Token
    "total_tokens",               # 👈 必须包含总Token
    "total_cost_usd"              # 👈 必须包含预估成本
]


def run_batch_evaluation():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到测试集文件: {INPUT_FILE}")
        return

    results = []

    print(f"🚀 开始批量评测，读取文件: {INPUT_FILE}")
    print("-" * 60)

    # ==========================================
    # 3. 逐行读取并执行测试集
    # ==========================================
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if not line.strip():
                continue

            item = json.loads(line)
            q_id = item.get("id", f"unknown_{line_num}")
            question = item.get("question", "")

            print(f"▶️ 正在测试 ID: [{q_id}] ...")

            # --- 🕒 开始掐表 ---
            start_time = time.time()
            # 初始化 Token 统计变量
            p_tokens, c_tokens, t_tokens, cost = 0, 0, 0, 0.0
            try:
                # 🌟 【核心修改】：使用 callback 监听这个代码块里发生的所有 LLM 调用！
                with get_openai_callback() as cb:
                    final_state = run_law_agent(question)

                    # 当图谱运行完毕后，提取累加的 Token 数据
                    p_tokens = cb.prompt_tokens
                    c_tokens = cb.completion_tokens
                    t_tokens = cb.total_tokens
                    cost = cb.total_cost
            except Exception as e:
                print(f"  ❌ 运行出错: {e}")
                # 为了防止一个报错中断整个几小时的评测，我们做好异常兜底
                final_state = {"original_request": question, "final_response": f"ERROR: {str(e)}"}

            # --- 🕒 结束掐表 ---
            end_time = time.time()
            e2e_time = round(end_time - start_time, 2)
            # 打印的时候也可以顺便看看消耗
            print(f"  ✅ 测试完成，耗时: {e2e_time} 秒 | 消耗 Token: {t_tokens}")

            # ==========================================
            # 4. 数据清洗与序列化 (将复杂对象转为字符串)
            # ==========================================
            intermediate_steps = final_state.get("intermediate_steps", [])

            # 🌟 【核心修改】：从中间步骤中提取出所有的 tool_name，并拼上 FINISH
            # 这样就能还原出 Agent 真实执行的工具调用链路
            actual_plan = [step.get("tool_name") for step in intermediate_steps if "tool_name" in step]
            actual_plan.append("FINISH")

            # 将组装好的真实计划转为 JSON 字符串存入 CSV
            plan_str = json.dumps(actual_plan, ensure_ascii=False)

            # 其他字段保持原样转为字符串
            steps_str = json.dumps(intermediate_steps, ensure_ascii=False)
            history_str = json.dumps(final_state.get("verification_history", []), ensure_ascii=False)

            # ⏱️ 提取时间统计信息
            timing_stats = final_state.get("timing_stats", {})
            planning_time = round(timing_stats.get('planning', 0), 4)
            rewrite_time = round(timing_stats.get('rewrite', 0), 4)
            retrieval_time = round(timing_stats.get('retrieval', 0), 4)
            generation_time = round(timing_stats.get('generation', 0), 4)
            web_search_time = round(timing_stats.get('web_search', 0), 4)
            # 其他工具时间 = 总时间 - 已知工具时间
            other_tools_time = round(e2e_time - planning_time - rewrite_time - retrieval_time - generation_time - web_search_time, 4)
            if other_tools_time < 0:
                other_tools_time = 0  # 防止负数

            # ==========================================
            # 5. 组装一行数据
            # ==========================================
            row_data = {
                "id": q_id,
                "original_request": final_state.get("original_request", question),
                "clarification_question": final_state.get("clarification_question") or "",
                "rewrite_request": final_state.get("rewrite_request") or "",
                "plan": plan_str,
                "intermediate_steps": steps_str,
                "verification_history": history_str,
                "final_response": final_state.get("final_response", ""),
                "loop_count": final_state.get("loop_count", 0),
                "planning_time_seconds": planning_time,         # 🌟 规划时间
                "rewrite_time_seconds": rewrite_time,           # 🌟 改写时间
                "retrieval_time_seconds": retrieval_time,       # 🌟 检索时间
                "generation_time_seconds": generation_time,     # 🌟 生成时间
                "web_search_time_seconds": web_search_time,     # 🌟 网络搜索时间
                "other_tools_time_seconds": other_tools_time,   # 🌟 其他工具时间
                "end_to_end_time_seconds": e2e_time,            # 🌟 端到端总时间
                "prompt_tokens": p_tokens,  # 🌟 写入 CSV
                "completion_tokens": c_tokens,  # 🌟 写入 CSV
                "total_tokens": t_tokens,  # 🌟 写入 CSV
                "total_cost_usd": round(cost, 6)  # 🌟 写入 CSV (保留6位小数)
            }
            results.append(row_data)

    # ==========================================
    # 6. 写入 CSV 文件
    # ==========================================
    print("-" * 60)
    print(f"💾 所有测试运行完毕，正在保存结果至: {OUTPUT_FILE}")

    with open(OUTPUT_FILE, 'w', encoding='utf-8-sig', newline='') as f:
        # 注意：encoding='utf-8-sig' 可以防止用 Excel 打开 CSV 时中文乱码
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(results)

    print("🎉 评测结果已成功保存！")


if __name__ == "__main__":
    run_batch_evaluation()