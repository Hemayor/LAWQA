import json
import csv
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_community.callbacks.manager import get_openai_callback

# 从你的主运行文件中导入调用函数
from app.main_graph import run_law_agent

# ==========================================
# 1. 配置文件与多线程参数
# ==========================================
INPUT_FILE = "../data/test_set/legal_test_set_final.jsonl"
OUTPUT_FILE = "../output/evaluation_results.csv"

# 🌟 【新增】：最大并发线程数
# 建议：如果是纯 API 调用，可以开到 5-10；
# 但因为你的本地检索用了 CUDA 模型，建议先设置为 2-3，防止显存溢出 (OOM) 或 API 触发限流
MAX_WORKERS = 5

# ==========================================
# 2. 定义 CSV 的表头
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
    "planning_time_seconds",
    "rewrite_time_seconds",
    "retrieval_time_seconds",
    "generation_time_seconds",
    "web_search_time_seconds",
    "other_tools_time_seconds",
    "end_to_end_time_seconds",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
]


# ==========================================
# 3. 核心处理逻辑 (被多线程调用的工作函数)
# ==========================================
def process_single_item(item, line_num):
    """
    处理单个测试用例的函数，设计为线程安全。
    """
    q_id = item.get("id", f"unknown_{line_num}")
    question = item.get("question", "")

    print(f"▶️ [线程启动] 正在测试 ID: [{q_id}] ...")

    start_time = time.time()
    p_tokens, c_tokens, t_tokens = 0, 0, 0

    try:
        # get_openai_callback 在当前线程内是安全的
        with get_openai_callback() as cb:
            final_state = run_law_agent(question)
            p_tokens = cb.prompt_tokens
            c_tokens = cb.completion_tokens
            t_tokens = cb.total_tokens
    except Exception as e:
        print(f"  ❌ ID [{q_id}] 运行出错: {e}")
        final_state = {"original_request": question, "final_response": f"ERROR: {str(e)}"}

    end_time = time.time()
    e2e_time = round(end_time - start_time, 2)
    print(f"  ✅ [线程完成] ID [{q_id}] 耗时: {e2e_time} 秒 | 消耗 Token: {t_tokens}")

    # 数据清洗与序列化
    intermediate_steps = final_state.get("intermediate_steps", [])
    actual_plan = [step.get("tool_name") for step in intermediate_steps if "tool_name" in step]
    actual_plan.append("FINISH")

    plan_str = json.dumps(actual_plan, ensure_ascii=False)
    steps_str = json.dumps(intermediate_steps, ensure_ascii=False)
    history_str = json.dumps(final_state.get("verification_history", []), ensure_ascii=False)

    timing_stats = final_state.get("timing_stats", {})
    planning_time = round(timing_stats.get('planning', 0), 4)
    rewrite_time = round(timing_stats.get('rewrite', 0), 4)
    retrieval_time = round(timing_stats.get('retrieval', 0), 4)
    generation_time = round(timing_stats.get('generation', 0), 4)
    web_search_time = round(timing_stats.get('web_search', 0), 4)

    other_tools_time = round(
        e2e_time - planning_time - rewrite_time - retrieval_time - generation_time - web_search_time, 4)
    if other_tools_time < 0:
        other_tools_time = 0

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
        "planning_time_seconds": planning_time,
        "rewrite_time_seconds": rewrite_time,
        "retrieval_time_seconds": retrieval_time,
        "generation_time_seconds": generation_time,
        "web_search_time_seconds": web_search_time,
        "other_tools_time_seconds": other_tools_time,
        "end_to_end_time_seconds": e2e_time,
        "prompt_tokens": p_tokens,
        "completion_tokens": c_tokens,
        "total_tokens": t_tokens,
    }

    return row_data


# ==========================================
# 4. 批量运行与线程池调度
# ==========================================
def run_batch_evaluation():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到测试集文件: {INPUT_FILE}")
        return

    print(f"🚀 开始多线程批量评测，读取文件: {INPUT_FILE}")
    print(f"⚙️  当前配置线程数 (MAX_WORKERS): {MAX_WORKERS}")
    print("-" * 60)

    # 1. 把文件数据先全部读进内存
    items_to_process = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if not line.strip():
                continue
            items_to_process.append((json.loads(line), line_num))

    results = []
    global_start_time = time.time()

    # 2. 🌟 【核心修改】：启动线程池
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 将任务提交给线程池
        futures = {executor.submit(process_single_item, item, line_num): item for item, line_num in items_to_process}

        # as_completed 会在某个线程一跑完时立刻返回其结果
        for future in as_completed(futures):
            try:
                row_data = future.result()
                results.append(row_data)
            except Exception as e:
                print(f"❌ 获取线程结果时发生致命错误: {e}")

    global_end_time = time.time()
    print("-" * 60)
    print(f"🏁 所有任务并发执行完毕！总计耗时: {round(global_end_time - global_start_time, 2)} 秒")

    # 3. 排序 (由于多线程完成顺序是随机的，这里根据 ID 重新排个序，让 CSV 更好看)
    results.sort(key=lambda x: str(x.get("id", "")))

    # 4. 写入 CSV 文件
    print(f"💾 正在保存结果至: {OUTPUT_FILE}")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(results)

    print("🎉 评测结果已成功保存！")


if __name__ == "__main__":
    run_batch_evaluation()