import os
import json
import time
import csv
from typing import Dict, Any

from langchain_core.tools import tool

from state import AgentState
from app.llm_factory import LLMFactory


@tool
def query_rewrite_tool(state: AgentState) -> Dict[str, Any]:
    """
    专门用于优化和扩展用户查询的工具。
    采用 Query Expansion 策略：保留原问题核心语义，剔除无意义废话，并补充缩写、别名及专业法律术语。
    如果系统之前进行了联网搜索，它会结合搜索到的新闻事实进行融合扩展。
    如果系统之前经历了审计失败，它会根据反思意见改变扩展方向。
    调用指南：直接调用 `query_rewrite_tool()` 即可，不需要传参数。
    """
    # ⏱️ 开始计时
    start_time = time.time()

    query = state.get('original_request', '')
    print(f"\n[Tool 1] 正在改写查询: '{query}'")

    # 🌟 核心升级 1：去 intermediate_steps 里寻找侦察兵（Scout）留下的线索
    web_context = ""
    intermediate_steps = state.get("intermediate_steps", [])

    for step in intermediate_steps:
        if step.get("tool_name") == "scout_web_search_tool":
            tool_output = step.get("tool_output", {})

            # 兼容处理：有时输出是列表，有时是包着 'search_results' 的字典
            search_results = []
            if isinstance(tool_output, list):
                search_results = tool_output
            elif isinstance(tool_output, dict):
                search_results = tool_output.get("search_results", [])

            # 把新闻标题和摘要拼接成背景资料
            if search_results:
                web_context = "【背景新闻事实】\n"
                for res in search_results:
                    title = res.get("title", "无标题")
                    content = res.get("content", "")
                    web_context += f"- 标题：{title}\n  内容摘要：{content}\n"
            break  # 找到就退出循环

    reflection_context = ""
    history = state.get("verification_history", [])
    if history and len(history) > 0:
        latest_audit = history[-1]
        reflection_context = f"【⚠️ 上一次检索失败的反思意见】\n{latest_audit.get('reasoning', '')}\n请你务必改变检索词的扩展侧重点，不要重复之前的错误方向！\n"

    # 获取模型
    llm = LLMFactory.get_deepseek()

    # 🌟 核心升级 2：动态 Prompt 策略 (Query Expansion)
    if web_context:
        print("  -> 📡 侦测到联网搜索背景知识，启用【上下文融合查询扩展模式】...")
        prompt = f"""你是一个专业的法律检索查询扩展（Query Expansion）专家。
你的【唯一任务】是：结合用户的原始提问和新闻事实进行查询扩充，以提升底层向量数据库的检索命中率。绝对不能仅仅提取几个关键词而丢失原始事实语义！

{web_context}
{reflection_context}

【用户的原始提问】:
{query}

【核心扩展规则】
1. 绝对不要回答问题，不要给出法律建议！
2. 剥离废话：删掉无意义的口语连词、情绪发泄和询问短语（如“怎么办”、“这合法吗”、“太恶心了”、“求助”）。
3. 保留事实语义：必须保留用户描述的核心事实短句（例如“试用期被无故辞退没给钱”）。
4. 实体与别名映射：根据新闻事实，补充涉及的实体全称、别名或特定项目名（如将“李佳琦带货的那个眉笔”扩充为“李佳琦 直播带货 花西子眉笔”）。
5. 术语升维：将口语场景平移映射为专业法律术语，追加在事实描述之后（如“退一赔三 欺诈 消费者权益保护法”）。
6. 输出格式：输出扩展后的文本，各维度词汇之间用空格隔开，不要加标点连句。总长度控制在 50 个汉字左右。

请严格按照上述规则输出扩展后的检索词："""

    else:
        if reflection_context:
            print("  -> 🔄 侦测到审计失败历史，启用【反思扩展模式】...")
        else:
            print("  -> 📝 未发现联网背景，启用【基础查询扩展模式】...")

        prompt = f"""你是一个专业的法律检索查询扩展（Query Expansion）专家。
你的【唯一任务】是将用户的口语化提问进行“清洗与扩充”，以提升底层向量数据库的检索命中率。绝对不能仅仅提取几个关键词而丢失原始事实语义！

{reflection_context}

【核心扩展规则】
1. 绝对不要回答问题，不要给出法律建议！
2. 剥离废话：删掉无意义的口语连词、情绪发泄和询问短语（如“怎么办”、“谁来管管”、“这能告他吗”）。
3. 保留事实语义：必须保留用户描述的核心场景和事实主干，不能只剩下干瘪的词汇。
4. 同义词与缩写展开：将常见的简称、别名展开为全称（如“离职竞业” -> “解除劳动合同 竞业限制”）。
5. 术语升维：将口语平移映射为相关的法律案由和专业术语，追加在事实之后。
6. 输出格式：输出扩展后的文本，各维度词汇之间用空格隔开，不要加标点连句。总长度控制在 50 个汉字左右。

【优秀扩充案例学习】
输入：别人欠钱不还怎么办，转账记录我都有。
输出：别人欠钱不还 有转账记录 民间借贷纠纷 逾期还款 支付令 强制执行 债务违约

输入：老板突然把我辞退了，没给钱，没提前通知。
输出：老板突然把员工辞退没给钱没提前通知 违法解除劳动合同 劳动争议 经济补偿金 赔偿金 拖欠薪资 代通知金

现在的用户输入：{query}
请严格按照上述规则输出扩展后的检索词："""

    # 调用大模型执行改写
    optimized_query = llm.invoke(prompt).content.strip()
    print(f"  -> ✅ 最终扩充结果: '{optimized_query}'")

    # ⏱️ 计时结束，累计到 timing_stats
    end_time = time.time()
    elapsed_time = round(end_time - start_time, 4)

    timing_stats = state.get('timing_stats', {}).copy()
    timing_stats['rewrite'] = timing_stats.get('rewrite', 0) + elapsed_time
    print(f"  -> 耗时: {elapsed_time}s | 累计改写时间: {timing_stats['rewrite']}s")

    return {
        "rewrite_request": optimized_query,
        "timing_stats": timing_stats
    }


# ==========================================
# 本地批量测试入口 (Main 函数)
# ==========================================
if __name__ == "__main__":
    # 为了模拟图谱的真实执行链路，我们需要引入 Scout 工具
    # 注意路径是否与你的项目结构一致
    from app.tool_scout_search import scout_web_search_tool

    print("🚀 开始批量测试【联网搜索 + 智能查询扩展】组合工具链路...")

    INPUT_FILE = "../data/test_set/web.jsonl"
    OUTPUT_FILE = "../output/暂时没用/rewrite_test_result.csv"

    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    headers = [
        "id", "original_question",
        "has_web_search", "web_context_snippet",
        "rewritten_query",
        "total_cost_time_sec", "error"
    ]

    success_count = 0
    error_count = 0

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
                open(OUTPUT_FILE, 'w', encoding='utf-8-sig', newline='') as f_out:

            writer = csv.DictWriter(f_out, fieldnames=headers)
            writer.writeheader()

            lines = f_in.readlines()
            total_cases = len([line for line in lines if line.strip()])
            print(f"📂 成功读取测试文件，共找到 {total_cases} 个测试用例。\n" + "=" * 50)

            for i, line in enumerate(lines):
                if not line.strip():
                    continue

                item = json.loads(line)
                q_id = item.get("id", "unknown")
                question = item.get("question", "")

                print(f"\n▶️ [进度 {success_count + error_count + 1}/{total_cases}] 测试 ID: {q_id}")

                # 1. 构造初始的 AgentState
                test_state = {
                    "original_request": question,
                    "clarification_question": None,
                    "rewrite_request": None,
                    "plan": [],
                    "intermediate_steps": [],
                    "verification_history": [],
                    "final_response": "",
                    "loop_count": 0,
                    "timing_stats": {}
                }

                row_data = {
                    "id": q_id,
                    "original_question": question,
                    "has_web_search": "False",
                    "web_context_snippet": "",
                    "rewritten_query": "",
                    "total_cost_time_sec": 0,
                    "error": ""
                }

                try:
                    total_time = 0

                    # --- 步骤一：模拟执行联网搜索 (Scout) ---
                    scout_result = scout_web_search_tool.invoke({"state": test_state})

                    # 假装 Router 把搜索结果装进了中间步骤列表
                    test_state["intermediate_steps"].append({
                        "tool_name": "scout_web_search_tool",
                        "tool_output": scout_result
                    })
                    # 更新时间状态
                    test_state["timing_stats"] = scout_result.get("timing_stats", {})

                    # 提取一部分新闻摘要用于写入 CSV (方便你事后查阅)
                    web_res = scout_result.get("search_results", [])
                    if web_res and isinstance(web_res, list) and 'content' in web_res[0]:
                        row_data["has_web_search"] = "True"
                        row_data["web_context_snippet"] = web_res[0]['content'][:50] + "..."  # 截取前50个字

                    # --- 步骤二：执行查询扩展 (Rewrite) ---
                    # 此时传入的 state 已经包含了步骤一的搜索结果
                    rewrite_result = query_rewrite_tool.invoke({"state": test_state})

                    # 填充 CSV 数据
                    row_data["rewritten_query"] = rewrite_result.get("rewrite_request", "")

                    # 计算两个工具叠加的总耗时
                    final_stats = rewrite_result.get("timing_stats", {})
                    total_time = final_stats.get("web_search", 0) + final_stats.get("rewrite", 0)
                    row_data["total_cost_time_sec"] = round(total_time, 2)

                    success_count += 1

                except Exception as e:
                    print(f"  ❌ ID {q_id} 链路执行出错: {e}")
                    row_data["error"] = str(e)
                    error_count += 1

                # 写入 CSV 并落盘
                writer.writerow(row_data)
                f_out.flush()

        print("\n" + "=" * 50)
        print(f"🏆 组合链路测试完成！")
        print(f"✅ 成功: {success_count} 条")
        print(f"❌ 失败: {error_count} 条")
        print(f"💾 结果已保存至: {OUTPUT_FILE}")

    except Exception as e:
        print(f"❌ 发生全局错误: {e}")