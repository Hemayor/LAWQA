import json
from typing import Dict, Any

# 引入状态定义和工具字典
from state import AgentState
from agent_graph import tool_map


def task_executor_node(state: AgentState) -> Dict[str, Any]:
    """
    工具执行节点：从计划中弹出第一个任务，执行它，并记录结果。
    """
    print("\n⚙️ [-- 工具执行节点 (Tool Executor) 启动 --]")

    current_plan = state.get('plan', [])

    # 防御性编程：如果计划为空，或者遇到了终点标识，直接返回
    if not current_plan or current_plan[0] == "FINISH":
        print("  -> 计划已空或遇到 'FINISH'，跳过执行。")
        return {"plan": current_plan}

    # 拿出计划中的下一步
    next_step = current_plan[0]

    # --- 安全的解析逻辑 ---
    # 我们不需要像原论文那样去抠括号里的参数，只需要提取工具名
    # 比如把 "query_rewrite_tool()" 变成 "query_rewrite_tool"
    tool_name = next_step.split('(')[0].strip()

    if tool_name not in tool_map:
        print(f"  -> ❌ 错误：找不到名为 '{tool_name}' 的工具。跳过该步骤。")
        return {
            "plan": current_plan[1:],  # 依然要把它从计划中踢出，防止死循环
            "intermediate_steps": state.get('intermediate_steps', [])
        }

    print(f"  -> 准备唤醒工具: {tool_name}")

    try:
        # 拿到具体的工具函数
        tool_to_call = tool_map[tool_name]

        # 【架构核心魔法】：直接把全局的 state 丢给工具！不传任何其他参数！
        result = tool_to_call.invoke({"state": state})

        print(f"  -> ✅ 工具 {tool_name} 成功返回数据。")

        # 【特殊处理】：某些工具返回字典格式，需要提取实际结果
        tool_output = result
        if tool_name == 'retrieve_and_rerank_tool' and isinstance(result, dict) and 'retrieval_results' in result:
            tool_output = result['retrieval_results']
        elif tool_name == 'scout_web_search_tool' and isinstance(result, dict) and 'search_results' in result:
            tool_output = result['search_results']

        # 记录执行足迹 (供最后的大模型参考)
        new_intermediate_step = {
            'tool_name': tool_name,
            'tool_output': tool_output
        }

        # 构造给 LangGraph 图谱的状态更新包
        state_update = {
            "intermediate_steps": state.get('intermediate_steps', []) + [new_intermediate_step],
            "plan": current_plan[1:]  # 任务完成，从计划列表中剔除这一步
        }

        # 【解决你之前提的漏洞】：动态更新 State 中的共享字段
        # 如果改写工具返回了 {"rewrite_request": "劳动争议"}，我们要把它顺手更新到外层的 State 中
        if isinstance(result, dict):
            # 将工具返回的字典数据，合并到更新包中
            for key, value in result.items():
                # 特殊处理 timing_stats：需要合并而不是覆盖
                if key == 'timing_stats' and isinstance(value, dict):
                    current_stats = state_update.get('timing_stats', state.get('timing_stats', {})).copy()
                    current_stats.update(value)
                    state_update['timing_stats'] = current_stats
                    print(f"  -> 🕐 时间统计已更新: {current_stats}")
                else:
                    state_update[key] = value
                    print(f"  -> 🔄 触发状态注入: 成功将 '{key}' 更新到全局 State 中。")

        return state_update

    except Exception as e:
        print(f"  -> ❌ 工具 {tool_name} 执行过程中发生错误: {e}")
        # 出错也不能卡死，跳过这一步继续
        return {
            "plan": current_plan[1:],
            "intermediate_steps": state.get('intermediate_steps', [])
        }


# ==========================================
# 本地独立测试入口
# ==========================================
if __name__ == "__main__":
    print("🚀 开始独立测试 Tool Executor 节点...")

    # 构造一个假状态，模拟 Planner 刚刚制定好的计划
    test_state = {
        "original_request": "某县政府正在制定年度国民经济和社会发展计划，准备将城市供水事业发展纳入其中，并考虑与本地绿源供水有限公司合作实施项目。但在初步审查中，发现绿源供水有限公司作为有限责任公司，其公司章程上只有公司盖章，没有股东签名。作为县政府法律顾问，请分析：这种情况是否影响绿源供水有限公司的合法性？如果不合法，是否会影响县政府将供水事业纳入计划的法定义务？",
        "clarification_question": None,
        "rewrite_request": None,  # 目前是空的
        "plan": ["query_rewrite_tool()", "retrieve_and_rerank_tool()", "FINISH"],  # Planner给的计划
        "intermediate_steps": [],
        "verification_history": [],
        "final_response": ""
    }

    # 执行第一步 (应该会执行 query_rewrite_tool)
    print("\n\n======== 第一次循环 ========")
    output_1 = task_executor_node(test_state)

    # 为了模拟 Graph 框架的行为，我们手动合并状态
    test_state.update(output_1)

    print(f"\n[状态快照 1] 剩余计划: {test_state['plan']}")
    print(f"[状态快照 1] 注入的新关键词: {test_state.get('rewrite_request')}")

    # 执行第二步 (此时状态里已经有 rewrite_request 了，应该会执行 retrieve_and_rerank_tool)
    print("\n\n======== 第二次循环 ========")
    output_2 = task_executor_node(test_state)
    print("---------------------------------------")
    print(test_state)