import json
from typing import Dict, Any
from langgraph.graph import StateGraph, END

# 1. 导入全局状态
from state import AgentState

# 2. 导入图谱节点
from node_ambiguity_judgment import ambiguity_judgment_node
from node_complexity_classifier import complexity_classifier_node
from node_task_planner import task_planner_node
from node_tool_executor import task_executor_node
from node_auditor import auditor_node
from node_generator import generator_node

# 3. 导入核心路由枢纽
from node_router import workflow_router

# ==========================================
# 第一步：初始化计算图并注册节点
# ==========================================
graph_builder = StateGraph(AgentState)

graph_builder.add_node("node_ambiguity", ambiguity_judgment_node)
graph_builder.add_node("node_complexity", complexity_classifier_node)  # 复杂度分类
graph_builder.add_node("node_task_planner", task_planner_node)
graph_builder.add_node("node_tool_executor", task_executor_node)
graph_builder.add_node("node_auditor", auditor_node)
graph_builder.add_node("node_synthesizer", generator_node)

# ==========================================
# 第二步：定义图谱的连线逻辑 (Edges)
# ==========================================
# 1. 设定入口点：模糊判断
graph_builder.set_entry_point("node_ambiguity")

# 2. 模糊判断分流
graph_builder.add_conditional_edges(
    "node_ambiguity",
    lambda state: "node_complexity" if state.get("clarification_question") is None else END,
    {
        "node_complexity": "node_complexity",
        END: END
    }
)

# 3. 🌟 复杂度分流：决定进规划器，还是直接带着硬编码 plan 进执行器
graph_builder.add_conditional_edges(
    "node_complexity",
    lambda state: "node_task_planner" if state.get("is_complex", True) else "node_tool_executor",
    {
        "node_task_planner": "node_task_planner",  # 复杂 -> 进规划器
        "node_tool_executor": "node_tool_executor"  # 简单 -> 直达执行器
    }
)

# 4. 规划节点 -> 执行节点
graph_builder.add_edge("node_task_planner", "node_tool_executor")


# 5. 🌟 执行节点内部循环及出口分流
def executor_router(state: AgentState) -> str:
    plan = state.get("plan", [])
    is_complex = state.get("is_complex", True)

    # 还有任务没执行完，继续循环
    if plan and plan[0] != "FINISH":
        return "node_tool_executor"

    # 任务全部执行完毕
    if not is_complex:
        # 如果是简单问题，直接去生成答案，跳过全局质检（Auditor）
        print("  -> ⚡ [快车道] 简单问题跳过质检，直达答案生成...")
        return "node_synthesizer"
    else:
        # 如果是复杂问题，必须交接给质检员
        return "node_auditor"


graph_builder.add_conditional_edges(
    "node_tool_executor",
    executor_router,
    {
        "node_auditor": "node_auditor",
        "node_synthesizer": "node_synthesizer",
        "node_tool_executor": "node_tool_executor"
    }
)

# 6. 审计打回判断
graph_builder.add_conditional_edges(
    "node_auditor",
    workflow_router,
    {
        "node_task_planner": "node_task_planner",
        "node_synthesizer": "node_synthesizer"
    }
)

# 7. 终点连线：合成节点结束后，流程彻底结束
graph_builder.add_edge("node_synthesizer", END)

# ==========================================
# 第三步：编译并固化图谱
# ==========================================
law_agent_app = graph_builder.compile()
print("🎉 Adaptive_Legal_Graph (自适应法律认知计算图) 编译成功！")


# ==========================================
# 第四步：封装最终的对外调用接口
# ==========================================
def run_law_agent(user_query: str):
    print(f"\n{'=' * 60}")
    print(f"⚖️ 接收到新的法律咨询案件：\n{user_query}")
    print(f"{'=' * 60}")

    initial_state = {
        "original_request": user_query,
        "clarification_question": None,
        "rewrite_request": None,
        "plan": [],
        "intermediate_steps": [],
        "verification_history": [],
        "final_response": "",
        "loop_count": 0,
        "is_complex": True,
        "timing_stats": {}
    }

    final_state = {}

    for output in law_agent_app.stream(initial_state, stream_mode="values"):
        final_state.update(output)

    print(f"\n{'=' * 60}")
    if final_state.get('clarification_question'):
        print("🧑‍⚖️ 【律师助理追问】")
        print(final_state['clarification_question'])
    else:
        print("🏛️ 【高级合伙人法律意见书】")
        print(final_state['final_response'])
    print(f"{'=' * 60}\n")

    return final_state


if __name__ == "__main__":
    # 【测试场景 1：极简常规常识】 -> 预期进入 Executor -> Synthesizer
    simple_query = "国家赔偿费用由哪个部门统一管理？"
    run_law_agent(simple_query)

    # 【测试场景 2：特定复杂新闻事实】 -> 预期进入 Task Planner -> Executor -> Auditor
    complex_query = "我是赵露思的粉丝，她跟银河酷娱闹解约违约要赔4亿，这合理吗？"
    run_law_agent(complex_query)