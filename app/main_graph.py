import json
from typing import Dict, Any

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, END

# 1. 导入我们之前写好的全局状态
from state import AgentState

# 2. 导入所有的图谱节点 (Nodes)
from node_ambiguity_judgment import ambiguity_judgment_node
from node_task_planner import task_planner_node
from node_tool_executor import task_executor_node
from node_auditor import auditor_node
from node_generator import generator_node

# 3. 导入我们的核心路由枢纽 (Router)
from node_router import workflow_router

# ==========================================
# 第一步：初始化计算图并注册节点
# ==========================================
# 明确告诉 LangGraph，我们的数据流体是 AgentState 字典
graph_builder = StateGraph(AgentState)

# 将我们写的 Python 函数注册为图谱上的“工作站”
graph_builder.add_node("node_ambiguity", ambiguity_judgment_node)
graph_builder.add_node("node_task_planner", task_planner_node)
graph_builder.add_node("node_tool_executor", task_executor_node)
graph_builder.add_node("node_auditor", auditor_node)
graph_builder.add_node("node_synthesizer", generator_node)

# ==========================================
# 第二步：定义图谱的连线逻辑 (Edges)
# ==========================================
# 1. 设定入口点：所有请求的第一站必须是“模糊判断”
graph_builder.set_entry_point("node_ambiguity")

# 2. 入口处的条件分流：如果判断模糊，直接走向图谱终点 (END)；否则去规划器。
# 这里用一个简单的 lambda 匿名函数处理
graph_builder.add_conditional_edges(
    "node_ambiguity",
    lambda state: "node_task_planner" if state.get("clarification_question") is None else END,
    {
        "node_task_planner": "node_task_planner",
        END: END
    }
)

# 3. 规划节点 -> 执行节点 (制定好计划后，必须去执行)
graph_builder.add_edge("node_task_planner", "node_tool_executor")

# 4. 执行节点内部循环：干完一个活，看看还有没有活，全干完再交接给质检
def executor_router(state: AgentState) -> str:
    plan = state.get("plan", [])
    # 如果计划空了，或者遇到了 FINISH 终点标识
    if not plan or plan[0] == "FINISH":
        return "node_auditor" # 任务全部执行完毕，送去全局质检
    else:
        return "node_tool_executor" # 还有任务，继续死磕执行节点

graph_builder.add_conditional_edges(
    "node_tool_executor",
    executor_router,
    {
        "node_auditor": "node_auditor",
        "node_tool_executor": "node_tool_executor"
    }
)

# 5. 【核心认知循环】：审计节点 -> 智能路由枢纽
# 路由枢纽会决定是打回重做(planner)、继续干活(executor)、还是生成报告(synthesizer)
graph_builder.add_conditional_edges(
    "node_auditor",
    workflow_router,
    {
        "node_task_planner": "node_task_planner",
        "node_synthesizer": "node_synthesizer"
    }
)

# 6. 终点连线：合成节点结束后，流程彻底结束
graph_builder.add_edge("node_synthesizer", END)

# ==========================================
# 第三步：编译并固化图谱
# ==========================================
law_agent_app = graph_builder.compile()
print("🎉 Legal_Cognitive_Graph (法律认知计算图) 编译成功！")


# ==========================================
# 第四步：封装最终的对外调用接口
# ==========================================
def run_law_agent(user_query: str):
    """
    对外的统一调用函数：输入用户问题，输出最终结果
    """
    print(f"\n{'=' * 60}")
    print(f"⚖️ 接收到新的法律咨询案件：\n{user_query}")
    print(f"{'=' * 60}")

    # 初始化干净的全局状态
    initial_state = {
        "original_request": user_query,
        "clarification_question": None,
        "rewrite_request": None,
        "plan": [],
        "intermediate_steps": [],
        "verification_history": [],
        "final_response": "",
        "loop_count": 0,  # 初始轮次设为 0
        "timing_stats": {}  # 初始化时间统计字典
    }

    final_state = {}

    # stream 模式可以在控制台实时看到节点之间的数据流转情况
    for output in law_agent_app.stream(initial_state, stream_mode="values"):
        final_state.update(output)

    print(f"\n{'=' * 60}")
    # 结果分发：如果是被守门员拦下，打印追问；否则打印最终意见书
    if final_state.get('clarification_question'):
        print("🧑‍⚖️ 【律师助理追问】")
        print(final_state['clarification_question'])
    else:
        print("🏛️ 【高级合伙人法律意见书】")
        print(final_state['final_response'])
    print(f"{'=' * 60}\n")

    return final_state


# ==========================================
# 本地测试：双场景实战演练
# ==========================================
if __name__ == "__main__":
    # 【测试场景 1：极其模糊的口水话】
    ambiguous_query = "打官司要多少钱啊？"
    run_law_agent(ambiguous_query)

    # 【测试场景 2：复杂的复合型法律问题】
    complex_query = "我是赵露思的粉丝，她跟银河酷娱闹解约，公司说她违约要赔4亿，这合理吗？"
    run_law_agent(complex_query)