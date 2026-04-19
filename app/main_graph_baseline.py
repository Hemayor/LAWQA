import time
import json
from typing import Dict, Any

# 导入所需的工具和生成节点
from app.tool_retrieve_rerank import retrieve_and_rerank_tool
from app.node_generator import generator_node


# ==========================================
# 对外统一调用接口 (对齐 Agent 版本的入参和出参)
# ==========================================
def run_law_agent(user_query: str) -> Dict[str, Any]:
    """
    基线 RAG (Baseline RAG) 对外调用接口：
    流程极简：接受请求 -> 检索+精排 -> 生成答案。
    """
    print(f"\n{'=' * 60}")
    print(f"⚖️ [Baseline RAG] 接收到查询：\n{user_query}")
    print(f"{'=' * 60}")

    # 1. 初始化对齐的状态结构
    state = {
        "original_request": user_query,
        "clarification_question": None,
        "rewrite_request": None,  # 基线系统没有改写
        "plan": ["retrieve_and_rerank_tool()", "FINISH"],  # 写死执行计划
        "intermediate_steps": [],
        "verification_history": [],
        "final_response": "",
        "loop_count": 1,
        "timing_stats": {}
    }

    # --------------------------------------------------
    # 步骤一：直接执行检索与精排
    # --------------------------------------------------
    print("  -> 📡 正在直接执行本地检索...")
    try:
        retrieve_result = retrieve_and_rerank_tool.invoke({"state": state})

        # 将检索结果装入 intermediate_steps
        state["intermediate_steps"].append({
            "tool_name": "retrieve_and_rerank_tool",
            "tool_output": retrieve_result
        })

        # 同步时间消耗
        if "timing_stats" in retrieve_result:
            state["timing_stats"].update(retrieve_result["timing_stats"])

    except Exception as e:
        print(f"  -> ❌ 检索失败: {e}")
        state["intermediate_steps"].append({
            "tool_name": "retrieve_and_rerank_tool",
            "tool_output": {"error": str(e), "timing_stats": {}}
        })

    # --------------------------------------------------
    # 步骤二：直接生成答案
    # --------------------------------------------------
    print("  -> 📝 正在综合生成最终答案...")
    # 🌟 核心修复：使用 update 更新字典，保留检索到的 intermediate_steps
    generator_updates = generator_node(state)
    state.update(generator_updates)

    # --------------------------------------------------
    # 步骤三：格式化输出
    # --------------------------------------------------
    print(f"\n{'=' * 60}")
    print("🏛️ 【Baseline 法律意见书】")
    print(state['final_response'])
    print(f"{'=' * 60}\n")

    return state


if __name__ == "__main__":
    complex_query = "根据《国家赔偿费用管理条例》的规定，国家赔偿费用由哪个部门统一管理？"
    run_law_agent(complex_query)