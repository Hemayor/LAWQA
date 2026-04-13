import config
from langgraph.graph import END
from state import AgentState


def workflow_router(state: AgentState) -> str:
    """
    智能路由枢纽：纯粹的条件判断函数（不调用大模型）。
    既然到达了这里，说明【执行器】已经把活干完，且【审计员】已经给出了全局打分。
    """
    print("\n🚦 [-- 智能路由枢纽 (Workflow Router) 启动 --]")

    # ==========================================
    # 优先级 1：事实不清，直接拦截
    # ==========================================
    if state.get("clarification_question"):
        print("  -> 🛑 决策：提问过于模糊，终止流程。")
        return END

    loop_count = state.get("loop_count", 0)
    verification_history = state.get("verification_history", [])

    # ==========================================
    # 优先级 2：检查审计成绩单
    # ==========================================
    if verification_history:
        last_audit = verification_history[-1]

        # 如果不及格 (< 3 分)
        if last_audit.get("confidence_score", 5) < config.PASSING_SCORE:

            # 熔断保护：如果已经重试了太多次，强制进入【生成器节点】作答！
            if loop_count >= config.MAX_LOOPS:
                print(f"  -> 🚨 触发死循环保护：已达最大重试次数 ({config.MAX_LOOPS})，放弃抢救，生成保底回答！")
                return "node_synthesizer"

            # 打回重做：指路给 Planner
            else:
                print(f"  -> ⚠️ 决策：全局审计未通过。打回重做 (已消耗轮次: {loop_count}/{config.MAX_LOOPS})")
                return "node_task_planner"

    # ==========================================
    # 优先级 3：完美通关 (及格了，或者没有打分记录)
    # ==========================================
    print("  -> 📝 决策：全局质检合格。前往【最终合成节点】输出答案。")
    return "node_synthesizer"