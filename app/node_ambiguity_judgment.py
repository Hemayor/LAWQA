from typing import Dict, Any
# 引入我们之前定义的 AgentState 和大模型工厂
from state import AgentState
from llm_factory import LLMFactory


def ambiguity_judgment_node(state: AgentState) -> Dict[str, Any]:
    """
    模糊判断节点：检查用户的法律咨询是否过于模糊，是否需要补充事实背景。
    """
    print("\n⚖️ [-- 模糊判断节点 (Ambiguity Judgment) 启动 --]")

    # 提取用户原始请求
    request = state['original_request']

    # 获取大模型实例 (在逻辑判断节点，我们需要模型尽可能严谨，温度设为 0)
    llm = LLMFactory.get_deepseek()
    llm.temperature = 0.0

    # 针对法律场景深度定制的 Prompt
    prompt = f"""你是一个资深的法律咨询助理。你的任务是在进行复杂的法律文书检索之前，判断用户的提问是否清晰。

判断标准：
- 【具体清晰的请求】：包含具体的案件事实、特定的法律关系、金额、时间或明确的诉求。
  （例如：“我在试用期被无故辞退，没有提前30天通知，能要求赔偿几个月的工资？”）
- 【模糊宽泛的请求】：缺乏事实背景，过于笼统，无法直接对应到具体的法律条文。
  （例如：“怎么打官司？”、“我被坑了怎么办？”、“签了合同对方违约了怎么算？”）

指令：
1. 如果用户的请求模糊，请站在律师助理的角度，生成一个礼貌、专业的反问，引导用户补充缺失的关键信息（如：标的额、有无书面证据、具体受损情况等）。
2. 如果请求已经足够清晰，能够直接进行法律分析，请严格且仅回复两个字：“明确”。

用户请求: "{request}"
你的回复:"""

    # 调用 LLM 进行判断
    response = llm.invoke(prompt).content.strip()

    # 解析输出结果
    if response == "明确":
        print("  -> ✅ 案件要素清晰，放行至规划节点 (Planner)。")
        return {"clarification_question": None}
    else:
        print(f"  -> ⚠️ 案件要素模糊，已生成追问: {response}")
        return {"clarification_question": response}


# ==========================================
# 本地独立测试入口
# ==========================================
if __name__ == "__main__":
    print("🚀 开始独立测试 Ambiguity Judgment 节点...")

    # 测试案例 1：极其模糊的法律提问
    test_state_1 = {"original_request": "老板把我开了，没给钱，怎么办？"}
    result_1 = ambiguity_judgment_node(test_state_1)
    print(f"\n【案例 1 测试结果】:\n {result_1}\n")
    print("-" * 50)

    # 测试案例 2：要素齐全的清晰提问
    test_state_2 = {
        "original_request": "我今年3月入职了一家公司，签了2年劳动合同，试用期3个月。但昨天老板突然口头通知我被辞退了，没有任何正当理由，也没提前通知。请问我能要求赔偿半个月的经济补偿金吗？"
    }
    result_2 = ambiguity_judgment_node(test_state_2)
    print(f"\n【案例 2 测试结果】:\n {result_2}\n")