import json
from typing import Dict, Any
from pydantic import BaseModel, Field

# 引入状态定义和模型工厂
from state import AgentState
from llm_factory import LLMFactory


# ==========================================
# 1. 定义审计结果的数据结构 (Pydantic)
# ==========================================
class AuditResult(BaseModel):
    """审计节点的结构化输出模型"""
    confidence_score: int = Field(
        description="对工具输出质量的信心评分，范围 1-5。1代表完全无关/空数据，5代表极度相关且充分。"
    )
    is_consistent: bool = Field(description="工具输出的数据内部逻辑是否一致？")
    is_relevant: bool = Field(description="工具输出的内容是否与用户原始请求的核心诉求直接相关？")
    reasoning: str = Field(description="给出打分的简要理由，特别是指出缺失了什么信息。")


# ==========================================
# 2. 审计节点逻辑
# ==========================================
def auditor_node(state: AgentState) -> Dict[str, Any]:
    """
    审计节点：【全局质检】审查所有工具执行完毕后的综合数据质量。
    """
    print("\n🔍 [-- 审计节点 (Auditor) 启动 - 全局质检模式 --]")

    request = state['original_request']
    intermediate_steps = state.get('intermediate_steps', [])

    # 如果没有任何执行记录，直接给最低分
    if not intermediate_steps:
        print("  -> ⚠️ 计划执行完毕，但没有任何工具返回数据，全局审计失败。")
        fallback_result = AuditResult(
            confidence_score=1,
            is_consistent=False,
            is_relevant=False,
            reasoning="工具执行队列为空，未获取到任何有效信息。"
        )
        return {"verification_history": state.get('verification_history', []) + [fallback_result.model_dump()]}

    # 【核心改动】：把所有执行步骤的数据拼接成一份“总卷宗”
    formatted_outputs = []
    for i, step in enumerate(intermediate_steps):
        tool_name = step.get('tool_name', '未知工具')
        tool_output = step.get('tool_output', '')
        # 安全处理复杂的 JSON 序列化
        try:
            output_str = json.dumps(tool_output, ensure_ascii=False)
        except Exception:
            output_str = str(tool_output)

        formatted_outputs.append(f"[步骤 {i + 1} | 工具: {tool_name}]\n{output_str}")

    global_context = "\n\n".join(formatted_outputs)

    # 修改 Prompt，让大模型基于“总卷宗”打分
    prompt = f"""你是一个苛刻的资深法务审核员。系统刚刚执行了一系列调查工具，请审查这些工具收集到的【综合数据】是否足以回答用户的法律问题。

**用户原始诉求:** {request}
**所有工具收集到的综合数据卷宗:** {global_context}

**质检标准:**
1. **全局相关性 (Relevance):** 综合来看，这些内容是否凑齐了回答用户问题所需的核心事实和法条？
2. **一致性 (Consistency):** 各个工具带回来的数据是否存在致命的逻辑矛盾？

**输出要求:**
请严格输出 JSON 格式，必须包含 confidence_score (1-5的整数), is_consistent (布尔值), is_relevant (布尔值) 以及 reasoning (字符串)。
"""

    # ... 下方调用 LLM 的代码保持不变 ...
    llm_deepseek = LLMFactory.get_deepseek()
    llm_deepseek.temperature = 0.0
    structured_ds = llm_deepseek.with_structured_output(AuditResult, method="json_mode")

    try:
        print("  -> 正在对工具输出进行质量评估...")
        audit_result = structured_ds.invoke(prompt)
    except Exception as e:
        print(f"  -> ⚠️ DeepSeek 审计解析失败: {e}，尝试切换 GPT...")
        llm_gpt = LLMFactory.get_gpt()
        llm_gpt.temperature = 0.0
        structured_gpt = llm_gpt.with_structured_output(AuditResult, method="json_mode")
        audit_result = structured_gpt.invoke(prompt)

    print(f"  -> 质检完成！信心评分: {audit_result.confidence_score}/5")

    # 将新的审计结果附加到验证历史中 (使用 model_dump() 兼容 Pydantic V2)
    current_history = state.get('verification_history', [])
    return {"verification_history": current_history + [audit_result.model_dump()]}


# ==========================================
# 本地独立测试入口
# ==========================================
if __name__ == "__main__":
    print("🚀 开始独立测试 Auditor 节点...")

    # 模拟一个用户问劳动法，但是检索工具却搜出了一堆“婚姻法”的错误结果
    test_state_bad = {
        'original_request': '公司没给我交社保，我被迫离职能要经济补偿金吗？',
        'intermediate_steps': [
            {
                'tool_name': 'retrieve_and_rerank_tool',
                'tool_output': [
                    {'content': '《中华人民共和国民法典》第一千零七十六条 夫妻双方自愿离婚的，应当签订书面离婚协议...'},
                    {'content': '《婚姻登记条例》...'}
                    ]
            }
        ],
        'verification_history': []
    }

    print("\n【测试案例 1：检索到毫不相关的法条 (预期低分)】")
    bad_result = auditor_node(test_state_bad)
    print(json.dumps(bad_result['verification_history'][0], indent=2, ensure_ascii=False))

    print("-" * 50)

    # 模拟检索到了正确的法条
    test_state_good = {
        'original_request': '公司没给我交社保，我被迫离职能要经济补偿金吗？',
        'intermediate_steps': [
            {
                'tool_name': 'retrieve_and_rerank_tool',
                'tool_output': [{
                                    'content': '《中华人民共和国劳动合同法》第三十八条 用人单位有下列情形之一的，劳动者可以解除劳动合同：...（三）未依法为劳动者缴纳社会保险费的...'},
                                {
                                    'content': '《中华人民共和国劳动合同法》第四十六条 有下列情形之一的，用人单位应当向劳动者支付经济补偿...'}
                                ]
            }
        ],
        'verification_history': []
    }

    print("\n【测试案例 2：检索到极其对口的法条 (预期高分)】")
    good_result = auditor_node(test_state_good)
    print(json.dumps(good_result['verification_history'][0], indent=2, ensure_ascii=False))