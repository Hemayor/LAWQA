import json
from typing import Dict, Any
import config
# 引入状态定义和模型工厂
from state import AgentState
from llm_factory import LLMFactory


def generator_node(state: AgentState) -> Dict[str, Any]:
    """
    生成器节点：整合所有工具的执行结果，运用法律逻辑进行深度推演，生成最终的法律意见。
    """
    print("\n🖋️ [-- 生成器节点 (Generator) 启动 --]")

    # 1. 提取原始请求
    request = state.get('original_request', '无')
    intermediate_steps = state.get('intermediate_steps', [])

    # 2. 如果没有任何执行步骤，直接兜底回复
    if not intermediate_steps:
        print("  -> ⚠️ 未发现任何工具执行线索，启动常识回复模式。")
        context_str = "无检索到额外的上下文信息。"
    else:
        # 3. 优雅地组装工具上下文 (Context Assembling)
        formatted_steps = []
        for i, step in enumerate(intermediate_steps):
            tool_name = step.get('tool_name', '未知工具')

            # 安全序列化工具输出，防止乱码
            try:
                output_str = json.dumps(step.get('tool_output'), ensure_ascii=False, indent=2)
            except Exception:
                output_str = str(step.get('tool_output'))

            formatted_steps.append(f"### [步骤 {i + 1}] 工具来源: {tool_name}\n**执行结果:**\n{output_str}")

        context_str = "\n\n".join(formatted_steps)

    # 4. 专为高级法律分析定制的 Prompt (杜绝纯总结，要求深度穿透)
    prompt = f"""你是一位律师。
你的任务是根据智能助手们收集到的所有线索（法条、新闻、数据），回答用户的法律问题。

**客户原始诉求：**
{request}

**案卷调查线索（来自各工具的汇编）：**
{context_str}

**撰写指令（必须严格遵循）：**
1. **精准回应：** 首先，用通俗易懂的语言直接回答客户的核心诉求，不要说废话。
2. **证据援引：** 结合提供的线索，引用具体的法条或事实来支撑你的结论。

你的最终法律意见："""

    # 5. 调用大模型 (保留少许温度值，赋予它将线索串联的“创造力和直觉”)
    # 这里非常适合使用 DeepSeek-Reasoner（深度思考模型），如果为了速度，使用普通 V3 也可以
    llm = LLMFactory.get_deepseek()
    llm.temperature = config.GENERATOR_TEMP  # 0.3 能够在保持严谨事实的同时，激发它的联想推演能力

    print("  -> 正在进行跨线索融合与法律逻辑推演...")
    final_answer = llm.invoke(prompt).content
    print("  -> ✅ 最终法律意见书生成完毕！")

    # 返回最终状态
    return {"final_response": final_answer}


# ==========================================
# 本地独立测试入口
# ==========================================
if __name__ == "__main__":
    print("🚀 开始独立测试 Generator 节点...")

    # 模拟一个非常前沿的复合型问题：不仅需要看法律，还需要看最新新闻动态
    test_state = {
        'original_request': "昨天一辆开启了自动驾驶的特斯拉撞了我的车，特斯拉官方说这是系统Bug。我该怎么索赔？",
        'intermediate_steps': [
            # 步骤 1：本地检索到了产品质量法和侵权责任法
            {
                'tool_name': 'retrieve_and_rerank_tool',
                'tool_output': [
                    {'title': '《中华人民共和国民法典》第一千二百零二条',
                     'content': '因产品存在缺陷造成他人损害的，被侵权人可以向产品的生产者请求赔偿，也可以向产品的销售者请求赔偿。'},
                    {'title': '《道路交通安全法》',
                     'content': '机动车发生交通事故造成人身伤亡、财产损失的，由保险公司在机动车第三者责任强制保险责任限额范围内予以赔偿；不足的部分，按照下列规定承担赔偿责任...'}
                ]
            },
            # 步骤 2：全网搜到了特斯拉的近期大新闻
            {
                'tool_name': 'scout_web_search_tool',
                'tool_output': [
                    {'title': '国家市场监督管理总局发文',
                     'content': '因自动驾驶系统FSD存在引发碰撞的风险，特斯拉(上海)有限公司宣布召回160万辆汽车，通过OTA升级修复该缺陷。'},
                ]
            }
        ],
        'verification_history': [{'confidence_score': 5}]  # 假定审计全票通过
    }

    output = generator_node(test_state)

    print("\n" + "=" * 50)
    print("🏛️ 最终交付给客户的法律意见书：")
    print("=" * 50 + "\n")
    print(output['final_response'])