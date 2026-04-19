from typing import Dict, Any
import json
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


# # ==========================================
# # 本地独立测试入口
# # ==========================================
# if __name__ == "__main__":
#     print("🚀 开始独立测试 Ambiguity Judgment 节点...")
#
#     # 测试案例 1：极其模糊的法律提问
#     test_state_1 = {"original_request": "老板把我开了，没给钱，怎么办？"}
#     result_1 = ambiguity_judgment_node(test_state_1)
#     print(f"\n【案例 1 测试结果】:\n {result_1}\n")
#     print("-" * 50)
#
#     # 测试案例 2：要素齐全的清晰提问
#     test_state_2 = {
#         "original_request": "我今年3月入职了一家公司，签了2年劳动合同，试用期3个月。但昨天老板突然口头通知我被辞退了，没有任何正当理由，也没提前通知。请问我能要求赔偿半个月的经济补偿金吗？"
#     }
#     result_2 = ambiguity_judgment_node(test_state_2)
#     print(f"\n【案例 2 测试结果】:\n {result_2}\n")
# ==========================================
# 本地独立测试入口
# ==========================================
if __name__ == "__main__":
    print("🚀 开始批量测试 Ambiguity Judgment 节点...")
    print(f"📂 测试文件: ../data/test_set/mini_testset3.jsonl")

    test_file = "../data/test_set/无用/mini_testset3.jsonl"

    # 统计变量，支持混合测试集
    total = 0
    correct = 0

    # 分类型统计
    clear_total = 0
    clear_correct = 0
    ambiguous_total = 0
    ambiguous_correct = 0

    error_cases = []

    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            total = len(lines)
            print(f"🔍 共加载 {total} 个测试用例\n")

            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    q_id = sample['id']
                    question = sample['question']
                    ground_truth = sample['ground_truth'].strip()
                    category = sample['category']
                    # 🔍 自动判断这个测试用例是【清晰】还是【模糊】
                    is_ground_clear = (category != "clarification")
                    case_type = "清晰问题" if is_ground_clear else "模糊问题"

                    print(f"\n{'=' * 60}")
                    print(f"【测试用例 {i + 1}/{total} | ID: {q_id} | 类型: {case_type}】")
                    print(f"用户问题: {question}")

                    # 构造状态，调用节点
                    test_state = {"original_request": question}
                    result = ambiguity_judgment_node(test_state)

                    # 模型的预测结果
                    pred_clarify = result['clarification_question']
                    is_pred_clear = (pred_clarify is None)

                    # ✅ 核心判断：预测和标准答案是否一致
                    if is_ground_clear == is_pred_clear:
                        # 预测正确
                        correct += 1
                        if is_ground_clear:
                            clear_correct += 1
                        else:
                            ambiguous_correct += 1

                        print(f"\n✅ 识别正确！")
                        if is_ground_clear:
                            print(f"  模型正确识别为清晰问题，直接放行。")
                        else:
                            print(f"  模型正确识别为模糊问题，生成了澄清。")
                    else:
                        # 预测错误
                        error_type = ""
                        if is_ground_clear and not is_pred_clear:
                            error_type = "把【清晰问题】误判成了模糊，生成了不必要的澄清"
                        else:
                            error_type = "把【模糊问题】误判成了清晰，错误地放行了"

                        error_cases.append({
                            "id": q_id,
                            "question": question,
                            "error": error_type,
                            "model_output": pred_clarify
                        })
                        print(f"\n❌ 识别错误！{error_type}")

                    # 统计总数
                    if is_ground_clear:
                        clear_total += 1
                    else:
                        ambiguous_total += 1

                except Exception as e:
                    print(f"\n⚠️  处理测试用例 {i + 1} 出错: {str(e)}")
                    continue

        # 全部跑完，输出汇总报告
        print("\n\n" + "=" * 60)
        print("🏆 批量测试完成！汇总报告:")
        print(f"  总测试用例: {total}")
        print(f"  整体准确率: {round(correct / total * 100, 2)}%")
        print("-" * 40)
        if clear_total > 0:
            print(f"  清晰问题: {clear_total} 个 | 识别准确率: {round(clear_correct / clear_total * 100, 2)}%")
        if ambiguous_total > 0:
            print(
                f"  模糊问题: {ambiguous_total} 个 | 识别准确率: {round(ambiguous_correct / ambiguous_total * 100, 2)}%")
        print("-" * 40)
        print(f"  错误用例数: {len(error_cases)}")

        if error_cases:
            print("\n❌ 错误的测试用例详情:")
            for err in error_cases:
                print(f"  - ID: {err['id']} | 问题: {err['question']}")
                print(f"    错误原因: {err['error']}")

    except FileNotFoundError:
        print(f"\n❌ 错误：找不到测试文件 {test_file}，请检查路径是否正确！")
    except Exception as e:
        print(f"\n❌ 运行出错: {str(e)}")