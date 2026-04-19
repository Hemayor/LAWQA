from typing import Dict, Any, List
from pydantic import BaseModel, Field
import time
import config # 引入全局配置
# 引入状态定义和已注册的工具箱
from agent_graph import tools
from state import AgentState
from llm_factory import LLMFactory
import json

# ==========================================
# 1. 定义 Pydantic 数据结构
# ==========================================
class AgentPlan(BaseModel):
    """大模型生成的执行计划模型"""
    plan: List[str] = Field(
        description="""一个由工具函数名字符串组成的列表。
        你只需要写出工具的名字加空括号即可（例如 "query_rewrite_tool()"）。绝对不要在括号里传递任何参数！
        如：["scout_web_search_tool()","query_rewrite_tool()", "retrieve_and_rerank_tool()", "FINISH"]
        列表的最后一个元素必须始终是固定的字符串: "FINISH"。
        """
    )


# ==========================================
# 2. 生成纯动态的系统提示词 (修改为每次动态传入 request)
# ==========================================
def create_planner_prompt(request: str, history: List[Dict] = None) -> str:
    tool_descriptions = "\n".join([f"- {tool.name}: {tool.description.strip()}" for tool in tools])

    # 【新增】：提取审计历史，形成反思上下文
    reflection_context = ""
    if history and len(history) > 0:
        latest_audit = history[-1]
        reflection_context = f"""
        【警告：上一次检索计划已被审计节点驳回！】
        上一轮的审计打分为: {latest_audit.get('confidence_score', 0)} / 5
        驳回理由与改进建议: {latest_audit.get('reasoning', '无具体理由')}

        你必须反思上述驳回理由，并改变策略。例如：缺乏背景事实则加scout_web_search_tool，法条不相关则先调query_rewrite_tool。
        """

    # 注意这里：JSON 示例部分使用双括号 {{ 和 }}，因为最外层是 f-string
    # 底部直接插入 {request}，不再需要后续的 .format()
    return f"""你是一个首席AI架构师（Supervisor）。
你的任务是阅读下方提供的【可用工具说明书】，根据用户的请求，自主决定需要按什么顺序调用哪些工具，制定多步执行计划。

**当前可用工具说明书：**
{tool_descriptions}

{reflection_context}

**极其重要的纪律：**
1. 你只能规划工具的名字，绝对不能传递任何参数！底层框架会自动处理参数的传递。
2. 你必须严格输出 JSON 格式。返回一个包含 `plan` 键的 JSON 对象，该键的值为工具名字符串数组。
格式示例：
{{
    "plan": ["scout_web_search_tool()","query_rewrite_tool()", "retrieve_and_rerank_tool()", "FINISH"]
}}
{{
    "plan": ["query_rewrite_tool()", "retrieve_and_rerank_tool()", "FINISH"]
}}
{{
    "plan": ["retrieve_and_rerank_tool()", "FINISH"]
}}

---
用户请求: {request}"""


# ==========================================
# 3. 节点逻辑 (双模型高可用版)
# ==========================================
def task_planner_node(state: AgentState) -> Dict[str, Any]:
    """
    任务规划节点：优先使用 DeepSeek，失败则降级使用 GPT-4o-mini
    """
    print("\n🧠 [-- 任务规划节点 (Task Planner) 启动 --]")

    # ⏱️ 开始计时
    start_time = time.time()

    current_loop = state.get("loop_count", 0)
    new_loop_count = current_loop + 1

    request = state['original_request']
    history = state.get('verification_history', [])

    # 【核心修复】：将历史反馈传入，生成动态 prompt
    prompt = create_planner_prompt(request, history)

    if history:
        print(f"  -> 发现审计驳回历史，已启用【反思纠错模式】进行重规划...")

    # 1. 实例化两个模型，都确保严谨性 (temperature=0.0)
    llm_deepseek = LLMFactory.get_deepseek()
    llm_deepseek.temperature = config.PLANNER_TEMP

    llm_gpt = LLMFactory.get_gpt()
    llm_gpt.temperature = config.PLANNER_TEMP

    # 2. 分别绑定 Pydantic (为了兼容 DeepSeek，统一指定 json_mode)
    structured_ds = llm_deepseek.with_structured_output(AgentPlan, method="json_mode")
    structured_gpt = llm_gpt.with_structured_output(AgentPlan, method="json_mode")

    # 3. 执行降级策略 (Fallback)
    try:
        print(f"  -> [尝试 1] 正在使用 DeepSeek 制定计划 (当前轮次: {new_loop_count})...")
        response_obj = structured_ds.invoke(prompt)
        print(f"  -> ✅ DeepSeek 制定计划成功: {response_obj.plan,}")

        # ⏱️ 计时结束，累计到 timing_stats
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 4)

        timing_stats = state.get('timing_stats', {}).copy()
        timing_stats['planning'] = timing_stats.get('planning', 0) + elapsed_time
        print(f"  -> 规划耗时: {elapsed_time}s | 累计规划时间: {timing_stats['planning']}s")

        return {
            "plan": response_obj.plan,
            "loop_count": new_loop_count,
            "intermediate_steps": [],
            "timing_stats": timing_stats
        }

    except Exception as e_ds:
        print(f"  -> ⚠️ DeepSeek 计划解析失败或网络异常: {e_ds}")
        print("  -> [尝试 2] 启动高可用降级策略，切换至 GPT-4o-mini...")

        try:
            response_obj = structured_gpt.invoke(prompt)
            plan_list = response_obj.plan
            print(f"  -> ✅ GPT-4o-mini 挽救成功，制定计划为: {plan_list}")

            # ⏱️ 计时结束，累计到 timing_stats
            end_time = time.time()
            elapsed_time = round(end_time - start_time, 4)

            timing_stats = state.get('timing_stats', {}).copy()
            timing_stats['planning'] = timing_stats.get('planning', 0) + elapsed_time
            print(f"  -> 规划耗时: {elapsed_time}s | 累计规划时间: {timing_stats['planning']}s")

            return {
                "plan": plan_list,
                "loop_count": new_loop_count,
                "intermediate_steps": [],
                "timing_stats": timing_stats
            }

        except Exception as e_gpt:
            print(f"  -> ❌ 致命错误：GPT-4o-mini 也解析失败: {e_gpt}")
            print("  -> 兜底机制：跳过工具，直接结束计划。")

            # ⏱️ 计时结束，累计到 timing_stats
            end_time = time.time()
            elapsed_time = round(end_time - start_time, 4)

            timing_stats = state.get('timing_stats', {}).copy()
            timing_stats['planning'] = timing_stats.get('planning', 0) + elapsed_time
            print(f"  -> 规划耗时: {elapsed_time}s | 累计规划时间: {timing_stats['planning']}s")

            return {
                "plan": ["FINISH"],
                "loop_count": new_loop_count,
                "intermediate_steps": [],
                "timing_stats": timing_stats
            }
# ==========================================
# 本地独立测试入口
# ==========================================
# if __name__ == "__main__":
#     print("🚀 开始独立测试 Task Planner 节点 (双模型高可用版)...")
#
#     # 构造完整的假 State
#     test_state_1 = {
#         "original_request": "我是赵露思的粉丝，她跟银河酷娱闹解约，公司说她违约要赔4亿，这合理吗？",
#         "clarification_question": None,
#         "rewrite_request": None,
#         "plan": [],
#         "intermediate_steps": [],
#         "verification_history": [],
#         "final_response": ""
#     }
#
#     print("\n【测试案例 1：本地法律检索】")
#     task_planner_node(test_state_1)


# ==========================================
# 本地独立测试入口
# ==========================================
if __name__ == "__main__":
    print("🚀 开始批量测试 Task Planner 节点 (双模型高可用版)...")

    # 🔧 你可以在这里修改测试文件的路径，默认用我们之前生成的多跳问答测试集
    test_file = "../data/test_set/无用/mini_testset3.jsonl"
    # 🔧 结果输出文件
    output_file = "../output/暂时没用/planner_test_result.txt"

    # 提前提取合法的工具名，用来校验模型输出
    valid_tool_names = {tool.name + "()" for tool in tools}
    print(f"✅ 合法工具列表: {valid_tool_names}")
    print(f"📂 测试文件: {test_file}")
    print(f"📝 结果将输出到: {output_file}")

    # 统计变量
    total = 0
    format_correct = 0
    plan_with_search = 0
    plan_without_search = 0
    error_cases = []

    try:
        with open(test_file, 'r', encoding='utf-8') as f, \
                open(output_file, 'w', encoding='utf-8') as out_f:

            # 写入 CSV 表头，方便你用 Excel 打开
            out_f.write("id,plan,is_format_correct\n")

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

                    print(f"\n{'=' * 60}")
                    print(f"【测试用例 {i + 1}/{total} | ID: {q_id}】")
                    print(f"用户问题: {question}")

                    # 构造状态，调用节点
                    test_state = {
                        "original_request": question,
                        "clarification_question": None,
                        "rewrite_request": None,
                        "plan": [],
                        "intermediate_steps": [],
                        "verification_history": [],
                        "final_response": "",
                        "loop_count": 0
                    }
                    result = task_planner_node(test_state)
                    plan = result['plan']

                    # 🔍 校验 plan 格式是否正确
                    # 1. 最后一个必须是 FINISH
                    # 2. 前面的工具名必须是合法的
                    is_format_ok = True
                    format_error = ""

                    if not plan:
                        is_format_ok = False
                        format_error = "Plan 为空"
                    else:
                        if plan[-1] != "FINISH":
                            is_format_ok = False
                            format_error = f"最后一个元素不是 FINISH，而是: {plan[-1]}"
                        else:
                            # 检查前面的工具名
                            for tool_name in plan[:-1]:
                                if tool_name not in valid_tool_names:
                                    is_format_ok = False
                                    format_error = f"包含非法工具名: {tool_name}"
                                    break

                    # 检查是否包含搜索工具
                    has_search = any("scout_web_search_tool" in p for p in plan)

                    # 📝 写入结果到文件
                    # 转义 plan 里的双引号，防止破坏 CSV 格式
                    plan_str = str(plan).replace('"', '\\"')
                    out_f.write(f'{q_id},"{plan_str}",{is_format_ok}\n')
                    out_f.flush()  # 实时写入，防止程序崩溃丢数据

                    # 统计
                    if is_format_ok:
                        format_correct += 1
                        if has_search:
                            plan_with_search += 1
                        else:
                            plan_without_search += 1
                        print(f"\n✅ 格式校验通过！")
                        if has_search:
                            print(f"  模型决定需要联网搜索: {plan}")
                        else:
                            print(f"  模型决定仅本地检索: {plan}")
                    else:
                        error_cases.append({
                            "id": q_id,
                            "question": question,
                            "error": format_error,
                            "plan": plan
                        })
                        print(f"\n❌ 格式校验失败！{format_error}")
                        print(f"  错误的 Plan: {plan}")

                except Exception as e:
                    print(f"\n⚠️  处理测试用例 {i + 1} 出错: {str(e)}")
                    continue

        # 全部跑完，输出汇总报告
        print("\n\n" + "=" * 60)
        print("🏆 批量测试完成！汇总报告:")
        print(f"  总测试用例: {total}")
        print(f"  格式正确的 Plan: {format_correct} 个 | 格式准确率: {round(format_correct / total * 100, 2)}%")
        print("-" * 40)
        if format_correct > 0:
            print(f"  其中，模型决定【需要联网搜索】的: {plan_with_search} 个")
            print(f"  其中，模型决定【仅本地检索】的: {plan_without_search} 个")
        print("-" * 40)
        print(f"  格式错误的 Plan: {len(error_cases)} 个")
        print(f"\n📄 所有测试结果已保存到: {output_file}")

        if error_cases:
            print("\n❌ 错误的测试用例详情:")
            for err in error_cases:
                print(f"  - ID: {err['id']} | 问题: {err['question']}")
                print(f"    错误原因: {err['error']}")
                print(f"    错误 Plan: {err['plan']}")

    except FileNotFoundError:
        print(f"\n❌ 错误：找不到测试文件 {test_file}，请检查路径是否正确！")
        # 如果找不到，就跑原来的单个测试
        print("\n🔄  fallback 到原来的单个测试...")
        # 构造完整的假 State
        test_state_1 = {
            "original_request": "我是赵露思的粉丝，她跟银河酷娱闹解约，公司说她违约要赔4亿，这合理吗？",
            "clarification_question": None,
            "rewrite_request": None,
            "plan": [],
            "intermediate_steps": [],
            "verification_history": [],
            "final_response": "",
            "loop_count": 0
        }
        print("\n【测试案例 1：本地法律检索】")
        task_planner_node(test_state_1)
    except Exception as e:
        print(f"\n❌ 运行出错: {str(e)}")