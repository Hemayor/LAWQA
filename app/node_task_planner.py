from typing import Dict, Any, List
from pydantic import BaseModel, Field
import config # 引入全局配置
# 引入状态定义和已注册的工具箱
from agent_graph import tools
from state import AgentState
from llm_factory import LLMFactory


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
def create_planner_prompt(request: str) -> str:
    tool_descriptions = "\n".join([f"- {tool.name}: {tool.description.strip()}" for tool in tools])

    # 注意这里：JSON 示例部分使用双括号 {{ 和 }}，因为最外层是 f-string
    # 底部直接插入 {request}，不再需要后续的 .format()
    return f"""你是一个首席AI架构师（Supervisor）。
你的任务是阅读下方提供的【可用工具说明书】，根据用户的请求，自主决定需要按什么顺序调用哪些工具，制定多步执行计划。

**当前可用工具说明书：**
{tool_descriptions}

**极其重要的纪律：**
1. 你只能规划工具的名字，绝对不能传递任何参数！底层框架会自动处理参数的传递。
2. 你必须严格输出 JSON 格式。返回一个包含 `plan` 键的 JSON 对象，该键的值为工具名字符串数组。
格式示例：
{{
    "plan": ["scout_web_search_tool()","query_rewrite_tool()", "retrieve_and_rerank_tool()", "FINISH"]
    "plan": ["query_rewrite_tool()", "retrieve_and_rerank_tool()", "FINISH"]
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

    # 【新增】：每次进入规划器，说明开启了新的一轮，轮次 +1
    current_loop = state.get("loop_count", 0)
    new_loop_count = current_loop + 1

    request = state['original_request']

    # 【核心修复】：直接调用函数并传入 request 生成最终的 prompt
    prompt = create_planner_prompt(request)

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
        return {
            "plan": response_obj.plan,
            "loop_count": new_loop_count,
            "intermediate_steps": []
        }

    except Exception as e_ds:
        print(f"  -> ⚠️ DeepSeek 计划解析失败或网络异常: {e_ds}")
        print("  -> [尝试 2] 启动高可用降级策略，切换至 GPT-4o-mini...")

        try:
            response_obj = structured_gpt.invoke(prompt)
            plan_list = response_obj.plan
            print(f"  -> ✅ GPT-4o-mini 挽救成功，制定计划为: {plan_list}")
            return {"plan": plan_list,
                    "loop_count": new_loop_count,
                    "intermediate_steps": []
            }

        except Exception as e_gpt:
            print(f"  -> ❌ 致命错误：GPT-4o-mini 也解析失败: {e_gpt}")
            print("  -> 兜底机制：跳过工具，直接结束计划。")
            return {"plan": ["FINISH"],
                    "loop_count": new_loop_count,
                    "intermediate_steps": []
            }
# ==========================================
# 本地独立测试入口
# ==========================================
if __name__ == "__main__":
    print("🚀 开始独立测试 Task Planner 节点 (双模型高可用版)...")

    # 构造完整的假 State
    test_state_1 = {
        "original_request": "我是赵露思的粉丝，她跟银河酷娱闹解约，公司说她违约要赔4亿，这合理吗？",
        "clarification_question": None,
        "rewrite_request": None,
        "plan": [],
        "intermediate_steps": [],
        "verification_history": [],
        "final_response": ""
    }

    print("\n【测试案例 1：本地法律检索】")
    task_planner_node(test_state_1)