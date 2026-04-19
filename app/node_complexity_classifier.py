import time
from typing import Dict, Any
from pydantic import BaseModel, Field
from state import AgentState
from app.llm_factory import LLMFactory
import config

class ComplexityResult(BaseModel):
    is_complex: bool = Field(
        description="If the query requires multi-step reasoning, web search, or involves specific recent events/companies, return true. If it is a straightforward legal fact question, return false."
    )

def complexity_classifier_node(state: AgentState) -> Dict[str, Any]:
    print("\n🔀 [-- 复杂度分类节点 (Complexity Classifier) 启动 --]")
    start_time = time.time()

    if not getattr(config, "ENABLE_COMPLEXITY_ROUTING", True):
        print("  -> ⚙️ 配置指示关闭分类器，默认按【复杂 (Complex)】处理...")
        return {"is_complex": True}

    request = state.get('original_request', '')

    prompt = f"""你是一个高级法律人工智能系统的【首席智能路由中枢（Router）】。
你的唯一任务是：精准评估用户法律诉求的复杂度，并将其分发到最匹配的底层处理引擎。
请务必严格以 JSON 格式输出结果！

【系统引擎差异认知】
- 静态基础 RAG (Baseline)：就像是直接翻阅法典目录。它只能处理一步到位的检索，适合能够通过简单的字面或语义直接在法律数据库中找到答案的通用问题。
- 多智能体 RAG (Agentic)：是一个拥有“思考-计划-调用工具-反思”循环的高级侦探。它被设计用来应对现实世界中案情交织、需要多维取证、或者涉及特定社会新闻事实的复杂案件。

【路由分发标准（极度严格）】

1. 判定为简单 (is_complex = false) ➡️ 走 Baseline RAG
   - 表现特征：问题极其简单、高度抽象、缺乏具体的现实案情支撑。
   - 核心测试：如果这个问题可以直接扔进搜索引擎搜出普法科普文章，或者直接对应某部特定法律的条款，即为简单。
   - 典型示例：
     * “《民法典》关于抵押权的规定是什么？”（纯法条询问）
     * “国家赔偿费用由谁管理？”（纯知识问答）
     * “产假国家规定多少天？”（通用普法）

2. 判定为复杂 (is_complex = true) ➡️ 走 Agentic RAG
   - 表现特征：问题是一段具体的、带有故事情节的现实纠纷（Scenario），或者包含多跳逻辑（Multi-hop）。
   - 强制触发条件（满足其一即为复杂）：
     * 提到了【特定的公司名】、【公众人物/明星】或【近期的热点新闻事件】（因为这需要调用联网搜索工具去查明新闻事实）。
     * 案情中包含多个需要拆解的要素（例如：既涉及试用期，又涉及未提前通知，还涉及赔偿金计算）。
   - 典型示例：
     * “我是芯启源的员工，网上曝光公司恶劣裁员，我该怎么维权？”（特定公司 + 新闻事件）
     * “我买了不少315晚会曝光的那种川渝漂白凤爪，拉肚子了能要求多重赔偿？”（新闻媒介 + 具体受害场景）
     * “老板突然把我辞退了，没给钱，也没提前30天通知，我能要几个月的补偿？”（复合型现实纠纷）

【当前用户诉求】
{request}

请仔细对照上述标准进行路由裁判。
"""

    llm = LLMFactory.get_gpt()
    llm.temperature = 0.0
    structured_llm = llm.with_structured_output(ComplexityResult, method="json_mode")

    try:
        result = structured_llm.invoke(prompt)
        is_complex = result.is_complex
    except Exception as e:
        print(f"  -> ⚠️ 分类器调用失败，降级为复杂模式: {e}")
        is_complex = True

    elapsed = round(time.time() - start_time, 4)
    print(f"  -> ✅ 判定结果: {'复杂 ➡️ [进规划器]' if is_complex else '简单 ➡️ [直达执行器]'} | 耗时: {elapsed}s")

    timing_stats = state.get('timing_stats', {}).copy()
    timing_stats['classification'] = timing_stats.get('classification', 0) + elapsed

    # 🌟 核心修改：如果是简单问题，我们提前帮它把极速直搜的 Plan 写好！
    updates = {
        "is_complex": is_complex,
        "timing_stats": timing_stats
    }
    if not is_complex:
        updates["plan"] = ["retrieve_and_rerank_tool()", "FINISH"]

    return updates