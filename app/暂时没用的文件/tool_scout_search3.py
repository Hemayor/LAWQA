#老包内容长度适中
import os
import json
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

from state import AgentState

# ==========================================
# 1. API 密钥配置区 (最佳实践)
# ==========================================
# 尝试从环境变量中读取 API Key
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# 严谨的校验逻辑：如果没有拿到 Key，直接报错提示，防止带着空凭证去请求报错
if not TAVILY_API_KEY:
    raise ValueError(
        "❌ 未找到 TAVILY_API_KEY！\n"
        "请确保你已经设置了环境变量，或者在项目根目录的 .env 文件中配置了该值。\n"
        "例如: TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxx"
    )

# 确保 LangChain 底层组件能正确读到这个变量（双重保险）
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# ==========================================
# 2. 实例化基础搜索工具
# ==========================================
tavily_base = TavilySearchResults(max_results=3)


# ==========================================
# 3. 包装为 Agent 可用的 Tool
# ==========================================
@tool
def scout_web_search_tool(state: AgentState) -> list:
    """
    网络搜索专家（Scout Agent）。
    用途：用于查找本地知识库中没有的【实时信息、新闻事实、热点事件或当前案件动态】。
    如果用户询问近期发生的某件事，应优先调用此工具查明事实。
    调用指南：直接调用 `scout_web_search_tool()` 即可，不需要传参数。
    """
    query = state['original_request']
    print(f"\n[Tool 3] 侦察兵出动 🚀 正在全网搜索: '{query}'")

    try:
        results = tavily_base.invoke({"query": query})
        print(f"  - 搜索完成：找到了 {len(results)} 条相关的优质网页摘要。")
        return results

    except Exception as e:
        print(f"  - ❌ 搜索失败: {e}")
        return [{"url": "error", "content": f"搜索工具发生错误: {str(e)}"}]


# ==========================================
# 4. 本地测试入口 (Main 函数)
# ==========================================
if __name__ == "__main__":
    print("🚀 开始独立测试 scout_web_search_tool 工具...")

    test_state = {
        "original_request": "单依纯抄袭",
        "clarification_question": None,
        "rewrite_request": None,
        "plan": [],
        "intermediate_steps": [],
        "verification_history": [],
        "final_response": ""
    }
    try:
        results = scout_web_search_tool.invoke({"state": test_state})

        print("\n🏆 --- Tavily 联网搜索精简结果 ---")
        print(json.dumps(results, indent=4, ensure_ascii=False))

    except Exception as e:
        print(f"\n❌ 运行出错: {e}")