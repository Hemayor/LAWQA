#新包超级长
import os
import json
from langchain_core.tools import tool
# 使用最新包中的正确类名
from langchain_tavily import TavilySearch
from state import AgentState

# 尝试从环境变量中读取 API Key
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError("❌ 未找到 TAVILY_API_KEY！请检查环境变量配置。")

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# 实例化基础搜索工具 (限制返回3条)
# tavily_base = TavilySearch(max_results=3)
tavily_base = TavilySearch(
    max_results=3,
    search_depth="advanced",  # 开启深度搜索（会调用性能更好的搜索模型，更精准）
    include_raw_content=True  # 开启抓取网页的完整纯文本原文
)


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
        # 新版本的 TavilySearch 返回的是一个包含全局信息的【字典】，而不是列表！
        response = tavily_base.invoke({"query": query})
        print(response)
        # 核心修复：把真正的网页数据从 "results" 键里面提取出来
        if isinstance(response, dict):
            search_results = response.get("results", [])
        else:
            search_results = response

        print(f"  - 搜索完成：找到了 {len(search_results)} 条相关的优质网页摘要。")
        return search_results

    except Exception as e:
        print(f"  - ❌ 搜索失败: {e}")
        return [{"url": "error", "content": f"搜索工具发生错误: {str(e)}"}]


# ==========================================
# 本地测试入口 (Main 函数)
# ==========================================
if __name__ == "__main__":
    print("🚀 开始独立测试 scout_web_search_tool 工具...")

    # 1. 构造一个模拟的 AgentState 字典
    # 填充所有定义的字段，保证符合 TypedDict 的结构规范
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
        # 2. 调用工具
        # 注意：LangChain 的 @tool 会将字典的 key 映射到函数的参数名上
        # 因为我们函数签名是 def scout_web_search_tool(state: AgentState)
        # 所以 invoke 时需要把装有状态的字典赋值给 "state" 键
        results = scout_web_search_tool.invoke({"state": test_state})

        print("\n🏆 --- 过滤后的“人类友好版”搜索结果 ---")

        for i, res in enumerate(results):
            # 加上类型判断的防弹衣，更加稳健
            if isinstance(res, dict):
                title = res.get('title', '无标题')
                url = res.get('url', '无链接')
                # raw_content = res.get('content', str(res))
                # 提取网页长文本原文，如果没有拿到原文，再退回使用简短的 content
                raw_content = res.get('raw_content') or res.get('content', str(res))
            else:
                title = "无标题"
                url = "未知"
                raw_content = str(res)

            # 净化文本，去掉换行符并截断显示，保证控制台干净
            # clean_content = raw_content.replace('\n', ' ')[:150] + "......"
            clean_content = raw_content

            print(f"\n[{i + 1}] 标题: {title}")
            print(f"    🔗 链接: {url}")
            print(f"    📄 摘要: {clean_content}")

    except Exception as e:
        print(f"\n❌ 运行出错: {e}")


