import os
import json
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from state import AgentState
from llm_factory import LLMFactory  # 👈 引入大模型工厂

# 尝试从环境变量中读取 API Key
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError("❌ 未找到 TAVILY_API_KEY！请检查环境变量配置。")

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# 👈 核心修改 1：开启 advanced 深度搜索，大幅减少页脚和侧边栏的垃圾文本
tavily_base = TavilySearch(max_results=3, search_depth="advanced")


@tool
def scout_web_search_tool(state: AgentState) -> list:
    """
    网络搜索专家（Scout Agent）。
    用途：用于查找本地知识库中没有的【实时信息、新闻事实、热点事件或当前案件动态】。
    只有用户询问近期发生的某件事，才会优先调用此工具查明事实。
    对于普通法律问题，不需要调用此工具
    调用指南：直接调用 `scout_web_search_tool()` 即可，不需要传参数。
    """
    raw_query = state['original_request']

    # 👈 核心修改 2：使用大模型提取搜索引擎友好的关键词
    print(f"\n[Tool 3] 侦察兵正在提炼搜索词...")
    llm = LLMFactory.get_deepseek()
    llm.temperature = 0.0

    prompt = f"""你是一个搜索引擎指令专家。
用户的原始提问通常是冗长且口语化的。你的任务是提取其中最关键的事实、公司名或核心法律争议，生成适合搜索引擎（如Google/Baidu）的简练关键词组合。
多个关键词之间用空格隔开，不要输出任何解释说明。

用户提问: {raw_query}
搜索关键词:"""

    optimized_query = llm.invoke(prompt).content.strip()

    print(f"  -> 🚀 侦察兵正式出动! 全网搜索: '{optimized_query}'")

    try:
        # 使用提炼后的关键词去搜索
        response = tavily_base.invoke({"query": optimized_query})

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

    test_state = {
        # "original_request": "我在北京的一家互联网公司被裁员了，公司以‘经营困难’为由拒绝支付N+1补偿。但我今天刚在新闻上看到这家公司完成了C轮两千万美元的融资。请问我该怎么维权？",
        "original_request": "单依纯到底咋了",

        "clarification_question": None,
        "rewrite_request": None,
        "plan": [],
        "intermediate_steps": [],
        "verification_history": [],
        "final_response": "",
        "loop_count": 0
    }

    try:
        results = scout_web_search_tool.invoke({"state": test_state})

        print("\n🏆 --- 过滤后的“人类友好版”搜索结果 ---")

        for i, res in enumerate(results):
            if isinstance(res, dict):
                title = res.get('title', '无标题')
                url = res.get('url', '无链接')
                raw_content = res.get('content', str(res))
            else:
                title = "无标题"
                url = "未知"
                raw_content = str(res)

            clean_content = raw_content

            print(f"\n[{i + 1}] 标题: {title}")
            print(f"    🔗 链接: {url}")
            print(f"    📄 摘要: {clean_content}...")  # 限制打印长度，防止刷屏

    except Exception as e:
        print(f"\n❌ 运行出错: {e}")