import os
import json
import time
import requests
from langchain_core.tools import tool
from dotenv import load_dotenv  # 新增：加载.env文件的库
from state import AgentState

# 🔧 自动加载项目根目录下的 .env 文件
load_dotenv()

# 尝试从环境变量中读取 API Key (现在会自动从 .env 里读)
QIANFAN_API_KEY = os.getenv("QIANFAN_API_KEY")

if not QIANFAN_API_KEY:
    raise ValueError("❌ 未找到 QIANFAN_API_KEY！请检查项目根目录下的 .env 文件配置。")

# 百度千帆 API 地址
API_URL = "https://qianfan.baidubce.com/v2/ai_search/web_search"



# """
#     网络搜索专家（Scout Agent）。
#     用途：用于查找本地知识库中没有的【实时信息、新闻事实、热点事件】。
#     只有用户询问关于时事热点，才会优先调用此工具查明事实。
#     对于普通法律问题，不需要调用此工具
#     调用指南：直接调用 `scout_web_search_tool()` 即可，不需要传参数。
# """
@tool
def scout_web_search_tool(state: AgentState) -> dict:
    """
    网络搜索专家（Scout Agent）。
    ⚠️ 【调用规则极其严格，请100%严格遵守】：
    1.  ❌ 【绝对禁止调用的场景】：
        如果用户只是询问通用的法律规定、法条解释、常规维权流程，比如：
        - "公司裁员不给N+1合法吗？"
        - "工伤认定需要什么材料？"
        - "消费者被欺诈能赔多少钱？"
        这些问题的答案本地法律知识库已经完全覆盖，**绝对不要调用此工具**！

    2.  ✅ 【必须调用的场景】：
        只有当用户的问题中提到了**具体的、近期的特定公司、热点事件、公众人物**，
        并且你需要查询该事件的背景事实、新闻详情才能回答时，才必须调用此工具，比如：
        - "我是芯启源的员工，公司裁员不给赔偿怎么办？"
        - "我买了315曝光的工业双氧水凤爪，能索赔吗？"
        - "在李佳琦直播间买的花西子眉笔，被曝虚假宣传能退一赔三吗？"
        这些特定事件的事实信息本地没有，必须联网搜索才能查明。

    调用指南：直接调用 `scout_web_search_tool()` 即可，不需要传参数。
    """
    # ⏱️ 开始计时
    start_time = time.time()

    query = state['original_request']
    print(f"\n[Tool 3] 侦察兵出动 🚀 正在全网搜索: '{query}'")

    try:
        # 构造百度千帆 API 请求
        headers = {
            "Authorization": f"Bearer {QIANFAN_API_KEY}",
            "Content-Type": "application/json"
        }

        # 🔧 严格对齐原 Tavily 的参数颗粒度：
        # - max_results=3 → 只返回3条结果
        # - 时间范围：最近1年
        # - 屏蔽低质量站点
        payload = {
            "messages": [{"role": "user", "content": query}],
            "search_source": "baidu_search_v2",
            "edition": "standard",
            "resource_type_filter": [{"type": "web", "top_k": 3}],  # 对齐原 max_results=3
            "search_recency_filter": "year",
            "block_websites": ["tieba.baidu.com", "zhidao.baidu.com"]
        }

        # 调用 API
        resp = requests.post(API_URL, headers=headers, json=payload)
        resp.raise_for_status()
        result = resp.json()

        # 检查 API 是否返回错误
        if result.get("code"):
            raise Exception(f"API Error: {result.get('message')}")

        references = result.get("references", [])

        # 🎯 核心：把百度的返回格式，**无损转换成 Tavily 完全一致的格式**
        # 保证上层代码完全感知不到切换，无缝兼容
        search_results = []
        for ref in references:
            search_results.append({
                "title": ref.get("title", "无标题"),
                "url": ref.get("url", "无链接"),
                "content": ref.get("content", ""),
                "published_date": ref.get("date", None),  # 对齐 Tavily 的时间字段
                "score": 1.0  # 占位，兼容原有字段
            })

        print(f"  - 搜索完成：找到了 {len(search_results)} 条相关的优质网页摘要。")

        # ⏱️ 计时结束，累计到 timing_stats
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 4)

        timing_stats = state.get('timing_stats', {}).copy()
        timing_stats['web_search'] = timing_stats.get('web_search', 0) + elapsed_time
        print(f"  - 搜索耗时: {elapsed_time}s | 累计搜索时间: {timing_stats['web_search']}s")

        return {
            "search_results": search_results,
            "timing_stats": timing_stats
        }

    except Exception as e:
        print(f"  - ❌ 搜索失败: {e}")

        # ⏱️ 计时结束（出错也要记录时间）
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 4)

        timing_stats = state.get('timing_stats', {}).copy()
        timing_stats['web_search'] = timing_stats.get('web_search', 0) + elapsed_time

        return {
            "search_results": [{"url": "error", "content": f"搜索工具发生错误: {str(e)}"}],
            "timing_stats": timing_stats
        }


# ==========================================
# 本地测试入口 (Main 函数)
# ==========================================
if __name__ == "__main__":
    print("🚀 开始独立测试 scout_web_search_tool 工具...")

    # 1. 构造一个模拟的 AgentState 字典
    test_state = {
        "original_request": "我是赵露思的粉丝，她跟银河酷娱闹解约，公司说她违约要赔4亿，这合理吗？",
        "clarification_question": None,
        "rewrite_request": None,
        "plan": [],
        "intermediate_steps": [],
        "verification_history": [],
        "final_response": "",
        "loop_count": 0
    }

    try:
        # 2. 调用工具
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
            print(f"    📄 摘要: {clean_content}")

    except Exception as e:
        print(f"\n❌ 运行出错: {e}")