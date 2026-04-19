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
# """
# 网络搜索专家（Scout Agent）。
# ⚠️ 【调用规则极其严格，请100%严格遵守】：
# 1.  ❌ 【绝对禁止调用的场景】：
#     如果用户只是询问通用的法律规定、法条解释、常规维权流程，比如：
#     - "公司裁员不给N+1合法吗？"
#     - "工伤认定需要什么材料？"
#     - "消费者被欺诈能赔多少钱？"
#     这些问题的答案本地法律知识库已经完全覆盖，**绝对不要调用此工具**！
#
# 2.  ✅ 【必须调用的场景】：
#     只有当用户的问题中提到了**具体的、近期的特定公司、热点事件、公众人物**，
#     并且你需要查询该事件的背景事实、新闻详情才能回答时，才必须调用此工具，比如：
#     - "我是芯启源的员工，公司裁员不给赔偿怎么办？"
#     - "我买了315曝光的工业双氧水凤爪，能索赔吗？"
#     - "在李佳琦直播间买的花西子眉笔，被曝虚假宣传能退一赔三吗？"
#     这些特定事件的事实信息本地没有，必须联网搜索才能查明。
#
# 调用指南：直接调用 `scout_web_search_tool()` 即可，不需要传参数。
# """
@tool
def scout_web_search_tool(state: AgentState) -> dict:
    """
        网络搜索专家（Scout Agent）。
        ⚠️ 【调用规则极其严格，请100%严格遵守，否则将导致系统严重超时】：
        本系统核心为本地法律知识库，联网搜索极其耗时，**默认禁止调用**！你必须把联网搜索作为“最后的手段”。

        1. ❌ 【绝对禁止调用的场景（通用法律咨询）】
           如果用户询问的是通用的法律规定、法条解释、常规纠纷维权，无论用户描述得多么长，**绝对不要联网**！
           【禁止调用示例】：
           - "试用期被老板开除，能要赔偿吗？"
           - "买到了烂尾楼，怎么要求开发商退款？"
           - "无人驾驶汽车出车祸了，责任怎么划分？"
           - "遇到知假买假的职业打假人，法院一般怎么判？"

        2. ✅ 【必须且唯一允许调用的场景（特定新闻/热点事件追踪）】
           只有当用户的提问中**明确命中以下特征**，且你发现自己缺乏该事件的客观背景事实时，才允许调用：
           - 特征 A：提问中包含了明确的媒介词，如“网上曝出”、“新闻里说”、“315晚会”、“热搜”、“网传”等。
           - 特征 B：提问中绑定了极其具体的【特定知名公司】、【公众人物/明星】或【近期引发热议的特定社会案件】。

           【允许调用示例】：
           - "看了【315晚会曝光】的【某品牌淀粉肠】事件，我吃出病了怎么维权？" (特定媒介 + 具体事件)
           - "网传【某大厂】强制实行【大小周】，这种操作合法吗？" (新闻词汇 + 特定公司群体)
           - "最近【某男星】被【私生饭】跟踪到家里，这种滋扰行为能直接拘留吗？" (公众人物 + 特定侵权行为)
           - "像昨天【新闻里那个知名火锅店吃出老鼠】的案子，消费者能索赔多少？" (特指某个具体社会新闻案件)

        🎯 【调用决策自检（Mental Check）】：
        在输出计划前请问自己：用户是在问“普适性的法律常识”，还是在吃瓜讨论“某个具体新闻热点/公众人物的特定事件”？如果是前者，严禁使用本工具！

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
                # "url": ref.get("url", "无链接"),
                "content": ref.get("content", ""),
                # "published_date": ref.get("date", None),  # 对齐 Tavily 的时间字段
                # "score": 1.0  # 占位，兼容原有字段
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
# 本地批量测试入口 (Main 函数)
# ==========================================
if __name__ == "__main__":
    import csv

    print("🚀 开始批量测试 scout_web_search_tool 工具...")

    INPUT_FILE = "../data/test_set/web.jsonl"
    OUTPUT_FILE = "../output/暂时没用/websearch_test_result.csv"

    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # 定义 CSV 表头
    headers = [
        "id", "question",
        "result_1_title", "result_1_content",
        "result_2_title", "result_2_content",
        "result_3_title", "result_3_content",
        "cost_time_sec", "error"
    ]

    success_count = 0
    error_count = 0

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
                open(OUTPUT_FILE, 'w', encoding='utf-8-sig', newline='') as f_out:  # utf-8-sig 防止 Excel 乱码

            writer = csv.DictWriter(f_out, fieldnames=headers)
            writer.writeheader()

            lines = f_in.readlines()
            total_cases = len([line for line in lines if line.strip()])
            print(f"📂 成功读取测试文件，共找到 {total_cases} 个测试用例。\n" + "=" * 50)

            for i, line in enumerate(lines):
                if not line.strip():
                    continue

                # 1. 解析每一行的 JSON 数据
                item = json.loads(line)
                q_id = item.get("id", "unknown")
                question = item.get("question", "")

                print(
                    f"▶️ [进度 {success_count + error_count + 1}/{total_cases}] 正在测试 ID: {q_id} | {question[:20]}...")

                # 2. 构造 AgentState
                test_state = {
                    "original_request": question,
                    "clarification_question": None,
                    "rewrite_request": None,
                    "plan": [],
                    "intermediate_steps": [],
                    "verification_history": [],
                    "final_response": "",
                    "loop_count": 0,
                    "timing_stats": {}
                }

                # 初始化这一行要写入 CSV 的数据
                row_data = {
                    "id": q_id,
                    "question": question,
                    "cost_time_sec": 0,
                    "error": ""
                }

                try:
                    # 3. 调用工具进行搜索
                    results = scout_web_search_tool.invoke({"state": test_state})

                    search_results = results.get("search_results", [])
                    timing_stats = results.get("timing_stats", {})

                    # 记录耗时
                    row_data["cost_time_sec"] = timing_stats.get("web_search", 0)

                    # 4. 将搜索结果填入对应的列 (最多取前3个)
                    for j in range(min(3, len(search_results))):
                        res = search_results[j]
                        # 防御性判断，万一返回的不是字典
                        if isinstance(res, dict):
                            row_data[f"result_{j + 1}_title"] = res.get("title", "无标题")
                            row_data[f"result_{j + 1}_content"] = res.get("content", "")
                        else:
                            row_data[f"result_{j + 1}_content"] = str(res)

                    success_count += 1

                except Exception as e:
                    print(f"  ❌ ID {q_id} 搜索出错: {e}")
                    row_data["error"] = str(e)
                    error_count += 1

                # 5. 写入 CSV 并立刻刷新缓存（防止中途崩溃丢数据）
                writer.writerow(row_data)
                f_out.flush()

        print("\n" + "=" * 50)
        print(f"🏆 批量测试完成！")
        print(f"✅ 成功: {success_count} 条")
        print(f"❌ 失败: {error_count} 条")
        print(f"💾 结果已保存至: {OUTPUT_FILE}")

    except FileNotFoundError:
        print(f"❌ 找不到输入文件: {INPUT_FILE}，请检查路径。")
    except Exception as e:
        print(f"❌ 发生全局错误: {e}")