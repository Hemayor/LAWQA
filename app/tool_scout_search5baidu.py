from langchain.tools import tool
import requests

# ⚠️  请把这里替换成你自己的百度千帆API Key！
API_KEY = "bce-v3/ALTAK-oLfRM7TtlAC6al91fPlEw/2df8da31178a065ddeac9718e2054e3ada1a0b70"
API_URL = "https://qianfan.baidubce.com/v2/ai_search/web_search"


# 把这个API封装成Agent的工具
@tool
def baidu_legal_search(query: str) -> str:
    """
    搜索全网的实时法律新闻、事件、政策信息，用来查询最新的热点事件、维权案例等。
    当你需要了解最新的社会事件、企业新闻的时候，调用这个工具。
    Args:
        query: 要搜索的关键词，比如「芯启源暴力裁员」
    """
    global API_KEY, API_URL

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    # 针对法律场景优化的搜索参数
    payload = {
        "messages": [{"role": "user", "content": query}],
        "search_source": "baidu_search_v2",
        # "edition": "standard",
        "edition": "lite",

        # 只返回10条网页结果
        "resource_type_filter": [{"type": "web", "top_k": 3}],
        # 只搜最近1年的内容
        "search_recency_filter": "year",
        # 屏蔽低质量论坛站点
        "block_websites": ["tieba.baidu.com", "zhidao.baidu.com"]
    }

    try:
        resp = requests.post(API_URL, headers=headers, json=payload)
        resp.raise_for_status()
        result = resp.json()
        references = result.get("references", [])

        if not references:
            return "没有搜索到相关的实时信息"

        # 把结果整理成Agent能看懂的格式
        output = []
        for ref in references:
            output.append(
                f"【标题】{ref['title']}\n"
                f"【摘要】{ref['content']}\n"
                f"【来源】{ref['url']}\n"
                f"【发布时间】{ref['date']}"
            )
        return "\n\n".join(output)
    except Exception as e:
        return f"搜索出错了：{str(e)}"


# 测试用的main函数，运行脚本会自动执行这个
def main():
    # 你可以把这里改成你要测试的其他关键词，比如"影石创新 竞业协议"
    test_query = "芯启源暴力裁员"
    print(f"=== 开始测试百度AI搜索，测试关键词：{test_query} ===")
    print("正在调用API，请稍候...\n")

    # LangChain Tool 标准调用方式：用 .invoke() 传参数字典
    search_result = baidu_legal_search.invoke({"query": test_query})

    # 打印结果
    print("=== 搜索结果如下 ===")
    print(search_result)


if __name__ == "__main__":
    # 先检查你有没有替换API Key，避免你直接运行报错
    if API_KEY == "你的百度千帆API_Key":
        print("⚠️  提醒：请先把代码里的「API_KEY」变量替换成你自己的API Key，再运行！")
        print("你可以在百度智能云千帆的应用详情里找到你的API Key")
    else:
        main()