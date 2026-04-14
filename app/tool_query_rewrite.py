from typing import Dict, Any
import time

from langchain_core.tools import tool

from state import AgentState
from app.llm_factory import LLMFactory


@tool
def query_rewrite_tool(state: AgentState) -> Dict[str, Any]:
    """
    专门用于优化和改写用户查询的工具。
    将用户模糊或口语化的提问转化为包含专业法律术语、具体案由或关键实体的精准检索词。
    调用指南：如果计划中包含此工具，直接调用 `query_rewrite_tool()` 即可，不要传任何自定义参数。它会自动读取原始问题。
    """
    # ⏱️ 开始计时
    start_time = time.time()

    query = state['original_request']
    print(f"\n[Tool 1] 正在改写查询: '{query}'")

    # 这里调用你配置好的 DeepSeek 模型
    llm = LLMFactory.get_deepseek()

    # 核心改进：引入了严格的格式限制和 Few-Shot 示例
    prompt = f"""你是一个专业的法律检索词优化专家。你的【唯一任务】是将用户的口语化提问，转化为适合在向量数据库（北大法宝）中进行精确检索的**高质量关键词组**。

    【核心绝对规则】
    1. 绝对不要回答问题，不要给出法律建议，不要解释概念！
    2. 剥离所有口语化词汇（如“怎么办”、“别人”、“怎么判”）。
    3. 提取标准案由、核心法律关系和专业术语。
    4. 输出格式必须是【用空格分隔的词组】，不要写完整的句子，不要用标点符号连句！
    5. 输出总长度严格控制在 25 个汉字以内。

    【改写案例学习】
    输入：别人欠钱不还怎么办
    输出：民间借贷纠纷 逾期还款 支付令 强制执行

    输入：老婆出轨了我想离婚怎么分财产
    输出：离婚财产纠纷 婚内过错 夫妻共同财产分割

    输入：老板突然把我辞退了，没给钱
    输出：劳动争议 违法解除劳动合同 经济补偿金

    输入：买的二手房漏水，中介不管
    输出：房屋买卖合同纠纷 隐瞒瑕疵 违约责任

    现在的用户输入：{query}
    请严格按照上述格式，只输出优化后的检索词："""

    optimized_query = llm.invoke(prompt).content.strip()
    print(f"  -> 优化后查询: '{optimized_query}'")

    # ⏱️ 计时结束，累计到 timing_stats
    end_time = time.time()
    elapsed_time = round(end_time - start_time, 4)

    timing_stats = state.get('timing_stats', {}).copy()
    timing_stats['rewrite'] = timing_stats.get('rewrite', 0) + elapsed_time
    print(f"  -> 改写耗时: {elapsed_time}s | 累计改写时间: {timing_stats['rewrite']}s")

    return {
        "rewrite_request": optimized_query,
        "timing_stats": timing_stats
    }

# (可选) 独立运行测试逻辑
if __name__ == "__main__":
    test_q = "我有一艘旧船需要处理，想了解一下相关的法律要求。"
    print(query_rewrite_tool.invoke(test_q))

