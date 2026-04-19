import pandas as pd
import json
import random
from openai import OpenAI

# --- 配置区 ---
client = OpenAI(api_key="sk-bdfd347ca40b4ba193ea4669afcee32c", base_url="https://api.deepseek.com")
INPUT_CSV = "../知识库/pkulaw_legalv2.csv"  # 你刚刚清理好的平铺式CSV
OUTPUT_JSONL = "legal_test_set_v3.jsonl"

# --- 1. 定义四类问题的核心 Prompt 策略 ---
# 每一类问题的 Prompt 都经过了法律逻辑强化
PROMPT_TEMPLATES = {
    "fact-based": (
        "你是一个法律考官。请基于提供的法条原文，设计一个简单的'事实型'咨询问题。"
        "要求：答案必须能直接从法条中提取，不涉及任何复杂推导。输出JSON。"
    ),
    "multi-hop": (
        "你是一个资深法官。请利用提供的法条，设计一个需要'多跳推理'的虚构案例咨询。"
        "要求：问题不能直接被单一法条回答，必须结合案情中的多个细节，通过逻辑链条得出答案。"
        "答案（ground_truth）应详细说明法律依据的推理过程。输出JSON。"
    ),
    "scenario": (
        "你是一个法律顾问。请设计一个'情景规划'问题。要求：案情描述应包含具体的人物（如张某、某公司）和明确的行为诉求。"
        "用户询问'我该怎么办'或'步骤是什么'等。答案必须分步骤提供可执行的行动指南。输出JSON。"
    ),
    "clarification": (
        "你是一个法律咨询师。请设计一个'模糊目标'的问题。要求：用户的问题必须描述含糊，关键法律要素严重缺失。"
        "例如：只说打架了但不说伤情，只说违约了但不说合同类型。答案（ground_truth）必须首先列出需要用户补充澄清的关键信息点。输出JSON。"
    )
}


# --- 2. 核心生成函数 ---
def generate_qa_pair(category, context_articles, current_id):
    # 将抽取的法条拼成上下文
    # 逻辑分流：复杂问题用推理模型 R1，简单问题用 V3
    chosen_model = "deepseek-reasoner" if category != "fact-based" else "deepseek-chat"
    context_text = ""
    relevant_laws_list = []
    for art in context_articles:
        context_text += f"法律名称：《{art['law_title']}》\n章节：{art['chapter_title']}\n条号：{art['article_number']}\n内容：{art['content']}\n---\n"
        relevant_laws_list.append(f"{art['law_title']} {art['article_number']}")

    sys_prompt = PROMPT_TEMPLATES[category]
    user_prompt = f"""
    参考法条原文：
    {context_text}

    请根据上述法条，生成一个ID为{current_id}的{category}类别测试数据。
    严格遵守以下JSON格式输出：
    {{
      "id": "{current_id}",
      "category": "{category}",
      "question": "法律咨询问题内容",
      "ground_truth": "标准答案内容（包含法律依据和逻辑推导）",
      "relevant_laws": {relevant_laws_list}
    }}
    """

    response = client.chat.completions.create(
        model=chosen_model,  # 动态选择模型
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ],
        # 注意：deepseek-reasoner 暂不支持 response_format={"type": "json_object"}
        # 建议通过 Prompt 强调 "请只输出 JSON 字符串"
    )

    # 解析逻辑 (针对 R1 可能返回的 Markdown 代码块进行清洗)
    content = response.choices[0].message.content
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    return json.loads(content)


# --- 3. 主程序逻辑 ---
def main():
    print("读取平铺版法律数据...")
    df = pd.read_csv(INPUT_CSV)
    # 将 dataframe 转为易处理的字典列表
    all_articles = df.to_dict('records')

    global_id = 1
    categories = ["fact-based", "multi-hop", "scenario", "clarification"]

    for cat in categories:
        print(f"正在生成类别: {cat}...")
        count = 0
        while count < 50:
            try:
                # 随机采样：多跳和情景类抽取2条相关法条，其余抽取1条
                sample_size = 2 if cat in ["multi-hop", "scenario"] else 1
                sampled_arts = random.sample(all_articles, sample_size)

                # 调用 LLM 生成
                case_json = generate_qa_pair(cat, sampled_arts, f"{global_id:03d}")

                # 实时保存到文件 (JSONL格式，一行一条，防止崩溃丢失)
                with open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
                    f.write(json.dumps(case_json, ensure_ascii=False) + "\n")

                count += 1
                global_id += 1
                print(f"进度: [{cat}] {count}/50 完成")
            except Exception as e:
                print(f"单条生成失败，正在重试... 错误: {e}")

    print(f"恭喜！200条高质量法律测试集已保存至: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()