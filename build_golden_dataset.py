import pandas as pd
import json
import re
import os

# ==========================================
# 1. 配置文件路径
# ==========================================
CSV_PATH = "data/data_set/pkulaw_combined_articles_chunked.csv"
INPUT_JSONL = "data/test_set/mini_testset3.jsonl"
OUTPUT_JSONL = "data/test_set/mini_testset4new.jsonl"


def parse_law_string(law_str: str):
    """
    智能解析法条字符串，提取 法律名称 和 条文号
    支持格式: "《法名》第X条" 或 "法名 第X条"
    """
    law_str = law_str.strip()

    # 格式 1: 带有《》的书名号格式，例如 "《消费者权益保护法》第二条"
    match = re.match(r'《(.*?)》(.*)', law_str)
    if match:
        return match.group(1).strip(), match.group(2).strip()

    # 格式 2: 空格分隔格式，例如 "国家赔偿费用管理条例 第四条"
    parts = law_str.rsplit(' ', 1)  # 从右向左切分一次，防止法律名称里本身带空格
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()

    return law_str, ""


def main():
    if not os.path.exists(CSV_PATH):
        print(f"❌ 找不到 CSV 文件: {CSV_PATH}")
        return

    print("📚 正在加载知识库 CSV 并建立快速索引...")
    # ==========================================
    # 2. 读取 CSV 并建立 O(1) 的哈希索引，极速匹配
    # ==========================================
    df = pd.read_csv(CSV_PATH)

    # 构建字典：{(law_title, article_number): [index1, index2, ...]}
    # 存为列表是因为如果一个长法条被 chunk（切片）了，它会对应多个行号
    kb_index_map = {}
    for index, row in df.iterrows():
        title = str(row.get('law_title', '')).strip()
        art_num = str(row.get('article_number', '')).strip()

        key = (title, art_num)
        if key not in kb_index_map:
            kb_index_map[key] = []
        kb_index_map[key].append(str(index))  # 保存为字符串格式的 ID

    print(f"✅ 成功索引了 {len(df)} 条知识库数据。")
    print("-" * 50)

    # ==========================================
    # 3. 逐行读取 JSONL，匹配 ID 并生成新文件
    # ==========================================
    processed_count = 0
    missing_laws = set()  # 记录没有匹配上的法条，方便排查

    with open(INPUT_JSONL, 'r', encoding='utf-8') as fin, \
            open(OUTPUT_JSONL, 'w', encoding='utf-8') as fout:

        for line in fin:
            if not line.strip():
                continue

            data = json.loads(line)
            laws = data.get('relevant_laws', [])

            golden_doc_ids = []

            for law in laws:
                title, art_num = parse_law_string(law)
                key = (title, art_num)

                if key in kb_index_map:
                    # 如果匹配成功，把对应的所有行号（含切片）加进去
                    golden_doc_ids.extend(kb_index_map[key])
                else:
                    print(
                        f"⚠️ 警告: 知识库中未找到 -> 法律: '{title}', 条文: '{art_num}' (对应题目ID: {data.get('id')})")
                    missing_laws.add(law)

            # 🌟 新增字段：去重后存入 JSON
            # 用 list(set()) 去重，防止同一个切片被添加多次
            data['golden_doc_ids'] = list(set(golden_doc_ids))

            # 写入新的 JSONL 文件
            fout.write(json.dumps(data, ensure_ascii=False) + '\n')
            processed_count += 1

    print("-" * 50)
    print(f"🎉 处理完成！共处理了 {processed_count} 道测试题。")
    print(f"💾 新的测试集已保存至: {OUTPUT_JSONL}")

    if missing_laws:
        print(f"\n[注意] 以下 {len(missing_laws)} 个法条在 CSV 中没有找到完全一致的匹配，请检查是否名称有出入：")
        for ml in missing_laws:
            print(f"  - {ml}")


if __name__ == "__main__":
    main()