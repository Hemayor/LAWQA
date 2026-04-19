import pandas as pd
import json
import ast


def clean_legal_data(input_path, output_path):
    print(f"🚀 开始读取原始数据: {input_path}")
    # 读取原始数据
    df = pd.read_csv(input_path, encoding='utf-8-sig')

    new_rows = []

    for _, row in df.iterrows():
        # 获取基础信息，这些信息将重复出现在该法的所有法条行中
        base_info = {
            'law_title': row['title'],
            'law_code': row['law_code'],
            'legal_level': row['legal_level'],
            'category': row['category'],
        }

        # 解析 chapters 字段
        chapters_str = row['chapters']
        try:
            # chapters 可能是字符串形式的列表，使用 ast.literal_eval 或 json.loads 安全解析
            if isinstance(chapters_str, str):
                # 针对你提供的格式，通常 ast 更稳健，如果是标准 JSON 则用 json.loads
                chapters_data = ast.literal_eval(chapters_str)
            else:
                chapters_data = chapters_str

            for chapter in chapters_data:
                chapter_title = chapter.get('chapter_title', '')
                articles = chapter.get('articles', [])

                for article in articles:
                    # 创建新行：基础信息 + 章节信息 + 法条信息
                    new_row = base_info.copy()
                    new_row.update({
                        'chapter_title': chapter_title,
                        'article_number': article.get('article_number', ''),
                        'content': article.get('content', ''),
                        'judicial_case': article.get('judicial_case', ''),
                        'relevant_laws': article.get('relevant_laws', [])
                    })
                    new_rows = [] if new_rows is None else new_rows  # 防御性
                    new_rows.append(new_row)

        except Exception as e:
            print(f"❌ 解析法律 '{row['title']}' 的章节时出错: {e}")
            continue

    # 创建新的 DataFrame
    new_df = pd.DataFrame(new_rows)

    # 保存结果
    new_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ 处理完成！")
    print(f"📊 原始法律部数: {len(df)}")
    print(f"📈 拆解后总条数: {len(new_df)}")
    print(f"💾 已保存至: {output_path}")


if __name__ == "__main__":
    input_file = 'pkulaw_legalv1.csv'
    output_file = 'pkulaw_legalv2.csv'
    clean_legal_data(input_file, output_file)