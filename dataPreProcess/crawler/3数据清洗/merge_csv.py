import pandas as pd
import os

# 定义文件路径列表
file_paths = [
    '知识库/01法律部分_(298)/pkulaw_legalv1.csv',
    '知识库/ad01_(588)/pkulaw_legalv1.csv',
    '知识库/司法/pkulaw_legalv1.csv'
]

# 定义输出路径
output_path = 'pkulaw_legalv1.csv'


def merge_legal_csv(paths, save_to):
    df_list = []

    for path in paths:
        if os.path.exists(path):
            print(f"正在读取: {path}")
            # 读取 CSV，如果文件中有中文，建议加上 encoding='utf-8-sig'
            df = pd.read_csv(path, encoding='utf-8-sig')
            df_list.append(df)
        else:
            print(f"⚠️ 警告: 未找到文件 {path}")

    if df_list:
        # 合并所有 DataFrame
        merged_df = pd.concat(df_list, ignore_index=True)

        # 确保输出目录存在
        os.makedirs(os.path.dirname(save_to), exist_ok=True)

        # 保存合并后的文件
        merged_df.to_csv(save_to, index=False, encoding='utf-8-sig')
        print(f"✅ 合并成功！总行数: {len(merged_df)}")
        print(f"文件已保存至: {save_to}")
    else:
        print("❌ 没有找到任何可合并的数据。")


if __name__ == "__main__":
    merge_legal_csv(file_paths, output_path)