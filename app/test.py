import pandas as pd

file_path = "../data/data_set/pkulaw_combined_articles_chunked.csv"
df = pd.read_csv(file_path)

# 情况 1：如果你指的 38054 是它的物理行号（即索引值，从 0 开始）
print("========== 【按行号查找 (Index: 38054)】 ==========")
print(df.iloc[16801])

print("\n" + "="*50 + "\n")

# 情况 2：如果你原始 CSV 里本来就有一列叫 'id' 或 'Unnamed: 0'，并且你想查值为 38054 的那一行
print("========== 【按 ID 列查找 (ID: 38054)】 ==========")
# 假设你的主键列名叫 'id'，如果叫别的请自行修改
if 'id' in df.columns:
    print(df[df['id'] == 38054] if pd.api.types.is_numeric_dtype(df['id']) else df[df['id'] == '38054'])
else:
    print("你的 CSV 里没有名为 'id' 的列。")