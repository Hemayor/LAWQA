import json
import os

# ==========================================
# 1. 配置文件路径
# ==========================================
# 假设你要处理的是这个文件，你可以根据实际情况修改
INPUT_FILE = "../data/test_set/legal_test_set_final.jsonl"
OUTPUT_FILE = "../data/test_set/web_renumbered.jsonl"


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到 JSONL 文件: {INPUT_FILE}")
        return

    print("🛠️ 正在重新编排行号...")

    processed_count = 0

    with open(INPUT_FILE, 'r', encoding='utf-8') as fin, \
            open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:

        for line in fin:
            if not line.strip():
                continue

            data = json.loads(line)

            # 生成新的 ID，自增并自动补齐 3 位数（001, 002, 003...）
            # 如果你的题目超过 1000 道，可以改成 :04d (0001)
            processed_count += 1
            new_id = f"{processed_count:03d}"

            # 替换旧的 ID
            data['id'] = new_id

            # 将更新后的数据写回新文件
            fout.write(json.dumps(data, ensure_ascii=False) + '\n')

    print("-" * 50)
    print(f"🎉 处理完成！共为 {processed_count} 道题目重新分配了连续的 ID。")
    print(f"💾 新的测试集已保存至: {OUTPUT_FILE}")
    print("💡 检查无误后，可以把旧的 web.jsonl 删掉，把这个新文件改名替换上去！")


if __name__ == "__main__":
    main()