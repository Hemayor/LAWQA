import json
import os

# ==========================================
# 1. 配置文件路径
# ==========================================
DATASET_FILE = "../data/test_set/legal_test_set_final.jsonl"


def check_dataset():
    print(f"🕵️ 开始全面体检数据集: {DATASET_FILE}\n" + "=" * 50)

    if not os.path.exists(DATASET_FILE):
        print(f"❌ 找不到文件: {DATASET_FILE}")
        return

    total_physical_lines = 0  # 物理总行数（包括空行）
    empty_lines = []  # 空行所在的行号
    json_error_lines = []  # JSON解析错误的行号

    seen_ids = {}  # 用于记录已出现的 ID 及其对应的行号 { "id": line_number }
    duplicate_reports = []  # 记录重复 ID 的详细信息

    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            total_physical_lines += 1

            # 检查空行
            if not line.strip():
                empty_lines.append(line_num)
                continue

            # 尝试解析 JSON
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                json_error_lines.append((line_num, str(e)))
                continue

            # 提取并检查 ID
            q_id = str(data.get('id', 'MISSING_ID')).strip()

            if q_id in seen_ids:
                # 发现内鬼！记录：当前 ID、第一次出现的行号、第二次出现的行号
                duplicate_reports.append({
                    "id": q_id,
                    "first_seen": seen_ids[q_id],
                    "duplicate_at": line_num
                })
            else:
                seen_ids[q_id] = line_num

    # ==========================================
    # 输出体检报告
    # ==========================================
    print("📋 【数据集体检报告】\n")
    print(f"📄 物理总行数: {total_physical_lines} 行")
    print(f"✅ 有效且独立的 ID 数量: {len(seen_ids)} 个\n")

    # 1. 汇报空行
    if empty_lines:
        print(f"⚠️ 发现 {len(empty_lines)} 个隐藏的空行，分别在第 {empty_lines} 行。")
    else:
        print("✔️ 没有发现隐藏空行。")

    # 2. 汇报 JSON 解析错误
    if json_error_lines:
        print(f"\n❌ 发现 {len(json_error_lines)} 处 JSON 格式错误:")
        for ln, err in json_error_lines:
            print(f"   - 第 {ln} 行: {err}")
    else:
        print("✔️ 所有数据行均符合严格的 JSON 格式。")

    # 3. 汇报重复 ID (高亮重点)
    if duplicate_reports:
        print(f"\n🚨 警报！发现 {len(duplicate_reports)} 个重复的 ID！")
        for report in duplicate_reports:
            print(
                f"   -> 抓到内鬼 ID: [{report['id']}] | 第一次出现在第 {report['first_seen']} 行，又在第 {report['duplicate_at']} 行重复出现了！")
        print("\n💡 建议: 请打开 JSONL 文件，直接跳转到对应的行号修改或删除重复数据。")
    else:
        print("\n✔️ 完美！没有发现任何重复的 ID。")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    check_dataset()