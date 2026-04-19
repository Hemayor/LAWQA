import pandas as pd
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings

# 忽略分词时的警告
warnings.filterwarnings("ignore")

# =============================
# 1. 全局字体与样式配置 (支持中文显示)
# =============================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# =============================
# 2. 文件路径配置
# =============================
csv_before_path = "../data/data_set/pkulaw_combined_articles.csv"
csv_after_path = "../data/data_set/pkulaw_combined_articles_chunked.csv"
output_dir = "../data/data_set/"
output_image_path = os.path.join(output_dir, "token_distribution_comparison.png")

# =============================
# 3. 初始化 tokenizer
# =============================
print("⏳ 正在加载 BGE-M3 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")


# =============================
# 4. 核心统计函数
# =============================
def get_token_stats(csv_path, desc):
    print(f"📂 正在处理【{desc}】数据: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"❌ 找不到文件: {csv_path}")
        return None, None

    df = pd.read_csv(csv_path)
    if "content" not in df.columns:
        print(f"❌ CSV 中缺少 'content' 列！")
        return None, None

    token_lengths = [len(tokenizer.encode(str(content), add_special_tokens=False)) for content in df["content"]]
    token_lengths = np.array(token_lengths)

    stats = {
        "总文档数": len(token_lengths),
        "平均 Token": token_lengths.mean(),
        "最短 Token": token_lengths.min(),
        "95%分位数": np.percentile(token_lengths, 95),
        "最长 Token": token_lengths.max()
    }

    print(f"--- 【{desc}】统计结果 ---")
    for k, v in stats.items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")
    print("-" * 30)

    return token_lengths, stats


# =============================
# 5. 读取并统计两侧数据
# =============================
lengths_before, stats_before = get_token_stats(csv_before_path, "切分前")
lengths_after, stats_after = get_token_stats(csv_after_path, "切分后")

if lengths_before is None or lengths_after is None:
    exit("⚠️ 数据读取失败，请检查路径。")

# =============================
# 6. 绘制并排对比直方图
# =============================
print("📊 正在生成对比分析图...")
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("知识库切片 (Chunking) 前后 Token 分布对比分析", fontsize=20, fontweight='bold', y=1.02)


def plot_histogram(ax, lengths, stats, title, color):
    max_len = lengths.max()
    step = max(50, int(max_len / 30))
    bins = np.arange(0, max_len + step, step)

    counts, edges, patches = ax.hist(lengths, bins=bins, edgecolor='black', color=color, alpha=0.8)

    # 🌟 核心修改：加大字号，改变颜色，并加粗字体
    for rect, count in zip(patches, counts):
        height = rect.get_height()
        if height > 0:
            ax.text(rect.get_x() + rect.get_width() / 2, height + (height * 0.01),
                    str(int(count)), ha='center', va='bottom',
                    fontsize=11, color='darkred', fontweight='bold')

    # 扩大Y轴上限，给粗大的数字留出足够的顶部空间
    ax.set_ylim(0, max(counts) * 1.15)

    ax.set_title(title, fontsize=16, pad=15)
    ax.set_xlabel("单段文本包含的 Token 数量", fontsize=14)
    ax.set_ylabel("文档/切片 数量 (频次)", fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    ax.axvline(stats['95%分位数'], color='red', linestyle='dashed', linewidth=2,
               label=f"95% 分位数 ({stats['95%分位数']:.0f})")
    ax.legend(fontsize=12, loc='upper right')

    stats_text = (
        f"数据总量: {stats['总文档数']}\n"
        f"平均长度: {stats['平均 Token']:.1f}\n"
        f"最大长度: {stats['最长 Token']}\n"
        f"最小长度: {stats['最短 Token']}\n"
        f"95%分位数: {stats['95%分位数']:.0f}"
    )
    ax.text(0.95, 0.75, stats_text, transform=ax.transAxes,
            fontsize=13, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="gray", alpha=0.9))


# 绘制左图：切分前
plot_histogram(axes[0], lengths_before, stats_before, "【切分前】原始知识库分布", color='#5DADE2')

# 绘制右图：切分后
plot_histogram(axes[1], lengths_after, stats_after, "【切分后】Chunk 知识库分布", color='#48C9B0')

# =============================
# 7. 调整排版并保存
# =============================
plt.tight_layout()
plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
print(f"🎉 成功！对比分析图已保存至: {output_image_path}")

plt.show()