import os
import pandas as pd
import chromadb
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ==========================================
# 常量配置区
# ==========================================
CSV_DATA_PATH = "../data/data_set/pkulaw_combined_articles_chunked.csv"
LOCAL_MODEL_PATH = "../models/bge-m3/Xorbits/bge-m3"
CHROMA_DB_PATH = "../data/chroma_local_db"
COLLECTION_NAME = "pkulaw_knowledge_base"

# 显存优化配置 (RTX 3060 6GB)
BATCH_SIZE_INFERENCE = 64  # GPU推理批大小
WRITE_CHUNK_SIZE = 1000  # 断点检查的步长，建议设小一点方便精确续传


def main():
    # 1. 检查 GPU 状态 (这一步非常关键，请确保看到的是 CUDA)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 [1/5] 检测到运行设备: {device.upper()}")
    if device == "cpu":
        print("⚠️ 警告：当前正在使用 CPU，速度会非常慢。建议检查 PyTorch 环境。")

    # 2. 加载数据集
    print(f"📅 [2/5] 正在加载数据集: {CSV_DATA_PATH}")
    try:
        df = pd.read_csv(CSV_DATA_PATH)
        text_column = 'content' if 'content' in df.columns else df.columns[0]
        # 预处理元数据为字符串
        df_meta = df.fillna("").astype(str)
        print(f"✅ 成功加载 {len(df)} 条原始数据。")
    except Exception as e:
        print(f"❌ 读取 CSV 失败: {e}")
        return

    # 3. 加载模型
    print(f"🧠 [3/5] 加载本地 Embedding 模型...")
    model = SentenceTransformer(LOCAL_MODEL_PATH, device=device)
    model.max_seq_length = 512
    print("✅ 模型加载完成。")

    # 4. 初始化 ChromaDB (支持断点续传的关键)
    print(f"🗄️ [4/5] 初始化本地 ChromaDB: {CHROMA_DB_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # 使用 get_or_create 而不是 delete，防止覆盖已有数据
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # 核心：检查已有数据量
    existing_count = collection.count()
    print(f"📊 数据库当前已有记录数: {existing_count}")

    # 5. 分批推理并入库
    print(f"\n⏳ [5/5] 开始处理数据 (自动跳过已存在记录)...")

    total_len = len(df)

    # 使用 tqdm 进度条查看总进度
    for i in range(0, total_len, WRITE_CHUNK_SIZE):
        # 断点续传逻辑：如果当前索引小于已有记录数，直接跳过
        if i + WRITE_CHUNK_SIZE <= existing_count:
            continue

        end_idx = min(i + WRITE_CHUNK_SIZE, total_len)
        batch_df = df.iloc[i:end_idx]
        batch_meta_df = df_meta.iloc[i:end_idx]

        # 准备数据
        documents = batch_df[text_column].fillna("").tolist()
        ids = [str(idx) for idx in batch_df.index]
        metadatas = batch_meta_df.to_dict(orient='records')

        # 生成向量
        embeddings = model.encode(
            documents,
            batch_size=BATCH_SIZE_INFERENCE,
            show_progress_bar=False,  # 外部已有大进度条，这里关掉
            convert_to_numpy=True
        ).tolist()

        # 写入数据库
        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )

        # 实时打印进度
        print(f"   📥 已成功处理并存入: {end_idx} / {total_len}")

        # 定期清理显存
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\n🎉 向量化全部完成！集合 '{COLLECTION_NAME}' 中总条数: {collection.count()}")


if __name__ == "__main__":
    main()