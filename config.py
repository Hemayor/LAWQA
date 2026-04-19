# app/config.py
import os
from dotenv import load_dotenv

# 加载可能存在于 .env 文件中的配置
load_dotenv()

# ==========================================
# 1. 核心循环与熔断配置 (Agent Control)
# ==========================================
MAX_LOOPS = int(os.getenv("MAX_LOOPS", 3))       # 整个 Agent 图谱最大允许的重试/循环轮次
PASSING_SCORE = int(os.getenv("PASSING_SCORE", 3)) # 审计节点 (Auditor) 的及格线，低于此分打回

# ==========================================
# 2. 向量检索配置 (Retrieval)
# ==========================================
RECALL_LIMIT = int(os.getenv("RECALL_LIMIT", 50)) # 粗排最大召回数
RERANK_LIMIT = int(os.getenv("RERANK_LIMIT", 5))  # 精排最终保留数

# ==========================================
# 3. 大模型参数 (LLM Settings)
# ==========================================
PLANNER_TEMP = float(os.getenv("PLANNER_TEMP", 0.0))    # 规划节点的温度 (要求严谨)
GENERATOR_TEMP = float(os.getenv("GENERATOR_TEMP", 0.3))# 生成节点的温度 (允许轻微创造力)


# 是否开启自适应复杂度路由
ENABLE_COMPLEXITY_ROUTING = True