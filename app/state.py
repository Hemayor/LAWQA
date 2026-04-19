from typing import Optional, List, Dict, Any
from typing_extensions import TypedDict


# ==========================================
#定义 Agent 的增强状态 (Enhanced State)
# ==========================================
class AgentState(TypedDict):
    """
    定义我们 Agent 计算图的全局状态。
    这是整个 Agentic RAG 的中枢神经系统，记录了从输入到输出的所有认知过程。
    """
    original_request: str                     # 用户最原始的查询/请求
    clarification_question: Optional[str]     # ambiguity_judgment节点如果觉得问题模糊，生成的反问/澄清问题
    rewrite_request: Optional[str]            # 存储改写工具生成的标准关键词
    plan: List[str]                           # Planner节点制定的分步执行计划
    intermediate_steps: List[Dict[str, Any]]  # 工具执行节点(Executor)记录的每次工具调用的输入和输出结果
    verification_history: List[Dict[str, Any]]# Reflector节点记录的自我检查/验证历史得分
    final_response: str                       # Generator节点最终合成的回答
    loop_count: int                           # 记录系统打回重做的次数
    timing_stats: Dict[str, float]            # 各工具的累计执行时间（秒）：{"rewrite": 0.5, "retrieval": 1.2, "generation": 0.8}
    is_complex: bool