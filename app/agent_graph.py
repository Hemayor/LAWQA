# app/agent_graph.py

# 1. 导入三大专门工具
from tool_query_rewrite import query_rewrite_tool
from tool_retrieve_rerank import retrieve_and_rerank_tool
from tool_scout_search import scout_web_search_tool

# 2. 注册工具箱并创建映射字典
tools = [scout_web_search_tool,query_rewrite_tool, retrieve_and_rerank_tool]
# tools = [scout_web_search_tool,retrieve_and_rerank_tool]

tool_map = {tool.name: tool for tool in tools}

# 测试入口
if __name__ == "__main__":
    print("\n🛠️ --- 当前 Agent 可用的工具清单 ---")
    for tool in tools:
        print(f"- Tool: {tool.name}")
        print(f"  Description: {tool.description.strip()}\n")