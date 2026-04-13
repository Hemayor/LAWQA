import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 1. 加载环境变量 (自动读取项目根目录下的 .env 文件)
load_dotenv()


class LLMFactory:

    @staticmethod
    def _get_env_var(key_name: str) -> str:
        """内部辅助方法：安全获取环境变量，拿不到就报错提示"""
        value = os.getenv(key_name)
        if not value:
            raise ValueError(f"❌ 致命错误: 未在环境变量或 .env 文件中找到 {key_name}！")
        return value

    @staticmethod
    def get_gpt():
        return ChatOpenAI(
            model="gpt-4o-mini",
            # 动态读取，不再硬编码
            api_key=LLMFactory._get_env_var("LINO_API_KEY"),
            base_url="https://linoapi.com.cn/v1",
            temperature=0.0
        )

    @staticmethod
    def get_deepseek():
        return ChatOpenAI(
            model="deepseek-chat",
            api_key=LLMFactory._get_env_var("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
            temperature=0.0
        )

    @staticmethod
    def get_deepseek_reasoner():
        return ChatOpenAI(
            model="deepseek-reasoner",
            api_key=LLMFactory._get_env_var("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
            temperature=0.0
        )


# ==========================================
# 本地测试入口
# ==========================================
def test_models():
    print("🚀 开始 LLM 配置测试...\n")

    # 1. 测试 GPT
    print("--- [测试 GPT-4o-mini] ---")
    try:
        gpt = LLMFactory.get_gpt()
        res_gpt = gpt.invoke("你好，请确认你已连接成功。")
        print(f"✅ GPT 响应成功: {res_gpt.content}\n")
    except Exception as e:
        print(f"❌ GPT 连接失败: {e}\n")

    # 2. 测试 DeepSeek
    print("--- [测试 DeepSeek-V3] ---")
    try:
        ds = LLMFactory.get_deepseek()
        res_ds = ds.invoke("你好，请确认你已连接成功。")
        print(f"✅ DeepSeek 响应成功: {res_ds.content}\n")
    except Exception as e:
        print(f"❌ DeepSeek 连接失败: {e}\n")

    # 3. 测试 DeepSeek-Reasoner
    print("--- [测试 DeepSeek-Reasoner] ---")
    try:
        ds_r = LLMFactory.get_deepseek_reasoner()
        res_ds_r = ds_r.invoke("你好，请确认你已连接成功。")
        print(f"✅ DeepSeek Reasoner 响应成功: {res_ds_r.content}\n")
    except Exception as e:
        print(f"❌ DeepSeek Reasoner 连接失败: {e}\n")


if __name__ == "__main__":
    test_models()