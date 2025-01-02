from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from zhipuai import ZhipuAI
from langchain_community.chat_models import ChatZhipuAI
import  os
from langchain.agents import tool
import re

LANGCHAIN_TRACING_V2=True
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="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"
LANGCHAIN_PROJECT="example"

os.environ["LANGCHAIN_TRACING_V2"] = "1"  # 规定langchain的版本
os.environ["LANGCHAIN_API_KEY"] = "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"

def ask_question(question):
    # 1. 调用api
    # zhipuai_chat = ChatZhipuAI(
    #     temperature=0.5,
    #     api_key="f037f80bb31908ef8d2159b5e4f17e6d.kDBaHZjpplHWU6pl",
    #     model_name="glm-4-flash",
    #
    # )
    model = ChatOpenAI(
        temperature=0.95,
        model="glm-4-flash",
        openai_api_key="f037f80bb31908ef8d2159b5e4f17e6d.kDBaHZjpplHWU6pl",
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
    )

    # 2. 创建提示模板
    # question = "你好我的朋友，请介绍一下你自己"
    prompt_template = ChatPromptTemplate.from_template("{input}")

    # 3. 创建Chain
    chain = (
            prompt_template
            | model
    )

    # 4. 运行Chain并获取回答
    response = chain.invoke({"input": question})

    # 5. 提取并打印模型的回答内容
    model_answer = response.content
    print(model_answer)

if __name__ == '__main__':
    question = input()
    ask_question(question)

