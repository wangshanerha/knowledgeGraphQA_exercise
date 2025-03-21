from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os

def get_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

if __name__ == '__main__':
    # text = "介绍一下你自己"
    # 1. 调用api
    model = ChatOpenAI(
        temperature=0.95,
        model="glm-4-flash",
        openai_api_key="f037f80bb31908ef8d2159b5e4f17e6d.kDBaHZjpplHWU6pl",
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
    )

    # 2. 创建提示模板
    prompt_template = ChatPromptTemplate([
        ("system", "你现在是个ai助手."),
        MessagesPlaceholder(variable_name='haha')
    ])

    # 3 创建数据响应器
    parser = StrOutputParser()

    # 4. 创建Chain
    chain = (
            prompt_template
            | model
            | parser
    )

    # 5 保存聊天历史记录
    store = {}  # key: session_id ,   value : 历史聊天对象
    do_message = RunnableWithMessageHistory(
        chain,
        get_history,  # 直接传递函数
        input_messages_key='haha' , # 每次聊天发送的key
    )
        # 第一轮
    config = {"configurable": {"session_id": "1"}}
    res1 = do_message.invoke(
        {
            'haha': [HumanMessage(content="你好，请记住我的名字是王山而")],
        },
        config=config
    )
    print(res1)
        # 第二轮
    res2 = do_message.invoke(
        {
            'haha': [HumanMessage(content="你好，我的名字是什么")],
        },
        config=config
    )
    print(res2)
    #     # 第三轮：流式输出
    # for res in do_message.stream(
    #     {
    #         'haha': [HumanMessage(content="你好，给我讲个笑话")],
    #     },
    #     config=config
    # ):
    #     # 每次的res都是一个token
    #     print(res,end='-')