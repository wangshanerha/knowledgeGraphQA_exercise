import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


# 初始化全局配置
def init_settings():
    st.set_page_config(
        page_title="GLM-4 Chatbot",
        page_icon="🤖",
        layout="centered"
    )
    if "messages" not in st.session_state:
        st.session_state.messages = []


# 侧边栏配置
def sidebar_config():
    with st.sidebar:
        st.title("🔧 配置中心")
        st.subheader("API 设置")
        api_key = st.text_input(
            "API 密钥",
            type="password",
            value="f037f80bb31908ef8d2159b5e4f17e6d.kDBaHZjpplHWU6pl"
        )
        api_base = st.text_input(
            "API 端点",
            value="https://open.bigmodel.cn/api/paas/v4/"
        )

        st.divider()
        st.subheader("模型参数")
        temperature = st.slider(
            "温度系数 (0-2)",
            min_value=0.0,
            max_value=2.0,
            value=0.95,
            step=0.1
        )
        model_name = st.selectbox(
            "模型选择",
            ("glm-4-flash", "glm-4", "glm-3-turbo"),
            index=0
        )

    return {
        "api_key": api_key,
        "api_base": api_base,
        "temperature": temperature,
        "model_name": model_name
    }


# 初始化对话链
@st.cache_resource
def init_chain(config):
    # 1. 初始化大模型
    llm = ChatOpenAI(
        temperature=config["temperature"],
        model=config["model_name"],
        openai_api_key=config["api_key"],
        openai_api_base=config["api_base"]
    )

    # 2. 构建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业且友好的AI助手"),
        MessagesPlaceholder(variable_name="chat_history"),
    ])

    # 3. 构建处理链
    chain = prompt | llm | StrOutputParser()

    # 4. 配置历史记录存储
    store = {}

    def get_history(session_id: str):
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    return RunnableWithMessageHistory(
        chain,
        get_history,
        input_messages_key="chat_history",
        history_messages_key="chat_history"
    )


# 流式输出处理
def stream_response(prompt, chain):
    with st.chat_message("user"):
        st.markdown(prompt)

    response_container = st.empty()
    full_response = ""

    # 流式调用大模型
    for chunk in chain.stream(
            {"chat_history": [HumanMessage(content=prompt)]},
            config={"configurable": {"session_id": "default"}}
    ):
        full_response += chunk
        response_container.markdown(full_response + "▌")

    response_container.markdown(full_response)
    return full_response


# 主界面
def main():
    init_settings()
    config = sidebar_config()

    # 初始化对话链
    try:
        chain = init_chain(config)
    except Exception as e:
        st.error(f"初始化失败: {str(e)}")
        return

    # 历史记录显示
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 用户输入处理
    if prompt := st.chat_input("输入您的问题..."):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 生成AI响应
        try:
            ai_response = stream_response(prompt, chain)
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
        except Exception as e:
            st.error(f"生成响应时出错: {str(e)}")


if __name__ == "__main__":
    main()