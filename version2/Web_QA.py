import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from entity import extract_entities
from Restatement import ask_question
from intent_detection import predict
from langchain_core.runnables.history import RunnableWithMessageHistory


# 初始化会话状态
def init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "model_initialized" not in st.session_state:
        st.session_state.model_initialized = False


# 自定义历史记录处理
def get_custom_history(session_id: str):
    if session_id not in st.session_state:
        st.session_state[session_id] = ChatMessageHistory()
    return st.session_state[session_id]


# 初始化模型链
def init_model_chain():
    return (
            ChatPromptTemplate.from_messages([
                ("system", "你是一个专业的人工智能助手"),
                MessagesPlaceholder(variable_name='chat_history'),
                ("human", "{input}")
            ])
            | ChatOpenAI(
        temperature=st.session_state.temperature,
        model="glm-4-flash",
        openai_api_key=st.session_state.api_key,
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
    )
            | StrOutputParser()
    )


# 设置Streamlit应用界面
st.sidebar.title("功能列表")
option = st.sidebar.selectbox("选择模型或解释", ["问答系统", "问题重述", "实体识别", "意图识别", "RAG","chatglm", "deepseek"])

# chatglm聊天
if option == "chatglm":
    # 侧边栏统一配置
    with st.sidebar:
        # st.subheader("通用配置")
        # api_key = st.text_input("API Key", type="password", key="api_key_input")
        # if api_key:
        #     st.session_state.api_key = api_key
        #     st.success("API Key已设置")

        st.session_state.temperature = st.slider(
            "Temperature", 0.0, 1.0, 0.7,
            help="控制生成文本的随机性，值越大输出越随机（0=确定性，1=高随机性）"
        )
    st.title("ChatGLM 智能问答系统")
    api_key = "f037f80bb31908ef8d2159b5e4f17e6d.kDBaHZjpplHWU6pl"
    st.session_state.api_key = api_key
    init_session_state()

    # 模型初始化
    if "api_key" in st.session_state and not st.session_state.model_initialized:
        with st.spinner("正在初始化模型..."):
            st.session_state.chain = init_model_chain()
            st.session_state.conversation = RunnableWithMessageHistory(
                st.session_state.chain,
                get_custom_history,
                input_messages_key="input",
                history_messages_key="chat_history"
            )
            st.session_state.model_initialized = True

    # 显示历史对话
    for msg in st.session_state.chat_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    # 处理用户输入
    if prompt := st.chat_input("请输入您的问题"):
        # 添加用户消息
        st.session_state.chat_history.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        # 生成回答
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            # 流式输出处理
            try:
                for chunk in st.session_state.conversation.stream(
                        {"input": prompt},
                        config={"configurable": {"session_id": "default"}}
                ):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)
                st.session_state.chat_history.append(AIMessage(content=full_response))
            except Exception as e:
                st.error(f"生成回答时出错: {str(e)}")

elif option == "实体识别":
    st.header("实体识别")
    query = st.text_input("请输入您的问题:", key="entity_input")
    if st.button("确认", key="entity_confirm"):
        if query.strip() == "":
            st.warning("请输入内容后再点击确认。")
        else:
            res = [query]
            answer = extract_entities(res)
            st.write("回答：", answer)

elif option == "意图识别":
    st.header("意图识别")
    query = st.text_input("请输入您的问题:", key="intent_input")
    if st.button("确认", key="intent_confirm"):
        if query.strip() == "":
            st.warning("请输入内容后再点击确认。")
        else:
            model_path = "D:/pycharm/example/nlp/graduate/version2/chinese-electra-large-Diabetes-question-intent"
            intent = predict(query, model_path)
            st.write("回答：", intent)

elif option == "问题重述":
    st.header("问题重述")
    query = st.text_input("请输入您的问题:", key="rewrite_input")
    if st.button("确认", key="rewrite_confirm"):
        if query.strip() == "":
            st.warning("请输入内容后再点击确认。")
        else:
            answer = ask_question(query)
            st.write(answer)