import streamlit as st
import torch
from streamlit_chat import message
from entity import extract_entities
from langchain_glm import ask_question
from intent_detection_bert import predict_intent
def on_input_change():
    user_input = st.session_state.user_input
    st.session_state.past.append(user_input)
    st.session_state.generated.append("感谢您的提问，我会尽力回答。")

def on_btn_click():
    del st.session_state.past[:]
    del st.session_state.generated[:]

# 初始化对话状态
st.session_state.setdefault('past', [])
st.session_state.setdefault('generated', [])

# 设置Streamlit应用界面
st.sidebar.title("功能列表")

# 左侧：功能选择
option = st.sidebar.selectbox("选择模型或解释", ["问答系统", "问题重述","实体识别", "意图识别"])

if option == "问答系统":
    st.header("问答系统")

    chat_placeholder = st.empty()

    with chat_placeholder.container():
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=f"{i}_user")
            message(st.session_state['generated'][i], key=f"{i}")

        st.button("清除对话", on_click=on_btn_click)

    with st.container():
        st.text_input("请输入您的问题:", on_change=on_input_change, key="user_input")

elif option == "实体识别":
    st.header("实体识别")
    query = st.text_input("请输入您的问题:", key="entity_input")
    if st.button("确认", key="entity_confirm"):
        if query.strip() == "":
            st.warning("请输入内容后再点击确认。")
        else:
            res = []
            res.append(query)
            answer = extract_entities(res)
            st.write("回答：", answer)

elif option == "意图识别":
    st.header("意图识别")
    query = st.text_input("请输入您的问题:", key="intent_input")
    if st.button("确认", key="intent_confirm"):
        if query.strip() == "":
            st.warning("请输入内容后再点击确认。")
        else:
            save_path = "saved_model"  # 模型保存路径
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            max_len = 128
            intent = predict_intent(query, save_path, device, max_len)
            st.write("回答：", intent)

elif option == "问题重述":
    st.header("问题重述")
    query = st.text_input("请输入您的问题:", key="rewrite_input")
    if st.button("确认", key="rewrite_confirm"):
        if query.strip() == "":
            st.warning("请输入内容后再点击确认。")
        else:
            answer = ask_question(query)
            st.write("回答：", answer)