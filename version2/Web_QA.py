import streamlit as st
from web_function import  ask

# 假设我们有两个模型的选择（可以替换为实际的模型加载逻辑）
def model_1_answer(query):
    answer = ask(query)
    # 模型1的回答逻辑（替换为实际模型）
    return f"模型1的回答：{(answer)}的答案是A。"

def model_2_answer(query):
    # 模型2的回答逻辑（替换为实际模型）
    return f"模型2的回答：{query}的答案是B。"

# 设置Streamlit应用界面
st.title("问答系统")
st.sidebar.title("功能列表")

# 左侧：模型选择
model_option = st.sidebar.selectbox("选择模型或解释", ["模型1", "模型2", "解释"])

# 右侧：问答框
query = st.text_input("请输入您的问题:")

# 显示答案
if query:
    if model_option == "模型1":
        answer = model_1_answer(query)
    elif model_option == "模型2":
        answer = model_2_answer(query)
    elif model_option == "解释":
        answer = "这是一个示例解释。如果您有具体问题，选择模型进行回答。"

    st.write("回答：", answer)