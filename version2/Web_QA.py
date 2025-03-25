import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from entity import extract_entities
from Restatement import *
from intent_detection import predict
from langchain_core.runnables.history import RunnableWithMessageHistory
from Web_KGshow import *
from RAG import *




def RAG_data_perpare():
    pdf_path = "../doc/国家基层糖尿病防治管理指南（ 2022）.pdf"

    # 嵌入模型加载
    embedding_model = SentenceTransformer("../../../nlp/doc/model/bce-embedding-base_v1")  # 用于生成嵌入向量

    # 1. 执行一次 RAG_prepare 来准备 PDF 内容和知识库
    pdf_content, knowledge_base = RAG_prepare(pdf_path, embedding_model)
    # st.write(f"PDF 提取文本完成，共 {len(pdf_content)} 页")
    # st.write(f"向量知识库构建完成，共包含 {len(knowledge_base)} 条知识项")

    return embedding_model,pdf_content,knowledge_base

def Comprehensive_question(query,embedding_model,pdf_content,knowledge_base):
    # 实体识别
    res = [query]
    answer = extract_entities(res)
    entities = []
    for entity in answer[0]:
        entities.append(entity[0])

    # # 意图识别
    model_path = "D:/pycharm/example/nlp/graduate/version2/chinese-electra-large-Diabetes-question-intent"
    intent = predict(query, model_path)

    # 图谱查询
    graph = KG_load()
    KG_results = neo4j_query(graph, entities, disease_relations[intent[:2]])# 意图截一下


    # RAG

    top_results = retrieve_answer(query, knowledge_base, embedding_model, top_k=3)
    rag_res = []
    for i, (score, entry) in enumerate(top_results):
        rag_res.append(entry['content'])

    # LLM 润色
    prompt = '''
               你好，我需要你的帮助。请扮演一名资深的糖尿病专家，现在病人对你提问，而你要根据我提供的信息回答问题。
               以下是病人的问题：
               %s
               此外，我还查询了专业的知识图谱，所以可以给你了以下提示信息（是一个字典，内部包含一下参考知识）：
               %s
               此外，我还查询了专门的文档：%s
               请你结合以上问句和提示，基于糖尿病领域的专业知识，为病人提供一个全面、详细、准确且易于理解的回答，可以适当使用我提示你的参考知识，但不能局限于我的参考知识。
                ''' % (query, KG_results, rag_res)

    return entities,intent,KG_results,top_results,rag_res,prompt,ask_question(prompt)


# 定义疾病和药物关系映射
disease_relations = {
    "A0": ["Symptom_Disease", "Test_Disease"],
    "A1": ["Symptom_Disease","Anatomy_Disease","Reason_Disease","Pathogenesis_Disease"],
    "A2": ["Symptom_Disease","Reason_Disease","Pathogenesis_Disease","Class_Disease"],
    "A3": ["Test_Disease","Treatment_Disease","Drug_Disease","Test_Items_Disease"],
    "B0": ["Duration_Drug","Test_Items_Disease"],
    "B1": ["Frequency_Drug","Duration_Drug","Amount_Drug","Method_Drug","ADE_Drug"],
    "B2": ["Drug_Disease", "Treatment_Disease"],
    "B3": ["Amount_Drug", "ADE_Drug","Method_Drug"],
    "B4": ["ADE_Drug"],
    "B5": ["Treatment_Disease","Operation_Disese"],
    "B6": ["Operation_Disease","Drug_Disease","Treatment_Disease","Test_Items_Disease"],
    "C0": ["Symptom_Disease", "Drug_Disease"],
    "C1": ["Class_Disease"],
    "C2": ["Reason_Disease","Pathogenesis_Disease"],
    "C3": ["Anatomy_Disease","Reason_Disease","Pathogenesis_Disease"],
    "C4": ["Pathogenesis_Disease","Reason_Disease"],
    "D0": ["Pathogenesis_Disease", "Reason_Disease"],
    "D1": ["Pathogenesis_Disease", "Reason_Disease"],
    "D2": ["Pathogenesis_Disease", "Reason_Disease"],
    "D3": ["Pathogenesis_Disease", "Reason_Disease"],
    "E1":  ["Pathogenesis_Disease", "Reason_Disease","Symptom_Disease"],
    "E2":  ["Pathogenesis_Disease", "Reason_Disease","Symptom_Disease"],
    "E3":  ["Pathogenesis_Disease", "Reason_Disease","Symptom_Disease"],
}


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
option = st.sidebar.selectbox("选择模型或解释", ["问答系统","流程演示", "问题重述", "实体识别", "意图识别", "RAG","chatglm", "deepseek"])

if option == "流程演示":
    query = st.text_input("请输入问题",key="query_input")
    if st.button("确认", key="entity_confirm"):
        if query.strip() == "":
            st.warning("请输入内容后再点击确认。")
        else:
            embedding_model,pdf_content, knowledge_base = RAG_data_perpare()  # RAG 的准备工作
            entities,intent,KG_results,top_results,rag_res,prompt,ask_question_results = Comprehensive_question(query,embedding_model,pdf_content,knowledge_base)
            # 实体识别
            st.write("识别到的实体为：")
            st.write(entities)

            # # 意图识别
            st.write("识别到的意图为：")
            st.write(intent)

            # 图谱查询
            st.write("知识图谱查询结果为：")
            st.write(KG_results)

            # RAG
            st.write(f"PDF 提取文本完成，共 {len(pdf_content)} 页")
            st.write(f"向量知识库构建完成，共包含 {len(knowledge_base)} 条知识项")

            print("\n检索结果（按相似度排序）：")
            st.write(f"从向量知识库中提取一下相似信息 {rag_res} ")

            # LLM 润色
            st.write("原始问句回答结果")
            st.write(ask_question(query))
            st.write("构建的提示词为：")
            st.write(prompt)
            st.write("提示后结果为：")
            st.write(ask_question_results)

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
            answer = Restatement_problem(query)
            st.write(answer)

elif option == "RAG":
    st.header("RAG演示")
    # 加载 PDF 文档并提取文本
    pdf_path = "../doc/国家基层糖尿病防治管理指南（ 2022）.pdf"

    # 嵌入模型加载
    embedding_model = SentenceTransformer("../../../nlp/doc/model/bce-embedding-base_v1")  # 用于生成嵌入向量

    # 1. 执行一次 RAG_prepare 来准备 PDF 内容和知识库
    pdf_content, knowledge_base = RAG_prepare(pdf_path, embedding_model)
    st.write(f"PDF 提取文本完成，共 {len(pdf_content)} 页")
    st.write(f"知识库构建完成，共包含 {len(knowledge_base)} 条知识项")

    query = st.text_input("请输入您的问题:", key="rewrite_input")
    if st.button("确认", key="rewrite_confirm"):
        if query.strip() == "":
            st.warning("请输入内容后再点击确认。")
        else:
            top_results = retrieve_answer(query, knowledge_base, embedding_model, top_k=3)

            print("\n检索结果（按相似度排序）：")
            for i, (score, entry) in enumerate(top_results):
                st.write(f"结果 {i + 1}:")
                st.write(f"相似度分数：{score:.4f}")
                st.write(f"所在页码：{entry['page_number']}")
                st.write(f"内容：{entry['content']}")
                st.write(f"所在页的所有内容：{pdf_content[entry['page_number'] - 1]['content']}")

if option == "问答系统":
    st.title("糖尿病智能问答系统")
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

    # 初始化RAG数据（只执行一次）
    if "rag_initialized" not in st.session_state:
        with st.spinner("正在初始化知识库..."):
            st.session_state.embedding_model, st.session_state.pdf_content, st.session_state.knowledge_base = RAG_data_perpare()
            st.session_state.rag_initialized = True

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

            try:
                # 执行综合处理流程
                with st.spinner("正在分析问题..."):
                    entities, intent, KG_results, top_results, rag_res, final_prompt, _ = Comprehensive_question(
                        prompt,
                        st.session_state.embedding_model,
                        st.session_state.pdf_content,
                        st.session_state.knowledge_base
                    )

                # 流式输出处理
                for chunk in st.session_state.conversation.stream(
                        {"input": final_prompt},  # 使用生成的最终提示词
                        config={"configurable": {"session_id": "default"}}
                ):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)
                st.session_state.chat_history.append(AIMessage(content=full_response))

                # 显示详细信息（可折叠）
                with st.expander("查看分析细节"):
                    st.subheader("实体识别结果")
                    st.write(entities)

                    st.subheader("意图识别结果")
                    st.write(intent)

                    st.subheader("知识图谱查询结果")
                    st.write(KG_results)

                    st.subheader("相关文档片段")
                    for i, content in enumerate(rag_res[:3]):  # 显示前3个相关结果
                        st.write(f"相关段落 {i + 1}:")
                        st.write(content[:200] + "...")  # 显示前200字符

            except Exception as e:
                st.error(f"生成回答时出错: {str(e)}")