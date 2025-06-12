from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder
from function.model_train.NER.Entity_Qwen2_5_predict import *
from Restatement import *
from function.model_train.TC.Intent_Recognition_BERT_CNN_predict import *
from langchain_core.runnables.history import RunnableWithMessageHistory
from Web_KGshow import *
from RAG import *

# 定义疾病和药物关系映射
disease_relations = {
    "A0": ["Symptom_Disease", "Test_Disease"],
    "A1": ["Reason_Disease","Pathogenesis_Disease"],
    "A2": ["Symptom_Disease","Anatomy_Disease","Pathogenesis_Disease","Class_Disease"],
    "A3": ["Test_Disease","Treatment_Disease","Test_Items_Disease"],
    "B0": ["Drug_Disease","Test_Items_Disease"],
    "B1": ["Frequency_Drug","Duration_Drug","Amount_Drug","Method_Drug","ADE_Drug"],
    "B2": ["Drug_Disease", "Pathogenesis_Disease"],
    "B3": ["ADE_Drug"],
    "B4": ["ADE_Drug"],
    "B5": ["Treatment_Disease","Operation_Disese"],
    "B6": ["Operation_Disease","Drug_Disease","Treatment_Disease"],
    "C0": ["Symptom_Disease", "Drug_Disease"],
    "C1": ["Reason_Disease","Drug_Disease"],
    "C2": ["Reason_Disease"],
    "C3": ["Symptom_Disease"],
    "C4": ["Pathogenesis_Disease"],
    "D0": ["Treatment_Disease"],
    "D1": ["Treatment_Disease"],
    "D2": ["Treatment_Disease"],
    "D3": ["Treatment_Disease"],
    "E1":  ["Pathogenesis_Disease", "Reason_Disease"],
    "E2":  ["Pathogenesis_Disease","Test_Disease"],
    "E3":  ["Symptom_Disease"],
}


relation_dict = {
    'Test_Disease': '检查方法',
    'Symptom_Disease': '临床表现',
    'Treatment_Disease': '非药治疗',
    'Drug_Disease': '药品名称',
    'Anatomy_Disease': '部位',
    'Reason_Disease': '病因',
    'Pathogenesis_Disease': '发病机制',
    'Operation_Disese': '手术',
    'Class_Disease': '分期分型',
    'Test_Items_Disease': '检查指标',
    'Frequency_Drug': '用药频率',
    'Duration_Drug': '持续时间',
    'Amount_Drug': '用药剂量',
    'Method_Drug': '用药方法',
    'ADE_Drug': '不良反应'
}


def f_entity(text):
    answer = entity_predict(text)
    return answer

def f_intent(text):
    intent_predictor = IntentPredictor()
    intent = intent_predictor.predict(text)
    return intent

def f_KG_query(graph, entities,intent):
    relations = disease_relations[intent]

    res1, res2 = neo4j_query(graph, entities, relations, relation_dict)
    return res1, res2

def f_RAG_prepare():
    pdf_path = "function/data/国家基层糖尿病防治管理指南（ 2022）.pdf"

    # 嵌入模型加载
    embedding_model = SentenceTransformer("function/models/bce-embedding-base_v1")  # 用于生成嵌入向量

    # 1. 执行一次 RAG_prepare 来准备 PDF 内容和知识库
    pdf_content, knowledge_base = RAG_prepare(pdf_path, embedding_model)
    # st.write(f"PDF 提取文本完成，共 {len(pdf_content)} 页")
    # st.write(f"向量知识库构建完成，共包含 {len(knowledge_base)} 条知识项")
    # pdf_content =[]
    # knowledge_base =[]
    return embedding_model,pdf_content,knowledge_base

def f_RAG_query(embedding_model,pdf_content,knowledge_base):
    top_results = retrieve_answer(query, knowledge_base, embedding_model, top_k=3)
    rag_res = []
    for i, (score, entry) in enumerate(top_results):
        rag_res.append(entry['content'])

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
option = st.sidebar.selectbox("选择模型或解释", ["问答系统","流程演示"])

if option == "流程演示":
    query = st.text_input("请输入问题", key="query_input")
    if st.button("确认", key="entity_confirm"):
        if query.strip() == "":
            st.warning("请输入内容后再点击确认。")
        else:
            # RAG 准备工作
            embedding_model, pdf_content, knowledge_base = f_RAG_prepare()

            # 实体识别
            entity_texts = f_entity(query)
            entities = [entity["text"] for entity in entity_texts]
            st.write("识别到的实体为：")
            st.write(entities)

            # 意图识别
            intent = f_intent(query)
            st.write("识别到的意图为：")
            st.write(intent)

            # 图谱查询
            graph = KG_load()
            res1, res2 = f_KG_query(graph, entities,intent)
            st.write(res1,res2)

            # RAG
            st.write(f"PDF 提取文本完成，共 {len(pdf_content)} 页")
            st.write(f"向量知识库构建完成，共包含 {len(knowledge_base)} 条知识项")

            top_results = retrieve_answer(query, knowledge_base, embedding_model, top_k=3)
            rag_res = []
            for i, (score, entry) in enumerate(top_results):
                rag_res.append(entry['content'])
            print("\n检索结果（按相似度排序）：")
            st.write(f"从向量知识库中提取一下相似信息 {rag_res} ")
                # LLM 润色
            prompt = '''
                              你好，我需要你的帮助。请扮演一名资深的糖尿病专家，现在病人对你提问，而你要根据我提供的信息回答问题。
                              以下是病人的问题：
                              %s
                              此外，我还查询了专业的知识图谱，所以可以给你了以下提示信息（是一个字典，内部包含一下参考知识）：
                              %s
                              此外，我还查询了专门的文档：%s 和 %s
                              请你结合以上问句和提示，基于糖尿病领域的专业知识，为病人提供一个全面、详细、准确且易于理解的回答，可以适当使用我提示你的参考知识，但不能局限于我的参考知识。
                               ''' % (query, res1, res2, rag_res)
            # LLM 润色
            st.write("原始问句回答结果")
            st.write(ask_question(query))
            st.write("构建的提示词为：")
            st.write(prompt)
            st.write("提示后结果为：")
            st.write(ask_question(prompt))


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
                st.session_state.embedding_model, st.session_state.pdf_content, st.session_state.knowledge_base = f_RAG_prepare()
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
            details = {}  # 存储处理细节

            try:
                # 执行综合处理流程
                with st.spinner("正在分析问题..."):
                    # 1. 实体识别
                    entity_result = f_entity(prompt)
                    details["entities"] = [entity["text"] for entity in entity_result]

                    # 2. 意图识别
                    details["intent"] = f_intent(prompt)


                    if details["intent"] == "F":
                        # 4. RAG检索
                        top_results = retrieve_answer(prompt, st.session_state.knowledge_base,
                                                      st.session_state.embedding_model, top_k=3)
                        details["rag_results"] = [entry['content'] for _, entry in top_results]

                        # 5. 构建最终提示词
                        final_prompt = f'''你好，请扮演一名资深的糖尿病专家，现在病人对你提问，而我可以根据提供的信息帮助回答问题。
                                                    以下是病人的问题：{prompt}
                                                    以下是我可以提供的信息
                                                    知识图谱查询结果：暂无
                                                    相关文档参考：{details["rag_results"]}
                                                     请结合以上信息，提供专业、准确的回答。请务必注意，不必回答我的问题，只需要回答病人问题'''
                    else:
                        # 3. 知识图谱查询
                        graph = KG_load()
                        kg_res1, kg_res2 = f_KG_query(graph, details["entities"], details["intent"])
                        details["kg_results"] = f"{kg_res1}\n{kg_res2}"

                        # 4. RAG检索
                        top_results = retrieve_answer(prompt, st.session_state.knowledge_base,
                                                          st.session_state.embedding_model, top_k=3)
                        details["rag_results"] = [entry['content'] for _, entry in top_results]

                        # 5. 构建最终提示词
                        final_prompt = f'''你好，请扮演一名资深的糖尿病专家，现在病人对你提问，而我可以根据提供的信息帮助回答问题。
                                                    以下是病人的问题：{prompt}
                                                    以下是我可以提供的信息
                                                    知识图谱查询结果：{details["kg_results"] }
                                                    相关文档参考：{details["rag_results"]}
                                                    请结合以上信息，提供专业、准确的回答。请务必注意，不必回答我的问题，只需要回答病人问题'''

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
                    st.write(details.get("entities", "未识别到实体"))

                    st.subheader("意图识别结果")
                    st.write(details.get("intent", "未识别到意图"))

                    st.subheader("知识图谱查询结果")
                    st.code(details.get("kg_results", "无相关结果"))

                    st.subheader("相关文档参考")
                    for i, content in enumerate(details.get("rag_results", [])):
                        st.markdown(f"**相关段落 {i + 1}**")
                        st.write(content[:300] + "...")  # 显示前300字符

            except Exception as e:
                    st.error(f"生成回答时出错: {str(e)}")