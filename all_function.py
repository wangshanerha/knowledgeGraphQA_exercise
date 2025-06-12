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


query = "介绍一下使用沙格列汀治疗糖尿病的细节？"

# 实体识别
entity_texts = f_entity(query)
entities = [entity["text"] for entity in entity_texts]
print("识别到的实体为：")
print(entities)


# 意图识别
intent = f_intent(query)
print("识别到的意图为：")
print(intent)

# 图谱查询
graph = KG_load()
res1, res2 = f_KG_query(graph, entities,intent)
print(res1, res2)

rag_res = []

# RAG
embedding_model, pdf_content, knowledge_base = f_RAG_prepare()

top_results = retrieve_answer(query, knowledge_base, embedding_model, top_k=3)

for i, (score, entry) in enumerate(top_results):
    rag_res.append(entry['content'])
print("\n检索结果（按相似度排序）：")
print(rag_res)

prompt = f'''你好，请扮演一名资深的糖尿病专家，现在病人对你提问，而我可以根据提供的信息帮助回答问题。
                以下是病人的问题：{query}
                以下是我可以提供的信息
                知识图谱查询结果：{res1, res2 }
                相关文档参考：{rag_res}
                请结合以上信息，提供专业、准确的回答。请务必注意，不必回答我的问题，只需要回答病人问题'''

# LLM 润色
# print("原始问句回答结果")
# print(ask_question(query))
print("构建的提示词为：")
print(prompt)
print("提示后结果为：")
print(ask_question(prompt))
