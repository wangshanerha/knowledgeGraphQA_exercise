import pandas as pd
from sentence_transformers import SentenceTransformer, util

# 构建表格知识库
def build_table_knowledge_base(df, embedding_model):
    table_knowledge_base = []
    for index, row in df.iterrows():
        question = row['question']
        # 将问题嵌入
        question_embedding = embedding_model.encode(question, convert_to_tensor=True)
        table_knowledge_base.append({
            "id": row['id'],
            "question": question,
            "answer": row['answer'],
            "embedding": question_embedding
        })
    return table_knowledge_base

# RAG 检索：根据问题检索表格知识库
def retrieve_answer_from_table(question, table_knowledge_base, embedding_model, top_k=3):
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)
    scores = []

    # 计算问题与每个表格条目之间的相似度
    for entry in table_knowledge_base:
        score = util.pytorch_cos_sim(question_embedding, entry["embedding"]).item()
        scores.append((score, entry))

    # 按相似度排序
    top_results = sorted(scores, key=lambda x: x[0], reverse=True)[:top_k]
    return top_results

# RAG 表格准备阶段
def RAG_Excel_prepare(df, embedding_model):
    # 构建表格知识库
    table_knowledge_base = build_table_knowledge_base(df, embedding_model)
    print(f"表格知识库构建完成，共包含 {len(table_knowledge_base)} 条知识项")
    return table_knowledge_base

# RAG 表格问答阶段
def RAG_Excel_ask(question, table_knowledge_base, embedding_model):
    # 进行检索
    results = retrieve_answer_from_table(question, table_knowledge_base, embedding_model)

    if results:
        print(f"\n问题：{question}")
        print("\n检索结果（按相似度排序）：")
        for result in results:
            score, entry = result  # 解包 result 为 score 和 entry
            print(f"\nID: {entry['id']}")
            print(f"相似度分数：{score:.4f}")
            print(f"问题：{entry['question']}")
            print(f"答案：{entry['answer']}")
    else:
        print("没有找到相关的答案。")

# 主函数
if __name__ == "__main__":
    # 示例：加载 Excel 文件（假设数据已经被加载到 DataFrame 中）
    excel_path = "data/intent_detection.xlsx"
    df = pd.read_excel(excel_path)  # 这里你可以使用自己的 Excel 文件路径

    # 加载嵌入模型
    embedding_model = SentenceTransformer('bce-embedding-base_v1')

    # 1. 准备表格知识库
    table_knowledge_base = RAG_Excel_prepare(df, embedding_model)

    # 2. 循环接收问题并进行检索，直到输入 "end" 为止
    while True:
        question = input("请输入问题（输入 'end' 结束）：")
        if question.lower() == 'end':
            print("程序结束。")
            break
        else:
            RAG_Excel_ask(question, table_knowledge_base, embedding_model)
