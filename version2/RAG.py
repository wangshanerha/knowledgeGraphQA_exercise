import pdfplumber
from sentence_transformers import SentenceTransformer, util

# 提取 PDF 所有页面的内容（合并为大块文本）
def extract_pdf_content(pdf_path):
    pdf_content = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:  # 跳过空页
                # 按段落合并内容，确保每页生成一个较大的文本块
                merged_text = " ".join(page_text.split("\n"))
                pdf_content.append({"page_number": page_number + 1, "content": merged_text})
    return pdf_content

# 构建知识库（知识嵌入）
def build_knowledge_base(pdf_content, embedding_model):
    knowledge_base = []
    for item in pdf_content:
        page_number = item["page_number"]
        content = item["content"]
        for paragraph in content.split("。"):  # 按句号切分并保留内容
            if paragraph.strip():  # 跳过空段落
                knowledge_base.append({
                    "id": f"page_{page_number}_para_{knowledge_base.count(paragraph) + 1}",
                    "page_number": page_number,
                    "content": paragraph,
                    "embedding": embedding_model.encode(paragraph, convert_to_tensor=True)
                })
    return knowledge_base

# RAG 检索
def retrieve_answer(question, knowledge_base, embedding_model, top_k=3):
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)
    scores = []

    for entry in knowledge_base:
        score = util.pytorch_cos_sim(question_embedding, entry["embedding"]).item()
        scores.append((score, entry))

    # 按相似度排序
    top_results = sorted(scores, key=lambda x: x[0], reverse=True)[:top_k]
    return top_results

# RAG 执行函数
def RAG_start(question, pdf_path, embedding_model):
    # 1. 加载并提取 PDF 文本
    pdf_content = extract_pdf_content(pdf_path)
    print(f"PDF 提取文本完成，共 {len(pdf_content)} 页")

    # 2. 构建知识库
    knowledge_base = build_knowledge_base(pdf_content, embedding_model)
    print(f"知识库构建完成，共包含 {len(knowledge_base)} 条知识项")

    # 3. 提问与检索
    print('**********************************************************')
    print(f"\n问题：{question}")

    top_results = retrieve_answer(question, knowledge_base, embedding_model, top_k=3)

    # 4. 输出检索结果
    print("\n检索结果（按相似度排序）：")
    for i, (score, entry) in enumerate(top_results):
        print(f"\n结果 {i + 1}:")
        print(f"相似度分数：{score:.4f}")
        print(f"所在页码：{entry['page_number']}")
        print(f"内容：{entry['content']}")
        print(f"所在页的所有内容{pdf_content[entry['page_number']]}")
# 主函数
if __name__ == "__main__":
    print("start")
    # 加载 PDF 文档并提取文本
    pdf_path = "data/国家基层糖尿病防治管理指南（ 2022）.pdf"

    # 嵌入模型加载
    embedding_model = SentenceTransformer("../../doc/bce-embedding-base_v1")  # 用于生成嵌入向量

    # 提问与检索
    question = "谈谈降糖药物的选择"  # 输入问题
    RAG_start(question, pdf_path, embedding_model)
