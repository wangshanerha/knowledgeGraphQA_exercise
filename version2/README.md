# 基于知识图谱的糖尿病问答系统

## 1 项目介绍

知识图谱是目前自然语言处理的一个热门方向。本项目是立足医药领域，以天池瑞金数据集和Dacorp问答数据为基础使用neo4j构建知识图谱并完成问答系统。主体思路为：

![version2](img/version2.jpg)

RAG的数据融合了多个来源，包括PDF数据（ [国家基层糖尿病防治管理指南（ 2022）.pdf](data\国家基层糖尿病防治管理指南（ 2022）.pdf) ）、Excel数据（ [intent_detection.xlsx](data\intent_detection.xlsx) ）等。





- **重要提示**

本项目用到的模型 `bce-embedding-base_v1`在本地，文件过大没有上传。

## 2 运行环境

| environment | version |
| :---------: | :-----: |
|   Windows   |   11    |
|   Python    |   3.9   |
|   pytorch   |  2.4.1  |
|    neo4j    | 社区版  |

 

## 3 代码解释

- RAG.py

对PDF进行RAG操作。embeding模型为`bce-embedding-base_v1`

-  RAG_Excel.py

对Excel进行RAG操作。embeding模型为`bce-embedding-base_v1`





## 4 工作进度

- 2024.12.19

更新`readme`文件

稍微修改`RAG.py`，实现多轮对话。

完成`RAG_Excel.py`，实现对Excel进行RAG。最初用的是`paraphrase-MiniLM-L6-v2`，但是效果奇差无比。

在后续的工作中可以尝试其他的embeding模型进行**对比评估**；也可以进行**多路召回和重排序**；可以考虑进行先**问题重述or意图识别再进行RAG**，PDF的知识非常专业，Excel的知识非常口语。

- 2024.12.16

更新RAG内容，调整RAG输出：

> **添加页码信息**：
>
> - 在 `extract_pdf_content` 中为每个页面的内容加上 `page_number` 属性。
> - 在输出结果时显示对应的页码。
>
> **合并更多内容**：
>
> - 在 `extract_pdf_content` 中，将每一页的所有段落合并为一个块，避免单独提取小段内容。
> - 在 `build_knowledge_base` 中按句号（`。`）再次分割大块内容，确保语义完整性。
>
> **输出详细内容**：
>
> - 输出检索结果时，显示完整的句子内容，不再截断

效果谈不上好，需要优化。

