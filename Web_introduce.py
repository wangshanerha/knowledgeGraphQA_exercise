import streamlit as st

st.title('基于知识图谱的糖尿病问答系统')

st.subheader('项目介绍')
st.write("知识图谱是目前自然语言处理的一个热门方向。本项目是立足医药领域，以天池瑞金数据集和Dacorp问答数据为基础使用neo4j构建知识图谱并完成问答系统。"
    )
st.write("设计并实现糖尿病领域知识图谱问答系统，结合自然语言处理与图数据库技术，构建从医学知识抽取到智能问答的全流程解决方案。基于BERT构建命名实体识别模块，融合TextCNN、BiLSTM及ELECTRA、DeBERTa等预训练模型优化问句解析算法，通过对比实验筛选最优模型组合，实现用户意图精准识别与语义关系映射。利用Neo4j图数据库构建糖尿病知识图谱并结合LangChain框架管理以及RAG技术实现大语言模型（ChatGLM）的推理与知识增强。基于Streamlit搭建交互式可视化界面，集成自然语言交互。"
    )
st.write("核心技术：知识图谱构建（Neo4j）、深度学习（PyTorch）、预训练模型调优（BERT/DeBERTa）、大模型集成（LangChain/ChatGLM）、开发（Python/Streamlit)。")
st.subheader('项目地址')
st.write("https://github.com/wangshanerha/knowledgeGraphQA_exercise/tree/master/version2")