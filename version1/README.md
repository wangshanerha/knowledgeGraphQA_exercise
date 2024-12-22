# 基于知识图谱的糖尿病问答系统

## 1 项目介绍

知识图谱是目前自然语言处理的一个热门方向。本项目是立足医药领域，以天池瑞金数据集和Dacorp问答数据为基础使用`neo4j`构建知识图谱并完成问答系统。目前项目主体思路如下：

![img1](../img/img1.jpg)

**知识图谱**选用天池瑞金数据集，读取json文件进行构建KG。

**命名实体识别**选择`jieba`分词构建自定义字典，对问句query进行识别。

**意图识别**抽象为基于`bert`的中文文本分类任务。共有23个意图label。如图所示：

![PixPin_2024-12-12_15-43-13](../img/PixPin_2024-12-12_15-43-13.jpg)

具体内容可以参考 [doc目录下2022.ccl-1.36.pdf](..\doc\2022.ccl-1.36.pdf) 或者Dacorp数据集相关论文-[中文糖尿病问题分类体系及标注语料库构建研究](https://aclanthology.org/2022.ccl-1.36/)。

- **重要提示**

bert模型（例如：`bert-base-chinese`）和训练好的模型（例如：`saved_model`）都在本地。文件过大没有上传。

## 2 运行环境

| environment |      version      |
| :---------: | :---------------: |
|   Windows   |        11         |
|   Python    |        3.9        |
|   pytorch   |       2.4.1       |
|    model    | bert-base-chinese |
|    neo4j    |      社区版       |
|     LLM     |   chatglm-flash   |

 

## 3 代码解释

- entity.py 

用于命名实体识别和构建jieba自定义字典。

- intent_detection.py

用于训练模型进行意图识别，将识别出的意图对应到图谱的关系，进行检索。

其中，`train_model`函数可以对模型进行训练，并将训练数据（损失、准确率等信息）写入Excel（文件要存在，这里用的是test.xlsx）。

- KG.py

读取data目录下的json文件从而构建知识图谱。

- QA_system.py

问答系统的主程序，在这里运行整个QA系统。

- ack_LLM.py

单独调用LLM用于测试。

- selfDefine.txt

自定义字典，用于命名实体识别。

## 4 效果评估

|       model        | params | epoch | optimality | loss |
| :----------------: | :----: | :---: | :--------: | :--: |
| bert-base-chinese  |  110M  |  10   |            |      |
| bert-large-uncased |  340M  |  10   |            |      |
|                    |        |       |            |      |
|                    |        |       |            |      |





## 5 工作进度

- 2024.12.22

`bert-large-uncased`因为language是英文所以效果不是很好，大概初始的准确率只有27%左右。

评估可以用的模型[中文文本分类-TextCNN+TextRNN+FastText+TextRCNN+TextRNN_Attention+DPCNN+Transformer_基于pytorch深度学习的nlp代码](https://blog.csdn.net/qq_31136513/article/details/131589556)

- 2024.12.21

修改了intent_detection.py代码，可以将模型训练损失、准确率等数据写入Excel。

实现对意图识别模型`bert-base-chinese`的效果评估。

下载其他微调bert模型[Models - Hugging Face](https://huggingface.co/models?other=base_model:finetune:google-bert/bert-base-chinese)

可以clone的代码进行重新上传，记的加时间。

- 2024.12.10

项目基本完成。

图谱中缺乏对Drug_Disease的详细描述，显得我们不够专业。可以考虑RAG。

输入的prompt为results的字典不够规范，建议建立标准的提示模板。

> 例如：某次query的results为{'Operation_Disease': [], 'Drug_Disease': ['甘舒霖', '二甲双胍片', '格列苯脲', '长效胰岛素', '甘舒霖30R注射液', '阿卡波糖胶囊', '诺和锐30R胰岛素', '二甲双胍', '格列齐特', '胰岛素', '诺和龙', '格华止', '格列吡嗪', '阿卡波糖'], 'Treatment_Disease': ['饮食控制'], 'Test_Items_Disease': ['总胆固醇', '血氯', '血钠', '尿A/C', '血清铁', '超敏C反应蛋白', '中性粒细胞', '血小板', '血白细胞', '血沉', 'HbA1C', '血红蛋白', '葡萄糖', '潜血', '蛋白质', 'HGB', 'RBC', 'WBC', '体温', '尿糖', '白细胞', 'BP', 'R', 'P', 'T', '手指毛细血糖', '甘油三酯', '血糖', '甲状腺抗体', '血色素', '红细胞', '血清肌酐', '血尿素氮', '尿酸', '血压', '餐后血糖', '餐前血糖', '血糖空腹', '餐后2h血糖', '糖化血红蛋白', '空腹血糖']}

某些情况LLM的回答并不规范。可以考虑LLM或者其他模型重述query

> 例如：
>
> query为：tan糖尿病如何治
>
> answer：首先，对于**tan糖尿病**的治疗，主要包括以下几个方面：

可以考虑与rag结合，完善图谱结构或者理解文档数据。

可以考虑建立patient图谱节点，存储病人信息。

可以做一下命名实体识别，实现可更新的知识图谱。

可以比较不同的意图识别模型，进行对比分析。

- 2024.11.14

交流，jieba分词效果比较依赖自定义字典。需要去优化字典。



