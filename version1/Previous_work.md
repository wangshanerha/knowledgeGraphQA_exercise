# 之前的工作

- 2024.12.24

完成textRNN（BiLSTM）和textCNN_att（BiLSTM+attention）代码。

- 2024.12.23

完成textCNN代码。最终效果大概76%左右。

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