from transformers import AutoModelForTokenClassification, BertTokenizerFast

def extract_entities(sentences):
    tokenizer = BertTokenizerFast.from_pretrained('D:/pycharm/example/nlp/graduate/version2/bert-base-chinese-medical-ner')
    model = AutoModelForTokenClassification.from_pretrained("D:/pycharm/example/nlp/graduate/version2/bert-base-chinese-medical-ner")
    # 对所有句子进行分词，不添加特殊符号，保持分词结果与原始句子对齐
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, add_special_tokens=False)
    # 获取模型预测的标签
    outputs = model(**inputs)
    logits = outputs.logits
    # 取出最大值的索引，并乘以attention_mask以忽略填充位置
    predictions = logits.argmax(-1).tolist()
    attention_masks = inputs['attention_mask'].tolist()

    # 处理每个句子
    results = []
    for i, sentence in enumerate(sentences):
        pred = predictions[i]
        mask = attention_masks[i]
        tokens = tokenizer.tokenize(sentence, add_special_tokens=False)
        # 获取每个token对应的字符跨度
        encoding = tokenizer(sentence, return_offsets_mapping=True, add_special_tokens=False)
        offsets = encoding['offset_mapping']

        entities = []
        entity = []
        for j, tag_id in enumerate(pred):
            if mask[j] == 0:
                continue  # 忽略填充位置
            tag = tag_id
            token = tokens[j]
            start_char, end_char = offsets[j]
            if tag == 1:  # B
                if entity:
                    entities.append(entity)
                entity = [(start_char, end_char, token)]
            elif tag == 2:  # I
                if entity:
                    entity.append((start_char, end_char, token))
                else:
                    # 没有B的情况，可能是模型预测错误
                    pass
            elif tag == 3:  # E
                if entity:
                    entity.append((start_char, end_char, token))
                    entities.append(entity)
                    entity = []
                else:
                    # 没有B的情况，可能是模型预测错误
                    pass
            elif tag == 4:  # O
                if entity:
                    entities.append(entity)
                    entity = []

        # 拼接实体文本
        final_entities = []
        for ent in entities:
            start = ent[0][0]
            end = ent[-1][1]
            text = sentence[start:end]
            final_entities.append((text, start, end))

        results.append(final_entities)

    return results


if __name__ == "__main__":

    # 测试函数
    sentences = ["糖尿病足部溃烂：这是糖尿病患者由于血管病变和神经病变导致的足部溃疡，常伴有感染和脓性分泌物","早期症状如突然性的多饮多食、体重减轻、四肢麻痹、肌肉抽筋、便秘腹泻、精神萎靡、性功能障碍等。"]
    entities = extract_entities(sentences)
    print(entities)
    for i, sentence in enumerate(sentences):
        print(f"Sentence {i + 1}: {sentence}")
        print("Entities:")
        for entity in entities[i]:
            print(f"  Text: {entity[0]}, Start: {entity[1]}, End: {entity[2]}")
        print()