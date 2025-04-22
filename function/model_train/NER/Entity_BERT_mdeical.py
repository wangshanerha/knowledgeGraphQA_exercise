import os
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, BertTokenizerFast
from sklearn.metrics import classification_report
from torch.nn import CrossEntropyLoss

def load_json_data(json_dir, tokenizer, model):
    sentences = []
    true_entities = []
    true_token_labels = []
    label_to_id = model.config.label2id

    for json_file in os.listdir(json_dir):
        if not json_file.endswith('.json'):
            continue
        with open(os.path.join(json_dir, json_file), 'r', encoding='utf-8') as f:
            data = json.load(f)
            for para in data['paragraphs']:
                for sent in para['sentences']:
                    # 处理句子文本
                    sent_text = sent['sentence']
                    sentences.append(sent_text)
                    
                    # 处理真实实体
                    entities = []
                    for e in sent['entities']:
                        entities.append({
                            'type': e['entity_type'],
                            'start': e['start_idx'],
                            'end': e['end_idx'],
                            'text': sent_text[e['start_idx']:e['end_idx']]
                        })
                    true_entities.append(entities)
                    
                    # 生成token级别标签
                    encoding = tokenizer(sent_text, return_offsets_mapping=True, add_special_tokens=False)
                    tokens = encoding.tokens()
                    offsets = encoding['offset_mapping']
                    
                    # 初始化标签为O
                    labels = ['O'] * len(offsets)
                    for e in sent['entities']:
                        ent_start = e['start_idx']
                        ent_end = e['end_idx']
                        ent_type = e['entity_type']
                        
                        # 标记实体范围内的token
                        entity_tokens = []
                        for token_idx, (token_start, token_end) in enumerate(offsets):
                            if token_start >= ent_start and token_end <= ent_end:
                                entity_tokens.append(token_idx)
                        
                        # 分配B/I标签
                        if entity_tokens:
                            labels[entity_tokens[0]] = f'B-{ent_type}'
                            for idx in entity_tokens[1:]:
                                labels[idx] = f'I-{ent_type}'
                    
                    # 转换为ID
                    label_ids = [label_to_id[label] for label in labels]
                    true_token_labels.append(label_ids)
    
    return sentences, true_entities, true_token_labels

def extract_entities(sentences, model, tokenizer):
    label_list = list(model.config.id2label.values())
    results = []
    
    for sent in tqdm(sentences):
        inputs = tokenizer(sent, return_tensors="pt", padding=True, 
                         add_special_tokens=False, return_offsets_mapping=True)
        with torch.no_grad():
            outputs = model(**inputs)
        
        preds = outputs.logits.argmax(-1).squeeze().tolist()
        offsets = inputs['offset_mapping'].squeeze().tolist()
        mask = inputs['attention_mask'].squeeze().tolist()
        
        entities = []
        current_entity = None
        
        for idx, (pred, offset) in enumerate(zip(preds, offsets)):
            if not mask[idx]:
                continue
            
            label = label_list[pred]
            if label == 'O':
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                continue
                
            pos, _, ent_type = label.partition('-')
            token_start, token_end = offset
            
            if pos == 'B':
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    'type': ent_type,
                    'start': token_start,
                    'end': token_end,
                    'tokens': [(token_start, token_end)]
                }
            elif pos in ['I', 'E'] and current_entity and current_entity['type'] == ent_type:
                current_entity['end'] = token_end
                current_entity['tokens'].append((token_start, token_end))
                if pos == 'E':
                    entities.append(current_entity)
                    current_entity = None
            else:
                if current_entity:
                    entities.append(current_entity)
                current_entity = None
        
        if current_entity:
            entities.append(current_entity)
            
        # 合并相邻token并去重
        final_entities = []
        for ent in entities:
            start = min(t[0] for t in ent['tokens'])
            end = max(t[1] for t in ent['tokens'])
            final_entities.append({
                'type': ent['type'],
                'start': start,
                'end': end,
                'text': sent[start:end]
            })
        
        results.append(final_entities)
    
    return results

def calculate_entity_metrics(true_entities_list, pred_entities_list, entity_type):
    tp = fp = fn = 0
    
    for true_ents, pred_ents in zip(true_entities_list, pred_entities_list):
        # 转换格式为集合
        true_set = set()
        for e in true_ents:
            if e['type'] == entity_type:
                true_set.add((e['start'], e['end'], e['type']))
                
        pred_set = set()
        for e in pred_ents:
            if e['type'] == entity_type:
                pred_set.add((e['start'], e['end'], e['type']))
        
        # 计算统计量
        tp += len(true_set & pred_set)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def evaluate():
    # 初始化模型
    model_path = '../../models/bert-base-chinese-medical-ner'
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 加载数据
    json_dir = "../../data/"
    sentences, true_entities, true_token_labels = load_json_data(json_dir, tokenizer, model)
    
    # 预测实体和token标签
    pred_entities = extract_entities(sentences, model, tokenizer)
    
    # 计算eval_loss
    loss_fct = CrossEntropyLoss(reduction='sum')
    total_loss = 0
    total_tokens = 0
    
    for sent, true_labels in zip(sentences, true_token_labels):
        inputs = tokenizer(sent, return_tensors='pt', padding=True, 
                          add_special_tokens=False).to(device)
        labels = torch.tensor(true_labels[:inputs['input_ids'].size(1)]).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, labels=labels)
        
        total_loss += outputs.loss.item() * labels.numel()
        total_tokens += labels.numel()
    
    eval_loss = total_loss / total_tokens
    
    # 计算token级别指标
    all_true = []
    all_pred = []
    
    for sent, true_labels in zip(sentences, true_token_labels):
        inputs = tokenizer(sent, return_tensors='pt', padding=True, 
                          add_special_tokens=False).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        
        preds = outputs.logits.argmax(-1).squeeze().tolist()
        mask = inputs['attention_mask'].squeeze().tolist()
        
        # 对齐长度
        min_len = min(len(true_labels), len(preds))
        all_true.extend(true_labels[:min_len])
        all_pred.extend(preds[:min_len])
    
    label_names = list(model.config.id2label.values())
    report = classification_report(all_true, all_pred, labels=range(len(label_names)),
                                   target_names=label_names, output_dict=True, zero_division=0)
    
    # 提取各项指标
    macro_precision = report['macro avg']['precision']
    macro_recall = report['macro avg']['recall']
    macro_f1 = report['macro avg']['f1-score']
    
    weighted_precision = report['weighted avg']['precision']
    weighted_recall = report['weighted avg']['recall']
    weighted_f1 = report['weighted avg']['f1-score']
    
    # 计算实体级别指标
    disease_precision, disease_recall, disease_f1 = calculate_entity_metrics(true_entities, pred_entities, 'Disease')
    drug_precision, drug_recall, drug_f1 = calculate_entity_metrics(true_entities, pred_entities, 'Drug')
    
    # 打印结果
    print(f"{'Metric':<20} {'Value':<10}")
    print(f"{'eval_loss':<20} {eval_loss:.4f}")
    print(f"{'macro_precision':<20} {macro_precision:.4f}")
    print(f"{'macro_recall':<20} {macro_recall:.4f}")
    print(f"{'macro_f1':<20} {macro_f1:.4f}")
    print(f"{'weighted_precision':<20} {weighted_precision:.4f}")
    print(f"{'weighted_recall':<20} {weighted_recall:.4f}")
    print(f"{'weighted_f1':<20} {weighted_f1:.4f}")
    print(f"{'disease_f1':<20} {disease_f1:.4f}")
    print(f"{'drug_f1':<20} {drug_f1:.4f}")

if __name__ == "__main__":
    evaluate()