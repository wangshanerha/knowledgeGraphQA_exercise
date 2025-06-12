import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, BitsAndBytesConfig
from peft import PeftModel
from typing import List, Dict
import json, os

def load_model(model_dir: str, use_lora_adapter: bool = False) -> (AutoTokenizer, torch.nn.Module):
    """加载 tokenizer 和量化后的模型"""
    # 1) 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2) 配置4-bit量化
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,  # 计算时使用float16
        bnb_4bit_quant_type="nf4",            # 量化类型
        bnb_4bit_use_double_quant=True        # 嵌套量化进一步压缩
    )

    # 3) 加载模型
    if use_lora_adapter:
        base_model = AutoModelForTokenClassification.from_pretrained(
            model_dir,
            quantization_config=quantization_config,
            device_map="auto",                # 自动分配设备
            torch_dtype=torch.float16,
            use_cache=False                   # 禁用缓存以避免警告
        )
        model = PeftModel.from_pretrained(base_model, model_dir)
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            model_dir,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            use_cache=False
        )

    model.eval()
    return tokenizer, model

def predict_entities(
    text: str,
    tokenizer: AutoTokenizer,
    model: torch.nn.Module,
    id2label: Dict[int, str]
) -> List[Dict]:
    """
    对单条文本做 NER 预测，返回 entity list。
    每个 entity 是一个 dict，包含 text、label、start, end 四个字段。
    """
    # 1) Tokenize
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        padding='longest',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids']   # [1, L]
    attention_mask = encoding['attention_mask']
    offsets = encoding['offset_mapping'][0].tolist()  # List[(start_char, end_char)]

    # 2) 模型前向
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [1, L, num_labels]
        preds = torch.argmax(logits, dim=-1)[0].cpu().tolist()  # [L]

    # 3) 解析 B-/I- 标签，抽取实体
    entities = []
    current_ent = None
    for idx, label_idx in enumerate(preds):
        label = id2label[label_idx]
        if label == 'O' or offsets[idx] == (0, 0):
            # 如果当前在构建实体，则先收尾
            if current_ent:
                entities.append(current_ent)
                current_ent = None
            continue

        prefix, ent_type = label.split('-', maxsplit=1)
        start_char, end_char = offsets[idx]

        if prefix == 'B':
            # 遇到新的实体
            if current_ent:
                entities.append(current_ent)
            current_ent = {
                'label': ent_type,
                'start': start_char,
                'end': end_char,
                'text': text[start_char:end_char]
            }
        elif prefix == 'I' and current_ent and current_ent['label'] == ent_type:
            # 实体续写，扩展 end 和文本
            current_ent['end'] = end_char
            current_ent['text'] = text[current_ent['start']:end_char]
        else:
            # 出现了 I-XXX 但类型或状态不一致，则重置
            if current_ent:
                entities.append(current_ent)
            current_ent = None

    # 若末尾仍有未关闭的实体
    if current_ent:
        entities.append(current_ent)

    return entities

def entity_predict(text,):
    MODEL_DIR = '../../saved_models/Qwen2.5_NER_results_DlearnRate/merged_model'
    USE_LORA_ADAPTER = False

 # 加载 label 配置（可选，从 config 中恢复）
    id2label_path = os.path.join(MODEL_DIR, 'id2label.json')
    if os.path.exists(id2label_path):
        with open(id2label_path, 'r', encoding='utf-8') as f:
            id2label = {int(k): v for k, v in json.load(f).items()}
    else:
        # 直接从模型 config 里拿
        temp_model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
        id2label = temp_model.config.id2label
        del temp_model

# --------- 加载模型 ---------
    tokenizer, model = load_model(MODEL_DIR, use_lora_adapter=USE_LORA_ADAPTER)
    ents = predict_entities(text, tokenizer, model, id2label)
    return ents

if __name__ == '__main__':
    text = "长时间使用胰岛素有什么不良反应吗"
    ents = entity_predict(text)
    if ents:
        print("预测到实体：")
        for e in ents:
            print(f"  - {e['text']}  ({e['label']})  [{e['start']}:{e['end']}]")
    else:
            print("未识别到任何实体。")
