import json
import os
import numpy as np
import pandas as pd  # 新增pandas库
from collections import Counter
from sklearn.model_selection import train_test_split
from seqeval.metrics import classification_report
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType

class NERDataset(Dataset):
    def __init__(self, samples, tokenizer, label2id, id2label, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = samples
        self.label2id = label2id
        self.id2label = id2label

    @classmethod
    def create_datasets(cls, json_dir, tokenizer, test_size=0.2):
        samples, label2id, id2label = cls.process_data(json_dir)
        
        if len(samples) < 10:
            test_size = 0.0
            
        train_samples, eval_samples = train_test_split(
            samples, test_size=test_size, random_state=42
        )
        return (
            cls(train_samples, tokenizer, label2id, id2label),
            cls(eval_samples, tokenizer, label2id, id2label),
            label2id,
            id2label
        )

    @staticmethod
    def process_data(json_dir):
        label_types = set()
        samples = []
        
        json_files = [
            os.path.join(json_dir, f) 
            for f in os.listdir(json_dir) 
            if f.endswith(".json")
        ]
        
        print(f"\n发现 {len(json_files)} 个数据文件：")
        print("\n".join([os.path.basename(f) for f in json_files]))
        
        for file_idx, json_path in enumerate(json_files, 1):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"\n处理文件中 ({file_idx}/{len(json_files)}): {os.path.basename(json_path)}")
                    
                    file_sentences = 0
                    file_entities = 0
                    
                    for para in data["paragraphs"]:
                        for sent in para["sentences"]:
                            file_sentences += 1
                            text = sent["sentence"]
                            entities = sent["entities"]
                            labels = ["O"] * len(text)

                            sorted_entities = sorted(entities, 
                                                   key=lambda x: (x["end_idx"] - x["start_idx"]), 
                                                   reverse=True)
                            for entity in sorted_entities:
                                start = entity["start_idx"]
                                end = entity["end_idx"]
                                entity_type = entity["entity_type"]
                                
                                valid = True
                                if start >= len(text) or end > len(text):
                                    print(f"跳过无效实体：{entity}，文本长度：{len(text)}")
                                    valid = False
                                if end <= start:
                                    print(f"跳过反向实体：{entity}")
                                    valid = False
                                
                                if valid:
                                    labels[start] = f"B-{entity_type}"
                                    for i in range(start+1, end):
                                        labels[i] = f"I-{entity_type}"
                                    
                                    label_types.add(f"B-{entity_type}")
                                    label_types.add(f"I-{entity_type}")
                                    file_entities += 1

                            samples.append({"text": text, "labels": labels})
                    
                    print(f"处理完成 → 句子数: {file_sentences} | 有效实体: {file_entities}")
                    
            except Exception as e:
                print(f"\n处理文件失败: {os.path.basename(json_path)}")
                print(f"错误信息: {str(e)}")
                continue

        label_list = ["O"] + sorted(label_types)
        label2id = {label: idx for idx, label in enumerate(label_list)}
        id2label = {idx: label for idx, label in enumerate(label_list)}
        
        print("\n数据加载汇总：")
        print(f"总文件数: {len(json_files)}")
        print(f"总样本数: {len(samples)}")
        print(f"发现实体类型: {', '.join(sorted(label_types))}")
        
        return samples, label2id, id2label

    def align_labels(self, tokenized_input, original_labels):
        offset_mapping = tokenized_input.pop("offset_mapping").squeeze().tolist()
        labels = []
        for (start, end) in offset_mapping:
            if start == 0 and end == 0:
                labels.append(-100)
                continue

            if start >= len(original_labels):
                main_char = original_labels[-1] if original_labels else "O"
            else:
                main_char = original_labels[start]

            labels.append(self.label2id.get(main_char, self.label2id["O"]))

        if len(labels) > self.max_length:
            labels = labels[:self.max_length]
        else:
            labels += [-100] * (self.max_length - len(labels))
        return labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        tokenized = self.tokenizer(
            sample["text"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        labels = self.align_labels(tokenized, sample["labels"])
        return {
            "input_ids": tokenized["input_ids"].squeeze(),
            "attention_mask": tokenized["attention_mask"].squeeze(),
            "labels": torch.LongTensor(labels)
        }

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = []
    true_predictions = []
    for pred, lab in zip(predictions, labels):
        valid_labels = []
        valid_preds = []
        for p, l in zip(pred, lab):
            if l != -100:
                valid_labels.append(label_list[l])
                valid_preds.append(label_list[p])
        
        if len(valid_labels) > 0:
            true_labels.append(valid_labels)
            true_predictions.append(valid_preds)

    if len(true_labels) == 0:
        return {
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "weighted_precision": 0.0,
            "weighted_recall": 0.0,
            "weighted_f1": 0.0,
            "disease_f1": 0.0,
            "drug_f1": 0.0
        }

    report = classification_report(true_labels, true_predictions, output_dict=True)
    
    return {
        "macro_precision": report.get("macro avg", {}).get("precision", 0.0),
        "macro_recall": report.get("macro avg", {}).get("recall", 0.0),
        "macro_f1": report.get("macro avg", {}).get("f1-score", 0.0),
        "weighted_precision": report.get("weighted avg", {}).get("precision", 0.0),
        "weighted_recall": report.get("weighted avg", {}).get("recall", 0.0),
        "weighted_f1": report.get("weighted avg", {}).get("f1-score", 0.0),
        "disease_f1": report.get("Disease", {}).get("f1-score", 0.0),
        "drug_f1": report.get("Drug", {}).get("f1-score", 0.0)
    }

# 参数配置
MODEL_NAME = "../../models/Qwen2.5-1.5B"
DATA_PATH = "../../data/"
OUTPUT_DIR = "../../saved_models/Qwen2.5_NER_results_DlearnRate"  # 修改输出目录
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 3e-4
WARMUP_STEPS = 500          # 新增参数
WEIGHT_DECAY = 0.01         # 新增参数
SCHEDULER_TYPE = "cosine"   # 新增参数

# 初始化组件
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 创建数据集
train_dataset, eval_dataset, label2id, id2label = NERDataset.create_datasets(
    DATA_PATH, tokenizer, test_size=0.2
)
label_list = list(label2id.keys())

# 保存标签配置
os.makedirs(f"{OUTPUT_DIR}/label_config", exist_ok=True)
with open(f"{OUTPUT_DIR}/label_config/id2label.json", "w") as f:
    json.dump(id2label, f)
with open(f"{OUTPUT_DIR}/label_config/label2id.json", "w") as f:
    json.dump(label2id, f)

# LoRA配置
lora_config = LoraConfig(
    task_type=TaskType.TOKEN_CLS,
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    inference_mode=False
)

# 加载模型
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 训练参数（含动态学习率）
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    # 动态学习率配置
    lr_scheduler_type=SCHEDULER_TYPE,
    warmup_steps=WARMUP_STEPS,
    weight_decay=WEIGHT_DECAY,
    # 日志和保存配置
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=20,
    evaluation_strategy="epoch" if len(eval_dataset) > 0 else "no",
    save_strategy="epoch",
    load_best_model_at_end=True if len(eval_dataset) > 0 else False,
    metric_for_best_model="eval_macro_f1",
    greater_is_better=True,
    # 优化配置
    fp16=True,
    gradient_accumulation_steps=4,
    eval_accumulation_steps=1,
    dataloader_num_workers=0,
    report_to="none"
)

def data_collator(features):
    return {
        "input_ids": torch.stack([f["input_ids"] for f in features]),
        "attention_mask": torch.stack([f["attention_mask"] for f in features]),
        "labels": torch.stack([f["labels"] for f in features])
    }

# 创建训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset if len(eval_dataset) > 0 else None,
    data_collator=data_collator,
    compute_metrics=compute_metrics if len(eval_dataset) > 0 else None
)

# 开始训练
print("\n>>> Starting training...")
trainer.train()

# 保存最终模型
model.save_pretrained(f"{OUTPUT_DIR}/best_model")
print(f"\n>>> Training complete! Model saved to {OUTPUT_DIR}/best_model")

# 合并并保存完整模型
print("\n>>> 合并LoRA适配器到基础模型...")
merged_model = model.merge_and_unload()

merged_model.save_pretrained(f"{OUTPUT_DIR}/merged_model")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/merged_model")
print(f"\n>>> 训练完成! 合并后的模型已保存至 {OUTPUT_DIR}/merged_model")

# 保存评估结果到Excel
if len(eval_dataset) > 0:
    try:
        # 提取所有评估结果
        eval_results = []
        for log in trainer.state.log_history:
            if "eval_macro_f1" in log:
                eval_results.append({
                    "epoch": log["epoch"],
                    "macro_precision": log["eval_macro_precision"],
                    "macro_recall": log["eval_macro_recall"],
                    "macro_f1": log["eval_macro_f1"],
                    "weighted_precision": log["eval_weighted_precision"],
                    "weighted_recall": log["eval_weighted_recall"],
                    "weighted_f1": log["eval_weighted_f1"],
                    "disease_f1": log["eval_disease_f1"],
                    "drug_f1": log["eval_drug_f1"],
                })
        
        if eval_results:
            # 创建DataFrame并保存
            df = pd.DataFrame(eval_results)
            excel_path = os.path.join(OUTPUT_DIR, "Qwen2.5_NER_results_DlearnRate.xlsx")
            df.to_excel(excel_path, index=False)
            print(f"\n>>> 评估结果已保存至 {excel_path}")
        else:
            print("\n>>> 未找到有效的评估结果")
    except Exception as e:
        print(f"\n>>> 保存评估结果时出错: {str(e)}")
else:
    print("\n>>> 未进行验证集评估，跳过结果保存")

print("\n>>> 所有流程已完成！")