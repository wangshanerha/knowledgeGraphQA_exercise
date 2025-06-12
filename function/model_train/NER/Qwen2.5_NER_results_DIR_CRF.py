import json
import os
import numpy as np
import pandas as pd
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
from torchcrf import CRF  # 新增CRF导入

class CRFModelWrapper(torch.nn.Module):
    """CRF模型包装器"""
    def __init__(self, base_model, num_labels):
        super().__init__()
        self.base_model = base_model
        self.crf = CRF(num_labels, batch_first=True)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        if labels is not None:
            mask = attention_mask.bool()
            loss = -self.crf(logits, labels, mask=mask, reduction='mean')
            return {'loss': loss, 'logits': logits}
        return {'logits': logits}

    def decode(self, logits, attention_mask):
        return self.crf.decode(logits, mask=attention_mask.bool())

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
    """支持CRF的评估函数"""
    model.eval()
    predictions = []
    labels = []
    
    # 获取原始数据
    logits = torch.tensor(p.predictions)
    label_ids = p.label_ids
    
    # 解码预测结果
    for i in range(len(logits)):
        mask = (label_ids[i] != -100)
        valid_logits = logits[i][mask.unsqueeze(0).bool()]
        preds = model.module.decode(valid_logits.unsqueeze(0), mask)
        predictions.extend(preds[0])
        
        valid_labels = label_ids[i][mask]
        labels.append(valid_labels.tolist())
    
    # 转换标签ID到文本
    true_labels = [[id2label[l] for l in seq] for seq in labels]
    true_predictions = [[id2label[p] for p in seq] for seq in predictions]

    # 生成评估报告
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
OUTPUT_DIR = "../../saved_models/Qwen2.5_NER_results_DlearnRate_CRF"
BATCH_SIZE = 16
EPOCHS = 15  # 减少训练轮次
LEARNING_RATE = 9e-5
MIN_LR = 1e-6  # 新增最小学习率
WARMUP_RATIO = 0.1  # 比例制预热
WEIGHT_DECAY = 0.005  # 调整权重衰减
SCHEDULER_TYPE = "cosine_with_restarts"  # 带重启的cosine
LOGGING_STEPS = 50

# 初始化组件
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 创建数据集
train_dataset, eval_dataset, label2id, id2label = NERDataset.create_datasets(
    DATA_PATH, tokenizer, test_size=0.2
)
label_list = list(label2id.keys())
id2label = {v: k for k, v in label2id.items()}  # 添加反向映射

# 保存标签配置
os.makedirs(f"{OUTPUT_DIR}/label_config", exist_ok=True)
with open(f"{OUTPUT_DIR}/label_config/id2label.json", "w") as f:
    json.dump(id2label, f)
with open(f"{OUTPUT_DIR}/label_config/label2id.json", "w") as f:
    json.dump(label2id, f)

# 初始化基础模型
base_model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# 创建CRF包装模型
model = CRFModelWrapper(base_model, len(label2id))

# 应用LoRA配置
lora_config = LoraConfig(
    task_type=TaskType.TOKEN_CLS,
    r=8,  
    lora_alpha=64,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="lora_only",
    modules_to_save=["classifier"]  # 保持分类层全参数
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 配置训练参数
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    lr_scheduler_type=SCHEDULER_TYPE,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=WEIGHT_DECAY,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=LOGGING_STEPS,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_macro_f1",
    greater_is_better=True,
    fp16=True,
    gradient_accumulation_steps=8,  # 增大有效批次
    lr_scheduler_kwargs={
        "num_cycles": 2,  # 重启次数
        "min_lr": MIN_LR
    },
    dataloader_num_workers=2,
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
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# 开始训练
print("\n>>> Starting CRF-enhanced training...")
trainer.train()

# 模型保存
model.save_pretrained(f"{OUTPUT_DIR}/best_model")
print(f"\n>>> Training complete! Model saved to {OUTPUT_DIR}/best_model")

# 合并LoRA适配器
print("\n>>> Merging LoRA adapters...")
merged_model = model.merge_and_unload()
merged_model.base_model.save_pretrained(f"{OUTPUT_DIR}/merged_model")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/merged_model")
print(f">>> Merged model saved to {OUTPUT_DIR}/merged_model")


if len(eval_dataset) > 0:
    try:
        # 收集训练损失（处理浮点epoch）
        train_loss_dict = {}
        for log in trainer.state.log_history:
            if 'loss' in log and 'epoch' in log:
                epoch = int(log['epoch'])
                train_loss_dict[epoch] = log['loss']

        # 收集评估结果
        eval_results = []
        for log in trainer.state.log_history:
            if "eval_macro_f1" in log and 'epoch' in log:
                epoch = int(log['epoch'])
                entry = {
                    "epoch": epoch,
                    "train_loss": train_loss_dict.get(epoch, None),
                    "eval_loss": log.get("eval_loss", 0.0),
                    "macro_precision": log.get("eval_macro_precision", 0.0),
                    "macro_recall": log.get("eval_macro_recall", 0.0),
                    "macro_f1": log.get("eval_macro_f1", 0.0),
                    "weighted_precision": log.get("eval_weighted_precision", 0.0),
                    "weighted_recall": log.get("eval_weighted_recall", 0.0),
                    "weighted_f1": log.get("eval_weighted_f1", 0.0),
                    "disease_f1": log.get("eval_disease_f1", 0.0),
                    "drug_f1": log.get("eval_drug_f1", 0.0)
                }
                eval_results.append(entry)

        if eval_results:
            df = pd.DataFrame(eval_results)
            # 按epoch排序并填充缺失值
            df = df.sort_values('epoch').ffill()
            
            # 创建评估目录并保存
            assess_dir = "../../assess/"
            os.makedirs(assess_dir, exist_ok=True)
            excel_path = os.path.join(assess_dir, "Qwen2.5_NER_results_DlearnRate_CRF.xlsx")
            df.to_excel(excel_path, index=False)
            print(f"\n>>> 评估结果已保存至 {excel_path}")

    except Exception as e:
        print(f"\n>>> 保存评估结果时出错: {str(e)}")

print("\n>>> 所有流程已完成！")