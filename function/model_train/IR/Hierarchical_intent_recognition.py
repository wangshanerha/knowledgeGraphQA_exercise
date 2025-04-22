import torch
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    BertTokenizer,  # 修改点1：替换tokenizer
    BertModel,       # 修改点2：替换模型
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset

# 配置参数
MODEL_NAME    = "../../models/bert-base-chinese"  # 修改点3：模型路径
DATA_PATH     = "../../data/intent_detection.xlsx"
MAX_LENGTH    = 64
TRAIN_RATIO   = 0.8
LEARNING_RATE = 2e-5
NUM_EPOCHS    = 3

class IntentDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        main_labels = [f.pop("main_labels") for f in features]
        sub_labels  = [f.pop("sub_labels") for f in features]
        batch = super().__call__(features)
        batch["labels"] = torch.stack([
            torch.tensor(main_labels, dtype=torch.long),
            torch.tensor(sub_labels, dtype=torch.long)
        ], dim=1)
        return batch

def preprocess_data(df: pd.DataFrame):
    df = df[["question", "main_intent", "intent"]].dropna()
    main_enc = LabelEncoder().fit(df["main_intent"])
    sub_enc  = LabelEncoder().fit(df["intent"])
    df["main_label"] = main_enc.transform(df["main_intent"])
    df["sub_label"]  = sub_enc.transform(df["intent"])

    M = len(main_enc.classes_)
    S = len(sub_enc.classes_)
    constraint_matrix = torch.zeros((M, S), dtype=torch.bool)
    for m, s in zip(df["main_label"], df["sub_label"]):
        constraint_matrix[m, s] = True

    return df, main_enc, sub_enc, constraint_matrix

class HierarchicalBERT(torch.nn.Module):  # 修改点4：类名调整
    def __init__(self, main_classes: int, sub_classes: int):
        super().__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)  # 修改点5：替换为BERT
        hidden_size    = self.bert.config.hidden_size
        self.main_head = torch.nn.Linear(hidden_size, main_classes)
        self.sub_head  = torch.nn.Linear(hidden_size, sub_classes)
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
        # 冻结部分层（可选）
        for param in self.bert.parameters():
            param.requires_grad = True  # 根据显存情况调整

    def forward(self, input_ids, attention_mask, **kwargs):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False
        )
        cls_vec = out.last_hidden_state[:, 0, :]
        return {
            "main_logits": self.main_head(cls_vec),
            "sub_logits":  self.sub_head(cls_vec)
        }

class HierarchicalTrainer(Trainer):
    def __init__(self, constraint_matrix: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraint_matrix = constraint_matrix.to(self.model.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        main_labels = labels[:, 0]
        sub_labels  = labels[:, 1]

        outputs     = model(**inputs)
        main_logits = outputs["main_logits"]
        sub_logits  = outputs["sub_logits"]

        main_loss = torch.nn.functional.cross_entropy(main_logits, main_labels)
        
        mask = self.constraint_matrix[main_labels]
        masked_sub_logits = sub_logits.masked_fill(~mask, -1e9)
        sub_loss = torch.nn.functional.cross_entropy(masked_sub_logits, sub_labels)

        total_loss = 0.7 * main_loss + 0.3 * sub_loss
        if return_outputs:
            return total_loss, outputs
        return total_loss

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = model(**inputs)
        
        main_logits = outputs["main_logits"].detach().cpu()
        sub_logits  = outputs["sub_logits"].detach().cpu()
        labels = inputs.get("labels")
        if labels is not None:
            labels = labels.detach().cpu()

        loss = None
        if labels is not None:
            loss = self.compute_loss(model, inputs)

        if prediction_loss_only:
            return (loss, None, None)

        return (
            loss,
            (main_logits, sub_logits),
            labels
        )

def compute_metrics(eval_pred):
    main_logits, sub_logits = eval_pred.predictions
    labels = eval_pred.label_ids

    main_logits = main_logits.numpy()
    sub_logits = sub_logits.numpy()
    labels = labels.numpy()

    main_labels = labels[:, 0]
    sub_labels  = labels[:, 1]

    main_preds = np.argmax(main_logits, axis=1)
    sub_preds  = np.argmax(sub_logits, axis=1)

    return {
        "main_accuracy": accuracy_score(main_labels, main_preds),
        "sub_accuracy":  accuracy_score(sub_labels, sub_preds),
        "main_f1":       f1_score(main_labels, main_preds, average="weighted"),
        "sub_f1":        f1_score(sub_labels, sub_preds, average="weighted")
    }

def main():
    try:
        raw_df = pd.read_excel(DATA_PATH, engine="openpyxl")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    df, main_enc, sub_enc, constraint_matrix = preprocess_data(raw_df)
    train_df, val_df = train_test_split(
        df, 
        test_size=1 - TRAIN_RATIO,
        stratify=df["main_label"],
        random_state=42
    )

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)  # 修改点6：BERT tokenizer
    
    def prepare_dataset(dframe):
        ds = Dataset.from_pandas(dframe.reset_index(drop=True))
        def tok_fn(examples):
            return tokenizer(
                examples["question"],
                truncation=True,
                max_length=MAX_LENGTH,
                return_token_type_ids=False  # BERT默认需要token_type_ids，但中文任务通常不需要
            )
        return ds.map(
            tok_fn, 
            batched=True,
            remove_columns=["question", "main_intent", "intent"]
        ).add_column("main_labels", dframe["main_label"].tolist()) \
         .add_column("sub_labels", dframe["sub_label"].tolist())

    train_ds = prepare_dataset(train_df)
    val_ds   = prepare_dataset(val_df)

    training_args = TrainingArguments(
        output_dir="../../saved_models/Hierarchical_BERT_result",  # 修改点7：输出目录
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=LEARNING_RATE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=1,
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False
    )

    model = HierarchicalBERT(  # 修改点8：使用BERT类
        main_classes=len(main_enc.classes_),
        sub_classes=len(sub_enc.classes_)
    )
    
    trainer = HierarchicalTrainer(
        constraint_matrix=constraint_matrix,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=IntentDataCollator(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )

    print("启动层次化意图识别训练(BERT)...")
    trainer.train()
    
    save_path = "../../saved_models/Hierarchical_BERT_result"
    os.makedirs(save_path, exist_ok=True)
    trainer.save_model(save_path)
    print(f"模型已保存至: {save_path}")

if __name__ == "__main__":
    main()