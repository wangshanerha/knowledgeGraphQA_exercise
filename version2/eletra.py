import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import torch

# 数据路径
data_path = "data/intent_detection.xlsx"  # 数据路径

# 加载数据
df = pd.read_excel(data_path)
data = df[["question", "intent_id"]].rename(columns={"question": "text", "intent_id": "label"})

# 将数据转换为Dataset对象
dataset = Dataset.from_pandas(data)

# 加载预训练模型和分词器
model_name = "D:/pycharm/example/nlp/doc/model/electra"  # 替换为你的模型路径
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(data["label"].unique()))
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 数据预处理
def preprocess_function(examples):
    # 确保所有输入特征的长度一致
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # 替换为 eval_strategy
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
)

# 定义Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,  # 提供一个验证数据集
)

# 开始训练
trainer.train()

# 保存微调后的模型
trainer.save_model("./fine_tuned_model")

# 推理示例
while True:
    text = input("请输入问题（输入'exit'退出）：")
    if text.lower() == "exit":
        break

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)

    intent_id = predictions.item()
    intent = df[df["intent_id"] == intent_id]["intent"].values[0]
    intent_explain = df[df["intent_id"] == intent_id]["intent_explain"].values[0]

    print(f"输入问题: {text}")
    print(f"预测意图: {intent} ({intent_id})")
    print(f"意图解释: {intent_explain}")