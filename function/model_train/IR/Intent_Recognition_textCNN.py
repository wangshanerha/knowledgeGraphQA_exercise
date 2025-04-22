import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import os

# 加载数据集
def load_dataset(file_path):
    data = pd.read_excel(file_path)
    texts = data['question'].tolist()
    labels = data['intent'].tolist()
    return texts, labels

# 数据预处理
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label_encoder, max_len):
        self.texts = texts
        self.labels = label_encoder.transform(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.texts[idx])
        if len(tokens) < self.max_len:
            tokens += [0] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]
        return torch.tensor(tokens), torch.tensor(self.labels[idx])

# 简单的分词器
def simple_tokenizer(text):
    vocab = {word: idx + 1 for idx, word in enumerate(set("".join(texts)))}
    return [vocab[char] for char in text if char in vocab]

# TextCNN模型
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, kernel_sizes, num_filters):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) for k in kernel_sizes
        ])
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)  # (batch_size, max_len, embed_dim)
        x = x.unsqueeze(1)  # (batch_size, 1, max_len, embed_dim)
        x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(batch_size, num_filters, max_len-kernel_size+1), ...]
        x = [torch.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(batch_size, num_filters), ...]
        x = torch.cat(x, 1)  # (batch_size, num_filters * len(kernel_sizes))
        x = self.dropout(x)
        x = self.fc(x)
        return x

# 模型训练

def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.tolist())
        all_targets.extend(targets.tolist())
    accuracy = accuracy_score(all_targets, all_preds)
    return total_loss / len(dataloader), accuracy

# 模型评估
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.tolist())
    accuracy = accuracy_score(all_targets, all_preds)
    return total_loss / len(dataloader), accuracy

# 模型保存
def save_model(model, path):
    torch.save(model.state_dict(), path)

# 模型加载
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

# 预测函数
def predict(model, texts, tokenizer, max_len):
    model.eval()
    tokens_list = []
    for text in texts:
        tokens = tokenizer(text)
        if len(tokens) < max_len:
            tokens += [0] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]
        tokens_list.append(tokens)
    inputs = torch.tensor(tokens_list)
    with torch.no_grad():
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
    return preds

if __name__ == "__main__":
    # 数据集加载
    file_path = "data/intent_detection.xlsx"
    texts, labels = load_dataset(file_path)

    # 标签编码
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    # 数据分割
    max_len = 20
    tokenizer = simple_tokenizer
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, label_encoder, max_len)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, label_encoder, max_len)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 模型初始化
    vocab_size = len(set("".join(texts))) + 1
    embed_dim = 50
    num_classes = len(label_encoder.classes_)
    kernel_sizes = [3, 4, 5]
    num_filters = 16
    model = TextCNN(vocab_size, embed_dim, num_classes, kernel_sizes, num_filters)

    # 损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 模型训练与验证
    epochs = 20
    model_save_path = "textcnn_model.pth"
    for epoch in range(epochs):
        train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # 保存模型
    save_model(model, model_save_path)
    print(f"Model saved to {model_save_path}")

    # 使用训练好的模型进行预测
    new_texts = ["糖尿病患者可以吃什么水果", "1型糖尿病能活多久"]
    predictions = predict(model, new_texts, tokenizer, max_len)
    label_names = label_encoder.inverse_transform(predictions.numpy())
    print("Predictions:", list(zip(new_texts, label_names)))


"""
max_len 控制文本长度；
embed_dim 控制词嵌入维度；
num_classes 控制分类的类别数；
kernel_sizes 控制卷积核的大小；
num_filters 控制卷积核的数量；
dropout 控制 Dropout 层的丢弃率；
batch_size 控制每次训练的样本数；
learning_rate 控制梯度下降的步长；
epochs 控制训练的周期数；
model_save_path 控制模型保存的路径。

"""