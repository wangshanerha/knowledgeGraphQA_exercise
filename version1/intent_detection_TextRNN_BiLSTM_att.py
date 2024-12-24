import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer
import numpy as np


# ===================== 数据加载与预处理 =====================
def load_and_preprocess_data(filepath, max_length):
    df = pd.read_excel(filepath)

    # 提取text和label
    texts = df['question'].values
    labels = df['intent'].values

    # 标签编码
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # 分词器（使用BertTokenizer作为示例）
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    # 将文本转换为token id
    def tokenize_text(text):
        return tokenizer.encode(text, add_special_tokens=True)

    # 补齐序列
    def pad_sequences(sequences, maxlen):
        return np.array([seq + [0] * (maxlen - len(seq)) if len(seq) < maxlen else seq[:maxlen] for seq in sequences])

    tokenized_texts = [tokenize_text(text) for text in texts]
    tokenized_texts = pad_sequences(tokenized_texts, max_length)

    return tokenized_texts, labels, label_encoder


class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = torch.tensor(texts)
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


# ===================== Bi-LSTM with Attention =====================
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_out):
        # lstm_out: [batch_size, seq_len, hidden_dim * 2]
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # [batch_size, seq_len, 1]
        context = torch.sum(attn_weights * lstm_out, dim=1)  # [batch_size, hidden_dim * 2]
        return context


class BiLSTMWithAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=2, dropout=0.5):
        super(BiLSTMWithAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=True, dropout=dropout, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)  # lstm_out: [batch_size, seq_len, hidden_dim * 2]
        context = self.attention(lstm_out)  # [batch_size, hidden_dim * 2]
        out = self.dropout(context)
        out = self.fc(out)
        return out


# ===================== 训练与评估 =====================
def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for texts, labels in train_loader:
        texts, labels = texts.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total
    return total_loss / len(train_loader), accuracy


def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return total_loss / len(test_loader), accuracy


# ===================== 主函数 =====================
if __name__ == "__main__":
    # 参数定义
    filepath = 'data/intent_detection.xlsx'
    max_length = 50
    batch_size = 32
    embedding_dim = 128
    hidden_dim = 128
    n_layers = 2
    dropout = 0.5
    num_epochs = 10
    learning_rate = 0.001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据加载与预处理
    tokenized_texts, labels, label_encoder = load_and_preprocess_data(filepath, max_length)
    X_train, X_test, y_train, y_test = train_test_split(tokenized_texts, labels, test_size=0.2, random_state=42)

    train_dataset = TextDataset(X_train, y_train)
    test_dataset = TextDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 模型定义
    vocab_size = len(BertTokenizer.from_pretrained('bert-base-chinese').vocab)
    output_dim = len(label_encoder.classes_)
    model = BiLSTMWithAttention(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练与评估
    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"  Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # 保存模型
    torch.save(model.state_dict(), 'bilstm_with_attention.pth')
    print("Model saved to 'bilstm_with_attention.pth'")
