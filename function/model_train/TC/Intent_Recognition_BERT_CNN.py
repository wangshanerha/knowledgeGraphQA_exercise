import torch
import pandas as pd
import numpy as np
import os
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 参数配置
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 2e-5
NUM_CLASSES = 24
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "../../saved_models/Intent_Recognition_BERT_CNN/best_model.bin"
RESULTS_SAVE_PATH = "../../assess/Intent_Recognition_BERT_CNN.xlsx"

# 确保目录存在
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(os.path.dirname(RESULTS_SAVE_PATH), exist_ok=True)

# 数据预处理
class DiabetesDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=False,
            return_attention_mask=True
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# BERT+CNN模型
class BERT_CNN_Classifier(nn.Module):
    def __init__(self, n_classes):
        super(BERT_CNN_Classifier, self).__init__()
        self.bert = BertModel.from_pretrained("../../models/bert-base-chinese")
        self.dropout = nn.Dropout(0.3)

        # CNN层配置
        self.conv1 = nn.Conv1d(768, 128, 3, padding=1)
        self.conv2 = nn.Conv1d(128, 64, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()

        # 全连接层
        seq_len_after_pool = MAX_LEN // 4
        self.fc = nn.Linear(64 * seq_len_after_pool, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        last_hidden_state = outputs.last_hidden_state
        x = last_hidden_state.permute(0, 2, 1)

        # CNN处理
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)

# 训练函数
def train_model(model, data_loader, optimizer, device):
    model.train()
    losses = []
    correct_predictions = 0

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

    # 转换为Python标量
    accuracy = correct_predictions.double().item() / len(data_loader.dataset)
    return accuracy, np.mean(losses)

# 验证函数
def eval_model(model, data_loader, device):
    model.eval()
    losses = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            losses.append(loss.item())

    # 计算评估指标
    avg_loss = np.mean(losses)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # 转换为Python float
    return (
        float(avg_loss),
        float(accuracy),
        float(precision),
        float(recall),
        float(f1)
    )

# 主流程
def main():
    # 加载数据
    df = pd.read_excel("../../data/intent_detection.xlsx", engine='openpyxl')
    texts = df['question'].values
    labels = df['intent_id'].values

    # 分层划分数据集
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, 
        test_size=0.2, 
        random_state=42,
        stratify=labels  # 添加分层抽样
    )

    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained("../../models/bert-base-chinese")

    # 创建DataLoader
    train_dataset = DiabetesDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_dataset = DiabetesDataset(val_texts, val_labels, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # 初始化模型
    model = BERT_CNN_Classifier(NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 初始化结果记录
    results = {
        'epoch': [],
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }

    best_accuracy = 0.0
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        # 训练
        train_acc, train_loss = train_model(model, train_loader, optimizer, DEVICE)
        print(f'Train loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}')

        # 验证
        val_loss, val_acc, val_precision, val_recall, val_f1 = eval_model(model, val_loader, DEVICE)
        print(f'Validation loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}')
        print(f'Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}\n')

        # 记录结果
        results['epoch'].append(epoch+1)
        results['train_loss'].append(float(train_loss))
        results['train_accuracy'].append(float(train_acc))
        results['val_loss'].append(val_loss)
        results['val_accuracy'].append(val_acc)
        results['val_precision'].append(val_precision)
        results['val_recall'].append(val_recall)
        results['val_f1'].append(val_f1)

        # 保存最佳模型
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            best_accuracy = val_acc
            print(f"New best model saved with accuracy {best_accuracy:.4f}")

    # 保存评估结果
    results_df = pd.DataFrame(results)
    results_df.to_excel(RESULTS_SAVE_PATH, index=False)
    print(f"Training metrics saved to {RESULTS_SAVE_PATH}")

if __name__ == '__main__':
    main()