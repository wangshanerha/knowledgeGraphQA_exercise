import torch
import pandas as pd
from transformers import BertModel, BertTokenizer


class IntentPredictor:
    def __init__(self):
        # 配置参数
        self.MAX_LEN = 128
        self.MODEL_SAVE_PATH = "function/saved_models/Intent_Recognition_BERT_CNN/best_model.bin"
        self.NUM_CLASSES = 24
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化组件
        self.tokenizer = None
        self.model = None
        self.id_to_name = {}

        # 加载资源
        self._load_resources()

    def _load_resources(self):
        """加载模型、tokenizer和标签映射"""
        # 加载标签映射
        df = pd.read_excel("function/data/intent_detection.xlsx", engine='openpyxl')
        self.id_to_name = df[['intent_id', 'intent']].drop_duplicates().set_index('intent_id')['intent'].to_dict()

        # 加载tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("function/models/bert-base-chinese")

        # 初始化模型
        self.model = BERT_CNN_Classifier(self.NUM_CLASSES).to(self.DEVICE)
        self.model.load_state_dict(torch.load(self.MODEL_SAVE_PATH, map_location=self.DEVICE))
        self.model.eval()

    def predict(self, text: str) -> str:
        """
        预测文本意图
        :param text: 输入文本（中文）
        :return: 意图名称字符串
        """
        # 预处理
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True
        )

        # 推理
        with torch.no_grad():
            input_ids = encoding['input_ids'].to(self.DEVICE)
            attention_mask = encoding['attention_mask'].to(self.DEVICE)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            _, pred_id = torch.max(outputs, dim=1)

        # 转换ID为名称
        return self.id_to_name.get(pred_id.item(), "未知意图")


# 模型定义需要保持与训练时一致
class BERT_CNN_Classifier(torch.nn.Module):
    def __init__(self, n_classes):
        super(BERT_CNN_Classifier, self).__init__()
        self.bert = BertModel.from_pretrained("function/models/bert-base-chinese")
        self.dropout = torch.nn.Dropout(0.3)
        self.conv1 = torch.nn.Conv1d(768, 128, 3, padding=1)
        self.conv2 = torch.nn.Conv1d(128, 64, 3, padding=1)
        self.pool = torch.nn.MaxPool1d(2)
        self.relu = torch.nn.ReLU()
        seq_len_after_pool = 128 // 4  # MAX_LEN=128
        self.fc = torch.nn.Linear(64 * seq_len_after_pool, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)


# 单例实例化
# predictor = IntentPredictor()
# print(predictor.predict("糖尿病有什么症状？"))