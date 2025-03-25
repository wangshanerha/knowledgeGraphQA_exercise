import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 定义标签映射表（根据分类体系）
label_map1 = [
    "A：诊断 A0：其他",
    "A：诊断 A1：interpretation of clinical（临床解释）",
    "A：诊断 A2：症状/表现",
    "A：诊断 A3：test（检查）",
    "B：治疗 B0：其他",
    "B：治疗 B1：how to use drug（如何用药）",
    "B：治疗 B2：drug choice（药物选择）",
    "B：治疗 B3：药物不良反应（药物副作用）",
    "B：治疗 B4：药物禁忌",
    "B：治疗 B5：Other Therapy（其他治疗方法）",
    "B：治疗 B6：Treatment Seeking（寻求治疗）",
    "C：常识 C0：other（其他）",
    "C：常识 C1：定义",
    "C：常识 C2：病因学",
    "C：常识 C3：Fertility（生育力）",
    "C：常识 C4：Hereditary（遗传性）",
    "D：健康生活方式 D0：other（其他）",
    "D：健康生活方式 D1：饮食",
    "D：健康生活方式 D2：Exercise（锻炼）",
    "D：健康生活方式 D3：weight-losing（减肥）",
    "E：流行病学 E1：感染（传染性）",
    "E：流行病学 E2：Prevention（预防）",
    "E：流行病学 E3：并发症",
    "F：其他 -"
]
label_map = [
    "A0：其他",
    "A1：interpretation of clinical（临床解释）",
    "A2：症状/表现",
    "A3：test（检查）",
    "B0：其他",
    "B1：how to use drug（如何用药）",
    "B2：drug choice（药物选择）",
    "B3：药物不良反应（药物副作用）",
    "B4：药物禁忌",
    "B5：Other Therapy（其他治疗方法）",
    "B6：Treatment Seeking（寻求治疗）",
    "C0：other（其他）",
    "C1：定义",
    "C2：病因学",
    "C3：Fertility（生育力）",
    "C4：Hereditary（遗传性）",
    "D0：other（其他）",
    "D1：饮食",
    "D2：Exercise（锻炼）",
    "D3：weight-losing（减肥）",
    "E1：感染（传染性）",
    "E2：Prevention（预防）",
    "E3：并发症",
    "F：其他 -"
]


# 定义预测函数
def predict(text, model_path):
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()

    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(model.device)
    attention_mask = encoding["attention_mask"].to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).item()

    return label_map[preds]  # 返回映射后的标签名称


if __name__ == "__main__":
    model_path = "chinese-electra-large-Diabetes-question-intent"
    while True:
        text = input("请输入文本进行预测（输入 'end' 退出）：")
        if text.lower() == "end":
            print("退出预测程序。")
            break

        intent_name = predict(text, model_path)
        print(f"预测的意图类别：{intent_name}")