import pandas as pd

# 定义main_category和sub_category到intent的映射关系
intent_mapping = {
    "Diagnosis(诊断）": {
        "other（其他）": "A0",
        "interpretation of clinical（临床解释）": "A1",
        "symptom/manifestations（症状/表现）": "A2",
        "test（检查）": "A3",
    },
    "Treatment（治疗）": {
        "other（其他）": "B0",
        "how to use drug（如何用药）": "B1",
        "drug choice（药物选择）": "B2",
        "adverse effects of drug（药物的副作用）": "B3",
        "contraindications of drug（药物的禁忌）": "B4",
        "Other Therapy（其他治疗方法）": "B5",
        "Treatment Seeking": "B6",
    },
    "Common Knowledge（常识）": {
        "other（其他）": "C0",
        "Definition（定义)": "C1",
        "Etiology（病因学）": "C2",
        "Fertility（生育力）": "C3",
        "Hereditary（遗传性）": "C4",
    },
    "healthy lifestyle（健康生活方式）": {
        "other（其他）": "D0",
        "Diet（饮食）": "D1",
        "Exercise（锻炼）": "D2",
        "weight-losing（减肥）": "D3",
    },
    "Epidemiololgy（流行病学）": {
        "Infect（传染性）": "E1",
        "Prevention（预防）": "E2",
        "Complication（并发症）": "E3",
    },
    "Other(其他）": {
        "other（其他）": "F",
    },
}

# 创建一个新字典，用于将编码转换为其ID
intent_id = {}
count = 0
for main_cat, sub_cats in intent_mapping.items():
    for sub_cat, code in sub_cats.items():
        intent_id[code] = count
        count += 1

print(intent_id)



# 创建一个新字典，用于将编码转换为中文解释
code_to_chinese = {
    "A0": "诊断大类其他意图",
    "A1": "临床解释",
    "A2": "症状/表现",
    "A3": "检查",
    "B0": "治疗大类其他意图",
    "B1": "如何用药",
    "B2": "药物选择",
    "B3": "药物的副作用",
    "B4": "药物的禁忌",
    "B5": "其他治疗方法",
    "B6": "寻求治疗",
    "C0": "常识大类其他意图",
    "C1": "定义",
    "C2": "病因学",
    "C3": "生育力",
    "C4": "遗传性",
    "D0": "健康生活方式大类其他意图",
    "D1": "饮食",
    "D2": "锻炼",
    "D3": "减肥",
    "E1": "传染性",
    "E2": "预防",
    "E3": "并发症",
    "F": "其他大类",
}

# 读取Excel文件
df = pd.read_excel('data/intent_detection.xlsx', sheet_name=0)

# 新增intent列，并填充数据
def map_to_intent(row):
    main_category = row['main_category']
    sub_category = row['sub_category']
    return intent_mapping.get(main_category, {}).get(sub_category, "")

df['intent_explain'] = df.apply(map_to_intent, axis=1)
df["intent_id"] = df.apply(map_to_intent, axis=1)

# 将编码转换为中文解释
df['intent_explain'] = df['intent_explain'].map(code_to_chinese)

# 将编码转换为其ID
df['intent_id'] = df['intent_id'].map(intent_id)
# 保存结果到新的Excel文件
df.to_excel('data/intent_detection.xlsx', index=False)

print("处理完成，文件已保存为 'data/intent_detection.xlsx")
print(df['intent'].head())