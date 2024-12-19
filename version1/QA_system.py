from py2neo import Graph
from entity import getEntityName, DefineEntity
import torch
from intent_detection import predict_intent
from zhipuai import ZhipuAI
import warnings
# 定义疾病和药物关系映射
disease_relations = {
    "A0": ["Symptom_Disease", "Test_Disease"],
    "A1": ["Symptom_Disease","Anatomy_Disease","Reason_Disease","Pathogenesis_Disease"],
    "A2": ["Symptom_Disease","Reason_Disease","Pathogenesis_Disease","Class_Disease"],
    "A3": ["Test_Disease","Treatment_Disease","Drug_Disease","Test_Items_Disease"],
    "B0": ["Duration_Drug","Test_Items_Disease"],
    "B1": ["Frequency_Drug","Duration_Drug","Amount_Drug","Method_Drug","ADE_Drug"],
    "B2": ["Drug_Disease", "Treatment_Disease"],
    "B3": ["Amount_Drug", "ADE_Drug","Method_Drug"],
    "B4": ["ADE_Drug"],
    "B5": ["Treatment_Disease","Operation_Disese"],
    "B6": ["Operation_Disease","Drug_Disease","Treatment_Disease","Test_Items_Disease"],
    "C0": ["Symptom_Disease", "Drug_Disease"],
    "C1": ["Class_Disease"],
    "C2": ["Reason_Disease","Pathogenesis_Disease"],
    "C3": ["Anatomy_Disease","Reason_Disease","Pathogenesis_Disease"],
    "C4": ["Pathogenesis_Disease","Reason_Disease"],
    "D0": ["Pathogenesis_Disease", "Reason_Disease"],
    "D1": ["Pathogenesis_Disease", "Reason_Disease"],
    "D2": ["Pathogenesis_Disease", "Reason_Disease"],
    "D3": ["Pathogenesis_Disease", "Reason_Disease"],
    "E1":  ["Pathogenesis_Disease", "Reason_Disease","Symptom_Disease"],
    "E2":  ["Pathogenesis_Disease", "Reason_Disease","Symptom_Disease"],
    "E3":  ["Pathogenesis_Disease", "Reason_Disease","Symptom_Disease"],


}


# 实体查找
def get_entity_category(entity_name, entity_dict):
    for category, entities in entity_dict.items():
        if entity_name in entities:
            return category
    return None  # 如果没有找到，返回None
# 映射编号到关系
def map_relations(outputs):
    mapped_relations = []
    for output in outputs:
        if output in disease_relations:
            mapped_relations.append((output, disease_relations[output]))
        else:
            mapped_relations.append((output, "Unknown_Relation"))
    return mapped_relations

def query_relation(relations, entity,category):
    result_dict = {}
    for relation in relations:
        query_result = link.run(
            """
            MATCH (n:%s{name: $entity_name})-[m:`%s`]-(s)
            RETURN s.name AS related_entity
            """ % (category, relation),
            entity_name=entity
        )
        result_dict[relation] = [record["related_entity"] for record in query_result]
    return result_dict

def ask_glm(content):
    # 初始化ZhipuAI客户端，填写你的API Key
    client = ZhipuAI(api_key="f037f80bb31908ef8d2159b5e4f17e6d.kDBaHZjpplHWU6pl")

    # 调用ZhipuAI的chat.completions接口，传入参数
    response = client.chat.completions.create(
        model="glm-4-flash",  # 指定模型名称
        messages=[
            {"role": "user", "content": content}  # 使用传入的content作为对话的用户输入
        ],
    )

    # 返回响应中的生成结果
    return response.choices[0].message

if __name__ == '__main__':
   # 忽略warning
   #  warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore")   # 忽略所有

    # 连接到 Neo4j 数据库
   # neo4j.bat console
    link = Graph('bolt://localhost:7687', auth=('neo4j', 'wangshaner1.'))
    # 问句
    query = input()
    # # 实体识别
    entityName = getEntityName(query)
    entity_dict = DefineEntity(link)
    category = get_entity_category(entityName, entity_dict)
    print('**********************************************************')
    print("问句为：" + query)
    print("提取到的实体为：" + entityName )
    print('类别为：' + category)

    # 意图识别
    data_path = "data/intent_detection.xlsx"  # 数据路径
    save_path = "saved_model"  # 模型保存路径
    max_len = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    intent = predict_intent(query, save_path, device, max_len)
    print(f"预测意图：{intent}")


    # 显示结果
    # 输入编号列表，例如模型输出
    model_outputs = []
    model_outputs.append(intent)
    result = map_relations(model_outputs)
    for code, relation in result:
        print(f"编号: {code} -> 关系: {relation}")

    # 查询图谱
    results =query_relation(relation, entityName,category)
    print(results)

    # LLM prompt
    prompt = '''
    你好，我需要你的帮助。请扮演一名资深的糖尿病专家，现在病人对你提问，而你要根据我提供的信息回答问题。
    以下是病人的问题：
    %s
    此外，我还查询了专业的知识图谱，所以可以给你了以下提示信息（是一个字典，key为关系代表可能问题关键词存在某种关系，value为实体代表一些专业知识）：
    %s
    请你结合以上问句和提示，基于糖尿病领域的专业知识，为病人提供一个详细、准确且易于理解的回答，请务必注意不需要对我的提示信息进行回答，只需要将我的提示信息融入到你对病人的回答即可。
    ''' % (query, results)
    answer = ask_glm(prompt).content
    print(answer)
