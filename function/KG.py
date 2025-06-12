import os
import json
from py2neo import Graph, Node, Relationship

# 检查节点是否存在的函数
def get_or_create_node(link, entity_type, entity_name):
    # 检查是否存在相同类型和名称的节点
    existing_node = link.evaluate(f"MATCH (n:{entity_type} {{name: $name}}) RETURN n", name=entity_name)
    if existing_node:
        return existing_node
    else:
        # 如果不存在，则创建新节点
        new_node = Node(entity_type, name=entity_name)
        link.create(new_node)
        return new_node

# 此函数用于读取json文件，提取其中的数据构建知识图谱
def process(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        # 提取doc_id
        doc_id = data['doc_id']
        # 遍历paragraphs并提取每个paragraph的信息
        for paragraph in data['paragraphs']:
            paragraph_id = paragraph['paragraph_id']
            paragraph_text = paragraph['paragraph']
            # print(f"\n处理字段: {paragraph_id}")
            # print(f"字段内容: {paragraph_text}")

            # 遍历sentences并提取每个sentence的信息
            for sentence in paragraph['sentences']:
                entity_nodes = {}
                # 遍历entities并提取每个entity的信息
                for entity in sentence['entities']:
                    entity_id = entity['entity_id']
                    entity_type = entity["entity_type"]
                    entity_name = entity["entity"]

                    # 创建或获取节点
                    node = get_or_create_node(link, entity_type, entity_name)
                    entity_nodes[entity_id] = node

                for relation in sentence['relations']:
                    # 创建关系
                    head_node = entity_nodes[relation["head_entity_id"]]
                    tail_node = entity_nodes[relation["tail_entity_id"]]
                    rel = Relationship(head_node, relation["relation_type"], tail_node)
                    link.create(rel)  # 将关系插入到 Neo4j 中

if __name__ == '__main__':
    # neo4j.bat console
    # 连接到 Neo4j 数据库
    link = Graph('bolt://localhost:7687', auth=('neo4j', 'wangshaner1.'))

    # 清除现有数据（确保数据库为空）
    link.run("MATCH (n) DETACH DELETE n")

    # 遍历data目录下的所有JSON文件
    data_directory = 'data'
    for filename in os.listdir(data_directory):
        if filename.endswith('.json'):
            file_path = os.path.join(data_directory, filename)
            print("开始处理文件："+file_path)
            process(file_path)

    # process('data/8.json')
    print("所有文件处理完成!")
