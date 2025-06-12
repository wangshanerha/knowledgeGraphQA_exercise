import streamlit as st
from PIL import Image
import base64
from py2neo import Graph
import pandas as pd

# neo4j.bat console
def viewerjs_image(image_path):
    with open(image_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode()

    html = f"""
    <link href="https://cdnjs.cloudflare.com/ajax/libs/viewerjs/1.11.3/viewer.min.css" rel="stylesheet">
    <img id="image" src="data:image/png;base64,{img_data}" alt="Picture" style="max-width: 100%; height: auto;">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/viewerjs/1.11.3/viewer.min.js"></script>
    <script>
        const viewer = new Viewer(document.getElementById('image'), {{
            inline: false,
            toolbar: {{
                zoomIn: 1,
                zoomOut: 1,
                oneToOne: 1,
                reset: 1,
                rotateLeft: 1,
                rotateRight: 1,
                flipHorizontal: 1,
                flipVertical: 1,
            }},
        }});
    </script>
    """
    st.components.v1.html(html, height=600)


@st.cache_resource
def KG_load():
    try:
        return Graph('bolt://localhost:7687',
                     auth=('neo4j', 'wangshaner1.'),
                     name='neo4j')
    except Exception as e:
        st.error(f"数据库连接失败: {str(e)}")
        return None

def neo4j_query(graph,entities, relations,relation_dict):
    # print(entities)
    # print(relations)
    results1, results2 = [], []
    # 处理单实体查询
    for entity in entities:
        for rel in relations:
            cypher = """
                    MATCH (d)-[r:%s]-(n)
                    WHERE d.name=$entity
                    RETURN type(r) AS relation, collect(n.name)[0..5] AS values
                    """ % rel
            data = graph.run(cypher, entity=entity).data()
            for dict_kg in data:
                results1.append({
                    "实体": entity,
                    "对应关系": relation_dict[dict_kg["relation"]],
                    "参考知识": dict_kg["values"]
                })
    # print("results1结果如下：")
    # print(results1)
    if len(entities) >= 2:
        # 遍历所有实体对
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                entity1 = entities[i]
                entity2 = entities[j]
                # print(entity1, entity2)
                # 查询直接关系
                direct_relations = []
                cypher_direct = """
                MATCH (e1)-[r]-(e2)
                WHERE e1.name = $entity1 AND e2.name = $entity2
                RETURN type(r) AS relation
                """
                data = graph.run(cypher_direct, entity1=entity1, entity2=entity2, relations=relations).data()
                # print(data)
                if data:
                    for record in data:
                        results2.append({
                            "实体对": f"{entity1} 与 {entity2}",
                            "关系类型": relation_dict[record["relation"]],
                            "关联信息": "直接关系",
                            "路径": "空"
                        })
                    # print(results2)
                else:
                    # 查询间接关系
                    cypher_indirect = """
                        MATCH path = (e1)-[rels*..3]-(e2)
                        WHERE e1.name = $entity1 AND e2.name = $entity2
                        AND ALL(r IN rels WHERE type(r) IN $relations)
                        RETURN 
                        [r IN rels | type(r)] AS relations,
                        [n IN nodes(path) | n.name] AS nodes,
                        length(path) AS hops
                        ORDER BY hops
                        LIMIT 5
                        """
                    indirect_data = graph.run(cypher_indirect, entity1=entity1, entity2=entity2,
                                              relations=relations).data()

                    if indirect_data:
                        for record in indirect_data:
                            relations_in_path = record["relations"]
                            translated_rels = [relation_dict[rel] for rel in relations_in_path]  # 转换关系类型

                            path_rels = record["relations"]  # 直接获取关系类型列表
                            path_nodes = [node for node in record["nodes"]
                                          if node not in (entity1, entity2)]

                            results2.append({
                                "实体对": f"{entity1} ↔ {entity2}",
                                "关联路径": " → ".join(translated_rels),  # 那个中文关系映射
                                "途经节点": [node for node in record["nodes"] if node not in (entity1, entity2)],
                                "跳数": record["hops"],
                                "关联类型": "直接" if record["hops"] == 1 else f"{record['hops']}跳关联"
                            })
                        # print(results2)
                    else:
                        results2.append({
                            "实体对": f"{entity1} 与 {entity2}",
                            "关系类型": "未知",
                            "关联信息": "未找到相关路径",
                            "路径": "空"
                        })

    # print("results2结果如下：")
    # print(results2)

    return results1,results2


# 主界面
st.title('糖尿病图谱的构建与查询')

# 静态图谱展示
st.subheader('糖尿病图谱的构建展示')
viewerjs_image("img/graph.png")

# 查询模块
st.subheader('🔍 知识图谱查询')
graph = KG_load()

if graph:
        # 输入查询语句
    cypher_query = st.text_input(
            "输入Cypher查询语句（示例：MATCH (d:Disease) WHERE d.name CONTAINS '糖尿病' RETURN d.name AS 疾病名称）",
            value="MATCH (d:Disease) WHERE d.name CONTAINS '糖尿病' RETURN d.name AS 疾病名称"
        )

        # 执行查询按钮
    if st.button("执行查询"):
        try:
                # 执行查询并转换为DataFrame
            result = graph.run(cypher_query).to_data_frame()

            # 显示结果
            if not result.empty:
                st.success("查询成功！找到 {} 条记录".format(len(result)))
                st.dataframe(result, use_container_width=True)
            else:
                st.warning("查询成功，但未找到匹配结果")

        except Exception as e:
            st.error(f"查询失败：{str(e)}")
else:
        st.error("无法连接到数据库，请检查Neo4j服务状态")