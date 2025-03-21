import streamlit as st
from PIL import Image
import base64
from py2neo import Graph
import pandas as pd


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


# 主界面
st.title('糖尿病图谱的构建与查询')

# 静态图谱展示
st.subheader('糖尿病图谱的构建展示')
viewerjs_image("../img/graph.png")

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