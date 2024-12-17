# 相关模块导入
import jieba.posseg as pseg
import jieba

from fuzzywuzzy import fuzz
from py2neo import Graph
from py2neo import NodeMatcher,RelationshipMatcher

## 问句实体的提取
## 结巴分词参考文献：https://blog.csdn.net/smilejiasmile/article/details/80958010
def getEntityName(text):
    '''
    :param text:用户输入内容
    :return: 输入内容中的实体名称
    '''
    EntityName = ''
    jieba.load_userdict('selfDefine.txt')
    words =pseg.cut(text)
    print(words)
    for w in words:
        ## 提取对话中的实体名称
        if w.flag == 'label':
            EntityName = w.word
    return EntityName

# 自定义实体文件
def DefineEntity(link):
    # 查询所有实体的标签和 name 属性
    cypher = "MATCH (n) RETURN labels(n) AS labels, n.name AS name"
    result = link.run(cypher)

    # 构建字典存储结果
    entities_by_label = {}

    for record in result:
        labels = record["labels"]  # 获取实体的标签列表
        name = record["name"]  # 获取实体的 name 属性

        # 遍历每个标签并存储 name，并构建自定义文件
        if name:  # 确保 name 属性存在
            for label in labels:
                if label not in entities_by_label:
                    entities_by_label[label] = []
                # with open('selfDefine.txt','a',encoding='utf-8' ) as file:
                #     # 将变量转换为字符串并写入文件
                #     file.write(name+" "+"100 label"+'\n')
                #  因为已经构建完成，所以不需要再写了
                entities_by_label[label].append(name)

    # 输出结果
    print("按标签分类的实体字典：")
    for label, names in entities_by_label.items():
        print(f"{label}: {names}")
    return entities_by_label

if __name__ == '__main__':
    # 连接到 Neo4j 数据库
    link = Graph('bolt://localhost:7687', auth=('neo4j', 'wangshaner1.'))

    # entity_class = ["Class", "Disease","Reason","Pathogenesis","Symptom","Test","Drug","Frequency","Amount","Method","Treatment","Operation","ADE","Anatomy"]

    # 图谱解释
    entity_dict = DefineEntity(link)

    # 问句
    queryText = '医生我得了糖尿病，我好害怕怎么办'
    # 实体提取
    entityName = getEntityName(queryText)
    print("问句为：" + queryText)
    print("提取到的实体为：" + entityName)


