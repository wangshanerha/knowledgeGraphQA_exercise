from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

def ask_question(text):
        # 1. 调用api
        # zhipuai_chat = ChatZhipuAI(
        #     temperature=0.5,
        #     api_key="f037f80bb31908ef8d2159b5e4f17e6d.kDBaHZjpplHWU6pl",
        #     model_name="glm-4-flash",
        #
        # )
        model = ChatOpenAI(
            temperature=0.95,
            model="glm-4-flash",
            openai_api_key="f037f80bb31908ef8d2159b5e4f17e6d.kDBaHZjpplHWU6pl",
            openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
        )

        # 2. 创建提示模板
        # prompt_template = ChatPromptTemplate.from_template("{input}")
        question = {"name": "糖尿病专家", "text": text}
        prompt_template = ChatPromptTemplate([
            ("system", "你现在扮演一个： {name}."),
            ("user",
             "现在你将接收到一段文字{text}"),
        ])

        # 3 创建数据响应器
        parser = StrOutputParser()

        # 4. 创建Chain
        chain = (
                prompt_template
                | model
                | parser
        )

        # 5. 运行Chain并获取回答
        response = chain.invoke(question)
        return (response)


def Restatement_problem(text):
    # 1. 调用api
    # zhipuai_chat = ChatZhipuAI(
    #     temperature=0.5,
    #     api_key="f037f80bb31908ef8d2159b5e4f17e6d.kDBaHZjpplHWU6pl",
    #     model_name="glm-4-flash",
    #
    # )
    model = ChatOpenAI(
        temperature=0.95,
        model="glm-4-flash",
        openai_api_key="f037f80bb31908ef8d2159b5e4f17e6d.kDBaHZjpplHWU6pl",
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
    )

    # 2. 创建提示模板
    # prompt_template = ChatPromptTemplate.from_template("{input}")
    question = {"name": "糖尿病专家", "text": text}
    # prompt_template = ChatPromptTemplate([
    #     ("system", "你现在扮演一个： {name}."),
    #     ("user", "现在你将接收到一段文字text{text}.，你认为解决text的问题需要哪些专业知识，不需要对text本身进行回答。请务必注意，专业知识不要包含“与”、“和”、“及”等有并列关系的文字，允许回答多个但是不要太多。"),
    # ])
    prompt_template = ChatPromptTemplate([
        ("system", "你现在扮演一个： {name}."),
        ("user", """请根据以下文本text{text}生成专业知识点提示词：
            请务必注意，不需要对text本身进行回答
            1. 识别关键医学要素：包括但不限于[诊断指标][用药情况][症状表现][治疗效果]
            2. 关联知识维度：匹配但不限于[治疗规范][药理机制][副作用监测][疗效评估][并发症预防]
            3. 输出格式要求：使用数字序号清单，每个提示词采用"领域关键词+具体方向"结构
            请务必注意，回答请围绕专业知识，严格按格式输出，每行的内容控制在20字以内
            示例输出：
            1. 药物治疗方案调整原则
            2. 二甲双胍药效动力学
            3. 血糖监测频率标准
            4. 胰岛素启用指征判断
            5. 微血管并发症筛查"""),
                ])


    # 3 创建数据响应器
    parser = StrOutputParser()

    # 4. 创建Chain
    chain = (
            prompt_template
            | model
            | parser
    )

    # 5. 运行Chain并获取回答
    response = chain.invoke(question)
    return  (response)

if __name__ == '__main__':
    # question = input()
    text =  "患者还出现视物模糊的症状，尤其是在看近处物体时感觉模糊不清。患者还提到，近期皮肤经常出现瘙痒，尤其是四肢部位，且容易感染，愈合缓慢。患者无明显低血糖发作史，无其他慢性病史。如何治疗"
    model_answer = Restatement_problem(text)
    print(model_answer)

