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
        ("user", "现在你将接收到一段文字{text}.，按照以下三个问题格式输出：按照background：的格式输出这段文字的背景。请顺理这段文字提出的问题，并按照query：的格式重述成用词更严谨的专业问题。请务必注意，不需要对query本身进行回答。请务必注意，专业问题不要包含“与”、“和”、“及”等有并列关系的文字。请务必注意，专业问题允许多个"),
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
    text =  "餐前低血糖、皮肤瘙痒或反复的皮肤感染、视物模糊、胃肠功能紊乱、反复发生的泌尿系感染、尿中泡沫增多或者蛋白尿等"
    model_answer = ask_question(text)
    print(model_answer)

