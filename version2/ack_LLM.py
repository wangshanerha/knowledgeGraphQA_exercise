from zhipuai import ZhipuAI
import re
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
    question = input()

    prompt1 = '''现在你将接收到一段文字%s，按照以下三个问题格式输出：
    按照"background："的格式输出这段文字的背景。
    按照"query："的格式输出这段文字中提出的问题。
    按照"knowledge1："的格式提问你认为回答此文本提到的问题需要了解的专业问题。
    请务必注意，不需要对query本身进行回答。
    请务必注意，专业问题knowledge不要包含“与”、“和”、“及”等有并列关系的文字。
    请务必注意，专业问题不是一定3个，可以是2个也可以是更多
        '''%(question)
    #     answer = ask_glm(prompt)['choices'][0]['message']['content']
    text = ask_glm(prompt1).content
    print(text)

