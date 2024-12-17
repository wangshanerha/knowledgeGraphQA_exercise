from zhipuai import ZhipuAI


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
    prompt = '''医生我得了糖尿病足，我应该吃什么药
        '''
    #     answer = ask_glm(prompt)['choices'][0]['message']['content']
    answer = ask_glm(prompt).content
    print(answer)

