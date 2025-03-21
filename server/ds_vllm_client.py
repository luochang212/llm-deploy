# # -*- coding: utf-8 -*-
# # USAGE: python3 ds_client.py

from openai import OpenAI


API_KEY = "token-abc123456"
BASE_URL = "http://localhost:9494/v1"


client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY
)

def chat_completion(messages, stream=False):
    response = client.chat.completions.create(
        model="DeepSeek-R1-Distill-Qwen-1.5B",
        messages=messages,
        max_tokens=8192,
        temperature=0.6,
        top_p=0.95,
        stream=stream
    )
    return response


if __name__ == "__main__":
    messages = [
        {"role": "user", "content": "你将扮演一个内心火热但是表面冷淡的小偶像，请用暗含深切热爱的态度，回复粉丝的晚安动态。"}
    ]

    result = chat_completion(messages)
    print(result.choices[0].message.content)


# import requests


# API_URL = "http://localhost:9494/v1/chat/completions"
# API_KEY = "token-abc123456"
# HEADERS = {
#     "Authorization": f"Bearer {API_KEY}",
#     "Content-Type": "application/json"
# }


# def chat_completion(messages, stream=False):
#     data = {
#         "messages": messages,
#         "max_tokens": 8192,
#         "temperature": 0.6,
#         "top_p": 0.95,
#         "stream": stream,
#     }
    
#     response = requests.post(API_URL, headers=HEADERS, json=data)
#     if response.status_code != 200:
#         raise Exception(f"Error: {response.status_code} - {response.text}")
    
#     return response.json()


# if __name__ == "__main__":
#     messages = [
#         {"role": "user", "content": "你将扮演一个内心火热但是表面冷淡的小偶像，请用暗含深切热爱的态度，回复粉丝的晚安动态。"}
#     ]

#     result = chat_completion(messages)
#     print(result['choices'][0]['message']['content'])
