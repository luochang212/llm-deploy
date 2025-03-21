# -*- coding: utf-8 -*-
# USAGE: python3 qwen_vllm_bash_client.py

from openai import OpenAI


openai_api_key = "token-kcgyrk"
openai_api_base = "http://localhost:8000/v1"


def chat_completion(prompt, model="Qwen/Qwen2.5-7B-Instruct"):
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    chat_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
        top_p=0.9,
        max_tokens=512,
        extra_body={
            "repetition_penalty": 1.05,
        },
    )

    return chat_response


if __name__ == "__main__":
    response = chat_completion(prompt="什么是大模型？")
    content = response.choices[0].message.content
    print(content)
