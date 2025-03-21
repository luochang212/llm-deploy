# -*- coding: utf-8 -*-
# USAGE: python3 ds_vllm_bash_client.py
# https://docs.vllm.ai/en/latest/features/reasoning_outputs.html
# pip install vllm --upgrade

from openai import OpenAI


openai_api_key = "token-abc123456"
openai_api_base = "http://localhost:8000/v1"


def chat_completion(prompt, model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
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
        temperature=0.6,
        top_p=0.95,
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
