# -*- coding: utf-8 -*-
# DESC: transformers openai server
# REFS:
#   - https://platform.openai.com/docs/api-reference/chat/create
#   - https://github.com/openai/openai-quickstart-python
#   - https://fastapi.tiangolo.com/advanced/events/
# USAGE: 
#   conda activate vllm_env
#   python3 ds_transformers_server.py

import os
import re
import torch
import transformers
import time
import uuid
import uvicorn

from typing import List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager


# 配置 API 密钥
API_KEY = "token-abc123456"
MODEL_NAME = "DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_PATH = "../model/DeepSeek-R1-Distill-Qwen-1.5B"


# 指定使用哪一块显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


llm_model = {}


def load_model():
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_PATH,
                                                              device_map='auto',
                                                              torch_dtype=torch.bfloat16)
    return tokenizer, model


@asynccontextmanager
async def lifespan(app: FastAPI):
    tokenizer, model = load_model()
    llm_model["tokenizer"] = tokenizer
    llm_model["model"] = model
    yield
    llm_model.clear()


app = FastAPI(lifespan=lifespan)
security = HTTPBearer()


class ChatMessage(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str


class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 8192
    temperature: Optional[float] = 0.6
    top_p: Optional[float] = 0.95
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0


def verify_token(credentials: HTTPBearer = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(401, "Invalid API Key")


def split_text(text):
    """文本分割函数"""
    pattern = re.compile(r'<think>(.*?)</think>(.*)', re.DOTALL)
    match = pattern.search(text) # 匹配思考过程

    if match: # 如果匹配到思考过程
        think_content = match.group(1) if match.group(1) is not None else ""
        think_content = think_content.strip()
        answer_content = match.group(2).strip()
    else:
        think_content = ""
        answer_content = text.strip()

    return think_content, answer_content


def model_infr(message: str, tokenizer, model, temperature=0.6, max_tokens=8192, top_p=0.95):
    # 应用对话模型
    input_ids = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([input_ids], return_tensors="pt").to(model.device)

    # 文本生成
    generated_ids = model.generate(model_inputs.input_ids,
                                   attention_mask=model_inputs.attention_mask,
                                   pad_token_id=tokenizer.eos_token_id,
                                   temperature=temperature,
                                   max_new_tokens=max_tokens,
                                   top_p=top_p)

    # 生成结果后处理：通过切片剔除输入部分，仅保留模型生成的内容
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # 解码：将 token id 转换为自然语言文本，并跳过特殊标记
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    response = f'<think>\n{response}'
    think_content, answer_content = split_text(response)

    return think_content, answer_content


def format_prompt(messages) -> list:
    """仅保留最后一轮 user 的对话"""
    # 倒序遍历找到最后一个用户消息
    for message in reversed(messages):
        if message.role == "user":
            return [message]
    return []


@app.get("/v1/models", dependencies=[Depends(verify_token)])
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "user",
                "permissions": []
            }
        ]
    }


@app.post("/v1/chat/completions", dependencies=[Depends(verify_token)])
async def create_chat_completion(request: ChatCompletionRequest):

    # 校验
    if request.n > 1 and not request.stream:
        raise HTTPException(400, "Only n=1 supported in non-streaming mode")
    if request.temperature is not None and request.temperature < 0:
        raise HTTPException(400, "Temperature must be ≥ 0")
    if request.top_p is not None and (request.top_p < 0 or request.top_p > 1):
        raise HTTPException(400, "Top_p must be between 0 and 1")

    # 模型推理
    prompt = format_prompt(request.messages)
    think_content, answer_content = model_infr(
        message=prompt,
        tokenizer=llm_model["tokenizer"],
        model=llm_model["model"],
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        top_p=request.top_p
    )

    # 输出排版
    lst = [
        "<think>",
        think_content,
        "</think>",
        answer_content
    ]
    content = "\n".join(lst)

    return {
        "id": f"chatcmpl-{str(uuid.uuid4())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content.strip()
            },
            "finish_reason": "stop"
        }]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9494, log_level="debug")
