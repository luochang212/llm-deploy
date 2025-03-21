# -*- coding: utf-8 -*-
# DESC: vLLM openai server
# REFS:
#   - https://platform.openai.com/docs/api-reference/chat/create
#   - https://github.com/openai/openai-quickstart-python
#   - https://fastapi.tiangolo.com/advanced/events/
# USAGE: 
#   conda activate vllm_env
#   python3 ds_vllm_server.py

import os
import re
import vllm
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
    llm = vllm.LLM(
        model=MODEL_PATH,
        gpu_memory_utilization=0.8,
        max_model_len=4096,
        tensor_parallel_size=1,
        enable_prefix_caching=True
    )

    return llm


@asynccontextmanager
async def lifespan(app: FastAPI):
    llm = load_model()
    llm_model[MODEL_NAME] = llm
    yield
    llm_model.clear()


app = FastAPI(lifespan=lifespan)
security = HTTPBearer()


class ChatMessage(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
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


def model_infr(message: str,
               model,
               temperature=0.6,
               max_tokens=8192,
               top_p=0.95,
               stop_token_ids=[151329, 151336, 151338]):

    # 定义采样参数
    sampling_params = vllm.SamplingParams(temperature=temperature,
                                          top_p=top_p,
                                          max_tokens=max_tokens,
                                          stop_token_ids=stop_token_ids)
    # stop_token_ids or [model.llm_engine.tokenizer.eos_token_id]

    # 应用对话模型
    output = model.generate(message, sampling_params)
    response = output[0].outputs[0].text
    response = f'<think>\n{response}'
    think_content, answer_content = split_text(response)

    return think_content, answer_content


def format_prompt(messages) -> str:
    """仅保留最后一轮 user 的对话"""
    # 倒序遍历找到最后一个用户消息
    for message in reversed(messages):
        if message.role == "user":
            return message.content  # 直接返回字符串内容
    return ""  # 没有用户消息时返回空字符串


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
    model_name = request.model
    models = [MODEL_NAME]
    if model_name not in models:
        raise HTTPException(400, f"{model_name} not in {models}")

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
        model=llm_model[model_name],
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
        "model": model_name,
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
