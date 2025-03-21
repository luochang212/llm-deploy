#!/bin/bash

# USAGE: sh hugging_face.sh

# 安装 huggingface_hub
pip install -U huggingface_hub

# 切换为国内镜像源
export HF_ENDPOINT=https://hf-mirror.com

# 下载 deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
huggingface-cli download --resume-download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local-dir ./DeepSeek-R1-Distill-Qwen-1.5B

# 下载 Qwen/Qwen2.5-1.5B-Instruct
huggingface-cli download --resume-download Qwen/Qwen2.5-1.5B-Instruct --local-dir ./Qwen2.5-1.5B-Instruct
