#!/bin/bash
# USAGE: bash ds_vllm_bash_server.sh
# HELP: vllm serve --help

source $(conda info --base)/etc/profile.d/conda.sh
conda activate vllm_env

deepseek_model_path="../model/DeepSeek-R1-Distill-Qwen-1.5B"
CUDA_VISIBLE_DEVICES=0 vllm serve $deepseek_model_path \
    --served-model-name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --enable-reasoning \
    --reasoning-parser deepseek_r1 \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.98 \
    --max-seq-len-to-capture 8192 \
    --tensor-parallel-size 1 \
    --api-key token-abc123456
