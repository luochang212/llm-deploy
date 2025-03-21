#!/bin/bash
# USAGE: bash qwen_vllm_bash_server.sh
# HELP: vllm serve --help

source $(conda info --base)/etc/profile.d/conda.sh
conda activate vllm_env

qwen_model_path="../model/Qwen2.5-1.5B-Instruct/"
vllm serve $qwen_model_path \
    --served-model-name Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.98 \
    --tensor-parallel-size 1 \
    --api-key token-kcgyrk
