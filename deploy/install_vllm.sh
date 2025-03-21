#!/bin/bash

# 1. 如果是 Windows，通过 wsl 进入 Ubuntu 虚拟机
wsl --list --verbose
wsl

# 2. 安装 miniconda：https://docs.anaconda.com/miniconda/install/#quick-command-line-install
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh

source ~/.bashrc

# 3. 安装 jupyterlab
conda activate base
conda install -c conda-forge jupyterlab

# 4. 为 vllm 创建虚拟环境，并绑定到 jupyterlab
conda create --name vllm_env python=3.12
conda activate vllm_env

python -m pip install ipykernel
python -m ipykernel install --user --name vllm_env --display-name "Python (vllm_env)"

# 5. 确认 cuda 相关组件已安装
nvidia-smi

# 6. 安装相关依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -i https://mirrors.aliyun.com/pypi/simple/
pip install transformers -i https://mirrors.aliyun.com/pypi/simple/
pip install accelerate huggingface_hub -i https://mirrors.aliyun.com/pypi/simple/

pip install importlib_metadata
pip install vllm -i https://mirrors.aliyun.com/pypi/simple/
pip install -U ipywidgets -i https://mirrors.aliyun.com/pypi/simple/

# 7. 下载模型 DeepSeek-R1-Distill-Qwen-1.5B
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local-dir ./model/DeepSeek-R1-Distill-Qwen-1.5B

# 8. 启动 jupyterlab
jupyter lab --ip='0.0.0.0' --port=3234 --notebook-dir=/ --no-browser --allow-root
