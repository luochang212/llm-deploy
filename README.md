
> 2023 年本地部署大模型的报价近千万，2024 年初便骤降至百万，如今是 2025 年，只需要一行 vLLM 命令就可以部署大模型，人工成几近于零。

本文内容包括：

- 三种方式部署 DeepSeek R1：Ollama, vLLM 和 Transformers
- 使用 vLLM 部署 Qwen2.5-1.5B 模型
- 安装 Open WebUI 作为本地部署模型的前端聊天框
- 通过 `vllm serve` 实现一行代码启动 vLLM 推理服务

✨ 快速部署说明在 [deploy](/deploy)，vLLM 服务启动脚本在 [server](/server).


## 一、3 种方式部署 DeepSeek R1

目前比较流行的推理引擎有：

|引擎|场景|介绍|
| -- | -- |-- |
|[Ollama](https://github.com/ollama/ollama)|适合个人开发者|基于 [llama.cpp](https://github.com/ggml-org/llama.cpp) 开发，故能在 CPU 机器上使用|
|[vLLM](https://github.com/vllm-project/vllm)|适合高并发、低延迟的场景|支持多 GPU 并行、批处理，是 GPU 服务器部署的不二之选|
|[Transformers](https://github.com/huggingface/transformers)|适合科研人员|既支持推理，也支持训练，可以深度自定义推理过程，但性能一般|

下面介绍如何部署这三款推理引擎，简要部署步骤见本项目的 [deploy](/deploy) 目录。

目录：

1. Ollama
2. vLLM
    - 安装 miniconda
    - 安装 jupyterlab
    - 为 vllm 创建虚拟环境，并绑定到 jupyterlab
    - 安装 vLLM 及相关依赖
    - 下载模型文件
    - 启动 jupyterlab
    - 验证 vLLM
3. Transformers


## 二、vLLM 部署 Qwen2.5

模型名带 `Instruct` 说明该模型是经过指令调优（Instruction Tuning）的版本，专为理解和执行用户指令优化，适合对话生成、任务导向型场景。

部署前要先下载模型文件：


推理效果：

![](/img/qwen_result.png)


## 三、Open WebUI 作为前端聊天框

第一节我用 3 种方式部署本地大模型：Ollama, vLLM 和 Transformers.

有了本地部署的大模型作为推理后端，还要连接到前端聊天框，才算是完整。开源聊天框有很多，比如 AnythingLLM, LM Studio. 比较了一圈，姑且使用 [Open WebUI](https://github.com/open-webui/open-webui). 它有预开发的账号系统和历史记录，支持快速开发 RAG 和联网搜索。

它的界面长这样：

![](/img/open_webui.jpg)

目录：

1. 安装 Open WebUI
    - 检查 Docker 环境
    - 下载镜像
    - 创建容器
    - 打开前端页面
2. 为 WebUI 配置推理后端
    - Ollama
    - vLLM


## 四、vLLM 推理服务

vLLM 官方支持一行 bash 命令，启动 API 服务：

```
vllm serve [YOUR MODEL or MODEL PATH]
```

当然对于不同的模型，启动参数也略有不同，需要做一定的配适。比如对于 `deepseek-r1` 这类推理模型，必须添加参数 `--enable-reasoning --reasoning-parser deepseek_r1`。具体每个模型如何使用 vLLM，可参考 vLLM 官方文档或模型文档。

> **PS:** vLLM 的 API 服务遵循 OpenAI 的接口协议。

目录：

1. DeepSeek-R1
    - 启动服务端
    - 运行客户端
2. Qwen
    - 启动服务端
    - 运行客户端
3. 符合 OpenAI 接口协议的 API 服务
4. 强制清理显存缓存
