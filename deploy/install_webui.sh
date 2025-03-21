#!/bin/bash

# 参考：https://docs.openwebui.com/getting-started/quick-start/

# 1. 确保已经安装 docker 且 docker 服务端已开启
docker info

# 2. 下载镜像，需科学上网环境
docker pull ghcr.io/open-webui/open-webui:cuda

# 3. 创建容器
# 请注意：
#   - 我们将容器的 /app/backend/data 目录挂载到本地的 ./data 目录，请确保 ./data 目录已被创建
#   - 将容器的 8080 端口映射到本机的 3215 端口，请确保本地 3215 端口未被占用，否则需要换端口
docker run -d -p 3215:8080 --gpus all -v ./data:/app/backend/data --name open-webui ghcr.io/open-webui/open-webui:cuda

# 查看容器日志
docker logs open-webui

# 4. 浏览器打开 http://127.0.0.1:3215
# 可能需要稍微加载一会儿，UI 界面才会出现
