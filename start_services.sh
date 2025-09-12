#!/bin/bash

# InternVL Streamlit Demo 启动脚本 - 支持 OpenAI API
# 设置环境变量
export WEB_SERVER_PORT=10003
export CHATGPT_WORKER_PORT=40002

# OpenAI API 配置

export OPENAI_MODEL="gpt-4o-mini"

# 设置最大图像限制，可以通过命令行参数覆盖
MAX_IMAGE_LIMIT=${1:-1000}  # 默认1000，可以通过第一个参数覆盖

# 进入streamlit_demo目录
cd streamlit_demo

echo "=== 启动 InternVL Streamlit Demo 服务 (OpenAI API 版本) ==="
echo "ChatGPT工作器端口: $CHATGPT_WORKER_PORT"
echo "Web服务器端口: $WEB_SERVER_PORT"
echo "最大图像限制: $MAX_IMAGE_LIMIT"
echo "使用模型: $OPENAI_MODEL"
echo ""

# 检查 OpenAI API Key 是否设置
if [ -z "$OPENAI_API_KEY" ]; then
    echo "错误: 未设置 OPENAI_API_KEY 环境变量"
    exit 1
fi

# 启动 ChatGPT worker
echo "1. 启动 ChatGPT worker..."
python chatgpt_worker.py --host 0.0.0.0 --port $CHATGPT_WORKER_PORT --api-key "$OPENAI_API_KEY" --model "$OPENAI_MODEL" &
CHATGPT_WORKER_PID=$!
echo "ChatGPT worker 已启动，PID: $CHATGPT_WORKER_PID"

# 等待 ChatGPT worker 启动
sleep 3

# 启动Streamlit应用
echo "2. 启动Streamlit应用..."
streamlit run app.py --server.port $WEB_SERVER_PORT --server.address 0.0.0.0 -- --use_chatgpt --chatgpt_worker_url http://localhost:$CHATGPT_WORKER_PORT --max_image_limit $MAX_IMAGE_LIMIT &
STREAMLIT_PID=$!
echo "Streamlit应用已启动，PID: $STREAMLIT_PID"

echo ""
echo "=== 所有服务已启动 ==="
echo "ChatGPT worker: http://0.0.0.0:$CHATGPT_WORKER_PORT"
echo "Streamlit应用: http://0.0.0.0:$WEB_SERVER_PORT"
echo "最大图像限制: $MAX_IMAGE_LIMIT"
echo "使用模型: $OPENAI_MODEL"
echo ""
echo "使用说明:"
echo "  ./start_services.sh [最大图像限制]"
echo "  例如: ./start_services.sh 2000  # 设置最大图像限制为2000"
echo "  默认最大图像限制: 1000"
echo ""
echo "按 Ctrl+C 停止所有服务"

# 等待用户中断
trap 'echo "正在停止所有服务..."; kill $CHATGPT_WORKER_PID $STREAMLIT_PID 2>/dev/null; exit 0' INT

# 保持脚本运行
wait
