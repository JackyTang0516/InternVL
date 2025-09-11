#!/bin/bash

# InternVL Streamlit Demo 启动脚本
# 设置环境变量
export SD_SERVER_PORT=39999
export WEB_SERVER_PORT=10003
export CONTROLLER_PORT=40000
export CONTROLLER_URL=http://0.0.0.0:$CONTROLLER_PORT
export SD_WORKER_URL=http://0.0.0.0:$SD_SERVER_PORT

# 设置最大图像限制，可以通过命令行参数覆盖
MAX_IMAGE_LIMIT=${1:-1000}  # 默认1000，可以通过第一个参数覆盖

# 进入streamlit_demo目录
cd streamlit_demo

echo "=== 启动 InternVL Streamlit Demo 服务 ==="
echo "控制器端口: $CONTROLLER_PORT"
echo "模型工作器端口: 40001"
echo "Web服务器端口: $WEB_SERVER_PORT"
echo "SD服务器端口: $SD_SERVER_PORT"
echo "最大图像限制: $MAX_IMAGE_LIMIT"
echo ""

# 检查设备支持情况
if python -c "import torch; print('CUDA available:', torch.cuda.is_available())" | grep -q "True"; then
    DEVICE="cuda"
    echo "检测到CUDA支持，使用GPU模式"
else
    DEVICE="cpu"
    echo "使用CPU模式"
fi

# 启动控制器
echo "1. 启动控制器..."
python controller.py --host 0.0.0.0 --port $CONTROLLER_PORT &
CONTROLLER_PID=$!
echo "控制器已启动，PID: $CONTROLLER_PID"

# 等待控制器启动
sleep 3

# 启动模型工作器
echo "2. 启动模型工作器..."
if [ "$DEVICE" = "cuda" ]; then
    CUDA_VISIBLE_DEVICES=0 python model_worker.py --host 0.0.0.0 --controller-address $CONTROLLER_URL --port 40001 --worker-address http://0.0.0.0:40001 --model-path OpenGVLab/InternVL2-1B --device cuda &
else
    python model_worker.py --host 0.0.0.0 --controller-address $CONTROLLER_URL --port 40001 --worker-address http://0.0.0.0:40001 --model-path OpenGVLab/InternVL2-1B --device $DEVICE &
fi
MODEL_WORKER_PID=$!
echo "模型工作器已启动，PID: $MODEL_WORKER_PID"

# 等待模型工作器启动
sleep 5

# 启动Streamlit应用
echo "3. 启动Streamlit应用..."
streamlit run app.py --server.port $WEB_SERVER_PORT --server.address 0.0.0.0 -- --controller_url $CONTROLLER_URL --max_image_limit $MAX_IMAGE_LIMIT &
STREAMLIT_PID=$!
echo "Streamlit应用已启动，PID: $STREAMLIT_PID"

echo ""
echo "=== 所有服务已启动 ==="
echo "控制器: http://0.0.0.0:$CONTROLLER_PORT"
echo "模型工作器: http://0.0.0.0:40001"
echo "Streamlit应用: http://0.0.0.0:$WEB_SERVER_PORT"
echo "最大图像限制: $MAX_IMAGE_LIMIT"
echo ""
echo "使用说明:"
echo "  ./start_services.sh [最大图像限制]"
echo "  例如: ./start_services.sh 2000  # 设置最大图像限制为2000"
echo "  默认最大图像限制: 1000"
echo ""
echo "按 Ctrl+C 停止所有服务"

# 等待用户中断
trap 'echo "正在停止所有服务..."; kill $CONTROLLER_PID $MODEL_WORKER_PID $STREAMLIT_PID 2>/dev/null; exit 0' INT

# 保持脚本运行
wait
