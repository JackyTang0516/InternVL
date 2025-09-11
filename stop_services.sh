#!/bin/bash

# 停止所有InternVL相关服务
echo "正在停止所有InternVL服务..."

# 停止控制器
pkill -f "controller.py"
echo "控制器已停止"

# 停止模型工作器
pkill -f "model_worker.py"
echo "模型工作器已停止"

# 停止Streamlit应用
pkill -f "streamlit run app.py"
echo "Streamlit应用已停止"

# 停止SD工作器（如果存在）
pkill -f "sd_worker.py"
echo "SD工作器已停止"

echo "所有服务已停止"
