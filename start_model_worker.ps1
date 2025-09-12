# ==========================================
# Start InternVL Model Worker
# ==========================================

$python_path = "C:\Program Files\Python38\python.exe"
$worker_port = 40001
$controller_url = "http://127.0.0.1:40000"
$model_path = "OpenGVLab/InternVL2-1B"
$device = "cuda"  # 或 "cpu"

# 切换到 streamlit_demo 目录
Set-Location "$PSScriptRoot\streamlit_demo"

Write-Host "Starting model worker on port $worker_port..."
$worker_args = @(
    "model_worker.py",
    "--host", "127.0.0.1",
    "--controller-address", $controller_url,
    "--port", $worker_port,
    "--worker-address", "http://127.0.0.1:$worker_port",
    "--model-path", $model_path,
    "--device", $device
)
Start-Process -NoNewWindow -PassThru -FilePath $python_path -ArgumentList $worker_args
Write-Host "Model worker started."
