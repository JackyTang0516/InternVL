# ==========================================
# Start InternVL Controller
# ==========================================

$python_path = "C:\Program Files\Python38\python.exe"
$controller_port = 40000

# 切换到 streamlit_demo 目录
Set-Location "$PSScriptRoot\streamlit_demo"

Write-Host "Starting controller on port $controller_port..."
$controller_args = @(
    "controller.py",
    "--host", "127.0.0.1",
    "--port", $controller_port
)
Start-Process -NoNewWindow -PassThru -FilePath $python_path -ArgumentList $controller_args
Write-Host "Controller started."
