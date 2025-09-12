# ==========================================
# Start InternVL Controller
# ==========================================

$python_path = "C:\Users\PDLP-013-Eric\Anaconda3\envs\video\python.exe"
$controller_port = 40000

# 强制杀掉占用 $controller_port 的进程
Get-NetTCPConnection -LocalPort $controller_port -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "Killing process on port $controller_port (PID=$($_.OwningProcess))..."
    Stop-Process -Id $_.OwningProcess -Force
}

# 切换到 streamlit_demo 目录
Set-Location "$PSScriptRoot\streamlit_demo"

Write-Host "Starting controller on port $controller_port..."
$controller_args = @(
    "controller.py",
    "--host", "localhost",
    "--port", $controller_port
)
Start-Process -NoNewWindow -PassThru -FilePath $python_path -ArgumentList $controller_args
Write-Host "Controller started."
