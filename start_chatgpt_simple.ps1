# ==========================================
# Simple ChatGPT Demo Start
# ==========================================

Write-Host "=== Starting ChatGPT Demo ===" -ForegroundColor Green

# 停止可能冲突的进程
Write-Host "`n🔄 Stopping conflicting processes..." -ForegroundColor Yellow
Get-Process | Where-Object {$_.ProcessName -like "*python*"} | Stop-Process -Force -ErrorAction SilentlyContinue

# 等待端口释放
Start-Sleep -Seconds 2

# 1. 启动ChatGPT worker
Write-Host "`n1. Starting ChatGPT worker..." -ForegroundColor Yellow
$python_path = "C:\Users\PDLP-013-Eric\Anaconda3\envs\video\python.exe"
$worker_port = 40002
$model = "gpt-4o"

Set-Location "streamlit_demo"

$worker_args = @(
    "chatgpt_worker.py",
    "--host", "127.0.0.1",
    "--port", $worker_port,
    "--api-key", $api_key,
    "--model", $model
)

Start-Process -NoNewWindow -PassThru -FilePath $python_path -ArgumentList $worker_args

Write-Host "Waiting for ChatGPT worker..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# 2. 启动Streamlit app
Write-Host "`n2. Starting Streamlit app..." -ForegroundColor Yellow
$web_port = 10003

$streamlit_args = @(
    "-m", "streamlit", "run", "app.py",
    "--server.port", $web_port,
    "--server.address", "localhost",
    "--",
    "--use_chatgpt"
)

Start-Process -NoNewWindow -PassThru -FilePath $python_path -ArgumentList $streamlit_args

# 等待app启动
Start-Sleep -Seconds 5

# 打开浏览器
Write-Host "`n3. Opening browser..." -ForegroundColor Yellow
Start-Process "http://localhost:$web_port"

Write-Host "`n=== ChatGPT Demo Started! ===" -ForegroundColor Green
Write-Host "🌐 Access at: http://localhost:$web_port" -ForegroundColor Cyan
Write-Host "🤖 Using: ChatGPT API ($model)" -ForegroundColor Cyan
Write-Host "`n💡 Note: This version supports text analysis" -ForegroundColor Yellow
