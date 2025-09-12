# ==========================================
# Start ChatGPT API Worker
# ==========================================

$python_path = "C:\Users\PDLP-013-Eric\Anaconda3\envs\video\python.exe"
$worker_port = 40002
$model = "gpt-4o"  # 使用更便宜的模型

# 切换到 streamlit_demo 目录
Set-Location "$PSScriptRoot\streamlit_demo"

Write-Host "Starting ChatGPT API worker..."
Write-Host "Model: $model"
Write-Host "Port: $worker_port"

if ($api_key -eq "YOUR_OPENAI_API_KEY_HERE") {
    Write-Host "ERROR: Please set your OpenAI API key in this script!" -ForegroundColor Red
    Write-Host "Get your API key from: https://platform.openai.com/api-keys" -ForegroundColor Yellow
    exit 1
}

$worker_args = @(
    "chatgpt_worker.py",
    "--host", "127.0.0.1",
    "--port", $worker_port,
    "--api-key", $api_key,
    "--model", $model
)

Start-Process -NoNewWindow -PassThru -FilePath $python_path -ArgumentList $worker_args
Write-Host "ChatGPT worker started."


