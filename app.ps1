# ==========================================
# Start InternVL Streamlit App (with auto browser)
# ==========================================

$python_path = "C:\Users\PDLP-013-Eric\Anaconda3\envs\video\python.exe"
$web_port = 10003
$controller_url = "http://localhost:40000"

# 切换到 streamlit_demo 目录
Set-Location "$PSScriptRoot\streamlit_demo"

Write-Host "Starting Streamlit app on port $web_port..."
$streamlit_args = @(
    "-m", "streamlit", "run", "app.py",
    "--server.port", "$web_port",
    "--server.address", "localhost",
    "--",
    "--controller_url", "$controller_url"
)

# 正确传参：Start-Process + ArgumentList
Start-Process -NoNewWindow -PassThru -FilePath $python_path -ArgumentList $streamlit_args

# 自动打开浏览器访问 Streamlit 页面
Start-Process "http://localhost:$web_port"

Write-Host "Streamlit app started at http://localhost:$web_port"
