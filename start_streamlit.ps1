# ==========================================
# Start InternVL Streamlit App (with auto browser)
# ==========================================

$python_path = "py"
$web_port = 10003
$controller_url = "http://127.0.0.1:40000"

# 切换到 streamlit_demo 目录
Set-Location "$PSScriptRoot\streamlit_demo"

Write-Host "Starting Streamlit app on port $web_port..."
$streamlit_args = @(
    "-3.10", "-m", "streamlit", "run", "app.py",
    "--server.port", "$web_port",
    "--server.address", "0.0.0.0",
    "--",
    "--controller_url", "$controller_url"
)

# 保持数组形式传参
Start-Process -NoNewWindow -PassThru -FilePath $python_path -ArgumentList $streamlit_args

# 自动打开浏览器访问 Streamlit 页面
Start-Process "http://127.0.0.1:$web_port"

Write-Host "Streamlit app started at http://127.0.0.1:$web_port"
