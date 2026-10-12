# 脱附启动 DeepSeek 调度器（供定时任务使用）
# 用法：powershell -ExecutionPolicy Bypass -File _launch_scheduler.ps1
# 说明：Start-Process 创建独立进程；实测存活情况见 _canary_survived.txt（2026-10-10 00:53 测试）

$py = if ($env:ABM_PYTHON) { $env:ABM_PYTHON } else { "C:\Users\21130\AppData\Local\Programs\Python\Python312\python.exe" }
$probe = if ($env:ABM_PROBE_ROOT) { $env:ABM_PROBE_ROOT } else { "D:\21130\Desktop\大创过程\probe_提示词明确性" }
$proj = if ($env:ABM_PROJ_ROOT) { $env:ABM_PROJ_ROOT } else { "D:\21130\Desktop\大创过程\实验代码\最终代码和数据" }
$wd = $probe
$lock = Join-Path $proj "results_direction\_scheduler.lock"
$done = Join-Path $proj "results_direction\_scheduler_done.flag"

if (Test-Path $done) { Write-Output "DONE_FLAG_EXISTS"; exit 0 }

# 锁存活检查
if (Test-Path $lock) {
    $pid = Get-Content $lock -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($pid -and (Get-Process -Id $pid -ErrorAction SilentlyContinue)) {
        Write-Output "SCHEDULER_ALREADY_RUNNING pid=$pid"; exit 0
    }
    Remove-Item $lock -Force -ErrorAction SilentlyContinue
}

Start-Process -FilePath $py -ArgumentList '-X','utf8','deepseek_scheduler.py' -WorkingDirectory $wd -WindowStyle Hidden
Write-Output "LAUNCHED"
