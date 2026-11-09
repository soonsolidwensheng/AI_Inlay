#!/bin/bash
# 启动脚本：使用 waitress，超时由应用层（multiprocessing Pipe + call_on_close）控制，进程退出后循环重启

PORT=5000

echo "Starting Flask app with waitress..."

while true; do
    python app.py
    EXIT_CODE=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Process exited with code ${EXIT_CODE}, restarting..."
    sleep 1
done
