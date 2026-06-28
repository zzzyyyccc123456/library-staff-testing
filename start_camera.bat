@echo off
chcp 65001 >nul
cd /d "d:\51.4\5555"
echo ================================================
echo            座位检测服务启动中...
echo ================================================
"C:\Users\小周\AppData\Local\Programs\Python\Python39\python.exe" start_server.py
pause
