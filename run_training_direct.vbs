' Run YOLO training in background
Set WshShell = CreateObject("WScript.Shell")
cmd = "d:\51.4\5555\venv\Scripts\python.exe -u d:\51.4\5555\train_final.py > d:\51.4\5555\train_output_log.txt 2>&1"
WshShell.Run cmd, 0, False
