@echo off
cd /d d:\51.4\5555
echo Starting training at %DATE% %TIME% > train_running.txt
python -u train_seat_model.py 2>&1
echo Training finished at %DATE% %TIME% >> train_running.txt
