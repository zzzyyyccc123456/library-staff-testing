@echo off
cd /d d:\51.4\5555
python -u convert_voc_to_yolo.py > convert_output.txt 2>&1
type convert_output.txt
