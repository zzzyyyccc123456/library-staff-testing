"""启动训练 - 所有输出重定向到文件"""
import sys, os

# 重定向 stdout/stderr 到文件
log_path = r"d:\51.4\5555\train_full_log.txt"
f = open(log_path, "w", encoding="utf-8")
sys.stdout = f
sys.stderr = f

print("Starting training...", flush=True)
print(f"Python: {sys.version}", flush=True)

import time
start = time.time()

from ultralytics import YOLO
print(f"Ultralytics imported in {time.time()-start:.1f}s", flush=True)

model = YOLO(r"d:\51.4\5555\yolo11n.pt")
print(f"Model loaded in {time.time()-start:.1f}s", flush=True)

print("Starting model.train()...", flush=True)
model.train(
    data=r"d:\51.4\5555\dataset_yolo_seat\data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    device="cpu",
    name="yolo11_seat_run",
    patience=20,
    seed=42,
    workers=0,
    amp=False,
)
print(f"Training complete! Total time: {time.time()-start:.1f}s", flush=True)

# 复制模型
from pathlib import Path
best = Path("runs/detect/yolo11_seat_run/weights/best.pt")
if best.exists():
    import shutil
    shutil.copy2(best, Path(r"d:\51.4\5555") / "yolo11_seat_best.pt")
    print(f"Model copied to d:\\51.4\\5555\\yolo11_seat_best.pt", flush=True)

f.close()
print(f"\nLog saved to {log_path}", flush=True)
