"""YOLOv11 训练 - 座位占用检测 (chair + person)"""
import sys, os, time
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

log_file = open(r"d:\51.4\5555\train_progress.txt", "w", encoding="utf-8")
def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    log_file.write(line + "\n")
    log_file.flush()
    print(line, flush=True)

log("=== YOLOv11 座位检测训练 ===")
log(f"Python: {sys.version}")
log(f"Executable: {sys.executable}")

from ultralytics import YOLO
log("Ultralytics 导入成功")

# 加载模型
model = YOLO(r"d:\51.4\5555\yolo11n.pt")
log("模型加载成功")

# 开始训练
log("开始训练 50 轮...")
log(f"数据: d:\\51.4\\5555\\dataset_yolo_seat\\data.yaml")
log(f"设备: cpu")

model.train(
    data=r"d:\51.4\5555\dataset_yolo_seat\data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    device="cpu",
    name="yolo11_seat_final",
    patience=20,
    seed=42,
    workers=0,
    amp=False,
    lr0=0.01,
    cos_lr=True,
)
log("训练完成!")

# 复制最佳模型
best = Path("runs/detect/yolo11_seat_final/weights/best.pt")
if best.exists():
    import shutil
    from pathlib import Path
    shutil.copy2(best, Path(r"d:\51.4\5555") / "yolo11_seat_best.pt")
    log("模型已复制到项目根目录")

log_file.close()
