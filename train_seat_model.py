"""
训练 YOLOv11 座位检测模型 (chair + person)
- 使用转换后的 YOLO 数据集
- 训练 50 轮
"""
import sys
import os
import time
from pathlib import Path

# 断网保护：禁用自动安装依赖
os.environ["ULTRALYTICS_DISABLE_REQUIREMENTS"] = "1"

from ultralytics import YOLO

LOG = open(r"d:\51.4\5555\train_log.txt", "w", encoding="utf-8")
def log(msg):
    t = time.strftime("%H:%M:%S")
    line = f"[{t}] {msg}"
    LOG.write(line + "\n")
    LOG.flush()
    print(line, flush=True)

log("=== 开始 YOLOv11 训练 ===")
data_yaml = r"d:\51.4\5555\dataset_yolo_seat\data.yaml"
model_name = "yolo11n.pt"
epochs = 50
imgsz = 640
batch = 16
device = "cpu"

log(f"模型: {model_name}")
log(f"数据: {data_yaml}")
log(f"Epochs: {epochs}")
log(f"Image size: {imgsz}")
log(f"Batch: {batch}")
log(f"Device: {device}")

# 加载模型
log("加载预训练模型...")
model = YOLO(model_name)

# 训练
log("开始训练...")
results = model.train(
    data=data_yaml,
    epochs=epochs,
    imgsz=imgsz,
    batch=batch,
    device=device,
    name="yolo11_seat",
    patience=20,
    seed=42,
    workers=0,
    amp=False,
    lr0=0.01,
    cos_lr=True,
)

log("训练完成!")

# 复制最佳模型
best_pt = Path("runs/detect/yolo11_seat/weights/best.pt")
if best_pt.exists():
    dst = Path(r"d:\51.4\5555") / "yolo11_seat_best.pt"
    import shutil
    shutil.copy2(best_pt, dst)
    log(f"模型已复制到: {dst}")

LOG.close()
