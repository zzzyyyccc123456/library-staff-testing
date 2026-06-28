"""
训练 YOLOv11 座位占用检测模型（chair + person 2类）
从 library_seatoccuped(DST1037) 数据集的 VOC XML 标注转换并训练
"""
import os
import sys
import xml.etree.ElementTree as ET
import shutil
from pathlib import Path
from ultralytics import YOLO

# ========== 配置 ==========
BASE_DIR = Path(r"d:\51.4\5555\library_seatoccuped(DST1037)")
OUTPUT_DIR = Path(r"d:\51.4\5555\dataset_seat")

# 类别映射
CLASS_NAMES = ["chair", "person"]

# 训练参数
EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 16
MODEL_NAME = "yolo11n.pt"  # 用 nano 版本的预训练权重
DEVICE = "cpu"  # 无GPU，使用CPU


def convert_voc_to_yolo(xml_path, img_width, img_height):
    """将单个 VOC XML 转换为 YOLO TXT 格式的标注行列表"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    yolo_lines = []

    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        if name not in CLASS_NAMES:
            print(f"  警告: 跳过未知类别 '{name}' 在 {xml_path.name}")
            continue
        class_id = CLASS_NAMES.index(name)

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        # 转换为 YOLO 格式 (归一化)
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        # 确保坐标在 [0, 1] 范围内
        x_center = min(max(x_center, 0.0), 1.0)
        y_center = min(max(y_center, 0.0), 1.0)
        width = min(max(width, 0.0), 1.0)
        height = min(max(height, 0.0), 1.0)

        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return yolo_lines


def process_split(split_name):
    """处理 train/valid/test 分割"""
    img_dir = BASE_DIR / "images" / split_name
    xml_dir = BASE_DIR / "VOC" / split_name
    out_img_dir = OUTPUT_DIR / "images" / split_name
    out_label_dir = OUTPUT_DIR / "labels" / split_name

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(xml_dir.glob("*.xml"))
    converted = 0
    skipped_noimg = 0

    for xml_path in xml_files:
        # 对应的图片文件名（与 XML 中 filename 一致）
        stem = xml_path.stem
        # 尝试常见图片扩展名
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = img_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            skipped_noimg += 1
            continue

        # 从 XML 读取图片尺寸
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find("size")
        img_width = int(size.find("width").text)
        img_height = int(size.find("height").text)

        # 转换标注
        yolo_lines = convert_voc_to_yolo(xml_path, img_width, img_height)
        if not yolo_lines:
            continue

        # 复制图片到输出目录
        dst_img = out_img_dir / img_path.name
        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)

        # 写入 TXT 标注文件
        txt_path = out_label_dir / f"{stem}.txt"
        with open(txt_path, "w") as f:
            f.write("\n".join(yolo_lines))

        converted += 1

    return converted, skipped_noimg


def main():
    print("=" * 60)
    print("YOLOv11 座位占用检测训练准备")
    print("=" * 60)

    # 1. 清理旧的输出目录
    if OUTPUT_DIR.exists():
        print(f"\n删除旧数据集目录: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    # 2. 转换所有数据集
    print("\n=== 转换 VOC XML → YOLO TXT ===")
    for split in ["train", "valid", "test"]:
        converted, skipped = process_split(split)
        print(f"  {split}: 转换 {converted} 张，跳过 {skipped} 张（无对应图片）")

    # 统计各 split 的图片数
    for split in ["train", "valid", "test"]:
        n_imgs = len(list((OUTPUT_DIR / "images" / split).glob("*")))
        n_labels = len(list((OUTPUT_DIR / "labels" / split).glob("*.txt")))
        print(f"  {split}: {n_imgs} 张图片, {n_labels} 个标注文件")

    # 3. 创建 data.yaml
    data_yaml = f"""
# YOLO 数据集配置 - library_seatoccuped(DST1037)
# 自动生成于训练准备阶段

path: {OUTPUT_DIR.as_posix()}
train: images/train
val: images/valid
test: images/test

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(data_yaml.strip())
    print(f"\n✅ 已创建 data.yaml: {yaml_path}")

    # 4. 检查输出
    n_train = len(list((OUTPUT_DIR / "images" / "train").glob("*")))
    n_valid = len(list((OUTPUT_DIR / "images" / "valid").glob("*")))
    print(f"\n📊 数据集统计:")
    print(f"  训练集: {n_train} 张")
    print(f"  验证集: {n_valid} 张")
    print(f"  类别: {CLASS_NAMES}")

    # 5. 开始训练
    print(f"\n{'=' * 60}")
    print(f"开始训练 YOLOv11 - {EPOCHS} 轮")
    print(f"  模型: {MODEL_NAME}")
    print(f"  图片尺寸: {IMG_SIZE}")
    print(f"  Batch: {BATCH_SIZE}")
    print(f"  设备: {DEVICE}")
    print(f"{'=' * 60}\n")

    # 加载模型
    model = YOLO(MODEL_NAME)

    # 训练
    results = model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        name="yolo11_seat",
        patience=15,  # early stopping
        seed=42,
        workers=0,  # Windows 兼容
        amp=False,  # CPU 训练关闭 AMP
    )

    print(f"\n✅ 训练完成!")
    print(f"   结果保存在: runs/detect/yolo11_seat")

    # 6. 复制最佳模型到项目根目录方便使用
    best_pt = Path("runs/detect/yolo11_seat/weights/best.pt")
    if best_pt.exists():
        dst = Path(__file__).parent / "yolo11_seat_best.pt"
        shutil.copy2(best_pt, dst)
        print(f"   ✅ 模型已复制到: {dst}")


if __name__ == "__main__":
    main()
