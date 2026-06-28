"""
VOC XML -> YOLO TXT 格式转换
来源: library_seatoccuped(DST1037)/VOC/ -> images/ 配对
输出: dataset_seat/
"""
import os
import sys
import xml.etree.ElementTree as ET
import shutil
import time
from pathlib import Path

BASE_DIR = Path(r"d:\51.4\5555\library_seatoccuped(DST1037)")
OUTPUT_DIR = Path(r"d:\51.4\5555\dataset_yolo_seat")
CLASS_NAMES = ["chair", "person"]

LOG = open(r"d:\51.4\5555\convert_progress.txt", "w", encoding="utf-8")
def log(msg):
    LOG.write(msg + "\n")
    LOG.flush()
    print(msg, flush=True)

for split in ["train", "valid", "test"]:
    img_dir = BASE_DIR / "images" / split
    xml_dir = BASE_DIR / "VOC" / split
    out_img_dir = OUTPUT_DIR / "images" / split
    out_label_dir = OUTPUT_DIR / "labels" / split

    # 创建目录（如果不存在则创建）
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    # 先删除旧的标注文件，确保干净
    for old_txt in out_label_dir.glob("*.txt"):
        try:
            old_txt.unlink(missing_ok=True)
        except:
            pass
    # 删除旧图片
    for old_img in out_img_dir.glob("*"):
        try:
            old_img.unlink(missing_ok=True)
        except:
            pass

    xml_files = sorted(xml_dir.glob("*.xml"))
    converted = 0
    skipped_noimg = 0
    skipped_empty = 0

    for xml_path in xml_files:
        stem = xml_path.stem
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = img_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            skipped_noimg += 1
            continue

        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find("size")
        img_width = int(size.find("width").text)
        img_height = int(size.find("height").text)

        yolo_lines = []
        for obj in root.findall("object"):
            name = obj.find("name").text.strip().lower()
            if name not in CLASS_NAMES:
                continue
            class_id = CLASS_NAMES.index(name)
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            x_center = ((xmin + xmax) / 2) / img_width
            y_center = ((ymin + ymax) / 2) / img_height
            w = (xmax - xmin) / img_width
            h = (ymax - ymin) / img_height

            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        if not yolo_lines:
            skipped_empty += 1
            continue

        # 复制图片
        shutil.copy2(img_path, out_img_dir / img_path.name)

        # 写标注
        txt_path = out_label_dir / f"{stem}.txt"
        with open(txt_path, "w") as f:
            f.write("\n".join(yolo_lines))

        converted += 1

    n_imgs = len(list(out_img_dir.glob("*")))
    n_lbls = len(list(out_label_dir.glob("*.txt")))
    log(f"{split}: 转换 {converted} 张 | 图片 {n_imgs} | 标注 {n_lbls} | 跳过(无图 {skipped_noimg}, 空标注 {skipped_empty})")

# 创建 data.yaml
yaml_content = f"""
path: {OUTPUT_DIR.as_posix()}
train: images/train
val: images/valid
test: images/test
nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
with open(OUTPUT_DIR / "data.yaml", "w") as f:
    f.write(yaml_content.strip())
log(f"data.yaml 已创建")

total_imgs = sum(len(list((OUTPUT_DIR / "images" / s).glob("*"))) for s in ["train", "valid", "test"])
total_lbls = sum(len(list((OUTPUT_DIR / "labels" / s).glob("*.txt"))) for s in ["train", "valid", "test"])
log(f"总计: {total_imgs} 张图片, {total_lbls} 个标注文件")
LOG.close()
