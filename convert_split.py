"""转换单个 split (train/valid/test)"""
import xml.etree.ElementTree as ET
import shutil
import sys
from pathlib import Path

split = sys.argv[1]  # train, valid, test
BASE_DIR = Path(r"d:\51.4\5555\library_seatoccuped(DST1037)")
OUTPUT_DIR = Path(r"d:\51.4\5555\dataset_yolo_seat")
CLASS_NAMES = ["chair", "person"]

img_dir = BASE_DIR / "images" / split
xml_dir = BASE_DIR / "VOC" / split
out_img_dir = OUTPUT_DIR / "images" / split
out_label_dir = OUTPUT_DIR / "labels" / split
out_img_dir.mkdir(parents=True, exist_ok=True)
out_label_dir.mkdir(parents=True, exist_ok=True)

xml_files = sorted(xml_dir.glob("*.xml"))
converted = 0
for xml_path in xml_files:
    stem = xml_path.stem
    img_path = None
    for ext in [".jpg", ".jpeg", ".png"]:
        candidate = img_dir / f"{stem}{ext}"
        if candidate.exists():
            img_path = candidate
            break
    if img_path is None:
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
        continue

    shutil.copy2(img_path, out_img_dir / img_path.name)
    with open(out_label_dir / f"{stem}.txt", "w") as f:
        f.write("\n".join(yolo_lines))
    converted += 1

n_imgs = len(list(out_img_dir.glob("*")))
n_lbls = len(list(out_label_dir.glob("*.txt")))
print(f"{split}: 转换 {converted} 张 | 图片 {n_imgs} | 标注 {n_lbls}")
