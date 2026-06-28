"""导入数据集 - 调试版"""
import shutil, xml.etree.ElementTree as ET
from pathlib import Path

SRC = Path(r"d:\51.4\5555\library_seatoccuped(DST1037)")
DST = Path(r"d:\51.4\5555\dataset_yolo_seat")
CLASS_NAMES = ["chair", "person"]

log = open("d:/51.4/5555/import_out.txt", "w", encoding="utf-8")

for split in ["train", "valid", "test"]:
    src_img = SRC / "images" / split
    src_xml = SRC / "VOC" / split
    dst_img = DST / "images" / split
    dst_lbl = DST / "labels" / split

    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(src_xml.glob("*.xml"))
    log.write(f"{split}: {len(xml_files)} xml files\n")
    
    # Check first 5 image-XML matches
    matched = 0
    for xf in xml_files[:5]:
        stem = xf.stem
        ip = None
        for ext in [".jpg", ".jpeg", ".png"]:
            p = src_img / f"{stem}{ext}"
            if p.exists():
                ip = p
                matched += 1
                break
        log.write(f"  {stem}: {'FOUND ' + ip.name if ip else 'NOT FOUND'}\n")
    log.write(f"  first 5: {matched}/5 matched\n")

    conv = 0
    noimg = 0
    noobj = 0
    err = 0
    err_details = []

    for xf in xml_files:
        stem = xf.stem
        ip = None
        for ext in [".jpg", ".jpeg", ".png"]:
            p = src_img / f"{stem}{ext}"
            if p.exists():
                ip = p
                break
        if ip is None:
            noimg += 1
            if noimg <= 3:
                log.write(f"  NOIMG: {stem}\n")
            continue
        try:
            tree = ET.parse(xf)
            r = tree.getroot()
            sz = r.find("size")
            if sz is None:
                err += 1
                err_details.append(f"nosize:{stem}")
                continue
            iw = int(sz.find("width").text)
            ih = int(sz.find("height").text)
            lines = []
            for o in r.findall("object"):
                n = o.find("name").text.strip().lower()
                if n not in CLASS_NAMES:
                    continue
                cid = CLASS_NAMES.index(n)
                bb = o.find("bndbox")
                x1 = float(bb.find("xmin").text)
                y1 = float(bb.find("ymin").text)
                x2 = float(bb.find("xmax").text)
                y2 = float(bb.find("ymax").text)
                lines.append(f"{cid} {((x1+x2)/2)/iw:.6f} {((y1+y2)/2)/ih:.6f} {(x2-x1)/iw:.6f} {(y2-y1)/ih:.6f}")
            if not lines:
                noobj += 1
                continue
            shutil.copy2(ip, dst_img / ip.name)
            (dst_lbl / f"{stem}.txt").write_text("\n".join(lines))
            conv += 1
        except Exception as e:
            err += 1
            err_details.append(f"err:{stem}:{e}")

    log.write(f"{split}: conv={conv} noimg={noimg} noobj={noobj} err={err}\n")
    if err_details:
        log.write(f"  errors: {'; '.join(err_details[:10])}\n")

for s in ["train", "valid", "test"]:
    ni = len(list((DST/"images"/s).iterdir()))
    nl = len(list((DST/"labels"/s).glob("*.txt")))
    log.write(f"{s}: {ni} imgs, {nl} lbls\n")

# data.yaml
with open(DST / "data.yaml", "w") as f:
    f.write(f"path: {DST.as_posix()}\ntrain: images/train\nval: images/valid\ntest: images/test\nnc: 2\nnames: ['chair', 'person']\n")

log.close()
print("DONE")
