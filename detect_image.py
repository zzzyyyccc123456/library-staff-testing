"""
图片检测 - 使用训练好的座位检测模型

用法:
    python detect_image.py <图片路径>
    
    或直接运行 -> 输入图片路径
"""
import sys, os
os.environ["ULTRALYTICS_DISABLE_REQUIREMENTS"] = "1"

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = r"d:\51.4\5555\yolo11_seat_best.pt"
CLASS_NAMES = ["chair", "person"]

def detect_image(image_path):
    """检测单张图片并显示结果"""
    if not os.path.exists(image_path):
        print("文件不存在:", image_path)
        return
    
    # 加载模型
    print("加载模型:", MODEL_PATH)
    model = YOLO(MODEL_PATH)
    
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print("无法读取图片:", image_path)
        return
    
    h, w = img.shape[:2]
    print("图片尺寸: %dx%d" % (w, h))
    
    # 执行检测
    print("检测中...")
    results = model(img, conf=0.45)[0]
    
    # 统计
    chairs = []
    persons = []
    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            if cls_id == 0:  # chair
                chairs.append([x1, y1, x2, y2])
            elif cls_id == 1:  # person
                persons.append([x1, y1, x2, y2])
    
    # 判断就座（简易 IoM）
    seated = 0
    seated_persons = []
    standing_persons = []
    
    for px1, py1, px2, py2 in persons:
        is_seated = False
        for cx1, cy1, cx2, cy2 in chairs:
            # 计算 IoM
            ix1 = max(px1, cx1)
            iy1 = max(py1, cy1)
            ix2 = min(px2, cx2)
            iy2 = min(py2, cy2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            p_area = (px2 - px1) * (py2 - py1)
            c_area = (cx2 - cx1) * (cy2 - cy1)
            iom = inter / min(p_area, c_area) if min(p_area, c_area) > 0 else 0
            if iom >= 0.3:
                is_seated = True
                break
        if is_seated:
            seated += 1
            seated_persons.append([px1, py1, px2, py2])
        else:
            standing_persons.append([px1, py1, px2, py2])
    
    # 打印结果
    print("\n检测结果:")
    print("  椅子数: %d" % len(chairs))
    print("  总人数: %d" % len(persons))
    print("  就座: %d  站立: %d" % (seated, len(persons) - seated))
    
    # 画框
    for cx1, cy1, cx2, cy2 in chairs:
        cv2.rectangle(img, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
        cv2.putText(img, "Chair", (cx1, cy1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    for px1, py1, px2, py2 in seated_persons:
        cv2.rectangle(img, (px1, py1), (px2, py2), (255, 0, 0), 2)
        cv2.putText(img, "Seated", (px1, py1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    for px1, py1, px2, py2 in standing_persons:
        cv2.rectangle(img, (px1, py1), (px2, py2), (0, 0, 255), 2)
        cv2.putText(img, "Standing", (px1, py1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # 信息
    info = "Chairs: %d  Persons: %d  Seated: %d  Standing: %d" % (len(chairs), len(persons), seated, len(persons)-seated)
    cv2.putText(img, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 保存
    out_path = os.path.splitext(image_path)[0] + "_result.jpg"
    cv2.imwrite(out_path, img)
    print("\n结果已保存:", out_path)
    
    # 显示
    cv2.imshow("Detection Result", img)
    print("按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        detect_image(sys.argv[1])
    else:
        print("=" * 50)
        print("座位检测 - 图片检测工具")
        print("=" * 50)
        path = input("输入图片路径: ").strip()
        if path:
            detect_image(path)
        else:
            print("未输入路径")
