# -*- coding: utf-8 -*-
"""
人-桌检测分析模块 (PersonTableAnalyzer)
=========================================

适配自 CameraWebServer 的 AIProcessor（原人脸检测处理器）
原功能: FaceDetectionResult + drawFaceRect
新功能: YOLOv11 检测 + IoM 重叠度分析 + 密度估算

核心算法:
  1. YOLOv11 检测 person 和 table
  2. IoM (Intersection over Minimum) 计算人与桌的重叠度
  3. 密度估算 (总体 + 网格)
"""

import numpy as np
import cv2

# 延迟导入 YOLO（避免在 esp32_server 中导入时卡死）
_YOLO_MODEL_CLS = None

def _get_yolo_cls():
    global _YOLO_MODEL_CLS
    if _YOLO_MODEL_CLS is None:
        from ultralytics import YOLO
        _YOLO_MODEL_CLS = YOLO
    return _YOLO_MODEL_CLS


class PersonTableAnalyzer:
    """
    人-桌检测分析器
    
    使用 YOLOv11 检测 person 和 table，
    通过 IoM 算法判断是否有人就座，
    并估算场景人员密度。
    """

    # 类别映射:
    #   COCO 模型:    person=0, dining table=60
    #   自定义3类:    face=0, person=1, table=2    (use_face_model=True)
    #   座位模型:     chair=0, person=1            (use_seat_model=True)
    CLASS_PERSON = 0    # 映射后类别 ID
    CLASS_TABLE = 1     # 映射后类别 ID (对于座位模型=椅子)
    CLASS_FACE = 0      # 自定义 face 类别 ID（仅用于 v8 模型）

    def __init__(self, model_path='yolo11n.pt', iom_threshold=0.3, confidence=0.45,
                 smooth_alpha=0.6, min_person_area=800, nms_iou=0.5,
                 use_face_model=False, use_seat_model=True):
        """
        初始化分析器
        
        参数:
            model_path: YOLO 模型路径
            iom_threshold: IoM 阈值，大于此值认为"人就座"
            confidence: 检测置信度阈值
            smooth_alpha: 帧间平滑系数 (0~1)，越大越跟随当前帧，越小越平滑
            min_person_area: 人框最小面积（像素），过滤太小/闪烁的框
            nms_iou: 非极大抑制 IoU 阈值，去除重叠框
            use_face_model: True=自定义3类模型(face,person,table)
            use_seat_model: True=座位模型(chair=0, person=1)
        """
        YOLO = _get_yolo_cls()
        self.model = YOLO(model_path)
        self.iom_threshold = iom_threshold
        self.confidence = confidence
        self.smooth_alpha = smooth_alpha
        self.min_person_area = min_person_area
        self.nms_iou = nms_iou
        self.use_face_model = use_face_model
        self.use_seat_model = use_seat_model

        # 帧间平滑缓存
        self._prev_person_boxes = None
        self._prev_table_boxes = None
        self._prev_face_boxes = None

        # 检测统计
        self.total_detections = 0

        print(f"[PersonTableAnalyzer] 模型: {model_path}")
        print(f"[PersonTableAnalyzer] IoM 阈值: {iom_threshold}")
        print(f"[PersonTableAnalyzer] 座位模型: {use_seat_model}")
        print(f"[PersonTableAnalyzer] 人脸模型: {use_face_model}")
        print(f"[PersonTableAnalyzer] 平滑系数: {smooth_alpha}, 最小面积: {min_person_area}, NMS IoU: {nms_iou}")

    def _nms_filter(self, boxes, iou_threshold=None):
        """对重叠框进行 NMS 去重，保留置信度最高的框"""
        if not boxes or len(boxes) <= 1:
            return boxes
        if iou_threshold is None:
            iou_threshold = self.nms_iou
        # 按面积降序排列（面积大的优先保留）
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        sorted_idx = sorted(range(len(boxes)), key=lambda i: areas[i], reverse=True)
        keep = []
        for i in sorted_idx:
            keep_i = True
            for j in keep:
                # 计算 IoU
                x1 = max(boxes[i][0], boxes[j][0])
                y1 = max(boxes[i][1], boxes[j][1])
                x2 = min(boxes[i][2], boxes[j][2])
                y2 = min(boxes[i][3], boxes[j][3])
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                area_i = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
                area_j = (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1])
                iou = inter / (area_i + area_j - inter) if (area_i + area_j - inter) > 0 else 0
                if iou > iou_threshold:
                    keep_i = False
                    break
            if keep_i:
                keep.append(i)
        return [boxes[i] for i in keep]

    def _smooth_boxes(self, current_boxes, prev_boxes, alpha=None):
        """指数移动平均平滑框坐标，减少帧间抖动"""
        if alpha is None:
            alpha = self.smooth_alpha
        if prev_boxes is None or len(prev_boxes) != len(current_boxes):
            return [list(b) for b in current_boxes]
        smoothed = []
        for cur, prev in zip(current_boxes, prev_boxes):
            smoothed.append([
                alpha * cur[0] + (1 - alpha) * prev[0],
                alpha * cur[1] + (1 - alpha) * prev[1],
                alpha * cur[2] + (1 - alpha) * prev[2],
                alpha * cur[3] + (1 - alpha) * prev[3],
            ])
        return smoothed

    def detect(self, image):
        """
        对图片执行 YOLO 检测，返回 face, person 和 table 的边界框列表
        
        参数:
            image: OpenCV 图像 (numpy array, BGR格式)
            
        返回:
            face_boxes:   [[x1,y1,x2,y2], ...] 人脸框
            person_boxes: [[x1,y1,x2,y2], ...] 人框
            table_boxes:  [[x1,y1,x2,y2], ...] 桌子框
        """
        results = self.model(image, conf=self.confidence, iou=self.nms_iou, verbose=False)[0]
        h, w = results.orig_shape[0], results.orig_shape[1]

        raw_face_boxes = []
        raw_person_boxes = []
        raw_table_boxes = []

        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                mapped_cls = self._map_class(cls_id)
                if mapped_cls == self.CLASS_FACE and self.use_face_model:
                    raw_face_boxes.append([x1, y1, x2, y2])
                elif mapped_cls == self.CLASS_PERSON:
                    raw_person_boxes.append([x1, y1, x2, y2])
                elif mapped_cls == self.CLASS_TABLE:
                    raw_table_boxes.append([x1, y1, x2, y2])

        # 1. 最小面积过滤（去掉闪烁的小框）
        face_boxes = [b for b in raw_face_boxes
                      if (b[2] - b[0]) * (b[3] - b[1]) >= 200]  # 人脸框可以小一些
        person_boxes = [b for b in raw_person_boxes
                        if (b[2] - b[0]) * (b[3] - b[1]) >= self.min_person_area]

        # 2. NMS 去重
        face_boxes = self._nms_filter(face_boxes)
        person_boxes = self._nms_filter(person_boxes)
        table_boxes = self._nms_filter(raw_table_boxes)

        # 3. 帧间平滑
        face_boxes = self._smooth_boxes(face_boxes, self._prev_face_boxes)
        person_boxes = self._smooth_boxes(person_boxes, self._prev_person_boxes)
        table_boxes = self._smooth_boxes(table_boxes, self._prev_table_boxes)

        # 缓存平滑后的框供下一帧使用
        self._prev_face_boxes = [list(b) for b in face_boxes]
        self._prev_person_boxes = [list(b) for b in person_boxes]
        self._prev_table_boxes = [list(b) for b in table_boxes]

        return face_boxes, person_boxes, table_boxes

    def _map_class(self, coco_cls_id):
        """
        将模型输出的类别 ID 映射到我们的类别 ID
        
        COCO 模型: person=0, dining table=60
        自定义3类(face): face=0, person=1, table=2
        座位模型:        chair=0, person=1
        
        映射: person -> 0, table/chair -> 1, face -> 0
        """
        if self.use_seat_model:
            # 座位模型: 0=chair, 1=person
            if coco_cls_id == 1:  # person
                return self.CLASS_PERSON
            elif coco_cls_id == 0:  # chair
                return self.CLASS_TABLE
        elif self.use_face_model:
            # 自定义 3 类模型: 0=face, 1=person, 2=table
            if coco_cls_id == 0:   # face
                return self.CLASS_FACE
            elif coco_cls_id == 1:  # person
                return self.CLASS_PERSON
            elif coco_cls_id == 2:  # table
                return self.CLASS_TABLE
        else:
            # COCO 模型: 0=person, 60=dining table
            if coco_cls_id == 0:    # person
                return self.CLASS_PERSON
            elif coco_cls_id == 60:  # dining table
                return self.CLASS_TABLE
        return -1

    # ========== IoM 算法 (替代原 AIProcessor 的人脸检测) ==========

    @staticmethod
    def compute_iom(box_person, box_table):
        """
        计算 IoM (Intersection over Minimum)
        
        IoM = area(交集) / min(area(人框), area(桌子框))
        
        相比 IoU，IoM 更能反映"人是否坐在桌前"：
        - 当人身体大部分与桌子重叠时，IoM 很高
        - 当人只是路过（小部分重叠）时，IoM 很低
        
        参数:
            box_person: [x1, y1, x2, y2] 人框
            box_table:  [x1, y1, x2, y2] 桌子框
            
        返回:
            IoM 值 (0.0 ~ 1.0)
        """
        # 计算交集区域
        x1 = max(box_person[0], box_table[0])
        y1 = max(box_person[1], box_table[1])
        x2 = min(box_person[2], box_table[2])
        y2 = min(box_person[3], box_table[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)

        # 计算各自面积
        person_area = (box_person[2] - box_person[0]) * \
                      (box_person[3] - box_person[1])
        table_area = (box_table[2] - box_table[0]) * \
                     (box_table[3] - box_table[1])

        # IoM = 交集 / min(人面积, 桌子面积)
        min_area = min(person_area, table_area)

        if min_area == 0:
            return 0.0

        return inter_area / min_area

    def judge_seated(self, person_boxes, table_boxes):
        """
        判断每个人是否就座
        
        对每个人框，遍历所有桌子框计算 IoM，
        若任一 IoM >= 阈值，则判定为"就座"
        
        参数:
            person_boxes: [[x1,y1,x2,y2], ...]
            table_boxes:  [[x1,y1,x2,y2], ...]
            
        返回:
            seated_list:   [True/False, ...] 每个人是否就座
            seated_count:  就座人数
            standing_count: 站立/走动人数
        """
        seated_list = []

        for p_box in person_boxes:
            is_seated = False
            max_iom = 0.0

            for t_box in table_boxes:
                iom = self.compute_iom(p_box, t_box)
                if iom > max_iom:
                    max_iom = iom
                if iom >= self.iom_threshold:
                    is_seated = True
                    break

            seated_list.append(is_seated)

        seated_count = sum(seated_list)
        standing_count = len(person_boxes) - seated_count

        return seated_list, seated_count, standing_count

    # ========== 密度估算 ==========

    @staticmethod
    def estimate_density(person_boxes, image_shape, grid_rows=3, grid_cols=3):
        """
        估算人员密度
        
        参数:
            person_boxes:  [[x1,y1,x2,y2], ...]
            image_shape:   (height, width)
            grid_rows:     网格行数
            grid_cols:     网格列数
            
        返回:
            dict: {total_people, overall_density, density_level, grid_stats}
        """
        h, w = image_shape[:2]
        image_area = w * h
        total_people = len(person_boxes)

        # 1. 总体密度：人数 / 图像面积
        if image_area > 0:
            overall_density = total_people / image_area
        else:
            overall_density = 0.0

        # 2. 密度等级划分
        # 不同分辨率的密度标准不同，这里按相对比例
        if total_people == 0:
            level = "无"
        elif total_people <= 2:
            level = "低"
        elif total_people <= 5:
            level = "中"
        elif total_people <= 10:
            level = "高"
        else:
            level = "极高"

        # 3. 网格密度统计
        grid_h = h // grid_rows
        grid_w = w // grid_cols
        grid_counts = np.zeros((grid_rows, grid_cols), dtype=int)

        for box in person_boxes:
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            row = min(int(cy // grid_h), grid_rows - 1)
            col = min(int(cx // grid_w), grid_cols - 1)
            grid_counts[row][col] += 1

        grid_stats = []
        for r in range(grid_rows):
            for c in range(grid_cols):
                grid_stats.append({
                    "row": r, "col": c,
                    "count": int(grid_counts[r][c])
                })

        return {
            "total_people": total_people,
            "overall_density": round(overall_density, 6),
            "density_level": level,
            "grid": f"{grid_rows}x{grid_cols}",
            "grid_stats": grid_stats
        }

    # ========== 完整分析流水线 ==========

    def analyze(self, image):
        """
        完整的分析流水线：检测 -> IoM -> 密度
        
        这是主入口，替代原 AIProcessor 的 FaceDetectionResult
        
        参数:
            image: OpenCV 图像
            
        返回:
            dict: {
                "total_persons": int,
                "table_count": int,
                "face_count": int,
                "seated": int,
                "standing": int,
                "seated_list": [bool],
                "density": {...},
                "face_boxes": [[x1,y1,x2,y2]],
                "person_boxes": [[x1,y1,x2,y2]],
                "table_boxes": [[x1,y1,x2,y2]],
                "image_shape": [h, w]
            }
        """
        # Step 1: YOLO 检测（返回 3 个值）
        face_boxes, person_boxes, table_boxes = self.detect(image)

        # Step 2: IoM 判断就座（仅对 person 判断）
        seated_list, seated_count, standing_count = \
            self.judge_seated(person_boxes, table_boxes)

        # Step 3: 密度估算
        density = self.estimate_density(person_boxes, image.shape)

        # Step 4: 构建结果
        result = {
            "total_persons": len(person_boxes),
            "table_count": len(table_boxes),
            "face_count": len(face_boxes),
            "seated": seated_count,
            "standing": standing_count,
            "seated_list": seated_list,
            "face_boxes": face_boxes,
            "person_boxes": person_boxes,
            "table_boxes": table_boxes,
            "density": density,
            "image_shape": [image.shape[0], image.shape[1]],
            "model_status": "active"
        }

        self.total_detections += 1

        return result

    def draw_results(self, image, result):
        """
        在图像上绘制检测结果（调试用）

        参数:
            image: OpenCV 图像（会被修改）
            result: analyze() 返回的结果 dict

        返回:
            绘制后的图像
        """
        # 优先使用 result 中已有的框坐标，避免重复检测
        face_boxes = result.get("face_boxes", [])
        person_boxes = result.get("person_boxes", [])
        table_boxes = result.get("table_boxes", [])
        if not face_boxes and not person_boxes and not table_boxes:
            face_boxes, person_boxes, table_boxes = self.detect(image)

        # 绘制人脸框 (黄色)
        for box in face_boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(image, "Face", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 绘制桌子框 (绿色) —— 座位模型显示 Chair
        label_table = "Chair" if self.use_seat_model else "Table"
        for box in table_boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label_table, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 绘制人框 (就座=蓝色, 站立=红色)
        for i, box in enumerate(person_boxes):
            x1, y1, x2, y2 = [int(v) for v in box]
            is_seated = result["seated_list"][i] if i < len(result["seated_list"]) else False
            color = (255, 0, 0) if is_seated else (0, 0, 255)  # 蓝=就座, 红=站立
            label = "Person(S)" if is_seated else "Person(U)"
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 绘制信息
        table_label = "Chairs" if self.use_seat_model else "Tables"
        info = [
            f"Face: {result.get('face_count', 0)}  "
            f"Person: {result['total_persons']}  "
            f"Seated: {result['seated']}  Standing: {result['standing']}",
            f"{table_label}: {result['table_count']}  "
            f"Density: {result['density']['density_level']}"
        ]
        for i, text in enumerate(info):
            cv2.putText(image, text, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return image


# ========== 独立测试 ==========
if __name__ == '__main__':
    import cv2

    print("PersonTableAnalyzer 自检...")
    analyzer = PersonTableAnalyzer()

    # 测试 IoM 计算
    person = [50, 50, 150, 200]   # 人框
    table = [100, 80, 300, 220]   # 桌子框

    iom = PersonTableAnalyzer.compute_iom(person, table)
    print(f"IoM 计算测试: person={person}, table={table}")
    print(f"  IoM = {iom:.3f}")

    # 测试密度估算
    density = PersonTableAnalyzer.estimate_density(
        [person, [200, 50, 300, 150], [400, 100, 500, 200]],
        (480, 640)
    )
    print(f"\n密度估算测试:")
    print(f"  总人数: {density['total_people']}")
    print(f"  密度等级: {density['density_level']}")
    print(f"  网格统计: {density['grid_stats']}")

    print("\n自检完成！")
