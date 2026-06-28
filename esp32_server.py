# -*- coding: utf-8 -*-
"""
PC 端检测服务 - 接收 ESP32-CAM 图片，进行 YOLOv11 检测 + IoM 分析
===================================================================

适配自 CameraWebServer（原人脸检测服务）
原功能: 人脸检测结果接收
新功能: YOLOv11 person + table 检测 + IoM 重叠度分析 + 密度估算

依赖:
  pip install flask ultralytics opencv-python numpy

启动:
  python esp32_server.py

ESP32-CAM 访问地址:
  http://<PC_IP>:5000/upload  (POST multipart image)
"""

import os
import sys

# 修复 Windows 中文主机名导致 socket.getfqdn 编码错误
import socket
_original_getfqdn = socket.getfqdn

def _patched_getfqdn(name=''):
    try:
        return _original_getfqdn(name)
    except UnicodeDecodeError:
        return 'localhost'

socket.getfqdn = _patched_getfqdn

import cv2
import numpy as np
import threading
import time
try:
    import requests
except ImportError:
    requests = None
    print("[警告] 未安装 requests，代理流功能将不可用")
from flask import Flask, request, jsonify, send_file, Response
from pathlib import Path

# 导入自定义分析模块
# 注意: 直接导入 ultralytics + 自定义 Analyzer，避免导入卡死
from person_table_analysis import PersonTableAnalyzer

app = Flask(__name__)

# 全局状态：缓存最新检测结果和 ESP32 IP
last_result = None
_esp32_ip = None
_proxy_running = False
_proxy_running_lock = threading.Lock()

# 代理视频流全局状态
latest_annotated_frame = None
stream_lock = threading.Lock()
proxy_thread_started = False

# ========== 模型配置 ==========
# 按优先级自动选择最佳可用模型：
#   1. 训练好的座位检测模型 (train_seat_model.py)
#   2. 旧微调模型
#   3. 默认预训练模型

SEAT_MODEL = r'd:\51.4\5555\yolo11_seat_best.pt'
FINETUNE_SMALL = r'd:\51.4\5555\train_output\yolo11s_person_table\weights\best.pt'
FINETUNE_NANO = r'd:\51.4\5555\train_output\yolo11_finetune\weights\best.pt'
BASE_MODEL = r'd:\51.4\5555\YOLOV\yolo11n.pt'

if os.path.exists(SEAT_MODEL):
    MODEL_PATH = SEAT_MODEL
    USE_SEAT_MODEL = True
    print(f"[配置] 使用座位检测模型 (chair+person): {MODEL_PATH}")
elif os.path.exists(FINETUNE_SMALL):
    MODEL_PATH = FINETUNE_SMALL
    USE_SEAT_MODEL = False
    print(f"[配置] 使用改进训练模型 (yolo11s): {MODEL_PATH}")
elif os.path.exists(FINETUNE_NANO):
    MODEL_PATH = FINETUNE_NANO
    USE_SEAT_MODEL = False
    print(f"[配置] 使用旧微调模型 (yolo11n): {MODEL_PATH}")
else:
    MODEL_PATH = BASE_MODEL
    USE_SEAT_MODEL = False
    print(f"[配置] 使用默认预训练模型: {MODEL_PATH}")

# ========== 检测配置 ==========
IOM_THRESHOLD = 0.3      # IoM 阈值：大于此值认为"人就座"
JPEG_QUALITY = 75        # 推流 JPEG 质量 (0-100)，越大画质越好
SMOOTH_ALPHA = 0.6       # 帧间平滑系数，越大框越灵敏，越小越平滑
MIN_PERSON_AREA = 800    # 人框最小面积(像素)，过滤闪烁小框
NMS_IOU = 0.5            # NMS IoU 阈值，去除重叠框

# ========== 全局初始化 ==========
print(f"[初始化] 加载 YOLOv11 模型: {MODEL_PATH}")
print(f"[初始化] IoM 阈值: {IOM_THRESHOLD}")

analyzer = PersonTableAnalyzer(
    model_path=MODEL_PATH,
    iom_threshold=IOM_THRESHOLD,
    smooth_alpha=SMOOTH_ALPHA,
    min_person_area=MIN_PERSON_AREA,
    nms_iou=NMS_IOU,
    use_seat_model=USE_SEAT_MODEL,
)

print("[初始化] 服务就绪，等待 ESP32-CAM 连接...")


@app.route('/upload', methods=['POST'])
def handle_upload():
    """
    接收 ESP32-CAM 上传的图片，进行检测分析
    
    请求: multipart/form-data, 字段名 "image"
    返回: JSON 格式的检测结果
    """
    # 检查是否有图片数据
    if 'image' not in request.files:
        # 也支持 raw binary 上传
        if request.data:
            img_data = request.data
        else:
            return jsonify({"error": "未找到图片数据"}), 400
    else:
        img_file = request.files['image']
        img_data = img_file.read()

    # 解码图片
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "图片解码失败"}), 400

    # 执行 YOLO 检测 + IoM 分析 + 密度估算
    result = analyzer.analyze(img)

    # 添加 IoM 阈值信息
    result['iom_threshold'] = IOM_THRESHOLD

    # 缓存最新结果和 ESP32 IP（供前端 Viewer 使用）
    global last_result, _esp32_ip
    last_result = result
    _esp32_ip = request.remote_addr

    # 打印日志
    print(f"[检测] 人数={result['total_persons']}, "
          f"就座={result['seated']}, 站立={result['standing']}, "
          f"桌子={result['table_count']}, 密度={result['density']['density_level']}")

    return jsonify(result)


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({
        "status": "ok",
        "model": str(MODEL_PATH),
        "use_seat_model": USE_SEAT_MODEL,
        "iom_threshold": IOM_THRESHOLD,
        "confidence": analyzer.confidence,
        "smooth_alpha": SMOOTH_ALPHA,
        "min_person_area": MIN_PERSON_AREA,
        "nms_iou": NMS_IOU,
        "jpeg_quality": JPEG_QUALITY,
    })


@app.route('/config', methods=['POST'])
def update_config():
    """
    动态更新配置（不需要重启服务）
    
    请求体 JSON:
    {
        "iom_threshold": 0.5,
        "confidence": 0.6
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "无效的 JSON"}), 400

    global IOM_THRESHOLD, SMOOTH_ALPHA, MIN_PERSON_AREA, NMS_IOU, JPEG_QUALITY

    if 'iom_threshold' in data:
        IOM_THRESHOLD = float(data['iom_threshold'])
        analyzer.iom_threshold = IOM_THRESHOLD
        print(f"[配置] IoM 阈值更新为: {IOM_THRESHOLD}")

    if 'confidence' in data:
        analyzer.confidence = float(data['confidence'])
        print(f"[配置] 置信度更新为: {analyzer.confidence}")

    if 'smooth_alpha' in data:
        SMOOTH_ALPHA = float(data['smooth_alpha'])
        analyzer.smooth_alpha = SMOOTH_ALPHA
        print(f"[配置] 平滑系数更新为: {SMOOTH_ALPHA}")

    if 'min_person_area' in data:
        MIN_PERSON_AREA = int(data['min_person_area'])
        analyzer.min_person_area = MIN_PERSON_AREA
        print(f"[配置] 最小人框面积更新为: {MIN_PERSON_AREA}")

    if 'nms_iou' in data:
        NMS_IOU = float(data['nms_iou'])
        analyzer.nms_iou = NMS_IOU
        print(f"[配置] NMS IoU 更新为: {NMS_IOU}")

    if 'jpeg_quality' in data:
        JPEG_QUALITY = int(data['jpeg_quality'])
        print(f"[配置] JPEG 质量更新为: {JPEG_QUALITY}")

    return jsonify({
        "status": "ok",
        "iom_threshold": IOM_THRESHOLD,
        "confidence": analyzer.confidence,
        "smooth_alpha": SMOOTH_ALPHA,
        "min_person_area": MIN_PERSON_AREA,
        "nms_iou": NMS_IOU,
        "jpeg_quality": JPEG_QUALITY,
    })


@app.route('/result', methods=['GET'])
def get_result():
    """返回最近一次检测的完整结果（含框坐标）"""
    if last_result is None:
        return jsonify({"error": "暂无检测结果"}), 404
    return jsonify(last_result)


@app.route('/esp32-ip', methods=['GET'])
def get_esp32_ip_route():
    """返回记录的 ESP32 IP，方便前端自动连接"""
    global _esp32_ip
    return jsonify({"esp32_ip": _esp32_ip})


@app.route('/')
def index():
    """提供 Viewer 前端页面"""
    return send_file(r'd:\51.4\5555\viewer.html')


# 预生成"等待中"占位帧，在代理流尚未就绪时显示
_waiting_img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(_waiting_img, "Connecting to ESP32 stream...", (60, 240),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
cv2.putText(_waiting_img, "Please wait...", (200, 290),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
_, _waiting_buf = cv2.imencode('.jpg', _waiting_img)
WAITING_FRAME = _waiting_buf.tobytes() if _ is not None else b''


@app.route('/annotated-stream')
def annotated_stream():
    """代理 ESP32 视频流并实时叠加检测框"""
    def generate():
        empty_count = 0
        while True:
            with stream_lock:
                frame = latest_annotated_frame
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Content-Length: ' + str(len(frame)).encode() + b'\r\n\r\n'
                       + frame + b'\r\n')
                empty_count = 0
            else:
                # 无数据时发送占位帧，避免浏览器黑屏等待
                if empty_count < 120:  # 最多持续约 6 秒
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n'
                           b'Content-Length: ' + str(len(WAITING_FRAME)).encode() + b'\r\n\r\n'
                           + WAITING_FRAME + b'\r\n')
                    empty_count += 1
                    time.sleep(0.05)
                else:
                    break  # 断开让浏览器重连
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/set-esp32-ip', methods=['POST'])
def set_esp32_ip_route():
    """前端手动设置 ESP32 IP，触发代理流连接"""
    global _esp32_ip
    data = request.get_json()
    if data and 'ip' in data:
        _esp32_ip = data['ip']
        print(f"[设置IP] 前端设置 IP={_esp32_ip}")
        return jsonify({"status": "ok", "esp32_ip": _esp32_ip})
    print("[设置IP] 缺少 ip 参数")
    return jsonify({"error": "缺少 ip 参数"}), 400


@app.route('/debug')
def debug():
    """调试端点：显示服务器内部状态"""
    global _esp32_ip, latest_annotated_frame, last_result
    thread_alive = False
    for t in threading.enumerate():
        if t.name and 'proxy' in t.name.lower():
            thread_alive = True
            break
    return jsonify({
        "esp32_ip": _esp32_ip,
        "proxy_thread_alive": thread_alive,
        "has_annotated_frame": latest_annotated_frame is not None,
        "latest_result_exists": last_result is not None,
        "thread_count": threading.active_count(),
        "thread_names": [t.name for t in threading.enumerate()],
    })


def proxy_stream_loop():
    """后台线程：持续拉取 ESP32 原始流，定时检测并画框"""
    import cv2 as _cv2
    import numpy as _np
    global latest_annotated_frame, last_result, _esp32_ip
    while True:
        ip = _esp32_ip
        if not ip:
            time.sleep(0.5)
            continue
        try:
            url = f"http://{ip}/stream"
            print(f"[代理流] 连接 ESP32: {url}")
            # 连接超时 5s，读取不设超时（由 ESP32 端主动关闭连接）
            r = requests.get(url, stream=True, timeout=(5, None))
            bytes_buffer = bytes()
            last_detect_time = 0
            current_result = None
            for chunk in r.iter_content(chunk_size=4096):
                bytes_buffer += chunk
                a = bytes_buffer.find(b'\xff\xd8')
                b = bytes_buffer.find(b'\xff\xd9')
                if a != -1 and b != -1 and b > a:
                    jpg = bytes_buffer[a:b+2]
                    bytes_buffer = bytes_buffer[b+2:]
                    img = _cv2.imdecode(_np.frombuffer(jpg, _np.uint8), _cv2.IMREAD_COLOR)
                    if img is not None:
                        now = time.time()
                        # 每 400ms 做一次检测
                        if now - last_detect_time > 0.4:
                            current_result = analyzer.analyze(img)
                            current_result['iom_threshold'] = IOM_THRESHOLD
                            last_result = current_result
                            last_detect_time = now
                        # 画框
                        if current_result:
                            img = analyzer.draw_results(img, current_result)
                        ret, buf = _cv2.imencode('.jpg', img, [int(_cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                        if ret:
                            with stream_lock:
                                latest_annotated_frame = buf.tobytes()
            r.close()
        except requests.exceptions.ReadTimeout:
            print("[代理流] 连接空闲超时，准备重连...")
            time.sleep(0.5)
        except Exception as e:
            print(f"[代理流] 断开或错误: {e}")
            time.sleep(0.5)


def start_proxy_thread():
    global proxy_thread_started
    if not proxy_thread_started and requests is not None:
        proxy_thread_started = True
        t = threading.Thread(target=proxy_stream_loop, name='proxy-stream-thread', daemon=True)
        t.start()
        print("[代理流] 后台拉流线程已启动")


start_proxy_thread()


if __name__ == '__main__':
    # 获取本机局域网 IP
    import socket as _socket
    try:
        s = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "0.0.0.0"

    print("\n" + "=" * 50)
    print("座位检测服务 (chair + person)")
    print("=" * 50)
    print(f"  监听地址: 0.0.0.0:8080 (本机IP: {local_ip})")
    print(f"  上传端点: POST /upload")
    print(f"  健康检查: GET  /health")
    print(f"  配置更新: POST /config")
    print(f"  模型: {MODEL_PATH}")
    print(f"  座位模型: {USE_SEAT_MODEL}")
    print(f"  IoM 阈值: {IOM_THRESHOLD}  置信度: {analyzer.confidence}")
    print(f"  平滑系数: {SMOOTH_ALPHA}  最小人框面积: {MIN_PERSON_AREA}")
    print(f"  NMS IoU: {NMS_IOU}  JPEG质量: {JPEG_QUALITY}")
    print("=" * 50 + "\n")

    # 使用 werkzeug 的 run_simple 直接启动，绑定所有网卡
    from werkzeug.serving import run_simple
    run_simple("0.0.0.0", 8080, app, threaded=True, use_reloader=False)
