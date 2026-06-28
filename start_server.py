"""启动检测服务 - 简化版"""
import os
os.environ["ULTRALYTICS_DISABLE_REQUIREMENTS"] = "1"
os.environ["YOLO_VERBOSE"] = "0"

import sys, time, cv2, numpy as np, json, threading
from flask import Flask, request, jsonify, send_file, Response
from pathlib import Path
from ultralytics import YOLO
import urllib.request
import io

app = Flask(__name__)

# ========== YOLO 配置 ==========
MODEL_PATH = r"d:\51.4\5555\yolo11_seat_best.pt"
IOM_THRESHOLD = 0.3
CONFIDENCE = 0.45

# ========== MQTT 配置（OneNet）==========
# 从 ESP32-S3 代码中获取，需在 OneNet 平台创建产品后填入
MQTT_CONFIG = {
    "host": "183.230.40.39",
    "port": 6002,
    "product_id": "你的产品ID",   # ← 改为与你ESP32-S3相同的产品ID
    "device_name": "pc-seat-server",
    "token": "你的PC设备TOKEN",    # ← 改为OneNet为pc-seat-server生成的TOKEN
    "seat_id": "A01",
}
CAMERA_TOPIC = "library/seat/%s/camera" % MQTT_CONFIG["seat_id"]

# 尝试导入 MQTT 库（非必需，不阻塞主服务）
try:
    import paho.mqtt.client as mqtt
    HAS_MQTT = True
except ImportError:
    HAS_MQTT = False
    print("[MQTT] paho-mqtt 未安装，摄像头结果不会推送到云")
    print("[MQTT] 如需推送请运行: pip install paho-mqtt")

# 加载模型
print("加载模型:", MODEL_PATH)
model = YOLO(MODEL_PATH)
print("模型加载成功!")

last_result = None
latest_frame = None
frame_lock = None

_esp32_ip = None
_proxy_running = False
_proxy_running_lock = threading.Lock()
latest_annotated_frame = None
annotated_stream_lock = threading.Lock()

# 等待帧
_waiting = np.zeros((480,640,3),dtype=np.uint8)
cv2.putText(_waiting, "Waiting for ESP32...", (120,240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
_, _waiting_buf = cv2.imencode('.jpg', _waiting)
WAITING_BYTES = _waiting_buf.tobytes()

def detect(img):
    """检测图片，返回结果"""
    h, w = img.shape[:2]
    results = model(img, conf=CONFIDENCE)[0]
    
    chairs, persons = [], []
    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls[0])
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            if cls_id == 0:
                chairs.append([x1,y1,x2,y2])
            elif cls_id == 1:
                persons.append([x1,y1,x2,y2])
    
    seated = 0
    seated_list = []
    for px1,py1,px2,py2 in persons:
        is_seated = False
        for cx1,cy1,cx2,cy2 in chairs:
            ix1 = max(px1,cx1); iy1 = max(py1,cy1)
            ix2 = min(px2,cx2); iy2 = min(py2,cy2)
            inter = max(0, ix2-ix1) * max(0, iy2-iy1)
            area_p = (px2-px1)*(py2-py1)
            area_c = (cx2-cx1)*(cy2-cy1)
            iom = inter / min(area_p, area_c) if min(area_p,area_c)>0 else 0
            if iom >= IOM_THRESHOLD:
                is_seated = True
                break
        seated_list.append(is_seated)
        if is_seated:
            seated += 1
    
    density = "无"
    if len(persons) == 0: density = "无"
    elif len(persons) <= 2: density = "低"
    elif len(persons) <= 5: density = "中"
    elif len(persons) <= 10: density = "高"
    else: density = "极高"
    
    return {
        "total_persons": len(persons),
        "table_count": len(chairs),
        "seated": seated,
        "standing": len(persons) - seated,
        "seated_list": seated_list,
        "person_boxes": persons,
        "table_boxes": chairs,
        "density": {"density_level": density},
        "image_shape": [h, w],
    }

def draw(img, result):
    """画框"""
    for cx1,cy1,cx2,cy2 in result["table_boxes"]:
        cv2.rectangle(img, (cx1,cy1), (cx2,cy2), (0,255,0), 2)
        cv2.putText(img, "Chair", (cx1,cy1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    for i, (px1,py1,px2,py2) in enumerate(result["person_boxes"]):
        seated = result["seated_list"][i] if i < len(result["seated_list"]) else False
        color = (255,0,0) if seated else (0,0,255)
        label = "Seated" if seated else "Standing"
        cv2.rectangle(img, (px1,py1), (px2,py2), color, 2)
        cv2.putText(img, label, (px1,py1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    info = "Chairs:%d Persons:%d Seated:%d Standing:%d Dens:%s" % (result["table_count"], result["total_persons"], result["seated"], result["standing"], result["density"]["density_level"])
    cv2.putText(img, info, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return img


def publish_mqtt(result):
    """推送摄像头检测结果到 OneNet MQTT，ESP32-S3 订阅后做 OR 运算"""
    if not HAS_MQTT:
        return
    try:
        cfg = MQTT_CONFIG
        client = mqtt.Client(client_id=cfg["device_name"])
        client.username_pw_set(cfg["product_id"], cfg["token"])
        client.connect(cfg["host"], cfg["port"], timeout=3)
        
        seated = result["seated"]
        total = result["total_persons"]
        payload = {
            "cameraOccupied": seated > 0,
            "seatedCount": seated,
            "totalPersons": total,
            "personDensity": seated / max(total, 1),
            "iou": IOM_THRESHOLD,
            "source": "yolo11_seat"
        }
        msg = json.dumps(payload)
        client.publish(CAMERA_TOPIC, msg)
        client.disconnect()
        print("[MQTT] 推送成功: cameraOccupied=%s seated=%d/%d" % (seated > 0, seated, total))
    except Exception as e:
        print("[MQTT] 推送失败:", e)

@app.route('/upload', methods=['POST'])
def handle_upload():
    global last_result
    data = request.data if not request.files else request.files['image'].read()
    np_arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "图片解码失败"}), 400
    result = detect(img)
    last_result = result
    result["iom_threshold"] = IOM_THRESHOLD
    
    # 异步推送 MQTT（不影响检测响应速度）
    publish_mqtt(result)
    
    return jsonify(result)

@app.route('/health')
def health():
    return jsonify({"status":"ok","model":MODEL_PATH,"iom":IOM_THRESHOLD})

@app.route('/result')
def get_result():
    if last_result is None:
        return jsonify({"error":"暂无结果"}), 404
    return jsonify(last_result)

@app.route('/')
def index():
    return send_file(r'd:\51.4\5555\viewer.html')

@app.route('/annotated-stream')
def annotated_stream():
    """代理 ESP32 视频流并实时叠加检测框"""
    def generate():
        empty_count = 0
        while True:
            with annotated_stream_lock:
                frame = latest_annotated_frame
            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                empty_count = 0
            else:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + WAITING_BYTES + b'\r\n')
                empty_count += 1
                time.sleep(0.05)
            if empty_count > 100:
                break
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/set-esp32-ip', methods=['POST'])
def set_esp32_ip_route():
    """前端手动设置 ESP32 IP，触发代理流连接"""
    global _esp32_ip
    data = request.get_json()
    if data and 'ip' in data:
        _esp32_ip = data['ip']
        print(f"[设置IP] 前端设置 IP={_esp32_ip}")
        # 启动代理线程
        start_proxy()
        return jsonify({"status": "ok", "esp32_ip": _esp32_ip})
    print("[设置IP] 缺少 ip 参数")
    return jsonify({"error": "缺少 ip 参数"}), 400


@app.route('/esp32-ip', methods=['GET'])
def get_esp32_ip_route():
    """返回记录的 ESP32 IP，方便前端自动连接"""
    global _esp32_ip
    return jsonify({"esp32_ip": _esp32_ip})


def proxy_worker():
    """后台线程：从 ESP32-CAM 拉取视频流 → 检测 → 缓存标注帧"""
    global _proxy_running, latest_annotated_frame, last_result, _esp32_ip
    stream_url = "http://%s/stream" % _esp32_ip
    print("[代理] 启动线程，拉取 %s" % stream_url)
    
    while True:
        with _proxy_running_lock:
            if not _proxy_running:
                break
        try:
            resp = urllib.request.urlopen(stream_url, timeout=10)
            buf = b""
            while True:
                with _proxy_running_lock:
                    if not _proxy_running:
                        return
                chunk = resp.read(4096)
                if not chunk:
                    break
                buf += chunk
                # 找 JPEG 帧边界
                while True:
                    start = buf.find(b'\xff\xd8')
                    end = buf.find(b'\xff\xd9')
                    if start != -1 and end != -1 and end > start:
                        jpg_data = buf[start:end+2]
                        buf = buf[end+2:]
                        # 检测
                        np_arr = np.frombuffer(jpg_data, np.uint8)
                        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        if img is not None:
                            result = detect(img)
                            last_result = result
                            annotated = draw(img.copy(), result)
                            _, buf_enc = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            with annotated_stream_lock:
                                latest_annotated_frame = buf_enc.tobytes()
                            # 异步推送 MQTT
                            publish_mqtt(result)
                    else:
                        if start == -1 and len(buf) > 2:
                            buf = buf[-2:]
                        break
        except Exception as e:
            print("[代理] 流中断:", e)
            time.sleep(2)


def start_proxy():
    """启动代理线程（单例）"""
    global _proxy_running
    with _proxy_running_lock:
        if _proxy_running:
            return
        _proxy_running = True
    t = threading.Thread(target=proxy_worker, name="proxy-stream", daemon=True)
    t.start()
    print("[代理] 已启动")

if __name__ == '__main__':
    import socket as _sk
    try:
        s = _sk.socket(_sk.AF_INET, _sk.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
    except:
        ip = "127.0.0.1"
    
    print("\n" + "="*50)
    print("座位检测服务已启动!")
    print("="*50)
    print("访问地址: http://%s:8080/" % ip)
    print("上传端点: POST /upload")
    print("="*50 + "\n")
    sys.stdout.flush()
    
    try:
        # 使用标准库 wsgiref 替代 werkzeug，绕过中文 hostname 问题
        from wsgiref.simple_server import make_server, WSGIServer, WSGIRequestHandler
        import socketserver as _ss
        import http.server as _hs
        class _NoFQDNServer(WSGIServer):
            allow_reuse_address = True
            def server_bind(self):
                _ss.TCPServer.server_bind(self)
                self.server_name = ip
                self.server_port = 8080
                self.setup_environ()
        server = make_server("0.0.0.0", 8080, app, server_class=_NoFQDNServer)
        print("[服务] 监听 0.0.0.0:8080 ...")
        server.serve_forever()
    except Exception as e:
        print("[错误]", e)
        import traceback
        traceback.print_exc()
