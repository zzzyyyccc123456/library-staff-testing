# 摄像头节点开发 — 使用手册

## 项目简介

本项目实现了一套基于 **ESP32-CAM + YOLOv11** 的摄像头检测系统，用于实时检测场景中的人、椅子/桌子，并通过 IoM（Intersection over Minimum）算法判断人员就座状态。系统由 PC 端检测服务和 ESP32-CAM 硬件固件两部分组成。

---

## 目录结构

```
摄像头节点开发/
├── ESP32_PersonTableCam/       ← ESP32-CAM 摄像头固件（拍照上传）
├── esp32s3_seat_node/          ← ESP32-S3 座位节点固件（接收摄像头数据）
├── dataset_yolo_seat/          ← YOLO 训练数据集（标注+图片）
├── training_outputs/           ← 训练输出（模型权重、曲线图）
├── images/                     ← 原始图片（train/test 分集）
├── library_seatoccuped(DST1037)/ ← VOC 格式原始数据集
│
├── esp32_server.py             ← PC 端主服务器（核心）
├── person_table_analysis.py    ← 检测分析模块（YOLO + IoM）
├── detect_image.py             ← 单张图片检测工具
├── train_seat_model.py         ← 模型训练入口
├── train_seat_yolo11.py        ← YOLOv11 训练脚本
├── train_final.py              ← 最终版训练脚本
├── start_server.py             ← 简化版服务器启动
├── start_train.py              ← 训练启动脚本
├── import_dataset.py           ← 数据集导入脚本
├── convert_voc_to_yolo.py      ← VOC → YOLO 格式转换
├── convert_split.py            ← 单分片格式转换
├── scan_esp32.py               ← 局域网扫描 ESP32
│
├── yolo11_seat_best.pt         ← 训练好的座位检测模型
├── yolo11n.pt                  ← YOLOv11n 预训练基础模型
│
├── viewer.html                 ← 实时检测网页查看器
├── start_camera.bat            ← 一键启动检测服务
├── run_train.bat               ← 一键运行训练
├── run_convert.bat             ← 一键转换数据集
├── run_test.bat                ← 测试脚本
└── run_training_direct.vbs     ← 直接训练启动
```

---

## 快速开始

### 环境要求

- **Python 3.8+**（推荐 3.9）
- 安装依赖：

```bash
pip install flask ultralytics opencv-python numpy
```

- **ESP32-CAM** 硬件（AI-Thinker 或兼容型号）
- Arduino IDE（用于编译上传固件）

### 一键启动检测服务

```bash
start_camera.bat
```

或手动运行：

```bash
python esp32_server.py
```

服务启动后会在 `http://localhost:8080` 监听。

---

## 一、PC 端检测服务

### 1.1 启动服务

```bash
python esp32_server.py
```

启动后终端会显示本机局域网 IP 和配置信息。服务默认监听 **0.0.0.0:8080**。

### 1.2 服务端点

| 端点 | 方法 | 功能 |
|------|------|------|
| `/upload` | POST | 接收 ESP32-CAM 上传的图片（multipart/form-data, 字段名 `image`） |
| `/health` | GET | 健康检查，返回当前配置 |
| `/config` | POST | 动态更新检测参数 |
| `/result` | GET | 返回最近一次检测结果 |
| `/annotated-stream` | GET | 代理 ESP32 视频流并实时叠加检测框（MJPEG） |
| `/` | GET | 提供 Viewer 前端页面 |
| `/debug` | GET | 查看服务器内部状态 |

### 1.3 动态配置

通过 `/config` 端点可在运行时调整参数，无需重启服务：

```bash
# 调整 IoM 阈值和置信度
curl -X POST http://localhost:8080/config \
  -H "Content-Type: application/json" \
  -d '{"iom_threshold": 0.4, "confidence": 0.5}'
```

可调参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `iom_threshold` | 0.3 | IoM 阈值，大于此值判定为"就座" |
| `confidence` | 0.45 | YOLO 检测置信度 |
| `smooth_alpha` | 0.6 | 帧间平滑系数（0~1），越大越灵敏 |
| `min_person_area` | 800 | 人框最小面积（像素），过滤闪烁小框 |
| `nms_iou` | 0.5 | NMS 去重 IoU 阈值 |
| `jpeg_quality` | 75 | 推流 JPEG 质量（0~100） |

### 1.4 模型优先级

服务启动时按以下优先级自动加载模型：

1. **座位检测模型** `yolo11_seat_best.pt`（检测 chair + person）→ 默认使用
2. 旧微调模型 `train_output/.../best.pt`（如果存在）
3. 基础预训练模型 `YOLOV/yolo11n.pt`（兜底）

---

## 二、ESP32-CAM 固件使用

### 2.1 固件位置

固件位于 `ESP32_PersonTableCam/PersonTableCam/` 目录：

- `PersonTableCam.ino` — 主程序
- `camera_pins.h` — 摄像头引脚定义

### 2.2 配置说明

在 `PersonTableCam.ino` 中修改以下配置：

```cpp
const char* ssid = "你的WiFi名称";
const char* password = "你的WiFi密码";

const char* pc_server_host = "PC的局域网IP";  // 运行 esp32_server.py 的电脑 IP
const int pc_server_port = 8080;
```

### 2.3 摄像头型号

固件默认使用 **AI-Thinker ESP32-CAM**。如需更换型号，取消对应宏定义注释：

```cpp
#define CAMERA_MODEL_AI_THINKER      // AI-Thinker（默认）
// #define CAMERA_MODEL_ESP_EYE
// #define CAMERA_MODEL_M5STACK_PSRAM
// #define CAMERA_MODEL_ESP32S3_EYE
```

### 2.4 上传固件

1. 在 Arduino IDE 中打开 `PersonTableCam.ino`
2. 选择开发板：`AI Thinker ESP32-CAM`
3. 配置 PSRAM：启用
4. 编译并上传到 ESP32-CAM

### 2.5 工作流程

```
ESP32-CAM 拍照 → HTTP POST 上传到 PC → YOLOv11 检测 → IoM 分析 → 返回 JSON 结果
```

ESP32-CAM 上电后会自动连接 WiFi，并在串口输出分配的 IP 地址。可通过以下端点交互：

| 端点 | 功能 |
|------|------|
| `http://<ESP32_IP>/capture` | 拍照并上传检测 |
| `http://<ESP32_IP>/status` | 查看最近检测结果 |
| `http://<ESP32_IP>/stream` | MJPEG 视频流 |

---

## 三、Web 查看器

打开浏览器访问 `http://localhost:8080` 即可进入 Viewer 页面。

功能：
- 输入 ESP32-CAM 的 IP 地址并点击"连接"
- 实时查看带检测框的视频流（蓝色=就座，红色=站立，绿色=椅子）
- 实时显示统计信息（总人数、就座数、站立数、椅子数、密度等级）

---

## 四、单张图片检测

无需 ESP32 硬件，可直接用图片测试检测效果：

```bash
python detect_image.py 图片路径.jpg
```

或直接运行后输入图片路径。检测结果会在窗口显示并保存为 `图片名_result.jpg`。

---

## 五、模型训练

### 5.1 数据集转换

如果使用 VOC 格式标注数据，先转换：

```bash
run_convert.bat
```

或手动运行：

```bash
python convert_voc_to_yolo.py
```

### 5.2 启动训练

```bash
run_train.bat
```

或手动运行：

```bash
python train_seat_model.py
```

训练参数（在 `train_seat_model.py` 中修改）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_name` | `yolo11n.pt` | 基础模型 |
| `epochs` | 50 | 训练轮数 |
| `imgsz` | 640 | 输入图片尺寸 |
| `batch` | 16 | 批大小 |
| `device` | "cpu" | 训练设备（可改为 "cuda"） |

### 5.3 训练输出

训练完成后，最佳模型会保存在 `training_outputs/best.pt`，同时自动复制到文件夹根目录作为 `yolo11_seat_best.pt`。训练曲线和评估结果存放在 `training_outputs/` 目录中。

---

## 六、局域网扫描

如需查找局域网内的 ESP32 设备：

```bash
python scan_esp32.py
```

该脚本会扫描当前网段中开放的 80 端口，打印出所有 ESP32-CAM 的 IP 地址。

---

## 七、常见问题

### Q: 服务启动后 ESP32-CAM 连接不上

1. 确保 PC 和 ESP32-CAM 在同一个 WiFi 网络
2. 检查 `pc_server_host` 是否为 PC 的正确局域网 IP
3. 检查 PC 防火墙是否放行 8080 端口
4. 运行 `scan_esp32.py` 确认 ESP32-CAM 在线

### Q: 检测结果不准确

- 降低 `confidence` 值可检测更多目标（但会增加误报）
- 调整 `iom_threshold` 改变就座判定灵敏度
- 重新训练模型，增加更多场景的标注数据

### Q: 视频流卡顿

- 提高 `CAPTURE_INTERVAL` 减少拍照频率
- 降低 `jpeg_quality` 值减小图片体积
- 检查 WiFi 信号强度

### Q: 训练时报错

- 训练脚本已设置 `ULTRALYTICS_DISABLE_REQUIREMENTS=1` 防止断网卡死
- 确保已安装 ultralytics 库：`pip install ultralytics`
- 查看 `train_log.txt` 获取详细错误信息
