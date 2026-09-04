# gatekeeper_V4 — 区域人数统计系统

> 运行于 RK3588 嵌入式平台的高精度区域人数统计算法。基于 YOLOv8 目标检测 + ByteTrack **决策级融合跟踪**（轨迹引导置信度校准 + IoU/运动/外观联合评分匹配），配合**跟踪位置记忆状态机**对抗遮挡场景，实现稳定的人员进出计数。

---

## 目录

- [功能特性](#功能特性)
- [系统架构](#系统架构)
- [依赖环境](#依赖环境)
- [编译构建](#编译构建)
- [快速开始](#快速开始)
  - [方式一：海康摄像头实时流](#方式一海康摄像头实时流)
  - [方式二：离线视频调试](#方式二离线视频调试)
  - [命令行参数说明](#命令行参数说明)
- [配置文件](#配置文件)
  - [格式说明](#格式说明)
  - [示例](#示例)
- [计数算法详解](#计数算法详解)
  - [状态机设计](#状态机设计)
  - [区域判定 (isInsideZone)](#区域判定-isinsidezone)
  - [计数平滑](#计数平滑)
- [Python 配置工具](#python-配置工具)
- [输出与通信](#输出与通信)
- [目录结构](#目录结构)
- [关键参数速查](#关键参数速查)

---

## 功能特性

| 特性 | 说明 |
|------|------|
| **目标检测** | YOLOv8 INT8 量化模型，640x640 输入，2 类 (people / umbrella) |
| **决策级融合跟踪** | ByteTrack + 轨迹引导置信度动态校准 + IoU/马氏距离/HSV外观联合评分匹配 |
| **区域计数** | 跟踪位置记忆 4 状态机，S_OUT → S_IN → S_LEAVE → S_OUT |
| **遮挡对抗** | track_id 跨帧保持 (max_lost=60)，8 帧离开宽限期，80px 位移防抖动 |
| **区域判定** | 脚底 92% 处单点 + 底部 35% 面积比 (阈值 0.2) 双保险 |
| **计数平滑** | 10 帧滑动窗口众数滤波，消除瞬时波动 |
| **多路并发** | 每路摄像头独立 NPU 线程池，支持多路同时运行 |
| **运行模式** | 实时流 (main_tracking) / 离线视频调试 (video_debug) / 无头模式 (main_headless) |
| **硬件加速** | RK3588 NPU 3 核并行，RGA 硬件预处理（可选回退 OpenCV） |
| **数据持久化** | SQLite3 WAL 模式，存储检测计数结果 |
| **串口通信** | RS-485 类 Modbus RTU 协议，LRC 校验，115200 baud |

---

## 系统架构

```
cameras_config.txt
      │
      ▼
parseCameraConfig() → 检测区域掩膜 (detection_mask)
      │
      ▼
┌──────────────────────────────────────────────────────────────────┐
│                     处理流程（每帧）                                │
│                                                                  │
│  原始帧 ──→ YOLOv8 线程池 ──→ Detection[]                        │
│                  │                                               │
│    ┌─────────────▼──────────────────────────┐                    │
│    │  ByteTrack.update()  决策级融合         │                    │
│    │                                        │                    │
│    │  ① calibrateConfidence                 │                    │
│    │     稳定track 马氏距离匹配低分det        │                    │
│    │     → 动态提升置信度，纳入高置信池       │                    │
│    │                                        │                    │
│    │  ② greedyJointMatch (两轮)              │                    │
│    │     S = 0.4×IoU + 0.25×Motion          │                    │
│    │       + 0.2×Appearance + 0.15×Score     │                    │
│    │                                        │                    │
│    │  ③ EMA 更新 HSV 外观直方图               │                    │
│    └────────────────┬───────────────────────┘                    │
│                     ▼                                            │
│           区域 4 状态机 (per track_id)                             │
│     S_OUT → S_ENTER → S_IN → S_LEAVE → S_OUT                    │
│                     │                                            │
│            ┌────────▼────────┐                                   │
│            │  10 帧众数滤波   │                                   │
│            └────────┬────────┘                                   │
│                     ▼                                            │
│         GUI 显示 + SQLite 持久化 + 串口输出                        │
└──────────────────────────────────────────────────────────────────┘
```

---

## 依赖环境

### 运行平台

- **硬件**: RK3588 (或兼容开发板)
- **系统**: Linux (aarch64), Ubuntu 20.04+
- **NPU 驱动**: Rockchip RKNN Runtime >= 2.0

### 运行时依赖

| 库 | 用途 |
|----|------|
| **librknnrt.so** | RK3588 NPU 推理引擎 |
| **librga.so** | 硬件图像预处理（RGA） |
| **OpenCV >= 4.x** | 图像处理 / 显示 / 视频读写 |
| **X11** | 多线程 GUI 显示窗口 |
| **海康 SDK** (libhcnetsdk.so, libPlayCtrl.so) | 海康摄像头 RTSP 流接入 |
| **SQLite3** | 检测结果持久化 |
| **pthread** | 多线程支持 |

### 编译依赖

- CMake >= 3.14
- aarch64-linux-gnu 交叉编译工具链
- RK3588 SDK (RKNN API, RGA 头文件)
- 海康设备网络 SDK 头文件

---

## 编译构建

```bash
cd gatekeeper_V4
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

编译产物：

| 文件 | 说明 |
|------|------|
| `main_tracking` | 海康实时流主程序（含 GUI） |
| `main_headless` | 无头模式（无 GUI，终端+SQLite+串口） |
| `video_debug` | 离线视频调试程序（自动输出标注视频） |
| `libyolov8_lib.so` | YOLOv8 推理封装 |
| `libbytetrack_lib.so` | ByteTrack 决策级融合跟踪器 |
| `libnn_process.so` | 预处理/后处理 |
| `librknn_engine.so` | RKNN 引擎封装 |

---

## 快速开始

### 方式一：海康摄像头实时流

```bash
# 1. 将 YOLOv8 模型放到 weights/ 目录
cp yolov8n_people.rknn weights/

# 2. 编辑配置文件
vim cameras_config.txt

# 3. 启动（2 路摄像头，每路 2 个 NPU 线程）
./main_tracking weights/yolov8n_people.rknn cameras_config.txt 2
```

### 方式二：离线视频调试

```bash
# 测试本地视频，自动输出 output_v4.avi（含众数滤波 + 决策级融合跟踪）
./video_debug weights/yolov8n_people.rknn cameras_config.txt videos/多人10人.mp4 2
```

### 方式三：无头模式（生产部署）

```bash
# 无 GUI 依赖，终端运行 + SQLite 持久化 + 串口输出
./main_headless weights/yolov8n_people.rknn cameras_config.txt 2
```

### 命令行参数说明

#### main_tracking

| 参数位置 | 说明 | 示例 |
|----------|------|------|
| `argv[1]` | RKNN 模型路径 | `weights/yolov8n.rknn` |
| `argv[2]` | 摄像头配置文件 | `cameras_config.txt` |
| `argv[3]` | 每路摄像头 NPU 线程数（默认 2） | `2` |

#### main_headless

| 参数位置 | 说明 | 示例 |
|----------|------|------|
| `argv[1]` | RKNN 模型路径 | `weights/yolov8n.rknn` |
| `argv[2]` | 摄像头配置文件 | `cameras_config.txt` |
| `argv[3]` | 每路摄像头 NPU 线程数（默认 2） | `2` |

#### video_debug

| 参数位置 | 说明 | 示例 |
|----------|------|------|
| `argv[1]` | RKNN 模型路径 | `weights/yolov8n.rknn` |
| `argv[2]` | 配置文件（仅用第一项） | `cameras_config.txt` |
| `argv[3]` | 输入视频文件路径 | `videos/test.mp4` |
| `argv[4]` | NPU 线程数（默认 2） | `4` |

---

## 配置文件

### 格式说明

```
# 注释以 # 开头，可插入任意位置
IP 用户名 密码 通道 分辨率 [x1,y1 x2,y2 x3,y3 ...]
```

| 字段 | 说明 | 示例 |
|------|------|------|
| IP | 海康摄像头 IP 地址 | `10.60.87.200` |
| 用户名 | 设备登录名 | `admin` |
| 密码 | 设备密码 | `peng2176` |
| 通道 | 视频通道号 | `1` |
| 分辨率 | 宽*高，与摄像头实际分辨率一致 | `1920*1080` |
| `[多边形]` | 检测区域多边形点，格式 `x,y`，空格分隔 | `[100,200 300,200 300,400 100,400]` |

### 示例

**单路摄像头** (`cameras_config.txt`)：

```
# gatekeeper_V4 检测区域配置
# IP user password channel resolution [polygon_points]
10.60.87.200 admin peng2176 1 1920*1080 [120,300 450,280 500,550 100,580]
```

**多路摄像头**：

```
10.60.87.200 admin peng2176 1 1920*1080 [120,300 450,280 500,550 100,580]
10.60.87.201 admin peng2176 1 1920*1080 [200,250 600,260 620,500 180,520]
```

> 程序会自动闭合多边形（首尾点相连），并为每路摄像头生成掩膜预览图 `mask_<IP>_ch<channel>.png`，方便确认检测区域是否正确。

---

## 计数算法详解

### 决策级融合跟踪 (ByteTrack 增强)

在标准 ByteTrack (IoU-only) 基础上，引入三层融合机制：

#### ① 轨迹引导置信度动态校准 (`calibrateConfidence`)

```
稳定 track（activated + 最近更新 + stable_count ≥ 5）
     │
     ├─→ 对每个低分 detection 计算马氏距离
     ├─→ 距离 < 3.0 → 置信度加权提升
     └─→ 提升后的 detection 纳入高置信度池参与匹配
```

**效果**：当稳定跟踪目标因遮挡/光线导致 YOLOv8 置信度骤降时，依靠历史轨迹"拉回来"，避免目标丢失。

#### ② 联合评分匹配 (`greedyJointMatch`)

替代纯 IoU 匹配，使用加权联合评分：

```
S = 0.4 × IoU        (空间交叠)
  + 0.25 × Motion    (马氏距离归一化到 [0,1]，越小越好)
  + 0.2 × Appearance (HSV 直方图余弦相似度)
  + 0.15 × Confidence (检测置信度)
```

分两轮执行：第一轮活跃 track vs 高置信度检测；第二轮未匹配 track vs 低置信度检测。

#### ③ HSV 外观直方图 + EMA 更新

每次成功匹配后用指数移动平均更新 track 的外观特征，增强对同一目标的辨识能力。

---

### 状态机设计

每个通过 ByteTrack 分配的 `track_id` 维护独立的状态记忆：

```cpp
struct TM {
    int         s      = S_OUT;   // 状态: S_OUT/S_ENTER/S_IN/S_LEAVE
    int         n      = 0;       // 连续帧计数器
    cv::Point2f p;                // 上次在区域内的脚底坐标
    int         out_n  = 0;       // 离开区域外的连续帧数
};
```

**四状态转移**：

```
             连续 ≥2 帧在区域内
  S_OUT/S_ENTER ───────────────→ S_IN (区域人数 +1)
                                       │
                                偏离 >80px 且不在区域
                                       │
                                       ▼
                                  S_LEAVE (宽限期 8 帧)
                                  ╱              ╲
                   重入区域 ╱                     ╲ 超 8 帧
                           ▼                       ▼
                        S_IN                     S_OUT
                     (保持计数)                (确认离开)
```

### 状态说明

| 状态 | 含义 | 行为 |
|------|------|------|
| **S_OUT** | 目标在区域外，或首次出现 | 每帧检测 isInsideZone，连续 2 帧满足 → S_IN，**计数 +1** |
| **S_ENTER** | 进入过渡（与 S_OUT 同逻辑） | 同 S_OUT，非区域 → 回 S_OUT（枚举预留扩展） |
| **S_IN** | 目标在区域内，已纳入统计 | 脚底位移 >80px 且非区域 → S_LEAVE；否则更新坐标 |
| **S_LEAVE** | 疑似离开（宽限期 8 帧） | 8 帧内重入 → S_IN；超时 → S_OUT；快速退出 (连续4帧 IoU<0.01) 立即确认 |

### 设计要点

1. **进入防抖**：连续 2 帧确认，避免单帧误检触发
2. **离开宽限**：8 帧宽限期内可重入 (约 0.32s @25fps)，防止短暂遮挡导致误减
3. **位移判定**：偏离上次脚底位置 >80px 才触发 S_LEAVE（约人体肩宽一半），避免原地抖动
4. **快速退出**：连续 4 帧 bbox 与区域几乎无交集 (IoU<0.01) → 立即确认离开
5. **跨遮挡保持**：ByteTrack max_lost=60 (约 2.4s)，遮挡期间 track_id 不变，状态保持

### 区域判定 (isInsideZone)

采用**双重条件 OR** 机制，任一满足即判为在区域内：

| 策略 | 方法 | 适用场景 |
|------|------|----------|
| **策略 1（优先）** | 取 bbox 底部 y=92% 处的单点，判断是否落在 mask 内 | 脚底可见的正常情况 |
| **策略 2（备选）** | 取 bbox 底部 35% 区域与 mask 的交叠面积比 ≥ 0.2 | 遮挡导致脚底被遮住 |

> 检测区域通常画在地板上，人可能上半身在区域外但脚在地板上。策略 1 用脚底点判断最为精准，策略 2 作为遮挡时的兜底。

### 计数平滑

**众数滤波** (10 帧滑动窗口)：取最近 10 帧区域计数值的众数作为最终输出，消除 YOLOv8 检测结果波动带来的计数抖动。终端输出同时显示 "Mode" (众数滤波结果) 和 "Zone" (当前原始计数)，实际串口上报以 Mode 为准。

---

## Python 配置工具

`闸门计数-网络摄像头配置文件绘制.py` 提供了完整的配置管理 GUI 工具。

### 功能

| 功能 | 说明 |
|------|------|
| **UDP 探测** | 自动扫描局域网中的 RK3588 设备（无需知道 IP） |
| **SCP 远程部署** | 将配置文件一键推送到设备端 |
| **多边形绘制** | 在摄像头画面上鼠标拖动绘制检测区域 |
| **RTSP 预览** | 连接海康摄像头实时预览画面 |
| **配置管理** | 加载 / 保存 / 编辑 cameras_config.txt |
| **本地图片** | 支持加载本地图片作为底图绘制区域 |

### 前置依赖

```bash
pip install PyQt5 opencv-python paramiko scp
```

### 使用方法

```bash
python 闸门计数-网络摄像头配置文件绘制.py
```

GUI 操作流程：

1. 输入海康摄像头 RTSP 地址或加载本地图片
2. 点击"开始预览"获取实时画面
3. 用鼠标在画面上绘制检测区域（左键添加点，右键闭合）
4. 配置设备的 IP / 用户名 / 密码
5. 点击"远程更新"通过 SCP 推送到 RK3588 设备

> 跨子网设备需要手动输入 IP，UDP 探测仅限同一子网。

---

## 输出与通信

### 画面显示

| 信息 | 位置 | 说明 |
|------|------|------|
| Mode: N | 左上角第1行 (黄色) | 众数滤波后的稳定人数 |
| Zone: N | 左上角第2行 (黄色) | 当前帧原始区域人数 |
| Tot: N | 左上角第3行 (黄色) | 当前帧总跟踪目标数 |
| FPS: N | 左上角第4行 (绿色) | 当前处理帧率 |
| 检测框 + track_id | 画面上 | 每个 track 固定随机颜色 |
| 检测区域轮廓 | 画面上 | 红色多边形 |

### 终端输出

```
video_name FPS:27.5 Mode:8 Zone:9 Tot:12 Max:15
```

- `Mode`: 10 帧众数滤波结果（稳定值）
- `Zone`: 当前帧区域内的原始统计人数
- `Tot`: 当前帧总跟踪目标数
- `Max`: 整个运行周期中单帧最大检测人数

### SQLite 持久化

| 数据库 | 文件名 | 说明 |
|--------|--------|------|
| 本地存储 | `results_v4.db` | 每路摄像头的检测计数结果 |
| 发送记录 | `detection_results_send.db` | 串口已发送记录（含伞状态字段） |

表结构 `detection_results`：

| 字段 | 类型 | 说明 |
|------|------|------|
| `timestamp` | TEXT | 毫秒精度时间戳 (YYYYMMDDHHMMSSmmm) |
| `max_count` | INTEGER | 当前时段最大人数 |
| `has_umbrella` | INTEGER | 是否检测到伞 (0/1) |

### 串口通信

- 协议：RS-485 类 Modbus RTU (ASCII)
- 波特率：115200, 8N1
- 校验：LRC (纵向冗余校验)
- 功能码：实时查询 (0x03) / 历史记录查询 (0x04) / 伞状态上报
- 编译时通过 `-DENABLE_SERIAL` 启用

---

## 目录结构

```
gatekeeper_V4/
├── CMakeLists.txt                      # CMake 构建配置
├── README.md                           # 本文档
├── cameras_config.txt                  # 摄像头配置（正式）
├── cameras_config_10.txt               # 摄像头配置（备用）
├── 闸门计数-网络摄像头配置文件绘制.py    # Python 配置工具（GUI: 远程部署 + 区域绘制）
│
├── include/                            # 头文件
│   ├── types.h                         # 核心类型定义 (Detection, CameraConfigEntry, NNError)
│   ├── mask_utils.h                    # 掩膜工具：解析配置 / 创建掩膜 / 区域判定
│   ├── yolov8_custom.h                 # YOLOv8 推理封装
│   ├── yolov8_thread_pool.h            # YOLOv8 线程池 (NPU 多核并行)
│   ├── engine.h                        # 推理引擎抽象接口
│   ├── rknn_engine.h                   # RK3588 NPU 引擎实现
│   ├── preprocess.h                    # 预处理 (letterbox + BGR→RGB)
│   ├── postprocess.h                   # 后处理 (解码 + NMS)
│   ├── cv_draw.h                       # 通用检测框绘制
│   ├── comm.h                          # 串口通信协议 (Modbus-like)
│   ├── LinuxPlayM4.h                   # 海康播放库头文件
│   └── bytetrack/
│       └── tracker.h                   # ByteTrack 决策级融合跟踪器
│
├── src/                                # 源文件
│   ├── main_tracking.cpp               # 主程序：海康实时流 + GUI
│   ├── main_headless.cpp               # 无头模式：终端 + SQLite + 串口
│   ├── video_debug.cpp                 # 调试程序：离线视频 → 标注输出
│   ├── mask_utils.cpp                  # 区域掩膜生成与判定位
│   ├── tracker.cpp                     # ByteTrack 实现 (含决策级融合)
│   ├── yolov8_custom.cpp               # YOLOv8 推理流程
│   ├── yolov8_thread_pool.cpp          # 线程池实现
│   ├── cv_draw.cpp                     # 检测框绘制
│   ├── comm.cpp                        # 串口协议实现 (完全对齐 gate_people_counting_hiksdk)
│   ├── rknn_engine.cpp                 # RKNN 引擎封装
│   ├── preprocess.cpp                  # 预处理（letterbox + BGR→RGB）
│   └── postprocess.cpp                 # 后处理（解码 + NMS）
│
├── weights/                            # 模型文件（用户自行放入）
├── videos/                             # 测试视频
│   ├── 多人10人.mp4                    # 测试视频
│   ├── output_v4.avi                   # video_debug 输出结果
│   └── DR10.jpg                        # 测试图片
│
└── build/                              # 编译产物目录
```

---

## 关键参数速查

### YOLOv8 检测

| 参数 | 默认值 | 位置 | 说明 |
|------|--------|------|------|
| 输入尺寸 | 640×640 | preprocess.cpp | 模型输入分辨率 |
| 置信度阈值 | 0.25 | yolov8_custom.cpp | 检测结果过滤 |
| NMS 阈值 | 0.5 | postprocess.cpp | 非极大值抑制 |
| 对象阈值 | 0.35 | postprocess.cpp | 后处理筛选 |

### ByteTrack 决策级融合

| 参数 | 默认值 | 位置 | 说明 |
|------|--------|------|------|
| track_thresh | 0.25 | main_tracking.cpp / video_debug.cpp | 低分检测的跟踪阈值 |
| high_thresh | 0.5 | main_tracking.cpp / video_debug.cpp | 高分检测阈值（低于此→低置信池） |
| match_thresh | 0.5 | main_tracking.cpp / video_debug.cpp | 匹配分数阈值 |
| max_lost | 60 | main_tracking.cpp / video_debug.cpp | 跟踪丢失最大帧数 (约 2.4s @25fps) |
| min_hits | 3 | main_tracking.cpp / video_debug.cpp | 确认跟踪的最小连续命中数 |
| 联合评分 α (IoU) | 0.4 | tracker.cpp:calcJointScore | 联合评分 IoU 权重 |
| 联合评分 β (Motion) | 0.25 | tracker.cpp:calcJointScore | 联合评分马氏距离权重 |
| 联合评分 γ (Appearance) | 0.2 | tracker.cpp:calcJointScore | 联合评分外观直方图权重 |
| 联合评分 δ (Confidence) | 0.15 | tracker.cpp:calcJointScore | 联合评分置信度权重 |
| 置信度校准马氏距离阈值 | 3.0 | tracker.cpp:calibrateConfidence | 稳定 track 匹配低分 det 的马氏距离上限 |
| 稳定 track 最小匹配帧数 | 5 | tracker.cpp:calibrateConfidence | stable_count 阈值 |
| 校准置信度提升系数 | 0.7×avg_score + 0.3×current | tracker.cpp:calibrateConfidence | EMA 加权公式 |

### 区域计数状态机

| 参数 | 默认值 | 位置 | 说明 |
|------|--------|------|------|
| 进入防抖帧数 | 2 | main_tracking.cpp / video_debug.cpp | OUTSIDE → INSIDE 需连续帧数 |
| 位移触发阈值 | 80px | main_tracking.cpp / video_debug.cpp | INSIDE → LEAVING 最小位移 |
| 离开宽限帧数 | 8 | main_tracking.cpp / video_debug.cpp | LEAVING → OUTSIDE 超时帧数 |
| 快速退出连续帧数 | 4 | main_tracking.cpp / video_debug.cpp | 完全出区域后的快速确认 |
| 快速退出面积比 | <0.01 | main_tracking.cpp / video_debug.cpp | bbox 与区域几乎无交集 |
| 区域判定面积比 | 0.2 | mask_utils.cpp | isInsideZone 底部 35% 面积比阈值 |
| 脚底判定位置 | y=92% | main_tracking.cpp / video_debug.cpp | bbox 高度的 92% 处 |
| 众数滤波窗口 | 10 帧 | main_tracking.cpp / video_debug.cpp | 计数平滑窗口大小 |

### 硬件

| 参数 | 默认值 | 位置 | 说明 |
|------|--------|------|------|
| NPU 核心 | 3 核 (mask=7) | rknn_engine.cpp | RKNN_CORE_MASK 使用 0/1/2 核 |
| NPU 线程数 | 2/路 | 命令行参数 | 每路摄像头的推理线程数 |
