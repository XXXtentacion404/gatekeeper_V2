# gatekeeper_V4 — 双摄闸口人数统计（RK3588）

YOLOv8 + ByteTrack 负责把人跟成轨迹，`GateIdCounter` 负责把"脚底跨过本相机判定线"翻译成 `IN / OUT / RETRO`。

核心设计：**每台相机只对自己画的判定线负责**，事件由"脚底相对判定线的带符号距离发生侧翻转"这个**边沿**触发，
与 track_id 生命周期解耦 —— 同一人反复往返，每跨一次计一次。

## 目录

- [依赖](#依赖)
- [编译](#编译)
- [运行](#运行)
- [配置文件](#配置文件)
- [目录结构](#目录结构)
- [注意事项](#注意事项)

---

## 依赖

```bash
sudo apt update
sudo apt install -y build-essential cmake libopencv-dev libsqlite3-dev libx11-dev
```

| 依赖 | 位置 / 说明 |
|------|------|
| 海康设备网络 SDK | `hiklib/`（`HCNetSDK.h` + `libhcnetsdk.so` + `libPlayCtrl.so` + **`HCNetSDKCom/`**） |
| RKNN Runtime | `librknn_api/`（`include/rknn_api.h` + `aarch64/librknnrt.so`）；板端已装则自动用系统的 |
| RGA（可选） | `3rdparty/rga/include/` 提供头文件，`.so` 用板端的；没有则回退 OpenCV 预处理 |

---

## 编译

```bash
mkdir -p build && cd build
cmake ..
make -j4
```

| 产物 | 说明 |
|------|------|
| `main_tracking` | ★ 板端主程序（实时流 + 显示） |
| `main_headless` | 无界面版 |
| `video_debug` | 离线视频调试 |
| `lib*.so` | `rknn_engine` / `nn_process` / `yolov8_lib` / `bytetrack_lib` / `draw_lib` |

## 运行

```bash
./build/main_tracking weights/v8s.rknn cameras_config.txt 2
```

| 参数 | 含义 |
|------|------|
| 1 | 模型路径 |
| 2 | **配置文件路径（必须是第 2 个参数）** |
| 3 | 每路相机的 NPU 推理线程数，默认 `2`（夹到 1~4） |

可选参数：`--headless` / `-n`（不创建显示窗口）、`--trace`（打印诊断信息）。

```bash
./build/main_headless <model> <config> [threads]                    # 无界面版
./build/video_debug  <model> <config> <video.mp4> [threads] [-c 0]  # 离线视频调试
```

启动必须出现 `[INFO] ... 判定线=折线(zin/zout) N 点, 角色=入口/出口`；
若是 `判定线=**无**`，说明配置没读进来，此时不会产生任何计数。

---

## 配置文件

每行一台相机，空白分隔，除前 4 项外 token 顺序任意：

```
<IP> <用户名> <密码> <通道号> [宽*高] [x,y x,y ...] <zin,... 或 zout,...> [zo,<平移量>]
```

| token | 说明 |
|------|------|
| 前 4 项 | IP / 用户名 / 密码 / 通道号，位置固定 |
| `1920*1080` | 分辨率，**星号**分隔，可省略（默认 1920×1080） |
| `[x,y x,y ...]` | 检测区域多边形，`[` 开始 `]` 结束 |
| `zin,...` / `zout,...` | 判定线：`zin,` = 入口相机，`zout,` = 出口相机；`x,y` 成对、≥2 点、点数不限 |
| `zo,100` | 区域沿 x 平移量，默认 100（只影响区域显示，不影响计数判定） |

区域 = 判定线 ∪ 其沿 x 平移 |平移量| px 的副本（闭环仅用于显示）。

**计数规则**：脚底从线外到线内 = 进入类，线内到线外 = 离开类。
入口相机：进入 = `IN`、离开 = `RETRO`；出口相机：进入 = `RETRO`、离开 = `OUT`。
首次观测只确立所在侧、不发事件。

配置 GUI 工具：`闸门计数-网络摄像头配置文件绘制.py`（Python + OpenCV + PyQt5，可预览 RTSP、绘制区域、生成判定线、SCP 下发）。

---

## 目录结构

```
gatekeeper_V4/
├── CMakeLists.txt
├── README.md
├── cameras_config.txt                 摄像头配置
├── 闸门计数-网络摄像头配置文件绘制.py   配置 GUI 工具
│
├── include/
│   ├── gate_id_counter.h              ★ 跨判定线计数状态机
│   ├── tripwire_counter.h             旧版计数器（仅 video_debug 用）
│   ├── mask_utils.h                   配置解析 / 掩膜 / 区域判定
│   ├── types.h                        基础类型
│   ├── yolov8_custom.h                YOLOv8 推理封装
│   ├── yolov8_thread_pool.h           推理线程池（NPU 多核）
│   ├── engine.h / engine_helper.h     推理引擎抽象
│   ├── rknn_engine.h                  RKNN 引擎实现
│   ├── preprocess.h / postprocess.h   预处理 / 后处理
│   ├── cv_draw.h                      检测框绘制
│   ├── comm.h                         串口协议
│   ├── LinuxPlayM4.h                  海康播放库头文件
│   └── bytetrack/tracker.h            ByteTrack 跟踪器
│
├── src/
│   ├── main_tracking.cpp              ★ 板端主程序
│   ├── main_headless.cpp              无界面版
│   ├── video_debug.cpp                离线视频调试
│   ├── gate_id_counter.cpp            ★ 计数状态机实现
│   ├── tripwire_counter.cpp           旧版计数器实现
│   ├── tracker.cpp                    ByteTrack
│   ├── mask_utils.cpp                 掩膜 / 区域工具
│   ├── yolov8_custom.cpp              YOLOv8 推理流程
│   ├── yolov8_thread_pool.cpp         推理线程池
│   ├── rknn_engine.cpp                RKNN 封装
│   ├── preprocess.cpp / postprocess.cpp
│   ├── cv_draw.cpp                    绘制
│   └── comm.cpp                       串口协议实现
│
├── hiklib/                            海康 SDK
├── librknn_api/                       RKNN Runtime
├── 3rdparty/rga/include/              RGA 头文件
├── weights/v8s.rknn                   部署模型（另有 v8s_v2.rknn、训练产出目录）
├── build-check/hosttest/              宿主机单元测试（test_gate_counter 等）
├── 文档/                              方案与迭代记录
└── vidoes/                            测试视频
```

---

## 注意事项

- **类别数写死在两处**：`postprocess.cpp:20` 的 `class_num = 2` 与 `yolov8_custom.cpp:5` 的类别名表
  `{"people","umbrella"}`。换模型必须同步改这两处；只有 `people` 参与人数统计。
- **人跨门那几帧必须被真实检测到才会计数**：计数模块只吃真实检测帧，而画面上的框包含跟踪器外推的
  预测框。跨线瞬间只有预测框 → 不计数（检测问题，不是计数逻辑问题）。
- **判定线画短会漏计**：绕到判定线末端 50px 以外就不算跨度内 → 0 事件。治本是画长一些。
- **`Zone` / `Tot` / `IDsIn` 来自检测区掩膜，与判定线无关**，不能当"这台相机该计数"的依据。
- **单摄计数不要拿另一台相机配平**：区域可从侧边进出，按设计不产生事件，`201.IN ≠ 200.RETRO` 正常。
- **单人反复往返会反复计数**（每跨一次计一次），这是设计行为。
- **`video_debug` 用的是旧计数实现**（`tripwire_counter.cpp`，含"出生补偿"），
  拿它当基准会偏向多计"进入类"，与板端 `main_tracking` 不一致。
- **2026-09-18 已修复的三类漏计**（现场"某人没有 IN / 没有逆行"的根因）：
  ① **交接封存吞掉待确认证据** → 现在**封存前先把旧 ID 已观测到的跨线结算掉**
  （`reason` 带 `-on-seal`，属正常路径）；
  ② **两套保留窗口错配**（tracker `max_lost=180` vs 计数器 120）→ `kStateKeepFrames` 提到 **240**；
  ③ **被 tracker 重连复活的旧 ID 冒充新人**去继承并封存真实轨迹 → 交接资格收紧为
  **"只有计数器第一次见到的 ID 才能发起/接受交接"**（复活者会被打 `[CAND] revived-id` 注记）。
