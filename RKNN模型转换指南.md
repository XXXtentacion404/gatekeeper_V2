# RKNN 模型转换指南（YOLOv8 → RK3588 板端 NPU）

> 适用项目：`gatekeeper_V4`（目标硬件 **RK3588**，三核 NPU）
> 适用模型：本仓库 `weights/` 下两个文件夹里的 YOLOv8 模型
> 目标：把训练好的 `.pt` 权重转成板端能跑的 `.rknn`

---

## 0. 先搞清楚你手里有什么

`weights/` 下有两个文件夹，每个里面还有一个 `weights/` 子目录，里面才是真正的 `.pt`：

```
weights/
├── yolov8n/
│   └── weights/
│       ├── last.pt      # 最后一个 epoch 的权重
│       └── v8n_2.pt     # 训练出的模型（建议用这个）
└── yolov8s/
    └── weights/
        ├── last.pt      # 最后一个 epoch 的权重
        └── v8s_2.pt     # 训练出的模型（建议用这个）
```

| 文件 | 说明 | 推荐 |
|------|------|------|
| `yolov8n/weights/v8n_2.pt` | YOLOv8n 训练权重 | ✅ 优先转这个 |
| `yolov8s/weights/v8s_2.pt` | YOLOv8s 训练权重 | ✅ 优先转这个 |
| `yolov8n/weights/last.pt` | 末轮权重（不一定是最好） | 备选 |
| `yolov8s/weights/last.pt` | 末轮权重（不一定是最好） | 备选 |

> 建议先用 `v8n_2.pt` / `v8s_2.pt`。`last.pt` 是训练跑完最后一轮保存的，不一定比 `v8n_2.pt` 好；如果你的 `results.csv` 里某个 epoch 最好，就用对应那一份。

---

## 1. 整体流程（为什么分两步、为什么要 WSL）

```
  ┌─────────────────────┐         ┌──────────────────────────────┐
  │  Windows (本机)     │         │  WSL Ubuntu-22.04 (Linux)     │
  │                     │         │                              │
  │  .pt  ──①──▶  .onnx │ ──拷贝──▶│  .onnx ──②──▶  .rknn         │
  │  (ultralytics)      │         │  (rknn-toolkit2)            │
  └─────────────────────┘         └──────────────────────────────┘
                                          │
                                          ▼
                                  板端 RK3588 推理
```

**关键原因：**
- **第①步 PT→ONNX** 用 `ultralytics` 的 `YOLO.export()`，Windows 上直接能跑（你之前用的 `pt_to_onnx.py` 就是干这个的）。
- **第②步 ONNX→RKNN** 必须用 **RKNN Toolkit2**，而它**只支持 x86_64 Linux**。Windows 原生跑不了，所以要在 **WSL（Ubuntu-22.04）** 里跑。
- 转换本身**纯 CPU**，WSL2 完全够用，不需要 GPU。
- 转换完的 `.rknn` 再拷回 Windows / 直接丢进板端 `weights/` 就能被 `rknn_engine.cpp` 加载。

---

## 2. 第①步：PT → ONNX（在 Windows 上做）

### 2.1 前提：确认 ultralytics 已装
在本机（Windows）打开终端：
```bash
pip show ultralytics
```
没有就装：`pip install ultralytics`。你之前能跑 `pt_to_onnx.py`，这套应该已经好了。

### 2.2 改你的脚本配置
你之前用的脚本：`D:\Projects\yolo_training\v8\ultralytics_yolov8-main\my_python\pt_to_onnx.py`

只需要改文件头部的 4 个变量，**每个模型改一次、跑一次**：

```python
# 核心配置（按每个模型改这里）
PT_MODEL_PATH  = r"D:\Projects\Research_Project\gatekeeper_V2\code\gatekeeper_V4\weights\yolov8s\weights\v8s_2.pt"
TARGET_ONNX_NAME = "yolov8s.onnx"     # 你想要的名字
ONNX_SAVE_DIR   = r"D:\Projects\Research_Project\gatekeeper_V2\code\gatekeeper_V4\weights\yolov8s\weights"
INPUT_IMGSZ    = 640                  # 必须和训练时一致（args.yaml 里 imgsz=640）
```

四个模型对应的配置表：

| 模型 | PT_MODEL_PATH | TARGET_ONNX_NAME |
|------|---------------|------------------|
| YOLOv8s（推荐） | `...weights\yolov8s\weights\v8s_2.pt` | `yolov8s.onnx` |
| YOLOv8n（推荐） | `...weights\yolov8n\weights\v8n_2.pt` | `yolov8n.onnx` |
| YOLOv8s（备选） | `...weights\yolov8s\weights\last.pt`  | `yolov8s_last.onnx` |
| YOLOv8n（备选） | `...weights\yolov8n\weights\last.pt`  | `yolov8n_last.onnx` |

> `ONNX_SAVE_DIR` 就填对应 `.pt` 同目录即可，脚本会自动把默认生成的 `best.onnx` 重命名成你要的名字。

### 2.3 运行 + 验证
```bash
cd D:\Projects\yolo_training\v8\ultralytics_yolov8-main\my_python
python pt_to_onnx.py
```
成功会打印类似：
```
✅ ONNX模型导出+重命名成功！
最终保存路径：...yolov8s.onnx
模型大小：xx.xx MB
```
到 `ONNX_SAVE_DIR` 里确认 `.onnx` 文件确实生成了，记下它的**绝对路径**（第②步在 WSL 里要用，路径会变成 `/mnt/d/...`）。

**导出参数说明（脚本里已写死，不必改）：**
- `opset=17`：RKNN Toolkit2 支持，没问题。
- `simplify=True`：ONNX 化简，板端更友好，**务必保持 True**。
- `imgsz=640`：和你训练、`rknn_engine.cpp` 推理的输入尺寸一致。

---

## 3. 第②步（上）：准备 WSL + 装 RKNN Toolkit2

### 3.1 进入你的 WSL
你之前的 WSL 在 `\\wsl.localhost\Ubuntu-22.04\home\xxx`。进入方式任选其一：
- 开始菜单搜 **Ubuntu-22.04** 打开终端；或
- 在 Windows 终端里输入 `wsl -d Ubuntu-22.04`

### 3.2 建一个干净的 Python 虚拟环境（强烈建议）
RKNN Toolkit2 **只支持 Python 3.8~3.11**（不支持 3.12/3.13）。Ubuntu-22.04 默认 `python3` 是 3.10，刚好。

```bash
# 确认版本（应为 3.8~3.11）
python3 --version

# 建虚拟环境
cd ~
python3 -m venv rknn-env
source rknn-env/bin/activate
# 激活后命令行前面会出现 (rknn-env)
pip install --upgrade pip
```

### 3.3 安装 RKNN Toolkit2
这个包**不在 PyPI**，要去 Rockchip 官方 release 下载对应的 wheel：
- 仓库：https://github.com/airockchip/rknn-toolkit2/releases
- 找 **2.x** 版本（rk3588 必须 2.0 以上），下载 **x86_64 Linux 的 cp310 wheel**，文件名类似：
  `rknn_toolkit2-2.3.0-cp310-cp310-manylinux_2_17_x86_64.whl`
  （注意是 `rknn_toolkit2`，**不是** `rknn_toolkit2_lite2`——lite 是给板端 ARM 用的，转换机要用完整版）

下载好后（假设放在 `~/downloads/`）：
```bash
pip install ~/downloads/rknn_toolkit2-2.3.0-cp310-cp310-manylinux_2_17_x86_64.whl
```

> 如果装完 import 报 `numpy` 相关错（如 `np.int` 不存在），是 numpy 版本太新，锁一下：
> `pip install "numpy==1.23.5"` 然后重开 Python 验证。

### 3.4 验证安装
```bash
python3 -c "from rknn.api import RKNN; print('rknn ok')"
```
打印 `rknn ok` 就说明装好了。顺便也装上 onnx 工具（可选，用来肉眼检查 onnx）：
```bash
pip install onnx onnxruntime
```

---

## 4. 第②步（下）：ONNX → RKNN（在 WSL 里跑）

### 4.1 WSL 怎么访问你的 Windows 文件
WSL 里 Windows 的 `D:\` 挂载在 `/mnt/d/`。所以：
- Windows 路径 `D:\Projects\Research_Project\gatekeeper_V2\code\gatekeeper_V4\weights\yolov8s\weights\yolov8s.onnx`
- 在 WSL 里变成 `/mnt/d/Projects/Research_Project/gatekeeper_V2/code/gatekeeper_V4/weights/yolov8s/weights/yolov8s.onnx`

### 4.2 转换脚本
把下面脚本存成 `onnx_to_rknn.py`（本指南同目录已附一份 `weights/onnx_to_rknn.py` 可直接用），**路径改成你自己的**：

```python
# onnx_to_rknx.py  —— 在 WSL Ubuntu-22.04 里运行
from rknn.api import RKNN

# ===== 按模型改这里 =====
ONNX_MODEL = '/mnt/d/Projects/Research_Project/gatekeeper_V2/code/gatekeeper_V4/weights/yolov8s/weights/yolov8s.onnx'
RKNN_MODEL = '/mnt/d/Projects/Research_Project/gatekeeper_V2/code/gatekeeper_V4/weights/yolov8s/weights/yolov8s.rknn'
DATASET    = './dataset.txt'   # INT8 量化用的校准图列表；float 模式可留空
DO_QUANT   = True              # True=INT8量化(精度高,板端快)；False=float(不量化,仅验证用)

if __name__ == '__main__':
    rknn = RKNN(verbose=True)

    # mean/std 用 [[0,0,0]]/[[255,255,255]] 表示“输入除以255归一化”，YOLOv8 标准做法
    rknn.config(
        target_platform='rk3588',
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
    )

    ret = rknn.load_onnx(model=ONNX_MODEL)
    assert ret == 0, 'load_onnx failed'

    if DO_QUANT:
        ret = rknn.build(do_quantization=True, dataset=DATASET)
    else:
        ret = rknn.build(do_quantization=False)
    assert ret == 0, 'build failed'

    ret = rknn.export_rknn(RKNN_MODEL)
    assert ret == 0, 'export_rknn failed'

    rknn.release()
    print('✅ RKNN 导出成功:', RKNN_MODEL)
```

### 4.3 要不要 INT8 量化？
- **INT8（`DO_QUANT=True`，默认）**：板端推理又快又省，是你项目（要求精度≥98%）的正式部署方式。
  **但需要一个 `dataset.txt` 校准集**——里面是 20~100 张代表性图片（闸机/通道场景的人）的绝对路径，每行一张。
  生成方法（在 WSL 里，假设校准图放 `~/calib/`）：
  ```bash
  find /mnt/d/你的校准图目录 -name '*.jpg' > dataset.txt
  head dataset.txt   # 确认每行是个绝对路径，如 /mnt/d/.../xxx.jpg
  ```
  > 没有现成图？可以从 `gatekeeper_V4/vidoes/` 的视频里抽帧，或拿训练集 val 图片用。图越多校准越准，20~50 张起步即可。
- **Float（`DO_QUANT=False`）**：不量化，转换秒过，用来**先验证整条链路通不通**、看推理能不能出框。验证完再走 INT8 正式版。

### 4.4 运行转换
```bash
# 激活环境
source ~/rknn-env/bin/activate
cd ~/你的脚本目录

# float 先试一次（快，验证链路）
python3 onnx_to_rknn.py

# 确认能出 .rknn 后，准备好 dataset.txt，改 DO_QUANT=True 再跑正式 INT8 版
```
跑完会在 `RKNN_MODEL` 指定的位置生成 `.rknn` 文件。

> ⚠️ 如果 `rknn.build()` 报 ONNX 算子不支持 / shape 不对：确认第①步 `simplify=True`、`opset=17` 已生效；必要时在 `pt_to_onnx.py` 里把 `opset` 降到 13 再试。

---

## 5. 部署 & 运行

### 5.1 把 .rknn 放到板端能读到的地方
转换时我已经把 `.rknn` 直接生成在 `weights/yolov8s/weights/` 下了（和 `.onnx` 同目录）。如果生成在别处，拷过去即可。最终板端加载的就是这个 `.rknn`。

建议你命名成和现有调用一致的（参考 README）：
- 入口用：`weights/yolov8n.rknn` 或 `weights/yolov8s.rknn`

### 5.2 运行（示例，板端或本地调试均可）
按你 `memory` 里的习惯命令：
```bash
# 本地调试（video_debug）
./video_debug ../weights/v8s.rknn ../cameras_config.txt ../vidoes/200.mp4 2 -n

# 板端正式（main_tracking）
./main_tracking ../weights/v8s.rknn ../cameras_config.txt 2
```
把 `v8s.rknn` 换成你实际生成的文件名即可。

### ⚠️ 5.3 必须检查的兼容性：类别数 `class_num`
你的 C++ 后处理 `src/postprocess.cpp` 第 20 行写死了：
```cpp
static int class_num = 2;
```
意思是板端后处理**假设模型输出 2 个类别**。所以：
- **转换前先确认你的 `.pt` 训的是几个类（nc）**。在 Windows 上跑：
  ```python
  from ultralytics import YOLO
  m = YOLO(r'...weights\yolov8s\weights\v8s_2.pt')
  print('nc =', m.model.model[-1].nc)   # 打印类别数
  ```
- 如果 `nc == 2`：✅ 直接和现有 `postprocess.cpp` 兼容，不用改 C++。
- 如果 `nc != 2`：要么把 `postprocess.cpp` 里的 `class_num` 改成你的真实类别数并重新编译，要么换一个 nc=2 的模型来转。**否则板端会数错/漏检。**

---

## 6. 常见问题 / 排错

| 现象 | 原因 | 解决 |
|------|------|------|
| `import rknn` 失败 / 装不上 | Python 版本不对（3.12/3.13） | 用 3.8~3.11 的 venv（Ubuntu22.04 默认 3.10） |
| `numpy` 报 `np.int` 不存在 | numpy≥1.24 删了旧 API | `pip install "numpy==1.23.5"` |
| 找不到 `rknn_toolkit2` wheel | 它不在 PyPI | 去 GitHub releases 下载 cp310 x86_64 wheel 手动 `pip install` |
| `build()` 说算子不支持 | ONNX 太复杂 / opset 太高 | 确认 `simplify=True`；`opset` 降到 13 重导 ONNX |
| INT8 校准慢 / 精度怪 | `dataset.txt` 图太少或不具代表性 | 加到 50~100 张真实场景图 |
| 板端出框但计数错 | `class_num` 和模型 nc 不符 | 改 `postprocess.cpp` 的 `class_num` 或换 nc=2 模型 |
| WSL 里找不到 Windows 文件 | 路径写错 | Windows `D:\x` → WSL `/mnt/d/x`，注意斜杠 |
| 转换报 AVX/非法指令 | wheel 架构不匹配 | 下 `manylinux_2_17_x86_64` 版本，别下 aarch64 |

---

## 7. 一键检查清单

- [ ] Windows 上用 `pt_to_onnx.py` 把 4 个（或选定的）`.pt` 导成 `.onnx`（imgsz=640, simplify=True, opset=17）
- [ ] 用 `m.model.model[-1].nc` 确认模型类别数 == 2（否则改 `postprocess.cpp`）
- [ ] WSL 里 `python3 --version` 是 3.8~3.11，并建好 venv
- [ ] 装好 `rknn_toolkit2` 2.x（cp310 x86_64 wheel），`from rknn.api import RKNN` 能过
- [ ] WSL 里 `.onnx` 路径写成 `/mnt/d/...`
- [ ] 准备 `dataset.txt`（INT8 用）；先 `DO_QUANT=False` 跑通，再 `DO_QUANT=True` 出正式版
- [ ] 生成的 `.rknn` 放到 `weights/`，用 `video_debug` 验证能出正确框和计数
```
