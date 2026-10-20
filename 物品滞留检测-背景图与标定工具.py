# -*- coding: utf-8 -*-
"""
gatekeeper_V4 — 滞留检测「取背景图 + A4 标定」PC 端工具（GUI）

和「闸门计数-网络摄像头配置文件绘制.py」是同一套用法（PyQt5 + RTSP + 可上传板端），
区别只有两点：
  1) 它不做绘制 —— **区域直接从 cameras_config.txt 导入**，就是计数用的那个检测区；
  2) 它干的是滞留检测的前置两件事：抓「标准空场景背景图」、用 A4 纸标定 px_per_cm。

为什么区域必须一样（本工具最关键的约束）
------------------------------------------------------------------
板端 main_tracking.cpp 把 `cfg.detection_mask`（= cameras_config.txt 里那串 `[...]`
检测区多边形生成的掩膜）同时喂给两处：
  · 计数：Zone/Tot/IDsIn 的人数统计、GateIdCounter 的区域上下文
  · 滞留检测：ItemAbandonmentDetector 的 region_mask → ROI 裁剪 + 候选块归属判定
所以「背景图 / 标定」用的区域必须**逐像素等于**该多边形，否则板上算的 ROI 和你在电脑上
量的不是同一块地方，px_per_cm 直接失效。

为了保证"逐像素等于"，本文件**不重新实现几何**，而是直接 import 现有的绘图脚本，
复用它里面与 C++ 端逐行对应的那套函数：
    _zone_ring() / _resolve_depth_sign() / _poly_mid() / _right_hand_normal()
    ENTRY_MARKER_DEPTH
以及它的 PolygonDrawer（区域 + 判定线的渲染与我们平时看到的完全一致）。
改区域约定时只改绘图脚本一处，两个工具同时跟着变 —— 这是刻意的。

用法
------------------------------------------------------------------
    D:/Software/Anaconda3/envs/yolov8/python.exe 物品滞留检测-背景图与标定工具.py
    ... 物品滞留检测-背景图与标定工具.py --config cameras_config.txt

典型流程
------------------------------------------------------------------
    1) 打开配置 → 选相机 → 连接摄像头
    2) 等人走空、光照稳定 → 抓取背景图（N 帧逐像素取中值，自动抹掉零星人影）
       → 存成 abandon_bg_<IP末段>.jpg，并自动载入画布
    3) 拿着 A4 纸（29.7cm）到检测区里横放 → 点「A4 标定」→ 从纸左边缘拖到右边缘
       近端 / 中段 / 远端各量一次（斜视角下像素密度差别很大，平均值没意义）
    4) 看"标定汇总"里的建议值 → 勾选要写的项 → 写入 abandon_config.txt
    5) 上传背景图 + abandon_config.txt 到板端，重启板端程序

依赖
    pip install PyQt5 opencv-python numpy        （上传功能另需 paramiko scp）
"""
import importlib.util
import io
import os
import re
import sys
import time

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("需要 opencv-python：请用 D:/Software/Anaconda3/envs/yolov8/python.exe 运行")

try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                                 QPushButton, QListWidget, QLabel, QFileDialog, QMessageBox,
                                 QLineEdit, QGroupBox, QTextEdit, QSpinBox, QDoubleSpinBox,
                                 QCheckBox)
    from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QObject
    from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
except ImportError:
    sys.exit("需要 PyQt5：pip install PyQt5")

ROOT = os.path.dirname(os.path.abspath(__file__))
ZONE_TOOL = os.path.join(ROOT, "闸门计数-网络摄像头配置文件绘制.py")
ABANDON_CFG = os.path.join(ROOT, "abandon_config.txt")
A4_LONG_CM = 29.7
A4_SHORT_CM = 21.0
DEFAULT_BOARD_IP = "192.168.1.104"


# ============================================================================
# 复用现有绘图脚本（区域几何的唯一真相来源）
# ============================================================================
def _load_zone_tool():
    """把「闸门计数-网络摄像头配置文件绘制.py」当模块加载。

    ⚠️ 文件名带中文和连字符，没法用普通 import，必须走 importlib 按路径加载。
    为什么非要 import 它：区域形状（判定线 ∪ 沿 x 平移副本）、平移量符号校正这些
    约定必须与计数端**同源**。复制一份代码到这里迟早会分叉，那时"区域和检测区一样"
    就成了一句空话。
    """
    if not os.path.exists(ZONE_TOOL):
        sys.exit("找不到 {}（区域几何依赖它，请确认文件还在项目根目录）".format(ZONE_TOOL))
    spec = importlib.util.spec_from_file_location("_gate_zone_tool", ZONE_TOOL)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:                      # noqa: BLE001
        sys.exit("加载 {} 失败：{}\n（它需要 PyQt5 / opencv-python / numpy 可用）".format(
            os.path.basename(ZONE_TOOL), e))
    # 安全阀：几何函数必须都在，否则后面会以 AttributeError 崩在半路
    for fn in ("_zone_ring", "_resolve_depth_sign", "_poly_mid", "_right_hand_normal",
               "PolygonDrawer", "ENTRY_MARKER_DEPTH"):
        if not hasattr(mod, fn):
            sys.exit("{} 里缺少 {} —— 两个工具的区域几何约定可能已经分叉，请先核对".format(
                os.path.basename(ZONE_TOOL), fn))
    return mod


ZONE = _load_zone_tool()


# ============================================================================
# 配置解析（与绘图脚本 _load 的口径一致）
# ============================================================================
def parse_cameras(path):
    """解析 cameras_config.txt：
        ip user pwd channel WxH [多边形…] <zin,/zout,/t1,判定线> [zo,平移量]

    zone_line/zone_role/zone_depth 的解析与「绘制」脚本**逐行同款**，
    zone_depth 的符号也用它同一个 _resolve_depth_sign 校正到"朝检测区那侧"。
    """
    cams = []
    with io.open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            t = ln.strip()
            if not t or t.startswith("#"):
                continue
            ps = t.split()
            if len(ps) < 4:
                continue
            ip, user, pwd, ch = ps[0], ps[1], ps[2], ps[3]
            res = "1920*1080"
            rest = ps[4:] if len(ps) > 4 else []
            if rest and "*" in rest[0]:
                res = rest[0]
                rest = rest[1:]

            # ---- 检测区多边形（区域几何的唯一来源）----
            polys = []
            for m in re.finditer(r"\[([^\]]*)\]", t):
                pts = []
                for xy in m.group(1).split():
                    if "," in xy:
                        try:
                            x, y = xy.split(",")[:2]
                            pts.append((int(float(x)), int(float(y))))
                        except ValueError:
                            pass
                if len(pts) >= 3:
                    polys.append(pts)

            # ---- 判定线（zin,/zout, 新格式；t1, 旧格式）+ 可选 zo, ----
            zone_line = None
            zone_role = None
            zone_depth_raw = None
            for tok in rest:
                if tok.startswith("zin,") or tok.startswith("zout,"):
                    role = "entry" if tok.startswith("zin,") else "exit"
                    # ⚠️ zout, 是 5 个字符，[4:] 会留下一个前导逗号；
                    #    靠 str.split() 自动吞掉空白才对齐 —— 与绘制脚本同款写法，别改。
                    parts = tok[4:].replace(",", " ").split()
                    pts = []
                    for k in range(0, len(parts) - 1, 2):
                        try:
                            pts.append((int(float(parts[k])), int(float(parts[k + 1]))))
                        except ValueError:
                            pass
                    if len(pts) >= 2:
                        zone_line, zone_role = pts, role
                elif tok.startswith("t1,"):
                    parts = tok[3:].replace(",", " ").split()
                    if len(parts) < 4:
                        continue
                    zone_line = [(int(float(parts[0])), int(float(parts[1]))),
                                 (int(float(parts[2])), int(float(parts[3])))]
                    zone_role = None
                    if len(parts) >= 5:
                        zone_depth_raw = int(float(parts[4]))
                elif tok.startswith("zo,"):
                    try:
                        zone_depth_raw = int(float(tok[3:]))
                    except ValueError:
                        pass

            zone_depth = 0
            if zone_line:
                base = zone_depth_raw if zone_depth_raw is not None else ZONE.ENTRY_MARKER_DEPTH
                zone_depth = ZONE._resolve_depth_sign(polys, zone_line, base)

            # ---- 角色：与 main_tracking.cpp 完全同款 ----
            # 显式写了 zin,/zout, 就用它；否则退回 IP 末段（200=出口，其余=入口）。
            if zone_role == "entry":
                role = 0
            elif zone_role == "exit":
                role = 1
            else:
                role = 1 if ip.split(".")[-1] == "200" else 0

            cams.append({
                "ip": ip, "user": user, "pwd": pwd, "channel": ch, "res": res,
                "polygons": polys, "zone_line": zone_line, "zone_role": zone_role,
                "zone_depth": zone_depth, "zone_depth_raw": zone_depth_raw,
                "role": role, "tag": ip.split(".")[-1],
            })
    return cams


def rtsp_url(cam):
    """海康 RTSP：主码流 = Streaming/Channels/<channel>01（channel=1 → 101）。"""
    ch = str(cam["channel"]).strip()
    if ch.isdigit() and len(ch) == 1:
        ch = ch + "01"
    elif not (ch.isdigit() and len(ch) == 3):
        ch = "101"
    from urllib.parse import quote
    return "rtsp://{}:{}@{}:554/Streaming/Channels/{}".format(
        quote(cam["user"], safe=""), quote(cam["pwd"], safe=""), cam["ip"], ch)


def read_image(path):
    """中文路径安全的读图（cv2.imread 在中文路径下会静默失败）。"""
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:                            # noqa: BLE001
        return None


def save_jpg(img, path, quality=95):
    """中文路径安全的写图。"""
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return False
    try:
        buf.tofile(path)
        return True
    except Exception:                            # noqa: BLE001
        return False


def roi_rect_of(cam, margin_px):
    """检测区 ROI 裁剪框（全分辨率）= 多边形 bbox 外扩 margin，再裁到画面内。

    与 C++ ItemAbandonmentDetector::prepareRoi() 同款 —— 它就是对 detection_mask
    取 boundingRect 后 ±roi_margin_px。板端启动日志那行
    「检测区 ROI 裁剪: 1129x852 @(582,228)」就是这个框，本工具显示同一组数字，
    方便你一眼核对"电脑上量的"和"板上裁的"是同一块地方。
    """
    polys = cam.get("polygons") or []
    if not polys:
        return None
    try:
        W, H = [int(v) for v in cam["res"].split("*")]
    except Exception:                            # noqa: BLE001
        return None
    xs = [p[0] for poly in polys for p in poly]
    ys = [p[1] for poly in polys for p in poly]
    if not xs or not ys:
        return None
    x0 = max(0, min(xs) - margin_px)
    y0 = max(0, min(ys) - margin_px)
    # ⚠️ 右/下边界要 +1：C++ 的 cv::boundingRect 宽度是 (max-min+1)，
    #    少写这个 1，本工具会比板端启动日志少 1px（1128 vs 1129），对着看会以为配错了。
    x1 = min(W, max(xs) + margin_px + 1)
    y1 = min(H, max(ys) + margin_px + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


# ============================================================================
# abandon_config.txt 读写（只动指定 key，注释一律保留）
# ============================================================================
def read_config_key(path, key):
    try:
        with io.open(path, encoding="utf-8", errors="replace") as f:
            for ln in f:
                s = ln.strip()
                if s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                if k.strip() == key:
                    return v.strip()
    except Exception:                            # noqa: BLE001
        pass
    return None


def write_config_keys(path, pairs):
    """按 key 覆盖写入；key 不存在则追加到文件末尾。返回实际改动的 key 列表。

    只匹配"行首（可有空白）+ key + =" 的形式 → `# min_item_px=45` 这种注释行不会被误改，
    `role1.ref_bg_path=` 也不会被 `ref_bg_path=` 命中（正则锚定在行首）。
    """
    with io.open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    done = []
    for key, val in pairs:
        pat = re.compile(r"^(?P<pre>\s*)" + re.escape(key) + r"\s*=")
        hit = False
        for i, ln in enumerate(lines):
            m = pat.match(ln)
            if m:
                lines[i] = "{}{}={}".format(m.group("pre"), key, val)
                done.append(key)
                hit = True
                break
        if not hit:
            lines.append("{}={}".format(key, val))
            done.append(key + "(新增)")
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    return done


# ============================================================================
# 线程间通信小工具
# ============================================================================
class LogRelay(QObject):
    """把工作线程里的日志搬到 GUI 线程。

    ⚠️ 子线程绝对不能直接碰控件。现有绘图脚本里的 scp_send/probe_network 都只要求
    传进来的对象有 .emit(str) —— 传一个**绑定的 pyqtSignal**（relay.msg）正好满足，
    而且是 Qt 的排队连接，天然线程安全。
    """
    msg = pyqtSignal(str)


def _mkdir(path):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        try:
            os.makedirs(d)
        except Exception:                        # noqa: BLE001
            pass


# ============================================================================
# 抓背景图 worker（复用同一条 RTSP 连接，避免相机并发拉流受限）
# ============================================================================
class GrabWorker(QThread):
    progress = pyqtSignal(int, int)
    done = pyqtSignal(object, str, float)     # (中值图 or None, 错误, 区域内最大帧间差)

    def __init__(self, cap, frames, zone_bbox=None, size=None, warmup=8):
        super().__init__()
        self.cap = cap
        self.n = max(1, int(frames))
        self.zone = zone_bbox
        self.size = size                       # (w,h)：对齐到配置分辨率
        self.warmup = warmup

    def _align(self, fr):
        """对齐到配置分辨率 —— 区域坐标是按配置分辨率存的，不对齐 ROI 判据就会错位。"""
        if self.size and (fr.shape[1], fr.shape[0]) != self.size:
            return cv2.resize(fr, self.size, interpolation=cv2.INTER_AREA)
        return fr

    def run(self):
        frames = []
        try:
            for _ in range(self.warmup):         # 丢掉开头几帧（码流刚起来可能花屏）
                self.cap.read()
            for i in range(self.n):
                ok, fr = self.cap.read()
                if ok and fr is not None:
                    frames.append(self._align(fr))
                self.progress.emit(i + 1, self.n)
        except Exception as e:                    # noqa: BLE001
            self.done.emit(None, "读流异常: {}".format(e), 0.0)
            return
        if not frames:
            self.done.emit(None, "连上了但一帧都没读到（码流格式/取流权限问题）", 0.0)
            return

        med = frames[0] if len(frames) == 1 else \
            np.median(np.stack(frames, 0), 0).astype(np.uint8)

        # ---- 背景图验收：抓取期间检测区内有没有东西在动 ----
        # 中值只能抹掉"偶发"的人影；如果有人一直在区里站着，中值图会留下他。
        # 这里用"各帧 vs 中值"在检测区内的最大平均差当指标：差得多 = 区里有人/在动。
        motion = 0.0
        try:
            x0, y0, x1, y1 = self.zone if self.zone else (0, 0, med.shape[1], med.shape[0])
            sub = med[y0:y1, x0:x1]
            if sub.size:
                g = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY).astype(np.float32)
                for fr in frames:
                    gf = cv2.cvtColor(fr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
                    motion = max(motion, float(np.abs(gf - g).mean()))
        except Exception:                         # noqa: BLE001
            motion = 0.0
        self.done.emit(med, "", motion)


# ============================================================================
# 画布：继承现有 PolygonDrawer，只加"A4 标尺 + ROI 框"
# ============================================================================
class CalibDrawer(ZONE.PolygonDrawer):
    """复用父类的区域/判定线渲染（与计数脚本画出来的一模一样），
    额外支持在图上拖拽量 A4 纸，并叠加板端会用的 ROI 裁剪框。

    ⚠️ 本类**不进入**父类的绘制模式：mode 恒为 MODE_MEASURE，
    区域一律来自 cameras_config.txt，不在电脑上重画（那就会出现第二套几何）。
    """
    MODE_MEASURE = 9

    def __init__(self):
        super().__init__()
        self.mode = self.MODE_MEASURE
        self.measures = []          # [(x1,y1,x2,y2)] 全分辨率坐标
        self.m1 = None
        self.m2 = None
        self.dragging = False
        self.roi_rect = None        # (x0,y0,x1,y1) 全分辨率
        self.show_roi = True
        self.a4_cm = A4_LONG_CM
        self.on_change = None       # 测量结果变化时的回调（主窗口用来刷新汇总）

    # ---- 鼠标：只做 A4 测量 ----
    def mousePressEvent(self, e):
        if self.mode != self.MODE_MEASURE or e.button() != Qt.LeftButton:
            return
        if self.frame is None or self.dw <= 0 or self.dh <= 0:
            return
        self.m1 = self._to(self._clamp(e.pos()))
        self.m2 = self.m1
        self.dragging = True
        self.update()

    def mouseMoveEvent(self, e):
        if self.mode != self.MODE_MEASURE or not self.dragging:
            return
        self.m2 = self._to(self._clamp(e.pos()))
        self.update()

    def mouseReleaseEvent(self, e):
        if self.mode != self.MODE_MEASURE or not self.dragging:
            return
        self.dragging = False
        self.m2 = self._to(self._clamp(e.pos()))
        if self.m1 and self.m2 and abs(self.m2[0] - self.m1[0]) >= 3:
            self.measures.append((self.m1[0], self.m1[1], self.m2[0], self.m2[1]))
            if self.on_change:
                self.on_change()
        self.m1 = self.m2 = None
        self.update()

    # ---- 渲染：先让父类画区域/判定线，再叠加标尺与 ROI ----
    def paintEvent(self, e):
        super().paintEvent(e)
        if self.frame is None or self.dw <= 0 or self.dh <= 0:
            return
        try:
            p = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing)
            p.setFont(QFont("Microsoft YaHei", 9))

            if self.show_roi and self.roi_rect:
                x0, y0, x1, y1 = self.roi_rect
                a, b = self._td(x0, y0), self._td(x1, y1)
                pen = QPen(QColor(0, 200, 220), 2, Qt.DashLine)
                p.setPen(pen)
                p.setBrush(Qt.NoBrush)
                p.drawRect(a.x(), a.y(), b.x() - a.x(), b.y() - a.y())
                p.drawText(a.x() + 4, a.y() + 16,
                           "ROI {}x{}".format(x1 - x0, y1 - y0))

            for i, (x1, y1, x2, y2) in enumerate(self.measures, 1):
                a, b = self._td(x1, y1), self._td(x2, y2)
                p.setPen(QPen(QColor(20, 200, 220), 2))
                p.drawLine(a, b)
                p.setBrush(QColor(255, 255, 255))
                p.setPen(QPen(QColor(0, 150, 170), 2))
                p.drawEllipse(a, 4, 4)
                p.drawEllipse(b, 4, 4)
                px = abs(x2 - x1)
                p.setPen(QPen(QColor(0, 110, 130), 1))
                p.drawText(min(a.x(), b.x()), max(16, min(a.y(), b.y()) - 8),
                           "#{} {}px  {:.2f}px/cm".format(
                               i, px, px / self.a4_cm if self.a4_cm else 0))
            if self.dragging and self.m1 and self.m2:
                a, b = self._td(*self.m1), self._td(*self.m2)
                p.setPen(QPen(QColor(220, 60, 60), 2))
                p.drawLine(a, b)
                px = abs(self.m2[0] - self.m1[0])
                p.setPen(QPen(QColor(200, 40, 40), 1))
                p.drawText(min(a.x(), b.x()), max(16, min(a.y(), b.y()) - 8),
                           "{:d}px  {:.2f}px/cm".format(px, px / self.a4_cm if self.a4_cm else 0))
            p.end()
        except Exception:                         # noqa: BLE001
            # 叠加层画不出来不该让整个界面挂掉：区域/判定线在前一步已经画好了
            pass

    def set_measurements(self, ms):
        self.measures = list(ms)
        self.update()


# ============================================================================
# 主窗口
# ============================================================================
class MainWindow(QMainWindow):
    def __init__(self, cfg_path):
        super().__init__()
        self.setWindowTitle("gatekeeper_V4 — 滞留检测：取背景图 + A4 标定（区域取自 cameras_config.txt）")
        self.setGeometry(70, 60, 1440, 880)

        self.cam_cfg_path = cfg_path if os.path.exists(cfg_path) else None
        self.cams = []
        self.cam_i = -1
        self.cap = None
        self.cam_thread = None
        self.grab_worker = None
        self._dead_threads = []
        self.frozen = False
        self.bg_image = None
        self.relay = LogRelay()
        self.relay.msg.connect(self._log)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

        self._ui()
        if self.cam_cfg_path:
            self._load_config(self.cam_cfg_path)
        else:
            self._log("未找到 cameras_config.txt —— 请点「打开配置」手动选一个")

    # ---------------- 日志 ----------------
    def _log(self, s):
        self.log_text.append(str(s))
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    # ---------------- 界面 ----------------
    def _ui(self):
        w = QWidget()
        h = QHBoxLayout()
        ctl = QWidget()
        ctl.setFixedWidth(392)
        cl = QVBoxLayout()

        # ---- 配置文件 ----
        g = QGroupBox("1. 相机配置（区域来源）")
        gl = QVBoxLayout()
        b = QPushButton("打开 cameras_config.txt")
        b.clicked.connect(self._pick_config)
        gl.addWidget(b)
        self.cfg_lbl = QLabel("（未加载）")
        self.cfg_lbl.setWordWrap(True)
        self.cfg_lbl.setStyleSheet("color:#666;")
        gl.addWidget(self.cfg_lbl)
        b = QPushButton("重新解析配置（改完文件点这个）")
        b.clicked.connect(lambda: self._load_config(self.cam_cfg_path) if self.cam_cfg_path else None)
        gl.addWidget(b)
        self.cam_lst = QListWidget()
        self.cam_lst.setMaximumHeight(76)
        self.cam_lst.itemClicked.connect(self._sel_cam)
        gl.addWidget(self.cam_lst)
        self.geo_lbl = QLabel("—")
        self.geo_lbl.setWordWrap(True)
        self.geo_lbl.setStyleSheet("font-family:Consolas,monospace;color:#333;")
        gl.addWidget(self.geo_lbl)
        g.setLayout(gl)
        cl.addWidget(g)

        # ---- 连接 / 抓背景图 ----
        g = QGroupBox("2. 连接 + 抓标准背景图")
        gl = QVBoxLayout()
        r = QHBoxLayout()
        b = QPushButton("连接摄像头"); b.clicked.connect(self._connect); r.addWidget(b)
        b = QPushButton("断开"); b.clicked.connect(self._disconnect); r.addWidget(b)
        b = QPushButton("冻结/继续"); b.clicked.connect(self._toggle_freeze); r.addWidget(b)
        gl.addLayout(r)
        r = QHBoxLayout()
        b = QPushButton("载入本地图片"); b.clicked.connect(self._load_image); r.addWidget(b)
        b = QPushButton("载入已有背景图"); b.clicked.connect(self._load_existing_bg); r.addWidget(b)
        gl.addLayout(r)
        r = QHBoxLayout()
        r.addWidget(QLabel("取中值帧数:"))
        self.frames_sb = QSpinBox(); self.frames_sb.setRange(1, 120); self.frames_sb.setValue(15)
        r.addWidget(self.frames_sb)
        self.save_as_cb = QCheckBox("另存为…")
        r.addWidget(self.save_as_cb)
        gl.addLayout(r)
        self.grab_btn = QPushButton("抓取背景图（人走空 → 逐像素取中值）")
        self.grab_btn.clicked.connect(self._grab)
        gl.addWidget(self.grab_btn)
        self.motion_lbl = QLabel("检测区内稳定性：—")
        self.motion_lbl.setStyleSheet("color:#666;")
        gl.addWidget(self.motion_lbl)
        g.setLayout(gl)
        cl.addWidget(g)

        # ---- A4 标定 ----
        g = QGroupBox("3. A4 标定（拖拽量像素）")
        gl = QVBoxLayout()
        r = QHBoxLayout()
        r.addWidget(QLabel("参考长度 cm:"))
        self.cm_sb = QDoubleSpinBox(); self.cm_sb.setRange(1.0, 500.0)
        self.cm_sb.setDecimals(1); self.cm_sb.setValue(A4_LONG_CM)
        self.cm_sb.valueChanged.connect(self._cm_changed)
        r.addWidget(self.cm_sb)
        b = QPushButton("A4 短边 21.0"); b.clicked.connect(lambda: self.cm_sb.setValue(A4_SHORT_CM))
        r.addWidget(b)
        gl.addLayout(r)
        gl.addWidget(QLabel("在图上从参考物左边缘拖到右边缘；近端/中段/远端各量一次"))
        self.meas_lst = QListWidget(); self.meas_lst.setMaximumHeight(92)
        gl.addWidget(self.meas_lst)
        r = QHBoxLayout()
        b = QPushButton("撤销上次"); b.clicked.connect(self._undo_meas); r.addWidget(b)
        b = QPushButton("清空测量"); b.clicked.connect(self._clear_meas); r.addWidget(b)
        b = QPushButton("显示/隐藏 ROI 框"); b.clicked.connect(self._toggle_roi); r.addWidget(b)
        gl.addLayout(r)
        self.summary_lbl = QLabel("（还没有测量）")
        self.summary_lbl.setWordWrap(True)
        self.summary_lbl.setStyleSheet("font-family:Consolas,monospace;color:#222;"
                                       "background:#f6f6f6;padding:6px;")
        gl.addWidget(self.summary_lbl)
        g.setLayout(gl)
        cl.addWidget(g)

        # ---- 写回配置 ----
        g = QGroupBox("4. 写回 abandon_config.txt")
        gl = QVBoxLayout()
        self.chk_minpx = QCheckBox("min_item_px"); self.chk_minpx.setChecked(True)
        self.chk_minarea = QCheckBox("min_item_area")
        self.chk_morph = QCheckBox("morph_open_px=0（小目标推荐）")
        gl.addWidget(self.chk_minpx)
        gl.addWidget(self.chk_minarea)
        gl.addWidget(self.chk_morph)
        r = QHBoxLayout()
        self.chk_rwcm = QCheckBox("region_width_cm"); self.chk_rwcm.setChecked(True)
        r.addWidget(self.chk_rwcm)
        self.rwcm_sb = QDoubleSpinBox(); self.rwcm_sb.setRange(0.0, 100000.0)
        self.rwcm_sb.setDecimals(1); self.rwcm_sb.setValue(0.0)
        r.addWidget(self.rwcm_sb)
        b = QPushButton("由标定反推"); b.clicked.connect(self._derive_width_cm)
        r.addWidget(b)
        gl.addLayout(r)
        self.chk_refbg = QCheckBox("ref_bg_path（板端路径）"); self.chk_refbg.setChecked(True)
        gl.addWidget(self.chk_refbg)
        b = QPushButton("写入 abandon_config.txt")
        b.clicked.connect(self._write_cfg)
        gl.addWidget(b)
        g.setLayout(gl)
        cl.addWidget(g)

        # ---- 上传板端 ----
        g = QGroupBox("5. 上传到板端")
        gl = QVBoxLayout()
        r = QHBoxLayout()
        r.addWidget(QLabel("设备IP:"))
        self.board_ip = QLineEdit(DEFAULT_BOARD_IP)
        r.addWidget(self.board_ip)
        b = QPushButton("探测"); b.clicked.connect(self._probe)
        r.addWidget(b)
        gl.addLayout(r)
        r = QHBoxLayout()
        b = QPushButton("上传背景图"); b.clicked.connect(lambda: self._upload(self._bg_path()))
        r.addWidget(b)
        b = QPushButton("上传 abandon_config.txt")
        b.clicked.connect(lambda: self._upload(ABANDON_CFG))
        r.addWidget(b)
        gl.addLayout(r)
        g.setLayout(gl)
        cl.addWidget(g)

        cl.addWidget(QLabel("日志:"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(110)
        cl.addWidget(self.log_text)
        cl.addStretch()
        ctl.setLayout(cl)
        h.addWidget(ctl)

        self.drawer = CalibDrawer()
        self.drawer.on_change = self._refresh_measurements
        h.addWidget(self.drawer, stretch=1)
        w.setLayout(h)
        self.setCentralWidget(w)

    # ---------------- 配置 ----------------
    def _pick_config(self):
        p, _ = QFileDialog.getOpenFileName(self, "打开配置", ROOT, "Text(*.txt);;All(*.*)")
        if p:
            self._load_config(p)

    def _load_config(self, path):
        try:
            cams = parse_cameras(path)
        except Exception as e:                    # noqa: BLE001
            QMessageBox.warning(self, "err", "解析失败: {}".format(e))
            return
        if not cams:
            QMessageBox.warning(self, "err", "没解析出相机行（前 5 项应为 ip user pwd channel 分辨率）")
            return
        self.cam_cfg_path = path
        self.cams = cams
        self.cfg_lbl.setText(path)
        self.cam_lst.clear()
        for c in cams:
            role = "入口" if c["role"] == 0 else "出口"
            zl = "折线{}点".format(len(c["zone_line"])) if c["zone_line"] else "**无判定线**"
            self.cam_lst.addItem("{} (Ch{})  {}  {}".format(c["ip"], c["channel"], role, zl))
        self._log("已解析 {}：{} 台相机".format(os.path.basename(path), len(cams)))
        self.cam_lst.setCurrentRow(0)
        self._sel_cam(self.cam_lst.item(0))

    def _sel_cam(self, item):
        if item is None:
            return
        i = self.cam_lst.row(item)
        if not (0 <= i < len(self.cams)):
            return
        self.cam_i = i
        cam = self.cams[i]
        self._disconnect()
        self.drawer.measures = []
        self._refresh_measurements()

        # ---- 把配置里的区域喂给画布（与计数脚本同一套渲染）----
        try:
            w, h = [int(v) for v in cam["res"].split("*")]
        except Exception:                         # noqa: BLE001
            w, h = 1920, 1080
        self.drawer.clear()
        self.drawer.set_polygons(cam["polygons"], w, h)
        self.drawer.set_zone(cam["zone_line"], cam["zone_role"], cam["zone_depth"] or 0, w, h)

        # ---- ROI 框（板端实际裁剪的那块）----
        margin = 24
        if os.path.exists(ABANDON_CFG):
            v = read_config_key(ABANDON_CFG, "roi_margin_px")
            if v and re.fullmatch(r"-?\d+", v):
                margin = int(v)
        roi = roi_rect_of(cam, margin)
        self.drawer.roi_rect = roi

        # ---- 几何信息面板（直接对照板端启动日志）----
        lines = ["相机 {} role={} ({})".format(cam["ip"], cam["role"],
                                              "入口" if cam["role"] == 0 else "出口"),
                 "分辨率 {}   多边形 {} 个 / {} 点".format(
                     cam["res"], len(cam["polygons"]),
                     sum(len(p) for p in cam["polygons"])),
                 "判定线 {}".format(
                     "{} 点, 角色 {}, 平移 {:+d}px".format(
                         len(cam["zone_line"]), cam["zone_role"], int(cam["zone_depth"]))
                     if cam["zone_line"] else "**无** → 该相机不会有计数/区域"),
                 "ROI {}".format("{}x{} @({},{})".format(
                     roi[2] - roi[0], roi[3] - roi[1], roi[0], roi[1]) if roi else "—")]
        if roi:
            lines.append("ROI 宽 {}px → px_per_cm = {} / 物理宽".format(
                roi[2] - roi[0], roi[2] - roi[0]))
        self.geo_lbl.setText("\n".join(lines))

        # ---- 自动尝试载入已有背景图（无人干扰，适合反复量）----
        p = self._bg_path()
        if os.path.exists(p):
            img = read_image(p)
            if img is not None:
                self.drawer.set_frame(self._fit(img))
                self._apply_geometry()
                self.bg_image = img
                self._log("已载入已有背景图 {}（可直接在上面量 A4）".format(os.path.basename(p)))
            else:
                self.drawer.set_frame(np.zeros((h, w, 3), np.uint8))
                self._apply_geometry()
        else:
            self.drawer.set_frame(np.zeros((h, w, 3), np.uint8))
            self._apply_geometry()
        self._log("已选 {}：{}".format(cam["ip"], lines[2]))

    def _apply_geometry(self):
        """set_polygons/set_zone 需要画布已知画面尺寸；画面换新后要重放一次。"""
        if not (0 <= self.cam_i < len(self.cams)):
            return
        cam = self.cams[self.cam_i]
        try:
            w, h = [int(v) for v in cam["res"].split("*")]
        except Exception:                         # noqa: BLE001
            w, h = 1920, 1080
        self.drawer.set_polygons(cam["polygons"], w, h)
        self.drawer.set_zone(cam["zone_line"], cam["zone_role"], cam["zone_depth"] or 0, w, h)

    def _fit(self, img):
        """把帧对齐到配置分辨率 —— 区域坐标是按配置分辨率存的，不对齐就会错位。"""
        if not (0 <= self.cam_i < len(self.cams)):
            return img
        try:
            w, h = [int(v) for v in self.cams[self.cam_i]["res"].split("*")]
        except Exception:                         # noqa: BLE001
            return img
        if img.shape[1] != w or img.shape[0] != h:
            return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        return img

    # ---------------- 连接 ----------------
    def _connect(self):
        if self.cam_i < 0:
            QMessageBox.warning(self, "err", "先选一台相机")
            return
        cam = self.cams[self.cam_i]
        self._disconnect()
        self._log("连接 {} … {}".format(cam["ip"], rtsp_url(cam).split("@")[-1]))
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(rtsp_url(cam), cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(rtsp_url(cam))
        if not cap.isOpened():
            self._log("✗ 连不上（检查 IP/账号/密码/channel，或本机是否与相机同网段）")
            cap.release()
            return
        self.cap = cap
        self.frozen = False
        self.timer.start(30)
        self._log("✓ 已连接 {}".format(cam["ip"]))

    def _disconnect(self):
        # 抓帧线程正独占这条连接：此刻 release 会和它的 read() 撞车（轻则报错重则崩溃）
        if self.grab_worker is not None and self.grab_worker.isRunning():
            self._log("正在抓帧，等它结束再断开")
            return
        if self.timer.isActive():
            self.timer.stop()
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:                     # noqa: BLE001
                pass
            self.cap = None

    def _toggle_freeze(self):
        self.frozen = not self.frozen
        self._log("画面已{}".format("冻结（可安心量 A4）" if self.frozen else "继续"))

    def _tick(self):
        if self.cap is None or self.frozen:
            return
        ok, f = self.cap.read()
        if not ok or f is None:
            self._disconnect()
            self._log("摄像头断开")
            return
        self.drawer.set_frame(self._fit(f))
        self._apply_geometry()

    # ---------------- 图像 ----------------
    def _load_image(self):
        p, _ = QFileDialog.getOpenFileName(self, "图片", ROOT, "Images(*.png *.jpg *.jpeg *.bmp)")
        if not p:
            return
        img = read_image(p)
        if img is None:
            self._log("读不到 " + p)
            return
        self.drawer.set_frame(self._fit(img))
        self._apply_geometry()
        self._log("已载入图片 {}".format(os.path.basename(p)))

    def _bg_path(self):
        if self.cam_i < 0:
            return ""
        return os.path.join(ROOT, "abandon_bg_{}.jpg".format(self.cams[self.cam_i]["tag"]))

    def _load_existing_bg(self):
        if self.cam_i < 0:
            QMessageBox.warning(self, "err", "先选一台相机")
            return
        p = self._bg_path()
        if not os.path.exists(p):
            QMessageBox.information(self, "提示", "还没有 {}，先抓一张".format(os.path.basename(p)))
            return
        img = read_image(p)
        if img is None:
            self._log("读不到 " + p)
            return
        self.drawer.set_frame(self._fit(img))
        self._apply_geometry()
        self.bg_image = img
        self._log("已载入 {}（可反复量 A4）".format(os.path.basename(p)))

    def _grab(self):
        if self.cap is None:
            QMessageBox.warning(self, "err", "先连接摄像头（没有实时流就抓不了背景图）")
            return
        if self.grab_worker is not None and self.grab_worker.isRunning():
            return
        self.timer.stop()                       # 抓流期间独占这条连接
        self.grab_btn.setEnabled(False)
        roi = self.drawer.roi_rect
        try:
            tw, th = [int(v) for v in self.cams[self.cam_i]["res"].split("*")]
        except Exception:                         # noqa: BLE001
            tw = th = None
        self._log("抓背景图：{} 帧取中值（检测区内请保持无人）…".format(self.frames_sb.value()))
        self.grab_worker = GrabWorker(self.cap, self.frames_sb.value(), roi, (tw, th))
        self.grab_worker.progress.connect(
            lambda i, n: self.grab_btn.setText("抓取中 {}/{}…".format(i, n)))
        self.grab_worker.done.connect(self._grab_done)
        self.grab_worker.start()

    def _grab_done(self, img, err, motion):
        self.grab_btn.setEnabled(True)
        self.grab_btn.setText("抓取背景图（人走空 → 逐像素取中值）")
        if not self.frozen and self.cap is not None:
            self.timer.start(30)
        if img is None:
            self._log("✗ " + err)
            return

        # ---- 稳定性判据（背景图验收）----
        if motion >= 12.0:
            self.motion_lbl.setText("检测区内稳定性：差（最大帧间差 {:.1f}）".format(motion))
            self.motion_lbl.setStyleSheet("color:#b00;")
            self._log("⚠️ 抓取期间检测区内有明显移动（最大帧间差 {:.1f}）→ 中值图可能残留人或物，"
                      "等人走空重抓".format(motion))
        elif motion >= 5.0:
            self.motion_lbl.setText("检测区内稳定性：一般（{:.1f}）".format(motion))
            self.motion_lbl.setStyleSheet("color:#b70;")
        else:
            self.motion_lbl.setText("检测区内稳定性：好（{:.1f}）".format(motion))
            self.motion_lbl.setStyleSheet("color:#070;")

        path = self._bg_path()
        if self.save_as_cb.isChecked():
            p, _ = QFileDialog.getSaveFileName(self, "另存背景图", path, "JPEG(*.jpg)")
            if p:
                path = p
        if not path:
            self._log("没有目标路径（先选相机）")
            return
        _mkdir(path)
        if not save_jpg(img, path):
            self._log("✗ 写文件失败: " + path)
            return
        h, w = img.shape[:2]
        self._log("✓ 背景图已保存 {}  {}x{}".format(os.path.basename(path), w, h))

        self.bg_image = img
        self.drawer.set_frame(self._fit(img))
        self._apply_geometry()
        self._log("已把背景图载入画布 → 现在可以放 A4 纸开始标定了")

    # ---------------- A4 标定 ----------------
    def _cm_changed(self, v):
        self.drawer.a4_cm = float(v)
        self.drawer.update()
        self._refresh_measurements()

    def _undo_meas(self):
        if self.drawer.measures:
            self.drawer.measures.pop()
        self._refresh_measurements()
        self.drawer.update()

    def _clear_meas(self):
        self.drawer.measures = []
        self._refresh_measurements()
        self.drawer.update()

    def _toggle_roi(self):
        self.drawer.show_roi = not self.drawer.show_roi
        self.drawer.update()

    def _stats(self):
        cm = float(self.cm_sb.value()) or A4_LONG_CM
        ms = self.drawer.measures
        if not ms:
            return None
        vals = [abs(x2 - x1) / cm for (x1, _y1, x2, _y2) in ms]
        return {"cm": cm, "vals": vals, "avg": sum(vals) / len(vals),
                "lo": min(vals), "hi": max(vals), "n": len(vals)}

    def _refresh_measurements(self):
        cm = float(self.cm_sb.value()) or A4_LONG_CM
        self.drawer.a4_cm = cm
        self.meas_lst.clear()
        for i, (x1, _y1, x2, _y2) in enumerate(self.drawer.measures, 1):
            px = abs(x2 - x1)
            self.meas_lst.addItem("#{}  {:>5}px   {:.2f} px/cm".format(i, px, px / cm))
        st = self._stats()
        if not st:
            self.summary_lbl.setText("（还没有测量）")
            return
        px8_lo = 8.0 * st["lo"]
        px8_hi = 8.0 * st["hi"]
        lines = [
            "参考物 {:.1f}cm · {} 次测量".format(st["cm"], st["n"]),
            "平均 px_per_cm = {:.2f}   最近端 {:.2f} / 最远端 {:.2f}".format(
                st["avg"], st["hi"], st["lo"]),
            "8cm × 8cm ≈ {:.0f}px（最近端）… {:.0f}px（最远端）".format(px8_hi, px8_lo),
        ]
        if px8_lo >= 20:
            lines.append("判定：可靠 ✓")
        elif px8_lo >= 15:
            lines.append("判定：勉强 —— 建议 morph_open_px=0、ref_diff_thresh 15~20")
        else:
            lines.append("判定：不可靠 —— 提高分辨率 / 挪近机位 / 缩小检测区")
        if st["n"] == 1:
            lines.append("⚠️ 只量了 1 处：请再到近端/远端各量一次")
        elif st["hi"] / max(st["lo"], 1e-6) >= 1.6:
            lines.append("ℹ 像素密度不均：最近/最远差 {:.1f} 倍 → 按**最远端**定门限".format(
                st["hi"] / st["lo"]))
        lines.append("建议 min_item_px = {}（按最远端，保证远端 8cm 不被滤掉）".format(
            int(round(px8_lo))))
        lines.append("建议 min_item_area = {}（≈远端 8cm 像素面积的一半）".format(
            int(round((px8_lo ** 2) * 0.5))))
        self.summary_lbl.setText("\n".join(lines))

    def _derive_width_cm(self):
        st = self._stats()
        roi = self.drawer.roi_rect
        if not st or not roi:
            QMessageBox.information(self, "提示", "需要至少 1 次测量 + 一个 ROI 框才能反推")
            return
        w_px = roi[2] - roi[0]
        cm = w_px / st["avg"] if st["avg"] else 0
        self.rwcm_sb.setValue(round(cm, 1))
        self._log("由标定反推检测区物理宽 ≈ {:.1f}cm（ROI {}px ÷ 平均 {:.2f}px/cm）"
                  "—— 仅作校核，建议再用卷尺量一次".format(cm, w_px, st["avg"]))

    # ---------------- 写回配置 ----------------
    def _write_cfg(self):
        if not os.path.exists(ABANDON_CFG):
            QMessageBox.warning(self, "err", "找不到 " + ABANDON_CFG)
            return
        pairs = []
        st = self._stats()
        if self.chk_minpx.isChecked():
            if not st:
                QMessageBox.warning(self, "err", "还没测量，算不出 min_item_px")
                return
            pairs.append(("min_item_px", str(int(round(8.0 * st["lo"])))))
        if self.chk_minarea.isChecked():
            if not st:
                QMessageBox.warning(self, "err", "还没测量，算不出 min_item_area")
                return
            pairs.append(("min_item_area", str(int(round((8.0 * st["lo"]) ** 2 * 0.5)))))
        if self.chk_morph.isChecked():
            pairs.append(("morph_open_px", "0"))
        if self.chk_rwcm.isChecked():
            if self.rwcm_sb.value() <= 0:
                QMessageBox.warning(self, "err", "region_width_cm 还是 0：点「由标定反推」或手填")
                return
            pairs.append(("region_width_cm", "{:g}".format(self.rwcm_sb.value())))
        if self.chk_refbg.isChecked():
            if self.cam_i < 0:
                QMessageBox.warning(self, "err", "先选一台相机")
                return
            cam = self.cams[self.cam_i]
            key = "ref_bg_path" if cam["role"] == 0 else "role{}.ref_bg_path".format(cam["role"])
            pairs.append((key, "/home/topeet/code/gatekeeper_V4/abandon_bg_{}.jpg".format(cam["tag"])))
        if not pairs:
            QMessageBox.information(self, "提示", "没有勾选任何要写的项")
            return
        try:
            done = write_config_keys(ABANDON_CFG, pairs)
        except Exception as e:                    # noqa: BLE001
            QMessageBox.warning(self, "err", "写入失败: {}".format(e))
            return
        for k, v in pairs:
            self._log("写入 {}={}".format(k, v))
        self._log("✓ 已更新 {}（改动 {} 项）".format(os.path.basename(ABANDON_CFG), len(done)))
        self._log("  ⚠️ 板端要重启才生效；上传用「上传 abandon_config.txt」")
        QMessageBox.information(self, "ok", "已写入：\n" + "\n".join(
            "{} = {}".format(k, v) for k, v in pairs))

    # ---------------- 上传 ----------------
    def _upload(self, local):
        if not local or not os.path.exists(local):
            QMessageBox.warning(self, "err", "文件不存在：{}".format(local))
            return
        ip = self.board_ip.text().strip()
        if not ip:
            QMessageBox.warning(self, "err", "填设备 IP")
            return
        self._log("上传 {} → {}:{}".format(os.path.basename(local), ip, ZONE.REMOTE_DIR))

        def run():
            try:
                ok, err = ZONE.scp_send(local, ip, self.relay.msg)
            except ImportError:
                self.relay.msg.emit("缺少 paramiko/scp：pip install paramiko scp")
                return
            except Exception as e:                # noqa: BLE001
                self.relay.msg.emit("上传异常: {}".format(e))
                return
            if ok:
                self.relay.msg.emit("✓ 上传成功，且板端旧程序已停止")
            elif err:
                self.relay.msg.emit("✗ 上传失败: {}".format(err))
            else:
                # scp_send 把"没停干净"也返回 False —— 文件其实上去了，别误导
                self.relay.msg.emit("⚠️ 文件已上传，但板端程序没停干净（可能有守护脚本在拉活，"
                                    "见上面的 pgrep 输出）")
        import threading
        threading.Thread(target=run, daemon=True).start()

    def _probe(self):
        ZONE.found_ip = None
        self._log("探测 3588 设备…")

        def run():
            try:
                for ni in ZONE.get_network_prefixes():
                    if ZONE.found_ip:
                        break
                    self.relay.msg.emit("扫描 {}0/24".format(ni["prefix"]))
                    ZONE.probe_network(ni["prefix"], self.relay.msg)
            except Exception as e:                # noqa: BLE001
                self.relay.msg.emit("探测异常: {}".format(e))
            if ZONE.found_ip:
                self.relay.msg.emit("找到设备: {}".format(ZONE.found_ip))
                self.board_ip.setText(ZONE.found_ip)
            else:
                self.relay.msg.emit("未找到设备（可手动填 IP）")
        import threading
        threading.Thread(target=run, daemon=True).start()

    def closeEvent(self, e):
        self._disconnect()
        e.accept()


def main():
    cfg = os.path.join(ROOT, "cameras_config.txt")
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        cfg = sys.argv[1]
    app = QApplication(sys.argv)
    w = MainWindow(cfg)
    w.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
