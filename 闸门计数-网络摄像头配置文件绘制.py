# -*- coding: utf-8 -*-
"""
gatekeeper_V4 — 检测区绘制 + 虚拟绊线 + 远程更新工具
功能：
  1. UDP探测 → 自动扫描3588设备IP
  2. SSH/SCP → 远程更新配置到设备
  3. 加载本地图片/RTSP → 画面绘制检测区多边形 + 虚拟绊线（逆行检测）
  4. 保存/上传配置文件

依赖: pip install PyQt5 opencv-python paramiko scp
"""
import os, sys, re, cv2, time, socket, threading, subprocess, ipaddress
import numpy as np

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QListWidget, QLabel, QFileDialog, QMessageBox,
                             QLineEdit, QGroupBox, QTextEdit, QComboBox, QInputDialog,
                             QSizePolicy, QSpinBox)
from PyQt5.QtCore import Qt, QTimer, QPoint, pyqtSignal, QObject, QThread, QMutex, QMutexLocker
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QPolygon, QFont, QPainterPath

# ==== 远程更新配置 ====
UDP_PORT = 8888
SSH_USER = "topeet"
SSH_PASS = "topeet"
REMOTE_DIR = "/home/topeet/code/gatekeeper_V4"
ALLOWED_FILES = {"cameras_config.txt", "cameras_config_10.txt", "*.txt"}

# ==== 全局变量 ====
found_ip = None
ip_lock = threading.Lock()

# ==================== UDP探测 + SSH传输 ====================
class Communicate(QObject):
    update_log = pyqtSignal(str)
    set_board_ip = pyqtSignal(str)

def get_network_prefixes():
    """获取本机所有网卡网段"""
    prefixes = []
    try:
        out = subprocess.check_output(["ipconfig"], shell=True,
                                       stderr=subprocess.DEVNULL,
                                       creationflags=0x08000000).decode("gbk", errors="ignore")
        lines = out.splitlines()
        cur_ip, cur_mask, cur_adapter, connected = "", "", "", True
        for line in lines:
            if "适配器" in line and ":" in line:
                if cur_ip and cur_mask and connected:
                    try:
                        net = int(ipaddress.IPv4Address(cur_ip)) & int(ipaddress.IPv4Address(cur_mask))
                        pref = ".".join(str(ipaddress.IPv4Address(net)).split(".")[:3]) + "."
                        if not cur_ip.startswith(("127.", "169.254.")):
                            prefixes.append({"adapter": cur_adapter, "prefix": pref, "ip": cur_ip})
                    except: pass
                cur_adapter = line.strip().replace(":", ""); cur_ip = ""; cur_mask = ""; connected = True
            if "媒体已断开" in line: connected = False
            if "IPv4 地址" in line and ":" in line:
                p = line.split(":")[-1].strip()
                if p.count(".") == 3: cur_ip = p
            if "子网掩码" in line and ":" in line:
                p = line.split(":")[-1].strip()
                if p.count(".") == 3: cur_mask = p
        if cur_ip and cur_mask and connected:
            try:
                net = int(ipaddress.IPv4Address(cur_ip)) & int(ipaddress.IPv4Address(cur_mask))
                pref = ".".join(str(ipaddress.IPv4Address(net)).split(".")[:3]) + "."
                if not cur_ip.startswith(("127.", "169.254.")):
                    prefixes.append({"adapter": cur_adapter, "prefix": pref, "ip": cur_ip})
            except: pass
    except: pass
    if not prefixes:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close()
            prefixes = [{"adapter": "默认", "prefix": ".".join(ip.split(".")[:3]) + ".", "ip": ip}]
        except: pass
    return prefixes


def probe_network(prefix, log_signal):
    """扫描单个网段"""
    global found_ip
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    udp.bind(("", 0))

    def recv_loop():
        global found_ip
        udp.settimeout(0.3)
        deadline = time.time() + 4
        while time.time() < deadline:
            with ip_lock:
                if found_ip: return
            try:
                data, addr = udp.recvfrom(1024)
                if data.decode().strip() == "3588_BOARD_RESPONSE":
                    with ip_lock: found_ip = addr[0]
                    log_signal.emit(f"找到设备: {addr[0]}")
                    return
            except socket.timeout: pass
            except: pass

    t = threading.Thread(target=recv_loop, daemon=True); t.start()
    try:
        udp.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        udp.sendto(b"3588_PROBE", (f"{prefix}255", UDP_PORT))
        for i in range(1, 255):
            with ip_lock:
                if found_ip: break
            try: udp.sendto(b"3588_PROBE", (f"{prefix}{i}", UDP_PORT))
            except: pass
            time.sleep(0.005)
    except: pass
    t.join(5)
    udp.close()


def scp_send(local_path, remote_ip, log_signal):
    """SCP发送文件到设备（30秒超时保护）"""
    result = [False, ""]
    done = threading.Event()
    def _upload():
        import paramiko
        from scp import SCPClient
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            ssh.connect(remote_ip, 22, SSH_USER, SSH_PASS, timeout=6, banner_timeout=5, auth_timeout=5)
            log_signal.emit(f"上传 {os.path.basename(local_path)}...")
            scp = SCPClient(ssh.get_transport(), socket_timeout=10)
            scp.put(local_path, f"{REMOTE_DIR}/")
            scp.close()
            log_signal.emit("终止旧进程...")
            ssh.exec_command("sudo pkill -9 -f main_tracking", timeout=5)
            ssh.exec_command("sudo pkill -9 -f video_debug", timeout=5)
            result[0] = True
        except Exception as e:
            result[1] = str(e)
        finally:
            try: ssh.close()
            except: pass
        done.set()
    t = threading.Thread(target=_upload, daemon=True)
    t.start()
    if not done.wait(30):
        log_signal.emit("上传超时(30秒)")
        return False, "上传超时"
    return result[0], result[1]


# ==================== 矩形<->线段换算（保持 5 参配置格式兼容 C++）====================
# 关键: 直角矩形方向 = 外侧边法向 (右手系旋转 -90°), depth 符号表示
#       矩形的"内侧"在外侧边法向的 + 侧 (depth>0) 或 - 侧 (depth<0)
#       C++ 端用同样算法还原 (drawZone 沿外侧边法向 * depth 画矩形)
def _rect_from_line_depth(p1, p2, depth):
    """5 参 (外侧边 p1->p2, 带符号 depth) -> 4 点矩形
       depth 正: 矩形朝 (p2-p1) 右手系法向方向; 负: 反向
    """
    if depth == 0:
        return p1, p2
    dx = p2[1] - p1[1]
    dy = -(p2[0] - p1[0])
    L = (dx*dx + dy*dy) ** 0.5
    if L < 1: return p1, p2
    ux, uy = dx/L, dy/L
    s = abs(depth)
    sign = 1 if depth > 0 else -1
    p3 = (int(round(p1[0] + ux*s*sign)), int(round(p1[1] + uy*s*sign)))
    p4 = (int(round(p2[0] + ux*s*sign)), int(round(p2[1] + uy*s*sign)))
    return p3, p4


def _rect_to_line_depth(zone):
    """4 点矩形 -> (外侧边 p1->p2, 带符号深度)
       深度 = (p3-p1) 和 (p4-p2) 在外侧边法向上的平均投影
       符号: 矩形的"内侧"在外侧边法向的哪一侧 (保留方向信息)
    """
    if not zone or len(zone) != 4:
        return None, 0
    p1, p2, p3, p4 = zone
    # 外侧边法向 (右手系旋转 -90°)
    dx = p2[1] - p1[1]
    dy = -(p2[0] - p1[0])
    L = (dx*dx + dy*dy) ** 0.5
    if L < 1: return [p1, p2], 0
    ux, uy = dx/L, dy/L
    # 平均深度 = (p3-p1) 和 (p4-p2) 在法向上的投影
    d1 = (p3[0]-p1[0])*ux + (p3[1]-p1[1])*uy
    d2 = (p4[0]-p2[0])*ux + (p4[1]-p2[1])*uy
    depth = (d1 + d2) / 2
    return [p1, p2], int(round(depth))


# ==================== 多边形绘制 ====================
# 入/出标记区域深度（px）：标记线向检测区内延伸 100px 为入标记区域，
# 向检测区外延伸 50px 为出标记区域。
ENTRY_MARKER_DEPTH = 100
EXIT_MARKER_DEPTH = 50


class PolygonDrawer(QWidget):
    MODE_POLYGON = 0
    MODE_MARKER_LINE = 1   # 标记线: 选检测区域边界 2 点 -> 自动算入/出标记区域

    def __init__(self):
        super().__init__()
        self.current = []         # 绘制中: [p1_disp, p2_disp]
        self.polygons = []
        # 4 点矩形: [(p1),(p2),(p3),(p4)]  p1->p2 标记线(外侧边), p3->p4 内侧边
        self.entry_zone = None    # 入标记区域（标记线向检测区内 ENTRY_MARKER_DEPTH）
        self.exit_zone = None     # 出标记区域（标记线向检测区外 EXIT_MARKER_DEPTH）
        self.marker_line = None   # 标记线 [p1, p2]（图像坐标）
        self.mode = self.MODE_POLYGON
        self.drawing = False
        self.frame = None
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ow = self.oh = 0
        self.scale = 1.0; self.ox = self.oy = 0; self.dw = self.dh = 0

    def set_mode(self, mode): self.mode = mode; self.drawing = False; self.current = []

    def set_frame(self, f):
        self.frame = f.copy(); self.oh, self.ow = f.shape[:2]; self._geo(); self.update()

    def _geo(self):
        if self.frame is None: return
        ww, wh = self.width(), self.height()
        self.scale = min(ww / self.ow, wh / self.oh)
        self.dw, self.dh = int(self.ow * self.scale), int(self.oh * self.scale)
        self.ox, self.oy = (ww - self.dw)//2, (wh - self.dh)//2

    def resizeEvent(self, e): self._geo(); self.update()

    def _td(self, x, y): return QPoint(int(x*self.scale)+self.ox, int(y*self.scale)+self.oy)
    def _to(self, p):
        return max(0,min(int((p.x()-self.ox)/self.scale), self.ow)), max(0,min(int((p.y()-self.oy)/self.scale), self.oh))
    def _clamp(self, p):
        return QPoint(max(self.ox,min(p.x(),self.ox+self.dw)), max(self.oy,min(p.y(),self.oy+self.dh)))

    # ---- 标记线 -> 入/出标记区域 ----
    def _compute_marker_zones(self, p1, p2):
        """由标记线 [p1,p2] 和检测区域多边形，自动计算入/出标记区域。
        入标记区域 = 标记线向检测区内延伸 ENTRY_MARKER_DEPTH px
        出标记区域 = 标记线向检测区外延伸 EXIT_MARKER_DEPTH px
        入侧方向 = 指向检测区域多边形中心的方向。"""
        dx = p2[1] - p1[1]
        dy = -(p2[0] - p1[0])
        L = (dx*dx + dy*dy) ** 0.5
        if L < 1:
            return None, None
        ux, uy = dx/L, dy/L   # 右手系法向 (与 C++ inwardNormal 约定一致)
        # 检测区域中心（所有多边形点的平均）
        cx = cy = None
        pts = [pt for poly in self.polygons for pt in poly]
        if pts:
            cx = sum(x for x, y in pts) / len(pts)
            cy = sum(y for x, y in pts) / len(pts)
        if cx is not None:
            dot = (cx - p1[0]) * ux + (cy - p1[1]) * uy
            if dot < 0:
                ux, uy = -ux, -uy   # 翻转，指向检测区内
        entry = [p1, p2,
                 (int(round(p1[0] + ux*ENTRY_MARKER_DEPTH)), int(round(p1[1] + uy*ENTRY_MARKER_DEPTH))),
                 (int(round(p2[0] + ux*ENTRY_MARKER_DEPTH)), int(round(p2[1] + uy*ENTRY_MARKER_DEPTH)))]
        exit_ = [p1, p2,
                 (int(round(p1[0] - ux*EXIT_MARKER_DEPTH)), int(round(p1[1] - uy*EXIT_MARKER_DEPTH))),
                 (int(round(p2[0] - ux*EXIT_MARKER_DEPTH)), int(round(p2[1] - uy*EXIT_MARKER_DEPTH)))]
        return entry, exit_

    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton: return
        p = self._clamp(e.pos())
        if self.mode == self.MODE_MARKER_LINE:
            # 标记线: 两次点击分别确定 p1 / p2
            if len(self.current) == 0:
                self.current = [p]
                self.drawing = True
            elif len(self.current) == 1:
                self.current.append(p)
                self.drawing = False
                self._finish_marker_line()
        else:
            self.drawing = True; self.current = [p]
        self.update()

    def mouseMoveEvent(self, e):
        if self.mode != self.MODE_POLYGON:
            return
        if not self.drawing: return
        p = self._clamp(e.pos())
        if not (e.buttons() & Qt.LeftButton): return
        if self.current:
            l = self.current[-1]
            if abs(l.x()-p.x())>5 or abs(l.y()-p.y())>5: self.current.append(p); self.update()

    def mouseReleaseEvent(self, e):
        if e.button() != Qt.LeftButton: return
        if self.mode == self.MODE_MARKER_LINE:
            self.update()
            return
        if self.drawing and self.mode == self.MODE_POLYGON and len(self.current) > 2:
            p = self._clamp(e.pos())
            l = self.current[-1]
            if abs(l.x()-p.x())>5 or abs(l.y()-p.y())>5: self.current.append(p)
            self.polygons.append([self._to(pt) for pt in self.current])
        self.update()

    def _finish_marker_line(self):
        """标记线 2 点已选，自动计算入/出标记区域。"""
        if len(self.current) < 2: return
        p1 = self._to(self.current[0]); p2 = self._to(self.current[1])
        self.marker_line = [p1, p2]
        self.entry_zone, self.exit_zone = self._compute_marker_zones(p1, p2)
        self.current = []
        self.drawing = False
        self.update()

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        if self.frame is None: return
        s = cv2.resize(self.frame, (self.dw, self.dh), interpolation=cv2.INTER_AREA)
        h, w = s.shape[:2]
        qi = QImage(s.data, w, h, 3*w, QImage.Format_RGB888).rgbSwapped()
        p.drawPixmap(self.ox, self.oy, QPixmap.fromImage(qi))

        if self.polygons:
            ov = QPixmap(self.width(), self.height()); ov.fill(Qt.transparent)
            op = QPainter(ov); op.setPen(Qt.NoPen); op.setBrush(QColor(0,120,255,40))
            for poly in self.polygons:
                op.drawPolygon(QPolygon([self._td(x,y) for x,y in poly]))
            op.end(); p.drawPixmap(0,0,ov)
            p.setPen(QPen(QColor(0,120,255), 3)); p.setBrush(Qt.NoBrush)
            for poly in self.polygons:
                p.drawPolygon(QPolygon([self._td(x,y) for x,y in poly]))

        # 入标记区域 (绿) + 出标记区域 (红)
        # 存储顺序: zone = [p1, p2, p3, p4]  p1->p2 标记线(外侧边), p3->p4 内侧边
        # 矩形顶点绘制顺序必须为 p1 -> p2 -> p4 -> p3 (顺时针或逆时针), 不能 p1->p2->p3->p4
        # 否则 Qt drawPolygon 会画成 2 个三角形
        for zone, fill, col, label in [
            (self.entry_zone, QColor(0,255,0,50), QColor(0,160,0), "入标记区域"),
            (self.exit_zone,  QColor(255,0,0,50), QColor(160,0,0), "出标记区域"),
        ]:
            if zone and len(zone) == 4:
                # 顶点顺序: p1, p2, p4, p3 (跳过 p3 到 p4 的对角)
                pts_disp = [self._td(*zone[0]), self._td(*zone[1]),
                            self._td(*zone[3]), self._td(*zone[2])]
                path = QPainterPath(); path.moveTo(pts_disp[0])
                for pp in pts_disp[1:]: path.lineTo(pp)
                path.closeSubpath()
                p.fillPath(path, fill)
                # 4 边
                p.setPen(QPen(col, 2))
                p.drawPolygon(QPolygon(pts_disp))
                # 标记线端点 (p1/p2) 高亮
                p.setBrush(QColor(255,255,255)); p.setPen(QPen(col, 2))
                p.drawEllipse(self._td(*zone[0]), 5, 5)
                p.drawEllipse(self._td(*zone[1]), 5, 5)
                # 标签
                cx = sum(x for x,y in zone) // 4
                cy = sum(y for x,y in zone) // 4
                tp = self._td(cx, cy)
                p.setPen(QPen(col, 1))
                p.drawText(tp.x()-20, tp.y()+5, label)

        # 标记线 (黄, 高亮)
        if self.marker_line and len(self.marker_line) == 2:
            p.setPen(QPen(QColor(255,255,0), 3))
            p.drawLine(self._td(*self.marker_line[0]), self._td(*self.marker_line[1]))

        # 画标记线过程中: 显示预览 (1 个点已放, 提示再点一次)
        if self.drawing and self.mode == self.MODE_MARKER_LINE and self.current:
            p1d = self.current[0]
            p.setBrush(QColor(255,255,255)); p.setPen(QPen(QColor(255,255,0), 2))
            p.drawEllipse(p1d, 5, 5)
            p.drawText(p1d.x()+10, p1d.y()-10, "标记线起点已放, 再点一次 = 标记线终点")

        if len(self.current) > 1 and self.mode == self.MODE_POLYGON:
                p.setPen(QPen(QColor(0,255,0), 2)); p.drawPolyline(QPolygon(self.current))
                if len(self.current) > 2: p.drawLine(self.current[-1], self.current[0])

    def clear(self):
        self.polygons = []; self.entry_zone = None; self.exit_zone = None
        self.marker_line = None; self.update()
    def delete_last(self):
        if self.polygons: self.polygons.pop(); self.update()
    def set_polygons(self, ps, ow, oh):
        sx = self.ow/ow if ow>0 else 1; sy = self.oh/oh if oh>0 else 1
        self.polygons = [[(int(x*sx),int(y*sy)) for x,y in p] for p in ps]; self.update()
    def set_zones(self, entry_zone, exit_zone, ow, oh):
        """设置 4 点矩形 (来自配置)，并据此重建标记线"""
        sx = self.ow/ow if ow>0 else 1; sy = self.oh/oh if oh>0 else 1
        if entry_zone and len(entry_zone) == 4:
            self.entry_zone = [(int(x*sx),int(y*sy)) for x,y in entry_zone]
            self.marker_line = [self.entry_zone[0], self.entry_zone[1]]
        if exit_zone and len(exit_zone) == 4:
            self.exit_zone = [(int(x*sx),int(y*sy)) for x,y in exit_zone]
        self.update()


# ==================== RTSP连接线程 ====================
class RTSPThread(QThread):
    ok = pyqtSignal(object, str)
    fail = pyqtSignal(str)

    def __init__(self, ip, user, pwd, ch):
        super().__init__(); self.ip=ip; self.u=user; self.p=pwd; self.c=int(ch)
    def run(self):
        url = f"rtsp://{self.u}:{self.p}@{self.ip}:554/Streaming/Channels/{(self.c-1)*100+1}"
        try:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            if not cap.isOpened(): self.fail.emit("打开失败"); return
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
            ret, _ = cap.read()
            if not ret: self.fail.emit("无画面"); cap.release(); return
            self.ok.emit(cap, self.ip)
        except Exception as e: self.fail.emit(str(e))


# ==================== 主窗口 ====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("gatekeeper_V4 — 检测区 + 绊线绘制 + 远程更新")
        self.setGeometry(80, 80, 1300, 750)
        self.cfg_path = None; self.cams = []; self.cam_i = -1
        self.cap = None; self.timer = QTimer(self); self.timer.timeout.connect(self._upd)
        self.cam_thread = None
        self.comm = Communicate()
        self.comm.update_log.connect(self._log)
        self.comm.set_board_ip.connect(self._set_ip)
        self._ui()
        self._log("gatekeeper_V4 启动就绪")

    def _log(self, t):
        self.log_text.append(t)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def _set_ip(self, ip):
        self.board_ip_edit.setText(ip)
        self.board_ip_edit.setEnabled(False)

    def _ui(self):
        w = QWidget(); l = QHBoxLayout()
        ctl = QWidget(); ctl.setFixedWidth(320); cl = QVBoxLayout()

        # ---- 远程更新 ----
        bg = QGroupBox("远程更新"); bl = QVBoxLayout()
        self.probe_btn = QPushButton("探测3588设备"); self.probe_btn.clicked.connect(self._probe)
        bl.addWidget(self.probe_btn)

        r1 = QHBoxLayout()
        r1.addWidget(QLabel("设备IP:")); self.board_ip_edit = QLineEdit()
        self.board_ip_edit.setPlaceholderText("手动或探测获取"); r1.addWidget(self.board_ip_edit)
        bl.addLayout(r1)

        r2 = QHBoxLayout()
        self.file_ed = QLineEdit(); self.file_ed.setReadOnly(True)
        b = QPushButton("选文件"); b.clicked.connect(self._sel_file); r2.addWidget(b)
        r2.addWidget(self.file_ed); bl.addLayout(r2)

        b = QPushButton("更新到设备"); b.clicked.connect(self._upload); bl.addWidget(b)

        self.log_text = QTextEdit(); self.log_text.setReadOnly(True); self.log_text.setMinimumHeight(100)
        bl.addWidget(QLabel("日志:")); bl.addWidget(self.log_text)
        bg.setLayout(bl); cl.addWidget(bg)

        # ---- 配置文件 ----
        fg = QGroupBox("配置文件"); fl = QVBoxLayout()
        b = QPushButton("打开配置"); b.clicked.connect(self._load); fl.addWidget(b)
        b = QPushButton("保存配置"); b.clicked.connect(self._save); fl.addWidget(b)
        fg.setLayout(fl); cl.addWidget(fg)

        # ---- 摄像头列表 ----
        cg = QGroupBox("摄像头"); cgl = QVBoxLayout()
        self.cam_lst = QListWidget(); self.cam_lst.itemClicked.connect(self._sel_cam); cgl.addWidget(self.cam_lst)
        cg.setLayout(cgl); cl.addWidget(cg)

        # ---- 摄像头信息 ----
        ig = QGroupBox("信息"); il = QVBoxLayout()
        self.ip_ed = QLineEdit(); self.user_ed = QLineEdit(); self.pass_ed = QLineEdit()
        self.ch_ed = QLineEdit(); self.res_cb = QComboBox()
        self.res_cb.addItems(["1920*1080","1280*720","704*576"])
        for t, ed in [("IP:",self.ip_ed),("用户:",self.user_ed),("密码:",self.pass_ed),("通道:",self.ch_ed)]:
            r = QHBoxLayout(); r.addWidget(QLabel(t)); r.addWidget(ed); il.addLayout(r)
        r = QHBoxLayout(); r.addWidget(QLabel("分辨率:")); r.addWidget(self.res_cb); il.addLayout(r)
        ig.setLayout(il); cl.addWidget(ig)

        # ---- 绘制 ----
        pg = QGroupBox("检测区绘制"); pl = QVBoxLayout()
        b = QPushButton("连接摄像头"); b.clicked.connect(self._cam); pl.addWidget(b)
        b = QPushButton("加载本地图片"); b.clicked.connect(self._img); pl.addWidget(b)

        r = QHBoxLayout()
        b = QPushButton("画区域"); b.clicked.connect(lambda:self.drawer.set_mode(PolygonDrawer.MODE_POLYGON)); r.addWidget(b)
        b = QPushButton("画标记线"); b.clicked.connect(lambda:self.drawer.set_mode(PolygonDrawer.MODE_MARKER_LINE)); r.addWidget(b)
        pl.addLayout(r)
        
        r = QHBoxLayout()
        b = QPushButton("删最后"); b.clicked.connect(lambda:self.drawer.delete_last()); r.addWidget(b)
        b = QPushButton("清空全部"); b.clicked.connect(lambda:self.drawer.clear()); r.addWidget(b)
        pl.addLayout(r)

        pl.addWidget(QLabel("步骤: 1)先画检测区域(多边形)  2)在区域边界点2点=标记线"))
        pl.addWidget(QLabel("自动: 入标记区域=向区域内100px | 出标记区域=向区域外50px"))
        b = QPushButton("保存当前"); b.clicked.connect(self._save_cur); pl.addWidget(b)
        pg.setLayout(pl); cl.addWidget(pg)
        cl.addStretch()
        ctl.setLayout(cl); l.addWidget(ctl)
        self.drawer = PolygonDrawer(); l.addWidget(self.drawer, stretch=1)
        w.setLayout(l); self.setCentralWidget(w)

    # ==== 远程探测 ====
    def _probe(self):
        global found_ip
        self._log("探测3588设备...")
        with ip_lock: found_ip = None
        self.board_ip_edit.setEnabled(True); self.probe_btn.setEnabled(False)
        def run():
            for ni in get_network_prefixes():
                with ip_lock:
                    if found_ip: break
                self._log(f"扫描 {ni['prefix']}0/24")
                probe_network(ni['prefix'], self.comm.update_log)
            with ip_lock:
                if found_ip:
                    self.comm.set_board_ip.emit(found_ip)
                    self._log(f"完成: {found_ip}")
                else: self._log("未找到设备")
            self.probe_btn.setEnabled(True)
        threading.Thread(target=run, daemon=True).start()

    def _sel_file(self):
        p, _ = QFileDialog.getOpenFileName(self, "选文件", "", "All(*.*)")
        if p: self.file_ed.setText(p)

    def _upload(self):
        ip = self.board_ip_edit.text().strip()
        fp = self.file_ed.text().strip() or self.cfg_path or ""
        if not ip: QMessageBox.warning(self,"err","无设备IP"); return
        if not fp: QMessageBox.warning(self,"err","无文件"); return
        pw, ok = QInputDialog.getText(self,"密码","更新密码:", QLineEdit.Password)
        if not ok or pw != "topeet": QMessageBox.warning(self,"err","密码错"); return
        self._log(f"上传 {fp}")
        def run():
            ok2, err = scp_send(fp, ip, self.comm.update_log)
            if ok2: self.comm.update_log.emit("上传成功"); QMessageBox.information(self,"ok","上传成功")
            else: self.comm.update_log.emit(f"失败:{err}"); QMessageBox.warning(self,"err",err)
        threading.Thread(target=run, daemon=True).start()

    # ==== 配置解析 ====
    def _load(self):
        p, _ = QFileDialog.getOpenFileName(self,"打开","","Text(*.txt);;All(*)")
        if not p: return
        self.cfg_path = p; self.cams = []; self._log(f"加载: {p}")
        try:
            with open(p,'r',encoding='utf-8') as f:
                for ln in f:
                    ln=ln.strip()
                    if not ln or ln.startswith('#'): continue
                    ps=ln.split()
                    if len(ps)<4: continue
                    ip,u,pwd,ch=ps[0],ps[1],ps[2],ps[3]
                    res="1920*1080"; rest=ps[4:] if len(ps)>4 else []
                    if rest and '*' in rest[0]: res=rest[0]; rest=rest[1:]
                    
                    polys=[]; entry=None; exit=None
                    for token in rest:
                        if token.startswith('t1,'):
                            parts=token[3:].replace(',',' ').split()
                            # 标记线 5 参: t1,x1,y1,x2,y2,depth
                            # depth 符号表入侧方向, |depth|=入标记区域深度
                            if len(parts) >= 5:
                                d = int(parts[4])
                                p1 = (int(parts[0]), int(parts[1]))
                                p2 = (int(parts[2]), int(parts[3]))
                                p3, p4 = _rect_from_line_depth(p1, p2, d)
                                entry = [p1, p2, p3, p4]
                                # 出标记区域 = 标记线向检测区外(反方向)延伸 EXIT_MARKER_DEPTH
                                exit_d = EXIT_MARKER_DEPTH if d < 0 else -EXIT_MARKER_DEPTH
                                e3, e4 = _rect_from_line_depth(p1, p2, exit_d)
                                exit = [p1, p2, e3, e4]
                            elif len(parts) >= 4:
                                p1 = (int(parts[0]), int(parts[1]))
                                p2 = (int(parts[2]), int(parts[3]))
                                entry = [p1, p2, p1, p2]
                        elif token.startswith('t2,'):
                            parts=token[3:].replace(',',' ').split()
                            if len(parts) >= 5:
                                d = int(parts[4])
                                p1 = (int(parts[0]), int(parts[1]))
                                p2 = (int(parts[2]), int(parts[3]))
                                p3, p4 = _rect_from_line_depth(p1, p2, d)
                                exit = [p1, p2, p3, p4]
                            elif len(parts) >= 4:
                                p1 = (int(parts[0]), int(parts[1]))
                                p2 = (int(parts[2]), int(parts[3]))
                                exit = [p1, p2, p1, p2]
                        elif token.startswith('['):
                            pts=[]
                            pstr=token[1:].rstrip(']')
                            # handle multi-token polygon
                            if ']' not in token:
                                # accumulate
                                pass
                            for pt_str in pstr.split():
                                if ',' in pt_str:
                                    xs,ys=pt_str.split(',')
                                    pts.append((int(xs),int(ys)))
                            if len(pts)>=3: polys.append(pts)

                    # handle multi-token polygons (re-parse properly)
                    pstr=' '.join(rest)
                    polys2=[]
                    start=0
                    while True:
                        a=pstr.find('[',start)
                        if a==-1: break
                        b=pstr.find(']',a)
                        if b==-1: break
                        pts=[]
                        for ps2 in pstr[a+1:b].split():
                            if ',' in ps2:
                                x,y=map(int,ps2.split(','))
                                pts.append((x,y))
                        if len(pts)>=3: polys2.append(pts)
                        start=b+1

                    self.cams.append({'ip':ip,'username':u,'password':pwd,'channel':ch,
                                      'resolution':res,'polygons':polys2,
                                      'entry_zone':entry,'exit_zone':exit})
        except Exception as e: QMessageBox.warning(self,"err",str(e)); return
        self.cam_lst.clear()
        for c in self.cams: self.cam_lst.addItem(f"{c['ip']}(Ch{c['channel']})")
        if self.cams: self.cam_lst.setCurrentRow(0); self._sel_cam(self.cam_lst.item(0))

    def _sel_cam(self, item):
        i=self.cam_lst.row(item)
        if i<0: return
        self.cam_i=i; c=self.cams[i]
        self.ip_ed.setText(c['ip']); self.user_ed.setText(c['username'])
        self.pass_ed.setText(c['password']); self.ch_ed.setText(c['channel'])
        self.res_cb.setCurrentText(c['resolution']); self.drawer.clear()
        if self.drawer.frame is not None:
            w,h=map(int,c['resolution'].split('*'))
            self.drawer.set_polygons(c['polygons'],w,h)
            self.drawer.set_zones(c.get('entry_zone'),c.get('exit_zone'),w,h)

    def _cam(self):
        if self.cam_i<0: QMessageBox.warning(self,"err","先选摄像头"); return
        c=self.cams[self.cam_i]
        self._stop_cam()
        self.cam_thread = RTSPThread(*[self.ip_ed.text().strip(),
            self.user_ed.text().strip(), self.pass_ed.text().strip(),
            self.ch_ed.text().strip()])
        self.cam_thread.ok.connect(self._cam_ok)
        self.cam_thread.fail.connect(self._cam_fail)
        self.cam_thread.start()

    def _cam_ok(self, cap, ip):
        self.cap=cap; self.timer.start(30); self._log(f"已连接 {ip}")
        # 摄像头连接后加载一次当前配置中的多边形和绊线
        if 0<=self.cam_i<len(self.cams):
            c=self.cams[self.cam_i]; w,h=map(int,c['resolution'].split('*'))
            self.drawer.set_polygons(c['polygons'],w,h)
            self.drawer.set_zones(c.get('entry_zone'),c.get('exit_zone'),w,h)

    def _cam_fail(self, msg): self._log(f"连接失败:{msg}")

    def _stop_cam(self):
        if self.timer.isActive(): self.timer.stop()
        if self.cap: self.cap.release(); self.cap=None
        if self.cam_thread and self.cam_thread.isRunning():
            self.cam_thread.quit(); self.cam_thread.wait(1000)

    def _img(self):
        p,_=QFileDialog.getOpenFileName(self,"图片","","Images(*.png *.jpg *.bmp)")
        if not p: return
        f=cv2.imread(p)
        if f is not None:
            self.drawer.set_frame(f)
            if 0<=self.cam_i<len(self.cams):
                c=self.cams[self.cam_i]; w,h=map(int,c['resolution'].split('*'))
                self.drawer.set_polygons(c['polygons'],w,h)
                self.drawer.set_zones(c.get('entry_zone'),c.get('exit_zone'),w,h)
            self._log(f"已加载: {os.path.basename(p)}")

    def _upd(self):
        if self.cap is None: return
        ret,f=self.cap.read()
        if not ret: self._stop_cam(); self._log("摄像头断开"); return
        w,h=map(int,self.res_cb.currentText().split('*'))
        f=cv2.resize(f,(w,h)); self.drawer.set_frame(f)

    def _save_cur(self):
        if self.cam_i<0: return
        c=self.cams[self.cam_i]
        c['resolution']=self.res_cb.currentText()
        c['polygons']=self.drawer.polygons
        c['entry_zone']=self.drawer.entry_zone
        c['exit_zone']=self.drawer.exit_zone
        QMessageBox.information(self,"ok","已保存到内存")

    def _save(self):
        if not self.cams: return
        p,_=QFileDialog.getSaveFileName(self,"保存",self.cfg_path or "","Text(*.txt)")
        if not p: return
        with open(p,'w',encoding='utf-8') as f:
            for c in self.cams:
                lp=[c['ip'],c['username'],c['password'],c['channel'],c['resolution']]
                for poly in c['polygons']:
                    # 跳过退化多边形（全是0或面积太小, <3个点）
                    if len(poly) < 3: continue
                    xs = [pt[0] for pt in poly]; ys = [pt[1] for pt in poly]
                    if max(xs)==0 and max(ys)==0: continue
                    lp.append('['+' '.join(f"{x},{y}" for x,y in poly)+']')
                # 标记线 4 点(入标记区域) -> 换算成 5 参格式 (标记线 + 带符号入深度)
                # 出标记区域(50px) 由 C++ 端按入深度反方向固定延伸，无需单独存储。
                if c.get('entry_zone') and len(c['entry_zone']) == 4:
                    line, depth = _rect_to_line_depth(c['entry_zone'])
                    if line:
                        lp.append(f"t1,{line[0][0]},{line[0][1]},{line[1][0]},{line[1][1]},{depth}")
                f.write(' '.join(lp)+'\n')
        QMessageBox.information(self,"ok",f"已保存: {p}"); self.cfg_path=p

    def closeEvent(self, e):
        self._stop_cam(); e.accept()


if __name__=="__main__":
    app=QApplication(sys.argv)
    w=MainWindow(); w.show()
    sys.exit(app.exec_())
