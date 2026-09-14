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
    probe_done = pyqtSignal()             # 探测结束（子线程不能直接碰控件，必须走信号回落 GUI 线程）
    upload_done = pyqtSignal(bool, str)   # 上传+停止结束(成功?, 错误信息)

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


def _ssh_run(ssh, cmd, log_signal, timeout=8):
    """执行一条远程命令，返回 (退出码, 全部输出)。
    必须 read() + recv_exit_status()，否则命令是否成功、报了什么错都看不到。"""
    try:
        _, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
        out = stdout.read().decode("utf-8", "ignore").strip()
        err = stderr.read().decode("utf-8", "ignore").strip()
        rc = stdout.channel.recv_exit_status()
        if out:
            log_signal.emit(f"  $ {cmd}\n    {out}")
        if err:
            log_signal.emit(f"  $ {cmd}  [stderr] {err}")
        return rc, (out + "\n" + err)
    except Exception as e:
        log_signal.emit(f"  $ {cmd}  执行异常: {e}")
        return -1, str(e)


def stop_remote_programs(ssh, log_signal):
    """停止板端 main_tracking / video_debug，并**验证**是否真的停了。

    为什么不用 sudo：SSH 非交互通道没有 tty，`sudo pkill ...` 会直接报
    "no tty present and no askpass program specified" 并失败——这正是
    "更新到设备后，板端程序还在跑"的根因。topeet 用户杀掉自己启的进程不需要 sudo。

    为什么还要验证：板端常见 `while true; do ./main_tracking; done` 或 rc.local/systemd
    之类守护，杀了会立刻被拉起来。这里用 pgrep 复查并把存活进程+排查线索写进日志，
    而不是假装成功了。
    """
    log_signal.emit("停止板端程序...")
    _ssh_run(ssh, "pkill -9 -f main_tracking", log_signal)
    _ssh_run(ssh, "pkill -9 -f video_debug", log_signal)
    time.sleep(1.0)
    _, left = _ssh_run(ssh, "pgrep -af 'main_tracking|video_debug' || true", log_signal)
    alive = [ln for ln in left.splitlines() if ln.strip()]
    if alive:
        log_signal.emit("⚠ 停止后仍有进程存活 —— 板端很可能有自启动/守护脚本在拉活：")
        for ln in alive:
            log_signal.emit(f"    {ln}")
        log_signal.emit("  排查: ps -ef | grep -v grep | grep -E 'main_tracking|video_debug'")
        log_signal.emit("  看这些进程的 PPID 是谁（while 循环 / rc.local / systemd），停掉对应的守护即可")
        return False
    log_signal.emit("已确认板端程序全部停止")
    return True


def scp_send(local_path, remote_ip, log_signal):
    """SCP发送文件到设备（45秒超时保护）+ 停止板端旧程序"""
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
            log_signal.emit("上传完成")
            result[0] = stop_remote_programs(ssh, log_signal)
        except Exception as e:
            result[1] = str(e)
        finally:
            try: ssh.close()
            except: pass
        done.set()
    t = threading.Thread(target=_upload, daemon=True)
    t.start()
    if not done.wait(45):
        log_signal.emit("上传超时(45秒)")
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
# 入口带深度（px）：入口绊线(t1)向检测区内延伸这么多像素 = C++ 端的 R1 区域。
# 出口区域已删除（req4）：检测区域多边形之外一律视为出口(R0)，配置文件不再写 t2。
ENTRY_MARKER_DEPTH = 100


def _depth_toward_detection(polys, p1, p2, depth=ENTRY_MARKER_DEPTH):
    """判断绊线 p1->p2 的哪一侧朝向检测区中心，返回**带符号**深度。

    右手系法向 u = (dy, -dx)/L 与 C++ inwardNormal 完全一致；depth>0 表示入口带
    朝 u 方向延伸。旧配置文件的 t1 只有 4 个参数（没有深度），若当成 depth=0，
    入口带会退化成一条线，保存时又会把板端 R1 区域抹掉，所以这里按检测区中心
    自动定向补一个默认深度。
    """
    dx = p2[1] - p1[1]
    dy = -(p2[0] - p1[0])
    L = (dx*dx + dy*dy) ** 0.5
    if L < 1:
        return depth
    ux, uy = dx/L, dy/L
    pts = [pt for poly in polys if len(poly) >= 3 for pt in poly]
    if not pts:
        return depth
    cx = sum(x for x, y in pts) / len(pts)
    cy = sum(y for x, y in pts) / len(pts)
    dot = (cx - p1[0]) * ux + (cy - p1[1]) * uy
    return depth if dot >= 0 else -depth


class PolygonDrawer(QWidget):
    MODE_POLYGON = 0
    MODE_MARKER_LINE = 1   # 标记线: 选检测区域边界 2 点 -> 自动算入/出标记区域

    def __init__(self):
        super().__init__()
        self.current = []         # 绘制中: [p1_disp, p2_disp]
        self.polygons = []
        # 4 点矩形: [(p1),(p2),(p3),(p4)]  p1->p2 绊线(外侧边), p3->p4 内侧边
        self.entry_zone = None    # 入口带（t1 向检测区内 ENTRY_MARKER_DEPTH）
        self.exit_zone = None     # 出口区域已删除（检测区外即出口），保留字段仅为兼容
        self.marker_line = None   # 入口绊线 [p1, p2]（图像坐标）
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

    # ---- 入口绊线 -> 入口带 ----
    def _compute_marker_zones(self, p1, p2):
        """由入口绊线 [p1,p2] 和检测区域多边形，自动计算入口带。
        入口带 = 绊线向检测区内延伸 ENTRY_MARKER_DEPTH px（C++ 端 R1 区域）。
        出口区域已删除：检测区域多边形之外一律视为出口(R0)。"""
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
        return entry, None

    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton: return
        if self.frame is None or self.dw <= 0 or self.dh <= 0: return   # 没有画面就没有坐标系，别画
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
            # 收尾必须复位：否则 paintEvent 会把刚完成的区域当成"绘制中"再描一遍绿线
            self.current = []
            self.drawing = False
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
        if self.frame is None or self.dw <= 0 or self.dh <= 0:
            return   # 没有画面 / 控件还没布局好：此时 cv2.resize((0,0)) 会直接抛异常
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
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

        # 入口带 (绿)；出口区域已删除
        # 存储顺序: zone = [p1, p2, p3, p4]  p1->p2 绊线(外侧边), p3->p4 内侧边
        # 矩形顶点绘制顺序必须为 p1 -> p2 -> p4 -> p3 (顺时针或逆时针), 不能 p1->p2->p3->p4
        # 否则 Qt drawPolygon 会画成 2 个三角形
        for zone, fill, col, label in [
            (self.entry_zone, QColor(0,255,0,50), QColor(0,160,0), "入口带"),
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
        # 画面还没加载时 self.ow/oh 为 0，无法把配置坐标换算到画面坐标；
        # 强行算会得到 scale=0 把所有点压到 (0,0)。交给后面加载画面时再应用。
        if not self.ow or not self.oh or ow <= 0 or oh <= 0: return
        sx, sy = self.ow/ow, self.oh/oh
        self.polygons = [[(int(x*sx),int(y*sy)) for x,y in p] for p in ps]; self.update()
    def set_zones(self, entry_zone, exit_zone, ow, oh):
        """设置 4 点矩形 (来自配置)，并据此重建标记线"""
        if not self.ow or not self.oh or ow <= 0 or oh <= 0: return
        sx, sy = self.ow/ow, self.oh/oh
        if entry_zone and len(entry_zone) == 4:
            self.entry_zone = [(int(x*sx),int(y*sy)) for x,y in entry_zone]
            self.marker_line = [self.entry_zone[0], self.entry_zone[1]]
        if exit_zone and len(exit_zone) == 4:
            self.exit_zone = [(int(x*sx),int(y*sy)) for x,y in exit_zone]
        self.update()
    def rebase(self, ow_old, oh_old, ow_new, oh_new):
        """把一个图像坐标空间里的图形整体换算到另一个空间（分辨率变化时用）。
        不做这一步的话，旧坐标会被当作新空间的坐标使用 → 多边形/入口带整体错位。"""
        if not (ow_old and oh_old and ow_new and oh_new): return
        sx, sy = ow_new/ow_old, oh_new/oh_old
        def f(p): return (int(round(p[0]*sx)), int(round(p[1]*sy)))
        self.polygons = [[f(pt) for pt in poly] for poly in self.polygons]
        if self.entry_zone:  self.entry_zone  = [f(pt) for pt in self.entry_zone]
        if self.exit_zone:   self.exit_zone   = [f(pt) for pt in self.exit_zone]
        if self.marker_line: self.marker_line = [f(pt) for pt in self.marker_line]
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
        self._dead_threads = []      # 仍在收尾的 RTSP 线程，保留引用避免被 GC 掉导致崩溃
        self._suppress_res = False   # 程序化改分辨率时抑制 rebase
        self.comm = Communicate()
        self.comm.update_log.connect(self._log)
        self.comm.set_board_ip.connect(self._set_ip)
        self.comm.probe_done.connect(self._probe_done)
        self.comm.upload_done.connect(self._upload_done)
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

        rb = QHBoxLayout()
        b = QPushButton("更新到设备"); b.clicked.connect(self._upload); rb.addWidget(b)
        b = QPushButton("停止设备程序"); b.clicked.connect(self._stop_remote); rb.addWidget(b)
        bl.addLayout(rb)

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
        # 改分辨率会改变绘制坐标系，必须同步换算已有图形，否则多边形/入口带会整体错位
        self.res_cb.currentTextChanged.connect(self._res_changed)
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
        pl.addWidget(QLabel("自动: 入口带 = 标记线向检测区内延伸100px（出口区域已删，检测区外即出口）"))
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
            # 子线程里绝不能直接调 self._log / setEnabled 这类操作控件的方法；
            # 一律通过信号回到 GUI 线程，否则会出现界面卡住、显示错乱甚至崩溃。
            for ni in get_network_prefixes():
                with ip_lock:
                    if found_ip: break
                self.comm.update_log.emit(f"扫描 {ni['prefix']}0/24")
                probe_network(ni['prefix'], self.comm.update_log)
            with ip_lock:
                if found_ip:
                    self.comm.set_board_ip.emit(found_ip)
                    self.comm.update_log.emit(f"完成: {found_ip}")
                else:
                    self.comm.update_log.emit("未找到设备")
            self.comm.probe_done.emit()
        threading.Thread(target=run, daemon=True).start()

    def _probe_done(self):
        self.probe_btn.setEnabled(True)

    def _sel_file(self):
        p, _ = QFileDialog.getOpenFileName(self, "选文件", "", "All(*.*)")
        if p: self.file_ed.setText(p)

    def _ask_pw(self):
        pw, ok = QInputDialog.getText(self,"密码","更新密码:", QLineEdit.Password)
        if not ok or pw != "topeet":
            QMessageBox.warning(self,"err","密码错"); return False
        return True

    def _upload(self):
        ip = self.board_ip_edit.text().strip()
        fp = self.file_ed.text().strip() or self.cfg_path or ""
        if not ip: QMessageBox.warning(self,"err","无设备IP"); return
        if not fp: QMessageBox.warning(self,"err","无文件"); return
        if not self._ask_pw(): return
        self._log(f"上传 {fp}")
        def run():
            ok2, err = scp_send(fp, ip, self.comm.update_log)
            # QMessageBox 只能在 GUI 线程弹；子线程直接弹会卡死/异常，所以走信号。
            self.comm.upload_done.emit(ok2, err or "")
        threading.Thread(target=run, daemon=True).start()

    def _upload_done(self, ok, err):
        if ok:
            self._log("上传并停止板端程序成功")
            QMessageBox.information(self, "ok",
                "上传成功，板端旧程序已停止。\n\n请到板子上用新配置重新启动。")
        else:
            self._log(f"失败: {err}")
            QMessageBox.warning(self, "err", err or "上传失败（详见日志）")

    def _stop_remote(self):
        """只停止板端程序（不传文件）：用于排查「杀了又自己起来」的自启动/守护脚本。"""
        ip = self.board_ip_edit.text().strip()
        if not ip: QMessageBox.warning(self,"err","无设备IP"); return
        if not self._ask_pw(): return
        def run():
            import paramiko
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            try:
                ssh.connect(ip, 22, SSH_USER, SSH_PASS, timeout=6, banner_timeout=5, auth_timeout=5)
                stop_remote_programs(ssh, self.comm.update_log)
            except Exception as e:
                self.comm.update_log.emit(f"连接失败: {e}")
            finally:
                try: ssh.close()
                except: pass
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
                    
                    # 1) 先解析检测区多边形（入口带定向需要检测区中心）
                    pstr = ' '.join(rest)
                    polys = []
                    start = 0
                    while True:
                        a = pstr.find('[', start)
                        if a == -1: break
                        b = pstr.find(']', a)
                        if b == -1: break
                        pts = []
                        for ps2 in pstr[a+1:b].split():
                            if ',' in ps2:
                                x, y = map(int, ps2.split(','))
                                pts.append((x, y))
                        if len(pts) >= 3: polys.append(pts)
                        start = b + 1

                    # 2) 再解析入口绊线 t1（5 参: t1,x1,y1,x2,y2,depth，depth 符号=入口带方向）
                    entry = None
                    for token in rest:
                        if token.startswith('t1,'):
                            parts = token[3:].replace(',', ' ').split()
                            if len(parts) < 4: continue
                            p1 = (int(parts[0]), int(parts[1]))
                            p2 = (int(parts[2]), int(parts[3]))
                            if len(parts) >= 5 and int(parts[4]) != 0:
                                depth = int(parts[4])
                            else:
                                # 旧格式 t1 只有 4 个参数（无深度）。若当成 depth=0，入口带会
                                # 退化成一条线，保存时还会把板端 R1 区域抹掉 —— 按检测区中心
                                # 方向补一个默认深度。
                                depth = _depth_toward_detection(polys, p1, p2)
                                self._log(f"t1 无深度，已按检测区方向补 {depth:+d}px")
                            p3, p4 = _rect_from_line_depth(p1, p2, depth)
                            entry = [p1, p2, p3, p4]
                        elif token.startswith('t2,'):
                            # 出口绊线已废弃：检测区域之外即出口，解析但忽略
                            self._log(f"忽略已废弃的出口绊线 t2: {token}")

                    self.cams.append({'ip':ip,'username':u,'password':pwd,'channel':ch,
                                      'resolution':res,'polygons':polys,
                                      'entry_zone':entry,'exit_zone':None})
        except Exception as e: QMessageBox.warning(self,"err",str(e)); return
        self.cam_lst.clear()
        for c in self.cams: self.cam_lst.addItem(f"{c['ip']}(Ch{c['channel']})")
        if self.cams: self.cam_lst.setCurrentRow(0); self._sel_cam(self.cam_lst.item(0))

    def _sel_cam(self, item):
        i=self.cam_lst.row(item)
        if i<0: return
        # 重复点同一个摄像头时不要清画布：否则刚画好、还没保存的区域/标记线会被悄悄清掉
        keep = (i == self.cam_i) and (self.drawer.polygons or self.drawer.marker_line)
        self.cam_i=i; c=self.cams[i]
        self.ip_ed.setText(c['ip']); self.user_ed.setText(c['username'])
        self.pass_ed.setText(c['password']); self.ch_ed.setText(c['channel'])
        self._suppress_res = True          # 程序化改分辨率：不要触发坐标换算
        self.res_cb.setCurrentText(c['resolution'])
        self._suppress_res = False
        if not keep:
            self.drawer.clear()
        self._apply_saved_geometry()

    def _apply_saved_geometry(self):
        """把当前配置里的多边形/入口带加载进画布。
        画布已有内容时不覆盖 —— 否则用户刚画好的区域/标记线会被悄悄清掉。"""
        if not (0 <= self.cam_i < len(self.cams)): return
        d = self.drawer
        if d.frame is None or d.polygons or d.marker_line: return
        c = self.cams[self.cam_i]
        try: w, h = map(int, c['resolution'].split('*'))
        except Exception: return
        d.set_polygons(c['polygons'], w, h)
        d.set_zones(c.get('entry_zone'), c.get('exit_zone'), w, h)

    def _res_changed(self, text):
        """分辨率下拉改变 → 绘制坐标空间随之改变，把已有图形按比例换算过去，
        否则旧坐标被当成新空间坐标 → 多边形/入口带整体错位（显示 bug）。"""
        if self._suppress_res: return
        try: nw, nh = map(int, text.split('*'))
        except Exception: return
        ow, oh = self.drawer.ow, self.drawer.oh
        if not (ow and oh): return
        if (ow, oh) != (nw, nh):
            self.drawer.rebase(ow, oh, nw, nh)
        if 0 <= self.cam_i < len(self.cams):
            self.cams[self.cam_i]['resolution'] = text

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
        # 摄像头连接后加载一次当前配置中的多边形和绊线（画布已有内容则不覆盖）
        self._apply_saved_geometry()

    def _cam_fail(self, msg): self._log(f"连接失败:{msg}")

    def _stop_cam(self):
        if self.timer.isActive(): self.timer.stop()
        if self.cap: self.cap.release(); self.cap=None
        if self.cam_thread and self.cam_thread.isRunning():
            self.cam_thread.quit()
            if not self.cam_thread.wait(3000):
                # 线程还卡在 rtsp 打开里。QThread 不能强杀，保留引用让它在后台自然结束；
                # 被 GC 回收一个仍在运行的 QThread 会直接崩溃。
                self._dead_threads.append(self.cam_thread)
        self.cam_thread = None

    def _img(self):
        p,_=QFileDialog.getOpenFileName(self,"图片","","Images(*.png *.jpg *.bmp)")
        if not p: return
        f=cv2.imread(p)
        if f is not None:
            self.drawer.set_frame(f)
            self._apply_saved_geometry()
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
                # 入口绊线 4 点(入口带) -> 换算成 5 参格式 (绊线 + 带符号深度)
                # 出口区域已删除：检测区域之外即出口，不再存储 t2。
                ez = c.get('entry_zone')
                if ez and len(ez) == 4:
                    line, depth = _rect_to_line_depth(ez)
                    if line and abs(depth) >= 1:
                        lp.append(f"t1,{line[0][0]},{line[0][1]},{line[1][0]},{line[1][1]},{depth}")
                    elif line:
                        # depth≈0 表示入口带退化成一条线。写出去等于告诉板端"没有入口带"，
                        # 会把 R1 区域悄悄删掉，所以宁可跳过并提示。
                        self._log("警告: 入口带深度≈0，已跳过 t1（避免板端 R1 区域消失）")
                f.write(' '.join(lp)+'\n')
        QMessageBox.information(self,"ok",f"已保存: {p}"); self.cfg_path=p

    def closeEvent(self, e):
        self._stop_cam(); e.accept()


if __name__=="__main__":
    app=QApplication(sys.argv)
    w=MainWindow(); w.show()
    sys.exit(app.exec_())
