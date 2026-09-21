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
import os, sys, re, cv2, time, socket, threading, subprocess, ipaddress, math
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


# ==================== 判定线(区域边缘折线) -> 区域 换算（与 C++ 端严格互逆）====================
# ⚠️ 三条约定必须与 C++ 端(gate_id_counter.cpp 的 evalLine / inwardNormal / zonePolygon)
#    完全一致，否则 Python 画出来的区域和板端算出来的会分居判定线两侧：
#   1) 判定线 = **区域边缘折线**（点数不限，区域边缘不一定是直线）。
#      它本身就**是判定线**：脚底跨过它 = 一次计数（"跨过本相机的线"这一动作本身）。
#   2) 区域的【形状】= 判定线 ∪ 其整体沿 x 平移 |平移量| px（y 不变）的副本，首尾相接成闭环。
#      为什么不做"沿法向偏移的等距带"：边缘是斜线/折线时法向带也是斜的，与相邻区域拼不
#      严实，走廊里会留缝；沿 x 平移后相邻区域共享同一条 x 方向边界，中间无缝。
#   3) 平移量的【符号】= 内侧法向（右手系 (dy,-dx)/L 按 depth 符号定向）的 x 分量所指的那
#      一侧（朝检测区），保证区域落在检测区一侧。
def _right_hand_normal(p1, p2):
    """右手系法向 (dy,-dx)/L —— 与 C++ inwardNormal 的基础约定一致。"""
    dx = p2[1] - p1[1]
    dy = -(p2[0] - p1[0])
    L = (dx*dx + dy*dy) ** 0.5
    if L < 1:
        return 0.0, 0.0
    return dx / L, dy / L


def _poly_mid(pts):
    """折线弧长中点 + 所属线段下标 —— 与 C++ midSegment()/lineMidPoint() 逐行对应。
    C++ 用弧长中点所在的那一段的法向做"内侧方向"探测，这里必须完全一致。"""
    n = len(pts) - 1
    if n < 1:
        return 0, (float(pts[0][0]), float(pts[0][1]))
    lens = [math.hypot(pts[i+1][0]-pts[i][0], pts[i+1][1]-pts[i][1]) for i in range(n)]
    half = sum(lens) * 0.5
    acc = 0.0
    for i, L in enumerate(lens):
        if L < 1e-3:
            continue
        if acc + L >= half:
            t = (half - acc) / L
            return i, (pts[i][0] + (pts[i+1][0]-pts[i][0])*t,
                       pts[i][1] + (pts[i+1][1]-pts[i][1])*t)
        acc += L
    return n - 1, (float(pts[-1][0]), float(pts[-1][1]))


def _polyline_len(pts):
    return sum(math.hypot(pts[i+1][0]-pts[i][0], pts[i+1][1]-pts[i][1])
               for i in range(len(pts)-1))


def _largest_polygon(polys):
    """取顶点最多的那条多边形（C++ 端同样挑点数最多的轮廓做探测）。返回 (N,1,2) float32。"""
    best, best_n = None, 0
    for poly in (polys or []):
        if len(poly) > best_n:
            best, best_n = poly, len(poly)
    if best is None or len(best) < 3:
        return None
    return np.array(best, dtype=np.float32).reshape(-1, 1, 2)


def _resolve_depth_sign(polys, pts, depth):
    """把平移量的符号校正为"指向检测区侧"。

    算法与 C++ GateIdCounter 构造函数**逐行对应**：探测点 = 折线弧长中点 ± 该段右手系法向
    * max(30,|depth|)，比较两点在检测区多边形内的深度，取更深的那一侧。
    不这么做的话，配置里方向画反时 Python 的预览与板端会分居判定线两侧。
    """
    seg, mid = _poly_mid(pts)
    ux, uy = _right_hand_normal(pts[seg], pts[seg + 1])
    if ux == 0.0 and uy == 0.0:
        return depth
    big = _largest_polygon(polys)
    if big is None:
        # 没有检测区多边形就无法判定内侧方向：沿用原值；|depth|<1 时按默认。
        return depth if abs(depth) >= 1 else -ENTRY_MARKER_DEPTH
    s = max(30.0, abs(depth))
    d_pos = cv2.pointPolygonTest(big, (mid[0] + ux * s, mid[1] + uy * s), True)
    d_neg = cv2.pointPolygonTest(big, (mid[0] - ux * s, mid[1] - uy * s), True)
    # 内侧方向 = "mid + 法向"那一侧在检测区内更深就取 +，否则取 −；幅值沿用配置。
    # ⚠️ 必须"先定方向、再乘幅值"，不能写成"d_neg>d_pos 就取反"：那样在配置里已经写成
    #    负值时会把方向又翻回去（正负来回错），区域整体反相（由"导出->再导入"幂等自检抓出）。
    side = 1.0 if d_pos >= d_neg else -1.0
    if abs(depth) < 1:
        return 100.0 * side
    return abs(depth) * side


def _x_offset_from_depth(pts, depth):
    """带符号 depth -> x 方向平移量（y 不变，符号即左右）。
    规则：平移方向 = 内侧法向的 x 分量所指的那一侧（与 C++ zoneOffsetX() 一致）。
    例：depth=+100 且弧长中点那段的内侧法向 x 分量为负 → 返回 -100（向左平移）。
    """
    seg, _ = _poly_mid(pts)
    ux, _ = _right_hand_normal(pts[seg], pts[seg + 1])
    sign = 1.0 if depth >= 0 else -1.0
    s = abs(depth)
    return s if ux * sign >= 0 else -s


def _zone_ring(pts, depth):
    """判定线折线 + 其沿 x 平移副本 -> 闭环（区域轮廓，图像坐标）。
    顺序 = [p1..pN, pN+off .. p1+off]，与 C++ zonePolygon(offset) 完全一致。
    ⚠️ 闭环只用于**显示**；计数判据是脚底到判定线的法向投影，与闭环无关。"""
    sx = _x_offset_from_depth(pts, depth)
    shifted = [(x + sx, y) for x, y in pts]
    return list(pts) + shifted[::-1]


# ==================== 检测区边界吸附（判定线 = 区域边缘的一段）====================
# 用户只点 2 个点，剩下的交给这两步：吸附到多边形边界，再取两点之间较短的那段弧。
def _snap_to_ring(ring, pt):
    """把点吸附到闭合折线上最近的位置。
    返回 (边下标, 边内参数 t, 吸附点, 距离)。ring 不重复首点，约定 ring[-1]->ring[0] 也算一条边。"""
    n = len(ring)
    if n < 2:
        return None
    best = None
    for i in range(n):
        ax, ay = ring[i]
        bx, by = ring[(i + 1) % n]
        dx, dy = bx - ax, by - ay
        L2 = dx*dx + dy*dy
        if L2 < 1e-6:
            continue
        t = ((pt[0]-ax)*dx + (pt[1]-ay)*dy) / L2
        t = max(0.0, min(1.0, t))
        px, py = ax + dx*t, ay + dy*t
        d = math.hypot(pt[0]-px, pt[1]-py)
        if best is None or d < best[3]:
            best = (i, t, (int(round(px)), int(round(py))), d)
    return best


def _ring_arc(ring, sa, sb):
    """闭合折线上 a->b 之间**较短**的那段，返回点列 [a, 中间顶点..., b]。
    取短的那段是刻意的：闸口的区域边缘是"一条边界"，两点之间当然走近路。"""
    n = len(ring)
    if n < 3:
        return [sa[2], sb[2]]
    ea, ta = sa[0], sa[1]
    eb, tb = sb[0], sb[1]
    lens = [math.hypot(ring[(i+1) % n][0]-ring[i][0], ring[(i+1) % n][1]-ring[i][1])
            for i in range(n)]
    total = sum(lens)
    cum = [0.0]
    for L in lens:
        cum.append(cum[-1] + L)
    # 正向顶点：ea+1, ea+2, ... 直到 eb
    fwd_idx, k = [], (ea + 1) % n
    for _ in range(n + 1):
        fwd_idx.append(k)
        if k == eb:
            break
        k = (k + 1) % n
    around = [sa[2]] + [ring[i] for i in fwd_idx] + [sb[2]]
    if ea == eb:
        # 两点落在同一条边上：较短的就是这条边自己的子段，另一段要绕整圈回来。
        direct = [sa[2], sb[2]]
        return direct if _polyline_len(direct) <= _polyline_len(around) else around
    # 反向顶点：ea, ea-1, ... 直到 eb
    bwd_idx, k = [], ea
    for _ in range(n + 1):
        bwd_idx.append(k)
        if k == eb:
            break
        k = (k - 1) % n
    back = [sa[2]] + [ring[i] for i in bwd_idx] + [sb[2]]
    return around if _polyline_len(around) <= _polyline_len(back) else back


# ==================== 多边形绘制 ====================
# 区域沿 x 的默认平移量（px）：把判定线的所有顶点整体沿 x 平移这么多像素（y 不变）。
# 与 C++ 端 zin,/zout, 未写 zo, 时的默认值一致。
ENTRY_MARKER_DEPTH = 100


class PolygonDrawer(QWidget):
    MODE_POLYGON    = 0
    MODE_ZONE_ENTRY = 1   # 画入口区域: 在检测区边界点 2 点 -> 自动取中间那段边缘 = 判定线
    MODE_ZONE_EXIT  = 2   # 画出口区域: 同上（由用户决定哪台相机画入口/哪台画出口）

    def __init__(self):
        super().__init__()
        self.current = []         # 绘制中: [p1_disp, p2_disp]
        self.polygons = []        # 检测区多边形（图像坐标，不重复首点）
        # ---- 判定线（= 区域边缘折线）+ 区域 ----
        # zone_line  = 判定线折线（图像坐标，点数不限）；脚底跨过它 = 一次计数
        # zone_role  = 'entry' / 'exit' / None(旧 t1 格式，角色交给板端按 IP 判定)
        # zone_depth = 带符号的 x 平移量；符号 = 朝检测区那一侧
        self.zone_line = None
        self.zone_role = None
        self.zone_depth = 0
        self.mode = self.MODE_POLYGON
        self.drawing = False
        self.frame = None
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ow = self.oh = 0
        self.scale = 1.0; self.ox = self.oy = 0; self.dw = self.dh = 0

    def set_mode(self, mode): self.mode = mode; self.drawing = False; self.current = []

    def _zone_role_of_mode(self):
        if self.mode == self.MODE_ZONE_ENTRY: return 'entry'
        if self.mode == self.MODE_ZONE_EXIT:  return 'exit'
        return None

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

    # ---- 标记线 -> 入口带（2 点整体沿 x 平移）----
    def _pick_ring(self, a, b):
        """在已有检测区多边形里挑一条"最贴近 a、b 两点"的闭合环（多区域配置时用）。"""
        best, best_d = None, None
        for poly in self.polygons:
            if len(poly) < 2:
                continue
            sa = _snap_to_ring(poly, a); sb = _snap_to_ring(poly, b)
            if sa is None or sb is None:
                continue
            d = sa[3] + sb[3]
            if best_d is None or d < best_d:
                best, best_d = poly, d
        return best

    def _edge_arc(self, a, b):
        """在检测区多边形的边界上取 a->b 之间**较短**的那一段折线（= 判定线）。

        用户只在边界上点 2 个点，这里把两点吸附到边界、再自动取中间那段 ——
        这样判定线天然就是"区域的那条边缘"，边缘是折的/弯的都能跟着走，
        不像以前那样只能得到一条直线。没有多边形可吸附时退化成两点直线。
        """
        ring = self._pick_ring(a, b)
        if ring is None:
            print("[提示] 还没有检测区多边形可吸附：判定线退化为这 2 点连成的直线。")
            return [list(a), list(b)]
        sa = _snap_to_ring(ring, a); sb = _snap_to_ring(ring, b)
        if sa is None or sb is None:
            return [list(a), list(b)]
        arc = _ring_arc(ring, sa, sb)
        # 两点几乎重合（点在边界上没挪动）→ 退化的判定线，没有任何意义
        if _polyline_len(arc) < 10.0:
            print("[提示] 这 2 点吸附到边界后几乎重合：判定线太短，已忽略这次点击。")
            return None
        return arc

    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton: return
        if self.frame is None or self.dw <= 0 or self.dh <= 0: return   # 没有画面就没有坐标系，别画
        p = self._clamp(e.pos())
        if self.mode in (self.MODE_ZONE_ENTRY, self.MODE_ZONE_EXIT):
            # 判定线: 在检测区边界上点 2 点
            if len(self.current) == 0:
                self.current = [p]
                self.drawing = True
            elif len(self.current) == 1:
                self.current.append(p)
                self.drawing = False
                self._finish_zone()
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
        if self.mode in (self.MODE_ZONE_ENTRY, self.MODE_ZONE_EXIT):
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

    def _finish_zone(self):
        """边界上 2 点已选：吸附出中间那段边缘 = 判定线，再据此算出区域。"""
        if len(self.current) < 2: return
        a = self._to(self.current[0]); b = self._to(self.current[1])
        role = self._zone_role_of_mode()
        if self.zone_line and self.zone_role and self.zone_role != role:
            print(f"[提示] 本相机原有「{'入口' if self.zone_role=='entry' else '出口'}区域」，"
                  f"已被新画的「{'入口' if role=='entry' else '出口'}区域」替换"
                  f"（一台相机只有一条判定线）。")
        arc = self._edge_arc(a, b)
        self.current = []; self.drawing = False
        if arc is None:
            self.update(); return
        self.zone_line = [[int(x), int(y)] for x, y in arc]
        self.zone_role = role
        # 平移方向朝检测区（算法与 C++ 构造器一致）；反向时这里会纠正回来
        self.zone_depth = _resolve_depth_sign(self.polygons, self.zone_line, ENTRY_MARKER_DEPTH)
        seg, _ = _poly_mid(self.zone_line)
        ux, _ = _right_hand_normal(self.zone_line[seg], self.zone_line[seg + 1])
        if abs(ux) < 0.25:
            print(f"[警告] 判定线的中间线段接近水平(内侧法向 x 分量={ux:+.2f})：该处区域带宽"
                  f"仅≈{abs(self.zone_depth*ux):.0f}px(趋于 0)。跨线判定不受影响；"
                  f"若希望区域看上去更宽，可把判定线画在接近竖直的那段边缘上。")
        print(f"[ok] {'入口' if role=='entry' else '出口'}判定线: "
              f"{len(self.zone_line)} 个顶点, 沿 x 平移 {self.zone_depth:+.0f}px")
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

        # ---- 区域（判定线 ∪ 其沿 x 平移副本）+ 判定线 ----
        # 区域形状 = 判定线折线 + 它的 x 平移副本首尾相接成闭环；入口绿 / 出口红。
        # ⚠️ 闭环只用于显示；板端计数判据是脚底到判定线的法向投影，与闭环无关。
        if self.zone_line and len(self.zone_line) >= 2:
            is_entry = (self.zone_role != 'exit')
            fill = QColor(0,255,0,50) if is_entry else QColor(0,0,255,50)
            col  = QColor(0,160,0)    if is_entry else QColor(0,0,160)
            label = "入口区域" if is_entry else "出口区域"
            depth = self.zone_depth if abs(self.zone_depth) >= 1 else ENTRY_MARKER_DEPTH
            ring = _zone_ring(self.zone_line, depth)
            pts  = [self._td(x, y) for x, y in ring]
            path = QPainterPath(); path.moveTo(pts[0])
            for pp in pts[1:]: path.lineTo(pp)
            path.closeSubpath()
            p.fillPath(path, fill)
            p.setPen(QPen(col, 2)); p.setBrush(Qt.NoBrush)
            p.drawPolygon(QPolygon(pts))
            # 判定线（黄，高亮）—— 跨过它 = 一次计数
            lpts = [self._td(x, y) for x, y in self.zone_line]
            p.setPen(QPen(QColor(255,255,0), 3))
            p.drawPolyline(QPolygon(lpts))
            # 折点高亮（首尾两点尤其重要：判定线的跨度端）
            p.setBrush(QColor(255,255,255)); p.setPen(QPen(col, 2))
            for pp in lpts: p.drawEllipse(pp, 4, 4)
            p.drawEllipse(lpts[0], 6, 6); p.drawEllipse(lpts[-1], 6, 6)
            # 标签 + 平移量
            cx = sum(x for x, y in ring) // len(ring)
            cy = sum(y for x, y in ring) // len(ring)
            tp = self._td(cx, cy)
            p.setPen(QPen(col, 1))
            p.drawText(tp.x()-24, tp.y()+5, label)
            p.drawText(tp.x()-24, tp.y()+22, f"x{self.zone_depth:+.0f}px")

        # 画判定线过程中: 显示预览 (1 个点已放, 提示再点一次)
        if self.drawing and self.mode in (self.MODE_ZONE_ENTRY, self.MODE_ZONE_EXIT) and self.current:
            p1d = self.current[0]
            p.setBrush(QColor(255,255,255)); p.setPen(QPen(QColor(255,255,0), 2))
            p.drawEllipse(p1d, 5, 5)
            p.drawText(p1d.x()+10, p1d.y()-10,
                       "边界起点已放, 再点一次 = 边界终点(自动取中间那段边缘)")

        if len(self.current) > 1 and self.mode == self.MODE_POLYGON:
                p.setPen(QPen(QColor(0,255,0), 2)); p.drawPolyline(QPolygon(self.current))
                if len(self.current) > 2: p.drawLine(self.current[-1], self.current[0])

    def clear(self):
        self.polygons = []
        self.zone_line = None; self.zone_role = None; self.zone_depth = 0
        self.update()
    def delete_last(self):
        if self.polygons: self.polygons.pop(); self.update()
    def set_polygons(self, ps, ow, oh):
        # 画面还没加载时 self.ow/oh 为 0，无法把配置坐标换算到画面坐标；
        # 强行算会得到 scale=0 把所有点压到 (0,0)。交给后面加载画面时再应用。
        if not self.ow or not self.oh or ow <= 0 or oh <= 0: return
        sx, sy = self.ow/ow, self.oh/oh
        self.polygons = [[(int(x*sx),int(y*sy)) for x,y in p] for p in ps]; self.update()
    def set_zone(self, line, role, depth, ow, oh):
        """设置判定线折线 (来自配置)，点数不限。"""
        if not self.ow or not self.oh or ow <= 0 or oh <= 0: return
        if not line or len(line) < 2: return
        sx, sy = self.ow/ow, self.oh/oh
        self.zone_line = [(int(x*sx),int(y*sy)) for x,y in line]
        self.zone_role = role
        self.zone_depth = depth
        self.update()
    def rebase(self, ow_old, oh_old, ow_new, oh_new):
        """把一个图像坐标空间里的图形整体换算到另一个空间（分辨率变化时用）。
        不做这一步的话，旧坐标会被当作新空间的坐标使用 → 多边形/区域整体错位。"""
        if not (ow_old and oh_old and ow_new and oh_new): return
        sx, sy = ow_new/ow_old, oh_new/oh_old
        def f(p): return (int(round(p[0]*sx)), int(round(p[1]*sy)))
        self.polygons = [[f(pt) for pt in poly] for poly in self.polygons]
        if self.zone_line: self.zone_line = [f(pt) for pt in self.zone_line]
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
        pl.addLayout(r)
        r = QHBoxLayout()
        b = QPushButton("画入口区域"); b.setToolTip("在检测区边界上点 2 点，自动取中间那段边缘作为判定线")
        b.clicked.connect(lambda:self.drawer.set_mode(PolygonDrawer.MODE_ZONE_ENTRY)); r.addWidget(b)
        b = QPushButton("画出口区域"); b.setToolTip("同上；由你决定哪台相机画入口、哪台画出口")
        b.clicked.connect(lambda:self.drawer.set_mode(PolygonDrawer.MODE_ZONE_EXIT)); r.addWidget(b)
        pl.addLayout(r)
        
        r = QHBoxLayout()
        b = QPushButton("删最后"); b.clicked.connect(lambda:self.drawer.delete_last()); r.addWidget(b)
        b = QPushButton("清空全部"); b.clicked.connect(lambda:self.drawer.clear()); r.addWidget(b)
        pl.addLayout(r)

        pl.addWidget(QLabel("步骤: 1)先画检测区域(多边形)  2)再画入口区域或出口区域"))
        pl.addWidget(QLabel("在检测区边界上点 2 点 → 自动取中间那段边缘 = 判定线(折线)"))
        pl.addWidget(QLabel("区域 = 判定线整体沿 x 平移 100px(y不变), 朝检测区侧; 跨过判定线 = 计数"))
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

                    # 2) 再解析判定线：新格式 zin,/zout,（区域边缘折线，点数不限）
                    #    或旧格式 t1（2 点直线），可选 zo, 单独指定区域沿 x 的平移量。
                    zone_line = None; zone_role = None
                    zone_depth_raw = None      # 文件里显式写的平移量（t1 的 depth 或 zo,）
                    for token in rest:
                        if token.startswith('zin,') or token.startswith('zout,'):
                            role = 'entry' if token.startswith('zin,') else 'exit'
                            parts = token[4:].replace(',', ' ').split()
                            pts = []
                            for k in range(0, len(parts) - 1, 2):
                                try:
                                    pts.append((int(float(parts[k])), int(float(parts[k+1]))))
                                except ValueError:
                                    pass
                            if len(pts) >= 2:
                                zone_line, zone_role = pts, role
                            else:
                                self._log(f"判定线点数不足(需要成对的顶点≥2点)，已忽略: {token}")
                        elif token.startswith('t1,'):
                            parts = token[3:].replace(',', ' ').split()
                            if len(parts) < 4: continue
                            p1 = (int(float(parts[0])), int(float(parts[1])))
                            p2 = (int(float(parts[2])), int(float(parts[3])))
                            zone_line = [p1, p2]
                            zone_role = None      # 旧格式不指定角色 → 角色交给板端按 IP 判定
                            if len(parts) >= 5:
                                zone_depth_raw = int(float(parts[4]))
                        elif token.startswith('zo,'):
                            try: zone_depth_raw = int(float(token[3:]))
                            except ValueError: pass
                        elif token.startswith('t2,'):
                            # 出口绊线已废弃：检测区域之外即出口，解析但忽略
                            self._log(f"忽略已废弃的出口绊线 t2: {token}")

                    # 平移量的符号校正为"朝检测区侧"（算法与 C++ 构造器一致）。
                    # 没有显式值时：新格式按默认 100，旧 t1 按 100 推断（板端也会这么做）。
                    zone_depth = 0
                    if zone_line:
                        base = zone_depth_raw if zone_depth_raw is not None else ENTRY_MARKER_DEPTH
                        zone_depth = _resolve_depth_sign(polys, zone_line, base)
                        if zone_depth_raw is None:
                            self._log(f"{'zin/zout' if zone_role else 't1'} 未给平移量，"
                                      f"已按检测区方向推断为 {zone_depth:+.0f}px")
                        elif abs(zone_depth - zone_depth_raw) >= 1:
                            self._log(f"区域平移量 {zone_depth_raw:+d} 与检测区方向相反，"
                                      f"已校正为 {zone_depth:+.0f}px")
                    self.cams.append({'ip':ip,'username':u,'password':pwd,'channel':ch,
                                      'resolution':res,'polygons':polys,
                                      'zone_line':zone_line,'zone_role':zone_role,
                                      'zone_depth':zone_depth,
                                      'zone_depth_raw':zone_depth_raw})
        except Exception as e: QMessageBox.warning(self,"err",str(e)); return
        self.cam_lst.clear()
        for c in self.cams: self.cam_lst.addItem(f"{c['ip']}(Ch{c['channel']})")
        if self.cams: self.cam_lst.setCurrentRow(0); self._sel_cam(self.cam_lst.item(0))

    def _sel_cam(self, item):
        i=self.cam_lst.row(item)
        if i<0: return
        # 重复点同一个摄像头时不要清画布：否则刚画好、还没保存的区域/判定线会被悄悄清掉
        keep = (i == self.cam_i) and (self.drawer.polygons or self.drawer.zone_line)
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
        """把当前配置里的多边形/判定线加载进画布。
        画布已有内容时不覆盖 —— 否则用户刚画好的区域/判定线会被悄悄清掉。"""
        if not (0 <= self.cam_i < len(self.cams)): return
        d = self.drawer
        if d.frame is None or d.polygons or d.zone_line: return
        c = self.cams[self.cam_i]
        try: w, h = map(int, c['resolution'].split('*'))
        except Exception: return
        d.set_polygons(c['polygons'], w, h)
        d.set_zone(c.get('zone_line'), c.get('zone_role'),
                   c.get('zone_depth') or 0, w, h)

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
        c['zone_line']=self.drawer.zone_line
        c['zone_role']=self.drawer.zone_role
        c['zone_depth']=self.drawer.zone_depth
        # 本相机新画了区域 → 平移量由画布决定；沿用原格式（t1 保持 t1，zin/zout 保持）
        if self.drawer.zone_line:
            c['zone_depth_raw'] = None if self.drawer.zone_role else self.drawer.zone_depth
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
                # 判定线（区域边缘折线）+ 区域沿 x 的平移量。
                #   新格式：zin,x1,y1,...,xN,yN  zo,<px>   —— 入口/出口由用户选择
                #   旧格式：t1,x1,y1,x2,y2,<depth>          —— 2 点直线(向后兼容)
                # 两种格式的 area 形状规则相同：判定线整体沿 x 平移 |平移量| px。
                line = c.get('zone_line')
                role = c.get('zone_role')
                depth = c.get('zone_depth') or 0
                if line and len(line) >= 2:
                    if role in ('entry', 'exit'):
                        token = 'zin' if role == 'entry' else 'zout'
                        lp.append(token + ',' + ','.join(f"{x},{y}" for x, y in line))
                        lp.append(f"zo,{int(round(depth))}")
                    else:
                        # 旧格式只存得下 2 点；点数更多的判定线自动升级成 zin/zout
                        if len(line) == 2:
                            raw = c.get('zone_depth_raw')
                            if raw is None: raw = int(round(depth))
                            lp.append(f"t1,{line[0][0]},{line[0][1]},"
                                      f"{line[1][0]},{line[1][1]},{int(raw)}")
                        else:
                            lp.append('zin,' + ','.join(f"{x},{y}" for x, y in line))
                            lp.append(f"zo,{int(round(depth))}")
                            self._log(f"警告: {c['ip']} 的判定线有 {len(line)} 个顶点，"
                                      f"旧格式 t1 存不下 → 已改写为 zin（请确认该相机是入口相机）")
                f.write(' '.join(lp)+'\n')
        QMessageBox.information(self,"ok",f"已保存: {p}"); self.cfg_path=p

    def closeEvent(self, e):
        self._stop_cam(); e.accept()


if __name__=="__main__":
    app=QApplication(sys.argv)
    w=MainWindow(); w.show()
    sys.exit(app.exec_())
