# -*- coding: utf-8 -*-
"""判定线几何诊断（离线辅助脚本，不属于正式测试套件）。

回答现场最常问的三个问题，口径与 C++ GateIdCounter 完全一致：
  1) 这个脚底点离本相机判定线多远？带符号（>0 = 线内侧/检测区侧）
  2) 脚底投影在判定线跨度内吗？（跨度外的事件不参与跨线判定）
  3) 该点属于哪个区域？0=区域外 1=入口带 2=检测区内

用法:
  python _diag_line_dist.py [cameras_config.txt] [ip] [x,y x,y ...]
  python _diag_line_dist.py            # 默认真配置 + 200 相机的几个现场点

为什么单独写一个：--trace 只能看"当前活跃 ID"，人一旦跟丢就看不到任何东西了；
而"这一趟到底该不该计"往往正是在跟丢那几帧决定的，必须在离线把几何摆出来。
"""
import importlib.util
import os
import sys

import cv2
import numpy as np

CROSS_LINE_DEPTH = 20.0     # 与 gate_id_counter.cpp 的 kCrossLineDepth 一致
SPAN_MARGIN = 50.0          # kLineSpanMargin
TOL = 10.0                  # kRegionGapTolerance
DEFAULT_DEPTH = 100.0


def load_py(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_root(start):
    root = start
    while not os.path.exists(os.path.join(root, "cameras_config.txt")):
        parent = os.path.dirname(root)
        if parent == root:
            raise SystemExit("cameras_config.txt not found")
        root = parent
    return root


def right_hand_normal(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = (dx * dx + dy * dy) ** 0.5
    if L < 1e-6:
        return (0.0, 0.0), 0.0
    return (dy / L, -dx / L), L


def arclen_mid(pts):
    """折线弧长中点所在段（= C++ midSegment + lineMidPoint 的口径）。"""
    total = 0.0
    seglen = []
    for i in range(len(pts) - 1):
        _, L = right_hand_normal(pts[i], pts[i + 1])
        seglen.append(L)
        total += L
    half = total * 0.5
    acc = 0.0
    for i, L in enumerate(seglen):
        if L < 1e-3:
            continue
        if acc + L >= half:
            t = (half - acc) / L
            a, b = pts[i], pts[i + 1]
            return i, (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
        acc += L
    return len(pts) - 2, pts[-1]


def inward_sign(polys, pts, depth):
    """复刻 C++ 构造函数的方向自校正：内侧法向 = 右手系法向 × sign。
    判据 = 弧长中点 ± 法向 * max(30,|depth|) 两点在检测区内谁更深。"""
    if len(pts) < 2:
        return 1.0
    seg, mid = arclen_mid(pts)
    n, _ = right_hand_normal(pts[seg], pts[seg + 1])
    if n[0] == 0.0 and n[1] == 0.0:
        return 1.0
    if not polys:
        return 1.0 if depth >= 0 else -1.0
    big = max(polys, key=len)
    arr = np.array(big, dtype=np.float32).reshape(-1, 1, 2)
    s = max(30.0, abs(depth))
    d_pos = cv2.pointPolygonTest(arr, (mid[0] + n[0] * s, mid[1] + n[1] * s), True)
    d_neg = cv2.pointPolygonTest(arr, (mid[0] - n[0] * s, mid[1] - n[1] * s), True)
    return 1.0 if d_pos >= d_neg else -1.0


def eval_line(pts, sign, pt):
    """复刻 C++ evalLine()。返回 dict。"""
    nseg = len(pts) - 1
    best = dict(seg=-1, best_dist=1e18, t=0.0, L=1.0, true_dist=1e18)
    for i in range(nseg):
        ax, ay = pts[i]
        dx, dy = pts[i + 1][0] - ax, pts[i + 1][1] - ay
        L2 = dx * dx + dy * dy
        if L2 < 1e-6:
            continue
        t = ((pt[0] - ax) * dx + (pt[1] - ay) * dy) / L2
        tc = min(1.0, max(0.0, t))
        cx, cy = ax + dx * tc, ay + dy * tc
        dist = ((pt[0] - cx) ** 2 + (pt[1] - cy) ** 2) ** 0.5
        if dist < best["best_dist"]:
            best = dict(seg=i, best_dist=dist, t=t, L=L2 ** 0.5)
    if best["seg"] < 0:
        return None
    i = best["seg"]
    nn, _ = right_hand_normal(pts[i], pts[i + 1])
    nx, ny = nn[0] * sign, nn[1] * sign
    a = pts[i]
    signed = (pt[0] - a[0]) * nx + (pt[1] - a[1]) * ny
    t_px = best["t"] * best["L"]
    is_first = (i == 0)
    is_last = (i == nseg - 1)
    in_span = ((not is_first) and (not is_last)) or \
              (-SPAN_MARGIN <= t_px <= best["L"] + SPAN_MARGIN)
    best.update(signed=signed, t_px=t_px, in_span=in_span,
                a=pts[i], b=pts[i + 1], n=(nx, ny),
                is_first=is_first, is_last=is_last)
    return best


def region_of(cam, ring, pt):
    if ring is not None and cv2.pointPolygonTest(ring, pt, True) >= -TOL:
        return 1
    for poly in cam["polys"]:
        arr = np.array(poly, dtype=np.float32).reshape(-1, 1, 2)
        if cv2.pointPolygonTest(arr, pt, True) >= -TOL:
            return 2
    return 0


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = find_root(here)
    chk = load_py(os.path.join(here, "_chk_zone_py.py"), "chkzone")
    tool = load_py(os.path.join(root, "闸门计数-网络摄像头配置文件绘制.py"), "gatedraw")

    cfg = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "cameras_config.txt")
    want_ip = sys.argv[2] if len(sys.argv) > 2 else "200"
    queries = []
    for tok in sys.argv[3:]:
        x, y = tok.split(",")
        queries.append((float(x), float(y)))

    cams = chk.parse_config(cfg)
    for cam in cams:
        if want_ip not in cam["ip"]:
            continue
        pts = cam["line"]
        depth = tool._resolve_depth_sign(cam["polys"], pts, cam["depth"])
        sign = 1.0 if depth >= 0 else -1.0
        ring = np.array(tool._zone_ring(pts, depth), dtype=np.float32).reshape(-1, 1, 2)
        seg, mid = arclen_mid(pts)
        n, _ = right_hand_normal(pts[seg], pts[seg + 1])
        nx, ny = n[0] * sign, n[1] * sign
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        print(f"== {cam['ip']}  判定线 {len(pts)} 点  区域沿 x 平移 {depth:+.0f}px ==")
        print(f"   bbox x[{min(xs)},{max(xs)}] y[{min(ys)},{max(ys)}]"
              f"   检测区多边形 {len(cam['polys'])} 个  区域环 {len(ring)} 顶点")
        print(f"   弧长中点在 seg#{seg} {pts[seg]}->{pts[seg+1]}  mid=({mid[0]:.0f},{mid[1]:.0f})"
              f"  内侧法向=({nx:+.2f},{ny:+.2f})")
        if queries:
            print("   " + "-" * 74)
            for pt in queries:
                ev = eval_line(pts, sign, pt)
                r = region_of(cam, ring, pt)
                rn = {0: "R0区域外", 1: "R1入口带", 2: "R2检测区"}[r]
                print(f"   脚底 ({pt[0]:.0f},{pt[1]:.0f})  {rn}")
                print(f"     最近线段 seg#{ev['seg']} {ev['a']}->{ev['b']}")
                print(f"     带符号垂距 line_signed = {ev['signed']:+.1f}px"
                      f"   (死区 ±{CROSS_LINE_DEPTH:.0f}, 推断窗口 ≤100)")
                print(f"     到折线真实最近距离 = {ev['best_dist']:.1f}px"
                      f"   投影参数 t={ev['t']:.3f} (t_px={ev['t_px']:.1f}, L={ev['L']:.1f})")
                print(f"     在跨度内 in_span = {ev['in_span']}"
                      f"   ({'首段' if ev['is_first'] else ''}{'末段' if ev['is_last'] else ''}"
                      f"{'内部段→强制 true' if not ev['is_first'] and not ev['is_last'] else ''})")
                print(f"     脚底到边界(best_dist) vs 垂距 差 {ev['best_dist'] - abs(ev['signed']):.1f}px"
                      f"   {'⚠ 垂距远小于真实距离 → 垂距不可信' if ev['best_dist'] - abs(ev['signed']) > 30 else ''}")


if __name__ == "__main__":
    main()
