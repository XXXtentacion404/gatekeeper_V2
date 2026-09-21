# -*- coding: utf-8 -*-
"""生成"判定线 = 区域边缘折线"的区域几何示意图（离线工具，不属于正式测试套件）。

要点（对应文档 §17）：
    判定线 = 检测区区域边缘的一段折线（点数不限，边缘是折的/弯的都能跟着走）；
    它**本身就是判定线** —— 脚底跨过它 = 一次计数；
    区域 = 判定线 ∪ 其整体沿 x 平移 |平移量| px（y 不变）的副本。
用法: python _fig_zone_poly.py [out.png]
"""
import importlib.util
import os
import sys

import cv2
import numpy as np

# 合成场景：一条"左边是折线"的走廊。判定线 = 走廊左边界的前 3 个顶点。
POLY = [(700, 200), (800, 450), (750, 900), (1300, 900), (1300, 200)]
LINE = [(700, 200), (800, 450), (750, 900)]
DEPTH = 100.0
ROI = (640, 140, 1120, 960)     # x0,y0,x1,y1
SCALE = 0.80
BG = 246
# ⚠️ cv2.putText 只支持 Hershey 矢量字体，画不了中文（会变成一串 ?）→ 图内文字一律 ASCII，
#    中文说明放在控制台输出和文档里。
T_OLD = "OLD: zone offset by NORMAL"
S_OLD = "kinks at the bend; leaves gaps beside the neighbour zone"
T_NEW = "NOW: zone = edge + x-shift"
S_NEW = "the edge itself IS the judging line; zone width = |dx|"


def load_tool(path):
    spec = importlib.util.spec_from_file_location("gatedraw", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def old_style_quads(mod, line, depth):
    """改前的口径：按每段的**法向**偏移 |depth| px（斜带）。折线处带子会拐弯、外边缘不等距。"""
    n = len(line) - 1
    seg, _ = mod._poly_mid(line)
    ux, uy = mod._right_hand_normal(line[seg], line[seg + 1])
    # 用检测区侧定向：合成场景里内部在 +x 侧
    sign = 1.0 if ux >= 0 else -1.0
    quads = []
    for i in range(n):
        ax, ay = line[i]
        bx, by = line[i + 1]
        nx, ny = mod._right_hand_normal(line[i], line[i + 1])
        dx, dy = nx * depth * sign, ny * depth * sign
        quads.append([(int(ax), int(ay)), (int(bx), int(by)),
                      (int(round(bx + dx)), int(round(by + dy))),
                      (int(round(ax + dx)), int(round(ay + dy)))])
    return quads


def draw_panel(title, subtitle, poly, line, fills, accent):
    x0, y0, x1, y1 = ROI
    img = np.full((y1 - y0, x1 - x0, 3), BG, np.uint8)

    def P(pts):
        return (np.array(pts, np.float32) - np.array([x0, y0])).astype(np.int32)

    cv2.polylines(img, [P(poly)], True, (200, 140, 60), 2)      # 检测区
    for quad in fills:                                          # 区域
        ov = img.copy()
        cv2.fillPoly(ov, [P(quad)], accent)
        cv2.addWeighted(ov, 0.38, img, 0.62, 0, img)
        cv2.polylines(img, [P(quad)], True, tuple(int(c * 0.7) for c in accent), 2)
    cv2.polylines(img, [P(line)], False, (0, 200, 245), 3)       # 判定线
    for p in line:
        cv2.circle(img, tuple(P([p])[0]), 5, (0, 200, 245), -1)
        cv2.circle(img, tuple(P([p])[0]), 5, (40, 40, 40), 1)

    img = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
    bar = np.full((52, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                tuple(int(c * 0.65) for c in accent), 2)
    cv2.putText(bar, subtitle, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                (90, 90, 90), 1)
    return np.vstack([bar, img])


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    while not os.path.exists(os.path.join(root, "cameras_config.txt")):
        root = os.path.dirname(root)
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        root, "文档", "判定线_折线边缘_区域几何.png")
    mod = load_tool(os.path.join(root, "闸门计数-网络摄像头配置文件绘制.py"))

    depth = mod._resolve_depth_sign([POLY], LINE, DEPTH)
    offset = mod._x_offset_from_depth(LINE, depth)
    ring = mod._zone_ring(LINE, depth)
    old = old_style_quads(mod, LINE, abs(depth))

    left = draw_panel(T_OLD, S_OLD, POLY, LINE, old, (235, 90, 90))
    right = draw_panel(T_NEW, f"{S_NEW}  (dx = {offset:+.0f}px)", POLY, LINE,
                       [ring], (60, 175, 60))

    h = max(left.shape[0], right.shape[0])
    pads = []
    for a in (left, right):
        pads.append(np.pad(a, ((0, h - a.shape[0]), (0, 0), (0, 0)), constant_values=255))
    img = np.hstack([pads[0], np.full((h, 14, 3), 120, np.uint8), pads[1]])

    print(f"判定线({len(LINE)} 点) = {LINE}")
    print(f"depth_signed = {depth:+.0f}  →  x 平移量 = {offset:+.0f}")
    print(f"区域环 {len(ring)} 点 = {ring}")
    # ⚠️ Windows 上 cv2.imwrite 用窄字符 fopen，中文路径会静默失败 → 走 imencode+tofile
    cv2.imencode(".png", img)[1].tofile(out)
    print("saved:", out, img.shape)


if __name__ == "__main__":
    main()
